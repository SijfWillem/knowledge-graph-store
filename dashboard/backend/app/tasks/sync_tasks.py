import asyncio
from datetime import datetime, timedelta
from typing import Optional
import logging
import json
import re
import ast

from celery import shared_task
import redis

from app.config import settings
from app.services.langfuse_service import LangFuseService
from app.services.cognee_service import CogneeService
from app.services.gap_analyzer import GapAnalyzer
from app.services.topic_classifier import TopicClassifier
from app.models.database import (
    async_session_maker,
    Trace,
    Score,
    KnowledgeGap,
    SyncState,
)
from sqlalchemy import select, update

logger = logging.getLogger(__name__)

# Patterns that indicate missing knowledge in AI responses
MISSING_KNOWLEDGE_PATTERNS = [
    r"i don'?t know",
    r"no information",
    r"not available",
    r"cannot find",
    r"don'?t have.*information",
    r"no.*context",
    r"unable to.*answer",
    r"not.*provided",
    r"no data",
    r"cannot.*determine",
    r"not sure",
    r"no relevant",
]


def extract_query_from_input(input_data) -> str:
    """Extract the actual query from the input data"""
    if input_data is None:
        return ""

    if isinstance(input_data, str):
        try:
            parsed = ast.literal_eval(input_data)
            if isinstance(parsed, dict):
                return parsed.get('query', str(input_data))
        except (ValueError, SyntaxError):
            pass
        return input_data

    if isinstance(input_data, dict):
        return input_data.get('query', str(input_data))

    return str(input_data)


def extract_answer_from_output(output_data) -> str:
    """Extract the actual answer from the output data"""
    if output_data is None:
        return ""

    if isinstance(output_data, str):
        try:
            parsed = ast.literal_eval(output_data)
            if isinstance(parsed, dict):
                for key in ['answer', 'response', 'result', 'text', 'content', 'error']:
                    if key in parsed:
                        return str(parsed[key])
                return str(parsed)
        except (ValueError, SyntaxError):
            pass
        return output_data

    if isinstance(output_data, dict):
        for key in ['answer', 'response', 'result', 'text', 'content', 'error']:
            if key in output_data:
                return str(output_data[key])
        return str(output_data)

    return str(output_data)


def is_missing_knowledge_response(response: str) -> bool:
    """Check if the response indicates missing knowledge"""
    if not response:
        return False

    response_lower = response.lower()
    for pattern in MISSING_KNOWLEDGE_PATTERNS:
        if re.search(pattern, response_lower):
            return True
    return False

# Redis client for publishing events
redis_client = redis.from_url(settings.redis_url)


def run_async(coro):
    """Helper to run async functions in sync context"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@shared_task(bind=True, max_retries=3)
def sync_langfuse_data(self):
    """
    Every 5 minutes:
    1. Fetch new traces from LangFuse
    2. Classify topics
    3. Detect new knowledge gaps
    4. Update analytics
    5. Publish real-time events
    """
    try:
        run_async(_sync_langfuse_data_async())
        return {"status": "success", "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"Error syncing LangFuse data: {e}")
        self.retry(exc=e, countdown=60)


async def _sync_langfuse_data_async():
    """Async implementation of LangFuse sync"""
    async with async_session_maker() as db:
        # Get last sync timestamp
        sync_state_query = select(SyncState).where(SyncState.source == "langfuse")
        result = await db.execute(sync_state_query)
        sync_state = result.scalar_one_or_none()

        last_sync = sync_state.last_sync_timestamp if sync_state else datetime.utcnow() - timedelta(days=1)

        # Initialize services
        langfuse = LangFuseService(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )

        cognee = CogneeService(llm_api_key=settings.cognee_llm_api_key)
        await cognee.initialize()

        topic_classifier = TopicClassifier()
        gap_analyzer = GapAnalyzer(langfuse, cognee)

        # Fetch new data
        sync_result = await langfuse.sync_new_data(
            last_sync_timestamp=last_sync,
            batch_size=100,
        )

        traces = sync_result.get("traces", [])
        new_traces_count = 0
        new_gaps_count = 0

        for trace_data in traces:
            # Classify topic
            topic, confidence = topic_classifier.classify(trace_data.get("input", ""))

            # Create trace record
            trace = Trace(
                trace_id=trace_data["trace_id"],
                session_id=trace_data.get("session_id"),
                user_id=trace_data.get("user_id"),
                input=trace_data.get("input"),
                output=trace_data.get("output"),
                name=trace_data.get("name"),
                tags=trace_data.get("tags", []),
                trace_metadata=trace_data.get("metadata", {}),
                timestamp=trace_data.get("timestamp"),
                latency=trace_data.get("latency"),
                total_cost=trace_data.get("total_cost"),
                prompt_tokens=trace_data.get("usage", {}).get("prompt_tokens", 0),
                completion_tokens=trace_data.get("usage", {}).get("completion_tokens", 0),
                total_tokens=trace_data.get("usage", {}).get("total_tokens", 0),
                feedback_negative=trace_data.get("feedback_negative", False),
                topic=topic,
                topic_confidence=confidence,
            )

            # Check if trace already exists
            existing_query = select(Trace).where(Trace.trace_id == trace_data["trace_id"])
            existing_result = await db.execute(existing_query)
            if not existing_result.scalar_one_or_none():
                db.add(trace)
                new_traces_count += 1

            # Save scores
            for score_data in trace_data.get("scores", []):
                score = Score(
                    score_id=score_data.get("score_id"),
                    trace_id=trace_data["trace_id"],
                    name=score_data.get("name"),
                    value=score_data.get("value"),
                    comment=score_data.get("comment"),
                    source=score_data.get("source"),
                    timestamp=score_data.get("timestamp"),
                )

                # Check if score already exists
                existing_score_query = select(Score).where(Score.score_id == score_data.get("score_id"))
                existing_score_result = await db.execute(existing_score_query)
                if not existing_score_result.scalar_one_or_none():
                    db.add(score)

        # Analyze for knowledge gaps from negative feedback
        failed_traces = [t for t in traces if t.get("feedback_negative")]
        if failed_traces:
            gaps = await gap_analyzer.analyze_failed_conversations(failed_traces)
            clustered_gaps = await gap_analyzer.cluster_similar_gaps(gaps)

            for gap_data in clustered_gaps:
                # Check if similar gap already exists
                existing_gap_query = select(KnowledgeGap).where(
                    KnowledgeGap.question == gap_data["question"],
                    KnowledgeGap.resolved == False
                )
                existing_gap_result = await db.execute(existing_gap_query)
                existing_gap = existing_gap_result.scalar_one_or_none()

                if existing_gap:
                    # Update frequency
                    existing_gap.frequency += gap_data.get("frequency", 1)
                    existing_gap.priority_score = gap_data.get("priority_score", existing_gap.priority_score)
                    existing_gap.priority_level = gap_data.get("priority_level", existing_gap.priority_level)
                    existing_gap.updated_at = datetime.utcnow()
                else:
                    # Create new gap
                    new_gap = KnowledgeGap(
                        question=gap_data["question"],
                        topic=gap_data.get("topic") or topic,
                        frequency=gap_data.get("frequency", 1),
                        priority_score=gap_data.get("priority_score", 0.5),
                        priority_level=gap_data.get("priority_level", "medium"),
                        confidence=gap_data.get("confidence"),
                        related_concepts=gap_data.get("related_concepts", []),
                        suggested_addition=gap_data.get("suggested_addition"),
                        sample_trace_ids=gap_data.get("sample_trace_ids", []),
                    )
                    db.add(new_gap)
                    new_gaps_count += 1

        # Also create knowledge gaps from pattern-detected missing knowledge
        # This catches cases where the AI admits it doesn't know, even without explicit negative feedback
        for trace_data in traces:
            answer = extract_answer_from_output(trace_data.get("output"))
            question = extract_query_from_input(trace_data.get("input"))

            if question and is_missing_knowledge_response(answer):
                # Check if this question already exists as a gap
                existing_gap_query = select(KnowledgeGap).where(
                    KnowledgeGap.question == question,
                    KnowledgeGap.resolved == False
                )
                existing_gap_result = await db.execute(existing_gap_query)
                existing_gap = existing_gap_result.scalar_one_or_none()

                if existing_gap:
                    # Update frequency and trace IDs
                    existing_gap.frequency += 1
                    existing_trace_ids = existing_gap.sample_trace_ids or []
                    if trace_data.get("trace_id") not in existing_trace_ids:
                        existing_trace_ids.append(trace_data.get("trace_id"))
                        existing_gap.sample_trace_ids = existing_trace_ids[:20]  # Keep max 20
                    existing_gap.updated_at = datetime.utcnow()
                else:
                    # Create new gap from missing knowledge detection
                    # Classify topic for this trace
                    gap_topic, gap_confidence = topic_classifier.classify(question)

                    new_gap = KnowledgeGap(
                        question=question,
                        topic=gap_topic,
                        frequency=1,
                        priority_score=0.6,  # Medium-high priority for auto-detected gaps
                        priority_level="medium",
                        confidence=gap_confidence,
                        related_concepts=[],
                        suggested_addition=f"The AI indicated it doesn't have information to answer: {question}",
                        sample_trace_ids=[trace_data.get("trace_id")] if trace_data.get("trace_id") else [],
                    )
                    db.add(new_gap)
                    new_gaps_count += 1
                    logger.info(f"Created knowledge gap from missing knowledge pattern: {question[:50]}...")

        # Update sync state
        if sync_state:
            sync_state.last_sync_timestamp = sync_result.get("sync_timestamp", datetime.utcnow())
            sync_state.last_sync_count = new_traces_count
            sync_state.updated_at = datetime.utcnow()
        else:
            new_sync_state = SyncState(
                source="langfuse",
                last_sync_timestamp=sync_result.get("sync_timestamp", datetime.utcnow()),
                last_sync_count=new_traces_count,
            )
            db.add(new_sync_state)

        await db.commit()

        # Publish real-time event
        event_data = {
            "type": "sync_complete",
            "new_traces": new_traces_count,
            "new_gaps": new_gaps_count,
            "timestamp": datetime.utcnow().isoformat(),
        }
        redis_client.publish("dashboard_events", json.dumps(event_data))

        logger.info(f"Sync complete: {new_traces_count} new traces, {new_gaps_count} new gaps")


@shared_task(bind=True, max_retries=2)
def recalculate_gap_priorities(self):
    """
    Every hour:
    1. Re-cluster similar gaps
    2. Recalculate priority scores based on frequency and recency
    """
    try:
        run_async(_recalculate_gap_priorities_async())
        return {"status": "success", "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"Error recalculating gap priorities: {e}")
        self.retry(exc=e, countdown=120)


async def _recalculate_gap_priorities_async():
    """Async implementation of gap priority recalculation"""
    async with async_session_maker() as db:
        # Get all unresolved gaps
        gaps_query = select(KnowledgeGap).where(KnowledgeGap.resolved == False)
        result = await db.execute(gaps_query)
        gaps = result.scalars().all()

        for gap in gaps:
            # Recalculate priority based on frequency and recency
            frequency_score = min(1.0, gap.frequency / 50)  # Max out at 50 occurrences

            # Recency factor
            days_since_update = (datetime.utcnow() - gap.updated_at).days if gap.updated_at else 30
            recency_score = max(0, 1 - (days_since_update / 30))

            # Knowledge gap certainty (inverse of confidence)
            gap_certainty = 1 - (gap.confidence or 0.5)

            # Calculate new priority score
            new_priority = (
                frequency_score * 0.4 +
                recency_score * 0.3 +
                gap_certainty * 0.3
            )

            # Determine priority level
            if new_priority >= 0.8:
                priority_level = "critical"
            elif new_priority >= 0.6:
                priority_level = "high"
            elif new_priority >= 0.4:
                priority_level = "medium"
            else:
                priority_level = "low"

            gap.priority_score = round(new_priority, 3)
            gap.priority_level = priority_level
            gap.updated_at = datetime.utcnow()

        await db.commit()

        # Publish update event
        event_data = {
            "type": "priorities_updated",
            "gaps_updated": len(gaps),
            "timestamp": datetime.utcnow().isoformat(),
        }
        redis_client.publish("dashboard_events", json.dumps(event_data))

        logger.info(f"Recalculated priorities for {len(gaps)} gaps")


@shared_task(bind=True, max_retries=2)
def update_topic_analytics(self):
    """
    Every 30 minutes:
    Update aggregated analytics per topic
    """
    try:
        run_async(_update_topic_analytics_async())
        return {"status": "success", "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"Error updating topic analytics: {e}")
        self.retry(exc=e, countdown=60)


async def _update_topic_analytics_async():
    """Async implementation of topic analytics update"""
    from app.services.analytics_service import AnalyticsService

    async with async_session_maker() as db:
        analytics_service = AnalyticsService(db)
        await analytics_service.update_topic_analytics()

        # Publish update event
        event_data = {
            "type": "analytics_updated",
            "timestamp": datetime.utcnow().isoformat(),
        }
        redis_client.publish("dashboard_events", json.dumps(event_data))

        logger.info("Topic analytics updated")
