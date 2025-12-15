from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from app.config import settings
from app.services.langfuse_service import LangFuseService
from app.services.cognee_service import CogneeService
from app.models.database import init_db

# Import routers
from app.api.routes import analytics, topics, knowledge_gaps, quality, realtime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global service instances
langfuse_service: LangFuseService = None
cognee_service: CogneeService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    global langfuse_service, cognee_service

    # Startup
    logger.info("Initializing services...")

    # Initialize database
    await init_db()
    logger.info("Database initialized")

    # Initialize LangFuse
    langfuse_service = LangFuseService(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )
    logger.info("LangFuse service initialized")

    # Initialize Cognee
    cognee_service = CogneeService(
        llm_api_key=settings.cognee_llm_api_key
    )
    await cognee_service.initialize()
    logger.info("Cognee service initialized")

    yield

    # Shutdown
    logger.info("Shutting down services...")
    if langfuse_service:
        langfuse_service.shutdown()


app = FastAPI(
    title="Knowledge Gap Dashboard API",
    description="API for identifying missing knowledge in AI chatbots",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",   # LangFuse (if needed)
        "http://localhost:3001",   # Dashboard frontend
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(analytics.router, prefix="/api")
app.include_router(topics.router, prefix="/api")
app.include_router(knowledge_gaps.router, prefix="/api")
app.include_router(quality.router, prefix="/api")
app.include_router(realtime.router, prefix="/api")


def get_langfuse_service() -> LangFuseService:
    """Dependency for getting the LangFuse service"""
    return langfuse_service


def get_cognee_service() -> CogneeService:
    """Dependency for getting the Cognee service"""
    return cognee_service


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok", "service": "knowledge-gap-dashboard"}


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Knowledge Gap Dashboard API",
        "docs": "/docs",
        "health": "/health",
        "api": {
            "analytics": "/api/analytics",
            "topics": "/api/topics",
            "knowledge_gaps": "/api/knowledge-gaps",
            "quality": "/api/quality",
            "realtime": "/api/realtime/stream",
            "debug_langfuse": "/api/debug/langfuse",
            "trigger_sync": "/api/sync/trigger",
        }
    }


@app.get("/api/debug/langfuse")
async def debug_langfuse():
    """Debug endpoint to test LangFuse connection and fetch traces"""
    try:
        # Test fetching traces
        traces = await langfuse_service.fetch_traces(limit=10)
        return {
            "status": "connected",
            "host": settings.langfuse_host,
            "traces_found": len(traces),
            "sample_traces": [
                {
                    "trace_id": t.get("trace_id"),
                    "name": t.get("name"),
                    "input": str(t.get("input", ""))[:100] + "..." if t.get("input") else None,
                    "timestamp": str(t.get("timestamp")),
                }
                for t in traces[:5]
            ]
        }
    except Exception as e:
        logger.error(f"LangFuse debug error: {e}")
        return {
            "status": "error",
            "host": settings.langfuse_host,
            "error": str(e),
        }


@app.post("/api/sync/trigger")
async def trigger_sync():
    """Manually trigger a LangFuse data sync"""
    from app.tasks.sync_tasks import sync_langfuse_data
    try:
        # Trigger the celery task
        task = sync_langfuse_data.delay()
        return {
            "status": "triggered",
            "task_id": str(task.id),
            "message": "Sync task has been queued"
        }
    except Exception as e:
        logger.error(f"Error triggering sync: {e}")
        return {
            "status": "error",
            "error": str(e),
        }


def make_naive(dt):
    """Convert timezone-aware datetime to naive UTC datetime"""
    if dt is None:
        return None
    if hasattr(dt, 'tzinfo') and dt.tzinfo is not None:
        from datetime import timezone
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


@app.post("/api/sync/now")
async def sync_now():
    """Run LangFuse sync immediately (blocking)"""
    from datetime import datetime, timedelta
    from app.models.database import async_session_maker, Trace, Score, KnowledgeGap, SyncState
    from app.services.topic_classifier import TopicClassifier
    from app.services.gap_analyzer import GapAnalyzer
    from sqlalchemy import select
    import re
    import ast

    # Patterns for detecting missing knowledge
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
        r"not stated",
        r"does not.*provide",
        r"does not.*list",
        r"does not.*mention",
    ]

    def extract_query_from_input(input_data) -> str:
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
        if not response:
            return False
        response_lower = response.lower()
        for pattern in MISSING_KNOWLEDGE_PATTERNS:
            if re.search(pattern, response_lower):
                return True
        return False

    try:
        async with async_session_maker() as db:
            # Get last sync timestamp
            sync_state_query = select(SyncState).where(SyncState.source == "langfuse")
            result = await db.execute(sync_state_query)
            sync_state = result.scalar_one_or_none()

            last_sync = sync_state.last_sync_timestamp if sync_state else datetime.utcnow() - timedelta(days=7)

            topic_classifier = TopicClassifier()
            gap_analyzer = GapAnalyzer(langfuse_service, cognee_service)

            # Fetch new data from LangFuse
            sync_result = await langfuse_service.sync_new_data(
                last_sync_timestamp=last_sync,
                batch_size=100,
            )

            traces = sync_result.get("traces", [])
            new_traces_count = 0
            new_gaps_count = 0

            for trace_data in traces:
                # Classify topic
                input_text = trace_data.get("input", "")
                if isinstance(input_text, dict):
                    input_text = str(input_text)
                topic, confidence = topic_classifier.classify(input_text or "")

                # Check for missing knowledge in the response
                output_text = str(trace_data.get("output", ""))
                answer = extract_answer_from_output(output_text)
                has_missing_knowledge = is_missing_knowledge_response(answer)

                # Create trace record (convert timestamps to naive UTC)
                trace = Trace(
                    trace_id=trace_data["trace_id"],
                    session_id=trace_data.get("session_id"),
                    user_id=trace_data.get("user_id"),
                    input=input_text,
                    output=output_text,
                    name=trace_data.get("name"),
                    tags=trace_data.get("tags", []),
                    trace_metadata=trace_data.get("metadata", {}),
                    timestamp=make_naive(trace_data.get("timestamp")),
                    latency=trace_data.get("latency"),
                    total_cost=trace_data.get("total_cost"),
                    prompt_tokens=trace_data.get("usage", {}).get("prompt_tokens", 0),
                    completion_tokens=trace_data.get("usage", {}).get("completion_tokens", 0),
                    total_tokens=trace_data.get("usage", {}).get("total_tokens", 0),
                    feedback_negative=trace_data.get("feedback_negative", False),
                    has_missing_knowledge=has_missing_knowledge,
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
                        timestamp=make_naive(score_data.get("timestamp")),
                    )

                    existing_score_query = select(Score).where(Score.score_id == score_data.get("score_id"))
                    existing_score_result = await db.execute(existing_score_query)
                    if not existing_score_result.scalar_one_or_none():
                        db.add(score)

            # Detect and save knowledge gaps from missing knowledge patterns
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
                            existing_gap.sample_trace_ids = existing_trace_ids[:20]
                        existing_gap.updated_at = datetime.utcnow()
                    else:
                        # Create new gap from missing knowledge detection
                        gap_topic, gap_confidence = topic_classifier.classify(question)

                        new_gap = KnowledgeGap(
                            question=question,
                            topic=gap_topic,
                            frequency=1,
                            priority_score=0.6,
                            priority_level="medium",
                            confidence=gap_confidence,
                            related_concepts=[],
                            suggested_addition=f"The AI indicated it doesn't have information to answer this question.",
                            sample_trace_ids=[trace_data.get("trace_id")] if trace_data.get("trace_id") else [],
                        )
                        db.add(new_gap)
                        new_gaps_count += 1
                        logger.info(f"Created knowledge gap from missing knowledge: {question[:50]}...")

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

            # Update topic analytics from traces
            from app.services.analytics_service import AnalyticsService
            analytics_service = AnalyticsService(db)
            await analytics_service.update_topic_analytics()

            return {
                "status": "success",
                "traces_fetched": len(traces),
                "new_traces_saved": new_traces_count,
                "new_gaps_found": new_gaps_count,
                "last_sync": last_sync.isoformat() if last_sync else None,
            }

    except Exception as e:
        logger.error(f"Sync error: {e}")
        import traceback
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


@app.post("/api/sync/detect-gaps")
async def detect_gaps_from_existing():
    """
    Scan existing traces for missing knowledge patterns and:
    1. Update the has_missing_knowledge flag on traces
    2. Create knowledge gaps for those traces
    """
    from datetime import datetime
    from app.models.database import async_session_maker, Trace, KnowledgeGap
    from app.services.topic_classifier import TopicClassifier
    from sqlalchemy import select
    import re
    import ast

    # Patterns for detecting missing knowledge
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
        r"not stated",
        r"does not.*provide",
        r"does not.*list",
        r"does not.*mention",
    ]

    def extract_query_from_input(input_data) -> str:
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
        if not response:
            return False
        response_lower = response.lower()
        for pattern in MISSING_KNOWLEDGE_PATTERNS:
            if re.search(pattern, response_lower):
                return True
        return False

    try:
        async with async_session_maker() as db:
            topic_classifier = TopicClassifier()

            # Fetch all traces
            query = select(Trace).order_by(Trace.timestamp.desc()).limit(500)
            result = await db.execute(query)
            traces = result.scalars().all()

            new_gaps_count = 0
            updated_gaps_count = 0
            traces_updated = 0

            for trace in traces:
                answer = extract_answer_from_output(trace.output)
                question = extract_query_from_input(trace.input)
                has_missing = is_missing_knowledge_response(answer)

                # Update trace if missing knowledge status changed
                if trace.has_missing_knowledge != has_missing:
                    trace.has_missing_knowledge = has_missing
                    trace.updated_at = datetime.utcnow()
                    traces_updated += 1

                if question and has_missing:
                    # Check if this question already exists as a gap
                    existing_gap_query = select(KnowledgeGap).where(
                        KnowledgeGap.question == question,
                        KnowledgeGap.resolved == False
                    )
                    existing_gap_result = await db.execute(existing_gap_query)
                    existing_gap = existing_gap_result.scalar_one_or_none()

                    if existing_gap:
                        # Update trace IDs if not already present
                        existing_trace_ids = existing_gap.sample_trace_ids or []
                        if trace.trace_id not in existing_trace_ids:
                            existing_trace_ids.append(trace.trace_id)
                            existing_gap.sample_trace_ids = existing_trace_ids[:20]
                            existing_gap.updated_at = datetime.utcnow()
                            updated_gaps_count += 1
                    else:
                        # Create new gap from missing knowledge detection
                        gap_topic, gap_confidence = topic_classifier.classify(question)

                        new_gap = KnowledgeGap(
                            question=question,
                            topic=gap_topic,
                            frequency=1,
                            priority_score=0.6,
                            priority_level="medium",
                            confidence=gap_confidence,
                            related_concepts=[],
                            suggested_addition=f"The AI indicated it doesn't have information to answer this question.",
                            sample_trace_ids=[trace.trace_id] if trace.trace_id else [],
                        )
                        db.add(new_gap)
                        new_gaps_count += 1
                        logger.info(f"Created knowledge gap: {question[:50]}...")

            await db.commit()

            # Update topic analytics from traces
            from app.services.analytics_service import AnalyticsService
            analytics_service = AnalyticsService(db)
            await analytics_service.update_topic_analytics()

            return {
                "status": "success",
                "traces_scanned": len(traces),
                "traces_updated": traces_updated,
                "new_gaps_created": new_gaps_count,
                "gaps_updated": updated_gaps_count,
            }

    except Exception as e:
        logger.error(f"Gap detection error: {e}")
        import traceback
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


@app.post("/api/sync/update-topics")
async def update_topic_analytics():
    """Recalculate topic analytics from traces (traces are source of truth)"""
    from datetime import datetime
    from app.models.database import async_session_maker, Trace, TopicAnalytics
    from sqlalchemy import select, func, and_, or_, delete

    try:
        async with async_session_maker() as db:
            # Get all unique topics from traces (source of truth)
            topics_query = select(Trace.topic).distinct().where(
                and_(Trace.topic.isnot(None), Trace.topic != "Uncategorized")
            )
            topics_result = await db.execute(topics_query)
            topics = [row[0] for row in topics_result.fetchall()]

            logger.info(f"Found {len(topics)} unique topics in traces: {topics}")

            # Clear all existing topic analytics to ensure consistency
            await db.execute(delete(TopicAnalytics))

            created_count = 0

            for topic in topics:
                # Calculate metrics for this topic
                total_query = select(func.count(Trace.id)).where(Trace.topic == topic)
                total_result = await db.execute(total_query)
                total = total_result.scalar() or 0

                success_query = select(func.count(Trace.id)).where(
                    and_(
                        Trace.topic == topic,
                        Trace.feedback_negative == False,
                        or_(Trace.has_missing_knowledge == False, Trace.has_missing_knowledge.is_(None))
                    )
                )
                success_result = await db.execute(success_query)
                success = success_result.scalar() or 0

                avg_conf_query = select(func.avg(Trace.topic_confidence)).where(Trace.topic == topic)
                avg_conf_result = await db.execute(avg_conf_query)
                avg_confidence = avg_conf_result.scalar() or 0

                logger.info(f"Topic '{topic}': total={total}, success={success}, conf={avg_confidence}")

                new_analytics = TopicAnalytics(
                    topic=topic,
                    question_count=total,
                    success_count=success,
                    failure_count=total - success,
                    success_rate=(success / total * 100) if total > 0 else 0,
                    avg_confidence=avg_confidence or 0,
                )
                db.add(new_analytics)
                created_count += 1

            await db.commit()

            return {
                "status": "success",
                "topics_found": len(topics),
                "topics_created": created_count,
            }

    except Exception as e:
        logger.error(f"Topic analytics update error: {e}")
        import traceback
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


@app.post("/api/sync/discover-topics")
async def discover_topics():
    """
    Use BERTopic to automatically discover topics from all questions.
    This will reclassify all traces with dynamically discovered topics.
    """
    from datetime import datetime
    from app.models.database import async_session_maker, Trace, TopicAnalytics
    from sqlalchemy import select, func, and_, or_, delete
    import re
    import ast

    def extract_query_from_input(input_data) -> str:
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

    try:
        # Import the dynamic topic modeler
        from app.services.dynamic_topic_modeler import get_topic_modeler

        async with async_session_maker() as db:
            # Get all questions from traces
            query = select(Trace).order_by(Trace.timestamp.desc())
            result = await db.execute(query)
            traces = result.scalars().all()

            if not traces:
                return {
                    "status": "error",
                    "error": "No traces found to discover topics from"
                }

            # Extract questions
            questions = []
            trace_ids = []
            for trace in traces:
                question = extract_query_from_input(trace.input)
                if question and len(question.strip()) > 5:  # Skip very short questions
                    questions.append(question)
                    trace_ids.append(trace.id)

            if len(questions) < 2:
                return {
                    "status": "error",
                    "error": f"Not enough questions to discover topics (found {len(questions)}, need at least 2)"
                }

            logger.info(f"Discovering topics from {len(questions)} questions...")

            # Get the topic modeler and fit it
            topic_modeler = get_topic_modeler()
            topic_modeler.fit(questions)

            if not topic_modeler.is_fitted:
                return {
                    "status": "error",
                    "error": "Failed to fit topic model"
                }

            # Get discovered topics info
            topic_info = topic_modeler.get_topic_info()
            logger.info(f"Discovered {len(topic_info)} topics")

            # Reclassify all traces with discovered topics
            classifications = topic_modeler.classify_batch(questions)
            traces_updated = 0

            for i, (trace_id, (topic_label, confidence)) in enumerate(zip(trace_ids, classifications)):
                # Find and update the trace
                trace_query = select(Trace).where(Trace.id == trace_id)
                trace_result = await db.execute(trace_query)
                trace = trace_result.scalar_one_or_none()

                if trace:
                    trace.topic = topic_label
                    trace.topic_confidence = confidence
                    trace.updated_at = datetime.utcnow()
                    traces_updated += 1

            await db.commit()

            # Now update topic analytics based on actual trace data (source of truth)
            # Clear existing topic analytics
            await db.execute(delete(TopicAnalytics))

            # Get all unique topics from traces (this is the source of truth)
            distinct_topics_query = select(Trace.topic).distinct().where(Trace.topic.isnot(None))
            distinct_result = await db.execute(distinct_topics_query)
            all_topics = [row[0] for row in distinct_result.fetchall()]

            logger.info(f"Creating analytics for {len(all_topics)} topics from traces: {all_topics}")

            for label in all_topics:
                if label == "Uncategorized":
                    continue  # Skip uncategorized

                # Count traces with this topic
                count_query = select(func.count(Trace.id)).where(Trace.topic == label)
                count_result = await db.execute(count_query)
                total = count_result.scalar() or 0

                success_query = select(func.count(Trace.id)).where(
                    and_(
                        Trace.topic == label,
                        Trace.feedback_negative == False,
                        or_(Trace.has_missing_knowledge == False, Trace.has_missing_knowledge.is_(None))
                    )
                )
                success_result = await db.execute(success_query)
                success = success_result.scalar() or 0

                avg_conf_query = select(func.avg(Trace.topic_confidence)).where(Trace.topic == label)
                avg_conf_result = await db.execute(avg_conf_query)
                avg_confidence = avg_conf_result.scalar() or 0

                new_analytics = TopicAnalytics(
                    topic=label,
                    question_count=total,
                    success_count=success,
                    failure_count=total - success,
                    success_rate=(success / total * 100) if total > 0 else 0,
                    avg_confidence=avg_confidence or 0,
                )
                db.add(new_analytics)

            await db.commit()

            return {
                "status": "success",
                "questions_processed": len(questions),
                "topics_discovered": len(topic_info),
                "traces_updated": traces_updated,
                "topics": [
                    {
                        "id": t.get("id"),
                        "label": t.get("label"),
                        "count": t.get("count"),
                        "top_words": t.get("top_words", [])[:5]
                    }
                    for t in topic_info
                ]
            }

    except ImportError as e:
        logger.error(f"BERTopic not available: {e}")
        return {
            "status": "error",
            "error": f"BERTopic not installed: {e}",
        }
    except Exception as e:
        logger.error(f"Topic discovery error: {e}")
        import traceback
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
