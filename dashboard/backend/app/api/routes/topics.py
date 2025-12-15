from fastapi import APIRouter, Depends, HTTPException
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.database import get_db, TopicAnalytics, Trace
from app.services.analytics_service import AnalyticsService

router = APIRouter(prefix="/topics", tags=["topics"])


@router.get("/")
async def get_topics(
    db: AsyncSession = Depends(get_db)
):
    """
    Get all topics with their analytics data.
    Topics are dynamically discovered using BERTopic.
    Returns list of topics with:
    - question_count
    - success_rate
    - handover_rate
    - avg_confidence
    - trend
    - priority
    """
    analytics_service = AnalyticsService(db)
    topics = await analytics_service.get_topics_analytics()

    if not topics:
        # Return empty list - topics need to be discovered first
        return []

    return topics


@router.get("/distribution")
async def get_topic_distribution(
    db: AsyncSession = Depends(get_db)
):
    """
    Get distribution of questions across topics.
    Returns percentage breakdown.
    """
    query = select(TopicAnalytics)
    result = await db.execute(query)
    topics = result.scalars().all()

    total_questions = sum(t.question_count for t in topics)

    distribution = []
    for topic in topics:
        percentage = (topic.question_count / total_questions * 100) if total_questions > 0 else 0
        distribution.append({
            "topic": topic.topic,
            "count": topic.question_count,
            "percentage": round(percentage, 2),
        })

    # Sort by count descending
    distribution.sort(key=lambda x: x["count"], reverse=True)

    return {
        "total_questions": total_questions,
        "distribution": distribution,
    }


@router.get("/{topic_name}")
async def get_topic_detail(
    topic_name: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Single topic deep dive with:
    - Full analytics
    - Sample questions
    - Recent trends
    - Common failure patterns
    """
    # Get topic analytics
    query = select(TopicAnalytics).where(TopicAnalytics.topic == topic_name)
    result = await db.execute(query)
    topic_analytics = result.scalar_one_or_none()

    if not topic_analytics:
        raise HTTPException(status_code=404, detail=f"Topic '{topic_name}' not found")

    # Get sample traces for this topic
    traces_query = select(Trace).where(Trace.topic == topic_name).limit(10)
    traces_result = await db.execute(traces_query)
    sample_traces = traces_result.scalars().all()

    # Get failed traces for pattern analysis
    failed_query = select(Trace).where(
        Trace.topic == topic_name,
        Trace.feedback_negative == True
    ).limit(5)
    failed_result = await db.execute(failed_query)
    failed_traces = failed_result.scalars().all()

    return {
        "topic": topic_name,
        "analytics": {
            "question_count": topic_analytics.question_count,
            "success_rate": round(topic_analytics.success_rate, 2),
            "failure_count": topic_analytics.failure_count,
            "handover_rate": round(topic_analytics.handover_rate, 2),
            "avg_confidence": round(topic_analytics.avg_confidence, 2),
            "avg_csat": round(topic_analytics.avg_csat, 2) if topic_analytics.avg_csat else None,
            "trend": round(topic_analytics.trend, 2),
            "priority": topic_analytics.priority,
        },
        "sample_questions": [
            {
                "question": t.input,
                "response": t.output[:200] + "..." if t.output and len(t.output) > 200 else t.output,
                "feedback_negative": t.feedback_negative,
                "timestamp": t.timestamp.isoformat() if t.timestamp else None,
            }
            for t in sample_traces
        ],
        "failure_patterns": [
            {
                "question": t.input,
                "response": t.output[:200] + "..." if t.output and len(t.output) > 200 else t.output,
            }
            for t in failed_traces
        ],
    }


@router.post("/classify")
async def classify_question(
    question: str
):
    """
    Classify a single question into a topic.
    Uses the dynamically discovered topics from BERTopic.
    """
    try:
        from app.services.dynamic_topic_modeler import get_topic_modeler
        topic_modeler = get_topic_modeler()

        if not topic_modeler.is_fitted:
            return {
                "question": question,
                "primary_topic": "Uncategorized",
                "confidence": 0.0,
                "message": "No topics discovered yet. Click 'Discover Topics' first."
            }

        topic, confidence = topic_modeler.classify(question)

        return {
            "question": question,
            "primary_topic": topic,
            "confidence": round(confidence, 3),
        }
    except ImportError:
        return {
            "question": question,
            "primary_topic": "Uncategorized",
            "confidence": 0.0,
            "error": "BERTopic not available"
        }
