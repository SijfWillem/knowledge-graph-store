from fastapi import APIRouter, Depends, Query
from typing import Optional, List
from datetime import datetime, timedelta
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.database import get_db, Trace
from app.services.quality_scorer import QualityScorer

router = APIRouter(prefix="/quality", tags=["quality"])

# Initialize scorer as module-level singleton
quality_scorer = QualityScorer()


class ConversationInput(BaseModel):
    """Schema for scoring a conversation"""
    input: str
    output: str


@router.get("/scores")
async def get_quality_scores(
    from_date: Optional[datetime] = Query(
        default=None,
        description="Start date"
    ),
    to_date: Optional[datetime] = Query(
        default=None,
        description="End date"
    ),
    sample_size: int = Query(
        default=100,
        le=500,
        description="Number of conversations to sample for scoring"
    ),
    db: AsyncSession = Depends(get_db)
):
    """
    Quality dimension scores aggregated across conversations.
    Returns average scores for:
    - Empathy
    - Understanding
    - Relevancy
    - Clarity
    - Proactiveness
    - Failure Recovery
    - Overall
    """
    if from_date is None:
        from_date = datetime.utcnow() - timedelta(days=30)
    if to_date is None:
        to_date = datetime.utcnow()

    # Get sample of conversations
    query = select(Trace).where(
        Trace.timestamp >= from_date,
        Trace.timestamp <= to_date,
        Trace.input.isnot(None),
        Trace.output.isnot(None)
    ).limit(sample_size)

    result = await db.execute(query)
    traces = result.scalars().all()

    if not traces:
        return {
            "scores": quality_scorer._empty_scores(),
            "sample_size": 0,
            "period": {
                "from": from_date.isoformat(),
                "to": to_date.isoformat(),
            }
        }

    # Convert to conversation format
    conversations = [
        {"input": t.input, "output": t.output}
        for t in traces
    ]

    # Get average scores
    average_scores = quality_scorer.get_dimension_averages(conversations)

    return {
        "scores": average_scores,
        "sample_size": len(conversations),
        "period": {
            "from": from_date.isoformat(),
            "to": to_date.isoformat(),
        }
    }


@router.get("/scores/by-topic")
async def get_quality_scores_by_topic(
    from_date: Optional[datetime] = Query(default=None),
    to_date: Optional[datetime] = Query(default=None),
    sample_per_topic: int = Query(default=50, le=100),
    db: AsyncSession = Depends(get_db)
):
    """
    Quality scores broken down by topic.
    """
    if from_date is None:
        from_date = datetime.utcnow() - timedelta(days=30)
    if to_date is None:
        to_date = datetime.utcnow()

    # Get unique topics
    topics_query = select(Trace.topic).distinct().where(
        Trace.topic.isnot(None),
        Trace.timestamp >= from_date,
        Trace.timestamp <= to_date
    )
    topics_result = await db.execute(topics_query)
    topics = [row[0] for row in topics_result.fetchall()]

    topic_scores = {}

    for topic in topics:
        # Get sample conversations for this topic
        query = select(Trace).where(
            Trace.topic == topic,
            Trace.timestamp >= from_date,
            Trace.timestamp <= to_date,
            Trace.input.isnot(None),
            Trace.output.isnot(None)
        ).limit(sample_per_topic)

        result = await db.execute(query)
        traces = result.scalars().all()

        if traces:
            conversations = [
                {"input": t.input, "output": t.output}
                for t in traces
            ]
            scores = quality_scorer.get_dimension_averages(conversations)
            topic_scores[topic] = {
                "scores": scores,
                "sample_size": len(conversations),
            }

    return {
        "by_topic": topic_scores,
        "period": {
            "from": from_date.isoformat(),
            "to": to_date.isoformat(),
        }
    }


@router.post("/score")
async def score_conversation(
    conversation: ConversationInput
):
    """
    Score a single conversation on quality dimensions.
    Useful for real-time scoring or testing.
    """
    scores = quality_scorer.score_conversation({
        "input": conversation.input,
        "output": conversation.output
    })

    return {
        "conversation": {
            "input": conversation.input[:100] + "..." if len(conversation.input) > 100 else conversation.input,
            "output": conversation.output[:100] + "..." if len(conversation.output) > 100 else conversation.output,
        },
        "scores": scores,
    }


@router.post("/score/batch")
async def score_conversations_batch(
    conversations: List[ConversationInput]
):
    """
    Score multiple conversations at once.
    """
    if len(conversations) > 100:
        conversations = conversations[:100]

    conv_dicts = [
        {"input": c.input, "output": c.output}
        for c in conversations
    ]

    individual_scores = quality_scorer.score_batch(conv_dicts)
    averages = quality_scorer.get_dimension_averages(conv_dicts)

    return {
        "individual_scores": individual_scores,
        "averages": averages,
        "count": len(conversations),
    }


@router.get("/dimensions")
async def get_quality_dimensions():
    """
    Get information about quality dimensions.
    """
    return {
        "dimensions": [
            {
                "name": "empathy",
                "description": "How well the response acknowledges user feelings and shows understanding",
                "weight": 0.15,
            },
            {
                "name": "understanding",
                "description": "How well the response addresses the actual question asked",
                "weight": 0.25,
            },
            {
                "name": "relevancy",
                "description": "How relevant and on-topic the response is",
                "weight": 0.25,
            },
            {
                "name": "clarity",
                "description": "How clear, well-structured, and easy to understand the response is",
                "weight": 0.15,
            },
            {
                "name": "proactiveness",
                "description": "Whether the response anticipates follow-up needs",
                "weight": 0.10,
            },
            {
                "name": "failure_recovery",
                "description": "How well errors or limitations are handled with alternatives",
                "weight": 0.10,
            },
        ]
    }
