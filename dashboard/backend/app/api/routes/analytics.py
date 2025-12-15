from fastapi import APIRouter, Query, Depends
from datetime import datetime, timedelta
from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
import re
import ast

from app.models.database import get_db, Trace
from app.services.analytics_service import AnalyticsService

router = APIRouter(prefix="/analytics", tags=["analytics"])

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

    # If it's a string that looks like a dict, parse it
    if isinstance(input_data, str):
        try:
            # Try to parse as Python dict literal
            parsed = ast.literal_eval(input_data)
            if isinstance(parsed, dict):
                return parsed.get('query', str(input_data))
        except (ValueError, SyntaxError):
            pass
        return input_data

    # If it's already a dict
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
                # Look for common answer fields
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


@router.get("/overview")
async def get_overview(
    from_date: Optional[datetime] = Query(
        default=None,
        description="Start date for analytics period"
    ),
    to_date: Optional[datetime] = Query(
        default=None,
        description="End date for analytics period"
    ),
    db: AsyncSession = Depends(get_db)
):
    """
    Dashboard overview metrics including:
    - Total conversations
    - Success rate
    - Average CSAT
    - Critical gaps count
    - Handover rate
    - Period-over-period trends
    """
    if from_date is None:
        from_date = datetime.utcnow() - timedelta(days=30)
    if to_date is None:
        to_date = datetime.utcnow()

    analytics_service = AnalyticsService(db)
    return await analytics_service.get_overview(from_date, to_date)


@router.get("/trends")
async def get_trends(
    from_date: Optional[datetime] = Query(
        default=None,
        description="Start date for trends"
    ),
    to_date: Optional[datetime] = Query(
        default=None,
        description="End date for trends"
    ),
    granularity: str = Query(
        default="day",
        description="Time granularity: hour, day, or week"
    ),
    db: AsyncSession = Depends(get_db)
):
    """
    Time-series trend data for charts.
    Returns conversation counts and success rates over time.
    """
    if from_date is None:
        from_date = datetime.utcnow() - timedelta(days=30)
    if to_date is None:
        to_date = datetime.utcnow()

    analytics_service = AnalyticsService(db)
    return await analytics_service.get_trends(from_date, to_date, granularity)


@router.get("/satisfaction")
async def get_satisfaction(
    from_date: Optional[datetime] = Query(
        default=None,
        description="Start date"
    ),
    to_date: Optional[datetime] = Query(
        default=None,
        description="End date"
    ),
    db: AsyncSession = Depends(get_db)
):
    """
    CSAT and feedback metrics including:
    - Thumbs up/down counts and ratio
    - CSAT score distribution (1-5)
    """
    if from_date is None:
        from_date = datetime.utcnow() - timedelta(days=30)
    if to_date is None:
        to_date = datetime.utcnow()

    analytics_service = AnalyticsService(db)
    return await analytics_service.get_satisfaction_data(from_date, to_date)


@router.get("/conversations")
async def get_conversations(
    limit: int = Query(default=50, description="Maximum conversations to return"),
    offset: int = Query(default=0, description="Offset for pagination"),
    only_missing_knowledge: bool = Query(default=False, description="Only show conversations with missing knowledge"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get all conversations with questions and answers.
    Identifies which ones have missing knowledge based on the response content.
    """
    query = select(Trace).order_by(desc(Trace.timestamp)).offset(offset).limit(limit)
    result = await db.execute(query)
    traces = result.scalars().all()

    conversations = []
    for trace in traces:
        # Extract clean question and answer
        question = extract_query_from_input(trace.input)
        answer = extract_answer_from_output(trace.output)

        # Check if this indicates missing knowledge
        has_missing_knowledge = is_missing_knowledge_response(answer)

        # Skip if we only want missing knowledge and this isn't one
        if only_missing_knowledge and not has_missing_knowledge:
            continue

        conversations.append({
            "id": trace.id,
            "trace_id": trace.trace_id,
            "timestamp": trace.timestamp.isoformat() if trace.timestamp else None,
            "name": trace.name,
            "question": question,
            "answer": answer,
            "has_missing_knowledge": has_missing_knowledge,
            "topic": trace.topic,
            "topic_confidence": trace.topic_confidence,
            "feedback_negative": trace.feedback_negative,
            "latency": trace.latency,
        })

    return {
        "conversations": conversations,
        "total": len(conversations),
        "offset": offset,
        "limit": limit,
    }


@router.get("/missing-knowledge")
async def get_missing_knowledge_conversations(
    limit: int = Query(default=50, description="Maximum to return"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get conversations where the AI indicated it doesn't know the answer.
    These are automatically detected knowledge gaps.
    """
    query = select(Trace).order_by(desc(Trace.timestamp)).limit(200)  # Get more to filter
    result = await db.execute(query)
    traces = result.scalars().all()

    missing_knowledge = []
    for trace in traces:
        question = extract_query_from_input(trace.input)
        answer = extract_answer_from_output(trace.output)

        if is_missing_knowledge_response(answer):
            missing_knowledge.append({
                "id": trace.id,
                "trace_id": trace.trace_id,
                "timestamp": trace.timestamp.isoformat() if trace.timestamp else None,
                "question": question,
                "answer": answer,
                "topic": trace.topic,
                "name": trace.name,
            })

            if len(missing_knowledge) >= limit:
                break

    return {
        "missing_knowledge": missing_knowledge,
        "total": len(missing_knowledge),
    }
