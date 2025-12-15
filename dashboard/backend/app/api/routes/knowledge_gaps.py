from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update

from app.models.database import get_db, KnowledgeGap

router = APIRouter(prefix="/knowledge-gaps", tags=["knowledge-gaps"])


class GapResolution(BaseModel):
    """Schema for gap resolution"""
    resolution_notes: str
    knowledge_added: Optional[str] = None


class GapCreate(BaseModel):
    """Schema for creating a new gap"""
    question: str
    topic: Optional[str] = None
    suggested_addition: Optional[str] = None
    sample_trace_ids: Optional[List[str]] = None


@router.get("/")
async def get_gaps(
    priority: Optional[str] = Query(
        default=None,
        description="Filter by priority: critical, high, medium, low"
    ),
    resolved: Optional[bool] = Query(
        default=False,
        description="Include resolved gaps"
    ),
    topic: Optional[str] = Query(
        default=None,
        description="Filter by topic"
    ),
    limit: int = Query(
        default=20,
        le=100,
        description="Maximum number of gaps to return"
    ),
    offset: int = Query(
        default=0,
        description="Offset for pagination"
    ),
    db: AsyncSession = Depends(get_db)
):
    """
    Prioritized list of knowledge gaps.
    Returns gaps sorted by priority score.
    """
    query = select(KnowledgeGap)

    # Apply filters
    if priority:
        query = query.where(KnowledgeGap.priority_level == priority)
    if not resolved:
        query = query.where(KnowledgeGap.resolved == False)
    if topic:
        query = query.where(KnowledgeGap.topic == topic)

    # Order by priority score descending
    query = query.order_by(KnowledgeGap.priority_score.desc())

    # Apply pagination
    query = query.offset(offset).limit(limit)

    result = await db.execute(query)
    gaps = result.scalars().all()

    # Get total count for pagination
    count_query = select(KnowledgeGap)
    if priority:
        count_query = count_query.where(KnowledgeGap.priority_level == priority)
    if not resolved:
        count_query = count_query.where(KnowledgeGap.resolved == False)
    if topic:
        count_query = count_query.where(KnowledgeGap.topic == topic)

    from sqlalchemy import func
    count_result = await db.execute(select(func.count()).select_from(count_query.subquery()))
    total_count = count_result.scalar()

    return {
        "gaps": [
            {
                "id": g.id,
                "question": g.question,
                "topic": g.topic,
                "frequency": g.frequency,
                "priority_score": round(g.priority_score, 3),
                "priority_level": g.priority_level,
                "confidence": round(g.confidence, 3) if g.confidence else None,
                "related_concepts": g.related_concepts or [],
                "suggested_addition": g.suggested_addition,
                "sample_trace_ids": g.sample_trace_ids or [],
                "resolved": g.resolved,
                "created_at": g.created_at.isoformat() if g.created_at else None,
                "updated_at": g.updated_at.isoformat() if g.updated_at else None,
            }
            for g in gaps
        ],
        "pagination": {
            "total": total_count,
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < total_count,
        }
    }


@router.get("/summary")
async def get_gaps_summary(
    db: AsyncSession = Depends(get_db)
):
    """
    Summary statistics for knowledge gaps.
    """
    from sqlalchemy import func

    # Count by priority level
    priority_query = select(
        KnowledgeGap.priority_level,
        func.count(KnowledgeGap.id)
    ).where(
        KnowledgeGap.resolved == False
    ).group_by(KnowledgeGap.priority_level)

    priority_result = await db.execute(priority_query)
    priority_counts = {row[0]: row[1] for row in priority_result.fetchall()}

    # Total unresolved
    total_unresolved = sum(priority_counts.values())

    # Total resolved
    resolved_query = select(func.count(KnowledgeGap.id)).where(KnowledgeGap.resolved == True)
    resolved_result = await db.execute(resolved_query)
    total_resolved = resolved_result.scalar() or 0

    # Recent gaps (last 7 days)
    recent_query = select(func.count(KnowledgeGap.id)).where(
        KnowledgeGap.created_at >= datetime.utcnow().replace(hour=0, minute=0, second=0) - __import__('datetime').timedelta(days=7)
    )
    recent_result = await db.execute(recent_query)
    recent_count = recent_result.scalar() or 0

    return {
        "total_unresolved": total_unresolved,
        "total_resolved": total_resolved,
        "by_priority": {
            "critical": priority_counts.get("critical", 0),
            "high": priority_counts.get("high", 0),
            "medium": priority_counts.get("medium", 0),
            "low": priority_counts.get("low", 0),
        },
        "recent_7_days": recent_count,
    }


@router.get("/{gap_id}")
async def get_gap_detail(
    gap_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Gap detail with sample questions and full information.
    """
    query = select(KnowledgeGap).where(KnowledgeGap.id == gap_id)
    result = await db.execute(query)
    gap = result.scalar_one_or_none()

    if not gap:
        raise HTTPException(status_code=404, detail="Knowledge gap not found")

    return {
        "id": gap.id,
        "question": gap.question,
        "topic": gap.topic,
        "frequency": gap.frequency,
        "priority_score": round(gap.priority_score, 3),
        "priority_level": gap.priority_level,
        "confidence": round(gap.confidence, 3) if gap.confidence else None,
        "related_concepts": gap.related_concepts or [],
        "suggested_addition": gap.suggested_addition,
        "sample_trace_ids": gap.sample_trace_ids or [],
        "resolved": gap.resolved,
        "resolution_notes": gap.resolution_notes,
        "resolved_at": gap.resolved_at.isoformat() if gap.resolved_at else None,
        "created_at": gap.created_at.isoformat() if gap.created_at else None,
        "updated_at": gap.updated_at.isoformat() if gap.updated_at else None,
    }


@router.post("/{gap_id}/resolve")
async def resolve_gap(
    gap_id: str,
    resolution: GapResolution,
    db: AsyncSession = Depends(get_db)
):
    """
    Mark gap as resolved with resolution notes.
    """
    query = select(KnowledgeGap).where(KnowledgeGap.id == gap_id)
    result = await db.execute(query)
    gap = result.scalar_one_or_none()

    if not gap:
        raise HTTPException(status_code=404, detail="Knowledge gap not found")

    gap.resolved = True
    gap.resolution_notes = resolution.resolution_notes
    gap.resolved_at = datetime.utcnow()
    gap.updated_at = datetime.utcnow()

    await db.commit()

    return {
        "id": gap.id,
        "resolved": True,
        "resolution_notes": gap.resolution_notes,
        "resolved_at": gap.resolved_at.isoformat(),
    }


@router.post("/{gap_id}/reopen")
async def reopen_gap(
    gap_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Reopen a previously resolved gap.
    """
    query = select(KnowledgeGap).where(KnowledgeGap.id == gap_id)
    result = await db.execute(query)
    gap = result.scalar_one_or_none()

    if not gap:
        raise HTTPException(status_code=404, detail="Knowledge gap not found")

    gap.resolved = False
    gap.resolved_at = None
    gap.updated_at = datetime.utcnow()

    await db.commit()

    return {
        "id": gap.id,
        "resolved": False,
    }


@router.post("/")
async def create_gap(
    gap_data: GapCreate,
    db: AsyncSession = Depends(get_db)
):
    """
    Manually create a knowledge gap.
    """
    new_gap = KnowledgeGap(
        question=gap_data.question,
        topic=gap_data.topic,
        suggested_addition=gap_data.suggested_addition,
        sample_trace_ids=gap_data.sample_trace_ids or [],
        priority_level="medium",
        priority_score=0.5,
        frequency=1,
    )

    db.add(new_gap)
    await db.commit()
    await db.refresh(new_gap)

    return {
        "id": new_gap.id,
        "question": new_gap.question,
        "created_at": new_gap.created_at.isoformat(),
    }
