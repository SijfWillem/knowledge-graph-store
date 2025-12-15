from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import Column, String, Float, DateTime, Integer, JSON, Boolean, Text
from datetime import datetime
import uuid

from app.config import settings


class Base(DeclarativeBase):
    pass


class Trace(Base):
    """Stored trace from LangFuse"""

    __tablename__ = "traces"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    trace_id = Column(String, unique=True, index=True, nullable=False)
    session_id = Column(String, index=True)
    user_id = Column(String, index=True)
    input = Column(Text)
    output = Column(Text)
    name = Column(String)
    tags = Column(JSON, default=list)
    trace_metadata = Column("metadata", JSON, default=dict)
    timestamp = Column(DateTime, index=True)
    latency = Column(Float)
    total_cost = Column(Float)
    prompt_tokens = Column(Integer, default=0)
    completion_tokens = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    feedback_negative = Column(Boolean, default=False)
    has_missing_knowledge = Column(Boolean, default=False, index=True)  # AI admitted it doesn't know
    topic = Column(String, index=True)
    topic_confidence = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Score(Base):
    """Feedback scores from LangFuse"""

    __tablename__ = "scores"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    score_id = Column(String, unique=True, index=True)
    trace_id = Column(String, index=True, nullable=False)
    name = Column(String, index=True)
    value = Column(Float)
    comment = Column(Text)
    source = Column(String)
    timestamp = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)


class KnowledgeGap(Base):
    """Identified knowledge gaps"""

    __tablename__ = "knowledge_gaps"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    question = Column(Text, nullable=False)
    topic = Column(String, index=True)
    frequency = Column(Integer, default=1)
    priority_score = Column(Float, default=0.0)
    priority_level = Column(String, default="medium")  # critical, high, medium, low
    confidence = Column(Float)
    related_concepts = Column(JSON, default=list)
    suggested_addition = Column(Text)
    sample_trace_ids = Column(JSON, default=list)
    resolved = Column(Boolean, default=False)
    resolution_notes = Column(Text)
    resolved_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class TopicAnalytics(Base):
    """Aggregated analytics per topic"""

    __tablename__ = "topic_analytics"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    topic = Column(String, unique=True, index=True, nullable=False)
    question_count = Column(Integer, default=0)
    success_count = Column(Integer, default=0)
    failure_count = Column(Integer, default=0)
    success_rate = Column(Float, default=0.0)
    handover_count = Column(Integer, default=0)
    handover_rate = Column(Float, default=0.0)
    avg_confidence = Column(Float, default=0.0)
    avg_csat = Column(Float)
    priority = Column(String, default="medium")
    trend = Column(Float, default=0.0)  # % change vs previous period
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class SyncState(Base):
    """Track sync state for incremental updates"""

    __tablename__ = "sync_state"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    source = Column(String, unique=True, nullable=False)  # e.g., "langfuse"
    last_sync_timestamp = Column(DateTime)
    last_sync_count = Column(Integer, default=0)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# Database engine and session
engine = create_async_engine(settings.database_url, echo=False)
async_session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db():
    """Initialize database tables"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncSession:
    """Dependency for getting database session"""
    async with async_session_maker() as session:
        try:
            yield session
        finally:
            await session.close()
