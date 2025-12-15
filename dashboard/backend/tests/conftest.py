import os
import sys

# Set environment variables BEFORE importing any app modules
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-test-key"
os.environ["LANGFUSE_SECRET_KEY"] = "sk-test-key"
os.environ["LANGFUSE_HOST"] = "https://test.langfuse.com"
os.environ["COGNEE_LLM_API_KEY"] = "sk-test-openai-key"
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"
os.environ["REDIS_URL"] = "redis://localhost:6379/0"

import pytest
import asyncio
from typing import AsyncGenerator, Generator
from unittest.mock import MagicMock, AsyncMock

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import StaticPool


# Use in-memory SQLite for tests
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create an event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_langfuse_client():
    """Mock LangFuse client."""
    mock = MagicMock()
    mock.async_api = MagicMock()
    mock.async_api.trace = MagicMock()
    mock.async_api.trace.list = AsyncMock(return_value=MagicMock(data=[]))
    mock.async_api.trace.get = AsyncMock(return_value=MagicMock(
        id="test-trace-id",
        session_id="test-session",
        user_id="test-user",
        input="Test question",
        output="Test response",
        name="test-trace",
        tags=[],
        metadata={},
        timestamp=None,
        scores=[],
        observations=[],
        usage=None,
        total_cost=0.0,
        latency=1.5
    ))
    mock.async_api.observations = MagicMock()
    mock.async_api.observations.get_many = AsyncMock(return_value=MagicMock(data=[]))
    mock.async_api.sessions = MagicMock()
    mock.async_api.sessions.list = AsyncMock(return_value=MagicMock(data=[]))
    mock.flush = MagicMock()
    return mock


@pytest.fixture
def mock_cognee():
    """Mock Cognee module."""
    mock = MagicMock()
    mock.add = AsyncMock()
    mock.cognify = AsyncMock()
    mock.memify = AsyncMock()
    mock.search = AsyncMock(return_value=[])
    mock.prune = MagicMock()
    mock.prune.prune_data = AsyncMock()
    mock.prune.prune_system = AsyncMock()
    return mock


@pytest.fixture
def sample_trace_data():
    """Sample trace data for testing."""
    return {
        "trace_id": "trace-123",
        "session_id": "session-456",
        "user_id": "user-789",
        "input": "How do I return a product?",
        "output": "You can return a product within 30 days of purchase.",
        "name": "chat-completion",
        "tags": ["production"],
        "metadata": {"model": "gpt-4"},
        "timestamp": "2024-01-15T10:30:00Z",
        "latency": 1.5,
        "total_cost": 0.02,
        "usage": {
            "prompt_tokens": 50,
            "completion_tokens": 30,
            "total_tokens": 80
        },
        "scores": [
            {
                "score_id": "score-1",
                "name": "thumbs",
                "value": 1,
                "comment": "Helpful!",
                "source": "user",
                "timestamp": "2024-01-15T10:31:00Z"
            }
        ],
        "feedback_negative": False
    }


@pytest.fixture
def sample_gap_data():
    """Sample knowledge gap data for testing."""
    return {
        "question": "How do I return an item bought with a gift card?",
        "topic": "Product Returns",
        "frequency": 15,
        "priority_score": 0.85,
        "priority_level": "high",
        "confidence": 0.2,
        "related_concepts": ["gift card", "return policy", "refund"],
        "suggested_addition": "Add documentation about gift card return procedures",
        "sample_trace_ids": ["trace-1", "trace-2", "trace-3"],
        "resolved": False
    }


@pytest.fixture
def sample_conversations():
    """Sample conversations for testing."""
    return [
        {
            "input": "How do I reset my password?",
            "output": "I understand you need help with your password. You can reset it by clicking 'Forgot Password' on the login page. Let me know if you need any other assistance!",
            "feedback_negative": False
        },
        {
            "input": "Why was my order cancelled?",
            "output": "I apologize for the inconvenience. Your order may have been cancelled due to payment issues or inventory problems. Please contact support for more details.",
            "feedback_negative": True
        },
        {
            "input": "What's your refund policy?",
            "output": "You can request a refund within 30 days.",
            "feedback_negative": False
        }
    ]
