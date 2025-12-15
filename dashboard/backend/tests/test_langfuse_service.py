import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime

from app.services.langfuse_service import LangFuseService


class TestLangFuseService:
    """Tests for LangFuseService."""

    @pytest.fixture
    def service(self, mock_langfuse_client):
        """Create a LangFuse service with mocked client."""
        with patch('app.services.langfuse_service.Langfuse', return_value=mock_langfuse_client):
            svc = LangFuseService(
                public_key="pk-test",
                secret_key="sk-test",
                host="https://test.langfuse.com"
            )
            return svc

    def test_init_creates_client(self):
        """Test that initialization creates a Langfuse client."""
        with patch('app.services.langfuse_service.Langfuse') as mock_langfuse:
            service = LangFuseService(
                public_key="pk-test",
                secret_key="sk-test",
                host="https://test.langfuse.com"
            )
            mock_langfuse.assert_called_once_with(
                public_key="pk-test",
                secret_key="sk-test",
                host="https://test.langfuse.com"
            )

    @pytest.mark.asyncio
    async def test_fetch_traces_returns_list(self, service, mock_langfuse_client):
        """Test that fetch_traces returns a list."""
        mock_trace = MagicMock()
        mock_trace.id = "trace-1"
        mock_trace.session_id = "session-1"
        mock_trace.user_id = "user-1"
        mock_trace.input = "Test question"
        mock_trace.output = "Test response"
        mock_trace.name = "test"
        mock_trace.tags = []
        mock_trace.metadata = {}
        mock_trace.timestamp = datetime.utcnow()
        mock_trace.total_cost = 0.01
        mock_trace.usage = MagicMock(input=10, output=20, total=30)
        mock_trace.latency = 1.5

        mock_langfuse_client.async_api.trace.list = AsyncMock(
            return_value=MagicMock(data=[mock_trace])
        )

        traces = await service.fetch_traces(limit=10)

        assert isinstance(traces, list)
        assert len(traces) == 1
        assert traces[0]["trace_id"] == "trace-1"
        assert traces[0]["input"] == "Test question"

    @pytest.mark.asyncio
    async def test_fetch_traces_empty(self, service, mock_langfuse_client):
        """Test fetch_traces with no traces."""
        mock_langfuse_client.async_api.trace.list = AsyncMock(
            return_value=MagicMock(data=[])
        )

        traces = await service.fetch_traces()
        assert traces == []

    @pytest.mark.asyncio
    async def test_fetch_trace_by_id(self, service, mock_langfuse_client):
        """Test fetching a single trace by ID."""
        trace = await service.fetch_trace_by_id("test-trace-id")

        assert trace is not None
        assert trace["trace_id"] == "test-trace-id"
        mock_langfuse_client.async_api.trace.get.assert_called_once_with("test-trace-id")

    @pytest.mark.asyncio
    async def test_fetch_trace_by_id_not_found(self, service, mock_langfuse_client):
        """Test fetching a non-existent trace."""
        mock_langfuse_client.async_api.trace.get = AsyncMock(
            side_effect=Exception("Not found")
        )

        trace = await service.fetch_trace_by_id("non-existent")
        assert trace is None

    @pytest.mark.asyncio
    async def test_fetch_scores_for_traces(self, service, mock_langfuse_client):
        """Test fetching scores for multiple traces."""
        mock_score = MagicMock()
        mock_score.id = "score-1"
        mock_score.name = "thumbs"
        mock_score.value = 1
        mock_score.comment = "Good"
        mock_score.source = "user"
        mock_score.timestamp = datetime.utcnow()

        mock_trace = MagicMock()
        mock_trace.scores = [mock_score]

        mock_langfuse_client.async_api.trace.get = AsyncMock(return_value=mock_trace)

        scores = await service.fetch_scores_for_traces(["trace-1", "trace-2"])

        assert "trace-1" in scores
        assert "trace-2" in scores
        assert len(scores["trace-1"]) == 1
        assert scores["trace-1"][0]["name"] == "thumbs"

    @pytest.mark.asyncio
    async def test_sync_new_data(self, service, mock_langfuse_client):
        """Test incremental sync of new data."""
        mock_trace = MagicMock()
        mock_trace.id = "trace-new"
        mock_trace.session_id = None
        mock_trace.user_id = None
        mock_trace.input = "New question"
        mock_trace.output = "New response"
        mock_trace.name = None
        mock_trace.tags = []
        mock_trace.metadata = {}
        mock_trace.timestamp = datetime.utcnow()
        mock_trace.total_cost = 0
        mock_trace.usage = None
        mock_trace.latency = None
        mock_trace.scores = []

        mock_langfuse_client.async_api.trace.list = AsyncMock(
            return_value=MagicMock(data=[mock_trace])
        )
        mock_langfuse_client.async_api.trace.get = AsyncMock(
            return_value=mock_trace
        )

        result = await service.sync_new_data(
            last_sync_timestamp=datetime(2024, 1, 1),
            batch_size=10
        )

        assert "traces" in result
        assert "sync_timestamp" in result
        assert "count" in result
        assert result["count"] == 1

    @pytest.mark.asyncio
    async def test_sync_new_data_empty(self, service, mock_langfuse_client):
        """Test sync when no new data."""
        mock_langfuse_client.async_api.trace.list = AsyncMock(
            return_value=MagicMock(data=[])
        )

        result = await service.sync_new_data(
            last_sync_timestamp=datetime(2024, 1, 1)
        )

        assert result["traces"] == []
        assert result["count"] == 0

    def test_has_negative_feedback_thumbs_down(self, service):
        """Test detection of negative feedback from thumbs down."""
        scores = [
            {"name": "thumbs", "value": 0},
        ]
        assert service._has_negative_feedback(scores) is True

    def test_has_negative_feedback_low_csat(self, service):
        """Test detection of negative feedback from low CSAT."""
        scores = [
            {"name": "csat", "value": 2},  # Below 3
        ]
        assert service._has_negative_feedback(scores) is True

    def test_has_negative_feedback_positive(self, service):
        """Test positive feedback detection."""
        scores = [
            {"name": "thumbs", "value": 1},
            {"name": "csat", "value": 4},
        ]
        assert service._has_negative_feedback(scores) is False

    def test_has_negative_feedback_empty(self, service):
        """Test with empty scores."""
        assert service._has_negative_feedback([]) is False

    @pytest.mark.asyncio
    async def test_fetch_observations(self, service, mock_langfuse_client):
        """Test fetching observations for a trace."""
        mock_obs = MagicMock()
        mock_obs.id = "obs-1"
        mock_obs.trace_id = "trace-1"
        mock_obs.type = "GENERATION"
        mock_obs.name = "completion"
        mock_obs.input = "input"
        mock_obs.output = "output"
        mock_obs.model = "gpt-4"
        mock_obs.model_parameters = {}
        mock_obs.usage = MagicMock(input=10, output=20, total=30)
        mock_obs.start_time = datetime.utcnow()
        mock_obs.end_time = datetime.utcnow()
        mock_obs.level = "DEFAULT"
        mock_obs.status_message = None

        mock_langfuse_client.async_api.observations.get_many = AsyncMock(
            return_value=MagicMock(data=[mock_obs])
        )

        observations = await service.fetch_observations("trace-1")

        assert len(observations) == 1
        assert observations[0]["observation_id"] == "obs-1"
        assert observations[0]["type"] == "GENERATION"

    def test_shutdown_flushes_client(self, service, mock_langfuse_client):
        """Test that shutdown flushes the client."""
        service.shutdown()
        mock_langfuse_client.flush.assert_called_once()
