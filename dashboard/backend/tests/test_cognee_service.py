import pytest
from unittest.mock import patch, AsyncMock, MagicMock
import os

from app.services.cognee_service import CogneeService


class TestCogneeService:
    """Tests for CogneeService."""

    @pytest.fixture
    def service(self):
        """Create a Cognee service instance."""
        return CogneeService(llm_api_key="test-api-key")

    @pytest.fixture
    def mock_cognee(self):
        """Create a mock cognee module."""
        mock = MagicMock()
        mock.prune = MagicMock()
        mock.prune.prune_data = AsyncMock()
        mock.prune.prune_system = AsyncMock()
        mock.add = AsyncMock()
        mock.cognify = AsyncMock()
        mock.memify = AsyncMock()
        mock.search = AsyncMock(return_value=[])
        return mock

    @pytest.mark.asyncio
    async def test_initialize_sets_api_key(self, service, mock_cognee):
        """Test that initialize sets the LLM API key."""
        with patch('app.services.cognee_service.COGNEE_AVAILABLE', True):
            with patch('app.services.cognee_service.cognee', mock_cognee):
                service._initialized = False
                await service.initialize()

                assert os.environ.get("LLM_API_KEY") == "test-api-key"
                assert service._initialized is True

    @pytest.mark.asyncio
    async def test_initialize_only_once(self, service, mock_cognee):
        """Test that initialize only runs once."""
        with patch('app.services.cognee_service.COGNEE_AVAILABLE', True):
            with patch('app.services.cognee_service.cognee', mock_cognee):
                service._initialized = False
                await service.initialize()
                await service.initialize()  # Second call

                # prune_data should only be called once
                mock_cognee.prune.prune_data.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_skips_when_cognee_unavailable(self, service):
        """Test that initialize skips when cognee is not available."""
        with patch('app.services.cognee_service.COGNEE_AVAILABLE', False):
            service._initialized = False
            await service.initialize()
            assert service._initialized is True

    @pytest.mark.asyncio
    async def test_ingest_knowledge_base(self, service, mock_cognee):
        """Test ingesting knowledge base documents."""
        with patch('app.services.cognee_service.COGNEE_AVAILABLE', True):
            with patch('app.services.cognee_service.cognee', mock_cognee):
                service._initialized = False
                documents = [
                    "Return policy: 30 days for full refund",
                    "Shipping takes 3-5 business days"
                ]

                result = await service.ingest_knowledge_base(documents)

                assert result["success"] is True
                assert result["ingested_count"] == 2
                assert mock_cognee.add.call_count == 2
                mock_cognee.cognify.assert_called_once()

    @pytest.mark.asyncio
    async def test_ingest_knowledge_base_with_errors(self, service, mock_cognee):
        """Test handling errors during ingestion."""
        mock_cognee.add = AsyncMock(side_effect=[None, Exception("Failed")])

        with patch('app.services.cognee_service.COGNEE_AVAILABLE', True):
            with patch('app.services.cognee_service.cognee', mock_cognee):
                service._initialized = False
                documents = ["Doc 1", "Doc 2"]
                result = await service.ingest_knowledge_base(documents)

                assert result["ingested_count"] == 1
                assert len(result["errors"]) == 1

    @pytest.mark.asyncio
    async def test_ingest_simulated_when_unavailable(self, service):
        """Test that ingestion is simulated when cognee is unavailable."""
        with patch('app.services.cognee_service.COGNEE_AVAILABLE', False):
            service._initialized = False
            documents = ["Doc 1", "Doc 2"]
            result = await service.ingest_knowledge_base(documents)

            assert result["success"] is True
            assert result["ingested_count"] == 2

    @pytest.mark.asyncio
    async def test_check_knowledge_exists_found(self, service, mock_cognee):
        """Test checking when knowledge exists."""
        # Include words from the question in the results to boost confidence
        mock_cognee.search = AsyncMock(return_value=[
            "What is the return policy? Return policy allows 30 days for refunds",
            "Return policy for is the return policy full refund available"
        ])

        with patch('app.services.cognee_service.COGNEE_AVAILABLE', True):
            with patch('app.services.cognee_service.cognee', mock_cognee):
                service._initialized = False
                result = await service.check_knowledge_exists(
                    "What is the return policy?",
                    confidence_threshold=0.3
                )

                assert result["found"] is True
                assert result["confidence"] > 0
                assert len(result["results"]) > 0

    @pytest.mark.asyncio
    async def test_check_knowledge_exists_not_found(self, service, mock_cognee):
        """Test checking when knowledge doesn't exist."""
        mock_cognee.search = AsyncMock(return_value=[])

        with patch('app.services.cognee_service.COGNEE_AVAILABLE', True):
            with patch('app.services.cognee_service.cognee', mock_cognee):
                service._initialized = False
                result = await service.check_knowledge_exists("Unknown topic question")

                assert result["found"] is False
                assert result["confidence"] == 0.0
                assert result["suggested_addition"] is not None

    @pytest.mark.asyncio
    async def test_check_knowledge_simulated_when_unavailable(self, service):
        """Test that check returns not found when cognee is unavailable."""
        with patch('app.services.cognee_service.COGNEE_AVAILABLE', False):
            service._initialized = False
            result = await service.check_knowledge_exists("Any question")

            assert result["found"] is False
            assert result["confidence"] == 0.0

    @pytest.mark.asyncio
    async def test_identify_gap_negative_feedback(self, service, mock_cognee):
        """Test gap identification with negative feedback."""
        mock_cognee.search = AsyncMock(return_value=[])

        with patch('app.services.cognee_service.COGNEE_AVAILABLE', True):
            with patch('app.services.cognee_service.cognee', mock_cognee):
                service._initialized = False
                result = await service.identify_gap(
                    question="How do I return a gift card purchase?",
                    ai_response="I'm not sure about that.",
                    feedback_negative=True
                )

                assert result is not None
                assert result["is_gap"] is True
                assert result["gap_type"] == "missing_knowledge"
                assert result["question"] == "How do I return a gift card purchase?"

    @pytest.mark.asyncio
    async def test_identify_gap_no_negative_feedback(self, service):
        """Test that no gap is identified without negative feedback."""
        result = await service.identify_gap(
            question="Test question",
            ai_response="Good response",
            feedback_negative=False
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_identify_gap_knowledge_exists(self, service, mock_cognee):
        """Test gap identification when knowledge exists."""
        # Return results that indicate high confidence
        mock_cognee.search = AsyncMock(return_value=[
            "return policy returns refund 30 days"
        ] * 5)

        with patch('app.services.cognee_service.COGNEE_AVAILABLE', True):
            with patch('app.services.cognee_service.cognee', mock_cognee):
                service._initialized = False
                result = await service.identify_gap(
                    question="return policy",
                    ai_response="Bad response",
                    feedback_negative=True
                )

                # When knowledge exists, it's a response quality issue, not a gap
                assert result is not None
                assert result["gap_type"] == "response_quality"

    @pytest.mark.asyncio
    async def test_add_knowledge(self, service, mock_cognee):
        """Test adding new knowledge."""
        with patch('app.services.cognee_service.COGNEE_AVAILABLE', True):
            with patch('app.services.cognee_service.cognee', mock_cognee):
                service._initialized = False
                result = await service.add_knowledge(
                    "Gift card returns: Contact support within 90 days"
                )

                assert result["success"] is True
                mock_cognee.add.assert_called_once()
                mock_cognee.cognify.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_knowledge_error(self, service, mock_cognee):
        """Test error handling when adding knowledge."""
        mock_cognee.add = AsyncMock(side_effect=Exception("Failed to add"))

        with patch('app.services.cognee_service.COGNEE_AVAILABLE', True):
            with patch('app.services.cognee_service.cognee', mock_cognee):
                service._initialized = False
                result = await service.add_knowledge("Test content")

                assert result["success"] is False
                assert "error" in result

    @pytest.mark.asyncio
    async def test_add_knowledge_simulated_when_unavailable(self, service):
        """Test that add knowledge succeeds when cognee is unavailable."""
        with patch('app.services.cognee_service.COGNEE_AVAILABLE', False):
            service._initialized = False
            result = await service.add_knowledge("Test content")

            assert result["success"] is True

    def test_calculate_confidence_empty_results(self, service):
        """Test confidence calculation with empty results."""
        confidence = service._calculate_confidence([], "test question")
        assert confidence == 0.0

    def test_calculate_confidence_with_matches(self, service):
        """Test confidence calculation with matching results."""
        results = ["return policy allows 30 days", "refund available"]
        confidence = service._calculate_confidence(results, "return policy refund")
        assert 0 < confidence <= 1.0

    def test_extract_concepts(self, service):
        """Test concept extraction from results."""
        results = ["return policy", "shipping information", "payment methods"]
        concepts = service._extract_concepts(results)
        assert isinstance(concepts, list)
        assert len(concepts) <= 10

    def test_serialize_results(self, service):
        """Test result serialization."""
        results = ["result 1", "result 2"]
        serialized = service._serialize_results(results)
        assert len(serialized) == 2
        assert all("content" in r and "type" in r for r in serialized)

    def test_generate_suggestion(self, service):
        """Test suggestion generation."""
        # With no results
        suggestion = service._generate_suggestion("How to do X?", [])
        assert "No relevant information found" in suggestion

        # With some results
        suggestion = service._generate_suggestion("How to do X?", ["partial info"])
        assert "Consider adding" in suggestion
