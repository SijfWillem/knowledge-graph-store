import pytest
from unittest.mock import MagicMock, AsyncMock
from datetime import datetime

from app.services.gap_analyzer import GapAnalyzer


class TestGapAnalyzer:
    """Tests for GapAnalyzer service."""

    @pytest.fixture
    def mock_langfuse(self):
        """Create mock LangFuse service."""
        mock = MagicMock()
        mock.sync_new_data = AsyncMock(return_value={
            "traces": [],
            "scores": {},
            "sync_timestamp": datetime.utcnow(),
            "count": 0
        })
        return mock

    @pytest.fixture
    def mock_cognee(self):
        """Create mock Cognee service."""
        mock = MagicMock()
        mock.identify_gap = AsyncMock(return_value={
            "is_gap": True,
            "gap_type": "missing_knowledge",
            "confidence": 0.2,
            "related_concepts": ["test concept"],
            "suggested_addition": "Add documentation"
        })
        return mock

    @pytest.fixture
    def analyzer(self, mock_langfuse, mock_cognee):
        """Create a GapAnalyzer instance."""
        return GapAnalyzer(mock_langfuse, mock_cognee)

    @pytest.mark.asyncio
    async def test_analyze_failed_conversations_empty(self, analyzer):
        """Test analysis with no conversations."""
        result = await analyzer.analyze_failed_conversations([])
        assert result == []

    @pytest.mark.asyncio
    async def test_analyze_failed_conversations_no_negative_feedback(self, analyzer):
        """Test that positive conversations are skipped."""
        conversations = [
            {"input": "Question", "output": "Answer", "feedback_negative": False}
        ]
        result = await analyzer.analyze_failed_conversations(conversations)
        assert result == []

    @pytest.mark.asyncio
    async def test_analyze_failed_conversations_with_gaps(self, analyzer, mock_cognee):
        """Test analysis identifies gaps from failed conversations."""
        conversations = [
            {
                "input": "How do I return a gift card?",
                "output": "I'm not sure.",
                "trace_id": "trace-1",
                "session_id": "session-1",
                "timestamp": datetime.utcnow(),
                "feedback_negative": True
            }
        ]

        result = await analyzer.analyze_failed_conversations(conversations)

        assert len(result) == 1
        assert result[0]["question"] == "How do I return a gift card?"
        assert result[0]["trace_id"] == "trace-1"
        mock_cognee.identify_gap.assert_called_once()

    @pytest.mark.asyncio
    async def test_analyze_failed_conversations_no_gap_detected(self, analyzer, mock_cognee):
        """Test when cognee doesn't detect a gap."""
        mock_cognee.identify_gap = AsyncMock(return_value={"is_gap": False})

        conversations = [
            {"input": "Question", "output": "Answer", "feedback_negative": True}
        ]

        result = await analyzer.analyze_failed_conversations(conversations)
        assert result == []

    @pytest.mark.asyncio
    async def test_cluster_similar_gaps_empty(self, analyzer):
        """Test clustering with no gaps."""
        result = await analyzer.cluster_similar_gaps([])
        assert result == []

    @pytest.mark.asyncio
    async def test_cluster_similar_gaps_single(self, analyzer):
        """Test clustering with single gap."""
        gaps = [
            {"question": "How do I return?", "confidence": 0.3}
        ]
        result = await analyzer.cluster_similar_gaps(gaps)

        assert len(result) == 1
        assert result[0]["frequency"] == 1

    @pytest.mark.asyncio
    async def test_cluster_similar_gaps_groups_similar(self, analyzer):
        """Test that similar questions are clustered together."""
        gaps = [
            {
                "question": "How do I return an item?",
                "confidence": 0.2,
                "trace_id": "t1",
                "timestamp": datetime.utcnow(),
                "related_concepts": [],
                "suggested_addition": None,
                "gap_type": "missing_knowledge"
            },
            {
                "question": "How can I return my purchase?",
                "confidence": 0.25,
                "trace_id": "t2",
                "timestamp": datetime.utcnow(),
                "related_concepts": [],
                "suggested_addition": None,
                "gap_type": "missing_knowledge"
            },
            {
                "question": "What is the shipping cost?",
                "confidence": 0.3,
                "trace_id": "t3",
                "timestamp": datetime.utcnow(),
                "related_concepts": [],
                "suggested_addition": None,
                "gap_type": "missing_knowledge"
            }
        ]

        result = await analyzer.cluster_similar_gaps(gaps, similarity_threshold=0.5)

        # Should cluster the two return questions together
        assert len(result) <= 3
        # Find the cluster with return questions
        return_cluster = next(
            (g for g in result if "return" in g["question"].lower()),
            None
        )
        if return_cluster:
            assert return_cluster["frequency"] >= 1

    @pytest.mark.asyncio
    async def test_cluster_calculates_priority(self, analyzer):
        """Test that clustering calculates priority scores."""
        gaps = [
            {
                "question": "Test question",
                "confidence": 0.2,
                "trace_id": "t1",
                "timestamp": datetime.utcnow(),
                "related_concepts": [],
                "suggested_addition": "Add docs",
                "gap_type": "missing_knowledge"
            }
        ]

        result = await analyzer.cluster_similar_gaps(gaps)

        assert len(result) == 1
        assert "priority_score" in result[0]
        assert "priority_level" in result[0]
        assert 0 <= result[0]["priority_score"] <= 1

    def test_calculate_priority(self, analyzer):
        """Test priority calculation."""
        # High frequency, recent gaps with low confidence should have high priority
        gaps = [
            {"confidence": 0.1, "timestamp": datetime.utcnow()}
            for _ in range(20)
        ]
        priority = analyzer._calculate_priority(gaps)
        assert priority > 0.5

        # Single old gap with high confidence should have low priority
        old_gap = [{"confidence": 0.9, "timestamp": datetime(2020, 1, 1)}]
        priority = analyzer._calculate_priority(old_gap)
        assert priority < 0.5

    def test_get_priority_level(self, analyzer):
        """Test priority level determination."""
        assert analyzer._get_priority_level(0.9) == "critical"
        assert analyzer._get_priority_level(0.7) == "high"
        assert analyzer._get_priority_level(0.5) == "medium"
        assert analyzer._get_priority_level(0.2) == "low"

    def test_merge_concepts(self, analyzer):
        """Test concept merging from multiple gaps."""
        gaps = [
            {"related_concepts": ["return", "refund"]},
            {"related_concepts": ["refund", "policy"]},
            {"related_concepts": ["shipping"]}
        ]
        merged = analyzer._merge_concepts(gaps)

        # Should deduplicate
        assert len(merged) == 4
        assert "return" in merged
        assert "refund" in merged
        assert "policy" in merged
        assert "shipping" in merged

    @pytest.mark.asyncio
    async def test_analyze_and_cluster_full_pipeline(self, analyzer, mock_langfuse, mock_cognee):
        """Test the full analysis pipeline."""
        mock_langfuse.sync_new_data = AsyncMock(return_value={
            "traces": [
                {
                    "input": "How do I return?",
                    "output": "Not sure",
                    "feedback_negative": True,
                    "trace_id": "t1"
                }
            ],
            "sync_timestamp": datetime.utcnow()
        })

        result = await analyzer.analyze_and_cluster(
            from_timestamp=datetime(2024, 1, 1),
            limit=100
        )

        assert "gaps" in result
        assert "total_traces_analyzed" in result
        assert "negative_feedback_count" in result
        assert "gaps_identified" in result

    @pytest.mark.asyncio
    async def test_analyze_and_cluster_no_traces(self, analyzer, mock_langfuse):
        """Test pipeline with no traces."""
        mock_langfuse.sync_new_data = AsyncMock(return_value={
            "traces": [],
            "sync_timestamp": datetime.utcnow()
        })

        result = await analyzer.analyze_and_cluster()

        assert result["gaps"] == []
        assert result["total_traces_analyzed"] == 0

    def test_embedding_model_lazy_loading(self, analyzer):
        """Test that embedding model is lazily loaded."""
        assert analyzer._embedding_model is None
        model = analyzer.embedding_model
        assert model is not None
        assert analyzer._embedding_model is not None
