import pytest
from app.services.quality_scorer import QualityScorer


class TestQualityScorer:
    """Tests for QualityScorer service."""

    @pytest.fixture
    def scorer(self):
        """Create a quality scorer instance."""
        return QualityScorer()

    def test_score_conversation_returns_all_dimensions(self, scorer):
        """Test that scoring returns all quality dimensions."""
        conversation = {
            "input": "How do I return my order?",
            "output": "I understand your concern. You can return your order within 30 days by visiting our returns portal. Is there anything else I can help you with?"
        }
        scores = scorer.score_conversation(conversation)

        assert "empathy" in scores
        assert "understanding" in scores
        assert "relevancy" in scores
        assert "clarity" in scores
        assert "proactiveness" in scores
        assert "failure_recovery" in scores
        assert "overall" in scores

    def test_score_conversation_all_in_range(self, scorer):
        """Test that all scores are in valid range 0-100."""
        conversation = {
            "input": "What's the status of my order?",
            "output": "Let me check that for you. Your order is currently being processed and should ship within 2-3 business days."
        }
        scores = scorer.score_conversation(conversation)

        for dimension, score in scores.items():
            assert 0 <= score <= 100, f"{dimension} score {score} out of range"

    def test_score_empathetic_response(self, scorer):
        """Test that empathetic responses score higher on empathy."""
        empathetic_response = {
            "input": "My order never arrived!",
            "output": "I'm so sorry to hear about this frustrating experience. I completely understand how disappointing this must be. Let me help you resolve this right away."
        }
        cold_response = {
            "input": "My order never arrived!",
            "output": "Check tracking number."
        }

        empathetic_score = scorer.score_conversation(empathetic_response)["empathy"]
        cold_score = scorer.score_conversation(cold_response)["empathy"]

        assert empathetic_score > cold_score

    def test_score_proactive_response(self, scorer):
        """Test that proactive responses score higher on proactiveness."""
        proactive_response = {
            "input": "How do I track my order?",
            "output": "You can track your order using the link in your confirmation email. Additionally, I can also help you with estimated delivery times or setting up delivery notifications. Would you like me to assist with anything else?"
        }
        basic_response = {
            "input": "How do I track my order?",
            "output": "Use the tracking link."
        }

        proactive_score = scorer.score_conversation(proactive_response)["proactiveness"]
        basic_score = scorer.score_conversation(basic_response)["proactiveness"]

        assert proactive_score > basic_score

    def test_score_clear_response(self, scorer):
        """Test that clear, structured responses score higher on clarity."""
        clear_response = {
            "input": "What's your return policy?",
            "output": "Our return policy includes: 1. 30-day return window. 2. Items must be unused. 3. Original packaging required. 4. Free return shipping for defective items."
        }
        unclear_response = {
            "input": "What's your return policy?",
            "output": "Well I think maybe you could possibly return things if they're not too old and I believe the packaging should be there but I'm not entirely sure about all the details."
        }

        clear_score = scorer.score_conversation(clear_response)["clarity"]
        unclear_score = scorer.score_conversation(unclear_response)["clarity"]

        assert clear_score > unclear_score

    def test_score_failure_recovery(self, scorer):
        """Test failure recovery scoring when limitations are handled well."""
        good_recovery = {
            "input": "Can you process a refund for me?",
            "output": "Unfortunately, I cannot process refunds directly. However, I can help you by connecting you with our refunds team, or you can try submitting a refund request through our website. Alternatively, I can provide you with the refund policy details."
        }
        poor_recovery = {
            "input": "Can you process a refund for me?",
            "output": "I cannot do that."
        }

        good_score = scorer.score_conversation(good_recovery)["failure_recovery"]
        poor_score = scorer.score_conversation(poor_recovery)["failure_recovery"]

        assert good_score > poor_score

    def test_score_empty_response(self, scorer):
        """Test scoring with empty response."""
        conversation = {
            "input": "Hello?",
            "output": ""
        }
        scores = scorer.score_conversation(conversation)
        assert scores == scorer._empty_scores()

    def test_score_batch(self, scorer, sample_conversations):
        """Test batch scoring of multiple conversations."""
        scores = scorer.score_batch(sample_conversations)
        assert len(scores) == len(sample_conversations)
        for score in scores:
            assert "overall" in score

    def test_get_dimension_averages(self, scorer, sample_conversations):
        """Test getting average scores across conversations."""
        averages = scorer.get_dimension_averages(sample_conversations)

        assert "empathy" in averages
        assert "overall" in averages
        for dimension, avg in averages.items():
            assert 0 <= avg <= 100

    def test_get_dimension_averages_empty(self, scorer):
        """Test dimension averages with empty list."""
        averages = scorer.get_dimension_averages([])
        assert averages == scorer._empty_scores()

    def test_understanding_keyword_matching(self, scorer):
        """Test that responses addressing keywords score higher on understanding."""
        good_understanding = {
            "input": "How long does shipping take?",
            "output": "Shipping typically takes 3-5 business days for standard delivery. Express shipping takes 1-2 days."
        }
        poor_understanding = {
            "input": "How long does shipping take?",
            "output": "Our return policy allows 30 days for refunds."
        }

        good_score = scorer.score_conversation(good_understanding)["understanding"]
        poor_score = scorer.score_conversation(poor_understanding)["understanding"]

        assert good_score > poor_score

    def test_relevancy_scoring(self, scorer):
        """Test that relevant responses score higher on relevancy."""
        relevant = {
            "input": "What payment methods do you accept?",
            "output": "We accept Visa, MasterCard, PayPal, and Apple Pay. All payments are securely processed."
        }
        irrelevant = {
            "input": "What payment methods do you accept?",
            "output": "Our store hours are 9 AM to 5 PM Monday through Friday."
        }

        relevant_score = scorer.score_conversation(relevant)["relevancy"]
        irrelevant_score = scorer.score_conversation(irrelevant)["relevancy"]

        assert relevant_score > irrelevant_score

    def test_overall_weighted_average(self, scorer):
        """Test that overall score is a weighted average of dimensions."""
        conversation = {
            "input": "Hello",
            "output": "Hello! I'm happy to help you today. How can I assist you? I can help with orders, returns, or any questions you might have. Let me know!"
        }
        scores = scorer.score_conversation(conversation)

        # Weights from QualityScorer
        weights = {
            "empathy": 0.15,
            "understanding": 0.25,
            "relevancy": 0.25,
            "clarity": 0.15,
            "proactiveness": 0.10,
            "failure_recovery": 0.10,
        }

        expected_overall = sum(
            scores[dim] * weights[dim]
            for dim in weights
        )

        # Allow small floating point difference
        assert abs(scores["overall"] - expected_overall) < 0.5
