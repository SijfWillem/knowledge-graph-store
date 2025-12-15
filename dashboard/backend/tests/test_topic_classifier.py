import pytest
from app.services.topic_classifier import TopicClassifier


class TestTopicClassifier:
    """Tests for TopicClassifier service."""

    @pytest.fixture
    def classifier(self):
        """Create a topic classifier instance."""
        return TopicClassifier()

    def test_init_has_default_topics(self, classifier):
        """Test that classifier initializes with default topics."""
        assert len(classifier.TOPICS) == 10
        assert "Product Returns" in classifier.TOPICS
        assert "Shipping & Delivery" in classifier.TOPICS
        assert "Payment Issues" in classifier.TOPICS

    def test_classify_returns_tuple(self, classifier):
        """Test that classify returns a tuple of (topic, confidence)."""
        result = classifier.classify("How do I return my order?")
        assert isinstance(result, tuple)
        assert len(result) == 2
        topic, confidence = result
        assert isinstance(topic, str)
        assert isinstance(confidence, float)
        assert 0 <= confidence <= 1

    def test_classify_product_returns_question(self, classifier):
        """Test classification of product returns question."""
        topic, confidence = classifier.classify("How do I return my order?")
        assert topic == "Product Returns"
        assert confidence > 0.4  # Semantic similarity can vary by model

    def test_classify_shipping_question(self, classifier):
        """Test classification of shipping question."""
        topic, confidence = classifier.classify("When will my package arrive?")
        assert topic == "Shipping & Delivery"
        assert confidence > 0.4  # Semantic similarity can vary by model

    def test_classify_payment_question(self, classifier):
        """Test classification of payment question."""
        topic, confidence = classifier.classify("My credit card was declined")
        assert topic == "Payment Issues"
        assert confidence > 0.4  # Semantic similarity can vary by model

    def test_classify_account_question(self, classifier):
        """Test classification of account question."""
        topic, confidence = classifier.classify("How do I reset my password?")
        assert topic == "Account Management"
        assert confidence > 0.4  # Semantic similarity can vary by model

    def test_classify_technical_question(self, classifier):
        """Test classification of technical support question."""
        topic, confidence = classifier.classify("The app keeps crashing on startup")
        assert topic == "Technical Support"
        assert confidence > 0.3  # Technical questions may have lower similarity

    def test_classify_batch_empty_list(self, classifier):
        """Test batch classification with empty list."""
        result = classifier.classify_batch([])
        assert result == []

    def test_classify_batch_multiple_questions(self, classifier):
        """Test batch classification with multiple questions."""
        questions = [
            "How do I return my order?",
            "When will my package arrive?",
            "My payment failed"
        ]
        results = classifier.classify_batch(questions)
        assert len(results) == 3
        assert results[0][0] == "Product Returns"
        assert results[1][0] == "Shipping & Delivery"
        assert results[2][0] == "Payment Issues"

    def test_get_top_topics(self, classifier):
        """Test getting top K topics."""
        results = classifier.get_top_topics("How do I return my order?", top_k=3)
        assert len(results) == 3
        assert all("topic" in r and "confidence" in r for r in results)
        # Results should be sorted by confidence (descending)
        confidences = [r["confidence"] for r in results]
        assert confidences == sorted(confidences, reverse=True)

    def test_add_custom_topic(self, classifier):
        """Test adding a custom topic."""
        initial_count = len(classifier.TOPICS)
        classifier.add_custom_topic("Custom Support", "custom help desk assistance")
        assert len(classifier.TOPICS) == initial_count + 1
        assert "Custom Support" in classifier.TOPICS
        assert "Custom Support" in classifier.TOPIC_DESCRIPTIONS

    def test_add_duplicate_topic(self, classifier):
        """Test that adding duplicate topic doesn't create duplicates."""
        initial_count = len(classifier.TOPICS)
        classifier.add_custom_topic("Product Returns", "returns and refunds")
        assert len(classifier.TOPICS) == initial_count

    def test_get_topic_distribution(self, classifier):
        """Test getting topic distribution."""
        questions = [
            "Return my order",
            "Return policy",
            "Shipping status",
            "Payment declined"
        ]
        distribution = classifier.get_topic_distribution(questions)
        assert isinstance(distribution, dict)
        assert "Product Returns" in distribution
        assert distribution["Product Returns"]["count"] >= 2
        total_percent = sum(d["percentage"] for d in distribution.values())
        assert abs(total_percent - 100) < 0.1  # Should sum to ~100%

    def test_get_topic_distribution_empty(self, classifier):
        """Test topic distribution with empty list."""
        distribution = classifier.get_topic_distribution([])
        assert distribution == {}

    def test_topic_embeddings_lazy_loading(self, classifier):
        """Test that topic embeddings are lazily loaded."""
        # Access the property
        embeddings = classifier.topic_embeddings
        assert embeddings is not None
        assert embeddings.shape[0] == len(classifier.TOPICS)
        # Verify it's cached
        assert classifier._topic_embeddings is not None
