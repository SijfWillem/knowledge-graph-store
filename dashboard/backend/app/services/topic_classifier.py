from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Tuple, Dict, Any


class TopicClassifier:
    """
    Semantic topic classifier using sentence embeddings.
    Classifies user questions into predefined topic categories.
    """

    TOPICS = [
        "Product Returns",
        "Shipping & Delivery",
        "Payment Issues",
        "Account Management",
        "Warranty & Repairs",
        "Product Information",
        "Discounts & Promotions",
        "Technical Support",
        "Order Status",
        "General Inquiry"
    ]

    # Extended topic descriptions for better matching
    TOPIC_DESCRIPTIONS = {
        "Product Returns": "returning items, refund policy, exchange products, return shipping, money back",
        "Shipping & Delivery": "delivery time, shipping cost, tracking package, shipping address, delivery status",
        "Payment Issues": "payment failed, credit card problem, billing error, payment method, transaction declined",
        "Account Management": "login problem, password reset, update profile, account settings, delete account",
        "Warranty & Repairs": "warranty claim, product repair, warranty period, broken item, replacement",
        "Product Information": "product features, specifications, product details, availability, product comparison",
        "Discounts & Promotions": "coupon code, discount offer, promotional deal, sale price, loyalty points",
        "Technical Support": "software issue, app not working, technical problem, bug report, installation help",
        "Order Status": "order tracking, order confirmation, order history, cancel order, order update",
        "General Inquiry": "general question, other inquiry, miscellaneous, help needed, information request"
    }

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self._topic_embeddings = None
        self._enriched_topic_texts = None

    @property
    def topic_embeddings(self) -> np.ndarray:
        """Lazy load topic embeddings with enriched descriptions"""
        if self._topic_embeddings is None:
            # Combine topic name with description for richer embedding
            self._enriched_topic_texts = [
                f"{topic}: {self.TOPIC_DESCRIPTIONS.get(topic, topic)}"
                for topic in self.TOPICS
            ]
            self._topic_embeddings = self.model.encode(self._enriched_topic_texts)
        return self._topic_embeddings

    def classify(self, question: str) -> Tuple[str, float]:
        """
        Classify a single question into a topic.

        Args:
            question: The user question to classify

        Returns:
            Tuple of (topic_name, confidence_score)
        """
        q_embedding = self.model.encode([question])
        similarities = cosine_similarity(q_embedding, self.topic_embeddings)[0]
        best_idx = np.argmax(similarities)
        return self.TOPICS[best_idx], float(similarities[best_idx])

    def classify_batch(self, questions: List[str]) -> List[Tuple[str, float]]:
        """
        Classify multiple questions efficiently.

        Args:
            questions: List of questions to classify

        Returns:
            List of (topic_name, confidence_score) tuples
        """
        if not questions:
            return []

        q_embeddings = self.model.encode(questions)
        similarities = cosine_similarity(q_embeddings, self.topic_embeddings)

        results = []
        for sim_row in similarities:
            best_idx = np.argmax(sim_row)
            results.append((self.TOPICS[best_idx], float(sim_row[best_idx])))

        return results

    def get_top_topics(
        self,
        question: str,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Get top K most likely topics for a question.

        Args:
            question: The question to classify
            top_k: Number of top topics to return

        Returns:
            List of dicts with topic and confidence
        """
        q_embedding = self.model.encode([question])
        similarities = cosine_similarity(q_embedding, self.topic_embeddings)[0]

        # Get indices of top K topics
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        return [
            {
                "topic": self.TOPICS[idx],
                "confidence": float(similarities[idx])
            }
            for idx in top_indices
        ]

    def add_custom_topic(self, topic_name: str, description: str):
        """
        Add a custom topic to the classifier.

        Args:
            topic_name: Name of the new topic
            description: Description for embedding
        """
        if topic_name not in self.TOPICS:
            self.TOPICS.append(topic_name)
            self.TOPIC_DESCRIPTIONS[topic_name] = description
            # Reset embeddings to regenerate
            self._topic_embeddings = None

    def get_topic_distribution(
        self,
        questions: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get distribution of topics across a list of questions.

        Args:
            questions: List of questions

        Returns:
            Dict mapping topic to count and percentage
        """
        if not questions:
            return {}

        classifications = self.classify_batch(questions)

        distribution = {topic: {"count": 0, "percentage": 0.0} for topic in self.TOPICS}

        for topic, confidence in classifications:
            distribution[topic]["count"] += 1

        total = len(questions)
        for topic in distribution:
            distribution[topic]["percentage"] = round(
                distribution[topic]["count"] / total * 100, 2
            )

        return distribution
