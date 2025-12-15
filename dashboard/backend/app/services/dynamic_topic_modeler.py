"""
Dynamic Topic Modeler using BERTopic for automatic topic discovery.
Discovers topics from questions without predefined categories.
"""
import os
import logging
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA

logger = logging.getLogger(__name__)

# Model storage path
MODEL_PATH = Path("/app/data/topic_model")


def normalize_question_for_clustering(question: str) -> str:
    """
    Normalize a question to focus on the theme/intent rather than specific entities.
    This helps cluster questions like "how old is John" and "how old is Mary" together.
    """
    import re

    text = question.lower().strip()

    # Common patterns to normalize - replace specific names/entities with placeholders
    # This helps BERTopic focus on the question TYPE rather than WHO it's about

    # Replace possessive patterns: "john's", "mary's", "sijf's", "sijfs" -> "[person]'s"
    # Handle both with and without apostrophe
    text = re.sub(r"\b[a-z]+['']s\b", "[person]'s", text)
    # Also handle possessives without apostrophe followed by relationship words
    text = re.sub(r"\b([a-z]+)s\s+(friend|girlfriend|boyfriend|brother|sister|mother|father|wife|husband|child|son|daughter)\b", r"[person]'s \2", text)

    # Replace relationship + name patterns: "friend jelte" -> "friend [person]"
    text = re.sub(r"\b(friend|girlfriend|boyfriend|brother|sister|mother|father|wife|husband|child|son|daughter)\s+([a-z]+)\b", r"\1 [person]", text)

    # Replace "is [name]" patterns at end of sentence (for questions like "how old is john")
    text = re.sub(r'\b(is|are|was|were)\s+([a-z]+)\s*\??$', r'\1 [person]?', text)

    # Replace "is [name]'s" patterns
    text = re.sub(r"\b(is|are|was|were)\s+\[person\]'s\b", r"\1 [person]'s", text)

    # Replace name patterns in "about [name]", "of [name]"
    text = re.sub(r'\b(about|of)\s+([a-z]+)\b', r'\1 [person]', text)

    # Replace "her/his/their" with generic form for consistency
    text = re.sub(r'\b(her|his|their)\b', '[person]\'s', text)

    # Clean up multiple spaces and normalize
    text = re.sub(r'\s+', ' ', text).strip()

    return text


class DynamicTopicModeler:
    """
    Dynamic topic modeler that discovers topics automatically using BERTopic.
    Supports incremental learning for streaming data.
    """

    def __init__(
        self,
        embedding_model_name: str = 'all-MiniLM-L6-v2',
        min_topic_size: int = 2,
        nr_topics: Optional[int] = None,
    ):
        self.embedding_model_name = embedding_model_name
        self.min_topic_size = min_topic_size
        self.nr_topics = nr_topics

        self._embedding_model = None
        self._topic_model = None
        self._is_fitted = False
        self._documents: List[str] = []
        self._topics: List[int] = []
        self._topic_labels: Dict[int, str] = {}

    @property
    def embedding_model(self) -> SentenceTransformer:
        """Lazy load embedding model"""
        if self._embedding_model is None:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
        return self._embedding_model

    @property
    def topic_model(self):
        """Lazy load or create BERTopic model"""
        if self._topic_model is None:
            self._initialize_topic_model()
        return self._topic_model

    def _initialize_topic_model(self, n_docs: int = 100):
        """Initialize BERTopic with parameters suitable for the dataset size"""
        try:
            from bertopic import BERTopic
            from bertopic.vectorizers import OnlineCountVectorizer
            from umap import UMAP
            from hdbscan import HDBSCAN

            # Use online-compatible components for incremental learning
            vectorizer_model = OnlineCountVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                min_df=1,
            )

            # Try to load existing model first
            if MODEL_PATH.exists():
                try:
                    logger.info(f"Loading existing topic model from {MODEL_PATH}")
                    self._topic_model = BERTopic.load(
                        str(MODEL_PATH),
                        embedding_model=self.embedding_model
                    )
                    self._is_fitted = True
                    self._load_topic_labels()
                    logger.info(f"Loaded model with {len(self.get_topics())} topics")
                    return
                except Exception as e:
                    logger.warning(f"Could not load existing model: {e}")

            # Configure UMAP for small datasets
            # n_neighbors must be less than the number of documents
            n_neighbors = min(15, max(2, n_docs - 1))
            n_components = min(5, max(2, n_docs - 2))

            logger.info(f"Configuring UMAP: n_neighbors={n_neighbors}, n_components={n_components} for {n_docs} docs")

            umap_model = UMAP(
                n_neighbors=n_neighbors,
                n_components=n_components,
                min_dist=0.0,
                metric='cosine',
                random_state=42,
            )

            # Configure HDBSCAN for small datasets
            min_cluster_size = max(2, min(self.min_topic_size, n_docs // 2))
            min_samples = max(1, min_cluster_size - 1)

            logger.info(f"Configuring HDBSCAN: min_cluster_size={min_cluster_size}, min_samples={min_samples}")

            hdbscan_model = HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric='euclidean',
                cluster_selection_method='eom',
                prediction_data=True,
            )

            # Create new model
            logger.info("Creating new BERTopic model")
            self._topic_model = BERTopic(
                embedding_model=self.embedding_model,
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                vectorizer_model=vectorizer_model,
                min_topic_size=self.min_topic_size,
                nr_topics=self.nr_topics,
                verbose=False,
                calculate_probabilities=True,
            )

        except ImportError as e:
            logger.error(f"BERTopic not available: {e}")
            self._topic_model = None

    def _load_topic_labels(self):
        """Load custom topic labels from disk"""
        labels_path = MODEL_PATH / "topic_labels.npy"
        if labels_path.exists():
            try:
                self._topic_labels = dict(np.load(labels_path, allow_pickle=True).item())
            except Exception as e:
                logger.warning(f"Could not load topic labels: {e}")
                self._topic_labels = {}

    def _save_topic_labels(self):
        """Save custom topic labels to disk"""
        MODEL_PATH.mkdir(parents=True, exist_ok=True)
        labels_path = MODEL_PATH / "topic_labels.npy"
        np.save(labels_path, self._topic_labels)

    def _generate_topic_label(self, topic_id: int) -> str:
        """Generate a human-readable label for a topic based on its top words"""
        if topic_id == -1:
            return "Uncategorized"

        if self._topic_model is None:
            return f"Topic {topic_id}"

        try:
            topic_words = self._topic_model.get_topic(topic_id)
            if topic_words and len(topic_words) > 0:
                # Filter out placeholder words and get meaningful keywords
                skip_words = {'person', 'persons', '[person]', "person's"}
                meaningful_words = []

                for word, score in topic_words:
                    # Skip placeholder words
                    if word.lower() in skip_words:
                        continue
                    # Skip words that are just "person" combinations
                    if 'person' in word.lower() and len(word.replace('person', '').strip()) < 3:
                        continue
                    # Extract the meaningful part from combined words like "old person"
                    clean_word = word.lower().replace('person', '').replace("'s", '').strip()
                    if clean_word and len(clean_word) >= 2 and clean_word not in meaningful_words:
                        meaningful_words.append(clean_word)

                    if len(meaningful_words) >= 3:
                        break

                if meaningful_words:
                    # Create a readable label from meaningful words
                    label = " & ".join(meaningful_words[:2]).title()
                    # Add "Questions" suffix for clarity
                    return f"{label} Questions"

        except Exception as e:
            logger.warning(f"Could not get topic words for topic {topic_id}: {e}")

        return f"Topic {topic_id}"

    def fit(self, documents: List[str]) -> 'DynamicTopicModeler':
        """
        Fit the topic model on a list of documents.

        Args:
            documents: List of text documents (questions)

        Returns:
            self for chaining
        """
        # Need at least 5 documents for meaningful topic discovery
        MIN_DOCS_FOR_TOPICS = 5

        if not documents or len(documents) < self.min_topic_size:
            logger.warning(f"Not enough documents to fit topic model (need at least {self.min_topic_size})")
            return self

        # For very small datasets, use simple keyword-based grouping instead
        if len(documents) < MIN_DOCS_FOR_TOPICS:
            logger.info(f"Using simple topic assignment for {len(documents)} documents (< {MIN_DOCS_FOR_TOPICS})")
            self._fit_simple(documents)
            return self

        try:
            logger.info(f"Fitting topic model on {len(documents)} documents")

            # Normalize questions to focus on theme/intent rather than specific entities
            normalized_docs = [normalize_question_for_clustering(doc) for doc in documents]
            logger.info(f"Sample normalizations:")
            for orig, norm in list(zip(documents, normalized_docs))[:3]:
                logger.info(f"  '{orig}' -> '{norm}'")

            # Reinitialize model with parameters suited to this dataset size
            self._topic_model = None  # Reset to force reinitialization
            self._initialize_topic_model(n_docs=len(documents))

            if self._topic_model is None:
                logger.error("Topic model not available")
                self._fit_simple(documents)
                return self

            # Use normalized docs for clustering
            topics, probs = self._topic_model.fit_transform(normalized_docs)
            self._documents = list(documents)
            self._topics = list(topics)
            self._is_fitted = True

            # Generate labels for all discovered topics
            self._generate_all_topic_labels()

            # Save model
            self._save_model()

            logger.info(f"Discovered {len(self.get_topics())} topics")

        except Exception as e:
            logger.error(f"Error fitting topic model: {e}")
            # Fall back to simple topic assignment on error
            logger.info("Falling back to simple topic assignment")
            self._fit_simple(documents)

        return self

    def _fit_simple(self, documents: List[str]):
        """
        Simple fallback topic assignment when we don't have enough documents for BERTopic.
        Groups all documents under a general topic.
        """
        self._documents = list(documents)
        self._topics = [0] * len(documents)  # All in one topic
        self._topic_labels = {
            0: "General Questions",
            -1: "Uncategorized"
        }
        self._is_fitted = True
        logger.info(f"Simple topic assignment: {len(documents)} documents assigned to 'General Questions'")

    def partial_fit(self, documents: List[str]) -> 'DynamicTopicModeler':
        """
        Incrementally update the topic model with new documents.

        Args:
            documents: List of new documents to add

        Returns:
            self for chaining
        """
        if not documents:
            return self

        if self.topic_model is None:
            logger.error("Topic model not available")
            return self

        try:
            if not self._is_fitted:
                # First fit
                return self.fit(documents)

            logger.info(f"Partially fitting topic model with {len(documents)} new documents")

            # Combine with existing documents and refit
            # Note: For true incremental learning, consider using River clustering
            all_docs = self._documents + list(documents)

            # Refit on all documents
            topics, probs = self.topic_model.fit_transform(all_docs)
            self._documents = all_docs
            self._topics = list(topics)

            # Regenerate labels
            self._generate_all_topic_labels()

            # Save model
            self._save_model()

            logger.info(f"Model now has {len(self.get_topics())} topics from {len(all_docs)} documents")

        except Exception as e:
            logger.error(f"Error in partial_fit: {e}")

        return self

    def _generate_all_topic_labels(self):
        """Generate human-readable labels for all topics"""
        if self.topic_model is None:
            return

        try:
            topics = self.topic_model.get_topics()
            for topic_id in topics.keys():
                if topic_id not in self._topic_labels:
                    self._topic_labels[topic_id] = self._generate_topic_label(topic_id)
        except Exception as e:
            logger.warning(f"Error generating topic labels: {e}")

    def _save_model(self):
        """Save the topic model to disk"""
        if self._topic_model is None:
            return

        try:
            MODEL_PATH.mkdir(parents=True, exist_ok=True)
            self._topic_model.save(
                str(MODEL_PATH),
                serialization="safetensors",
                save_ctfidf=True,
                save_embedding_model=self.embedding_model_name
            )
            self._save_topic_labels()
            logger.info(f"Saved topic model to {MODEL_PATH}")
        except Exception as e:
            logger.error(f"Error saving topic model: {e}")

    def classify(self, question: str) -> Tuple[str, float]:
        """
        Classify a single question into a topic.

        Args:
            question: The user question to classify

        Returns:
            Tuple of (topic_label, confidence_score)
        """
        if not self._is_fitted:
            return "Uncategorized", 0.0

        # If we used simple fallback, return the general topic
        if self._topic_model is None or not hasattr(self._topic_model, 'transform'):
            label = self._topic_labels.get(0, "General Questions")
            return label, 0.5

        try:
            # Normalize question to match how we trained the model
            normalized_question = normalize_question_for_clustering(question)
            topics, probs = self._topic_model.transform([normalized_question])
            topic_id = topics[0]
            confidence = float(probs[0].max()) if probs is not None and len(probs) > 0 else 0.5

            # Get human-readable label
            label = self._topic_labels.get(topic_id, self._generate_topic_label(topic_id))

            return label, confidence

        except Exception as e:
            logger.error(f"Error classifying question: {e}")
            # Return the first available topic label as fallback
            if self._topic_labels:
                first_topic = next((t for t in self._topic_labels.keys() if t != -1), 0)
                return self._topic_labels.get(first_topic, "General Questions"), 0.3
            return "Uncategorized", 0.0

    def classify_batch(self, questions: List[str]) -> List[Tuple[str, float]]:
        """
        Classify multiple questions efficiently.

        Args:
            questions: List of questions to classify

        Returns:
            List of (topic_label, confidence_score) tuples
        """
        if not questions:
            return []

        if not self._is_fitted:
            return [("Uncategorized", 0.0) for _ in questions]

        # If we used simple fallback, return the general topic for all
        if self._topic_model is None or not hasattr(self._topic_model, 'transform'):
            label = self._topic_labels.get(0, "General Questions")
            return [(label, 0.5) for _ in questions]

        try:
            # Normalize questions to match how we trained the model
            normalized_questions = [normalize_question_for_clustering(q) for q in questions]
            topics, probs = self._topic_model.transform(normalized_questions)

            results = []
            for i, topic_id in enumerate(topics):
                confidence = float(probs[i].max()) if probs is not None else 0.5
                label = self._topic_labels.get(topic_id, self._generate_topic_label(topic_id))
                results.append((label, confidence))

            return results

        except Exception as e:
            logger.error(f"Error in batch classification: {e}")
            # Fallback: return the first available topic
            if self._topic_labels:
                first_topic = next((t for t in self._topic_labels.keys() if t != -1), 0)
                label = self._topic_labels.get(first_topic, "General Questions")
                return [(label, 0.3) for _ in questions]
            return [("Uncategorized", 0.0) for _ in questions]

    def get_topics(self) -> Dict[int, List[Tuple[str, float]]]:
        """
        Get all discovered topics with their top words.

        Returns:
            Dict mapping topic_id to list of (word, weight) tuples
        """
        if self._topic_model is None:
            return {}

        try:
            return self._topic_model.get_topics()
        except Exception as e:
            logger.error(f"Error getting topics: {e}")
            return {}

    def get_topic_labels(self) -> Dict[int, str]:
        """
        Get human-readable labels for all topics.

        Returns:
            Dict mapping topic_id to label string
        """
        return self._topic_labels.copy()

    def get_topic_info(self) -> List[Dict[str, Any]]:
        """
        Get information about all discovered topics.

        Returns:
            List of topic info dicts with id, label, count, and top words
        """
        # Handle simple fallback case
        if self._topic_model is None or not hasattr(self._topic_model, 'get_topic_info'):
            if self._is_fitted and self._topic_labels:
                result = []
                for topic_id, label in self._topic_labels.items():
                    if topic_id == -1:
                        continue
                    count = sum(1 for t in self._topics if t == topic_id) if self._topics else 0
                    result.append({
                        'id': topic_id,
                        'label': label,
                        'count': count,
                        'top_words': [],
                    })
                return result
            return []

        try:
            topic_info = self._topic_model.get_topic_info()

            result = []
            for _, row in topic_info.iterrows():
                topic_id = row['Topic']
                if topic_id == -1:
                    continue  # Skip outlier topic

                result.append({
                    'id': topic_id,
                    'label': self._topic_labels.get(topic_id, f"Topic {topic_id}"),
                    'count': row['Count'],
                    'top_words': row.get('Representation', [])[:5] if 'Representation' in row else [],
                })

            return result

        except Exception as e:
            logger.error(f"Error getting topic info: {e}")
            return []

    def set_topic_label(self, topic_id: int, label: str):
        """
        Set a custom label for a topic.

        Args:
            topic_id: The topic ID
            label: The custom label
        """
        self._topic_labels[topic_id] = label
        self._save_topic_labels()

    @property
    def is_fitted(self) -> bool:
        """Check if the model has been fitted"""
        return self._is_fitted

    @property
    def n_topics(self) -> int:
        """Get the number of discovered topics (excluding outliers)"""
        topics = self.get_topics()
        return len([t for t in topics.keys() if t != -1])


# Singleton instance for the application
_topic_modeler: Optional[DynamicTopicModeler] = None


def get_topic_modeler() -> DynamicTopicModeler:
    """Get the singleton topic modeler instance"""
    global _topic_modeler
    if _topic_modeler is None:
        _topic_modeler = DynamicTopicModeler(min_topic_size=2)
    return _topic_modeler
