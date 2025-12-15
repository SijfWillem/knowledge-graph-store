from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

from .cognee_service import CogneeService
from .langfuse_service import LangFuseService

logger = logging.getLogger(__name__)


class GapAnalyzer:
    """
    Combines LangFuse traces with Cognee analysis to find knowledge gaps.
    """

    def __init__(self, langfuse: LangFuseService, cognee: CogneeService):
        self.langfuse = langfuse
        self.cognee = cognee
        self._embedding_model = None

    @property
    def embedding_model(self) -> SentenceTransformer:
        """Lazy load the embedding model"""
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        return self._embedding_model

    async def analyze_failed_conversations(
        self,
        conversations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        For conversations with negative feedback:
        1. Extract the user question
        2. Check Cognee if knowledge exists
        3. If not, create a knowledge gap record

        Returns list of identified gaps with:
        - question
        - topic (classified)
        - frequency (how many similar questions)
        - priority_score
        - suggested_knowledge_addition
        """
        gaps = []

        for conv in conversations:
            # Only analyze conversations with negative feedback
            if not conv.get('feedback_negative', False):
                continue

            question = conv.get('input')
            if not question:
                continue

            try:
                gap_info = await self.cognee.identify_gap(
                    question=question,
                    ai_response=conv.get('output', ''),
                    feedback_negative=True
                )

                if gap_info and gap_info.get('is_gap'):
                    gap_record = {
                        "question": question,
                        "ai_response": conv.get('output'),
                        "trace_id": conv.get('trace_id'),
                        "session_id": conv.get('session_id'),
                        "timestamp": conv.get('timestamp'),
                        "confidence": gap_info.get('confidence', 0.0),
                        "related_concepts": gap_info.get('related_concepts', []),
                        "suggested_addition": gap_info.get('suggested_addition'),
                        "gap_type": gap_info.get('gap_type', 'missing_knowledge'),
                    }
                    gaps.append(gap_record)

            except Exception as e:
                logger.error(f"Error analyzing conversation {conv.get('trace_id')}: {e}")

        return gaps

    async def cluster_similar_gaps(
        self,
        gaps: List[Dict[str, Any]],
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Group similar questions together using embeddings.
        This prevents showing 50 variations of the same gap.

        Args:
            gaps: List of gap records
            similarity_threshold: Minimum similarity to cluster together

        Returns:
            Clustered gaps with frequency counts
        """
        if not gaps:
            return []

        if len(gaps) == 1:
            gaps[0]["frequency"] = 1
            gaps[0]["sample_questions"] = [gaps[0]["question"]]
            gaps[0]["sample_trace_ids"] = [gaps[0].get("trace_id")]
            gaps[0]["priority_score"] = self._calculate_priority(gaps)
            gaps[0]["priority_level"] = self._get_priority_level(gaps[0]["priority_score"])
            return gaps

        # Get embeddings for all questions
        questions = [g["question"] for g in gaps]
        embeddings = self.embedding_model.encode(questions)

        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(embeddings)

        # Convert to distance matrix (1 - similarity)
        distance_matrix = 1 - similarity_matrix

        # Cluster using agglomerative clustering
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1 - similarity_threshold,
            metric='precomputed',
            linkage='average'
        )
        cluster_labels = clustering.fit_predict(distance_matrix)

        # Group gaps by cluster
        clusters: Dict[int, List[Dict[str, Any]]] = {}
        for idx, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(gaps[idx])

        # Create clustered gap records
        clustered_gaps = []
        for cluster_id, cluster_gaps in clusters.items():
            # Find the most representative question (closest to centroid)
            cluster_embeddings = embeddings[[
                i for i, g in enumerate(gaps)
                if cluster_labels[i] == cluster_id
            ]]
            centroid = np.mean(cluster_embeddings, axis=0)
            distances_to_centroid = [
                np.linalg.norm(emb - centroid)
                for emb in cluster_embeddings
            ]
            representative_idx = np.argmin(distances_to_centroid)
            representative_gap = cluster_gaps[representative_idx]

            # Aggregate information
            all_trace_ids = [g.get("trace_id") for g in cluster_gaps if g.get("trace_id")]
            all_questions = [g["question"] for g in cluster_gaps]

            # Calculate priority based on frequency and other factors
            priority_score = self._calculate_priority(cluster_gaps)

            clustered_gap = {
                "question": representative_gap["question"],
                "frequency": len(cluster_gaps),
                "sample_questions": all_questions[:5],
                "sample_trace_ids": all_trace_ids[:10],
                "priority_score": priority_score,
                "priority_level": self._get_priority_level(priority_score),
                "confidence": np.mean([g.get("confidence", 0) for g in cluster_gaps]),
                "related_concepts": self._merge_concepts(cluster_gaps),
                "suggested_addition": representative_gap.get("suggested_addition"),
                "gap_type": representative_gap.get("gap_type"),
                "first_seen": min(
                    (g.get("timestamp") for g in cluster_gaps if g.get("timestamp")),
                    default=None
                ),
                "last_seen": max(
                    (g.get("timestamp") for g in cluster_gaps if g.get("timestamp")),
                    default=None
                ),
            }
            clustered_gaps.append(clustered_gap)

        # Sort by priority score (descending)
        clustered_gaps.sort(key=lambda x: x["priority_score"], reverse=True)

        return clustered_gaps

    def _calculate_priority(self, cluster_gaps: List[Dict[str, Any]]) -> float:
        """
        Calculate priority score based on multiple factors:
        - Frequency (more occurrences = higher priority)
        - Recency (recent gaps are more important)
        - Confidence that it's truly missing (low knowledge confidence)
        """
        frequency = len(cluster_gaps)

        # Frequency factor (logarithmic scaling)
        frequency_score = min(1.0, np.log1p(frequency) / np.log1p(20))

        # Recency factor
        recency_score = 0.5
        timestamps = [g.get("timestamp") for g in cluster_gaps if g.get("timestamp")]
        if timestamps:
            most_recent = max(timestamps)
            if isinstance(most_recent, datetime):
                days_ago = (datetime.utcnow() - most_recent).days
                recency_score = max(0, 1 - (days_ago / 30))

        # Knowledge gap confidence (inverse - lower cognee confidence = higher gap priority)
        avg_confidence = np.mean([g.get("confidence", 0.5) for g in cluster_gaps])
        gap_certainty = 1 - avg_confidence

        # Weighted combination
        priority_score = (
            frequency_score * 0.4 +
            recency_score * 0.3 +
            gap_certainty * 0.3
        )

        return round(priority_score, 3)

    def _get_priority_level(self, score: float) -> str:
        """Convert priority score to level"""
        if score >= 0.8:
            return "critical"
        elif score >= 0.6:
            return "high"
        elif score >= 0.4:
            return "medium"
        return "low"

    def _merge_concepts(self, cluster_gaps: List[Dict[str, Any]]) -> List[str]:
        """Merge and deduplicate related concepts from all gaps in cluster"""
        all_concepts = []
        for gap in cluster_gaps:
            concepts = gap.get("related_concepts", [])
            all_concepts.extend(concepts)

        # Deduplicate while preserving order
        seen = set()
        unique_concepts = []
        for concept in all_concepts:
            if concept.lower() not in seen:
                seen.add(concept.lower())
                unique_concepts.append(concept)

        return unique_concepts[:15]

    async def analyze_and_cluster(
        self,
        from_timestamp: Optional[datetime] = None,
        limit: int = 500
    ) -> Dict[str, Any]:
        """
        Full pipeline: fetch traces, analyze for gaps, cluster results.

        Returns comprehensive analysis results.
        """
        # Fetch traces from LangFuse
        sync_result = await self.langfuse.sync_new_data(
            last_sync_timestamp=from_timestamp or datetime.min,
            batch_size=limit
        )

        traces = sync_result.get("traces", [])
        if not traces:
            return {
                "gaps": [],
                "total_traces_analyzed": 0,
                "negative_feedback_count": 0,
                "gaps_identified": 0,
            }

        # Filter to only negative feedback traces
        negative_traces = [t for t in traces if t.get("feedback_negative")]

        # Analyze for knowledge gaps
        gaps = await self.analyze_failed_conversations(negative_traces)

        # Cluster similar gaps
        clustered_gaps = await self.cluster_similar_gaps(gaps)

        return {
            "gaps": clustered_gaps,
            "total_traces_analyzed": len(traces),
            "negative_feedback_count": len(negative_traces),
            "gaps_identified": len(clustered_gaps),
            "sync_timestamp": sync_result.get("sync_timestamp"),
        }
