import os
from typing import List, Optional, Dict, Any
import logging

import cognee

logger = logging.getLogger(__name__)


class CogneeService:
    """
    Uses Cognee to build knowledge graphs and identify gaps.

    Workflow:
    1. Ingest your chatbot's knowledge base docs
    2. Build knowledge graph with cognee.cognify()
    3. When a question fails, check if knowledge exists in graph
    4. If not found = knowledge gap identified
    """

    def __init__(self, llm_api_key: str):
        self.llm_api_key = llm_api_key
        self._initialized = False

    async def initialize(self):
        """Setup Cognee with your LLM provider"""
        if self._initialized:
            return

        # Set the OpenAI API key for Cognee
        os.environ["LLM_API_KEY"] = self.llm_api_key

        # Reset any previous state if needed
        try:
            await cognee.prune.prune_data()
            await cognee.prune.prune_system(metadata=True)
        except Exception as e:
            logger.warning(f"Could not prune existing data: {e}")

        self._initialized = True
        logger.info("Cognee service initialized")

    async def ingest_knowledge_base(
        self,
        documents: List[str],
        dataset_name: str = "knowledge_base"
    ) -> Dict[str, Any]:
        """
        Add your existing knowledge base to Cognee.
        This creates the reference graph to compare against.

        Args:
            documents: List of document texts to ingest
            dataset_name: Name for this dataset in Cognee

        Returns:
            Dict with ingestion stats
        """
        if not self._initialized:
            await self.initialize()

        ingested_count = 0
        errors = []

        for i, doc in enumerate(documents):
            try:
                await cognee.add(doc, dataset_name)
                ingested_count += 1
            except Exception as e:
                logger.error(f"Error ingesting document {i}: {e}")
                errors.append({"index": i, "error": str(e)})

        # Build the knowledge graph
        try:
            await cognee.cognify()
            logger.info(f"Knowledge graph built with {ingested_count} documents")
        except Exception as e:
            logger.error(f"Error building knowledge graph: {e}")
            errors.append({"stage": "cognify", "error": str(e)})

        # Add memory algorithms for better retrieval
        try:
            await cognee.memify()
            logger.info("Memory algorithms applied")
        except Exception as e:
            logger.warning(f"Could not apply memory algorithms: {e}")

        return {
            "ingested_count": ingested_count,
            "total_documents": len(documents),
            "errors": errors,
            "success": len(errors) == 0,
        }

    async def check_knowledge_exists(
        self,
        question: str,
        confidence_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Search the knowledge graph for relevant info.

        Args:
            question: The question to check
            confidence_threshold: Minimum confidence to consider as "found"

        Returns:
            {
                "found": bool,
                "confidence": float,
                "related_concepts": List[str],
                "results": List of search results,
                "suggested_addition": str if not found
            }
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Search the knowledge graph
            results = await cognee.search(question)

            if not results or len(results) == 0:
                return {
                    "found": False,
                    "confidence": 0.0,
                    "related_concepts": [],
                    "results": [],
                    "suggested_addition": self._generate_suggestion(question, []),
                }

            # Analyze results to determine confidence
            confidence = self._calculate_confidence(results, question)
            related_concepts = self._extract_concepts(results)

            found = confidence >= confidence_threshold

            return {
                "found": found,
                "confidence": confidence,
                "related_concepts": related_concepts,
                "results": self._serialize_results(results),
                "suggested_addition": None if found else self._generate_suggestion(question, results),
            }

        except Exception as e:
            logger.error(f"Error checking knowledge for question '{question}': {e}")
            return {
                "found": False,
                "confidence": 0.0,
                "related_concepts": [],
                "results": [],
                "error": str(e),
                "suggested_addition": self._generate_suggestion(question, []),
            }

    async def identify_gap(
        self,
        question: str,
        ai_response: str,
        feedback_negative: bool
    ) -> Optional[Dict[str, Any]]:
        """
        Determine if a failed conversation indicates a knowledge gap.

        Logic:
        1. If feedback is negative
        2. AND knowledge graph doesn't have strong match
        3. THEN this is a knowledge gap

        Args:
            question: The user's question
            ai_response: The AI's response
            feedback_negative: Whether feedback was negative

        Returns:
            Gap info dict if gap identified, None otherwise
        """
        if not feedback_negative:
            return None

        knowledge_check = await self.check_knowledge_exists(question)

        # If knowledge exists with high confidence, this might not be a knowledge gap
        # but rather a response quality issue
        if knowledge_check["found"] and knowledge_check["confidence"] > 0.7:
            return {
                "is_gap": False,
                "gap_type": "response_quality",
                "question": question,
                "explanation": "Knowledge exists but response quality may be the issue",
                "confidence": knowledge_check["confidence"],
                "related_concepts": knowledge_check["related_concepts"],
            }

        # This is a knowledge gap
        return {
            "is_gap": True,
            "gap_type": "missing_knowledge",
            "question": question,
            "ai_response": ai_response,
            "confidence": knowledge_check["confidence"],
            "related_concepts": knowledge_check["related_concepts"],
            "suggested_addition": knowledge_check.get("suggested_addition"),
        }

    async def add_knowledge(
        self,
        content: str,
        dataset_name: str = "knowledge_base"
    ) -> Dict[str, Any]:
        """
        Add new knowledge to fill a gap.

        Args:
            content: The knowledge content to add
            dataset_name: Dataset to add to

        Returns:
            Status of the addition
        """
        if not self._initialized:
            await self.initialize()

        try:
            await cognee.add(content, dataset_name)
            await cognee.cognify()

            return {
                "success": True,
                "content_added": content[:100] + "..." if len(content) > 100 else content,
            }
        except Exception as e:
            logger.error(f"Error adding knowledge: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    def _calculate_confidence(
        self,
        results: List[Any],
        question: str
    ) -> float:
        """
        Calculate confidence score based on search results.
        Uses heuristics based on result quality and relevance.
        """
        if not results:
            return 0.0

        # Base confidence from having results
        base_confidence = min(0.3, len(results) * 0.1)

        # Analyze result content for relevance
        relevance_boost = 0.0
        question_words = set(question.lower().split())

        for result in results[:5]:  # Check top 5 results
            result_text = str(result).lower()
            matching_words = sum(1 for word in question_words if word in result_text)
            relevance_boost += (matching_words / max(len(question_words), 1)) * 0.14

        return min(1.0, base_confidence + relevance_boost)

    def _extract_concepts(self, results: List[Any]) -> List[str]:
        """Extract key concepts from search results"""
        concepts = []

        for result in results[:10]:
            result_str = str(result)
            # Extract notable phrases (simplified - in production use NLP)
            words = result_str.split()
            for i in range(len(words) - 1):
                phrase = f"{words[i]} {words[i+1]}"
                if len(phrase) > 5 and phrase not in concepts:
                    concepts.append(phrase)
                    if len(concepts) >= 10:
                        break

        return concepts[:10]

    def _serialize_results(self, results: List[Any]) -> List[Dict[str, Any]]:
        """Convert Cognee results to serializable format"""
        serialized = []
        for result in results[:10]:
            if hasattr(result, "__dict__"):
                serialized.append({
                    "content": str(result),
                    "type": type(result).__name__,
                })
            else:
                serialized.append({
                    "content": str(result),
                    "type": "text",
                })
        return serialized

    def _generate_suggestion(
        self,
        question: str,
        results: List[Any]
    ) -> str:
        """Generate a suggestion for knowledge to add"""
        if results:
            return f"Consider adding detailed documentation about: {question}. Related information exists but may need expansion."
        return f"No relevant information found. Consider adding comprehensive documentation addressing: {question}"
