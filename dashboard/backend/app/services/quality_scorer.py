from typing import Dict, Any, List, Optional
import re
import logging

logger = logging.getLogger(__name__)


class QualityScorer:
    """
    Scores conversations on quality dimensions:
    - Empathy: How well the response acknowledges user feelings
    - Understanding: How well the response addresses the actual question
    - Relevancy: How relevant the response is to the query
    - Clarity: How clear and well-structured the response is
    - Proactiveness: Whether the response anticipates follow-up needs
    - Failure Recovery: How well errors/limitations are handled
    """

    # Empathy indicators
    EMPATHY_PHRASES = [
        "understand", "sorry", "apologize", "appreciate",
        "thank you", "certainly", "of course", "happy to help",
        "i can see", "frustrating", "inconvenient"
    ]

    # Proactiveness indicators
    PROACTIVE_PHRASES = [
        "also", "additionally", "you might also",
        "let me know if", "is there anything else",
        "would you like", "i can also", "for your reference"
    ]

    # Clarity indicators (negative)
    UNCLEAR_PATTERNS = [
        r"i think",
        r"maybe",
        r"i'm not sure",
        r"possibly",
        r"it could be",
        r"i believe"
    ]

    # Failure handling indicators
    FAILURE_RECOVERY_PHRASES = [
        "however", "alternatively", "instead",
        "unfortunately", "while i cannot", "but i can",
        "let me help you", "here's what we can do"
    ]

    def score_conversation(
        self,
        conversation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Score a conversation on quality dimensions.

        Args:
            conversation: Dict with 'input' (user question) and 'output' (AI response)

        Returns:
            Scores 0-100 for each dimension plus overall score
        """
        user_input = conversation.get("input", "")
        ai_response = conversation.get("output", "")

        if not ai_response:
            return self._empty_scores()

        response_lower = ai_response.lower()
        input_lower = user_input.lower()

        scores = {
            "empathy": self._score_empathy(response_lower),
            "understanding": self._score_understanding(input_lower, response_lower),
            "relevancy": self._score_relevancy(input_lower, response_lower),
            "clarity": self._score_clarity(ai_response),
            "proactiveness": self._score_proactiveness(response_lower),
            "failure_recovery": self._score_failure_recovery(response_lower),
        }

        # Calculate overall score (weighted average)
        weights = {
            "empathy": 0.15,
            "understanding": 0.25,
            "relevancy": 0.25,
            "clarity": 0.15,
            "proactiveness": 0.10,
            "failure_recovery": 0.10,
        }

        overall = sum(scores[dim] * weights[dim] for dim in scores)
        scores["overall"] = round(overall, 1)

        return scores

    def score_batch(
        self,
        conversations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Score multiple conversations"""
        return [self.score_conversation(conv) for conv in conversations]

    def get_dimension_averages(
        self,
        conversations: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Get average scores across all dimensions for a set of conversations"""
        if not conversations:
            return self._empty_scores()

        all_scores = self.score_batch(conversations)

        dimensions = ["empathy", "understanding", "relevancy", "clarity", "proactiveness", "failure_recovery", "overall"]
        averages = {}

        for dim in dimensions:
            values = [s[dim] for s in all_scores]
            averages[dim] = round(sum(values) / len(values), 1)

        return averages

    def _score_empathy(self, response: str) -> float:
        """Score empathy based on presence of empathetic language"""
        matches = sum(1 for phrase in self.EMPATHY_PHRASES if phrase in response)
        # Normalize to 0-100, max score at 4+ matches
        return min(100, matches * 25)

    def _score_understanding(self, input_text: str, response: str) -> float:
        """Score understanding based on keyword overlap and response structure"""
        # Extract key words from input (simple approach)
        input_words = set(word for word in input_text.split() if len(word) > 3)

        if not input_words:
            return 50  # Neutral score for very short inputs

        # Check how many input keywords appear in response
        matches = sum(1 for word in input_words if word in response)
        overlap_ratio = matches / len(input_words)

        # Also check for question word patterns
        question_addressed = 0
        if "how" in input_text and any(w in response for w in ["can", "you can", "to", "by"]):
            question_addressed = 20
        elif "what" in input_text and any(w in response for w in ["is", "are", "the"]):
            question_addressed = 20
        elif "why" in input_text and any(w in response for w in ["because", "reason", "due to"]):
            question_addressed = 20
        elif "when" in input_text and any(w in response for w in ["time", "date", "within"]):
            question_addressed = 20

        return min(100, overlap_ratio * 80 + question_addressed)

    def _score_relevancy(self, input_text: str, response: str) -> float:
        """Score relevancy based on semantic similarity (simplified)"""
        # In production, use embedding similarity
        # Here we use word overlap as approximation

        input_words = set(word.lower() for word in re.findall(r'\b\w+\b', input_text) if len(word) > 2)
        response_words = set(word.lower() for word in re.findall(r'\b\w+\b', response) if len(word) > 2)

        if not input_words:
            return 50

        # Calculate Jaccard-like overlap
        intersection = len(input_words & response_words)
        relevancy = intersection / len(input_words)

        # Check for off-topic indicators
        off_topic_penalty = 0
        off_topic_phrases = ["i don't understand", "not related", "different topic"]
        for phrase in off_topic_phrases:
            if phrase in response:
                off_topic_penalty = 30
                break

        return min(100, max(0, relevancy * 100 - off_topic_penalty))

    def _score_clarity(self, response: str) -> float:
        """Score clarity based on structure and hedging language"""
        score = 100

        # Penalize hedging/unclear language
        for pattern in self.UNCLEAR_PATTERNS:
            if re.search(pattern, response.lower()):
                score -= 10

        # Reward structured responses (lists, paragraphs)
        if re.search(r'\d+\.', response) or re.search(r'•|-\s', response):
            score += 10

        # Penalize very short responses
        if len(response) < 50:
            score -= 20
        # Penalize very long responses (may be unfocused)
        elif len(response) > 2000:
            score -= 10

        # Check sentence structure
        sentences = response.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        if avg_sentence_length > 40:  # Very long sentences
            score -= 15
        elif avg_sentence_length < 5 and len(sentences) > 1:  # Very choppy
            score -= 10

        return max(0, min(100, score))

    def _score_proactiveness(self, response: str) -> float:
        """Score proactiveness based on anticipating user needs"""
        matches = sum(1 for phrase in self.PROACTIVE_PHRASES if phrase in response)
        return min(100, matches * 20)

    def _score_failure_recovery(self, response: str) -> float:
        """Score failure recovery based on handling limitations gracefully"""
        # Check if response indicates inability
        indicates_limitation = any(
            phrase in response for phrase in ["cannot", "can't", "unable", "don't have", "unfortunately"]
        )

        if not indicates_limitation:
            # No failure to recover from, return neutral-high score
            return 70

        # Check for recovery attempts
        recovery_matches = sum(1 for phrase in self.FAILURE_RECOVERY_PHRASES if phrase in response)

        # Also check for alternative suggestions
        alternative_patterns = [
            r"you can",
            r"you could",
            r"try",
            r"suggest",
            r"recommend",
            r"option",
            r"alternative"
        ]
        alternative_matches = sum(1 for pattern in alternative_patterns if re.search(pattern, response))

        return min(100, (recovery_matches + alternative_matches) * 15)

    def _empty_scores(self) -> Dict[str, float]:
        """Return empty/zero scores"""
        return {
            "empathy": 0,
            "understanding": 0,
            "relevancy": 0,
            "clarity": 0,
            "proactiveness": 0,
            "failure_recovery": 0,
            "overall": 0,
        }
