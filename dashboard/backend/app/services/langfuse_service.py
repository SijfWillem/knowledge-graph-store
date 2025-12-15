from langfuse import Langfuse
from datetime import datetime
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class LangFuseService:
    """
    Connects to LangFuse to fetch conversation traces.

    Key methods:
    - fetch_traces(): Get traces with input/output/feedback
    - fetch_scores(): Get thumbs up/down, CSAT scores
    - sync_new_data(): Incremental sync since last fetch
    """

    def __init__(self, public_key: str, secret_key: str, host: str):
        self.client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host
        )

    async def fetch_traces(
        self,
        from_timestamp: Optional[datetime] = None,
        to_timestamp: Optional[datetime] = None,
        limit: int = 100,
        user_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch traces and extract:
        - trace_id, session_id
        - input (user question)
        - output (AI response)
        - latency, token usage
        - Any scores/feedback attached
        """
        try:
            # Build query parameters
            params = {"limit": limit}
            if from_timestamp:
                params["from_timestamp"] = from_timestamp
            if to_timestamp:
                params["to_timestamp"] = to_timestamp
            if user_id:
                params["user_id"] = user_id
            if tags:
                params["tags"] = tags

            # Use async API for non-blocking operations
            traces_response = await self.client.async_api.trace.list(**params)

            processed_traces = []
            for trace in traces_response.data:
                # Handle different LangFuse SDK versions - use getattr for safety
                usage = getattr(trace, 'usage', None)
                trace_data = {
                    "trace_id": trace.id,
                    "session_id": getattr(trace, 'session_id', None),
                    "user_id": getattr(trace, 'user_id', None),
                    "input": getattr(trace, 'input', None),
                    "output": getattr(trace, 'output', None),
                    "name": getattr(trace, 'name', None),
                    "tags": getattr(trace, 'tags', None) or [],
                    "metadata": getattr(trace, 'metadata', None) or {},
                    "timestamp": getattr(trace, 'timestamp', None),
                    "latency": self._calculate_latency(trace),
                    "total_cost": getattr(trace, 'total_cost', None),
                    "usage": {
                        "prompt_tokens": getattr(usage, 'input', 0) if usage else 0,
                        "completion_tokens": getattr(usage, 'output', 0) if usage else 0,
                        "total_tokens": getattr(usage, 'total', 0) if usage else 0,
                    },
                }
                processed_traces.append(trace_data)

            return processed_traces

        except Exception as e:
            logger.error(f"Error fetching traces: {e}")
            raise

    async def fetch_trace_by_id(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single trace by ID with full details"""
        try:
            trace = await self.client.async_api.trace.get(trace_id)
            return {
                "trace_id": trace.id,
                "session_id": getattr(trace, 'session_id', None),
                "user_id": getattr(trace, 'user_id', None),
                "input": getattr(trace, 'input', None),
                "output": getattr(trace, 'output', None),
                "name": getattr(trace, 'name', None),
                "tags": getattr(trace, 'tags', None) or [],
                "metadata": getattr(trace, 'metadata', None) or {},
                "timestamp": getattr(trace, 'timestamp', None),
                "observations": getattr(trace, 'observations', None) or [],
                "scores": getattr(trace, 'scores', None) or [],
            }
        except Exception as e:
            logger.error(f"Error fetching trace {trace_id}: {e}")
            return None

    async def fetch_scores_for_traces(
        self, trace_ids: List[str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all feedback (thumbs, CSAT) for traces.

        Returns a dict mapping trace_id to list of scores.
        """
        scores_by_trace: Dict[str, List[Dict[str, Any]]] = {
            tid: [] for tid in trace_ids
        }

        try:
            for trace_id in trace_ids:
                trace = await self.client.async_api.trace.get(trace_id)
                if trace.scores:
                    for score in trace.scores:
                        score_data = {
                            "score_id": score.id,
                            "name": score.name,
                            "value": score.value,
                            "comment": score.comment,
                            "source": score.source,
                            "timestamp": score.timestamp,
                        }
                        scores_by_trace[trace_id].append(score_data)

            return scores_by_trace

        except Exception as e:
            logger.error(f"Error fetching scores: {e}")
            raise

    async def fetch_observations(
        self,
        trace_id: str,
        observation_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Fetch observations (generations, spans) for a trace.

        Args:
            trace_id: The trace to fetch observations for
            observation_type: Filter by type (GENERATION, SPAN, EVENT)
            limit: Maximum observations to return
        """
        try:
            params = {"trace_id": trace_id, "limit": limit}
            if observation_type:
                params["type"] = observation_type

            observations = await self.client.async_api.observations.get_many(**params)

            result = []
            for obs in observations.data:
                usage = getattr(obs, 'usage', None)
                result.append({
                    "observation_id": obs.id,
                    "trace_id": getattr(obs, 'trace_id', None),
                    "type": getattr(obs, 'type', None),
                    "name": getattr(obs, 'name', None),
                    "input": getattr(obs, 'input', None),
                    "output": getattr(obs, 'output', None),
                    "model": getattr(obs, 'model', None),
                    "model_parameters": getattr(obs, 'model_parameters', None),
                    "usage": {
                        "prompt_tokens": getattr(usage, 'input', 0) if usage else 0,
                        "completion_tokens": getattr(usage, 'output', 0) if usage else 0,
                        "total_tokens": getattr(usage, 'total', 0) if usage else 0,
                    },
                    "start_time": getattr(obs, 'start_time', None),
                    "end_time": getattr(obs, 'end_time', None),
                    "level": getattr(obs, 'level', None),
                    "status_message": getattr(obs, 'status_message', None),
                })
            return result

        except Exception as e:
            logger.error(f"Error fetching observations for trace {trace_id}: {e}")
            raise

    async def sync_new_data(
        self,
        last_sync_timestamp: datetime,
        batch_size: int = 100,
    ) -> Dict[str, Any]:
        """
        Incremental sync of new data since last fetch.

        Returns:
            Dict with new traces, their scores, and sync metadata
        """
        new_traces = await self.fetch_traces(
            from_timestamp=last_sync_timestamp,
            limit=batch_size,
        )

        if not new_traces:
            return {
                "traces": [],
                "scores": {},
                "sync_timestamp": datetime.utcnow(),
                "count": 0,
            }

        trace_ids = [t["trace_id"] for t in new_traces]
        scores = await self.fetch_scores_for_traces(trace_ids)

        # Attach scores to traces
        for trace in new_traces:
            trace["scores"] = scores.get(trace["trace_id"], [])
            trace["feedback_negative"] = self._has_negative_feedback(trace["scores"])

        return {
            "traces": new_traces,
            "scores": scores,
            "sync_timestamp": datetime.utcnow(),
            "count": len(new_traces),
        }

    def _calculate_latency(self, trace) -> Optional[float]:
        """Calculate latency in seconds from trace timing"""
        if hasattr(trace, "latency") and trace.latency is not None:
            return trace.latency
        return None

    def _has_negative_feedback(self, scores: List[Dict[str, Any]]) -> bool:
        """
        Determine if scores indicate negative feedback.

        Checks for:
        - Thumbs down (value = 0 or negative)
        - Low CSAT scores (< 3 out of 5)
        - Explicit negative feedback names
        """
        for score in scores:
            name = score.get("name", "").lower()
            value = score.get("value")

            if value is None:
                continue

            # Thumbs up/down scoring
            if "thumb" in name or "feedback" in name:
                if value <= 0:
                    return True

            # CSAT scoring (typically 1-5 scale)
            if "csat" in name or "satisfaction" in name:
                if value < 3:
                    return True

            # Generic negative score detection
            if "quality" in name or "rating" in name:
                if value < 0.5:  # Assuming normalized 0-1 scale
                    return True

        return False

    async def get_sessions(
        self,
        limit: int = 50,
        from_timestamp: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch conversation sessions"""
        try:
            params = {"limit": limit}
            if from_timestamp:
                params["from_timestamp"] = from_timestamp

            sessions = await self.client.async_api.sessions.list(**params)

            return [
                {
                    "session_id": session.id,
                    "user_id": session.user_id,
                    "created_at": session.created_at,
                    "project_id": session.project_id,
                }
                for session in sessions.data
            ]

        except Exception as e:
            logger.error(f"Error fetching sessions: {e}")
            raise

    def shutdown(self):
        """Flush any pending data and shutdown the client"""
        self.client.flush()
