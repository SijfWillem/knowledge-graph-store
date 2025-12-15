from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_
import logging

from app.models.database import Trace, Score, KnowledgeGap, TopicAnalytics

logger = logging.getLogger(__name__)


class AnalyticsService:
    """Aggregates metrics for dashboard display"""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_overview(
        self,
        from_date: datetime,
        to_date: datetime
    ) -> Dict[str, Any]:
        """
        Returns overview metrics:
        {
            "total_conversations": int,
            "success_rate": float,  # % with positive feedback
            "avg_csat": float,
            "critical_gaps_count": int,
            "handover_rate": float,
            "trends": {...}
        }
        """
        # Get total conversations in period
        total_query = select(func.count(Trace.id)).where(
            and_(
                Trace.timestamp >= from_date,
                Trace.timestamp <= to_date
            )
        )
        total_result = await self.db.execute(total_query)
        total_conversations = total_result.scalar() or 0

        # Get success rate (conversations without negative feedback AND without missing knowledge)
        from sqlalchemy import or_
        success_query = select(func.count(Trace.id)).where(
            and_(
                Trace.timestamp >= from_date,
                Trace.timestamp <= to_date,
                Trace.feedback_negative == False,
                or_(Trace.has_missing_knowledge == False, Trace.has_missing_knowledge.is_(None))
            )
        )
        success_result = await self.db.execute(success_query)
        success_count = success_result.scalar() or 0
        success_rate = (success_count / total_conversations * 100) if total_conversations > 0 else 0

        # Get average CSAT
        csat_query = select(func.avg(Score.value)).where(
            and_(
                Score.timestamp >= from_date,
                Score.timestamp <= to_date,
                Score.name.ilike('%csat%')
            )
        )
        csat_result = await self.db.execute(csat_query)
        avg_csat = csat_result.scalar() or 0

        # Get critical gaps count
        gaps_query = select(func.count(KnowledgeGap.id)).where(
            and_(
                KnowledgeGap.resolved == False,
                KnowledgeGap.priority_level == 'critical'
            )
        )
        gaps_result = await self.db.execute(gaps_query)
        critical_gaps_count = gaps_result.scalar() or 0

        # Calculate handover rate (traces with handover tag or metadata)
        # For JSON columns in PostgreSQL, use cast and text matching
        from sqlalchemy import cast, String, text
        handover_query = select(func.count(Trace.id)).where(
            and_(
                Trace.timestamp >= from_date,
                Trace.timestamp <= to_date,
                cast(Trace.tags, String).ilike('%handover%')
            )
        )
        handover_result = await self.db.execute(handover_query)
        handover_count = handover_result.scalar() or 0
        handover_rate = (handover_count / total_conversations * 100) if total_conversations > 0 else 0

        # Calculate trends vs previous period
        period_length = to_date - from_date
        prev_from = from_date - period_length
        prev_to = from_date

        prev_total_query = select(func.count(Trace.id)).where(
            and_(
                Trace.timestamp >= prev_from,
                Trace.timestamp <= prev_to
            )
        )
        prev_total_result = await self.db.execute(prev_total_query)
        prev_total = prev_total_result.scalar() or 0

        conversation_trend = self._calculate_trend(total_conversations, prev_total)

        return {
            "total_conversations": total_conversations,
            "success_rate": round(success_rate, 2),
            "avg_csat": round(avg_csat, 2) if avg_csat else None,
            "critical_gaps_count": critical_gaps_count,
            "handover_rate": round(handover_rate, 2),
            "trends": {
                "conversations": conversation_trend,
            },
            "period": {
                "from": from_date.isoformat(),
                "to": to_date.isoformat(),
            }
        }

    async def get_topics_analytics(self) -> List[Dict[str, Any]]:
        """
        Per-topic breakdown:
        - question_count
        - success_rate
        - handover_rate
        - avg_confidence
        - trend (vs previous period)
        - priority (critical/high/medium/low)
        """
        query = select(TopicAnalytics).order_by(TopicAnalytics.question_count.desc())
        result = await self.db.execute(query)
        topics = result.scalars().all()

        return [
            {
                "topic": t.topic,
                "question_count": t.question_count,
                "success_rate": round(t.success_rate, 2),
                "handover_rate": round(t.handover_rate, 2),
                "avg_confidence": round(t.avg_confidence, 2),
                "avg_csat": round(t.avg_csat, 2) if t.avg_csat else None,
                "trend": round(t.trend, 2),
                "priority": t.priority,
            }
            for t in topics
        ]

    async def get_satisfaction_data(
        self,
        from_date: datetime,
        to_date: datetime
    ) -> Dict[str, Any]:
        """Thumbs up/down counts, CSAT distribution"""
        # Get thumbs up/down
        thumbs_up_query = select(func.count(Score.id)).where(
            and_(
                Score.timestamp >= from_date,
                Score.timestamp <= to_date,
                Score.name.ilike('%thumb%'),
                Score.value > 0
            )
        )
        thumbs_up_result = await self.db.execute(thumbs_up_query)
        thumbs_up = thumbs_up_result.scalar() or 0

        thumbs_down_query = select(func.count(Score.id)).where(
            and_(
                Score.timestamp >= from_date,
                Score.timestamp <= to_date,
                Score.name.ilike('%thumb%'),
                Score.value <= 0
            )
        )
        thumbs_down_result = await self.db.execute(thumbs_down_query)
        thumbs_down = thumbs_down_result.scalar() or 0

        # Get CSAT distribution (1-5)
        csat_distribution = {}
        for rating in range(1, 6):
            count_query = select(func.count(Score.id)).where(
                and_(
                    Score.timestamp >= from_date,
                    Score.timestamp <= to_date,
                    Score.name.ilike('%csat%'),
                    Score.value >= rating,
                    Score.value < rating + 1
                )
            )
            count_result = await self.db.execute(count_query)
            csat_distribution[rating] = count_result.scalar() or 0

        return {
            "thumbs": {
                "up": thumbs_up,
                "down": thumbs_down,
                "ratio": round(thumbs_up / (thumbs_up + thumbs_down), 2) if (thumbs_up + thumbs_down) > 0 else 0
            },
            "csat_distribution": csat_distribution,
            "period": {
                "from": from_date.isoformat(),
                "to": to_date.isoformat(),
            }
        }

    async def get_trends(
        self,
        from_date: datetime,
        to_date: datetime,
        granularity: str = "day"
    ) -> List[Dict[str, Any]]:
        """
        Time-series data for trend charts.

        Args:
            from_date: Start date
            to_date: End date
            granularity: "hour", "day", or "week"
        """
        # Determine the interval
        if granularity == "hour":
            interval = timedelta(hours=1)
            date_format = "%Y-%m-%d %H:00"
        elif granularity == "week":
            interval = timedelta(weeks=1)
            date_format = "%Y-W%W"
        else:  # day
            interval = timedelta(days=1)
            date_format = "%Y-%m-%d"

        trends = []
        current = from_date

        while current <= to_date:
            next_period = current + interval

            # Count conversations in this period
            count_query = select(func.count(Trace.id)).where(
                and_(
                    Trace.timestamp >= current,
                    Trace.timestamp < next_period
                )
            )
            count_result = await self.db.execute(count_query)
            total = count_result.scalar() or 0

            # Count successful (no negative feedback AND no missing knowledge)
            from sqlalchemy import or_
            success_query = select(func.count(Trace.id)).where(
                and_(
                    Trace.timestamp >= current,
                    Trace.timestamp < next_period,
                    Trace.feedback_negative == False,
                    or_(Trace.has_missing_knowledge == False, Trace.has_missing_knowledge.is_(None))
                )
            )
            success_result = await self.db.execute(success_query)
            success = success_result.scalar() or 0

            trends.append({
                "date": current.strftime(date_format),
                "timestamp": current.isoformat(),
                "total_conversations": total,
                "successful_conversations": success,
                "success_rate": round(success / total * 100, 2) if total > 0 else 0,
            })

            current = next_period

        return trends

    async def update_topic_analytics(self):
        """
        Recalculate and update topic analytics.
        Called periodically by background jobs.
        """
        # Get all unique topics from traces
        topics_query = select(Trace.topic).distinct().where(Trace.topic.isnot(None))
        topics_result = await self.db.execute(topics_query)
        topics = [row[0] for row in topics_result.fetchall()]

        for topic in topics:
            # Calculate metrics for this topic
            total_query = select(func.count(Trace.id)).where(Trace.topic == topic)
            total_result = await self.db.execute(total_query)
            total = total_result.scalar() or 0

            from sqlalchemy import or_
            success_query = select(func.count(Trace.id)).where(
                and_(
                    Trace.topic == topic,
                    Trace.feedback_negative == False,
                    or_(Trace.has_missing_knowledge == False, Trace.has_missing_knowledge.is_(None))
                )
            )
            success_result = await self.db.execute(success_query)
            success = success_result.scalar() or 0

            avg_conf_query = select(func.avg(Trace.topic_confidence)).where(Trace.topic == topic)
            avg_conf_result = await self.db.execute(avg_conf_query)
            avg_confidence = avg_conf_result.scalar() or 0

            # Update or create analytics record
            existing_query = select(TopicAnalytics).where(TopicAnalytics.topic == topic)
            existing_result = await self.db.execute(existing_query)
            existing = existing_result.scalar_one_or_none()

            if existing:
                existing.question_count = total
                existing.success_count = success
                existing.failure_count = total - success
                existing.success_rate = (success / total * 100) if total > 0 else 0
                existing.avg_confidence = avg_confidence
                existing.updated_at = datetime.utcnow()
            else:
                new_analytics = TopicAnalytics(
                    topic=topic,
                    question_count=total,
                    success_count=success,
                    failure_count=total - success,
                    success_rate=(success / total * 100) if total > 0 else 0,
                    avg_confidence=avg_confidence,
                )
                self.db.add(new_analytics)

        await self.db.commit()

    def _calculate_trend(self, current: int, previous: int) -> float:
        """Calculate percentage change between periods"""
        if previous == 0:
            return 100.0 if current > 0 else 0.0
        return round((current - previous) / previous * 100, 2)
