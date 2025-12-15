from celery import Celery
from celery.schedules import crontab

from app.config import settings

# Create Celery app
celery_app = Celery(
    "knowledge_gap_dashboard",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["app.tasks.sync_tasks"],
)

# Celery configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=600,  # 10 minutes max per task
    worker_prefetch_multiplier=1,
    result_expires=3600,  # Results expire after 1 hour
)

# Beat schedule for periodic tasks
celery_app.conf.beat_schedule = {
    # Sync LangFuse data every 5 minutes
    "sync-langfuse-data": {
        "task": "app.tasks.sync_tasks.sync_langfuse_data",
        "schedule": crontab(minute="*/5"),
    },
    # Recalculate gap priorities every hour
    "recalculate-gap-priorities": {
        "task": "app.tasks.sync_tasks.recalculate_gap_priorities",
        "schedule": crontab(minute=0),  # Every hour at minute 0
    },
    # Update topic analytics every 30 minutes
    "update-topic-analytics": {
        "task": "app.tasks.sync_tasks.update_topic_analytics",
        "schedule": crontab(minute="*/30"),
    },
}
