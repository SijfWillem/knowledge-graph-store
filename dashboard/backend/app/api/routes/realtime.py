from fastapi import APIRouter
from fastapi.responses import StreamingResponse
import asyncio
import json
import redis.asyncio as redis
from datetime import datetime

from app.config import settings

router = APIRouter(prefix="/realtime", tags=["realtime"])


async def event_generator():
    """
    Server-Sent Events generator for real-time dashboard updates.
    Listens to Redis pub/sub for events from background tasks.
    """
    redis_client = redis.from_url(settings.redis_url)
    pubsub = redis_client.pubsub()
    await pubsub.subscribe("dashboard_events")

    try:
        # Send initial connection event
        yield f"event: connected\ndata: {json.dumps({'timestamp': datetime.utcnow().isoformat()})}\n\n"

        # Keep connection alive with heartbeat
        heartbeat_interval = 30  # seconds
        last_heartbeat = asyncio.get_event_loop().time()

        while True:
            try:
                # Check for messages with timeout
                message = await asyncio.wait_for(
                    pubsub.get_message(ignore_subscribe_messages=True),
                    timeout=1.0
                )

                if message and message.get("type") == "message":
                    data = message.get("data")
                    if isinstance(data, bytes):
                        data = data.decode("utf-8")

                    event_data = json.loads(data)
                    event_type = event_data.get("type", "update")

                    yield f"event: {event_type}\ndata: {json.dumps(event_data)}\n\n"

                # Send heartbeat
                current_time = asyncio.get_event_loop().time()
                if current_time - last_heartbeat >= heartbeat_interval:
                    yield f"event: heartbeat\ndata: {json.dumps({'timestamp': datetime.utcnow().isoformat()})}\n\n"
                    last_heartbeat = current_time

            except asyncio.TimeoutError:
                # No message received, check for heartbeat
                current_time = asyncio.get_event_loop().time()
                if current_time - last_heartbeat >= heartbeat_interval:
                    yield f"event: heartbeat\ndata: {json.dumps({'timestamp': datetime.utcnow().isoformat()})}\n\n"
                    last_heartbeat = current_time
                continue

    except asyncio.CancelledError:
        pass
    finally:
        await pubsub.unsubscribe("dashboard_events")
        await redis_client.close()


@router.get("/stream")
async def realtime_stream():
    """
    SSE endpoint for live dashboard updates.

    Events:
    - connected: Initial connection established
    - sync_complete: New data synced from LangFuse
    - priorities_updated: Gap priorities recalculated
    - analytics_updated: Topic analytics refreshed
    - heartbeat: Keep-alive signal
    """
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


@router.get("/status")
async def get_realtime_status():
    """Check real-time connection status and Redis health"""
    try:
        redis_client = redis.from_url(settings.redis_url)
        await redis_client.ping()
        await redis_client.close()

        return {
            "status": "healthy",
            "redis": "connected",
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "redis": "disconnected",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
        }
