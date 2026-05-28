"""
jobs.py — Redis-backed job lifecycle for the recommendation queue.

Job lifecycle:
    1. Client POSTs to /api/recommend/* → API creates a job, pushes to Redis queue, returns 202
    2. Worker BLPOPs from the queue, processes the job, stores results
    3. Client polls GET /api/jobs/{job_id} → transitions: queued → running → completed | failed

Jobs are stored as Redis hashes with key `job:{uuid}`.
The queue uses Redis lists: `jobs:high` (text queries) and `jobs:default` (everything else).

Usage:
    from src.jobs import create_job, get_job, mark_running, mark_completed, mark_failed

    job_id = create_job("similar", {"title": "Inception", "limit": 12})
    # Returns uuid string

    job = get_job(job_id)
    # Returns {"id": "...", "type": "similar", "status": "queued", ...}

    mark_running(job_id)
    mark_completed(job_id, results=[...])
    mark_failed(job_id, error="Something went wrong")
"""
from __future__ import annotations

import json
import time
import uuid
from typing import Any

import redis

from .config import REDIS_URL, JOB_TTL_SECONDS, WORKER_POLL_TIMEOUT


def get_redis() -> redis.Redis:
    """Create and return a Redis client."""
    return redis.from_url(REDIS_URL, decode_responses=True)


def create_job(
    job_type: str,
    payload: dict[str, Any],
    redis_client: redis.Redis | None = None,
) -> str:
    """
    Create a new job and push it to the appropriate Redis queue.

    Args:
        job_type: One of "similar", "query", "profile", "diverse", "explore".
        payload: The request payload (will be JSON-serialized).
        redis_client: Optional existing Redis client. Creates one if None.

    Returns:
        The job UUID string.
    """
    r = redis_client or get_redis()
    job_id = str(uuid.uuid4())
    now = time.time()

    job_data = {
        "id": job_id,
        "type": job_type,
        "status": "queued",
        "payload": json.dumps(payload),
        "results": "",
        "error": "",
        "created_at": str(now),
        "started_at": "",
        "finished_at": "",
    }

    # Store job metadata with TTL
    r.hset(f"job:{job_id}", mapping=job_data)
    r.expire(f"job:{job_id}", JOB_TTL_SECONDS)

    # Enqueue: /recommend/query goes to high priority (Ollama is slow),
    # everything else goes to default
    queue_name = "jobs:high" if job_type == "query" else "jobs:default"
    r.lpush(queue_name, job_id)

    return job_id


def get_job(job_id: str, redis_client: redis.Redis | None = None) -> dict[str, Any] | None:
    """
    Retrieve job metadata from Redis.

    Returns:
        Dict with job fields, or None if job doesn't exist (expired or never created).
        The 'payload' and 'results' fields are JSON-decoded if non-empty.
    """
    r = redis_client or get_redis()
    data = r.hgetall(f"job:{job_id}")
    if not data:
        return None

    # Decode JSON fields
    if data.get("payload"):
        try:
            data["payload"] = json.loads(data["payload"])
        except (json.JSONDecodeError, TypeError):
            pass
    if data.get("results"):
        try:
            data["results"] = json.loads(data["results"])
        except (json.JSONDecodeError, TypeError):
            pass

    # Numeric fields
    for key in ("created_at", "started_at", "finished_at"):
        if data.get(key):
            try:
                data[key] = float(data[key])
            except (ValueError, TypeError):
                data[key] = None

    return data


def mark_running(job_id: str, redis_client: redis.Redis | None = None) -> None:
    """Mark a job as running. Called by the worker when it starts processing."""
    r = redis_client or get_redis()
    r.hset(f"job:{job_id}", mapping={
        "status": "running",
        "started_at": str(time.time()),
    })


def mark_completed(
    job_id: str,
    results: list[dict[str, Any]],
    redis_client: redis.Redis | None = None,
) -> None:
    """
    Mark a job as completed and store results.

    Args:
        job_id: The job UUID.
        results: The list of movie result dicts from the recommendation engine.
    """
    r = redis_client or get_redis()
    r.hset(f"job:{job_id}", mapping={
        "status": "completed",
        "results": json.dumps(results),
        "finished_at": str(time.time()),
    })
    # Reset TTL so completed jobs stick around for polling
    r.expire(f"job:{job_id}", JOB_TTL_SECONDS)


def mark_failed(
    job_id: str,
    error: str,
    redis_client: redis.Redis | None = None,
) -> None:
    """Mark a job as failed with an error message."""
    r = redis_client or get_redis()
    r.hset(f"job:{job_id}", mapping={
        "status": "failed",
        "error": error,
        "finished_at": str(time.time()),
    })
    r.expire(f"job:{job_id}", JOB_TTL_SECONDS)


def dequeue_job(timeout: int = WORKER_POLL_TIMEOUT) -> str | None:
    """
    Block and wait for the next job from the priority queues.
    Checks high-priority first, then default.

    Args:
        timeout: Seconds to wait before returning None.

    Returns:
        Job UUID string, or None if timeout.
    """
    r = get_redis()

    # Check high-priority first (non-blocking)
    result = r.rpop("jobs:high")
    if result:
        return result

    # Check default queue (blocking)
    result = r.brpop(["jobs:high", "jobs:default"], timeout=timeout)
    if result:
        # brpop returns (queue_name, value)
        return result[1] if isinstance(result, (list, tuple)) else result
    return None
