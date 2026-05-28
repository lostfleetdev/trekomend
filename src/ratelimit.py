"""
ratelimit.py — Redis-backed per-IP rate limiting for FastAPI.

Uses a sliding-window approach with Redis sorted sets for accurate
per-IP rate limiting. Falls back to a permissive in-memory limiter
if Redis is unavailable (dev mode).

Usage as a FastAPI dependency:
    from src.ratelimit import rate_limit

    @app.get("/api/endpoint")
    def my_endpoint(client_ip: str = Depends(rate_limit)):
        ...

Or as middleware for all routes (applied automatically in main.py).
"""
from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

from fastapi import HTTPException, Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from .config import REDIS_URL, RATE_LIMIT_PER_MINUTE

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


def _get_redis_client() -> "redis.Redis | None":
    """Create a Redis client if available. Returns None if not available."""
    if not REDIS_AVAILABLE:
        return None
    try:
        r = redis.from_url(REDIS_URL, decode_responses=True, socket_connect_timeout=2)
        r.ping()
        return r
    except Exception:
        return None


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware-based rate limiter for all /api/* routes.

    Applies a sliding-window per-IP limit. Adds rate-limit headers
    to every response:
        X-RateLimit-Limit:     max requests per window
        X-RateLimit-Remaining: requests left in current window
        X-RateLimit-Reset:     Unix timestamp when window resets

    Returns 429 Too Many Requests when limit is exceeded.
    """

    def __init__(self, app, requests_per_minute: int = RATE_LIMIT_PER_MINUTE):
        super().__init__(app)
        self.rpm = requests_per_minute
        self._redis = _get_redis_client()
        self._local_counts: dict[str, list[float]] = defaultdict(list)

        if self._redis:
            print(f"  Rate limiting: Redis-backed sliding window ({self.rpm} req/min)")
        else:
            print(f"  Rate limiting: in-memory fallback ({self.rpm} req/min) "
                  f"— Redis not available")

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP, respecting X-Forwarded-For from trusted proxies."""
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _check_rate_limit_redis(self, ip: str) -> tuple[int, int]:
        """Check rate limit via Redis sorted set. Returns (remaining, reset_ts)."""
        now = time.time()
        window = 60.0  # 1 minute window
        key = f"rl:{ip}"
        cutoff = now - window

        pipe = self._redis.pipeline()
        # Remove entries outside the window
        pipe.zremrangebyscore(key, 0, cutoff)
        # Count current entries
        pipe.zcard(key)
        # Add current request
        pipe.zadd(key, {str(now): now})
        # Expire the key
        pipe.expire(key, int(window) + 1)
        results = pipe.execute()

        current = results[1]  # zcard result
        remaining = max(0, self.rpm - current)
        reset_ts = int(now + window)
        return remaining, reset_ts

    def _check_rate_limit_local(self, ip: str) -> tuple[int, int]:
        """In-memory fallback rate limiter."""
        now = time.time()
        window = 60.0
        cutoff = now - window

        # Prune old entries
        self._local_counts[ip] = [
            t for t in self._local_counts[ip] if t > cutoff
        ]

        current = len(self._local_counts[ip])
        remaining = max(0, self.rpm - current)
        reset_ts = int(now + window)

        if current < self.rpm:
            self._local_counts[ip].append(now)
        return remaining, reset_ts

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Only rate-limit /api/* routes
        if not request.url.path.startswith("/api"):
            return await call_next(request)

        # Skip rate limiting for health checks
        if request.url.path in ("/api/health",):
            return await call_next(request)

        # Skip rate limiting entirely when Redis is unavailable (local dev)
        if not self._redis:
            return await call_next(request)

        ip = self._get_client_ip(request)

        if self._redis:
            try:
                remaining, reset_ts = self._check_rate_limit_redis(ip)
            except Exception:
                # Redis down → fall through to local
                remaining, reset_ts = self._check_rate_limit_local(ip)
        else:
            remaining, reset_ts = self._check_rate_limit_local(ip)

        if remaining == 0:
            return Response(
                content='{"detail":"Rate limit exceeded. Try again shortly."}',
                status_code=429,
                media_type="application/json",
                headers={
                    "X-RateLimit-Limit": str(self.rpm),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(reset_ts),
                    "Retry-After": "60",
                },
            )

        response = await call_next(request)

        # Add rate-limit headers to all API responses
        response.headers["X-RateLimit-Limit"] = str(self.rpm)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(reset_ts)

        return response
