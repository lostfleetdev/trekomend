"""
api_recommend.py — Recommendation endpoints with optional job queue.

When Redis is available: submits jobs to the queue, returns 202 with job_id.
When Redis is unavailable (local dev): runs recommendations synchronously,
returns 200 with results directly.

Endpoints:
    POST /api/recommend/similar    Movies like a given title
    POST /api/recommend/query      Movies matching a text description
    POST /api/recommend/profile    Profile from liked/disliked films + mood
    POST /api/recommend/diverse    Maximum diversity mode (DPP)
    POST /api/recommend/explore    Serendipity and novelty
    GET  /api/jobs/{job_id}        Poll job status (only when queue is active)
"""
from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .jobs import create_job, get_job

router = APIRouter(prefix="/api")


# ============================================================================
# State references — set by main.py via set_recommend_state()
# ============================================================================

_searcher = None
_hdf5_path: str = ""
_redis_available: bool = False


def set_recommend_state(searcher, hdf5_path: str = ""):
    """Inject shared state from main.py lifespan."""
    global _searcher, _hdf5_path
    _searcher = searcher
    _hdf5_path = hdf5_path


def set_redis_available(available: bool):
    """Called by main.py after checking Redis connectivity."""
    global _redis_available
    _redis_available = available


# ============================================================================
# Redis check
# ============================================================================

def _check_redis() -> bool:
    """Return True if Redis is reachable and jobs can be queued."""
    global _redis_available
    if not _redis_available:
        return False
    try:
        from .jobs import get_redis
        r = get_redis()
        r.ping()
        _redis_available = True
        return True
    except Exception:
        _redis_available = False
        return False


# ============================================================================
# Synchronous fallback (no Redis)
# ============================================================================

def _run_sync(job_type: str, payload: dict) -> list[dict]:
    """
    Run a recommendation synchronously using the worker's processor functions.
    This is the fallback when Redis is unavailable (local dev).
    """
    import src.worker as worker

    # Wire up the worker's globals with the API's state
    worker.searcher = _searcher
    worker.hdf5_path = _hdf5_path
    worker.phase2_loaded = True  # API already loaded Phase 2

    # Import Phase 2 modules from the API's state
    from . import api_read
    worker.feature_builder = api_read._feature_builder
    worker.lightgbm_model = api_read._lightgbm_model

    # Lazy-import diversity modules
    try:
        from .diversity import GenreRoundRobin, DPPSelector, RoundRobinConfig, DPPConfig
        from .config import (
            GENRE_MAX_PER_GENRE, GENRE_MIN_UNIQUE, SERENDIPITY_POSITION,
            DPP_RANK, DPP_LAMBDA_QD,
        )
        if worker.genre_round_robin is None:
            rr_cfg = RoundRobinConfig(
                max_per_genre=GENRE_MAX_PER_GENRE,
                min_unique_genres=GENRE_MIN_UNIQUE,
                serendipity_position=SERENDIPITY_POSITION,
            )
            worker.genre_round_robin = GenreRoundRobin(_searcher._db, config=rr_cfg)
        if worker.dpp_selector is None:
            dpp_cfg = DPPConfig(rank=DPP_RANK, lambda_qd=DPP_LAMBDA_QD, kernel_mode="full")
            worker.dpp_selector = DPPSelector(config=dpp_cfg)
    except Exception:
        pass

    processor = worker.PROCESSORS.get(job_type)
    if processor is None:
        raise ValueError(f"Unknown recommendation type: {job_type}")

    return processor(payload)


# ============================================================================
# Pydantic models for request validation
# ============================================================================

class SimilarRequest(BaseModel):
    title: str = Field(..., description="Movie title to find similar to")
    limit: int = Field(default=12, ge=1, le=100)
    nprobe: int | None = Field(default=None, ge=1, le=2048)
    use_phase2: bool = Field(default=True)


class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language query")
    limit: int = Field(default=12, ge=1, le=100)
    nprobe: int | None = Field(default=None, ge=1, le=2048)
    use_phase2: bool = Field(default=True)


class ProfileRequest(BaseModel):
    liked: list[str] = Field(..., min_length=1, description="Movies the user likes")
    mood: str | None = Field(default=None, description="Mood or context text")
    dislike: list[str] = Field(default_factory=list, description="Movies to avoid")
    mood_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    limit: int = Field(default=12, ge=1, le=100)
    nprobe: int | None = Field(default=None, ge=1, le=2048)
    use_phase2: bool = Field(default=True)


class DiverseRequest(BaseModel):
    title: str = Field(..., description="Movie title to find similar to")
    limit: int = Field(default=12, ge=1, le=50)
    lambda_qd: float = Field(default=0.5, ge=0.0, le=1.0,
                              description="Quality/diversity tradeoff (lower = more diverse)")
    nprobe: int | None = Field(default=None, ge=1, le=2048)


class ExploreRequest(BaseModel):
    liked: list[str] = Field(..., min_length=1, description="Movies the user likes")
    limit: int = Field(default=12, ge=1, le=50)
    nprobe: int | None = Field(default=None, ge=1, le=2048)


# ============================================================================
# Validation helpers
# ============================================================================

def _validate_movie_exists(title: str) -> int:
    """Validate that a movie title exists in the database. Returns TMDB ID."""
    if not _searcher:
        raise HTTPException(503, "Searcher not initialized")
    movie = _searcher.lookup_by_title(title)
    if movie is None:
        raise HTTPException(404, f"Movie not found: '{title}'")
    return movie["id"]


def _validate_movies_exist(titles: list[str]) -> list[dict]:
    """Validate that all movie titles exist. Returns list of movie dicts."""
    if not _searcher:
        raise HTTPException(503, "Searcher not initialized")
    movies = []
    for title in titles:
        movie = _searcher.lookup_by_title(title)
        if movie is None:
            raise HTTPException(404, f"Movie not found: '{title}'")
        movies.append(movie)
    return movies


# ============================================================================
# Unified dispatch: queue if Redis, sync if not
# ============================================================================

def _dispatch(job_type: str, payload: dict, sync_msg: str) -> dict[str, Any]:
    """
    Queue a job if Redis is available, otherwise run synchronously.
    Returns a response dict with either job info or direct results.
    """
    if _check_redis():
        try:
            job_id = create_job(job_type, payload)
            return {
                "job_id": job_id,
                "status": "queued",
                "type": job_type,
                "message": f"Queued {job_type} recommendation",
                "mode": "queue",
            }
        except Exception:
            pass

    # Synchronous fallback (no Redis)
    start = time.time()
    try:
        results = _run_sync(job_type, payload)
        elapsed = time.time() - start
        return {
            "job_id": None,
            "status": "completed",
            "type": job_type,
            "message": sync_msg,
            "mode": "sync",
            "elapsed_seconds": round(elapsed, 2),
            "results": results,
        }
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Recommendation failed: {type(e).__name__}: {e}")


# ============================================================================
# Recommendation endpoints
# ============================================================================

@router.post("/recommend/similar", response_model=dict[str, Any])
def recommend_similar(req: SimilarRequest) -> dict[str, Any]:
    """Movies similar to a given title."""
    _validate_movie_exists(req.title)
    return _dispatch(
        "similar",
        req.model_dump(),
        f"Found movies similar to '{req.title}'",
    )


@router.post("/recommend/query", response_model=dict[str, Any])
def recommend_query(req: QueryRequest) -> dict[str, Any]:
    """Movies matching a text description. Uses Ollama for embedding."""
    if not req.query.strip():
        raise HTTPException(400, "Query cannot be empty")
    return _dispatch(
        "query",
        req.model_dump(),
        f"Found movies for: '{req.query[:50]}'",
    )


@router.post("/recommend/profile", response_model=dict[str, Any])
def recommend_profile(req: ProfileRequest) -> dict[str, Any]:
    """Profile-based recommendations from liked movies, mood, and dislikes."""
    _validate_movies_exist(req.liked)
    return _dispatch(
        "profile",
        req.model_dump(),
        f"Built profile from {len(req.liked)} liked movies",
    )


@router.post("/recommend/diverse", response_model=dict[str, Any])
def recommend_diverse(req: DiverseRequest) -> dict[str, Any]:
    """Maximum diversity mode. Lower lambda_qd for more variety."""
    _validate_movie_exists(req.title)
    return _dispatch(
        "diverse",
        req.model_dump(),
        f"Found diverse picks for '{req.title}'",
    )


@router.post("/recommend/explore", response_model=dict[str, Any])
def recommend_explore(req: ExploreRequest) -> dict[str, Any]:
    """Serendipity mode. Favors less popular, less obvious movies."""
    _validate_movies_exist(req.liked)
    return _dispatch(
        "explore",
        req.model_dump(),
        f"Found serendipity picks from {len(req.liked)} liked movies",
    )


# ============================================================================
# Job status polling (only meaningful with Redis queue)
# ============================================================================

@router.get("/jobs/{job_id}")
def get_job_status(job_id: str) -> dict[str, Any]:
    """
    Poll job status. Returns current state and results if completed.

    Response shapes:
        queued:    {"id": "...", "status": "queued", ...}
        running:   {"id": "...", "status": "running", ...}
        completed: {"id": "...", "status": "completed", "results": [...]}
        failed:    {"id": "...", "status": "failed", "error": "..."}
        expired:   404 {"detail": "Job not found or expired"}
    """
    job = get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Job not found or expired. Jobs expire after 5 minutes.",
        )
    return job
