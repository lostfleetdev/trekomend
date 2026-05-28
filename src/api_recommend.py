"""
api_recommend.py — Expensive recommendation endpoints with job queue.

These endpoints submit jobs to Redis and return 202 Accepted immediately.
The actual FAISS + Phase 2 pipeline runs in a separate worker process.

Validation (movie existence, input bounds) happens synchronously before
the job is created — the user gets immediate 400/404 for bad input.

Endpoints:
    POST /api/recommend/similar    Movies like a given title
    POST /api/recommend/query      Movies matching a text description
    POST /api/recommend/profile    Profile from liked/disliked films + mood
    POST /api/recommend/diverse    Maximum diversity mode (DPP)
    POST /api/recommend/explore    Serendipity and novelty
    GET  /api/jobs/{job_id}        Poll job status and retrieve results
"""
from __future__ import annotations

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


def set_recommend_state(searcher, hdf5_path: str = ""):
    """Inject shared state from main.py lifespan."""
    global _searcher, _hdf5_path
    _searcher = searcher
    _hdf5_path = hdf5_path


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


class MovieResult(BaseModel):
    rank: int
    tmdb_id: int
    title: str
    primary_genre: str
    genres: str | None = None
    year: int | None = None
    overview: str | None = None
    vote_average: float | None = None
    score: float | None = None


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
# Recommendation endpoints (job queue)
# ============================================================================

@router.post(
    "/recommend/similar",
    status_code=202,
    response_model=dict[str, Any],
)
def recommend_similar(req: SimilarRequest) -> dict[str, Any]:
    """Movies similar to a given title. Validates input, queues the job."""
    _validate_movie_exists(req.title)

    job_id = create_job("similar", req.model_dump())
    return {
        "job_id": job_id,
        "status": "queued",
        "type": "similar",
        "message": f"Queued recommendation for '{req.title}'",
    }


@router.post(
    "/recommend/query",
    status_code=202,
    response_model=dict[str, Any],
)
def recommend_query(req: QueryRequest) -> dict[str, Any]:
    """Movies matching a text description. Uses Ollama for embedding."""
    if not req.query.strip():
        raise HTTPException(400, "Query cannot be empty")

    job_id = create_job("query", req.model_dump())
    return {
        "job_id": job_id,
        "status": "queued",
        "type": "query",
        "message": f"Queued text search: '{req.query[:50]}...' (uses Ollama, may take 3-10s)",
    }


@router.post(
    "/recommend/profile",
    status_code=202,
    response_model=dict[str, Any],
)
def recommend_profile(req: ProfileRequest) -> dict[str, Any]:
    """Profile-based recommendations from liked movies, mood, and dislikes."""
    _validate_movies_exist(req.liked)
    # Disliked movies are optional — skip validation if a dislike is not found

    job_id = create_job("profile", req.model_dump())
    return {
        "job_id": job_id,
        "status": "queued",
        "type": "profile",
        "message": f"Queued profile from {len(req.liked)} liked movies",
    }


@router.post(
    "/recommend/diverse",
    status_code=202,
    response_model=dict[str, Any],
)
def recommend_diverse(req: DiverseRequest) -> dict[str, Any]:
    """Maximum diversity mode. Lower lambda_qd for more variety."""
    _validate_movie_exists(req.title)

    job_id = create_job("diverse", req.model_dump())
    return {
        "job_id": job_id,
        "status": "queued",
        "type": "diverse",
        "message": f"Queued diverse recommendation for '{req.title}'",
    }


@router.post(
    "/recommend/explore",
    status_code=202,
    response_model=dict[str, Any],
)
def recommend_explore(req: ExploreRequest) -> dict[str, Any]:
    """Serendipity mode. Favors less popular, less obvious movies."""
    _validate_movies_exist(req.liked)

    job_id = create_job("explore", req.model_dump())
    return {
        "job_id": job_id,
        "status": "queued",
        "type": "explore",
        "message": f"Queued serendipity from {len(req.liked)} liked movies",
    }


# ============================================================================
# Job status polling
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
