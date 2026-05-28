"""
api_read.py — Cheap (synchronous) read-only endpoints.

These endpoints run without the job queue. They are fast SQLite lookups
(~1-3ms) and don't require FAISS heavy computation.

Endpoints:
    GET  /api/health                Liveness check
    GET  /api/stats                 Index stats
    GET  /api/movies/search         Full-text search (FTS5)
    GET  /api/movies/genres         All primary genres
    GET  /api/movies/{tmdb_id}      Single movie by TMDB ID
    GET  /api/ranker/features       LightGBM feature importance
"""
from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from .config import LIGHTGBM_MODEL_PATH

router = APIRouter(prefix="/api")


# ============================================================================
# State references — set by main.py via set_read_state()
# ============================================================================

_searcher = None
_feature_builder = None
_lightgbm_model = None


def set_read_state(searcher, feature_builder=None, lightgbm_model=None):
    """Inject shared state from main.py lifespan."""
    global _searcher, _feature_builder, _lightgbm_model
    _searcher = searcher
    _feature_builder = feature_builder
    _lightgbm_model = lightgbm_model


# ============================================================================
# Health & Stats
# ============================================================================

@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/stats")
def stats() -> dict[str, Any]:
    if not _searcher:
        raise HTTPException(503, "Searcher not initialized")
    return _searcher.stats()


# ============================================================================
# Movie lookup and browsing
# ============================================================================

@router.get("/movies/search")
def search_movies(
    q: str = Query(..., description="Search query"),
    limit: int = Query(default=20, ge=1, le=100),
) -> list[dict[str, Any]]:
    """Full-text search across titles, overviews, keywords, and genres."""
    if not _searcher:
        raise HTTPException(503, "Searcher not initialized")
    return _searcher.search_text(q, limit=limit)


@router.get("/movies/genres")
def list_genres() -> list[str]:
    """All distinct primary genres in the database."""
    if not _searcher:
        raise HTTPException(503, "Searcher not initialized")
    return _searcher.get_genres()


@router.get("/movies/{tmdb_id}")
def get_movie(tmdb_id: int) -> dict[str, Any]:
    """A single movie by TMDB ID."""
    if not _searcher:
        raise HTTPException(503, "Searcher not initialized")
    movie = _searcher._lookup_movie(tmdb_id)
    if movie is None:
        raise HTTPException(404, f"Movie #{tmdb_id} not found")
    return movie


# ============================================================================
# Ranker info
# ============================================================================

@router.get("/ranker/features")
def ranker_features() -> dict[str, Any]:
    """Feature importance for the LightGBM re-ranker."""
    if not _lightgbm_model:
        raise HTTPException(404, "LightGBM model not loaded. Train it first.")

    importance_path = LIGHTGBM_MODEL_PATH.with_suffix(".importance.json")
    if importance_path.exists():
        with open(importance_path) as f:
            return {
                "model_path": str(LIGHTGBM_MODEL_PATH),
                "feature_importance": json.load(f),
                "feature_names": list(_feature_builder.FEATURE_NAMES) if _feature_builder else [],
            }

    importance = _lightgbm_model.feature_importance(importance_type="gain")
    names = (_feature_builder.FEATURE_NAMES if _feature_builder
             else [f"f{i}" for i in range(len(importance))])
    return {
        "model_path": str(LIGHTGBM_MODEL_PATH),
        "feature_importance": {names[i]: float(importance[i]) for i in range(len(importance))},
        "feature_names": list(names),
    }
