"""
main.py — trekomend movie recommendation API.

Runs a FastAPI server backed by a FAISS IVF-PQ index and a SQLite metadata
database. Big files (FAISS index, HDF5 embeddings, SQLite database) live in
new-kaggle-output/ and are hosted on HuggingFace, not in git.

Start the server:
    uv run uvicorn main:app --host 0.0.0.0 --port 8080 --workers 1

Or just:
    uv run python main.py

Phase 2 pipeline (on by default):
  FAISS top-200 -> Genre Round-Robin -> LightGBM re-ranker -> DPP -> Top-12
  Default path skips DPP. Use /recommend/diverse to opt in.

Endpoints:
    GET  /health                    Liveness check
    GET  /stats                     Index stats
    POST /recommend/similar         Movies like another movie
    POST /recommend/query           Movies matching a text description
    POST /recommend/profile         Profile from liked/disliked films + mood
    POST /recommend/diverse         Max diversity (DPP mode)
    POST /recommend/explore         Serendipity/novelty mode
    GET  /movies/{tmdb_id}          Single movie by TMDB ID
    GET  /movies/search?q=...       Full-text search (FTS5)
    GET  /movies/browse?genre=...   Browse with filters
    GET  /movies/genres             All primary genres
    GET  /ranker/features           LightGBM feature importance
"""
import os as _os
import json as _json
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.faiss_search import FaissSearcher
from src.io import embed_query, embed_hybrid
from src.config import (
    DEFAULT_FAISS_INDEX, DEFAULT_SQLITE_DB, DEFAULT_EMB_DIR,
    MERGED_H5_NAME, QUERY_INSTRUCTION,
    PHASE2_ENABLED, LIGHTGBM_MODEL_PATH,
    DPP_RANK, DPP_LAMBDA_QD, DPP_CANDIDATE_POOL,
    GENRE_MAX_PER_GENRE, GENRE_MIN_UNIQUE, SERENDIPITY_POSITION,
)


# ============================================================================
# Global state
# ============================================================================

searcher: FaissSearcher | None = None
feature_builder = None
genre_round_robin = None
dpp_selector = None
lightgbm_model = None
hdf5_path: str = ""
phase2_loaded: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the FAISS index, SQLite DB, and Phase 2 modules at startup."""
    global searcher, hdf5_path, phase2_loaded
    global feature_builder, genre_round_robin, dpp_selector, lightgbm_model

    hdf5_path = _os.environ.get(
        "TREKOMEND_HDF5", str(DEFAULT_EMB_DIR / MERGED_H5_NAME)
    )
    faiss_path = _os.environ.get("TREKOMEND_FAISS", str(DEFAULT_FAISS_INDEX))
    sqlite_path = _os.environ.get("TREKOMEND_SQLITE", str(DEFAULT_SQLITE_DB))
    searcher = FaissSearcher(faiss_path, sqlite_path, hdf5_path=hdf5_path)
    searcher.load()
    s = searcher.stats()
    print(f"API ready: {s['faiss_vectors']:,} vectors x {s['faiss_dim']}d  "
          f"nprobe={s['faiss_nprobe']}  DB={s['db_movies']:,} movies")

    # Phase 2 modules
    phase2_enabled = _os.environ.get("TREKOMEND_PHASE2", "1") not in ("0", "false", "no")
    if phase2_enabled and PHASE2_ENABLED:
        try:
            from src.features import FeatureBuilder
            from src.diversity import (
                GenreRoundRobin, DPPSelector,
                RoundRobinConfig, DPPConfig,
            )

            feature_builder = FeatureBuilder(searcher._db, hdf5_path=hdf5_path)
            feature_builder.compute_catalog_stats()
            print(f"  Phase 2: FeatureBuilder ready ({FeatureBuilder.N_FEATURES} features)")

            rr_cfg = RoundRobinConfig(
                max_per_genre=GENRE_MAX_PER_GENRE,
                min_unique_genres=GENRE_MIN_UNIQUE,
                serendipity_position=SERENDIPITY_POSITION,
            )
            genre_round_robin = GenreRoundRobin(searcher._db, config=rr_cfg)
            print("  Phase 2: GenreRoundRobin ready")

            dpp_cfg = DPPConfig(
                rank=DPP_RANK,
                lambda_qd=DPP_LAMBDA_QD,
                kernel_mode="full",
            )
            dpp_selector = DPPSelector(config=dpp_cfg)
            print(f"  Phase 2: DPPSelector ready (lambda_qd={DPP_LAMBDA_QD})")

            if LIGHTGBM_MODEL_PATH.exists():
                import lightgbm as lgb
                lightgbm_model = lgb.Booster(model_file=str(LIGHTGBM_MODEL_PATH))
                print(f"  Phase 2: LightGBM model loaded from {LIGHTGBM_MODEL_PATH}")
            else:
                print(f"  Phase 2: No LightGBM model at {LIGHTGBM_MODEL_PATH}. "
                      f"Train with: uv run python -m src.train_ranker")

            phase2_loaded = True
        except Exception as e:
            print(f"  Phase 2: failed to load ({e})")
            phase2_loaded = False
    else:
        print("  Phase 2: disabled")
        phase2_loaded = False

    yield
    if searcher:
        searcher.close()
    print("API shut down.")


app = FastAPI(
    title="trekomend API",
    description="Movie recommendation API using FAISS and Qwen3 embeddings",
    version="0.4.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ============================================================================
# Pydantic models
# ============================================================================

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


class StatsResponse(BaseModel):
    faiss_vectors: int
    faiss_dim: int
    faiss_nprobe: int
    db_movies: int


# ============================================================================
# Phase 2 pipeline
# ============================================================================

def _phase2_rerank(
    query_tmdb_id: int | None,
    faiss_results: list[dict],
    top_k: int = 12,
    lambda_qd: float | None = None,
    skip_lightgbm: bool = False,
    use_dpp: bool = False,
) -> list[dict]:
    """Three-layer post-retrieval pipeline.

    Layer 1: LightGBM re-ranker (applies learned relevance scores)
    Layer 2: Genre round-robin (spreads results across genres)
    Layer 3: DPP set selection (opt-in fine-grained diversity)

    DPP is off by default. Round-robin alone adds near-zero overhead
    and gives 6-9 unique genres for most queries.
    """
    global feature_builder, genre_round_robin, dpp_selector, lightgbm_model

    if not phase2_loaded or not faiss_results:
        return faiss_results[:top_k]

    enriched = list(faiss_results)

    # Layer 1: LightGBM re-ranking (before round-robin, so within-genre
    # picks benefit from the improved relevance scores)
    if not skip_lightgbm and lightgbm_model is not None and query_tmdb_id is not None:
        try:
            candidate_ids = [r["tmdb_id"] for r in enriched]
            candidate_scores = [r.get("score", 0) or 0 for r in enriched]

            X = feature_builder.build_features(
                query_tmdb_id=query_tmdb_id,
                candidate_tmdb_ids=candidate_ids,
                candidate_scores=candidate_scores,
            )
            new_scores = lightgbm_model.predict(X)

            for i, r in enumerate(enriched):
                r["raw_score"] = r.get("score", 0)
                r["score"] = round(float(new_scores[i]), 4)

            enriched.sort(key=lambda r: r.get("score", 0) or 0, reverse=True)
        except Exception:
            pass

    # Layer 2: Genre round-robin (always applied)
    if genre_round_robin is not None:
        enriched = genre_round_robin.diversify(enriched, top_k=max(200, top_k * 4))

    # Layer 3: DPP (opt-in)
    if use_dpp and dpp_selector is not None and len(enriched) > top_k:
        dpp_pool = enriched[:DPP_CANDIDATE_POOL]
        dpp_ids = [r["tmdb_id"] for r in dpp_pool]

        raw_scores = np.array([r.get("score", 0) or 0 for r in dpp_pool],
                              dtype=np.float32)
        rr_position_bonus = np.linspace(0.15, 0.0, len(dpp_pool), dtype=np.float32)
        dpp_scores = raw_scores + rr_position_bonus

        emb_map = searcher.get_exact_embeddings_batch(dpp_ids)
        embs = []
        valid_indices = []
        for i, tid in enumerate(dpp_ids):
            vec = emb_map.get(tid)
            if vec is not None:
                embs.append(vec)
                valid_indices.append(i)

        if len(embs) >= top_k:
            emb_matrix = np.stack(embs).astype(np.float32)
            valid_scores = dpp_scores[valid_indices]

            if lambda_qd is not None:
                dpp_selector._cfg.lambda_qd = lambda_qd

            try:
                selected_indices, _ = dpp_selector.select(
                    emb_matrix, valid_scores, top_k=top_k,
                )
                result = [dpp_pool[valid_indices[i]] for i in selected_indices]
                for i, r in enumerate(result):
                    r["rank"] = i + 1
                return result
            except Exception:
                pass

    return enriched[:top_k]


# ============================================================================
# Health & Stats
# ============================================================================

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/stats", response_model=StatsResponse)
def stats() -> dict[str, Any]:
    if not searcher:
        raise HTTPException(503, "Searcher not initialized")
    return searcher.stats()


# ============================================================================
# Recommendations
# ============================================================================

@app.post("/recommend/similar", response_model=list[MovieResult])
def recommend_similar(req: SimilarRequest) -> list[dict]:
    """Movies similar to a given title."""
    if not searcher:
        raise HTTPException(503, "Searcher not initialized")

    try:
        movie = searcher.lookup_by_title(req.title)
        if movie is None:
            raise ValueError(f"Movie not found: '{req.title}'")
    except ValueError as e:
        raise HTTPException(404, str(e))

    query_tmdb_id = movie["id"]
    vec = searcher.get_exact_embedding(query_tmdb_id)
    if vec is None:
        raise HTTPException(500,
            f"No embedding for '{req.title}' (TMDB {query_tmdb_id}). "
            f"Check HDF5 at {hdf5_path}.")

    faiss_k = min(200, req.limit * 4) if req.use_phase2 else req.limit
    results = searcher.search(vec, k=faiss_k, nprobe=req.nprobe)

    if req.use_phase2 and phase2_loaded:
        results = _phase2_rerank(
            query_tmdb_id=query_tmdb_id,
            faiss_results=results,
            top_k=req.limit,
            use_dpp=False,
        )
    else:
        results = results[:req.limit]

    return results


@app.post("/recommend/query", response_model=list[MovieResult])
def recommend_query(req: QueryRequest) -> list[dict]:
    """Movies matching a text description. Uses Ollama for the embedding."""
    if not searcher:
        raise HTTPException(503, "Searcher not initialized")

    try:
        query_vec = embed_query(req.query, instruction=QUERY_INSTRUCTION)
    except Exception as e:
        raise HTTPException(503, f"Ollama embedding failed: {e}")

    faiss_k = min(200, req.limit * 4) if req.use_phase2 else req.limit
    results = searcher.search(query_vec, k=faiss_k, nprobe=req.nprobe)

    if req.use_phase2 and phase2_loaded:
        results = _phase2_rerank(
            query_tmdb_id=None,
            faiss_results=results,
            top_k=req.limit,
            use_dpp=False,
        )
    else:
        results = results[:req.limit]

    return results


@app.post("/recommend/profile", response_model=list[MovieResult])
def recommend_profile(req: ProfileRequest) -> list[dict]:
    """Profile-based recommendations from liked movies, mood, and dislikes.

    Builds a taste centroid from liked movies, pushes away from disliked
    ones, blends with mood text if given, then searches and reranks.
    """
    if not searcher:
        raise HTTPException(503, "Searcher not initialized")

    liked_movies: list[dict] = []
    for title in req.liked:
        movie = searcher.lookup_by_title(title)
        if movie is None:
            raise HTTPException(404, f"Movie not found: '{title}'")
        liked_movies.append(movie)

    disliked_movies: list[dict] = []
    for title in req.dislike:
        movie = searcher.lookup_by_title(title)
        if movie is not None:
            disliked_movies.append(movie)

    all_tmdb_ids = [m["id"] for m in liked_movies] + [m["id"] for m in disliked_movies]
    emb_map = searcher.get_exact_embeddings_batch(all_tmdb_ids)

    positive_vecs = []
    for movie in liked_movies:
        vec = emb_map.get(movie["id"])
        if vec is None:
            raise HTTPException(500,
                f"No embedding for '{movie['title']}' (TMDB {movie['id']}). "
                f"Check HDF5 at {hdf5_path}.")
        positive_vecs.append(vec.astype(np.float32))

    user_vector = np.mean(positive_vecs, axis=0, dtype=np.float32)

    if disliked_movies:
        negative_vecs = []
        for movie in disliked_movies:
            vec = emb_map.get(movie["id"])
            if vec is not None:
                negative_vecs.append(vec.astype(np.float32))

        if negative_vecs:
            dislike_centroid = np.mean(negative_vecs, axis=0, dtype=np.float32)
            dislike_centroid = dislike_centroid / (np.linalg.norm(dislike_centroid) + 1e-10)
            projection = float(np.dot(user_vector, dislike_centroid))
            if projection > 0:
                user_vector = user_vector - 0.3 * projection * dislike_centroid

    if req.mood:
        try:
            mood_vec = embed_hybrid(req.mood)
            user_vector = (1 - req.mood_weight) * user_vector + req.mood_weight * mood_vec
        except Exception as e:
            raise HTTPException(503, f"Mood embedding failed: {e}")

    norm = np.linalg.norm(user_vector) + 1e-10
    user_vector = user_vector / norm

    faiss_k = min(200, req.limit * 4) if req.use_phase2 else req.limit
    results = searcher.search(user_vector, k=faiss_k, nprobe=req.nprobe)

    liked_titles_lower = {t.lower() for t in req.liked}
    results = [r for r in results if r["title"].lower() not in liked_titles_lower]

    if req.use_phase2 and phase2_loaded and results:
        pseudo_query_id = liked_movies[0]["id"]
        results = _phase2_rerank(
            query_tmdb_id=pseudo_query_id,
            faiss_results=results,
            top_k=req.limit,
            use_dpp=False,
        )
    else:
        results = results[:req.limit]

    return results


# ============================================================================
# Specialized diversity endpoints
# ============================================================================

@app.post("/recommend/diverse", response_model=list[MovieResult])
def recommend_diverse(req: DiverseRequest) -> list[dict]:
    """Maximum diversity mode. Lower lambda_qd for more variety."""
    if not searcher:
        raise HTTPException(503, "Searcher not initialized")

    movie = searcher.lookup_by_title(req.title)
    if movie is None:
        raise HTTPException(404, f"Movie not found: '{req.title}'")

    query_tmdb_id = movie["id"]
    vec = searcher.get_exact_embedding(query_tmdb_id)
    if vec is None:
        raise HTTPException(500, f"No embedding for '{req.title}'")

    faiss_k = min(200, req.limit * 5)
    results = searcher.search(vec, k=faiss_k, nprobe=req.nprobe)

    if phase2_loaded:
        results = _phase2_rerank(
            query_tmdb_id=query_tmdb_id,
            faiss_results=results,
            top_k=req.limit,
            lambda_qd=req.lambda_qd,
            use_dpp=True,
        )
    else:
        results = results[:req.limit]

    return results


@app.post("/recommend/explore", response_model=list[MovieResult])
def recommend_explore(req: ExploreRequest) -> list[dict]:
    """Serendipity mode. Favors less popular, less obvious movies."""
    if not searcher:
        raise HTTPException(503, "Searcher not initialized")

    liked_movies = []
    for title in req.liked:
        movie = searcher.lookup_by_title(title)
        if movie is None:
            raise HTTPException(404, f"Movie not found: '{title}'")
        liked_movies.append(movie)

    if not liked_movies:
        raise HTTPException(404, "No valid liked movies found")

    all_ids = [m["id"] for m in liked_movies]
    emb_map = searcher.get_exact_embeddings_batch(all_ids)
    positive_vecs = []
    for movie in liked_movies:
        vec = emb_map.get(movie["id"])
        if vec is not None:
            positive_vecs.append(vec.astype(np.float32))

    if not positive_vecs:
        raise HTTPException(500, "Could not retrieve embeddings for liked movies")

    user_vector = np.mean(positive_vecs, axis=0, dtype=np.float32)
    user_vector = user_vector / (np.linalg.norm(user_vector) + 1e-10)

    faiss_k = min(200, req.limit * 5)
    results = searcher.search(user_vector, k=faiss_k, nprobe=req.nprobe)

    liked_titles_lower = {t.lower() for t in req.liked}
    results = [r for r in results if r["title"].lower() not in liked_titles_lower]

    if phase2_loaded and results:
        pseudo_id = liked_movies[0]["id"]
        results = _phase2_rerank(
            query_tmdb_id=pseudo_id,
            faiss_results=results,
            top_k=req.limit,
            lambda_qd=0.35,
            use_dpp=True,
        )
    else:
        results = results[:req.limit]

    return results


@app.get("/ranker/features")
def ranker_features() -> dict[str, Any]:
    """Feature importance for the LightGBM re-ranker."""
    if not lightgbm_model:
        raise HTTPException(404, "LightGBM model not loaded. Train it first.")

    importance_path = LIGHTGBM_MODEL_PATH.with_suffix(".importance.json")
    if importance_path.exists():
        with open(importance_path) as f:
            return {
                "model_path": str(LIGHTGBM_MODEL_PATH),
                "feature_importance": _json.load(f),
                "feature_names": list(feature_builder.FEATURE_NAMES) if feature_builder else [],
            }

    importance = lightgbm_model.feature_importance(importance_type="gain")
    names = (feature_builder.FEATURE_NAMES if feature_builder
             else [f"f{i}" for i in range(len(importance))])
    return {
        "model_path": str(LIGHTGBM_MODEL_PATH),
        "feature_importance": {names[i]: float(importance[i]) for i in range(len(importance))},
        "feature_names": list(names),
    }


# ============================================================================
# Movie lookup and browsing
# ============================================================================

@app.get("/movies/search")
def search_movies(
    q: str = Query(..., description="Search query"),
    limit: int = Query(default=20, ge=1, le=100),
) -> list[dict[str, Any]]:
    """Full-text search across titles, overviews, keywords, and genres."""
    if not searcher:
        raise HTTPException(503, "Searcher not initialized")
    return searcher.search_text(q, limit=limit)


@app.get("/movies/browse")
def browse_movies(
    genre: str | None = Query(default=None),
    year_min: int | None = Query(default=None),
    year_max: int | None = Query(default=None),
    min_votes: int | None = Query(default=None),
    min_rating: float | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> list[dict[str, Any]]:
    """Browse movies with optional filters on genre, year, and rating."""
    if not searcher:
        raise HTTPException(503, "Searcher not initialized")
    return searcher.browse(
        genre=genre, year_min=year_min, year_max=year_max,
        min_votes=min_votes, min_rating=min_rating,
        limit=limit, offset=offset,
    )


@app.get("/movies/genres")
def list_genres() -> list[str]:
    """All distinct primary genres in the database."""
    if not searcher:
        raise HTTPException(503, "Searcher not initialized")
    return searcher.get_genres()


@app.get("/movies/{tmdb_id}")
def get_movie(tmdb_id: int) -> dict[str, Any]:
    """A single movie by TMDB ID."""
    if not searcher:
        raise HTTPException(503, "Searcher not initialized")
    movie = searcher._lookup_movie(tmdb_id)
    if movie is None:
        raise HTTPException(404, f"Movie #{tmdb_id} not found")
    return movie
