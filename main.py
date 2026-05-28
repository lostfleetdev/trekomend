"""
main.py — trekomend movie recommendation API (refactored).

Thin entrypoint that loads the FAISS index and Phase 2 modules at startup,
then delegates to two routers:
    - api_read         : cheap synchronous endpoints (movies/*, health, stats)
    - api_recommend    : expensive endpoints with Redis job queue

The job queue is handled by a separate worker process (src/worker.py).
This file only serves the API.

Start the server:
    uv run uvicorn main:app --host 127.0.0.1 --port 6767 --workers 3 --proxy-headers

Or just:
    uv run python main.py
"""
import os as _os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.faiss_search import FaissSearcher
from src.config import (
    DEFAULT_FAISS_INDEX, DEFAULT_SQLITE_DB, DEFAULT_EMB_DIR,
    MERGED_H5_NAME,
    PHASE2_ENABLED, LIGHTGBM_MODEL_PATH,
    DPP_RANK, DPP_LAMBDA_QD,
    GENRE_MAX_PER_GENRE, GENRE_MIN_UNIQUE, SERENDIPITY_POSITION,
    PRODUCTION_MODE,
)
from src.api_read import router as read_router, set_read_state
from src.api_recommend import router as recommend_router, set_recommend_state, set_redis_available
from src.ratelimit import RateLimitMiddleware


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

    # Inject state into routers
    set_read_state(searcher, feature_builder, lightgbm_model)
    set_recommend_state(searcher, hdf5_path)

    # Check Redis availability
    try:
        from src.jobs import get_redis
        r = get_redis()
        r.ping()
        set_redis_available(True)
        print("  Redis: connected — recommendations will use job queue")
    except Exception:
        set_redis_available(False)
        print("  Redis: not available — recommendations run synchronously")

    yield

    if searcher:
        searcher.close()
    print("API shut down.")


# ============================================================================
# App creation
# ============================================================================

# In production, disable docs to prevent API schema exposure
docs_url = None if PRODUCTION_MODE else "/docs"
redoc_url = None if PRODUCTION_MODE else "/redoc"
openapi_url = None if PRODUCTION_MODE else "/openapi.json"

app = FastAPI(
    title="trekomend API",
    description="Movie recommendation API using FAISS and Qwen3 embeddings",
    version="0.5.0",
    lifespan=lifespan,
    docs_url=docs_url,
    redoc_url=redoc_url,
    openapi_url=openapi_url,
)

# Rate limiting middleware (covers all /api/* routes)
app.add_middleware(RateLimitMiddleware)

# CORS — allow the site domain and localhost for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://trekomend.chaospunk.space",
        "http://localhost:4321",  # Astro dev server
        "http://127.0.0.1:4321",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Include routers
app.include_router(read_router)
app.include_router(recommend_router)


# ============================================================================
# Root redirect (for when someone visits the API directly)
# ============================================================================

@app.get("/")
def root():
    return {
        "message": "Trekomend API",
        "docs": "Visit https://trekomend.chaospunk.space for the website",
        "health": "/api/health",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=6767,
        proxy_headers=True,
        log_level="info",
    )
