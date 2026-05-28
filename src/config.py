"""
Shared settings for trekomend.
"""
import os
from pathlib import Path

# Data paths (large files hosted on HuggingFace, not in git)
DEFAULT_EMB_DIR = Path("new-kaggle-output")
DEFAULT_FAISS_INDEX = Path("new-kaggle-output/tmdb_qwen06b_1024d.faiss")
DEFAULT_SQLITE_DB = Path("new-kaggle-output/tmdb_movies.db")
DEFAULT_DATASET = Path("dataset/TMDB_movie_dataset_v11.csv")

# Embedding model
OUT_DIM = 1024
OLLAMA_MODEL = "qwen3-embedding:0.6b"

# Merged HDF5 file name (checked first, then fallbacks)
MERGED_H5_NAMES = [
    "tmdb_qwen06b_1024d.h5",
    "tmdb_qwen06b_768d.h5",
    "tmdb_qwen4b_1536d_v2.h5",
]
MERGED_H5_NAME = MERGED_H5_NAMES[0]

# Query instructions for asymmetric embedding (Qwen3 prompt format)
QUERY_INSTRUCTION = (
    "Given a description of movie preferences including liked films, desired genres, "
    "themes, mood, and watching context, retrieve movies that best match these criteria "
    "for personalized content-based recommendation."
)
MOOD_INSTRUCTION = (
    "Given a description of the viewer's current mood, emotional state, and desired "
    "atmosphere, retrieve movies whose tone, energy, and emotional arc best match "
    "this feeling for a satisfying viewing experience."
)
GENRE_INSTRUCTION = (
    "Given a description of desired genres, plot elements, themes, narrative style, "
    "and cinematic qualities, retrieve movies that share these characteristics "
    "in terms of story, genre conventions, and thematic content."
)
HYBRID_INSTRUCTION = (
    "Given a list of movies the user enjoys and a description of their current "
    "preferences or mood, retrieve movies that bridge the user's established taste "
    "with their immediate viewing context for a personalized, timely recommendation."
)

# FAISS index settings
FAISS_DEFAULT_NPROBE = 32

# Phase 2: learned re-ranker and diversity
PHASE2_ENABLED = True
LIGHTGBM_MODEL_PATH = Path("models/ranker_v1.txt")

# DPP (Determinantal Point Process)
DPP_RANK = 20
DPP_LAMBDA_QD = 0.7
DPP_CANDIDATE_POOL = 50

# Genre round-robin
GENRE_MAX_PER_GENRE = 3
GENRE_MIN_UNIQUE = 3
SERENDIPITY_POSITION = 7

# Redis / job queue settings
REDIS_URL = os.environ.get("TREKOMEND_REDIS_URL", "redis://127.0.0.1:6379/0")
JOB_TTL_SECONDS = 300          # Jobs expire after 5 minutes
JOB_POLL_INTERVAL_MS = 500     # Frontend polls every 500ms
WORKER_POLL_TIMEOUT = 1        # BLPOP timeout in seconds
RATE_LIMIT_PER_MINUTE = 60     # Per-IP requests per minute

# Production mode — disable /docs, /redoc, /openapi.json
PRODUCTION_MODE = os.environ.get("TREKOMEND_PRODUCTION", "0").lower() in ("1", "true", "yes")
