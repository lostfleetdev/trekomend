"""
Shared settings for the entire project.
"""
import time
from pathlib import Path

import numpy as np

# Paths
DEFAULT_EMB_DIR = Path("embeddings")
DEFAULT_DATASET = Path("dataset/TMDB_movie_dataset_v11.csv")

# Embedding model — these defaults are detected from the HDF5 at runtime.
# Set them here as fallbacks for when no files are present yet.
TARGET_ROWS = 1_427_380
OUT_DIM = 1024          # fallback; actual dim is read from HDF5 attributes
OLLAMA_MODEL = "qwen3-embedding:0.6b"

# Known HDF5 shard prefixes (we scan for all of them)
SHARD_GLOBS = ["shard_*.h5", "qwen_*.h5"]

# Single merged file names (checked in order, first found wins).
# Newest first: v2 uses 1024-dim with research-optimized text template.
MERGED_H5_NAMES = [
    "tmdb_qwen06b_1024d.h5",    # 0.6B model, 1024-dim native (trekomend_v2_1024d.ipynb)
    "tmdb_qwen06b_768d.h5",     # 0.6B model, 768-dim (legacy)
    "tmdb_qwen4b_1536d_v2.h5",  # 4B model, 1536-dim (legacy)
]
MERGED_H5_NAME = MERGED_H5_NAMES[0]  # default

# Qwen3 query instruction for movie recommendation.
# Research-backed: instruction improves retrieval 1-5% per Qwen3 docs.
# Write in English even for multilingual queries.

# General-purpose instruction — balances all signals
QUERY_INSTRUCTION = (
    "Given a description of movie preferences including liked films, desired genres, "
    "themes, mood, and watching context, retrieve movies that best match these criteria "
    "for personalized content-based recommendation."
)

# Mood-focused instruction — prioritizes emotional/aesthetic fit over exact plot match
MOOD_INSTRUCTION = (
    "Given a description of the viewer's current mood, emotional state, and desired "
    "atmosphere, retrieve movies whose tone, energy, and emotional arc best match "
    "this feeling for a satisfying viewing experience."
)

# Genre/plot instruction — prioritizes narrative and thematic similarity
GENRE_INSTRUCTION = (
    "Given a description of desired genres, plot elements, themes, narrative style, "
    "and cinematic qualities, retrieve movies that share these characteristics "
    "in terms of story, genre conventions, and thematic content."
)

# Hybrid instruction — combines loved movies + current mood
HYBRID_INSTRUCTION = (
    "Given a list of movies the user enjoys and a description of their current "
    "preferences or mood, retrieve movies that bridge the user's established taste "
    "with their immediate viewing context for a personalized, timely recommendation."
)

# Random number generator — seeded from current time so results vary each run
RNG = np.random.default_rng(int(time.time() * 1e6) % (2 ** 31))

# RRF constant (higher = less weight on ranking position)
RRF_K = 60
