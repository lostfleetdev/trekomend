"""
Shared settings for the entire project.
"""
import time
from pathlib import Path

import numpy as np

# Paths
DEFAULT_EMB_DIR = Path("embeddings")
DEFAULT_DATASET = Path("dataset/TMDB_movie_dataset_v11.csv")

# Embedding model
TARGET_ROWS = 1_427_380
OUT_DIM = 768
OLLAMA_MODEL = "qwen3:0.6b"

# Qwen3 query instruction (docs recommend writing in English even for multilingual)
QUERY_INSTRUCTION = "Given a movie search query, retrieve the most relevant movies."

# Random number generator — seeded from current time so results vary each run
RNG = np.random.default_rng(int(time.time() * 1e6) % (2 ** 31))

# RRF constant (higher = less weight on ranking position)
RRF_K = 60
