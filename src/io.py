"""
Read/write HDF5 shards, load TMDB metadata, and embed queries via Ollama.
"""
import json as _json
import urllib.request as _urllib
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from .config import OUT_DIM, QUERY_INSTRUCTION


# ═══════════════════════════════════════════════════════════════════════════════
#  Shard discovery
# ═══════════════════════════════════════════════════════════════════════════════

def scan_shards(emb_dir: Path) -> list[dict]:
    """Find all qwen_*.h5 files. Returns list of {index, file, start, end, rows, size_mb}."""
    shards = []
    for h5f in sorted(emb_dir.glob("qwen_*.h5")):
        parts = h5f.stem.split("_")
        start, end = int(parts[1]), int(parts[2])
        with h5py.File(h5f, "r") as f:
            n_rows = f["embeddings"].shape[0]
        shards.append({
            "index": len(shards),
            "file": h5f,
            "start": start,
            "end": end,
            "rows": n_rows,
            "size_mb": h5f.stat().st_size / 1e6,
        })
    return shards


# ═══════════════════════════════════════════════════════════════════════════════
#  Metadata loading (titles, genres, etc. from the big CSV)
# ═══════════════════════════════════════════════════════════════════════════════

def _primary_genre(s) -> str:
    """Extract the first genre from a comma-separated string."""
    if pd.isna(s) or str(s).strip() == "":
        return "Unknown"
    return str(s).split(",")[0].strip()


def load_metadata(h5_path: Path, dataset_path: str) -> pd.DataFrame:
    """
    Load the TMDB CSV rows that correspond to a single .h5 shard.

    The shard stores `/rows` (CSV row indices) and `/ids` (TMDB movie IDs).
    We look up those rows in the big CSV using chunked isin() matching to keep
    RAM low — never loads the full 632 MB CSV at once.

    Returns a DataFrame with columns like title, overview, genres, etc.
    plus a 'primary_genre' column.
    """
    with h5py.File(h5_path, "r") as hf:
        row_indices = hf["rows"][:]

    needed = set(int(r) for r in row_indices)
    parts = []
    for chunk in pd.read_csv(dataset_path, chunksize=250_000, low_memory=False):
        mask = chunk.index.isin(needed)
        if mask.any():
            parts.append(chunk[mask])
        # Stop early once we have all the rows we need
        if sum(len(c) for c in parts) >= len(needed):
            break

    df = pd.concat(parts, ignore_index=False)
    # Reorder rows to match the shard's order
    df = df.loc[row_indices].reset_index(drop=True)
    df["primary_genre"] = df["genres"].apply(_primary_genre)
    return df


def load_all(emb_dir: Path, dataset_path: str):
    """
    Load ALL shards into memory at once.
    Returns (embeddings: np.ndarray [N x 768],
             metadata: pd.DataFrame,
             row_indices: np.ndarray [N],
             shards: list)
    """
    shards = scan_shards(emb_dir)
    if not shards:
        return None, None, None, []

    emb_parts, df_parts, rows_parts = [], [], []
    for s in shards:
        with h5py.File(s["file"], "r") as hf:
            emb_parts.append(hf["embeddings"][:].astype(np.float32))
        df_c = load_metadata(s["file"], dataset_path)
        with h5py.File(s["file"], "r") as hf:
            rows_parts.append(hf["rows"][:])

    emb = np.concatenate(emb_parts, axis=0)
    df = pd.concat(df_parts, ignore_index=True)
    rows_all = np.concatenate(rows_parts, axis=0)
    return emb, df, rows_all, shards


def load_shard_array(h5_path: Path) -> np.ndarray:
    """Read just the embeddings from one shard. Fast, suitable for mmap-style access."""
    with h5py.File(h5_path, "r") as hf:
        return hf["embeddings"][:].astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
#  Ollama query embedding
# ═══════════════════════════════════════════════════════════════════════════════

def embed_query(query_text: str, model: str = "qwen3:0.6b") -> np.ndarray:
    """
    Send a search query to a local Ollama instance for embedding.

    Wraps the query in Qwen3's asymmetric prompt format: instruction for
    the query side, no instruction on the document side (movies are stored
    as raw text without instructions).

    Returns a 768-dim unit-normalized float32 vector.
    """
    prompt = f"Instruct: {QUERY_INSTRUCTION}\nQuery: {query_text}"

    payload = _json.dumps({
        "model": model,
        "prompt": prompt,
    }).encode("utf-8")

    req = _urllib.Request(
        "http://localhost:11434/api/embeddings",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with _urllib.urlopen(req) as resp:
        data = _json.loads(resp.read().decode("utf-8"))

    vec = np.array(data["embedding"], dtype=np.float32)
    # Ollama returns the full 1024-dim vector. Truncate to OUT_DIM (Matryoshka).
    vec = vec[:OUT_DIM]
    # Normalize to unit length — cosine similarity = dot product
    vec = vec / np.linalg.norm(vec)
    return vec
