"""
Read/write HDF5 shards, load TMDB metadata, and embed queries via Ollama.

Handles two formats:
  - Legacy: qwen_*.h5 or shard_*.h5 shards (768 or 1536 dim)
  - Merged:  tmdb_qwen4b_1.4M.h5 single file (1536 dim from Kaggle notebook)

Embedding dimension is auto-detected from HDF5 attributes.
"""
import json as _json
import urllib.request as _urllib
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from .config import OUT_DIM, QUERY_INSTRUCTION, SHARD_GLOBS, MERGED_H5_NAMES, MERGED_H5_NAME


# ═══════════════════════════════════════════════════════════════════════════════
#  Dimension detection
# ═══════════════════════════════════════════════════════════════════════════════

def _detect_dim(h5_path: Path, fallback: int = OUT_DIM) -> int:
    """Read embedding dimension from HDF5 attributes, with fallback."""
    try:
        with h5py.File(h5_path, "r") as f:
            if "dim" in f.attrs:
                return int(f.attrs["dim"])
            return f["embeddings"].shape[1]
    except Exception:
        return fallback


# ═══════════════════════════════════════════════════════════════════════════════
#  Shard discovery
# ═══════════════════════════════════════════════════════════════════════════════

def scan_shards(emb_dir: Path) -> list[dict]:
    """
    Find all shard HDF5 files (qwen_*.h5 and shard_*.h5) in emb_dir.
    Also detects the merged single-file format if present.

    Returns list of {index, file, start, end, rows, size_mb, dim}.
    """
    shards = []

    for glob_pattern in SHARD_GLOBS:
        for h5f in sorted(emb_dir.glob(glob_pattern)):
            try:
                with h5py.File(h5f, "r") as f:
                    n_rows = f["embeddings"].shape[0]

                # Try to extract row range from filename
                parts = h5f.stem.split("_")
                if len(parts) >= 3:
                    try:
                        start, end = int(parts[-2]), int(parts[-1])
                    except (ValueError, IndexError):
                        start, end = 0, n_rows
                else:
                    start, end = 0, n_rows

                shards.append({
                    "index": len(shards),
                    "file": h5f,
                    "start": start,
                    "end": end,
                    "rows": n_rows,
                    "size_mb": h5f.stat().st_size / 1e6,
                    "dim": _detect_dim(h5f),
                })
            except Exception:
                continue

    return shards


def load_merged(emb_dir: Path) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, str | None]:
    """
    Load a single merged HDF5 file if present (checks multiple possible names).
    Returns (embeddings, ids, rows, filename) or (None, None, None, None).
    """
    for name in MERGED_H5_NAMES:
        merged = emb_dir / name
        if merged.exists():
            print(f"  Found merged file: {merged.name}")
            with h5py.File(merged, "r") as f:
                emb  = f["embeddings"][:].astype(np.float32)
                ids  = f["ids"][:]
                # /rows may not exist in merged HDF5 (v2 notebook writes only embeddings+ids)
                if "rows" in f:
                    rows = f["rows"][:]
                else:
                    n_rows = int(f.attrs.get("n_rows", emb.shape[0]))
                    rows = np.arange(n_rows, dtype=np.int32)
                dim  = int(f.attrs.get("dim", emb.shape[1]))
                print(f"  Shape: {emb.shape[0]:,} x {dim}")
                return emb, ids, rows, merged.name
    return None, None, None, None


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
    Load ALL embeddings into memory.

    Prefers the single merged HDF5 if present, otherwise loads and
    concatenates all shards.

    Returns (embeddings: np.ndarray [N x D],
             metadata: pd.DataFrame,
             row_indices: np.ndarray [N],
             shards: list,
             ids: np.ndarray [N] | None)
    """
    # ── Try merged file first ────────────────────────────────────────
    emb, ids, rows_all, merged_name = load_merged(emb_dir)
    if emb is not None:
        n = emb.shape[0]
        dim = emb.shape[1]
        print(f"Loaded merged file: {merged_name}  ({n:,} rows × {dim} dims)")

        # Load metadata for merged file
        df_parts = []
        needed = set(int(r) for r in rows_all[:n])
        for chunk in pd.read_csv(dataset_path, chunksize=250_000, low_memory=False):
            mask = chunk.index.isin(needed)
            if mask.any():
                df_parts.append(chunk[mask])
            if sum(len(c) for c in df_parts) >= len(needed):
                break

        df = pd.concat(df_parts, ignore_index=False)
        df = df.loc[rows_all[:n]].reset_index(drop=True)
        df["primary_genre"] = df["genres"].apply(_primary_genre)
        return emb, df, rows_all, [], ids

    # ── Fall back to shard-by-shard loading ───────────────────────────
    shards = scan_shards(emb_dir)
    if not shards:
        return None, None, None, [], None

    emb_parts, df_parts, rows_parts = [], [], []
    for s in shards:
        with h5py.File(s["file"], "r") as hf:
            emb_parts.append(hf["embeddings"][:].astype(np.float32))
        df_c = load_metadata(s["file"], str(dataset_path))
        if df_c is None or len(df_c) == 0:
            print(f"  WARNING: load_metadata returned empty for {s['file'].name}")
            continue
        df_parts.append(df_c)
        with h5py.File(s["file"], "r") as hf:
            rows_parts.append(hf["rows"][:])

    if not emb_parts:
        return None, None, None, [], None
    emb = np.concatenate(emb_parts, axis=0)
    if not df_parts:
        return emb, None, None, shards, None
    df = pd.concat(df_parts, ignore_index=True)
    rows_all = np.concatenate(rows_parts, axis=0)
    return emb, df, rows_all, shards, None


def load_shard_array(h5_path: Path) -> np.ndarray:
    """Read just the embeddings from one shard. Fast, suitable for mmap-style access."""
    with h5py.File(h5_path, "r") as hf:
        return hf["embeddings"][:].astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
#  Ollama query embedding
# ═══════════════════════════════════════════════════════════════════════════════

def embed_query(query_text: str, model: str = "qwen3:0.6b",
                 out_dim: int | None = None,
                 instruction: str | None = None) -> np.ndarray:
    """
    Send a search query to a local Ollama instance for embedding.

    Wraps the query in Qwen3's asymmetric prompt format: instruction for
    the query side, no instruction on the document side (movies are stored
    as raw text without instructions).

    Research: different instruction types improve retrieval quality by
    biasing the embedding toward mood, genre, or hybrid matching.

    Args:
        query_text: The user's search query.
        model: Ollama model name.
        out_dim: Target dimension. Auto-detected from embeddings if None.
        instruction: Custom instruction. Uses QUERY_INSTRUCTION if None.
    Returns a unit-normalized float32 vector.
    """
    from .config import QUERY_INSTRUCTION
    instr = instruction if instruction is not None else QUERY_INSTRUCTION
    prompt = f"Instruct: {instr}\nQuery: {query_text}"

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
    # Truncate to output dimension (Matryoshka-compatible)
    dim = out_dim if out_dim is not None else OUT_DIM
    vec = vec[:dim]
    # Normalize to unit length — cosine similarity = dot product
    vec = vec / np.linalg.norm(vec)
    return vec


def embed_mood(mood_text: str, model: str = "qwen3:0.6b",
               out_dim: int | None = None) -> np.ndarray:
    """Embed a mood/atmosphere query using mood-biased instruction."""
    from .config import MOOD_INSTRUCTION
    return embed_query(mood_text, model=model, out_dim=out_dim,
                       instruction=MOOD_INSTRUCTION)


def embed_genre(genre_text: str, model: str = "qwen3:0.6b",
                out_dim: int | None = None) -> np.ndarray:
    """Embed a genre/plot query using narrative-biased instruction."""
    from .config import GENRE_INSTRUCTION
    return embed_query(genre_text, model=model, out_dim=out_dim,
                       instruction=GENRE_INSTRUCTION)


def embed_hybrid(hybrid_text: str, model: str = "qwen3:0.6b",
                 out_dim: int | None = None) -> np.ndarray:
    """Embed for hybrid (taste + mood) using the hybrid instruction."""
    from .config import HYBRID_INSTRUCTION
    return embed_query(hybrid_text, model=model, out_dim=out_dim,
                       instruction=HYBRID_INSTRUCTION)
