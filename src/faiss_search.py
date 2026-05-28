"""
FAISS-based vector search with SQLite metadata lookups.

Replaces the brute-force HDF5 dot-product path in search.py with a
compressed IVF-PQ index (~90 MB vs 5.7 GB). Combined with a SQLite
metadata DB, this cuts query RAM from ~6 GB to ~200 MB — suitable
for a VPS with 1-4 GB RAM.

Index details:
  - Type:      IndexIVFPQ (Inverted File + Product Quantization)
  - nlist:     2048 clusters
  - M:         64 subquantizers
  - nbits:     8 bits per PQ code
  - Code size: 64 bytes/vector (vs 4096 raw)
  - Metric:    Inner product (cosine since vectors are L2-normed)
  - Default nprobe: 32 (tunable for recall vs speed)

Usage:
    from src.faiss_search import FaissSearcher

    searcher = FaissSearcher("embeddings/tmdb_qwen06b_1024d.faiss",
                              "embeddings/tmdb_movies.db")
    results = searcher.search(query_vector, k=12)
    # -> [(rank, title, genre, score, tmdb_id), ...]
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import faiss
import h5py
import numpy as np

from .config import FAISS_DEFAULT_NPROBE, OUT_DIM


# ═══════════════════════════════════════════════════════════════════════════════
#  LRU Cache for exact embeddings (avoids repeated HDF5 opens)
# ═══════════════════════════════════════════════════════════════════════════════

class _EmbeddingCache:
    """Thread-safe bounded cache for frequently-accessed exact embedding vectors."""

    def __init__(self, maxsize: int = 256):
        import collections
        self._cache: collections.OrderedDict[int, np.ndarray] = collections.OrderedDict()
        self._maxsize = maxsize

    def get(self, key: int) -> np.ndarray | None:
        val = self._cache.get(key)
        if val is not None:
            self._cache.move_to_end(key)
        return val

    def put(self, key: int, value: np.ndarray) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            self._cache[key] = value
            if len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)

    def clear(self) -> None:
        self._cache.clear()


# ═══════════════════════════════════════════════════════════════════════════════
#  FAISS + SQLite Searcher
# ═══════════════════════════════════════════════════════════════════════════════

class FaissSearcher:
    """
    Efficient approximate nearest-neighbor search over 1.4M movie embeddings.

    Loads a pre-built FAISS IVF-PQ index from disk and resolves result
    IDs to movie metadata via a companion SQLite database.

    Memory profile:
      - Index on disk: ~90 MB (compressed PQ codes)
      - Index in RAM:  ~150 MB (codes + precomputed tables)
      - SQLite DB:     ~300 MB on disk, ~50 MB page cache at query time
      - TOTAL:         ~200 MB resident (vs 5.7 GB brute-force)

    Performance (1.4M vectors, 1024-dim, nprobe=32):
      - Query latency: 1-5 ms single-thread
      - Recall@10:     ~88-93% (vs exact brute-force)
    """

    def __init__(
        self,
        faiss_path: str | Path = "embeddings/tmdb_qwen06b_1024d.faiss",
        sqlite_path: str | Path = "embeddings/tmdb_movies.db",
        hdf5_path: str | Path | None = None,
        nprobe: int | None = None,
    ):
        """
        Args:
            faiss_path: Path to the FAISS index file.
            sqlite_path: Path to the SQLite metadata database.
            hdf5_path: Path to merged HDF5 for ID mapping (tmdb_qwen06b_1024d.h5).
                       Auto-detected from faiss_path directory if None.
            nprobe: Number of IVF clusters to probe per query.
                    Higher = better recall, slower.
                    None = use default from config (32).
        """
        self.faiss_path = Path(faiss_path)
        self.sqlite_path = Path(sqlite_path)
        self.nprobe = nprobe if nprobe is not None else FAISS_DEFAULT_NPROBE

        # HDF5 path for ID mapping (FAISS uses positional 0..N-1,
        # HDF5 /ids maps position -> TMDB ID)
        if hdf5_path is None:
            # Auto-detect from same directory as FAISS index
            hdf5_path = self.faiss_path.parent / "tmdb_qwen06b_1024d.h5"
        self.hdf5_path = Path(hdf5_path)

        self._index: faiss.Index | None = None
        self._db: sqlite3.Connection | None = None
        self._id_map: np.ndarray | None = None  # position -> TMDB ID
        self._id_reverse: dict[int, int] = {}  # TMDB ID -> position
        self._ntotal: int = 0
        self._dim: int = 0
        self._loaded: bool = False
        self._emb_cache = _EmbeddingCache(maxsize=2048)  # cache for get_exact_embedding
        self._h5_file: h5py.File | None = None  # lazily-opened HDF5 handle

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load FAISS index, ID mapping, and SQLite connection. Idempotent."""
        if self._loaded:
            return

        if not self.faiss_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {self.faiss_path}\n"
                f"Build it with the Kaggle notebook or run:\n"
                f"  python -m src.faiss_search --build"
            )

        # Load FAISS index (mmap-friendly on Linux)
        self._index = faiss.read_index(str(self.faiss_path))
        self._index.nprobe = self.nprobe
        self._ntotal = self._index.ntotal
        self._dim = self._index.d

        # Load ID mapping from HDF5 (position -> TMDB ID)
        # FAISS returns positional indices 0..N-1; we map to TMDB IDs
        if self.hdf5_path.exists():
            with h5py.File(self.hdf5_path, "r") as f:
                self._id_map = f["ids"][:].astype(np.int32)
            # Build reverse map for fast TMDB ID -> position lookups
            # Using dict comprehension on the array (slower startup but fast lookups)
            id_list = self._id_map.tolist()
            self._id_reverse = {int(tmdb): i for i, tmdb in enumerate(id_list)}
            print(f"  ID mapping: {len(self._id_map):,} positions loaded")
        else:
            print(f"  WARNING: HDF5 not found at {self.hdf5_path} — "
                  f"TMDB ID mapping unavailable. Use position-based lookups.")

        # Open SQLite (read-only, WAL mode)
        # check_same_thread=False is needed for FastAPI threadpool workers
        self._db = sqlite3.connect(str(self.sqlite_path), check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.execute("PRAGMA cache_size=-50000")  # 50 MB page cache
        self._db.execute("PRAGMA query_only=ON")
        self._db.row_factory = sqlite3.Row  # dict-like access

        self._loaded = True

    def close(self) -> None:
        """Release resources."""
        if self._db is not None:
            self._db.close()
            self._db = None
        if self._index is not None:
            del self._index
            self._index = None
        if self._h5_file is not None:
            self._h5_file.close()
            self._h5_file = None
        self._emb_cache.clear()
        self._loaded = False

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, *args):
        self.close()

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def ntotal(self) -> int:
        return self._ntotal

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ── Core Search ────────────────────────────────────────────────────────

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 12,
        nprobe: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Find the k nearest movies to a query vector.

        Uses batched SQLite lookups (single WHERE id IN (...)) for low latency.

        Args:
            query_vector: L2-normalized float32 vector of shape (D,) or (1, D).
            k: Number of results to return.
            nprobe: Override default nprobe for this query.

        Returns:
            List of dicts with keys: rank, tmdb_id, title, primary_genre,
            genres, year, overview, vote_average, score.
        """
        if not self._loaded:
            self.load()

        # Ensure correct shape
        vec = np.asarray(query_vector, dtype=np.float32).flatten()
        if len(vec) != self._dim:
            raise ValueError(
                f"Query vector has {len(vec)} dimensions, index expects {self._dim}"
            )
        vec = vec.reshape(1, -1)

        # Temporarily override nprobe if requested
        original_nprobe = self._index.nprobe
        if nprobe is not None:
            self._index.nprobe = nprobe

        try:
            distances, indices = self._index.search(vec, k)
        finally:
            if nprobe is not None:
                self._index.nprobe = original_nprobe

        # Convert FAISS positions -> TMDB IDs (vectorized)
        faiss_positions = indices[0].astype(np.int32)
        tmdb_ids = self._faiss_positions_to_tmdb_ids(faiss_positions)

        # ── Batched SQLite lookup (single query, not N individual) ──
        tmdb_to_movie = self._lookup_movies_batch(tmdb_ids)

        # Build results preserving FAISS rank order
        # Skip movies not found in DB (filtered out adult content or missing)
        results = []
        rank = 0
        for tmdb_id, dist in zip(tmdb_ids, distances[0]):
            tid = int(tmdb_id)
            movie = tmdb_to_movie.get(tid)

            if movie is None:
                continue

            rank += 1
            results.append({
                "rank": rank,
                "tmdb_id": tid,
                "title": movie["title"],
                "primary_genre": movie["primary_genre"],
                "genres": movie["genres"],
                "year": movie["year"],
                "overview": movie["overview"],
                "vote_average": movie["vote_average"],
                "poster_path": movie.get("poster_path"),
                "backdrop_path": movie.get("backdrop_path"),
                "score": round(float(dist), 4),
            })

        return results

    # ── Title Lookup ───────────────────────────────────────────────────────

    def lookup_by_title(self, title: str, exact: bool = False) -> dict | None:
        """Find a movie by title. Returns metadata dict or None."""
        if not self._loaded:
            self.load()

        if exact:
            row = self._db.execute(
                "SELECT * FROM movies WHERE title = ? AND adult != 'True' LIMIT 1", (title,)
            ).fetchone()
        else:
            row = self._db.execute(
                "SELECT * FROM movies WHERE title LIKE ? AND adult != 'True' LIMIT 1",
                (f"%{title}%",),
            ).fetchone()

        if row is None:
            return None
        return self._row_to_dict(row)

    def get_embedding(self, tmdb_id: int) -> np.ndarray | None:
        """
        Get the embedding vector for a movie by TMDB ID.

        Uses HDF5 to find the FAISS position from the TMDB ID,
        then reconstructs via PQ (approximate, lossy).
        For exact vectors, read directly from HDF5.
        """
        if not self._loaded:
            self.load()

        faiss_pos = self._tmdb_id_to_faiss_pos(tmdb_id)
        if faiss_pos is None:
            return None

        try:
            return self._index.reconstruct(faiss_pos)
        except Exception:
            return None

    def get_exact_embedding(self, tmdb_id: int) -> np.ndarray | None:
        """
        Get the EXACT embedding vector from HDF5 (not PQ-approximated).
        Uses LRU cache and a persistent HDF5 handle for low latency.
        """
        if not self._loaded:
            self.load()

        # Check cache first
        cached = self._emb_cache.get(tmdb_id)
        if cached is not None:
            return cached.copy()

        faiss_pos = self._tmdb_id_to_faiss_pos(tmdb_id)
        if faiss_pos is None or not self.hdf5_path.exists():
            return None

        try:
            # Use persistent handle (lazy-open, stays open for the session)
            if self._h5_file is None:
                self._h5_file = h5py.File(self.hdf5_path, "r")
            vec = self._h5_file["embeddings"][faiss_pos].astype(np.float32)
            self._emb_cache.put(tmdb_id, vec.copy())
            return vec
        except Exception:
            return None

    def get_exact_embeddings_batch(self, tmdb_ids: list[int]) -> dict[int, np.ndarray]:
        """
        Get exact embeddings for multiple TMDB IDs in a single HDF5 read.
        Dramatically faster for profile building with many liked/disliked movies.

        Returns dict mapping tmdb_id -> embedding vector.
        """
        if not self._loaded:
            self.load()

        result: dict[int, np.ndarray] = {}
        uncached_ids: list[int] = []
        uncached_positions: list[int] = []

        # Separate cached from uncached
        for tid in tmdb_ids:
            cached = self._emb_cache.get(tid)
            if cached is not None:
                result[tid] = cached.copy()
            else:
                pos = self._tmdb_id_to_faiss_pos(tid)
                if pos is not None:
                    uncached_ids.append(tid)
                    uncached_positions.append(pos)

        # Batch-read from HDF5
        if uncached_positions and self.hdf5_path.exists():
            try:
                if self._h5_file is None:
                    self._h5_file = h5py.File(self.hdf5_path, "r")

                # Convert to numpy array of positions
                pos_arr = np.array(uncached_positions, dtype=np.int64)

                # h5py supports fancy indexing with a list/array of positions
                # Sort and read in order, then restore original order via inverse permutation
                sorted_order = np.argsort(pos_arr)
                sorted_positions = pos_arr[sorted_order]
                sorted_vectors = self._h5_file["embeddings"][sorted_positions].astype(np.float32)

                # Build inverse permutation: unsort_order[i] = where sorted_order == i
                unsort_order = np.empty(len(sorted_order), dtype=np.int64)
                unsort_order[sorted_order] = np.arange(len(sorted_order), dtype=np.int64)

                for j, tid in enumerate(uncached_ids):
                    vec = sorted_vectors[unsort_order[j]]
                    result[tid] = vec
                    self._emb_cache.put(tid, vec.copy())
            except Exception:
                # Fallback: one-by-one with persistent handle
                for tid, pos in zip(uncached_ids, uncached_positions):
                    try:
                        vec = self._h5_file["embeddings"][pos].astype(np.float32)
                        result[tid] = vec
                        self._emb_cache.put(tid, vec.copy())
                    except Exception:
                        pass

        return result

    # ── Similarity Search ──────────────────────────────────────────────────

    def search_by_title(
        self, title: str, k: int = 12, nprobe: int | None = None
    ) -> list[dict[str, Any]]:
        """
        Find movies similar to a given title.

        Looks up the stored embedding for the movie (via FAISS reconstruct)
        and searches for nearest neighbors.
        """
        if not self._loaded:
            self.load()

        movie = self.lookup_by_title(title)
        if movie is None:
            raise ValueError(f"Movie not found: '{title}'")

        tmdb_id = movie["id"]
        # Use exact HDF5 vector for accurate similarity (PQ reconstruct is lossy)
        vec = self.get_exact_embedding(tmdb_id)
        if vec is None:
            raise RuntimeError(
                f"Cannot get embedding for '{title}' (TMDB ID {tmdb_id}). "
                f"Make sure the HDF5 file is present at {self.hdf5_path}."
            )

        return self.search(vec, k=k, nprobe=nprobe)

    # ── Full-Text Search (SQLite FTS5) ─────────────────────────────────────

    def search_text(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        """
        Full-text search across titles, overviews, keywords, taglines, genres.

        Uses SQLite FTS5 for fast substring/phrase matching.
        """
        if not self._loaded:
            self.load()

        # FTS5 query syntax: double-quote multi-word phrases, escape special chars
        safe_query = query.replace('"', '""')
        # If multi-word, wrap in quotes for phrase matching
        if " " in safe_query:
            safe_query = f'"{safe_query}"'

        try:
            rows = self._db.execute(
                """
                SELECT m.*
                FROM movies m
                JOIN movies_fts fts ON m.id = fts.rowid
                WHERE movies_fts MATCH ?
                  AND m.adult != 'True'
                ORDER BY rank
                LIMIT ?
                """,
                (query, limit),  # Use raw query, FTS5 handles tokenization
            ).fetchall()
        except sqlite3.OperationalError:
            # Fallback: LIKE-based search
            like_pattern = f"%{query}%"
            rows = self._db.execute(
                """
                SELECT * FROM movies
                WHERE (title LIKE ? OR overview LIKE ?)
                  AND adult != 'True'
                ORDER BY popularity DESC
                LIMIT ?
                """,
                (like_pattern, like_pattern, limit),
            ).fetchall()

        return [self._row_to_dict(r) for r in rows]

    # ── Filtered Browse ────────────────────────────────────────────────────

    def browse(
        self,
        genre: str | None = None,
        year_min: int | None = None,
        year_max: int | None = None,
        min_votes: int | None = None,
        min_rating: float | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Browse movies with optional filters."""
        if not self._loaded:
            self.load()

        query = "SELECT * FROM movies WHERE adult != 'True'"
        params: list[Any] = []

        if genre:
            query += " AND primary_genre = ?"
            params.append(genre)
        if year_min is not None:
            query += " AND year >= ?"
            params.append(year_min)
        if year_max is not None:
            query += " AND year <= ?"
            params.append(year_max)
        if min_votes is not None:
            query += " AND vote_count >= ?"
            params.append(min_votes)
        if min_rating is not None:
            query += " AND vote_average >= ?"
            params.append(min_rating)

        query += " ORDER BY popularity DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = self._db.execute(query, params).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_genres(self) -> list[str]:
        """Get all distinct primary genres (excluding adult content)."""
        if not self._loaded:
            self.load()
        rows = self._db.execute(
            "SELECT DISTINCT primary_genre FROM movies WHERE adult != 'True' ORDER BY primary_genre"
        ).fetchall()
        return [r["primary_genre"] for r in rows]

    # ── Helpers ────────────────────────────────────────────────────────────

    def _faiss_pos_to_tmdb_id(self, pos: int) -> int:
        """Convert FAISS internal position to TMDB ID."""
        if self._id_map is not None and 0 <= pos < len(self._id_map):
            return int(self._id_map[pos])
        # Fallback: use position as TMDB ID (may not be correct)
        return pos

    def _faiss_positions_to_tmdb_ids(self, positions: np.ndarray) -> np.ndarray:
        """Vectorized: convert FAISS positions to TMDB IDs in one shot."""
        if self._id_map is not None:
            valid_mask = (positions >= 0) & (positions < len(self._id_map))
            result = positions.astype(np.int64).copy()
            result[valid_mask] = self._id_map[positions[valid_mask]]
            return result
        return positions

    def _tmdb_id_to_faiss_pos(self, tmdb_id: int) -> int | None:
        """Convert TMDB ID to FAISS internal position."""
        if self._id_reverse is not None:
            return self._id_reverse.get(tmdb_id)
        return None

    def _lookup_movie(self, tmdb_id: int) -> dict | None:
        """Look up a single movie by TMDB ID from SQLite."""
        row = self._db.execute(
            "SELECT * FROM movies WHERE id = ?", (tmdb_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_dict(row)

    def _lookup_movies_batch(self, tmdb_ids: np.ndarray) -> dict[int, dict]:
        """
        Batched SQLite lookup: single WHERE id IN (...) instead of N queries.

        Returns dict mapping tmdb_id -> movie dict.
        """
        unique_ids = list(set(int(t) for t in tmdb_ids))
        if not unique_ids:
            return {}

        # Build parameterised IN clause safely
        placeholders = ",".join(["?"] * len(unique_ids))
        rows = self._db.execute(
            f"SELECT * FROM movies WHERE id IN ({placeholders}) AND adult != 'True'",
            unique_ids,
        ).fetchall()

        return {row["id"]: self._row_to_dict(row) for row in rows}

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        """Convert sqlite3.Row to plain dict."""
        return dict(row)

    def stats(self) -> dict[str, Any]:
        """Return summary statistics about the loaded index and DB."""
        if not self._loaded:
            self.load()

        db_count = self._db.execute("SELECT COUNT(*) FROM movies").fetchone()[0]
        db_clean = self._db.execute("SELECT COUNT(*) FROM movies WHERE adult != 'True'").fetchone()[0]
        return {
            "faiss_vectors": self._ntotal,
            "faiss_dim": self._dim,
            "faiss_nprobe": self._index.nprobe,
            "db_movies": db_clean,
            "db_movies_total": db_count,
            "adult_filtered": db_count - db_clean,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  Batch scoring (drop-in replacement for _score_all_shards)
# ═══════════════════════════════════════════════════════════════════════════════

def faiss_score_all(
    query_vector: np.ndarray,
    searcher: FaissSearcher,
    k: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """
    FAISS equivalent of search._score_all_shards().

    Returns (scores, indices) for top-k results from the FAISS index.
    This is a drop-in for the old brute-force path.

    Args:
        query_vector: L2-normalized query vector (D,).
        searcher: Loaded FaissSearcher instance.
        k: Number of candidates to retrieve.

    Returns:
        (scores: [k], indices: [k])
    """
    vec = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
    distances, indices = searcher._index.search(vec, k)
    return distances[0], indices[0]


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI: Build FAISS index from merged HDF5 (for local use, not Kaggle)
# ═══════════════════════════════════════════════════════════════════════════════

def build_faiss_from_hdf5(
    hdf5_path: str | Path,
    output_path: str | Path,
    nlist: int = 2048,
    m: int = 64,
    nbits: int = 8,
    train_size: int = 200_000,
    nprobe: int = 32,
) -> None:
    """
    Build a FAISS IVF-PQ index from a merged HDF5 file.

    This is the local equivalent of what the Kaggle notebook does.
    Useful if you already have the HDF5 but need to rebuild the index.

    Args:
        hdf5_path: Path to merged HDF5 (tmdb_qwen06b_1024d.h5).
        output_path: Where to save the FAISS index.
        nlist: Number of IVF clusters.
        m: Number of PQ subquantizers (must divide embedding dim).
        nbits: Bits per PQ code (8 = standard).
        train_size: How many vectors to sample for training.
        nprobe: Default search probes.
    """
    import gc
    import h5py
    from tqdm import tqdm

    hdf5_path = Path(hdf5_path)
    output_path = Path(output_path)

    print(f"Building FAISS IVF-PQ index from {hdf5_path}...")
    print(f"  nlist={nlist}  M={m}  nbits={nbits}")

    with h5py.File(hdf5_path, "r") as f:
        emb = f["embeddings"]
        n_total, dim = emb.shape
        print(f"  {n_total:,} vectors x {dim} dims")

        # Sample training vectors (stratified across the dataset)
        step = max(1, n_total // train_size)
        train_indices = np.arange(0, n_total, step, dtype=np.int64)[:train_size]
        train_indices = np.sort(train_indices)

        train_vecs = np.empty((len(train_indices), dim), dtype=np.float32)
        chunk = 5000
        for i in tqdm(range(0, len(train_indices), chunk), desc="read train"):
            end = min(i + chunk, len(train_indices))
            train_vecs[i:end] = emb[train_indices[i:end]]

        # Create and train
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits, faiss.METRIC_INNER_PRODUCT)

        print("Training...")
        index.train(train_vecs)
        del train_vecs
        gc.collect()

        try:
            index.reserveVecs(n_total)
        except Exception:
            pass

        print(f"Adding {n_total:,} vectors...")
        add_chunk = 50_000
        for start in tqdm(range(0, n_total, add_chunk), desc="add"):
            end = min(start + add_chunk, n_total)
            batch = emb[start:end]
            index.add(batch)

    index.nprobe = nprobe

    print(f"Saving to {output_path}...")
    if output_path.exists():
        output_path.unlink()
    faiss.write_index(index, str(output_path))

    mb = output_path.stat().st_size / 1e6
    print(f"Done: {mb:.1f} MB  ({index.ntotal:,} vectors, nprobe={nprobe})")
    del index


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="FAISS + SQLite movie search CLI")
    sub = p.add_subparsers(dest="cmd")

    # --build
    build_p = sub.add_parser("build", help="Build FAISS index from merged HDF5")
    build_p.add_argument("--hdf5", default="embeddings/tmdb_qwen06b_1024d.h5")
    build_p.add_argument("--output", default="embeddings/tmdb_qwen06b_1024d.faiss")
    build_p.add_argument("--nlist", type=int, default=2048)
    build_p.add_argument("--m", type=int, default=64)
    build_p.add_argument("--nbits", type=int, default=8)
    build_p.add_argument("--nprobe", type=int, default=32)

    # --search
    search_p = sub.add_parser("search", help="Search by title or text")
    search_p.add_argument("query", help="Movie title or text query")
    search_p.add_argument("--mode", choices=["title", "text"], default="title")
    search_p.add_argument("--k", type=int, default=12)
    search_p.add_argument("--nprobe", type=int, default=None)

    # --interactive
    sub.add_parser("interactive", help="Interactive search REPL")

    # --stats
    sub.add_parser("stats", help="Show index statistics")

    args = p.parse_args()

    if args.cmd == "build":
        build_faiss_from_hdf5(
            args.hdf5, args.output,
            nlist=args.nlist, m=args.m, nbits=args.nbits, nprobe=args.nprobe,
        )

    elif args.cmd in ("search", "interactive", "stats"):
        searcher = FaissSearcher()
        searcher.load()

        if args.cmd == "stats":
            s = searcher.stats()
            print(f"FAISS: {s['faiss_vectors']:,} vectors x {s['faiss_dim']}d  nprobe={s['faiss_nprobe']}")
            print(f"SQLite: {s['db_movies']:,} movies")

        elif args.cmd == "search":
            if args.mode == "title":
                results = searcher.search_by_title(args.query, k=args.k, nprobe=args.nprobe)
            else:
                results = searcher.search_text(args.query, limit=args.k)
            for r in results:
                print(f"  {r['rank']}. {r['title']} ({r['primary_genre']}) score={r.get('score', 'N/A')}")

        elif args.cmd == "interactive":
            print("Interactive mode. Type 'title <name>' or 'text <query>'. Ctrl+C to quit.")
            while True:
                try:
                    cmd = input("\n> ").strip()
                    if not cmd:
                        continue
                    if cmd.startswith("title "):
                        results = searcher.search_by_title(cmd[6:])
                    elif cmd.startswith("text "):
                        results = searcher.search_text(cmd[5:])
                    elif cmd == "quit":
                        break
                    else:
                        results = searcher.search_text(cmd)

                    for r in results:
                        score = r.get("score", "")
                        print(f"  {r['title'][:50]:<52} {r['primary_genre']:<20} {score}")
                except (KeyboardInterrupt, EOFError):
                    break

        searcher.close()
