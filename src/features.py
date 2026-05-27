"""
features.py — Feature builder for LightGBM learned re-ranker (Phase 2).

Builds a 15-feature vector for each (query, candidate) movie pair.
Uses batched SQLite and HDF5 lookups for efficiency — all features
for the top-200 candidates are computed in a single batch pass.

Feature list:
  [0]  cosine_sim              float   FAISS cosine / inner product score
  [1]  genre_jaccard           float   Jaccard: |G_q ∩ G_c| / |G_q ∪ G_c|
  [2]  genre_match_count       int     Number of overlapping genre tags
  [3]  year_diff_abs           float   |year_q - year_c| (None → 50 years)
  [4]  year_diff_bucket        int     0=same, 1=±5yr, 2=±15yr, 3=far
  [5]  popularity_percentile   float   Percentile of candidate vote_count
  [6]  embedding_norm          float   L2 norm of candidate embedding
  [7]  same_language           int     1 if original_language matches, else 0
  [8]  keyword_jaccard         float   Jaccard on parsed keywords
  [9]  vote_average_candidate  float   Candidate rating (TMDB vote_average)
  [10] vote_average_query      float   Query movie rating
  [11] rating_diff             float   |rating_q - rating_c|
  [12] runtime_ratio           float   min(r_q, r_c) / max(r_q, r_c)
  [13] same_primary_genre      int     1 if primary_genre matches, else 0
  [14] embedding_norm_query    float   L2 norm of query embedding

References:
  - Feature #6 (embedding norm): Meehan & Pauwels (RecSys 2025) —
    embedding magnitude encodes popularity even without explicit signals.
  - Feature #5 (popularity percentile): Stitch Fix (2021) —
    percentile binning for quick quality signals.
"""
from __future__ import annotations

import sqlite3
from typing import Any

import numpy as np


class FeatureBuilder:
    """
    Builds feature matrices for LightGBM ranker training and inference.

    Maintains precomputed catalog-wide statistics (vote_count percentiles,
    embedding norms) for normalization of popularity features.
    """

    # Feature names (for reporting and feature importance)
    FEATURE_NAMES = [
        "cosine_sim",
        "genre_jaccard",
        "genre_match_count",
        "year_diff_abs",
        "year_diff_bucket",
        "popularity_percentile",
        "embedding_norm_candidate",
        "same_language",
        "keyword_jaccard",
        "vote_average_candidate",
        "vote_average_query",
        "rating_diff",
        "runtime_ratio",
        "same_primary_genre",
        "embedding_norm_query",
    ]

    N_FEATURES = len(FEATURE_NAMES)

    def __init__(
        self,
        db: sqlite3.Connection,
        hdf5_path: str | None = None,
    ):
        """
        Args:
            db: SQLite connection to tmdb_movies.db.
            hdf5_path: Path to HDF5 for embedding loads (optional,
                       only needed for embedding_norm features).
        """
        self._db = db
        self._hdf5_path = hdf5_path
        self._h5_file = None  # lazy-open

        # Catalog-wide stats (computed once on demand)
        self._vote_count_percentiles: np.ndarray | None = None
        self._vote_count_bins: int = 1000
        self._stats_computed: bool = False

    # ── Catalog Statistics (precompute once) ──────────────────────────────

    def compute_catalog_stats(self) -> None:
        """
        Pre-compute catalog-wide statistics for feature normalization.
        This queries the SQLite DB once and caches percentile arrays.
        Should be called once at startup.
        """
        if self._stats_computed:
            return

        # Vote count percentile bins for popularity feature
        rows = self._db.execute(
            "SELECT vote_count FROM movies WHERE vote_count IS NOT NULL AND vote_count > 0"
        ).fetchall()
        vote_counts = np.array([r["vote_count"] for r in rows], dtype=np.float64)

        # Build percentile reference (1,000 bins for fast lookup)
        self._vote_count_percentiles = np.percentile(
            vote_counts, np.linspace(0, 100, self._vote_count_bins + 1)
        )

        self._stats_computed = True

    def _popularity_percentile(self, vote_count: int | None) -> float:
        """Map vote_count to percentile [0, 1] using precomputed bins."""
        if vote_count is None or vote_count <= 0:
            return 0.0
        if self._vote_count_percentiles is None:
            return 0.5  # fallback
        # Binary search in percentile bins
        idx = np.searchsorted(self._vote_count_percentiles, vote_count)
        return min(idx / self._vote_count_bins, 1.0)

    # ── Batch SQLite Lookups ─────────────────────────────────────────────

    def _lookup_movies_batch(self, tmdb_ids: list[int]) -> dict[int, dict[str, Any]]:
        """Fetch movie metadata for a batch of TMDB IDs (auto-chunked for SQLite limits)."""
        unique_ids = list(set(tmdb_ids))
        if not unique_ids:
            return {}

        result: dict[int, dict[str, Any]] = {}
        chunk_size = 900  # SQLite max 999 variables, keep safe margin
        for i in range(0, len(unique_ids), chunk_size):
            chunk = unique_ids[i:i + chunk_size]
            placeholders = ",".join(["?"] * len(chunk))
            rows = self._db.execute(
                f"SELECT * FROM movies WHERE id IN ({placeholders})",
                chunk,
            ).fetchall()
            for row in rows:
                result[row["id"]] = dict(row)
        return result

    # ── Feature Computation ──────────────────────────────────────────────

    def build_features(
        self,
        query_tmdb_id: int,
        candidate_tmdb_ids: list[int],
        candidate_scores: list[float] | np.ndarray,
        candidate_embeddings: np.ndarray | None = None,
        query_embedding: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Build feature matrix for a single query against N candidates.

        Args:
            query_tmdb_id: TMDB ID of the query movie.
            candidate_tmdb_ids: List of N candidate TMDB IDs.
            candidate_scores: N FAISS cosine scores.
            candidate_embeddings: (N, D) array of candidate embeddings.
                                 Required for embedding_norm features.
                                 If None, skips norm features.
            query_embedding: (D,) query embedding.
                            Required for embedding_norm features.

        Returns:
            np.ndarray of shape (N, 15), float32. Missing data → 0.
        """
        N = len(candidate_tmdb_ids)
        if N == 0:
            return np.zeros((0, self.N_FEATURES), dtype=np.float32)

        if not self._stats_computed:
            self.compute_catalog_stats()

        # ── Batch-fetch all metadata ──
        all_ids = [query_tmdb_id] + list(candidate_tmdb_ids)
        metadata = self._lookup_movies_batch(all_ids)
        query_meta = metadata.get(query_tmdb_id)

        if query_meta is None:
            # Query not in DB — build features with zero query signal
            query_meta = {}

        # ── Pre-compute query feature values (same for all candidates) ──
        q_genres_set = _parse_genre_set(query_meta.get("genres", ""))
        q_vote_average = float(query_meta.get("vote_average") or 0)
        q_year = _safe_int(query_meta.get("year"))
        q_language = str(query_meta.get("original_language") or "").lower()
        q_runtime = _safe_float(query_meta.get("runtime"))
        q_primary_genre = str(query_meta.get("primary_genre") or "").lower()
        q_keywords_set = _parse_keyword_set(query_meta.get("keywords", ""))
        q_emb_norm = float(np.linalg.norm(query_embedding)) if query_embedding is not None else 1.0

        # ── Allocate feature matrix ──
        X = np.zeros((N, self.N_FEATURES), dtype=np.float32)
        scores = np.asarray(candidate_scores, dtype=np.float32)

        for i, (tid, score) in enumerate(zip(candidate_tmdb_ids, scores)):
            cand_meta = metadata.get(tid)
            if cand_meta is None:
                # Missing metadata — fill with zeros + cosine score only
                X[i, 0] = score
                continue

            # [0] cosine_sim
            X[i, 0] = score

            # Parse candidate metadata
            c_genres_set = _parse_genre_set(cand_meta.get("genres", ""))
            c_vote_average = float(cand_meta.get("vote_average") or 0)
            c_year = _safe_int(cand_meta.get("year"))
            c_language = str(cand_meta.get("original_language") or "").lower()
            c_runtime = _safe_float(cand_meta.get("runtime"))
            c_primary_genre = str(cand_meta.get("primary_genre") or "").lower()
            c_keywords_set = _parse_keyword_set(cand_meta.get("keywords", ""))
            c_vote_count = _safe_int(cand_meta.get("vote_count"))

            # [1] genre_jaccard
            union = q_genres_set | c_genres_set
            if union and (q_genres_set or c_genres_set):
                X[i, 1] = len(q_genres_set & c_genres_set) / len(union)

            # [2] genre_match_count
            X[i, 2] = float(len(q_genres_set & c_genres_set))

            # [3] year_diff_abs
            if q_year is not None and c_year is not None:
                X[i, 3] = float(abs(q_year - c_year))
            else:
                X[i, 3] = 50.0  # default "unknown" distance

            # [4] year_diff_bucket
            diff = abs((q_year or 0) - (c_year or 0)) if q_year and c_year else None
            X[i, 4] = float(_year_diff_bucket(diff))

            # [5] popularity_percentile
            X[i, 5] = self._popularity_percentile(c_vote_count)

            # [6] embedding_norm (candidate)
            if candidate_embeddings is not None and i < len(candidate_embeddings):
                X[i, 6] = float(np.linalg.norm(candidate_embeddings[i]))

            # [7] same_language
            X[i, 7] = 1.0 if (q_language and c_language and q_language == c_language) else 0.0

            # [8] keyword_jaccard
            kw_union = q_keywords_set | c_keywords_set
            if kw_union and (q_keywords_set or c_keywords_set):
                X[i, 8] = len(q_keywords_set & c_keywords_set) / len(kw_union)

            # [9] vote_average_candidate
            X[i, 9] = c_vote_average

            # [10] vote_average_query
            X[i, 10] = q_vote_average

            # [11] rating_diff
            X[i, 11] = abs(q_vote_average - c_vote_average)

            # [12] runtime_ratio
            if q_runtime and c_runtime and q_runtime > 0 and c_runtime > 0:
                X[i, 12] = min(q_runtime, c_runtime) / max(q_runtime, c_runtime)
            else:
                X[i, 12] = 1.0

            # [13] same_primary_genre
            X[i, 13] = 1.0 if (q_primary_genre and c_primary_genre and
                               q_primary_genre == c_primary_genre) else 0.0

            # [14] embedding_norm (query)
            X[i, 14] = q_emb_norm

        return X

    @staticmethod
    def feature_matrix_for_pairs(
        db: sqlite3.Connection,
        pairs: list[tuple[int, int, float]],  # (query_id, candidate_id, score)
        query_embeddings: dict[int, np.ndarray] | None = None,
        candidate_embeddings: dict[int, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build features for a list of (query_id, candidate_id, score) pairs.
        Used by the training script — one call handles all pairs.

        Args:
            db: SQLite connection.
            pairs: List of (query_tmdb_id, candidate_tmdb_id, score) tuples.
            query_embeddings: Dict mapping tmdb_id -> embedding for queries.
            candidate_embeddings: Dict mapping tmdb_id -> embedding for candidates.

        Returns:
            (X, y) where X is (N, 15) feature matrix, y is (N,) pseudo-label vector.
            y is the composite relevance score: 0.5*cosine + 0.2*rating_norm + ...
        """
        builder = FeatureBuilder(db)
        builder.compute_catalog_stats()

        N = len(pairs)
        X = np.zeros((N, FeatureBuilder.N_FEATURES), dtype=np.float32)
        y = np.zeros(N, dtype=np.float32)

        # Collect all unique TMDB IDs
        all_ids = set()
        for qid, cid, _ in pairs:
            all_ids.add(qid)
            all_ids.add(cid)

        # Batch-fetch metadata
        metadata = builder._lookup_movies_batch(list(all_ids))

        for i, (qid, cid, score) in enumerate(pairs):
            q_meta = metadata.get(qid)
            c_meta = metadata.get(cid)

            if q_meta is None or c_meta is None:
                X[i, 0] = score
                y[i] = score  # fallback
                continue

            # Build single-row features
            row = _build_single_row(
                q_meta, c_meta, score,
                builder._popularity_percentile,
                query_embeddings.get(qid) if query_embeddings else None,
                candidate_embeddings.get(cid) if candidate_embeddings else None,
            )
            X[i] = row

            # Pseudo-label: composite relevance score
            y[i] = _compute_pseudo_label(q_meta, c_meta, score)

        return X, y


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers — feature sub-computations
# ═══════════════════════════════════════════════════════════════════════════════

def _safe_int(val: Any) -> int | None:
    """Convert to int or return None."""
    if val is None:
        return None
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None


def _safe_float(val: Any) -> float | None:
    """Convert to float or return None."""
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _parse_genre_set(genres_str: str) -> set[str]:
    """Parse comma-separated genre string into a normalized set."""
    if not genres_str or genres_str.lower() in ("nan", "none", ""):
        return set()
    return {g.strip().lower() for g in genres_str.split(",") if g.strip()}


def _parse_keyword_set(keywords_str: str) -> set[str]:
    """Parse comma-separated keyword string into a normalized set."""
    if not keywords_str or keywords_str.lower() in ("nan", "none", ""):
        return set()
    return {k.strip().lower() for k in keywords_str.split(",") if k.strip()}


def _year_diff_bucket(diff: int | None) -> int:
    """Bucket year difference into 4 categories."""
    if diff is None:
        return 3  # unknown → far
    if diff == 0:
        return 0  # same year
    if diff <= 5:
        return 1  # same era (±5 years)
    if diff <= 15:
        return 2  # adjacent era (±15 years)
    return 3      # far apart


def _build_single_row(
    q_meta: dict[str, Any],
    c_meta: dict[str, Any],
    score: float,
    popularity_fn,
    q_emb: np.ndarray | None = None,
    c_emb: np.ndarray | None = None,
) -> np.ndarray:
    """Build a single (15,) feature row for one (query, candidate) pair."""
    row = np.zeros(FeatureBuilder.N_FEATURES, dtype=np.float32)

    q_genres_set = _parse_genre_set(q_meta.get("genres", ""))
    c_genres_set = _parse_genre_set(c_meta.get("genres", ""))

    row[0] = score

    # Genre features
    union = q_genres_set | c_genres_set
    if union:
        row[1] = len(q_genres_set & c_genres_set) / len(union)
    row[2] = float(len(q_genres_set & c_genres_set))

    # Year features
    q_year = _safe_int(q_meta.get("year"))
    c_year = _safe_int(c_meta.get("year"))
    if q_year is not None and c_year is not None:
        row[3] = float(abs(q_year - c_year))
    else:
        row[3] = 50.0
    diff = abs((q_year or 0) - (c_year or 0)) if q_year and c_year else None
    row[4] = float(_year_diff_bucket(diff))

    # Popularity
    row[5] = popularity_fn(_safe_int(c_meta.get("vote_count")))

    # Embedding norms
    if c_emb is not None:
        row[6] = float(np.linalg.norm(c_emb))
    if q_emb is not None:
        row[14] = float(np.linalg.norm(q_emb))
    else:
        row[14] = 1.0

    # Language
    q_lang = str(q_meta.get("original_language") or "").lower()
    c_lang = str(c_meta.get("original_language") or "").lower()
    row[7] = 1.0 if (q_lang and c_lang and q_lang == c_lang) else 0.0

    # Keywords
    q_kw = _parse_keyword_set(q_meta.get("keywords", ""))
    c_kw = _parse_keyword_set(c_meta.get("keywords", ""))
    kw_union = q_kw | c_kw
    if kw_union:
        row[8] = len(q_kw & c_kw) / len(kw_union)

    # Vote averages
    q_va = float(q_meta.get("vote_average") or 0)
    c_va = float(c_meta.get("vote_average") or 0)
    row[9] = c_va
    row[10] = q_va
    row[11] = abs(q_va - c_va)

    # Runtime ratio
    q_rt = _safe_float(q_meta.get("runtime"))
    c_rt = _safe_float(c_meta.get("runtime"))
    if q_rt and c_rt and q_rt > 0 and c_rt > 0:
        row[12] = min(q_rt, c_rt) / max(q_rt, c_rt)
    else:
        row[12] = 1.0

    # Primary genre match
    q_pg = str(q_meta.get("primary_genre") or "").lower()
    c_pg = str(c_meta.get("primary_genre") or "").lower()
    row[13] = 1.0 if (q_pg and c_pg and q_pg == c_pg) else 0.0

    return row


def _compute_pseudo_label(
    q_meta: dict[str, Any],
    c_meta: dict[str, Any],
    cosine_score: float,
) -> float:
    """
    Compute a composite relevance score as pseudo-label for training.

    Formula (weighted):
        0.50 × cosine_sim
      + 0.15 × normalized_rating
      + 0.10 × log_vote_count_normalized
      + 0.10 × genre_match_score
      + 0.10 × keyword_overlap_score
      + 0.05 × (1.0 - year_diff_normalized)

    This provides a continuous relevance signal that LightGBM learns to
    predict — capturing feature interactions beyond the hand-crafted formula.
    """
    # Rating: normalize from [0, 10] to [0, 1]
    rating = float(c_meta.get("vote_average") or 0)
    rating_norm = np.clip(rating / 10.0, 0.0, 1.0)

    # Vote count: log-normalize
    vc = _safe_int(c_meta.get("vote_count")) or 0
    vc_norm = np.clip(np.log1p(vc) / np.log1p(100_000), 0.0, 1.0)

    # Genre match
    q_genres = _parse_genre_set(q_meta.get("genres", ""))
    c_genres = _parse_genre_set(c_meta.get("genres", ""))
    union = q_genres | c_genres
    genre_match = len(q_genres & c_genres) / len(union) if union else 0.0

    # Keyword overlap
    q_kw = _parse_keyword_set(q_meta.get("keywords", ""))
    c_kw = _parse_keyword_set(c_meta.get("keywords", ""))
    kw_union = q_kw | c_kw
    kw_overlap = len(q_kw & c_kw) / len(kw_union) if kw_union else 0.0

    # Year diff
    q_year = _safe_int(q_meta.get("year"))
    c_year = _safe_int(c_meta.get("year"))
    if q_year is not None and c_year is not None:
        year_norm = 1.0 - min(abs(q_year - c_year) / 50.0, 1.0)
    else:
        year_norm = 0.5

    return float(
        0.50 * np.clip(cosine_score, 0.0, 1.0)
        + 0.15 * rating_norm
        + 0.10 * vc_norm
        + 0.10 * genre_match
        + 0.10 * kw_overlap
        + 0.05 * year_norm
    )
