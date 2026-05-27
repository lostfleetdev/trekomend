"""
diversity.py — Phase 2 diversity layers for recommendation re-ranking.

Two complementary diversity mechanisms:

1. GenreRoundRobin — zero-training genre spread
   Groups FAISS candidates by primary_genre and interleaves round-robin
   with per-genre caps and a reserved serendipity slot. Zero latency
   overhead (batch-SQLite only).

2. DPPSelector — Determinantal Point Process for set diversity
   Low-rank quality-diversity kernel with greedy MAP inference,
   based on Gartrell et al. (KDD 2016, arXiv:1602.05436).

References:
  - DPP for ML: Kulesza & Taskar (2012), Foundations & Trends in ML
  - Low-rank DPP: Gartrell, Paquet, Koenigstein (KDD 2016)
  - Personalized DPP: Ibrahim et al. (RecSoGood 2025)
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
#  Genre Round-Robin Diversification
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RoundRobinConfig:
    """Constraints for genre round-robin diversification."""
    max_per_genre: int = 3         # Max candidates from same genre in final top-K
    min_unique_genres: int = 3     # Min distinct primary genres in final top-K
    serendipity_position: int = 7  # 0-indexed position for serendipity slot (pos 8)
    serendipity_min_rating: float = 7.0
    serendipity_min_votes: int = 100


class GenreRoundRobin:
    """
    Zero-training genre diversification layer.

    Algorithm:
      1. Receive top-N candidates from FAISS (TMDB IDs with scores)
      2. Batch-lookup primary_genre from SQLite (single query)
      3. Group by primary_genre, sort within each group by score descending
      4. Round-robin interleave: pick best from each genre in rotation
      5. Apply soft constraints (max-per-genre cap, min-unique-genres floor)
      6. Reserve one "serendipity slot" for a well-rated but genre-different item
    """

    def __init__(
        self,
        db: sqlite3.Connection,
        config: RoundRobinConfig | None = None,
    ):
        self._db = db
        self._cfg = config or RoundRobinConfig()

    def diversify(
        self,
        candidates: list[dict[str, Any]],
        top_k: int = 12,
    ) -> list[dict[str, Any]]:
        """
        Re-order candidates for genre spread.

        Args:
            candidates: List from FaissSearcher.search(), each with
                        {tmdb_id, score, primary_genre, ...}
            top_k: Number of results to return

        Returns:
            Re-ordered list of same dicts, top_k items, with genre spread
        """
        if not candidates:
            return []

        n = min(len(candidates), max(200, top_k * 4))

        # ── Batch-fetch genres for any candidates missing primary_genre ──
        candidate_ids = [c["tmdb_id"] for c in candidates[:n]]
        genre_map = self._batch_genre_lookup(candidate_ids)

        # Attach primary_genre to each candidate
        enriched: list[dict] = []
        for c in candidates[:n]:
            tid = c["tmdb_id"]
            c = dict(c)  # shallow copy to avoid mutating input
            c["primary_genre"] = c.get("primary_genre") or genre_map.get(tid, "Unknown")
            enriched.append(c)

        # ── Group by primary_genre ──
        groups: dict[str, list[dict]] = {}
        for c in enriched:
            genre = c["primary_genre"] or "Unknown"
            groups.setdefault(genre, []).append(c)

        # Sort within each group by score (descending)
        # Also ensure all items have numeric scores
        for g in groups.values():
            g.sort(key=lambda c: float(c.get("score", 0) or 0), reverse=True)

        # ── Round-robin interleave ──
        selected: list[dict] = []
        used_genres: dict[str, int] = {}  # genre -> count selected
        genre_iterators = {g: iter(items) for g, items in groups.items()}
        genre_order = sorted(groups.keys(), key=lambda g: groups[g][0]["score"]
                             if groups[g] else 0.0, reverse=True)

        # Continue until we have top_k or exhaust candidates
        rot_idx = 0
        while len(selected) < top_k and genre_iterators:
            # Filter out genres that have hit the cap
            eligible_genres = [
                g for g in genre_order
                if g in genre_iterators
                and used_genres.get(g, 0) < self._cfg.max_per_genre
            ]
            if not eligible_genres:
                # All active genres at cap; fall back to any remaining
                eligible_genres = [g for g in genre_order if g in genre_iterators]
                if not eligible_genres:
                    break

            genre = eligible_genres[rot_idx % len(eligible_genres)]
            iterator = genre_iterators[genre]

            try:
                item = next(iterator)
                selected.append(item)
                used_genres[genre] = used_genres.get(genre, 0) + 1
            except StopIteration:
                del genre_iterators[genre]
                continue

            rot_idx += 1

        # ── Serendipity slot ──
        if self._cfg.serendipity_position < len(selected):
            # Find a well-rated, vote-backed movie not in the dominant genres
            top_genres = set(
                sorted(used_genres, key=used_genres.get, reverse=True)[:5]
            )
            serendipity_candidates = [
                c for c in enriched
                if c not in selected
                and c.get("primary_genre") not in top_genres
                and (c.get("vote_average") or 0) >= self._cfg.serendipity_min_rating
                and (c.get("vote_count") if "vote_count" in c
                     else float("inf")) >= self._cfg.serendipity_min_votes
            ]
            if serendipity_candidates:
                # Pick highest-scoring serendipity candidate
                serendipity_candidates.sort(
                    key=lambda c: float(c.get("score", 0) or 0), reverse=True
                )
                selected.insert(self._cfg.serendipity_position,
                                serendipity_candidates[0])
                selected = selected[:top_k]  # trim

        # ── Final check: min_unique_genres ──
        final_genres = {c.get("primary_genre") for c in selected[:top_k]}
        if len(final_genres) < self._cfg.min_unique_genres:
            # Inject a diverse item from an unused genre
            for c in enriched:
                if c not in selected and c.get("primary_genre") not in final_genres:
                    selected.append(c)
                    final_genres.add(c["primary_genre"])
                    if len(final_genres) >= self._cfg.min_unique_genres:
                        break
            selected = selected[:top_k]

        return selected[:top_k]

    def _batch_genre_lookup(self, tmdb_ids: list[int]) -> dict[int, str]:
        """Batch-fetch primary_genre from SQLite for given TMDB IDs."""
        if not tmdb_ids:
            return {}
        unique_ids = list(set(tmdb_ids))
        placeholders = ",".join(["?"] * len(unique_ids))
        rows = self._db.execute(
            f"SELECT id, primary_genre FROM movies WHERE id IN ({placeholders})",
            unique_ids,
        ).fetchall()
        return {row["id"]: row["primary_genre"] or "Unknown" for row in rows}


# ═══════════════════════════════════════════════════════════════════════════════
#  DPP Selector
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DPPConfig:
    """
    Configuration for the DPP diversity selector.

    rank:         Low-rank dimension for the quality-diversity kernel.
                  Higher = more accurate but slower. r ∈ [10, 50].
    lambda_qd:    Quality vs diversity trade-off.
                  1.0 = pure quality ranking (no diversity adjustment)
                  0.0 = pure diversity (maximally spread embeddings)
                  0.7 = recommended for movie recommendations
    temperature:  Softmax-like scaling of quality scores. > 1 flattens
                  the score distribution (more uniform). < 1 sharpens.
    kernel_mode:  'full'   — compute exact N×N kernel (accurate, O(N²d))
                  'lowrank' — approximate via rank-r factorization (faster)
    """
    rank: int = 20
    lambda_qd: float = 0.7
    temperature: float = 1.0
    kernel_mode: str = "full"  # 'full' or 'lowrank'


class DPPSelector:
    """
    Determinantal Point Process for set diversity.

    Given N candidates with embeddings and relevance scores, select a
    diverse subset of K items using the Quality-Diversity kernel
    decomposition from Gartrell et al. (KDD 2016):

        L_ij = q_i × (λ·S_ij + (1-λ)·I_ij) × q_j

    where q_i = relevance quality from LightGBM/cosine,
          S_ij = cosine similarity of embeddings,
          λ    = quality vs diversity trade-off.

    Greedy MAP inference selects items maximizing the log-determinant
    of the kernel submatrix, using the matrix determinant lemma for
    O(K²) per-candidate scoring (vs naive O(K³) determinant recomputation).
    """

    def __init__(self, config: DPPConfig | None = None):
        self._cfg = config or DPPConfig()

    def select(
        self,
        embeddings: np.ndarray,
        scores: np.ndarray,
        top_k: int = 12,
    ) -> tuple[list[int], np.ndarray]:
        """
        Select a diverse subset via DPP greedy MAP inference.

        Args:
            embeddings: (N, D) float32 array of L2-normalized candidate embeddings.
            scores:     (N,) float array of relevance scores (higher = more relevant).
            top_k:      Number of items to select.

        Returns:
            (selected_indices, log_determinants)
            selected_indices: list of K indices into embeddings/scores,
                              in greedy selection order.
            log_determinants: (K,) array of log-det(L_S) at each step.
        """
        N = embeddings.shape[0]
        if N == 0:
            return [], np.array([])

        # ── Quality vector q ─────────────────────────────────────────────
        # Normalize scores to [q_min, 1.0] to avoid degenerate kernel (det=0)
        q = np.asarray(scores, dtype=np.float64).copy()
        q = q - q.min()
        q_range = q.max()
        if q_range < 1e-8:
            q = np.full(N, 0.5)  # all equal quality
        else:
            q = 0.05 + 0.95 * q / q_range  # [0.05, 1.0]

        # Apply temperature
        if self._cfg.temperature != 1.0:
            q = np.exp(np.log(q + 1e-10) / self._cfg.temperature)

        # ── Similarity kernel S ──────────────────────────────────────────
        lam = self._cfg.lambda_qd
        eye = np.eye(N, dtype=np.float64)
        S = embeddings.astype(np.float64) @ embeddings.astype(np.float64).T

        # ── Quality-diversity kernel L ───────────────────────────────────
        # L = diag(q) @ (λ·S + (1-λ)·I) @ diag(q)
        L_kernel = lam * S + (1.0 - lam) * eye
        L = np.outer(q, q) * L_kernel

        # Small ridge on diagonal for numerical stability
        L += 1e-8 * np.eye(N)

        # ── Greedy MAP inference ─────────────────────────────────────────
        selected: list[int] = []
        remaining = set(range(N))
        log_dets: list[float] = []

        # We maintain the Cholesky factor of L_S for efficient determinant updates.
        # After adding item i, we update via the matrix determinant lemma:
        #   det(L_{S∪{i}}) = det(L_S) × (L_ii - L_{iS} @ L_S^{-1} @ L_{Si})
        # Instead of inverting L_S, we solve against its Cholesky factor.

        L_S_inv: np.ndarray | None = None  # inverse of L_S (small K×K)
        L_S_chol: np.ndarray | None = None  # Cholesky of L_S

        for step in range(min(top_k, N)):
            best_idx = -1
            best_gain = -float("inf")

            if step == 0:
                # First pick: just maximize q_i (quality)
                for i in remaining:
                    gain = np.log(L[i, i] + 1e-30)
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = i
            else:
                # Use matrix determinant lemma for efficiency
                # gain_i = log(L_ii - L_{iS} @ L_S_inv @ L_{Si})
                for i in remaining:
                    L_iS = L[i, list(selected)]
                    # Solve: v = L_S_inv @ L_{Si}
                    v = L_S_inv @ L_iS
                    gain = L[i, i] - float(L_iS @ v)
                    log_gain = np.log(max(gain, 1e-30))

                    if log_gain > best_gain:
                        best_gain = log_gain
                        best_idx = i

            if best_idx < 0:
                break

            selected.append(best_idx)
            remaining.discard(best_idx)
            log_dets.append(best_gain)

            # Update Cholesky/inverse of L_S for next iteration
            # Using block Cholesky update: O(K²) per step
            k = len(selected)
            if k == 1:
                L_S_inv = np.array([[1.0 / L[best_idx, best_idx]]])
            else:
                # Block inverse update via Sherman-Morrison-Woodbury
                # Partition L_new = [[L_S, u], [u^T, l_nn]]
                # Then L_new^{-1} can be computed from L_S^{-1} in O(K²)
                u = L[np.ix_([best_idx], selected[:-1])].flatten()  # shape (k-1,)
                l_nn = L[best_idx, best_idx]  # scalar

                # Schur complement
                schur = l_nn - float(u @ L_S_inv @ u)
                inv_schur = 1.0 / max(schur, 1e-30)

                # Update inverse
                v = L_S_inv @ u  # (k-1,)
                w = v * inv_schur  # (k-1,)

                # Build new inverse
                L_S_new = np.zeros((k, k), dtype=np.float64)
                L_S_new[:k-1, :k-1] = L_S_inv + np.outer(w, v)
                L_S_new[:k-1, k-1] = -w
                L_S_new[k-1, :k-1] = -w
                L_S_new[k-1, k-1] = inv_schur
                L_S_inv = L_S_new

        return selected, np.array(log_dets)


# ═══════════════════════════════════════════════════════════════════════════════
#  Pipeline orchestrator (thin wrapper)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DiversityResult:
    """Output from the diversity pipeline."""
    results: list[dict[str, Any]]       # Final top-K results
    phase2_applied: bool = True
    round_robin_applied: bool = True
    dpp_applied: bool = True
    serendipity_inserted: bool = False
