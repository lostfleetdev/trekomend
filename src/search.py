"""
Search strategies for finding similar movies.

All functions return: list of (rank, title, genre, score) tuples.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RRF_K
from .io import (
    scan_shards, load_all, load_shard_array, load_merged,
    embed_query, embed_mood, embed_genre, embed_hybrid,
)


def _total_movies(emb_dir: Path) -> tuple[int, int]:
    """Return (n_movies, n_shards) — prefers merged file count."""
    emb_m, _ids_m, _rows_m, _name = load_merged(emb_dir)
    if emb_m is not None:
        return emb_m.shape[0], 1
    shards = scan_shards(emb_dir)
    return sum(s["rows"] for s in shards), len(shards)


def _score_all_shards(query_vector: np.ndarray, emb_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Dot every stored movie against a single query vector.
    Prefers merged HDF5 if present (single-pass, faster); falls back to shard scanning.

    Returns (scores: [N], global_indices: [N]).
    """
    # Prefer merged file — one contiguous array, single matmul
    emb_m, ids_m, rows_m, _name = load_merged(emb_dir)
    if emb_m is not None:
        scores = emb_m @ query_vector
        indices = np.arange(len(scores))
        return scores, indices

    # Fallback: score shards independently
    shards = scan_shards(emb_dir)
    all_scores = []
    all_indices = []
    offset = 0
    for s in shards:
        emb = load_shard_array(s["file"])
        if emb.shape[1] != len(query_vector):
            continue  # skip dimension-mismatched shards (e.g. old 768-dim)
        scores = emb @ query_vector
        all_scores.append(scores)
        all_indices.append(np.arange(offset, offset + len(scores)))
        offset += len(scores)

    if not all_scores:
        raise RuntimeError("No compatible embeddings found (dimension mismatch?)")
    return np.concatenate(all_scores), np.concatenate(all_indices)


def _top_results(scores: np.ndarray, indices: np.ndarray,
                 df, top_n: int = 12) -> list[tuple]:
    """Convert raw scores into human-readable (rank, title, genre, score) tuples."""
    order = np.argsort(scores)[::-1][:top_n]
    results = []
    for rank, idx in enumerate(order):
        title = str(df.iloc[idx]["title"]) if not pd.isna(df.iloc[idx]["title"]) else f"#{idx}"
        genre = df.iloc[idx]["primary_genre"]
        results.append((rank + 1, title, genre, float(scores[idx])))
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  Single-title lookup (find a stored movie by name, search around it)
# ═══════════════════════════════════════════════════════════════════════════════

def search_by_title(title: str, emb_dir: Path, dataset_path: str, top_n: int = 12):
    """Find movies similar to the stored vector of a movie matching `title`."""
    emb, df, rows, shards, _ids = load_all(emb_dir, dataset_path)
    if emb is None:
        print("No shards found.")
        return []
    if df is None:
        print("Metadata loading failed. Check dataset path.")
        return []

    match = df[df["title"].str.lower() == title.lower()]
    if match.empty:
        match = df[df["title"].str.contains(title, case=False, na=False)]
    if match.empty:
        print(f"'{title}' not found.")
        return []

    qi = match.index[0]
    q_title = df.iloc[qi]["title"]
    q_genre = df.iloc[qi]["primary_genre"]
    sims = emb @ emb[qi]

    results = _top_results(sims, np.arange(len(emb)), df, top_n)

    print(f"\n  Most similar to: '{q_title}'  ({q_genre})")
    print(f"  {emb.shape[0]:,} movies in {len(shards)} shards\n")
    _print_results(results, self_index=qi)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  Text query (Ollama) — single query
# ═══════════════════════════════════════════════════════════════════════════════

def search_by_query(query_text: str, emb_dir: Path, dataset_path: str,
                    ollama_model: str, top_n: int = 12,
                    instruction: str | None = None):
    """
    Embed `query_text` via Ollama and search all movies.

    Uses Qwen3's asymmetric prompt: instruction on query, raw text on movies.
    Pass `instruction` to bias toward mood/genre/hybrid matching.
    """
    print(f"  Embedding via Ollama ({ollama_model})...")
    try:
        q_vec = embed_query(query_text, model=ollama_model, instruction=instruction)
    except Exception as e:
        print(f"  Error contacting Ollama: {e}")
        print(f"  Make sure Ollama is running and '{ollama_model}' is pulled:")
        print(f"    ollama pull {ollama_model}")
        return []

    print(f"  Vector: {len(q_vec)}-dim, norm={np.linalg.norm(q_vec):.4f}")

    total, n_shards = _total_movies(emb_dir)
    print(f"  Searching {total:,} movies...")

    scores, indices = _score_all_shards(q_vec, emb_dir)

    print(f"  Loading metadata...")
    _, df, _, _, _ = load_all(emb_dir, dataset_path)

    print(f"\n  Search: '{query_text}'")
    print(f"  Across: {total:,} movies\n")
    results = _top_results(scores, indices, df, top_n)
    _print_results(results)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  Multi-query strategies
# ═══════════════════════════════════════════════════════════════════════════════

def _get_vectors(items: list[str], mode: str, emb_dir: Path,
                 dataset_path: str, ollama_model: str) -> tuple[list[np.ndarray], list[str]]:
    """
    Convert a list of items (titles or text queries) into embedding vectors.

    mode = "titles": look up each item as a stored movie title
    mode = "text":   embed each item via Ollama

    Returns (vectors, labels) where labels are the display names.
    """
    if mode == "titles":
        emb, df, _, _, _ = load_all(emb_dir, dataset_path)
        vectors, labels = [], []
        for title in items:
            match = df[df["title"].str.lower() == title.lower()]
            if match.empty:
                match = df[df["title"].str.contains(title, case=False, na=False)]
            if match.empty:
                print(f"  Warning: '{title}' not found — skipping.")
                continue
            qi = match.index[0]
            vectors.append(emb[qi].copy())
            labels.append(str(df.iloc[qi]["title"]))
        return vectors, labels
    else:
        vectors, labels = [], []
        for text in items:
            try:
                v = embed_query(text, model=ollama_model)
                vectors.append(v)
                labels.append(text)
            except Exception as e:
                print(f"  Warning: failed to embed '{text[:40]}' — {e}")
        return vectors, labels


# ── Strategy implementations ──────────────────────────────────────────────────

def search_blend(items: list[str], mode: str, emb_dir: Path,
                 dataset_path: str, ollama_model: str, top_n: int = 12):
    """Average the query vectors, search once."""
    vectors, labels = _get_vectors(items, mode, emb_dir, dataset_path, ollama_model)
    if not vectors:
        return []

    blended = np.mean(vectors, axis=0)
    blended = blended / np.linalg.norm(blended)  # re-normalize

    total, n_shards = _total_movies(emb_dir)

    scores, indices = _score_all_shards(blended, emb_dir)
    _, df, _, _, _ = load_all(emb_dir, dataset_path)

    print(f"\n  Blend: {', '.join(labels)}")
    print(f"  Across: {total:,} movies\n")
    results = _top_results(scores, indices, df, top_n)
    _print_results(results)
    return results


def search_max(items: list[str], mode: str, emb_dir: Path,
               dataset_path: str, ollama_model: str, top_n: int = 12):
    """
    For each candidate movie, score = max(similarity to any query).
    Good when combined queries are from different genres — no "averaged mush."
    """
    vectors, labels = _get_vectors(items, mode, emb_dir, dataset_path, ollama_model)
    if not vectors:
        return []

    total, n_shards = _total_movies(emb_dir)

    # Score every candidate against each query vector, take the max per candidate
    best_scores = None
    best_indices = None
    for v in vectors:
        scores, indices = _score_all_shards(v, emb_dir)
        if best_scores is None:
            best_scores = scores
            best_indices = indices
        else:
            best_scores = np.maximum(best_scores, scores)

    _, df, _, _, _ = load_all(emb_dir, dataset_path)

    print(f"\n  Max-similarity across: {', '.join(labels)}")
    print(f"  Across: {total:,} movies\n")
    results = _top_results(best_scores, best_indices, df, top_n)
    _print_results(results)
    return results


def search_rrf(items: list[str], mode: str, emb_dir: Path,
               dataset_path: str, ollama_model: str, top_n: int = 12):
    """
    Run an independent search per query, then merge rankings using Reciprocal
    Rank Fusion. Each query contributes equally — a niche query's top match
    isn't drowned out by a blockbuster query's high scores.
    """
    vectors, labels = _get_vectors(items, mode, emb_dir, dataset_path, ollama_model)
    if not vectors:
        return []

    total, n_shards = _total_movies(emb_dir)

    # Run each query independently to get ranked lists
    ranked_lists = []
    for v in vectors:
        scores, indices = _score_all_shards(v, emb_dir)
        order = np.argsort(scores)[::-1]  # best first
        ranked_lists.append(indices[order].tolist())

    # Reciprocal Rank Fusion: sum 1/(k + rank) across all query results
    scores_rrf = {}
    for ranked in ranked_lists:
        for rank, doc_idx in enumerate(ranked):
            scores_rrf[doc_idx] = scores_rrf.get(doc_idx, 0.0) + 1.0 / (RRF_K + rank + 1)

    # Sort by RRF score descending
    sorted_docs = sorted(scores_rrf.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_indices = np.array([d[0] for d in sorted_docs])
    top_scores = np.array([d[1] for d in sorted_docs])

    _, df, _, _, _ = load_all(emb_dir, dataset_path)

    print(f"\n  RRF fusion across: {', '.join(labels)}")
    print(f"  Across: {total:,} movies\n")
    results = _top_results(top_scores, top_indices, df, top_n)
    _print_results(results)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  Combined search — mix titles AND text queries in one call
# ═══════════════════════════════════════════════════════════════════════════════

def search_combined(liked_titles: list[str], query_texts: list[str],
                    emb_dir: Path, dataset_path: str, ollama_model: str,
                    strategy: str = "blend", title_weight: float = 0.6,
                    top_n: int = 12):
    """
    Mix liked movie TITLES with natural-language QUERIES in one search.

    This is the most powerful vague-search mode:
      --search "Inception" "The Matrix" --also "something more philosophical"

    How it works:
      1. Look up stored vectors for each liked title
      2. Embed each query text via Ollama (with appropriate instruction)
      3. Blend all vectors with weighted averaging (title_weight)
      4. Run the selected strategy (blend/max/rrf) across all vectors

    Args:
        liked_titles: List of movie titles the user likes.
        query_texts: List of free-text queries (mood, genre, plot, etc.).
        strategy: One of 'blend', 'max', 'rrf'.
        title_weight: How much to weight title vectors vs text queries (0-1).
    """
    vectors = []
    labels = []

    # Step 1: Load stored vectors for liked titles
    if liked_titles:
        print(f"  Looking up {len(liked_titles)} title(s)...")
        emb, df, _, _, _ = load_all(emb_dir, dataset_path)
        for title in liked_titles:
            match = df[df["title"].str.lower() == title.lower()]
            if match.empty:
                match = df[df["title"].str.contains(title, case=False, na=False)]
            if match.empty:
                print(f"    Warning: '{title}' not found — skipping.")
                continue
            qi = match.index[0]
            v = emb[qi].copy()
            vectors.append(v * title_weight)
            labels.append(str(df.iloc[qi]["title"]))
            print(f"    + {df.iloc[qi]['title']}  ({df.iloc[qi]['primary_genre']})")

    # Step 2: Embed text queries via Ollama
    if query_texts:
        print(f"  Embedding {len(query_texts)} query(s) via Ollama...")
        for i, text in enumerate(query_texts):
            try:
                v = embed_query(text, model=ollama_model)
                vectors.append(v * (1 - title_weight) if liked_titles else v)
                labels.append(f'"{text}"')
                print(f"    + query: \"{text[:60]}\"  embedded ({len(v)}d)")
            except Exception as e:
                print(f"    Warning: failed to embed '{text[:40]}' — {e}")

    if not vectors:
        print("No valid vectors. Check titles and Ollama connection.")
        return []

    total, n_shards = _total_movies(emb_dir)

    # Step 3: Apply strategy
    if strategy == "max":
        best_scores = None
        best_indices = None
        for v in vectors:
            scores, indices = _score_all_shards(v, emb_dir)
            if best_scores is None:
                best_scores = scores
                best_indices = indices
            else:
                best_scores = np.maximum(best_scores, scores)
        final_scores, final_indices = best_scores, best_indices
    elif strategy == "rrf":
        ranked_lists = []
        for v in vectors:
            scores, indices = _score_all_shards(v, emb_dir)
            order = np.argsort(scores)[::-1]
            ranked_lists.append(indices[order].tolist())
        scores_rrf = {}
        for ranked in ranked_lists:
            for rank, doc_idx in enumerate(ranked):
                scores_rrf[doc_idx] = scores_rrf.get(doc_idx, 0.0) + 1.0 / (RRF_K + rank + 1)
        sorted_docs = sorted(scores_rrf.items(), key=lambda x: x[1], reverse=True)[:top_n]
        final_indices = np.array([d[0] for d in sorted_docs])
        final_scores = np.array([d[1] for d in sorted_docs])
    else:  # blend (default)
        blended = np.mean(vectors, axis=0)
        blended = blended / np.linalg.norm(blended)
        final_scores, final_indices = _score_all_shards(blended, emb_dir)

    _, df, _, _, _ = load_all(emb_dir, dataset_path)

    print(f"\n  Combined search [{strategy}]: {', '.join(labels)}")
    print(f"  Across: {total:,} movies\n")
    results = _top_results(final_scores, final_indices, df, top_n)
    _print_results(results)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  User preference profile search
# ═══════════════════════════════════════════════════════════════════════════════

def search_profile(liked_titles: list[str],
                   mood_text: str | None = None,
                   disliked_titles: list[str] | None = None,
                   emb_dir: Path | None = None, dataset_path: str = "",
                   ollama_model: str = "qwen3:0.6b",
                   mood_weight: float = 0.3, top_n: int = 12):
    """
    Build a user PREFERENCE VECTOR from liked movies + optional mood,
    then search. This is the "vague search" powerhouse.

    Algorithm (research-backed, see RESEARCH_FINDINGS.md):
      1. Weighted centroid of liked movies (all equally weighted by default)
      2. Optionally push AWAY from disliked movies
      3. Blend with mood encoding (mood_weight controls influence)
      4. L2-normalize the final preference vector
      5. Search all 1.4M movies by cosine similarity

    Example usage:
      --profile "Inception" "The Matrix" "Interstellar"
      --profile "Inception" --mood "something lighter but still smart"
      --profile "The Godfather" --dislike "Twilight" --mood "modern crime thriller"

    Args:
        liked_titles: Movies the user enjoys.
        mood_text: Natural language description of current mood/context.
        disliked_titles: Movies to push away from (negative signal).
        mood_weight: How much the mood text influences the profile (0-1).
    """
    emb, df, _, _, _ = load_all(emb_dir, dataset_path)
    if emb is None:
        print("No embeddings found.")
        return []

    dim = emb.shape[1]
    positive_vecs = []

    # Step 1: Gather liked-movie vectors
    print(f"\n  Building user profile from {len(liked_titles)} liked movie(s)...")
    for title in liked_titles:
        match = df[df["title"].str.lower() == title.lower()]
        if match.empty:
            match = df[df["title"].str.contains(title, case=False, na=False)]
        if match.empty:
            print(f"    Warning: '{title}' not found — skipping.")
            continue
        qi = match.index[0]
        positive_vecs.append(emb[qi].copy())
        print(f"    + {df.iloc[qi]['title']}  ({df.iloc[qi]['primary_genre']})")

    if not positive_vecs:
        print("No liked movies found. Try different titles.")
        if mood_text:
            print("  Falling back to mood-only search...")
            return search_by_query(mood_text, emb_dir, dataset_path, ollama_model, top_n)
        return []

    # Step 2: Compute positive centroid
    user_vector = np.mean(positive_vecs, axis=0)

    # Step 3: Push away from disliked movies
    if disliked_titles:
        print(f"\n  Pushing away from {len(disliked_titles)} disliked movie(s)...")
        negative_vecs = []
        for title in disliked_titles:
            match = df[df["title"].str.lower() == title.lower()]
            if match.empty:
                match = df[df["title"].str.contains(title, case=False, na=False)]
            if match.empty:
                print(f"    Warning: '{title}' not found — skipping.")
                continue
            qi = match.index[0]
            negative_vecs.append(emb[qi].copy())
            print(f"    - {df.iloc[qi]['title']}  ({df.iloc[qi]['primary_genre']})")

        if negative_vecs:
            dislike_centroid = np.mean(negative_vecs, axis=0)
            dislike_centroid = dislike_centroid / np.linalg.norm(dislike_centroid)
            # Project user_vector onto dislike direction and subtract
            projection = np.dot(user_vector, dislike_centroid)
            if projection > 0:
                user_vector = user_vector - 0.3 * projection * dislike_centroid
                print(f"    Adjusted away from disliked direction (proj={projection:.3f})")

    # Step 4: Blend with mood
    if mood_text:
        print(f"\n  Blending with mood: \"{mood_text[:80]}\" (weight={mood_weight:.1f})...")
        try:
            mood_vec = embed_hybrid(mood_text, model=ollama_model)
            user_vector = (1 - mood_weight) * user_vector + mood_weight * mood_vec
            print(f"    Mood vector: {len(mood_vec)}d, norm={np.linalg.norm(mood_vec):.4f}")
        except Exception as e:
            print(f"    Warning: mood embedding failed — {e}")
            print(f"    Continuing with title-only profile.")

    # Step 5: Normalize and search
    user_vector = user_vector / np.linalg.norm(user_vector)
    print(f"  Final profile vector: norm={np.linalg.norm(user_vector):.4f}")

    total, n_shards = _total_movies(emb_dir)
    scores, indices = _score_all_shards(user_vector, emb_dir)

    # Exclude already-liked movies from results
    liked_ids = set()
    for title in liked_titles:
        match = df[df["title"].str.lower() == title.lower()]
        if not match.empty:
            liked_ids.add(match.index[0])
    for lid in liked_ids:
        if lid < len(scores):
            scores[lid] = -2.0

    print(f"\n  Profile search across {total:,} movies\n")
    results = _top_results(scores, indices, df, top_n)
    _print_results(results)
    return results

def _print_results(results: list[tuple], self_index: int = None):
    print(f"  {'Rank':<6}{'Title':<45}{'Genre':<22}{'Score'}")
    print(f"  {'-'*5} {'-'*45} {'-'*22} {'-'*5}")
    for rank, title, genre, score in results:
        mark = " <--" if self_index is not None and rank == 1 else ""
        # Sanitize title for Windows console (replace non-ASCII with ?)
        safe_title = str(title).encode("ascii", errors="replace").decode("ascii")
        if len(safe_title) > 43:
            safe_title = safe_title[:40] + "..."
        safe_genre = str(genre).encode("ascii", errors="replace").decode("ascii")
        print(f"  {rank:<6}{safe_title:<45}{safe_genre:<22}{score:.4f}{mark}")
