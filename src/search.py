"""
Search strategies for finding similar movies.

All functions return: list of (rank, title, genre, score) tuples.
"""
from pathlib import Path

import numpy as np

from .config import RRF_K
from .io import scan_shards, load_all, load_shard_array, embed_query


def _score_all_shards(query_vector: np.ndarray, emb_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Dot every stored movie against a single query vector.
    Does NOT load all shards into RAM — scores each shard independently.

    Returns (scores: [N], global_indices: [N]).
    """
    shards = scan_shards(emb_dir)
    all_scores = []
    all_indices = []
    offset = 0
    for s in shards:
        emb = load_shard_array(s["file"])
        scores = emb @ query_vector
        all_scores.append(scores)
        all_indices.append(np.arange(offset, offset + len(scores)))
        offset += len(scores)

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
    emb, df, rows, shards = load_all(emb_dir, dataset_path)
    if emb is None:
        print("No shards found.")
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
    results[0] = (results[0][0], results[0][1], results[0][2], results[0][3])

    print(f"\n  Most similar to: '{q_title}'  ({q_genre})")
    print(f"  {emb.shape[0]:,} movies in {len(shards)} shards\n")
    _print_results(results, self_index=qi)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  Text query (Ollama) — single query
# ═══════════════════════════════════════════════════════════════════════════════

def search_by_query(query_text: str, emb_dir: Path, dataset_path: str,
                    ollama_model: str, top_n: int = 12):
    """
    Embed `query_text` via Ollama and search all shards.

    This is the intended Qwen3 search flow — query gets an instruction prefix,
    stored movie embeddings are raw text.
    """
    print(f"  Embedding via Ollama ({ollama_model})...")
    try:
        q_vec = embed_query(query_text, model=ollama_model)
    except Exception as e:
        print(f"  Error contacting Ollama: {e}")
        print(f"  Make sure Ollama is running and '{ollama_model}' is pulled:")
        print(f"    ollama pull {ollama_model}")
        return []

    print(f"  Vector: {len(q_vec)}-dim, norm={np.linalg.norm(q_vec):.4f}")

    shards = scan_shards(emb_dir)
    total = sum(s["rows"] for s in shards)
    print(f"  Searching {total:,} movies across {len(shards)} shards...")

    scores, indices = _score_all_shards(q_vec, emb_dir)

    print(f"  Loading metadata...")
    _, df, _, _ = load_all(emb_dir, dataset_path)

    print(f"\n  Search: '{query_text}'")
    print(f"  Across: {total:,} movies in {len(shards)} shards\n")
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
        emb, df, _, _ = load_all(emb_dir, dataset_path)
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

    shards = scan_shards(emb_dir)
    total = sum(s["rows"] for s in shards)

    scores, indices = _score_all_shards(blended, emb_dir)
    _, df, _, _ = load_all(emb_dir, dataset_path)

    print(f"\n  Blend: {', '.join(labels)}")
    print(f"  Across: {total:,} movies in {len(shards)} shards\n")
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

    shards = scan_shards(emb_dir)
    total = sum(s["rows"] for s in shards)

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

    _, df, _, _ = load_all(emb_dir, dataset_path)

    print(f"\n  Max-similarity across: {', '.join(labels)}")
    print(f"  Across: {total:,} movies in {len(shards)} shards\n")
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

    shards = scan_shards(emb_dir)
    total = sum(s["rows"] for s in shards)

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

    _, df, _, _ = load_all(emb_dir, dataset_path)

    print(f"\n  RRF fusion across: {', '.join(labels)}")
    print(f"  Across: {total:,} movies in {len(shards)} shards\n")
    results = _top_results(top_scores, top_indices, df, top_n)
    _print_results(results)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  Display
# ═══════════════════════════════════════════════════════════════════════════════

def _print_results(results: list[tuple], self_index: int = None):
    print(f"  {'Rank':<6}{'Title':<45}{'Genre':<22}{'Score'}")
    print(f"  {'-'*5} {'-'*45} {'-'*22} {'-'*5}")
    for rank, title, genre, score in results:
        mark = " <--" if self_index is not None and rank == 1 else ""
        print(f"  {rank:<6}{title:<45}{genre:<22}{score:.4f}{mark}")
