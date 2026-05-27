"""
train_ranker.py — LightGBM LambdaRank training script for Phase 2.

Generates pseudo-labeled training pairs, extracts 15 features per pair,
and trains a LightGBM regression model for learned re-ranking.

Usage:
    # Full training
    uv run python -m src.train_ranker --output models/ranker_v1.txt

    # Quick dev iteration (fewer queries)
    uv run python -m src.train_ranker --queries 50 --output models/ranker_dev.txt

    # Cross-validate only (no model save)
    uv run python -m src.train_ranker --cv-only

    # With custom LightGBM params
    uv run python -m src.train_ranker --num-leaves 63 --lr 0.02 --output models/ranker_v2.txt

Training pipeline:
    1. Select query movies (stratified by genre, minimum quality threshold)
    2. For each query, retrieve top-200 FAISS neighbors + random negatives
    3. Build 15-feature matrix for all (query, candidate) pairs
    4. Compute pseudo-labels (composite relevance score)
    5. Train LightGBM regressor with 5-fold CV
    6. Report feature importance and validation metrics
    7. Save model to disk (~2 MB)

Memory: ~200 MB peak (150K pairs × 15 features × 4 bytes ≈ 9 MB,
                        plus embeddings in cache).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ── CLI ──

def main():
    parser = argparse.ArgumentParser(
        description="Train LightGBM re-ranker for trekomend Phase 2"
    )
    parser.add_argument("--output", type=str, default="models/ranker_v1.txt",
                        help="Path to save trained LightGBM model")
    parser.add_argument("--queries", type=int, default=250,
                        help="Number of query movies (default: 250)")
    parser.add_argument("--candidates-per-query", type=int, default=200,
                        help="FAISS candidates per query (default: 200)")
    parser.add_argument("--negatives-per-query", type=int, default=50,
                        help="Random negatives per query (default: 50)")
    parser.add_argument("--cv-only", action="store_true",
                        help="Run cross-validation only, don't save model")
    parser.add_argument("--num-leaves", type=int, default=31,
                        help="LightGBM num_leaves (default: 31)")
    parser.add_argument("--lr", type=float, default=0.05,
                        help="Learning rate (default: 0.05)")
    parser.add_argument("--n-estimators", type=int, default=200,
                        help="Number of trees (default: 200)")
    parser.add_argument("--max-depth", type=int, default=6,
                        help="Max tree depth (default: 6)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")

    args = parser.parse_args()

    # ── Import LightGBM (deferred — validates installation) ──
    try:
        import lightgbm as lgb
    except ImportError:
        print("Error: lightgbm not installed. Run: uv pip install lightgbm")
        sys.exit(1)

    # ── Initialize searcher ──
    from src.faiss_search import FaissSearcher
    from src.config import DEFAULT_FAISS_INDEX, DEFAULT_SQLITE_DB, DEFAULT_EMB_DIR, MERGED_H5_NAME

    hdf5_path = str(DEFAULT_EMB_DIR / MERGED_H5_NAME)
    searcher = FaissSearcher(
        faiss_path=str(DEFAULT_FAISS_INDEX),
        sqlite_path=str(DEFAULT_SQLITE_DB),
        hdf5_path=hdf5_path,
    )
    print("Loading FAISS index + SQLite DB...")
    searcher.load()
    stats = searcher.stats()
    print(f"  FAISS: {stats['faiss_vectors']:,} vectors x {stats['faiss_dim']}d")
    print(f"  SQLite: {stats['db_movies']:,} movies")

    # ── Initialize feature builder ──
    from src.features import FeatureBuilder, _compute_pseudo_label, _build_single_row

    feature_builder = FeatureBuilder(searcher._db, hdf5_path=hdf5_path)
    feature_builder.compute_catalog_stats()
    print("  Catalog stats computed.")

    # ═══════════════════════════════════════════════════════════════
    #  Phase 1: Select Query Movies
    # ═══════════════════════════════════════════════════════════════

    print(f"\n{'='*60}")
    print(f"  Phase 1: Selecting {args.queries} query movies...")
    print(f"{'='*60}")

    query_ids = _select_query_movies(
        searcher, feature_builder,
        n_queries=args.queries,
        min_votes=100,
        min_rating=5.0,
        seed=args.seed,
    )
    print(f"  Selected {len(query_ids)} query movies")

    # ═══════════════════════════════════════════════════════════════
    #  Phase 2: Retrieve Candidates
    # ═══════════════════════════════════════════════════════════════

    print(f"\n{'='*60}")
    print(f"  Phase 2: Retrieving candidates for {len(query_ids)} queries...")
    print(f"{'='*60}")

    all_pairs: list[tuple[int, int, float]] = []
    # (query_tmdb_id, candidate_tmdb_id, cosine_score)

    for i, qid in enumerate(query_ids):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Query {i+1}/{len(query_ids)}...")

        # Get query embedding
        q_emb = searcher.get_exact_embedding(qid)
        if q_emb is None:
            continue

        # FAISS top-200
        try:
            faiss_results = searcher.search(q_emb, k=args.candidates_per_query)
        except Exception:
            continue

        for r in faiss_results:
            if r["tmdb_id"] != qid:  # exclude self
                all_pairs.append((qid, r["tmdb_id"], r.get("score", 0) or 0))

        # Random negatives (movies with low cosine to query)
        # Sample random indices from the full catalog
        n_negatives = args.negatives_per_query
        random_ids = _sample_random_movies(
            searcher, n=n_negatives,
            exclude=set(r["tmdb_id"] for r in faiss_results) | {qid},
            seed=args.seed + i,
        )
        for rid in random_ids:
            all_pairs.append((qid, rid, 0.0))  # approximate score as 0

    print(f"  Total pairs: {len(all_pairs):,}")

    # ═══════════════════════════════════════════════════════════════
    #  Phase 3: Build Features
    # ═══════════════════════════════════════════════════════════════

    print(f"\n{'='*60}")
    print(f"  Phase 3: Building features for {len(all_pairs):,} pairs...")
    print(f"{'='*60}")

    t0 = time.perf_counter()

    # Batch-fetch all metadata
    all_ids = set()
    for qid, cid, _ in all_pairs:
        all_ids.add(qid)
        all_ids.add(cid)
    print(f"  Fetching metadata for {len(all_ids):,} unique movie IDs...")
    metadata = feature_builder._lookup_movies_batch(list(all_ids))

    # Batch-fetch all needed embeddings (queries only, for norm feature)
    # This is the expensive part — we minimize it by only loading query embeddings
    unique_queries = list(set(qid for qid, _, _ in all_pairs))
    print(f"  Fetching embeddings for {len(unique_queries)} queries...")
    query_embs = searcher.get_exact_embeddings_batch(unique_queries)

    # Build feature matrix
    N = len(all_pairs)
    X = np.zeros((N, FeatureBuilder.N_FEATURES), dtype=np.float32)
    y = np.zeros(N, dtype=np.float32)

    for i, (qid, cid, score) in enumerate(all_pairs):
        if (i + 1) % 50_000 == 0:
            print(f"  Feature {i+1}/{N}...")

        q_meta = metadata.get(qid)
        c_meta = metadata.get(cid)

        if q_meta is None or c_meta is None:
            X[i, 0] = score
            y[i] = score
            continue

        X[i] = _build_single_row(
            q_meta, c_meta, score,
            feature_builder._popularity_percentile,
            query_embs.get(qid),
            None,  # skip candidate norms for training (expensive, minor feature)
        )
        y[i] = _compute_pseudo_label(q_meta, c_meta, score)

    elapsed = time.perf_counter() - t0
    print(f"  Feature matrix: {X.shape}  labels shape: {y.shape}")
    print(f"  Build time: {elapsed:.1f}s")

    # ── Handle NaN/Inf ──
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    # ═══════════════════════════════════════════════════════════════
    #  Phase 4: Train / Cross-Validate
    # ═══════════════════════════════════════════════════════════════

    print(f"\n{'='*60}")
    print(f"  Phase 4: Training LightGBM...")
    print(f"{'='*60}")

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=args.seed,
    )

    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": args.num_leaves,
        "max_depth": args.max_depth,
        "learning_rate": args.lr,
        "n_estimators": args.n_estimators,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": args.seed,
        "n_jobs": -1,
        "verbosity": 1,
        "force_row_wise": True,  # faster on many features
    }

    print(f"  Params: num_leaves={args.num_leaves}, lr={args.lr}, "
          f"n_estimators={args.n_estimators}, max_depth={args.max_depth}")
    print(f"  Train: {X_train.shape[0]:,} pairs  Val: {X_val.shape[0]:,} pairs")

    t0 = time.perf_counter()

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=[
            lgb.early_stopping(20),
            lgb.log_evaluation(50),
        ],
    )

    train_time = time.perf_counter() - t0
    print(f"  Training time: {train_time:.1f}s")
    print(f"  Best iteration: {model.best_iteration_}")

    # ── Validation metrics ──
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)
    print(f"  Validation RMSE: {rmse:.4f}")
    print(f"  Validation R²:   {r2:.4f}")

    # ── Feature importance ──
    # ── Feature importance ──
    print(f"\n  Feature Importance (gain):")
    print(f"  {'#':<4} {'Feature':<30} {'Gain':<10}")
    print(f"  {'-'*4} {'-'*30} {'-'*10}")

    importance = model.feature_importances_
    ranked = np.argsort(importance)[::-1]
    for rank, idx in enumerate(ranked):
        name = FeatureBuilder.FEATURE_NAMES[idx]
        gain = importance[idx]
        print(f"  {rank+1:<4} {name:<30} {gain:.3f}")

    # Save feature importance to JSON
    importance_dict = {
        FeatureBuilder.FEATURE_NAMES[i]: float(importance[i])
        for i in range(len(importance))
    }
    importance_path = Path(args.output).with_suffix(".importance.json")
    importance_path.parent.mkdir(parents=True, exist_ok=True)
    with open(importance_path, "w") as f:
        json.dump(importance_dict, f, indent=2)
    print(f"\n  Feature importance saved to {importance_path}")

    # ═══════════════════════════════════════════════════════════════
    #  Phase 5: Save Model
    # ═══════════════════════════════════════════════════════════════

    if not args.cv_only:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        model.booster_.save_model(str(output_path))
        size_kb = output_path.stat().st_size / 1024
        print(f"\n  Model saved to {output_path} ({size_kb:.0f} KB)")
    else:
        print(f"\n  CV-only mode — model not saved.")

    print(f"\n{'='*60}")
    print(f"  Done.")
    print(f"{'='*60}")

    searcher.close()


# ═══════════════════════════════════════════════════════════════════════════════
#  Helper functions
# ═══════════════════════════════════════════════════════════════════════════════

def _select_query_movies(
    searcher,
    feature_builder,
    n_queries: int = 250,
    min_votes: int = 100,
    min_rating: float = 5.0,
    seed: int = 42,
) -> list[int]:
    """
    Select query movies for training — stratified by genre.
    Returns list of TMDB IDs.
    """
    rng = np.random.default_rng(seed)

    # Get all distinct primary genres
    genres = searcher.get_genres()
    # Keep only major genres (exclude tiny ones)
    major_genres = [
        g for g in genres
        if g not in ("Unknown", "", None)
    ][:15]  # top 15 genres

    queries_per_genre = max(10, n_queries // len(major_genres))

    query_ids = []
    for genre in major_genres:
        rows = searcher._db.execute(
            """
            SELECT id FROM movies
            WHERE primary_genre = ?
              AND vote_count >= ?
              AND vote_average >= ?
              AND year IS NOT NULL
            ORDER BY vote_count DESC
            LIMIT ?
            """,
            (genre, min_votes, min_rating, queries_per_genre * 2),
        ).fetchall()

        ids = [r["id"] for r in rows]
        if len(ids) > queries_per_genre:
            ids = list(rng.choice(ids, queries_per_genre, replace=False))
        query_ids.extend(ids)

    # Trim to n_queries
    if len(query_ids) > n_queries:
        query_ids = list(rng.choice(query_ids, n_queries, replace=False))

    return query_ids


def _sample_random_movies(
    searcher,
    n: int,
    exclude: set[int],
    seed: int = 42,
) -> list[int]:
    """Sample random movie IDs from the catalog, excluding a set."""
    rng = np.random.default_rng(seed)

    # Use random rowid sampling — much faster than OFFSET
    n_total = searcher._db.execute("SELECT MAX(id) FROM movies").fetchone()[0]
    if n_total is None:
        n_total = 2_000_000  # fallback

    sampled: list[int] = []
    max_attempts = n * 20
    attempts = 0
    while len(sampled) < n and attempts < max_attempts:
        # Generate random TMDB IDs in the known range
        random_id = int(rng.integers(1, int(n_total * 1.5)))
        if random_id in exclude:
            attempts += 1
            continue
        row = searcher._db.execute(
            "SELECT id FROM movies WHERE id = ?", (random_id,)
        ).fetchone()
        if row:
            sampled.append(row["id"])
        attempts += 1

    return sampled


if __name__ == "__main__":
    main()
