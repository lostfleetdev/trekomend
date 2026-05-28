"""
worker.py — Standalone worker process for the recommendation job queue.

Pulls jobs from Redis, executes the FAISS + Phase 2 pipeline, and stores
results back in Redis for the client to poll.

Run as:
    uv run python -m src.worker [--worker-id N] [--once]

With --once, processes one job then exits (useful for testing).
Otherwise, runs continuously until SIGTERM/SIGINT.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
import traceback
from typing import Any

import numpy as np

from src.jobs import dequeue_job, get_job, mark_running, mark_completed, mark_failed, get_redis
from src.config import (
    DEFAULT_FAISS_INDEX, DEFAULT_SQLITE_DB, DEFAULT_EMB_DIR,
    MERGED_H5_NAME, QUERY_INSTRUCTION,
    PHASE2_ENABLED, LIGHTGBM_MODEL_PATH,
    DPP_RANK, DPP_LAMBDA_QD, DPP_CANDIDATE_POOL,
    GENRE_MAX_PER_GENRE, GENRE_MIN_UNIQUE, SERENDIPITY_POSITION,
    REDIS_URL,
)

# ============================================================================
# Global state (loaded once at startup)
# ============================================================================

searcher = None
feature_builder = None
genre_round_robin = None
dpp_selector = None
lightgbm_model = None
hdf5_path: str = ""
phase2_loaded: bool = False
_running: bool = True


def _signal_handler(sig, frame):
    global _running
    print(f"\nWorker received signal {sig}, shutting down gracefully...")
    _running = False


def load_state():
    """Load FAISS index, SQLite DB, Phase 2 modules, and Ollama connection."""
    global searcher, hdf5_path, phase2_loaded
    global feature_builder, genre_round_robin, dpp_selector, lightgbm_model

    from src.faiss_search import FaissSearcher

    hdf5_path = os.environ.get(
        "TREKOMEND_HDF5", str(DEFAULT_EMB_DIR / MERGED_H5_NAME)
    )
    faiss_path = os.environ.get("TREKOMEND_FAISS", str(DEFAULT_FAISS_INDEX))
    sqlite_path = os.environ.get("TREKOMEND_SQLITE", str(DEFAULT_SQLITE_DB))

    print(f"Worker loading FAISS index from {faiss_path}...")
    searcher = FaissSearcher(faiss_path, sqlite_path, hdf5_path=hdf5_path)
    searcher.load()
    s = searcher.stats()
    print(f"Worker ready: {s['faiss_vectors']:,} vectors x {s['faiss_dim']}d  "
          f"nprobe={s['faiss_nprobe']}  DB={s['db_movies']:,} movies")

    # Phase 2 modules
    phase2_enabled = os.environ.get("TREKOMEND_PHASE2", "1") not in ("0", "false", "no")
    if phase2_enabled and PHASE2_ENABLED:
        try:
            from src.features import FeatureBuilder
            from src.diversity import (
                GenreRoundRobin, DPPSelector,
                RoundRobinConfig, DPPConfig,
            )

            feature_builder = FeatureBuilder(searcher._db, hdf5_path=hdf5_path)
            feature_builder.compute_catalog_stats()
            print(f"  Phase 2: FeatureBuilder ready ({FeatureBuilder.N_FEATURES} features)")

            rr_cfg = RoundRobinConfig(
                max_per_genre=GENRE_MAX_PER_GENRE,
                min_unique_genres=GENRE_MIN_UNIQUE,
                serendipity_position=SERENDIPITY_POSITION,
            )
            genre_round_robin = GenreRoundRobin(searcher._db, config=rr_cfg)
            print("  Phase 2: GenreRoundRobin ready")

            dpp_cfg = DPPConfig(
                rank=DPP_RANK,
                lambda_qd=DPP_LAMBDA_QD,
                kernel_mode="full",
            )
            dpp_selector = DPPSelector(config=dpp_cfg)
            print(f"  Phase 2: DPPSelector ready (lambda_qd={DPP_LAMBDA_QD})")

            if LIGHTGBM_MODEL_PATH.exists():
                import lightgbm as lgb
                lightgbm_model = lgb.Booster(model_file=str(LIGHTGBM_MODEL_PATH))
                print(f"  Phase 2: LightGBM model loaded from {LIGHTGBM_MODEL_PATH}")
            else:
                print(f"  Phase 2: No LightGBM model at {LIGHTGBM_MODEL_PATH}. "
                      f"Train with: uv run python -m src.train_ranker")

            phase2_loaded = True
        except Exception as e:
            print(f"  Phase 2: failed to load ({e})")
            phase2_loaded = False
    else:
        print("  Phase 2: disabled")
        phase2_loaded = False


# ============================================================================
# Phase 2 pipeline (copied from main.py for worker independence)
# ============================================================================

def _phase2_rerank(
    query_tmdb_id: int | None,
    faiss_results: list[dict],
    top_k: int = 12,
    lambda_qd: float | None = None,
    skip_lightgbm: bool = False,
    use_dpp: bool = False,
) -> list[dict]:
    """Three-layer post-retrieval pipeline."""
    global feature_builder, genre_round_robin, dpp_selector, lightgbm_model

    if not phase2_loaded or not faiss_results:
        return faiss_results[:top_k]

    enriched = list(faiss_results)

    if not skip_lightgbm and lightgbm_model is not None and query_tmdb_id is not None:
        try:
            candidate_ids = [r["tmdb_id"] for r in enriched]
            candidate_scores = [r.get("score", 0) or 0 for r in enriched]

            X = feature_builder.build_features(
                query_tmdb_id=query_tmdb_id,
                candidate_tmdb_ids=candidate_ids,
                candidate_scores=candidate_scores,
            )
            new_scores = lightgbm_model.predict(X)

            for i, r in enumerate(enriched):
                r["raw_score"] = r.get("score", 0)
                r["score"] = round(float(new_scores[i]), 4)

            enriched.sort(key=lambda r: r.get("score", 0) or 0, reverse=True)
        except Exception:
            pass

    if genre_round_robin is not None:
        enriched = genre_round_robin.diversify(enriched, top_k=max(200, top_k * 4))

    if use_dpp and dpp_selector is not None and len(enriched) > top_k:
        dpp_pool = enriched[:DPP_CANDIDATE_POOL]
        dpp_ids = [r["tmdb_id"] for r in dpp_pool]

        raw_scores = np.array([r.get("score", 0) or 0 for r in dpp_pool],
                              dtype=np.float32)
        rr_position_bonus = np.linspace(0.15, 0.0, len(dpp_pool), dtype=np.float32)
        dpp_scores = raw_scores + rr_position_bonus

        emb_map = searcher.get_exact_embeddings_batch(dpp_ids)
        embs = []
        valid_indices = []
        for i, tid in enumerate(dpp_ids):
            vec = emb_map.get(tid)
            if vec is not None:
                embs.append(vec)
                valid_indices.append(i)

        if len(embs) >= top_k:
            emb_matrix = np.stack(embs).astype(np.float32)
            valid_scores = dpp_scores[valid_indices]

            if lambda_qd is not None:
                dpp_selector._cfg.lambda_qd = lambda_qd

            try:
                selected_indices, _ = dpp_selector.select(
                    emb_matrix, valid_scores, top_k=top_k,
                )
                result = [dpp_pool[valid_indices[i]] for i in selected_indices]
                for i, r in enumerate(result):
                    r["rank"] = i + 1
                return result
            except Exception:
                pass

    return enriched[:top_k]


# ============================================================================
# Job processors
# ============================================================================

def _process_similar(payload: dict[str, Any]) -> list[dict]:
    """Process a 'similar' recommendation job."""
    title = payload["title"]
    limit = payload.get("limit", 12)
    nprobe = payload.get("nprobe")
    use_phase2 = payload.get("use_phase2", True)

    movie = searcher.lookup_by_title(title)
    if movie is None:
        raise ValueError(f"Movie not found: '{title}'")

    query_tmdb_id = movie["id"]
    vec = searcher.get_exact_embedding(query_tmdb_id)
    if vec is None:
        raise ValueError(f"No embedding for '{title}' (TMDB {query_tmdb_id})")

    faiss_k = min(200, limit * 4) if use_phase2 else limit
    results = searcher.search(vec, k=faiss_k, nprobe=nprobe)

    if use_phase2 and phase2_loaded:
        results = _phase2_rerank(
            query_tmdb_id=query_tmdb_id,
            faiss_results=results,
            top_k=limit,
            use_dpp=False,
        )
    else:
        results = results[:limit]

    return results


def _process_query(payload: dict[str, Any]) -> list[dict]:
    """Process a 'query' recommendation job (needs Ollama)."""
    from src.io import embed_query
    from src.config import QUERY_INSTRUCTION

    query_text = payload["query"]
    limit = payload.get("limit", 12)
    nprobe = payload.get("nprobe")
    use_phase2 = payload.get("use_phase2", True)

    query_vec = embed_query(query_text, instruction=QUERY_INSTRUCTION)

    faiss_k = min(200, limit * 4) if use_phase2 else limit
    results = searcher.search(query_vec, k=faiss_k, nprobe=nprobe)

    if use_phase2 and phase2_loaded:
        results = _phase2_rerank(
            query_tmdb_id=None,
            faiss_results=results,
            top_k=limit,
            use_dpp=False,
        )
    else:
        results = results[:limit]

    return results


def _process_profile(payload: dict[str, Any]) -> list[dict]:
    """Process a 'profile' recommendation job."""
    from src.io import embed_hybrid

    liked_titles = payload["liked"]
    dislike_titles = payload.get("dislike", [])
    mood = payload.get("mood")
    mood_weight = payload.get("mood_weight", 0.3)
    limit = payload.get("limit", 12)
    nprobe = payload.get("nprobe")
    use_phase2 = payload.get("use_phase2", True)

    liked_movies = []
    for title in liked_titles:
        movie = searcher.lookup_by_title(title)
        if movie is None:
            raise ValueError(f"Movie not found: '{title}'")
        liked_movies.append(movie)

    disliked_movies = []
    for title in dislike_titles:
        movie = searcher.lookup_by_title(title)
        if movie is not None:
            disliked_movies.append(movie)

    all_ids = [m["id"] for m in liked_movies] + [m["id"] for m in disliked_movies]
    emb_map = searcher.get_exact_embeddings_batch(all_ids)

    positive_vecs = []
    for movie in liked_movies:
        vec = emb_map.get(movie["id"])
        if vec is None:
            raise ValueError(
                f"No embedding for '{movie['title']}' (TMDB {movie['id']})")
        positive_vecs.append(vec.astype(np.float32))

    user_vector = np.mean(positive_vecs, axis=0, dtype=np.float32)

    if disliked_movies:
        negative_vecs = []
        for movie in disliked_movies:
            vec = emb_map.get(movie["id"])
            if vec is not None:
                negative_vecs.append(vec.astype(np.float32))
        if negative_vecs:
            dislike_centroid = np.mean(negative_vecs, axis=0, dtype=np.float32)
            dislike_centroid = dislike_centroid / (np.linalg.norm(dislike_centroid) + 1e-10)
            projection = float(np.dot(user_vector, dislike_centroid))
            if projection > 0:
                user_vector = user_vector - 0.3 * projection * dislike_centroid

    if mood:
        mood_vec = embed_hybrid(mood)
        user_vector = (1 - mood_weight) * user_vector + mood_weight * mood_vec

    norm = np.linalg.norm(user_vector) + 1e-10
    user_vector = user_vector / norm

    faiss_k = min(200, limit * 4) if use_phase2 else limit
    results = searcher.search(user_vector, k=faiss_k, nprobe=nprobe)

    liked_titles_lower = {t.lower() for t in liked_titles}
    results = [r for r in results if r["title"].lower() not in liked_titles_lower]

    if use_phase2 and phase2_loaded and results:
        pseudo_query_id = liked_movies[0]["id"]
        results = _phase2_rerank(
            query_tmdb_id=pseudo_query_id,
            faiss_results=results,
            top_k=limit,
            use_dpp=False,
        )
    else:
        results = results[:limit]

    return results


def _process_diverse(payload: dict[str, Any]) -> list[dict]:
    """Process a 'diverse' recommendation job (DPP mode)."""
    title = payload["title"]
    limit = payload.get("limit", 12)
    lambda_qd = payload.get("lambda_qd", 0.5)
    nprobe = payload.get("nprobe")

    movie = searcher.lookup_by_title(title)
    if movie is None:
        raise ValueError(f"Movie not found: '{title}'")

    query_tmdb_id = movie["id"]
    vec = searcher.get_exact_embedding(query_tmdb_id)
    if vec is None:
        raise ValueError(f"No embedding for '{title}'")

    faiss_k = min(200, limit * 5)
    results = searcher.search(vec, k=faiss_k, nprobe=nprobe)

    if phase2_loaded:
        results = _phase2_rerank(
            query_tmdb_id=query_tmdb_id,
            faiss_results=results,
            top_k=limit,
            lambda_qd=lambda_qd,
            use_dpp=True,
        )
    else:
        results = results[:limit]

    return results


def _process_explore(payload: dict[str, Any]) -> list[dict]:
    """Process an 'explore' recommendation job (serendipity)."""
    liked_titles = payload["liked"]
    limit = payload.get("limit", 12)
    nprobe = payload.get("nprobe")

    liked_movies = []
    for title in liked_titles:
        movie = searcher.lookup_by_title(title)
        if movie is None:
            raise ValueError(f"Movie not found: '{title}'")
        liked_movies.append(movie)

    all_ids = [m["id"] for m in liked_movies]
    emb_map = searcher.get_exact_embeddings_batch(all_ids)
    positive_vecs = []
    for movie in liked_movies:
        vec = emb_map.get(movie["id"])
        if vec is not None:
            positive_vecs.append(vec.astype(np.float32))

    if not positive_vecs:
        raise ValueError("Could not retrieve embeddings for liked movies")

    user_vector = np.mean(positive_vecs, axis=0, dtype=np.float32)
    user_vector = user_vector / (np.linalg.norm(user_vector) + 1e-10)

    faiss_k = min(200, limit * 5)
    results = searcher.search(user_vector, k=faiss_k, nprobe=nprobe)

    liked_titles_lower = {t.lower() for t in liked_titles}
    results = [r for r in results if r["title"].lower() not in liked_titles_lower]

    if phase2_loaded and results:
        pseudo_id = liked_movies[0]["id"]
        results = _phase2_rerank(
            query_tmdb_id=pseudo_id,
            faiss_results=results,
            top_k=limit,
            lambda_qd=0.35,
            use_dpp=True,
        )
    else:
        results = results[:limit]

    return results


# ============================================================================
# Job dispatcher
# ============================================================================

PROCESSORS = {
    "similar": _process_similar,
    "query": _process_query,
    "profile": _process_profile,
    "diverse": _process_diverse,
    "explore": _process_explore,
}


def process_job(job_id: str) -> None:
    """Process a single job by ID."""
    redis_client = get_redis()

    job = get_job(job_id, redis_client=redis_client)
    if job is None:
        print(f"  Job {job_id} not found (expired?)")
        return

    job_type = job.get("type", "unknown")
    payload = job.get("payload", {})
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except (json.JSONDecodeError, TypeError):
            payload = {}

    processor = PROCESSORS.get(job_type)
    if processor is None:
        mark_failed(job_id, f"Unknown job type: {job_type}", redis_client=redis_client)
        print(f"  Job {job_id} failed: unknown type '{job_type}'")
        return

    mark_running(job_id, redis_client=redis_client)
    start_time = time.time()

    try:
        results = processor(payload)
        elapsed = time.time() - start_time
        mark_completed(job_id, results, redis_client=redis_client)
        print(f"  Job {job_id} completed ({job_type}) in {elapsed:.2f}s — "
              f"{len(results)} results")
    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = f"{type(e).__name__}: {e}"
        mark_failed(job_id, error_msg, redis_client=redis_client)
        print(f"  Job {job_id} failed ({job_type}) after {elapsed:.2f}s: {error_msg}")
        traceback.print_exc()


# ============================================================================
# Main loop
# ============================================================================

def main():
    global _running

    parser = argparse.ArgumentParser(description="Trekomend recommendation worker")
    parser.add_argument("--worker-id", type=int, default=0,
                        help="Worker ID for logging")
    parser.add_argument("--once", action="store_true",
                        help="Process one job then exit (for testing)")
    args = parser.parse_args()

    worker_tag = f"worker-{args.worker_id}"
    print(f"[{worker_tag}] Starting trekomend recommendation worker...")

    # Register signal handlers
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    # Load state
    load_state()

    print(f"[{worker_tag}] Entering job loop...")

    if args.once:
        # Process exactly one job
        print(f"[{worker_tag}] Waiting for one job (timeout 5s)...")
        job_id = dequeue_job(timeout=5)
        if job_id:
            process_job(job_id)
        else:
            print(f"[{worker_tag}] No job available, exiting.")
        return

    # Continuous loop
    while _running:
        try:
            job_id = dequeue_job(timeout=1)
            if job_id:
                process_job(job_id)
        except Exception as e:
            print(f"[{worker_tag}] Error in main loop: {e}")
            traceback.print_exc()
            time.sleep(1)

    # Shutdown
    if searcher:
        searcher.close()
    print(f"[{worker_tag}] Shut down.")


if __name__ == "__main__":
    main()
