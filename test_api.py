"""
test_api.py — Verify all trekomend API endpoints work locally.

Run with:  uv run python test_api.py

Requires the data files in new-kaggle-output/ (download from HuggingFace).
Does NOT require Redis — tests graceful degradation when Redis is offline.
"""
import os
import sys

os.environ.setdefault("TREKOMEND_HDF5", "new-kaggle-output/tmdb_qwen06b_1024d.h5")

from main import app
from fastapi.testclient import TestClient


def main():
    passed = 0
    failed = 0
    errors = []

    with TestClient(app, raise_server_exceptions=False) as client:
        def check(name: str, condition: bool, detail: str = ""):
            nonlocal passed, failed
            if condition:
                print(f"  [OK]   {name}")
                passed += 1
            else:
                msg = f"  [FAIL] {name}" + (f" — {detail}" if detail else "")
                print(msg)
                errors.append(msg)
                failed += 1

        # -- Root --
        print("\n-- Root --")
        r = client.get("/")
        check("GET /", r.status_code == 200, f"status={r.status_code}")

        # -- Health & Stats --
        print("\n-- Health & Stats --")
        r = client.get("/api/health")
        check("GET /api/health", r.json().get("status") == "ok")

        r = client.get("/api/stats")
        check("GET /api/stats", "faiss_vectors" in r.json(),
              f"keys={list(r.json().keys())}")
        if r.status_code == 200:
            s = r.json()
            print(f"        {s['faiss_vectors']:,} vectors × {s['faiss_dim']}d  "
                  f"DB: {s['db_movies']:,} movies")

        # -- Movies --
        print("\n-- Movies --")
        r = client.get("/api/movies/genres")
        genres = r.json()
        check("GET /api/movies/genres", len(genres) > 10,
              f"count={len(genres)}")

        r = client.get("/api/movies/search?q=matrix&limit=3")
        check("GET /api/movies/search?q=matrix", len(r.json()) > 0,
              f"results={len(r.json())}")

        r = client.get("/api/movies/search?q=xyznotamovie")
        check("GET /api/movies/search?q=xyznotamovie (empty)", len(r.json()) == 0)

        r = client.get("/api/movies/browse?genre=Action&limit=5")
        check("GET /api/movies/browse?genre=Action", len(r.json()) == 5,
              f"results={len(r.json())}")

        r = client.get("/api/movies/browse?genre=Horror&year_min=2020&limit=3")
        check("GET /api/movies/browse?genre=Horror&year_min=2020",
              len(r.json()) > 0, f"results={len(r.json())}")

        r = client.get("/api/movies/550")
        check("GET /api/movies/550 (Fight Club)",
              r.status_code == 200 and r.json().get("title") == "Fight Club",
              f"title={r.json().get('title')}")

        r = client.get("/api/movies/99999999")
        check("GET /api/movies/99999999 (404)", r.status_code == 404)

        # -- Ranker --
        print("\n-- Ranker --")
        r = client.get("/api/ranker/features")
        check("GET /api/ranker/features", "feature_importance" in r.json(),
              f"features={len(r.json().get('feature_names', []))}")

        # -- Jobs (no Redis) --
        print("\n-- Jobs (Redis offline) --")
        r = client.get("/api/jobs/nonexistent")
        check("GET /api/jobs/nonexistent (404)", r.status_code == 404)

        # -- Rate Limiting (skipped when Redis is down) --
        print("\n-- Rate Limiting (skipped, no Redis) --")
        r = client.get("/api/stats")
        check("Rate limit headers absent (no Redis)",
              r.headers.get("x-ratelimit-limit") is None)

        # -- Recommend (no Redis -> synchronous) --
        print("\n-- Recommend (synchronous, no Redis) --")
        r = client.post("/api/recommend/similar",
                        json={"title": "Inception", "limit": 5})
        check("POST /recommend/similar (sync)",
              r.status_code == 200 and r.json().get("mode") == "sync",
              f"status={r.status_code} mode={r.json().get('mode')}")
        if r.status_code == 200 and r.json().get("results"):
            n = len(r.json()["results"])
            print(f"        {n} results in {r.json().get('elapsed_seconds', '?')}s")

        # recommend/query requires Ollama, so we just check it doesn't crash
        r = client.post("/api/recommend/query",
                        json={"query": "dark sci-fi with robots", "limit": 5})
        # May succeed (Ollama running) or 500 (Ollama not running) - both OK
        check("POST /recommend/query (sync or Ollama error)",
              r.status_code in (200, 500),
              f"status={r.status_code}")

        # -- Validation (works before Redis) --
        print("\n-- Validation --")
        r = client.post("/api/recommend/query", json={"query": ""})
        check("POST /recommend/query empty (400)", r.status_code == 400)

        r = client.post("/api/recommend/similar",
                        json={"title": "xyznotamovie"})
        check("POST /recommend/similar bad title (404)", r.status_code == 404)

    # -- Summary --
    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    if errors:
        print("\nFailures:")
        for e in errors:
            print(e)
    print()
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
