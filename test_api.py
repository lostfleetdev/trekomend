"""
test_api.py — Comprehensive integration test for the trekomend API.

Tests all endpoints, measures latency, and validates response structure.

Usage:
    uv run python test_api.py

The API server must be running on localhost:8080.
Start with:
    uv run uvicorn main:app --host 127.0.0.1 --port 8080 --workers 1
"""
import json
import sys
import time
import urllib.request
import urllib.error
from typing import Any

# Fix Windows console encoding for Unicode movie titles
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


BASE = "http://127.0.0.1:8081"
PASSED = 0
FAILED = 0
TIMINGS: list[tuple[str, float]] = []


def request(method: str, path: str, body: dict | None = None,
            timeout: int = 30) -> tuple[int, Any, float]:
    """Make an HTTP request, return (status, data, elapsed_ms)."""
    url = f"{BASE}{path}"
    data_bytes = json.dumps(body).encode("utf-8") if body else None

    req = urllib.request.Request(
        url,
        data=data_bytes,
        headers={"Content-Type": "application/json"} if data_bytes else {},
        method=method,
    )

    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.status
            raw = resp.read()
            elapsed = (time.perf_counter() - t0) * 1000
            try:
                return status, json.loads(raw), elapsed
            except json.JSONDecodeError:
                return status, raw.decode("utf-8"), elapsed
    except urllib.error.HTTPError as e:
        elapsed = (time.perf_counter() - t0) * 1000
        try:
            body_data = json.loads(e.read())
        except Exception:
            body_data = {"error": str(e)}
        return e.code, body_data, elapsed
    except Exception as e:
        elapsed = (time.perf_counter() - t0) * 1000
        return 0, {"error": str(e)}, elapsed


def check(name: str, condition: bool, detail: str = "") -> None:
    """Assert a condition, counting pass/fail."""
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  ✓ {name}")
    else:
        FAILED += 1
        print(f"  ✗ {name}  — {detail}")


def record(name: str, elapsed_ms: float) -> None:
    TIMINGS.append((name, elapsed_ms))


# ═══════════════════════════════════════════════════════════════════
#  Test Suite
# ═══════════════════════════════════════════════════════════════════

def test_health():
    print("\n── /health ──")
    status, data, elapsed = request("GET", "/health")
    record("GET /health", elapsed)
    check("status 200", status == 200, f"got {status}")
    check("status is ok", data.get("status") == "ok", f"got {data}")
    print(f"  ⏱ {elapsed:.1f} ms")


def test_stats():
    print("\n── /stats ──")
    status, data, elapsed = request("GET", "/stats")
    record("GET /stats", elapsed)
    check("status 200", status == 200, f"got {status}")
    check("has faiss_vectors", "faiss_vectors" in data, f"keys: {list(data.keys())}")
    check("has db_movies", "db_movies" in data)
    check("vectors > 100K", data.get("faiss_vectors", 0) > 100_000,
          f"got {data.get('faiss_vectors')}")
    check("dimension is 1024", data.get("faiss_dim") == 1024,
          f"got {data.get('faiss_dim')}")
    print(f"  FAISS: {data.get('faiss_vectors', '?'):,} vectors x "
          f"{data.get('faiss_dim')}d  nprobe={data.get('faiss_nprobe')}  "
          f"DB: {data.get('db_movies', '?'):,} movies")
    print(f"  ⏱ {elapsed:.1f} ms")


def test_recommend_similar():
    print("\n── POST /recommend/similar ──")

    # Valid request
    body = {"title": "Inception", "limit": 5}
    status, data, elapsed = request("POST", "/recommend/similar", body)
    record("POST /recommend/similar (Inception, k=5)", elapsed)
    check("status 200", status == 200, f"got {status}: {data}")
    check("returns list", isinstance(data, list), f"type: {type(data)}")
    check("returns 5 results", len(data) == 5, f"got {len(data)}")

    if isinstance(data, list) and len(data) > 0:
        r = data[0]
        check("has rank", "rank" in r)
        check("has tmdb_id", "tmdb_id" in r)
        check("has title", "title" in r)
        check("has score", "score" in r and r["score"] is not None)
        print(f"  Top result: {r.get('title')} ({r.get('primary_genre')}) "
              f"score={r.get('score')}")
    print(f"  ⏱ {elapsed:.1f} ms")

    # Not-found movie
    body = {"title": "xyznonexistentmovie12345", "limit": 5}
    status, data, _ = request("POST", "/recommend/similar", body)
    check("404 for unknown movie", status == 404, f"got {status}: {data}")

    # nprobe override
    body = {"title": "The Matrix", "limit": 3, "nprobe": 64}
    status, data, elapsed = request("POST", "/recommend/similar", body)
    record("POST /recommend/similar (Matrix, nprobe=64)", elapsed)
    check("nprobe override ok", status == 200 and len(data) == 3,
          f"status={status}, count={len(data) if isinstance(data, list) else '?'}")
    print(f"  ⏱ {elapsed:.1f} ms (nprobe=64)")


def test_recommend_query():
    print("\n── POST /recommend/query ──")
    body = {"query": "sci-fi thriller with mind-bending plot twists", "limit": 5}
    status, data, elapsed = request("POST", "/recommend/query", body)
    record("POST /recommend/query", elapsed)
    if status == 503 and "Ollama" in str(data):
        print(f"  ⚠ Skipped — Ollama not available: {data}")
        return
    check("status 200", status == 200, f"got {status}: {data}")
    check("returns list", isinstance(data, list))
    if isinstance(data, list) and len(data) > 0:
        print(f"  Top result: {data[0].get('title')} ({data[0].get('primary_genre')}) "
              f"score={data[0].get('score')}")
    print(f"  ⏱ {elapsed:.1f} ms")


def test_recommend_profile():
    print("\n── POST /recommend/profile ──")

    # Title-only profile
    body = {"liked": ["Inception", "The Matrix"], "limit": 5}
    status, data, elapsed = request("POST", "/recommend/profile", body)
    record("POST /recommend/profile (2 liked)", elapsed)
    check("status 200", status == 200, f"got {status}: {data}")
    check("returns list", isinstance(data, list), f"type: {type(data)}")
    check("returns results", len(data) > 0, f"got {len(data)}")
    if isinstance(data, list) and len(data) > 0:
        print(f"  Top result: {data[0].get('title')} ({data[0].get('primary_genre')}) "
              f"score={data[0].get('score')}")
    print(f"  ⏱ {elapsed:.1f} ms")

    # With dislike
    body = {
        "liked": ["The Godfather", "Goodfellas"],
        "dislike": ["Twilight"],
        "limit": 5,
    }
    status, data, elapsed = request("POST", "/recommend/profile", body)
    record("POST /recommend/profile (with dislike)", elapsed)
    check("dislike ok", status == 200 and len(data) > 0)
    print(f"  ⏱ {elapsed:.1f} ms")

    # Profile with mood (needs Ollama)
    body = {
        "liked": ["Toy Story", "Finding Nemo"],
        "mood": "something more grown up but still uplifting",
        "mood_weight": 0.4,
        "limit": 5,
    }
    status, data, elapsed = request("POST", "/recommend/profile", body)
    record("POST /recommend/profile (with mood)", elapsed)
    if status == 503 and "Ollama" in str(data):
        print(f"  ⚠ Mood test skipped — Ollama not available")
    else:
        check("mood ok", status == 200)
        print(f"  ⏱ {elapsed:.1f} ms")

    # Not-found liked movie
    body = {"liked": ["xyznonexistent12345"], "limit": 3}
    status, data, _ = request("POST", "/recommend/profile", body)
    check("404 for unknown liked movie", status == 404, f"got {status}: {data}")


def test_movie_lookup():
    print("\n── GET /movies/{tmdb_id} ──")
    status, data, elapsed = request("GET", "/movies/27205")  # Inception
    record("GET /movies/27205", elapsed)
    check("status 200", status == 200, f"got {status}")
    check("has title", "title" in data and data["title"] is not None)
    check("is Inception", data.get("title") == "Inception",
          f"got '{data.get('title')}'")
    print(f"  Movie: {data.get('title')} ({data.get('primary_genre')}, "
          f"{data.get('year')})")
    print(f"  ⏱ {elapsed:.1f} ms")

    # Not-found
    status, data, _ = request("GET", "/movies/-1")
    check("404 for unknown id", status == 404, f"got {status}: {data}")


def test_movie_search():
    print("\n── GET /movies/search ──")
    status, data, elapsed = request("GET", "/movies/search?q=star+wars&limit=5")
    record("GET /movies/search?q=star+wars", elapsed)
    check("status 200", status == 200, f"got {status}")
    check("returns list", isinstance(data, list) and len(data) > 0,
          f"got {len(data) if isinstance(data, list) else type(data)}")
    if isinstance(data, list) and len(data) > 0:
        print(f"  #1: {data[0].get('title')} ({data[0].get('year')})")
    print(f"  ⏱ {elapsed:.1f} ms")


def test_browse():
    print("\n── GET /movies/browse ──")

    # Browse with genre filter
    status, data, elapsed = request(
        "GET", "/movies/browse?genre=Science+Fiction&limit=5&min_votes=1000"
    )
    record("GET /movies/browse (Sci-Fi)", elapsed)
    check("status 200", status == 200)
    check("returns list", isinstance(data, list) and len(data) > 0)
    if isinstance(data, list) and len(data) > 0:
        check("genre matches", data[0].get("primary_genre") == "Science Fiction",
              f"got '{data[0].get('primary_genre')}'")
        print(f"  #1: {data[0].get('title')} ({data[0].get('primary_genre')}, "
              f"{data[0].get('year')})")
    print(f"  ⏱ {elapsed:.1f} ms")

    # Browse with year range
    status, data, elapsed = request(
        "GET", "/movies/browse?year_min=2010&year_max=2020&limit=5&min_rating=7.0"
    )
    record("GET /movies/browse (2010-2020)", elapsed)
    check("year range ok", status == 200 and len(data) > 0)
    print(f"  ⏱ {elapsed:.1f} ms")


def test_genres():
    print("\n── GET /movies/genres ──")
    status, data, elapsed = request("GET", "/movies/genres")
    record("GET /movies/genres", elapsed)
    check("status 200", status == 200)
    check("returns list", isinstance(data, list) and len(data) > 5,
          f"got {len(data) if isinstance(data, list) else type(data)}")
    if isinstance(data, list):
        print(f"  Genres: {', '.join(data[:10])}... ({len(data)} total)")
    print(f"  ⏱ {elapsed:.1f} ms")


def test_latency_warm():
    """Repeated query to measure warm-cache latency."""
    print("\n── Warm Cache Latency (5x /recommend/similar Inception k=12) ──")
    times: list[float] = []
    body = {"title": "Inception", "limit": 12}
    for i in range(5):
        status, data, elapsed = request("POST", "/recommend/similar", body)
        times.append(elapsed)
        if status != 200:
            print(f"  ✗ Request {i+1} failed: {status}")
            return
    avg = sum(times) / len(times)
    print(f"  Times: {[f'{t:.1f}' for t in times]} ms")
    print(f"  Average: {avg:.1f} ms")
    record("POST /recommend/similar (warm avg)", avg)


# ═══════════════════════════════════════════════════════════════════
#  Phase 2 Tests
# ═══════════════════════════════════════════════════════════════════

def test_phase2_diverse():
    """Test the /recommend/diverse endpoint for genre spread."""
    print("\n── POST /recommend/diverse ──")
    body = {"title": "Inception", "limit": 12, "lambda_qd": 0.5}
    status, data, elapsed = request("POST", "/recommend/diverse", body)
    record("POST /recommend/diverse", elapsed)
    check("status 200", status == 200, f"got {status}: {data}")
    check("returns list", isinstance(data, list), f"type: {type(data)}")
    if isinstance(data, list) and len(data) > 0:
        check("has results", len(data) >= 3, f"got {len(data)}")
        genres = [r.get("primary_genre", "") for r in data]
        unique_genres = len(set(genres))
        print(f"  Unique genres: {unique_genres} / {len(data)} results")
        print(f"  Genres: {', '.join(genres[:8])}")
        check("genre diversity >= 5 unique", unique_genres >= 5,
              f"only {unique_genres} unique genres (target >= 5)")
    print(f"  ⏱ {elapsed:.1f} ms")

    # Test with aggressive diversity
    body = {"title": "The Matrix", "limit": 8, "lambda_qd": 0.2}
    status, data, elapsed = request("POST", "/recommend/diverse", body)
    record("POST /recommend/diverse (aggressive)", elapsed)
    check("aggressive diversity ok", status == 200)
    print(f"  ⏱ {elapsed:.1f} ms (lambda_qd=0.2)")


def test_phase2_explore():
    """Test the /recommend/explore endpoint."""
    print("\n── POST /recommend/explore ──")
    body = {"liked": ["Inception", "The Matrix"], "limit": 10}
    status, data, elapsed = request("POST", "/recommend/explore", body)
    record("POST /recommend/explore", elapsed)
    check("status 200", status == 200, f"got {status}: {data}")
    check("returns list", isinstance(data, list), f"type: {type(data)}")
    if isinstance(data, list) and len(data) > 0:
        print(f"  Results: {len(data)} movies")
        for r in data[:3]:
            print(f"    {r.get('title')} ({r.get('primary_genre')}) score={r.get('score')}")
    print(f"  ⏱ {elapsed:.1f} ms")


def test_phase2_profile_diversity():
    """Test profile endpoint with Phase 2 for genre diversity."""
    print("\n── POST /recommend/profile (Phase 2 diversity) ──")
    body = {
        "liked": ["The Godfather", "Pulp Fiction", "The Dark Knight"],
        "limit": 12,
        "use_phase2": True,
    }
    status, data, elapsed = request("POST", "/recommend/profile", body)
    record("POST /recommend/profile (Phase 2)", elapsed)
    check("status 200", status == 200, f"got {status}")
    if isinstance(data, list) and len(data) > 0:
        genres = [r.get("primary_genre", "") for r in data]
        unique_genres = len(set(genres))
        print(f"  Unique genres: {unique_genres} / {len(data)} results")
        print(f"  Genres: {', '.join(genres[:8])}")
        check("genre diversity >= 5 unique", unique_genres >= 5,
              f"only {unique_genres} unique genres (target >= 5)")
    print(f"  ⏱ {elapsed:.1f} ms")

    # Test Phase 1 fallback (use_phase2=False)
    body["use_phase2"] = False
    status, data, elapsed = request("POST", "/recommend/profile", body)
    record("POST /recommend/profile (Phase 1 fallback)", elapsed)
    check("phase1 fallback ok", status == 200 and len(data) > 0)
    print(f"  ⏱ {elapsed:.1f} ms (Phase 1 fallback)")


def test_ranker_features():
    """Test the /ranker/features endpoint."""
    print("\n── GET /ranker/features ──")
    status, data, elapsed = request("GET", "/ranker/features")
    record("GET /ranker/features", elapsed)
    if status == 404:
        print(f"  ⚠ Skipped — LightGBM model not trained yet: {data}")
        return
    check("status 200", status == 200, f"got {status}")
    if isinstance(data, dict):
        fi = data.get("feature_importance", {})
        if fi:
            print(f"  Top features: {list(fi.keys())[:5]}")
    print(f"  ⏱ {elapsed:.1f} ms")


def test_phase2_latency():
    """Measure Phase 2 pipeline latency."""
    print("\n── Phase 2 Pipeline Latency (3x /recommend/similar Inception k=12) ──")
    times: list[float] = []
    body = {"title": "Inception", "limit": 12, "use_phase2": True}
    for i in range(3):
        status, data, elapsed = request("POST", "/recommend/similar", body)
        times.append(elapsed)
        if status != 200:
            print(f"  ✗ Request {i+1} failed: {status}")
            return
    avg = sum(times) / len(times)
    print(f"  Times: {[f'{t:.1f}' for t in times]} ms")
    print(f"  Average: {avg:.1f} ms")
    record("POST Phase 2 pipeline (warm avg)", avg)
    check("under 100ms", avg < 100, f"{avg:.1f} ms — exceeds 100ms budget")


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    global PASSED, FAILED

    print("=" * 60)
    print("  TREKOMEND API INTEGRATION TEST")
    print(f"  Target: {BASE}")
    print("=" * 60)

    # Quick connectivity check
    try:
        urllib.request.urlopen(f"{BASE}/health", timeout=5)
    except Exception as e:
        print(f"\n✗ Cannot reach API at {BASE}")
        print(f"  Error: {e}")
        print(f"\n  Start the server first:")
        print(f"    uv run uvicorn main:app --host 127.0.0.1 --port 8080")
        sys.exit(1)

    test_health()
    test_stats()
    test_recommend_similar()
    test_recommend_query()
    test_recommend_profile()
    test_movie_lookup()
    test_movie_search()
    test_browse()
    test_genres()
    test_latency_warm()

    # Phase 2 tests
    print("\n" + "─" * 60)
    print("  PHASE 2 TESTS")
    print("─" * 60)
    test_phase2_diverse()
    test_phase2_explore()
    test_phase2_profile_diversity()
    test_ranker_features()
    test_phase2_latency()

    # Summary
    print("\n" + "=" * 60)
    total = PASSED + FAILED
    print(f"  Results: {PASSED}/{total} passed"
          + (f", {FAILED} FAILED" if FAILED else ""))
    print(f"\n  Latency Summary:")
    for name, ms in sorted(TIMINGS, key=lambda x: x[1]):
        print(f"    {name:<55s} {ms:6.1f} ms")
    avg_lat = sum(ms for _, ms in TIMINGS) / len(TIMINGS) if TIMINGS else 0
    print(f"\n  Average latency: {avg_lat:.1f} ms")
    print("=" * 60)

    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
