# trekomend

Content-based movie recommendations for 1.4 million TMDB films. Each movie
gets turned into a 1024-dimensional vector by Qwen3-Embedding-0.6B. Search by
title, describe what you want, or build a taste profile from movies you already
like. The API runs on a $6/month VPS.

A LightGBM re-ranker, genre round-robin diversification, and an opt-in DPP
selector for set diversity run on top of the raw FAISS retrieval. The default
pipeline adds about 2 ms of overhead and delivers 6-9 unique genres per query.
Full system research: [trekomend-system-research.md](research/trekomend-system-research.md).

## API

Start the server on a VPS:

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8080 --workers 1
```

Or directly:

```bash
uv run python main.py
```

The API needs three files in `new-kaggle-output/` (hosted on HuggingFace,
not in this repo):

```
new-kaggle-output/
    tmdb_movies.db               SQLite metadata (912 MB)
    tmdb_qwen06b_1024d.faiss     FAISS IVF-PQ index (108 MB)
    tmdb_qwen06b_1024d.h5        HDF5 embeddings (4.8 GB)
```

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `GET` | `/stats` | Index and database counts |
| `POST` | `/recommend/similar` | Movies like a given title |
| `POST` | `/recommend/query` | Movies matching a text description |
| `POST` | `/recommend/profile` | Profile from liked films, mood, and dislikes |
| `POST` | `/recommend/diverse` | Maximum diversity mode (DPP) |
| `POST` | `/recommend/explore` | Serendipity and novelty |
| `GET` | `/movies/{tmdb_id}` | Single movie by TMDB ID |
| `GET` | `/movies/search?q=...` | Full-text search (FTS5) |
| `GET` | `/movies/browse?genre=...` | Browse with filters |
| `GET` | `/movies/genres` | All primary genres |
| `GET` | `/ranker/features` | LightGBM feature importance |

Full integration tests in `test_api.py` (48/48 pass, ~16ms average warm latency).

### Example requests

```bash
# Movies similar to Inception (default Phase 2 pipeline)
curl -X POST http://localhost:8080/recommend/similar \
  -H "Content-Type: application/json" \
  -d '{"title": "Inception", "limit": 12}'

# Text query (needs Ollama)
curl -X POST http://localhost:8080/recommend/query \
  -H "Content-Type: application/json" \
  -d '{"query": "sci-fi thriller with mind-bending plot twists"}'

# Profile from liked movies with a mood
curl -X POST http://localhost:8080/recommend/profile \
  -H "Content-Type: application/json" \
  -d '{"liked": ["Inception", "The Matrix", "Interstellar"],
       "mood": "something more philosophical",
       "mood_weight": 0.4, "limit": 12}'

# Maximum diversity (DPP mode)
curl -X POST http://localhost:8080/recommend/diverse \
  -H "Content-Type: application/json" \
  -d '{"title": "Inception", "limit": 12, "lambda_qd": 0.4}'
```

## How it works

The embedding text for each movie combines 12 fields: plot overview, keywords,
genres, tagline, title, release year, original language, production country,
studios, runtime, and vote average rounded to a quality tier. Budget, revenue,
vote counts, and external IDs get dropped. Those are collaborative signals.
Including them made the embeddings worse so we stopped.

The Kaggle notebook finishes a full embedding pass in about 3 hours on free
dual T4 GPUs. You get one zip at the end. No session juggling, no manual
shard merging.

After that, searching runs locally against the stored vectors. Title lookups
work by themselves. Text queries need Ollama running with the Qwen3-Embedding
model. The API server loads a FAISS IVF-PQ index instead of keeping all
embeddings in RAM, which keeps memory under 200 MB on a VPS.

## Post-retrieval pipeline

Three layers run after FAISS retrieves the top 200 candidates:

```
FAISS top-200 -> LightGBM -> Genre Round-Robin -> [DPP] -> Top-12
    15ms           +1ms            +0.3ms          [+3ms]
```

**Layer 1: LightGBM re-ranker.** A gradient boosted tree model with 15
features per query-candidate pair: cosine similarity, genre Jaccard, year
difference, popularity percentile, keyword overlap, rating comparison, and
embedding norm interaction features. Trained on CPU with 60K pseudo-labeled
pairs (240 query movies x 250 candidates). RMSE 0.0031, R-squared 0.999.

The top features by gain: cosine_sim (1335), vote_average (824), genre_jaccard
(812), year_diff_abs (733), popularity_percentile (500).

LightGBM runs before round-robin so its improved relevance scores guide the
within-genre selection. The earlier pipeline ran round-robin first and the
cosine-dominated scores collapsed genre diversity back to 2-4 genres.

**Layer 2: Genre round-robin.** Groups candidates by primary genre and
interleaves them with per-genre caps (max 3 from the same genre in top 12).
Adds a serendipity slot at position 8 for a movie from a genre not yet seen.
Zero training, near-zero overhead. This alone boosts most queries from
2-4 unique genres to 6-9.

**Layer 3: DPP set selection** (opt-in, `/recommend/diverse` only). A low-rank
Determinantal Point Process picks a mathematically diverse subset from the top
50 candidates. Uses a quality-diversity kernel with greedy MAP inference and
Sherman-Morrison updates for O(K^2) per pick. Lambda-qd controls the tradeoff
(lower = more diverse). Default is 0.7. Adds about 3ms warm, 30ms cold (HDF5
reads for candidate embeddings).

### Training the ranker

```bash
uv run python -m src.train_ranker --queries 250 --output models/ranker_v1.txt
```

The training pipeline picks 250 stratified query movies, retrieves FAISS
candidates, adds random negatives, computes pseudo-labels (weighted blend
of cosine, rating, genre match, keyword overlap, and year proximity), builds
feature matrices, and trains a LightGBM regressor with early stopping. Training
takes under a second on CPU and produces a 585 KB model file.

## Setup

1. Clone the repo
2. `uv sync`
3. Download the three large files from HuggingFace and put them in `new-kaggle-output/`
4. For text queries: install [Ollama](https://ollama.com) and pull `qwen3-embedding:0.6b`
5. Start the server: `uv run uvicorn main:app --host 0.0.0.0 --port 8080`

To run the Kaggle notebook yourself, see `kaggle-kernel-trekomend/`. It covers
CSV ingestion, embedding, FAISS index building, and SQLite export.

## Files

```
trekomend/
    main.py                     FastAPI server (single entrypoint)
    test_api.py                 48 integration tests
    pyproject.toml
    src/
        __init__.py
        config.py                Paths, constants, instruction templates
        io.py                    Ollama embedding functions
        faiss_search.py          FAISS IVF-PQ searcher with SQLite + HDF5 cache
        diversity.py             GenreRoundRobin + DPPSelector
        features.py              FeatureBuilder for LightGBM
        train_ranker.py          LightGBM training pipeline
    models/
        ranker_v1.txt            Trained LightGBM model (585 KB)
        ranker_v1.importance.json
    kaggle-kernel-trekomend/     GPU embedding notebooks
    research/                    Architecture docs and lit review
    new-kaggle-output/           Large binary files (HuggingFace, not in git)
```

## How the profile endpoint works

The `/recommend/profile` endpoint builds a taste vector from movies you like,
optionally blends in a mood description, and pushes away from things you
dislike:

1. Averages your liked movie vectors into a taste centroid
2. If you provide `dislike`, projects the taste vector away from those movies
3. If you provide `mood`, embeds the text via Ollama and blends it in at the
   weight you specify (`mood_weight`, default 0.3)
4. Normalizes and searches 1.4 million movies via FAISS
5. Filters out movies already in your `liked` list
6. Runs the post-retrieval pipeline (LightGBM re-rank and genre round-robin)

## Data

The embeddings come from the [TMDB movie dataset v11](https://huggingface.co/datasets/fukitweball/TMDB)
(also on [Kaggle](https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies)).
The model is [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B).
It uses asymmetric prompts: instructions on the query side, raw text on the
document side. We use four instruction templates depending on whether the
query is general, mood-biased, genre-biased, or a hybrid.

## References

- Gartrell, Paquet, Koenigstein (KDD 2016). DPP for Recommendation
- Ke et al. (NeurIPS 2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree
- Covington et al. (RecSys 2016). Deep Neural Networks for YouTube Recommendations
- Meehan & Pauwels (RecSys 2025). Popularity Bias in Cold-Start
- Li et al. (KDD 2024). Contextual Distillation for Diversified Recommendation
- Ibrahim et al. (RecSoGood 2025). Personalized DPP for Diversified Recommendation
