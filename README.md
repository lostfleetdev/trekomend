# trekomend

Embed all 1.4 million TMDB movies with Qwen3‑0.6B on a free Colab T4. Search them from your terminal.

## How it works

**Colab** — `moviedata_v2.ipynb` downloads the TMDB CSV, loads Qwen3, and processes movies in 20K‑row chunks. Each chunk lands as a single `.h5` file. The free tier lasts five hours or so. Download the zip before the runtime dies.

**Local** — drop the `.h5` files into `embeddings/`, the CSV into `dataset/`, then run the CLI.

```bash
uv sync
```

## Commands

```bash
uv run python main.py                              # stats + validate (default)
uv run python main.py --validate                   # norm/NaN check per shard
uv run python main.py --visualize                  # PCA + heatmaps + 10‑movie comparison
uv run python main.py --visualize 3                # just shard 3
uv run python main.py --similar "Inception"        # nearest neighbors by stored vector
uv run python main.py --similar Inception Interstellar --blend
uv run python main.py --similar "Die Hard" "Toy Story" --max
uv run python main.py --query "sci-fi mind-bender" # search via Ollama
uv run python main.py --query "scary" "funny" --max
```

| Flag | Does |
|------|------|
| `--stats` | Progress bar, shard count, disk usage |
| `--validate` | Norm and NaN check for every `.h5` file |
| `--visualize` | PCA scatter, heatmap, 10‑movie pair comparison, genre classification per shard |
| `--visualize N` | Same, one shard only |
| `--similar TITLE` | Looks up a stored movie by title, returns its nearest neighbors |
| `--similar A B C --blend` | Averages A/B/C vectors into one, searches once |
| `--similar A B C --max` | Each candidate keeps its highest similarity to any query |
| `--similar A B C --rrf` | Three independent searches, merged by Reciprocal Rank Fusion |
| `--query TEXT` | Embeds your text through Ollama, searches all shards |
| `--query A B --rrf` | Multiple Ollama queries, RRF‑merged |
| `--emb-dir PATH` | Path to embeddings folder (default `embeddings/`) |
| `--dataset PATH` | Path to TMDB CSV (default `dataset/TMDB_movie_dataset_v11.csv`) |
| `--ollama-model NAME` | Which Ollama model for `--query` (default `qwen3:0.6b`) |

## Multi‑query strategies

When you pass more than one `--similar` or `--query`, pick how they combine:

| Strategy | How | When |
|----------|-----|------|
| `--blend` (default) | Average vectors, search once | All picks are similar — "Inception" + "Interstellar" |
| `--max` | Score by best match to any query | Mixing genres — "Die Hard" + "Toy Story" |
| `--rrf` | Independent searches, rank fusion | You want every query to pull equal weight |

## Files

```
trekomend/
  main.py                 CLI entrypoint
  moviedata_v2.ipynb      Colab notebook (the embedding half)
  src/
    __init__.py
    config.py             Paths, constants, RNG seed
    io.py                 HDF5 reading, CSV metadata, Ollama embedding
    search.py             Search strategies (single, blend, max, rrf)
    analyze.py            Validate, stats, PCA/heatmap/classification
  embeddings/             ← drop .h5 shards here
  dataset/                ← TMDB CSV here (git‑ignored)
```

## What the numbers mean

**Vector norm = 1.0.** Qwen3 normalizes every output. If it drifts past 0.01 in either direction, something went wrong during embedding.

**PCA‑50 variance > 0.3.** Values below that usually mean the vectors have no real semantic structure — either corrupt data or text that's too uniform to matter.

**Classification > 2× baseline.** If logistic regression on raw vectors guesses genre better than random chance, the embeddings carry meaning. Scores near baseline mean they don't.

**Search results should be coherent.** Inception returning Memento is a good sign. Inception returning a documentary called "Useless" means two CSV rows had the same empty‑field fallback text — the v2 notebook fixed this.

**10‑movie comparison.** `--visualize` picks ten random movies, prints the full pairwise similarity table, and flags pairs above 0.25. Same‑genre matches get a `**`. Good for spot‑checking new shards.

## Setup

1. Clone the repo
2. `uv sync`
3. Drop `.h5` shards into `embeddings/`
4. Put `TMDB_movie_dataset_v11.csv` in `dataset/`
5. For `--query`: install [Ollama](https://ollama.com), then `ollama pull qwen3:0.6b`
6. `uv run python main.py`

## Data

[TMDB movie dataset v11](https://huggingface.co/datasets/fukitweball/TMDB) — 1.4 million movies, 632 MB CSV.

[Qwen3‑Embedding‑0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) — 1.2 GB VRAM on a T4. 768‑dim output. Matryoshka support. Asymmetric prompts.
