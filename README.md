# recommend

Embed 1.4 million TMDB movies with Qwen3-0.6B on a free Colab T4, then search, cluster, and classify on your own machine.

## How it works

Two halves, two machines.

**Colab** — `moviedata_optimized.ipynb` downloads the TMDB CSV, loads Qwen3, and processes movies in 20K-row chunks. Each chunk saves as a `.npy` shard. Free tier gives about 5 hours of GPU per day. Download the zip before the session ends.

**Local** — drop `.npy` and `.rows.csv` files into `embeddings/`, then run `main.py`.

## Commands

```bash
uv sync
```

```bash
uv run python main.py                          # stats + validate (default)
uv run python main.py --validate               # norm/NaN check per shard
uv run python main.py --visualize              # PCA plots + classification
uv run python main.py --similar "Inception"    # cross-shard search
uv run python main.py --shard 3                # inspect one shard
uv run python main.py --validate --similar "The Matrix"
```

| Flag | What it does |
|------|-------------|
| `--stats` | Progress bar, shard list, disk usage |
| `--validate` | Norm and NaN check for every `.npy` file |
| `--visualize` | Per-shard PCA scatter, similarity heatmap, genre classification |
| `--similar TITLE` | Search all 140K+ movies for the nearest neighbors to TITLE |
| `--shard N` | Full analysis on a single shard |
| `--emb-dir PATH` | Path to embeddings folder (default `embeddings/`) |
| `--dataset PATH` | Path to TMDB CSV (default `dataset/TMDB_movie_dataset_v11.csv`) |

No flags = `--stats` + `--validate`.

## Files

| File | Purpose |
|------|---------|
| `main.py` | Local CLI — validate, visualize, search |
| `moviedata_optimized.ipynb` | Colab notebook — embed movies on T4 GPU |
| `embeddings/` | Drop `.npy` shards here |
| `dataset/` | TMDB CSV (git-ignored, download from HuggingFace) |

## What the numbers mean

**Vector norm = 1.0.** The model normalizes every output. Drift above 1.01 or below 0.99 means something broke.

**PCA-50 variance > 0.3.** Real structure in the data. Below 0.3 usually means noise or corrupted vectors. Typical is 0.53-0.55.

**Classification > 2x baseline.** Logistic regression on raw vectors guesses genre better than random. If it drops near 1x, the embeddings aren't encoding semantic meaning.

**Search results should be thematically similar.** Action movies should pull up action movies. If Inception returns documentaries, something is wrong with the embeddings (or the dataset has duplicate rows producing identical vectors — check for empty fields).

## Data

[TMDB movie dataset v11](https://huggingface.co/datasets/fukitweball/TMDB) — 1.4M movies, 632 MB CSV.

[Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) — ~1.2 GB VRAM on T4, 768-dim normalized output.
