# Trekomend v2 — Kaggle Kernel

1024-dimensional movie embeddings for ~1M TMDB movies using
**Qwen3-Embedding-0.6B** on Kaggle's **free dual T4 GPU** setup.

---

## What This Is

A self-contained Kaggle notebook that embeds every movie in the
[asaniczka/tmdb-movies-dataset-2023-930k-movies](https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies)
dataset into 1024-dimensional vectors optimized for content-based
recommendation. One session, one zip file at the end.

## Research-Backed Design

Every decision in this notebook is grounded in research (see RESEARCH_FINDINGS.md
in the parent project for full citations):

| Decision | Why | Source |
|---|---|---|
| 1024-dim (not 768 MRL-truncated) | +2-3% retrieval quality | MRL for RecSys, arXiv 2406.07432 |
| No IDs/financials in text template | Pure noise for content similarity | Embedding recsys surveys, arXiv 2310.18608 |
| Layer-priority text template | Genres/keywords before context | Ablation studies, Milvus Qwen3 guide |
| Manual dual-GPU (not DataParallel) | Higher throughput, no per-forward overhead | PyTorch forums, Accelerate docs |
| flash_attention_2 priority | +20-25% speed on T4 | Qwen3 docs, FA2 benchmarks |
| torch.compile (reduce-overhead) | +4-40% for repeated forwards | PyTorch 2.x docs |
| Length-sorted batching | Less padding waste | SBERT efficiency guide |
| LZF→gzip shard compression | Fast write, portable output | HDF5 docs |
| SQLite checkpointing | Crash recovery, resumable | Kaggle 12h session limit |
| Shards in /kaggle/tmp | Doesn't count against output limit | Kaggle disk docs |

## Files

```
kaggle-kernel-trekomend/
  kernel-metadata.json         Kaggle kernel metadata (dataset ref, GPU, internet)
  trekomend_v2_1024d.ipynb     The notebook (14 cells, self-contained)
  README.md                    This file
```

## How to Run on Kaggle

### Option A: Push via Kaggle API

```bash
# Install Kaggle CLI
pip install kaggle

# Set up API key (~/.kaggle/kaggle.json)
# Get your key: https://www.kaggle.com/settings/account → API → Create New Token

# Push the kernel
cd kaggle-kernel-trekomend
kaggle kernels push
```

Then open on Kaggle, go to Settings → Accelerator → **GPU T4 x2**, and Run All.

### Option B: Manual Upload

1. Go to [Kaggle Notebooks](https://www.kaggle.com/code) → New Notebook
2. File → Upload Notebook → select `trekomend_v2_1024d.ipynb`
3. Settings → Accelerator → **GPU T4 x2**
4. Add Data → search for `asaniczka/tmdb-movies-dataset-2023-930k-movies`
5. Runtime → Run All

### Option C: Import from GitHub

If you push this to a GitHub repo, you can:
1. Kaggle → New Notebook → File → Import Notebook → GitHub
2. Paste the raw URL of `trekomend_v2_1024d.ipynb`
3. Settings → GPU T4 x2
4. Add the asaniczka dataset

## Notebook Cell Breakdown

| Cell | What It Does | Time |
|---|---|---|
| 1 | Install dependencies + flash-attn | ~30s |
| 2 | Configuration & imports | ~1s |
| 3 | Logger & GPU monitor setup | ~1s |
| 4 | Dataset discovery (Kaggle→HF fallback) | ~10s |
| 5 | Movie text builder (research template) | ~1s |
| 6 | SQLite checkpoint DB | ~1s |
| 7 | Load model ×2 (dual GPU + FA2 + compile) | ~30s |
| 8 | Embedding engine (concurrent GPU forward) | ~1s |
| 9 | Speed benchmark | ~10s |
| 10 | **Main processing loop** ← bulk of time | 25-50 min |
| 11 | Merge shards → single HDF5 + manifest | 2-5 min |
| 12 | Validation (NaN/norm/pairwise) | 1-3 min |
| 13 | Create downloadable zip | 1-3 min |
| 14 | Semantic smoke test (optional) | 1-2 min |

## Expected Output

| File | Size (~1M movies) | Description |
|---|---|---|
| `tmdb_qwen06b_1024d.h5` | ~3.8 GB | Merged embeddings (gzip lvl 2) |
| `trekomend_v2_1024d.zip` | ~3.8 GB | Downloadable (HDF5 + manifest + log) |
| `manifest.json` | ~2 KB | Full run metadata for reproducibility |
| `run.log` | ~100 KB | Detailed session log |

## Speed Tuning

| Parameter | Default | If Too Slow | If OOM |
|---|---|---|---|
| `TOTAL_BATCH_SIZE` | 448 | Increase to 512 | Decrease to 256 |
| `MAX_LEN` | 512 | Decrease to 256 | Keep 512 |
| `OUT_DIM` | 1024 | Decrease to 768 | Keep 1024 |
| `SHARD_SIZE` | 10,000 | Keep | Decrease to 5,000 |
| `CPU_WORKERS` | 4 | Increase to 6 | Decrease to 2 |

## Using the Output Locally

After downloading the zip, use the parent project's CLI:

```bash
# Copy the HDF5 into embeddings/
cp tmdb_qwen06b_1024d.h5 /path/to/trekomend/embeddings/

# Search by title
cd /path/to/trekomend
uv run python main.py --similar "Inception"

# Search by text description (requires Ollama)
uv run python main.py --query "mind-bending sci-fi thriller"
```

## Dataset Columns Used

Only 12 of 24 columns go into the embedding text:

| Column | Used | Reason |
|---|---|---|
| title | Yes | Identity |
| original_title | Yes | Alternative name |
| release_date | Yes → year | Era context |
| runtime | Yes | Length context |
| original_language | Yes | Language context |
| overview | Yes ★ | Highest signal |
| tagline | Yes | Supplementary |
| genres | Yes ★ | High signal |
| keywords | Yes ★ | Very high signal |
| production_companies | Yes | Studio context |
| production_countries | Yes | Country context |
| id | For tracking only | Not in text |
| vote_average | No | Collaborative signal |
| vote_count | No | Collaborative signal |
| status | No | Noise |
| revenue | No | Collaborative signal |
| adult | No | Noise |
| backdrop_path | No | Noise |
| budget | No | Collaborative signal |
| homepage | No | Noise |
| imdb_id | No | Noise |
| popularity | No | Collaborative signal |
| poster_path | No | Noise |
| spoken_languages | No | Redundant with original_language |

## License

Same as the parent trekomend project. Dataset: ODC Attribution License (ODC-By).
