# trekomend

1.4 million TMDB movies, each turned into a 1024-dimensional vector.
Find similar movies, search by description, or build a taste profile
from what you already like. All of it runs locally.

## How it works

Every movie gets embedded by Qwen3-Embedding-0.6B. The text fed into
the model covers 12 fields picked for content-based similarity: plot
overview, keywords, genres, tagline, title, year, language, country,
studio, runtime. Budget, revenue, ratings, and TMDB IDs get dropped.
Those are collaborative signals. They made the embeddings worse when
we included them, so we stopped.

The Kaggle notebook finishes a full embedding pass in about 3 hours on
free dual T4 GPUs. You get one zip file at the end. No multi-session
juggling, no shard merging.

After that, searching runs locally against the stored vectors. If you
want text-based queries you need Ollama running with the Qwen3 model.
Title lookups work without it.

## Quick start

```bash
# Install dependencies
uv sync

# Drop the HDF5 file into embeddings/
cp tmdb_qwen06b_1024d.h5 embeddings/

# Optional: put the TMDB CSV in dataset/ for metadata
```

If you do not have the embeddings yet, run the Kaggle notebook.
Instructions are in `kaggle-kernel-trekomend/README.md`.

For text queries, install [Ollama](https://ollama.com) and pull the model:

```bash
ollama pull qwen3-embedding:0.6b
```

## Commands

```bash
uv run python main.py                              # stats + validate
uv run python main.py --validate                   # norm/NaN check
uv run python main.py --visualize                  # PCA + heatmaps + genre classification
```

### Title lookup

```bash
uv run python main.py --similar "Inception"          # movies like Inception
uv run python main.py --similar Inception Interstellar --blend
uv run python main.py --similar "Die Hard" "Toy Story" --max
```

### Text query (needs Ollama)

```bash
uv run python main.py --query "mind-bending sci-fi thriller with plot twists"
uv run python main.py --query "uplifting" "dark" --max
uv run python main.py --query "space" "ocean" "war" --rrf
```

### Combined: titles plus text (the good stuff)

This mixes stored movie vectors with live Ollama embeddings in one
search. You anchor on movies you know and steer with natural language.

```bash
uv run python main.py --search "Inception" --also "but more philosophical and dreamlike"
uv run python main.py --search "The Godfather" --also "modern crime" --max
```

### Preference profile (your taste, modeled)

Builds a personal taste vector from movies you like, optionally
blends in a mood description, and pushes away from stuff you do
not want. This is the closest thing to a "recommendation engine"
in one command.

```bash
uv run python main.py --profile "Inception" "The Matrix" "Interstellar"
uv run python main.py --profile "Toy Story" --mood "something more grown up"
uv run python main.py --profile "The Godfather" "Goodfellas" \
                        --dislike "Twilight" --mood "modern crime thriller"
uv run python main.py --profile "Parasite" --mood "lighter, funnier" --mood-weight 0.5
```

## Multi-query strategies

When you pass more than one `--similar`, `--query`, or `--search` item,
pick how they combine.

| Strategy | Does | When |
|---|---|---|
| `--blend` (default) | Averages all vectors, searches once | Items are similar: "Inception" + "Interstellar" |
| `--max` | Each candidate gets its best score from any query | Mixing genres: "Die Hard" + "Toy Story" |
| `--rrf` | Independent searches merged by Reciprocal Rank Fusion | You want every query to pull equal weight |

## What the numbers mean

**Vector norm should be about 1.0.** Qwen3 normalizes every output.
If it drifts past 1.01 or below 0.99, something went wrong during
embedding. Probably a half-finished shard.

**PCA-50 variance above 0.3 is good.** Below that, the vectors do
not carry much structure. Either the source text is too uniform or
the data is noise.

**Classification accuracy should be at least 2x the baseline.**
We run logistic regression on the raw vectors to guess the primary
genre. If it barely beats the most-common-genre baseline, the
embeddings are not encoding anything useful. On the 1.4M movie set
we see 89.7% accuracy against a 44.1% baseline. That is 2.0x.

**Search results should make sense.** Inception returning Memento
is a good sign. Inception returning a documentary called "Useless"
means two rows had identical empty-field fallback text and the
deduplication missed it.

**The 10-movie comparison** in `--visualize` picks ten random
movies, prints their pairwise similarity table, and flags pairs
above 0.25 cosine. Same-genre matches get marked. Horror should
cluster with horror. If it does not, the embeddings are broken.

## Files

```
trekomend/
  main.py                         CLI entrypoint
  src/
    __init__.py
    config.py                     Paths, constants, instruction templates, RNG seed
    io.py                         HDF5 reading, CSV metadata, Ollama embedding
    search.py                     All search strategies (title, query, combined, profile)
    analyze.py                    Validate, stats, PCA, heatmaps, genre classification
  kaggle-kernel-trekomend/        Kaggle notebook + metadata for GPU embedding
    kernel-metadata.json
    trekomend_v2_1024d.ipynb
    README.md
  embeddings/                     Drop the HDF5 here (git-ignored)
  dataset/                        TMDB CSV here (git-ignored)
```

## Setup from scratch

1. Clone the repo
2. `uv sync`
3. Run the Kaggle notebook to generate embeddings, or download a pre-built HDF5
4. Drop `tmdb_qwen06b_1024d.h5` into `embeddings/`
5. Put `TMDB_movie_dataset_v11.csv` in `dataset/`
6. For text queries: install [Ollama](https://ollama.com), then `ollama pull qwen3-embedding:0.6b`
7. `uv run python main.py`

## How the preference profile works

The `--profile` command runs a lightweight version of a content-based
recommender. The algorithm:

1. Takes your liked movies and averages their stored vectors into a
   taste centroid.
2. If you gave `--dislike` titles, it projects your taste vector away
   from those movies in embedding space so they stop showing up.
3. If you gave `--mood`, it embeds that text via Ollama using a
   hybrid instruction that biases toward bridging established taste
   with immediate context. The mood vector gets blended in at the
   weight you set (default 0.3, meaning 70% taste, 30% mood).
4. L2-normalizes the result and searches all 1.4 million movies by
   cosine similarity.
5. Filters out the movies you already said you like so you only see
   new stuff.

The mood weight parameter is the tuning knob. At 0.0 you get pure
"more like what I already like." At 1.0 you get pure "whatever I
feel like right now." The default 0.3 leans toward taste but lets
the mood nudge results in a direction.

## Data

The embeddings come from the [TMDB movie dataset v11](https://huggingface.co/datasets/fukitweball/TMDB)
(also available on [Kaggle](https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies)).
The CSV has 24 columns. Only 12 go into the embedding text. The rest
are either collaborative signals (budget, revenue, ratings) or noise
(poster paths, IMDB IDs, homepage URLs).

The model is [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B).
It fits in about 1.2 GB of VRAM on a T4. Output is 1024-dim with
Matryoshka support so you can truncate to any dimension from 32 to
1024. The model uses asymmetric prompts: instructions on the query
side, raw text on the document side. We use four different
instruction templates depending on whether the query is general,
mood-biased, genre-biased, or a hybrid of taste and context.

## Going faster

- Kaggle gives you two free T4 GPUs. The notebook finishes 1.4M
  movies in about 3 hours. One session. One download.
- A local RTX 3060 finishes in roughly 2 hours if you lower the
  batch size a bit.
- Colab's single T4 takes longer but still works overnight.
- If you have the VRAM, the 4B model gives better embeddings
  (MTEB ~75 versus ~62 for 0.6B) but runs 3-4x slower and needs
  more GPU memory.
