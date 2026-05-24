"""
main.py — CLI for TMDB movie embeddings.

╔══════════════════════════════════════════════════════════════════════════════╗
║                         QUICK COMMANDS                                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  uv run python main.py                                    stats + validate ║
║  uv run python main.py --validate                        health check only ║
║  uv run python main.py --visualize                       PCA + heatmaps    ║
║  uv run python main.py --visualize 3                     inspect shard #3 ║
║                                                                             ║
║  uv run python main.py --similar "Inception"             look up by title  ║
║  uv run python main.py --similar Inception Interstellar --blend            ║
║  uv run python main.py --query "mind-bending sci-fi"     Ollama search     ║
║  uv run python main.py --query "scary" "funny" --max                       ║
║  uv run python main.py --query "space" "ocean" "war" --rrf                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

SEARCH STRATEGIES (for multiple --similar or --query items)
  --blend    Average all query vectors into one, search once.
             Best when items are similar (all sci-fi, all horror).
             Fast — only one search pass.

  --max      Each candidate scored by its highest similarity to any query.
             Best when items are different genres (action + comedy).
             No "averaged mush" — keeps extremes.

  --rrf      Run independent searches, merge via Reciprocal Rank Fusion.
             Best when you want every query to contribute equally.
             A niche movie's top matches aren't drowned out.

  Default strategy if you give multiple items: --blend

MORE FLAGS
  --emb-dir PATH       Embeddings folder (default: embeddings/)
  --dataset PATH       TMDB CSV location (default: dataset/TMDB_movie_dataset_v11.csv)
  --ollama-model NAME  Ollama model for --query (default: qwen3:0.6b)
  --shard N            Analyze a single shard by index (0, 1, 2, ...)

WHAT EACH MODE DOES
  --similar "Title"    Finds the stored embedding for a movie whose title
                       matches, then returns its nearest neighbors.
                       Good for: "show me more like this specific film."

  --query "text"       Embeds your text description via Ollama using Qwen3's
                       asymmetric prompt (instruction on query, raw text on
                       movies — the intended Qwen3 search pattern).
                       Good for: "I know what I want but not a title."

SETUP
  1. Drop .h5 shards into embeddings/
  2. Put TMDB CSV at dataset/TMDB_movie_dataset_v11.csv
  3. For --query: install Ollama and pull qwen3:0.6b
       ollama pull qwen3:0.6b
"""
import argparse
from pathlib import Path

from src import config
from src.analyze import show_stats, validate, visualize, inspect_shard
from src.search import (
    search_by_title,
    search_by_query,
    search_blend,
    search_max,
    search_rrf,
)


def main():
    p = argparse.ArgumentParser(
        description="Validate, visualize, and search TMDB movie embeddings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES
  uv run python main.py                                    stats + validate
  uv run python main.py --validate                         norm/NaN check
  uv run python main.py --visualize                        all shards
  uv run python main.py --visualize 3                      shard #3 only
  uv run python main.py --shard 3                          same as above
  uv run python main.py --similar "Inception"              movies like Inception
  uv run python main.py --similar Inception Interstellar --blend
  uv run python main.py --similar "Die Hard" "Toy Story" --max
  uv run python main.py --similar "The Matrix" "Inception" "Interstellar" --rrf
  uv run python main.py --query "sci-fi thriller with twists"
  uv run python main.py --query "space" "ocean" --blend
  uv run python main.py --query "scary" "funny" --max
  uv run python main.py --query "space" "ocean" "war" --rrf

For --query to work, install Ollama and pull the model:
  ollama pull qwen3:0.6b
""",
    )

    # ── Paths ────────────────────────────────────────────────────────────────
    p.add_argument("--emb-dir", default=str(config.DEFAULT_EMB_DIR),
                   help="Folder with .h5 shard files")
    p.add_argument("--dataset", default=str(config.DEFAULT_DATASET),
                   help="Path to TMDB CSV")

    # ── Analysis ─────────────────────────────────────────────────────────────
    p.add_argument("--stats", action="store_true",
                   help="Show progress bar and shard list")
    p.add_argument("--validate", action="store_true",
                   help="Check norm and NaN for every shard")
    p.add_argument("--visualize", nargs="?", const=-1, type=int, default=None, metavar="N",
                   help="Per-shard PCA + heatmaps + 10-movie comparison. "
                        "No value = all shards. --visualize 3 = shard 3 only.")
    p.add_argument("--shard", type=int, default=None, metavar="N",
                   help="Alias for --visualize N (inspect one shard)")

    # ── Search ───────────────────────────────────────────────────────────────
    p.add_argument("--similar", nargs="+", default=None, metavar="TITLE",
                   help="Search using stored vectors of movies matching these titles")
    p.add_argument("--query", nargs="+", default=None, metavar="TEXT",
                   help="Embed text via Ollama and search. Accepts multiple queries")
    p.add_argument("--ollama-model", default=config.OLLAMA_MODEL,
                   help=f"Ollama model for --query (default: {config.OLLAMA_MODEL})")

    # ── Strategy (for multiple --similar or --query items) ────────────────────
    p.add_argument("--blend", action="store_true",
                   help="Average query vectors, search once (fast, best for similar items)")
    p.add_argument("--max", action="store_true",
                   help="Score each candidate by its best match to any query")
    p.add_argument("--rrf", action="store_true",
                   help="Merge independent searches via Reciprocal Rank Fusion")

    args = p.parse_args()

    # ── Default action ───────────────────────────────────────────────────────
    has_action = any([
        args.stats, args.validate, args.visualize is not None,
        args.similar, args.query, args.shard is not None,
    ])
    if not has_action:
        args.stats = True
        args.validate = True

    emb_dir = Path(args.emb_dir)

    print("=" * 60)
    print("  EMBEDDING VALIDATOR")
    print(f"  Directory: {emb_dir}")
    print("=" * 60)

    # ── Analysis commands ────────────────────────────────────────────────────
    if args.stats:
        show_stats(emb_dir)
    if args.validate:
        validate(emb_dir)

    if args.visualize is not None:
        # --visualize (no value) = const=-1 → all shards
        # --visualize 3 = int → just shard 3
        single = None if args.visualize == -1 else args.visualize
        visualize(emb_dir, args.dataset, single_shard=single)

    if args.shard is not None:
        # --shard N is an alias for --visualize N
        inspect_shard(args.shard, emb_dir, args.dataset)

    # ── Search commands ──────────────────────────────────────────────────────
    if args.similar:
        if len(args.similar) == 1:
            search_by_title(args.similar[0], emb_dir, args.dataset)
        else:
            _dispatch_multi(args.similar, "titles", emb_dir, args.dataset,
                            args.ollama_model, args)

    if args.query:
        if len(args.query) == 1:
            search_by_query(args.query[0], emb_dir, args.dataset, args.ollama_model)
        else:
            _dispatch_multi(args.query, "text", emb_dir, args.dataset,
                            args.ollama_model, args)


def _dispatch_multi(items, mode, emb_dir, dataset_path, ollama_model, args):
    """Route multi-item search to the right strategy."""
    # Pick strategy — rrf > max > blend (arbitrary precedence if multiple given)
    if args.rrf:
        search_rrf(items, mode, emb_dir, dataset_path, ollama_model)
    elif args.max:
        search_max(items, mode, emb_dir, dataset_path, ollama_model)
    else:
        # Default to blend
        search_blend(items, mode, emb_dir, dataset_path, ollama_model)


if __name__ == "__main__":
    main()
