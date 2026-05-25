"""
main.py — CLI for TMDB movie embeddings (Qwen3-0.6B, 1024-dim).

Supports merged HDF5 files from the Kaggle notebook and legacy shard files.
Auto-detects embedding dimension from HDF5 attributes.

============================================================================
                         QUICK COMMANDS
============================================================================
  uv run python main.py                                    stats + validate
  uv run python main.py --validate                        health check only
  uv run python main.py --visualize                       PCA + heatmaps

  uv run python main.py --similar "Inception"              look up by title
  uv run python main.py --similar Inception Interstellar --blend
  uv run python main.py --similar "Die Hard" "Toy Story" --max

  uv run python main.py --query "sci-fi thriller"          Ollama text search
  uv run python main.py --query "scary" "funny" --max

  uv run python main.py --search "Inception" --also "but more philosophical"
  uv run python main.py --profile "The Matrix" "Interstellar"
  uv run python main.py --profile "Toy Story" --mood "more grown up"
  uv run python main.py --profile "Godfather" --dislike "Twilight" --mood "crime"
============================================================================

SEARCH STRATEGIES (for multiple --similar or --query items)
  --blend    Average all query vectors into one, search once.
             Best when items are similar (all sci-fi, all horror).

  --max      Each candidate scored by its highest similarity to any query.
             Best when items are different genres (action + comedy).
             No "averaged mush" — keeps extremes.

  --rrf      Independent searches merged via Reciprocal Rank Fusion.
             Best when every query should contribute equally.

  Default strategy if you give multiple items: --blend

MORE FLAGS
  --emb-dir PATH       Embeddings folder (default: embeddings/)
  --dataset PATH       TMDB CSV location (default: dataset/TMDB_movie_dataset_v11.csv)
  --ollama-model NAME  Ollama model for queries (default: qwen3-embedding:0.6b)

For --query, --search, --also, --mood to work, install Ollama and pull:
  ollama pull qwen3-embedding:0.6b
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
    search_combined,
    search_profile,
)


def main():
    p = argparse.ArgumentParser(
        description="Validate, visualize, and search TMDB movie embeddings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES
  uv run python main.py                                    stats + validate
  uv run python main.py --validate                         norm/NaN check
  uv run python main.py --visualize                        PCA + heatmaps
  uv run python main.py --visualize 3                      shard #3 only
  uv run python main.py --shard 3                          same as above

  # Title-based search
  uv run python main.py --similar "Inception"              movies like Inception
  uv run python main.py --similar Inception Interstellar --blend
  uv run python main.py --similar "Die Hard" "Toy Story" --max
  uv run python main.py --similar "The Matrix" "Inception" "Interstellar" --rrf

  # Text query (Ollama required)
  uv run python main.py --query "sci-fi thriller with twists"
  uv run python main.py --query "space" "ocean" --blend
  uv run python main.py --query "scary" "funny" --max
  uv run python main.py --query "space" "ocean" "war" --rrf

  # Combined: titles + text
  uv run python main.py --search "Inception" --also "but more philosophical"
  uv run python main.py --search "The Godfather" --also "modern crime" --max

  # User preference profile (liked movies + mood + dislikes)
  uv run python main.py --profile "Inception" "The Matrix" "Interstellar"
  uv run python main.py --profile "Toy Story" --mood "something more grown up"
  uv run python main.py --profile "The Godfather" --dislike "Twilight" --mood "crime thriller"

For --query/--search/--profile to work, install Ollama and pull the model:
  ollama pull qwen3-embedding:0.6b
""",
    )

    # ── Paths ──
    p.add_argument("--emb-dir", default=str(config.DEFAULT_EMB_DIR),
                   help="Folder with .h5 shard files")
    p.add_argument("--dataset", default=str(config.DEFAULT_DATASET),
                   help="Path to TMDB CSV")

    # ── Analysis ──
    p.add_argument("--stats", action="store_true",
                   help="Show progress bar and shard list")
    p.add_argument("--validate", action="store_true",
                   help="Check norm and NaN for every shard")
    p.add_argument("--visualize", nargs="?", const=-1, type=int, default=None, metavar="N",
                   help="Per-shard PCA + heatmaps + 10-movie comparison. "
                        "No value = all shards. --visualize 3 = shard 3 only.")
    p.add_argument("--shard", type=int, default=None, metavar="N",
                   help="Alias for --visualize N (inspect one shard)")

    # ── Search ──
    p.add_argument("--similar", nargs="+", default=None, metavar="TITLE",
                   help="Search using stored vectors of movies matching these titles")
    p.add_argument("--query", nargs="+", default=None, metavar="TEXT",
                   help="Embed text via Ollama and search. Accepts multiple queries")

    # ── Enhanced: combine titles + text ──
    p.add_argument("--search", nargs="+", default=None, metavar="TITLE",
                   help="Like --similar, but can combine with --also text queries")
    p.add_argument("--also", nargs="+", default=None, metavar="TEXT",
                   help="Extra text queries to combine with --search titles")

    # ── User preference profile ──
    p.add_argument("--profile", nargs="+", default=None, metavar="TITLE",
                   help="Build preference vector from liked movies + optional mood")
    p.add_argument("--mood", type=str, default=None, metavar="TEXT",
                   help="Mood/context text for --profile (blended with liked movies)")
    p.add_argument("--dislike", nargs="+", default=None, metavar="TITLE",
                   help="Movies to push AWAY from in --profile results")
    p.add_argument("--mood-weight", type=float, default=0.3, metavar="W",
                   help="How much mood influences --profile (0-1, default 0.3)")

    p.add_argument("--ollama-model", default=config.OLLAMA_MODEL,
                   help=f"Ollama model for queries (default: {config.OLLAMA_MODEL})")

    # ── Strategy (for multiple --similar or --query items) ──
    p.add_argument("--blend", action="store_true",
                   help="Average query vectors, search once (fast, best for similar items)")
    p.add_argument("--max", action="store_true",
                   help="Score each candidate by its best match to any query")
    p.add_argument("--rrf", action="store_true",
                   help="Merge independent searches via Reciprocal Rank Fusion")

    args = p.parse_args()

    # ── Default action ──
    has_action = any([
        args.stats, args.validate, args.visualize is not None,
        args.similar, args.query, args.shard is not None,
        args.search, args.profile,
    ])
    if not has_action:
        args.stats = True
        args.validate = True

    emb_dir = Path(args.emb_dir)

    print("=" * 60)
    print("  EMBEDDING VALIDATOR")
    print(f"  Directory: {emb_dir}")
    print("=" * 60)

    # ── Analysis commands ──
    if args.stats:
        show_stats(emb_dir)
    if args.validate:
        validate(emb_dir)

    if args.visualize is not None:
        single = None if args.visualize == -1 else args.visualize
        visualize(emb_dir, args.dataset, single_shard=single)

    if args.shard is not None:
        inspect_shard(args.shard, emb_dir, args.dataset)

    # ── Enhanced: combined titles + text search ──
    if args.search or args.also:
        liked = args.search or []
        queries = args.also or []
        strategy = "blend"
        if args.rrf:
            strategy = "rrf"
        elif args.max:
            strategy = "max"
        search_combined(liked, queries, emb_dir, args.dataset,
                        args.ollama_model, strategy=strategy)

    # ── User preference profile ──
    elif args.profile:
        search_profile(
            liked_titles=args.profile,
            mood_text=args.mood,
            disliked_titles=args.dislike,
            emb_dir=emb_dir,
            dataset_path=args.dataset,
            ollama_model=args.ollama_model,
            mood_weight=args.mood_weight,
        )

    # ── Classic title search ──
    elif args.similar:
        if len(args.similar) == 1:
            search_by_title(args.similar[0], emb_dir, args.dataset)
        else:
            _dispatch_multi(args.similar, "titles", emb_dir, args.dataset,
                            args.ollama_model, args)

    # ── Classic text query ──
    elif args.query:
        if len(args.query) == 1:
            search_by_query(args.query[0], emb_dir, args.dataset, args.ollama_model)
        else:
            _dispatch_multi(args.query, "text", emb_dir, args.dataset,
                            args.ollama_model, args)


def _dispatch_multi(items, mode, emb_dir, dataset_path, ollama_model, args):
    """Route multi-item search to the right strategy."""
    if args.rrf:
        search_rrf(items, mode, emb_dir, dataset_path, ollama_model)
    elif args.max:
        search_max(items, mode, emb_dir, dataset_path, ollama_model)
    else:
        search_blend(items, mode, emb_dir, dataset_path, ollama_model)


if __name__ == "__main__":
    main()
