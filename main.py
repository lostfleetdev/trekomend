"""
main.py — Validate, visualize, and search TMDB movie embeddings.

Usage:
  uv run python main.py --stats                              # progress bar
  uv run python main.py --validate                           # quick health check
  uv run python main.py --visualize                          # per-shard PCA + classification
  uv run python main.py --similar "Inception"                 # search by stored movie vector
  uv run python main.py --query "sci-fi mind-bending thriller" # Ollama query embedding + search
  uv run python main.py --shard 3                            # inspect one shard

Flags can be combined:  uv run python main.py --validate --query "space exploration"
"""
import argparse
import time
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------
DEFAULT_EMB_DIR = "embeddings"
DEFAULT_DATASET = "dataset/TMDB_movie_dataset_v11.csv"
TARGET_ROWS = 1_427_380
OUT_DIM = 768
RNG = np.random.default_rng(int(time.time() * 1e6) % (2**31))


# ===========================================================================
#  Helpers
# ===========================================================================

def scan_shards(emb_dir: Path) -> list[dict]:
    """Find all qwen_*.h5 files and return shard metadata sorted by index."""
    h5_files = sorted(emb_dir.glob("qwen_*.h5"))
    shards = []
    for h5f in h5_files:
        parts = h5f.stem.split("_")
        start, end = int(parts[1]), int(parts[2])
        with h5py.File(h5f, "r") as f:
            n_rows = f["embeddings"].shape[0]
        shards.append({
            "index": len(shards),
            "file": h5f,
            "start": start,
            "end": end,
            "rows": n_rows,
            "size_mb": h5f.stat().st_size / 1e6,
        })
    return shards


def primary_genre(s) -> str:
    if pd.isna(s) or str(s).strip() == "":
        return "Unknown"
    return str(s).split(",")[0].strip()


def load_metadata_for_shard(h5_path: Path, dataset_path: str):
    """
    Load TMDB CSV rows matching a single .h5 shard's /rows dataset.
    Returns (DataFrame with title/genre/... on those rows, ids array, rows array).
    """
    with h5py.File(h5_path, "r") as hf:
        row_indices = hf["rows"][:]
        ids_arr = hf["ids"][:]

    needed = set(int(r) for r in row_indices)
    parts = []
    for chunk in pd.read_csv(dataset_path, chunksize=250_000, low_memory=False):
        mask = chunk.index.isin(needed)
        if mask.any():
            parts.append(chunk[mask])
        if sum(len(c) for c in parts) >= len(needed):
            break
    df = pd.concat(parts, ignore_index=False)
    # Reorder to match shard order
    df = df.loc[row_indices].reset_index(drop=True)
    df["primary_genre"] = df["genres"].apply(primary_genre)
    return df, ids_arr, row_indices


def load_all(emb_dir: Path, dataset_path: str):
    """Load all shard embeddings + metadata into memory. Returns (emb, df, rows_list, shards)."""
    shards = scan_shards(emb_dir)
    if not shards:
        return None, None, None, []
    emb_parts, df_parts, rows_parts = [], [], []
    for s in shards:
        with h5py.File(s["file"], "r") as hf:
            chunk = hf["embeddings"][:]
            emb_parts.append(np.array(chunk, dtype=np.float32))
        df_c, ids_c, rows_c = load_metadata_for_shard(s["file"], dataset_path)
        df_parts.append(df_c)
        rows_parts.append(rows_c)
    emb = np.concatenate(emb_parts, axis=0)
    df = pd.concat(df_parts, ignore_index=True)
    rows_all = np.concatenate(rows_parts, axis=0)
    return emb, df, rows_all, shards


def embed_query_via_ollama(query_text: str, model: str = "qwen3:0.6b") -> np.ndarray:
    """
    Embed a search query using a local Ollama API endpoint.
    Uses Qwen3 asymmetric format: Instruct: ...\nQuery: ...
    """
    import json as _json
    import urllib.request as _urllib

    instruction = "Given a movie search query, retrieve the most relevant movies."
    prompt = f"Instruct: {instruction}\nQuery: {query_text}"

    payload = _json.dumps({
        "model": model,
        "prompt": prompt,
    }).encode("utf-8")

    req = _urllib.Request("http://localhost:11434/api/embeddings",
                         data=payload,
                         headers={"Content-Type": "application/json"},
                         method="POST")

    with _urllib.urlopen(req) as resp:
        data = _json.loads(resp.read().decode("utf-8"))

    vec = np.array(data["embedding"], dtype=np.float32)
    # Ollama returns the full 1024-dim vector; truncate to OUT_DIM for Matryoshka
    vec = vec[:OUT_DIM]
    # Normalize (Ollama vectors may not be unit-normalized)
    vec = vec / np.linalg.norm(vec)
    return vec


# ===========================================================================
#  Commands
# ===========================================================================

def cmd_stats(emb_dir: Path) -> None:
    """Print progress bar and shard list."""
    shards = scan_shards(emb_dir)
    if not shards:
        print("No shards found in", emb_dir)
        return
    total = sum(s["rows"] for s in shards)
    pct = 100 * total / TARGET_ROWS
    bar_w = 40
    filled = int(bar_w * total / TARGET_ROWS)
    bar = "#" * filled + "-" * (bar_w - filled)

    print(f"Shards: {len(shards)}")
    print(f"Movies: {total:,} / {TARGET_ROWS:,}  ({pct:.1f}%)")
    print(f"Size:   {sum(s['size_mb'] for s in shards):.0f} MB")
    print(f"[{bar}]")
    for s in shards:
        print(f"  {s['file'].name}  ({s['rows']:,} rows)")


def cmd_validate(emb_dir: Path) -> None:
    """Check every shard for norm, NaN, basic integrity."""
    shards = scan_shards(emb_dir)
    if not shards:
        print("No shards found.")
        return

    print(f"{'Shard':<6}{'Rows':>7}{'Norm':>8}{'NaN':>6}{'Status'}")
    print("-" * 45)
    all_ok = True
    for s in shards:
        with h5py.File(s["file"], "r") as hf:
            emb = hf["embeddings"][:]
        norm = np.linalg.norm(emb, axis=1).mean()
        has_nan = not np.isfinite(emb).all()
        ok = abs(norm - 1.0) < 0.01 and not has_nan
        if not ok:
            all_ok = False
        print(f"{s['index']:<6}{s['rows']:>7,}{norm:>8.4f}{str(has_nan):>6}  {'PASS' if ok else 'FAIL'}")

    print(f"\nOverall: {'PASS' if all_ok else 'FAIL — check flagged shards'}")


def cmd_visualize(emb_dir: Path, dataset_path: str) -> None:
    """Per-shard PCA plots, similarity heatmaps, genre classification."""
    shards = scan_shards(emb_dir)
    if not shards:
        print("No shards found.")
        return

    summary = []
    for s in shards:
        idx = s["index"]
        print(f"\n{'='*60}")
        print(f"  SHARD {idx} — {s['file'].name}  ({s['rows']:,} movies)")
        print(f"{'='*60}")

        df, ids_arr, rows_arr = load_metadata_for_shard(s["file"], dataset_path)
        with h5py.File(s["file"], "r") as hf:
            emb = hf["embeddings"][:]
        emb = np.array(emb, dtype=np.float32)
        genre_counts = df["primary_genre"].value_counts()

        norm = np.linalg.norm(emb, axis=1).mean()
        nan = not np.isfinite(emb).all()
        print(f"  Norm: {norm:.4f}  |  NaN: {nan}  |  Genres: {len(genre_counts)}")

        # --- PCA ---
        pca50 = PCA(n_components=50, random_state=42)
        emb50 = pca50.fit_transform(emb)
        pca2 = PCA(n_components=2, random_state=42)
        emb2d = pca2.fit_transform(emb50)
        pca50_var = pca50.explained_variance_ratio_.sum()
        pca2_var = pca2.explained_variance_ratio_.sum()
        print(f"  PCA-50: {pca50_var:.3f}  |  PCA-2: {pca2_var:.3f}")

        # --- PCA plot ---
        _plot_pca(emb2d, df, genre_counts, idx, s)
        # --- Similarity heatmap ---
        _plot_similarity(emb, df, idx)
        # --- Classification ---
        acc, baseline, per_genre = _classify(emb, df)

        if acc is not None:
            xbase = acc / baseline
            print(f"  Classify: {acc:.3f} +/- {acc*0.01:.3f}  ({xbase:.1f}x baseline)")
            best = per_genre[:3]
            worst = per_genre[-3:]
            print(f"    Best:  {', '.join(f'{g}({s:.2f})' for g,s in best)}")
            print(f"    Worst: {', '.join(f'{g}({s:.2f})' for g,s in worst)}")
        else:
            acc, baseline, xbase = None, None, None
            print(f"  Classify: SKIP")

        summary.append({"shard": idx, "file": s["file"].name, "rows": s["rows"],
                        "norm": norm, "nan": nan, "pca50": pca50_var,
                        "acc": acc, "xbase": xbase, "genres": len(genre_counts)})

    # Summary table
    print(f"\n{'='*70}")
    print(f"  SUMMARY — {len(shards)} shards")
    print(f"{'='*70}")
    print(f"  {'Shard':<6}{'Rows':>7}{'Norm':>8}{'NaN':>6}{'PCA50':>7}{'Acc':>8}{'xBase':>7}{'Genres':>7}")
    print(f"  {'-'*6}{'-'*7}{'-'*8}{'-'*6}{'-'*7}{'-'*8}{'-'*7}{'-'*7}")
    for r in summary:
        acc_s = f"{r['acc']:.3f}" if r["acc"] else "N/A"
        xb_s = f"{r['xbase']:.1f}x" if r["xbase"] else "N/A"
        print(f"  {r['shard']:<6}{r['rows']:>7,}{r['norm']:>8.4f}{str(r['nan']):>6}"
              f"{r['pca50']:>7.3f}{acc_s:>8}{xb_s:>7}{r['genres']:>7}")
    print("=" * 70)


def cmd_similar(query: str, emb_dir: Path, dataset_path: str) -> None:
    """Search across ALL shards using the vector of a movie matching the given title."""
    print("Loading all shards...")
    emb, df, rows_all, shards = load_all(emb_dir, dataset_path)
    if emb is None:
        print("No shards found.")
        return

    match = df[df["title"].str.lower() == query.lower()]
    if match.empty:
        match = df[df["title"].str.contains(query, case=False, na=False)]
    if match.empty:
        print(f"'{query}' not found in any shard.")
        return

    qi = match.index[0]
    q_title = df.iloc[qi]["title"]
    q_genre = df.iloc[qi]["primary_genre"]
    sims = emb @ emb[qi]
    top_k = np.argsort(sims)[::-1][:12]

    print(f"\n  Most similar to: '{q_title}'  ({q_genre})")
    print(f"  Searching across {emb.shape[0]:,} movies in {len(shards)} shards\n")
    print(f"  {'Rank':<6}{'Title':<45}{'Genre':<22}{'Score'}")
    print(f"  {'-'*5} {'-'*45} {'-'*22} {'-'*5}")
    for rank, idx in enumerate(top_k):
        mark = " <--" if idx == qi else ""
        t = str(df.iloc[idx]["title"]) if not pd.isna(df.iloc[idx]["title"]) else f"#{idx}"
        g = df.iloc[idx]["primary_genre"]
        dup = " DUP" if (idx != qi and sims[idx] > 0.9999) else ""
        print(f"  {rank+1:<6}{t:<45}{g:<22}{sims[idx]:.4f}{mark}{dup}")

    dup_count = sum(1 for idx in top_k if idx != qi and sims[idx] > 0.9999)
    if dup_count:
        print(f"\n  [!] {dup_count} result(s) marked DUP — identical vectors.")
        print(f"      Caused by identical movie_text() output (empty fields).")
        print(f"      Re-embed with the updated v2 notebook to fix.")


def cmd_query(query_text: str, emb_dir: Path, dataset_path: str, ollama_model: str = "qwen3:0.6b") -> None:
    """Search across ALL shards using Ollama to embed the query text."""
    print(f"Embedding query via Ollama ({ollama_model})...")
    try:
        q_vec = embed_query_via_ollama(query_text, model=ollama_model)
    except Exception as e:
        print(f"  Error: {e}")
        print(f"  Make sure Ollama is running and has '{ollama_model}' pulled:")
        print(f"    ollama pull {ollama_model}")
        return

    print(f"  Query vector: {OUT_DIM}-dim, norm={np.linalg.norm(q_vec):.4f}")

    shards = scan_shards(emb_dir)
    if not shards:
        print("No shards found.")
        return

    # Score each shard independently to keep RAM low
    print(f"  Searching {sum(s['rows'] for s in shards):,} movies across {len(shards)} shards...")

    all_scores = []
    all_indices = []
    offset = 0
    for s in shards:
        with h5py.File(s["file"], "r") as hf:
            chunk = hf["embeddings"][:]
        chunk = np.array(chunk, dtype=np.float32)
        scores = chunk @ q_vec
        all_scores.append(scores)
        all_indices.append(np.arange(offset, offset + len(scores)))
        offset += len(scores)

    scores = np.concatenate(all_scores)
    indices = np.concatenate(all_indices)
    top_k = np.argsort(scores)[::-1][:12]

    # Load metadata for top results only
    print(f"  Loading metadata for top results...")
    emb, df, rows_all, _ = load_all(emb_dir, dataset_path)

    print(f"\n  Search: '{query_text}'")
    print(f"  Across: {emb.shape[0]:,} movies in {len(shards)} shards\n")
    print(f"  {'Rank':<6}{'Title':<45}{'Genre':<22}{'Score'}")
    print(f"  {'-'*5} {'-'*45} {'-'*22} {'-'*5}")
    for rank, idx in enumerate(top_k):
        t = str(df.iloc[idx]["title"]) if not pd.isna(df.iloc[idx]["title"]) else f"#{idx}"
        g = df.iloc[idx]["primary_genre"]
        print(f"  {rank+1:<6}{t:<45}{g:<22}{scores[idx]:.4f}")


def cmd_shard(index: int, emb_dir: Path, dataset_path: str) -> None:
    """Inspect a single shard: PCA plot, similarity, classification."""
    shards = scan_shards(emb_dir)
    if index < 0 or index >= len(shards):
        print(f"Shard {index} out of range (0-{len(shards)-1})")
        return

    s = shards[index]
    df, ids_arr, rows_arr = load_metadata_for_shard(s["file"], dataset_path)
    with h5py.File(s["file"], "r") as hf:
        emb = hf["embeddings"][:]
    emb = np.array(emb, dtype=np.float32)
    genre_counts = df["primary_genre"].value_counts()
    norm = np.linalg.norm(emb, axis=1).mean()
    nan = not np.isfinite(emb).all()

    print(f"\n  Shard {index}: {s['file'].name}  ({s['rows']:,} movies)")
    print(f"  Norm: {norm:.4f}  |  NaN: {nan}  |  Genres: {len(genre_counts)}")
    print(f"  Top genres: {dict(genre_counts.head(6))}")

    # PCA
    pca50 = PCA(n_components=50, random_state=42)
    emb50 = pca50.fit_transform(emb)
    pca2 = PCA(n_components=2, random_state=42)
    emb2d = pca2.fit_transform(emb50)
    print(f"  PCA-50 var: {pca50.explained_variance_ratio_.sum():.3f}")

    _plot_pca(emb2d, df, genre_counts, index, s)
    _plot_similarity(emb, df, index)

    acc, baseline, per_genre = _classify(emb, df)
    if acc is not None:
        xbase = acc / baseline
        print(f"  Classify: {acc:.3f}  ({xbase:.1f}x baseline)")
        for g, sc in per_genre[:5]:
            print(f"    {g:<20} f1={sc:.3f}")


# ===========================================================================
#  Plot helpers
# ===========================================================================

def _plot_pca(emb2d: np.ndarray, df: pd.DataFrame, genre_counts,
              shard_idx: int, shard_info: dict) -> None:
    fig, ax = plt.subplots(figsize=(13, 9))
    top8 = genre_counts.head(8).index.tolist()
    colors = plt.cm.tab10(np.linspace(0, 1, len(top8)))
    for i, g in enumerate(top8):
        m = df["primary_genre"] == g
        ax.scatter(emb2d[m, 0], emb2d[m, 1], c=[colors[i]],
                   label=f"{g} ({m.sum():,})", s=8, alpha=0.5, edgecolors="none")
    other = ~df["primary_genre"].isin(top8)
    ax.scatter(emb2d[other, 0], emb2d[other, 1],
               c="lightgrey", label=f"Other ({other.sum():,})", s=4, alpha=0.2)

    n_label = min(12, len(df))
    for i in sorted(RNG.choice(len(df), size=n_label, replace=False)):
        t = str(df.iloc[i]["title"]) if not pd.isna(df.iloc[i]["title"]) else f"#{i}"
        ax.annotate(t, (emb2d[i, 0], emb2d[i, 1]),
                    fontsize=7, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
                    ha="center")

    ax.set_title(f"Shard {shard_idx}: {shard_info['rows']:,} Movies — PCA 2D\n{shard_info['file'].name}",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=7, markerscale=2, framealpha=0.9, title="Genre")
    ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
    ax.grid(True, alpha=0.10)
    plt.tight_layout()
    fname = f"embeddings_pca_shard_{shard_idx}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


def _plot_similarity(emb: np.ndarray, df: pd.DataFrame, shard_idx: int) -> None:
    fig, ax = plt.subplots(figsize=(9, 8))
    n_show = min(12, len(df))
    rand_idx = sorted(RNG.choice(len(df), size=n_show, replace=False))
    rand_emb = emb[rand_idx]
    rand_titles = [str(df.iloc[i]["title"]) if not pd.isna(df.iloc[i]["title"]) else f"#{i}"
                   for i in rand_idx]
    sim = rand_emb @ rand_emb.T

    ax.imshow(sim, cmap="YlOrRd", vmin=0.15, vmax=1.0, aspect="equal")
    ax.set_xticks(range(n_show))
    ax.set_yticks(range(n_show))
    ax.set_xticklabels(rand_titles, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(rand_titles, fontsize=7)
    ax.set_title(f"Shard {shard_idx} — Similarity ({n_show} random movies)",
                 fontsize=12, fontweight="bold")
    for i in range(n_show):
        for j in range(i + 1, n_show):
            if sim[i, j] > 0.3:
                ax.text(j, i, f"{sim[i,j]:.2f}", ha="center", va="center", fontsize=5)
    plt.tight_layout()
    fname = f"embeddings_sim_shard_{shard_idx}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


def _classify(emb: np.ndarray, df: pd.DataFrame):
    """5-fold logistic regression on primary genre. Returns (acc, baseline, per_genre_list)."""
    genre_counts = df["primary_genre"].value_counts()
    valid = genre_counts[genre_counts >= 15].index.tolist()
    if len(valid) < 2:
        return None, None, "need >=2 genres with 15+ samples"

    df_v = df[df["primary_genre"].isin(valid)]
    emb_v = emb[df_v.index]
    le = LabelEncoder()
    y_raw = le.fit_transform(df_v["primary_genre"])
    cls_mask = np.isin(y_raw, np.where(np.bincount(y_raw) >= 2)[0])
    X = StandardScaler().fit_transform(emb_v[cls_mask])
    y = LabelEncoder().fit_transform(y_raw[cls_mask])

    if len(np.unique(y)) < 2:
        return None, None, "need >=2 classes after filtering"

    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    n_splits = min(5, np.bincount(y).min())
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    baseline = np.bincount(y).max() / len(y)

    y_pred = cross_val_predict(clf, X, y, cv=cv)
    report = classification_report(y, y_pred, target_names=le.classes_,
                                   output_dict=True, zero_division=0)
    per_genre = [(g, report[g]["f1-score"]) for g in report
                 if g not in ("accuracy", "macro avg", "weighted avg")]
    per_genre.sort(key=lambda x: -x[1])
    return scores.mean(), baseline, per_genre


# ===========================================================================
#  CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate, visualize, and search TMDB movie embeddings.")
    p.add_argument("--emb-dir", default=DEFAULT_EMB_DIR, help="Path to embeddings folder")
    p.add_argument("--dataset", default=DEFAULT_DATASET, help="Path to TMDB CSV")
    p.add_argument("--stats", action="store_true", help="Show progress and shard list")
    p.add_argument("--validate", action="store_true", help="Quick health check across all shards")
    p.add_argument("--visualize", action="store_true", help="Per-shard PCA plots + classification")
    p.add_argument("--similar", type=str, default=None, metavar="TITLE",
                   help="Search using the vector of a movie whose title matches TITLE")
    p.add_argument("--query", type=str, default=None, metavar="TEXT",
                   help="Embed TEXT via Ollama and search across all shards")
    p.add_argument("--ollama-model", type=str, default="qwen3:0.6b",
                   help="Ollama model name for --query (default: qwen3:0.6b)")
    p.add_argument("--shard", type=int, default=None, metavar="N",
                   help="Inspect a single shard by index")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    emb_dir = Path(args.emb_dir)

    # If no flags given, default to stats + validate
    if not any([args.stats, args.validate, args.visualize,
                args.similar, args.query, args.shard is not None]):
        args.stats = True
        args.validate = True

    print("=" * 60)
    print("  EMBEDDING VALIDATOR")
    print(f"  Directory: {emb_dir}")
    print("=" * 60)

    if args.stats:
        cmd_stats(emb_dir)
    if args.validate:
        cmd_validate(emb_dir)
    if args.visualize:
        cmd_visualize(emb_dir, args.dataset)
    if args.similar:
        cmd_similar(args.similar, emb_dir, args.dataset)
    if args.query:
        cmd_query(args.query, emb_dir, args.dataset, args.ollama_model)
    if args.shard is not None:
        cmd_shard(args.shard, emb_dir, args.dataset)


if __name__ == "__main__":
    main()
