"""
Validation, statistics, visualization, and classification for embedding shards.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .config import TARGET_ROWS, RNG
from .io import scan_shards, load_metadata, load_shard_array


# ═══════════════════════════════════════════════════════════════════════════════
#  Stats
# ═══════════════════════════════════════════════════════════════════════════════

def show_stats(emb_dir: Path) -> None:
    """Progress bar and shard list."""
    shards = scan_shards(emb_dir)
    if not shards:
        print("No shards found in", emb_dir)
        return

    total = sum(s["rows"] for s in shards)
    pct = 100 * total / TARGET_ROWS
    bar_w = 40
    filled = min(bar_w, int(bar_w * total / TARGET_ROWS))
    bar = "#" * filled + "-" * (bar_w - filled)

    print(f"Shards: {len(shards)}")
    print(f"Movies: {total:,} / {TARGET_ROWS:,}  ({pct:.1f}%)")
    print(f"Size:   {sum(s['size_mb'] for s in shards):.0f} MB")
    print(f"[{bar}]")
    for s in shards:
        print(f"  {s['file'].name}  ({s['rows']:,} rows)")


# ═══════════════════════════════════════════════════════════════════════════════
#  Validate
# ═══════════════════════════════════════════════════════════════════════════════

def validate(emb_dir: Path) -> bool:
    """Check every shard: norm ~1.0, no NaN. Returns True if all pass."""
    shards = scan_shards(emb_dir)
    if not shards:
        print("No shards found.")
        return False

    print(f"{'Shard':<6}{'Rows':>7}{'Norm':>8}{'NaN':>6}{'Status'}")
    print("-" * 45)
    all_ok = True
    for s in shards:
        emb = load_shard_array(s["file"])
        norm = np.linalg.norm(emb, axis=1).mean()
        has_nan = not np.isfinite(emb).all()
        ok = abs(norm - 1.0) < 0.01 and not has_nan
        if not ok:
            all_ok = False
        print(f"{s['index']:<6}{s['rows']:>7,}{norm:>8.4f}{str(has_nan):>6}  {'PASS' if ok else 'FAIL'}")

    print(f"\nOverall: {'PASS' if all_ok else 'FAIL — check flagged shards'}")
    return all_ok


# ═══════════════════════════════════════════════════════════════════════════════
#  Visualization
# ═══════════════════════════════════════════════════════════════════════════════
#
#  Each shard gets:
#    1. PCA scatter plot (colored by genre, random titles labeled)
#    2. Similarity heatmap (random sample of 12 movies)
#    3. 10-movie comparison — picks 10 random movies, prints a text
#       similarity matrix and highlights the strongest cross-movie pairs
#    4. Genre classification score (logistic regression on embeddings)

def visualize(emb_dir: Path, dataset_path: str, single_shard: int = None) -> None:
    """
    Visualize embeddings.

    If single_shard is None → process every shard.
    If single_shard is an int → process only that shard (0-indexed).
    """
    shards = scan_shards(emb_dir)
    if not shards:
        print("No shards found.")
        return

    # Filter to one shard if requested
    if single_shard is not None:
        if single_shard < 0 or single_shard >= len(shards):
            print(f"Shard {single_shard} out of range (0-{len(shards) - 1})")
            return
        shards = [shards[single_shard]]
        print(f"  Visualizing shard {single_shard} only")

    summary = []
    for s in shards:
        idx = s["index"]
        print(f"\n{'='*60}")
        print(f"  SHARD {idx} — {s['file'].name}  ({s['rows']:,} movies)")
        print(f"{'='*60}")

        emb = load_shard_array(s["file"])
        df = load_metadata(s["file"], dataset_path)
        genre_counts = df["primary_genre"].value_counts()

        norm = np.linalg.norm(emb, axis=1).mean()
        nan = not np.isfinite(emb).all()
        print(f"  Norm: {norm:.4f}  |  NaN: {nan}  |  Genres: {len(genre_counts)}")

        # ── PCA ───
        pca50 = PCA(n_components=50, random_state=42)
        emb50 = pca50.fit_transform(emb)
        pca2 = PCA(n_components=2, random_state=42)
        emb2d = pca2.fit_transform(emb50)
        pca50_var = pca50.explained_variance_ratio_.sum()
        print(f"  PCA-50 var: {pca50_var:.3f}")

        # ── Plots ──
        _plot_pca(emb2d, df, genre_counts, idx, s)
        _plot_heatmap(emb, df, idx)

        # ── 10-movie similarity comparison ──
        _compare_random_sample(emb, df, idx, n_sample=10)

        # ── Classification ──
        acc, baseline, per_genre = _classify_genre(emb, df)
        if acc is not None:
            xbase = acc / baseline
            print(f"\n  Classify: {acc:.3f}  ({xbase:.1f}x baseline)")
            for g, sc in per_genre[:5]:
                print(f"    {g:<25} f1={sc:.3f}")
        else:
            acc, xbase = None, None
            print("\n  Classify: SKIP")

        summary.append({
            "shard": idx, "file": s["file"].name, "rows": s["rows"],
            "norm": norm, "nan": nan, "pca50": pca50_var,
            "acc": acc, "xbase": xbase, "genres": len(genre_counts),
        })

    # Summary table (only when multiple shards)
    if len(shards) > 1:
        _print_summary(shards, summary)


# ═══════════════════════════════════════════════════════════════════════════════
#  Single-shard inspection
# ═══════════════════════════════════════════════════════════════════════════════

def inspect_shard(index: int, emb_dir: Path, dataset_path: str) -> None:
    """Deep dive on one shard — same as visualize(..., single_shard=index)."""
    visualize(emb_dir, dataset_path, single_shard=index)


# ═══════════════════════════════════════════════════════════════════════════════
#  Plot helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _plot_pca(emb2d, df, genre_counts, shard_idx, shard_info):
    fig, ax = plt.subplots(figsize=(14, 10))
    top8 = genre_counts.head(8).index.tolist()
    colors = plt.cm.tab10(np.linspace(0, 1, len(top8)))

    for i, g in enumerate(top8):
        mask = df["primary_genre"] == g
        ax.scatter(emb2d[mask, 0], emb2d[mask, 1], c=[colors[i]],
                   label=f"{g} ({mask.sum():,})", s=10, alpha=0.55, edgecolors="none")

    other = ~df["primary_genre"].isin(top8)
    ax.scatter(emb2d[other, 0], emb2d[other, 1],
               c="lightgrey", label=f"Other ({other.sum():,})", s=4, alpha=0.2)

    n_label = min(12, len(df))
    picks = sorted(RNG.choice(len(df), size=n_label, replace=False))
    for i in picks:
        t = str(df.iloc[i]["title"]) if not pd.isna(df.iloc[i]["title"]) else f"#{i}"
        ax.annotate(t, (emb2d[i, 0], emb2d[i, 1]),
                    fontsize=7, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85),
                    ha="center")

    ax.set_title(f"Shard {shard_idx}: {shard_info['rows']:,} Movies — PCA 2D\n{shard_info['file'].name}",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=7, markerscale=2, framealpha=0.9, title="Genre")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.grid(True, alpha=0.10)
    plt.tight_layout()
    fname = f"embeddings_pca_shard_{shard_idx}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


def _plot_heatmap(emb, df, shard_idx):
    """Similarity heatmap of 12 random movies."""
    fig, ax = plt.subplots(figsize=(9, 8))
    n_show = min(12, len(df))
    picks = sorted(RNG.choice(len(df), size=n_show, replace=False))
    sim = emb[picks] @ emb[picks].T

    titles = [str(df.iloc[i]["title"]) if not pd.isna(df.iloc[i]["title"]) else f"#{i}"
              for i in picks]

    ax.imshow(sim, cmap="YlOrRd", vmin=0.10, vmax=1.0, aspect="equal")
    ax.set_xticks(range(n_show))
    ax.set_yticks(range(n_show))
    ax.set_xticklabels(titles, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(titles, fontsize=7)
    ax.set_title(f"Shard {shard_idx} — Similarity ({n_show} random movies)",
                 fontsize=12, fontweight="bold")

    # Annotate strong similarities (cos > 0.4 or > 0.3 for more)
    for i in range(n_show):
        for j in range(i + 1, n_show):
            if sim[i, j] > 0.3:
                ax.text(j, i, f"{sim[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        fontweight="bold" if sim[i, j] > 0.5 else "normal")

    plt.tight_layout()
    fname = f"embeddings_sim_shard_{shard_idx}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


def _compare_random_sample(emb, df, shard_idx, n_sample=10, threshold=0.25):
    """
    Pick N random movies from the shard, compute their pairwise similarity,
    print a matrix and highlight the strongest cross-movie pairs.
    """
    n = len(df)
    if n < n_sample:
        print(f"\n  Skipping comparison — only {n} movies in shard")
        return

    picks = sorted(RNG.choice(n, size=n_sample, replace=False))
    sample_emb = emb[picks]
    sim_matrix = sample_emb @ sample_emb.T

    titles = []
    genres = []
    for i in picks:
        t = str(df.iloc[i]["title"]) if not pd.isna(df.iloc[i]["title"]) else f"#{i}"
        g = df.iloc[i]["primary_genre"]
        titles.append(t)
        genres.append(g)

    print(f"\n  ── Similarity Matrix ({n_sample} random movies) ──")

    # Header row (abbreviated titles)
    print(f"  {'':>25}", end="")
    for t in titles:
        print(f"{t[:10]:>10}", end="")
    print()

    # Matrix rows
    for i in range(n_sample):
        print(f"  {titles[i]:>25}", end="")
        for j in range(n_sample):
            v = sim_matrix[i, j]
            if i == j:
                print(f"  {'·':>9}", end="")
            else:
                print(f"  {v:>8.3f}", end="")
        print()

    # Highlight the strongest cross-movie similarities
    pairs = []
    for i in range(n_sample):
        for j in range(i + 1, n_sample):
            if sim_matrix[i, j] >= threshold:
                pairs.append((sim_matrix[i, j], i, j))

    pairs.sort(reverse=True)

    if pairs:
        print(f"\n  Most similar pairs (cos >= {threshold}):")
        for score, i, j in pairs:
            genre_match = "**" if genres[i] == genres[j] else ""
            print(f"    {titles[i][:35]:<35} ↔ {titles[j][:35]:<35}  "
                  f"cos={score:.3f}  {genre_match}")
    else:
        print(f"\n  No pairs with cos >= {threshold} — vectors are well-separated.")

    # Save a dedicated comparison heatmap
    _save_comparison_heatmap(sim_matrix, titles, genres, shard_idx, n_sample)


def _save_comparison_heatmap(sim_matrix, titles, genres, shard_idx, n_sample):
    """Save a larger, annotated heatmap of the random sample comparison."""
    fig, ax = plt.subplots(figsize=(11, 10))

    im = ax.imshow(sim_matrix, cmap="RdYlBu_r", vmin=-0.1, vmax=1.0, aspect="equal")

    # Color bar
    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Cosine similarity", fontsize=9)

    # Tick labels with genre
    labels = [f"{t[:25]}\n({g})" for t, g in zip(titles, genres)]
    ax.set_xticks(range(n_sample))
    ax.set_yticks(range(n_sample))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)

    # Annotate all off-diagonal values
    for i in range(n_sample):
        for j in range(n_sample):
            if i == j:
                continue
            v = sim_matrix[i, j]
            color = "white" if v < 0.3 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    fontweight="bold" if v >= 0.5 else "normal", color=color)

    ax.set_title(f"Shard {shard_idx} — {n_sample} Random Movies Similarity",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fname = f"embeddings_compare_shard_{shard_idx}.png"
    plt.savefig(fname, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


def _print_summary(shards, summary):
    """Summary table across all shards."""
    print(f"\n{'='*70}")
    print(f"  SUMMARY — {len(shards)} shards")
    print(f"{'='*70}")
    hdr = f"  {'Shard':<6}{'Rows':>7}{'Norm':>8}{'NaN':>6}{'PCA50':>7}{'Acc':>8}{'xBase':>7}{'Genres':>7}"
    print(hdr)
    print("  " + "-" * 61)
    for r in summary:
        acc_s = f"{r['acc']:.3f}" if r["acc"] else "N/A"
        xb_s = f"{r['xbase']:.1f}x" if r["xbase"] else "N/A"
        na_s = "Y" if r["nan"] else "N"
        print(f"  {r['shard']:<6}{r['rows']:>7,}{r['norm']:>8.4f}{na_s:>6}"
              f"{r['pca50']:>7.3f}{acc_s:>8}{xb_s:>7}{r['genres']:>7}")
    print("=" * 70)


def _classify_genre(emb, df):
    """5-fold logistic regression on primary genre. Returns (accuracy, baseline, f1_per_genre)."""
    genre_counts = df["primary_genre"].value_counts()
    valid = genre_counts[genre_counts >= 15].index.tolist()
    if len(valid) < 2:
        return None, None, []

    mask = df["primary_genre"].isin(valid)
    df_v = df[mask]
    emb_v = emb[mask.values]

    le = LabelEncoder()
    y = le.fit_transform(df_v["primary_genre"])

    # Remove classes with only 1 sample
    keep = np.isin(y, np.where(np.bincount(y) >= 2)[0])
    X = StandardScaler().fit_transform(emb_v[keep])
    y = LabelEncoder().fit_transform(y[keep])

    if len(np.unique(y)) < 2:
        return None, None, []

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
