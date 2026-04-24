"""
geometry.py — Embed figurative expressions and visualise their geometry.

For each classified sentence the *expression* field (the key word/phrase
identified by the classifier) is embedded with sentence-transformers.
UMAP projects the embeddings into 2D; the plot shows METAPHOR / METONYMY /
LITERAL as three colour-coded clouds.

Theory (Jakobson): metaphor links distant semantic domains → points should be
*spread* across the embedding space. Metonymy substitutes within a domain →
points should be *locally clustered*.

We quantify this with two measures printed to stdout:
  - Mean pairwise cosine distance within each class (higher = more spread)
  - Davies-Bouldin index per class pair (lower = tighter, more separated)

No API calls. Runs entirely on CPU (slow but feasible for 2200 points).

Usage:
    python3 geometry.py --input results.parquet
    python3 geometry.py --input results.parquet --output geometry.png --method umap
    python3 geometry.py --input results.parquet --output geometry.png --method tsne
"""

import argparse
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

COLORS = {
    "METAPHOR": "#e05c3a",   # red-orange
    "METONYMY":  "#3a7ec8",  # blue
    "LITERAL":   "#aaaaaa",  # grey
}
ALPHA = {"METAPHOR": 0.55, "METONYMY": 0.75, "LITERAL": 0.35}
SIZE  = {"METAPHOR": 18,   "METONYMY": 22,   "LITERAL": 10}
ORDER = ["LITERAL", "METAPHOR", "METONYMY"]   # paint order: METONYMY on top


def embed(texts: list[str], model_name: str) -> np.ndarray:
    log.info("Loading model %s ...", model_name)
    model = SentenceTransformer(model_name)
    log.info("Embedding %d texts ...", len(texts))
    vecs = model.encode(texts, batch_size=64, show_progress_bar=True,
                        convert_to_numpy=True)
    return normalize(vecs)   # unit-norm for cosine geometry


def reduce_umap(vecs: np.ndarray, seed: int) -> np.ndarray:
    from umap import UMAP
    log.info("Running UMAP ...")
    return UMAP(n_components=2, random_state=seed, metric="cosine",
                n_neighbors=15, min_dist=0.1).fit_transform(vecs)


def reduce_tsne(vecs: np.ndarray, seed: int) -> np.ndarray:
    from sklearn.manifold import TSNE
    log.info("Running t-SNE ...")
    return TSNE(n_components=2, random_state=seed, metric="cosine",
                perplexity=40, n_iter=1000, init="pca").fit_transform(vecs)


def spread_stats(vecs: np.ndarray, labels: np.ndarray) -> dict:
    """Mean pairwise cosine distance within each class (higher = more spread)."""
    stats = {}
    for cls in np.unique(labels):
        mask = labels == cls
        sub = vecs[mask]
        if len(sub) < 2:
            stats[cls] = float("nan")
            continue
        # cosine distance = 1 - cosine_similarity; vecs are unit-normed so
        # cosine_sim = dot product, distance = 1 - dot
        dots = sub @ sub.T
        n = len(sub)
        # upper triangle only, excluding diagonal
        upper = dots[np.triu_indices(n, k=1)]
        stats[cls] = float(np.mean(1 - upper))
    return stats


def plot(df: pd.DataFrame, coords: np.ndarray, output_path: str, method: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # --- Left: all three classes ---
    ax = axes[0]
    for cls in ORDER:
        mask = df["type"] == cls
        if mask.sum() == 0:
            continue
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=COLORS[cls], label=f"{cls} (n={mask.sum()})",
            s=SIZE[cls], alpha=ALPHA[cls], linewidths=0,
        )
    ax.set_title(f"All classes — {method.upper()} projection\nof figurative expression embeddings",
                 fontsize=11)
    ax.legend(fontsize=9, markerscale=1.8)
    ax.set_xticks([]); ax.set_yticks([])

    # --- Right: METAPHOR vs METONYMY only (LITERAL hidden to reduce noise) ---
    ax = axes[1]
    for cls in ["METAPHOR", "METONYMY"]:
        mask = df["type"] == cls
        if mask.sum() == 0:
            continue
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=COLORS[cls], label=f"{cls} (n={mask.sum()})",
            s=SIZE[cls] + 4, alpha=0.65, linewidths=0,
        )
    ax.set_title("METAPHOR vs METONYMY\n(LITERAL hidden)", fontsize=11)
    ax.legend(fontsize=9, markerscale=1.8)
    ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(
        "Semantic geometry of figurative language (Jakobson hypothesis)\n"
        "Embeddings of key expressions via sentence-transformers/all-MiniLM-L6-v2",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    log.info("Plot saved to %s", output_path)
    plt.close(fig)


def run(input_path: str, output_path: str, method: str, model_name: str, seed: int, n: int | None) -> None:
    df = pd.read_parquet(input_path)
    log.info("Loaded %d rows", len(df))

    df["type"] = df["type"].str.upper().str.strip()

    if n is not None:
        # Stratified sample: take proportionally from each class, balanced toward METAPHOR/METONYMY
        rng = np.random.default_rng(seed)
        parts = []
        classes = df["type"].unique()
        per_class = max(1, n // len(classes))
        for cls in classes:
            sub = df[df["type"] == cls]
            k = min(per_class, len(sub))
            parts.append(sub.sample(k, random_state=int(rng.integers(1e6))))
        df = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
        log.info("Sampled %d rows (target n=%d, stratified by class)", len(df), n)

    # Use expression if non-empty, otherwise fall back to sentence
    texts = [
        expr.strip() if isinstance(expr, str) and expr.strip() else sent
        for expr, sent in zip(df["expression"], df["sentence"])
    ]
    log.info("  %d expressions, %d sentence fallbacks",
             sum(1 for e in df["expression"] if isinstance(e, str) and e.strip()),
             sum(1 for e in df["expression"] if not (isinstance(e, str) and e.strip())))

    vecs = embed(texts, model_name)

    if method == "umap":
        coords = reduce_umap(vecs, seed)
    else:
        coords = reduce_tsne(vecs, seed)

    # --- Spread statistics ---
    labels = df["type"].values
    stats = spread_stats(vecs, labels)

    print("\n=== Within-class spread (mean pairwise cosine distance) ===")
    print("  Higher = more spread across semantic space")
    for cls in ["METAPHOR", "METONYMY", "LITERAL"]:
        if cls in stats:
            print(f"  {cls:10s}: {stats[cls]:.4f}")

    metaphor_spread = stats.get("METAPHOR", 0)
    metonymy_spread = stats.get("METONYMY", 0)
    print("\n  Jakobson prediction: METAPHOR spread > METONYMY spread")
    if metaphor_spread > metonymy_spread:
        diff = metaphor_spread - metonymy_spread
        print(f"  Result: CONFIRMED — METAPHOR is {diff:.4f} more spread than METONYMY")
    else:
        diff = metonymy_spread - metaphor_spread
        print(f"  Result: NOT CONFIRMED — METONYMY is {diff:.4f} more spread than METAPHOR")

    # --- Centroid distance between METAPHOR and METONYMY ---
    m_mask = labels == "METAPHOR"
    n_mask = labels == "METONYMY"
    if m_mask.sum() > 0 and n_mask.sum() > 0:
        centroid_m = vecs[m_mask].mean(axis=0)
        centroid_n = vecs[n_mask].mean(axis=0)
        cosine_sim = float(centroid_m @ centroid_n)
        print(f"\n  Centroid cosine similarity METAPHOR↔METONYMY: {cosine_sim:.4f}")
        print(f"  (0 = orthogonal, 1 = identical — lower means more distinct regions)")

    plot(df, coords, output_path, method)


def main():
    parser = argparse.ArgumentParser(
        description="Embed figurative expressions and visualise METAPHOR/METONYMY/LITERAL geometry"
    )
    parser.add_argument("--input", required=True, help="Parquet from classifier.py")
    parser.add_argument("--output", default="geometry.png", help="Output PNG path")
    parser.add_argument("--method", choices=["umap", "tsne"], default="umap",
                        help="Dimensionality reduction method (default: umap)")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Sentence-transformers model name")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=None,
                        help="Randomly sample N rows (stratified by class) before embedding")
    args = parser.parse_args()
    run(args.input, args.output, args.method, args.model, args.seed, args.n)


if __name__ == "__main__":
    main()
