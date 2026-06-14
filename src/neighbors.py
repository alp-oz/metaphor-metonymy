"""
k-NN geometry for the extension analyses.

For each pair (term1, term2) computes:
  s1  — first-order cosine similarity (already in dataset, repeated here for convenience)
  s2  — second-order similarity: Jaccard of top-k neighbor sets against the full vocabulary
  asym — rank asymmetry: rank(term2 in term1's NN list) - rank(term1 in term2's NN list)
         positive = term2 is a closer neighbor of term1 than vice versa

Also places Goossens (1990) metaphtonymy mixed cases on the (s1, s2) plane.

Usage:
    python3 src/neighbors.py          # uses cached embeddings, k=20
    python3 src/neighbors.py --k 50
    python3 src/neighbors.py --plot   # save figures/neighbors_plane.png
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.dataset import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

FIGURES = Path(__file__).parent.parent / "figures"

# Goossens (1990) metaphtonymy mixed cases — (term1, term2, label)
# These are embedded on the fly using the cached matrix if terms exist,
# or re-encoded if not (rare; they are common English words).
METAPHTONYMY_CASES = [
    ("lend",     "ear",        "metaphtonymy"),   # lend an ear
    ("shoot",    "mouth",      "metaphtonymy"),   # shoot one's mouth off
    ("keep",     "eye",        "metaphtonymy"),   # keep an eye on
    ("lose",     "head",       "metaphtonymy"),   # lose one's head
    ("put",      "foot",       "metaphtonymy"),   # put one's foot in it
    ("bite",     "tongue",     "metaphtonymy"),   # bite one's tongue
]


def build_knn(matrix: np.ndarray, k: int) -> np.ndarray:
    """Returns (N, k) array of neighbor indices, excluding self."""
    # cosine similarity matrix (matrix rows are unit vectors)
    sim = matrix @ matrix.T          # (N, N)
    np.fill_diagonal(sim, -np.inf)   # exclude self
    return np.argsort(sim, axis=1)[:, -k:][:, ::-1]  # (N, k), descending


def jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def compute_neighbor_features(
    df: pd.DataFrame,
    terms: list[str],
    matrix: np.ndarray,
    term_to_idx: dict,
    k: int = 20,
) -> pd.DataFrame:
    log.info("Building k-NN graph (k=%d, %d terms) ...", k, len(terms))
    knn = build_knn(matrix, k)          # (N, k)
    knn_sets = [set(row) for row in knn]

    idx1 = df.term1.map(term_to_idx).values
    idx2 = df.term2.map(term_to_idx).values

    s2 = np.array([
        jaccard(knn_sets[i], knn_sets[j])
        for i, j in zip(idx1, idx2)
    ], dtype=np.float32)

    # rank asymmetry: position of term2 in term1's ranked neighbor list
    # and vice versa; lower rank = closer neighbor
    full_sim = matrix @ matrix.T
    np.fill_diagonal(full_sim, -np.inf)
    ranks = np.argsort(np.argsort(-full_sim, axis=1), axis=1)  # (N, N), rank 0 = closest

    rank_1_of_2 = ranks[idx1, idx2].astype(np.float32)  # rank of term2 in term1's list
    rank_2_of_1 = ranks[idx2, idx1].astype(np.float32)  # rank of term1 in term2's list
    asym = rank_1_of_2 - rank_2_of_1   # positive: term2 closer to term1 than reverse

    df = df.copy()
    df["s2"] = s2
    df["rank_1_of_2"] = rank_1_of_2
    df["rank_2_of_1"] = rank_2_of_1
    df["asym"] = asym
    return df


def embed_extra_terms(new_terms: list[str], terms: list[str], matrix: np.ndarray) -> dict:
    """Returns {term: vector} for terms not already in the cache."""
    missing = [t for t in new_terms if t not in set(terms)]
    result = {t: matrix[terms.index(t)] for t in new_terms if t in set(terms)}
    if missing:
        log.info("Embedding %d metaphtonymy terms not in cache ...", len(missing))
        from sentence_transformers import SentenceTransformer
        from sklearn.preprocessing import normalize as sk_normalize
        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        vecs = sk_normalize(model.encode(missing, convert_to_numpy=True).astype(np.float32))
        result.update(dict(zip(missing, vecs)))
    return result


def compute_metaphtonymy_points(
    cases: list[tuple],
    terms: list[str],
    matrix: np.ndarray,
    knn_sets: list[set],
    term_to_idx: dict,
    k: int,
) -> pd.DataFrame:
    extra_terms = list({t for t1, t2, _ in cases for t in (t1, t2)})
    vecs = embed_extra_terms(extra_terms, terms, matrix)

    rows = []
    for t1, t2, label in cases:
        v1, v2 = vecs[t1], vecs[t2]
        s1 = float(v1 @ v2)

        # second-order: use cached knn_sets if term in vocab, else recompute
        if t1 in term_to_idx and t2 in term_to_idx:
            s2 = jaccard(knn_sets[term_to_idx[t1]], knn_sets[term_to_idx[t2]])
        else:
            # approximate: top-k against full matrix
            sim1 = matrix @ v1
            sim2 = matrix @ v2
            nn1 = set(np.argsort(sim1)[-k:])
            nn2 = set(np.argsort(sim2)[-k:])
            s2 = jaccard(nn1, nn2)

        rows.append(dict(term1=t1, term2=t2, label=label, s1=s1, s2=s2))
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> None:
    print("\n=== Second-order similarity (s2) by category ===")
    print(df.groupby(["lifecycle", "structural_type"])[["sim", "s2"]].mean().round(3))

    print("\n=== Rank asymmetry by structural type ===")
    print(df.groupby("structural_type")[["asym"]].agg(["mean", "median"]).round(1))
    print("  (positive = term2 is a closer neighbor of term1 than vice versa)")


def plot_plane(df: pd.DataFrame, mixed: pd.DataFrame | None = None) -> None:
    import matplotlib.pyplot as plt

    FIGURES.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))

    colors = {
        ("metonymy", "dead"): "#1f77b4",
        ("metonymy", "live"): "#aec7e8",
        ("metaphor", "dead"): "#d62728",
        ("metaphor", "live"): "#ff9896",
    }
    for (stype, lc), grp in df.groupby(["structural_type", "lifecycle"]):
        ax.scatter(grp.sim, grp.s2, c=colors[(stype, lc)], alpha=0.45, s=18,
                   label=f"{lc} {stype}")

    if mixed is not None and len(mixed):
        ax.scatter(mixed.s1, mixed.s2, c="green", marker="*", s=120, zorder=5,
                   label="metaphtonymy")
        for _, row in mixed.iterrows():
            ax.annotate(f"{row.term1}/{row.term2}", (row.s1, row.s2),
                        fontsize=7, xytext=(4, 2), textcoords="offset points")

    ax.set_xlabel("s₁  (first-order cosine)", fontsize=12)
    ax.set_ylabel("s₂  (second-order: neighbour Jaccard)", fontsize=12)
    ax.set_title("Contiguity vs substitutability plane", fontsize=13)
    ax.legend(fontsize=9, markerscale=1.4)
    fig.tight_layout()
    out = FIGURES / "neighbors_plane.png"
    fig.savefig(out, dpi=150)
    log.info("Saved %s", out)
    plt.close(fig)

    # asymmetry strip plot by type
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    for stype, grp in df.groupby("structural_type"):
        ax2.violinplot(grp.asym.dropna(), positions=[0 if stype == "metonymy" else 1],
                       showmedians=True)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["metonymy", "metaphor"])
    ax2.axhline(0, color="k", linewidth=0.8, linestyle="--")
    ax2.set_ylabel("rank asymmetry  (term2 closer → positive)")
    ax2.set_title("Directional asymmetry by structural type")
    fig2.tight_layout()
    out2 = FIGURES / "neighbors_asymmetry.png"
    fig2.savefig(out2, dpi=150)
    log.info("Saved %s", out2)
    plt.close(fig2)


def run(k: int = 20, plot: bool = False) -> pd.DataFrame:
    df, terms, matrix, term_to_idx = load_dataset()
    df = compute_neighbor_features(df, terms, matrix, term_to_idx, k=k)

    knn = build_knn(matrix, k)
    knn_sets = [set(row) for row in knn]
    mixed = compute_metaphtonymy_points(
        METAPHTONYMY_CASES, terms, matrix, knn_sets, term_to_idx, k
    )

    summarise(df)

    print("\n=== Metaphtonymy mixed cases on (s1, s2) plane ===")
    print(mixed[["term1", "term2", "s1", "s2"]].round(3).to_string(index=False))

    if plot:
        plot_plane(df, mixed)

    out = Path(__file__).parent.parent / "data" / "cache" / "pairs_neighbors.parquet"
    df.to_parquet(out, index=False)
    log.info("Saved neighbor features to %s", out)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--plot", action="store_true")
    run(k=ap.parse_args().k, plot=ap.parse_args().plot)
