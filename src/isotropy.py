"""
Anisotropy analysis and isotropisation robustness check.

Sentence-transformer embeddings are anisotropic: vectors cluster in a
narrow cone, so random pairs have cosine >> 0. This inflates all our
similarity scores. Here we check whether the type/lifecycle ordering
survives three corrections:

  1. Mean-centring: subtract the mean vector
  2. All-but-the-top (Mu & Viswanath 2018): also remove projection onto
     top-d principal directions (d chosen by % variance explained)
  3. Whitening: full covariance normalisation → isotropic unit sphere

For each corrected space we recompute pair cosines and report the
category means and regression coefficients. If the type effect survives
whitening, it is robust to anisotropy; if it shrinks or disappears,
the original result was partly an anisotropy artifact.

Usage:
    python3 src/isotropy.py
    python3 src/isotropy.py --d 5 --plot
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import statsmodels.formula.api as smf

from src.dataset import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

CACHE = Path(__file__).parent.parent / "data" / "cache"
FIGURES = Path(__file__).parent.parent / "figures"


def mean_centre(matrix: np.ndarray) -> np.ndarray:
    return matrix - matrix.mean(axis=0)


def all_but_top(matrix: np.ndarray, d: int) -> np.ndarray:
    """Remove projection onto top-d principal components."""
    pca = PCA(n_components=d)
    pca.fit(matrix)
    projection = matrix @ pca.components_.T @ pca.components_
    corrected = matrix - projection
    var_explained = pca.explained_variance_ratio_.sum()
    log.info("All-but-top-%d removes %.1f%% of variance.", d, 100 * var_explained)
    return corrected


def whiten(matrix: np.ndarray) -> np.ndarray:
    """ZCA whitening: maps covariance to identity."""
    cov = np.cov(matrix.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 1e-10)   # numerical stability
    W = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    return matrix @ W


def pair_cosines(matrix: np.ndarray, idx1: np.ndarray, idx2: np.ndarray) -> np.ndarray:
    mat = normalize(matrix)
    v1 = mat[idx1]
    v2 = mat[idx2]
    return np.einsum("ij,ij->i", v1, v2).astype(np.float32)


def anisotropy_score(matrix: np.ndarray, n_sample: int = 1000, seed: int = 42) -> float:
    """
    Average cosine between random pairs.
    Isotropic space → 0. Anisotropic → positive (all vectors near same direction).
    """
    rng = np.random.default_rng(seed)
    mat = normalize(matrix)
    n = len(mat)
    i = rng.integers(0, n, n_sample)
    j = rng.integers(0, n, n_sample)
    mask = i != j
    return float(np.einsum("ij,ij->i", mat[i[mask]], mat[j[mask]]).mean())


def run_regression(df: pd.DataFrame, sim_col: str = "sim") -> dict:
    """Returns key coefficients from OLS (fast, ICC≈0 so equivalent to mixed)."""
    d = df.copy()
    d["is_metonymy"] = (d.structural_type == "metonymy").astype(float)
    d["is_live"]     = (d.lifecycle == "live").astype(float)
    for col in ["log_freq_mean", "length_gap"]:
        if col in d.columns:
            mu, sd = d[col].mean(), d[col].std()
            d[f"{col}_z"] = (d[col] - mu) / sd

    covs = " + log_freq_mean_z + length_gap_z" if "log_freq_mean_z" in d.columns else ""
    formula = f"{sim_col} ~ is_metonymy + is_live{covs}"
    res = smf.ols(formula, d).fit()
    return {
        "coef_metonymy": res.params.get("is_metonymy", np.nan),
        "p_metonymy":    res.pvalues.get("is_metonymy", np.nan),
        "coef_live":     res.params.get("is_live", np.nan),
        "p_live":        res.pvalues.get("is_live", np.nan),
    }


def summarise_means(df: pd.DataFrame, sim_col: str) -> pd.Series:
    return df.groupby(["lifecycle", "structural_type"])[sim_col].mean().round(3)


def run(d: int = 5, plot: bool = False) -> pd.DataFrame:
    # Load base data
    df_cov_path = CACHE / "pairs_covariates.parquet"
    base_df = pd.read_parquet(df_cov_path) if df_cov_path.exists() else None

    _, terms, matrix, term_to_idx = load_dataset()

    # Reload the pairs table with language/type info
    pairs_df = pd.read_parquet(CACHE / "pairs_long.parquet")
    if base_df is not None:
        # bring in covariates
        pairs_df = base_df[["language", "structural_type", "lifecycle",
                             "term1", "term2", "sim",
                             "log_freq_mean", "length_gap"]].copy()

    idx1 = pairs_df.term1.map(term_to_idx).values
    idx2 = pairs_df.term2.map(term_to_idx).values

    corrections = {
        "original":      matrix,
        "mean_centred":  mean_centre(matrix),
        "all_but_top":   all_but_top(mean_centre(matrix), d=d),
        "whitened":      whiten(mean_centre(matrix)),
    }

    print("\n=== Anisotropy scores (avg cosine of random pairs; 0 = isotropic) ===")
    for name, mat in corrections.items():
        score = anisotropy_score(mat)
        print(f"  {name:20s}  {score:.4f}")

    rows = []
    for name, mat in corrections.items():
        sims = pair_cosines(mat, idx1, idx2)
        df_tmp = pairs_df.copy()
        df_tmp["sim_corrected"] = sims

        means = summarise_means(df_tmp, "sim_corrected")
        reg   = run_regression(df_tmp, sim_col="sim_corrected")

        rows.append({
            "correction":      name,
            "dead_metonymy":   means.get(("dead", "metonymy"),  np.nan),
            "dead_metaphor":   means.get(("dead", "metaphor"),  np.nan),
            "live_metonymy":   means.get(("live", "metonymy"),  np.nan),
            "live_metaphor":   means.get(("live", "metaphor"),  np.nan),
            "coef_metonymy":   reg["coef_metonymy"],
            "p_metonymy":      reg["p_metonymy"],
            "coef_live":       reg["coef_live"],
            "p_live":          reg["p_live"],
        })

    results = pd.DataFrame(rows).set_index("correction")
    pd.set_option("display.float_format", "{:.4f}".format)

    print("\n=== Category means under each correction ===")
    print(results[["dead_metonymy", "dead_metaphor", "live_metonymy", "live_metaphor"]])

    print("\n=== Regression coefficients under each correction ===")
    print(results[["coef_metonymy", "p_metonymy", "coef_live", "p_live"]])

    if plot:
        _plot(results)

    out = CACHE / "isotropy_results.parquet"
    results.reset_index().to_parquet(out, index=False)
    log.info("Saved %s", out)
    return results


def _plot(results: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    FIGURES.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    labels = results.index.tolist()
    x = np.arange(len(labels))
    w = 0.2

    ax = axes[0]
    ax.bar(x - 1.5*w, results.dead_metonymy, w, label="dead metonymy",  color="#1f77b4")
    ax.bar(x - 0.5*w, results.dead_metaphor, w, label="dead metaphor",  color="#d62728")
    ax.bar(x + 0.5*w, results.live_metonymy, w, label="live metonymy",  color="#aec7e8")
    ax.bar(x + 1.5*w, results.live_metaphor, w, label="live metaphor",  color="#ff9896")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Mean cosine similarity")
    ax.set_title("Category means under isotropisation")
    ax.legend(fontsize=8)

    ax2 = axes[1]
    ax2.plot(labels, results.coef_metonymy, "o-", label="metonymy coef", color="#1f77b4")
    ax2.plot(labels, results.coef_live,     "s--", label="live coef",    color="#d62728")
    ax2.axhline(0, color="k", linewidth=0.7, linestyle=":")
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax2.set_ylabel("Regression coefficient")
    ax2.set_title("Type & lifecycle effects under isotropisation")
    ax2.legend(fontsize=9)

    fig.tight_layout()
    out = FIGURES / "isotropy_robustness.png"
    fig.savefig(out, dpi=150)
    log.info("Saved %s", out)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=5,
                    help="Number of top PCs to remove in all-but-top correction.")
    ap.add_argument("--plot", action="store_true")
    run(d=ap.parse_args().d, plot=ap.parse_args().plot)
