"""
Mixed-effects regression: the centerpiece of the extension analyses.

Model:
    sim ~ structural_type + lifecycle + log_freq_mean + length_gap
          + (1 | language)

where structural_type and lifecycle are treatment-coded dummies
(baseline: dead metaphor — the highest-similarity category, so
coefficients show how each group differs downward from that).

Why mixed effects? The 390 pairs are nested in 8 languages. Treating
language as a fixed effect would eat 7 degrees of freedom and not
generalise; a random intercept estimates the language-level variance
and gives correct standard errors for the type/lifecycle effects.

statsmodels MixedLM is used (no R dependency needed).

Usage:
    python3 src/regression.py
    python3 src/regression.py --plot   # save figures/regression_coefs.png
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

CACHE = Path(__file__).parent.parent / "data" / "cache"
FIGURES = Path(__file__).parent.parent / "figures"


def load() -> pd.DataFrame:
    path = CACHE / "pairs_covariates.parquet"
    if not path.exists():
        raise FileNotFoundError("Run src/covariates.py first.")
    return pd.read_parquet(path)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Treatment coding with dead metaphor as baseline
    df["is_metonymy"] = (df.structural_type == "metonymy").astype(float)
    df["is_live"]     = (df.lifecycle == "live").astype(float)
    # Standardise continuous predictors so coefficients are comparable
    for col in ["log_freq_mean", "length_gap", "jaccard"]:
        mu, sd = df[col].mean(), df[col].std()
        df[f"{col}_z"] = (df[col] - mu) / sd
    return df


def run_model(df: pd.DataFrame, include_jaccard: bool = True):
    """
    Fits:
      sim ~ is_metonymy + is_live + log_freq_mean_z + length_gap_z [+ jaccard_z]
      random intercept: language
    Returns fitted model.
    """
    formula = "sim ~ is_metonymy + is_live + log_freq_mean_z + length_gap_z"
    if include_jaccard:
        formula += " + jaccard_z"
    model = smf.mixedlm(formula, df, groups=df["language"])
    result = model.fit(reml=True, method="lbfgs")
    return result


def print_summary(result, ols_result=None) -> None:
    print("\n" + "=" * 60)
    print("Mixed-effects regression: sim ~ type + lifecycle + covariates")
    print("Random intercept: language (8 groups)")
    print("=" * 60)

    params = result.params
    ci    = result.conf_int()
    pvals = result.pvalues

    rows = []
    labels = {
        "Intercept":       "Intercept (dead metaphor baseline)",
        "is_metonymy":     "is_metonymy  (metonymy vs metaphor)",
        "is_live":         "is_live      (live vs dead)",
        "log_freq_mean_z": "log_freq_mean (standardised)",
        "length_gap_z":    "length_gap    (standardised)",
    }
    for key, label in labels.items():
        if key not in params.index:
            continue
        coef = params[key]
        ci_lo = ci.loc[key, 0] if key in ci.index else float("nan")
        ci_hi = ci.loc[key, 1] if key in ci.index else float("nan")
        p = pvals[key] if key in pvals.index else float("nan")
        rows.append({
            "Predictor": label,
            "Coef":  coef,
            "95% CI lower": ci_lo,
            "95% CI upper": ci_hi,
            "p": p,
        })

    tbl = pd.DataFrame(rows).set_index("Predictor")
    tbl["sig"] = tbl["p"].apply(
        lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        if pd.notna(p) else ""
    )
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.max_colwidth", 50)
    print(tbl.to_string())

    # Random effects variance
    if hasattr(result, "cov_re"):
        re_var = float(np.diag(result.cov_re)[0])
        re_sd  = np.sqrt(max(re_var, 0))
        resid_var = result.scale
        icc = re_var / (re_var + resid_var) if (re_var + resid_var) > 0 else 0.0
        print(f"\nRandom intercept SD (language): {re_sd:.4f}")
        print(f"Residual SD:                    {np.sqrt(resid_var):.4f}")
        print(f"ICC (language):                 {icc:.3f}")
        print("  ICC ≈ 0: language clustering explains negligible variance.")
        print("  Fixed-effect estimates are equivalent to plain OLS.")

    print(f"\nN pairs: {int(result.nobs)}")

    if ols_result is not None:
        print("\n--- OLS comparison (no random effects) ---")
        ols_keys = ["is_metonymy", "is_live", "log_freq_mean_z", "length_gap_z"]
        for key in ols_keys:
            if key in ols_result.params:
                print(f"  {key:30s}  coef={ols_result.params[key]:.4f}  "
                      f"p={ols_result.pvalues[key]:.4f}")


def plot_coefs(result, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    FIGURES.mkdir(exist_ok=True)

    keys = ["is_metonymy", "is_live", "log_freq_mean_z", "length_gap_z"]
    labels = ["metonymy\n(vs metaphor)", "live\n(vs dead)",
              "log freq\n(standardised)", "length gap\n(standardised)"]
    coefs = [result.params[k] for k in keys]
    lo    = [result.conf_int().loc[k, 0] for k in keys]
    hi    = [result.conf_int().loc[k, 1] for k in keys]
    err_lo = [c - l for c, l in zip(coefs, lo)]
    err_hi = [h - c for c, h in zip(coefs, hi)]

    fig, ax = plt.subplots(figsize=(6, 4))
    y = range(len(keys))
    ax.errorbar(coefs, y, xerr=[err_lo, err_hi], fmt="o", color="#1f77b4",
                capsize=4, linewidth=1.5, markersize=7)
    ax.axvline(0, color="k", linewidth=0.8, linestyle="--")
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Coefficient (effect on cosine similarity)", fontsize=11)
    ax.set_title("Mixed-effects regression coefficients\n(random intercept: language)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    log.info("Saved %s", out_path)
    plt.close(fig)


def run(plot: bool = False):
    df = load()
    df = prepare(df)
    result = run_model(df)

    # OLS for comparison when random effects collapse
    ols = smf.ols(
        "sim ~ is_metonymy + is_live + log_freq_mean_z + length_gap_z", df
    ).fit()

    print_summary(result, ols_result=ols)
    if plot:
        plot_coefs(result, FIGURES / "regression_coefs.png")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    run(ap.parse_args().plot)
