"""
analyzer.py — Compute metaphor/metonymy ratio per decade and test Jakobson's
hypothesis: 19th-century realist prose is metonymic; 20th-century modernist
prose shifts toward metaphor.

Input:  Parquet from classifier.py (columns: decade, sentence, expression, type, reason)
Output: PNG plot + printed statistical summary

Usage:
    python3 analyzer.py --input results.parquet
    python3 analyzer.py --input results.parquet --output ratio_plot.png
"""

import argparse
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, kendalltau

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Decade order
DECADE_ORDER = [
    "1850s","1860s","1870s","1880s","1890s",
    "1900s","1910s","1920s","1930s","1940s","1950s",
]

# The Jakobson split: realist 19th century vs modernist 20th century
C19 = {"1850s","1860s","1870s","1880s","1890s"}
C20 = {"1900s","1910s","1920s","1930s","1940s","1950s"}


def compute_ratio(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["type"] = df["type"].str.upper().str.strip()

    grouped = (
        df.groupby("decade")["type"]
        .value_counts()
        .unstack(fill_value=0)
        .reset_index()
    )
    for col in ["METAPHOR", "METONYMY", "LITERAL"]:
        if col not in grouped.columns:
            grouped[col] = 0

    grouped["n_figurative"] = grouped["METAPHOR"] + grouped["METONYMY"]
    grouped["ratio"] = grouped["METAPHOR"] / grouped["n_figurative"].replace(0, np.nan)
    grouped["metaphor_pct"] = 100 * grouped["METAPHOR"] / (
        grouped["METAPHOR"] + grouped["METONYMY"] + grouped["LITERAL"]
    ).replace(0, np.nan)

    # Sort by canonical decade order
    cat = pd.CategoricalDtype(categories=DECADE_ORDER, ordered=True)
    grouped["decade"] = grouped["decade"].astype(cat)
    grouped = grouped.sort_values("decade").reset_index(drop=True)
    return grouped


def statistical_tests(ratio_df: pd.DataFrame) -> dict:
    valid = ratio_df.dropna(subset=["ratio"])

    c19_vals = valid.loc[valid["decade"].isin(C19), "ratio"].values
    c20_vals = valid.loc[valid["decade"].isin(C20), "ratio"].values

    results = {}

    # Mann-Whitney: 19th vs 20th century
    if len(c19_vals) >= 2 and len(c20_vals) >= 2:
        u, p = mannwhitneyu(c19_vals, c20_vals, alternative="two-sided")
        results["mw_U"] = u
        results["mw_p"] = p
        results["c19_mean"] = float(np.mean(c19_vals))
        results["c20_mean"] = float(np.mean(c20_vals))

    # Kendall tau: monotone trend over time
    x = np.arange(len(valid))
    tau, p_tau = kendalltau(x, valid["ratio"].values)
    results["kendall_tau"] = tau
    results["kendall_p"] = p_tau

    return results


def plot(ratio_df: pd.DataFrame, output_path: str) -> None:
    valid = ratio_df.dropna(subset=["ratio"])

    fig, ax = plt.subplots(figsize=(13, 6))

    # Shade 19th vs 20th century background
    decades = list(valid["decade"].astype(str))
    switch_idx = None
    for i, d in enumerate(decades):
        if d in C20:
            switch_idx = i
            break

    if switch_idx is not None:
        # 19th century
        ax.axvspan(-0.5, switch_idx - 0.5, alpha=0.06, color="steelblue", label="19th century")
        # 20th century
        ax.axvspan(switch_idx - 0.5, len(decades) - 0.5, alpha=0.06, color="darkorange", label="20th century")

    # Plot ratio line
    xs = np.arange(len(valid))
    ys = valid["ratio"].values

    ax.plot(xs, ys, "o-", color="midnightblue", linewidth=2.2, markersize=8,
            label="Metaphor / (Metaphor + Metonymy)")
    ax.fill_between(xs, ys, alpha=0.12, color="midnightblue")

    # Trend line (linear fit)
    if len(xs) >= 3:
        z = np.polyfit(xs, ys, 1)
        p_line = np.poly1d(z)
        ax.plot(xs, p_line(xs), "--", color="crimson", linewidth=1.4,
                alpha=0.7, label=f"Linear trend (slope {z[0]:+.3f}/decade)")

    # Annotate n_figurative per decade
    for x, row in zip(xs, valid.itertuples()):
        ax.annotate(
            f"n={row.n_figurative}",
            xy=(x, row.ratio),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center", fontsize=7, color="gray",
        )

    ax.set_xticks(xs)
    ax.set_xticklabels(decades, fontsize=10)
    ax.set_ylabel("Metaphor ratio  [METAPHOR / (METAPHOR + METONYMY)]", fontsize=11)
    ax.set_xlabel("Decade", fontsize=11)
    ax.set_title(
        "Metaphor vs. Metonymy Ratio across Literary Decades 1850–1950\n"
        "Testing Jakobson's hypothesis: realist prose is metonymic, modernist is metaphoric",
        fontsize=12,
    )
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    # Century boundary label
    if switch_idx is not None:
        ax.axvline(switch_idx - 0.5, color="gray", linewidth=1, linestyle=":")
        ax.text(switch_idx - 0.5, 0.97, "1900 →", ha="left", va="top",
                fontsize=8, color="gray", style="italic")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    log.info("Plot saved to %s", output_path)
    plt.close(fig)


def run(input_path: str, output_path: str) -> None:
    df = pd.read_parquet(input_path)
    log.info("Loaded %d rows from %s", len(df), input_path)

    if "type" not in df.columns:
        raise ValueError("Parquet must have a 'type' column.")

    ratio_df = compute_ratio(df)

    print("\n=== Per-Decade Summary ===")
    print(ratio_df[["decade","METAPHOR","METONYMY","LITERAL","n_figurative","ratio"]].to_string(index=False))

    stats = statistical_tests(ratio_df)
    print("\n=== Statistical Tests ===")
    if "mw_U" in stats:
        direction = "higher" if stats["c20_mean"] > stats["c19_mean"] else "lower"
        print(f"  19th-century mean ratio : {stats['c19_mean']:.3f}")
        print(f"  20th-century mean ratio : {stats['c20_mean']:.3f}")
        print(f"  Mann-Whitney U={stats['mw_U']:.0f}, p={stats['mw_p']:.4f}")
        sig = "significant" if stats["mw_p"] < 0.05 else "not significant"
        print(f"  -> {sig}: metaphor ratio is {direction} in the 20th century.")
    if "kendall_tau" in stats:
        print(f"\n  Kendall τ = {stats['kendall_tau']:.3f}, p = {stats['kendall_p']:.4f}")
        direction = "increasing" if stats["kendall_tau"] > 0 else "decreasing"
        sig = "significant" if stats["kendall_p"] < 0.05 else "not significant"
        print(f"  -> {sig} {direction} trend over time.")

    print(f"\n  Jakobson prediction: metaphor ratio should INCREASE from 1850s to 1950s.")
    if "kendall_tau" in stats:
        supported = stats["kendall_tau"] > 0 and stats["kendall_p"] < 0.05
        print(f"  Result: {'SUPPORTED' if supported else 'NOT SUPPORTED (or inconclusive)'}")

    plot(ratio_df, output_path)


def main():
    parser = argparse.ArgumentParser(description="Analyze metaphor/metonymy ratio over literary decades")
    parser.add_argument("--input", required=True, help="Parquet file from classifier.py")
    parser.add_argument("--output", default="ratio_plot.png", help="Output PNG path")
    args = parser.parse_args()
    run(args.input, args.output)


if __name__ == "__main__":
    main()
