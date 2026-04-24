"""
wordnet_check.py — WordNet depth asymmetry across three figurative-language categories.

Prediction (highest → lowest asymmetry):
  1. Live metonymy  — instrument/person, part/whole cross taxonomic levels
  2. Dead metaphor  — body-part / functional equivalent, similar but not identical depth
  3. Live metaphor  — cross-domain pairs chosen for structural parallelism

Uses WordNet max_depth (most common noun sense) for each term.
No embeddings, no API calls.

Usage:
    python3 wordnet_check.py
    python3 wordnet_check.py --output wordnet_plot.png
"""

import argparse
import math
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nltk
import numpy as np
from scipy.stats import mannwhitneyu
from wordfreq import word_frequency

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
from nltk.corpus import wordnet as wn

LIVE_METONYMY = [
    ("press",       "journalist"),
    ("stage",       "theatre"),
    ("bench",       "judge"),
    ("pulpit",      "clergy"),
    ("microphone",  "singer"),
    ("hand",        "sailor"),
    ("robe",        "magistrate"),
    ("pen",         "writer"),
    ("brush",       "painter"),
    ("lens",        "photographer"),
]

DEAD_METAPHOR = [
    ("leg",     "support"),
    ("foot",    "base"),
    ("arm",     "side"),
    ("eye",     "hole"),
    ("mouth",   "opening"),
    ("head",    "leader"),
    ("heart",   "core"),
    ("spine",   "back"),
    ("tongue",  "flap"),
    ("crown",   "top"),
]

LIVE_METAPHOR = [
    ("cheek",     "apple"),
    ("mind",      "machine"),
    ("love",      "fire"),
    ("anger",     "heat"),
    ("idea",      "seed"),
    ("society",   "organism"),
    ("word",      "weapon"),
    ("hope",      "light"),
    ("grief",     "weight"),
    ("knowledge", "food"),
]

CATEGORIES = {
    "Live metonymy": LIVE_METONYMY,
    "Dead metaphor": DEAD_METAPHOR,
    "Live metaphor": LIVE_METAPHOR,
}

CAT_COLORS = {
    "Live metonymy": "#6aaee0",
    "Dead metaphor": "#c94a1a",
    "Live metaphor": "#f0a080",
}


def wordnet_depth(word: str) -> int | None:
    synsets = wn.synsets(word, pos=wn.NOUN)
    if not synsets:
        return None
    return max(s.max_depth() for s in synsets)


def freq_specificity(word: str) -> float:
    freq = word_frequency(word.lower(), "en")
    return -math.log(freq if freq > 0 else 1e-9)


def run(output_path: str) -> None:
    cat_rows = {cat: [] for cat in CATEGORIES}
    cat_asyms = {cat: [] for cat in CATEGORIES}

    # --- Per-pair table ---
    for cat, pairs in CATEGORIES.items():
        print(f"\n{'='*72}")
        print(f"  {cat}")
        print(f"  {'Term 1':<14} {'Term 2':<14} {'WN1':>4} {'WN2':>4} {'WN|Δ|':>6}  "
              f"{'Sp1':>6} {'Sp2':>6} {'Sp|Δ|':>6}")
        print(f"  {'-'*70}")
        for t1, t2 in pairs:
            d1, d2 = wordnet_depth(t1), wordnet_depth(t2)
            s1, s2 = freq_specificity(t1), freq_specificity(t2)
            if d1 is None or d2 is None:
                wn_asym_str = f"{'?':>4} {'?':>4} {'?':>6}"
                wn_asym = None
            else:
                wn_asym = abs(d1 - d2)
                wn_asym_str = f"{d1:>4} {d2:>4} {wn_asym:>6.1f}"
                cat_asyms[cat].append(wn_asym)
                cat_rows[cat].append((t1, t2, d1, d2, wn_asym, s1, s2, abs(s1 - s2)))
            print(f"  {t1:<14} {t2:<14} {wn_asym_str}  {s1:>6.3f} {s2:>6.3f} {abs(s1-s2):>6.3f}")

    # --- Summary table ---
    print(f"\n{'='*60}")
    print(f"  Category summary (WordNet depth asymmetry)")
    print(f"  {'Category':<18} {'Mean WN|Δ|':>10} {'SD':>7} {'N':>4}")
    print(f"  {'-'*42}")
    for cat in CATEGORIES:
        arr = np.array(cat_asyms[cat])
        print(f"  {cat:<18} {arr.mean():>10.4f} {arr.std():>7.4f} {len(arr):>4}")

    # --- Pairwise Mann-Whitney ---
    cat_names = list(CATEGORIES.keys())
    print(f"\n{'='*60}")
    print("  Pairwise Mann-Whitney U (WordNet asymmetry)")
    print(f"  {'Comparison':<38} {'U':>5} {'p':>8} {'sig':>5}")
    print(f"  {'-'*58}")
    for ca, cb in combinations(cat_names, 2):
        a, b = np.array(cat_asyms[ca]), np.array(cat_asyms[cb])
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        print(f"  {ca + '  vs  ' + cb:<38} {u:>5.0f} {p:>8.4f} {sig:>5}")

    # --- Prediction check ---
    means = {cat: np.mean(cat_asyms[cat]) for cat in cat_names}
    print(f"\n  Prediction: live metonymy > dead metaphor > live metaphor")
    print(f"  Observed:   " + " > ".join(
        f"{c} ({means[c]:.3f})" for c in sorted(cat_names, key=lambda c: -means[c])
    ))
    predicted_order = cat_names  # live met, dead meta, live meta
    observed_order  = sorted(cat_names, key=lambda c: -means[c])
    if observed_order == predicted_order:
        print("  → ORDER CONFIRMED")
    else:
        print("  → ORDER NOT CONFIRMED")

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: bar chart with individual points
    ax = axes[0]
    x = np.arange(len(cat_names))
    for i, cat in enumerate(cat_names):
        arr = np.array(cat_asyms[cat])
        ax.bar(i, arr.mean(), color=CAT_COLORS[cat], alpha=0.75, width=0.5, label=cat,
               yerr=arr.std(), capsize=5)
        ax.scatter(np.random.default_rng(i).normal(i, 0.06, size=len(arr)), arr,
                   color=CAT_COLORS[cat], s=40, zorder=3, alpha=0.7, edgecolors="white", lw=0.5)
        ax.text(i, arr.mean() + arr.std() + 0.1, f"{arr.mean():.2f}",
                ha="center", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace(" ", "\n") for c in cat_names], fontsize=10)
    ax.set_ylabel("Mean WordNet depth asymmetry |d₁ − d₂|", fontsize=10)
    ax.set_title("WordNet depth asymmetry by category\n(± SD, with individual pairs)", fontsize=10)
    ax.set_ylim(0, ax.get_ylim()[1] + 0.5)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    # Right: violin
    ax = axes[1]
    data   = [np.array(cat_asyms[cat]) for cat in cat_names]
    colors = [CAT_COLORS[cat] for cat in cat_names]
    parts  = ax.violinplot(data, positions=range(len(cat_names)),
                           showmedians=True, showextrema=True)
    for pc, col in zip(parts["bodies"], colors):
        pc.set_facecolor(col); pc.set_alpha(0.65)
    parts["cmedians"].set_color("black")
    parts["cbars"].set_color("black")
    parts["cmaxes"].set_color("black")
    parts["cmins"].set_color("black")
    for i, (arr, col) in enumerate(zip(data, colors)):
        ax.scatter(np.random.default_rng(i + 10).normal(i, 0.05, size=len(arr)), arr,
                   color=col, alpha=0.5, s=20, zorder=3)

    ax.set_xticks(range(len(cat_names)))
    ax.set_xticklabels([c.replace(" ", "\n") for c in cat_names], fontsize=10)
    ax.set_ylabel("WordNet depth asymmetry |d₁ − d₂|", fontsize=10)
    ax.set_title("Distribution of depth asymmetry", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.suptitle(
        "WordNet depth asymmetry: live metonymy > dead metaphor > live metaphor?\n"
        "English noun pairs only  |  WordNet max_depth (most common sense)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"\n  Plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="WordNet depth asymmetry across three figurative-language categories"
    )
    parser.add_argument("--output", default="wordnet_plot.png")
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
