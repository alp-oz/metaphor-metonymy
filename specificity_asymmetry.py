"""
specificity_asymmetry.py — Word-frequency specificity asymmetry across pair types.

Hypothesis: metonymic pairs link a specific term to a general one
(producer→product, place→institution), so |spec(src) − spec(tgt)| is larger
than for metaphoric pairs, where both terms are drawn from semantically
distant but comparably specific domains.

Uses wordfreq for frequency estimates; no API calls, no embeddings.

Usage:
    python3 specificity_asymmetry.py
    python3 specificity_asymmetry.py --output specificity_plot.png
"""

import argparse
import logging
import math
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu
from wordfreq import word_frequency

# Import pair dictionaries from pairs_geometry.py
from pairs_geometry import DEAD_METONYMY, LIVE_METONYMY, DEAD_METAPHOR, LIVE_METAPHOR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

LANG_MAP = {
    "English": "en",
    "French":  "fr",
    "Turkish": "tr",
    "Russian": "ru",
    "Swedish": "sv",
}

CATEGORIES = {
    "Dead metonymy": DEAD_METONYMY,
    "Live metonymy": LIVE_METONYMY,
    "Dead metaphor": DEAD_METAPHOR,
    "Live metaphor": LIVE_METAPHOR,
}

CAT_COLORS = {
    "Dead metonymy": "#1a5fa8",
    "Live metonymy": "#6aaee0",
    "Dead metaphor": "#c94a1a",
    "Live metaphor": "#f0a080",
}

METONYMY_CATS = {"Dead metonymy", "Live metonymy"}
METAPHOR_CATS = {"Dead metaphor", "Live metaphor"}


def tokenize(phrase: str) -> set[str]:
    return {tok.strip(".,!?\"'()-").lower() for tok in phrase.split() if tok.strip(".,!?\"'()-")}


def rarest_unique_token(phrase: str, other_phrase: str, lang_code: str) -> float:
    """
    Specificity of the most specific token in `phrase` that does NOT appear
    in `other_phrase`. This isolates the figurative/literal term that actually
    differs between the two halves of a pair.

    Fallback chain:
      1. rarest token unique to this phrase
      2. rarest token in this phrase (if all tokens are shared)
      3. floor 1e-9
    """
    my_tokens = tokenize(phrase)
    other_tokens = tokenize(other_phrase)
    unique = my_tokens - other_tokens
    candidates = unique if unique else my_tokens

    best_freq = None
    for tok in candidates:
        freq = word_frequency(tok, lang_code)
        if freq > 0 and (best_freq is None or freq < best_freq):
            best_freq = freq

    if best_freq is None or best_freq == 0:
        best_freq = 1e-9
    return -math.log(best_freq)


def pair_specificity(pair: tuple[str, str], lang_code: str) -> tuple[float, float, float]:
    """Return (spec_src, spec_tgt, asymmetry) using tokens unique to each side."""
    s1 = rarest_unique_token(pair[0], pair[1], lang_code)
    s2 = rarest_unique_token(pair[1], pair[0], lang_code)
    return s1, s2, abs(s1 - s2)


def run(output_path: str) -> None:
    LANGUAGES = list(DEAD_METONYMY.keys())

    # Collect rows: {cat, lang, src, tgt, spec_src, spec_tgt, asymmetry}
    rows = []
    for cat, cat_data in CATEGORIES.items():
        for lang in LANGUAGES:
            lang_code = LANG_MAP[lang]
            for pair in cat_data[lang]:
                s1, s2, asym = pair_specificity(pair, lang_code)
                rows.append({
                    "cat":      cat,
                    "lang":     lang,
                    "src":      pair[0],
                    "tgt":      pair[1],
                    "spec_src": s1,
                    "spec_tgt": s2,
                    "asym":     asym,
                })

    # --- Per-pair table ---
    print(f"\n{'='*90}")
    print(f"  {'Lang':<10} {'Category':<18} {'Source':<30} {'Target':<30} {'Sp(src)':>7} {'Sp(tgt)':>7} {'|Δ|':>6}")
    print(f"  {'-'*88}")
    for r in rows:
        print(f"  {r['lang']:<10} {r['cat']:<18} {r['src'][:28]:<30} {r['tgt'][:28]:<30} "
              f"{r['spec_src']:>7.3f} {r['spec_tgt']:>7.3f} {r['asym']:>6.3f}")

    # --- Category summary ---
    cat_asyms = {cat: [] for cat in CATEGORIES}
    for r in rows:
        cat_asyms[r["cat"]].append(r["asym"])

    print(f"\n{'='*60}")
    print(f"  Category summary (combined across all languages)")
    print(f"  {'Category':<18} {'Mean |Δ|':>9} {'SD':>7} {'N':>4}")
    print(f"  {'-'*42}")
    for cat in CATEGORIES:
        arr = np.array(cat_asyms[cat])
        print(f"  {cat:<18} {arr.mean():>9.4f} {arr.std():>7.4f} {len(arr):>4}")

    # --- Metonymy vs Metaphor (2-way) ---
    met_asym = [r["asym"] for r in rows if r["cat"] in METONYMY_CATS]
    meta_asym = [r["asym"] for r in rows if r["cat"] in METAPHOR_CATS]
    u2, p2 = mannwhitneyu(met_asym, meta_asym, alternative="two-sided")
    sig2 = "***" if p2 < 0.001 else "**" if p2 < 0.01 else "*" if p2 < 0.05 else "n.s."
    print(f"\n  2-way: METONYMY (mean={np.mean(met_asym):.4f}) vs "
          f"METAPHOR (mean={np.mean(meta_asym):.4f})")
    print(f"  Mann-Whitney U={u2:.0f}, p={p2:.4f} {sig2}  n={len(met_asym)} vs {len(meta_asym)}")

    # --- All pairwise Mann-Whitney ---
    cat_names = list(CATEGORIES.keys())
    print(f"\n{'='*60}")
    print("  Pairwise Mann-Whitney U (combined across languages)")
    print(f"  {'Comparison':<40} {'U':>6} {'p':>8} {'sig':>5}")
    print(f"  {'-'*62}")
    for ca, cb in combinations(cat_names, 2):
        a_arr = np.array(cat_asyms[ca])
        b_arr = np.array(cat_asyms[cb])
        u, p = mannwhitneyu(a_arr, b_arr, alternative="two-sided")
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        label = f"{ca}  vs  {cb}"
        print(f"  {label:<40} {u:>6.0f} {p:>8.4f} {sig:>5}")

    # --- Per-language breakdown ---
    print(f"\n{'='*60}")
    print("  Per-language mean asymmetry")
    header = f"  {'Language':<12}" + "".join(f"  {c[:14]:>14}" for c in cat_names)
    print(header)
    print(f"  {'-'*70}")
    for lang in LANGUAGES:
        row_str = f"  {lang:<12}"
        for cat in cat_names:
            vals = [r["asym"] for r in rows if r["cat"] == cat and r["lang"] == lang]
            row_str += f"  {np.mean(vals):>14.4f}"
        print(row_str)

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Left: per-language grouped bar chart
    ax = axes[0]
    lang_labels = LANGUAGES + ["COMBINED"]
    n_cats = len(cat_names)
    w = 0.18
    offsets = np.linspace(-(n_cats - 1) * w / 2, (n_cats - 1) * w / 2, n_cats)
    x = np.arange(len(lang_labels))

    for ci, cat in enumerate(cat_names):
        means, sds = [], []
        for lang in LANGUAGES:
            vals = [r["asym"] for r in rows if r["cat"] == cat and r["lang"] == lang]
            means.append(np.mean(vals))
            sds.append(np.std(vals))
        means.append(np.mean(cat_asyms[cat]))
        sds.append(np.std(cat_asyms[cat]))
        ax.bar(x + offsets[ci], means, w, yerr=sds, capsize=3,
               color=CAT_COLORS[cat], alpha=0.85, label=cat)

    ax.set_xticks(x)
    ax.set_xticklabels(lang_labels, fontsize=9)
    ax.set_ylabel("Mean |Δ specificity|  (higher = more asymmetric)", fontsize=10)
    ax.set_title("Specificity asymmetry by language and category", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    # Right: violin per category (combined)
    ax = axes[1]
    data   = [np.array(cat_asyms[cat]) for cat in cat_names]
    colors = [CAT_COLORS[cat] for cat in cat_names]
    parts  = ax.violinplot(data, positions=range(n_cats), showmedians=True, showextrema=True)
    for pc, col in zip(parts["bodies"], colors):
        pc.set_facecolor(col)
        pc.set_alpha(0.7)
    parts["cmedians"].set_color("black")
    parts["cbars"].set_color("black")
    parts["cmaxes"].set_color("black")
    parts["cmins"].set_color("black")

    for i, (arr, col) in enumerate(zip(data, colors)):
        ax.scatter(np.random.default_rng(i).normal(i, 0.04, size=len(arr)), arr,
                   color=col, alpha=0.4, s=14, zorder=3)
    for i, arr in enumerate(data):
        ax.text(i, arr.mean() + 0.1, f"{arr.mean():.3f}",
                ha="center", fontsize=8, fontweight="bold")

    ax.set_xticks(range(n_cats))
    ax.set_xticklabels([c.replace(" ", "\n") for c in cat_names], fontsize=9)
    ax.set_ylabel("|Δ specificity|", fontsize=11)
    ax.set_title("Distribution across all languages (combined)", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.suptitle(
        "Specificity asymmetry (wordfreq −log freq)\n"
        "Prediction: metonymy > metaphor  |  5 languages, 4 categories",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    log.info("Plot saved to %s", output_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Specificity asymmetry test across four figurative-language categories"
    )
    parser.add_argument("--output", default="specificity_plot.png")
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
