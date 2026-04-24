"""
pairs_geometry.py — Four-way cross-lingual test of semantic proximity.

Four categories (predicted similarity order high→low):
  1. DEAD METONYMY   — producer/product, container/contents (near-paraphrases)
  2. LIVE METONYMY   — institutional place-names, instrument-for-person, part-for-whole
  3. DEAD METAPHOR   — conventionalised body-part / structural metaphors
  4. LIVE METAPHOR   — vivid cross-domain mappings

Key comparison: dead metaphor vs live metonymy.
  If dead metaphor > live metonymy → conventionalisation dominates.
  If live metonymy > dead metaphor → contiguity signal dominates.

Uses paraphrase-multilingual-MiniLM-L12-v2 (cached locally).
Pair data is loaded from data/pairs/ (one module per language).

Usage:
    python3 src/pairs_geometry.py
    python3 src/pairs_geometry.py --output results/figures/pairs_plot.png
"""

import argparse
import logging
import sys
import os
from itertools import combinations
from pathlib import Path

# Allow imports from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

from data.pairs import (
    DEAD_METONYMY, LIVE_METONYMY, DEAD_METAPHOR, LIVE_METAPHOR, LANGUAGES
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

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


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b)


def run(output_path: str, model_name: str) -> None:
    log.info("Loading model %s ...", model_name)
    model = SentenceTransformer(model_name)

    all_strings = list({s for cat_data in CATEGORIES.values()
                        for lang_pairs in cat_data.values()
                        for pair in lang_pairs for s in pair})
    log.info("Encoding %d unique strings across %d languages ...",
             len(all_strings), len(LANGUAGES))
    raw_vecs = model.encode(all_strings, batch_size=64, show_progress_bar=True,
                            convert_to_numpy=True)
    vecs = normalize(raw_vecs)
    s2v = {s: vecs[i] for i, s in enumerate(all_strings)}

    cat_lang_sims = {cat: {} for cat in CATEGORIES}
    for cat, cat_data in CATEGORIES.items():
        for lang in LANGUAGES:
            cat_lang_sims[cat][lang] = [
                cosine_sim(s2v[a], s2v[b]) for a, b in cat_data[lang]
            ]

    cat_all_sims = {
        cat: [s for lang in LANGUAGES for s in cat_lang_sims[cat][lang]]
        for cat in CATEGORIES
    }

    # --- Per-pair table ---
    for lang in LANGUAGES:
        print(f"\n{'='*72}")
        print(f"  {lang}")
        print(f"{'='*72}")
        for cat in CATEGORIES:
            print(f"\n  [{cat}]")
            for (a, b), s in zip(CATEGORIES[cat][lang], cat_lang_sims[cat][lang]):
                print(f"    {a[:35]:<36} {b[:35]:<36} {s:>6.4f}")

    # --- Summary table ---
    print(f"\n{'='*72}")
    print("  Category summary (combined across all languages)")
    print(f"  {'Category':<18} {'Mean':>6} {'SD':>6} {'Min':>6} {'Max':>6} {'N':>4}")
    print(f"  {'-'*50}")
    for cat in CATEGORIES:
        arr = np.array(cat_all_sims[cat])
        print(f"  {cat:<18} {arr.mean():>6.4f} {arr.std():>6.4f} "
              f"{arr.min():>6.4f} {arr.max():>6.4f} {len(arr):>4}")

    # --- All pairwise Mann-Whitney tests ---
    cat_names = list(CATEGORIES.keys())
    print(f"\n{'='*72}")
    print("  Pairwise Mann-Whitney U (combined across languages)")
    print(f"  {'Comparison':<40} {'U':>6} {'p':>8} {'sig':>5}")
    print(f"  {'-'*62}")
    for ca, cb in combinations(cat_names, 2):
        a_arr = np.array(cat_all_sims[ca])
        b_arr = np.array(cat_all_sims[cb])
        u, p = mannwhitneyu(a_arr, b_arr, alternative="two-sided")
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        label = f"{ca}  vs  {cb}"
        print(f"  {label:<40} {u:>6.0f} {p:>8.4f} {sig:>5}")

    dm_arr = np.array(cat_all_sims["Dead metaphor"])
    lm_arr = np.array(cat_all_sims["Live metonymy"])
    print(f"\n  KEY: dead metaphor mean={dm_arr.mean():.4f}  "
          f"live metonymy mean={lm_arr.mean():.4f}")
    if dm_arr.mean() > lm_arr.mean():
        print("  → Conventionalisation dominates: dead metaphor > live metonymy")
    else:
        print("  → Contiguity dominates: live metonymy > dead metaphor")

    # --- Per-language table ---
    print(f"\n{'='*72}")
    print("  Per-language means")
    header = f"  {'Language':<12}" + "".join(f"  {c[:14]:>14}" for c in cat_names)
    print(header)
    print(f"  {'-'*70}")
    for lang in LANGUAGES:
        row = f"  {lang:<12}"
        for cat in cat_names:
            row += f"  {np.mean(cat_lang_sims[cat][lang]):>14.4f}"
        print(row)
    combined_row = f"  {'COMBINED':<12}"
    for cat in cat_names:
        combined_row += f"  {np.mean(cat_all_sims[cat]):>14.4f}"
    print(combined_row)

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    ax = axes[0]
    lang_labels = LANGUAGES + ["COMBINED"]
    n_cats = len(cat_names)
    w = 0.18
    offsets = np.linspace(-(n_cats - 1) * w / 2, (n_cats - 1) * w / 2, n_cats)
    x = np.arange(len(lang_labels))

    for ci, cat in enumerate(cat_names):
        means = [np.mean(cat_lang_sims[cat][l]) for l in LANGUAGES] + \
                [np.mean(cat_all_sims[cat])]
        sds   = [np.std(cat_lang_sims[cat][l])  for l in LANGUAGES] + \
                [np.std(cat_all_sims[cat])]
        ax.bar(x + offsets[ci], means, w, yerr=sds, capsize=3,
               color=CAT_COLORS[cat], alpha=0.85, label=cat)

    ax.set_xticks(x)
    ax.set_xticklabels(lang_labels, fontsize=9)
    ax.set_ylabel("Mean cosine similarity (± sd)", fontsize=11)
    ax.set_title("Four-way similarity by language\n(left→right: dead met., live met., dead meta., live meta.)",
                 fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(0, min(1.05, ax.get_ylim()[1] + 0.08))
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    ax = axes[1]
    data   = [np.array(cat_all_sims[cat]) for cat in cat_names]
    colors = [CAT_COLORS[cat] for cat in cat_names]
    parts  = ax.violinplot(data, positions=range(n_cats), showmedians=True,
                           showextrema=True)
    for i, (pc, col) in enumerate(zip(parts["bodies"], colors)):
        pc.set_facecolor(col)
        pc.set_alpha(0.7)
    parts["cmedians"].set_color("black")
    parts["cbars"].set_color("black")
    parts["cmaxes"].set_color("black")
    parts["cmins"].set_color("black")

    for i, (arr, col) in enumerate(zip(data, colors)):
        ax.scatter(np.random.normal(i, 0.04, size=len(arr)), arr,
                   color=col, alpha=0.35, s=12, zorder=3)

    ax.set_xticks(range(n_cats))
    ax.set_xticklabels([c.replace(" ", "\n") for c in cat_names], fontsize=9)
    ax.set_ylabel("Cosine similarity", fontsize=11)
    ax.set_title("Distribution across all languages\n(combined)", fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for i, arr in enumerate(data):
        ax.text(i, arr.mean() + 0.03, f"{arr.mean():.3f}",
                ha="center", fontsize=8, color="black", fontweight="bold")

    fig.suptitle(
        "Semantic proximity: dead metonymy > live metonymy > dead metaphor > live metaphor?\n"
        f"Model: paraphrase-multilingual-MiniLM-L12-v2  |  {len(LANGUAGES)} languages",
        fontsize=11,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
    fig.savefig(output_path, dpi=150)
    log.info("Plot saved to %s", output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Four-way cross-lingual cosine similarity test"
    )
    parser.add_argument("--output", default="results/figures/pairs_plot.png")
    parser.add_argument("--model",
                        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    args = parser.parse_args()
    run(args.output, args.model)


if __name__ == "__main__":
    main()
