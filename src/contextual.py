"""
Contextual vs bare-term embedding comparison for English pairs.

For each term: encodes it bare (term string alone) and in context
(carrier sentence, pooling only the target term's token span).
Compares pair similarities under both conditions and measures
self-similarity (how much context shifts a term's representation).

Usage:
    python3 src/contextual.py
    python3 src/contextual.py --plot
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CACHE      = Path(__file__).parent.parent / "data" / "cache"
FIGURES    = Path(__file__).parent.parent / "figures"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model():
    log.info("Loading tokenizer and model from %s ...", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model


# ---------------------------------------------------------------------------
# Token-span matching
# ---------------------------------------------------------------------------

def find_span(term_ids: list[int], sentence_ids: list[int]) -> tuple[int, int]:
    """
    Find the first contiguous occurrence of term_ids inside sentence_ids.
    Returns (start, end) as half-open slice indices into sentence_ids.
    Raises ValueError if not found.
    """
    n, m = len(sentence_ids), len(term_ids)
    for i in range(n - m + 1):
        if sentence_ids[i:i + m] == term_ids:
            return i, i + m
    raise ValueError(f"Term token ids {term_ids} not found in sentence token ids {sentence_ids}")


def term_token_ids(tokenizer, term: str) -> list[int]:
    """
    Token ids for a term encoded in isolation, stripping special tokens.
    """
    enc = tokenizer(term, add_special_tokens=False)
    return enc["input_ids"]


def find_span_robust(
    tokenizer, term: str, sentence: str
) -> tuple[list[int], list[int], int, int]:
    """
    Tries to find the term's token span inside the sentence token ids,
    using several fallback strategies for capitalisation and inflection:

    1. Exact match (term as-is, sentence as-is).
    2. Lowercase sentence — catches terms that start the sentence.
    3. Lowercase both — catches both capitalisation directions.

    Returns (term_ids, sentence_ids, start, end) for the first strategy
    that succeeds. Raises ValueError if all strategies fail.
    """
    strategies = [
        (term,            sentence),
        (term,            sentence[0].lower() + sentence[1:]),
        (term.lower(),    sentence.lower()),
    ]
    for t_str, s_str in strategies:
        t_ids = tokenizer(t_str, add_special_tokens=False)["input_ids"]
        s_ids = tokenizer(s_str, add_special_tokens=True)["input_ids"]
        try:
            start, end = find_span(t_ids, s_ids)
            # Return original (non-lowercased) sentence ids for hidden state lookup
            orig_s_ids = tokenizer(sentence, add_special_tokens=True)["input_ids"]
            return t_ids, orig_s_ids, start, end
        except ValueError:
            continue

    # All strategies failed — raise informative error
    t_ids = tokenizer(term, add_special_tokens=False)["input_ids"]
    s_ids = tokenizer(sentence, add_special_tokens=True)["input_ids"]
    raise ValueError(
        f"\n  Term:     {term!r}\n"
        f"  Sentence: {sentence!r}\n"
        f"  Term token ids:     {t_ids}\n"
        f"  Sentence token ids: {s_ids}\n"
        f"  → Term span not found after all fallback strategies.\n"
        f"  The term may be inflected in the carrier sentence. "
        f"Fix the carrier so the term appears verbatim."
    )


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_bare(tokenizer, model, term: str) -> np.ndarray:
    """Mean-pool all token embeddings for a term encoded alone."""
    enc = tokenizer(term, return_tensors="pt", add_special_tokens=True)
    out = model(**enc)
    # last_hidden_state: (1, seq_len, hidden)
    hidden = out.last_hidden_state[0]  # (seq_len, hidden)
    # exclude special tokens: use attention_mask
    mask = enc["attention_mask"][0].bool()
    vec = hidden[mask].mean(dim=0).numpy()
    return vec / (np.linalg.norm(vec) + 1e-10)


@torch.no_grad()
def encode_contextual(
    tokenizer, model, term: str, sentence: str
) -> np.ndarray:
    """
    Mean-pool only the token positions of `term` within `sentence`.
    Uses robust span matching to handle capitalisation differences.
    Raises ValueError with a clear message if span matching fails.
    """
    _, _, start, end = find_span_robust(tokenizer, term, sentence)

    enc    = tokenizer(sentence, return_tensors="pt", add_special_tokens=True)
    out    = model(**enc)
    hidden = out.last_hidden_state[0]  # (seq_len, hidden)
    vec    = hidden[start:end].mean(dim=0).numpy()
    return vec / (np.linalg.norm(vec) + 1e-10)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

def compute(tokenizer, model, df: pd.DataFrame, carriers: dict) -> pd.DataFrame:
    """
    For each pair in df, compute bare_sim, ctx_sim, delta_sim,
    self_sim_t1, self_sim_t2.
    """
    # Pre-compute bare and contextual vectors for every unique term
    all_terms = sorted(set(df.term1) | set(df.term2))
    log.info("Encoding %d unique terms (bare) ...", len(all_terms))
    bare_vecs: dict[str, np.ndarray] = {}
    for t in all_terms:
        bare_vecs[t] = encode_bare(tokenizer, model, t)

    log.info("Encoding %d unique terms (contextual) ...", len(all_terms))
    ctx_vecs: dict[str, np.ndarray] = {}
    for t in all_terms:
        sentence = carriers.get(t)
        if sentence is None:
            raise KeyError(
                f"No carrier sentence for term {t!r}. "
                f"Add it to data/carriers/english_carriers.py."
            )
        ctx_vecs[t] = encode_contextual(tokenizer, model, t, sentence)

    rows = []
    for _, row in df.iterrows():
        t1, t2 = row.term1, row.term2
        bare_sim    = cosine(bare_vecs[t1], bare_vecs[t2])
        ctx_sim     = cosine(ctx_vecs[t1],  ctx_vecs[t2])
        self_sim_t1 = cosine(bare_vecs[t1], ctx_vecs[t1])
        self_sim_t2 = cosine(bare_vecs[t2], ctx_vecs[t2])
        rows.append({
            "term1":          t1,
            "term2":          t2,
            "structural_type": row.structural_type,
            "lifecycle":      row.lifecycle,
            "bare_sim":       bare_sim,
            "ctx_sim":        ctx_sim,
            "delta_sim":      ctx_sim - bare_sim,
            "self_sim_t1":    self_sim_t1,
            "self_sim_t2":    self_sim_t2,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summary and figures
# ---------------------------------------------------------------------------

def print_summary(results: pd.DataFrame) -> None:
    print("\n=== Contextual embedding summary (English pairs) ===")
    grp = results.groupby(["lifecycle", "structural_type"])
    summary = grp.agg(
        n=("bare_sim", "count"),
        bare_sim_mean=("bare_sim", "mean"),
        ctx_sim_mean=("ctx_sim", "mean"),
        delta_mean=("delta_sim", "mean"),
        self_sim_mean=("self_sim_t1", "mean"),
    ).round(3)
    print(summary.to_string())

    print("\nExpected bare_sim_mean (full-dataset): "
          "dead metonymy ~0.82, dead metaphor ~0.69, "
          "live metonymy ~0.57, live metaphor ~0.46")


def plot(results: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    FIGURES.mkdir(exist_ok=True)

    cats = [
        ("dead",  "metonymy",  "#1f77b4", "Dead metonymy"),
        ("dead",  "metaphor",  "#d62728", "Dead metaphor"),
        ("live",  "metonymy",  "#aec7e8", "Live metonymy"),
        ("live",  "metaphor",  "#ff9896", "Live metaphor"),
    ]

    # --- Figure 1: bare vs contextual scatter per category ---
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=True, sharex=True)
    for ax, (lc, stype, color, label) in zip(axes, cats):
        sub = results[(results.lifecycle == lc) & (results.structural_type == stype)]
        ax.scatter(sub.bare_sim, sub.ctx_sim, c=color, alpha=0.7, s=30, edgecolors="k", linewidths=0.3)
        lo = min(sub.bare_sim.min(), sub.ctx_sim.min()) - 0.05
        hi = max(sub.bare_sim.max(), sub.ctx_sim.max()) + 0.05
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, label="y = x")
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("bare sim", fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("contextual sim", fontsize=9)
        ax.set_aspect("equal")

    fig.suptitle("Bare vs contextual pair similarity (English)", fontsize=12, y=1.01)
    fig.tight_layout()
    out1 = FIGURES / "contextual_shift.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    log.info("Saved %s", out1)
    plt.close(fig)

    # --- Figure 2: self-similarity boxplot ---
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    data_t1 = [
        results[(results.lifecycle == lc) & (results.structural_type == st)].self_sim_t1.values
        for lc, st, _, _ in cats
    ]
    data_t2 = [
        results[(results.lifecycle == lc) & (results.structural_type == st)].self_sim_t2.values
        for lc, st, _, _ in cats
    ]
    labels  = [label for _, _, _, label in cats]
    colors  = [c for _, _, c, _ in cats]
    x = np.arange(len(cats))
    w = 0.3
    bp1 = ax2.boxplot(data_t1, positions=x - w/2, widths=w, patch_artist=True,
                      medianprops=dict(color="k"))
    bp2 = ax2.boxplot(data_t2, positions=x + w/2, widths=w, patch_artist=True,
                      medianprops=dict(color="k", linestyle="--"))
    for patch, color in zip(bp1["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    for patch, color in zip(bp2["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.4)

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("Self-similarity (bare vs contextual)", fontsize=10)
    ax2.set_title("Stability of term representations under contextualisation", fontsize=11)
    ax2.set_ylim(0, 1.05)
    from matplotlib.patches import Patch
    ax2.legend(
        handles=[Patch(facecolor="grey", alpha=0.8, label="term1"),
                 Patch(facecolor="grey", alpha=0.4, label="term2")],
        fontsize=9,
    )
    fig2.tight_layout()
    out2 = FIGURES / "self_similarity.png"
    fig2.savefig(out2, dpi=150)
    log.info("Saved %s", out2)
    plt.close(fig2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(make_plot: bool = False) -> pd.DataFrame:
    # Load English pairs
    cov_path = CACHE / "pairs_covariates.parquet"
    if not cov_path.exists():
        raise FileNotFoundError("Run src/covariates.py first.")
    df = pd.read_parquet(cov_path)
    df = df[df.language == "English"].copy()
    log.info("Loaded %d English pairs.", len(df))

    # Load carriers
    from data.carriers.english_carriers import (
        DEAD_METONYMY, LIVE_METONYMY, DEAD_METAPHOR, LIVE_METAPHOR,
    )
    carriers: dict[str, str] = {
        **DEAD_METONYMY, **LIVE_METONYMY, **DEAD_METAPHOR, **LIVE_METAPHOR,
    }

    tokenizer, model = load_model()
    results = compute(tokenizer, model, df, carriers)

    print_summary(results)

    out = CACHE / "contextual_results.parquet"
    results.to_parquet(out, index=False)
    log.info("Saved %s", out)

    if make_plot:
        plot(results)

    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    run(ap.parse_args().plot)
