"""
Covariate enrichment for the extension analyses.

Adds to the pairs long table:
  log_freq1, log_freq2   — log10 of wordfreq unigram frequency (multilingual)
  log_freq_mean          — mean of the two, proxy for pair familiarity
  length1, length2       — character counts of term1, term2
  length_gap             — |len(term1) - len(term2)|, confound for embedding geometry
  concreteness1, concreteness2  — Brysbaert (2014) norms, English pairs only
  concreteness_gap       — |conc1 - conc2|

Brysbaert norms are English-only; rows for other languages get NaN in those columns.

Usage:
    python3 src/covariates.py
    python3 src/covariates.py --norms path/to/Concreteness_ratings_Brysbaert.csv
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from wordfreq import word_frequency

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

CACHE = Path(__file__).parent.parent / "data" / "cache"
DF_PATH = CACHE / "pairs_long.parquet"
NEIGHBORS_PATH = CACHE / "pairs_neighbors.parquet"
OUT_PATH = CACHE / "pairs_covariates.parquet"

# wordfreq language codes matching our LANGUAGES list
LANG_CODE = {
    "English":  "en",
    "French":   "fr",
    "Turkish":  "tr",
    "Russian":  "ru",
    "Swedish":  "sv",
    "German":   "de",
    "Arabic":   "ar",
    "Japanese": "ja",
}

# Brysbaert (2014) column names vary by file version; we try common ones
_BRYSBAERT_WORD_COLS = ["Word", "word", "WORD"]
_BRYSBAERT_CONC_COLS = ["Conc.M", "conc.M", "ConcM", "Mean", "Rating.Mean"]


def load_brysbaert(path: Path) -> dict[str, float]:
    """Returns {word_lower: concreteness_rating}."""
    df = pd.read_csv(path)
    word_col = next((c for c in _BRYSBAERT_WORD_COLS if c in df.columns), None)
    conc_col = next((c for c in _BRYSBAERT_CONC_COLS if c in df.columns), None)
    if word_col is None or conc_col is None:
        raise ValueError(
            f"Cannot find word/concreteness columns in {path}.\n"
            f"Available columns: {list(df.columns)}"
        )
    log.info("Loaded Brysbaert norms from %s (%d words).", path, len(df))
    return dict(zip(df[word_col].str.lower(), df[conc_col].astype(float)))


def get_freq(term: str, lang_code: str) -> float:
    """Log10 frequency; returns -6 (i.e. < 1-per-million) when unknown or unsupported."""
    try:
        f = word_frequency(term.lower(), lang_code)
        return np.log10(f) if f > 0 else -6.0
    except Exception:
        return -6.0


def add_covariates(df: pd.DataFrame, brysbaert: dict | None) -> pd.DataFrame:
    df = df.copy()

    log.info("Computing frequency covariates ...")
    df["log_freq1"] = [
        get_freq(t, LANG_CODE[lang]) for t, lang in zip(df.term1, df.language)
    ]
    df["log_freq2"] = [
        get_freq(t, LANG_CODE[lang]) for t, lang in zip(df.term2, df.language)
    ]
    df["log_freq_mean"] = (df.log_freq1 + df.log_freq2) / 2

    log.info("Computing length covariates ...")
    df["length1"] = df.term1.str.len()
    df["length2"] = df.term2.str.len()
    df["length_gap"] = (df.length1 - df.length2).abs()

    if brysbaert is not None:
        log.info("Computing concreteness covariates (English only) ...")
        english_mask = df.language == "English"
        df["concreteness1"] = np.nan
        df["concreteness2"] = np.nan
        df.loc[english_mask, "concreteness1"] = df.loc[english_mask, "term1"].str.lower().map(brysbaert)
        df.loc[english_mask, "concreteness2"] = df.loc[english_mask, "term2"].str.lower().map(brysbaert)
        df["concreteness_gap"] = (df.concreteness1 - df.concreteness2).abs()
        n_found = english_mask.sum() - df.loc[english_mask, "concreteness1"].isna().sum()
        log.info("Concreteness: %d/%d English pairs have both terms in norms.", n_found, english_mask.sum())
    else:
        log.info("No Brysbaert norms provided; skipping concreteness columns.")
        df["concreteness1"] = np.nan
        df["concreteness2"] = np.nan
        df["concreteness_gap"] = np.nan

    return df


def summarise(df: pd.DataFrame) -> None:
    print("\n=== Covariate means by [lifecycle × structural_type] ===")
    cols = ["log_freq_mean", "length_gap"]
    if df.concreteness_gap.notna().any():
        cols.append("concreteness_gap")
    print(df.groupby(["lifecycle", "structural_type"])[cols].mean().round(3))

    print("\n=== Covariate means by structural_type ===")
    print(df.groupby("structural_type")[cols].mean().round(3))


def run(norms_path: Path | None = None) -> pd.DataFrame:
    # Prefer neighbor-enriched table if available, fall back to base table
    source = NEIGHBORS_PATH if NEIGHBORS_PATH.exists() else DF_PATH
    if not source.exists():
        raise FileNotFoundError("Run src/dataset.py (and optionally src/neighbors.py) first.")
    df = pd.read_parquet(source)
    log.info("Loaded %d pairs from %s.", len(df), source)

    brysbaert = load_brysbaert(norms_path) if norms_path else None
    df = add_covariates(df, brysbaert)
    summarise(df)

    df.to_parquet(OUT_PATH, index=False)
    log.info("Saved enriched table to %s", OUT_PATH)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--norms",
        type=Path,
        default=None,
        help="Path to Brysbaert (2014) concreteness CSV. Optional.",
    )
    run(ap.parse_args().norms)
