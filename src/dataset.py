"""
Consolidated source of truth for the extension analyses.

Builds a tidy long table (one row per pair) and a cached embedding
matrix for every unique term, so downstream modules never re-embed or drift.

Columns: language, structural_type {metonymy,metaphor},
         lifecycle {dead,live}, term1, term2, sim

sim = cosine similarity of independently embedded terms (matches pairs_geometry.py).

Usage:
    python3 src/dataset.py            # build + cache
    python3 src/dataset.py --rebuild  # ignore cache, recompute
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

from data.pairs import (
    DEAD_METONYMY, LIVE_METONYMY, DEAD_METAPHOR, LIVE_METAPHOR, LANGUAGES,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
CACHE = Path(__file__).parent.parent / "data" / "cache"
DF_PATH = CACHE / "pairs_long.parquet"
EMB_PATH = CACHE / "embeddings.npz"

_SPEC = [
    (DEAD_METONYMY,  "metonymy", "dead"),
    (LIVE_METONYMY,  "metonymy", "live"),
    (DEAD_METAPHOR,  "metaphor", "dead"),
    (LIVE_METAPHOR,  "metaphor", "live"),
]


def build_pairs_table() -> pd.DataFrame:
    rows = []
    for cat_dict, structural, lifecycle in _SPEC:
        for lang in LANGUAGES:
            for t1, t2 in cat_dict[lang]:
                rows.append(dict(
                    language=lang,
                    structural_type=structural,
                    lifecycle=lifecycle,
                    term1=t1,
                    term2=t2,
                ))
    return pd.DataFrame(rows)


def embed_terms(terms: list[str], model_name: str = MODEL_NAME) -> np.ndarray:
    log.info("Loading model %s ...", model_name)
    model = SentenceTransformer(model_name)
    log.info("Embedding %d unique terms ...", len(terms))
    mat = model.encode(
        terms, batch_size=64, show_progress_bar=True, convert_to_numpy=True,
    ).astype(np.float32)
    return normalize(mat)  # unit vectors → dot product == cosine


def build(rebuild: bool = False) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    df = build_pairs_table()
    terms = sorted(set(df.term1) | set(df.term2))
    idx = {t: i for i, t in enumerate(terms)}

    if EMB_PATH.exists() and not rebuild:
        log.info("Loading cached embeddings.")
        z = np.load(EMB_PATH, allow_pickle=True)
        cached_terms = list(z["terms"])
        if cached_terms == terms:
            mat = z["matrix"]
        else:
            log.info("Term set changed; re-embedding.")
            mat = embed_terms(terms)
            np.savez(EMB_PATH, terms=np.array(terms, dtype=object), matrix=mat)
    else:
        mat = embed_terms(terms)
        np.savez(EMB_PATH, terms=np.array(terms, dtype=object), matrix=mat)

    v1 = mat[df.term1.map(idx).values]
    v2 = mat[df.term2.map(idx).values]
    df["sim"] = np.einsum("ij,ij->i", v1, v2).astype(np.float32)

    df.to_parquet(DF_PATH, index=False)
    log.info("Wrote %s (%d pairs).", DF_PATH, len(df))
    log.info("Wrote %s (%d terms).", EMB_PATH, len(terms))

    summary = df.groupby(["lifecycle", "structural_type"]).sim.agg(["count", "mean"]).round(3)
    log.info("Sanity check vs README:\n%s", summary)
    return df


def load_dataset():
    """For downstream modules: returns (df, terms, matrix, term_to_idx)."""
    if not (DF_PATH.exists() and EMB_PATH.exists()):
        raise FileNotFoundError("Run `python3 src/dataset.py` first.")
    df = pd.read_parquet(DF_PATH)
    z = np.load(EMB_PATH, allow_pickle=True)
    terms = list(z["terms"])
    return df, terms, z["matrix"], {t: i for i, t in enumerate(terms)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    build(ap.parse_args().rebuild)
