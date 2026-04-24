"""
sampler.py — Extract and filter sentences from COHA decade files.

Expects COHA text files organised as one file per decade, named like:
    1810s.txt, 1820s.txt, ... or 1810.txt, 1820.txt, ...
or a directory structure where each decade folder contains .txt files.

Filters for sentences containing candidate figurative expressions using:
  1. Metonymy triggers: place/institution names used as actors, container words,
     producer-for-product patterns.
  2. Metaphor triggers: word list derived from the VU Amsterdam Metaphor Corpus
     (built-in subset; swap for full list via --metaphor-list).

Output: CSV with columns: decade, sentence

Usage:
    python sampler.py --coha-dir /data/coha --output sampled.csv --n 200
    python sampler.py --coha-dir /data/coha --output sampled.csv --n 200 \
        --metaphor-list vu_amsterdam_words.txt
"""

import argparse
import csv
import logging
import random
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Trigger word lists
# ---------------------------------------------------------------------------

# Metonymy triggers: place names / institutions used as agents
METONYMY_PLACE_NAMES = {
    "washington", "the white house", "the kremlin", "brussels", "beijing",
    "london", "moscow", "paris", "berlin", "wall street", "hollywood",
    "silicon valley", "the pentagon", "downing street", "the senate",
    "congress", "the house", "the court", "the bench",
}

# Container/producer-product metonymy triggers
METONYMY_CONTAINER = {
    "bottle", "glass", "cup", "barrel", "barrel", "can", "tin", "jug",
    "kettle", "pot", "flask",
}

# Producer-for-product (brand / author names used as products)
METONYMY_PRODUCER = {
    "ford", "chevy", "rolls", "boeing", "hemingway", "dickens", "shakespeare",
    "picasso", "rembrandt", "levis", "coke", "pepsi",
}

ALL_METONYMY = METONYMY_PLACE_NAMES | METONYMY_CONTAINER | METONYMY_PRODUCER

# Core VU Amsterdam metaphor-related words (high-precision subset).
# Replace or augment via --metaphor-list for the full list.
VU_AMSTERDAM_SUBSET = {
    "attack", "battle", "battle", "beat", "bury", "capture", "carry", "cast",
    "catch", "chase", "choke", "clash", "climb", "collapse", "colour",
    "crash", "crush", "cut", "damage", "dark", "dead", "decay", "deep",
    "destroy", "devour", "dig", "drown", "erode", "explode", "fall", "fight",
    "flood", "flow", "freeze", "fuel", "grasp", "grip", "ground", "grow",
    "hammer", "harvest", "heal", "heavy", "hit", "ignite", "illuminate",
    "kill", "launch", "leap", "light", "lose", "melt", "mirror", "mount",
    "navigate", "overshadow", "paint", "pierce", "plant", "poison", "pull",
    "push", "race", "rain", "rise", "root", "run", "shadow", "sharp",
    "shatter", "shed", "shield", "shoot", "sink", "slide", "slip", "spark",
    "squeeze", "stem", "stir", "strike", "struggle", "sweep", "target",
    "tear", "throw", "track", "trap", "twist", "undermine", "wave", "weigh",
    "wrestle",
}


def load_metaphor_list(path: str) -> set:
    words = set()
    with open(path) as f:
        for line in f:
            w = line.strip().lower()
            if w:
                words.add(w)
    log.info("Loaded %d metaphor trigger words from %s", len(words), path)
    return words


def is_candidate(sentence: str, metaphor_words: set) -> bool:
    """Return True if the sentence contains at least one figurative trigger."""
    lower = sentence.lower()
    tokens = set(re.findall(r"\b\w+\b", lower))

    # Metonymy: substring match for multi-word triggers
    for trigger in ALL_METONYMY:
        if trigger in lower:
            return True

    # Metaphor: token match
    if tokens & metaphor_words:
        return True

    return False


def sentence_split(text: str):
    """Rough sentence splitter; good enough for COHA prose."""
    # Split on . ! ? followed by whitespace and a capital letter
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [p.strip() for p in parts if len(p.strip()) > 20]


def collect_sentences_from_file(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        log.warning("Could not read %s: %s", path, e)
        return []
    return sentence_split(text)


def decade_label(name: str) -> str:
    """Extract decade string like '1860s' from a filename or directory name."""
    m = re.search(r"(1[89]\d0)", name)
    if m:
        return m.group(1) + "s"
    m = re.search(r"(1[89]\d\d)", name)
    if m:
        year = int(m.group(1))
        return f"{(year // 10) * 10}s"
    return name


def discover_files(coha_dir: Path) -> dict[str, list[Path]]:
    """
    Map decade label → list of .txt files.
    Handles flat layout (1810s.txt) and directory layout (1810s/*.txt).
    """
    mapping: dict[str, list[Path]] = {}

    txt_files = list(coha_dir.glob("*.txt"))
    if txt_files:
        for f in txt_files:
            d = decade_label(f.stem)
            mapping.setdefault(d, []).append(f)
    else:
        # Try subdirectory layout
        for sub in sorted(coha_dir.iterdir()):
            if sub.is_dir():
                d = decade_label(sub.name)
                txts = list(sub.glob("**/*.txt"))
                if txts:
                    mapping.setdefault(d, []).extend(txts)

    return mapping


def sample_decade(
    files: list[Path],
    decade: str,
    n: int,
    metaphor_words: set,
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)
    all_candidates: list[str] = []

    for f in files:
        sentences = collect_sentences_from_file(f)
        candidates = [s for s in sentences if is_candidate(s, metaphor_words)]
        all_candidates.extend(candidates)

    if not all_candidates:
        log.warning("Decade %s: no candidates found in %d file(s).", decade, len(files))
        return []

    chosen = rng.sample(all_candidates, min(n, len(all_candidates)))
    log.info("Decade %s: %d candidates → sampled %d", decade, len(all_candidates), len(chosen))
    return [{"decade": decade, "sentence": s} for s in chosen]


def run(coha_dir: str, output: str, n: int, metaphor_list: str | None, seed: int) -> None:
    coha_path = Path(coha_dir)
    if not coha_path.is_dir():
        raise FileNotFoundError(f"COHA directory not found: {coha_dir}")

    metaphor_words = load_metaphor_list(metaphor_list) if metaphor_list else VU_AMSTERDAM_SUBSET

    decade_files = discover_files(coha_path)
    if not decade_files:
        raise RuntimeError(f"No .txt files found under {coha_dir}")

    log.info("Found %d decades: %s", len(decade_files), sorted(decade_files))

    rows: list[dict] = []
    for decade in sorted(decade_files):
        rows.extend(sample_decade(decade_files[decade], decade, n, metaphor_words, seed))

    import pandas as pd
    df = pd.DataFrame(rows, columns=["decade", "sentence"])
    df.to_csv(output, index=False)
    log.info("Saved %d rows to %s", len(df), output)
    log.info("Decade counts:\n%s", df["decade"].value_counts().sort_index().to_string())


def main():
    parser = argparse.ArgumentParser(description="Sample candidate figurative sentences from COHA")
    parser.add_argument("--coha-dir", required=True, help="Root directory of COHA .txt files")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--n", type=int, default=200, help="Sentences per decade (default: 200)")
    parser.add_argument(
        "--metaphor-list",
        default=None,
        help="Path to plain-text file of metaphor trigger words (one per line). "
        "Defaults to built-in VU Amsterdam subset.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()
    run(args.coha_dir, args.output, args.n, args.metaphor_list, args.seed)


if __name__ == "__main__":
    main()
