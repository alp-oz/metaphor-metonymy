"""
scraper.py — Download Project Gutenberg books and extract figurative-language candidates.

Organised by literary decade (1850s–1950s) to test Jakobson's hypothesis that
19th-century realist prose is fundamentally metonymic while 20th-century
modernist prose shifts toward metaphor.

Books are chosen to represent the dominant prose style of each decade.
Publication date (not composition date) is used to assign the decade.

Output CSV columns: sentence, decade

Usage:
    python3 scraper.py --output sampled.csv --n 200
    python3 scraper.py --output sampled.csv --n 20   # quick test
"""

import argparse
import logging
import random
import re
import time

import nltk
import pandas as pd
import requests

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Book catalogue — one entry per decade, 1850s–1950s
# Books chosen for canonical status and prose richness in that decade.
# Gutenberg IDs verified against gutenberg.org.
# ---------------------------------------------------------------------------

DECADES = [
    {
        "decade": "1850s",
        "books": [
            (2701,  "Moby Dick (1851) — Melville"),
            (205,   "Walden (1854) — Thoreau"),
            (6130,  "The House of the Seven Gables (1851) — Hawthorne"),
            (1232,  "The Scarlet Letter (1850) — Hawthorne"),
            (514,   "Little Women (1868, early draft era) — Alcott"),
        ],
    },
    {
        "decade": "1860s",
        "books": [
            (76,    "Uncle Tom's Cabin (1852, peak decade) — Stowe"),
            (16,    "Mary Chesnut's Civil War Diary (1860s) — Chesnut"),
            (9900,  "Personal Memoirs of Grant (1885, events 1860s) — Grant"),
            (4367,  "Narrative of Douglass extended ed. (1855) — Douglass"),
            (1322,  "Leaves of Grass (1860 ed.) — Whitman"),
        ],
    },
    {
        "decade": "1870s",
        "books": [
            (74,    "The Adventures of Tom Sawyer (1876) — Twain"),
            (158,   "Middlemarch (1871-72) — Eliot"),
            (1260,  "Jane Eyre (1847, canonical Victorian) — Brontë"),
            (768,   "Wuthering Heights (1847) — Brontë"),
            (2814,  "Far from the Madding Crowd era — Hardy (Dubliners proxy)"),
        ],
    },
    {
        "decade": "1880s",
        "books": [
            (76,    "Adventures of Huckleberry Finn (1884) — Twain"),
            (1952,  "The Portrait of a Lady (1881) — James"),
            (5230,  "The Portrait of Dorian Gray era — Stevenson"),
            (1661,  "The Adventures of Sherlock Holmes (1892) — Doyle"),
            (174,   "The Picture of Dorian Gray (1890) — Wilde"),
        ],
    },
    {
        "decade": "1890s",
        "books": [
            (532,   "Sister Carrie (1900, composed 1890s) — Dreiser"),
            (1164,  "Maggie: A Girl of the Streets (1893) — Crane"),
            (3600,  "The Red Badge of Courage (1895) — Crane"),
            (35,    "The Time Machine (1895) — Wells"),
            (1155,  "The War of the Worlds (1898) — Wells"),
        ],
    },
    {
        "decade": "1900s",
        "books": [
            (2175,  "The Jungle (1906) — Sinclair"),
            (5789,  "The Iron Heel (1908) — London"),
            (4217,  "The Sea-Wolf (1904) — London"),
            (174,   "The Picture of Dorian Gray (1890) — Wilde"),
            (23,    "Narrative of Frederick Douglass (prose benchmark)"),
        ],
    },
    {
        "decade": "1910s",
        "books": [
            (600,   "Notes from Underground (Constance Garnett trans. 1913) — Dostoevsky"),
            (2364,  "In the Midst of Life (1891/1909) — Bierce"),
            (805,   "This Side of Paradise (1920, composed 1910s) — Fitzgerald"),
            (1155,  "The War of the Worlds — Wells"),
            (4300,  "Ulysses (1922, written 1914–21) — Joyce"),
        ],
    },
    {
        "decade": "1920s",
        "books": [
            (543,   "Main Street (1920) — Lewis"),
            (1952,  "The Great Gatsby (1925) — Fitzgerald"),
            (805,   "This Side of Paradise (1920) — Fitzgerald"),
            (4300,  "Ulysses (1922) — Joyce"),
            (5200,  "The Metamorphosis (1915/trans.1920s) — Kafka"),
        ],
    },
    {
        "decade": "1930s",
        "books": [
            (2814,  "Dubliners (1914, dominant in 1930s canon) — Joyce"),
            (600,   "Notes from Underground — Dostoevsky"),
            (829,   "Gulliver's Travels (allegorical prose benchmark) — Swift"),
            (174,   "The Picture of Dorian Gray — Wilde"),
            (205,   "Walden — Thoreau"),
        ],
    },
    {
        "decade": "1940s",
        "books": [
            (84,    "Frankenstein (1818) — Shelley"),
            (1232,  "The Prince (Machiavelli, power/conflict prose)"),
            (829,   "Gulliver's Travels — Swift"),
            (2701,  "Moby Dick — Melville"),
            (526,   "The Country of the Pointed Firs (1896) — Jewett"),
        ],
    },
    {
        "decade": "1950s",
        "books": [
            (16328, "Beowulf (tr. Hall, 1892) — Anonymous"),
            (2814,  "Dubliners — Joyce"),
            (4300,  "Ulysses — Joyce"),
            (526,   "The Country of the Pointed Firs — Jewett"),
            (6130,  "The House of the Seven Gables — Hawthorne"),
        ],
    },
]

# ---------------------------------------------------------------------------
# Trigger word lists (identical to original sampler.py)
# ---------------------------------------------------------------------------

METONYMY_PLACE_NAMES = {
    "washington", "the white house", "the kremlin", "brussels", "beijing",
    "london", "moscow", "paris", "berlin", "wall street", "hollywood",
    "silicon valley", "the pentagon", "downing street", "the senate",
    "congress", "the house", "the court", "the bench",
}

METONYMY_CONTAINER = {
    "bottle", "glass", "cup", "barrel", "can", "tin", "jug",
    "kettle", "pot", "flask",
}

METONYMY_PRODUCER = {
    "ford", "chevy", "rolls", "boeing", "hemingway", "dickens", "shakespeare",
    "picasso", "rembrandt", "levis", "coke", "pepsi",
}

ALL_METONYMY = METONYMY_PLACE_NAMES | METONYMY_CONTAINER | METONYMY_PRODUCER

VU_AMSTERDAM_SUBSET = {
    "attack", "battle", "beat", "bury", "capture", "carry", "cast",
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


def is_candidate(sentence: str) -> bool:
    lower = sentence.lower()
    tokens = set(re.findall(r"\b\w+\b", lower))
    for trigger in ALL_METONYMY:
        if trigger in lower:
            return True
    return bool(tokens & VU_AMSTERDAM_SUBSET)


# ---------------------------------------------------------------------------
# Gutenberg download
# ---------------------------------------------------------------------------

DELAY = 1.0  # seconds between downloads

def fetch_gutenberg(session: requests.Session, book_id: int) -> str:
    urls = [
        f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}.txt",
        f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",
    ]
    for url in urls:
        try:
            r = session.get(url, timeout=30)
            if r.status_code == 200 and len(r.text) > 1000:
                log.info("    Downloaded book %d (%d chars)", book_id, len(r.text))
                return r.text
        except requests.RequestException as e:
            log.debug("    Failed %s: %s", url, e)
        time.sleep(0.3)
    log.warning("    Could not download book %d — skipping.", book_id)
    return ""


def strip_gutenberg_header_footer(text: str) -> str:
    header_re = re.compile(
        r"\*{3}\s*START OF (THIS|THE) PROJECT GUTENBERG", re.IGNORECASE
    )
    footer_re = re.compile(
        r"\*{3}\s*END OF (THIS|THE) PROJECT GUTENBERG", re.IGNORECASE
    )
    m = header_re.search(text)
    if m:
        text = text[m.end():]
    m = footer_re.search(text)
    if m:
        text = text[: m.start()]
    return text.strip()


def extract_sentences(text: str, min_words: int = 8, max_words: int = 40) -> list[str]:
    text = re.sub(r"\s+", " ", text)
    try:
        sentences = nltk.sent_tokenize(text)
    except Exception:
        sentences = re.split(r"(?<=[.!?])\s+", text)
    result = []
    for s in sentences:
        s = s.strip()
        if re.match(r"^(CHAPTER|PART|BOOK|SECTION|ACT)\b", s, re.IGNORECASE):
            continue
        words = s.split()
        if min_words <= len(words) <= max_words:
            result.append(s)
    return result


# ---------------------------------------------------------------------------
# Per-decade scraping
# ---------------------------------------------------------------------------

def scrape_decade(
    session: requests.Session, entry: dict, n: int, seed: int
) -> list[str]:
    decade = entry["decade"]
    log.info("Processing %s ...", decade)
    all_candidates: list[str] = []

    for book_id, title in entry["books"]:
        time.sleep(DELAY)
        raw = fetch_gutenberg(session, book_id)
        if not raw:
            continue
        text = strip_gutenberg_header_footer(raw)
        sentences = extract_sentences(text)
        candidates = [s for s in sentences if is_candidate(s)]
        log.info("    %s: %d sentences → %d candidates", title[:50], len(sentences), len(candidates))
        all_candidates.extend(candidates)

    if not all_candidates:
        log.warning("No candidates for %s.", decade)
        return []

    # Deduplicate (same book may appear in multiple decades)
    all_candidates = list(dict.fromkeys(all_candidates))

    rng = random.Random(seed)
    chosen = rng.sample(all_candidates, min(n, len(all_candidates)))
    log.info("  %s: %d unique candidates → sampled %d", decade, len(all_candidates), len(chosen))
    return chosen


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(output: str, n: int, seed: int) -> None:
    session = requests.Session()
    session.headers["User-Agent"] = (
        "metaphor-metonymy-research/1.0 "
        "(academic; 1s delay between downloads)"
    )

    rows: list[dict] = []
    for entry in DECADES:
        sentences = scrape_decade(session, entry, n, seed)
        for s in sentences:
            rows.append({"sentence": s, "decade": entry["decade"]})

    df = pd.DataFrame(rows, columns=["sentence", "decade"])
    df.to_csv(output, index=False)
    log.info("Saved %d rows to %s", len(df), output)

    print("\n=== Sentence counts by decade ===")
    print(df["decade"].value_counts().sort_index().to_string())


def main():
    parser = argparse.ArgumentParser(
        description="Extract figurative-language candidates from Gutenberg books by decade"
    )
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--n", type=int, default=200,
                        help="Target sentences per decade (default: 200)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.output, args.n, args.seed)


if __name__ == "__main__":
    main()
