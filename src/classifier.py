"""
classifier.py — Classify sentences as METAPHOR, METONYMY, or LITERAL.

Input:  CSV with at minimum a 'sentence' column and optionally 'decade'.
Output: Parquet with columns: decade, sentence, expression, type, reason.

Usage:
    python classifier.py --input sampled.csv --output results.parquet
    python classifier.py --input sampled.csv --output results.parquet --model claude-haiku-4-5
"""

import argparse
import json
import time
import logging
from pathlib import Path

import anthropic
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a precise linguistic analyst trained in Jakobson's theory of metaphor and metonymy.

Given a sentence, identify the primary figurative expression (if any) and classify it.

Definitions:
- METAPHOR: a cross-domain mapping where one concept is understood in terms of another (e.g. "the stock market collapsed", "he has a sharp mind").
- METONYMY: a within-domain substitution where one entity stands for a related entity (e.g. "Washington announced" meaning the US government, "the White House said", "the bottle is his friend" meaning alcohol).
- LITERAL: no significant figurative usage; the sentence is interpreted at face value.

Respond only with valid JSON:
{"expression": "<the key word or phrase>", "type": "METAPHOR|METONYMY|LITERAL", "reason": "<one sentence explanation>"}

If the sentence contains no candidate figurative expression, set expression to "" and type to "LITERAL"."""

SYSTEM_BLOCKS = [
    {
        "type": "text",
        "text": SYSTEM_PROMPT,
        "cache_control": {"type": "ephemeral"},
    }
]

FALLBACK = {"expression": "", "type": "LITERAL", "reason": "parse error"}


def classify_sentence(client: anthropic.Anthropic, sentence: str, model: str) -> dict:
    """Call the API for one sentence. Returns parsed JSON dict."""
    try:
        response = client.messages.create(
            model=model,
            max_tokens=256,
            system=SYSTEM_BLOCKS,
            messages=[{"role": "user", "content": sentence}],
        )
        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)
    except json.JSONDecodeError as e:
        log.warning("JSON parse error for sentence %r: %s", sentence[:60], e)
        return FALLBACK
    except anthropic.RateLimitError as e:
        retry_after = int(getattr(e.response, "headers", {}).get("retry-after", "60"))
        log.warning("Rate limited. Sleeping %ds.", retry_after)
        time.sleep(retry_after)
        return classify_sentence(client, sentence, model)
    except anthropic.APIStatusError as e:
        log.error("API error %d: %s", e.status_code, e.message)
        return FALLBACK


def run(input_path: str, output_path: str, model: str, batch_size: int) -> None:
    df = pd.read_csv(input_path)

    if "sentence" not in df.columns:
        raise ValueError("Input CSV must have a 'sentence' column.")
    if "decade" not in df.columns:
        df["decade"] = None

    client = anthropic.Anthropic()

    results = []
    total = len(df)
    log.info("Classifying %d sentences with model=%s", total, model)

    for i, row in df.iterrows():
        sentence = str(row["sentence"])
        decade = row.get("decade")

        parsed = classify_sentence(client, sentence, model)

        results.append(
            {
                "decade": decade,
                "sentence": sentence,
                "expression": parsed.get("expression", ""),
                "type": parsed.get("type", "LITERAL"),
                "reason": parsed.get("reason", ""),
            }
        )

        if (i + 1) % batch_size == 0 or (i + 1) == total:
            out = pd.DataFrame(results)
            out.to_parquet(output_path, index=False)
            log.info("Progress: %d/%d — checkpoint saved to %s", i + 1, total, output_path)

    log.info("Done. Final results in %s", output_path)
    log.info("Type distribution:\n%s", pd.DataFrame(results)["type"].value_counts().to_string())


def main():
    parser = argparse.ArgumentParser(description="Classify sentences as METAPHOR/METONYMY/LITERAL")
    parser.add_argument("--input", required=True, help="Input CSV path (must have 'sentence' column)")
    parser.add_argument("--output", required=True, help="Output parquet path")
    parser.add_argument(
        "--model",
        default="claude-haiku-4-5",
        help="Claude model ID (default: claude-haiku-4-5)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Save checkpoint every N sentences (default: 100)",
    )
    args = parser.parse_args()
    run(args.input, args.output, args.model, args.batch_size)


if __name__ == "__main__":
    main()
