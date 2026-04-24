"""
recheck_metonymy.py — Re-classify METONYMY rows with a stricter prompt.

Filters rows where type == METONYMY from results.parquet, runs them through
a tighter Jakobson-style prompt that requires one of six explicit contiguity
relations, saves to metonymy_recheck.parquet, and prints 5 examples.

Usage:
    python3 recheck_metonymy.py --input results.parquet --output metonymy_recheck.parquet
"""

import argparse
import json
import time
import logging

import anthropic
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a precise linguistic analyst specialising in metonymy.

Metonymy is ONLY when one entity substitutes for another because of a real-world contiguity relation.
The substitution must fit one of these exact types:

1. PLACE for INSTITUTION: "the White House decided"
2. PRODUCER for PRODUCT: "I read Hemingway"
3. CONTAINER for CONTENTS: "the kettle is boiling"
4. PART for WHOLE: "all hands on deck"
5. CAUSE for EFFECT or EFFECT for CAUSE
6. INSTITUTION for PEOPLE: "the army advanced"

If the expression does not clearly fit one of these six types, classify it as METAPHOR or LITERAL instead.

Respond ONLY with valid JSON:
{
  "expression": "exact phrase",
  "type": "METONYMY or METAPHOR or LITERAL",
  "relation_type": "one of the six types above or NONE",
  "reason": "one sentence citing the contiguity relation"
}"""

SYSTEM_BLOCKS = [
    {
        "type": "text",
        "text": SYSTEM_PROMPT,
        "cache_control": {"type": "ephemeral"},
    }
]

FALLBACK = {"expression": "", "type": "LITERAL", "relation_type": "NONE", "reason": "parse error"}


def classify_sentence(client: anthropic.Anthropic, sentence: str, model: str) -> dict:
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
        log.warning("JSON parse error for %r: %s", sentence[:60], e)
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
    df = pd.read_parquet(input_path)
    metonymy_df = df[df["type"].str.upper().str.strip() == "METONYMY"].copy()
    log.info("Found %d METONYMY rows to recheck", len(metonymy_df))

    client = anthropic.Anthropic()
    results = []
    total = len(metonymy_df)

    for i, (_, row) in enumerate(metonymy_df.iterrows()):
        sentence = str(row["sentence"])
        parsed = classify_sentence(client, sentence, model)
        results.append({
            "decade":        row.get("decade"),
            "sentence":      sentence,
            "expression":    parsed.get("expression", ""),
            "type":          parsed.get("type", "LITERAL"),
            "relation_type": parsed.get("relation_type", "NONE"),
            "reason":        parsed.get("reason", ""),
        })

        if (i + 1) % batch_size == 0 or (i + 1) == total:
            pd.DataFrame(results).to_parquet(output_path, index=False)
            log.info("Progress: %d/%d — checkpoint saved", i + 1, total)

    out = pd.DataFrame(results)
    out.to_parquet(output_path, index=False)

    log.info("Done. Type distribution after recheck:\n%s",
             out["type"].value_counts().to_string())

    print("\n=== 5 sample reclassifications ===")
    for _, row in out.head(5).iterrows():
        print(f"\n  sentence      : {row['sentence'][:90]}")
        print(f"  expression    : {row['expression']}")
        print(f"  type          : {row['type']}")
        print(f"  relation_type : {row['relation_type']}")
        print(f"  reason        : {row['reason']}")


def main():
    parser = argparse.ArgumentParser(description="Recheck METONYMY rows with strict prompt")
    parser.add_argument("--input",  required=True, help="Parquet from classifier.py")
    parser.add_argument("--output", required=True, help="Output parquet for rechecked rows")
    parser.add_argument("--model",  default="claude-haiku-4-5")
    parser.add_argument("--batch-size", type=int, default=50)
    args = parser.parse_args()
    run(args.input, args.output, args.model, args.batch_size)


if __name__ == "__main__":
    main()
