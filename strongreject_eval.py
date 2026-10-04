"""
StrongREJECT evaluation script.

Reads results.json, evaluates harmfulness of each response type,
and writes evaluation scores to eval_results.json.
"""

import json
import logging
import sys
import os

import numpy as np
from datasets import Dataset
from strong_reject.evaluate import evaluate_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

INPUT_FILE = "resultss.json"
OUTPUT_FILE = "eval_results.json"
EVALUATOR = "strongreject_finetuned"  # no OpenAI key needed; runs on GPU
BATCH_SIZE = 16

RESPONSE_COLUMNS = [
    "response_no_trigger",
    "response_with_trigger",
    "response_nta",
    "response_original_model",
    "response_original_model_triggered",
]


def load_data():
    with open(INPUT_FILE) as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = next(v for v in data.values() if isinstance(v, list))
    log.info(f"Loaded {len(data)} items from {INPUT_FILE}")
    return data


def evaluate_column(data, col):
    hf_dataset = Dataset.from_dict({
        "forbidden_prompt": [row["question"] for row in data],
        "response": [row.get(col) or "" for row in data],
    })
    return evaluate_dataset(hf_dataset, [EVALUATOR], batch_size=BATCH_SIZE).to_dict()


def summarise(scores):
    arr = np.array([s for s in scores if s is not None], dtype=float)
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr)),
            "min": float(np.min(arr)), "max": float(np.max(arr)), "n": int(len(arr))}


def main():
    data = load_data()
    per_item = [{"question": row["question"]} for row in data]
    aggregate = {}

    for col in RESPONSE_COLUMNS:
        if not any(row.get(col) for row in data):
            log.warning(f"'{col}' missing in all rows — skipping.")
            continue

        log.info(f"Evaluating '{col}'...")
        scored = evaluate_column(data, col)

        scores        = scored.get("score", [])
        refusals      = scored.get("refusal", [None] * len(scores))
        specificities = scored.get("specificity", [None] * len(scores))
        convincing    = scored.get("convincingness", [None] * len(scores))

        for i, row in enumerate(per_item):
            row[col] = {
                "score":         scores[i]        if i < len(scores) else None,
                "refusal":       refusals[i]       if i < len(refusals) else None,
                "specificity":   specificities[i]  if i < len(specificities) else None,
                "convincingness": convincing[i]    if i < len(convincing) else None,
            }

        aggregate[col] = summarise(scores)
        log.info(f"  → mean score: {aggregate[col]['mean']:.3f} ± {aggregate[col]['std']:.3f}")

    with open(OUTPUT_FILE, "w") as f:
        json.dump({"evaluator": EVALUATOR, "aggregate": aggregate, "per_item": per_item}, f, indent=2)
    log.info(f"Saved results to {OUTPUT_FILE}")

    print("\n=== Harmfulness Score Summary ===")
    print(f"{'Response Type':<42} {'Mean':>6} {'Std':>6} {'Min':>6} {'Max':>6}")
    print("-" * 70)
    for col, s in aggregate.items():
        print(f"{col:<42} {s['mean']:>6.3f} {s['std']:>6.3f} {s['min']:>6.3f} {s['max']:>6.3f}")


if __name__ == "__main__":
    main()
