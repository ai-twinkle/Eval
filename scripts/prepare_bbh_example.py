"""Download BIG-Bench Hard (BBH) samples from the original GitHub repo and create example dataset.

Source: https://github.com/suzgunmirac/BIG-Bench-Hard
"""

import json
import urllib.request
from pathlib import Path

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "datasets" / "example" / "bbh" / "bbh.jsonl"
BASE_URL = "https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/main/bbh"

# MC subtasks: targets like (A), (B), (C), (D)
MC_SUBTASKS = [
    "disambiguation_qa",
    "date_understanding",
    "logical_deduction_three_objects",
    "movie_recommendation",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "snarks",
]

# Binary subtasks: targets like Yes, No, True, False, valid, invalid
BINARY_SUBTASKS = [
    "boolean_expressions",
    "causal_judgement",
    "formal_fallacies",
    "navigate",
]

# Free-form subtasks: targets are integers or space-separated strings
FREEFORM_SUBTASKS = [
    "multistep_arithmetic_two",
    "object_counting",
    "word_sorting",
    "dyck_languages",
]


def fetch_subtask(subtask: str) -> list[dict]:
    """Fetch examples for a subtask from the BBH GitHub repo."""
    url = f"{BASE_URL}/{subtask}.json"
    resp = urllib.request.urlopen(url)
    data = json.loads(resp.read())
    return data["examples"]


def main() -> None:
    samples: list[dict] = []

    # MC: 7 samples, one per subtask (index 0)
    for subtask in MC_SUBTASKS:
        examples = fetch_subtask(subtask)
        row = examples[0]
        samples.append({
            "id": f"bbh_{subtask}_0",
            "question": row["input"],
            "answer": row["target"],
            "subtask": subtask,
        })

    # Binary: 4 samples, one per subtask (index 0)
    for subtask in BINARY_SUBTASKS:
        examples = fetch_subtask(subtask)
        row = examples[0]
        samples.append({
            "id": f"bbh_{subtask}_0",
            "question": row["input"],
            "answer": row["target"],
            "subtask": subtask,
        })

    # Free-form: 4 samples, one per subtask (index 0)
    for subtask in FREEFORM_SUBTASKS:
        examples = fetch_subtask(subtask)
        row = examples[0]
        samples.append({
            "id": f"bbh_{subtask}_0",
            "question": row["input"],
            "answer": row["target"],
            "subtask": subtask,
        })

    # Write JSONL
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Wrote {len(samples)} samples to {OUTPUT_PATH}")
    for s in samples:
        print(f"  {s['id']}: answer={s['answer']!r}")


if __name__ == "__main__":
    main()
