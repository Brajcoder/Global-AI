from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    dataset_path = Path("benchmarks/datasets/temporal_traps.json")
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    cohorts = {}
    for row in data:
        cohorts[row["cohort"]] = cohorts.get(row["cohort"], 0) + 1

    print("Temporal trap tasks:")
    for k, v in sorted(cohorts.items()):
        print(f"- {k}: {v}")


if __name__ == "__main__":
    main()
