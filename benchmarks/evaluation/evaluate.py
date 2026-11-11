"""Command-line benchmark runner.

Summarise the dataset:
    python -m benchmarks.evaluation.evaluate

Run a benchmark against a running API server and print the leaderboard:
    python -m benchmarks.evaluation.evaluate --run --agents parametric_baseline disciplined_verifier --limit 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx

DATASET_PATH = Path(__file__).resolve().parents[1] / "datasets" / "temporal_traps.json"


def summarize() -> dict[str, int]:
    data = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    cohorts: dict[str, int] = {}
    for row in data:
        cohorts[row["cohort"]] = cohorts.get(row["cohort"], 0) + 1
    return cohorts


def run(base_url: str, agents: list[str], limit: int | None, api_key: str) -> int:
    headers = {"X-API-Key": api_key} if api_key else {}
    body = {"agent_ids": agents, "background": True, **({"limit_tasks": limit} if limit else {})}
    with httpx.Client(base_url=base_url.rstrip("/"), headers=headers, timeout=30) as client:
        res = client.post("/api/v1/benchmarks/run", json=body)
        if res.status_code >= 400:
            print(f"Could not start benchmark: HTTP {res.status_code} {res.text}", file=sys.stderr)
            return 1
        state = res.json()
        while state["status"] == "running":
            print(f"  {state['tasks_done']}/{state['tasks_total']} tasks done...", flush=True)
            time.sleep(2)
            state = client.get(f"/api/v1/benchmarks/{state['benchmark_id']}").json()

    print(f"\nBenchmark {state['benchmark_id']}: {state['status']} (telemetry: {state['telemetry_source']})")
    if state.get("error"):
        print(f"Error: {state['error']}", file=sys.stderr)
    print(f"{'agent':<28}{'model':<28}{'truth':>8}{'eff':>8}{'chronos':>9}{'tokens':>9}{'failed':>8}")
    for row in state["leaderboard"]:
        print(
            f"{row['agent_id']:<28}{row['model_name'][:26]:<28}{row['truth_score']:>8.2f}{row['eff_score']:>8.2f}"
            f"{row['chronos_score']:>9.2f}{row['tokens_used']:>9}{row['tasks_failed']:>8}"
        )
    return 0 if state["status"] == "completed" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="store_true", help="run a benchmark against the API server")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument(
        "--agents", nargs="+", default=["parametric_baseline", "unbounded_deep_research", "disciplined_verifier"]
    )
    parser.add_argument("--limit", type=int, default=None, help="only run the first N tasks")
    parser.add_argument("--api-key", default=os.environ.get("API_KEY", ""), help="X-API-Key, if the server requires one")
    args = parser.parse_args()

    if args.run:
        return run(args.base_url, args.agents, args.limit, args.api_key)

    print("Temporal trap tasks:")
    for cohort, count in sorted(summarize().items()):
        print(f"- {cohort}: {count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
