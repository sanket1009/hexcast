"""
eval_gate.py
============
A9 regression gate. Free, offline, runs in CI on every pull request.

    PYTHONPATH=. python3 scripts/eval_gate.py
    PYTHONPATH=. python3 scripts/eval_gate.py --update-baseline

Checks A2/A4/A5 on the frozen judge sample and A1 metrics on the frozen
retrieval rankings against evals/baselines/ci_*.json (see src/evals/gate.py).
Exit code 1 on any regression or missing baseline.

When a change is intentional (new prompt, new retrieval settings): regenerate
the inputs (scripts/run_judge.py --rebuild-sample, scripts/eval_retrieval.py
--save-rankings), then --update-baseline, in the same PR.
"""

import argparse
import sys

import yaml

from src.evals import gate
from src.evals.harness import write_baseline


def load_config() -> dict:
    return {
        **yaml.safe_load(open("config/pipeline.yaml")),
        **yaml.safe_load(open("config/cities/los_angeles.yaml")),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--update-baseline", action="store_true")
    args = parser.parse_args()
    config = load_config()

    run = gate.briefing_run(config)
    a1_metrics = gate.retrieval_metrics()

    print(f"Frozen briefings ({run.metrics['n_briefings']}):")
    for name in sorted(run.metrics):
        print(f"  {name:50} {run.metrics[name]}")
    print("Frozen A1 rankings:")
    for name, value in a1_metrics.items():
        print(f"  {name:50} {value}")

    if args.update_baseline:
        print(f"\nBaseline updated: {write_baseline(run, gate.BRIEFING_BASELINE)}")
        manifest = {"git_sha": run.manifest["git_sha"], "rankings": str(gate.RANKINGS)}
        print(
            f"Baseline updated: {gate.write_retrieval_baseline(a1_metrics, manifest)}"
        )
        return 0

    regressions = gate.briefing_regressions(run) + gate.retrieval_regressions(
        a1_metrics
    )
    if regressions:
        print("\nREGRESSIONS:")
        for r in regressions:
            print(f"  {r}")
        return 1
    print("\nNo regressions against the CI baselines.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
