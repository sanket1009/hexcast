"""
run_robustness.py
=================
A7 robustness tests (src/evals/robustness.py). About 30 Haiku calls (under
$0.05); the perturbation tests are free.

    PYTHONPATH=. .venv/bin/python scripts/run_robustness.py --week 2026-06-22

Uses the week's cached parts (data/briefing_runs/<week>_groups/plans.jsonl and
groups.jsonl) for the excerpts each part was written from, and the week's
briefing run (full_checkpoint.jsonl) for the perturbation tracts. Writes
evals/results/robustness_<week>_<time>.json/.csv and logs to MLflow.

Options:
    --skip-live      perturbation tests only (no calls)
    --max-calls N    cap on live calls (default evals.robustness.max_calls)
    --no-mlflow
"""

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

from src.evals import robustness as a7
from src.evals.cost import call_cost
from src.evals.dataset import load_cases_jsonl
from src.evals.excerpts import fetch_chunks
from src.evals.harness import EvalRun, _git, _hash, log_to_mlflow, save_results
from src.evals.sampling import stratified_sample
from src.rag.action_plans import (
    GROUP_SYSTEM_PROMPT,
    PLAN_SYSTEM_PROMPT,
    PlanBriefingGenerator,
)


def load_config() -> dict:
    return {
        **yaml.safe_load(open("config/pipeline.yaml")),
        **yaml.safe_load(open("config/cities/los_angeles.yaml")),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--week", required=True)
    parser.add_argument("--run-dir", help="default data/briefing_runs/<week>_groups")
    parser.add_argument("--skip-live", action="store_true")
    parser.add_argument("--max-calls", type=int)
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()
    load_dotenv(override=False)  # ANTHROPIC_API_KEY from .env, if not already set
    # Before any output file is written, so the manifest shows the code's state
    git_sha, git_dirty = (
        _git("rev-parse", "--short", "HEAD"),
        bool(_git("status", "--porcelain")),
    )

    config = load_config()
    cfg = config["evals"]["robustness"]
    run_dir = Path(args.run_dir or f"data/briefing_runs/{args.week}_groups")
    plans_path, groups_path = run_dir / "plans.jsonl", run_dir / "groups.jsonl"
    rows, calls, cost = [], [], 0.0
    model = config["rag"]["anthropic"]["model"]

    if not args.skip_live:
        units = a7.load_units(plans_path, groups_path)
        plan = a7.live_plan(units)
        n_calls = sum(len(v) for k, v in plan.items() if k != "consistency")
        n_calls += a7.REPEATS * len(plan["consistency"])
        print(
            f"Live tests: {n_calls} calls planned ({', '.join(f'{k} {len(v)}' for k, v in plan.items())})"
        )
        chunks = fetch_chunks(config, a7.needed_chunk_ids(units, plan))
        gen = PlanBriefingGenerator(
            config,
            retriever=a7.NoCalls(),  # excerpts come from the caches
            max_model_calls=args.max_calls or cfg["max_calls"],
        )
        live = a7.LiveTests(gen, units, chunks)
        rows += live.run(plan)
        calls = live.calls
        cost = call_cost(config, gen.model, gen.input_tokens, gen.output_tokens)

    # Perturbation: cached parts only; NoCalls fails the test if anything is missing
    offline = PlanBriefingGenerator(
        config,
        retriever=a7.NoCalls(),
        client=a7.NoCalls(),
        plan_cache_path=plans_path,
        group_cache_path=groups_path,
    )
    offline.plan_cache_path = offline.group_cache_path = None  # never write the caches
    cases = load_cases_jsonl(run_dir / "full_checkpoint.jsonl")
    tracts = stratified_sample(
        cases, cfg["perturbation_tracts"], seed=cfg["seed"], floor=1
    )
    rows += a7.perturbation_rows(tracts, offline)

    metrics = a7.robustness_metrics(rows, calls) | {"cost_usd": cost}
    manifest = {
        "week": args.week,
        "suite": "robustness",
        "model": model,
        "plan_prompt_hash": _hash(PLAN_SYSTEM_PROMPT),
        "group_prompt_hash": _hash(GROUP_SYSTEM_PROMPT),
        "live": not args.skip_live,
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "evaluated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    df = pd.DataFrame(rows)
    run = EvalRun(args.week, manifest, metrics, df)

    print(f"\nRobustness, week {args.week}")
    for name, value in metrics.items():
        print(f"  {name:45} {value}")
    failed = df[~df["passed"].astype(bool)]
    for _, r in failed.iterrows():
        print(f"  FAILED {r['test']} {r['unit']}: {r['reason']}")
    print(f"\nSaved: {save_results(run, prefix='robustness_')}")
    if not args.no_mlflow:
        run_id = log_to_mlflow(
            run, config["ml"]["mlflow_tracking_uri"], f"robustness {args.week}"
        )
        print(f"MLflow run: {run_id} (experiment chbindex-evals)")
    print(f"Cost this run: ${cost:.3f} ({len(calls)} live calls)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
