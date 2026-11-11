"""
run_judge.py
============
A6 LLM-as-judge on a stratified sample of 60 briefings. Paid (Sonnet 5.5,
about $1 for 60), cached: a rerun on the same sample and rubric is free.

    PYTHONPATH=. .venv/bin/python scripts/run_judge.py --week 2026-06-22 --build-only
    PYTHONPATH=. .venv/bin/python scripts/run_judge.py --week 2026-06-22

Step 1 (--build-only, free): draws the sample from the week's briefing run
(data/briefing_runs/<week>_groups/full_checkpoint.jsonl, the file the
warehouse rows were written from), reads the excerpt text from Chroma, and
writes:
    evals/datasets/judge_sample_v1.jsonl   frozen sample (also the CI fixture)
    evals/judge/human_review_v1.md         the 20 briefings to label, as the judge sees them
    evals/judge/human_labels_v1.csv        blank label sheet (never overwritten)
Step 2: judges every item, writes evals/results/judge_<week>_<time>.json/.csv,
logs to MLflow (experiment chbindex-evals).

Options:
    --rebuild-sample   draw a new sample even if one exists (after a prompt change)
    --max-calls N      cap on judge calls (default evals.judge.max_calls)
    --no-mlflow
"""

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

from src.evals import judge as a6
from src.evals.cost import call_cost
from src.evals.dataset import load_cases_jsonl
from src.evals.excerpts import fetch_chunks
from src.evals.harness import EvalRun, _git, log_to_mlflow, save_results


def load_config() -> dict:
    return {
        **yaml.safe_load(open("config/pipeline.yaml")),
        **yaml.safe_load(open("config/cities/los_angeles.yaml")),
    }


def build(config: dict, week: str, run_dir: Path) -> list[a6.JudgeItem]:
    cfg = config["evals"]["judge"]
    cases = load_cases_jsonl(run_dir / "full_checkpoint.jsonl")
    ids = sorted({cid for c in cases for cid in c.briefing.retrieved_chunk_ids})
    chunks = fetch_chunks(config, ids)
    items = a6.build_sample(
        cases, chunks, week, cfg["sample_size"], cfg["human_labels"], cfg["seed"]
    )
    a6.write_sample(items)
    md, csv = a6.write_human_sheet(items, Path(cfg["rubric"]).read_text())
    strata = len({i.stratum for i in items})
    print(
        f"Sample: {len(items)} briefings from {len(cases)}, {strata} strata -> {a6.SAMPLE_PATH}"
    )
    print(f"Human sheet: {sum(i.human for i in items)} items -> {md}, labels in {csv}")
    return items


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--week", required=True)
    parser.add_argument("--run-dir", help="default data/briefing_runs/<week>_groups")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--rebuild-sample", action="store_true")
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
    run_dir = Path(args.run_dir or f"data/briefing_runs/{args.week}_groups")
    if args.max_calls is not None:
        config["evals"]["judge"]["max_calls"] = args.max_calls

    if args.rebuild_sample or not a6.SAMPLE_PATH.exists():
        items = build(config, args.week, run_dir)
    else:
        items = a6.load_sample()
        print(f"Using frozen sample {a6.SAMPLE_PATH} ({len(items)} items)")
    if args.build_only:
        return 0

    judge = a6.Judge(config)
    rows, failures = [], {}
    for item in items:
        try:
            rows.append(a6.scores_row(item, judge.judge(item)))
        except a6.JudgeError as exc:
            failures[item.item_id] = str(exc)
            print(f"  {item.item_id}: {exc}")
    if not rows:
        print("No verdicts.")
        return 1

    cost = call_cost(config, judge.model, judge.input_tokens, judge.output_tokens)
    metrics = a6.judge_metrics(rows) | {
        "judge_calls": judge.calls,
        "cache_hits": judge.cache_hits,
        "failures": len(failures),
        "cost_usd": cost,
    }
    manifest = {
        "week": args.week,
        "suite": "judge",
        "judge_model": judge.model,
        "temperature": str(judge.temperature),
        "rubric": config["evals"]["judge"]["rubric"],
        "rubric_hash": judge.rubric_hash,
        "sample_hash": hashlib.sha256(a6.SAMPLE_PATH.read_bytes()).hexdigest()[:12],
        "briefing_models": ",".join(sorted({i.briefing.model for i in items})),
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "evaluated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    extras = {"by_stratum": a6.by_stratum(rows), "failures": failures}
    run = EvalRun(args.week, manifest, metrics, pd.DataFrame(rows), extras)

    print(f"\nJudge ({judge.model}), {len(rows)} briefings")
    for name, value in metrics.items():
        print(f"  {name:40} {value}")
    unsafe = [r for r in rows if r["unsafe_advice"]]
    for r in unsafe:
        print(f"  UNSAFE {r['item_id']} {r['geoid']}: {r['unsafe_reason']}")
    print(f"\nSaved: {save_results(run, prefix='judge_')}")
    if not args.no_mlflow:
        run_id = log_to_mlflow(
            run, config["ml"]["mlflow_tracking_uri"], f"judge {args.week}"
        )
        print(f"MLflow run: {run_id} (experiment chbindex-evals)")
    print(
        f"Cost this run: ${cost:.2f} ({judge.calls} calls, {judge.cache_hits} cached)"
    )
    print(
        json.dumps(
            {
                "next": "fill evals/judge/human_labels_v1.csv, then scripts/judge_agreement.py"
            }
        )
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
