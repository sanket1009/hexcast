"""
run_briefings_week.py
=====================
Generate one week of LA briefings in two steps.

Step 1, sample (default): about 25 tracts, stratified by urgency, written to a
scratch copy of the warehouse. Writes a review file to read by hand. The live
warehouse is not touched.

Step 2, full: every HIGH and CRITICAL tract. Reuses the sample's briefings, so
only the remaining tracts cost model calls. Writes to the live warehouse only
with --confirm-full, only after the sample exists, and only if every tract
succeeded (or with --allow-partial).

Usage (from the project folder, with ANTHROPIC_API_KEY set):
    python3 scripts/run_briefings_week.py --week 2026-06-22 --mode sample
    python3 scripts/run_briefings_week.py --week 2026-06-22 --mode full --confirm-full

By default each briefing is a shared action plan (one model call per driver
combination) plus group actions for the tract's top Census ACS groups (one
call per hazard and group, at most 21), see src/rag/action_plans.py, plus a
tract summary from a template. --per-tract uses the old one-call-per-tract
format. --max-calls caps model calls for the run (default 30); the run stops
before a call that would exceed it.

Outputs go to data/briefing_runs/<week>_groups/ (or <week>/ with --per-tract):
    plans.jsonl, groups.jsonl, sample_checkpoint.jsonl,
    sample_warehouse.duckdb, sample_review.md, full_checkpoint.jsonl
plans.jsonl starts as a copy of <week>_plans/plans.jsonl when that exists:
driver plans are keyed the same way, so they are not paid for twice.
"""

import argparse
import copy
import os
import shutil
import sys
from datetime import date
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # config and warehouse paths are relative

from src.rag.action_plans import PlanBriefingGenerator  # noqa: E402
from src.rag.briefing import BriefingGenerator  # noqa: E402
from src.rag.briefing_runner import (  # noqa: E402
    commit_week,
    run_week,
    select_sample,
    write_review,
)
from src.rag.risk_inputs import build_tract_risks  # noqa: E402


def load_config() -> dict:
    return {
        **yaml.safe_load(open("config/pipeline.yaml")),
        **yaml.safe_load(open("config/cities/los_angeles.yaml")),
    }


def make_generator(
    config: dict, out_dir: Path, per_tract: bool, max_calls: int
) -> BriefingGenerator:
    """Shared plans and group actions by default; the caches are reused by the full run."""
    if per_tract:
        return BriefingGenerator(config)
    return PlanBriefingGenerator(
        config,
        plan_cache_path=out_dir / "plans.jsonl",
        group_cache_path=out_dir / "groups.jsonl",
        max_model_calls=max_calls,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--week", required=True, help="Week start date, e.g. 2026-06-22"
    )
    parser.add_argument("--mode", choices=["sample", "full"], default="sample")
    parser.add_argument("--confirm-full", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument(
        "--per-tract",
        action="store_true",
        help="One model call per tract (old format) instead of shared action plans",
    )
    parser.add_argument(
        "--max-calls",
        type=int,
        default=30,
        help="Stop before the run makes more than this many model calls",
    )
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY first.")

    week_start = date.fromisoformat(args.week)
    config = load_config()
    live_db = Path(config["storage"]["local_db_path"])
    # Plan-format runs get their own folder, so old per-tract checkpoints are not reused
    run_name = args.week if args.per_tract else f"{args.week}_groups"
    out_dir = ROOT / "data" / "briefing_runs" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    old_plans = ROOT / "data" / "briefing_runs" / f"{args.week}_plans" / "plans.jsonl"
    if (
        not args.per_tract
        and old_plans.exists()
        and not (out_dir / "plans.jsonl").exists()
    ):
        shutil.copy(old_plans, out_dir / "plans.jsonl")
    sample_ckpt = out_dir / "sample_checkpoint.jsonl"
    full_ckpt = out_dir / "full_checkpoint.jsonl"

    if args.mode == "sample":
        scratch_db = out_dir / "sample_warehouse.duckdb"
        if not scratch_db.exists():
            shutil.copy(live_db, scratch_db)
        sample_config = copy.deepcopy(config)
        sample_config["storage"]["local_db_path"] = str(scratch_db)

        risks = build_tract_risks(sample_config, args.week)
        chosen = select_sample(risks)
        print(f"Sample: {len(chosen)} tracts out of {len(risks)} HIGH or CRITICAL")

        generator = make_generator(
            sample_config, out_dir, args.per_tract, args.max_calls
        )
        briefings, failures = run_week(chosen, generator, sample_ckpt)
        if briefings:
            commit_week(
                briefings, failures, week_start, sample_config, allow_partial=True
            )
        write_review(
            out_dir / "sample_review.md", args.week, chosen, briefings, failures
        )

        print(f"Written: {len(briefings)}, failed: {len(failures)}")
        print(f"Review: {out_dir / 'sample_review.md'}")
        print("Read the review before running --mode full.")
        return 1 if failures else 0

    # Full mode
    if not args.confirm_full:
        sys.exit(
            "Full mode writes to the live warehouse. Add --confirm-full to run it."
        )
    if not sample_ckpt.exists():
        sys.exit(
            "No sample found for this week. Run --mode sample and review it first."
        )

    risks = build_tract_risks(config, args.week)
    if not full_ckpt.exists():
        shutil.copy(sample_ckpt, full_ckpt)  # reuse the sample's briefings

    generator = make_generator(config, out_dir, args.per_tract, args.max_calls)
    briefings, failures = run_week(risks, generator, full_ckpt)
    commit_week(
        briefings, failures, week_start, config, allow_partial=args.allow_partial
    )

    print(f"Written to live warehouse: {len(briefings)} of {len(risks)} tracts")
    if failures:
        print(f"Failed: {len(failures)}. See {full_ckpt} for what succeeded.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
