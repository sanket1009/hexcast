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

Outputs go to data/briefing_runs/<week>/:
    sample_checkpoint.jsonl, sample_warehouse.duckdb, sample_review.md
    full_checkpoint.jsonl
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--week", required=True, help="Week start date, e.g. 2026-06-22")
    parser.add_argument("--mode", choices=["sample", "full"], default="sample")
    parser.add_argument("--confirm-full", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY first.")

    week_start = date.fromisoformat(args.week)
    config = load_config()
    live_db = Path(config["storage"]["local_db_path"])
    out_dir = ROOT / "data" / "briefing_runs" / args.week
    out_dir.mkdir(parents=True, exist_ok=True)
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

        generator = BriefingGenerator(sample_config)
        briefings, failures = run_week(chosen, generator, sample_ckpt)
        if briefings:
            commit_week(briefings, failures, week_start, sample_config, allow_partial=True)
        write_review(out_dir / "sample_review.md", args.week, chosen, briefings, failures)

        print(f"Written: {len(briefings)}, failed: {len(failures)}")
        print(f"Review: {out_dir / 'sample_review.md'}")
        print("Read the review before running --mode full.")
        return 1 if failures else 0

    # Full mode
    if not args.confirm_full:
        sys.exit("Full mode writes to the live warehouse. Add --confirm-full to run it.")
    if not sample_ckpt.exists():
        sys.exit("No sample found for this week. Run --mode sample and review it first.")

    risks = build_tract_risks(config, args.week)
    if not full_ckpt.exists():
        shutil.copy(sample_ckpt, full_ckpt)  # reuse the sample's briefings

    generator = BriefingGenerator(config)
    briefings, failures = run_week(risks, generator, full_ckpt)
    commit_week(briefings, failures, week_start, config, allow_partial=args.allow_partial)

    print(f"Written to live warehouse: {len(briefings)} of {len(risks)} tracts")
    if failures:
        print(f"Failed: {len(failures)}. See {full_ckpt} for what succeeded.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
