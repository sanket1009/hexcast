"""
judge_agreement.py
==================
A6 judge validation: compare your 20 hand labels with the judge's scores.
Free: reads the judge's cached verdicts, makes no calls.

    PYTHONPATH=. .venv/bin/python scripts/judge_agreement.py

Reads evals/judge/human_labels_v1.csv (rows with any blank are skipped) and
the judge cache for the same items. Reports quadratic weighted Cohen's kappa,
exact and within-1 agreement, and judge-minus-human mean per criterion, and
which criteria the judge can be trusted on (kappa >= evals.judge.min_kappa).
Writes evals/results/judge_agreement_<time>.json and logs to MLflow.
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml

from src.evals import judge as a6
from src.evals.agreement import agreement, read_human_labels
from src.evals.harness import EXPERIMENT, RESULTS_DIR, _git


class Offline:
    """No API client: every verdict must already be in the cache."""

    def __getattr__(self, name):
        raise RuntimeError("judge_agreement makes no model calls")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default=str(a6.HUMAN_CSV))
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(open("config/pipeline.yaml"))
    cfg = config["evals"]["judge"]
    human = read_human_labels(Path(args.labels))
    items = {i.item_id: i for i in a6.load_sample() if i.human}
    print(f"Human labels: {len(human)} of {len(items)} rows filled")
    if not human:
        return 1

    judge = a6.Judge(config, client=Offline())
    judged = {}
    for item_id in human:
        record = judge.cached(items[item_id])
        if record is None:
            print(f"{item_id}: no cached verdict; run scripts/run_judge.py first")
            return 1
        v = record["verdict"]
        judged[item_id] = {c: v[c]["score"] for c in a6.CRITERIA} | {
            "unsafe_advice": v["unsafe_advice"]
        }

    result = agreement(human, judged, cfg["min_kappa"])
    print(
        f"\nAgreement on {result['n']} briefings (trusted at kappa >= {cfg['min_kappa']})"
    )
    print(
        f"  {'criterion':20} {'kappa':>6} {'exact':>6} {'within1':>8} {'judge-human':>12}  trusted"
    )
    for c, r in result["criteria"].items():
        kappa = "n/a" if r["kappa"] is None else f"{r['kappa']:.2f}"
        print(
            f"  {c:20} {kappa:>6} {r['exact']:>6.2f} {r['within_1']:>8.2f} "
            f"{r['mean_diff']:>+12.2f}  {'yes' if r['trusted'] else 'NO'}"
        )
    u = result["unsafe_advice"]
    print(
        f"  unsafe_advice: agreement {u['agreement']:.2f} (you {u['human_yes']} yes, judge {u['judge_yes']} yes)"
    )
    print(f"\nTrusted criteria: {', '.join(result['trusted_criteria']) or 'none'}")

    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    manifest = {
        "judge_model": judge.model,
        "rubric_hash": judge.rubric_hash,
        "labels": args.labels,
        "git_sha": _git("rev-parse", "--short", "HEAD"),
        "evaluated_at": stamp,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"judge_agreement_{stamp[:19].replace(':', '')}.json"
    out.write_text(json.dumps({"manifest": manifest, **result}, indent=2))
    print(f"Saved: {out}")

    if not args.no_mlflow:
        import mlflow

        mlflow.set_tracking_uri(
            Path(config["ml"]["mlflow_tracking_uri"]).resolve().as_uri()
        )
        mlflow.set_experiment(EXPERIMENT)
        with mlflow.start_run(run_name="judge agreement") as active:
            mlflow.log_params(manifest)
            metrics = {"n": result["n"], "unsafe_agreement": u["agreement"]}
            for c, r in result["criteria"].items():
                metrics[f"{c}/exact"] = r["exact"]
                metrics[f"{c}/within_1"] = r["within_1"]
                if r["kappa"] is not None:
                    metrics[f"{c}/kappa"] = r["kappa"]
            mlflow.log_metrics(metrics)
            mlflow.log_dict(result, "agreement.json")
            print(f"MLflow run: {active.info.run_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
