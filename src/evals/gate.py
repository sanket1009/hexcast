"""
gate.py
=======
A9: the regression gate CI runs on every pull request. Free and offline.

What it checks:
    1. A2 faithfulness, A4 rules and A5 specificity on the frozen judge sample
       (evals/datasets/judge_sample_v1.jsonl: 60 stored briefings with their
       inputs), against evals/baselines/ci_briefings.json.
    2. A1 metric code on frozen retrieval rankings
       (tests/fixtures/evals/retrieval_rankings_v1.json: the production
       retriever's page rankings for every gold query), against
       evals/baselines/ci_retrieval.json.

The inputs are frozen, so a failure means the scoring code changed what it
measures, or someone regenerated the inputs after a prompt or retrieval
change without updating the baseline. Either way, the PR has to explain it.

Not covered in CI: live A1 retrieval. It needs the 1.2 GB Qwen3 model and the
Chroma index, so it stays a local pre-merge check (scripts/eval_retrieval.py).
"""

import hashlib
import json
from pathlib import Path
from typing import Any

from src.evals import retrieval as a1
from src.evals.dataset import load_cases_jsonl
from src.evals.harness import BASELINES_DIR, EvalRun, compare_to_baseline, run_suites

SAMPLE = Path("evals/datasets/judge_sample_v1.jsonl")
RANKINGS = Path("tests/fixtures/evals/retrieval_rankings_v1.json")
BRIEFING_BASELINE = "ci_briefings"
RETRIEVAL_BASELINE = "ci_retrieval"
BRIEFING_SUITES = ["faithfulness", "rules", "specificity"]

# A1 metric -> direction. Frozen rankings give exact numbers, so the slack only
# absorbs float rounding.
RETRIEVAL_GATED = {
    "recall@3": "higher",
    "recall@10": "higher",
    "mrr": "higher",
    "ndcg@10": "higher",
    "negative_rejection": "higher",
    "empty_rate": "lower",
}
RETRIEVAL_SLACK = 0.005


def briefing_run(config: dict[str, Any], sample: Path = SAMPLE) -> EvalRun:
    cases = load_cases_jsonl(sample)
    week = json.loads(sample.read_text().splitlines()[0]).get("week", "frozen")
    return run_suites(cases, week, BRIEFING_SUITES, config, rank_by="frozen sample")


def briefing_regressions(run: EvalRun) -> list[str]:
    if not (BASELINES_DIR / f"{BRIEFING_BASELINE}.json").exists():
        return [f"missing baseline evals/baselines/{BRIEFING_BASELINE}.json"]
    return compare_to_baseline(run, BRIEFING_BASELINE)


def save_rankings(
    path: Path,
    gold_path: str,
    cutoff: float,
    rankings: dict[str, a1.Ranked],
    manifest: dict[str, Any],
) -> Path:
    """Freeze A1 rankings (written by scripts/eval_retrieval.py --save-rankings)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "gold": gold_path,
        "gold_hash": _hash(Path(gold_path).read_text()),
        "cutoff": cutoff,
        "manifest": manifest,
        "rankings": {
            qid: [[src, page, round(d, 6)] for (src, page), d in ranked]
            for qid, ranked in rankings.items()
        },
    }
    path.write_text(json.dumps(data, indent=1))
    return path


def retrieval_metrics(path: Path = RANKINGS) -> dict[str, float]:
    data = json.loads(path.read_text())
    gold_text = Path(data["gold"]).read_text()
    if _hash(gold_text) != data["gold_hash"]:
        raise ValueError(
            f"{data['gold']} changed since the rankings were frozen; rerun "
            "scripts/eval_retrieval.py --save-rankings and update the baseline"
        )
    gold = a1.load_gold(data["gold"])
    rankings = {
        qid: [((src, int(page)), float(d)) for src, page, d in ranked]
        for qid, ranked in data["rankings"].items()
    }
    metrics, _ = a1.evaluate(gold, rankings, data["cutoff"])
    return metrics


def retrieval_regressions(metrics: dict[str, float]) -> list[str]:
    path = BASELINES_DIR / f"{RETRIEVAL_BASELINE}.json"
    if not path.exists():
        return [f"missing baseline {path}"]
    baseline = json.loads(path.read_text())["metrics"]
    out = []
    for name, direction in RETRIEVAL_GATED.items():
        if name not in baseline or name not in metrics:
            continue
        old, new = baseline[name], metrics[name]
        worse = (
            new < old - RETRIEVAL_SLACK
            if direction == "higher"
            else new > old + RETRIEVAL_SLACK
        )
        if worse:
            out.append(f"A1 {name}: {old} -> {new} ({direction} is better)")
    return out


def write_retrieval_baseline(
    metrics: dict[str, float], manifest: dict[str, Any]
) -> Path:
    BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    path = BASELINES_DIR / f"{RETRIEVAL_BASELINE}.json"
    gated = {k: metrics[k] for k in RETRIEVAL_GATED if k in metrics}
    path.write_text(json.dumps({"manifest": manifest, "metrics": gated}, indent=2))
    return path


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]
