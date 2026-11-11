"""
harness.py
==========
Run the eval suites on one week of briefings, the same way every time.

    1. Load cases (dataset.load_cases)
    2. Run every scorer on every briefing, plus the week-level metrics
    3. Record exactly what was evaluated (version manifest)
    4. Save results to evals/results/ and log them to MLflow
    5. Compare gated metrics to evals/baselines/ and report regressions

Suites: faithfulness (A2), rules (A4), specificity (A5). All are free: they
score stored briefings and never call a model.

MLflow note: mlflow.genai.evaluate needs Databricks in MLflow 3.1.1, so runs
are logged with the standard tracking API (log_params, log_metrics,
log_table). The scorers are plain functions and can be wrapped as MLflow
@scorer functions after an upgrade.
"""

import hashlib
import json
import statistics
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.evals import faithfulness, rules, specificity
from src.evals.base import Scorer
from src.evals.dataset import EvalCase
from src.rag.briefing import SYSTEM_PROMPT
from src.utils.logger import get_logger

logger = get_logger(__name__)

SUITES: dict[str, dict[str, Scorer]] = {
    "faithfulness": faithfulness.SCORERS,
    "rules": rules.SCORERS,
    "specificity": specificity.SCORERS,
}

# Scorers whose value is a measurement, not pass/fail: report the median.
MEASUREMENTS = {
    "rules.reading_grade",
    "specificity.action_count",
    "specificity.summary_words",
}

# Metrics the baseline gate checks: name -> (direction, allowed slack).
# "higher" means a drop is a regression; "lower" means a rise is.
GATED: dict[str, tuple[str, float]] = {
    "faithfulness.numbers_match_data": ("higher", 0.01),
    "faithfulness.urgency_matches": ("higher", 0.01),
    "faithfulness.drivers_mentioned": ("higher", 0.01),
    "faithfulness.drivers_accounted": ("higher", 0.01),
    "rules.no_source_named": ("higher", 0.01),
    "rules.no_system_talk": ("higher", 0.01),
    "rules.no_jargon": ("higher", 0.02),
    "rules.reading_grade.median": ("lower", 0.5),
    "specificity.driver_separation": ("higher", 0.03),
    "specificity.group_separation": ("higher", 0.03),
    "specificity.duplicate_source_sets": ("lower", 0.02),
}

EXPERIMENT = "chbindex-evals"
RESULTS_DIR = Path("evals/results")
BASELINES_DIR = Path("evals/baselines")


@dataclass
class EvalRun:
    week: str
    manifest: dict[str, Any]
    metrics: dict[str, float]  # aggregate numbers, logged to MLflow
    per_briefing: pd.DataFrame  # one row per briefing: score and reason per scorer
    extras: dict[str, Any] = field(default_factory=dict)  # non-numeric outputs


def run_suites(
    cases: list[EvalCase],
    week: str,
    suites: list[str],
    config: dict[str, Any],
    rank_by: str,
) -> EvalRun:
    rows = [
        {
            "geoid": c.briefing.geoid,
            "urgency": c.tract.urgency,
            "drivers": ",".join(c.tract.drivers),
        }
        for c in cases
    ]
    metrics: dict[str, float] = {}

    for suite in suites:
        for name, scorer in SUITES[suite].items():
            key = f"{suite}.{name}"
            scores = [scorer(c) for c in cases]
            for row, s in zip(rows, scores, strict=True):
                row[key] = s.value
                row[f"{key}.reason"] = s.reason
            values = [s.value for s in scores]
            if key in MEASUREMENTS:
                metrics[f"{key}.median"] = statistics.median(values)
            else:
                metrics[key] = round(statistics.fmean(values), 4)  # pass rate

    extras: dict[str, Any] = {}
    if "specificity" in suites:
        week_level = specificity.specificity_metrics(cases)
        extras["top5_source_pages"] = week_level.pop("top5_source_pages")
        metrics |= {f"specificity.{k}": v for k, v in week_level.items()}
    if "rules" in suites:
        grades = [r["rules.reading_grade"] for r in rows]
        metrics["rules.reading_grade.share_at_or_above_target"] = round(
            sum(g >= rules.READING_GRADE_TARGET for g in grades) / len(grades), 4
        )

    metrics["n_briefings"] = len(cases)
    manifest = version_manifest(cases, week, suites, config, rank_by)
    return EvalRun(week, manifest, metrics, pd.DataFrame(rows), extras)


def version_manifest(
    cases: list[EvalCase],
    week: str,
    suites: list[str],
    config: dict[str, Any],
    rank_by: str,
) -> dict[str, Any]:
    """Everything needed to tell two eval runs apart, or reproduce one."""
    rag = config["rag"]
    return {
        "week": week,
        "suites": ",".join(suites),
        "rank_by": rank_by,
        "git_sha": _git("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "system_prompt_hash": _hash(SYSTEM_PROMPT),
        "rag_config_hash": _hash(json.dumps(rag, sort_keys=True, default=str)),
        "briefing_models": ",".join(sorted({c.briefing.model for c in cases})),
        "embedding_model": rag["embedding_model"],
        "max_distance": rag["max_distance"],
        "evaluated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def save_results(run: EvalRun, prefix: str = "") -> Path:
    """Write evals/results/<prefix><week>_<time>.json (summary) and .csv (per row)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{prefix}{run.week}_{run.manifest['evaluated_at'][:19].replace(':', '')}"
    summary = {"manifest": run.manifest, "metrics": run.metrics, **run.extras}
    (RESULTS_DIR / f"{stem}.json").write_text(json.dumps(summary, indent=2))
    run.per_briefing.to_csv(RESULTS_DIR / f"{stem}.csv", index=False)
    return RESULTS_DIR / f"{stem}.json"


def log_to_mlflow(run: EvalRun, tracking_uri: str, run_name: str | None = None) -> str:
    """One MLflow run per eval run: params = manifest, metrics = aggregates, table = per briefing."""
    import mlflow  # imported here so the scorers and tests do not need MLflow

    mlflow.set_tracking_uri(Path(tracking_uri).resolve().as_uri())
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=run_name or f"briefings {run.week}") as active:
        mlflow.log_params(run.manifest)
        mlflow.log_metrics(
            {k.replace(".", "/"): float(v) for k, v in run.metrics.items()}
        )
        mlflow.log_table(run.per_briefing, artifact_file="per_briefing.json")
        if run.extras:
            mlflow.log_dict(run.extras, "extras.json")
        return active.info.run_id


def compare_to_baseline(run: EvalRun, name: str = "briefings") -> list[str]:
    """Regressions against evals/baselines/<name>.json. Empty list = no regressions."""
    path = BASELINES_DIR / f"{name}.json"
    if not path.exists():
        logger.warning(
            f"No baseline at {path}; run with --update-baseline to create one"
        )
        return []
    baseline = json.loads(path.read_text())["metrics"]

    regressions = []
    for metric, (direction, slack) in GATED.items():
        if metric not in baseline or metric not in run.metrics:
            continue
        old, new = baseline[metric], run.metrics[metric]
        worse = new < old - slack if direction == "higher" else new > old + slack
        if worse:
            regressions.append(
                f"{metric}: {old} -> {new} ({direction} is better, slack {slack})"
            )
    return regressions


def write_baseline(run: EvalRun, name: str = "briefings") -> Path:
    BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    path = BASELINES_DIR / f"{name}.json"
    gated = {k: run.metrics[k] for k in GATED if k in run.metrics}
    path.write_text(json.dumps({"manifest": run.manifest, "metrics": gated}, indent=2))
    return path


# ---------------------------------------------------------------------------
# Private functions
# ---------------------------------------------------------------------------


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
