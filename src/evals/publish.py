"""
publish.py
==========
Copies the accepted evaluation results into the warehouse table eval_summary,
so the API and dashboard can show them without reading files or MLflow.

Sources (the files are the record; the table is a mirror of them):
    ml          evals/results/ml/compare_summary.json  (scripts/ml_compare.py)
    briefings   evals/baselines/briefings.json         (A2/A4/A5 on the 2026-06-22 briefings)
    retrieval   evals/baselines/ci_retrieval.json      (A1 on the production retriever's rankings)

Long format, one row per number:

    suite      ml | briefings | retrieval
    section    panel, pooled, window, comparison, calibration, reliability,
               equity, equity_summary (ml); metric (briefings, retrieval)
    model      scorer or model name (index, recent_critical, logreg_tuned, ...)
    subgroup   window number, reliability bin, "attribute:tercile", ...
    metric     pr_auc, ci_low, topk_recall, recall@3, ...
    value      the number (flags are 0/1)
    n          rows or tracts behind the number, where it has one
    detail     text that is not a number (week range, embedding model)

Why long: compare_summary.json is nested (models x metrics, attributes x
terciles, bootstrap intervals). A wide table would need a schema change for
every new metric; a long one is filtered in the API instead, and works the
same in DuckDB and BigQuery.

The whole table is rebuilt from the three files on every publish
(WRITE_TRUNCATE): it is small, and MLflow already keeps the history of runs.

Run:
    PYTHONPATH=. python scripts/publish_evals.py
ml_compare.py publishes after it writes compare_summary.json.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.bigquery import write_dataframe
from src.utils.logger import get_logger

logger = get_logger(__name__)

EVAL_TABLE = "eval_summary"
ML_SUMMARY = Path("evals/results/ml/compare_summary.json")
BRIEFING_BASELINE = Path("evals/baselines/briefings.json")
RETRIEVAL_BASELINE = Path("evals/baselines/ci_retrieval.json")
COLUMNS = [
    "city_fips", "suite", "section", "model", "subgroup", "metric",
    "value", "n", "detail", "git_sha", "evaluated_at",
]  # fmt: skip


def _row(section: str, metric: str, value: Any, **extra: Any) -> dict[str, Any]:
    return {"section": section, "metric": metric, "value": float(value), **extra}


def ml_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten compare_summary.json into rows."""
    rows: list[dict[str, Any]] = []
    ref = summary["reference"]

    # The panel: what was evaluated
    k = None
    for model_metrics in summary["pooled"].values():
        for name in model_metrics:
            if m := re.fullmatch(r"precision_at_(\d+)_per_week", name):
                k = int(m.group(1))
    rows += [
        _row("panel", "rows", summary["rows"], detail=summary["weeks"]),
        _row("panel", "test_rows", summary["oof_rows"]),
        _row("panel", "base_rate", summary["oof_base_rate"], n=summary["oof_rows"]),
        _row("panel", "test_windows", len(summary["test_windows"])),
        _row("panel", "reference", 1, model=ref, detail=summary["target"]),
    ]
    if k:
        rows.append(_row("panel", "top_k_per_week", k))

    # Pooled out-of-fold scores, every scorer (baselines and models)
    for model, metrics in summary["pooled"].items():
        for metric, value in metrics.items():
            if metric.startswith("precision_at_"):
                metric = "precision_at_k"
            rows.append(_row("pooled", metric, value, model=model))

    # PR-AUC per test window
    for w, window in summary["per_window"].items():
        sub = str(int(w) + 1)
        rows.append(_row("window", "base_rate", window["base_rate"], subgroup=sub))
        for model, value in window.items():
            if model not in ("weeks", "base_rate"):
                rows.append(_row("window", "pr_auc", value, model=model, subgroup=sub))

    # Reference vs each other scorer: block bootstrap of the PR-AUC gap, DeLong on ROC-AUC
    for key, comp in summary["comparisons"].items():
        other = key.removeprefix(f"{ref}_vs_")
        boot, delong = comp["pr_auc_bootstrap"], comp["delong"]
        for metric in ("diff", "ci_low", "ci_high", "share_le_0"):
            rows.append(
                _row("comparison", metric, boot[metric], model=other, n=boot["n_boot"])
            )
        rows.append(_row("comparison", "delong_p", delong["p_value"], model=other))

    cal = summary["calibration"]
    for metric in ("brier", "brier_base_rate_only", "ece"):
        rows.append(_row("calibration", metric, cal[metric], model=ref))
    for i, b in enumerate(cal["reliability"]):
        for metric in ("predicted", "observed"):
            rows.append(
                _row(
                    "reliability",
                    metric,
                    b[metric],
                    model=ref,
                    subgroup=str(i + 1),
                    n=b["n"],
                )
            )

    for attr, audit in summary["equity"].items():
        for tercile, r in audit["terciles"].items():
            for metric in ("base_rate", "pr_auc", "topk_recall", "calibration_gap"):
                rows.append(
                    _row(
                        "equity", metric, r[metric], model=ref,
                        subgroup=f"{attr}:{tercile}", n=r["tracts"],
                    )
                )  # fmt: skip
        rows += [
            _row("equity_summary", "recall_ratio_min_max", audit["recall_ratio_min_max"],
                 model=ref, subgroup=attr, detail=audit["lowest_recall_tercile"]),
            _row("equity_summary", "flag", int(audit["flag"]), model=ref, subgroup=attr),
        ]  # fmt: skip

    git_sha = summary.get("git_sha")
    for r in rows:
        r.update(suite="ml", git_sha=git_sha, evaluated_at=None)
    return rows


def baseline_rows(suite: str, baseline: dict[str, Any]) -> list[dict[str, Any]]:
    """An eval-harness baseline file: manifest + flat metrics."""
    man = baseline.get("manifest", {})
    model = man.get("briefing_models") or man.get("embedding_model")
    detail = man.get("week") or man.get("rankings")
    return [
        {
            "suite": suite,
            "section": "metric",
            "model": model,
            "metric": metric,
            "value": float(value),
            "detail": detail,
            "git_sha": man.get("git_sha"),
            "evaluated_at": man.get("evaluated_at"),
        }
        for metric, value in baseline["metrics"].items()
    ]


def _mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(
        timespec="seconds"
    )


def build_eval_summary(city_fips: str, root: Path = Path(".")) -> pd.DataFrame:
    """All three suites as one DataFrame; a missing file is skipped with a warning."""
    rows: list[dict[str, Any]] = []
    ml_path = root / ML_SUMMARY
    if ml_path.exists():
        ml = ml_rows(json.loads(ml_path.read_text()))
        for r in ml:
            r["evaluated_at"] = _mtime(ml_path)
        rows += ml
    else:
        logger.warning(f"No ML evaluation at {ml_path}; run scripts/ml_compare.py")
    for suite, rel in (
        ("briefings", BRIEFING_BASELINE),
        ("retrieval", RETRIEVAL_BASELINE),
    ):
        path = root / rel
        if not path.exists():
            logger.warning(f"No {suite} baseline at {path}")
            continue
        suite_rows = baseline_rows(suite, json.loads(path.read_text()))
        for r in suite_rows:
            r["evaluated_at"] = r["evaluated_at"] or _mtime(path)
        rows += suite_rows

    df = pd.DataFrame(rows).reindex(columns=COLUMNS)
    df["city_fips"] = city_fips
    df["n"] = df["n"].astype("Int64")
    for col in ("model", "subgroup", "detail", "git_sha", "evaluated_at"):
        df[col] = df[col].astype("string")
    return df


def publish_eval_summary(
    config: dict[str, Any], root: Path = Path(".")
) -> pd.DataFrame:
    df = build_eval_summary(config["full_fips"], root)
    if df.empty:
        raise FileNotFoundError("No evaluation results to publish")
    write_dataframe(df, EVAL_TABLE, config, write_disposition="WRITE_TRUNCATE")
    counts = df.groupby("suite").size().to_dict()
    logger.info(f"Wrote {len(df)} rows to {EVAL_TABLE}: {counts}")
    return df
