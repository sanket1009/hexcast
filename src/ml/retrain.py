"""
retrain.py
==========
Weekly champion/challenger retraining and scoring for enters_critical_4w.

retrain(config, as_of)
    1. Build the panel from weeks <= as_of (features.load_panel).
    2. Labeled rows end at as_of - 4 weeks. Hold out the last HOLDOUT_WEEKS of
       them; the candidate trains on rows up to HORIZON_WEEKS before the
       holdout (the same embargo as the CV), so no training label window
       overlaps the holdout.
    3. Quality gate: the candidate must beat both baselines on the holdout
       (this week's burden percentile, and "CRITICAL in the last 4 weeks").
    4. Champion check: the current production model was refit on everything
       labeled when it was promoted, so part of today's holdout may be its own
       training data. It is compared with the candidate only on holdout weeks
       after its training ended, and only when those weeks hold at least
       MIN_COMPARE_POSITIVES entrants; otherwise the gate alone decides.
    5. A promoted candidate is refit on all labeled rows and saved as a new
       version; current.json points at it.

score_week(config, week)
    Score every non-CRITICAL tract of the week with the current model and
    write ml_predictions (probability of becoming CRITICAL within 4 weeks,
    rank within the week, model version).

The model is logistic regression on Model A features with sklearn defaults:
in the Phase 6 comparison it beat tuned XGBoost and monotone LightGBM, and
nested tuning of C and the feature set did not change its score.

Storage: data/models/enters_critical_4w/<version>.joblib plus current.json.
A pointer file instead of the MLflow model registry, which needs a database
backend; every run is still logged to MLflow (experiment chbindex-ml-retrain).
"""

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.ml.features import (
    HORIZON_WEEKS,
    MODEL_A_FEATURES,
    TARGET,
    load_panel,
    scoring_rows,
)
from src.ml.report import git_sha
from src.ml.tuning import make_logreg
from src.utils.bigquery import REPLACE_PARTITION, write_dataframe
from src.utils.logger import get_logger

logger = get_logger(__name__)

MODEL_NAME = TARGET
MODEL_DIR = Path("data/models") / MODEL_NAME
FEATURES = MODEL_A_FEATURES
HOLDOUT_WEEKS = 8
MIN_COMPARE_POSITIVES = 50
EXPERIMENT = "chbindex-ml-retrain"
PREDICTIONS_TABLE = "ml_predictions"


# ---------------------------------------------------------------------------
# Model store
# ---------------------------------------------------------------------------


def load_current(model_dir: Path = MODEL_DIR) -> tuple[Any, dict] | tuple[None, None]:
    pointer = model_dir / "current.json"
    if not pointer.exists():
        return None, None
    meta = json.loads(pointer.read_text())
    return joblib.load(model_dir / meta["file"]), meta


def save_version(model: Any, meta: dict, model_dir: Path = MODEL_DIR) -> dict:
    """Write the model file, then swap current.json in one rename."""
    model_dir.mkdir(parents=True, exist_ok=True)
    file = f"{meta['version']}.joblib"
    joblib.dump(model, model_dir / file)
    meta = {**meta, "file": file}
    tmp = model_dir / "current.json.tmp"
    tmp.write_text(json.dumps(meta, indent=2))
    os.replace(tmp, model_dir / "current.json")
    return meta


# ---------------------------------------------------------------------------
# Retrain
# ---------------------------------------------------------------------------


def promotion_split(labeled: pd.DataFrame, holdout_weeks: int = HOLDOUT_WEEKS):
    """(candidate training rows, holdout rows) from the eligible, labeled rows."""
    last = labeled["week_idx"].max()
    hold_start = last - holdout_weeks + 1
    train = labeled[labeled["week_idx"] <= hold_start - HORIZON_WEEKS - 1]
    hold = labeled[labeled["week_idx"] >= hold_start]
    return train, hold


def decide(
    candidate: float,
    baselines: dict[str, float],
    candidate_unseen: float | None = None,
    current_unseen: float | None = None,
) -> bool:
    """Beat every baseline on the holdout, and the current model on the holdout
    weeks it never trained on (when there were enough of them to compare)."""
    if not all(candidate > v for v in baselines.values()):
        return False
    return current_unseen is None or candidate_unseen >= current_unseen


def baseline_scores(rows: pd.DataFrame) -> dict[str, np.ndarray]:
    pct = rows["burden_percentile"].to_numpy()
    return {
        "index": pct,
        "recent_critical": rows["weeks_critical_prev4"].fillna(0).to_numpy()
        + pct / 1000,
    }


def _ap(rows: pd.DataFrame, score: np.ndarray) -> float:
    return float(average_precision_score(rows[TARGET], score))


def retrain(
    config: dict[str, Any],
    as_of: Any,
    model_dir: Path = MODEL_DIR,
    log_mlflow: bool = True,
) -> dict[str, Any]:
    panel = load_panel(config, as_of=as_of)
    labeled = panel[panel["eligible"]].copy()
    labeled[TARGET] = labeled[TARGET].astype(int)
    train, hold = promotion_split(labeled)
    if train.empty or hold[TARGET].nunique() < 2:
        raise ValueError(
            f"Not enough labeled history before {as_of} to train and compare"
        )

    candidate = make_logreg().fit(train[FEATURES], train[TARGET])
    cand_score = _ap(hold, candidate.predict_proba(hold[FEATURES])[:, 1])
    base = {k: _ap(hold, v) for k, v in baseline_scores(hold).items()}

    current, meta = load_current(model_dir)
    cand_oos = cur_score = None
    compare_rows = 0
    if current is not None and meta.get("features") == FEATURES:
        unseen = hold[hold["week_start_date"] > pd.Timestamp(meta["trained_through"])]
        compare_rows = len(unseen)
        if unseen[TARGET].sum() >= MIN_COMPARE_POSITIVES:
            cand_oos = _ap(unseen, candidate.predict_proba(unseen[FEATURES])[:, 1])
            cur_score = _ap(unseen, current.predict_proba(unseen[FEATURES])[:, 1])
    promote = decide(cand_score, base, cand_oos, cur_score)

    weeks = labeled["week_start_date"]
    result = {
        "as_of": str(pd.Timestamp(as_of).date()),
        "labeled_rows": len(labeled),
        "labeled_weeks": f"{weeks.min():%Y-%m-%d}..{weeks.max():%Y-%m-%d}",
        "holdout_weeks": (
            f"{hold['week_start_date'].min():%Y-%m-%d}..{hold['week_start_date'].max():%Y-%m-%d}"
        ),
        "holdout_base_rate": float(hold[TARGET].mean()),
        "candidate_pr_auc": cand_score,
        "baseline_index_pr_auc": base["index"],
        "baseline_recent_critical_pr_auc": base["recent_critical"],
        "compare_rows_unseen_by_current": compare_rows,
        "candidate_pr_auc_unseen": cand_oos,
        "current_pr_auc_unseen": cur_score,
        "current_version": meta["version"] if meta else None,
        "promoted": promote,
    }
    if promote:
        final = make_logreg().fit(labeled[FEATURES], labeled[TARGET])
        version = f"{result['as_of']}_{datetime.now(UTC):%Y%m%dT%H%M%S}"
        saved = save_version(
            final,
            {
                "version": version,
                "model": "logistic_regression",
                "features": FEATURES,
                "trained_on": result["labeled_weeks"],
                "trained_through": f"{weeks.max():%Y-%m-%d}",
                "holdout_pr_auc": cand_score,
                "git_sha": git_sha(),
                "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            },
            model_dir,
        )
        result["version"] = saved["version"]
    else:
        result["version"] = meta["version"]
    logger.info(
        f"Retrain as of {result['as_of']}: candidate PR-AUC {cand_score:.3f} (index "
        f"{base['index']:.3f}, recent {base['recent_critical']:.3f}); current on unseen weeks "
        f"{'not compared' if cur_score is None else f'{cur_score:.3f} vs {cand_oos:.3f}'} -> "
        f"{'promoted ' + result['version'] if promote else 'kept ' + result['version']}"
    )
    if log_mlflow:
        _log_retrain(config, result)
    return result


def _log_retrain(config: dict[str, Any], result: dict[str, Any]) -> None:
    """Best effort: a tracking failure must not fail the retrain."""
    try:
        import mlflow

        mlflow.set_tracking_uri(
            Path(config["ml"]["mlflow_tracking_uri"]).resolve().as_uri()
        )
        mlflow.set_experiment(EXPERIMENT)
        with mlflow.start_run(run_name=f"retrain {result['as_of']}"):
            mlflow.log_params(
                {
                    k: v
                    for k, v in result.items()
                    if not isinstance(v, float) or v is None
                }
            )
            mlflow.log_metrics(
                {k: v for k, v in result.items() if isinstance(v, float) and v == v}
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"MLflow logging skipped: {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Weekly scoring
# ---------------------------------------------------------------------------


def predict_week(panel: pd.DataFrame, model: Any, meta: dict) -> pd.DataFrame:
    rows = scoring_rows(panel)
    prob = model.predict_proba(rows[meta["features"]])[:, 1]
    if not np.isfinite(prob).all():
        raise ValueError("Non-finite predicted probabilities")
    out = pd.DataFrame(
        {
            "geoid": rows["geoid"].to_numpy(),
            "prob_critical_4w": prob.round(5),
            "burden_percentile": rows["burden_percentile"].to_numpy(),
            "model_version": meta["version"],
            "week_start_date": pd.to_datetime(
                rows["week_start_date"]
            ).dt.date.to_numpy(),
        }
    )
    out["rank"] = (
        out["prob_critical_4w"].rank(ascending=False, method="first").astype(int)
    )
    return out.sort_values("rank").reset_index(drop=True)


def score_week(
    config: dict[str, Any], week: Any, model_dir: Path = MODEL_DIR
) -> pd.DataFrame:
    model, meta = load_current(model_dir)
    if model is None:
        raise ValueError(f"No current model in {model_dir}; run retrain first")
    panel = load_panel(config, as_of=week)
    if panel["week_start_date"].max() != pd.Timestamp(week):
        raise ValueError(f"Week {week} is not in the warehouse yet")
    out = predict_week(panel, model, meta)
    out["city_fips"] = config["full_fips"]
    write_dataframe(out, PREDICTIONS_TABLE, config, write_disposition=REPLACE_PARTITION)
    logger.info(
        f"Scored {len(out):,} tracts for {week} with {meta['version']}; "
        f"top probability {out['prob_critical_4w'].max():.3f}"
    )
    return out
