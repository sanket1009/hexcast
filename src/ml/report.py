"""
report.py
=========
Calibration, equity audit and MLflow logging for the Phase 6 champion.

Calibration
    Brier score against the base-rate-only reference, a 10-bin reliability
    table (equal-count bins of predicted probability) and the expected
    calibration error (ECE, count-weighted mean |predicted - observed|).
    Also mean predicted vs observed per test window, because the base rate
    moves with the season (0.069 to 0.099 across our windows).

Equity audit
    Splits tracts into terciles of an attribute (median income, low-income
    share, no-vehicle share, limited-English share) and reports per tercile:
    base rate, PR-AUC, recall of the weekly top-k list (share of true entrants
    that the county-wide top k flags), and calibration gap (mean predicted
    minus observed). Records which tercile has the lowest recall, since a gap
    against advantaged tracts reads differently from one against vulnerable
    tracts. Flags a gap when the lowest tercile recall is below
    RECALL_RATIO_FLAG of the highest (the four-fifths rule used in fairness
    audits) or a calibration gap exceeds CALIBRATION_GAP_FLAG.
    Race/ethnicity is not in the warehouse yet (needs an ACS B03002 pull), so
    majority group is not audited.
"""

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

RECALL_RATIO_FLAG = 0.8
CALIBRATION_GAP_FLAG = 0.02
EQUITY_ATTRIBUTES = [
    "median_income",
    "low_income_pct",
    "no_vehicle_pct",
    "limited_english_pct",
]
EXPERIMENT = "chbindex-ml"


def calibration_report(
    oof: pd.DataFrame, score: str, target: str, n_bins: int = 10
) -> dict:
    y, p = oof[target].to_numpy(), oof[score].to_numpy()
    bins = pd.qcut(p, n_bins, duplicates="drop")
    table = (
        oof.assign(_bin=bins)
        .groupby("_bin", observed=True)
        .agg(predicted=(score, "mean"), observed=(target, "mean"), n=(score, "size"))
        .reset_index(drop=True)
    )
    ece = float(
        (table["predicted"] - table["observed"]).abs().mul(table["n"]).sum() / len(p)
    )
    windows = oof.groupby("time_window").agg(
        predicted=(score, "mean"), observed=(target, "mean")
    )
    return {
        "brier": float(brier_score_loss(y, p)),
        "brier_base_rate_only": float(brier_score_loss(y, np.full(len(y), y.mean()))),
        "ece": ece,
        "reliability": table.round(4).to_dict("records"),
        "per_window": {int(k): v for k, v in windows.round(4).to_dict("index").items()},
    }


def topk_flags(oof: pd.DataFrame, score: str, k: int) -> pd.Series:
    """True for rows in their week's county-wide top k by score."""
    rank = oof.groupby("week_idx")[score].rank(ascending=False, method="first")
    return rank <= k


def equity_audit(
    oof: pd.DataFrame,
    tracts: pd.DataFrame,
    score: str,
    target: str,
    k: int,
    attributes: list[str] | None = None,
) -> dict:
    """Per-tercile metrics for each attribute; tercile cut over tracts, not rows."""
    df = oof.merge(tracts, on="geoid", how="left")
    df["_flag"] = topk_flags(df, score, k)
    out = {}
    for attr in attributes or EQUITY_ATTRIBUTES:
        t = tracts.dropna(subset=[attr])
        cuts = pd.qcut(t[attr], 3, labels=["low", "mid", "high"])
        df["_tercile"] = pd.Categorical(
            df["geoid"].map(dict(zip(t["geoid"], cuts.astype(str), strict=True))),
            categories=["low", "mid", "high"],
            ordered=True,
        )
        rows = {}
        for name, g in df.dropna(subset=["_tercile"]).groupby(
            "_tercile", observed=True
        ):
            pos = g[g[target] == 1]
            rows[str(name)] = {
                "tracts": int(g["geoid"].nunique()),
                "rows": int(len(g)),
                "base_rate": float(g[target].mean()),
                "pr_auc": float(average_precision_score(g[target], g[score])),
                "topk_recall": float(pos["_flag"].mean()) if len(pos) else float("nan"),
                "calibration_gap": float(g[score].mean() - g[target].mean()),
            }
        recalls = [r["topk_recall"] for r in rows.values()]
        ratio = min(recalls) / max(recalls) if max(recalls) > 0 else float("nan")
        worst_gap = max(abs(r["calibration_gap"]) for r in rows.values())
        out[attr] = {
            "terciles": rows,
            "recall_ratio_min_max": float(ratio),
            "lowest_recall_tercile": min(rows, key=lambda r: rows[r]["topk_recall"]),
            "max_abs_calibration_gap": float(worst_gap),
            "flag": bool(ratio < RECALL_RATIO_FLAG or worst_gap > CALIBRATION_GAP_FLAG),
        }
    return out


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def log_to_mlflow(
    summary: dict[str, Any], tracking_uri: str, artifacts: list[Path]
) -> str:
    """One MLflow run: params = data/code versions and chosen settings, metrics = scores."""
    import mlflow

    mlflow.set_tracking_uri(Path(tracking_uri).resolve().as_uri())
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"compare {summary['reference']}") as run:
        mlflow.log_params(
            {
                "git_sha": summary["git_sha"],
                "target": summary["target"],
                "weeks": summary["weeks"],
                "rows": summary["rows"],
                "oof_rows": summary["oof_rows"],
                "test_windows": json.dumps(summary["test_windows"]),
                "reference": summary["reference"],
                **{
                    f"chosen_{k}": json.dumps(v["counts"])
                    for k, v in summary["chosen_params"].items()
                },
            }
        )
        metrics = {}
        for s, m in summary["pooled"].items():
            for name, v in m.items():
                metrics[f"{s}/{name}"] = v
        for w, row in summary["per_window"].items():
            for s, v in row.items():
                if isinstance(v, float):
                    metrics[
                        f"window{w}/{s}_pr_auc"
                        if s != "base_rate"
                        else f"window{w}/base_rate"
                    ] = v
        for pair, c in summary.get("comparisons", {}).items():
            b = c["pr_auc_bootstrap"]
            metrics[f"{pair}/pr_auc_diff"] = b["diff"]
            metrics[f"{pair}/ci_low"] = b["ci_low"]
            metrics[f"{pair}/ci_high"] = b["ci_high"]
        cal = summary.get("calibration", {})
        for name in ("brier", "brier_base_rate_only", "ece"):
            if name in cal:
                metrics[f"calibration/{name}"] = cal[name]
        for attr, a in summary.get("equity", {}).items():
            metrics[f"equity/{attr}/recall_ratio"] = a["recall_ratio_min_max"]
            metrics[f"equity/{attr}/max_calibration_gap"] = a["max_abs_calibration_gap"]
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v == v})
        for path in artifacts:
            if path.exists():
                mlflow.log_artifact(str(path))
        return run.info.run_id
