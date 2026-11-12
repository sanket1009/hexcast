"""
ml_first_run.py
===============
Phase 6 part 1: first run of the "enters CRITICAL within 4 weeks" models
under space-time blocked CV. Free: no API calls, no MLflow yet (part 2).

    PYTHONPATH=. .venv/bin/python scripts/ml_first_run.py

Compares on the same out-of-fold rows:
    baseline_index   this week's burden percentile (no model)
    xgb_a            XGBoost on temporal + static features
    xgb_b            XGBoost on Model A features + 8-nearest-neighbor lags

Writes (both gitignored):
    data/processed/ml/part1_oof.parquet     out-of-fold scores per tract-week
    evals/results/ml/part1_summary.json     metrics, folds, leakage screen
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

from src.ml.features import (
    MODEL_A_FEATURES,
    MODEL_B_FEATURES,
    TARGET,
    check_no_leakage,
    engineer_features,
)
from src.ml.spatial_cv import SpaceTimeBlockCV, build_spatial_groups

OOF_PATH = Path("data/processed/ml/part1_oof.parquet")
SUMMARY_PATH = Path("evals/results/ml/part1_summary.json")
SCREEN_FLAG_AUC = 0.98  # a single feature this good on its own is suspect
TOP_K_PER_WEEK = 50  # about the median weekly number of new CRITICAL tracts


def load_config() -> dict:
    return {
        **yaml.safe_load(open("config/pipeline.yaml")),
        **yaml.safe_load(open("config/cities/los_angeles.yaml")),
    }


def single_feature_screen(df: pd.DataFrame, cols: list[str]) -> list[dict]:
    """ROC-AUC of each feature alone (direction-free). Catches label leaks."""
    rows = []
    for c in cols:
        m = df[c].notna()
        if df.loc[m, TARGET].nunique() < 2:
            continue
        auc = roc_auc_score(df.loc[m, TARGET], df.loc[m, c])
        rows.append(
            {
                "feature": c,
                "auc": round(max(auc, 1 - auc), 4),
                "missing": round(1 - m.mean(), 4),
            }
        )
    return sorted(rows, key=lambda r: -r["auc"])


def precision_at_k(oof: pd.DataFrame, score: str, k: int) -> float:
    """Mean over weeks of the share of true entrants among the top-k scored tracts."""
    vals = []
    for _, wk in oof.groupby("week_idx"):
        top = wk.nlargest(k, score)
        vals.append(top[TARGET].mean())
    return float(np.mean(vals))


def metrics(y: np.ndarray, s: np.ndarray) -> dict:
    return {
        "pr_auc": float(average_precision_score(y, s)),
        "roc_auc": float(roc_auc_score(y, s)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators", type=int, default=None)
    args = parser.parse_args()

    config = load_config()
    ml = config["ml"]
    params = ml["xgboost"]
    seed = ml["random_seed"]

    check_no_leakage(MODEL_B_FEATURES)
    df = engineer_features(config)

    # Spatial blocks: H3 res-6 parent of each tract centroid.
    tracts = df.drop_duplicates("geoid")[["geoid", "centroid_lat", "centroid_lon"]]
    tract_groups = dict(
        zip(
            tracts["geoid"],
            build_spatial_groups(
                tracts["geoid"],
                tracts["centroid_lat"],
                tracts["centroid_lon"],
                h3_resolution=6,
            ),
            strict=True,
        )
    )
    groups = df["geoid"].map(tract_groups).to_numpy()

    print(
        f"\nPanel: {len(df):,} rows, {df['geoid'].nunique():,} tracts, "
        f"{df['week_idx'].nunique()} weeks, base rate {df[TARGET].mean():.3f}, "
        f"{len(set(groups))} H3 res-6 blocks"
    )

    screen = single_feature_screen(df, MODEL_B_FEATURES)
    print("\nSingle-feature screen (ROC-AUC alone, top 8):")
    for r in screen[:8]:
        print(f"  {r['feature']:<24} {r['auc']:.3f}  missing {r['missing']:.1%}")
    flagged = [r["feature"] for r in screen if r["auc"] >= SCREEN_FLAG_AUC]
    if flagged:
        print(f"  FLAG: features at or above {SCREEN_FLAG_AUC}: {flagged}")

    cv = SpaceTimeBlockCV(n_spatial_splits=ml["cv_folds"], random_seed=seed)
    print(f"\nTest windows (week_idx): {cv.time_windows(df['week_idx'].to_numpy())}")

    models = {"xgb_a": MODEL_A_FEATURES, "xgb_b": MODEL_B_FEATURES}
    oof_parts, folds, gains = [], [], {m: [] for m in models}
    t0 = time.time()
    for info, tr, te in cv.split(df["week_idx"].to_numpy(), groups):
        part = df.iloc[te][
            ["geoid", "week_idx", "week_start_date", TARGET, "burden_percentile"]
        ].copy()
        part["baseline_index"] = part["burden_percentile"]
        part["time_window"], part["spatial_fold"] = (
            info["time_window"],
            info["spatial_fold"],
        )
        y_tr = df[TARGET].to_numpy()[tr]
        fold_row = {
            **info,
            "n_train": len(tr),
            "n_test": len(te),
            "test_base_rate": float(part[TARGET].mean()),
            "baseline_index": metrics(part[TARGET], part["baseline_index"]),
        }
        for name, cols in models.items():
            model = xgb.XGBClassifier(
                n_estimators=args.n_estimators or params["n_estimators"],
                learning_rate=params["learning_rate"],
                max_depth=params["max_depth"],
                subsample=params["subsample"],
                colsample_bytree=params["colsample_bytree"],
                tree_method="hist",
                eval_metric="aucpr",
                random_state=seed,
                n_jobs=-1,
            )
            model.fit(df.iloc[tr][cols], y_tr)
            part[name] = model.predict_proba(df.iloc[te][cols])[:, 1]
            fold_row[name] = metrics(part[TARGET], part[name])
            gains[name].append(model.get_booster().get_score(importance_type="gain"))
        oof_parts.append(part)
        folds.append(fold_row)
        print(
            f"  window {info['time_window']} fold {info['spatial_fold']}: "
            f"train {len(tr):>6,} test {len(te):>6,}  PR-AUC "
            + "  ".join(
                f"{n} {fold_row[n]['pr_auc']:.3f}" for n in ["baseline_index", *models]
            )
        )
    oof = pd.concat(oof_parts, ignore_index=True)
    print(f"  {len(folds)} fits in {time.time() - t0:.0f} s")

    scorers = ["baseline_index", *models]
    summary = {
        "rows": len(df),
        "oof_rows": len(oof),
        "oof_base_rate": float(oof[TARGET].mean()),
        "pooled": {},
        "per_window": {},
        "fold_mean_sd": {},
        "screen": screen,
        "folds": folds,
    }
    print(
        f"\nOut-of-fold results ({len(oof):,} rows, base rate {oof[TARGET].mean():.3f}):"
    )
    print(
        f"  {'scorer':<16}{'PR-AUC':>8}{'ROC-AUC':>9}{'fold PR-AUC mean+-sd':>24}"
        f"{f'prec@{TOP_K_PER_WEEK}/wk':>14}"
    )
    for s in scorers:
        pooled = metrics(oof[TARGET], oof[s])
        pk = precision_at_k(oof, s, TOP_K_PER_WEEK)
        fa = np.array([f[s]["pr_auc"] for f in folds])
        summary["pooled"][s] = {**pooled, f"precision_at_{TOP_K_PER_WEEK}_per_week": pk}
        summary["fold_mean_sd"][s] = {
            "pr_auc_mean": float(fa.mean()),
            "pr_auc_sd": float(fa.std()),
        }
        print(
            f"  {s:<16}{pooled['pr_auc']:>8.3f}{pooled['roc_auc']:>9.3f}"
            f"{fa.mean():>15.3f} +- {fa.std():.3f}{pk:>14.3f}"
        )

    print("\nPooled PR-AUC by time window:")
    for w, wk in oof.groupby("time_window"):
        weeks = sorted(wk["week_start_date"].dt.strftime("%Y-%m-%d").unique())
        row = {s: float(average_precision_score(wk[TARGET], wk[s])) for s in scorers}
        summary["per_window"][int(w)] = {
            "weeks": [weeks[0], weeks[-1]],
            "base_rate": float(wk[TARGET].mean()),
            **row,
        }
        print(
            f"  {weeks[0]} to {weeks[-1]} (base {wk[TARGET].mean():.3f}): "
            + "  ".join(f"{s} {row[s]:.3f}" for s in scorers)
        )

    print("\nxgb_b mean gain, top 12:")
    g = pd.DataFrame(gains["xgb_b"]).fillna(0).mean().sort_values(ascending=False)
    summary["xgb_b_gain"] = {k: float(v) for k, v in g.items()}
    for f, v in g.head(12).items():
        print(f"  {f:<24} {v:8.1f}")

    OOF_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    oof.to_parquet(OOF_PATH, index=False)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote {OOF_PATH} and {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
