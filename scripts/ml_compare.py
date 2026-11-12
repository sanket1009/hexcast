"""
ml_compare.py
=============
Phase 6 part 2: baselines and tuned models for "enters CRITICAL within
4 weeks", scored on the space-time test windows. Settings are chosen on a
nested validation split (src/ml/tuning.py), never on the test windows.
Free: no API calls.

    PYTHONPATH=. .venv/bin/python scripts/ml_compare.py

Scorers (same out-of-fold rows for all):
    index            this week's burden percentile
    recent_critical  weeks CRITICAL among the previous 4, ties by percentile
    logreg           logistic regression on Model A features (fold-fit
                     median fill + standardization)
    xgb_a            tuned XGBoost, Model A features
    xgb_b            tuned XGBoost, Model A + 8-nearest-neighbor lags

Writes (gitignored):
    data/processed/ml/compare_oof.parquet
    evals/results/ml/compare_summary.json
"""

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.ml.features import (
    MODEL_A_FEATURES,
    MODEL_B_FEATURES,
    TARGET,
    engineer_features,
)
from src.ml.scoring import precision_at_k, ranking_metrics
from src.ml.spatial_cv import (
    SpaceTimeBlockCV,
    assign_blocks_to_folds,
    build_spatial_groups,
)
from src.ml.tuning import fit_xgb, inner_split, tune_xgb

OOF_PATH = Path("data/processed/ml/compare_oof.parquet")
SUMMARY_PATH = Path("evals/results/ml/compare_summary.json")
TOP_K_PER_WEEK = 50
MODELS = {"xgb_a": MODEL_A_FEATURES, "xgb_b": MODEL_B_FEATURES}
SCORERS = ["index", "recent_critical", "logreg", *MODELS]


def load_config() -> dict:
    return {
        **yaml.safe_load(open("config/pipeline.yaml")),
        **yaml.safe_load(open("config/cities/los_angeles.yaml")),
    }


def spatial_groups(df: pd.DataFrame) -> np.ndarray:
    tracts = df.drop_duplicates("geoid")
    cells = build_spatial_groups(
        tracts["geoid"], tracts["centroid_lat"], tracts["centroid_lon"], h3_resolution=6
    )
    return df["geoid"].map(dict(zip(tracts["geoid"], cells, strict=True))).to_numpy()


def logreg() -> object:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=2000),
    )


def main() -> int:
    config = load_config()
    ml = config["ml"]
    seed, n_splits = ml["random_seed"], ml["cv_folds"]

    df = engineer_features(config)
    groups = spatial_groups(df)
    week_idx = df["week_idx"].to_numpy()
    y = df[TARGET].to_numpy()
    cv = SpaceTimeBlockCV(n_spatial_splits=n_splits, random_seed=seed)
    fold = assign_blocks_to_folds(groups, n_splits, seed)  # same assignment the CV uses

    print(
        f"\nPanel: {len(df):,} rows, {df['week_idx'].nunique()} weeks, base rate {y.mean():.3f}"
    )
    print(f"Test windows (week_idx): {cv.time_windows(week_idx)}")

    parts, folds, chosen = [], [], {m: [] for m in MODELS}
    t0 = time.time()
    for info, tr, te in cv.split(week_idx, groups):
        k, (start, _) = info["spatial_fold"], info["test_weeks"]
        part = df.iloc[te][["geoid", "week_idx", "week_start_date", TARGET]].copy()
        part["time_window"], part["spatial_fold"] = info["time_window"], k
        part["index"] = df["burden_percentile"].to_numpy()[te]
        part["recent_critical"] = (
            df["weeks_critical_prev4"].fillna(0).to_numpy()[te] + part["index"] / 1000
        )
        lr = logreg().fit(df.iloc[tr][MODEL_A_FEATURES], y[tr])
        part["logreg"] = lr.predict_proba(df.iloc[te][MODEL_A_FEATURES])[:, 1]

        in_tr, in_val, inner = inner_split(week_idx, fold, k, start, n_splits)
        fold_row = {
            **info,
            **inner,
            "n_train": len(tr),
            "n_test": len(te),
            "n_inner_train": len(in_tr),
            "n_inner_val": len(in_val),
        }
        for name, cols in MODELS.items():
            best, trials = tune_xgb(df[cols], y, in_tr, in_val, seed)
            model = fit_xgb(df.iloc[tr][cols], y[tr], best, seed)
            part[name] = model.predict_proba(df.iloc[te][cols])[:, 1]
            chosen[name].append(best)
            fold_row[f"{name}_params"] = best
            fold_row[f"{name}_trials"] = trials
        for s in SCORERS:
            fold_row[s] = ranking_metrics(part[TARGET], part[s])
        parts.append(part)
        folds.append(fold_row)
        print(
            f"  window {info['time_window']} fold {k}: "
            + "  ".join(f"{s} {fold_row[s]['pr_auc']:.3f}" for s in SCORERS)
            + f"  | a {best_str(chosen['xgb_a'][-1])}  ({time.time() - t0:.0f} s)"
        )
    oof = pd.concat(parts, ignore_index=True)

    summary = {
        "rows": len(df),
        "oof_rows": len(oof),
        "oof_base_rate": float(oof[TARGET].mean()),
        "pooled": {},
        "per_window": {},
        "chosen_params": {},
        "folds": folds,
    }
    print(
        f"\nOut-of-fold results ({len(oof):,} rows, base rate {oof[TARGET].mean():.3f}):"
    )
    print(
        f"  {'scorer':<16}{'PR-AUC':>8}{'ROC-AUC':>9}{'fold PR-AUC mean+-sd':>24}"
        f"{f'prec@{TOP_K_PER_WEEK}/wk':>14}"
    )
    for s in SCORERS:
        pooled = ranking_metrics(oof[TARGET], oof[s])
        pk = precision_at_k(oof, s, TARGET, TOP_K_PER_WEEK)
        fa = np.array([f[s]["pr_auc"] for f in folds])
        summary["pooled"][s] = {
            **pooled,
            f"precision_at_{TOP_K_PER_WEEK}_per_week": pk,
            "fold_pr_auc_mean": float(fa.mean()),
            "fold_pr_auc_sd": float(fa.std()),
        }
        print(
            f"  {s:<16}{pooled['pr_auc']:>8.3f}{pooled['roc_auc']:>9.3f}"
            f"{fa.mean():>15.3f} +- {fa.std():.3f}{pk:>14.3f}"
        )

    print("\nPooled PR-AUC by time window:")
    for w, wk in oof.groupby("time_window"):
        weeks = sorted(wk["week_start_date"].dt.strftime("%Y-%m-%d").unique())
        row = {s: float(average_precision_score(wk[TARGET], wk[s])) for s in SCORERS}
        summary["per_window"][int(w)] = {
            "weeks": [weeks[0], weeks[-1]],
            "base_rate": float(wk[TARGET].mean()),
            **row,
        }
        print(
            f"  {weeks[0]} to {weeks[-1]} (base {wk[TARGET].mean():.3f}): "
            + "  ".join(f"{s} {row[s]:.3f}" for s in SCORERS)
        )

    print("\nSettings chosen on inner validation (count of 15 outer folds):")
    for name, picks in chosen.items():
        counts = Counter(best_str(p, trees=False) for p in picks)
        trees = [p["n_estimators"] for p in picks]
        summary["chosen_params"][name] = {"counts": dict(counts), "trees": trees}
        print(
            f"  {name}: {dict(counts)}; trees median {int(np.median(trees))} "
            f"(range {min(trees)}-{max(trees)})"
        )

    OOF_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    oof.to_parquet(OOF_PATH, index=False)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote {OOF_PATH} and {SUMMARY_PATH}")
    return 0


def best_str(p: dict, trees: bool = True) -> str:
    s = f"d{p['max_depth']} mcw{p['min_child_weight']}"
    return f"{s} t{p['n_estimators']}" if trees else s


if __name__ == "__main__":
    raise SystemExit(main())
