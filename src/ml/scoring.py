"""Metrics shared by the Phase 6 ML scripts."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def ranking_metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    return {
        "pr_auc": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
    }


def precision_at_k(df: pd.DataFrame, score: str, target: str, k: int) -> float:
    """Mean over weeks of the share of true positives among the top-k scored rows."""
    return float(
        np.mean(
            [wk.nlargest(k, score)[target].mean() for _, wk in df.groupby("week_idx")]
        )
    )
