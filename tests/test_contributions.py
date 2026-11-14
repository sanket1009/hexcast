"""Per-feature log-odds contributions (src/ml/contributions.py)."""

import numpy as np
import pandas as pd
import pytest

from src.ml.contributions import feature_contributions
from src.ml.tuning import make_logreg


@pytest.fixture
def fitted():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "a": rng.normal(10, 3, 300),
            "b": rng.normal(0, 1, 300),
            "c": rng.normal(5, 2, 300),
        }
    )
    y = (X["a"] / 3 + X["b"] + rng.normal(0, 1, 300) > 3.5).astype(int)
    X.loc[::17, "c"] = np.nan  # some missing values, filled by the median imputer
    return make_logreg().fit(X, y), X


def test_terms_add_up_to_model_log_odds(fitted):
    model, X = fitted
    rows = X.head(20).assign(geoid=[f"g{i}" for i in range(20)])
    out = feature_contributions(model, rows, ["a", "b", "c"])
    assert len(out) == 60
    intercept = model.steps[-1][1].intercept_[0]
    sums = out.groupby("geoid", sort=False)["contribution"].sum() + intercept
    expected = model.decision_function(X.head(20))
    assert np.allclose(sums.to_numpy(), expected, atol=1e-4)
    assert np.allclose(
        out.groupby("geoid", sort=False)["logit"].first(), expected, atol=1e-4
    )


def test_missing_values_are_flagged_and_scaled_not_raw(fitted):
    model, X = fitted
    rows = X.head(1).assign(geoid="g0", c=np.nan)
    out = feature_contributions(model, rows, ["a", "b", "c"]).set_index("feature")
    assert bool(out.loc["c", "imputed"]) and np.isnan(out.loc["c", "value"])
    assert not out.loc["a", "imputed"]
    # coef x (value - mean) / sd, not coef x raw value
    imp, sc, lr = (s for _, s in model.steps)
    j = 0
    scaled = (X.iloc[0, j] - sc.mean_[j]) / sc.scale_[j]
    assert out.loc["a", "contribution"] == pytest.approx(
        lr.coef_[0][j] * scaled, abs=1e-5
    )


def test_rejects_other_models():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(StandardScaler(), LogisticRegression())
    with pytest.raises(ValueError, match="imputer"):
        feature_contributions(model, pd.DataFrame({"geoid": ["x"]}), [])
