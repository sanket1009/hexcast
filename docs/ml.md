# Forecasting model

The Watch list predicts which tracts that are not CRITICAL this week will be CRITICAL within the next 4 weeks. Back to the [README](../README.md).

## Target

`enters_critical_4w` is 1 when a tract that is not CRITICAL in week *t* is CRITICAL in any of weeks *t*+1 to *t*+4. Only non-CRITICAL tracts are eligible.

I started with "CRITICAL next week" and dropped it. Next week's class is almost a copy of this week's: PM2.5 enters the index as a 4-week mean and heat as a 32-day composite, so this week's percentile alone scored ROC-AUC 0.972 on it. A 4-week horizon reaches a heat window that shares no days with week *t*, and it answers the question a health department would ask: which tracts that are not on the list yet will be on it within a month?

The first year of data had no spring in the training folds, so a second year (June 2024 to June 2025) was backfilled before the final comparison. The panel has 217,862 eligible tract-weeks over 97 labeled weeks, with a base rate of 7.6%.

## Features

33 features, all known at the end of week *t*:

- **Index history:** percentile now, lags of 1, 2 and 4 weeks, changes, 4-week mean and spread, weeks CRITICAL among the previous 4.
- **Components now:** the PM2.5, heat, access and vulnerability z-scores, plus the newest single-week Kriged PM2.5 and its gap to the 4-week mean (the single week moves first).
- **Static tract data:** ACS groups, CDC PLACES outcomes, population, income, nearest ER.

Left out on purpose: week of year (with little history it acts as a week ID and leaks how many tracts enter later that week) and a year-over-year heat anomaly (it needs more years than exist). A second feature set added spatial lags from the 8 nearest tracts; k-nearest was used instead of H3 rings because ring membership swings from dozens of tracts downtown to none in the desert. It did not help.

## Validation without leakage

Random cross-validation would leak twice: neighboring tracts share PM2.5 and heat, and neighboring weeks share labels. The splitter holds out both at once:

- **Time:** the last three 8-week windows are test windows, each with an expanding training window before it and a 4-week embargo, so no training label looks into the test window.
- **Space:** within each test window, tracts are split into spatial blocks, and each block is predicted by a model that never saw it.

Every tract-week in the test windows is predicted exactly once: 53,904 out-of-fold predictions, base rate 8.1%. Hyperparameters were tuned in an inner loop on training data only.

## Results

| Scorer | PR-AUC | ROC-AUC | Precision in weekly top 50 |
|---|---|---|---|
| Rule: burden percentile this week | 0.405 | 0.884 | 0.589 |
| Rule: CRITICAL in the last 4 weeks | 0.434 | 0.897 | 0.650 |
| **Logistic regression (in use)** | **0.529** | **0.916** | **0.722** |
| Logistic regression, tuned | 0.530 | 0.917 | 0.721 |
| XGBoost, feature set A | 0.437 | 0.895 | 0.581 |
| XGBoost, feature set B (spatial lags) | 0.421 | 0.894 | 0.578 |
| LightGBM, monotone constraints | 0.454 | 0.899 | 0.651 |

PR-AUC is the headline metric because positives are rare (8%); a random ranking scores 0.08.

**Is the gap real?** A spatial block bootstrap (2,000 draws) gives the tuned logistic regression's PR-AUC gain with 95% intervals: +0.126 [+0.096, +0.156] over the percentile rule, +0.096 [+0.069, +0.122] over the recent-CRITICAL rule, +0.076 [+0.044, +0.102] over LightGBM, and +0.001 [-0.003, +0.006] over the untuned logistic regression. Tuning adds nothing, so production runs the defaults. The boosted trees were unstable too: XGBoost picked 10 different settings across 15 folds.

**Calibration.** Brier score 0.051 against 0.074 for predicting the base rate everywhere; expected calibration error 0.006; predicted and observed rates within 2.6 points in every decile. Calibration is what lets the dashboard add probabilities into an expected count ("169 tracts expected to enter CRITICAL").

**Equity audit.** Recall in the weekly top 50, by thirds of tracts:

| Attribute | Lowest third | Middle third | Highest third |
|---|---|---|---|
| Median household income | 0.23 | 0.17 | 0.09 |
| Share with low income | 0.12 | 0.16 | 0.23 |
| Households without a car | 0.12 | 0.18 | 0.24 |
| Limited-English households | 0.12 | 0.19 | 0.22 |

The model catches the fewest entrants among better-off tracts, where entering CRITICAL is also much rarer (2.0% of tract-weeks in the highest-income third against 17.3% in the lowest). Calibration gaps stay under 0.015 in every third. The audit's rule (flag when one third's recall is under 80% of another's) flags all four attributes, and the README lists this as a limitation rather than tuning it away.

## Explaining a prediction

Logistic regression's terms add up exactly to its log-odds: for each tract and feature, contribution = coefficient times the scaled value. These are stored per tract (`ml_contributions`, checked to sum to the model's decision function within 1e-6) and shown as "why the model flags it". No second explainer model is needed. The caveat on the dashboard: correlated features, such as the burden score and its own components, can share or offset each other's weight, so these are the model's arithmetic, not causes.

## Retraining and drift

- `model_retrain` holds out the last 8 labeled weeks. A candidate is promoted only if it beats both rules on them, and the current model is compared only on weeks after its own training data. Models are saved with a timestamped version, and the pointer to the current one is replaced atomically.
- After a rescore of past weeks, `ml_retrain.py --force` refits even if the current model looks fine on its old data. The year-one rescore moved the production holdout PR-AUC from 0.492 to 0.518.
- Every run logs to MLflow with the git SHA, so a number in this document can be traced to the code that produced it.
- Drift checks (8 per week, see [methods](methods.md#weekly-checks)) warn without blocking. All 8 passed for the week of June 22, 2026.

Code: `src/ml/` (features, spatial_cv, tuning, scoring, compare_stats, contributions, retrain, drift) and `scripts/ml_compare.py`, `scripts/ml_retrain.py`.
