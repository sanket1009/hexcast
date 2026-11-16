# Decisions

The choices behind Hexcast and the results that settled them. Most started as something that went wrong. The dashboard's Methods page links here. Back to the [README](../README.md).

## PM2.5 between 21 monitors

Tract PM2.5 is interpolated with ordinary Kriging from about 21 regulatory monitors a week. The first fit used a zero nugget, which makes the surface pass exactly through every monitor. It looked perfect at the monitors and said nothing honest about the space between them.

The nugget now has a floor set by the sampling noise of a weekly mean (with a finite population correction for the days a monitor reports). Leave-one-out validation refits the variogram in every fold, so the held-out monitor never shapes the model that predicts it. The error is reported as MAE, not R²: over 21 points, R² mostly reflects how far apart the monitors are. Median LOO MAE across 105 weeks: 1.38 µg/m³.

## Ocean pixels in the heat layer

Landsat surface temperature averaged over tract polygons pulled 39 coastal tracts about 14 °C too cool, because cold ocean pixels fell inside their boundaries. The coastal minimum went from 31.5 °C to 38.5 °C once water was masked with the Landsat QA water bit and a majority-vote land mask. Tracts that are only water get no heat value instead of an ocean temperature.

## Drive times that reach every road

Three bugs in emergency care access, each found by looking at a map:

- **A false desert.** One tract's origin snapped to a road segment cut off from the rest of the network, so no hospital could reach it. Origins now snap to the largest strongly connected component.
- **Mountain tracts dropping out.** Polygon centroids sat far from any road. Origins moved to the Census 2020 population-weighted center of each tract.
- **County lines.** 56 border tracts have their nearest emergency department in another county. A 30 km buffer of hospitals and population (46 more emergency departments, 1,228 more tracts) credits them correctly.

Public Overpass servers also failed on the buffered area, so the road network comes from a Geofabrik extract parsed locally.

## A composite that knows which way is bad

The first index was spatial PCA. It put healthcare access on the burden side (correlation +0.86 with the first component), so the three tracts with no emergency department within 30 minutes scored LOW. PCA finds directions of variance; it cannot be told which direction is bad.

The index is now a signed z-score composite: heat and PM2.5 raise burden, access lowers it, and a run fails validation if burden rises with access. PCA stays as a sensitivity check only.

Population vulnerability from the ACS joined later as a fourth equal-weight component. It is added to the hazards rather than multiplied with them: multiplying would hide a tract with extreme heat and average vulnerability, and the point of the index is that these risks stack.

## A four-week PM2.5 window

With weekly PM2.5, only 59% of CRITICAL tracts stayed CRITICAL the next week. Most of that churn was interpolation noise from about 21 monitors, not real change. The index uses a 4-week rolling mean of Kriged PM2.5 instead, and 81% stay. The weekly values are kept for the drift checks, and the forecasting model uses the newest single week as a feature because it moves first.

## Weights left alone after the CalEnviroScreen benchmark

The index was compared with CalEnviroScreen, California's cumulative pollution and population index, overall and component by component: rank correlation 0.67, and 78% of CRITICAL tracts in its top quarter.

I did not tune the weights to raise that number. CalEnviroScreen measures a broader, slower-moving burden (many more indicators, multi-year averages); Hexcast measures this week's heat and air with care access. Close but not full agreement is the expected result, and matching another index is not the goal. Equal weights stay, and the comparison is reported as it is.

## Confidence from correlated error

The first confidence layer drew Kriging error independently for each tract and said 98% of tracts could change class. That is wrong: a monitor reading too high pushes all its neighbors up together, so errors are correlated in space and partly cancel in rankings.

Each of the 200 draws now samples the full tract-by-tract error covariance from the week's variogram (a symmetric square-root factor, because eigenvector factors have arbitrary signs and a shared draw would cancel across weeks), with one draw shared by the 4 weeks in the PM2.5 window, plus ±10% on the component weights. The weights barely matter; PM2.5 interpolation drives almost all of the doubt.

The badge is about the decision, CRITICAL or not, rather than the exact class: the bands between ELEVATED and HIGH are narrow enough that exact-class agreement gave no HIGH badges at all to tracts in them.

## Drift checks that do not cry wolf

A population stability index on raw PM2.5 or heat flagged every one of the 97 checkable weeks as a major shift, because the whole county's level moves with the weather. An alarm that always rings gets ignored.

The checks now separate "did the whole county move" (level, in standard deviations from the trailing 8 weeks) from "did the spatial pattern change" (PSI on the within-week spatial pattern), with thresholds set from the 2024-2026 history so each fires on roughly the most unusual 5 to 10% of weeks. On that history the major flags land on real events: smoke weeks, cloud-masked heat weeks, and holiday Kriging errors. They warn and never stop the pipeline.

## Forecasting entry, not next week's class

"CRITICAL next week" was too easy: this week's percentile alone scored ROC-AUC 0.972, because the inputs are rolling windows. The model predicts whether a tract that is not CRITICAL now enters CRITICAL within 4 weeks, which is the question a health department would ask.

A second year of data was backfilled first, so the training folds include a spring. Cross-validation holds out later weeks and whole spatial blocks at once. Logistic regression beat tuned XGBoost and monotone LightGBM (PR-AUC 0.53 against 0.44 and 0.45), and its terms add up exactly to its log-odds, which is how the dashboard explains each prediction without a second model. Details in [ml.md](ml.md).

## Retrieval chosen by an eval

A hand-labeled gold set of policy questions decided the embedding model: production retrieval moved from all-MiniLM-L6-v2 to Qwen3-Embedding-0.6B, which ranked relevant pages higher and was the only model that rejected every off-topic question at a usable recall. A sweep on the same set set the distance cutoff, and a cross-encoder reranker tested on it did not make it into production. See [evals.md](evals.md#a1-retrieval).

The briefings are also checked without model calls for numbers that are not in the tract data, talk about the pipeline, and actions for hazards the tract does not have. An early prompt asked the model to explain its urgency level, and 24 of 25 sample briefings narrated how the pipeline set it.

## Hexes where people live

Area maps let large, empty desert tracts dominate the picture. The map paints H3 cells only where Census 2020 blocks have residents, each colored by the tract holding most of its people. Resolution 8 left 448 small, dense tracts (1.38M people) without a cell of their own; resolution 9 leaves 6 (10,250 people). Whole-tract fill is one click away to show the difference.

## Backfill scored out of order

The backfill DAG triggered weekly scoring runs as mapped tasks. Airflow's `max_active_runs=1` limits how many run at once but does not order them, so 30 of 105 weeks were scored before the weeks in their PM2.5 window, with shortened windows.

The fix has two parts. The backfill now triggers weeks in sorted order and waits for each to finish. And a guard in scoring raises if any week in the window has raw PM2.5 but no Kriged surface, so the bug cannot come back silently. Rescoring changed 6.5% of year-one CRITICAL tract-weeks. The forecasting baselines did not move, which showed the test windows had been clean, and the model's out-of-fold PR-AUC rose from 0.518 to 0.529.

## A private API behind a public dashboard

The dashboard is public; the API behind it is not. Cloud Run IAM allows only the dashboard's service account to call the API, with Google ID tokens. Anyone can look at the results without being able to query the warehouse directly. GitHub Actions deploys with Workload Identity Federation limited to this repository's main branch, so there are no stored service account keys to leak.

## Copy to BigQuery now, switch later

Airflow keeps writing DuckDB, and a sync script mirrors every table to BigQuery after each run (partitioned by week, clustered by city and tract). This put the dashboard on BigQuery without rewriting a working pipeline. A check script runs every API query against both stores and compares the answers; all 11 agree. Moving the pipeline itself to BigQuery is the next step.
