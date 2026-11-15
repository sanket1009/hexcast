"""
Validation and pipeline health: can the numbers on the other pages be trusted?

    This week      drift checks, agreement with CalEnviroScreen, who lives in
                   CRITICAL tracts
    Every week     Kriging leave-one-out error, satellite coverage, and how
                   sure each week's CRITICAL list is
    Watch-list model  PR-AUC against simple rules, calibration, recall by
                   kind of tract (from eval_summary, src/evals/publish.py)
    Briefings      the accepted A1 retrieval and A2/A4/A5 briefing scores

Headings state what the data shows; every number comes from the API.
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from ui import api, fmt
from ui.charts import (
    confidence_mix_chart,
    equity_chart,
    pr_auc_chart,
    recall_chart,
    reliability_chart,
    weekly_chart,
)

# metric -> (label, what it measures and when it is flagged, value format)
# Thresholds and directions are those in src/ml/drift.py.
DRIFT = {
    "pm25_level": ("PM2.5 countywide level", "SD from the trailing 8 weeks; flagged beyond ±2", "{:+.2f} SD"),
    "pm25_shape": ("PM2.5 spatial pattern", "PSI against the trailing 8 weeks; flagged above 0.5", "{:.3f}"),
    "lst_level": ("Heat countywide level", "SD from the trailing 8 weeks; flagged beyond ±2", "{:+.2f} SD"),
    "lst_shape": ("Heat spatial pattern", "PSI against the trailing 8 weeks; flagged above 0.10", "{:.3f}"),
    "lst_coverage": ("Tracts with clear heat pixels", "Flagged below 95%", "{:.1%}"),
    "monitors_ratio": ("PM2.5 monitors reporting", "Ratio to the trailing median; flagged when off by 10%", "{:.2f}×"),
    "kriging_mae_ratio": ("Kriging error", "Ratio to the trailing median; flagged above 1.5×", "{:.2f}×"),
    "critical_jaccard": ("CRITICAL list overlap with last week", "Jaccard index; flagged below 0.45", "{:.2f}"),
}  # fmt: skip

IN_USE = "logreg"  # src/ml/retrain.py: logistic regression, Model A features, defaults
BASELINES = {"index", "recent_critical"}
# Short column names for the per-window table
WINDOW_LABELS = {
    "index": "Burden rule",
    "recent_critical": "Recent-CRITICAL rule",
    "logreg": "Model in use",
}
MODEL_LABELS = {
    "index": "Rule: burden percentile this week",
    "recent_critical": "Rule: CRITICAL in the last 4 weeks",
    "logreg": "Logistic regression (in use)",
    "logreg_tuned": "Logistic regression, tuned",
    "xgb_a": "XGBoost, feature set A",
    "xgb_b": "XGBoost, feature set B",
    "lgbm_c": "LightGBM, monotone",
}
REF_SHORT = {
    "logreg_tuned": "tuned logistic regression",
    "logreg": "logistic regression",
}
ATTRIBUTES = {
    "median_income": "Median household income",
    "low_income_pct": "Low-income residents",
    "no_vehicle_pct": "Households without a car",
    "limited_english_pct": "Limited-English households",
}
THIRDS = {"low": "lowest", "mid": "middle", "high": "highest"}

# metric -> (label, format); "%" = share of briefings or queries
BRIEFING_METRICS = {
    "faithfulness.numbers_match_data": ("Every number comes from the tract data", "%"),
    "faithfulness.urgency_matches": ("Urgency matches the burden class", "%"),
    "faithfulness.drivers_mentioned": ("Drivers named in the summary", "%"),
    "faithfulness.drivers_accounted": (
        "Each driver gets an action or a stated data gap",
        "%",
    ),
    "rules.no_source_named": ("Names no pollution or heat source", "%"),
    "rules.no_system_talk": ("No talk of the pipeline or model", "%"),
    "rules.no_jargon": ("No technical jargon", "%"),
    "rules.on_driver_actions": ("No actions for hazards the tract does not have", "%"),
    "rules.concrete_actions": ("Actions that start with a concrete verb", "%"),
    "rules.reading_grade.median": ("Reading grade, median (target under 12)", ".1f"),
    "specificity.driver_separation": ("Text difference between driver mixes", ".2f"),
    "specificity.group_separation": (
        "Text difference between vulnerable groups",
        ".2f",
    ),
    "specificity.urgency_separation": ("Text difference between urgency levels", ".2f"),
    "specificity.duplicate_source_sets": (
        "Cite the same pages as another briefing",
        "%",
    ),
}
RETRIEVAL_METRICS = {
    "recall@3": ("Relevant pages in the top 3 (what a briefing sees)", "%"),
    "recall@10": ("Relevant pages in the top 10", "%"),
    "mrr": ("Mean reciprocal rank of the first relevant page", ".2f"),
    "ndcg@10": ("nDCG@10", ".2f"),
    "negative_rejection": ("Unanswerable questions that correctly get nothing", "%"),
    "empty_rate": ("Answerable questions that get nothing", "%"),
}

week = st.session_state["week"]

try:
    val = api.validation(week)
    equity = api.equity(week)
except api.ApiError as exc:
    st.error(str(exc), title="Could not load validation results")
    st.stop()
try:
    ev = api.evaluation()  # not tied to a week
    ev_error = None
except api.ApiError as exc:
    ev, ev_error = None, str(exc).split(": ", 1)[-1]

drift = pd.DataFrame(val["drift_checks"])
weekly = pd.DataFrame(val["weekly"])
weekly["week_start_date"] = pd.to_datetime(weekly["week_start_date"])
this = weekly[weekly["week_start_date"] == pd.Timestamp(week)]


def _fmt(value: float, how: str) -> str:
    return f"{value:.0%}" if how == "%" else format(value, how)


# ------------------------------------------------------------------ headline

if drift.empty:
    st.header(f"No data checks recorded for {fmt.week_label(week)}")
    st.caption(
        "Drift checks run each week after the compound score; this week has none stored."
    )
else:
    flagged = drift[drift["status"].isin(["moderate", "major"])]
    if flagged.empty:
        st.header(
            f"All {len(drift)} data checks pass for {fmt.week_label(week)}"
            if len(drift) > 1
            else f"The data check passes for {fmt.week_label(week)}"
        )
    else:
        names = ", ".join(DRIFT.get(m, (m,))[0].lower() for m in flagged["metric"])
        st.header(
            f"{len(flagged)} of {len(drift)} data checks flag {fmt.week_label(week)}: {names}"
        )
    st.caption(
        "Checks compare this week with the trailing 8 weeks. They warn and never stop the "
        "pipeline; thresholds were set so each fires on roughly the most unusual 5 to 10% of weeks."
    )

week_tab, history_tab, model_tab, rag_tab = st.tabs(
    ["This week", "Every week", "Watch-list model", "Briefings and retrieval"]
)

# ----------------------------------------------------------------- this week

with week_tab:
    st.subheader("Data checks")
    if drift.empty:
        st.caption("None stored for this week.")
    else:
        rows = []
        for r in drift.itertuples():
            label, rule, how = DRIFT.get(r.metric, (r.metric, "", "{:.3f}"))
            rows.append(
                {
                    "Check": label,
                    "Value": "n/a"
                    if r.value is None or pd.isna(r.value)
                    else how.format(r.value),
                    "Status": str(r.status).capitalize(),
                    "Rule": rule,
                }
            )
        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            column_config={
                "Check": st.column_config.TextColumn(width="medium"),
                "Value": st.column_config.TextColumn(width=90),
                "Status": st.column_config.TextColumn(width=80),
                "Rule": st.column_config.TextColumn(width="large"),
            },
            alt="Table of this week's data drift checks",
        )

    left, right = st.columns([1, 1.2], gap="large")
    with left:
        st.subheader("Agreement with CalEnviroScreen")
        ces = val["ces_agreement"]
        if ces is None:
            st.caption("Too few tracts with a CalEnviroScreen score to compare.")
        else:
            c1, c2 = st.columns(2, gap="small")
            c1.metric(
                "Rank correlation",
                f"{ces['spearman']:.2f}",
                help=f"Spearman correlation of burden percentile with the CalEnviroScreen percentile, {ces['tracts']:,} tracts.",
                border=True,
            )
            c2.metric(
                "In CES top quarter",
                "n/a"
                if ces["critical_in_ces_top_quarter"] is None
                else f"{ces['critical_in_ces_top_quarter']:.0%}",
                help="Share of this week's CRITICAL tracts that CalEnviroScreen also puts in its top 25% of tracts.",
                border=True,
            )
            st.caption(
                "CalEnviroScreen is California's cumulative pollution and population index. The weights "
                "here were not tuned to match it: it measures a broader, slower-moving burden, so close "
                "but not full agreement is the expected result."
            )

    with right:
        all_row = equity[equity["group"] == "all"].iloc[0]
        rest = equity[equity["group"] != "all"]
        hi = rest.loc[rest["ratio_to_all"].idxmax()]
        lo = rest.loc[rest["ratio_to_all"].idxmin()]
        st.subheader(
            f"{hi['label']}: {hi['ratio_to_all']:.1f} times as likely as residents overall "
            "to live in a CRITICAL tract"
        )
        st.altair_chart(
            equity_chart(equity),
            alt="Bar chart of each group's share in CRITICAL tracts relative to all residents",
        )
        st.caption(
            f"{all_row['share_in_critical']:.1%} of all residents live in CRITICAL tracts this week. "
            f"Lowest ratio: {lo['label']} ({lo['ratio_to_all']:.2f}). "
            "Income thirds split tracts by median household income; car and English counts are households, "
            "the rest are people (ACS 2020-2024)."
        )

# ---------------------------------------------------------------- every week

with history_tab:
    worst = weekly.loc[weekly["kriging_loo_mae"].idxmax()]
    st.subheader(
        f"Kriging error was {weekly['kriging_loo_mae'].median():.2f} µg/m³ in the median week; "
        f"the worst, {fmt.week_label(worst['week_start_date'])}, reached {worst['kriging_loo_mae']:.2f}"
    )
    st.altair_chart(
        weekly_chart(
            weekly, "kriging_loo_mae", "LOO MAE (µg/m³)", week, ".2f",
            extra_tooltip=[alt.Tooltip("n_monitors:Q", title="Monitors")],
        ),
        alt="Line chart of weekly Kriging leave-one-out mean absolute error",
    )  # fmt: skip
    st.caption(
        "Leave-one-out: each monitor is predicted from the others with the variogram refit, "
        "so the error is honest. Spikes are weeks when one monitor reads far from its neighbors "
        "(fireworks, smoke). The dashed line marks the week picked in the sidebar"
        + (f" ({this['kriging_loo_mae'].iloc[0]:.2f})." if len(this) else ".")
    )

    left, right = st.columns(2, gap="large")
    with left:
        low_cov = int((weekly["lst_coverage"] < 0.95).sum())
        st.markdown(
            f"**Clear heat pixels covered under 95% of tracts in {low_cov} of {len(weekly)} weeks**"
        )
        st.altair_chart(
            weekly_chart(
                weekly, "lst_coverage", "Share of tracts", week, ".1%", domain=(0, 1)
            ),
            alt="Line chart of the share of tracts with a land surface temperature value each week",
        )
        st.caption(
            "Share of tracts with a land surface temperature value in the 32-day Landsat composite. "
            "When too few tracts have one (clouds), the index reuses the last good week's heat scores."
        )
    with right:
        mix = weekly[
            ["week_start_date", "critical_high", "critical_medium", "critical_low"]
        ]
        if len(this):
            t = this.iloc[0]
            n = int(t["critical_high"] + t["critical_medium"] + t["critical_low"])
            head = f"{int(t['critical_high'])} of {n} CRITICAL tracts this week are HIGH confidence"
        else:
            head = "How sure each week's CRITICAL list is"
        st.markdown(f"**{head}**")
        st.altair_chart(
            confidence_mix_chart(mix, week),
            alt="Stacked bars of CRITICAL tracts by confidence each week",
        )
        st.caption(
            "Confidence comes from 200 versions of the index with correlated Kriging error and "
            "±10% component weights. HIGH: CRITICAL in at least 90% of them; LOW: under half. "
            "Doubt comes mostly from PM2.5 interpolation."
        )

# ------------------------------------------------------------ watch-list model

with model_tab:
    if ev_error:
        st.info(f"No evaluation results published. {ev_error}")
    ml = ev["ml"] if ev else None
    if ml is None:
        if ev:
            st.caption(
                "The watch-list model has not been evaluated yet (scripts/ml_compare.py)."
            )
    else:
        scorers = pd.DataFrame(ml["scorers"])
        scorers["label"] = scorers["model"].map(MODEL_LABELS).fillna(scorers["model"])
        scorers["kind"] = scorers["model"].map(
            lambda m: "baseline" if m in BASELINES else "model"
        )
        by = scorers.set_index("model")
        in_use = by.loc[IN_USE] if IN_USE in by.index else by.loc[ml["reference"]]
        best_rule = by.loc[[m for m in BASELINES if m in by.index], "pr_auc"]
        k = ml["top_k"] or 50

        verdict = (
            "ranks future CRITICAL tracts better than either simple rule"
            if in_use["pr_auc"] > best_rule.max()
            else "does not beat the better simple rule"
        )
        st.subheader(
            f"The model {verdict}: PR-AUC {in_use['pr_auc']:.2f} against {best_rule.max():.2f}"
        )
        st.caption(
            f"Target: a tract that is not CRITICAL enters CRITICAL within 4 weeks ({ml['base_rate']:.1%} of "
            f"tract-weeks do). Scored out of fold on {ml['test_rows']:,} tract-weeks in {ml['test_windows']} "
            f"later test windows, with spatial blocks held out ({ml['weeks'].replace('..', ' to ')}). "
            f"Evaluated {fmt.week_label(ml['evaluated_at'][:10]) if ml['evaluated_at'] else 'n/a'}, "
            f"git {ml['git_sha']}."
        )
        left, right = st.columns([1.1, 1], gap="large")
        with left:
            st.altair_chart(
                pr_auc_chart(scorers, ml["base_rate"]),
                alt="Bar chart of PR-AUC for each model and baseline rule",
            )
            st.caption(
                f"Dashed line: a random ranking ({ml['base_rate']:.2f}). In each week's top {k}, "
                f"{in_use['precision_at_k']:.0%} of the model's tracts did turn CRITICAL, against "
                f"{by.loc[best_rule.idxmax(), 'precision_at_k']:.0%} for the better rule."
            )
        with right:
            comps = pd.DataFrame(ml["comparisons"])
            ref_label = REF_SHORT.get(ml["reference"], ml["reference"])
            st.markdown(
                f"**PR-AUC gain of the {ref_label} over each scorer (95% interval)**"
            )
            st.dataframe(
                pd.DataFrame(
                    {
                        "Compared with": comps["model"]
                        .map(MODEL_LABELS)
                        .fillna(comps["model"]),
                        "Gap": comps["diff"],
                        "Interval": [
                            f"{a:+.3f} to {b:+.3f}"
                            for a, b in zip(
                                comps["ci_low"], comps["ci_high"], strict=True
                            )
                        ],
                    }
                ).sort_values("Gap", ascending=False),
                hide_index=True,
                column_config={
                    "Gap": st.column_config.NumberColumn(format="%+.3f", width=70),
                },
                alt="Table of PR-AUC differences with bootstrap intervals",
            )
            n_boot = comps["n_boot"].dropna().max()
            st.caption(
                f"Spatial block bootstrap, {n_boot:,.0f} draws. The tuned and default logistic "
                "regressions tie, so the pipeline runs the default; boosted trees did not beat it."
            )
            windows = pd.DataFrame(
                [{"Test window": w["window"], "Base rate": w["base_rate"],
                  **{WINDOW_LABELS[m]: w["pr_auc"][m] for m in WINDOW_LABELS if m in w["pr_auc"]}}
                 for w in ml["windows"]]
            )  # fmt: skip
            st.markdown("**PR-AUC in each test window**")
            st.dataframe(
                windows,
                hide_index=True,
                column_config={
                    c: st.column_config.NumberColumn(format="%.2f")
                    for c in windows.columns
                    if c != "Test window"
                },
                alt="Table of PR-AUC per test window",
            )

        cal = ml["calibration"]
        left, right = st.columns([1, 1.1], gap="large")
        with left:
            rel = pd.DataFrame(cal["reliability"])
            worst_bin = (rel["predicted"] - rel["observed"]).abs().max()
            st.subheader(
                f"Predicted and observed rates are within {worst_bin * 100:.1f} points in every decile"
            )
            st.altair_chart(
                reliability_chart(rel),
                alt="Reliability chart of predicted against observed rates by probability decile",
            )
            st.caption(
                f"Deciles of out-of-fold probability from the {ref_label}. Brier score {cal['brier']:.3f}, "
                f"against {cal['brier_base_rate_only']:.3f} for predicting the base rate everywhere; "
                f"expected calibration error {cal['ece']:.3f}. Calibration is why the watch list can "
                "add probabilities into an expected count."
            )
        with right:
            eq = pd.DataFrame(ml["equity"])
            eq["attribute_label"] = (
                eq["attribute"].map(ATTRIBUTES).fillna(eq["attribute"])
            )
            summ = pd.DataFrame(ml["equity_summary"])
            first = summ.sort_values("recall_ratio_min_max").iloc[0]
            cells = eq[eq["attribute"] == first["attribute"]].set_index("tercile")
            lo_t = first["lowest_recall_tercile"]
            hi_t = cells["topk_recall"].idxmax()
            st.subheader(
                f"Recall is lowest in the {THIRDS.get(lo_t, lo_t)} third of tracts by "
                f"{ATTRIBUTES.get(first['attribute'], first['attribute']).lower()}"
            )
            st.altair_chart(
                recall_chart(eq),
                alt="Grouped bars of top-50 recall by thirds of tracts for four attributes",
            )
            gap = eq["calibration_gap"].abs().max()
            st.caption(
                f"{cells.loc[lo_t, 'topk_recall']:.0%} of entrants there are caught in the weekly top {k}, against "
                f"{cells.loc[hi_t, 'topk_recall']:.0%} in the {THIRDS.get(hi_t, hi_t)} third, where tracts turn "
                f"CRITICAL {'more' if cells.loc[hi_t, 'base_rate'] > cells.loc[lo_t, 'base_rate'] else 'less'} often ({cells.loc[lo_t, 'base_rate']:.1%} vs {cells.loc[hi_t, 'base_rate']:.1%} "
                f"of tract-weeks). The audit flags an attribute when the lowest third's recall is under 80% "
                f"of the highest's, or when probabilities miss by more than 0.02 in any third: "
                f"{int(summ['flag'].sum())} of {len(summ)} are flagged. Predicted and observed rates "
                f"stay within {gap:.3f} of each other in every third."
            )

# ----------------------------------------------------- briefings and retrieval

with rag_tab:
    left, right = st.columns(2, gap="large")
    for col, key, labels, title in (
        (left, "briefings", BRIEFING_METRICS, "Policy briefings"),
        (right, "retrieval", RETRIEVAL_METRICS, "Retrieval"),
    ):
        with col:
            st.subheader(title)
            suite = ev.get(key) if ev else None
            if not suite:
                st.caption("No accepted baseline published.")
                continue
            rows = [
                {
                    "Check": labels.get(m, (m, ""))[0],
                    "Score": _fmt(v, labels.get(m, (m, ".3f"))[1]),
                }
                for m, v in suite["metrics"].items()
            ]
            st.dataframe(
                pd.DataFrame(rows),
                hide_index=True,
                column_config={"Check": st.column_config.TextColumn(width="large")},
                height=38 + 35 * len(rows),
                alt=f"Table of {title.lower()} evaluation scores",
            )
            when = (
                fmt.week_label(suite["evaluated_at"][:10])
                if suite["evaluated_at"]
                else "n/a"
            )
            if key == "briefings":
                st.caption(
                    f"{suite['model']} briefings for the week of {fmt.week_label(suite['detail'])}, "
                    f"scored without model calls (A2 faithfulness, A4 rules, A5 specificity). "
                    f"Evaluated {when}, git {suite['git_sha']}. Briefings are pre-generated; "
                    "the dashboard never calls a language model."
                )
            else:
                st.caption(
                    "A1 on the hand-labeled gold set, scored per policy page on the production "
                    f"retriever's frozen rankings. Evaluated {when}, git {suite['git_sha']}."
                )
