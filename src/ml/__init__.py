"""
Machine learning modules for CHBIndex (Phase 6).

Predicts which tracts that are not CRITICAL this week become CRITICAL within
the next 4 weeks (enters_critical_4w):
    features.py       tract-week panel, target, leakage checks
    spatial_cv.py     space-time blocked CV (held-out H3 blocks, later weeks, embargo)
    tuning.py         nested validation split; XGBoost, monotone LightGBM, logistic regression
    scoring.py        PR-AUC, ROC-AUC, weekly precision@k
    compare_stats.py  spatial block bootstrap CIs and DeLong's test
    report.py         calibration, equity audit, MLflow logging
"""
