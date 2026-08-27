"""
Machine learning modules for CHBIndex.

Trains and evaluates models that predict next-week high-risk tracts:
    features.py     — Feature engineering with H3 spatial lag features
    spatial_cv.py   — Blocked spatial cross-validation splitter
    train.py        — Model A (XGBoost), B (spatial), C (LightGBM+constraints)
    evaluate.py     — DeLong's test A/B comparison, Brier score
    calibrate.py    — Platt scaling probability calibration
    equity_audit.py — Fairness analysis by race and income quartile
"""
