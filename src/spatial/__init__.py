"""
Spatial computation modules for CHBIndex.

Transforms raw ingested data into spatial signals per census tract:
    kriging.py          — PM2.5 interpolation from sparse EPA monitors
    lst_processing.py   — LST zonal statistics from Landsat rasters
    healthcare_access.py — 2SFCA travel-time healthcare access scores
    compound_score.py   — Spatial PCA compound burden index
    hotspots.py         — Moran's I + Gi* hotspot detection
"""
