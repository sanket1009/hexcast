"""Compute healthcare access for one week on the host and record its fingerprint.

The Airflow worker runs out of memory loading the road network GraphML. Run
this once from the local venv (with spatial_transform paused): later weeks
with the same inputs copy this result instead of loading the network.

    python3 scripts/seed_healthcare.py [YYYY-MM-DD]
"""

import sys
from datetime import datetime

from src.spatial.healthcare_access import compute_2sfca
from src.utils.config_loader import load_config

week = sys.argv[1] if len(sys.argv) > 1 else "2026-06-22"
df = compute_2sfca(load_config("los_angeles"), week_start=datetime.fromisoformat(week))
print(f"Seeded healthcare access for {week}: {len(df)} tracts")
