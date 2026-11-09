"""Re-pull EPA AQS PM2.5 for one or more weeks into epa_pm25_raw.

Use after a week was skipped (for example 2026-01-12, two faulty monitors).
Pause spatial_transform first: DuckDB allows one writer at a time.

    PYTHONPATH=. python3 scripts/repull_pm25.py 2026-01-12 [2026-01-19]
"""

import sys
from datetime import datetime

from src.ingestion.epa_aqs import pull_pm25_range
from src.utils.config_loader import load_config

first = datetime.fromisoformat(sys.argv[1])
last = datetime.fromisoformat(sys.argv[2]) if len(sys.argv) > 2 else first
result = pull_pm25_range(load_config("los_angeles"), first, last)
print(f"Written: {result['weeks_written']}  Missing: {result['weeks_missing']}")
