"""
check_dags.py
=============
DAG integrity check: every file in airflow/dags imports cleanly in Airflow
and every expected DAG is there. Run in CI with Airflow installed; tasks are
not executed and src/ is not imported (DAG files import it only inside tasks).

Usage:
    AIRFLOW__CORE__DAGS_FOLDER=airflow/dags python scripts/check_dags.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "airflow" / "dags"))  # as Airflow does for the DAGs folder

from airflow.dag_processing.dagbag import DagBag  # noqa: E402
from chbindex_common import CITIES  # noqa: E402

EXPECTED = ["ingest_weekly", "spatial_transform", "alert_dispatch", "backfill"]


def main() -> int:
    bag = DagBag(dag_folder=str(ROOT / "airflow" / "dags"))
    problems = [f"{path}: {err.strip().splitlines()[-1]}" for path, err in bag.import_errors.items()]

    expected = {f"{name}_{city}" for name in EXPECTED for city in CITIES}
    missing = sorted(expected - set(bag.dags))
    problems += [f"missing DAG: {dag_id}" for dag_id in missing]

    for dag_id, dag in sorted(bag.dags.items()):
        if not dag.tags or "chbindex" not in dag.tags:
            problems.append(f"{dag_id}: no 'chbindex' tag")
        print(f"ok  {dag_id}: {len(dag.tasks)} tasks")

    if problems:
        print("\n".join(["", "DAG check failed:"] + problems))
        return 1
    print(f"\nAll {len(bag.dags)} DAGs imported; {len(expected)} expected DAGs present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
