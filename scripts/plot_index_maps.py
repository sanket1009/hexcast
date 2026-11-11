"""
plot_index_maps.py
==================
Two README maps for the Phase 2 index validation, from stored weeks:

  1. critical_persistence.png : how many weeks each tract was CRITICAL.
     Persistent (at least half of the weeks) vs churning (some weeks)
     vs never. A program can plan outreach around the persistent set.
  2. top_decile_vs_ces.png    : our top 10% by year-average burden vs the
     CalEnviroScreen 5.0 top 10%. Shows where the two indexes agree and
     where the access component moves our list (east San Gabriel Valley,
     San Pedro) away from CES (central LA, Compton, El Monte).

Usage (from the repo root, after rescoring):
    python scripts/plot_index_maps.py

Maps are saved under the object store's plots/index/ folder.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import duckdb  # noqa: E402
import geopandas as gpd  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from src.evals.index_validation import top_decile  # noqa: E402
from src.utils.config_loader import load_config  # noqa: E402

CITY = "los_angeles"
ISLANDS = "06037599"  # Catalina and San Clemente tract prefix
NEVER = "#e9ecef"


def out_dir(config) -> Path:
    d = (
        Path(config["storage"]["local_object_dir"])
        / config["gcp"]["gcs_bucket"]
        / "plots"
        / "index"
    )
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_tracts(config) -> gpd.GeoDataFrame:
    path = Path("data/processed/tract_geometries") / f"{config['full_fips']}.parquet"
    return gpd.read_parquet(path)[["geoid", "geometry"]].to_crs(config["crs_projected"])


def query(config, sql: str):
    con = duckdb.connect(config["storage"]["local_db_path"], read_only=True)
    schema = config["gcp"]["bigquery_dataset"]
    df = con.execute(sql.replace("{schema}", f'"{schema}"')).df()
    con.close()
    return df


def mainland(ax, g: gpd.GeoDataFrame) -> None:
    """Zoom to the mainland: the islands sit far offshore and shrink the map."""
    b = g[~g["geoid"].str.startswith(ISLANDS)].total_bounds
    pad = 4000
    ax.set_xlim(b[0] - pad, b[2] + pad)
    ax.set_ylim(b[1] - pad, b[3] + pad)
    ax.set_axis_off()


def plot_persistence(config, tracts, out: Path) -> None:
    df = query(
        config,
        """
        SELECT geoid,
            SUM(CASE WHEN burden_class = 'CRITICAL' THEN 1 ELSE 0 END) AS weeks_critical,
            COUNT(*) AS n_weeks,
            MIN(week_start_date) AS first_week,
            MAX(week_start_date) AS last_week
        FROM {schema}.compound_scores
        GROUP BY geoid
        """,
    )
    n_weeks = int(df["n_weeks"].max())
    span = f"{str(df['first_week'].min())[:10]} to {str(df['last_week'].max())[:10]}"
    g = tracts.merge(df, on="geoid", how="left")
    persistent = g[g["weeks_critical"] >= n_weeks / 2]
    churning = g[(g["weeks_critical"] > 0) & (g["weeks_critical"] < n_weeks / 2)]
    never = g[g["weeks_critical"].fillna(0) == 0]

    fig, ax = plt.subplots(figsize=(13, 13), facecolor="white")
    never.plot(ax=ax, color=NEVER, linewidth=0)
    churning.plot(
        ax=ax,
        column="weeks_critical",
        cmap="Oranges",
        vmin=-int(n_weeks / 4),  # keep 1-week tracts visible against grey
        vmax=n_weeks / 2,
        linewidth=0,
    )
    persistent.plot(ax=ax, color="#a61e4d", linewidth=0)
    mainland(ax, g)
    ax.legend(
        handles=[
            Patch(
                color="#a61e4d",
                label=f"Persistent: CRITICAL in at least half of weeks ({len(persistent):,})",
            ),
            Patch(
                color="#f08c00",
                label=f"Churning: CRITICAL in some weeks, darker = more ({len(churning):,})",
            ),
            Patch(color=NEVER, label=f"Never CRITICAL ({len(never):,})"),
        ],
        loc="lower left",
        fontsize=10,
    )
    ax.set_title(
        f"Weeks each tract was CRITICAL (top 10% burden), {n_weeks} weeks, {span}",
        fontsize=15,
        loc="left",
    )
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"Saved {out}")


def plot_vs_ces(config, tracts, out: Path) -> None:
    df = query(
        config,
        """
        SELECT s.geoid, AVG(s.raw_burden) AS burden, ANY_VALUE(c.ces_score) AS ces_score
        FROM {schema}.compound_scores s
        JOIN {schema}.calenviroscreen c USING (geoid)
        GROUP BY s.geoid
        """,
    ).dropna(subset=["ces_score"])
    ours = top_decile(df["burden"], df["geoid"])
    ces = top_decile(df["ces_score"], df["geoid"])
    both, ours_only, ces_only = ours & ces, ours - ces, ces - ours

    g = tracts.copy()
    groups = [
        (both, "#7048e8", f"Both top 10% ({len(both)})"),
        (ours_only, "#e8590c", f"CHBIndex top 10% only ({len(ours_only)})"),
        (ces_only, "#1c7ed6", f"CalEnviroScreen 5.0 top 10% only ({len(ces_only)})"),
    ]
    fig, ax = plt.subplots(figsize=(13, 13), facecolor="white")
    g.plot(ax=ax, color=NEVER, linewidth=0)
    for ids, color, _ in groups:
        g[g["geoid"].isin(ids)].plot(ax=ax, color=color, linewidth=0)
    mainland(ax, g)
    ax.legend(
        handles=[Patch(color=c, label=label) for _, c, label in groups],
        loc="lower left",
        fontsize=10,
    )
    ax.set_title(
        f"Top 10% tracts: CHBIndex year average vs CalEnviroScreen 5.0 "
        f"({len(df):,} tracts with a CES score)",
        fontsize=15,
        loc="left",
    )
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"Saved {out}")


def main() -> None:
    config = load_config(CITY)
    tracts = load_tracts(config)
    d = out_dir(config)
    plot_persistence(config, tracts, d / "critical_persistence.png")
    plot_vs_ces(config, tracts, d / "top_decile_vs_ces.png")


if __name__ == "__main__":
    main()
