"""
plot_healthcare_access.py
=========================
Two diagnostic maps for the healthcare access step:

  1. road_network.png  : the cached drive network (freeways highlighted),
                         ED hospitals in the city and buffer, county outline,
                         and the buffered bbox the network covers
  2. access_score.png  : 2SFCA access per tract (beds per 1,000), ER deserts
                         in red, unreachable tracts hatched

Usage (from the repo root):
    python scripts/plot_healthcare_access.py            # both maps
    python scripts/plot_healthcare_access.py --access   # access map only (fast)

Maps are saved under the object store's plots/healthcare/ folder.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import duckdb  # noqa: E402
import geopandas as gpd  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from shapely.geometry import box  # noqa: E402

from src.utils.config_loader import get_buffered_bbox, load_config  # noqa: E402

CITY = "los_angeles"
FREEWAYS = {"motorway", "motorway_link", "trunk", "trunk_link"}


def out_dir(config) -> Path:
    d = (
        Path(config["storage"]["local_object_dir"])
        / config["gcp"]["gcs_bucket"]
        / "plots"
        / "healthcare"
    )
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_tracts(config) -> gpd.GeoDataFrame:
    path = Path("data/processed/tract_geometries") / f"{config['full_fips']}.parquet"
    return gpd.read_parquet(path)[["geoid", "geometry"]].to_crs(config["crs_projected"])


def load_hospitals(config) -> gpd.GeoDataFrame:
    path = Path("data/processed/hospitals") / f"{config['full_fips']}.parquet"
    h = gpd.read_parquet(path)
    return h[h["has_ed"]].to_crs(config["crs_projected"])


def plot_network(config, tracts, hospitals, out: Path) -> None:
    import osmnx as ox

    hc = config["healthcare"]
    graph = (
        Path("data/processed/road_network")
        / f"{config['full_fips']}_{config['osmnx']['network_type']}_buffer{hc['buffer_km']:g}km.graphml"
    )
    print(f"Loading {graph} (a minute or two)...")
    G = ox.load_graphml(graph)
    edges = ox.graph_to_gdfs(G, nodes=False)[["highway", "geometry"]]
    del G
    edges["fwy"] = edges["highway"].apply(
        lambda h: bool(set(h if isinstance(h, list) else [h]) & FREEWAYS)
    )
    edges = edges.to_crs(config["crs_projected"])
    print(f"Plotting {len(edges):,} road segments...")

    county = tracts.dissolve()
    bb = gpd.GeoSeries(
        [box(*get_buffered_bbox(config, hc["buffer_km"]))], crs=config["crs"]
    ).to_crs(config["crs_projected"])

    fig, ax = plt.subplots(figsize=(14, 13), facecolor="white")
    edges[~edges["fwy"]].plot(ax=ax, color="#9aa3ad", linewidth=0.08, alpha=0.6)
    edges[edges["fwy"]].plot(ax=ax, color="#d9822b", linewidth=0.6)
    county.boundary.plot(ax=ax, color="#1f2a44", linewidth=1.4)
    bb.boundary.plot(ax=ax, color="#1f2a44", linewidth=1, linestyle="--")
    hospitals[~hospitals["in_city"]].plot(
        ax=ax, color="#2b8a3e", markersize=28, edgecolor="white", linewidth=0.6, zorder=5
    )
    hospitals[hospitals["in_city"]].plot(
        ax=ax, color="#c92a2a", markersize=28, edgecolor="white", linewidth=0.6, zorder=6
    )

    n_city = int(hospitals["in_city"].sum())
    n_buf = int((~hospitals["in_city"]).sum())
    ax.legend(
        handles=[
            Line2D([], [], color="#9aa3ad", lw=1.5, label="Drivable road"),
            Line2D([], [], color="#d9822b", lw=2, label="Freeway / trunk"),
            Line2D([], [], marker="o", ls="", color="#c92a2a", mec="white", ms=8,
                   label=f"LA County ED hospital ({n_city})"),
            Line2D([], [], marker="o", ls="", color="#2b8a3e", mec="white", ms=8,
                   label=f"Buffer ED hospital ({n_buf})"),
            Line2D([], [], color="#1f2a44", lw=1.4, label="LA County (tracts)"),
            Line2D([], [], color="#1f2a44", lw=1, ls="--",
                   label=f"Network extent (bbox + {hc['buffer_km']:g} km)"),
        ],
        loc="lower left", frameon=True, fontsize=10,
    )
    ax.set_title(
        f"Road network used for ER travel times: {len(edges):,} segments",
        fontsize=15, loc="left",
    )
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"Saved {out}")


def plot_access(config, tracts, hospitals, out: Path) -> None:
    con = duckdb.connect(config["storage"]["local_db_path"], read_only=True)
    schema = config["gcp"]["bigquery_dataset"]
    acc = con.execute(
        f"""
        SELECT geoid, access_score_er, scored, reachable, er_in_catchment, week_start_date
        FROM "{schema}".healthcare_access
        WHERE week_start_date = (SELECT max(week_start_date) FROM "{schema}".healthcare_access)
        """
    ).df()
    con.close()
    week = str(acc["week_start_date"].iloc[0])[:10]
    g = tracts.merge(acc, on="geoid", how="left")

    scored = g[g["scored"] & g["er_in_catchment"]]
    desert = g[g["scored"] & ~g["er_in_catchment"]]
    unreach = g[~g["reachable"].fillna(False)]
    nopop = g[g["reachable"].fillna(False) & ~g["scored"].fillna(False)]

    fig, ax = plt.subplots(figsize=(13, 13), facecolor="white")
    scored.plot(
        ax=ax, column="access_score_er", cmap="viridis", linewidth=0,
        legend=True, vmin=scored["access_score_er"].quantile(0.02),
        vmax=scored["access_score_er"].quantile(0.98),
        legend_kwds={"label": "ER beds per 1,000 residents (2SFCA, 30 min, Gaussian decay)",
                     "shrink": 0.55, "orientation": "horizontal", "pad": 0.02},
    )
    nopop.plot(ax=ax, color="#e9ecef", linewidth=0)
    desert.plot(ax=ax, color="#e03131", linewidth=0)
    unreach.plot(ax=ax, facecolor="white", edgecolor="#868e96", hatch="////", linewidth=0.4)
    hospitals[hospitals["in_city"]].plot(
        ax=ax, color="white", markersize=14, edgecolor="black", linewidth=0.6, zorder=5
    )

    # zoom to the mainland: islands sit far offshore and shrink everything
    main = g[g["reachable"].fillna(False)].total_bounds
    pad = 4000
    ax.set_xlim(main[0] - pad, main[2] + pad)
    ax.set_ylim(main[1] - pad, main[3] + pad)

    ax.legend(
        handles=[
            Patch(color="#e03131", label=f"No ER within 30 min ({len(desert)})"),
            Patch(facecolor="white", edgecolor="#868e96", hatch="////",
                  label=f"Unreachable by road ({len(unreach)})"),
            Patch(color="#e9ecef", label=f"No residents ({len(nopop)})"),
            Line2D([], [], marker="o", ls="", mfc="white", mec="black", ms=6,
                   label="LA County ED hospital"),
        ],
        loc="lower left", fontsize=10,
    )
    ax.set_title(f"Emergency department access by census tract, week of {week}",
                 fontsize=15, loc="left")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"Saved {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--access", action="store_true", help="access map only")
    ap.add_argument("--network", action="store_true", help="network map only")
    args = ap.parse_args()

    config = load_config(CITY)
    tracts = load_tracts(config)
    hospitals = load_hospitals(config)
    d = out_dir(config)

    if not args.access:
        plot_network(config, tracts, hospitals, d / "road_network.png")
    if not args.network:
        plot_access(config, tracts, hospitals, d / "access_score.png")


if __name__ == "__main__":
    main()
