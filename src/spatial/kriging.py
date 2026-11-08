"""
kriging.py
==========
Ordinary Kriging interpolation of PM2.5 from sparse EPA monitor
locations to all census tract centroids in the configured city.

This module solves the fundamental spatial data problem in the
compound burden index: LA County has 8 regulatory PM2.5 monitors for
2,498 census tracts. Kriging learns how PM2.5 similarity decays with
distance (the variogram) and uses it to estimate every tract centroid.
Monitors from neighboring counties within a buffer (see epa_aqs.py)
are included to give the variogram enough pairs and to cover the
county edges.

Method: Ordinary Kriging (PyKrige) with our own variogram fit
  - Empirical variogram: max lag = half the max pair distance,
    equal-count bins with >= spatial.kriging_min_pairs_per_bin pairs
  - Spherical model fit by weighted least squares (weights = pairs)
  - nugget/psill/range handed to PyKrige as fixed parameters
  - Variogram plot saved every run (plots/variogram/)
  - Leave-one-out CV, refitting the variogram in every fold

Why Ordinary Kriging over IDW or Nearest Neighbor:
  - IDW assumes correlation decays with 1/distance — arbitrary
  - Kriging learns the actual decay from the data (variogram)
  - Kriging is BLUE: Best Linear Unbiased Estimator
  - Kriging provides uncertainty estimates (variance) per prediction

Input:
    BigQuery table: epa_pm25_raw (monitor readings)
    BigQuery table: tract_geometries (tract centroids)

Output:
    BigQuery table: pm25_kriged (one row per tract per week)
    Columns: geoid, pm25_kriged, pm25_kriged_variance,
             week_start_date, city_fips

Usage:
    from src.spatial.kriging import run_kriging
    df = run_kriging(config)
"""

import io
import warnings
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from pykrige.ok import OrdinaryKriging
from scipy.optimize import OptimizeWarning, curve_fit
from scipy.spatial.distance import pdist

from src.utils.bigquery import REPLACE_PARTITION, read_query, write_dataframe
from src.utils.gcs import upload_bytes
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def run_kriging(
    config: dict[str, Any],
    week_start: datetime | None = None,
) -> pd.DataFrame:
    """
    Interpolate PM2.5 from EPA monitors to all census tract centroids.

    Reads monitor readings and tract centroids from BigQuery,
    fits an Ordinary Kriging model, predicts PM2.5 at all tract
    centroids, validates with leave-one-out CV, and writes results
    to BigQuery.

    Parameters
    ----------
    config : dict
        Merged pipeline config from config_loader.load_config().
        Uses: spatial.kriging_variogram_model, spatial.kriging_n_jobs,
              full_fips, crs_projected

    week_start : datetime, optional
        Week to run Kriging for. Defaults to most recent Monday.

    Returns
    -------
    pd.DataFrame
        One row per tract with columns:
        geoid, pm25_kriged, pm25_kriged_variance,
        pm25_percentile, week_start_date, city_fips

    Examples
    --------
    >>> config = load_config("los_angeles")
    >>> df = run_kriging(config)
    >>> len(df)
    2310
    >>> df["pm25_kriged"].describe()
    count    2310.000000
    mean       13.24...
    std         3.81...
    min         8.12...
    max        28.94...
    """
    if week_start is None:
        today = datetime.now()
        week_start = today - timedelta(days=today.weekday())
        week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)

    full_fips = config["full_fips"]
    week_str = week_start.strftime("%Y-%m-%d")

    logger.info(f"Running PM2.5 Kriging for {config['city_name']} week of {week_str}")

    # Step 1 — Load monitor readings from BigQuery
    monitors_df = _load_monitor_readings(config, week_str)
    logger.info(f"Loaded {len(monitors_df)} monitor readings for week of {week_str}")

    # Step 2 — Load tract centroids from BigQuery
    tracts_df = _load_tract_centroids(config)
    logger.info(f"Loaded {len(tracts_df):,} tract centroids")

    # Step 3 — Project coordinates to UTM for distance calculations
    # Kriging requires distances in meters — not degrees
    # Using degrees produces wrong variograms because
    # 1 degree longitude ≠ 1 degree latitude in distance
    monitors_utm, tracts_utm = _project_to_utm(
        monitors_df, tracts_df, config
    )

    # Step 4 — Empirical variogram, weighted spherical fit, plot
    bins, params = _variogram_for(monitors_utm, config)
    logger.info(
        "Empirical variogram bins:\n"
        + bins.assign(lag_km=(bins["lag_m"] / 1000).round(1))
              [["lag_km", "semivariance", "n_pairs"]].round(3).to_string(index=False)
    )
    logger.info(
        f"Nugget floor from sampling noise: {params['nugget_floor']:.3f}. "
        f"Fitted spherical variogram: nugget={params['nugget']:.3f}, "
        f"sill={params['nugget'] + params['psill']:.3f}, "
        f"range={params['range'] / 1000:.1f} km"
    )
    _save_variogram_plot(bins, params, week_str, config)

    # Step 5 — Krige every tract centroid with the fixed variogram
    kriged_df = _fit_and_predict(monitors_utm, tracts_utm, params)

    # Step 5b — Leave-one-out cross-validation
    cv = _leave_one_out_cv(monitors_utm, config)
    cv_r2 = cv["r2"]

    # Step 6 — Join GEOID back to kriged predictions
    # tracts_utm still has geoid column — merge on index
    result_df = pd.DataFrame({
        "geoid": tracts_df["geoid"].values,
        "pm25_kriged": kriged_df["pm25_kriged"].values,
        "pm25_kriged_variance": kriged_df["pm25_kriged_variance"].values,
    })

    # Step 7 — Compute percentile rank within city
    # Used for compound score normalization
    result_df["pm25_percentile"] = (
        result_df["pm25_kriged"]
        .rank(pct=True)
        .mul(100)
        .round(1)
    )

    # Step 8 — Add metadata
    result_df["week_start_date"] = week_start.date()
    result_df["city_fips"] = full_fips
    result_df["cv_r2"] = round(cv["r2"], 4)
    result_df["cv_r2_city"] = round(cv["r2_city"], 4)
    result_df["cv_mae"] = round(cv["mae"], 4)
    result_df["n_monitors"] = len(monitors_utm)
    result_df["variogram_nugget"] = round(params["nugget"], 4)
    result_df["variogram_nugget_floor"] = round(params["nugget_floor"], 4)
    result_df["variogram_psill"] = round(params["psill"], 4)
    result_df["variogram_range_m"] = round(params["range"], 1)

    # Step 9 — Validate output
    _validate_output(result_df, len(tracts_df))

    # Step 10 — Write to BigQuery
    write_dataframe(result_df, "pm25_kriged", config, write_disposition=REPLACE_PARTITION)

    logger.info(
        f"Kriging complete: {len(result_df):,} tracts, "
        f"PM2.5 range {result_df['pm25_kriged'].min():.1f}–"
        f"{result_df['pm25_kriged'].max():.1f} μg/m³, "
        f"CV R²={cv_r2:.3f}"
    )

    return result_df


# ---------------------------------------------------------------------------
# Private functions
# ---------------------------------------------------------------------------

def _load_monitor_readings(
    config: dict[str, Any],
    week_str: str,
) -> pd.DataFrame:
    """
    Load EPA PM2.5 monitor readings for the current week from BigQuery.

    Returns monitor locations (lat/lon) and their weekly mean PM2.5.
    These are the known values Kriging interpolates from.

    Parameters
    ----------
    config : dict
        Pipeline config.
    week_str : str
        Week start date as 'YYYY-MM-DD' string.

    Returns
    -------
    pd.DataFrame
        Monitor readings with columns:
        monitor_id, latitude, longitude, pm25_mean, pm25_std,
        reading_count, in_city
    """
    query = """
        SELECT
            monitor_id,
            latitude,
            longitude,
            pm25_mean,
            pm25_std,
            reading_count,
            in_city
        FROM {dataset}.epa_pm25_raw
        WHERE
            week_start_date = '{week}'
            AND city_fips = '{fips}'
            AND pm25_mean IS NOT NULL
            AND latitude IS NOT NULL
            AND longitude IS NOT NULL
        ORDER BY monitor_id
    """.format(
        dataset="{dataset}",  # substituted by read_query
        week=week_str,
        fips=config["full_fips"],
    )

    df = read_query(query, config)

    if len(df) < 3:
        raise ValueError(
            f"Need at least 3 monitors for Kriging, got {len(df)}. "
            f"Check EPA AQS data for week {week_str}."
        )

    return df


def _load_tract_centroids(config: dict[str, Any]) -> pd.DataFrame:
    """
    Load census tract centroids from BigQuery.

    These are the prediction locations — Kriging predicts PM2.5
    at each centroid. Centroids were computed in census_tiger.py
    using UTM projection for accuracy.

    Parameters
    ----------
    config : dict
        Pipeline config.

    Returns
    -------
    pd.DataFrame
        Tract centroids with columns: geoid, centroid_lat, centroid_lon
    """
    query = """
        SELECT
            geoid,
            centroid_lat,
            centroid_lon
        FROM {dataset}.tract_geometries
        WHERE
            city_fips = '{fips}'
            AND centroid_lat IS NOT NULL
            AND centroid_lon IS NOT NULL
        ORDER BY geoid
    """.format(
        dataset="{dataset}",
        fips=config["full_fips"],
    )

    df = read_query(query, config)

    if df.empty:
        raise ValueError(
            f"No tract centroids found for FIPS {config['full_fips']}. "
            f"Run census_tiger ingestion first."
        )

    return df


def _project_to_utm(
    monitors_df: pd.DataFrame,
    tracts_df: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Project monitor and tract coordinates from WGS84 to UTM.

    Kriging computes distances between points. In geographic
    coordinates (degrees), distances are distorted — 1 degree
    of longitude is ~95km in LA but ~111km near the equator,
    and these units are incompatible with each other.

    UTM (Universal Transverse Mercator) projects to a flat plane
    with meters as units — consistent distances in all directions.
    LA uses UTM Zone 11N (EPSG:26911).
    Chicago uses UTM Zone 16N (EPSG:26916).

    The correct UTM zone comes from config["crs_projected"] —
    set per city in los_angeles.yaml / chicago.yaml.

    Parameters
    ----------
    monitors_df : pd.DataFrame
        Monitor readings with latitude/longitude in WGS84.
    tracts_df : pd.DataFrame
        Tract centroids with centroid_lat/centroid_lon in WGS84.
    config : dict
        Pipeline config containing crs_projected.

    Returns
    -------
    tuple of (monitors_utm, tracts_utm)
        Both DataFrames with x_utm, y_utm columns added (meters).
    """
    import pyproj

    crs_projected = config["crs_projected"]

    # Create transformer: WGS84 → UTM
    # always_xy=True ensures consistent x=longitude, y=latitude order
    transformer = pyproj.Transformer.from_crs(
        "EPSG:4326",
        crs_projected,
        always_xy=True,
    )

    logger.debug(f"Projecting coordinates to {crs_projected}")

    # Project monitor coordinates
    monitors_utm = monitors_df.copy()
    monitors_utm["x_utm"], monitors_utm["y_utm"] = transformer.transform(
        monitors_df["longitude"].values,
        monitors_df["latitude"].values,
    )

    # Project tract centroid coordinates
    tracts_utm = tracts_df.copy()
    tracts_utm["x_utm"], tracts_utm["y_utm"] = transformer.transform(
        tracts_df["centroid_lon"].values,
        tracts_df["centroid_lat"].values,
    )

    logger.debug(
        f"Monitor UTM range: x={monitors_utm['x_utm'].min():.0f}–"
        f"{monitors_utm['x_utm'].max():.0f}m, "
        f"y={monitors_utm['y_utm'].min():.0f}–"
        f"{monitors_utm['y_utm'].max():.0f}m"
    )

    return monitors_utm, tracts_utm


def _empirical_variogram(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    min_pairs_per_bin: int,
) -> pd.DataFrame:
    """
    Compute the empirical variogram from monitor pairs.

    For every pair of monitors: distance h between them and
    semivariance 0.5 * (z_i - z_j)^2. Pairs farther apart than half
    the maximum pair distance are dropped: long lags are made only of
    edge-to-edge pairs and are unreliable.

    Pairs are then grouped into equal-count bins (quantiles of h)
    rather than PyKrige's equal-width bins, so every bin holds at
    least min_pairs_per_bin pairs. Each bin's point is the mean
    distance and mean semivariance of its pairs.

    Returns
    -------
    pd.DataFrame
        One row per bin: lag_m, semivariance, n_pairs.
    """
    coords = np.column_stack([x, y])
    h = pdist(coords)                                   # meters
    gamma = 0.5 * pdist(z.reshape(-1, 1), "sqeuclidean")

    max_lag = h.max() / 2
    keep = h <= max_lag
    h, gamma = h[keep], gamma[keep]

    n_bins = len(h) // min_pairs_per_bin
    if n_bins < 3:
        # A spherical model has 3 parameters; fewer than 3 points
        # cannot pin it down. Accept smaller bins and say so.
        n_bins = 3
        logger.warning(
            f"Only {len(h)} pairs within max lag; using 3 bins with "
            f"~{len(h) // 3} pairs each (below {min_pairs_per_bin})"
        )

    edges = np.quantile(h, np.linspace(0, 1, n_bins + 1))
    bin_idx = np.clip(np.searchsorted(edges, h, side="right") - 1, 0, n_bins - 1)

    bins = (
        pd.DataFrame({"bin": bin_idx, "h": h, "gamma": gamma})
        .groupby("bin")
        .agg(lag_m=("h", "mean"), semivariance=("gamma", "mean"),
             n_pairs=("h", "size"))
        .reset_index(drop=True)
    )
    return bins


def _spherical(h: np.ndarray, psill: float, range_m: float, nugget: float) -> np.ndarray:
    """Spherical model, same form PyKrige uses."""
    r = h / range_m
    return np.where(
        h < range_m,
        nugget + psill * (1.5 * r - 0.5 * r ** 3),
        nugget + psill,
    )


def _nugget_floor(monitors: pd.DataFrame) -> float:
    """
    Smallest nugget the data justifies: the sampling noise of a weekly mean.

    A monitor that samples n of the week's 7 days estimates the weekly
    mean with variance s^2 / n * (1 - n/7), where s is the day-to-day std.
    The (1 - n/7) finite population correction makes a full 7-day week
    exactly 0: it is the weekly mean, not an estimate. Only 1-in-3-day
    samplers add noise. Averaged over monitors, this is noise present even
    between two monitors at the same spot, which is what the nugget
    represents. Instrument error is not included, so this is a minimum.

    The closest variogram bin is ~19 km out, so the fit alone cannot see
    this and would otherwise collapse the nugget to 0.
    """
    days = 7
    n = monitors["reading_count"].clip(upper=days)
    var = (monitors["pm25_std"] ** 2 / n) * (1 - n / days)
    # Full weeks are 0 even when std is missing; a single reading has no
    # std to estimate from and is left out
    var = var.where(n < days, 0.0)
    var = var[var.notna()]
    return float(var.mean()) if len(var) else 0.0


def _fit_variogram(bins: pd.DataFrame, nugget_floor: float = 0.0) -> dict[str, float]:
    """
    Fit a spherical model to the binned variogram by weighted least squares.

    Weights = number of pairs per bin, so a bin backed by 40 pairs pulls
    the curve harder than one backed by 30. curve_fit takes sigma, the
    per-point standard deviation, so sigma = 1 / sqrt(n_pairs).

    Returns
    -------
    dict
        {"psill", "range", "nugget"} in the form PyKrige accepts.
    """
    lag = bins["lag_m"].values
    sv = bins["semivariance"].values

    # Starting guesses: nugget near the first bin, sill near the top,
    # range in the middle of the lags
    # Nugget may not go below the measured sampling noise
    # (kept under the top bin so the bounds stay valid)
    floor = min(nugget_floor, sv.max() * 0.9)

    p0 = [max(sv.max() - sv.min(), 1e-6), lag.mean(), max(sv.min() * 0.5, floor * 1.1 + 1e-9)]
    lower = [0, lag.min() * 0.5, floor]
    upper = [sv.max() * 3, lag.max() * 3, sv.max()]
    p0 = [min(max(v, lo + 1e-9), hi - 1e-9) for v, lo, hi in zip(p0, lower, upper)]

    # A parameter resting on its bound (nugget at the floor is expected)
    # leaves no covariance to estimate; the fit itself is still fine
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        (psill, range_m, nugget), _ = curve_fit(
            _spherical, lag, sv,
            p0=p0, bounds=(lower, upper),
            sigma=1 / np.sqrt(bins["n_pairs"].values),
            maxfev=10_000,
        )
    return {"psill": float(psill), "range": float(range_m), "nugget": float(nugget)}


def _save_variogram_plot(
    bins: pd.DataFrame,
    params: dict[str, float],
    week_str: str,
    config: dict[str, Any],
) -> str:
    """
    Plot the binned variogram and fitted curve; store it as a PNG.

    Dot size = pairs per bin. Goes through upload_bytes, so it lands in
    data/object_store locally and in the city's bucket on GCP.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(
        bins["lag_m"] / 1000, bins["semivariance"],
        s=bins["n_pairs"] * 4, alpha=0.7, label="Empirical (size = pairs)",
    )
    for _, row in bins.iterrows():
        ax.annotate(int(row["n_pairs"]), (row["lag_m"] / 1000, row["semivariance"]),
                    textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)

    h_line = np.linspace(0, bins["lag_m"].max() * 1.1, 200)
    ax.plot(h_line / 1000, _spherical(h_line, **{
        "psill": params["psill"], "range_m": params["range"], "nugget": params["nugget"]
    }), color="crimson", label="Spherical fit")

    ax.axhline(params["nugget"] + params["psill"], ls=":", color="grey")
    ax.axvline(params["range"] / 1000, ls=":", color="grey")
    ax.set_xlabel("Distance between monitors (km)")
    ax.set_ylabel("Semivariance (μg/m³)²")
    ax.set_title(
        f"PM2.5 variogram, {config['city_name']}, week of {week_str}\n"
        f"nugget={params['nugget']:.2f} (floor {params['nugget_floor']:.2f})  sill={params['nugget'] + params['psill']:.2f}  "
        f"range={params['range'] / 1000:.1f} km"
    )
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(loc="lower right")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)

    path = f"plots/variogram/variogram_{week_str}.png"
    upload_bytes(buf.getvalue(), path, config, content_type="image/png")
    logger.info(f"Saved variogram plot to {path}")
    return path


def _variogram_for(monitors_utm: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, float]]:
    """Empirical variogram + spherical fit for one set of monitors."""
    bins = _empirical_variogram(
        monitors_utm["x_utm"].values,
        monitors_utm["y_utm"].values,
        monitors_utm["pm25_mean"].values,
        config["spatial"].get("kriging_min_pairs_per_bin", 30),
    )
    floor = _nugget_floor(monitors_utm)
    params = _fit_variogram(bins, floor)
    params["nugget_floor"] = floor
    return bins, params


def _pykrige_params(params: dict[str, float]) -> dict[str, float]:
    """Only the keys PyKrige's spherical model accepts."""
    return {k: params[k] for k in ("psill", "range", "nugget")}


def _fit_and_predict(
    monitors_utm: pd.DataFrame,
    tracts_utm: pd.DataFrame,
    params: dict[str, float],
) -> pd.DataFrame:
    """
    Ordinary Kriging at tract centroids with a fixed, pre-fitted variogram.

    PyKrige is given our nugget/psill/range instead of fitting its own,
    so it only solves the Kriging weights. All monitors are used for
    every tract (21 monitors is small enough for the full system).

    The spherical variogram model:
        γ(h) = nugget + psill × [1.5(h/range) - 0.5(h/range)³]   h < range
        γ(h) = nugget + psill                                    h ≥ range
    """
    logger.info(f"Kriging {len(tracts_utm):,} tract centroids from {len(monitors_utm)} monitors")

    ok = OrdinaryKriging(
        x=monitors_utm["x_utm"].values,
        y=monitors_utm["y_utm"].values,
        z=monitors_utm["pm25_mean"].values,
        variogram_model="spherical",
        variogram_parameters=_pykrige_params(params),
        verbose=False,
        enable_plotting=False,
    )

    pm25_pred, pm25_var = ok.execute(
        "points",
        tracts_utm["x_utm"].values,
        tracts_utm["y_utm"].values,
        backend="vectorized",
    )

    pm25_pred = np.clip(np.asarray(pm25_pred, dtype=float), 0, None)
    pm25_var = np.clip(np.asarray(pm25_var, dtype=float), 0, None)

    logger.info(
        f"Kriging predictions: mean={pm25_pred.mean():.2f} μg/m³, "
        f"std={pm25_pred.std():.2f}, mean variance={pm25_var.mean():.4f}"
    )

    return pd.DataFrame({
        "pm25_kriged": pm25_pred.round(3),
        "pm25_kriged_variance": pm25_var.round(6),
    })


def _leave_one_out_cv(
    monitors_utm: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, float]:
    """
    Leave-one-out cross-validation.

    For each monitor: drop it, rebuild the variogram from the remaining
    monitors with the same binning and fit, Krige at the dropped
    monitor's location, compare. Refitting per fold keeps the held-out
    reading out of the variogram too, so the score is honest.

    Reported on all monitors and on city monitors only. The city-only
    numbers describe accuracy where the tracts actually are, but rest
    on few points (8 in LA), so read them with care.

    Returns
    -------
    dict
        r2, mae, rmse (all monitors), r2_city, mae_city, n_folds.
    """
    n = len(monitors_utm)
    if n < 4:
        logger.warning(f"Only {n} monitors; LOO-CV needs at least 4. Skipping.")
        return {"r2": np.nan, "mae": np.nan, "rmse": np.nan,
                "r2_city": np.nan, "mae_city": np.nan, "n_folds": 0}

    logger.info(f"Running leave-one-out CV on {n} monitors")
    rows = []
    for i in range(n):
        train = monitors_utm.drop(index=monitors_utm.index[i])
        held = monitors_utm.iloc[i]
        try:
            _, params = _variogram_for(train, config)
            ok_cv = OrdinaryKriging(
                x=train["x_utm"].values, y=train["y_utm"].values,
                z=train["pm25_mean"].values,
                variogram_model="spherical", variogram_parameters=_pykrige_params(params),
                verbose=False, enable_plotting=False,
            )
            pred, _ = ok_cv.execute("points", np.array([held["x_utm"]]), np.array([held["y_utm"]]))
            rows.append({"monitor_id": held["monitor_id"], "in_city": bool(held["in_city"]),
                         "actual": held["pm25_mean"], "pred": float(np.asarray(pred)[0])})
        except Exception as e:  # one bad fold should not sink the run
            logger.warning(f"LOO-CV failed for {held['monitor_id']}: {e}")

    cv = pd.DataFrame(rows)

    def _r2(a, p):
        ss_tot = np.sum((a - a.mean()) ** 2)
        return float(1 - np.sum((a - p) ** 2) / ss_tot) if ss_tot > 0 else np.nan

    city = cv[cv["in_city"]]
    out = {
        "r2": _r2(cv["actual"], cv["pred"]),
        "mae": float(np.mean(np.abs(cv["actual"] - cv["pred"]))),
        "rmse": float(np.sqrt(np.mean((cv["actual"] - cv["pred"]) ** 2))),
        "r2_city": _r2(city["actual"], city["pred"]) if len(city) >= 3 else np.nan,
        "mae_city": float(np.mean(np.abs(city["actual"] - city["pred"]))) if len(city) else np.nan,
        "n_folds": len(cv),
    }

    logger.info(
        "LOO-CV per monitor:\n"
        + cv.assign(error=(cv["pred"] - cv["actual"]).round(2))
            .round({"pred": 2})
            .sort_values("error")
            .to_string(index=False)
    )
    logger.info(
        f"LOO-CV: R²={out['r2']:.3f}, MAE={out['mae']:.2f}, RMSE={out['rmse']:.2f} μg/m³ "
        f"(city only: R²={out['r2_city']:.3f}, MAE={out['mae_city']:.2f})"
    )
    return out


def _validate_output(df: pd.DataFrame, expected_tract_count: int) -> None:
    """
    Validate Kriging output before writing to BigQuery.

    Parameters
    ----------
    df : pd.DataFrame
        Kriging output DataFrame.
    expected_tract_count : int
        Expected number of rows (one per tract).

    Raises
    ------
    ValueError
        If validation fails.
    """
    # Check row count matches tract count
    if len(df) != expected_tract_count:
        raise ValueError(
            f"Kriging output has {len(df)} rows, "
            f"expected {expected_tract_count} (one per tract)"
        )

    # Check no negative PM2.5 predictions
    negative = (df["pm25_kriged"] < 0).sum()
    if negative > 0:
        raise ValueError(
            f"{negative} tracts have negative PM2.5 predictions. "
            f"Kriging clip to 0 should have prevented this."
        )

    # Check for extreme values — above 500 μg/m³ is instrument error
    extreme = (df["pm25_kriged"] > 500).sum()
    if extreme > 0:
        raise ValueError(
            f"{extreme} tracts have PM2.5 > 500 μg/m³. "
            f"Check monitor readings for outliers."
        )

    # Check no null predictions
    null_preds = df["pm25_kriged"].isna().sum()
    if null_preds > 0:
        raise ValueError(
            f"{null_preds} tracts have null PM2.5 predictions."
        )

    # Warn if CV R² is low — predictions may be unreliable
    if "cv_r2" in df.columns:
        r2 = df["cv_r2"].iloc[0]
        if r2 < 0.5:
            logger.warning(
                f"LOO-CV R²={r2:.3f} is below 0.5. "
                f"Kriging predictions may be unreliable this week. "
                f"Check for monitor outages or unusual pollution events."
            )

    logger.info(
        f"Kriging validation passed: {len(df):,} tracts, "
        f"PM2.5 range {df['pm25_kriged'].min():.1f}–"
        f"{df['pm25_kriged'].max():.1f} μg/m³"
    )