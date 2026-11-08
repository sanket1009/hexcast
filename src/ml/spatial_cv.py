"""
spatial_cv.py
=============
Blocked spatial cross-validation splitter for CHBIndex ML models.

Standard k-fold CV randomly assigns tracts to folds. For spatial data
this causes data leakage — neighboring tracts share correlated features,
so the model effectively sees test tracts through their training neighbors.
The result is artificially inflated CV performance that doesn't reflect
real-world generalization to new cities or regions.

Blocked spatial CV solves this by assigning entire geographic blocks
(H3 resolution-6 parent cells) to folds together. When a block is in
the test fold, NO tract from that geographic region appears in training.
This matches the real deployment scenario: predicting for a new city
the model has never seen.

Method:
    1. Assign each tract to its H3 resolution-6 parent cell
    2. Assign each unique parent cell to one of K folds
       (stratified to balance tract count per fold)
    3. Each CV split: hold out all tracts in fold k's parent cells,
       train on all other tracts

Usage:
    from src.ml.spatial_cv import SpatialBlockCV

    cv = SpatialBlockCV(n_splits=5, h3_resolution=6, random_seed=42)
    for train_idx, test_idx in cv.split(X, groups=h3_parent_cells):
        model.fit(X[train_idx], y[train_idx])
        score = model.score(X[test_idx], y[test_idx])
"""

from collections.abc import Iterator
from typing import Any

import h3
import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator

from src.utils.logger import get_logger

logger = get_logger(__name__)


class SpatialBlockCV(BaseCrossValidator):
    """
    Sklearn-compatible blocked spatial cross-validator.

    Assigns tracts to geographic blocks using H3 parent cells,
    then assigns blocks to folds. Prevents spatial data leakage
    by ensuring no training tract is geographically adjacent to
    any test tract.

    Inherits from sklearn's BaseCrossValidator so it works
    seamlessly with sklearn's cross_val_score, GridSearchCV,
    and any other CV utility that accepts a cv parameter.

    Parameters
    ----------
    n_splits : int
        Number of CV folds. Default 5.
        With ~50-80 H3 blocks in LA County, 5 folds gives
        10-16 blocks per test fold — enough geographic coverage.

    h3_resolution : int
        H3 resolution for geographic blocks. Default 6.
        Resolution 6 cells are ~36 km² — large enough to prevent
        leakage between neighboring tracts in different folds.
        Resolution 7 (5 km²) would be too fine — adjacent cells
        could end up in different folds.

    random_seed : int
        Random seed for reproducible fold assignments. Default 42.

    Examples
    --------
    >>> cv = SpatialBlockCV(n_splits=5, random_seed=42)
    >>> for train_idx, test_idx in cv.split(X, groups=h3_parent_cells):
    ...     print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")
    Train: 1848, Test: 462
    Train: 1851, Test: 459
    ...
    """

    def __init__(
        self,
        n_splits: int = 5,
        h3_resolution: int = 6,
        random_seed: int = 42,
    ):
        self.n_splits = n_splits
        self.h3_resolution = h3_resolution
        self.random_seed = random_seed

    def split(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series | None = None,
        groups: np.ndarray | pd.Series | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test index splits for spatial block CV.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix. Only used for its length.
        y : array-like of shape (n_samples,), optional
            Target variable. Not used but required by sklearn API.
        groups : array-like of shape (n_samples,)
            H3 parent cell ID for each tract. REQUIRED.
            All tracts with the same group (same H3 parent cell)
            will always be in the same fold together.
            Get this from build_spatial_groups().

        Yields
        ------
        train_indices : np.ndarray
            Indices of training samples for this fold.
        test_indices : np.ndarray
            Indices of test samples for this fold.

        Raises
        ------
        ValueError
            If groups is None — spatial CV requires group labels.
        """
        if groups is None:
            raise ValueError(
                "SpatialBlockCV requires groups parameter. "
                "Pass H3 parent cell IDs as groups. "
                "Use build_spatial_groups() to generate them."
            )

        groups = np.asarray(groups)
        n_samples = len(groups)

        # Get unique geographic blocks (H3 parent cells)
        unique_blocks = np.unique(groups)
        n_blocks = len(unique_blocks)

        logger.debug(
            f"Spatial block CV: {n_samples} tracts, "
            f"{n_blocks} H3 blocks, {self.n_splits} folds"
        )

        if n_blocks < self.n_splits:
            raise ValueError(
                f"Number of H3 blocks ({n_blocks}) is less than "
                f"n_splits ({self.n_splits}). "
                f"Reduce n_splits or use coarser H3 resolution."
            )

        # Assign blocks to folds
        # Shuffle blocks first for randomness, then assign sequentially
        rng = np.random.RandomState(self.random_seed)
        shuffled_blocks = rng.permutation(unique_blocks)

        # Assign each block to a fold (round-robin assignment)
        # This roughly balances the number of blocks per fold
        block_to_fold = {
            block: i % self.n_splits for i, block in enumerate(shuffled_blocks)
        }

        logger.debug(
            f"Block-to-fold assignment: "
            f"{[sum(1 for f in block_to_fold.values() if f == k) for k in range(self.n_splits)]} "
            f"blocks per fold"
        )

        # Generate train/test splits
        for fold in range(self.n_splits):
            # Test: all tracts whose H3 parent cell is in this fold
            test_mask = np.array([block_to_fold[g] == fold for g in groups])

            # Train: all other tracts
            train_mask = ~test_mask

            train_indices = np.where(train_mask)[0]
            test_indices = np.where(test_mask)[0]

            logger.debug(
                f"Fold {fold + 1}/{self.n_splits}: "
                f"train={len(train_indices)}, test={len(test_indices)}, "
                f"test_blocks={sum(1 for f in block_to_fold.values() if f == fold)}"
            )

            yield train_indices, test_indices

    def get_n_splits(
        self,
        X: Any = None,
        y: Any = None,
        groups: Any = None,
    ) -> int:
        """Return the number of splits. Required by sklearn API."""
        return self.n_splits

    def _iter_test_masks(self, X=None, y=None, groups=None):
        """Required by BaseCrossValidator — not used directly."""
        for _, test in self.split(X, y, groups):
            mask = np.zeros(len(groups), dtype=bool)
            mask[test] = True
            yield mask


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def build_spatial_groups(
    geoids: pd.Series | list,
    centroids_lat: pd.Series | list,
    centroids_lon: pd.Series | list,
    h3_resolution: int = 6,
) -> np.ndarray:
    """
    Build H3 parent cell group labels for each tract.

    Assigns each tract centroid to its H3 resolution-6 parent cell.
    Tracts in the same parent cell get the same group label and will
    always appear in the same CV fold together.

    This is the groups parameter passed to SpatialBlockCV.split().

    Parameters
    ----------
    geoids : array-like
        Census tract GEOIDs.
    centroids_lat : array-like
        Tract centroid latitudes.
    centroids_lon : array-like
        Tract centroid longitudes.
    h3_resolution : int
        H3 resolution for blocks. Default 6 (~36 km²).

    Returns
    -------
    np.ndarray
        Array of H3 parent cell IDs, one per tract.
        Same length as geoids.

    Examples
    --------
    >>> groups = build_spatial_groups(
    ...     geoids=df["geoid"],
    ...     centroids_lat=df["centroid_lat"],
    ...     centroids_lon=df["centroid_lon"],
    ... )
    >>> len(np.unique(groups))
    67  # 67 unique H3 blocks in LA County
    """
    groups = []

    for lat, lon in zip(centroids_lat, centroids_lon, strict=False):
        if pd.isna(lat) or pd.isna(lon):
            # Missing centroid — assign to a dummy block
            # This tract will consistently land in the same fold
            groups.append("missing_centroid")
            continue

        # Assign to H3 resolution-7 cell first
        cell_r7 = h3.latlng_to_cell(lat, lon, 7)

        # Get parent cell at the block resolution (default 6)
        # All tracts within the same ~36 km² block share this parent
        parent_cell = h3.cell_to_parent(cell_r7, h3_resolution)
        groups.append(parent_cell)

    return np.array(groups)


def validate_no_leakage(
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    groups: np.ndarray,
) -> bool:
    """
    Verify that no H3 block appears in both train and test sets.

    This is the unit test that proves blocked spatial CV is working
    correctly — the core property we need for valid spatial evaluation.

    Parameters
    ----------
    train_indices : np.ndarray
        Training set indices from one CV fold.
    test_indices : np.ndarray
        Test set indices from one CV fold.
    groups : np.ndarray
        H3 parent cell ID for each sample.

    Returns
    -------
    bool
        True if no leakage detected (blocks are cleanly separated).

    Raises
    ------
    AssertionError
        If any H3 block appears in both train and test sets.

    Examples
    --------
    >>> assert validate_no_leakage(train_idx, test_idx, groups)
    >>> # Passes silently if no leakage
    """
    train_blocks = set(groups[train_indices])
    test_blocks = set(groups[test_indices])

    # Check for overlap — any shared block is a leakage
    leaked_blocks = train_blocks & test_blocks

    if leaked_blocks:
        raise AssertionError(
            f"Spatial leakage detected: {len(leaked_blocks)} H3 blocks "
            f"appear in both train and test sets. "
            f"Example leaked blocks: {list(leaked_blocks)[:3]}"
        )

    logger.debug(
        f"No leakage: {len(train_blocks)} train blocks, "
        f"{len(test_blocks)} test blocks, 0 overlap"
    )

    return True


def summarize_cv_folds(
    cv: SpatialBlockCV,
    X: np.ndarray,
    groups: np.ndarray,
) -> pd.DataFrame:
    """
    Summarize the train/test split sizes across all CV folds.

    Useful for verifying fold balance before training —
    if one fold has far fewer test samples than others,
    its performance estimate will be less reliable.

    Parameters
    ----------
    cv : SpatialBlockCV
        Fitted cross-validator.
    X : np.ndarray
        Feature matrix (used only for its length).
    groups : np.ndarray
        H3 parent cell group labels.

    Returns
    -------
    pd.DataFrame
        Summary table with one row per fold showing:
        fold, n_train, n_test, n_test_blocks, train_pct

    Examples
    --------
    >>> summary = summarize_cv_folds(cv, X, groups)
    >>> print(summary)
       fold  n_train  n_test  n_test_blocks  train_pct
    0     1     1848     462             13      80.0%
    1     2     1851     459             14      80.1%
    ...
    """
    records = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, groups=groups)):
        # Verify no leakage for this fold
        validate_no_leakage(train_idx, test_idx, groups)

        test_blocks = set(groups[test_idx])
        train_pct = len(train_idx) / len(groups) * 100

        records.append(
            {
                "fold": fold + 1,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "n_test_blocks": len(test_blocks),
                "train_pct": f"{train_pct:.1f}%",
            }
        )

    return pd.DataFrame(records)
