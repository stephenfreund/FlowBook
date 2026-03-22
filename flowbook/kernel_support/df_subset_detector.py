"""
DataFrame Subset Detection for Checkpoint Optimization

Detects when DataFrames in a namespace are row-subsets of other DataFrames,
enabling storage of indices instead of full copies during checkpointing.

Algorithm:
1. Identify all DataFrames in namespace
2. Sort by row count (descending) - larger DataFrames are potential parents
3. For each potential child, find smallest valid parent
4. Validate subset relationship (index subset + value equality for common columns)
5. Compute storage: parent_var_name + row_indices + extra_columns_data

Example:
    df = load_data()                          # 100 MB
    df_filtered = df[df['country'] != 'X']    # 80 MB, row subset of df
    df_filtered2 = df_filtered[df['val'] > 0] # 60 MB, row subset of df_filtered

    Without optimization: 100 + 80 + 60 = 240 MB checkpoint
    With optimization: 100 + ~1 MB indices = ~101 MB checkpoint
"""

from dataclasses import dataclass, field
from typing import Any
import time

import numpy as np
import pandas as pd


@dataclass
class SubsetRelation:
    """Represents a parent-child DataFrame subset relationship."""

    child_var: str  # Variable name of subset DataFrame
    parent_var: str  # Variable name of parent DataFrame
    row_indices: np.ndarray  # Integer positions in parent (not index labels)
    common_columns: list[str]  # Columns shared with parent (in order)
    extra_columns: list[str]  # Columns in child but not parent
    extra_data: pd.DataFrame | None  # Data for extra columns (None if no extras)
    estimated_savings_bytes: int  # Bytes saved by using this relation
    child_columns: list[str] = field(default_factory=list)  # Original column order

    def to_dict(self) -> dict:
        """Serialize for storage (indices as list for JSON compatibility)."""
        return {
            "child_var": self.child_var,
            "parent_var": self.parent_var,
            "row_indices": self.row_indices.tolist(),
            "common_columns": self.common_columns,
            "extra_columns": self.extra_columns,
            "extra_data": self.extra_data.to_dict("list")
            if self.extra_data is not None
            else None,
            "estimated_savings_bytes": self.estimated_savings_bytes,
            "child_columns": self.child_columns,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SubsetRelation":
        """Deserialize from storage."""
        extra_data = None
        if d["extra_data"] is not None:
            extra_data = pd.DataFrame(d["extra_data"])
        return cls(
            child_var=d["child_var"],
            parent_var=d["parent_var"],
            row_indices=np.array(d["row_indices"], dtype=np.intp),
            common_columns=d["common_columns"],
            extra_columns=d["extra_columns"],
            extra_data=extra_data,
            estimated_savings_bytes=d["estimated_savings_bytes"],
            child_columns=d.get("child_columns", []),
        )


@dataclass
class SubsetDetectionResult:
    """Result of subset detection analysis."""

    relations: list[SubsetRelation]  # Detected subset relationships
    parent_vars: set[str]  # Variables that are parents (need full copy)
    child_vars: set[str]  # Variables that are children (store as indices)
    standalone_vars: set[str]  # DataFrames with no subset relationship
    total_estimated_savings_bytes: int  # Total bytes saved
    detection_time_ms: float  # Time spent detecting


class DataFrameSubsetDetector:
    """
    Detects DataFrame subset relationships for checkpoint optimization.

    Configuration:
        min_rows: Minimum rows for a DataFrame to be considered (default: 100)
        min_savings_bytes: Minimum savings to create a relation (default: 10 KB)
        max_dataframes: Maximum DataFrames to analyze (default: 20)
        timeout_ms: Maximum time for detection (default: 1000 ms)

    Caching:
        Results are cached based on DataFrame identity (id()). Cache is invalidated
        when a DataFrame's id changes (indicating a new or modified DataFrame).
    """

    def __init__(
        self,
        min_rows: int = 100,
        min_savings_bytes: int = 10 * 1024,  # 10 KB
        max_dataframes: int = 20,
        timeout_ms: float = 1000,
    ):
        self.min_rows = min_rows
        self.min_savings_bytes = min_savings_bytes
        self.max_dataframes = max_dataframes
        self.timeout_ms = timeout_ms
        # Cache: (parent_id, child_id) -> SubsetRelation or None
        # Also store (parent_id, child_id) -> (parent_shape, child_shape) for validation
        self._cache: dict[tuple[int, int], SubsetRelation | None] = {}
        self._cache_shapes: dict[tuple[int, int], tuple[tuple, tuple]] = {}

    def detect(self, variables: dict[str, Any]) -> SubsetDetectionResult:
        """
        Analyze variables to find DataFrame subset relationships.

        Args:
            variables: Dictionary of variable names to values

        Returns:
            SubsetDetectionResult with detected relationships
        """
        start_time = time.time()

        # 1. Extract DataFrames meeting minimum size
        dataframes = {}
        for name, value in variables.items():
            if isinstance(value, pd.DataFrame) and len(value) >= self.min_rows:
                dataframes[name] = value

        # Early exit: need at least 2 DataFrames for subset relationships
        if len(dataframes) < 2:
            return SubsetDetectionResult(
                relations=[],
                parent_vars=set(),
                child_vars=set(),
                standalone_vars=set(dataframes.keys()),
                total_estimated_savings_bytes=0,
                detection_time_ms=(time.time() - start_time) * 1000,
            )

        # 2. Limit to max_dataframes (largest first)
        if len(dataframes) > self.max_dataframes:
            sorted_names = sorted(
                dataframes.keys(), key=lambda k: len(dataframes[k]), reverse=True
            )
            dataframes = {k: dataframes[k] for k in sorted_names[: self.max_dataframes]}

        # 3. Sort by row count descending (larger = potential parents)
        sorted_by_size = sorted(
            dataframes.items(), key=lambda x: len(x[1]), reverse=True
        )

        # 4. Find subset relationships
        relations = []
        child_vars = set()

        for i, (child_name, child_df) in enumerate(sorted_by_size):
            # Skip if already identified as a child
            if child_name in child_vars:
                continue

            # Check timeout
            elapsed_ms = (time.time() - start_time) * 1000
            if elapsed_ms > self.timeout_ms:
                break

            # Check against all larger DataFrames (potential parents)
            for parent_name, parent_df in sorted_by_size[:i]:
                # Skip if parent is already a child of something else
                if parent_name in child_vars:
                    continue

                # Skip if child has more rows than parent
                if len(child_df) >= len(parent_df):
                    continue

                # Check subset relationship
                from flowbook.util.output import timer
                with timer(key="subset:00_check_call"):
                    relation = self._check_subset(
                        child_name, child_df, parent_name, parent_df
                    )
                if (
                    relation is not None
                    and relation.estimated_savings_bytes >= self.min_savings_bytes
                ):
                    relations.append(relation)
                    child_vars.add(child_name)
                    break  # Found a parent, move to next potential child

        # 5. Identify parent and standalone variables
        parent_vars = {r.parent_var for r in relations}
        all_df_vars = set(dataframes.keys())
        standalone_vars = all_df_vars - child_vars - parent_vars

        detection_time_ms = (time.time() - start_time) * 1000
        total_savings = sum(r.estimated_savings_bytes for r in relations)

        return SubsetDetectionResult(
            relations=relations,
            parent_vars=parent_vars,
            child_vars=child_vars,
            standalone_vars=standalone_vars,
            total_estimated_savings_bytes=total_savings,
            detection_time_ms=detection_time_ms,
        )

    def _check_subset(
        self,
        child_name: str,
        child_df: pd.DataFrame,
        parent_name: str,
        parent_df: pd.DataFrame,
    ) -> SubsetRelation | None:
        """
        Check if child_df is a row subset of parent_df.

        Uses fast checks only:
        1. Cache lookup (instant if cached)
        2. Index type compatibility (instant)
        3. Index subset check (fast) - required for row_indices
        4. Single-value spot check (fast) - quick validation
        """
        from flowbook.util.output import timer

        # 0. Check cache first
        with timer(key="subset:00a_cache_check"):
            cache_key = (id(parent_df), id(child_df))
            shapes = (parent_df.shape, child_df.shape)
            if cache_key in self._cache:
                # Validate shapes haven't changed (DataFrame was modified in place)
                if self._cache_shapes.get(cache_key) == shapes:
                    cached = self._cache[cache_key]
                    if cached is not None:
                        # Update variable names in case they changed
                        return SubsetRelation(
                            child_var=child_name,
                            parent_var=parent_name,
                            row_indices=cached.row_indices,
                            common_columns=cached.common_columns,
                            extra_columns=cached.extra_columns,
                            extra_data=cached.extra_data,
                            estimated_savings_bytes=cached.estimated_savings_bytes,
                            child_columns=cached.child_columns,
                        )
                    return None

        # Helper to cache and return None
        def _cache_none():
            self._cache[cache_key] = None
            self._cache_shapes[cache_key] = shapes
            return None

        # 1. Quick index type check - different types can't be subsets
        with timer(key="subset:00b_index_type"):
            child_idx_type = type(child_df.index)
            parent_idx_type = type(parent_df.index)
            if child_idx_type != parent_idx_type:
                # Allow RangeIndex to match Int64Index
                if not (
                    child_idx_type.__name__ in ("RangeIndex", "Int64Index", "Index")
                    and parent_idx_type.__name__ in ("RangeIndex", "Int64Index", "Index")
                    and child_df.index.dtype == parent_df.index.dtype
                ):
                    return _cache_none()

        # 2. Identify common columns
        with timer(key="subset:01_parent_cols"):
            parent_cols = set(parent_df.columns)

        with timer(key="subset:02_common_cols"):
            common_columns = [c for c in child_df.columns if c in parent_cols]

        if not common_columns:
            return _cache_none()

        # 3. Check index subset (required for row_indices)
        with timer(key="subset:03_get_indexer"):
            try:
                row_indices = parent_df.index.get_indexer(child_df.index)
                if -1 in row_indices:
                    return _cache_none()
            except Exception:
                return _cache_none()

        # 4. Validate ALL values in ALL common columns (vectorized, fast)
        # This guarantees zero false positives - critical for data integrity on restore
        with timer(key="subset:04_full_validation"):
            for col in common_columns:
                try:
                    child_arr = child_df[col].values
                    parent_arr = parent_df[col].values
                    parent_subset = parent_arr[row_indices]

                    # Handle different dtypes appropriately:
                    # - Floating point: use equal_nan=True to handle NaN correctly
                    # - Object/string: use pandas isna + element comparison
                    # - Other (int, bool, etc.): basic array_equal works
                    if isinstance(child_arr.dtype, np.dtype) and np.issubdtype(child_arr.dtype, np.floating):
                        if not np.array_equal(child_arr, parent_subset, equal_nan=True):
                            return _cache_none()
                    elif child_arr.dtype == object or (hasattr(child_arr.dtype, 'kind') and child_arr.dtype.kind in ('U', 'S')) or pd.api.types.is_string_dtype(child_arr.dtype):
                        # For object/string dtypes, handle NaN specially
                        # Use pandas isna which handles None, NaN, NaT correctly
                        child_na = pd.isna(child_arr)
                        parent_na = pd.isna(parent_subset)
                        if not np.array_equal(child_na, parent_na):
                            return _cache_none()
                        # Compare non-NA values
                        non_na_mask = ~child_na
                        if non_na_mask.any():
                            if not np.array_equal(
                                child_arr[non_na_mask], parent_subset[non_na_mask]
                            ):
                                return _cache_none()
                    else:
                        # For non-float, non-object types (int, bool, etc.)
                        if not np.array_equal(child_arr, parent_subset):
                            return _cache_none()
                except Exception:
                    return _cache_none()

        # 5. Passed full validation - confirmed valid subset
        with timer(key="subset:05_extra_cols"):
            extra_columns = [c for c in child_df.columns if c not in parent_cols]

        # 6. Extract extra column data
        with timer(key="subset:06_extra_copy"):
            extra_data = None
            if extra_columns:
                extra_data = child_df[extra_columns].copy()

        # 7. Estimate savings
        with timer(key="subset:07_estimate"):
            estimated_savings = self._estimate_savings(child_df, row_indices, extra_data)

        result = SubsetRelation(
            child_var=child_name,
            parent_var=parent_name,
            row_indices=row_indices,
            common_columns=common_columns,
            extra_columns=extra_columns,
            extra_data=extra_data,
            estimated_savings_bytes=estimated_savings,
            child_columns=list(child_df.columns),
        )

        # Cache the successful result
        self._cache[cache_key] = result
        self._cache_shapes[cache_key] = shapes

        return result

    def _estimate_savings(
        self,
        child_df: pd.DataFrame,
        row_indices: np.ndarray,
        extra_data: pd.DataFrame | None,
    ) -> int:
        """Estimate bytes saved by storing indices instead of full DataFrame.

        Uses fast shallow memory estimate (deep=False) to avoid slow object inspection.
        """
        try:
            # Fast shallow estimate - avoids inspecting object dtype contents
            full_size = child_df.memory_usage(deep=False).sum()

            # Size of indices (numpy array bytes + small overhead)
            indices_size = row_indices.nbytes + 128

            # Size of extra columns data (also shallow)
            extra_size = 0
            if extra_data is not None:
                extra_size = extra_data.memory_usage(deep=False).sum()

            optimized_size = indices_size + extra_size
            return max(0, int(full_size - optimized_size))
        except Exception:
            return 0


def reconstruct_from_subset(
    parent_df: pd.DataFrame,
    relation: SubsetRelation,
) -> pd.DataFrame:
    """
    Reconstruct a child DataFrame from its parent and subset relation.

    Args:
        parent_df: The parent DataFrame (already restored)
        relation: The SubsetRelation describing how to reconstruct

    Returns:
        Reconstructed child DataFrame
    """
    # 1. Select rows from parent using integer positions
    child_df = parent_df.iloc[relation.row_indices][relation.common_columns].copy()

    # 2. Add extra columns if any
    if relation.extra_columns and relation.extra_data is not None:
        for col in relation.extra_columns:
            child_df[col] = relation.extra_data[col].values

    # 3. Reorder columns to match original child column order
    # Use stored child_columns if available, otherwise fall back to common + extra
    if relation.child_columns:
        child_df = child_df[relation.child_columns]
    else:
        all_columns = relation.common_columns + relation.extra_columns
        child_df = child_df[all_columns]

    return child_df


def topological_sort_relations(relations: list[SubsetRelation]) -> list[SubsetRelation]:
    """
    Sort relations so parents are restored before children.

    Handles chains like: df -> df_filtered -> df_filtered_more
    """
    if not relations:
        return []

    # Build dependency graph
    child_to_parent = {r.child_var: r.parent_var for r in relations}
    relation_by_child = {r.child_var: r for r in relations}

    # Find relations whose parent is not in remaining (ready to restore)
    sorted_relations = []
    remaining = set(r.child_var for r in relations)

    while remaining:
        # Find relations whose parent is not in remaining
        ready = [
            child for child in remaining if child_to_parent[child] not in remaining
        ]

        if not ready:
            # Cycle detected or error - just add remaining in any order
            ready = list(remaining)

        for child in ready:
            sorted_relations.append(relation_by_child[child])
            remaining.remove(child)

    return sorted_relations
