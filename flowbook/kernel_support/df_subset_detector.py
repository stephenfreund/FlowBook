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
import pickle

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

        Returns SubsetRelation if valid, None otherwise.
        """
        # 1. Check if child's index is subset of parent's index
        child_index_set = set(child_df.index)
        parent_index_set = set(parent_df.index)
        if not child_index_set.issubset(parent_index_set):
            return None

        # 2. Get integer positions (handles non-integer indices)
        try:
            row_indices = parent_df.index.get_indexer(child_df.index)
            if -1 in row_indices:
                return None  # Some indices not found
        except Exception:
            return None

        # 3. Identify common and extra columns
        child_cols = set(child_df.columns)
        parent_cols = set(parent_df.columns)
        # Preserve child's column order for common columns
        common_columns = [c for c in child_df.columns if c in parent_cols]
        extra_columns = [c for c in child_df.columns if c not in parent_cols]

        if not common_columns:
            return None  # No common columns, can't be a meaningful subset

        # 4. Verify common columns have matching values
        try:
            parent_subset = parent_df.iloc[row_indices][common_columns]
            child_common = child_df[common_columns]

            # Reset indices for comparison (they should have same values)
            parent_subset_reset = parent_subset.reset_index(drop=True)
            child_common_reset = child_common.reset_index(drop=True)

            if not child_common_reset.equals(parent_subset_reset):
                return None  # Values differ
        except Exception:
            return None

        # 5. Extract extra column data
        extra_data = None
        if extra_columns:
            extra_data = child_df[extra_columns].copy()

        # 6. Estimate savings
        estimated_savings = self._estimate_savings(child_df, row_indices, extra_data)

        return SubsetRelation(
            child_var=child_name,
            parent_var=parent_name,
            row_indices=row_indices,
            common_columns=common_columns,
            extra_columns=extra_columns,
            extra_data=extra_data,
            estimated_savings_bytes=estimated_savings,
            child_columns=list(child_df.columns),
        )

    def _estimate_savings(
        self,
        child_df: pd.DataFrame,
        row_indices: np.ndarray,
        extra_data: pd.DataFrame | None,
    ) -> int:
        """Estimate bytes saved by storing indices instead of full DataFrame."""
        try:
            # Size of full DataFrame
            full_size = len(pickle.dumps(child_df, protocol=pickle.HIGHEST_PROTOCOL))

            # Size of indices + extra data
            indices_size = len(
                pickle.dumps(row_indices, protocol=pickle.HIGHEST_PROTOCOL)
            )
            extra_size = 0
            if extra_data is not None:
                extra_size = len(
                    pickle.dumps(extra_data, protocol=pickle.HIGHEST_PROTOCOL)
                )

            optimized_size = indices_size + extra_size
            return max(0, full_size - optimized_size)
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
