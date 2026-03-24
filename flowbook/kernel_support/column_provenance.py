"""
Column Provenance Tracking via df.attrs.

Records which cell first created each column of a DataFrame by storing
provenance metadata in df.attrs['_flowbook_col_origins'].

This enables the enforcer to distinguish ColAdd (new column, affects
structural attributes like shape/columns) from Col (existing column
modification, does not affect structure) even after re-execution,
where the diff can no longer detect ColumnAdded.

Design:
    - First writer wins: record_column_write only sets origin if the
      column doesn't already have one.
    - record_var_write resets ALL origins (DataFrame was replaced/created).
    - Provenance travels with the object via df.attrs, surviving
      copy(deep=False), aliasing, and checkpoint/restore.
"""

from typing import Dict, Optional, Set

import pandas as pd

PROVENANCE_KEY = '_flowbook_col_origins'


class ColumnProvenanceTracker:
    """Manages df.attrs['_flowbook_col_origins'] = {col_name: cell_id}."""

    @staticmethod
    def record_var_write(df: pd.DataFrame, cell_id: str) -> None:
        """Record that a DataFrame was assigned to a variable.

        All existing columns are attributed to cell_id. This covers
        pd.read_csv(), pd.DataFrame(), df.merge(), pd.concat(), etc.
        """
        df.attrs[PROVENANCE_KEY] = {str(col): cell_id for col in df.columns}

    @staticmethod
    def record_column_write(df: pd.DataFrame, col_name: str, cell_id: str) -> None:
        """Record a column write. First writer wins — does not overwrite
        an existing origin. This ensures that re-execution of
        df['x'] = 5 preserves the original creator's cell_id."""
        origins = df.attrs.get(PROVENANCE_KEY, {})
        if col_name not in origins:
            origins[col_name] = cell_id
            df.attrs[PROVENANCE_KEY] = origins

    @staticmethod
    def record_column_delete(df: pd.DataFrame, col_name: str) -> None:
        """Remove provenance entry when a column is dropped."""
        origins = df.attrs.get(PROVENANCE_KEY, {})
        origins.pop(col_name, None)

    @staticmethod
    def get_origins(df: pd.DataFrame) -> Dict[str, str]:
        """Read the column origins from a DataFrame."""
        return df.attrs.get(PROVENANCE_KEY, {})

    @staticmethod
    def get_columns_from_cell(df: pd.DataFrame, cell_id: str) -> Set[str]:
        """Return the set of columns whose origin is the given cell_id."""
        origins = df.attrs.get(PROVENANCE_KEY, {})
        return {col for col, cid in origins.items() if cid == cell_id}

    @staticmethod
    def is_column_added_by(
        df: pd.DataFrame, col_name: str, cell_id: str
    ) -> bool:
        """Check if a column was first created by the given cell."""
        return df.attrs.get(PROVENANCE_KEY, {}).get(col_name) == cell_id
