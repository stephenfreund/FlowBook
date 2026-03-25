"""
DataFrame Provenance Tracking via df.attrs.

Records which cell first caused each structural change to a DataFrame by
storing a single DataFrameProvenance object in df.attrs['_flowbook_provenance'].

This enables the enforcer to recover the correct WriteLoc types on
re-execution, where checkpoint diffs may miss idempotent structural changes
(e.g. ColumnAdded degrades to ColumnModified, RowsAdded vanishes entirely).

Tracked structural effects:
    - Column addition:  col_origins   {col: cell_id}  — first writer wins
    - Column deletion:  col_deletions {col: cell_id}  — first deleter wins
    - Dtype change:     dtype_origins {col: cell_id}  — first changer wins
    - Row mutation:     row_mutators  set of cell_ids
    - Index mutation:   index_mutators set of cell_ids

Design:
    - First writer/deleter/changer wins: provenance records the original
      creator so re-execution doesn't overwrite.
    - record_var_write resets ALL provenance (DataFrame was replaced/created).
    - Provenance travels with the object via df.attrs, surviving
      copy(deep=False), aliasing, and checkpoint/restore.
"""

import copy as copy_module
from typing import Dict, Optional, Set

import pandas as pd

PROVENANCE_KEY = '_flowbook_provenance'


class DataFrameProvenance:
    """All structural provenance for a DataFrame.

    Stored as a single object in df.attrs[PROVENANCE_KEY].
    """

    __slots__ = (
        'col_origins',
        'col_deletions',
        'dtype_origins',
        'row_mutators',
        'index_mutators',
    )

    def __init__(self):
        self.col_origins: Dict[str, str] = {}
        self.col_deletions: Dict[str, str] = {}
        self.dtype_origins: Dict[str, str] = {}
        self.row_mutators: Set[str] = set()
        self.index_mutators: Set[str] = set()

    def __copy__(self):
        new = DataFrameProvenance()
        new.col_origins = self.col_origins.copy()
        new.col_deletions = self.col_deletions.copy()
        new.dtype_origins = self.dtype_origins.copy()
        new.row_mutators = self.row_mutators.copy()
        new.index_mutators = self.index_mutators.copy()
        return new

    def __deepcopy__(self, memo):
        # All fields are flat (str keys/values), shallow copy of each suffices.
        return self.__copy__()


class DataFrameProvenanceTracker:
    """Static methods to read/write DataFrameProvenance on a DataFrame."""

    @staticmethod
    def _get_or_create(df: pd.DataFrame) -> DataFrameProvenance:
        """Get existing provenance or create a fresh one."""
        prov = df.attrs.get(PROVENANCE_KEY)
        if prov is None:
            prov = DataFrameProvenance()
            df.attrs[PROVENANCE_KEY] = prov
        return prov

    # ── Recording methods ──────────────────────────────────────────────

    @staticmethod
    def record_var_write(df: pd.DataFrame, cell_id: str) -> None:
        """Record that a DataFrame was assigned to a variable.

        Resets ALL provenance. All existing columns are attributed to
        cell_id. This covers pd.read_csv(), pd.DataFrame(), df.merge(),
        pd.concat(), etc.
        """
        prov = DataFrameProvenance()
        prov.col_origins = {str(col): cell_id for col in df.columns}
        df.attrs[PROVENANCE_KEY] = prov

    @staticmethod
    def record_column_write(
        df: pd.DataFrame, col_name: str, cell_id: str
    ) -> None:
        """Record a column write. First writer wins — does not overwrite
        an existing origin."""
        prov = DataFrameProvenanceTracker._get_or_create(df)
        if col_name not in prov.col_origins:
            prov.col_origins[col_name] = cell_id

    @staticmethod
    def record_column_delete(
        df: pd.DataFrame, col_name: str, cell_id: Optional[str] = None
    ) -> None:
        """Remove column from origins and optionally record who deleted it.

        First deleter wins — does not overwrite an existing deletion record.
        """
        prov = DataFrameProvenanceTracker._get_or_create(df)
        prov.col_origins.pop(col_name, None)
        if cell_id is not None and col_name not in prov.col_deletions:
            prov.col_deletions[col_name] = cell_id

    @staticmethod
    def record_dtype_change(
        df: pd.DataFrame, col_name: str, cell_id: str
    ) -> None:
        """Record a dtype change. First changer wins per column."""
        prov = DataFrameProvenanceTracker._get_or_create(df)
        if col_name not in prov.dtype_origins:
            prov.dtype_origins[col_name] = cell_id

    @staticmethod
    def record_row_mutation(df: pd.DataFrame, cell_id: str) -> None:
        """Record that this cell mutated rows (add/remove)."""
        prov = DataFrameProvenanceTracker._get_or_create(df)
        prov.row_mutators.add(cell_id)

    @staticmethod
    def record_index_mutation(df: pd.DataFrame, cell_id: str) -> None:
        """Record that this cell mutated the index."""
        prov = DataFrameProvenanceTracker._get_or_create(df)
        prov.index_mutators.add(cell_id)

    # ── Query methods ──────────────────────────────────────────────────

    @staticmethod
    def get_provenance(df: pd.DataFrame) -> Optional[DataFrameProvenance]:
        """Get the provenance object, or None if not set."""
        return df.attrs.get(PROVENANCE_KEY)

    @staticmethod
    def get_origins(df: pd.DataFrame) -> Dict[str, str]:
        """Read the column origins dict."""
        prov = df.attrs.get(PROVENANCE_KEY)
        return prov.col_origins if prov is not None else {}

    @staticmethod
    def get_columns_from_cell(
        df: pd.DataFrame, cell_id: str
    ) -> Set[str]:
        """Return the set of columns whose origin is the given cell_id."""
        prov = df.attrs.get(PROVENANCE_KEY)
        if prov is None:
            return set()
        return {col for col, cid in prov.col_origins.items() if cid == cell_id}

    @staticmethod
    def is_column_added_by(
        df: pd.DataFrame, col_name: str, cell_id: str
    ) -> bool:
        """Check if a column was first created by the given cell."""
        prov = df.attrs.get(PROVENANCE_KEY)
        return prov is not None and prov.col_origins.get(col_name) == cell_id

    @staticmethod
    def is_column_deleted_by(
        df: pd.DataFrame, col_name: str, cell_id: str
    ) -> bool:
        """Check if a column was first deleted by the given cell."""
        prov = df.attrs.get(PROVENANCE_KEY)
        return prov is not None and prov.col_deletions.get(col_name) == cell_id

    @staticmethod
    def is_dtype_changed_by(
        df: pd.DataFrame, col_name: str, cell_id: str
    ) -> bool:
        """Check if a column's dtype was first changed by the given cell."""
        prov = df.attrs.get(PROVENANCE_KEY)
        return prov is not None and prov.dtype_origins.get(col_name) == cell_id

    @staticmethod
    def is_row_mutator(df: pd.DataFrame, cell_id: str) -> bool:
        """Check if the given cell mutated rows."""
        prov = df.attrs.get(PROVENANCE_KEY)
        return prov is not None and cell_id in prov.row_mutators

    @staticmethod
    def is_index_mutator(df: pd.DataFrame, cell_id: str) -> bool:
        """Check if the given cell mutated the index."""
        prov = df.attrs.get(PROVENANCE_KEY)
        return prov is not None and cell_id in prov.index_mutators
