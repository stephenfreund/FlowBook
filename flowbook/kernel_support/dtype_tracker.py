"""
Dtype Origin Tracker for Checkpoint Save/Restore

Tracks the original dtypes of pandas DataFrame columns and Series before
checkpoint deepcopy converts object-dtype columns to specialized dtypes
(e.g., StringDtype, Int64). On restore, converts back to original dtypes
so user code sees the same types it originally had.

This follows the same pattern as CuDFOriginTracker in cudf_compat.py.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from flowbook.util.output import log


class DtypeOriginTracker:
    """
    Tracks original column dtypes for DataFrames and dtype for Series.

    Used during checkpoint save to record what dtypes the user's variables
    had before deepcopy converts object columns to specialized types.
    During restore, converts columns/Series back to original dtypes.

    Only tracks top-level variables (same scope as CuDFOriginTracker).
    """

    def __init__(self):
        # For DataFrames: var_name -> {col_name: original_dtype}
        self._df_dtypes: Dict[str, Dict[Any, Any]] = {}
        # For Series: var_name -> original_dtype
        self._series_dtypes: Dict[str, Any] = {}

    def record(self, name: str, value: Any) -> None:
        """
        Record original dtypes for a variable if it's a DataFrame or Series.

        Args:
            name: Variable name in user namespace
            value: The variable's value
        """
        if isinstance(value, pd.DataFrame):
            # Record dtypes for all columns
            col_dtypes = {}
            for col in value.columns:
                col_dtypes[col] = value[col].dtype
            if col_dtypes:
                self._df_dtypes[name] = col_dtypes
        elif isinstance(value, pd.Series):
            self._series_dtypes[name] = value.dtype

    def restore_value(self, name: str, value: Any) -> Any:
        """
        Restore original dtypes for a variable after checkpoint restore.

        Args:
            name: Variable name
            value: The restored value (may have converted dtypes)

        Returns:
            Value with original dtypes restored
        """
        if isinstance(value, pd.DataFrame) and name in self._df_dtypes:
            original_dtypes = self._df_dtypes[name]
            for col in value.columns:
                if col in original_dtypes:
                    current_dtype = value[col].dtype
                    target_dtype = original_dtypes[col]
                    if current_dtype != target_dtype:
                        try:
                            value[col] = value[col].astype(target_dtype)
                            log(f"Restored column '{col}' dtype from {current_dtype} to {target_dtype}")
                        except (TypeError, ValueError):
                            log(f"Could not restore column '{col}' dtype from {current_dtype} to {target_dtype}")
            return value
        elif isinstance(value, pd.Series) and name in self._series_dtypes:
            target_dtype = self._series_dtypes[name]
            if value.dtype != target_dtype:
                try:
                    value = value.astype(target_dtype)
                    log(f"Restored Series '{name}' dtype from {value.dtype} to {target_dtype}")
                except (TypeError, ValueError):
                    log(f"Could not restore Series '{name}' dtype to {target_dtype}")
            return value
        return value

    def clear(self) -> None:
        """Clear all recorded dtypes."""
        self._df_dtypes.clear()
        self._series_dtypes.clear()
