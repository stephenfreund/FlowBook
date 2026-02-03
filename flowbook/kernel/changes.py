"""
Changes - Typed records of what actually changed between checkpoints.

This module defines a hierarchy of change types that replace the unstructured
MemoryCheckpointDiffResult with explicit typed objects representing specific kinds of changes.

Design Principles:
- Each change type has clear, specific semantics
- Changes are detected by diffing pre/post checkpoints
- Changes are matched against prior AccessEvents to detect conflicts

Change Detection Flow:
    pre_checkpoint ---> [cell execution] ---> post_checkpoint
                               |
                               v
                    diff(pre, post) -> MemoryCheckpointDiffResult
                               |
                               v
                    detect_changes(diff) -> List[Change]
"""

from abc import ABC
from typing import Optional, Union

from pydantic import BaseModel


class Change(BaseModel, ABC):
    """
    Base class for all detected changes.

    All changes track which variable was modified. Subclasses specify
    what kind of modification occurred.
    """

    variable: str

    class Config:
        frozen = True  # Immutable


class ValueChanged(Change):
    """
    Variable's value completely changed (replacement or non-DataFrame mutation).

    This is the most general change type, used when:
    - A variable is reassigned to a new value
    - A non-DataFrame/Series object is mutated
    - We can't determine more specific change type

    Examples:
        x = 5; x = 10              -> ValueChanged(variable='x')
        config['key'] = new_value  -> ValueChanged(variable='config')
        df = pd.DataFrame(...)     -> ValueChanged(variable='df')  # reassignment
    """

    def __repr__(self) -> str:
        return f"ValueChanged({self.variable})"


class ColumnAdded(Change):
    """
    New column added to a DataFrame.

    Detected when a column exists in post-checkpoint but not in pre-checkpoint.

    Examples:
        df['new_col'] = values        -> ColumnAdded(variable='df', column='new_col')
        df.insert(0, 'first', vals)   -> ColumnAdded(variable='df', column='first')
        df.assign(new=lambda x: ...)  -> ColumnAdded(variable='df', column='new')

    Conflict Implications:
        - Does NOT conflict with ColumnRead of other columns
        - DOES conflict with StructuralRead of columns/shape/dtypes (structural)
    """

    column: str

    def __repr__(self) -> str:
        return f"ColumnAdded({self.variable}['{self.column}'])"


class ColumnModified(Change):
    """
    Existing column's values changed.

    Detected when a column exists in both checkpoints but values differ.

    Examples:
        df['price'] = df['price'] * 1.1  -> ColumnModified(variable='df', column='price')
        df.loc[:, 'x'] = 0               -> ColumnModified(variable='df', column='x')
        df['count'] += 1                 -> ColumnModified(variable='df', column='count')

    Conflict Implications:
        - DOES conflict with ColumnRead of the SAME column
        - Does NOT conflict with ColumnRead of other columns
        - Does NOT conflict with StructuralRead (structure unchanged)
    """

    column: str

    def __repr__(self) -> str:
        return f"ColumnModified({self.variable}['{self.column}'])"


class ColumnRemoved(Change):
    """
    Column removed from a DataFrame.

    Detected when a column exists in pre-checkpoint but not in post-checkpoint.

    Examples:
        del df['old_col']                        -> ColumnRemoved(variable='df', column='old_col')
        df.drop(columns=['x'], inplace=True)     -> ColumnRemoved(variable='df', column='x')
        df.pop('temp')                           -> ColumnRemoved(variable='df', column='temp')

    Conflict Implications:
        - DOES conflict with ColumnRead of the removed column
        - DOES conflict with StructuralRead of columns/shape/dtypes (structural)
    """

    column: str

    def __repr__(self) -> str:
        return f"ColumnRemoved({self.variable}['{self.column}'])"


class RowsAdded(Change):
    """
    Rows added to a DataFrame.

    Detected when post-checkpoint has more rows than pre-checkpoint.

    Examples:
        df.loc[len(df)] = new_row               -> RowsAdded(variable='df', count=1)
        df = pd.concat([df, new_rows])          -> RowsAdded(variable='df', count=N)
        df.append(row_dict, ignore_index=True)  -> RowsAdded(variable='df', count=1)

    Conflict Implications:
        - DOES conflict with ColumnRead (column now has more values!)
        - DOES conflict with StructuralRead of shape/len/index/size (structural)
        - Does NOT conflict with StructuralRead of columns/dtypes only
    """

    count: int

    def __repr__(self) -> str:
        return f"RowsAdded({self.variable}, count={self.count})"


class RowsRemoved(Change):
    """
    Rows removed from a DataFrame.

    Detected when post-checkpoint has fewer rows than pre-checkpoint.

    Examples:
        df = df[df['x'] > 0]                  -> RowsRemoved(variable='df', count=N)
        df.drop(index=[0, 1], inplace=True)   -> RowsRemoved(variable='df', count=2)
        df.dropna(inplace=True)               -> RowsRemoved(variable='df', count=N)

    Conflict Implications:
        - DOES conflict with ColumnRead (column now has fewer/different values!)
        - DOES conflict with StructuralRead of shape/len/index/size (structural)
    """

    count: int

    def __repr__(self) -> str:
        return f"RowsRemoved({self.variable}, count={self.count})"


class IndexChanged(Change):
    """
    DataFrame/Series index changed (labels, not just length).

    Detected when index values or type changed between checkpoints.

    Examples:
        df.reset_index(inplace=True)          -> IndexChanged(variable='df')
        df.set_index('col', inplace=True)     -> IndexChanged(variable='df')
        df.index = new_index                  -> IndexChanged(variable='df')
        df.reindex(new_labels)                -> IndexChanged(variable='df')

    Conflict Implications:
        - DOES conflict with StructuralRead of index/axes (structural)
        - Does NOT conflict with ColumnRead (column values unchanged)

    Note: Row additions/removals also change the index, but are reported as
    RowsAdded/RowsRemoved instead. IndexChanged is for label-only changes.
    """

    def __repr__(self) -> str:
        return f"IndexChanged({self.variable})"


class DtypeChanged(Change):
    """
    Column dtype changed.

    Detected when a column's dtype differs between checkpoints.

    Examples:
        df['x'] = df['x'].astype(float)       -> DtypeChanged(variable='df', column='x', ...)
        df['date'] = pd.to_datetime(df['date']) -> DtypeChanged(variable='df', column='date', ...)

    Conflict Implications:
        - DOES conflict with StructuralRead of dtypes (structural)
        - May conflict with ColumnRead (same values, different type behavior)
    """

    column: str
    old_dtype: str
    new_dtype: str

    def __repr__(self) -> str:
        return f"DtypeChanged({self.variable}['{self.column}']: {self.old_dtype} -> {self.new_dtype})"


# Type alias for any change
AnyChange = Union[
    ValueChanged,
    ColumnAdded,
    ColumnModified,
    ColumnRemoved,
    RowsAdded,
    RowsRemoved,
    IndexChanged,
    DtypeChanged,
]
