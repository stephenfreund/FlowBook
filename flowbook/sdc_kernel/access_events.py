"""
Access Events - Typed records of variable/column/structural access during cell execution.

This module defines a hierarchy of access event types that replace the multiple
dictionaries in TrackingData (reads_before_writes, column_reads_before_writes,
structural_reads, etc.) with explicit typed objects.

Design Principles:
- Each event type has clear, specific semantics
- No optional fields with overloaded meanings
- Immutable (frozen) for use in sets and as dict keys
- No VariableRead - dynamic analysis always gives us specific access info

Structure-Using vs Structure-Revealing:
- StructuralRead is only recorded for structure-REVEALING access (df.columns, df.shape)
- Structure-USING methods (repr, __getitem__, mean) do NOT record StructuralRead
  even though they internally access structural attributes
- This distinction is handled at recording time, not in these types
"""

from abc import ABC
from typing import Union

from pydantic import BaseModel


class AccessEvent(BaseModel, ABC):
    """
    Base class for all access events during cell execution.

    All events track which variable was accessed. Subclasses add
    specificity about what kind of access occurred.
    """

    variable: str

    class Config:
        frozen = True  # Immutable for use in sets


class ColumnRead(AccessEvent):
    """
    Read of a specific DataFrame/Series column.

    Recorded when code accesses column data for computation.

    Examples:
        df['price'].sum()     -> ColumnRead(variable='df', column='price')
        df.price.mean()       -> ColumnRead(variable='df', column='price')
        df.loc[:, 'x']        -> ColumnRead(variable='df', column='x')
        df[['a', 'b']]        -> ColumnRead(variable='df', column='a'),
                                 ColumnRead(variable='df', column='b')
    """

    column: str

    def __repr__(self) -> str:
        return f"ColumnRead({self.variable}['{self.column}'])"


class ColumnWrite(AccessEvent):
    """
    Write to a specific DataFrame column.

    Recorded when code modifies or creates a column.

    Examples:
        df['price'] = values      -> ColumnWrite(variable='df', column='price')
        df.loc[:, 'x'] = 0        -> ColumnWrite(variable='df', column='x')
        df['new'] = df['a'] + 1   -> ColumnWrite(variable='df', column='new')
        df.insert(0, 'col', val)  -> ColumnWrite(variable='df', column='col')
    """

    column: str

    def __repr__(self) -> str:
        return f"ColumnWrite({self.variable}['{self.column}'])"


class StructuralRead(AccessEvent):
    """
    Read of a structural attribute that reveals DataFrame/Series structure.

    Only recorded for EXPLICIT structure-revealing access, NOT for internal
    access by structure-using methods like repr(), __getitem__, mean(), etc.

    Structural Attributes Tracked:
        DataFrame:
            columns  - Column names (Index)
            shape    - (rows, cols) tuple
            index    - Row labels (Index)
            dtypes   - Column dtypes (Series)
            size     - Total elements (rows * cols)
            empty    - Whether DataFrame is empty
            axes     - [index, columns]
            keys     - Same as columns
            len      - Number of rows (via __len__)

        Series:
            index    - Element labels
            shape    - (length,) tuple
            dtype    - Data type
            name     - Series name
            size     - Number of elements
            empty    - Whether Series is empty

    Examples:
        cols = df.columns           -> StructuralRead(variable='df', attr='columns')
        n_rows, n_cols = df.shape   -> StructuralRead(variable='df', attr='shape')
        length = len(df)            -> StructuralRead(variable='df', attr='len')
        for col in df:              -> StructuralRead(variable='df', attr='iter')
    """

    attr: str

    def __repr__(self) -> str:
        return f"StructuralRead({self.variable}.{self.attr})"


class VariableRead(AccessEvent):
    """
    Read of a variable at the whole-variable level (not column-specific).

    Used for non-DataFrame/Series variables where we track that the entire
    variable was read, but have no column-level detail.

    Examples:
        x = config['value']   -> VariableRead(variable='config')
        result = x + 1        -> VariableRead(variable='x')
        print(data)           -> VariableRead(variable='data')

    Conflict Implications:
        - Any change to this variable (ValueChanged) is a violation
        - This is the most conservative read type
    """

    def __repr__(self) -> str:
        return f"VariableRead({self.variable})"


# Type aliases for clarity
ReadEvent = Union[ColumnRead, StructuralRead, VariableRead]
WriteEvent = ColumnWrite
AnyAccessEvent = Union[ColumnRead, ColumnWrite, StructuralRead]
