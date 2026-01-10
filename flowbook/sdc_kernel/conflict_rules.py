"""
Conflict Rules - Declarative specification of SDC conflict detection.

This module defines the rules that determine when a Change conflicts with
a prior AccessEvent. Instead of procedural if/else chains, conflicts are
detected by matching against this rules table.

Design Principles:
- All conflict logic in one place (this file)
- Rules are explicit and self-documenting
- Each rule is independently testable
- Adding new cases = adding rows to the table

Structural Mode:
- Rules with is_structural=True have mode-dependent severity
- ENFORCE mode: structural conflicts are VIOLATION
- WARN mode: structural conflicts are WARNING
- OFF mode: structural conflicts are OK (ignored)
"""

from enum import Enum
from typing import FrozenSet, List, Optional, Tuple, Type

from pydantic import BaseModel

from .access_events import AccessEvent, ColumnRead, StructuralRead, VariableRead
from .changes import (
    Change,
    ColumnAdded,
    ColumnModified,
    ColumnRemoved,
    DtypeChanged,
    IndexChanged,
    RowsAdded,
    RowsRemoved,
    ValueChanged,
)


class ConflictSeverity(str, Enum):
    """The severity of a detected conflict."""

    OK = "ok"  # No conflict
    WARNING = "warning"  # Warning (shown but not blocking)
    VIOLATION = "violation"  # Hard block (cell rolled back)


class StructuralMode(str, Enum):
    """User-configurable mode for structural conflict handling."""

    OFF = "off"  # Ignore structural conflicts entirely
    WARN = "warn"  # Generate warnings for structural conflicts
    ENFORCE = "enforce"  # Treat structural conflicts as violations


class ConflictRule(BaseModel):
    """
    A declarative rule for detecting conflicts between a Change and an AccessEvent.

    Rules are evaluated in order. The first matching rule determines the result.

    Attributes:
        change_types: Tuple of Change subclasses this rule applies to
        read_types: Tuple of AccessEvent subclasses this rule applies to
        same_column: If True, only match when change.column == read.column
        structural_attrs: If set, only match StructuralRead with attr in this set
        severity: The conflict severity when this rule matches
        is_structural: If True, actual severity depends on structural_mode
        description: Human-readable explanation for docs and error messages
    """

    change_types: Tuple[Type[Change], ...]
    read_types: Tuple[Type[AccessEvent], ...]
    same_column: bool = False
    structural_attrs: Optional[FrozenSet[str]] = None
    severity: ConflictSeverity
    is_structural: bool = False
    description: str

    class Config:
        frozen = True

    def matches(self, change: Change, read: AccessEvent) -> bool:
        """
        Check if this rule applies to the given change/read pair.

        Args:
            change: The detected change from current cell
            read: The access event from a prior cell

        Returns:
            True if this rule matches and should determine the result
        """
        # Must match change type
        if not isinstance(change, self.change_types):
            return False

        # Must match read type
        if not isinstance(read, self.read_types):
            return False

        # Check column matching if required
        if self.same_column:
            change_col = getattr(change, "column", None)
            read_col = getattr(read, "column", None)
            if change_col is None or read_col is None:
                return False
            if change_col != read_col:
                return False

        # Check structural attribute matching if specified
        if self.structural_attrs is not None:
            if not isinstance(read, StructuralRead):
                return False
            if read.attr not in self.structural_attrs:
                return False

        return True


# =============================================================================
# Attribute Sets for Readability
# =============================================================================

# Attributes that reveal column structure
COLUMN_STRUCTURE_ATTRS: FrozenSet[str] = frozenset(
    {
        "columns",  # Column names Index
        "keys",  # Same as columns
        "dtypes",  # Column dtypes
        "axes",  # [index, columns]
        "T",  # Transpose (exposes columns)
        "values",  # Full array (shape visible)
        "iter",  # Iteration over DataFrame yields columns
        "describe",  # describe() includes all columns
    }
)

# Attributes that reveal row structure
ROW_STRUCTURE_ATTRS: FrozenSet[str] = frozenset(
    {
        "index",  # Row labels
        "shape",  # (rows, cols)
        "size",  # rows * cols
        "len",  # Number of rows
        "empty",  # Whether empty
    }
)

# Attributes that reveal both row AND column structure
SHAPE_ATTRS: FrozenSet[str] = frozenset(
    {
        "shape",  # (rows, cols)
        "size",  # rows * cols
    }
)

# All structural attributes
ALL_STRUCTURE_ATTRS: FrozenSet[str] = COLUMN_STRUCTURE_ATTRS | ROW_STRUCTURE_ATTRS


# =============================================================================
# THE CONFLICT RULES TABLE
# =============================================================================
#
# Rules are evaluated in order. First matching rule wins.
# Be careful with rule ordering - more specific rules should come before
# more general rules.
#

CONFLICT_RULES: List[ConflictRule] = [
    # =========================================================================
    # VARIABLE-LEVEL READS (catch-all for non-DataFrame variables)
    # =========================================================================
    # Any change to a variable that was read at the variable level is a violation
    ConflictRule(
        change_types=(
            ValueChanged,
            ColumnAdded,
            ColumnModified,
            ColumnRemoved,
            RowsAdded,
            RowsRemoved,
            IndexChanged,
            DtypeChanged,
        ),
        read_types=(VariableRead,),
        severity=ConflictSeverity.VIOLATION,
        description="Any change to variable invalidates variable-level reads",
    ),
    # =========================================================================
    # VALUE CHANGES (complete variable replacement)
    # =========================================================================
    ConflictRule(
        change_types=(ValueChanged,),
        read_types=(ColumnRead, StructuralRead),
        severity=ConflictSeverity.VIOLATION,
        description="Value replacement invalidates all prior reads of the variable",
    ),
    # =========================================================================
    # COLUMN MODIFICATIONS
    # =========================================================================
    # Same column: VIOLATION
    ConflictRule(
        change_types=(ColumnModified,),
        read_types=(ColumnRead,),
        same_column=True,
        severity=ConflictSeverity.VIOLATION,
        description="Modifying column X invalidates prior reads of column X",
    ),
    # Different column: OK
    ConflictRule(
        change_types=(ColumnModified,),
        read_types=(ColumnRead,),
        same_column=False,
        severity=ConflictSeverity.OK,
        description="Modifying column X does not affect reads of column Y",
    ),
    # Structural read: OK (modifying values doesn't change structure)
    ConflictRule(
        change_types=(ColumnModified,),
        read_types=(StructuralRead,),
        severity=ConflictSeverity.OK,
        description="Modifying column values does not affect structural reads",
    ),
    # =========================================================================
    # COLUMN ADDITIONS
    # =========================================================================
    # Column read: OK (new column doesn't affect existing column reads)
    ConflictRule(
        change_types=(ColumnAdded,),
        read_types=(ColumnRead,),
        severity=ConflictSeverity.OK,
        description="Adding new column does not affect reads of existing columns",
    ),
    # Structural read of column-revealing attrs: STRUCTURAL
    ConflictRule(
        change_types=(ColumnAdded,),
        read_types=(StructuralRead,),
        structural_attrs=COLUMN_STRUCTURE_ATTRS,
        severity=ConflictSeverity.WARNING,
        is_structural=True,
        description="Adding column affects reads of column-revealing attributes (columns, dtypes, etc.)",
    ),
    # Structural read of shape/size: STRUCTURAL (shape changes)
    ConflictRule(
        change_types=(ColumnAdded,),
        read_types=(StructuralRead,),
        structural_attrs=SHAPE_ATTRS,
        severity=ConflictSeverity.WARNING,
        is_structural=True,
        description="Adding column changes shape/size",
    ),
    # Structural read of row-only attrs (index, len, empty): OK
    ConflictRule(
        change_types=(ColumnAdded,),
        read_types=(StructuralRead,),
        structural_attrs=ROW_STRUCTURE_ATTRS - SHAPE_ATTRS,
        severity=ConflictSeverity.OK,
        description="Adding column does not affect row-only structural reads (index, len, empty)",
    ),
    # =========================================================================
    # COLUMN REMOVALS
    # =========================================================================
    # Same column read: VIOLATION
    ConflictRule(
        change_types=(ColumnRemoved,),
        read_types=(ColumnRead,),
        same_column=True,
        severity=ConflictSeverity.VIOLATION,
        description="Removing column X invalidates prior reads of column X",
    ),
    # Different column read: OK
    ConflictRule(
        change_types=(ColumnRemoved,),
        read_types=(ColumnRead,),
        same_column=False,
        severity=ConflictSeverity.OK,
        description="Removing column X does not affect reads of column Y",
    ),
    # Structural read of column-revealing attrs: STRUCTURAL
    ConflictRule(
        change_types=(ColumnRemoved,),
        read_types=(StructuralRead,),
        structural_attrs=COLUMN_STRUCTURE_ATTRS | SHAPE_ATTRS,
        severity=ConflictSeverity.WARNING,
        is_structural=True,
        description="Removing column affects column-structure reads",
    ),
    # =========================================================================
    # ROW ADDITIONS
    # =========================================================================
    # Column read: VIOLATION (column now has more data!)
    ConflictRule(
        change_types=(RowsAdded,),
        read_types=(ColumnRead,),
        severity=ConflictSeverity.VIOLATION,
        description="Adding rows changes all column values (more data in each column)",
    ),
    # Structural read of row-revealing attrs: STRUCTURAL
    ConflictRule(
        change_types=(RowsAdded,),
        read_types=(StructuralRead,),
        structural_attrs=ROW_STRUCTURE_ATTRS | SHAPE_ATTRS,
        severity=ConflictSeverity.WARNING,
        is_structural=True,
        description="Adding rows affects row-structure reads (shape, len, index, size)",
    ),
    # Structural read of column-only attrs (no shape): OK
    ConflictRule(
        change_types=(RowsAdded,),
        read_types=(StructuralRead,),
        structural_attrs=COLUMN_STRUCTURE_ATTRS - SHAPE_ATTRS,
        severity=ConflictSeverity.OK,
        description="Adding rows does not affect column-only structural reads (columns, dtypes)",
    ),
    # =========================================================================
    # ROW REMOVALS
    # =========================================================================
    # Column read: VIOLATION (column now has less/different data!)
    ConflictRule(
        change_types=(RowsRemoved,),
        read_types=(ColumnRead,),
        severity=ConflictSeverity.VIOLATION,
        description="Removing rows changes all column values (less data in each column)",
    ),
    # Structural read of row-revealing attrs: STRUCTURAL
    ConflictRule(
        change_types=(RowsRemoved,),
        read_types=(StructuralRead,),
        structural_attrs=ROW_STRUCTURE_ATTRS | SHAPE_ATTRS,
        severity=ConflictSeverity.WARNING,
        is_structural=True,
        description="Removing rows affects row-structure reads (shape, len, index, size)",
    ),
    # Structural read of column-only attrs (no shape): OK
    ConflictRule(
        change_types=(RowsRemoved,),
        read_types=(StructuralRead,),
        structural_attrs=COLUMN_STRUCTURE_ATTRS - SHAPE_ATTRS,
        severity=ConflictSeverity.OK,
        description="Removing rows does not affect column-only structural reads (columns, dtypes)",
    ),
    # =========================================================================
    # INDEX CHANGES
    # =========================================================================
    # Structural read of index-revealing attrs: STRUCTURAL
    ConflictRule(
        change_types=(IndexChanged,),
        read_types=(StructuralRead,),
        structural_attrs=frozenset({"index", "axes"}),
        severity=ConflictSeverity.WARNING,
        is_structural=True,
        description="Index change affects index-revealing reads",
    ),
    # Column read: OK (index change doesn't affect column values)
    ConflictRule(
        change_types=(IndexChanged,),
        read_types=(ColumnRead,),
        severity=ConflictSeverity.OK,
        description="Index change does not affect column value reads",
    ),
    # Other structural attrs: OK
    ConflictRule(
        change_types=(IndexChanged,),
        read_types=(StructuralRead,),
        structural_attrs=ALL_STRUCTURE_ATTRS - frozenset({"index", "axes"}),
        severity=ConflictSeverity.OK,
        description="Index change does not affect non-index structural reads",
    ),
    # =========================================================================
    # DTYPE CHANGES
    # =========================================================================
    # Structural read of dtype-revealing attrs: STRUCTURAL
    ConflictRule(
        change_types=(DtypeChanged,),
        read_types=(StructuralRead,),
        structural_attrs=frozenset({"dtypes", "dtype"}),
        severity=ConflictSeverity.WARNING,
        is_structural=True,
        description="Dtype change affects dtype-revealing reads",
    ),
    # Column read of same column: WARNING (values same, type different)
    ConflictRule(
        change_types=(DtypeChanged,),
        read_types=(ColumnRead,),
        same_column=True,
        severity=ConflictSeverity.WARNING,
        description="Dtype change on column may affect column reads (same values, different type behavior)",
    ),
    # Column read of different column: OK
    ConflictRule(
        change_types=(DtypeChanged,),
        read_types=(ColumnRead,),
        same_column=False,
        severity=ConflictSeverity.OK,
        description="Dtype change on column X does not affect reads of column Y",
    ),
    # Other structural attrs: OK
    ConflictRule(
        change_types=(DtypeChanged,),
        read_types=(StructuralRead,),
        structural_attrs=ALL_STRUCTURE_ATTRS - frozenset({"dtypes", "dtype"}),
        severity=ConflictSeverity.OK,
        description="Dtype change does not affect non-dtype structural reads",
    ),
]


def get_rule_by_description(description: str) -> Optional[ConflictRule]:
    """Find a rule by its description (for testing)."""
    for rule in CONFLICT_RULES:
        if rule.description == description:
            return rule
    return None
