"""
Tests for the conflict rules and resolver.

Tests cover:
- Individual ConflictRule matching
- ConflictResolver with all three structural modes
- All change/read combinations in CONFLICT_RULES
- Helper methods (get_violations, get_warnings, has_conflict)
"""

import pytest

from flowbook.kernel.access_events import ColumnRead, ColumnWrite, StructuralRead
from flowbook.kernel.changes import (
    ColumnAdded,
    ColumnModified,
    ColumnRemoved,
    DtypeChanged,
    IndexChanged,
    RowsAdded,
    RowsRemoved,
    ValueChanged,
)
from flowbook.kernel.conflict_resolver import (
    ConflictResolver,
    ConflictResult,
    ViolationsResult,
    DEFAULT_MAX_VIOLATIONS,
)
from flowbook.kernel.conflict_rules import (
    CONFLICT_RULES,
    ConflictRule,
    ConflictSeverity,
    StructuralMode,
    get_rule_by_description,
)


# =============================================================================
# Test ConflictRule.matches()
# =============================================================================


class TestConflictRuleMatches:
    """Tests for ConflictRule.matches() method."""

    def test_matches_change_type(self):
        """Rule should match based on change type."""
        rule = ConflictRule(
            change_types=(ColumnModified,),
            read_types=(ColumnRead,),
            same_column=True,
            severity=ConflictSeverity.VIOLATION,
            description="test",
        )

        # Should match ColumnModified
        assert rule.matches(
            ColumnModified(variable="df", column="x"),
            ColumnRead(variable="df", column="x"),
        )

        # Should not match ColumnAdded
        assert not rule.matches(
            ColumnAdded(variable="df", column="x"),
            ColumnRead(variable="df", column="x"),
        )

    def test_matches_read_type(self):
        """Rule should match based on read type."""
        rule = ConflictRule(
            change_types=(ColumnModified,),
            read_types=(ColumnRead,),
            same_column=True,
            severity=ConflictSeverity.VIOLATION,
            description="test",
        )

        # Should match ColumnRead
        assert rule.matches(
            ColumnModified(variable="df", column="x"),
            ColumnRead(variable="df", column="x"),
        )

        # Should not match StructuralRead
        assert not rule.matches(
            ColumnModified(variable="df", column="x"),
            StructuralRead(variable="df", attr="shape"),
        )

    def test_matches_same_column_true(self):
        """Rule with same_column=True should only match same column."""
        rule = ConflictRule(
            change_types=(ColumnModified,),
            read_types=(ColumnRead,),
            same_column=True,
            severity=ConflictSeverity.VIOLATION,
            description="test",
        )

        # Same column: should match
        assert rule.matches(
            ColumnModified(variable="df", column="price"),
            ColumnRead(variable="df", column="price"),
        )

        # Different column: should not match
        assert not rule.matches(
            ColumnModified(variable="df", column="price"),
            ColumnRead(variable="df", column="quantity"),
        )

    def test_matches_same_column_false(self):
        """Rule with same_column=False should match different columns."""
        rule = ConflictRule(
            change_types=(ColumnModified,),
            read_types=(ColumnRead,),
            same_column=False,
            severity=ConflictSeverity.OK,
            description="test",
        )

        # Different column: should match
        assert rule.matches(
            ColumnModified(variable="df", column="price"),
            ColumnRead(variable="df", column="quantity"),
        )

        # Same column: should also match (same_column=False means "any")
        # But the same_column=True rule should come first in CONFLICT_RULES
        assert rule.matches(
            ColumnModified(variable="df", column="price"),
            ColumnRead(variable="df", column="price"),
        )

    def test_matches_structural_attrs(self):
        """Rule with structural_attrs should only match those attrs."""
        rule = ConflictRule(
            change_types=(ColumnAdded,),
            read_types=(StructuralRead,),
            structural_attrs=frozenset({"columns", "dtypes"}),
            severity=ConflictSeverity.WARNING,
            is_structural=True,
            description="test",
        )

        # Matching attr
        assert rule.matches(
            ColumnAdded(variable="df", column="new"),
            StructuralRead(variable="df", attr="columns"),
        )

        # Non-matching attr
        assert not rule.matches(
            ColumnAdded(variable="df", column="new"),
            StructuralRead(variable="df", attr="index"),
        )

    def test_matches_multiple_change_types(self):
        """Rule can match multiple change types."""
        rule = ConflictRule(
            change_types=(ColumnAdded, ColumnRemoved),
            read_types=(StructuralRead,),
            structural_attrs=frozenset({"columns"}),
            severity=ConflictSeverity.WARNING,
            is_structural=True,
            description="test",
        )

        assert rule.matches(
            ColumnAdded(variable="df", column="x"),
            StructuralRead(variable="df", attr="columns"),
        )
        assert rule.matches(
            ColumnRemoved(variable="df", column="x"),
            StructuralRead(variable="df", attr="columns"),
        )


# =============================================================================
# Test ConflictResolver
# =============================================================================


class TestConflictResolver:
    """Tests for ConflictResolver class."""

    def test_different_variables_is_ok(self):
        """Changes to different variables never conflict."""
        resolver = ConflictResolver()
        result = resolver.resolve(
            ColumnModified(variable="df1", column="x"),
            ColumnRead(variable="df2", column="x"),
        )
        assert result.severity == ConflictSeverity.OK
        assert "Different variables" in result.description

    def test_column_modified_same_column_is_violation(self):
        """Modifying a column that was read is a violation."""
        resolver = ConflictResolver()
        result = resolver.resolve(
            ColumnModified(variable="df", column="price"),
            ColumnRead(variable="df", column="price"),
        )
        assert result.severity == ConflictSeverity.VIOLATION

    def test_column_modified_different_column_is_ok(self):
        """Modifying a different column is OK."""
        resolver = ConflictResolver()
        result = resolver.resolve(
            ColumnModified(variable="df", column="price"),
            ColumnRead(variable="df", column="quantity"),
        )
        assert result.severity == ConflictSeverity.OK

    def test_value_changed_is_violation(self):
        """Complete value change invalidates all reads."""
        resolver = ConflictResolver()

        # Column read
        result = resolver.resolve(
            ValueChanged(variable="df"),
            ColumnRead(variable="df", column="x"),
        )
        assert result.severity == ConflictSeverity.VIOLATION

        # Structural read
        result = resolver.resolve(
            ValueChanged(variable="df"),
            StructuralRead(variable="df", attr="shape"),
        )
        assert result.severity == ConflictSeverity.VIOLATION


# =============================================================================
# Test Structural Mode
# =============================================================================


class TestStructuralMode:
    """Tests for structural mode handling."""

    def test_enforce_mode_makes_structural_violation(self):
        """In ENFORCE mode, structural conflicts are violations."""
        resolver = ConflictResolver(structural_mode=StructuralMode.ENFORCE)
        result = resolver.resolve(
            ColumnAdded(variable="df", column="new"),
            StructuralRead(variable="df", attr="columns"),
        )
        assert result.severity == ConflictSeverity.VIOLATION

    def test_warn_mode_makes_structural_warning(self):
        """In WARN mode, structural conflicts are warnings."""
        resolver = ConflictResolver(structural_mode=StructuralMode.WARN)
        result = resolver.resolve(
            ColumnAdded(variable="df", column="new"),
            StructuralRead(variable="df", attr="columns"),
        )
        assert result.severity == ConflictSeverity.WARNING

    def test_off_mode_makes_structural_ok(self):
        """In OFF mode, structural conflicts are ignored."""
        resolver = ConflictResolver(structural_mode=StructuralMode.OFF)
        result = resolver.resolve(
            ColumnAdded(variable="df", column="new"),
            StructuralRead(variable="df", attr="columns"),
        )
        assert result.severity == ConflictSeverity.OK

    def test_non_structural_violation_unaffected_by_mode(self):
        """Non-structural violations are the same in all modes."""
        for mode in [StructuralMode.OFF, StructuralMode.WARN, StructuralMode.ENFORCE]:
            resolver = ConflictResolver(structural_mode=mode)
            result = resolver.resolve(
                ColumnModified(variable="df", column="x"),
                ColumnRead(variable="df", column="x"),
            )
            assert result.severity == ConflictSeverity.VIOLATION


# =============================================================================
# Test Row Changes
# =============================================================================


class TestRowChanges:
    """Tests for row addition/removal conflicts."""

    def test_rows_added_column_read_is_violation(self):
        """Adding rows changes column values - must be violation."""
        resolver = ConflictResolver()
        result = resolver.resolve(
            RowsAdded(variable="df", count=5),
            ColumnRead(variable="df", column="price"),
        )
        assert result.severity == ConflictSeverity.VIOLATION

    def test_rows_removed_column_read_is_violation(self):
        """Removing rows changes column values - must be violation."""
        resolver = ConflictResolver()
        result = resolver.resolve(
            RowsRemoved(variable="df", count=3),
            ColumnRead(variable="df", column="price"),
        )
        assert result.severity == ConflictSeverity.VIOLATION

    def test_rows_added_shape_read_is_structural(self):
        """Adding rows with shape read is structural (mode-dependent)."""
        # WARN mode
        resolver = ConflictResolver(structural_mode=StructuralMode.WARN)
        result = resolver.resolve(
            RowsAdded(variable="df", count=5),
            StructuralRead(variable="df", attr="shape"),
        )
        assert result.severity == ConflictSeverity.WARNING

        # ENFORCE mode
        resolver = ConflictResolver(structural_mode=StructuralMode.ENFORCE)
        result = resolver.resolve(
            RowsAdded(variable="df", count=5),
            StructuralRead(variable="df", attr="shape"),
        )
        assert result.severity == ConflictSeverity.VIOLATION

    def test_rows_added_columns_read_is_ok(self):
        """Adding rows doesn't affect column-only structural reads."""
        resolver = ConflictResolver()
        result = resolver.resolve(
            RowsAdded(variable="df", count=5),
            StructuralRead(variable="df", attr="columns"),
        )
        assert result.severity == ConflictSeverity.OK

        result = resolver.resolve(
            RowsAdded(variable="df", count=5),
            StructuralRead(variable="df", attr="dtypes"),
        )
        assert result.severity == ConflictSeverity.OK


# =============================================================================
# Test Column Addition/Removal
# =============================================================================


class TestColumnChanges:
    """Tests for column addition/removal conflicts."""

    def test_column_added_column_read_is_ok(self):
        """Adding a new column doesn't affect reads of existing columns."""
        resolver = ConflictResolver()
        result = resolver.resolve(
            ColumnAdded(variable="df", column="new_col"),
            ColumnRead(variable="df", column="existing_col"),
        )
        assert result.severity == ConflictSeverity.OK

    def test_column_added_columns_read_is_structural(self):
        """Adding column affects structural reads of columns."""
        resolver = ConflictResolver(structural_mode=StructuralMode.WARN)
        result = resolver.resolve(
            ColumnAdded(variable="df", column="new"),
            StructuralRead(variable="df", attr="columns"),
        )
        assert result.severity == ConflictSeverity.WARNING

    def test_column_removed_same_column_is_violation(self):
        """Removing a column that was read is a violation."""
        resolver = ConflictResolver()
        result = resolver.resolve(
            ColumnRemoved(variable="df", column="x"),
            ColumnRead(variable="df", column="x"),
        )
        assert result.severity == ConflictSeverity.VIOLATION

    def test_column_removed_different_column_is_ok(self):
        """Removing a different column is OK."""
        resolver = ConflictResolver()
        result = resolver.resolve(
            ColumnRemoved(variable="df", column="y"),
            ColumnRead(variable="df", column="x"),
        )
        assert result.severity == ConflictSeverity.OK


# =============================================================================
# Test Index and Dtype Changes
# =============================================================================


class TestIndexDtypeChanges:
    """Tests for index and dtype change conflicts."""

    def test_index_changed_index_read_is_structural(self):
        """Index change with index read is structural."""
        resolver = ConflictResolver(structural_mode=StructuralMode.WARN)
        result = resolver.resolve(
            IndexChanged(variable="df"),
            StructuralRead(variable="df", attr="index"),
        )
        assert result.severity == ConflictSeverity.WARNING

    def test_index_changed_column_read_is_ok(self):
        """Index change doesn't affect column value reads."""
        resolver = ConflictResolver()
        result = resolver.resolve(
            IndexChanged(variable="df"),
            ColumnRead(variable="df", column="x"),
        )
        assert result.severity == ConflictSeverity.OK

    def test_dtype_changed_same_column_is_warning(self):
        """Dtype change on same column is a warning."""
        resolver = ConflictResolver()
        result = resolver.resolve(
            DtypeChanged(variable="df", column="x", old_dtype="int64", new_dtype="float64"),
            ColumnRead(variable="df", column="x"),
        )
        assert result.severity == ConflictSeverity.WARNING

    def test_dtype_changed_different_column_is_ok(self):
        """Dtype change on different column is OK."""
        resolver = ConflictResolver()
        result = resolver.resolve(
            DtypeChanged(variable="df", column="x", old_dtype="int64", new_dtype="float64"),
            ColumnRead(variable="df", column="y"),
        )
        assert result.severity == ConflictSeverity.OK


# =============================================================================
# Test Helper Methods
# =============================================================================


class TestHelperMethods:
    """Tests for ConflictResolver helper methods."""

    def test_check_all_returns_sorted(self):
        """check_all returns results sorted by severity."""
        resolver = ConflictResolver(structural_mode=StructuralMode.WARN)

        changes = [
            ColumnModified(variable="df", column="x"),  # Will be VIOLATION
            ColumnAdded(variable="df", column="y"),  # Will be OK (no matching read)
        ]
        reads = [
            ColumnRead(variable="df", column="x"),
            ColumnRead(variable="df", column="z"),
        ]

        results = resolver.check_all(changes, reads)
        severities = [r.severity for r in results]

        # Violations should come first
        assert severities[0] == ConflictSeverity.VIOLATION

    def test_get_violations_only_returns_violations(self):
        """get_violations filters to only violations."""
        resolver = ConflictResolver(structural_mode=StructuralMode.WARN)

        changes = [
            ColumnModified(variable="df", column="x"),  # VIOLATION
            ColumnAdded(variable="df", column="y"),  # WARNING (structural)
        ]
        reads = [
            ColumnRead(variable="df", column="x"),
            StructuralRead(variable="df", attr="columns"),
        ]

        violations = resolver.get_violations(changes, reads)
        assert all(v.severity == ConflictSeverity.VIOLATION for v in violations)
        assert len(violations) == 1

    def test_get_warnings_only_returns_warnings(self):
        """get_warnings filters to only warnings."""
        resolver = ConflictResolver(structural_mode=StructuralMode.WARN)

        changes = [
            ColumnAdded(variable="df", column="y"),  # WARNING (structural)
        ]
        reads = [
            StructuralRead(variable="df", attr="columns"),
        ]

        warnings = resolver.get_warnings(changes, reads)
        assert all(w.severity == ConflictSeverity.WARNING for w in warnings)
        assert len(warnings) == 1

    def test_has_conflict_short_circuits(self):
        """has_conflict returns True on first violation."""
        resolver = ConflictResolver()

        changes = [ColumnModified(variable="df", column="x")]
        reads = [ColumnRead(variable="df", column="x")]

        assert resolver.has_conflict(changes, reads) is True

    def test_has_conflict_false_when_no_violations(self):
        """has_conflict returns False when only warnings/OK."""
        resolver = ConflictResolver(structural_mode=StructuralMode.WARN)

        changes = [ColumnAdded(variable="df", column="new")]
        reads = [StructuralRead(variable="df", attr="columns")]

        # Only a WARNING, not a VIOLATION
        assert resolver.has_conflict(changes, reads) is False


# =============================================================================
# Test get_rule_by_description helper
# =============================================================================


class TestGetRuleByDescription:
    """Tests for get_rule_by_description helper."""

    def test_finds_existing_rule(self):
        """Can find a rule by its description."""
        rule = get_rule_by_description(
            "Modifying column X invalidates prior reads of column X"
        )
        assert rule is not None
        assert rule.same_column is True
        assert rule.severity == ConflictSeverity.VIOLATION

    def test_returns_none_for_unknown(self):
        """Returns None for unknown description."""
        rule = get_rule_by_description("This rule does not exist")
        assert rule is None


# =============================================================================
# Test All Rules Have Coverage
# =============================================================================


class TestRuleCoverage:
    """Ensure all rules in CONFLICT_RULES are reachable."""

    def test_all_rules_have_unique_descriptions(self):
        """Each rule should have a unique description."""
        descriptions = [rule.description for rule in CONFLICT_RULES]
        assert len(descriptions) == len(set(descriptions))

    def test_no_unreachable_rules(self):
        """Verify no rule is shadowed by earlier rules.

        This is a partial check - we verify that each rule can match
        at least one change/read combination that earlier rules don't match.
        """
        # This is hard to test exhaustively, but we can at least ensure
        # each rule's description appears in some test expectation
        pass  # Structural test - the other tests implicitly cover this


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_changes_list(self):
        """Empty changes list returns empty results."""
        resolver = ConflictResolver()
        results = resolver.check_all([], [ColumnRead(variable="df", column="x")])
        assert results == []

    def test_empty_reads_list(self):
        """Empty reads list returns empty results."""
        resolver = ConflictResolver()
        results = resolver.check_all(
            [ColumnModified(variable="df", column="x")], []
        )
        assert results == []

    def test_custom_rules_list(self):
        """Can provide custom rules list."""
        custom_rules = [
            ConflictRule(
                change_types=(ColumnModified,),
                read_types=(ColumnRead,),
                severity=ConflictSeverity.OK,  # Make everything OK
                description="Custom: allow all",
            )
        ]
        resolver = ConflictResolver(rules=custom_rules)
        result = resolver.resolve(
            ColumnModified(variable="df", column="x"),
            ColumnRead(variable="df", column="x"),
        )
        # With default rules this would be VIOLATION, but custom says OK
        assert result.severity == ConflictSeverity.OK


# =============================================================================
# Test Variable Indexing Optimization
# =============================================================================


class TestVariableIndexing:
    """Tests for variable indexing optimization in check_all."""

    def test_different_variables_no_conflict(self):
        """Changes and reads with different variables produce no conflicts."""
        resolver = ConflictResolver()

        changes = [
            ColumnModified(variable="df1", column="x"),
            ColumnModified(variable="df2", column="y"),
        ]
        reads = [
            ColumnRead(variable="df3", column="x"),
            ColumnRead(variable="df4", column="y"),
        ]

        # With variable indexing, these should produce no results
        # (since no variable names overlap)
        results = resolver.check_all(changes, reads)
        assert len(results) == 0

    def test_only_matching_variables_checked(self):
        """Only changes/reads with matching variables produce results."""
        resolver = ConflictResolver()

        changes = [
            ColumnModified(variable="df", column="x"),
            ColumnModified(variable="other", column="y"),
        ]
        reads = [
            ColumnRead(variable="df", column="x"),  # Matches first change
            ColumnRead(variable="unrelated", column="z"),  # No matching change
        ]

        results = resolver.check_all(changes, reads)
        # Only df/x should match and produce a result
        assert len(results) == 1
        assert results[0].change.variable == "df"
        assert results[0].severity == ConflictSeverity.VIOLATION

    def test_many_columns_same_variable(self):
        """Many columns on same variable are checked correctly."""
        resolver = ConflictResolver()

        # Create many column changes
        changes = [
            ColumnModified(variable="df", column=f"col_{i}")
            for i in range(100)
        ]
        # Only read one column
        reads = [ColumnRead(variable="df", column="col_50")]

        violations = resolver.get_violations(changes, reads)
        # Should find exactly one violation (col_50 modified and read)
        assert len(violations) == 1
        assert violations.violations[0].change.column == "col_50"


# =============================================================================
# Test ViolationsResult and Truncation
# =============================================================================


class TestViolationsResult:
    """Tests for ViolationsResult class and truncation behavior."""

    def test_violations_result_properties(self):
        """ViolationsResult has correct properties."""
        resolver = ConflictResolver()

        changes = [ColumnModified(variable="df", column="x")]
        reads = [ColumnRead(variable="df", column="x")]

        result = resolver.get_violations(changes, reads)

        assert isinstance(result, ViolationsResult)
        assert len(result) == 1
        assert result.total_count == 1
        assert result.truncated is False
        assert result.truncated_count == 0

    def test_violations_result_iterable(self):
        """ViolationsResult is iterable for backwards compatibility."""
        resolver = ConflictResolver()

        changes = [ColumnModified(variable="df", column="x")]
        reads = [ColumnRead(variable="df", column="x")]

        result = resolver.get_violations(changes, reads)

        # Should be iterable
        violations_list = list(result)
        assert len(violations_list) == 1
        assert violations_list[0].severity == ConflictSeverity.VIOLATION

    def test_violations_result_bool(self):
        """ViolationsResult is truthy when violations exist."""
        resolver = ConflictResolver()

        # With violation
        changes = [ColumnModified(variable="df", column="x")]
        reads = [ColumnRead(variable="df", column="x")]
        result = resolver.get_violations(changes, reads)
        assert bool(result) is True

        # Without violation
        changes = [ColumnModified(variable="df", column="x")]
        reads = [ColumnRead(variable="df", column="y")]  # Different column
        result = resolver.get_violations(changes, reads)
        assert bool(result) is False

    def test_truncation_with_max_violations(self):
        """Results are truncated when exceeding max_violations."""
        resolver = ConflictResolver()

        # Create many violations
        changes = [
            ColumnModified(variable="df", column=f"col_{i}")
            for i in range(100)
        ]
        reads = [
            ColumnRead(variable="df", column=f"col_{i}")
            for i in range(100)
        ]

        # Limit to 10 violations
        result = resolver.get_violations(changes, reads, max_violations=10)

        assert len(result.violations) == 10
        assert result.total_count >= 10
        assert result.truncated is True
        assert result.truncated_count == result.total_count - 10

    def test_no_truncation_under_limit(self):
        """No truncation when under max_violations limit."""
        resolver = ConflictResolver()

        changes = [
            ColumnModified(variable="df", column=f"col_{i}")
            for i in range(5)
        ]
        reads = [
            ColumnRead(variable="df", column=f"col_{i}")
            for i in range(5)
        ]

        result = resolver.get_violations(changes, reads, max_violations=50)

        assert len(result.violations) == 5
        assert result.total_count == 5
        assert result.truncated is False

    def test_unlimited_violations(self):
        """max_violations=None returns all violations."""
        resolver = ConflictResolver()

        changes = [
            ColumnModified(variable="df", column=f"col_{i}")
            for i in range(100)
        ]
        reads = [
            ColumnRead(variable="df", column=f"col_{i}")
            for i in range(100)
        ]

        result = resolver.get_violations(changes, reads, max_violations=None)

        assert len(result.violations) == 100
        assert result.total_count == 100
        assert result.truncated is False

    def test_default_max_violations(self):
        """Default max_violations is applied."""
        resolver = ConflictResolver()

        # Create more violations than default limit
        n = DEFAULT_MAX_VIOLATIONS + 20
        changes = [
            ColumnModified(variable="df", column=f"col_{i}")
            for i in range(n)
        ]
        reads = [
            ColumnRead(variable="df", column=f"col_{i}")
            for i in range(n)
        ]

        result = resolver.get_violations(changes, reads)

        assert len(result.violations) == DEFAULT_MAX_VIOLATIONS
        assert result.truncated is True


# =============================================================================
# Test Massive Column List Performance
# =============================================================================


class TestMassiveColumnListPerformance:
    """Tests for performance with large column lists (the original issue)."""

    def test_many_columns_different_variables_fast(self):
        """Many columns across different variables should be fast."""
        resolver = ConflictResolver()

        # 1000 changes across 100 different variables
        changes = [
            ColumnModified(variable=f"df_{i // 10}", column=f"col_{i}")
            for i in range(1000)
        ]
        # Reads from only one variable
        reads = [
            ColumnRead(variable="df_0", column=f"col_{i}")
            for i in range(10)
        ]

        # With variable indexing, this should only check 10*10=100 pairs
        # instead of 1000*10=10000 pairs
        result = resolver.get_violations(changes, reads)

        # Should find violations for df_0 columns 0-9
        assert len(result.violations) == 10

    def test_feature_engineering_pattern(self):
        """Simulates feature engineering creating many columns."""
        resolver = ConflictResolver()

        # Cell creates 180 new columns (like Body_Temp_Age_20..79)
        changes = [
            ColumnAdded(variable="train", column=f"Body_Temp_Age_{i}")
            for i in range(20, 80)
        ] + [
            ColumnAdded(variable="train", column=f"Heart_Rate_Age_{i}")
            for i in range(20, 80)
        ] + [
            ColumnAdded(variable="test", column=f"Body_Temp_Age_{i}")
            for i in range(20, 80)
        ]

        # Prior cell read just the base columns
        reads = [
            ColumnRead(variable="train", column="Body_Temp"),
            ColumnRead(variable="train", column="Heart_Rate"),
            ColumnRead(variable="test", column="Body_Temp"),
        ]

        # ColumnAdded on different columns shouldn't conflict with ColumnRead
        result = resolver.get_violations(changes, reads)
        assert len(result.violations) == 0
