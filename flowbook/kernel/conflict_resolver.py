"""
Conflict Resolver - Matches Changes against AccessEvents using declarative rules.

This module provides the ConflictResolver class that evaluates whether a Change
conflicts with a prior AccessEvent by matching against the CONFLICT_RULES table.

Usage:
    resolver = ConflictResolver(structural_mode=StructuralMode.WARN)

    # Check single change against single read
    result = resolver.resolve(change, prior_read)
    if result.severity == ConflictSeverity.VIOLATION:
        print(f"Conflict: {result.description}")

    # Check all changes against all prior reads
    conflicts = resolver.check_all(changes, prior_reads)
    violations = [c for c in conflicts if c.severity == ConflictSeverity.VIOLATION]
"""

from typing import List, Optional

from pydantic import BaseModel, ConfigDict

from flowbook.kernel.access_events import AccessEvent
from flowbook.kernel.changes import Change
from flowbook.kernel.conflict_rules import (
    CONFLICT_RULES,
    ConflictRule,
    ConflictSeverity,
    StructuralMode,
)


class ConflictResult(BaseModel):
    """
    Result of checking a Change against an AccessEvent.

    Attributes:
        change: The change that was checked
        read: The prior read that was checked against
        severity: The conflict severity (OK, WARNING, VIOLATION)
        rule: The rule that matched (None if no rule matched -> OK)
        description: Human-readable explanation
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    change: Change
    read: AccessEvent
    severity: ConflictSeverity
    rule: Optional[ConflictRule] = None
    description: str


class ConflictResolver:
    """
    Resolves conflicts between Changes and AccessEvents using declarative rules.

    The resolver iterates through CONFLICT_RULES in order, returning the result
    of the first matching rule. If no rule matches, the result is OK.

    Structural Mode:
        Rules marked with is_structural=True have mode-dependent severity:
        - ENFORCE: structural conflicts are VIOLATION
        - WARN: structural conflicts are WARNING
        - OFF: structural conflicts are OK (ignored)

    Thread Safety:
        ConflictResolver is stateless and safe to use from multiple threads.
        The rules list is immutable and shared.

    Example:
        resolver = ConflictResolver(structural_mode=StructuralMode.WARN)

        # Single check
        result = resolver.resolve(
            ColumnModified(variable='df', column='price'),
            ColumnRead(variable='df', column='price')
        )
        # result.severity == ConflictSeverity.VIOLATION

        # Batch check
        conflicts = resolver.check_all(changes, prior_reads)
        for c in conflicts:
            if c.severity != ConflictSeverity.OK:
                print(f"{c.change} conflicts with {c.read}: {c.description}")
    """

    def __init__(
        self,
        structural_mode: StructuralMode = StructuralMode.WARN,
        rules: Optional[List[ConflictRule]] = None,
    ):
        """
        Initialize the resolver.

        Args:
            structural_mode: How to handle structural conflicts
            rules: Custom rules list (defaults to CONFLICT_RULES)
        """
        self.structural_mode = structural_mode
        self.rules = rules if rules is not None else CONFLICT_RULES

    def _apply_mode(self, rule: ConflictRule) -> ConflictSeverity:
        """
        Apply structural_mode to determine actual severity for a rule.

        For non-structural rules, returns the rule's severity unchanged.
        For structural rules, maps based on structural_mode:
            - ENFORCE -> VIOLATION
            - WARN -> WARNING
            - OFF -> OK

        Args:
            rule: The matched rule

        Returns:
            The effective severity after applying structural_mode
        """
        if not rule.is_structural:
            return rule.severity

        # Structural rule - apply mode
        if self.structural_mode == StructuralMode.ENFORCE:
            return ConflictSeverity.VIOLATION
        elif self.structural_mode == StructuralMode.WARN:
            return ConflictSeverity.WARNING
        else:  # OFF
            return ConflictSeverity.OK

    def resolve(self, change: Change, read: AccessEvent) -> ConflictResult:
        """
        Check if a change conflicts with a prior read.

        Iterates through rules in order, returning the result of the first
        matching rule. If no rule matches, returns OK.

        Args:
            change: The detected change from current cell
            read: The access event from a prior cell

        Returns:
            ConflictResult with severity and explanation
        """
        # Must be same variable
        if change.variable != read.variable:
            return ConflictResult(
                change=change,
                read=read,
                severity=ConflictSeverity.OK,
                rule=None,
                description="Different variables - no conflict",
            )

        # Find first matching rule
        for rule in self.rules:
            if rule.matches(change, read):
                severity = self._apply_mode(rule)
                return ConflictResult(
                    change=change,
                    read=read,
                    severity=severity,
                    rule=rule,
                    description=rule.description,
                )

        # No rule matched - default to OK
        return ConflictResult(
            change=change,
            read=read,
            severity=ConflictSeverity.OK,
            rule=None,
            description="No matching rule - no conflict",
        )

    def check_all(
        self,
        changes: List[Change],
        prior_reads: List[AccessEvent],
    ) -> List[ConflictResult]:
        """
        Check all changes against all prior reads.

        Returns a list of ConflictResults for every change/read pair.
        Non-OK results appear first (violations, then warnings), sorted
        by severity for easy processing.

        Args:
            changes: List of changes from current cell
            prior_reads: List of access events from prior cells

        Returns:
            List of ConflictResults, sorted by severity (worst first)
        """
        results: List[ConflictResult] = []

        for change in changes:
            for read in prior_reads:
                result = self.resolve(change, read)
                results.append(result)

        # Sort by severity: VIOLATION > WARNING > OK
        severity_order = {
            ConflictSeverity.VIOLATION: 0,
            ConflictSeverity.WARNING: 1,
            ConflictSeverity.OK: 2,
        }
        results.sort(key=lambda r: severity_order[r.severity])

        return results

    def get_violations(
        self,
        changes: List[Change],
        prior_reads: List[AccessEvent],
    ) -> List[ConflictResult]:
        """
        Get only the violations (not warnings or OK).

        Convenience method for checking if a cell should be rolled back.

        Args:
            changes: List of changes from current cell
            prior_reads: List of access events from prior cells

        Returns:
            List of ConflictResults with VIOLATION severity
        """
        all_results = self.check_all(changes, prior_reads)
        return [r for r in all_results if r.severity == ConflictSeverity.VIOLATION]

    def get_warnings(
        self,
        changes: List[Change],
        prior_reads: List[AccessEvent],
    ) -> List[ConflictResult]:
        """
        Get only the warnings (not violations or OK).

        Convenience method for structural warnings that don't block execution.

        Args:
            changes: List of changes from current cell
            prior_reads: List of access events from prior cells

        Returns:
            List of ConflictResults with WARNING severity
        """
        all_results = self.check_all(changes, prior_reads)
        return [r for r in all_results if r.severity == ConflictSeverity.WARNING]

    def has_conflict(
        self,
        changes: List[Change],
        prior_reads: List[AccessEvent],
    ) -> bool:
        """
        Quick check if any violations exist.

        More efficient than get_violations() when you only need a boolean.
        Short-circuits on first violation found.

        Args:
            changes: List of changes from current cell
            prior_reads: List of access events from prior cells

        Returns:
            True if any change/read pair is a VIOLATION
        """
        for change in changes:
            for read in prior_reads:
                result = self.resolve(change, read)
                if result.severity == ConflictSeverity.VIOLATION:
                    return True
        return False
