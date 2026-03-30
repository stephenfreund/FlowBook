"""
Fix reproducibility violations command implementation.

This command operates in two phases:
1. Propose: Generate a proposed_fix and store it in cell metadata
2. Apply: Apply the stored fix to modify the notebook cells
"""

import argparse
from typing import Any, Dict, Optional, List

from pydantic import BaseModel, Field

from flowbook.server.base import NotebookCommand, ProcessingResult
from flowbook.server.kernel_manager import FlowbookKernelClient
from flowbook.server.config import FlowbookConfig
from flowbook.agent.agent import FlowbookAgent, FlowbookStats
from flowbook.util.prompts import get_prompt
from flowbook.util.output import log
from flowbook.util.notebook_to_python import notebook_to_python, python_to_notebook_cells
from flowbook.util.flowbook_metadata import (
    ProposedFix,
    ProposedFixEntry,
    set_proposed_fix_flowbook_metadata,
    clear_proposed_fix_flowbook_metadata,
    get_flowbook_metadata_from_cell,
)


class FixProposal(BaseModel):
    """LLM output model for fix proposal."""

    strategy: str = Field(
        description="Fix strategy: 'alpha_rename', 'copy_value', 'merge_cells', or 'reorder'"
    )
    fix_entries: List[ProposedFixEntry] = Field(
        description="List of cell modifications to apply"
    )
    explanation: str = Field(
        description="Overall explanation of the fix strategy"
    )


class FixProposalAndStats(BaseModel):
    """Result and statistics from fix proposal generation."""

    proposal: FixProposal
    stats: FlowbookStats


class FixCommand(NotebookCommand):
    """Fix reproducibility violations using LLM-generated proposals."""

    @property
    def command_name(self) -> str:
        return "fix"

    @property
    def display_name(self) -> str:
        return "Fix Violation"

    @property
    def icon_name(self) -> str:
        return "ui-components:refresh"

    @property
    def tooltip(self) -> str:
        return "Fix reproducibility violation"

    @property
    def requires_kernel(self) -> bool:
        return False

    def make_subparser(
        self, subparsers: argparse._SubParsersAction
    ) -> argparse.ArgumentParser:
        """Create CLI subparser with fix-specific arguments."""
        subparser = subparsers.add_parser(
            self.command_name,
            help=self.display_name,
        )
        subparser.add_argument(
            "--silent",
            action="store_true",
            help="Generate and apply fix in one shot (no intermediate proposal step)",
        )
        subparser.add_argument(
            "--apply-only",
            action="store_true",
            help="Only apply existing proposed fixes, don't generate new ones",
        )
        return subparser

    def _fix_to_dict(self, fix: ProposedFix, applied: bool = False) -> Dict[str, Any]:
        """Convert a fix to a dictionary for metadata reporting."""
        return {
            "status": "applied" if applied else "proposed",
            "strategy": fix.strategy,
            "violation_type": fix.violation_type,
            "mutating_cell": fix.mutating_cell,
            "affected_cell": fix.affected_cell,
            "explanation": fix.explanation,
            "changes": [
                {
                    "cells": entry.cell_ids,
                    "explanation": entry.explanation,
                    "new_source": entry.modified_source if entry.modified_source.strip() else "(deleted)",
                }
                for entry in fix.fix_entries
            ],
        }

    def _get_violation_from_cell(self, cell: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract violation info from cell metadata."""
        metadata = cell.get("metadata", {})

        # Check flowbook.errors (canonical source from kernel metadata)
        flowbook = metadata.get("flowbook", {})
        if isinstance(flowbook, dict):
            errors = flowbook.get("errors", [])
            if errors:
                return errors[0]

        return None

    def _get_proposed_fix_from_cell(self, cell: Dict[str, Any]) -> Optional[ProposedFix]:
        """Extract proposed fix from cell metadata."""
        flowbook = get_flowbook_metadata_from_cell(cell)
        if flowbook and flowbook.get("proposed_fix"):
            return ProposedFix.model_validate(flowbook["proposed_fix"])
        return None

    def _find_cell_by_id(
        self, cells: List[Dict[str, Any]], cell_id: str
    ) -> Optional[Dict[str, Any]]:
        """Find a cell by its ID."""
        for cell in cells:
            if cell.get("id") == cell_id:
                return cell
        return None

    async def _generate_fix_proposal(
        self,
        notebook: Dict[str, Any],
        violation: Dict[str, Any],
        model: str,
    ) -> Optional[FixProposalAndStats]:
        """
        Generate a fix proposal for a violation using LLM.

        Args:
            notebook: The notebook content
            violation: Violation info dict with mutating_cell, affected_cell, variables
            model: The LLM model to use

        Returns:
            FixProposalAndStats if successful, None otherwise
        """
        violation_type = violation.get("type", violation.get("violation_type", "backward_mutation"))
        mutating_cell = violation.get("mutating_cell", "")
        affected_cell = violation.get("affected_cell", "")
        variables = violation.get("variables", [])
        message = violation.get("message", "")

        log(f"Generating fix proposal for violation: {violation_type}")

        # Convert notebook to annotated Python
        notebook_python = notebook_to_python(notebook)

        # Build changes detail if available
        changes_detail = ""
        if "column_changes" in violation:
            changes_detail = f"\nColumn changes: {violation['column_changes']}"

        # Create the agent
        agent = FlowbookAgent[FixProposal](
            key="fix_violation",
            model=model,
            instructions=get_prompt("fix_violation_instructions"),
            output_type=FixProposal,
        )

        # Format the input
        input_text = get_prompt(
            "fix_violation_input",
            violation_type=violation_type,
            mutating_cell_id=mutating_cell,
            affected_cell_id=affected_cell,
            variables=", ".join(variables),
            violation_message=message,
            changes_detail=changes_detail,
            notebook_python=notebook_python,
        )

        # Run the agent
        result, stats = await agent.run(input_text)

        log(
            f"Generated fix proposal | Strategy: {result.strategy} | "
            f"Tokens: {stats.usage.total_tokens} | Cost: ${stats.cost:.4f}"
        )

        return FixProposalAndStats(proposal=result, stats=stats)

    def _apply_fix(
        self, notebook: Dict[str, Any], proposed_fix: ProposedFix
    ) -> Dict[str, Any]:
        """
        Apply a proposed fix to the notebook.

        Args:
            notebook: The notebook content
            proposed_fix: The fix to apply

        Returns:
            Modified notebook with fixes applied
        """
        cells = list(notebook.get("cells", []))
        cells_by_id = {cell.get("id"): i for i, cell in enumerate(cells)}

        # Track cells to delete (indices)
        cells_to_delete: set = set()

        for entry in proposed_fix.fix_entries:
            cell_ids = entry.cell_ids

            if not cell_ids:
                continue

            if len(cell_ids) == 1:
                # Single cell modification
                cell_id = cell_ids[0]
                if cell_id not in cells_by_id:
                    log(f"Warning: Cell {cell_id} not found in notebook")
                    continue

                idx = cells_by_id[cell_id]

                if not entry.modified_source.strip():
                    # Empty source means delete cell
                    cells_to_delete.add(idx)
                    log(f"Marking cell {cell_id} for deletion")
                else:
                    # Update cell source
                    cells[idx] = dict(cells[idx])
                    cells[idx]["source"] = entry.modified_source
                    # Clear proposed_fix metadata after applying
                    clear_proposed_fix_flowbook_metadata(cells[idx])
                    log(f"Updated cell {cell_id}")

            else:
                # Multi-cell merge: combine into first cell, delete rest
                first_id = cell_ids[0]
                if first_id not in cells_by_id:
                    log(f"Warning: Cell {first_id} not found for merge")
                    continue

                first_idx = cells_by_id[first_id]
                cells[first_idx] = dict(cells[first_idx])
                cells[first_idx]["source"] = entry.modified_source
                clear_proposed_fix_flowbook_metadata(cells[first_idx])
                log(f"Merged cells {cell_ids} into {first_id}")

                # Mark other cells for deletion
                for other_id in cell_ids[1:]:
                    if other_id in cells_by_id:
                        cells_to_delete.add(cells_by_id[other_id])

        # Remove deleted cells (in reverse order to preserve indices)
        for idx in sorted(cells_to_delete, reverse=True):
            del cells[idx]

        # Return modified notebook
        result = dict(notebook)
        result["cells"] = cells
        return result

    async def process(
        self,
        notebook_content: Dict[str, Any],
        kernel_client: Optional[FlowbookKernelClient] = None,
        selected_cell_ids: Optional[List[str]] = None,
        config: Optional[FlowbookConfig] = None,
        cell_id: Optional[str] = None,
        silent: bool = False,
        **kwargs,
    ) -> ProcessingResult:
        """
        Process the fix command.

        If a cell has a violation but no proposed_fix, generate one.
        If a cell has a proposed_fix, apply it.
        If silent=True, generate and apply in one shot.

        Args:
            notebook_content: The notebook content
            kernel_client: Optional kernel client (not used)
            selected_cell_ids: Optional list of selected cell IDs
            config: Configuration with model settings
            cell_id: Optional specific cell ID to fix
            silent: If True, apply fix immediately after generating
        """
        with self.timing_context() as get_elapsed:
            log("Processing fix command")

            nb = notebook_content
            cells = nb.get("cells", [])
            total_cost = 0.0
            proposals_generated = 0
            fixes_applied = 0
            fixes_list: List[Dict[str, Any]] = []

            # Determine which cells to process
            if cell_id:
                target_cells = [c for c in cells if c.get("id") == cell_id]
            elif selected_cell_ids:
                target_cells = [c for c in cells if c.get("id") in selected_cell_ids]
            else:
                # Process all cells with violations
                target_cells = [c for c in cells if self._get_violation_from_cell(c)]

            for cell in target_cells:
                cell_id_current = cell.get("id", "unknown")
                violation = self._get_violation_from_cell(cell)
                proposed_fix = self._get_proposed_fix_from_cell(cell)

                if proposed_fix and not silent:
                    # Apply existing proposed fix
                    log(f"Applying proposed fix for cell {cell_id_current}")
                    nb = self._apply_fix(nb, proposed_fix)
                    fixes_applied += 1
                    fixes_list.append(self._fix_to_dict(proposed_fix, applied=True))

                elif violation and not proposed_fix:
                    # Generate new proposal
                    result = await self._generate_fix_proposal(
                        nb,
                        violation,
                        config.model if config else "claude-opus-4-5",
                    )

                    if result:
                        # Create ProposedFix from LLM output
                        fix = ProposedFix(
                            violation_type=violation.get("type", violation.get("violation_type", "backward_mutation")),
                            mutating_cell=violation.get("mutating_cell", ""),
                            affected_cell=violation.get("affected_cell", ""),
                            strategy=result.proposal.strategy,
                            fix_entries=result.proposal.fix_entries,
                            explanation=result.proposal.explanation,
                        )

                        if silent:
                            # Apply immediately
                            nb = self._apply_fix(nb, fix)
                            fixes_applied += 1
                            fixes_list.append(self._fix_to_dict(fix, applied=True))
                        else:
                            # Store in metadata
                            # Find and update the cell in nb
                            for i, c in enumerate(nb["cells"]):
                                if c.get("id") == cell_id_current:
                                    nb["cells"][i] = dict(c)
                                    set_proposed_fix_flowbook_metadata(nb["cells"][i], fix)
                                    break
                            fixes_list.append(self._fix_to_dict(fix, applied=False))

                        proposals_generated += 1
                        total_cost += result.stats.cost

            metadata = {
                "status": "success",
                "command": self.command_name,
                "proposals_generated": proposals_generated,
                "fixes_applied": fixes_applied,
                "total_cost": total_cost,
                "fixes": fixes_list,
            }

            log(
                f"Fix command completed | Proposals: {proposals_generated} | "
                f"Applied: {fixes_applied} | Cost: ${total_cost:.4f}"
            )

            total_time = get_elapsed()

        return ProcessingResult(
            notebook=nb,
            metadata=metadata,
            total_cost=total_cost,
            total_time=total_time,
        )
