"""
Prepare Code for FlowBook command implementation.

This command transforms pandas chained assignment patterns to prevent
ChainedAssignmentError by rewriting them to explicit assignment form.
"""

import copy
from typing import Any, Dict, List, Optional

import libcst as cst
from libcst import matchers as m

from flowbook.server.base import NotebookCommand, ProcessingResult
from flowbook.server.kernel_manager import FlowbookKernelClient


class ChainedAssignmentRewriter(cst.CSTTransformer):
    """
    LibCST transformer that rewrites chained assignment patterns.

    Transforms:
        df[subscript].method(..., inplace=True)
    To:
        df[subscript] = df[subscript].method(...)

    Preserves all formatting, comments, and code style.
    """

    def __init__(self):
        super().__init__()
        self.transformations: List[Dict[str, Any]] = []

    def leave_SimpleStatementLine(
        self, original_node: cst.SimpleStatementLine, updated_node: cst.SimpleStatementLine
    ) -> cst.SimpleStatementLine:
        """
        Process simple statements to detect and rewrite chained assignments.

        Pattern: df[subscript].method(..., inplace=True)
        Where df[subscript] is any subscript expression.
        """
        # Only process if there's exactly one statement (an Expr)
        if len(updated_node.body) != 1:
            return updated_node

        stmt = updated_node.body[0]
        if not isinstance(stmt, cst.Expr):
            return updated_node

        expr_value = stmt.value
        if not isinstance(expr_value, cst.Call):
            return updated_node

        # Check if it's a method call on a subscript
        # Pattern: subscript.method(...)
        if not isinstance(expr_value.func, cst.Attribute):
            return updated_node

        obj = expr_value.func.value
        if not isinstance(obj, cst.Subscript):
            return updated_node

        # Check if inplace=True is in the arguments
        has_inplace_true = False
        inplace_arg_index = -1

        for i, arg in enumerate(expr_value.args):
            if isinstance(arg.keyword, cst.Name) and arg.keyword.value == "inplace":
                # Check if the value is True
                if m.matches(arg.value, m.Name("True")):
                    has_inplace_true = True
                    inplace_arg_index = i
                elif m.matches(arg.value, m.Name("False")):
                    # Explicitly False, don't transform
                    return updated_node

        if not has_inplace_true:
            return updated_node

        # Build new args list without inplace=True and fix trailing commas
        new_args = []
        for i, arg in enumerate(expr_value.args):
            if i == inplace_arg_index:
                # Skip the inplace argument
                continue

            # If this is the last remaining arg and the inplace arg came after it,
            # remove any trailing comma
            is_last_arg = (i == len(expr_value.args) - 2 and inplace_arg_index == len(expr_value.args) - 1)
            if is_last_arg and arg.comma:
                # Remove trailing comma from last arg
                arg = arg.with_changes(comma=cst.MaybeSentinel.DEFAULT)

            new_args.append(arg)

        # Build the transformation:
        # subscript = subscript.method(...)

        # Create the target (left side of assignment)
        target = obj  # The subscript expression

        # Create the value (right side of assignment)
        # This is the same method call but without inplace=True
        new_call = expr_value.with_changes(args=new_args)

        # Create the assignment statement
        assignment = cst.Assign(
            targets=[cst.AssignTarget(target=target)],
            value=new_call,
        )

        # Track the transformation
        try:
            original_code = original_node.body[0].value  # type: ignore
            self.transformations.append({
                "from": cst.Module([original_node]).code.strip(),
                "to": cst.Module([cst.SimpleStatementLine([assignment])]).code.strip(),
            })
        except Exception:
            # If we can't extract the code, still do the transformation
            pass

        # Return the updated statement line with the assignment
        return updated_node.with_changes(
            body=[assignment]
        )


class PrepareCodeForFlowbookCommand(NotebookCommand):
    """
    Transform pandas chained assignment patterns to prevent ChainedAssignmentError.

    This command uses LibCST to rewrite problematic patterns like:
        df['col'].fillna(0, inplace=True)
    to:
        df['col'] = df['col'].fillna(0)

    All formatting, comments, and code style are preserved.
    """

    @property
    def command_name(self) -> str:
        return "prepare_code"

    @property
    def display_name(self) -> str:
        return "Prepare Code for FlowBook"

    @property
    def icon_name(self) -> str:
        return "ui-components:code"

    @property
    def tooltip(self) -> str:
        return "Transform chained assignments to prevent pandas errors"

    def process(
        self,
        notebook_content: Dict[str, Any],
        kernel_client: Optional[FlowbookKernelClient] = None,
        selected_cell_ids: Optional[List[str]] = None,
        config: Optional[Any] = None,
        **kwargs,
    ) -> ProcessingResult:
        """
        Process the notebook and transform chained assignment patterns.

        Args:
            notebook_content: The parsed JSON content of a Jupyter notebook
            kernel_client: Optional kernel client (not used)
            selected_cell_ids: Optional list of cell IDs to transform
            config: Optional configuration (not used)
            **kwargs: Additional parameters

        Returns:
            ProcessingResult containing the transformed notebook and metadata
        """
        with self.timing_context() as get_elapsed:
            new_notebook = copy.deepcopy(notebook_content)
            cells = new_notebook.get("cells", [])

            cells_modified = 0
            total_transformations = 0
            transformation_summary = []

            for cell in cells:
                # Skip non-code cells
                if cell.get("cell_type") != "code":
                    continue

                # Get cell ID
                cell_id = cell.get("id", "")

                # If selected_cell_ids is provided, only process those cells
                if selected_cell_ids is not None and cell_id not in selected_cell_ids:
                    continue

                # Get cell source
                source = cell.get("source", "")
                if isinstance(source, list):
                    source = "".join(source)

                # Skip empty cells
                if not source.strip():
                    continue

                # Try to parse and transform the code
                try:
                    # Parse the code with LibCST
                    module = cst.parse_module(source)

                    # Apply the transformer
                    rewriter = ChainedAssignmentRewriter()
                    transformed = module.visit(rewriter)

                    # Check if any transformations were made
                    if rewriter.transformations:
                        # Update the cell source
                        new_source = transformed.code
                        cell["source"] = new_source

                        # Track transformations
                        cells_modified += 1
                        total_transformations += len(rewriter.transformations)

                        transformation_summary.append({
                            "cell_id": cell_id,
                            "transformations": rewriter.transformations,
                        })

                except Exception as e:
                    # If parsing fails, skip this cell
                    # Could be syntax error or other issue
                    continue

            # Build metadata
            metadata = {
                "status": "success",
                "command": self.command_name,
                "transformations": {
                    "cells_modified": cells_modified,
                    "total_transformations": total_transformations,
                    "summary": transformation_summary,
                },
                "message": f"Applied {total_transformations} transformation{'s' if total_transformations != 1 else ''} to {cells_modified} cell{'s' if cells_modified != 1 else ''}",
            }

            total_time = get_elapsed()

        return ProcessingResult(
            notebook=new_notebook,
            metadata=metadata,
            total_cost=0.0,
            total_time=total_time,
        )
