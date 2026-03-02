"""
AST-based inference of reads/writes from Python code for litmus tests.

This module provides the `infer_rw` function that analyzes Python code
to extract variable reads, writes, column accesses, and structural reads.
"""

import ast
from dataclasses import dataclass, field
from typing import Dict, Set


@dataclass
class InferredRW:
    """Result of inferring reads/writes from Python code."""
    reads: Set[str] = field(default_factory=set)
    writes: Set[str] = field(default_factory=set)
    column_reads: Dict[str, Set[str]] = field(default_factory=dict)  # var -> {cols}
    column_writes: Dict[str, Set[str]] = field(default_factory=dict)  # var -> {cols}
    structural_reads: Dict[str, Set[str]] = field(default_factory=dict)  # var -> {"shape"}


def infer_rw(code: str) -> InferredRW:
    """
    AST-based inference of reads/writes from Python code.

    Handles:
    - Variables: x = 1 → writes x
    - Variable reads: y = x + 1 → reads x, writes y
    - Column reads: df['price'] → column_reads={'df': {'price'}}
    - Column writes: df['new'] = ... → column_writes={'df': {'new'}}
    - Structural: df.shape → structural_reads={'df': {'shape'}}

    >>> infer_rw("x = 1")
    InferredRW(reads=set(), writes={'x'}, ...)

    >>> infer_rw("y = df['price'] * 2")
    InferredRW(reads={'df'}, writes={'y'}, column_reads={'df': {'price'}}, ...)

    >>> infer_rw("n = len(df)")
    InferredRW(reads={'df'}, writes={'n'}, structural_reads={'df': {'shape'}}, ...)

    >>> infer_rw("df['new'] = df['x'] + 1")
    InferredRW(reads={'df'}, writes={'df'},
               column_reads={'df': {'x'}}, column_writes={'df': {'new'}}, ...)
    """
    reads: Set[str] = set()
    writes: Set[str] = set()
    column_reads: Dict[str, Set[str]] = {}
    column_writes: Dict[str, Set[str]] = {}
    structural_reads: Dict[str, Set[str]] = {}

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return InferredRW()

    class RWVisitor(ast.NodeVisitor):
        def __init__(self):
            self.in_subscript_store = False
            self.subscript_store_var = None

        def visit_Name(self, node: ast.Name) -> None:
            if isinstance(node.ctx, ast.Store):
                writes.add(node.id)
            elif isinstance(node.ctx, ast.Load):
                reads.add(node.id)
            self.generic_visit(node)

        def visit_Subscript(self, node: ast.Subscript) -> None:
            if isinstance(node.value, ast.Name):
                var = node.value.id
                # Extract string key if present
                key = None
                if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                    key = node.slice.value

                if key is not None:
                    if isinstance(node.ctx, ast.Store):
                        column_writes.setdefault(var, set()).add(key)
                        writes.add(var)  # Writing to a column means writing to the var
                    else:
                        column_reads.setdefault(var, set()).add(key)
                        reads.add(var)

            # Continue visiting children
            self.generic_visit(node)

        def visit_Attribute(self, node: ast.Attribute) -> None:
            if isinstance(node.value, ast.Name):
                var = node.value.id
                # Track structural reads
                if node.attr in ('shape', 'columns', 'index', 'dtypes'):
                    structural_reads.setdefault(var, set()).add(node.attr)
                    reads.add(var)

            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:
            # len(df) → structural read of shape
            if isinstance(node.func, ast.Name) and node.func.id == 'len':
                if node.args and isinstance(node.args[0], ast.Name):
                    var = node.args[0].id
                    structural_reads.setdefault(var, set()).add('shape')
                    reads.add(var)

            self.generic_visit(node)

    visitor = RWVisitor()
    visitor.visit(tree)

    # reads_before_writes = reads that aren't also written (at variable level)
    # But we need to be careful: if we write to a column of df, we still read df first
    final_reads = reads - (writes - set(column_writes.keys()))

    return InferredRW(
        reads=final_reads,
        writes=writes,
        column_reads=column_reads,
        column_writes=column_writes,
        structural_reads=structural_reads,
    )
