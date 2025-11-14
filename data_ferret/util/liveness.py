"""
Liveness analysis for Jupyter notebooks.

This module provides backward dataflow analysis to determine which global variables
are "live" (will be used in the future) at the end of each notebook cell.

OVERVIEW
========

Liveness analysis answers: "Which variables defined so far will be needed later?"
This complements dependency analysis (which answers "What does this cell need?") by
identifying variables that are:
- LIVE: Will be read by subsequent cells
- DEAD: Will never be read again (safe to discard/garbage collect)

ALGORITHM
=========

Uses classic backward dataflow analysis with gen/kill sets:

1. **Build gen/kill sets from dependencies.py**:
   - gen[cell] = variables READ by this cell (includes transitive closure)
   - kill[cell] = variables WRITTEN by this cell (new definitions)

2. **Backward propagation**:
   - Start from last cell: live_out[last] = ∅ (nothing needed after)
   - For each cell (backwards):
       live_in[cell] = (live_out[cell] - kill[cell]) ∪ gen[cell]
       live_out[prev] = live_in[cell]

3. **Dataflow equations**:
   - live_in[cell] = variables live BEFORE cell executes
   - live_out[cell] = variables live AFTER cell executes
   - A variable is live if it will be read in the future (before being redefined)

FUNCTION HANDLING
=================

**No special logic needed!** Functions work correctly through standard dataflow:

1. **Function Definitions**:
   - `def f(): ...` is treated as writing to variable `f`
   - Appears in kill[cell] just like any variable assignment

2. **Function Calls**:
   - `f()` reads `f` plus ALL variables `f` uses (transitively)
   - dependencies.py computes transitive closure automatically
   - Appears in gen[cell] with complete dependency set

3. **Python Semantics**:
   - Functions capture variable NAMES not VALUES (late binding)
   - When variable redefined between function def and call, function sees new value
   - This is correctly modeled: old definition is killed, new one is live

Examples:

    # Example 1: Function called - dependencies stay live
    # Cell 1: x = 5
    # Cell 2: def f(): return x + 10
    # Cell 3: result = f()
    #
    # Result: Cell 1 live_out = {x}, Cell 2 live_out = {f, x}
    # Both x and f needed for Cell 3

    # Example 2: Function never called - dependencies die
    # Cell 1: x = 5
    # Cell 2: def f(): return x + 10
    # Cell 3: y = 20
    #
    # Result: Cell 1 live_out = {}, Cell 2 live_out = {}
    # Neither x nor f are ever used

    # Example 3: Variable redefined between def and call
    # Cell 1: x = 5
    # Cell 2: def f(): return x + 10
    # Cell 3: x = 20  # Redefine x
    # Cell 4: result = f()  # f() sees x=20, not x=5
    #
    # Result: Cell 1 live_out = {} (x from Cell 1 is DEAD)
    #         Cell 2 live_out = {f}
    #         Cell 3 live_out = {f, x}
    # Correctly models Python late binding

    # Example 4: Nested function calls
    # Cell 1: def g(): return y
    # Cell 2: def f(): return g() + 10
    # Cell 3: y = 5
    # Cell 4: result = f()
    #
    # Result: Cell 1 live_out = {g}
    #         Cell 2 live_out = {f, g}
    #         Cell 3 live_out = {f, g, y}
    # Transitive closure includes y in f's dependencies

PRECISION AND CONSERVATIVENESS
==============================

This analysis inherits precision characteristics from dependencies.py:

**Conservative (Over-Approximation)**:
- May mark variables as LIVE when they won't actually be used
- Never marks LIVE variables as DEAD (safe for optimization)
- Conservative assumptions from dependencies.py propagate to liveness

**Flow-Insensitive Across Cells**:
- Assumes sequential cell execution (cell_0, cell_1, ..., cell_n)
- Does not model control flow or conditional execution of cells
- Does not track how many times cells execute

**Flow-Sensitive Within Cells**:
- Inherits within-cell flow sensitivity from dependencies.py
- Variables written before being read don't appear in gen set

**Writes as New Definitions**:
- When cell writes to `x`, it creates a NEW definition
- Previous definition of `x` is killed (no longer live)
- Example: `x = 5; ... later cells ...; x = 10` - first x is killed by second assignment

USE CASES
=========

1. **Dead Code Detection**: Identify cells that define variables never used
2. **Checkpoint Optimization**: Don't checkpoint dead variables
3. **Garbage Collection**: Variables in (live_in - live_out) can be freed after cell
4. **Notebook Optimization**: Remove/reorder cells that define unused variables
5. **Memory Profiling**: Focus on live variables for memory analysis

COMPARISON WITH DEPENDENCIES
============================

| Analysis      | Question                           | Direction | Result              |
|---------------|------------------------------------| --------- |---------------------|
| Dependencies  | What does this cell need?          | Forward   | globals_read        |
| Liveness      | What will be needed in the future? | Backward  | live_out            |

Both analyses are complementary:
- Dependencies: "This cell reads {x, y, f}"
- Liveness: "After this cell, {y, f} are still needed but {x} is dead"

LIMITATIONS
===========

1. **Sequential Execution Only**: Assumes cells run in order (0, 1, 2, ...)
2. **No Iteration**: Doesn't model notebooks with cells executed multiple times
3. **No Conditional Execution**: Doesn't model skip/re-run patterns
4. **Inherits Dependencies Limitations**: Under-approximations in dependencies.py
   (eval, dynamic imports, etc.) may cause variables to incorrectly appear dead

Example Usage
-------------
    from data_ferret.util.liveness import analyze_notebook_liveness, get_live_out_variables

    notebook = {"cells": [...]}

    # Full analysis
    liveness = analyze_notebook_liveness(notebook)
    for cell_id, info in liveness.items():
        print(f"Cell {cell_id}:")
        print(f"  Live out: {info.live_out}")
        print(f"  Dead after: {info.live_in - info.live_out}")

    # Simplified API
    live_vars = get_live_out_variables(notebook)
    for cell_id, vars_list in live_vars.items():
        print(f"After cell {cell_id}, live: {vars_list}")
"""

from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, field

from data_ferret.util.dependencies import analyze_notebook, CellDependencies


@dataclass
class CellLiveness:
    """Represents liveness information for a single cell."""

    cell_id: str
    live_in: Set[str] = field(default_factory=set)   # Variables live BEFORE this cell
    live_out: Set[str] = field(default_factory=set)  # Variables live AFTER this cell
    gen: Set[str] = field(default_factory=set)       # Variables READ by this cell
    kill: Set[str] = field(default_factory=set)      # Variables WRITTEN by this cell

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'cell_id': self.cell_id,
            'live_in': sorted(list(self.live_in)),
            'live_out': sorted(list(self.live_out)),
            'gen': sorted(list(self.gen)),
            'kill': sorted(list(self.kill)),
            'dead_after': sorted(list(self.live_in - self.live_out))
        }


def analyze_notebook_liveness(
    notebook: Dict[str, Any],
    dependencies: Optional[Dict[str, CellDependencies]] = None
) -> Dict[str, CellLiveness]:
    """
    Analyze liveness for all cells in a notebook using backward dataflow analysis.

    This function determines which global variables are "live" (will be used in the
    future) at each point in the notebook. A variable is live at the end of a cell
    if it will be read by a subsequent cell before being redefined.

    Args:
        notebook: Jupyter notebook content as a dictionary
        dependencies: Pre-computed dependencies (optional, will compute if None)
                     Providing this avoids re-parsing the notebook

    Returns:
        Dictionary mapping cell IDs to CellLiveness objects containing:
        - live_in: Variables live before the cell executes
        - live_out: Variables live after the cell executes
        - gen: Variables read by this cell (from dependencies)
        - kill: Variables written by this cell (from dependencies)

    Algorithm:
        1. Compute dependencies if not provided (or reuse provided ones)
        2. Build gen/kill sets from dependencies:
           - gen[cell] = dependencies[cell].globals_read
           - kill[cell] = dependencies[cell].globals_written
        3. Backward dataflow analysis:
           - Initialize: live_out[last_cell] = ∅
           - For each cell (in reverse order):
               live_in[cell] = (live_out[cell] - kill[cell]) ∪ gen[cell]
               live_out[prev_cell] = live_in[cell]

    Example:
        >>> notebook = {
        ...     "cells": [
        ...         {"id": "c1", "cell_type": "code", "source": "x = 5"},
        ...         {"id": "c2", "cell_type": "code", "source": "y = x + 10"},
        ...         {"id": "c3", "cell_type": "code", "source": "z = y * 2"}
        ...     ]
        ... }
        >>> liveness = analyze_notebook_liveness(notebook)
        >>> liveness["c1"].live_out
        {'x'}
        >>> liveness["c2"].live_out
        {'y'}
        >>> liveness["c3"].live_out
        set()
    """
    # Step 1: Get or compute dependencies
    if dependencies is None:
        dependencies = analyze_notebook(notebook)

    # Step 2: Get ordered list of code cells
    cells = notebook.get('cells', [])
    code_cells = []
    for cell in cells:
        if cell.get('cell_type') != 'code':
            continue
        cell_id = cell.get('id', '')
        if cell_id and cell_id in dependencies:
            code_cells.append(cell_id)

    if not code_cells:
        return {}

    # Step 3: Build gen/kill sets and initialize liveness info
    liveness: Dict[str, CellLiveness] = {}

    for cell_id in code_cells:
        deps = dependencies[cell_id]
        liveness[cell_id] = CellLiveness(
            cell_id=cell_id,
            gen=deps.globals_read.copy(),
            kill=deps.globals_written.copy()
        )

    # Step 4: Backward dataflow analysis
    # Initialize last cell: nothing is live after the last cell
    liveness[code_cells[-1]].live_out = set()

    # Process cells in reverse order
    for i in range(len(code_cells) - 1, -1, -1):
        cell_id = code_cells[i]
        cell_liveness = liveness[cell_id]

        # Compute live_in using dataflow equation:
        # live_in[cell] = (live_out[cell] - kill[cell]) ∪ gen[cell]
        #
        # Explanation:
        # - Start with variables live after this cell (live_out)
        # - Remove variables this cell redefines (kill) - they're new definitions
        # - Add variables this cell reads (gen) - they must be live before
        cell_liveness.live_in = (
            (cell_liveness.live_out - cell_liveness.kill) | cell_liveness.gen
        )

        # Propagate to previous cell: what's live before this cell must be
        # live after the previous cell
        if i > 0:
            prev_cell_id = code_cells[i - 1]
            liveness[prev_cell_id].live_out = cell_liveness.live_in.copy()

    return liveness


def get_live_out_variables(notebook: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Get variables that are live at the END of each cell.

    This is the primary use case: determining which variables are still needed
    after each cell completes execution.

    Args:
        notebook: Jupyter notebook content as a dictionary

    Returns:
        Dictionary mapping cell IDs to sorted lists of live variable names

    Example:
        >>> live_vars = get_live_out_variables(notebook)
        >>> print(live_vars["cell_5"])
        ['df', 'model', 'results']  # These variables needed by later cells
    """
    liveness = analyze_notebook_liveness(notebook)
    return {
        cell_id: sorted(list(info.live_out))
        for cell_id, info in liveness.items()
    }


def get_live_in_variables(notebook: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Get variables that are live at the START of each cell.

    These are variables that the cell or subsequent cells will read.

    Args:
        notebook: Jupyter notebook content as a dictionary

    Returns:
        Dictionary mapping cell IDs to sorted lists of live variable names

    Example:
        >>> live_in = get_live_in_variables(notebook)
        >>> print(live_in["cell_5"])
        ['data', 'config', 'model']  # Cell 5 or later cells need these
    """
    liveness = analyze_notebook_liveness(notebook)
    return {
        cell_id: sorted(list(info.live_in))
        for cell_id, info in liveness.items()
    }


def get_dead_after_cell(notebook: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Get variables that become DEAD after each cell executes.

    A variable is dead after a cell if it was live before the cell but not after.
    This can happen when:
    - The cell is the last use of the variable
    - The cell redefines the variable (killing the old value)

    These variables can be safely garbage collected after the cell.

    Args:
        notebook: Jupyter notebook content as a dictionary

    Returns:
        Dictionary mapping cell IDs to sorted lists of newly-dead variable names

    Example:
        >>> dead_vars = get_dead_after_cell(notebook)
        >>> print(dead_vars["cell_5"])
        ['temp_data', 'intermediate_result']  # Can be freed after cell 5
    """
    liveness = analyze_notebook_liveness(notebook)
    return {
        cell_id: sorted(list(info.live_in - info.live_out))
        for cell_id, info in liveness.items()
    }


def get_never_used_variables(notebook: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Get variables that are written but never used in the notebook.

    Identifies dead code: cells that define variables that are never read by
    any subsequent cell (including transitively through functions).

    Args:
        notebook: Jupyter notebook content as a dictionary

    Returns:
        Dictionary mapping cell IDs to lists of variables defined but never used

    Example:
        >>> unused = get_never_used_variables(notebook)
        >>> print(unused["cell_3"])
        ['debug_temp', 'old_version']  # These are written but never read
    """
    dependencies = analyze_notebook(notebook)
    liveness = analyze_notebook_liveness(notebook, dependencies)

    result = {}
    for cell_id, deps in dependencies.items():
        if cell_id not in liveness:
            continue

        # Variables written by this cell but not live after it
        # (and not read by the cell itself after writing)
        written = deps.globals_written
        live_after = liveness[cell_id].live_out

        # A variable is never used if it's written but not live after this cell
        never_used = written - live_after

        if never_used:
            result[cell_id] = sorted(list(never_used))

    return result


if __name__ == '__main__':
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(
        description='Analyze variable liveness in a Jupyter notebook'
    )
    parser.add_argument(
        'notebook',
        help='Path to the Jupyter notebook file (.ipynb)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed information including gen/kill sets'
    )
    parser.add_argument(
        '--live-out-only',
        action='store_true',
        help='Show only variables live after each cell'
    )
    parser.add_argument(
        '--dead-only',
        action='store_true',
        help='Show only variables that become dead after each cell'
    )
    parser.add_argument(
        '--unused-only',
        action='store_true',
        help='Show only variables that are never used'
    )

    args = parser.parse_args()

    # Read the notebook file
    try:
        with open(args.notebook, 'r') as f:
            notebook = json.load(f)
    except FileNotFoundError:
        print(f"Error: Notebook file '{args.notebook}' not found", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: '{args.notebook}' is not a valid JSON file", file=sys.stderr)
        sys.exit(1)

    # Analyze the notebook
    liveness = analyze_notebook_liveness(notebook)

    if not liveness:
        print("No code cells found in notebook")
        sys.exit(0)

    # Handle special display modes
    if args.unused_only:
        unused = get_never_used_variables(notebook)
        if not unused:
            print("No unused variables found")
        else:
            print(f"\nUnused Variables in: {args.notebook}")
            print("=" * 80)
            for cell_id, vars_list in unused.items():
                print(f"\nCell {cell_id[:8]}...")
                print(f"  Never used: {', '.join(vars_list)}")
        sys.exit(0)

    if args.dead_only:
        dead = get_dead_after_cell(notebook)
        print(f"\nDead Variables After Each Cell: {args.notebook}")
        print("=" * 80)
        for cell_id in liveness.keys():
            dead_vars = dead.get(cell_id, [])
            if dead_vars:
                print(f"\nCell {cell_id[:8]}...")
                print(f"  Dead after: {', '.join(dead_vars)}")
        sys.exit(0)

    if args.live_out_only:
        live_out = get_live_out_variables(notebook)
        print(f"\nLive Variables After Each Cell: {args.notebook}")
        print("=" * 80)
        for cell_id, vars_list in live_out.items():
            print(f"\nCell {cell_id[:8]}...")
            if vars_list:
                print(f"  Live: {', '.join(vars_list)}")
            else:
                print(f"  Live: (none)")
        sys.exit(0)

    # Regular display
    print(f"\nLiveness Analysis for: {args.notebook}")
    print("=" * 80)

    cells = notebook.get('cells', [])
    for cell_id, info in liveness.items():
        # Get cell index for display
        cell_index = None
        for idx, cell in enumerate(cells):
            if cell.get('id') == cell_id:
                cell_index = idx
                break

        # Print cell header
        if cell_index is not None:
            print(f"\nCell [{cell_index}] (ID: {cell_id[:8]}...)")
        else:
            print(f"\nCell (ID: {cell_id[:8]}...)")
        print("-" * 80)

        # Show live_in
        if info.live_in:
            print(f"  Live in:  {', '.join(sorted(info.live_in))}")
        else:
            print("  Live in:  (none)")

        # Show live_out
        if info.live_out:
            print(f"  Live out: {', '.join(sorted(info.live_out))}")
        else:
            print("  Live out: (none)")

        # Show dead after
        dead_after = info.live_in - info.live_out
        if dead_after:
            print(f"  Dead:     {', '.join(sorted(dead_after))}")

        # Show gen/kill in verbose mode
        if args.verbose:
            if info.gen:
                print(f"  Gen:      {', '.join(sorted(info.gen))}")
            else:
                print("  Gen:      (none)")
            if info.kill:
                print(f"  Kill:     {', '.join(sorted(info.kill))}")
            else:
                print("  Kill:     (none)")

    # Print summary
    print("\n" + "=" * 80)
    print("Summary")
    print("-" * 80)

    all_live_out = set()
    for info in liveness.values():
        all_live_out.update(info.live_out)

    all_dead = set()
    for info in liveness.values():
        all_dead.update(info.live_in - info.live_out)

    print(f"Total cells analyzed: {len(liveness)}")
    print(f"Variables live at end: {len(all_live_out)}")
    if all_live_out:
        print(f"  {', '.join(sorted(all_live_out))}")
    else:
        print("  (none - all variables consumed)")

    print(f"Variables that become dead: {len(all_dead)}")
    if all_dead:
        print(f"  {', '.join(sorted(all_dead))}")

    # Identify never-used variables
    unused = get_never_used_variables(notebook)
    if unused:
        total_unused = sum(len(vars_list) for vars_list in unused.values())
        print(f"\n⚠ Variables defined but never used: {total_unused}")
        for cell_id, vars_list in unused.items():
            print(f"  Cell {cell_id[:8]}...: {', '.join(vars_list)}")

    print()
