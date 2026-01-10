"""
Notebook analysis encapsulation.

This module provides the NotebookAnalysis class, which encapsulates dependency
and liveness analysis for Jupyter notebooks. It provides a clean, type-safe API
for accessing cell-level information needed during optimization and validation.

Example Usage:
    from flowbook.util.notebook_analysis import NotebookAnalysis

    analysis = NotebookAnalysis(notebook)

    # Get variables to validate (only live outputs)
    validation_vars = analysis.get_validation_variables(cell_id)

    # Get dependencies for LLM context
    dependencies = analysis.get_dependency_variables(cell_id)

    # Filter environment to only dependencies
    filtered_env = analysis.filter_env_to_dependencies(cell_id, full_env)
"""

from typing import Dict, Set, Optional, Any
import nbformat

from flowbook.util.dependencies import (
    analyze_notebook,
    CellDependencies,
)
from flowbook.util.liveness import (
    analyze_notebook_liveness,
    CellLiveness,
)


class NotebookAnalysis:
    """
    Encapsulates dependency and liveness analysis for a notebook.

    This class runs static analysis once on notebook creation and provides
    efficient access to analysis results through a clean API. It eliminates
    the need to pass dependency/liveness dictionaries around and provides
    specialized methods for optimization use cases.

    Key Features:
    - Single analysis run, multiple accesses (cached results)
    - Type-safe API (no raw dict access needed)
    - Specialized methods for optimization (validation vars, dependency filtering)
    - Handles missing cells gracefully

    Attributes:
        notebook: The notebook being analyzed
        _dependencies: Cached dependency analysis results
        _liveness: Cached liveness analysis results
    """

    def __init__(self, notebook: nbformat.NotebookNode):
        """
        Initialize and run analysis on the notebook.

        Args:
            notebook: Jupyter notebook to analyze
        """
        self.notebook = notebook

        # Convert to dict format expected by analysis functions
        nb_dict = nbformat.from_dict(notebook) if hasattr(notebook, 'cells') else notebook

        # Run analyses once and cache results
        self._dependencies: Dict[str, CellDependencies] = analyze_notebook(nb_dict)
        self._liveness: Dict[str, CellLiveness] = analyze_notebook_liveness(
            nb_dict, self._dependencies
        )

    # Core analysis accessors

    def get_dependencies(self, cell_id: str) -> Optional[CellDependencies]:
        """
        Get complete dependency information for a cell.

        Args:
            cell_id: ID of the cell

        Returns:
            CellDependencies object or None if cell not found
        """
        return self._dependencies.get(cell_id)

    def get_liveness(self, cell_id: str) -> Optional[CellLiveness]:
        """
        Get complete liveness information for a cell.

        Args:
            cell_id: ID of the cell

        Returns:
            CellLiveness object or None if cell not found
        """
        return self._liveness.get(cell_id)

    # Convenience accessors for common operations

    def get_globals_written(self, cell_id: str) -> Set[str]:
        """
        Get all global variables written by a cell.

        Args:
            cell_id: ID of the cell

        Returns:
            Set of variable names written (empty if cell not found)
        """
        deps = self.get_dependencies(cell_id)
        return deps.globals_written.copy() if deps else set()

    def get_globals_read(self, cell_id: str) -> Set[str]:
        """
        Get all global variables read by a cell (includes transitive dependencies).

        Args:
            cell_id: ID of the cell

        Returns:
            Set of variable names read (empty if cell not found)
        """
        deps = self.get_dependencies(cell_id)
        return deps.globals_read.copy() if deps else set()

    def get_live_out_variables(self, cell_id: str) -> Set[str]:
        """
        Get variables that are live after this cell executes.

        These are variables that will be read by subsequent cells.

        Args:
            cell_id: ID of the cell

        Returns:
            Set of live variable names (empty if cell not found)
        """
        liveness = self.get_liveness(cell_id)
        return liveness.live_out.copy() if liveness else set()

    def get_live_in_variables(self, cell_id: str) -> Set[str]:
        """
        Get variables that are live before this cell executes.

        Args:
            cell_id: ID of the cell

        Returns:
            Set of live variable names (empty if cell not found)
        """
        liveness = self.get_liveness(cell_id)
        return liveness.live_in.copy() if liveness else set()

    # Specialized methods for optimization

    def get_validation_variables(self, cell_id: str) -> Set[str]:
        """
        Get variables that should be validated for correctness after optimization.

        This returns ALL variables that are live after this cell executes,
        ensuring that:
        1. Variables written by this cell produce correct values
        2. Variables from previous cells are not accidentally modified

        This is more precise than checking all namespace variables because:
        - Dead variables (never used again) don't need validation
        - Only variables that will actually be used later are checked

        Example:
            Cell 1: x = 1, temp = 2
            Cell 2: y = x + 1
            Cell 3: z = y * 2

            After Cell 2:
            - live_out = {y, x} (temp is dead, not used)
            - validation vars = {y, x}

            This catches both:
            - Incorrect calculation of y (output)
            - Accidental modification of x (should be unchanged)

        Args:
            cell_id: ID of the cell

        Returns:
            Set of variable names to validate (empty if cell not found)
        """
        # Return ALL live variables - anything that will be used later
        return self.get_live_out_variables(cell_id)

    def get_dependency_variables(self, cell_id: str) -> Set[str]:
        """
        Get variables this cell depends on (for LLM context filtering).

        Returns the set of global variables read by this cell, which should
        be included in the environment context when generating optimizations.

        Args:
            cell_id: ID of the cell

        Returns:
            Set of variable names this cell depends on
        """
        return self.get_globals_read(cell_id)

    def filter_env_to_dependencies(
        self,
        cell_id: str,
        env: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Filter an environment dictionary to only include this cell's dependencies.

        This is useful for reducing LLM context size by only including relevant
        variable type information.

        Args:
            cell_id: ID of the cell
            env: Full environment dictionary (e.g., from profile metadata)

        Returns:
            Filtered environment with only dependency variables
        """
        dependencies = self.get_dependency_variables(cell_id)

        # Filter env to only include dependencies
        return {k: v for k, v in env.items() if k in dependencies}

    # Utility methods

    def has_cell(self, cell_id: str) -> bool:
        """
        Check if a cell exists in the analysis.

        Args:
            cell_id: ID of the cell

        Returns:
            True if cell was analyzed, False otherwise
        """
        return cell_id in self._dependencies

    def get_all_cell_ids(self) -> Set[str]:
        """
        Get all cell IDs that were analyzed.

        Returns:
            Set of all cell IDs in the analysis
        """
        return set(self._dependencies.keys())

    def get_summary(self, cell_id: str) -> Dict[str, Any]:
        """
        Get a summary of analysis results for a cell.

        Useful for debugging and logging.

        Args:
            cell_id: ID of the cell

        Returns:
            Dictionary with summary information
        """
        deps = self.get_dependencies(cell_id)
        liveness = self.get_liveness(cell_id)

        if not deps:
            return {"error": "Cell not found"}

        return {
            "cell_id": cell_id,
            "reads": sorted(deps.globals_read),
            "writes": sorted(deps.globals_written),
            "live_out": sorted(liveness.live_out) if liveness else [],
            "validation_vars": sorted(self.get_validation_variables(cell_id)),
            "dependencies": sorted(self.get_dependency_variables(cell_id)),
        }

    # Backward compatibility methods (for gradual migration)

    def to_dependencies_dict(self) -> Dict[str, CellDependencies]:
        """
        Get the raw dependencies dictionary.

        This is provided for backward compatibility during migration.
        New code should use the typed accessor methods instead.

        Returns:
            Dictionary mapping cell IDs to CellDependencies objects
        """
        return self._dependencies.copy()

    def to_liveness_dict(self) -> Dict[str, CellLiveness]:
        """
        Get the raw liveness dictionary.

        This is provided for backward compatibility during migration.
        New code should use the typed accessor methods instead.

        Returns:
            Dictionary mapping cell IDs to CellLiveness objects
        """
        return self._liveness.copy()
