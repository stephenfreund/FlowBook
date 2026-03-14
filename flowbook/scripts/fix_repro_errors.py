#!/usr/bin/env python3
"""
Fix reproducibility errors in Jupyter notebooks.

This script modifies notebooks to fix reproducibility errors detected by FlowBook.
It creates/modifies a -fixed.ipynb copy, never touching the original.

Usage:
    python fix_repro_errors.py NOTEBOOK CELL_ID --fix-type TYPE [--variable VAR]

Fix types:
    inplace-reassign    - Deep-copy variable and alpha-rename downstream
    sequential-chain    - Same as inplace-reassign
    diagnostic-split    - Add %diagnostic magic to inspection code
    visualization-split - Same as diagnostic-split
    variable-reuse      - Alpha-rename reused variable downstream
    model-copy          - Copy ML model before mutation (fit/predict)
    inplace-to-copy     - Convert df.method(inplace=True) to df = df.method()
    struct-copy         - Insert df.copy() before structural assignment

Examples:
    python fix_repro_errors.py nb.ipynb abcd --fix-type inplace-reassign --variable train
    python fix_repro_errors.py nb.ipynb efgh --fix-type diagnostic-split
    python fix_repro_errors.py nb.ipynb ijkl --fix-type model-copy --variable model
    python fix_repro_errors.py nb.ipynb mnop --fix-type inplace-to-copy --variable df
"""

import argparse
import ast
import copy
import json
import re
import sys
from pathlib import Path
from typing import Optional


# Comment marker for FlowBook fixes
FLOWBOOK_FIX_MARKER = "# [FLOWBOOK FIX]"


def load_notebook(path: Path) -> dict:
    """Load a Jupyter notebook from disk."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_notebook(notebook: dict, path: Path) -> None:
    """Save a Jupyter notebook to disk."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
        f.write("\n")


def get_fixed_path(original_path: Path) -> Path:
    """Get the path for the -fixed.ipynb version.

    If the input is already a -fixed.ipynb, return it as-is (modify in place).
    Otherwise, append -fixed to the stem.
    """
    stem = original_path.stem
    # If already a -fixed notebook, return as-is to avoid -fixed-fixed
    if stem.endswith("-fixed"):
        return original_path
    return original_path.parent / f"{stem}-fixed.ipynb"


def initialize_fixed_notebook(original_path: Path, force: bool = False) -> Path:
    """
    Create a fresh -fixed.ipynb copy from the original notebook.

    This should be called once at the start of a batch of fixes.
    Subsequent fix operations will modify this copy.

    Args:
        original_path: Path to the original notebook
        force: If True, overwrite existing -fixed.ipynb

    Returns:
        Path to the -fixed.ipynb copy
    """
    fixed_path = get_fixed_path(original_path)

    if fixed_path.exists() and not force:
        print(f"Note: {fixed_path.name} already exists, will modify it")
        return fixed_path

    # Load original and save as -fixed
    notebook = load_notebook(original_path)
    save_notebook(notebook, fixed_path)
    print(f"Created: {fixed_path.name}")
    return fixed_path


def get_code_cell_by_id(notebook: dict, cell_id: str) -> tuple[int, dict] | None:
    """Find a code cell by its ID or code cell index, returning (notebook_index, cell) or None.

    Args:
        notebook: The notebook dict
        cell_id: Either a cell ID string, or a code cell index prefixed with '@' (e.g., '@18')

    Returns:
        Tuple of (notebook_index, cell) or None if not found
    """
    # Check if cell_id is a code cell index (e.g., "@18")
    if cell_id.startswith("@"):
        try:
            target_code_idx = int(cell_id[1:])
            code_idx = 0
            for i, cell in enumerate(notebook.get("cells", [])):
                if cell.get("cell_type") == "code":
                    if code_idx == target_code_idx:
                        return i, cell
                    code_idx += 1
        except ValueError:
            pass
        return None

    # Otherwise, search by cell ID
    for i, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") == "code":
            # Check both 'id' field and metadata
            cid = cell.get("id") or cell.get("metadata", {}).get("id")
            if cid == cell_id:
                return i, cell
    return None


def get_code_cells_from_index(notebook: dict, start_idx: int) -> list[tuple[int, dict]]:
    """Get all code cells from start_idx onwards."""
    result = []
    for i, cell in enumerate(notebook.get("cells", [])):
        if i >= start_idx and cell.get("cell_type") == "code":
            result.append((i, cell))
    return result


def get_cell_source(cell: dict) -> str:
    """Get cell source as a string."""
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(source)
    return source


def set_cell_source(cell: dict, source: str) -> None:
    """Set cell source (as list of lines for nbformat compatibility)."""
    # Split into lines, keeping line endings
    lines = source.splitlines(keepends=True)
    # Ensure last line doesn't have trailing newline if original didn't
    if lines and not source.endswith("\n"):
        lines[-1] = lines[-1].rstrip("\n")
    cell["source"] = lines if lines else [""]


def get_cell_id(cell: dict) -> str:
    """Get the cell ID."""
    return cell.get("id") or cell.get("metadata", {}).get("id", "unknown")


def split_cell_magic(source: str) -> tuple[str, str]:
    """
    Split cell source into cell magic prefix and remaining code.

    Cell magics (%%time, %%timeit, %%capture, etc.) MUST be the first line(s)
    of a cell. This function separates them so we can insert code after them.

    Returns:
        Tuple of (magic_prefix, remaining_code) where magic_prefix includes
        any cell magic lines and remaining_code is everything after.
    """
    lines = source.splitlines(keepends=True)
    if not lines:
        return "", source

    magic_lines = []
    rest_start = 0

    for i, line in enumerate(lines):
        stripped = line.lstrip()
        # Cell magics start with %% (must be at start of cell)
        if stripped.startswith("%%"):
            magic_lines.append(line)
            rest_start = i + 1
            # Cell magics can span multiple lines if they have arguments
            # but typically it's just one line, so we stop after first %%
            break
        # Skip blank lines and comments at the start (before magic)
        elif stripped == "" or stripped.startswith("#"):
            # Only skip if we haven't found a magic yet and might still find one
            if i == len(magic_lines):
                magic_lines.append(line)
                rest_start = i + 1
            else:
                break
        else:
            # Non-magic, non-blank line - stop looking
            break

    # If we collected leading blanks/comments but no magic, reset
    if magic_lines and not any(
        ln.lstrip().startswith("%%") for ln in magic_lines
    ):
        return "", source

    magic_prefix = "".join(magic_lines)
    remaining = "".join(lines[rest_start:])
    return magic_prefix, remaining


def prepend_to_cell_source(source: str, prefix: str) -> str:
    """
    Prepend code/comments to cell source, preserving cell magics.

    Cell magics (%%time, etc.) must remain at the very top of the cell.
    This function inserts the prefix after any cell magics.

    Args:
        source: Original cell source
        prefix: Code/comments to prepend

    Returns:
        New source with prefix inserted after any cell magics
    """
    magic_prefix, remaining = split_cell_magic(source)
    if magic_prefix:
        return magic_prefix + prefix + remaining
    return prefix + source


class VariableRenamer(ast.NodeTransformer):
    """AST transformer that renames a variable throughout the code."""

    def __init__(self, old_name: str, new_name: str):
        self.old_name = old_name
        self.new_name = new_name
        self.renamed = False

    def visit_Name(self, node):
        if node.id == self.old_name:
            node.id = self.new_name
            self.renamed = True
        return node

    def visit_arg(self, node):
        # Don't rename function arguments
        return node


def rename_variable_in_code(code: str, old_name: str, new_name: str) -> tuple[str, bool]:
    """
    Rename all occurrences of a variable in code using AST.

    Returns (new_code, was_renamed).
    Falls back to regex if AST parsing fails.
    """
    try:
        tree = ast.parse(code)
        renamer = VariableRenamer(old_name, new_name)
        new_tree = renamer.visit(tree)
        if renamer.renamed:
            return ast.unparse(new_tree), True
        return code, False
    except SyntaxError:
        # Fall back to regex for code that doesn't parse
        # Match word boundaries
        pattern = rf"\b{re.escape(old_name)}\b"
        new_code, count = re.subn(pattern, new_name, code)
        return new_code, count > 0


def find_actual_variable_name(source: str, base_variable: str) -> str:
    """
    Find the actual variable name in the source code.

    If a previous fix renamed the variable (e.g., df -> df_flow_1234),
    this function finds that renamed version. PRIORITIZES the _flow_ version
    over the original, since if both exist, we want to chain from the renamed one.

    Args:
        source: The cell source code
        base_variable: The original variable name (e.g., "df")

    Returns:
        The actual variable name found (either base_variable or a _flow_ variant)
    """
    # FIRST check for previously renamed versions: {base}_flow_XXXX
    # We prioritize these because if a previous fix renamed the variable,
    # we want to chain from the renamed version, not the original
    # Pattern matches both old @N style and new clean N style
    flow_pattern = rf"\b({re.escape(base_variable)}_flow_\w+)\b"
    matches = re.findall(flow_pattern, source)
    if matches:
        # Return the first renamed version found
        return matches[0]

    # Then check if the base variable exists
    pattern = rf"\b{re.escape(base_variable)}\b"
    if re.search(pattern, source):
        return base_variable

    # Variable not found at all - return base and let caller handle
    return base_variable


def add_deepcopy_and_rename(
    notebook: dict,
    cell_idx: int,
    variable: str,
    new_suffix: str,
) -> None:
    """
    Add a deep copy of variable and rename all downstream uses.

    This fix pattern works for:
    - In-place variable reassignment
    - Sequential transformation chains

    If the variable was already renamed by a previous fix (e.g., df_flow_1234),
    this function will find that renamed version, deep copy it to a new name
    (e.g., df_flow_5678), and rename downstream uses.

    Args:
        notebook: The notebook dict
        cell_idx: Index of the cell to start fixing
        variable: The base variable name to copy and rename
        new_suffix: Suffix to add to create new name (e.g., "abcd" -> "train_flow_abcd")
    """
    # Get the target cell
    cells = notebook.get("cells", [])
    if cell_idx >= len(cells):
        return

    target_cell = cells[cell_idx]
    source = get_cell_source(target_cell)

    # Find the actual variable name (might be renamed from previous fix)
    actual_var = find_actual_variable_name(source, variable)

    # Determine the base name for the new variable
    # If actual_var is already a _flow_ variant, extract the original base
    if "_flow_" in actual_var:
        base_name = actual_var.split("_flow_")[0]
    else:
        base_name = variable

    new_name = f"{base_name}_flow_{new_suffix}"

    # Add the deep copy at the start of the cell
    copy_line = f"import copy\n{new_name} = copy.deepcopy({actual_var})  {FLOWBOOK_FIX_MARKER} Deep copy to avoid in-place mutation\n"

    # Add comment explaining the fix
    if actual_var != variable:
        fix_comment = f"""{FLOWBOOK_FIX_MARKER} Original error: In-place reassignment of '{variable}' (now '{actual_var}')
{FLOWBOOK_FIX_MARKER} Fix: Created deep copy '{new_name}' from '{actual_var}' and renamed downstream uses
"""
    else:
        fix_comment = f"""{FLOWBOOK_FIX_MARKER} Original error: In-place reassignment of '{variable}'
{FLOWBOOK_FIX_MARKER} Fix: Created deep copy '{new_name}' and renamed downstream uses
"""

    # Rename variable in this cell
    renamed_source, _ = rename_variable_in_code(source, actual_var, new_name)

    # Set new source with copy and comment, preserving cell magics (%%time, etc.)
    new_source = prepend_to_cell_source(renamed_source, fix_comment + copy_line)
    set_cell_source(target_cell, new_source)

    # Rename in all downstream cells
    for i in range(cell_idx + 1, len(cells)):
        cell = cells[i]
        if cell.get("cell_type") == "code":
            cell_source = get_cell_source(cell)
            new_cell_source, renamed = rename_variable_in_code(
                cell_source, actual_var, new_name
            )
            if renamed:
                set_cell_source(cell, new_cell_source)


def split_diagnostic_cell(
    notebook: dict,
    cell_idx: int,
    diagnostic_lines: Optional[list[int]] = None,
) -> None:
    """
    Add %diagnostic magic to a cell that does inspection/visualization.

    For simple cases, just adds %diagnostic to the entire cell.
    For complex cases where the cell mixes inspection and mutation,
    this would need to split the cell (not implemented here - that requires
    more sophisticated analysis).

    Args:
        notebook: The notebook dict
        cell_idx: Index of the cell to fix
        diagnostic_lines: Optional list of line indices that are diagnostic
    """
    cells = notebook.get("cells", [])
    if cell_idx >= len(cells):
        return

    target_cell = cells[cell_idx]
    source = get_cell_source(target_cell)

    # Add %diagnostic magic and comment, preserving cell magics (%%time, etc.)
    fix_comment = f"""{FLOWBOOK_FIX_MARKER} Original error: Diagnostic inspection before mutation
{FLOWBOOK_FIX_MARKER} Fix: Added %diagnostic to skip reproducibility tracking for this inspection cell
"""

    new_source = prepend_to_cell_source(source, fix_comment + "%diagnostic\n")
    set_cell_source(target_cell, new_source)


def alpha_rename_reused_variable(
    notebook: dict,
    cell_idx: int,
    variable: str,
    new_suffix: str,
) -> None:
    """
    Alpha-rename a variable that is reused for different purposes.

    This creates a new variable name from the point of reuse onwards.
    If the variable was already renamed by a previous fix (e.g., model_flow_1234),
    this function will find that renamed version and create a new name.

    Args:
        notebook: The notebook dict
        cell_idx: Index of the cell where reuse starts
        variable: The base variable name being reused
        new_suffix: Suffix for new name
    """
    cells = notebook.get("cells", [])
    if cell_idx >= len(cells):
        return

    target_cell = cells[cell_idx]
    source = get_cell_source(target_cell)

    # Find the actual variable name (might be renamed from previous fix)
    actual_var = find_actual_variable_name(source, variable)

    # Determine the base name for the new variable
    if "_flow_" in actual_var:
        base_name = actual_var.split("_flow_")[0]
    else:
        base_name = variable

    new_name = f"{base_name}_{new_suffix}"

    # Add comment explaining the fix
    if actual_var != variable:
        fix_comment = f"""{FLOWBOOK_FIX_MARKER} Original error: Variable '{variable}' (now '{actual_var}') reused for different purpose
{FLOWBOOK_FIX_MARKER} Fix: Renamed to '{new_name}' to distinguish from earlier use
"""
    else:
        fix_comment = f"""{FLOWBOOK_FIX_MARKER} Original error: Variable '{variable}' reused for different purpose
{FLOWBOOK_FIX_MARKER} Fix: Renamed to '{new_name}' to distinguish from earlier use
"""

    # Rename in this cell and all downstream, preserving cell magics (%%time, etc.)
    renamed_source, _ = rename_variable_in_code(source, actual_var, new_name)
    new_source = prepend_to_cell_source(renamed_source, fix_comment)
    set_cell_source(target_cell, new_source)

    # Rename in all downstream cells
    for i in range(cell_idx + 1, len(cells)):
        cell = cells[i]
        if cell.get("cell_type") == "code":
            cell_source = get_cell_source(cell)
            new_cell_source, renamed = rename_variable_in_code(
                cell_source, actual_var, new_name
            )
            if renamed:
                set_cell_source(cell, new_cell_source)


def add_model_copy_and_rename(
    notebook: dict,
    cell_idx: int,
    variable: str,
    new_suffix: str,
) -> None:
    """
    Copy an ML model before mutation and rename downstream uses.

    This fix handles unrecoverable mutations from ML model operations:
    - model.fit(X, y)
    - scaler.fit_transform(X)
    - model.predict(X) when it modifies internal state

    Uses safe_model_copy() which:
    - Uses sklearn.clone() for unfitted sklearn estimators (fast)
    - Uses copy.deepcopy() for fitted models (correct for all frameworks)
    - Handles PyTorch, XGBoost, LightGBM, CatBoost correctly

    Args:
        notebook: The notebook dict
        cell_idx: Index of the cell with the mutation
        variable: The model variable name
        new_suffix: Suffix to add to create new name
    """
    cells = notebook.get("cells", [])
    if cell_idx >= len(cells):
        return

    target_cell = cells[cell_idx]
    source = get_cell_source(target_cell)

    # Find the actual variable name (might be renamed from previous fix)
    actual_var = find_actual_variable_name(source, variable)

    # Determine the base name for the new variable
    if "_flow_" in actual_var:
        base_name = actual_var.split("_flow_")[0]
    else:
        base_name = variable

    new_name = f"{base_name}_flow_{new_suffix}"

    # Use safe_model_copy for correct handling of all ML frameworks
    copy_code = f"""from flowbook.util.model_copy import safe_model_copy
{new_name} = safe_model_copy({actual_var})  {FLOWBOOK_FIX_MARKER} Copy model before mutation
"""

    # Add comment explaining the fix
    if actual_var != variable:
        fix_comment = f"""{FLOWBOOK_FIX_MARKER} Original error: Unrecoverable mutation of '{variable}' (now '{actual_var}')
{FLOWBOOK_FIX_MARKER} Fix: Created model copy '{new_name}' and renamed downstream uses
"""
    else:
        fix_comment = f"""{FLOWBOOK_FIX_MARKER} Original error: Unrecoverable mutation of '{variable}' (e.g., .fit() modifies internal state)
{FLOWBOOK_FIX_MARKER} Fix: Created model copy '{new_name}' and renamed downstream uses
"""

    # Rename variable in this cell
    renamed_source, _ = rename_variable_in_code(source, actual_var, new_name)

    # Set new source with copy and comment
    new_source = prepend_to_cell_source(renamed_source, fix_comment + copy_code)
    set_cell_source(target_cell, new_source)

    # Rename in all downstream cells
    for i in range(cell_idx + 1, len(cells)):
        cell = cells[i]
        if cell.get("cell_type") == "code":
            cell_source = get_cell_source(cell)
            new_cell_source, renamed = rename_variable_in_code(
                cell_source, actual_var, new_name
            )
            if renamed:
                set_cell_source(cell, new_cell_source)


class InplaceRemover(ast.NodeTransformer):
    """AST transformer that removes inplace=True from method calls on a variable."""

    def __init__(self, target_var: str):
        self.target_var = target_var
        self.modified = False
        self.method_calls_fixed = []

    def visit_Expr(self, node):
        """Convert df.method(inplace=True) to df = df.method()."""
        if isinstance(node.value, ast.Call):
            call = node.value
            # Check if it's a method call on our target variable
            if isinstance(call.func, ast.Attribute):
                # Get the object being called on
                if isinstance(call.func.value, ast.Name):
                    if call.func.value.id == self.target_var:
                        # Check for inplace=True in keyword arguments
                        new_keywords = []
                        had_inplace = False
                        for kw in call.keywords:
                            if kw.arg == "inplace":
                                # Check if it's True
                                if isinstance(kw.value, ast.Constant) and kw.value.value is True:
                                    had_inplace = True
                                    continue  # Skip this keyword
                            new_keywords.append(kw)

                        if had_inplace:
                            # Remove inplace from keywords
                            call.keywords = new_keywords
                            # Convert to assignment: df = df.method(...)
                            # Create proper AST nodes with all required attributes
                            target = ast.Name(id=self.target_var, ctx=ast.Store())
                            assign = ast.Assign(
                                targets=[target],
                                value=call,
                            )
                            # Copy location info from original node
                            ast.copy_location(assign, node)
                            ast.copy_location(target, node)
                            # Fix missing attributes for ast.unparse
                            ast.fix_missing_locations(assign)
                            self.modified = True
                            self.method_calls_fixed.append(call.func.attr)
                            return assign

        return node


def convert_inplace_to_assignment(
    notebook: dict,
    cell_idx: int,
    variable: str,
) -> None:
    """
    Convert df.method(inplace=True) to df = df.method().

    This fix handles unrecoverable mutations from pandas inplace operations:
    - df.drop(columns=['x'], inplace=True) -> df = df.drop(columns=['x'])
    - df.fillna(0, inplace=True) -> df = df.fillna(0)
    - df.reset_index(inplace=True) -> df = df.reset_index()

    Args:
        notebook: The notebook dict
        cell_idx: Index of the cell with the inplace operation
        variable: The DataFrame variable name
    """
    cells = notebook.get("cells", [])
    if cell_idx >= len(cells):
        return

    target_cell = cells[cell_idx]
    source = get_cell_source(target_cell)

    # Find the actual variable name (might be renamed from previous fix)
    actual_var = find_actual_variable_name(source, variable)

    # Try AST transformation
    try:
        tree = ast.parse(source)
        remover = InplaceRemover(actual_var)
        new_tree = remover.visit(tree)

        if remover.modified:
            methods_fixed = ", ".join(remover.method_calls_fixed)
            fix_comment = f"""{FLOWBOOK_FIX_MARKER} Original error: Unrecoverable mutation via inplace=True
{FLOWBOOK_FIX_MARKER} Fix: Converted {actual_var}.{methods_fixed}(inplace=True) to assignment
"""
            new_source = prepend_to_cell_source(ast.unparse(new_tree), fix_comment)
            set_cell_source(target_cell, new_source)
            return
    except SyntaxError:
        pass

    # Fallback: regex-based replacement
    # Pattern: variable.method(..., inplace=True, ...) or variable.method(..., inplace = True, ...)
    pattern = rf"(\b{re.escape(actual_var)}\.(\w+)\([^)]*),\s*inplace\s*=\s*True([^)]*)\)"
    replacement = rf"{actual_var} = \1\3)"

    new_source, count = re.subn(pattern, replacement, source)
    if count > 0:
        fix_comment = f"""{FLOWBOOK_FIX_MARKER} Original error: Unrecoverable mutation via inplace=True
{FLOWBOOK_FIX_MARKER} Fix: Converted inplace=True to assignment
"""
        new_source = prepend_to_cell_source(new_source, fix_comment)
        set_cell_source(target_cell, new_source)


def add_copy_before_structural_assign(
    notebook: dict,
    cell_idx: int,
    variable: str,
    new_suffix: str,
) -> None:
    """
    Insert df.copy() before structural attribute assignment.

    This fix handles unrecoverable mutations from structural assignments:
    - df.columns = ['a', 'b', 'c']
    - df.index = new_index

    Args:
        notebook: The notebook dict
        cell_idx: Index of the cell with structural assignment
        variable: The DataFrame variable name
        new_suffix: Suffix for the new variable name
    """
    cells = notebook.get("cells", [])
    if cell_idx >= len(cells):
        return

    target_cell = cells[cell_idx]
    source = get_cell_source(target_cell)

    # Find the actual variable name (might be renamed from previous fix)
    actual_var = find_actual_variable_name(source, variable)

    # Determine the base name for the new variable
    if "_flow_" in actual_var:
        base_name = actual_var.split("_flow_")[0]
    else:
        base_name = variable

    new_name = f"{base_name}_flow_{new_suffix}"

    # Create copy before structural assignment
    copy_code = f"{new_name} = {actual_var}.copy()  {FLOWBOOK_FIX_MARKER} Copy before structural mutation\n"

    fix_comment = f"""{FLOWBOOK_FIX_MARKER} Original error: Unrecoverable structural mutation of '{actual_var}'
{FLOWBOOK_FIX_MARKER} Fix: Created copy '{new_name}' and renamed downstream uses
"""

    # Rename variable in this cell
    renamed_source, _ = rename_variable_in_code(source, actual_var, new_name)

    # Set new source with copy and comment
    new_source = prepend_to_cell_source(renamed_source, fix_comment + copy_code)
    set_cell_source(target_cell, new_source)

    # Rename in all downstream cells
    for i in range(cell_idx + 1, len(cells)):
        cell = cells[i]
        if cell.get("cell_type") == "code":
            cell_source = get_cell_source(cell)
            new_cell_source, renamed = rename_variable_in_code(
                cell_source, actual_var, new_name
            )
            if renamed:
                set_cell_source(cell, new_cell_source)


def apply_fix(
    notebook_path: Path,
    cell_id: str,
    fix_type: str,
    variable: Optional[str] = None,
) -> Path:
    """
    Apply a reproducibility fix to a notebook.

    Args:
        notebook_path: Path to the original notebook
        cell_id: ID of the cell to fix
        fix_type: Type of fix to apply
        variable: Variable name (required for some fix types)

    Returns:
        Path to the fixed notebook
    """
    # Load or create the fixed notebook
    fixed_path = get_fixed_path(notebook_path)
    if fixed_path.exists():
        notebook = load_notebook(fixed_path)
    else:
        notebook = load_notebook(notebook_path)

    # Find the cell
    result = get_code_cell_by_id(notebook, cell_id)
    if result is None:
        raise ValueError(f"Cell '{cell_id}' not found in notebook")

    cell_idx, cell = result

    # Get suffix from cell_id (last 4 chars, strip @ for index notation)
    clean_id = cell_id.lstrip("@")  # Remove @ prefix if using index notation
    suffix = clean_id[-4:] if len(clean_id) >= 4 else clean_id

    # Apply the appropriate fix
    if fix_type in ("inplace-reassign", "sequential-chain"):
        if not variable:
            raise ValueError(f"--variable required for {fix_type}")
        add_deepcopy_and_rename(notebook, cell_idx, variable, suffix)

    elif fix_type in ("diagnostic-split", "visualization-split"):
        split_diagnostic_cell(notebook, cell_idx)

    elif fix_type == "variable-reuse":
        if not variable:
            raise ValueError(f"--variable required for {fix_type}")
        alpha_rename_reused_variable(notebook, cell_idx, variable, suffix)

    elif fix_type == "model-copy":
        if not variable:
            raise ValueError(f"--variable required for {fix_type}")
        add_model_copy_and_rename(notebook, cell_idx, variable, suffix)

    elif fix_type == "inplace-to-copy":
        if not variable:
            raise ValueError(f"--variable required for {fix_type}")
        convert_inplace_to_assignment(notebook, cell_idx, variable)

    elif fix_type == "struct-copy":
        if not variable:
            raise ValueError(f"--variable required for {fix_type}")
        add_copy_before_structural_assign(notebook, cell_idx, variable, suffix)

    else:
        raise ValueError(f"Unknown fix type: {fix_type}")

    # Save the fixed notebook
    save_notebook(notebook, fixed_path)
    return fixed_path


def main():
    parser = argparse.ArgumentParser(
        description="Fix reproducibility errors in Jupyter notebooks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("notebook", type=Path, help="Path to the notebook")
    parser.add_argument("cell_id", nargs="?", help="ID of the cell to fix")
    parser.add_argument(
        "--fix-type",
        choices=[
            "inplace-reassign",
            "sequential-chain",
            "diagnostic-split",
            "visualization-split",
            "variable-reuse",
            "model-copy",
            "inplace-to-copy",
            "struct-copy",
        ],
        help="Type of fix to apply",
    )
    parser.add_argument(
        "--variable",
        help="Variable name (required for most fix types except diagnostic-split)",
    )
    parser.add_argument(
        "--init",
        action="store_true",
        help="Initialize a fresh -fixed.ipynb copy from the original (call once before applying fixes)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="With --init, overwrite existing -fixed.ipynb",
    )

    args = parser.parse_args()

    if not args.notebook.exists():
        print(f"Error: Notebook not found: {args.notebook}", file=sys.stderr)
        sys.exit(1)

    # Handle --init mode
    if args.init:
        try:
            fixed_path = initialize_fixed_notebook(args.notebook, force=args.force)
            print(f"Initialized: {fixed_path}")
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        return

    # Handle fix mode
    if not args.cell_id:
        print("Error: cell_id is required when not using --init", file=sys.stderr)
        sys.exit(1)

    if not args.fix_type:
        print("Error: --fix-type is required when not using --init", file=sys.stderr)
        sys.exit(1)

    try:
        fixed_path = apply_fix(
            args.notebook,
            args.cell_id,
            args.fix_type,
            args.variable,
        )
        print(f"Fixed notebook saved to: {fixed_path}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
