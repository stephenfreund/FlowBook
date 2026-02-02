#!/usr/bin/env python3
"""Analyze type annotation coverage in Jupyter notebooks."""

import ast
import json
import sys
from pathlib import Path


def extract_code_cells(notebook_path: str) -> list[str]:
    """Extract source code from all code cells in a notebook."""
    with open(notebook_path) as f:
        nb = json.load(f)
    sources = []
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', '')
            if isinstance(source, list):
                source = ''.join(source)
            sources.append(source)
    return sources


def analyze_functions(tree: ast.Module) -> tuple[int, int]:
    """Return (funcs_with_any_annotation, total_funcs)."""
    funcs_annotated = 0
    funcs_total = 0

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs_total += 1
            has_annotation = False
            if node.returns is not None:
                has_annotation = True
            if not has_annotation:
                all_args = node.args.args + node.args.posonlyargs + node.args.kwonlyargs
                extras = []
                if node.args.vararg:
                    extras.append(node.args.vararg)
                if node.args.kwarg:
                    extras.append(node.args.kwarg)
                for i, arg in enumerate(all_args):
                    if i == 0 and arg.arg in ('self', 'cls'):
                        continue
                    if arg.annotation is not None:
                        has_annotation = True
                        break
                if not has_annotation:
                    for arg in extras:
                        if arg.annotation is not None:
                            has_annotation = True
                            break
            if has_annotation:
                funcs_annotated += 1

    return funcs_annotated, funcs_total


def analyze_global_variables(tree: ast.Module) -> tuple[int, int]:
    """Return (typed_globals, total_globals) for top-level assignments."""
    typed = 0
    total = 0

    for node in tree.body:
        if isinstance(node, ast.AnnAssign):
            # x: int = ... or x: int
            total += 1
            typed += 1
        elif isinstance(node, ast.Assign):
            # Count each target name
            for target in node.targets:
                if isinstance(target, ast.Name):
                    total += 1
                elif isinstance(target, ast.Tuple):
                    for elt in ast.walk(target):
                        if isinstance(elt, ast.Name):
                            total += 1
                # Skip attribute/subscript assignments

    return typed, total


def pct(num: int, den: int) -> str:
    if den == 0:
        return 'N/A'
    return f'{num}/{den} ({100 * num / den:.1f}%)'


def analyze_notebook(path: str) -> dict:
    cells = extract_code_cells(path)

    total_funcs_annotated = 0
    total_funcs = 0
    total_globals_typed = 0
    total_globals = 0

    for source in cells:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue

        fa, ft = analyze_functions(tree)
        total_funcs_annotated += fa
        total_funcs += ft

        gt, g = analyze_global_variables(tree)
        total_globals_typed += gt
        total_globals += g

    return {
        'funcs_annotated': total_funcs_annotated,
        'funcs_total': total_funcs,
        'globals_typed': total_globals_typed,
        'globals_total': total_globals,
    }


def main():
    if len(sys.argv) < 2:
        print(f'Usage: {sys.argv[0]} <notebook.ipynb> [notebook2.ipynb ...]',
              file=sys.stderr)
        sys.exit(1)

    agg = {
        'funcs_annotated': 0, 'funcs_total': 0,
        'globals_typed': 0, 'globals_total': 0,
    }

    # Expand arguments: .txt files contain lists of notebooks (relative to the txt file)
    notebook_paths = []
    for arg in sys.argv[1:]:
        if arg.endswith('.txt'):
            txt_path = Path(arg)
            if not txt_path.exists():
                print(f'Warning: {arg} not found, skipping', file=sys.stderr)
                continue
            txt_dir = txt_path.parent
            with open(txt_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        notebook_paths.append(str(txt_dir / line))
        else:
            notebook_paths.append(arg)

    results = []
    for path in notebook_paths:
        if not Path(path).exists():
            print(f'Warning: {path} not found, skipping', file=sys.stderr)
            continue
        stats = analyze_notebook(path)
        results.append((path, stats))
        for k in agg:
            agg[k] += stats[k]

    for path, stats in results:
        print(f'\n=== {path} ===')
        print(f'  Annotated functions:   {pct(stats["funcs_annotated"], stats["funcs_total"])}')
        print(f'  Typed global vars:     {pct(stats["globals_typed"], stats["globals_total"])}')

    if len(results) > 1:
        print(f'\n=== AGGREGATE ({len(results)} notebooks) ===')
        print(f'  Annotated functions:   {pct(agg["funcs_annotated"], agg["funcs_total"])}')
        print(f'  Typed global vars:     {pct(agg["globals_typed"], agg["globals_total"])}')


if __name__ == '__main__':
    main()
