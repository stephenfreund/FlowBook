import difflib
from typing import List
from nbformat import NotebookNode


def notebook_diff(nb_a: NotebookNode, nb_b: NotebookNode) -> str:
    """
    Compute a diff between two NotebookNode instances.

    - Cells are aligned by their .id.
    - Cells only in nb_a are reported as removed.
    - Cells only in nb_b are reported as added.
    - Cells in both get a unified diff of their .source lists.

    Returns a single string containing the diff.
    """
    # Build id → source-lines maps
    cells_a = {cell.id: cell.source.splitlines(keepends=False) for cell in nb_a.cells}
    cells_b = {cell.id: cell.source.splitlines(keepends=False) for cell in nb_b.cells}

    # Preserve original ordering
    order_a = [cell.id for cell in nb_a.cells]
    order_b = [cell.id for cell in nb_b.cells]

    diffs: List[str] = []

    # 1) Removed or modified cells (from A)
    for cid in order_a:
        sa = cells_a[cid]
        if cid not in cells_b:
            # removed cell
            diffs.append(f"--- cell {cid}")
            for line in sa:
                diffs.append(f"- {line}")
            diffs.append("")
        else:
            sb = cells_b[cid]
            if sa != sb:
                # unified diff for changed cell
                hunk = difflib.unified_diff(
                    sa, sb, fromfile=f"cell {cid}", tofile=f"cell {cid}", lineterm=""
                )
                diffs.extend(hunk)
                diffs.append("")

    # 2) Added cells (in B only)
    for cid in order_b:
        if cid not in cells_a:
            sb = cells_b[cid]
            diffs.append(f"+++ cell {cid}")
            for line in sb:
                diffs.append(f"+ {line}")
            diffs.append("")

    return "\n".join(diffs)


# a helper to print it with ANSI colors
def print_colored_diff(diff_text: str, prefix="") -> None:
    print(colored_diff_string(diff_text, prefix))


def colored_diff_string(diff_text: str, prefix="") -> str:
    RED = "\033[31m"
    GREEN = "\033[32m"
    CYAN = "\033[36m"
    RESET = "\033[0m"

    lines = []
    for line in diff_text.splitlines():
        if line.startswith("+") and not line.startswith("+++ "):
            lines.append(f"{prefix}{GREEN}{line}{RESET}")
        elif line.startswith("-") and not line.startswith("--- "):
            lines.append(f"{prefix}{RED}{line}{RESET}")
        elif line.startswith("@@"):
            lines.append(f"{prefix}{CYAN}{line}{RESET}")
        else:
            lines.append(f"{prefix}{line}")
    return "\n".join(lines)


def main():
    import sys
    import nbformat

    nb_a = nbformat.read(sys.argv[1], as_version=4)
    nb_b = nbformat.read(sys.argv[2], as_version=4)
    print_colored_diff(notebook_diff(nb_a, nb_b))


if __name__ == "__main__":
    main()
