#!/usr/bin/env python3
"""
Generate LaTeX document from litmus tests.

This script runs all litmus tests and compiles their LaTeX outputs
into a single document for inclusion in formal documentation.
"""

import os
import sys
from pathlib import Path

import yaml

# Add the project root to path so we can import flowbook modules
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from flowbook.kernel.tests.test_litmus import LitmusTestRunner, load_litmus_tests


OUTPUT_DIR = Path(__file__).parent / "litmus_output"


def generate_all_latex():
    """Generate LaTeX for all litmus tests."""
    tests = load_litmus_tests()
    OUTPUT_DIR.mkdir(exist_ok=True)

    all_latex = []

    # Document header
    all_latex.append(r"""\documentclass{article}
\usepackage{multirow}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{geometry}
\geometry{margin=1in}

\title{FlowBook Reproducibility Litmus Tests}
\author{Generated from LITMUS\_TESTS.yaml}
\date{\today}

\begin{document}
\maketitle

\section{Introduction}
This document contains the litmus tests for FlowBook's reproducibility enforcement system.
Each test verifies a specific behavior of the state machine that tracks cell staleness and
detects reproducibility violations.

\tableofcontents
\newpage
""")

    # Group tests by operation type
    groups = {
        "RUN": [],
        "EDIT": [],
        "DELETE": [],
        "INSERT": [],
        "MOVE": [],
    }

    for test in tests:
        name = test.get("name", "unnamed")
        ops = test.get("operations", [])
        # Determine primary operation type
        for op in ops:
            op_type = op.get("type", "").upper()
            if op_type in groups:
                groups[op_type].append(test)
                break
        else:
            # Default to RUN if no clear type
            groups["RUN"].append(test)

    # Generate sections for each operation type
    for op_type, op_tests in groups.items():
        if not op_tests:
            continue

        all_latex.append(f"\n\\section{{{op_type} Operations}}\n")

        for test in op_tests:
            name = test.get("name", "unnamed")
            description = test.get("description", "")
            cell_order = test.get("cell_order", [])
            operations = test.get("operations", [])

            print(f"Generating: {name}")

            # Run the test to capture state evolution
            try:
                runner = LitmusTestRunner(cell_order)
                for op in operations:
                    runner.execute_operation(op)

                # Generate LaTeX
                latex = runner.render_latex(name, description)

                # Write individual file
                with open(OUTPUT_DIR / f"{name}.tex", "w") as f:
                    f.write(latex)

                # Add to combined document
                all_latex.append(f"\\subsection{{{name.replace('_', ' ').title()}}}")
                all_latex.append(f"\\label{{test:{name}}}")
                all_latex.append(latex)
                all_latex.append("")

            except Exception as e:
                print(f"  ERROR: {e}")
                all_latex.append(f"\\subsection{{{name.replace('_', ' ').title()}}}")
                all_latex.append(f"Test failed to generate: {e}")
                all_latex.append("")

    # Document footer
    all_latex.append(r"\end{document}")

    # Write combined document
    combined_path = OUTPUT_DIR / "all_litmus_tests.tex"
    with open(combined_path, "w") as f:
        f.write("\n".join(all_latex))

    print(f"\nGenerated combined document: {combined_path}")
    print(f"Individual tests written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    generate_all_latex()
