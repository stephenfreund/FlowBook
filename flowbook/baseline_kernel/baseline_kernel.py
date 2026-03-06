"""Baseline kernel for fair comparison with FlowBook.

This kernel provides the same memory measurement capabilities as FlowBook
but without any reproducibility tracking, checkpointing, or other FlowBook features.
It's used for fair A/B comparison of memory overhead.

Features:
- Standard IPython kernel behavior
- %memory magic command (same as FlowBook)
- Memory reporting in same format as FlowBook for comparison
- Kernel-side timing reported via metadata (for fair timing comparison)

Does NOT include:
- Reproducibility tracking
- Checkpointing
- Variable access tracking
- Any FlowBook-specific features
"""

import time
import types
from typing import Any, Dict, Optional

from ipykernel.ipkernel import IPythonKernel
from IPython.core.magic import line_magic, magics_class, Magics


@magics_class
class MemoryMagics(Magics):
    """Memory magic commands for baseline comparison using HeapSizer."""

    @line_magic
    def memory(self, line: str) -> None:
        """Show memory usage using HeapSizer.

        Usage:
            %memory         - Show namespace memory summary
            %memory vars    - Show per-variable breakdown
            %memory vars 10 - Show top 10 variables
        """
        from flowbook.kernel_support.heap_size import HeapSizer

        args = line.strip().lower().split()
        if not args:
            args = [""]

        cmd = args[0]

        if cmd == "" or cmd == "?" or cmd == "status":
            # Show namespace summary
            sizer = HeapSizer()
            user_ns = self.shell.user_ns
            # Filter to user variables
            user_vars = {
                k: v for k, v in user_ns.items()
                if not k.startswith('_') and not isinstance(v, (type, types.FunctionType, types.BuiltinFunctionType, types.ModuleType))
            }
            ns_size = sizer.sizeof_namespace(user_vars)
            print(f"Namespace memory: {ns_size.total_bytes / (1024*1024):.1f} MB")
            print(f"Variables: {len(ns_size.by_variable)}")

        elif cmd == "vars":
            # Show per-variable breakdown
            limit = 20
            if len(args) > 1:
                try:
                    limit = int(args[1])
                except ValueError:
                    pass

            sizer = HeapSizer()
            user_ns = self.shell.user_ns
            user_vars = {
                k: v for k, v in user_ns.items()
                if not k.startswith('_') and not isinstance(v, (type, types.FunctionType, types.BuiltinFunctionType, types.ModuleType))
            }
            ns_size = sizer.sizeof_namespace(user_vars)

            # Build list of (name, type, size)
            var_sizes = []
            for name, size in ns_size.by_variable.items():
                type_name = type(user_vars[name]).__name__
                var_sizes.append((name, type_name, size))

            # Sort by size descending
            var_sizes.sort(key=lambda x: x[2], reverse=True)
            var_sizes = var_sizes[:limit]

            # Format output
            print("Variable         Type            Size")
            print("─" * 50)
            for name, type_name, size in var_sizes:
                size_str = self._format_bytes(size)
                print(f"{name:<16} {type_name:<15} {size_str:>10}")

        else:
            print(f"Unknown command: {cmd}")
            print("Usage: %memory [vars [limit]]")

    def _format_bytes(self, size: int) -> str:
        """Format bytes as human-readable string."""
        if size < 1024:
            return f"{size} B"
        elif size < 1024 * 1024:
            return f"{size / 1024:.1f} KB"
        elif size < 1024 * 1024 * 1024:
            return f"{size / (1024 * 1024):.1f} MB"
        else:
            return f"{size / (1024 * 1024 * 1024):.1f} GB"



class BaselineKernel(IPythonKernel):
    """IPython kernel for baseline comparison with FlowBook.

    This kernel provides standard IPython behavior with memory introspection
    via HeapSizer, but without any FlowBook features like reproducibility
    tracking or checkpointing.
    """

    implementation = "baseline_kernel"
    implementation_version = "2.0"
    banner = "Baseline Kernel - For comparison testing"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._memory_metadata: Optional[Dict[str, Any]] = None

    def start(self) -> None:
        """Start the kernel and register magic commands."""
        super().start()

        # Register memory magics
        if self.shell:
            self.shell.register_magics(MemoryMagics)

            # Pre-import pandas and enable same options as FlowBook kernel
            # These are set in memory_checkpoint.py for FlowBook - we need them
            # here for fair comparison
            try:
                import pandas as pd

                # Enable copy-on-write mode for better performance with DataFrame copies
                # (always enabled in pandas >= 3.0, but needs to be set for pandas 2.x)
                if hasattr(pd.options.mode, 'copy_on_write'):
                    pd.options.mode.copy_on_write = True

                # Enable string inference so read_csv() returns StringDtype instead of object dtype
                # (always enabled in pandas >= 3.0, but needs to be set for pandas 2.x)
                if hasattr(pd.options, 'future') and hasattr(pd.options.future, 'infer_string'):
                    pd.options.future.infer_string = True
            except ImportError:
                pass

    async def do_execute(
        self,
        code: str,
        silent: bool,
        store_history: bool = True,
        user_expressions: Optional[Dict[str, Any]] = None,
        allow_stdin: bool = False,
        *,
        cell_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute code with standard IPython behavior and timing.

        Reports kernel-side execution time via display_data metadata
        for fair comparison with FlowBook kernel.
        """
        start_time = time.perf_counter()

        # Execute code normally (IPythonKernel.do_execute is async)
        result = await super().do_execute(
            code,
            silent,
            store_history,
            user_expressions,
            allow_stdin,
            cell_id=cell_id,
        )

        code_duration_ms = (time.perf_counter() - start_time) * 1000

        # Emit timing metadata (same pattern as FlowBook kernel)
        if not silent:
            self.send_response(
                self.iopub_socket,
                "display_data",
                {
                    "data": {"text/plain": f"✓ Code: {code_duration_ms:.0f} ms"},
                    "metadata": {
                        "baseline": {
                            "code_duration_ms": code_duration_ms,
                        }
                    },
                },
            )

        return result
