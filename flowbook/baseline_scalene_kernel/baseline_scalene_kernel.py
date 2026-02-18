"""Baseline kernel with Scalene memory tracking for fair comparison.

This kernel provides the same Scalene memory measurement capabilities as FlowBook
but without any reproducibility tracking, checkpointing, or other FlowBook features.
It's used for fair A/B comparison of memory overhead.

Features:
- Standard IPython kernel behavior
- %scalene_memory magic command (same as FlowBook)
- Memory reporting in same format as FlowBook for comparison

Does NOT include:
- Reproducibility tracking
- Checkpointing
- Variable access tracking
- Any FlowBook-specific features
"""

from typing import Any, Dict, Optional

from ipykernel.ipkernel import IPythonKernel
from IPython.core.magic import line_magic, magics_class, Magics

from flowbook.kernel_support.scalene_memory import ScaleneMemoryTracker


@magics_class
class ScaleneMemoryMagics(Magics):
    """Scalene memory magic commands for baseline comparison."""

    @line_magic
    def scalene_memory(self, line: str) -> None:
        """Control Scalene-based memory tracking.

        Usage:
            %scalene_memory on      - Enable memory + GPU tracking
            %scalene_memory off     - Disable tracking
            %scalene_memory ?       - Show total memory + GPU stats
            %scalene_memory reset   - Reset all statistics
        """
        args = line.strip().lower().split()
        if not args:
            args = ["?"]

        cmd = args[0]

        if cmd == "on":
            if not ScaleneMemoryTracker.is_available():
                print(
                    "Scalene not available. Kernel must be launched with "
                    "DYLD_INSERT_LIBRARIES (macOS) or LD_PRELOAD (Linux) set."
                )
                return
            if ScaleneMemoryTracker.start():
                print("Scalene memory tracking enabled")
            else:
                print("Failed to enable Scalene tracking")

        elif cmd == "off":
            if ScaleneMemoryTracker.stop():
                print("Scalene memory tracking disabled")
            else:
                print("Failed to disable Scalene tracking (or not running)")

        elif cmd == "?" or cmd == "status":
            if not ScaleneMemoryTracker.is_available():
                print("Scalene not available")
                return
            stats = ScaleneMemoryTracker.get_memory()
            print(ScaleneMemoryTracker.format_stats(stats))
            tracking = "enabled" if ScaleneMemoryTracker.is_tracking() else "disabled"
            print(f"Tracking: {tracking}")

        elif cmd == "reset":
            if ScaleneMemoryTracker.reset():
                print("Scalene statistics reset")
            else:
                print("Failed to reset (Scalene not available)")

        else:
            print(f"Unknown command: {cmd}")
            print("Usage: %scalene_memory on|off|?|reset")

    @line_magic
    def scalene(self, line: str) -> None:
        """Alias for %scalene_memory."""
        return self.scalene_memory(line)


class BaselineScaleneKernel(IPythonKernel):
    """IPython kernel with Scalene memory tracking for baseline comparison.

    This kernel mirrors FlowBook's Scalene usage but without reproducibility
    features, allowing fair A/B comparison of memory overhead.
    """

    implementation = "baseline_scalene_kernel"
    implementation_version = "1.0"
    banner = "Baseline Scalene Kernel - For memory comparison testing"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._memory_metadata: Optional[Dict[str, Any]] = None

    def start(self) -> None:
        """Start the kernel and register magic commands."""
        super().start()

        # Register scalene memory magics
        if self.shell:
            self.shell.register_magics(ScaleneMemoryMagics)

    def do_execute(
        self,
        code: str,
        silent: bool,
        store_history: bool = True,
        user_expressions: Optional[Dict[str, Any]] = None,
        allow_stdin: bool = False,
        *,
        cell_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute code with optional Scalene memory tracking.

        If Scalene tracking is enabled, records memory stats before and after
        execution and includes them in output metadata for comparison.
        """
        # Get memory state before execution (if tracking)
        before_stats = None
        if ScaleneMemoryTracker.is_tracking():
            before_stats = ScaleneMemoryTracker.get_memory()

        # Execute code normally
        result = super().do_execute(
            code,
            silent,
            store_history,
            user_expressions,
            allow_stdin,
            cell_id=cell_id,
        )

        # Get memory state after execution (if tracking)
        if ScaleneMemoryTracker.is_tracking() and before_stats:
            after_stats = ScaleneMemoryTracker.get_memory()

            # Calculate deltas
            memory_delta = {
                "current_footprint_mb": after_stats["current_footprint_mb"],
                "max_footprint_mb": after_stats["max_footprint_mb"],
                "allocation_delta_mb": (
                    after_stats["total_malloc_mb"] - before_stats["total_malloc_mb"]
                ),
                "free_delta_mb": (
                    after_stats["total_free_mb"] - before_stats["total_free_mb"]
                ),
                "net_delta_mb": (
                    after_stats["net_allocation_mb"] - before_stats["net_allocation_mb"]
                ),
                "gpu_samples": after_stats.get("gpu_samples", 0),
                "gpu_mem_samples": after_stats.get("gpu_mem_samples", 0),
            }

            # Store for later retrieval via metadata
            self._memory_metadata = {
                "baseline_scalene": {
                    "cell_id": cell_id,
                    "memory": memory_delta,
                }
            }

            # Also publish as display_data for comparison tools
            if not silent:
                self.send_response(
                    self.iopub_socket,
                    "display_data",
                    {
                        "data": {"text/plain": ""},
                        "metadata": {"baseline_scalene": memory_delta},
                    },
                )

        return result
