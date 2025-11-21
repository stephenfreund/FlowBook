"""
Optimization pipeline CLI for ferret notebook processing.

This CLI iterates over cells and runs a three-step optimization pipeline:
1. Profile - Gather performance and memory metrics
2. Inspect - Identify optimization opportunities
3. Optimize - Apply optimizations based on inspection results
"""

import argparse
import json
import sys
import asyncio
from contextlib import nullcontext
import traceback
from typing import Optional, List, Dict, Any

from data_ferret.server.registry import CommandRegistry
from data_ferret.server.config import FerretConfig
from data_ferret.util.output import error, indent, log, timer, quiet, print
from data_ferret.util.ferret_metadata import FerretMetadata, OptimizationPotential
from data_ferret.util.text import wrap_markdown

from .helpers import (
    load_notebook,
    setup_kernel,
    save_notebook,
    cleanup_kernel,
)
from .optimization_metadata import (
    FerretOptimizationMetadata,
    SplitResultsSummary,
    OptimizationResultsSummary,
    CellOptimizationResult,
    LLMCostSummary,
)
from .stats_display import (
    render_split_results_table,
    render_optimization_results_table,
    render_llm_cost_summary_table,
)


def print_inspection_report(
    optimization_potential: OptimizationPotential, cell_index: int
) -> None:
    """
    Print a formatted inspection report for a cell.

    Args:
        optimization_potential: The optimization potential metadata
        cell_index: 1-based index for display
    """
    # Print potential score
    potential_bar = "█" * optimization_potential.potential + "░" * (
        5 - optimization_potential.potential
    )
    print(
        f"- Optimization Potential: {optimization_potential.potential}/5 [{potential_bar}]"
    )

    # Print optimization plan
    if optimization_potential.optimization_plan:
        with indent(message="- Optimization Plan"):
            for i, step in enumerate(optimization_potential.optimization_plan, 1):
                target_desc = (
                    f"function '{step.function_name}'"
                    if step.function_name
                    else "whole cell"
                )
                with indent(message=f"{i}. {step.target_cell_id} ({target_desc})"):
                    print(wrap_markdown("\n".join(step.description), width=100))
    else:
        print(f"- Optimization Plan: None")


def get_code_cell_ids(notebook_content: Dict[str, Any]) -> List[str]:
    """
    Get all code cell IDs from a notebook.

    Args:
        notebook_content: The notebook JSON

    Returns:
        List of code cell IDs
    """
    code_cell_ids = []
    for cell in notebook_content["cells"]:
        if cell.get("cell_type") == "code":
            code_cell_ids.append(cell.get("id"))
    return code_cell_ids


async def simplify_cell_code(
    cell_id: str,
    notebook_content: Dict[str, Any],
    config: FerretConfig,
    registry: CommandRegistry,
) -> Dict[str, Any]:
    """
    Simplify the code in a cell using LLM.

    This is a placeholder implementation. In the future, this could:
    - Remove unnecessary code
    - Clean up formatting
    - Simplify complex expressions
    - Remove redundant computations

    Args:
        cell_id: The cell to simplify
        notebook_content: The notebook (modified in place)
        config: Ferret configuration
        registry: Command registry

    Returns:
        Result dictionary with cost/time info
    """
    # TODO: Implement actual simplification using LLM
    # For now, this is a no-op placeholder
    log(f"Simplification not yet implemented (placeholder)")

    return {
        "cost": 0.0,
        "time": 0.0,
        "simplified": False,
    }


async def optimize_cell(
    cell_id: str,
    cell_index: int,
    total_cells: int,
    notebook_content: Dict[str, Any],
    kernel_client,
    config: FerretConfig,
    registry: CommandRegistry,
    quiet: bool = False,
    simplify: bool = False,
    optimize_iterations: int = 1,
) -> Dict[str, Any]:
    """
    Run the optimization pipeline on a single cell.

    Pipeline:
    0. Simplify (optional) - Simplify code before profiling
    1. Profile - Run ProfileCommand
    2. Inspect - Run InspectCommand
    3. Optimize - Run OptimizeCommand

    Args:
        cell_id: The cell ID to optimize
        cell_index: 1-based index for display
        total_cells: Total number of cells being processed
        notebook_content: The notebook content (will be modified in place)
        kernel_client: The kernel client
        config: Ferret configuration
        registry: Command registry
        quiet: If True, suppress progress messages
        simplify: If True, simplify code before profiling
        optimize_iterations: Number of times to run optimization step

    Returns:
        Dictionary with results from each step
    """
    with indent(message=f"[{cell_index}/{total_cells}] Processing cell {cell_id}"):
        results = {
            "cell_id": cell_id,
            "simplify": None,
            "profile": None,
            "inspect": None,
            "optimize": None,
            "total_cost": 0.0,
            "total_time": 0.0,
        }

        # Step 0: Simplify (optional, NEW - runs BEFORE profile)
        if simplify:
            with indent(message=f"* Simplifying..."):
                with timer(
                    key=f"simplify_cell_{cell_index}",
                    message=f"Simplify cell {cell_index}",
                ):
                    try:
                        simplify_result = await simplify_cell_code(
                            cell_id, notebook_content, config, registry
                        )
                        results["simplify"] = simplify_result
                        results["total_cost"] += simplify_result.get("cost", 0.0)
                        results["total_time"] += simplify_result.get("time", 0.0)
                        log(f"Simplification completed for cell {cell_index}")
                    except Exception as e:
                        error(f"Simplification failed for cell {cell_index}: {e}")
                        results["simplify"] = {"error": str(e)}

        # Step 1: Profile
        with indent(message=f"* Profiling..."):

            # Save environment BEFORE profiling
            from data_ferret.kernel.kernel_command_client import KernelCommandClient

            cmd_client = KernelCommandClient(kernel_client)

            checkpoint_name_before = f"cell_{cell_id}_before_profile"
            cmd_client.checkpoint_save(checkpoint_name_before)
            log(f"Saved checkpoint before profile: {checkpoint_name_before}")

            with timer(
                key=f"profile_cell_{cell_index}", message=f"Profile cell {cell_index}"
            ):
                try:
                    profile_cmd = registry.get_command("profile")
                    profile_result = await profile_cmd.process(
                        notebook_content,
                        kernel_client=kernel_client,
                        selected_cell_ids=[cell_id],
                        config=config,
                    )
                    # Extract cost and time from ProcessingResult
                    notebook_content = profile_result.notebook
                    results["profile"] = profile_result.metadata
                    results["total_cost"] += profile_result.total_cost
                    results["total_time"] += profile_result.total_time
                    log(
                        f"Profile completed for cell {cell_index} (cost: ${profile_result.total_cost:.4f}, time: {profile_result.total_time:.2f}s)"
                    )
                except Exception as e:
                    error(f"Profile failed for cell {cell_index}: {e}")
                    results["profile"] = {"error": str(e)}

            # Save environment AFTER profiling
            checkpoint_name_after = f"cell_{cell_id}_after_profile"
            cmd_client.checkpoint_save(checkpoint_name_after)
            log(f"Saved checkpoint after profile: {checkpoint_name_after}")

            # Extract duration from profile metadata
            profile_duration = None
            for c in notebook_content.get("cells", []):
                if c.get("id") == cell_id:
                    from data_ferret.util.ferret_metadata import FerretMetadata

                    ferret_metadata = FerretMetadata.from_cell(c)
                    profile_data = ferret_metadata.get_profile()
                    if profile_data:
                        profile_duration = profile_data.duration
                        log(f"Extracted profile duration: {profile_duration:.2f}s")
                    break

            # Store checkpoint names and duration for later use
            results["checkpoint_before"] = checkpoint_name_before
            results["checkpoint_after"] = checkpoint_name_after
            results["profile_duration"] = profile_duration

            if profile_duration is not None:
                print(f"- Duration: {profile_duration:.2f}s")
            else:
                print(f"- Duration: No profile data found")

        # Step 2: Inspect
        with indent(message=f"* Inspecting..."):
            with timer(
                key=f"inspect_cell_{cell_index}", message=f"Inspect cell {cell_index}"
            ):
                try:
                    inspect_cmd = registry.get_command("inspect")
                    inspect_result = await inspect_cmd.process(
                        notebook_content,
                        kernel_client=kernel_client,
                        selected_cell_ids=[cell_id],
                        config=config,
                    )
                    # Extract cost and time from ProcessingResult
                    notebook_content = inspect_result.notebook
                    results["inspect"] = inspect_result.metadata
                    results["total_cost"] += inspect_result.total_cost
                    results["total_time"] += inspect_result.total_time
                    log(
                        f"Inspection completed for cell {cell_index} (cost: ${inspect_result.total_cost:.4f}, time: {inspect_result.total_time:.2f}s)"
                    )

                    # Print the full inspection report
                    cell = None
                    for c in notebook_content["cells"]:
                        if c.get("id") == cell_id:
                            cell = c
                            break

                    if cell:
                        ferret_metadata = FerretMetadata.from_cell(cell)
                        optimization_potential = (
                            ferret_metadata.get_optimization_potential()
                        )
                        if optimization_potential:
                            print_inspection_report(optimization_potential, cell_index)

                except Exception as e:
                    error(f"Inspection failed for cell {cell_index}: {e}")
                    traceback.print_exc()
                    results["inspect"] = {"error": str(e)}

            # Check if cell meets optimization criteria
            should_optimize = False
            skip_reason = None

            # Find the cell in notebook
            cell = None
            for c in notebook_content["cells"]:
                if c.get("id") == cell_id:
                    cell = c
                    break

            if cell:
                # Get metadata from cell
                ferret_metadata = FerretMetadata.from_cell(cell)
                profile_data = ferret_metadata.get_profile()
                optimization_potential = ferret_metadata.get_optimization_potential()

                # Check criteria: potential >= 4 and duration > 3.0 seconds
                if profile_data and optimization_potential:
                    duration = profile_data.duration
                    potential = optimization_potential.potential

                    if potential < 4:
                        skip_reason = f"potential too low ({potential} < 4)"
                    elif duration <= 3.0:
                        skip_reason = f"duration too short ({duration:.2f}s <= 3.0s)"
                    else:
                        should_optimize = True
                        log(
                            f"Cell meets optimization criteria: potential={potential}, duration={duration:.2f}s"
                        )
                else:
                    skip_reason = "missing profile or optimization metadata"
            else:
                skip_reason = "cell not found in notebook"

        # Step 3: Optimize (only if criteria met)
        if should_optimize:
            with indent(message=f"* Optimizing..."):
                # Import PrePostEnvironments model
                from data_ferret.server.commands.optimize import PrePostEnvironments

                # Create pre_post_envs using the checkpoints and duration from profile
                pre_post_envs = PrePostEnvironments(
                    original_environment=results.get("checkpoint_before"),
                    original_result=results.get("checkpoint_after"),
                    original_duration=results.get("profile_duration"),
                )

                # Run optimization multiple times if optimize_iterations > 1
                # Keep track of all iterations and use the last successful one
                best_result = None
                total_optimize_cost = 0.0
                total_optimize_time = 0.0

                for iteration in range(optimize_iterations):
                    if optimize_iterations > 1:
                        print(
                            f"Optimization iteration {iteration + 1}/{optimize_iterations}"
                        )

                    with timer(
                        key=f"optimize_cell_{cell_index}_iter_{iteration+1}",
                        message=f"Optimize cell {cell_index} iteration {iteration+1}",
                    ):
                        try:
                            optimize_cmd = registry.get_command("optimize")
                            optimize_result = await optimize_cmd.process(
                                notebook_content,
                                kernel_client=kernel_client,
                                selected_cell_ids=[cell_id],
                                config=config,
                                pre_post_envs=pre_post_envs,
                            )
                            # TODO: Store best for real
                            best_result = optimize_result
                            total_optimize_cost += optimize_result.total_cost
                            total_optimize_time += optimize_result.total_time
                            log(
                                f"Optimization iteration {iteration + 1} completed (cost: ${optimize_result.total_cost:.4f}, time: {optimize_result.total_time:.2f}s)"
                            )
                        except Exception as e:
                            error(
                                f"Optimization iteration {iteration + 1} failed for cell {cell_index}: {e}"
                            )
                            if (
                                iteration == optimize_iterations - 1
                                and best_result is None
                            ):
                                # All iterations failed
                                results["optimize"] = {"error": str(e)}
                                break

                # Store the best (last successful) result
                if best_result:
                    results["optimize"] = best_result.metadata
                    results["total_cost"] += total_optimize_cost
                    results["total_time"] += total_optimize_time
                    if optimize_iterations > 1:
                        log(
                            f"All {optimize_iterations} optimization iterations completed (total cost: ${total_optimize_cost:.4f}, total time: {total_optimize_time:.2f}s)"
                        )
                    else:
                        log(
                            f"Optimization completed for cell {cell_index} (cost: ${total_optimize_cost:.4f}, time: {total_optimize_time:.2f}s)"
                        )
        else:
            with indent(message=f"* Skipping optimization: {skip_reason}"):
                results["optimize"] = {"skipped": True, "reason": skip_reason}
                log(f"Optimization skipped for cell {cell_index}: {skip_reason}")

        with indent(message=f"* Cleaning up..."):
            # Clean up checkpoints created during profiling
            # These are no longer needed after optimization completes
            if results.get("checkpoint_before"):
                try:
                    cmd_client.checkpoint_delete(results["checkpoint_before"])
                except Exception as e:
                    error(
                        f"Failed to delete checkpoint {results['checkpoint_before']}: {e}"
                    )

            if results.get("checkpoint_after"):
                try:
                    cmd_client.checkpoint_delete(results["checkpoint_after"])
                except Exception as e:
                    error(
                        f"Failed to delete checkpoint {results['checkpoint_after']}: {e}"
                    )

        return results


async def run_split_preprocessing(
    notebook_content: Dict[str, Any], config: FerretConfig, registry: CommandRegistry
) -> Dict[str, Any]:
    """
    Run the split command on the notebook before optimization.

    Args:
        notebook_content: The notebook to split
        config: Ferret configuration
        registry: Command registry

    Returns:
        Dictionary with split notebook and metadata including cost/time
    """
    with indent(message="Running Split Preprocessing"):
        log("Splitting cells before optimization")

        # Get split command from registry
        split_cmd = registry.get_command("split")

        if not split_cmd:
            error("Split command not found in registry")
            raise RuntimeError("Split command not available")

        # Execute split command
        with timer(key="split_preprocessing", message="Split preprocessing"):
            split_result = await split_cmd.process(
                notebook_content=notebook_content,
                kernel_client=None,  # Split doesn't need kernel
                selected_cell_ids=None,  # Process all cells
                config=config,
            )

        # Extract results
        split_notebook = split_result.notebook
        split_metadata = split_result.metadata
        split_cost = split_result.total_cost
        split_time = split_result.total_time

        # Log results
        if split_metadata.get("status") == "success":
            log(f"Split complete:")
            log(f"  - Cells analyzed: {split_metadata['cells_analyzed']}")
            log(f"  - Cells split: {split_metadata['cells_split']}")
            log(f"  - Total new cells: {split_metadata['total_new_cells']}")
            log(f"  - LLM cost: ${split_cost:.4f}")
            log(f"  - Total time: {split_time:.2f}s")
        else:
            error(f"Split failed: {split_metadata.get('error', 'Unknown error')}")
            raise RuntimeError("Split preprocessing failed")

        return {
            "notebook": split_notebook,
            "metadata": split_metadata,
            "total_cost": split_cost,
            "total_time": split_time,
        }


async def run_optimization_pipeline(
    notebook_content: Dict[str, Any],
    kernel_client,
    selected_cell_ids: Optional[List[str]],
    config: FerretConfig,
    registry: CommandRegistry,
    simplify: bool = False,
    optimize_iterations: int = 1,
) -> Dict[str, Any]:
    """
    Run the optimization pipeline on selected cells.

    Args:
        notebook_content: The notebook to optimize
        kernel_client: The kernel client
        selected_cell_ids: Optional list of cell IDs to process
        config: Ferret configuration
        registry: Command registry
        simplify: If True, simplify code before profiling
        optimize_iterations: Number of times to run optimization step

    Returns:
        Dictionary with notebook and metadata
    """
    # Determine which cells to process
    if selected_cell_ids:
        cell_ids = selected_cell_ids
    else:
        cell_ids = get_code_cell_ids(notebook_content)

    total_cells = len(cell_ids)
    log(f"Starting optimization pipeline for {total_cells} cell(s)")

    all_results = []

    # Process each cell through the pipeline
    for idx, cell_id in enumerate(cell_ids, start=1):
        cell_results = await optimize_cell(
            cell_id=cell_id,
            cell_index=idx,
            total_cells=total_cells,
            notebook_content=notebook_content,
            kernel_client=kernel_client,
            config=config,
            registry=registry,
            simplify=simplify,
            optimize_iterations=optimize_iterations,
        )
        all_results.append(cell_results)

    # Compile metadata and aggregate costs
    total_cost = sum(result.get("total_cost", 0.0) for result in all_results)
    total_time = sum(result.get("total_time", 0.0) for result in all_results)

    metadata = {
        "total_cells_processed": total_cells,
        "cell_results": all_results,
        "total_cost": total_cost,
        "total_time": total_time,
    }

    return {
        "notebook": notebook_content,
        "metadata": metadata,
    }


def optimize_cli_main():
    """Command-line interface for the optimization pipeline."""
    parser = argparse.ArgumentParser(
        description="Run optimization pipeline (profile -> inspect -> optimize) on notebook cells"
    )

    parser.add_argument("notebook_path", help="Notebook file (.ipynb) to optimize")

    parser.add_argument(
        "--kernel-name",
        default="ferret_kernel",
        help="Kernel name for new kernel (default: ferret_kernel)",
    )

    parser.add_argument(
        "--output",
        "-o",
        help="Output file for the optimized notebook (default: adds _optimized suffix)",
    )

    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="AI model to use for optimization (default: gpt-4o)",
    )

    parser.add_argument(
        "--fast-model",
        default="gpt-4o-mini",
        help="Fast AI model to use for lightweight operations (default: gpt-4o-mini)",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress log and timer messages (errors and main output still shown)",
    )

    parser.add_argument(
        "--split",
        action="store_true",
        help="Run split command on notebook before optimization pipeline",
    )

    parser.add_argument(
        "--simplify",
        action="store_true",
        help="Simplify code before profiling (applies per-cell)",
    )

    parser.add_argument(
        "--optimize-iterations",
        type=int,
        default=1,
        help="Number of times to run optimization step per cell (default: 1, for testing)",
    )

    args = parser.parse_args()

    # Use quiet context manager if --quiet flag is set
    quiet_context = quiet() if args.quiet else nullcontext()

    with quiet_context:

        # Create config from CLI arguments
        config = FerretConfig(model=args.model, fast_model=args.fast_model)

        kernel_manager = None
        kernel_client = None
        registry = CommandRegistry()

        try:
            # Load notebook
            notebook_content = load_notebook(args.notebook_path)

            # Initialize split tracking
            split_result = None

            # Run split preprocessing if requested
            if args.split:
                split_result = asyncio.run(
                    run_split_preprocessing(
                        notebook_content=notebook_content,
                        config=config,
                        registry=registry,
                    )
                )
                # Use the split notebook for optimization
                notebook_content = split_result["notebook"]

            # Setup kernel - always start a new kernel for optimization
            kernel_manager, kernel_client = setup_kernel(kernel_name=args.kernel_name)

            # Run optimization pipeline
            with indent(message="Running Optimization Pipeline"):

                result = asyncio.run(
                    run_optimization_pipeline(
                        notebook_content,
                        kernel_client=kernel_client,
                        selected_cell_ids=None,  # Always process all code cells
                        config=config,
                        registry=registry,
                        simplify=args.simplify,
                        optimize_iterations=args.optimize_iterations,
                    )
                )

            # Display summary metadata
            metadata = result["metadata"]
            print(f"\nProcessed {metadata['total_cells_processed']} cell(s)")

            # Collect timing data for ALL code cells
            timing_summary = []

            for cell_result in metadata["cell_results"]:
                cell_id = cell_result["cell_id"]

                # Find the actual cell in the notebook to get its ferret metadata
                cell = None
                for c in notebook_content["cells"]:
                    if c.get("id") == cell_id:
                        cell = c
                        break

                # Get profile duration and optimization potential from cell's ferret metadata
                profile_duration = 0.0
                potential = None
                if cell:
                    ferret_metadata = FerretMetadata.from_cell(cell)
                    profile_data = ferret_metadata.get_profile()
                    if profile_data:
                        profile_duration = profile_data.duration
                    optimization_potential = (
                        ferret_metadata.get_optimization_potential()
                    )
                    if optimization_potential:
                        potential = optimization_potential.potential

                # Get optimization timing data and status
                optimize_meta = cell_result.get("optimize", {})

                # Default values
                status = "no improvement"
                initial_time = profile_duration
                final_time = profile_duration
                speedup = 1.0

                # Check if optimization was skipped
                if optimize_meta and optimize_meta.get("skipped"):
                    status = "not attempted"
                elif optimize_meta and not optimize_meta.get("error"):
                    # Get timing data from optimize command metadata
                    opt_metadata = (
                        optimize_meta if isinstance(optimize_meta, dict) else {}
                    )
                    cell_timing = opt_metadata.get("cell_timing", {})

                    # Get timing and status for this cell
                    if cell_id in cell_timing:
                        timing_info = cell_timing[cell_id]
                        status = timing_info.get("status", "no improvement")
                        initial_time = timing_info.get(
                            "original_duration", profile_duration
                        )
                        final_time = timing_info.get(
                            "modified_duration", profile_duration
                        )
                        speedup = timing_info.get("speedup", 1.0)

                        # If optimization wasn't applied, final time = initial time
                        if status in ("no improvement", "error"):
                            final_time = initial_time
                            speedup = 1.0

                timing_summary.append(
                    {
                        "cell_id": cell_id,
                        "potential": potential,
                        "initial_time": initial_time,
                        "final_time": final_time,
                        "speedup": speedup,
                        "status": status,
                    }
                )

            # Build optimization results model
            total_initial = 0
            total_final = 0

            cell_results = []
            for timing in timing_summary:
                cell_results.append(
                    CellOptimizationResult(
                        cell_id=timing["cell_id"],
                        potential=timing["potential"],
                        initial_time=timing["initial_time"],
                        final_time=timing["final_time"],
                        speedup=timing["speedup"],
                        status=timing["status"],
                    )
                )
                total_initial += timing["initial_time"]
                total_final += timing["final_time"]

            # Calculate overall stats
            overall_speedup = total_initial / total_final if total_final > 0 else 1.0
            time_saved = total_initial - total_final
            time_saved_percent = (
                (time_saved / total_initial * 100) if total_initial > 0 else 0.0
            )

            optimization_results_summary = OptimizationResultsSummary(
                cells=cell_results,
                total_initial_time=total_initial,
                total_final_time=total_final,
                overall_speedup=overall_speedup,
                time_saved=time_saved,
                time_saved_percent=time_saved_percent,
            )

            # Build LLM cost summary model
            optimization_cost = metadata.get("total_cost", 0.0)
            optimization_time = metadata.get("total_time", 0.0)

            if split_result:
                split_cost = split_result["total_cost"]
                split_time = split_result["total_time"]
                total_cost = split_cost + optimization_cost
                total_time = split_time + optimization_time

                llm_cost_summary = LLMCostSummary(
                    split_cost=split_cost,
                    split_time=split_time,
                    optimization_cost=optimization_cost,
                    optimization_time=optimization_time,
                    total_cost=total_cost,
                    total_time=total_time,
                )
            else:
                total_cost = optimization_cost
                total_time = optimization_time

                llm_cost_summary = LLMCostSummary(
                    optimization_cost=optimization_cost,
                    optimization_time=optimization_time,
                    total_cost=total_cost,
                    total_time=total_time,
                )

            # Build split results summary if applicable
            split_results_summary = None
            if split_result:
                split_metadata = split_result["metadata"]
                split_results_summary = SplitResultsSummary(
                    cells_analyzed=split_metadata["cells_analyzed"],
                    cells_split=split_metadata["cells_split"],
                    total_new_cells=split_metadata["total_new_cells"],
                    llm_cost=split_result["total_cost"],
                    time=split_result["total_time"],
                )

            # Build complete optimization metadata
            ferret_metadata = FerretOptimizationMetadata(
                split_results=split_results_summary,
                optimization_results=optimization_results_summary,
                llm_costs=llm_cost_summary,
                timestamp=FerretOptimizationMetadata.create_timestamp(),
                model=config.model,
                fast_model=config.fast_model,
            )

            # Store metadata in notebook
            if "metadata" not in result["notebook"]:
                result["notebook"]["metadata"] = {}
            result["notebook"]["metadata"][
                "ferret_optimization"
            ] = ferret_metadata.model_dump()

            # Determine output path
            if args.output:
                output_path = args.output
            else:
                base_name = args.notebook_path.rsplit(".", 1)[0]
                if args.split:
                    output_path = f"{base_name}_split_optimized.ipynb"
                else:
                    output_path = f"{base_name}_optimized.ipynb"

            # Save notebook with embedded metadata
            save_notebook(result["notebook"], output_path=output_path)
            print(f"\n{'='*70}")
            print(f"Optimized notebook written to {output_path}")
            print(f"{'='*70}")

            # Display results using rendering functions
            if split_results_summary:
                render_split_results_table(split_results_summary)

            render_optimization_results_table(optimization_results_summary)
            render_llm_cost_summary_table(llm_cost_summary)

            # # Save full metadata to JSON file
            # metadata_path = output_path.rsplit(".", 1)[0] + "_metadata.json"
            # full_metadata = {
            #     'optimization': metadata
            # }
            # if split_result:
            #     full_metadata['split'] = split_result['metadata']
            #     full_metadata['total_cost'] = total_cost
            #     full_metadata['total_time'] = total_time
            # with open(metadata_path, "w", encoding="utf-8") as f:
            #     json.dump(full_metadata, f, indent=2)
            # print(f"\nFull metadata written to {metadata_path}")

            return 0

        except FileNotFoundError as e:
            print(f"Error: File not found: {e}", file=sys.stderr)
            return 1
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in notebook: {e}", file=sys.stderr)
            return 1
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            import traceback

            traceback.print_exc()
            return 1
        finally:
            cleanup_kernel(kernel_client, kernel_manager)


if __name__ == "__main__":
    sys.exit(optimize_cli_main())
