# FlowBook CLI Reference

FlowBook provides command-line tools for interactive notebook development, overhead benchmarking, result analysis, and batch execution on clusters.

## `flowlab` — Launch JupyterLab

Start JupyterLab with FlowBook extensions enabled.

```bash
flowlab
```

This is a wrapper around `jupyter lab` that ensures FlowBook's server extension and frontend plugins are active.

## `flowbook` — Notebook Processing

Unified CLI for running any registered notebook command. The most important subcommand for benchmarking is `compare-baseline`.

```bash
flowbook <command> [options] <notebook.ipynb>
```

### `flowbook compare-baseline` — Benchmark FlowBook Overhead

Runs a notebook through multiple phases to measure FlowBook's timing and memory overhead compared to a vanilla Python kernel.

**Default phases** (3 of 4 run by default):

1. **FlowBook Timing** — Execute with FlowBook kernel, measure per-cell timing breakdown (code, state, check)
2. **Baseline Timing** — _Skipped by default_ (enable with `--baseline-timing`)
3. **Baseline Memory** — Execute with vanilla kernel, measure namespace size before/after each cell
4. **FlowBook Memory** — Execute with FlowBook kernel, measure namespace + checkpoint + enforcer overhead

**Output:** A JSON file (v3.0 format) with per-cell timing and memory data for both kernels.

#### Usage

```bash
# Default: FlowBook timing + baseline memory + FlowBook memory
flowbook compare-baseline notebook.ipynb

# Skip baseline memory (FlowBook-only mode, less accurate memory plots)
flowbook compare-baseline notebook.ipynb --skip-baseline

# Also run baseline timing phase
flowbook compare-baseline notebook.ipynb --baseline-timing

# Timing only, no memory measurement
flowbook compare-baseline notebook.ipynb --skip-memory

# Multiple trials
flowbook compare-baseline notebook.ipynb --trials 3

# With cell timeout and rerun passes
flowbook compare-baseline notebook.ipynb --timeout 120 --rerun-k 2

```

#### Options

| Flag                | Default  | Description                                                                       |
| ------------------- | -------- | --------------------------------------------------------------------------------- |
| `--timeout TIMEOUT` | no limit | Timeout in seconds per cell                                                       |
| `--skip-memory`     | off      | Skip all memory measurement phases                                                |
| `--skip-baseline`   | off      | Skip baseline memory run (disables cross-run comparison)                          |
| `--baseline-timing` | off      | Also run baseline timing phase                                                    |
| `--rerun-k K`       | 0        | Number of extra top-to-bottom rerun passes after initial execution                |
| `--trials N`        | 1        | Number of independent trials (saved as `notebook-1.json`, `notebook-2.json`, ...) |
| `--start N`         | 1        | Starting trial number                                                             |

#### Output JSON (v3.0)

The output JSON contains timing and memory data for both kernels:

```
kernels.baseline.timing   — per-cell execution time (if --baseline-timing)
kernels.baseline.memory   — per-cell namespace size before/after each cell
kernels.flowbook.timing   — per-cell execution time with state/check breakdown
kernels.flowbook.memory   — per-cell namespace + checkpoint + enforcer overhead
```

Memory cells include `pre_namespace_mb` and `namespace_mb` (before/after each cell) enabling cross-run subtraction to isolate FlowBook's overhead.

## `flowbook_compare_overhead` — Generate Overhead Plots

Processes one or more comparison JSON files (from `flowbook compare-baseline`) and generates statistics tables and PDF plots.

```bash
flowbook_compare_overhead [options] <files_or_dirs...>
```

#### Usage

```bash
# Process all JSON files in a directory, generate plots
flowbook_compare_overhead results/ --plot

# Table output (default)
flowbook_compare_overhead results/

# JSON or CSV output
flowbook_compare_overhead results/ --format json
flowbook_compare_overhead results/ --format csv

# Sort by memory overhead instead of slowdown
flowbook_compare_overhead results/ --sort-by memory --plot

# Custom output filename
flowbook_compare_overhead results/ --plot --output my_plots.pdf

# Large fonts for paper figures
flowbook_compare_overhead results/ --plot --large-fonts

# Process remote files (scp-style paths)
flowbook_compare_overhead user@host:/path/to/*.json --plot
```

#### Options

| Flag                                       | Default          | Description                                           |
| ------------------------------------------ | ---------------- | ----------------------------------------------------- |
| `--format {table,json,csv}`                | table            | Output format for statistics                          |
| `--sort-by {slowdown,memory,runtime,name}` | slowdown         | Sort order for notebooks                              |
| `--plot`                                   | off              | Generate PDF plots                                    |
| `--output FILE`                            | all_overhead.pdf | Output PDF filename                                   |
| `--output-dir DIR`                         | current dir      | Directory for plot output                             |
| `--large-fonts`                            | off              | Larger fonts for publication-ready plots              |
| `--top-n N`                                | 10               | Number of top variables to show individually in plots |
| `--force-download`                         | off              | Re-download remote files (ignore cache)               |
| `--clear-cache`                            | off              | Clear cached remote files and exit                    |

#### Generated Plots (6 panels per notebook)

| Panel            | Title                         | Content                                                                                                |
| ---------------- | ----------------------------- | ------------------------------------------------------------------------------------------------------ |
| 1 (top-left)     | Timing                        | Cumulative time: baseline vs FlowBook (code + state + check + other)                                   |
| 2 (top-right)    | Checkpoint Time by Variable   | Per-variable stacked area of checkpoint deepcopy time                                                  |
| 3 (mid-left)     | Memory Overhead               | Cross-run: baseline namespace + FlowBook overhead. Fallback: FlowBook namespace + checkpoint_var_costs |
| 4 (mid-right)    | Checkpoint Memory by Variable | Per-variable stacked area of checkpoint memory                                                         |
| 5 (bottom-left)  | Overhead Time per Cell        | Per-cell bar chart of state + check + other overhead                                                   |
| 6 (bottom-right) | Checkpoint Overhead Ratio     | Cross-run: Checkpoint_i / Base_i. Fallback: checkpoint_var_costs delta / namespace                     |

Additional pages: aggregate histograms and CDFs across all notebooks.

## `flowbook_timers` — Analyze Timing Data

Analyzes `flowbook-times.json` files produced during notebook processing.

```bash
flowbook_timers [options] <files...>
```

#### Usage

```bash
# Analyze timing data
flowbook_timers flowbook-times.json

# Sort by total time, show top 10
flowbook_timers flowbook-times.json --sort-by total --top 10

# Filter to specific timer keys
flowbook_timers flowbook-times.json --keys "checkpoint" "diff"

# Show ASCII histograms
flowbook_timers flowbook-times.json --histograms

# Generate histogram PDFs
flowbook_timers flowbook-times.json --histplot "checkpoint_ms"

# Generate scatter plot PDFs
flowbook_timers flowbook-times.json --scatterplot "state_ms%check_ms"

# JSON or CSV output
flowbook_timers flowbook-times.json --format json

# Process remote files
flowbook_timers user@host:/path/flowbook-times.json
```

#### Options

| Flag                                   | Default     | Description                            |
| -------------------------------------- | ----------- | -------------------------------------- |
| `--format {table,json,csv}`            | table       | Output format                          |
| `--sort-by {total,mean,count,max,key}` | key         | Sort order                             |
| `--top N`                              | all         | Show only top N timers                 |
| `--keys KEY [KEY ...]`                 | all         | Filter to specific timer keys          |
| `--histograms`                         | off         | Show ASCII histograms (table mode)     |
| `--histplot KEY`                       | —           | Generate histogram PDF for a timer key |
| `--scatterplot KEY%KEY`                | —           | Generate scatter plot PDF for two keys |
| `--output-dir DIR`                     | current dir | Directory for plot output              |
| `--clip PERCENTILE`                    | —           | Clip outliers above this percentile    |
| `--force-download`                     | off         | Re-download remote files               |
| `--clear-cache`                        | off         | Clear cached remote files              |

## `flowbook_slurm` — Batch Execution on Slurm Clusters

Submits one Slurm job per notebook, with automatic per-notebook conda environment discovery. Wraps any `flowbook` subcommand for cluster execution.

```bash
flowbook_slurm [options] <input_files...> -- <flowbook subcommand and args>
```

The `flowbook` command prefix with `--timings-file` and `--metadata-file` is automatically prepended. Everything after `--` is the FlowBook subcommand to run on each notebook.

#### Usage

```bash
# Benchmark all notebooks listed in a file
flowbook_slurm notebooks.txt -- compare-baseline --rerun-k 1

# Benchmark specific notebooks
flowbook_slurm notebook1.ipynb notebook2.ipynb -- compare-baseline

# Dry run to see what would be submitted
flowbook_slurm notebooks.txt --dry-run -- compare-baseline

# Run locally instead of submitting to Slurm
flowbook_slurm notebooks.txt --local -- compare-baseline --rerun-k 1

# Custom Slurm resources
flowbook_slurm notebooks.txt --mem 32G --cpus 8 --gpus 2 --time 48:00:00 -- compare-baseline

# Use a specific conda environment for all notebooks
flowbook_slurm notebooks.txt --env myenv -- compare-baseline

# Create/recreate per-notebook conda environments before submitting
flowbook_slurm notebooks.txt --make-env -- compare-baseline
```

#### Options

| Flag                    | Default        | Description                                                 |
| ----------------------- | -------------- | ----------------------------------------------------------- |
| `--partition PARTITION` | gpmoo-b        | Slurm partition                                             |
| `--time TIME_LIMIT`     | 24:00:00       | Time limit                                                  |
| `--mem MEM`             | 16G            | Memory request                                              |
| `--cpus CPUS`           | 4              | CPUs per task                                               |
| `--gpus GPUS`           | 1              | GPUs per task                                               |
| `--job-name JOB_NAME`   | flowbook-batch | Slurm job name prefix                                       |
| `--env ENV`             | auto-discover  | Conda environment (overrides per-notebook `_env` discovery) |
| `--make-env`            | off            | Create/recreate `_env` environments before submitting       |
| `--dry-run`             | off            | Print sbatch commands without submitting                    |
| `--local`               | off            | Run commands locally in sequence instead of Slurm           |
| `--no-wait`             | off            | Submit jobs and exit without waiting for completion         |
| `--poll-interval SECS`  | —              | Seconds between job status polls                            |
| `--log-dir DIR`         | —              | Directory for Slurm log files                               |

#### Input files

- **`.txt` files** — One notebook path per line (supports comments with `#`)
- **`.ipynb` files** — Direct notebook paths

## Use Cases

### Benchmarking a single notebook locally

Measure FlowBook's overhead on one notebook and generate plots:

```bash
flowbook compare-baseline notebook.ipynb --rerun-k 1
flowbook_compare_overhead . --plot --large-fonts
```

### Benchmarking a suite of notebooks on a cluster

Run benchmarks across many notebooks using Slurm, then aggregate results:

```bash
# Create a list of notebooks
find kaggle_notebooks/ -name "*.ipynb" > notebooks.txt

# Submit to Slurm (3 trials each, with reruns)
flowbook_slurm notebooks.txt --mem 32G -- compare-baseline --trials 3 --rerun-k 1

# After jobs complete, collect results and generate aggregate plots
flowbook_compare_overhead results/ --plot --large-fonts --output benchmark_results.pdf
```

### Analyzing timing bottlenecks

Identify which FlowBook operations are slowest:

```bash
# Run benchmark
flowbook compare-baseline notebook.ipynb

# Find slowest timer keys
flowbook_timers flowbook-times.json --sort-by total --top 20

# Deep-dive into checkpoint timing distribution
flowbook_timers flowbook-times.json --histplot "state_total_ms" --histplot "check_total_ms"

# Correlate state vs check time
flowbook_timers flowbook-times.json --scatterplot "state_total_ms%check_total_ms"
```

### Interactive development

Launch JupyterLab with FlowBook's reproducibility tracking and UI extensions:

```bash
flowlab
```

Then select the `flowbook_kernel` for reproducibility enforcement or `experimental_kernel` for AI commands and profiling.
