"""Lightweight session for the FlowBook NBI extension.

Tracks checkpoints (cell source snapshots) and an event log of tool calls.
All notebook state lives in JupyterLab's frontend --- this session only stores
data that can't be derived from the frontend.
"""

import time
import json
from datetime import datetime


class FlowBookSession:
    """Lightweight session that tracks checkpoints and an event log."""

    def __init__(self):
        self._checkpoints: dict[str, list[dict]] = {}
        self._event_log: list[dict] = []
        self._next_checkpoint_id: int = 0
        self._start_time: float = time.time()

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------

    def save_checkpoint(self, cells: list[dict]) -> str:
        """Save a snapshot of cell sources/types.

        Args:
            cells: List of dicts with keys: 'label', 'cell_type', 'source'
                   (as returned by flowbook:get-cell bridge command)

        Returns:
            Checkpoint ID string (e.g., 'ckpt_0', 'ckpt_1', ...)
        """
        checkpoint_id = f'ckpt_{self._next_checkpoint_id}'
        self._next_checkpoint_id += 1
        # Store a defensive copy so later mutations don't affect the snapshot.
        self._checkpoints[checkpoint_id] = [dict(c) for c in cells]
        return checkpoint_id

    def get_checkpoint(self, checkpoint_id: str) -> list[dict]:
        """Retrieve a checkpoint by ID.

        Raises KeyError if checkpoint_id not found.
        """
        if checkpoint_id not in self._checkpoints:
            raise KeyError(f'Checkpoint not found: {checkpoint_id}')
        return self._checkpoints[checkpoint_id]

    def list_checkpoints(self) -> list[dict]:
        """List all checkpoints.

        Returns list of: {'id': str, 'cell_count': int, 'timestamp': str}
        """
        result = []
        for cp_id, cells in self._checkpoints.items():
            result.append({
                'id': cp_id,
                'cell_count': len(cells),
                'timestamp': datetime.now().isoformat(),
            })
        return result

    # ------------------------------------------------------------------
    # Event Log
    # ------------------------------------------------------------------

    def log_event(
        self,
        tool: str,
        args: dict,
        result: str,
        duration_ms: float,
        error: str = None,
    ) -> None:
        """Record a tool invocation in the event log.

        Args:
            tool: Tool name
            args: Tool arguments (already serializable)
            result: Result string (truncated to 2000 chars)
            duration_ms: Execution duration in milliseconds
            error: Error string if tool failed, None otherwise
        """
        now = time.time()
        truncated_result = result[:2000] if result and len(result) > 2000 else result
        entry = {
            'seq': len(self._event_log) + 1,
            'tool': tool,
            'args': args,
            'result': truncated_result,
            'duration_ms': duration_ms,
            'error': error,
            'timestamp': datetime.now().isoformat(),
            'relative_time_s': round(now - self._start_time, 3),
        }
        self._event_log.append(entry)

    def get_log(self) -> list[dict]:
        """Return the full event log as a list of dicts."""
        return list(self._event_log)

    def format_log(self) -> str:
        """Return a human-readable timeline of tool calls.

        Format:
        [001] 0.0s  load_notebook(path="test.ipynb")  -> 234ms
        [002] 0.5s  run_cell(cell="@A")  -> 1205ms
        [003] 1.8s  get_status()  -> 12ms  ERROR: ...
        """
        lines = []
        for entry in self._event_log:
            seq = entry['seq']
            rel = entry['relative_time_s']
            tool = entry['tool']
            args = entry['args']

            # Format arguments
            if args:
                arg_parts = [f'{k}="{v}"' if isinstance(v, str) else f'{k}={v}'
                             for k, v in args.items()]
                arg_str = ', '.join(arg_parts)
            else:
                arg_str = ''

            dur = entry['duration_ms']
            line = f'[{seq:03d}] {rel:.1f}s  {tool}({arg_str})  -> {dur:.0f}ms'

            if entry.get('error'):
                line += f'  ERROR: {entry["error"]}'

            lines.append(line)
        return '\n'.join(lines)

    def save_log_to_file(self, path: str) -> str:
        """Write event log to a JSON file. Returns the path written."""
        with open(path, 'w') as f:
            json.dump(self._event_log, f, indent=2)
        return path
