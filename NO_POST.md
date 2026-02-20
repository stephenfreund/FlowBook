# Plan: Remove Post Checkpoints via Predecessor Re-execution

## Context

**Problem**: The FlowBook kernel stores two full namespace checkpoints per cell execution:
- `_pre_{cell_id}`: Before execution (used for backward mutation detection + staleness)
- `_post_{cell_id}`: After execution (used ONLY for EXEC-RESTORE)

This effectively doubles memory usage. Post checkpoints are rarely used (only when user explicitly runs `%exec_restore` on a contaminated cell).

**Goal**: Eliminate post checkpoint storage by reconstructing the post state on-demand via re-executing the predecessor cell from its pre-checkpoint.

**Memory Savings**: ~50% reduction in checkpoint memory (one checkpoint per cell instead of two).

---

## Implementation Plan

### Step 1: Add Source Code Storage to Execution Records

**File**: `flowbook/kernel/models.py`

Add `source: str` field to `ReproducibilityExecutionRecord`:

```python
@dataclass
class ReproducibilityExecutionRecord:
    """Record of a cell's most recent execution — Rec[i] in the formalism (§1.6)."""
    cell_id: str
    tracking: TrackingData
    execution_seq: int
    source: str  # NEW: Cell source code for replay
    structural_reads_values: Dict[str, Dict[str, str]] = field(default_factory=dict)
    typed_changes: List["Change"] = field(default_factory=list)
```

### Step 2: Update Execution Record Creation to Store Source

**File**: `flowbook/kernel/reproducibility_enforcer.py`

Update all places where `ReproducibilityExecutionRecord` is created to pass source code:
- `_check_live()` (line ~443): Add `source` parameter
- `_check_exec_restore()` (line ~1143): Add `source` parameter
- `check()` method: Accept and pass through `source` parameter

### Step 3: Pass Source Code from Kernel to Enforcer

**File**: `flowbook/kernel/flowbook_kernel.py`

Update `_do_execute_impl()` to pass the `code` parameter to `self._enforcer.check()`:

```python
sdc_result = self._enforcer.check(
    cell_id=self._cell_id,
    pre_checkpoint=pre_checkpoint,
    post_checkpoint=post_checkpoint,  # Will become live namespace
    tracking=tracking,
    continue_on_violation=self._continue_after_violation,
    namespace=self.shell.user_ns,
    is_exec_restore=_is_exec_restore,
    old_live_checkpoint=_old_live_checkpoint,
    source=code,  # NEW: Pass source for storage
)
```

### Step 4: Remove Post Checkpoint Creation

**File**: `flowbook/kernel/flowbook_kernel.py` (lines 1181-1186)

Remove the post-checkpoint creation block:

```python
# REMOVE THIS:
with timer(key="kernel:checkpoint", message="Post-execution checkpoint") as post_timer:
    post_checkpoint = self._take_checkpoint(f"{POST_CHECKPOINT_PREFIX}{self._cell_id}")
```

### Step 5: Use Live Namespace for Diff Computation

**File**: `flowbook/kernel/flowbook_kernel.py`

Instead of taking a deep-copy post checkpoint, create a lightweight wrapper around the live namespace for the immediate diff computation:

```python
# After execution, wrap live namespace for diff (NO deep copy)
# The live namespace IS the post state at this moment
live_post = LiveNamespaceCheckpoint(self.shell.user_ns, f"_live_{self._cell_id}")

sdc_result = self._enforcer.check(
    ...
    post_checkpoint=live_post,  # Use live state, not deep copy
    ...
)
# Note: live_post is only valid until namespace is mutated
# It's used immediately for diff and then discarded
```

**File**: `flowbook/kernel_support/memory_checkpoint.py`

Add a lightweight wrapper class that mimics MemoryCheckpoint interface for diff:

```python
class LiveNamespaceCheckpoint:
    """Lightweight wrapper around live namespace for immediate diff computation.

    Unlike MemoryCheckpoint, this does NOT deep copy the namespace.
    It's only valid until the namespace is mutated.
    Used to avoid the memory overhead of post checkpoints.
    """

    def __init__(self, user_ns: dict, name: str):
        self.name = name
        # Filter but don't copy - direct reference to live objects
        self.user_ns = filter_user_namespace(user_ns)
        self.reverse_memo = {}
        self.cudf_origins = None
        self._reachable_ids = None
        self._id_to_vars = None
        self._id_to_paths = None
        self._alias_index_built = False

    # The diff() function accesses .user_ns, so this wrapper is compatible
    # No need to implement full MemoryCheckpoint interface
```

**Alternative (simpler)**: If `MemoryCheckpoint.diff()` only needs `.user_ns`, we can pass a simple object:

```python
# Simplest approach - just wrap the namespace
class _LiveNS:
    def __init__(self, ns): self.user_ns = filter_user_namespace(ns)

live_post = _LiveNS(self.shell.user_ns)
```

### Step 6: Add Post Checkpoint Reconstruction Method

**File**: `flowbook/kernel/flowbook_kernel.py`

Add method to reconstruct post checkpoint by re-executing predecessor:

```python
async def _reconstruct_post_checkpoint(self, cell_id: str) -> Optional[MemoryCheckpoint]:
    """Reconstruct post checkpoint by re-executing cell from its pre checkpoint.

    Used for EXEC-RESTORE when we need the prefix state but don't have
    a stored post checkpoint.

    Args:
        cell_id: The cell whose post state we need to reconstruct

    Returns:
        MemoryCheckpoint representing the post state, or None if reconstruction fails
    """
    # Get execution record with source code
    record = self._enforcer.records.get(cell_id)
    if record is None:
        log(f"[reconstruct] No execution record for {cell_id}")
        return None
    if not record.source:
        log(f"[reconstruct] No source code stored for {cell_id}")
        return None

    pre_name = f"{PRE_CHECKPOINT_PREFIX}{cell_id}"
    if pre_name not in self._checkpoints.memory.saved:
        log(f"[reconstruct] No pre checkpoint for {cell_id}")
        return None

    # Save current live state
    temp_name = f"_temp_live_{cell_id}"
    current_state = self._take_checkpoint(temp_name)

    try:
        # Restore to pre state
        self._restore_checkpoint(pre_name)

        # Re-execute the cell's source code (silently, without tracking)
        # Use raw IPython execution to avoid recursive reproducibility checks
        self.shell.run_cell(record.source, silent=True, store_history=False)

        # Capture the reconstructed post state
        reconstructed_name = f"_reconstructed_post_{cell_id}"
        reconstructed = self._take_checkpoint(reconstructed_name)

        return reconstructed

    finally:
        # Restore to original live state
        self._restore_checkpoint(temp_name)
        self._checkpoints.delete(temp_name)
```

### Step 7: Update EXEC-RESTORE Logic

**File**: `flowbook/kernel/flowbook_kernel.py` (lines 995-1054)

Modify EXEC-RESTORE to reconstruct post checkpoint when not found:

```python
elif prefix_name in self._checkpoints.memory.saved:
    # Post checkpoint exists (rare - from cache or legacy)
    _old_live_checkpoint = self._take_checkpoint(f"_old_live_{self._cell_id}")
    self._restore_checkpoint(prefix_name)
    _is_exec_restore = True

else:
    # Post checkpoint not stored - reconstruct by re-executing predecessor
    prev_cell_id = prefix_name.replace(POST_CHECKPOINT_PREFIX, "")
    log(f"[exec_restore] Reconstructing post checkpoint for {prev_cell_id}")

    _old_live_checkpoint = self._take_checkpoint(f"_old_live_{self._cell_id}")
    reconstructed = await self._reconstruct_post_checkpoint(prev_cell_id)

    if reconstructed is not None:
        # Don't need to restore - reconstruction left us in the right state
        # The _reconstruct_post_checkpoint restores to pre, executes, then
        # restores back. We need to restore to the reconstructed state.
        self._restore_checkpoint(reconstructed.name)
        _is_exec_restore = True
        log(f"[exec_restore] Restored reconstructed prefix for cell {self._cell_id}")

        # Optionally cache the reconstructed checkpoint under the standard name
        # for future EXEC-RESTORE calls (uncomment if desired)
        # self._checkpoints.memory.saved[prefix_name] = reconstructed
    else:
        # Reconstruction failed
        error_msg = f"Cannot restore {self._cell_id}: failed to reconstruct prefix state"
        log(f"[exec_restore] {error_msg}")
        self._display.display_icon_and_text("...", error_msg)
        self._pending_exec_restore = None
        return {"status": "error", ...}
```

### Step 8: Update `get_prefix_checkpoint_name()` Signature

**File**: `flowbook/kernel/reproducibility_enforcer.py`

Update to return the predecessor cell_id as well (for reconstruction):

```python
def get_prefix_checkpoint_info(self, cell_id: str) -> Tuple[Optional[str], Optional[str]]:
    """[PrefixStore] (Def 1.8.4)

    Returns (checkpoint_name, predecessor_cell_id) for EXEC-RESTORE.
    checkpoint_name is the post-checkpoint name of the predecessor.
    predecessor_cell_id is the cell_id of the predecessor (for reconstruction).

    Returns (None, None) if cell_id is the first cell.
    """
    try:
        my_position = self._cell_order.index(cell_id)
    except ValueError:
        return None, None

    if my_position == 0:
        return None, None  # First cell — restore to initial state

    prev_cell_id = self._cell_order[my_position - 1]
    return f"{POST_CHECKPOINT_PREFIX}{prev_cell_id}", prev_cell_id
```

### Step 9: Update Formal Specification

**File**: `FORMAL_DEVELOPMENT.md`

Update Definition 1.8.4 (Prefix Checkpoint) to reflect reconstruction approach:

```markdown
**Definition 1.8.4 (Prefix Checkpoint — Reconstructed)**

The prefix store σ^post_{i-1} is not persisted after every execution.
Instead, it is reconstructed on demand for EXEC-RESTORE:

1. Restore namespace to σ^pre_{i-1} (the pre-checkpoint of cell i-1)
2. Re-execute Code[i-1] (the stored source code of cell i-1)
3. The resulting namespace state is σ^post_{i-1}

**Precondition**: Cell i-1 must be fresh (not stale), ensuring deterministic replay.

**Trade-offs**:
- Memory: Eliminates O(N) post checkpoint storage
- Time: Adds O(execution_time) to EXEC-RESTORE (rare operation)
- Correctness: Assumes cell execution is deterministic given its pre-state
```

### Step 10: Update Tests

**Files affected**:
- `flowbook/kernel/tests/test_exec_restore_magic.py`
- `flowbook/kernel/tests/test_forward_dependency.py`
- `flowbook/kernel/tests/test_reproducibility_enforcer.py`

Tests that explicitly create `post_{cell_id}` or `_post_{cell_id}` checkpoints need updates:
1. Either use the new reconstruction approach
2. Or explicitly create post checkpoints for test isolation (legacy mode)

Add new tests:
- Test that source code is stored in execution records
- Test reconstruction of post checkpoint from pre + source
- Test EXEC-RESTORE with reconstruction
- Test reconstruction failure handling

---

## Files to Modify

| File | Changes |
|------|---------|
| `flowbook/kernel/models.py` | Add `source: str` field to `ReproducibilityExecutionRecord` |
| `flowbook/kernel/reproducibility_enforcer.py` | Update record creation, add `get_prefix_checkpoint_info()` |
| `flowbook/kernel/flowbook_kernel.py` | Remove post checkpoint creation, add reconstruction method, update EXEC-RESTORE |
| `flowbook/kernel_support/memory_checkpoint.py` | Add `from_live_namespace()` classmethod |
| `FORMAL_DEVELOPMENT.md` | Update Definition 1.8.4 |
| `flowbook/kernel/tests/test_exec_restore_magic.py` | Update tests |
| `flowbook/kernel/tests/test_forward_dependency.py` | Update tests |

---

## Verification

1. **Unit tests**: Run `pytest flowbook/kernel/tests/` - all tests should pass
2. **Memory check**: Use `%memory` magic to verify checkpoint count is halved
3. **EXEC-RESTORE test**:
   - Execute cells A, B in order
   - Execute A again (out of order) - B becomes contaminated
   - Run `%exec_restore` on B - should succeed via reconstruction
4. **Backward mutation test**: Verify backward mutation detection still works (uses pre checkpoint + live diff)
5. **Staleness test**: Verify staleness propagation still works

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Non-deterministic cells (random, time) | Document limitation. Fresh predecessor requirement means inputs are fixed. User can set seeds. |
| Side effects (file writes, network) | Document limitation. Most data science cells are pure. Side effects would be re-executed. |
| Reconstruction performance | Only happens for EXEC-RESTORE (rare). Can optionally cache reconstructed checkpoints. |
| Source code not stored for old records | Handle gracefully - fail with clear error message. |

---

## Implementation Order

1. Step 1-3: Add source storage (backwards compatible, no behavior change)
2. Step 5: Add `from_live_namespace()` method
3. Step 6: Add reconstruction method
4. Step 7-8: Update EXEC-RESTORE to use reconstruction
5. Step 4: Remove post checkpoint creation (the breaking change)
6. Step 9-10: Update docs and tests
