# Plan: Class Attribute Tracking in FlowBook

## Overview

Extend FlowBook's checkpointing and diffing systems to track class-level attributes for user-defined classes, enabling full state tracking for notebooks that use class variables.

**Updated for reorganized codebase (flowbook/ structure).**

---

## Part 1: Scope and Definitions

### What We're Tracking

**In scope:**

- Data attributes on classes defined in the notebook (`__module__ == '__main__'`)
- Mutable class-level state (lists, dicts, custom objects as class attributes)
- Class-level configuration values

**Out of scope:**

- Classes from imported modules (already excluded via module filtering)
- Methods, classmethods, staticmethods, properties (code, not state)
- Dunder attributes (`__doc__`, `__module__`, etc.)
- Metaclass attributes

### Key Invariant

> Class attribute tracking must maintain the same **isomorphic pointer structure** guarantee as instance attribute tracking.

If `MyClass.data` and `my_instance.shared` reference the same object, the checkpoint must preserve that relationship.

---

## Part 2: Changes to `flowbook/kernel/checkpoint.py`

### 2.1 New Helper: Identify User-Defined Classes

Add to the helpers section (after line ~50):

```python
def is_user_defined_class(obj: Any) -> bool:
    """
    Check if obj is a class defined in the notebook (not imported).

    Classes defined in notebook cells have __module__ == '__main__'
    or sometimes the cell's generated module name.
    """
    if not isinstance(obj, type):
        return False
    module = getattr(obj, '__module__', None)
    if module is None:
        return False
    return module == '__main__' or module.startswith('__main__')
```

### 2.2 New Helper: Extract Class Data Attributes

```python
def get_class_data_attributes(cls: type) -> Dict[str, Any]:
    """
    Extract data attributes from a class, excluding methods and descriptors.

    Only looks at cls.__dict__ (not inherited attributes) because:
    1. Parent classes are separately checkpointed
    2. Pointer structure tracking handles cross-references
    """
    data_attrs = {}

    for name, value in cls.__dict__.items():
        # Skip private/dunder
        if name.startswith('_'):
            continue
        # Skip methods and functions
        if callable(value) and not isinstance(value, (list, dict, set)):
            continue
        # Skip descriptors
        if isinstance(value, (classmethod, staticmethod, property)):
            continue
        # Skip if it's a function (method not yet bound)
        if isinstance(value, types.FunctionType):
            continue

        data_attrs[name] = value

    return data_attrs
```

### 2.3 Modify `Checkpoint` Dataclass

Update the Checkpoint class (~line 87):

```python
@dataclass
class Checkpoint:
    name: str
    user_ns: Dict[str, Any]
    reverse_memo: Dict[int, Any]
    class_attributes: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # NEW

    # Existing fields for deep alias detection
    _reachable_ids: Dict[str, Set[int]] = field(default_factory=dict, repr=False)
    _id_to_vars: Dict[int, Set[str]] = field(default_factory=dict, repr=False)
    _id_to_paths: Dict[int, Dict[str, str]] = field(default_factory=dict, repr=False)
    _alias_index_built: bool = field(default=False, repr=False)
```

### 2.4 Modify `Checkpoints.save()`

Update the save method to capture class attributes. Key integration points:

```python
def save(
    self,
    name: str,
    user_ns: Dict[str, Any],
    sanity_check: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Save a checkpoint of the user namespace including class attributes."""
    from flowbook.kernel.deepcopy import deepcopy  # Use our custom deepcopy

    memo: Dict[int, Any] = {}
    cp: Dict[str, Any] = {}
    class_attrs: Dict[str, Dict[str, Any]] = {}  # NEW
    saved: Dict[str, Any] = {}
    removed: Dict[str, Any] = {}

    # ... existing variable filtering logic ...

    # Checkpoint instance/value state (existing logic)
    for k, v in checkpointable_values.items():
        try:
            cp[k] = deepcopy(v, memo)  # Use flowbook's deepcopy
            saved[k] = get_type_model(v)
        except Exception:
            removed[k] = get_type_model(v)

    # NEW: Checkpoint class attributes
    for var_name, obj in checkpointable_values.items():
        if is_user_defined_class(obj):
            cls_data = get_class_data_attributes(obj)
            if cls_data:
                try:
                    # Use same memo to preserve cross-references!
                    class_attrs[var_name] = {
                        attr_name: deepcopy(attr_val, memo)
                        for attr_name, attr_val in cls_data.items()
                        if _checkpointable_value(attr_val)
                    }
                except Exception as e:
                    log(f"Failed to checkpoint class attributes for {var_name}: {e}")

    # Build reverse memo for alias tracking
    reverse_memo = {id(v): k for k, v in memo.items()}

    checkpoint = Checkpoint(
        name=name,
        user_ns=cp,
        reverse_memo=reverse_memo,
        class_attributes=class_attrs,  # NEW
    )

    self._checkpoints[name] = checkpoint
    return saved, removed
```

### 2.5 Modify `Checkpoints.restore()`

```python
def restore(self, name: str, user_ns: Dict[str, Any]) -> None:
    """Restore namespace from checkpoint including class attributes."""
    from flowbook.kernel.deepcopy import deepcopy

    cp = self._checkpoints[name]
    checkpointable_vars = _checkpointable_vars(user_ns)

    # Clear current namespace
    for k in checkpointable_vars.keys():
        del user_ns[k]

    # Restore namespace values
    memo: Dict[int, Any] = {}
    for k, v in cp.user_ns.items():
        user_ns[k] = deepcopy(v, memo)

    # NEW: Restore class attributes
    for class_name, attrs in cp.class_attributes.items():
        if class_name in user_ns and is_user_defined_class(user_ns[class_name]):
            cls = user_ns[class_name]

            # Clear existing data attributes first
            for attr_name in list(get_class_data_attributes(cls).keys()):
                try:
                    delattr(cls, attr_name)
                except AttributeError:
                    pass

            # Restore saved attributes using same memo for cross-references
            for attr_name, value in attrs.items():
                setattr(cls, attr_name, deepcopy(value, memo))
```

### 2.6 Update Deep Alias Detection

The `_build_alias_index()` method (~line 300) needs to also traverse class attributes:

```python
def _build_alias_index(self) -> None:
    """Build the deep alias index for this checkpoint."""
    if self._alias_index_built:
        return

    # ... existing initialization ...

    # Traverse all variables in namespace
    for var_name, value in self.user_ns.items():
        self._traverse_for_aliases(var_name, value, var_name, seen=set())

    # NEW: Also traverse class attributes
    for class_name, attrs in self.class_attributes.items():
        for attr_name, attr_value in attrs.items():
            path = f"{class_name}.{attr_name}"
            self._traverse_for_aliases(class_name, attr_value, path, seen=set())

    self._alias_index_built = True
```

---

## Part 3: Changes to `flowbook/kernel/diff.py`

### 3.1 Add User Class Detection to Dispatch

Add after line ~440 (the dispatch table setup):

```python
def _is_user_defined_class(obj: Any) -> bool:
    """Check if obj is a user-defined class (for diff dispatch)."""
    if not isinstance(obj, type):
        return False
    module = getattr(obj, '__module__', None)
    return module is not None and (module == '__main__' or module.startswith('__main__'))
```

### 3.2 New Method: `_compare_user_class()`

Add to the Diff class after `_compare_callable()`:

```python
def _compare_user_class(self, cls_a: type, cls_b: type, path: str) -> Optional[DiffNode]:
    """
    Compare two user-defined classes.

    Compares class data attributes (not methods/descriptors).
    Uses same pointer tracking as other comparisons.
    """
    from flowbook.kernel.checkpoint import get_class_data_attributes

    children: Dict[str, DiffNode] = {}

    # Get data attributes from both classes
    attrs_a = get_class_data_attributes(cls_a)
    attrs_b = get_class_data_attributes(cls_b)

    # In LEQ mode, extra attributes in b are allowed
    if not self.use_leq:
        # Check for attributes only in a
        only_in_a = set(attrs_a.keys()) - set(attrs_b.keys())
        for attr in sorted(only_in_a):
            children[f".{attr}"] = ValueComparison(
                status="different",
                value1=attrs_a[attr],
                value2=None,
                message=f"Class attribute removed at {path}.{attr}",
            )

    # Check for attributes only in b
    only_in_b = set(attrs_b.keys()) - set(attrs_a.keys())
    if not self.use_leq:  # In LEQ mode, additions are allowed
        for attr in sorted(only_in_b):
            children[f".{attr}"] = ValueComparison(
                status="different",
                value1=None,
                value2=attrs_b[attr],
                message=f"Class attribute added at {path}.{attr}",
            )

    # Compare common attributes
    common_attrs = set(attrs_a.keys()) & set(attrs_b.keys())
    for attr in sorted(common_attrs):
        attr_diff = self._compare_values(
            attrs_a[attr], attrs_b[attr], f"{path}.{attr}"
        )
        if attr_diff:
            children[f".{attr}"] = attr_diff

    if children:
        return CompoundDiff(
            source_type="class",
            children=children,
            truncated=False
        )
    return None
```

### 3.3 Update `_compare_values()` Dispatch

Add to the isinstance fallback chain in `_compare_values()` (~line 1200):

```python
# After the callable check, before the final object fallback:
elif _is_user_defined_class(val_a) and _is_user_defined_class(val_b):
    # User-defined classes - compare class attributes
    result = self._compare_user_class(val_a, val_b, path)
```

### 3.4 Update `Checkpoint.diff()` Static Method

Update the static `diff()` method to include class attributes:

```python
@staticmethod
def diff(
    a: "Checkpoint",
    b: "Checkpoint",
    keys_to_include: Optional[Set[str]] = None,
    use_leq: bool = False,
    column_rbw: Optional[Dict[str, Set[str]]] = None,
    structural_reads: Optional[Dict[str, Set[str]]] = None,
    structural_mode: StructuralTrackingMode = StructuralTrackingMode.OFF,
) -> DiffResult:
    """Compare two checkpoints including class attributes."""
    from flowbook.kernel.diff import Diff

    differ = Diff(
        strict=False,
        report_close=False,
        atol=1e-6,
        rtol=1e-5,
        use_leq=use_leq,
        column_rbw=column_rbw or {},
        structural_reads=structural_reads or {},
        structural_mode=structural_mode,
    )

    # Diff namespace values
    ns_diff = differ.diff(a.user_ns, b.user_ns, keys_to_include)

    # NEW: Class attribute comparison is automatic via _compare_user_class
    # when comparing class objects in the namespace

    return ns_diff
```

---

## Part 4: Changes to `flowbook/kernel/deepcopy.py`

### 4.1 Handle User-Defined Classes in Deepcopy

The custom deepcopy module needs to properly handle user-defined classes:

```python
def _deepcopy_user_class(cls: type, memo: Dict[int, Any]) -> type:
    """
    Deep copy a user-defined class including its data attributes.

    Note: This creates a NEW class object with copied attributes.
    The class itself is copied (not just returned as-is) to enable
    proper checkpoint isolation.
    """
    obj_id = id(cls)
    if obj_id in memo:
        return memo[obj_id]

    # For classes, we typically want to preserve identity
    # (the same class object) but copy the mutable attributes.
    # Since checkpoint.restore() handles attribute restoration,
    # we just return the class itself here.
    memo[obj_id] = cls
    return cls

# Add to dispatch table if full class copying is needed:
# d[type] = _deepcopy_user_class  # Only if we want to copy class objects
```

**Note:** For most use cases, classes should preserve identity (same class object). The class _attributes_ are separately deep copied in checkpoint.save(). This maintains Python's expectation that `isinstance(obj, MyClass)` works correctly across checkpoint restore.

---

## Part 5: Changes to `flowbook/sdc_kernel/sdc_enforcer.py`

### 5.1 Include Class Attributes in Change Detection

The SDC enforcer's `_expand_with_deep_aliases()` function needs no changes if checkpoint alias detection includes class attributes (done in Part 2.6).

### 5.2 Update TrackingData for Class Attributes (Optional Enhancement)

For full SDC enforcement, the tracking system could track reads/writes to class attributes. This would require:

1. Wrapping class attribute access in `TrackingDict`
2. Reporting class attribute access in `TrackingData`

**Recommendation:** Start without SDC tracking of class attribute access. The checkpoint/diff integration provides the foundation. SDC integration can be a follow-up.

---

## Part 6: Edge Cases and Handling

### 6.1 Class Redefinition

**Scenario:** User re-executes cell that defines a class.

```python
# Cell 1 (execution 1)
class Config:
    debug = True

# Cell 2
Config.debug = False

# Cell 1 (execution 2) - NEW class object created
class Config:
    debug = True
```

**Handling:**

- On restore, match by class name in namespace, not object identity
- Restore attributes to whichever class currently has that name
- If class was deleted, skip (no error)

### 6.2 Inheritance

**Scenario:** Subclass inherits class attribute from parent.

```python
class Base:
    config = {'a': 1}

class Child(Base):
    pass

Child.config['b'] = 2  # Modifies Base.config!
```

**Handling:**

- `get_class_data_attributes()` only looks at `cls.__dict__`
- Parent class (`Base`) is separately checkpointed
- Changes via subclass appear in parent's diff
- Pointer structure tracking links them correctly

### 6.3 Metaclasses

**Scenario:** Class uses custom metaclass with state.

```python
class SingletonMeta(type):
    _instances = {}

class MyClass(metaclass=SingletonMeta):
    pass
```

**Handling:**

- Metaclass attributes are out of scope for v1
- Metaclass state won't be tracked unless the metaclass itself is in namespace
- Document as known limitation

### 6.4 Class Attribute References Instance (Circular)

```python
class Node:
    root = None

Node.root = Node()
Node.root.parent = Node.root
```

**Handling:**

- The memo mechanism handles circular references
- Same memo used for namespace and class attributes
- No special handling needed

### 6.5 Opaque Objects as Class Attributes

**Scenario:** Keras model stored as class attribute.

```python
class ModelRegistry:
    current_model = keras.Sequential([...])
```

**Handling:**

- The opaque handler pattern in `flowbook/kernel/opaque.py` applies
- `_checkpointable_value()` should include opaque-handled objects
- Keras/PyTorch models as class attributes work automatically

---

## Part 7: Testing Plan

### 7.1 Unit Tests for `checkpoint.py`

Location: `flowbook/kernel/test_checkpoint.py`

```python
def test_class_attribute_checkpoint_save():
    """Class attributes are saved in checkpoint."""
    class Config:
        debug = True
        values = [1, 2, 3]

    user_ns = {'Config': Config}
    checkpoints = Checkpoints()
    checkpoints.save('test', user_ns)

    cp = checkpoints.get('test')
    assert 'Config' in cp.class_attributes
    assert cp.class_attributes['Config']['debug'] == True
    assert cp.class_attributes['Config']['values'] == [1, 2, 3]


def test_class_attribute_checkpoint_restore():
    """Class attributes are restored from checkpoint."""
    class Config:
        debug = True

    user_ns = {'Config': Config}
    checkpoints = Checkpoints()
    checkpoints.save('test', user_ns)

    Config.debug = False  # Mutate
    checkpoints.restore('test', user_ns)

    assert Config.debug == True  # Restored


def test_class_attribute_cross_reference():
    """Cross-references between class and instance are preserved."""
    class Model:
        shared = [1, 2, 3]

    m = Model()
    m.data = Model.shared  # Same object

    user_ns = {'Model': Model, 'm': m}
    checkpoints = Checkpoints()
    checkpoints.save('test', user_ns)

    cp = checkpoints.get('test')
    # Verify aliasing is preserved in checkpoint
    restored_shared = cp.class_attributes['Model']['shared']
    restored_data = cp.user_ns['m'].data
    assert restored_shared is restored_data


def test_class_attribute_deep_alias_detection():
    """Deep alias detection includes class attributes."""
    class Registry:
        items = {'key': [1, 2, 3]}

    other = {'ref': Registry.items['key']}  # Shares nested object

    user_ns = {'Registry': Registry, 'other': other}
    checkpoints = Checkpoints()
    checkpoints.save('test', user_ns)

    cp = checkpoints.get('test')
    aliases = cp.get_aliases_for_vars({'other'})
    # Should include Registry because they share nested data
    assert 'Registry' in aliases
```

### 7.2 Unit Tests for `diff.py`

Location: `flowbook/kernel/test_diff.py`

```python
def test_class_attribute_diff_equal():
    """Classes with same attributes show no diff."""
    class A:
        x = 1
    class B:
        x = 1

    differ = Diff()
    result = differ._compare_user_class(A, B, 'cls')
    assert result is None


def test_class_attribute_diff_different():
    """Classes with different attributes show diff."""
    class A:
        x = 1
    class B:
        x = 2

    differ = Diff()
    result = differ._compare_user_class(A, B, 'cls')
    assert result is not None
    assert '.x' in result.children


def test_class_attribute_diff_leq_mode():
    """In LEQ mode, extra class attributes are allowed."""
    class A:
        x = 1
    class B:
        x = 1
        y = 2  # Extra attribute

    differ = Diff(use_leq=True)
    result = differ._compare_user_class(A, B, 'cls')
    assert result is None  # No diff - y is allowed in LEQ mode


def test_class_attribute_pointer_structure():
    """Pointer structure in class attributes is checked."""
    shared = [1, 2, 3]

    class A:
        x = shared
        y = shared  # Same object

    class B:
        x = [1, 2, 3]
        y = [1, 2, 3]  # Different objects!

    differ = Diff()
    result = differ._compare_user_class(A, B, 'cls')
    assert result is not None
    # Should detect pointer structure mismatch
```

### 7.3 Integration Tests

Location: `flowbook/kernel/test_class_attributes_integration.py`

```python
def test_full_checkpoint_with_class_mutation():
    """End-to-end test of checkpoint/restore with class mutation."""
    user_ns = {}
    exec("""
class Counter:
    count = 0
    history = []

counter = Counter()
""", user_ns)

    checkpoints = Checkpoints()
    checkpoints.save('before', user_ns)

    # Mutate class attributes
    exec("""
Counter.count = 5
Counter.history.append('incremented')
""", user_ns)

    # Verify mutation
    assert user_ns['Counter'].count == 5

    # Restore
    checkpoints.restore('before', user_ns)

    # Verify restoration
    assert user_ns['Counter'].count == 0
    assert user_ns['Counter'].history == []


def test_sdc_backward_mutation_class_attribute():
    """SDC detects backward mutation via class attribute."""
    # This test validates SDC integration (Phase 2)
    pass  # Implement after SDC tracking integration
```

---

## Part 8: Implementation Order

### Phase 1: Core Infrastructure (checkpoint.py)

1. Add `is_user_defined_class()` helper
2. Add `get_class_data_attributes()` helper
3. Update `Checkpoint` dataclass with `class_attributes` field
4. Update `Checkpoints.save()` to capture class attributes
5. Update `Checkpoints.restore()` to restore class attributes
6. Update `_build_alias_index()` to include class attributes

### Phase 2: Diff Integration (diff.py)

7. Add `_is_user_defined_class()` helper
8. Add `_compare_user_class()` method
9. Update `_compare_values()` dispatch to handle user classes
10. Add unit tests for class attribute diffing

### Phase 3: Testing and Validation

11. Add unit tests for checkpoint save/restore
12. Add integration tests for end-to-end flows
13. Add tests for edge cases (inheritance, circular refs)
14. Verify existing tests still pass

### Phase 4: SDC Integration (Optional Enhancement)

15. Add class attribute access tracking to TrackingDict
16. Update TrackingData to include class attribute reads/writes
17. Update SDC enforcer to detect class attribute mutations
18. Add SDC-specific tests

---

## Part 9: Documentation Updates

### 9.1 Update `flowbook-execution-guarantees-report.md`

Change section 2.10 from "Class Variables Are Not Tracked" to:

> **2.10 Class Variables Are Tracked for User-Defined Classes**
>
> FlowBook tracks class-level data attributes for classes defined in the notebook (`__module__ == '__main__'`). This includes:
>
> - Mutable class attributes (lists, dicts, custom objects)
> - Class-level configuration values
>
> **Not tracked:**
>
> - Classes from imported modules (already excluded)
> - Methods, class methods, static methods, properties
> - Dunder attributes (`__doc__`, `__module__`, etc.)
> - Metaclass attributes
>
> Class attributes participate in the same pointer structure tracking as instance attributes. If `MyClass.data` and `instance.shared` reference the same object, checkpoints preserve that relationship.

### 9.2 User-Facing Summary

**Before:** "FlowBook tracks what you assign to notebook variables, not side effects made through class attributes."

**After:** "FlowBook tracks notebook variables and class attributes for classes you define in the notebook."

---

## Part 10: Migration and Compatibility

### 10.1 Checkpoint Format Change

The `Checkpoint` dataclass gains a new field with default:

```python
class_attributes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
```

Old checkpoints without `class_attributes` work (default to empty dict).

### 10.2 Diff Result Format

Class attribute diffs appear with `.` prefix:

```
MyClass.x    -> Class attribute diff
x            -> Regular variable diff
```

This matches Python attribute access syntax and is intuitive for users.
