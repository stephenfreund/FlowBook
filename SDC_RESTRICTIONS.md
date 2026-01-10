# SDC Programming Model Restrictions

This document describes the restrictions and limitations of the Sequential Dataflow Consistency (SDC) enforcement system in FlowBook.

## 1. Checkpointable Objects Only

SDC can only track changes to objects it can checkpoint. The following **cannot be checkpointed**:

| Type | Reason |
|------|--------|
| Modules | Cannot be deepcopied |
| Generators/Coroutines | Maintain internal execution state |
| File handles/IO objects | System resources |
| Sockets/network connections | External state |
| Matplotlib figures | Circular refs, C-level resources |
| Thread/Lock objects | Not serializable |

## 2. Class Variables Not Restored

This is a critical limitation. Only **instance attributes** are checkpointed:

```python
class Counter:
    count = 0  # CLASS VARIABLE - NOT restored!

cp.save('before', user_ns)
Counter.count = 100
cp.restore('before', user_ns)
# Counter.count is STILL 100 (not 0!)
```

**Workaround:** Use instance attributes instead of class variables.

## 3. Variable-Level Tracking Only for Namespace Access

SDC tracks reads/writes at the **namespace level** (variable assignment). It does **not** track:

- **Attribute assignments**: `obj.attr = value` is not tracked as a write to `obj`
- **Container mutations**: `list.append()`, `dict.update()` are not tracked as writes
- **In-place modifications**: `x += 1` where `x` is inside a container

SDC relies on **checkpoint diffing** to detect these changes, not the tracking itself.

## 4. Column Tracking via Monkey-Patching

Column-level tracking only works for pandas operations that go through **patched methods**:

```python
# TRACKED - uses __getitem__
df['col']
df.loc[:, 'col']
df.iloc[:, 0]

# NOT TRACKED as column read - direct array access
df.values[:, 0]
df.to_numpy()[:, 0]
```

## 5. Deep Alias Detection Skips Singletons

Alias tracking skips "singleton" types to avoid false positives:

- `type` objects (classes)
- `FunctionType`, `BuiltinFunctionType`
- `ModuleType`
- Method descriptors

This means if you store a class object in multiple variables, modifications through one won't be detected as affecting the other (by design—class objects are shared singletons).

## 6. External Side Effects Not Tracked

SDC **cannot** track:

- File I/O (`open()`, `read()`, `write()`)
- Network requests
- Database operations
- Random number generation (`np.random.seed()` state)
- Environment variables
- Global state in C extensions

```python
# Cell A
with open('data.txt', 'w') as f:
    f.write('hello')

# Cell B (earlier in notebook)
with open('data.txt') as f:
    data = f.read()  # SDC can't know this depends on Cell A
```

## 7. ML Model Restrictions

### Keras

- Must be **built** (architecture frozen)
- Distribute strategy context lost after restore
- Callbacks are separate objects (not captured)

### PyTorch

- Lazy modules must be **initialized**
- JIT/TorchScript models not supported
- Quantized models not supported
- DataParallel wrappers not supported
- Gradients (`.grad`) not preserved

## 8. Dynamic Code Execution

Static analysis cannot track dependencies through:

```python
eval("x + y")       # What does this read?
exec(code_string)   # Unknown dependencies
getattr(obj, name)  # Dynamic attribute access
```

## 9. Callbacks and Higher-Order Functions

When functions/objects are passed to library code, their internal accesses are not tracked:

```python
# SDC sees: read 'df', write 'result'
# Does NOT see what columns 'custom_func' accesses
result = df.apply(custom_func)

# Library may call your function with partial data
df.groupby('key').apply(my_aggregator)
```

## 10. Structural Tracking is Opt-In

Structural changes (`.columns`, `.shape`, `len()`) are only tracked when:
- Structural tracking mode is `WARN` or `ENFORCE`
- The attribute access goes through the patched properties

```python
# TRACKED
len(df)
df.shape
df.columns

# NOT TRACKED as structural
df.values.shape  # Goes through numpy
```

---

## Summary: What SDC Can Reliably Track

| Feature | Coverage |
|---------|----------|
| Variable creation/reassignment | Full |
| DataFrame column reads via `[]`, `.loc`, `.iloc` | Full |
| DataFrame column writes via `[]` | Full |
| Structural reads (shape, columns, len) | Opt-in |
| Deep aliasing across containers | Full (for mutable types) |
| ML model weights (Keras, PyTorch) | Full (with restrictions) |
| Arbitrary object mutations | Via diff only |

## Key Insight: Two Complementary Mechanisms

SDC uses **two complementary mechanisms**:

1. **Tracking** - Records what a cell accessed (may miss some)
2. **Diffing** - Detects what actually changed (always correct)

The diff catches mutations that tracking misses, so SDC is **sound** (won't miss real violations) but may have **false positives** for untracked accesses.

## Thread Safety

SDC is **NOT thread-safe**. Concurrent cell executions or modifications to the namespace during checkpoint operations will produce undefined behavior.
