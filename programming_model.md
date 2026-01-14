
#### FlowBook Programming Model

**Basic Model**

* Cells containing Python code.
* Each cell is either UpToDate, Stale, or Unexecuted.
* At kernel start, all cells are Unexecuted (no execution state).
* A cell becomes UpToDate upon successful execution.
* A cell becomes Stale when variables it read have changed since its last execution.
* Users can execute cells in any order, but SDC tracks whether the resulting state is reproducible.
* Global variables contain "flowable" values.

**Reproducibility Guarantee**: If a notebook has no errors and no stale cells, then the cell outputs are consistent with a run of all cells in sequential order in a new kernel.

**Flowable Values:** values that can flow from one cell to the next

| Category | Types |
| :---- | :---- |
| Immutable Atomics | None, True, False, int, float, complex, str, bytes, range, datetime.date, datetime.time, datetime.datetime, datetime.timedelta, decimal.Decimal |
| NumPy | np.ndarray (all dtypes, including object dtype if elements are flowable), np.generic (all NumPy scalar types), np.matrix (deprecated) |
| Pandas | pd.DataFrame (if all columns are flowable), pd.Series (if all elements are flowable), pd.Index (all types), pd.Timestamp, pd.Timedelta, pd.Period, pd.NA |
| Standard Containers | list, tuple, dict, set, frozenset, collections.deque, collections.OrderedDict, collections.defaultdict, collections.Counter |
| Functions & Classes | types.FunctionType (user-defined functions, including closures with mutable defaults), types.LambdaType, types.MethodType (bound methods, if instance is flowable), type (class objects themselves, but see Class Attributes limitation below) |
| ML Models (Special-Cased) | Keras Sequential, Functional, custom Model subclasses that have been "built", nn.Module subclasses (only if no uninitialized lazy modules), CatBoost Pool **[These are hardwired special cases with optimized handlers]** |
| ML Models (Via Pickle) | sklearn models (all), XGBoost models, LightGBM models, statsmodels models **[These work via the generic \_\_reduce\_\_ path]** |
| SciPy | scipy.sparse matrices (all types) **[Via pickle]** |
| cuDF | cudf.DataFrame, cudf.Series, cudf.Index, cudf.pandas proxies **[GPU -> CPU transfer during checkpoint]** |
| User-Defined Objects | Objects with \_\_dict\_\_ (if all attributes are flowable), Objects with \_\_slots\_\_ (if all slots are flowable), Objects with custom \_\_deepcopy\_\_ (if it succeeds) |

**Non-Flowable Values:**

matplotlib.\* (figures, axes, artists, backends)

Generators & Coroutines
* types.GeneratorType
* types.CoroutineType
* types.AsyncGeneratorType

Code & Runtime Objects
* types.CodeType
* types.FrameType
* types.TracebackType

Built-in Functions & Descriptors
* types.BuiltinFunctionType
* types.BuiltinMethodType
* types.MethodWrapperType
* types.WrapperDescriptorType
* types.MethodDescriptorType
* types.ClassMethodDescriptorType
* types.GetSetDescriptorType
* types.MemberDescriptorType

I/O & Network
* io.IOBase (file handles, streams)
* socket.socket
* ssl.SSLSocket, ssl.SSLContext

Database Connections
* sqlite3.Connection, sqlite3.Cursor
* Database driver connections (psycopg2, pymysql, etc.)

Threading & Multiprocessing Primitives
* threading.Lock, RLock, Condition, Semaphore, BoundedSemaphore, Event, Barrier, Thread, Timer
* multiprocessing.Lock, RLock, Condition, Semaphore, BoundedSemaphore, Event, Barrier, Queue, JoinableQueue, Pool, Process

Asyncio Primitives
* asyncio.Lock, asyncio.Event, asyncio.Condition, asyncio.Semaphore, asyncio.BoundedSemaphore
* asyncio.Queue, asyncio.PriorityQueue, asyncio.LifoQueue

Weak References
* weakref.ref, weakref.proxy
* weakref.WeakSet, weakref.WeakKeyDictionary, weakref.WeakValueDictionary

Memory & Low-Level
* memoryview objects
* ctypes pointers and arrays

Unbuilt/Uninitialized Models
* Keras models that are not built (architecture not frozen)
* PyTorch models with uninitialized LazyLinear, LazyConv\*, etc.

**Important Model Features that Differ From Jupyter Notebooks and Python**

| Feature | Rationale |
| :---- | :---- |
| Only variables containing flowable values are preserved at the end of cell execution | If we can't copy them, we can't checkpoint them. **Not a problem:** isolate work on these objects to single cells. |
| Backward Mutation Detection: A cell may not modify any variable or DataFrame column read by an earlier cell | This would violate our reproducibility requirement. Detected as a violation with rollback. |
| Forward Dependency Detection: A cell may not read a variable written by a later cell that has already executed | Reading "future" state that wouldn't exist in top-to-bottom order breaks reproducibility. Detected as a violation. |
| Column-level tracking for DataFrames | Modifying `df['price']` does NOT conflict with reading `df['quantity']`. Finer granularity reduces false positives. |
| Structural attribute tracking (optional) | Reading `.columns`, `.shape`, or `len(df)` can be tracked. Adding rows/columns triggers warnings or violations depending on mode (OFF/WARN/ENFORCE). |
| Internal library state not stored in Python variables is unstable | Example: `np.random.seed()` sets C-level state not checkpointed. Use `rng = np.random.RandomState(42)` instead (the `rng` object IS flowable). |
| C-extension internal object state is unstable | Object state managed by C code may not be captured by Python-level checkpointing. |
| Object ids are unstable | Deep copy creates new objects with new ids. Do not rely on `id()` across cell executions. |
| Circular references are handled correctly | The memo mechanism prevents infinite loops during deep copy. |
| Mutable function default arguments are copied | Functions with mutable defaults like `def f(x=[]): ...` have those defaults deep-copied, preventing cross-checkpoint contamination. |
| The set of loaded modules is monotonically increasing but unstable | Modules may be loaded between executions of a cell. |
| File contents are unstable | Files can change between cell executions. File modification tracking is NOT implemented. Recommendation: load files into variables at notebook start. |
| Class attributes (class-level variables) are unstable | Class-level attributes are NOT checkpointed or restored. Only instance attributes (in `__dict__`) are preserved. Use instance attributes instead of class variables. |
| Large object-dtype columns are expensive | Object-dtype DataFrame columns require O(n) element-wise deep copy. Consider using specialized dtypes (Int64, string, datetime64). |
| Diff truncation warnings | Very large data structures may have truncated diffs, potentially causing incomplete change tracking. |

**Error Handling**

* When a cell execution raises an exception, the namespace is rolled back to its pre-execution checkpoint state.
* SDC violations also trigger rollback to the pre-execution state.

**Corner Cases**

| Case | Behavior |
| :---- | :---- |
| Empty cells | No-op, no state change |
| Cells with only comments | No-op, no state change |
| Cell creates then modifies same variable | Not a backward mutation (variable didn't exist before this cell) |
| Cell reads its own writes | Not tracked as a read-before-write |
| Nested DataFrames (e.g., dict of DataFrames) | Column tracking applies to all DataFrames found via namespace traversal |
| DataFrame views vs copies | With Copy-on-Write enabled (default), writing to a view triggers a copy first, so the original is NOT modified. CoW ensures correctness. |
| NumPy array views | Views tracked via `.base` attribute for alias detection |

**Alias Detection**

Variables that share internal references (deep aliases) are tracked:
* If `a["b"]` and `c["b"]` point to the same object, modifying `a["b"]` also changes `c`
* If `df1` and `df2` share a column's underlying array, modifications propagate
* Alias detection uses precomputed indexes for O(accessed + aliases) lookup

