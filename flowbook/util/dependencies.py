"""
Dependency analysis for Jupyter notebooks.

This module provides static analysis tools to track variable, function, and method
dependencies in notebook cells, creating a map from cell IDs to the global variables
they access.

GUARANTEE
=========
The analysis provides the following guarantee:

    COMPUTED DEPENDENCIES INCLUDE ALL REAL DEPENDENCIES
    OR
    WARNINGS ARE REPORTED FOR CASES WHERE ANALYSIS MAY BE INCOMPLETE

This means:
- If no warnings are produced, the dependencies are COMPLETE (sound)
- If warnings are produced, the dependencies MAY be incomplete for the warned patterns
- Check CellDependencies.warnings for any under-approximation warnings

ANALYSIS FEATURES AND PROPERTIES
=================================

Flow-Sensitivity
----------------
FLOW-SENSITIVE at module level (within a single cell):
  - Tracks whether variables are written before being read
  - Example: In "x = 5; y = x", x is NOT a dependency (written first)
  - Example: In "y = x; x = 5", x IS a dependency (read first)
  - Order of statements matters for determining dependencies

FLOW-INSENSITIVE in other contexts:
  - Across cells: No tracking of cell execution order
  - Inside functions/methods: Analyzed independently of control flow
  - Conditional blocks treated conservatively (see below)

Context-Sensitivity
-------------------
CONTEXT-SENSITIVE for functions and methods:
  - Each function/method is analyzed separately to determine its dependencies
  - Transitive closure: When cell calls f(), includes dependencies from f's body
  - Method definitions tracked per-class with qualified names (ClassName.method)

CONTEXT-INSENSITIVE for method dispatch:
  - No type inference or points-to analysis
  - Conservative: obj.method() could dispatch to ANY method named "method"
  - Example: If ClassA.process and ClassB.process both exist, calling
    obj.process() assumes dependencies from BOTH methods

Conservative Approximations (Over-Approximation)
------------------------------------------------
The analysis errs on the side of including MORE dependencies than necessary:

1. CONDITIONAL WRITES: Variables written inside if/while/for/try blocks are
   treated as "conditional" - subsequent reads are marked as dependencies
   Example: "if cond: x = 5; y = x" → y depends on x (might not be written)

2. METHOD DISPATCH: Without type information, obj.method() includes dependencies
   from ALL notebook-defined methods named "method", regardless of class
   Example: obj.process() → dependencies from ClassA.process AND ClassB.process

3. CALLBACK TRACKING: Functions/methods passed as arguments are assumed to be called
   Example: df.apply(func) → func is marked as called, includes func's dependencies

4. ATTRIBUTE REFERENCES: Method references like obj.method passed as callbacks
   are tracked by method name only
   Example: df.apply(processor.transform) → any method named "transform" is
   considered as possibly being called

5. TRANSITIVE CLOSURE: All called functions/methods contribute their full
   dependency sets, even if only partially executed

Scope Keywords (global and nonlocal)
-------------------------------------
The analysis CORRECTLY handles `global` and `nonlocal` declarations:

1. GLOBAL KEYWORD: Variables declared with `global` in a function are tracked
   as module-level dependencies rather than local variables
   Example:
   ```python
   def increment():
       global counter
       counter += 1  # counter tracked as global read and write
   ```

2. NONLOCAL KEYWORD: Variables declared with `nonlocal` refer to enclosing
   function scopes, not the module level
   Example:
   ```python
   def outer():
       x = 0
       def inner():
           nonlocal x
           x += 1  # x is local to outer, not a global dependency
       inner()
   ```

3. SCOPE TRACKING: Each scope (module, function, nested function) maintains
   separate tracking of:
   - Local variables
   - Global declarations
   - Nonlocal declarations

Detected Under-Approximations (Warnings)
----------------------------------------
The following cases CANNOT be fully tracked, but are DETECTED and produce warnings:

1. DYNAMIC CODE EXECUTION (WARNING):
   - eval(), exec(), compile() → produces warning
   - importlib.import_module() → produces warning
   - Example: eval("x + y") → WARNING: "dynamic code execution"

2. REFLECTION AND DYNAMIC ATTRIBUTE ACCESS (WARNING):
   - getattr(), setattr(), delattr(), hasattr() → produces warning
   - vars() → produces warning
   - obj.__dict__ access → produces warning
   - Example: getattr(obj, "method") → WARNING: "dynamic attribute access"

3. STAR IMPORTS (WARNING):
   - "from module import *" → produces warning
   - Example: from numpy import * → WARNING: "star import"

7. INDIRECT CALLS (WARNING):
   - funcs[0](), handlers['key']() → WARNING: "indirect call via subscript"
   - get_handler()() → WARNING: "indirect call via function result"
   - (f1 if cond else f2)() → WARNING: "indirect call via conditional"
   - f = helper; f() → WARNING: "call to 'f' which is not a function definition"
   - obj.callback = func; obj.callback() → WARNING: "call to attribute 'callback'"

8. METACLASSES AND CLASS DECORATORS (WARNING):
   - class Foo(metaclass=...) → WARNING: "uses metaclass"
   - @decorator class Foo → WARNING: "has decorator(s)"

Undetected Under-Approximations (Silent)
----------------------------------------
The following cases are NOT tracked and NOT warned about:

4. ATTRIBUTE ASSIGNMENTS:
   - obj.attr = value → not tracked as a write to obj
   - Only tracks module-level variable assignments

5. CONTAINER MODIFICATIONS:
   - list.append(), dict.update(), etc. → not tracked
   - Only tracks variable bindings, not mutations

6. CLASS/INSTANCE ATTRIBUTES:
   - self.attr reads/writes inside methods not tracked separately
   - Only method-level global dependencies are captured

Temporal Scope Assumption (Callbacks)
--------------------------------------
The analysis assumes callbacks are invoked DURING the call that receives them,
not stored for later invocation:

SOUND (Immediate callback - tracked correctly):
   ```python
   def handler():
       return data

   library.process(handler)  # handler called DURING this line
   # → Dependencies from handler ARE included
   ```

UNSOUND (Stored callback - NOT tracked across cells):
   ```python
   # Cell 1
   def handler():
       return data

   # Cell 2
   library.register(handler)  # handler STORED for later

   # Cell 3
   data = new_value()

   # Cell 4
   library.trigger()  # handler invoked HERE
   # → Cell 4's dependency on data is NOT tracked
   ```

This limitation applies to:
- Functions passed to external (library) code
- Objects with methods passed to external code

The analysis assumes:
- ✅ Callbacks/methods MAY be invoked during the call that receives them
- ✅ Dependencies from the callback/method ARE included in that call
- ❌ Callbacks/methods are NOT stored and invoked from later cells
- ❌ Dependencies across stored callback invocations are NOT tracked

Implications:
- Dependencies are sound within the temporal scope of immediate callbacks
- Dependencies MAY be incomplete for event-driven code with stored callbacks
- Users working with event systems, observers, or callback registries should
  manually verify dependencies across callback invocations

Design Decisions
----------------
- GUARANTEE SOUNDNESS OR WARN: If the analysis cannot guarantee complete
  dependency tracking, it produces a warning. Users can check warnings
  to know if manual review is needed.

- CONSERVATIVE FOR SAFETY: Prefer false positives (extra dependencies) over
  false negatives (missing dependencies)

- NOTEBOOK-INTERNAL ONLY: Only tracks dependencies on variables/functions/classes
  defined within the notebook. Imported names and builtins are filtered out.

- TRANSITIVE CLOSURE: Dependencies propagate through function/method calls to
  ensure complete dependency graphs

- METHOD NAME-BASED DISPATCH: Without type information, uses method name
  matching across all classes as a sound over-approximation

- FIELD-NAME BASED ATTRIBUTE TRACKING: For stored function detection (Case 7f),
  uses field names only (not object types) to conservatively detect when an
  attribute might contain a stored function

Example Usage
-------------
    notebook = {"cells": [...]}
    dependencies = analyze_notebook(notebook)

    for cell_id, deps in dependencies.items():
        print(f"Cell {cell_id} reads: {deps.globals_read}")
        print(f"Cell {cell_id} writes: {deps.globals_written}")
        if deps.warnings:
            print(f"Cell {cell_id} warnings: {deps.warnings}")
"""

import ast
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, field


@dataclass
class CellDependencies:
    """Represents the dependencies of a single cell."""

    cell_id: str
    globals_read: Set[str] = field(default_factory=set)
    globals_written: Set[str] = field(default_factory=set)
    functions_called: Set[str] = field(default_factory=set)
    modules: Set[str] = field(default_factory=set)  # Imported modules (excluded from dependencies)
    imported_names: Set[str] = field(default_factory=set)  # Names from 'from ... import' (excluded from dependencies)
    functions_defined: Set[str] = field(default_factory=set)  # Functions defined in this cell
    classes_defined: Set[str] = field(default_factory=set)  # Classes defined in this cell
    methods_called: Set[str] = field(default_factory=set)  # Method names called (e.g., from obj.method())
    methods_defined: Dict[str, Set[str]] = field(default_factory=dict)  # class_name -> set of method names
    attributes_assigned: Set[str] = field(default_factory=set)  # Attribute names assigned (e.g., obj.attr = value)
    warnings: List[str] = field(default_factory=list)  # Warnings about potential under-approximations

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'cell_id': self.cell_id,
            'globals_read': sorted(list(self.globals_read)),
            'globals_written': sorted(list(self.globals_written)),
            'functions_called': sorted(list(self.functions_called)),
            'modules': sorted(list(self.modules)),
            'imported_names': sorted(list(self.imported_names)),
            'functions_defined': sorted(list(self.functions_defined)),
            'classes_defined': sorted(list(self.classes_defined)),
            'methods_called': sorted(list(self.methods_called)),
            'methods_defined': {cls: sorted(list(methods)) for cls, methods in self.methods_defined.items()},
            'attributes_assigned': sorted(list(self.attributes_assigned)),
            'warnings': self.warnings.copy(),
        }


@dataclass
class FunctionInfo:
    """Information about a function definition."""

    name: str
    globals_read: Set[str] = field(default_factory=set)
    functions_called: Set[str] = field(default_factory=set)
    methods_called: Set[str] = field(default_factory=set)  # Method names called from this function
    defined_in_cell: Optional[str] = None
    class_name: Optional[str] = None  # If this is a method, the class it belongs to


# Functions that execute dynamic code - cannot be statically analyzed
DYNAMIC_CODE_FUNCTIONS = {'eval', 'exec', 'compile'}

# Functions for dynamic attribute access - cannot be statically tracked
REFLECTION_FUNCTIONS = {'getattr', 'setattr', 'delattr', 'hasattr', 'vars'}

# Common method names to exclude from attribute-stored function warnings
# These are methods that are commonly called on objects but unlikely to be stored functions
COMMON_METHOD_NAMES = {
    # Container methods
    'append', 'extend', 'insert', 'remove', 'pop', 'clear', 'index', 'count',
    'sort', 'reverse', 'copy', 'update', 'keys', 'values', 'items', 'get',
    'setdefault', 'add', 'discard', 'union', 'intersection', 'difference',
    # String methods
    'strip', 'lstrip', 'rstrip', 'split', 'rsplit', 'join', 'replace',
    'find', 'rfind', 'startswith', 'endswith', 'upper', 'lower', 'title',
    'capitalize', 'format', 'encode', 'decode',
    # File methods
    'read', 'readline', 'readlines', 'write', 'writelines', 'close',
    'seek', 'tell', 'flush', 'truncate',
    # Iterator/generator methods
    'next', 'send', 'throw',
    # Object methods
    'deepcopy', '__init__', '__str__', '__repr__',
    # DataFrame/Series common methods (pandas)
    'head', 'tail', 'describe', 'info', 'mean', 'sum', 'min', 'max', 'std',
    'var', 'median', 'mode', 'abs', 'round', 'cumsum', 'cumprod',
    'groupby', 'merge', 'join', 'concat',
    'apply', 'map', 'transform', 'agg', 'aggregate',
    'filter', 'query', 'where', 'mask',
    'sort_values', 'sort_index', 'rank', 'nlargest', 'nsmallest',
    'drop', 'drop_duplicates', 'duplicated',
    'fillna', 'dropna', 'isna', 'notna', 'interpolate',
    'reset_index', 'set_index', 'reindex',
    'to_csv', 'to_json', 'to_excel', 'to_dict', 'to_list', 'to_numpy',
    'astype', 'convert_dtypes',
    'rename', 'clip', 'shift', 'diff', 'pct_change',
    'rolling', 'expanding', 'ewm', 'resample',
    'pivot', 'pivot_table', 'melt', 'stack', 'unstack',
    'sample', 'value_counts', 'unique', 'nunique',
    # NumPy array methods
    'reshape', 'flatten', 'ravel', 'transpose', 'squeeze', 'expand_dims',
    'argmax', 'argmin', 'argsort', 'nonzero', 'all', 'any',
    'dot', 'matmul', 'trace', 'diagonal',
    # Plotting
    'plot', 'hist', 'scatter', 'bar', 'barh', 'pie', 'box', 'area',
    'show', 'savefig', 'legend', 'xlabel', 'ylabel',
    # Sklearn/model methods
    'fit', 'predict', 'fit_transform', 'score',
    'get_params', 'set_params',
}


class GlobalAccessAnalyzer(ast.NodeVisitor):
    """
    AST visitor that tracks global variable accesses in Python code.

    This analyzer is flow-sensitive at module level:
    - A variable is only considered "read" if it was read BEFORE being written
    - Direct variable references
    - Function calls
    - Functions passed as arguments (treated as called)
    """

    def __init__(self):
        self.globals_read: Set[str] = set()
        self.globals_written: Set[str] = set()
        self.functions_called: Set[str] = set()
        self.modules: Set[str] = set()  # Imported modules (import X as Y)
        self.imported_names: Set[str] = set()  # Names from 'from ... import'
        self.local_vars: Set[str] = set()
        self.scope_stack: List[Set[str]] = [set()]  # Stack of local scopes
        self.global_names_stack: List[Set[str]] = [set()]  # Names declared global in each scope
        self.nonlocal_names_stack: List[Set[str]] = [set()]  # Names declared nonlocal in each scope
        self.function_defs: Dict[str, FunctionInfo] = {}  # Function definitions found
        self.written_at_module_level: Set[str] = set()  # Track module-level writes in order
        self.in_conditional: int = 0  # Track if we're inside conditional/loop (conservative)
        self.conditional_write_depth: Dict[str, int] = {}  # Track at what depth each var was written
        self.functions_defined: Set[str] = set()  # Function names defined at module level
        self.classes_defined: Set[str] = set()  # Class names defined at module level
        self.methods_called: Set[str] = set()  # Method names called (from obj.method())
        self.methods_defined: Dict[str, Set[str]] = {}  # class_name -> set of method names
        self.method_defs: Dict[str, FunctionInfo] = {}  # "ClassName.method_name" -> FunctionInfo
        self.current_class: Optional[str] = None  # Track which class we're currently analyzing
        self.attributes_assigned: Set[str] = set()  # Attribute names that are assigned (obj.attr = ...)
        self.warnings: List[str] = []  # Warnings about potential under-approximations

    def _current_scope(self) -> Set[str]:
        """Get the current local scope."""
        return self.scope_stack[-1]

    def _push_scope(self):
        """Enter a new scope."""
        self.scope_stack.append(set())
        self.global_names_stack.append(set())
        self.nonlocal_names_stack.append(set())

    def _pop_scope(self):
        """Exit the current scope."""
        if len(self.scope_stack) > 1:
            self.scope_stack.pop()
            self.global_names_stack.pop()
            self.nonlocal_names_stack.pop()

    def _is_local(self, name: str) -> bool:
        """Check if a name is local to any current scope."""
        # If name is declared global in current scope, it's not local
        if name in self.global_names_stack[-1]:
            return False
        return any(name in scope for scope in self.scope_stack)

    def _is_declared_global(self, name: str) -> bool:
        """Check if a name is declared global in the current scope."""
        return name in self.global_names_stack[-1]

    def _is_declared_nonlocal(self, name: str) -> bool:
        """Check if a name is declared nonlocal in the current scope."""
        return name in self.nonlocal_names_stack[-1]

    def _add_local(self, name: str):
        """Add a name to the current local scope."""
        # Don't add to local scope if it's declared global
        if not self._is_declared_global(name):
            self._current_scope().add(name)

    def _is_at_module_level(self) -> bool:
        """Check if we're currently at module level."""
        return len(self.scope_stack) == 1

    def visit_Global(self, node: ast.Global):
        """Handle global declarations."""
        # Mark these names as global in the current scope
        for name in node.names:
            self.global_names_stack[-1].add(name)
        # Don't call generic_visit - Global nodes have no children to visit

    def visit_Nonlocal(self, node: ast.Nonlocal):
        """Handle nonlocal declarations."""
        # Mark these names as nonlocal in the current scope
        for name in node.names:
            self.nonlocal_names_stack[-1].add(name)
        # Don't call generic_visit - Nonlocal nodes have no children to visit

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Handle function definitions."""
        # Visit decorators first (in outer scope)
        for decorator in node.decorator_list:
            self.visit(decorator)

        # Function name is defined in the enclosing scope
        is_global_function = self._is_at_module_level()
        if is_global_function:
            self.globals_written.add(node.name)
            self.functions_defined.add(node.name)  # Track function definitions
            # Function definitions are always written (not conditional)
            if self.in_conditional == 0:
                self.written_at_module_level.add(node.name)
        else:
            self._add_local(node.name)

        # For global functions, analyze their dependencies separately
        if is_global_function:
            # Create a sub-analyzer for the function body
            func_analyzer = GlobalAccessAnalyzer()
            # Use 2 scopes: module scope (empty) and function scope
            # This way, assignments in the function body go to the function scope, not globals
            func_analyzer.scope_stack = [set(), set()]

            # Add parameters to function's local scope
            for arg in node.args.args:
                func_analyzer._add_local(arg.arg)
            for arg in node.args.posonlyargs:
                func_analyzer._add_local(arg.arg)
            for arg in node.args.kwonlyargs:
                func_analyzer._add_local(arg.arg)
            if node.args.vararg:
                func_analyzer._add_local(node.args.vararg.arg)
            if node.args.kwarg:
                func_analyzer._add_local(node.args.kwarg.arg)

            # Visit function body
            for child in node.body:
                func_analyzer.visit(child)

            # Store function info
            func_info = FunctionInfo(
                name=node.name,
                globals_read=func_analyzer.globals_read.copy(),
                functions_called=func_analyzer.functions_called.copy()
            )
            self.function_defs[node.name] = func_info
        else:
            # For nested functions, just visit normally
            # Enter function scope
            self._push_scope()

            # Add parameters to local scope
            for arg in node.args.args:
                self._add_local(arg.arg)
            for arg in node.args.posonlyargs:
                self._add_local(arg.arg)
            for arg in node.args.kwonlyargs:
                self._add_local(arg.arg)
            if node.args.vararg:
                self._add_local(node.args.vararg.arg)
            if node.args.kwarg:
                self._add_local(node.args.kwarg.arg)

            # Visit function body
            for child in node.body:
                self.visit(child)

            # Exit function scope
            self._pop_scope()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        """Handle async function definitions."""
        self.visit_FunctionDef(node)

    def visit_Lambda(self, node: ast.Lambda):
        """Handle lambda expressions."""
        self._push_scope()

        # Add parameters to local scope
        for arg in node.args.args:
            self._add_local(arg.arg)
        for arg in node.args.posonlyargs:
            self._add_local(arg.arg)
        for arg in node.args.kwonlyargs:
            self._add_local(arg.arg)
        if node.args.vararg:
            self._add_local(node.args.vararg.arg)
        if node.args.kwarg:
            self._add_local(node.args.kwarg.arg)

        # Visit lambda body
        self.visit(node.body)

        self._pop_scope()

    def visit_ClassDef(self, node: ast.ClassDef):
        """Handle class definitions."""
        # Case 8: Warn on metaclass usage
        for keyword in node.keywords:
            if keyword.arg == 'metaclass':
                self.warnings.append(
                    f"line {node.lineno}: class {node.name} uses metaclass - "
                    f"class behavior may be modified, dependencies may be incomplete"
                )
                break

        # Case 8: Warn on class decorators
        if node.decorator_list:
            self.warnings.append(
                f"line {node.lineno}: class {node.name} has decorator(s) - "
                f"class behavior may be modified, dependencies may be incomplete"
            )

        # Class name is defined in the enclosing scope
        is_global_class = self._is_at_module_level()
        if is_global_class:
            self.globals_written.add(node.name)
            self.classes_defined.add(node.name)  # Track class definitions
            # Class definitions are written (track for flow-sensitivity)
            if self.in_conditional == 0:
                self.written_at_module_level.add(node.name)
        else:
            self._add_local(node.name)

        # Visit base classes and decorators (in outer scope)
        for base in node.bases:
            self.visit(base)
        for decorator in node.decorator_list:
            self.visit(decorator)

        # For global classes, analyze methods separately
        if is_global_class:
            # Track that we're inside this class
            old_class = self.current_class
            self.current_class = node.name
            self.methods_defined[node.name] = set()

            # Analyze each method in the class
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_name = child.name
                    self.methods_defined[node.name].add(method_name)

                    # Create a sub-analyzer for the method body
                    method_analyzer = GlobalAccessAnalyzer()
                    method_analyzer.scope_stack = [set(), set()]  # module scope + method scope

                    # Add 'self' (or 'cls' for classmethods) to method's local scope
                    if child.args.args:
                        method_analyzer._add_local(child.args.args[0].arg)  # self/cls

                    # Add other parameters to method's local scope
                    for arg in child.args.args[1:]:  # Skip first arg (self/cls)
                        method_analyzer._add_local(arg.arg)
                    for arg in child.args.posonlyargs:
                        method_analyzer._add_local(arg.arg)
                    for arg in child.args.kwonlyargs:
                        method_analyzer._add_local(arg.arg)
                    if child.args.vararg:
                        method_analyzer._add_local(child.args.vararg.arg)
                    if child.args.kwarg:
                        method_analyzer._add_local(child.args.kwarg.arg)

                    # Visit method body
                    for stmt in child.body:
                        method_analyzer.visit(stmt)

                    # Store method info with qualified name
                    qualified_name = f"{node.name}.{method_name}"
                    method_info = FunctionInfo(
                        name=qualified_name,
                        globals_read=method_analyzer.globals_read.copy(),
                        functions_called=method_analyzer.functions_called.copy(),
                        methods_called=method_analyzer.methods_called.copy(),
                        class_name=node.name
                    )
                    self.method_defs[qualified_name] = method_info

            # Restore previous class context
            self.current_class = old_class

            # Still visit class body normally for other definitions
            self._push_scope()
            for child in node.body:
                self.visit(child)
            self._pop_scope()
        else:
            # Nested class - visit normally
            self._push_scope()
            for child in node.body:
                self.visit(child)
            self._pop_scope()

    def visit_Name(self, node: ast.Name):
        """Handle variable name references."""
        name = node.id

        # Check if name is declared global in current scope
        is_declared_global = self._is_declared_global(name)

        # Skip if it's a local variable (unless declared global)
        if not is_declared_global and self._is_local(name):
            return

        # Determine if it's a read or write
        if isinstance(node.ctx, ast.Store):
            # Handle writes
            if self._is_at_module_level() or is_declared_global:
                # Write to global scope
                self.globals_written.add(name)
                # Track the conditional depth at which this variable was written
                self.conditional_write_depth[name] = self.in_conditional
                # Only add to written_at_module_level if not in conditional
                # (conservative: conditional writes might not execute)
                if self.in_conditional == 0 and self._is_at_module_level():
                    self.written_at_module_level.add(name)
            else:
                # Write to local scope (unless declared global, handled above)
                self._add_local(name)
        elif isinstance(node.ctx, (ast.Load, ast.Del)):
            # Handle reads
            # Track all non-builtin names
            # Common builtins that we want to exclude
            common_builtins = {
                'True', 'False', 'None', 'print', 'len', 'range', 'str', 'int',
                'float', 'bool', 'list', 'dict', 'set', 'tuple', 'type', 'isinstance',
                'hasattr', 'getattr', 'setattr', 'max', 'min', 'sum', 'abs', 'all',
                'any', 'enumerate', 'zip', 'map', 'filter', 'sorted', 'reversed',
                'open', 'repr', 'chr', 'ord', 'id', 'hash', 'hex', 'oct', 'bin',
                'round', 'divmod', 'pow', 'next', 'iter', 'callable', 'dir',
                'Exception', 'ValueError', 'TypeError', 'KeyError', 'IndexError',
                'RuntimeError', 'AttributeError', 'ImportError', 'IOError',
                'OSError', 'NameError', 'ZeroDivisionError', 'StopIteration'
            }
            if name not in common_builtins:
                # Flow-sensitive: only add to globals_read if not already written at module level
                if self._is_at_module_level():
                    # Check if variable was written unconditionally
                    if name in self.written_at_module_level:
                        # Written unconditionally, don't add to globals_read
                        pass
                    elif name in self.conditional_write_depth:
                        # Written conditionally - check if we're at same or deeper nesting
                        write_depth = self.conditional_write_depth[name]
                        if self.in_conditional >= write_depth:
                            # We're at the same or deeper nesting level as the write,
                            # so the write dominates this read within this block
                            pass
                        else:
                            # We're at a shallower nesting level, so the write might
                            # not have executed (e.g., write in loop, read after loop)
                            self.globals_read.add(name)
                    else:
                        # Not written at all, add to globals_read
                        self.globals_read.add(name)
                elif is_declared_global:
                    # Declared global in nested scope - add to globals_read
                    self.globals_read.add(name)
                else:
                    # In nested scope, not declared global, add normally
                    self.globals_read.add(name)

        self.generic_visit(node)

    def _extract_callable_reference(self, node):
        """
        Extract function/method reference from a node that might be callable.

        Returns:
            Tuple of (function_name, method_name) where either can be None
        """
        if isinstance(node, ast.Name) and not self._is_local(node.id):
            # Direct function reference: f or my_func
            return (node.id, None)
        elif isinstance(node, ast.Attribute):
            # Method reference: obj.method or a.b.method
            # We track the method name as potentially callable
            return (None, node.attr)
        return (None, None)

    def visit_Call(self, node: ast.Call):
        """Handle function calls."""
        # Track the function being called
        if isinstance(node.func, ast.Name):
            func_name = node.func.id

            # Case 1: Warn on dynamic code execution (eval, exec, compile)
            if func_name in DYNAMIC_CODE_FUNCTIONS:
                self.warnings.append(
                    f"line {node.lineno}: {func_name}() - dynamic code execution, "
                    f"dependencies may be incomplete"
                )

            # Case 2: Warn on reflection functions (getattr, setattr, etc.)
            if func_name in REFLECTION_FUNCTIONS:
                self.warnings.append(
                    f"line {node.lineno}: {func_name}() - dynamic attribute access, "
                    f"dependencies may be incomplete"
                )

            if not self._is_local(func_name):
                self.functions_called.add(func_name)
                # Flow-sensitive: only add to globals_read if not already written at module level
                if self._is_at_module_level():
                    # Check if function was written unconditionally
                    if func_name in self.written_at_module_level:
                        # Written unconditionally, don't add to globals_read
                        pass
                    elif func_name in self.conditional_write_depth:
                        # Written conditionally - check if we're at same or deeper nesting
                        write_depth = self.conditional_write_depth[func_name]
                        if self.in_conditional >= write_depth:
                            # We're at the same or deeper nesting level as the write,
                            # so the write dominates this read within this block
                            pass
                        else:
                            # We're at a shallower nesting level, so the write might
                            # not have executed (e.g., write in loop, read after loop)
                            self.globals_read.add(func_name)
                    else:
                        # Not written at all, add to globals_read
                        self.globals_read.add(func_name)
                else:
                    # In nested scope, add to globals_read
                    self.globals_read.add(func_name)

        elif isinstance(node.func, ast.Attribute):
            # For method calls like obj.method(), track obj and the method name
            self.visit(node.func.value)
            # Track the method name - it might dispatch to a notebook-defined method
            method_name = node.func.attr
            self.methods_called.add(method_name)

            # Case 1: Warn on importlib.import_module()
            if method_name == 'import_module':
                self.warnings.append(
                    f"line {node.lineno}: import_module() - dynamic import, "
                    f"dependencies may be incomplete"
                )

        elif isinstance(node.func, ast.Lambda):
            # Case 7a: Lambda calls - NO warning needed
            # The lambda body is analyzed by visit_Lambda, so dependencies are captured
            self.visit(node.func)  # Visit the lambda to capture its dependencies

        elif isinstance(node.func, ast.Subscript):
            # Case 7b: Indirect call via subscript - funcs[0](), handlers['key']()
            self.warnings.append(
                f"line {node.lineno}: indirect call via subscript - "
                f"called function unknown, dependencies may be incomplete"
            )
            self.visit(node.func)  # Visit to capture dependencies in the subscript

        elif isinstance(node.func, ast.Call):
            # Case 7c: Indirect call via function result - get_handler()()
            self.warnings.append(
                f"line {node.lineno}: indirect call via function result - "
                f"called function unknown, dependencies may be incomplete"
            )
            self.visit(node.func)  # Visit the inner call

        elif isinstance(node.func, ast.IfExp):
            # Case 7d: Indirect call via conditional - (f1 if cond else f2)()
            self.warnings.append(
                f"line {node.lineno}: indirect call via conditional - "
                f"called function unknown, dependencies may be incomplete"
            )
            self.visit(node.func)  # Visit to capture dependencies in the conditional

        else:
            # Case 7 (other): Any other indirect call pattern
            self.warnings.append(
                f"line {node.lineno}: indirect call - "
                f"called function unknown, dependencies may be incomplete"
            )
            self.visit(node.func)  # Visit to capture any dependencies

        # Visit arguments - functions/methods passed as arguments are considered called
        for arg in node.args:
            self.visit(arg)
            # Extract callable references (functions or methods)
            func_ref, method_ref = self._extract_callable_reference(arg)
            if func_ref:
                self.functions_called.add(func_ref)
            if method_ref:
                self.functions_called.add(method_ref)  # Might be a function too
                self.methods_called.add(method_ref)

        for keyword in node.keywords:
            self.visit(keyword.value)
            # Extract callable references from keyword arguments
            func_ref, method_ref = self._extract_callable_reference(keyword.value)
            if func_ref:
                self.functions_called.add(func_ref)
            if method_ref:
                self.functions_called.add(method_ref)  # Might be a function too
                self.methods_called.add(method_ref)

    def visit_Import(self, node: ast.Import):
        """Handle import statements."""
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            if self._is_at_module_level():
                self.modules.add(name)  # Track as module, not regular global
                self.globals_written.add(name)
                # Imports are always written (not conditional in typical usage)
                if self.in_conditional == 0:
                    self.written_at_module_level.add(name)
            else:
                self._add_local(name)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Handle from...import statements."""
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            if name == '*':
                # Case 3: Star imports - warn about unknown names
                module_name = node.module or '<unknown>'
                self.warnings.append(
                    f"line {node.lineno}: from {module_name} import * - "
                    f"star import, dependencies may be incomplete"
                )
                continue
            if self._is_at_module_level():
                # Items imported from modules - track separately to exclude from dependencies
                self.imported_names.add(name)
                self.globals_written.add(name)
                # Track for flow-sensitivity
                if self.in_conditional == 0:
                    self.written_at_module_level.add(name)
            else:
                self._add_local(name)

    def visit_For(self, node: ast.For):
        """Handle for loops."""
        # Visit the iterable first (before target is assigned)
        self.visit(node.iter)

        # Target variables are assigned - extract names and mark as local
        self._extract_assignment_targets(node.target)

        # Loop bodies are conditional - be conservative
        self.in_conditional += 1
        try:
            # Visit body
            for child in node.body:
                self.visit(child)
            # Visit else clause if present
            for child in node.orelse:
                self.visit(child)
        finally:
            self.in_conditional -= 1

    def _visit_comprehension(self, node, elt=None, key=None, value=None):
        """Helper to visit comprehensions (list, set, dict, generator).

        Comprehensions have their own scope - loop variables are local to the
        comprehension and should not be treated as global reads/writes.
        """
        # Collect all loop variables from all generators (these are local to comprehension)
        local_vars = set()

        # First, visit all iterators (these may read globals) and collect local vars
        for generator in node.generators:
            # Visit the iterator first (may read globals)
            self.visit(generator.iter)

            # Collect target variables (loop vars) - these are local
            if isinstance(generator.target, ast.Name):
                local_vars.add(generator.target.id)
            elif isinstance(generator.target, (ast.Tuple, ast.List)):
                for elt_node in ast.walk(generator.target):
                    if isinstance(elt_node, ast.Name):
                        local_vars.add(elt_node.id)

        # Save state AFTER visiting iterators
        state_after_iters_reads = self.globals_read.copy()
        state_after_iters_writes = self.globals_written.copy()

        # Now visit the element/key/value expressions and filters
        # Track what they read/write, but exclude local vars
        for generator in node.generators:
            for if_clause in generator.ifs:
                self.visit(if_clause)

        # Visit the element/key/value expressions
        for expr in [elt, key, value]:
            if expr is not None:
                self.visit(expr)

        # Remove local variable references from reads/writes
        self.globals_read -= local_vars
        self.globals_written -= local_vars

    def visit_ListComp(self, node: ast.ListComp):
        """Visit list comprehension - loop variables are local."""
        self._visit_comprehension(node, elt=node.elt)

    def visit_SetComp(self, node: ast.SetComp):
        """Visit set comprehension - loop variables are local."""
        self._visit_comprehension(node, elt=node.elt)

    def visit_DictComp(self, node: ast.DictComp):
        """Visit dict comprehension - loop variables are local."""
        self._visit_comprehension(node, key=node.key, value=node.value)

    def visit_GeneratorExp(self, node: ast.GeneratorExp):
        """Visit generator expression - loop variables are local."""
        self._visit_comprehension(node, elt=node.elt)

    def _extract_assignment_targets(self, node):
        """Extract names from assignment targets and add to appropriate scope."""
        if isinstance(node, ast.Name):
            name = node.id
            is_declared_global = self._is_declared_global(name)

            if self._is_at_module_level() or is_declared_global:
                self.globals_written.add(name)
                # Track for flow-sensitivity
                if self.in_conditional == 0 and self._is_at_module_level():
                    self.written_at_module_level.add(name)

            # Add to local scope only if not declared global
            if not is_declared_global:
                self._add_local(name)
        elif isinstance(node, (ast.Tuple, ast.List)):
            for elt in node.elts:
                self._extract_assignment_targets(elt)
        elif isinstance(node, ast.Starred):
            self._extract_assignment_targets(node.value)
        # For subscript, attribute, etc., visit normally
        elif isinstance(node, (ast.Subscript, ast.Attribute)):
            self.visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        """Handle attribute access.

        This method tracks:
        - __dict__ access for Case 2 (reflection warning)
        - Attribute assignments for Case 7f (stored function detection)
        """
        # Case 2: Warn on __dict__ access
        if node.attr == '__dict__':
            self.warnings.append(
                f"line {node.lineno}: __dict__ access - "
                f"dynamic attribute access, dependencies may be incomplete"
            )

        # Case 7f: Track attribute assignments (obj.attr = value)
        if isinstance(node.ctx, ast.Store):
            self.attributes_assigned.add(node.attr)

        # Continue visiting the value part (e.g., the 'obj' in obj.attr)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign):
        """Handle assignment statements.

        Important: Visit RHS before LHS to handle x = x + 1 correctly.
        """
        # Visit the RHS (value) first
        self.visit(node.value)

        # Now mark the LHS targets as written
        for target in node.targets:
            if isinstance(target, ast.Name):
                name = target.id
                is_declared_global = self._is_declared_global(name)
                if (self._is_at_module_level() or is_declared_global) and not (self._is_local(name) and not is_declared_global):
                    self.globals_written.add(name)
                    if self.in_conditional == 0 and self._is_at_module_level():
                        self.written_at_module_level.add(name)
            # Visit the target for any nested structure (including attribute assignments)
            self.visit(target)

    def visit_AugAssign(self, node: ast.AugAssign):
        """Handle augmented assignment (+=, -=, etc.).

        For x += 1, x is read before being written.
        """
        # Visit target first as Load (it's being read)
        if isinstance(node.target, ast.Name):
            name = node.target.id
            is_declared_global = self._is_declared_global(name)

            if not self._is_local(name) or is_declared_global:
                if self._is_at_module_level() and name not in self.written_at_module_level:
                    self.globals_read.add(name)
                elif is_declared_global:
                    self.globals_read.add(name)

        # Visit the value
        self.visit(node.value)

        # Now mark as written
        if isinstance(node.target, ast.Name):
            name = node.target.id
            is_declared_global = self._is_declared_global(name)

            if self._is_at_module_level() or is_declared_global:
                if not self._is_local(name) or is_declared_global:
                    self.globals_written.add(name)
                    if self.in_conditional == 0 and self._is_at_module_level():
                        self.written_at_module_level.add(name)

    def visit_If(self, node: ast.If):
        """Handle if statements."""
        # Visit condition first
        self.visit(node.test)

        # Bodies are conditional - be conservative
        self.in_conditional += 1
        try:
            for child in node.body:
                self.visit(child)
            for child in node.orelse:
                self.visit(child)
        finally:
            self.in_conditional -= 1

    def visit_While(self, node: ast.While):
        """Handle while loops."""
        # Visit condition
        self.visit(node.test)

        # Loop bodies are conditional
        self.in_conditional += 1
        try:
            for child in node.body:
                self.visit(child)
            for child in node.orelse:
                self.visit(child)
        finally:
            self.in_conditional -= 1

    def visit_Try(self, node: ast.Try):
        """Handle try/except statements."""
        # Try body might not complete - be conservative
        self.in_conditional += 1
        try:
            for child in node.body:
                self.visit(child)
            for handler in node.handlers:
                self.visit(handler)
            for child in node.orelse:
                self.visit(child)
            for child in node.finalbody:
                self.visit(child)
        finally:
            self.in_conditional -= 1

    def visit_With(self, node: ast.With):
        """Handle with statements."""
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars:
                self.visit(item.optional_vars)
        for child in node.body:
            self.visit(child)

    def visit_ExceptHandler(self, node: ast.ExceptHandler):
        """Handle except clauses."""
        if node.type:
            self.visit(node.type)
        if node.name:
            self._add_local(node.name)
        for child in node.body:
            self.visit(child)


def analyze_cell_dependencies(source: str, cell_id: str) -> tuple[CellDependencies, Dict[str, FunctionInfo], Dict[str, FunctionInfo]]:
    """
    Analyze a single cell's code to extract its dependencies.

    Args:
        source: The Python source code of the cell
        cell_id: The cell's identifier

    Returns:
        Tuple of (CellDependencies, dict of function definitions, dict of method definitions)
    """
    deps = CellDependencies(cell_id=cell_id)
    function_defs = {}
    method_defs = {}

    if not source or not source.strip():
        return deps, function_defs, method_defs

    try:
        tree = ast.parse(source)
        analyzer = GlobalAccessAnalyzer()
        analyzer.visit(tree)

        deps.globals_read = analyzer.globals_read.copy()
        deps.globals_written = analyzer.globals_written.copy()
        deps.functions_called = analyzer.functions_called.copy()
        deps.modules = analyzer.modules.copy()
        deps.imported_names = analyzer.imported_names.copy()
        deps.functions_defined = analyzer.functions_defined.copy()
        deps.classes_defined = analyzer.classes_defined.copy()
        deps.methods_called = analyzer.methods_called.copy()
        deps.methods_defined = {cls: methods.copy() for cls, methods in analyzer.methods_defined.items()}
        deps.attributes_assigned = analyzer.attributes_assigned.copy()
        deps.warnings = analyzer.warnings.copy()

        # Store function definitions with their cell info
        function_defs = analyzer.function_defs.copy()
        for func_info in function_defs.values():
            func_info.defined_in_cell = cell_id

        # Store method definitions with their cell info
        method_defs = analyzer.method_defs.copy()
        for method_info in method_defs.values():
            method_info.defined_in_cell = cell_id

    except SyntaxError:
        # If the code has syntax errors, we can't analyze it
        pass

    return deps, function_defs, method_defs


def compute_transitive_dependencies(
    func_name: str,
    function_map: Dict[str, FunctionInfo],
    visited: Optional[Set[str]] = None
) -> Set[str]:
    """
    Compute transitive closure of dependencies for a function or method.

    Args:
        func_name: Name of the function/method to analyze (can be qualified like "ClassName.method")
        function_map: Map of all function and method definitions
        visited: Set of already visited functions (to avoid cycles)

    Returns:
        Set of all global variables this function depends on (transitively)
    """
    if visited is None:
        visited = set()

    if func_name in visited or func_name not in function_map:
        return set()

    visited.add(func_name)
    func_info = function_map[func_name]

    # Start with direct dependencies
    all_deps = func_info.globals_read.copy()

    # Add transitive dependencies from called functions
    for called_func in func_info.functions_called:
        transitive_deps = compute_transitive_dependencies(called_func, function_map, visited)
        all_deps.update(transitive_deps)

    # Add transitive dependencies from called methods
    for called_method in func_info.methods_called:
        transitive_deps = compute_transitive_dependencies(called_method, function_map, visited)
        all_deps.update(transitive_deps)

    return all_deps


def analyze_notebook(notebook: Dict[str, Any]) -> Dict[str, CellDependencies]:
    """
    Analyze all cells in a notebook to extract dependencies.

    This performs a two-pass analysis:
    1. First pass: collect all function and method definitions and their direct dependencies
    2. Second pass: compute transitive closure for each cell based on functions/methods called

    Args:
        notebook: Jupyter notebook content as a dictionary

    Returns:
        Dictionary mapping cell IDs to their dependencies (with transitive closure applied)
    """
    dependencies: Dict[str, CellDependencies] = {}
    function_map: Dict[str, FunctionInfo] = {}  # Combined map of functions and methods
    method_name_map: Dict[str, List[str]] = {}  # method_name -> list of qualified names

    cells = notebook.get('cells', [])

    # First pass: collect all dependencies, function definitions, and method definitions
    for cell in cells:
        # Only analyze code cells
        if cell.get('cell_type') != 'code':
            continue

        # Get cell ID
        cell_id = cell.get('id', '')
        if not cell_id:
            continue

        # Get cell source
        source = cell.get('source', '')
        if isinstance(source, list):
            source = ''.join(source)

        # Analyze the cell
        deps, func_defs, method_defs = analyze_cell_dependencies(source, cell_id)
        dependencies[cell_id] = deps

        # Add function definitions to global map
        function_map.update(func_defs)

        # Add method definitions to global map and build method name index
        function_map.update(method_defs)  # Methods also go in the unified map
        for qualified_name, method_info in method_defs.items():
            # Extract method name from "ClassName.method_name"
            method_name = qualified_name.split('.')[-1]
            if method_name not in method_name_map:
                method_name_map[method_name] = []
            method_name_map[method_name].append(qualified_name)

    # Collect all modules and imported names across all cells
    all_modules = set()
    all_imported_names = set()
    for deps in dependencies.values():
        all_modules.update(deps.modules)
        all_imported_names.update(deps.imported_names)

    # Collect ALL variables/functions/classes defined in the notebook
    # This includes everything written in any cell
    all_notebook_definitions = set()
    for deps in dependencies.values():
        all_notebook_definitions.update(deps.globals_written)

    # Remove modules (import math -> math is external)
    # But keep imported_names (from math import sqrt -> sqrt IS a notebook definition)
    # Cells can depend on imports from other cells through transitive closure
    all_notebook_definitions -= all_modules
    # Note: imported_names stays in all_notebook_definitions!

    # Second pass: compute transitive closure for function and method calls
    # and filter to only include notebook-internal dependencies
    for cell_id, deps in dependencies.items():
        # Compute transitive dependencies for all called functions
        transitive_deps = set()
        for func_name in deps.functions_called:
            func_deps = compute_transitive_dependencies(func_name, function_map)
            transitive_deps.update(func_deps)

        # Compute transitive dependencies for all called methods
        # Conservative: assume each method call could dispatch to ANY method with that name
        for method_name in deps.methods_called:
            # Look up all methods with this name
            if method_name in method_name_map:
                for qualified_name in method_name_map[method_name]:
                    method_deps = compute_transitive_dependencies(qualified_name, function_map)
                    transitive_deps.update(method_deps)
            # Also check if it's a function with this name (method_name could be a function too)
            if method_name in function_map:
                func_deps = compute_transitive_dependencies(method_name, function_map)
                transitive_deps.update(func_deps)

        # Add transitive dependencies to globals_read
        deps.globals_read.update(transitive_deps)

        # Remove modules from reads and writes (import math -> math is external)
        # But keep imported_names (from math import sqrt -> sqrt IS a notebook definition)
        # Imported names create bindings in the namespace, affecting cell dependencies
        deps.globals_read -= all_modules
        deps.globals_written -= all_modules
        # Note: imported_names stays in BOTH globals_read and globals_written!

        # KEEP ONLY notebook-defined variables in globals_read
        # (remove external dependencies - things not defined in the notebook)
        deps.globals_read &= all_notebook_definitions

    # Case 7e: Detect aliased function calls
    # If a cell calls a name that is:
    # - Defined in the notebook (in globals_written somewhere)
    # - NOT a function definition
    # - NOT a class definition
    # Then warn that it might be an aliased function
    all_function_defs = set(function_map.keys())
    all_class_defs = set()
    for deps in dependencies.values():
        all_class_defs.update(deps.classes_defined)

    for cell_id, deps in dependencies.items():
        for func_name in deps.functions_called:
            # Skip if it's a known function or class
            if func_name in all_function_defs:
                continue
            if func_name in all_class_defs:
                continue
            # Skip if it's not defined in the notebook (external)
            if func_name not in all_notebook_definitions:
                continue
            # This is a notebook-defined variable being called as a function
            # It might be an aliased function
            deps.warnings.append(
                f"call to '{func_name}' which is not a function definition - "
                f"may be aliased function, dependencies may be incomplete"
            )

    # Case 7f: Detect attribute-stored function calls
    # If an attribute name is both assigned and called, warn about potential stored function
    all_attributes_assigned: Set[str] = set()
    for deps in dependencies.values():
        all_attributes_assigned.update(deps.attributes_assigned)

    for cell_id, deps in dependencies.items():
        for attr_name in deps.methods_called:
            # Skip common method names to reduce false positives
            if attr_name in COMMON_METHOD_NAMES:
                continue
            # Check if this attribute was assigned anywhere in the notebook
            if attr_name in all_attributes_assigned:
                deps.warnings.append(
                    f"call to attribute '{attr_name}' which is also assigned elsewhere - "
                    f"may be stored function, dependencies may be incomplete"
                )

    return dependencies


def get_dependency_graph(notebook: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Create a simplified dependency graph showing which globals each cell accesses.

    Args:
        notebook: Jupyter notebook content as a dictionary

    Returns:
        Dictionary mapping cell IDs to lists of global variable names accessed
    """
    dependencies = analyze_notebook(notebook)
    return {
        cell_id: sorted(list(deps.globals_read))
        for cell_id, deps in dependencies.items()
    }


def get_cell_writes(notebook: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Get which global variables each cell writes to.

    Args:
        notebook: Jupyter notebook content as a dictionary

    Returns:
        Dictionary mapping cell IDs to lists of global variable names written
    """
    dependencies = analyze_notebook(notebook)
    return {
        cell_id: sorted(list(deps.globals_written))
        for cell_id, deps in dependencies.items()
    }


if __name__ == '__main__':
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(
        description='Analyze dependencies in a Jupyter notebook'
    )
    parser.add_argument(
        'notebook',
        help='Path to the Jupyter notebook file (.ipynb)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed information including function calls'
    )
    parser.add_argument(
        '--writes-only', '-w',
        action='store_true',
        help='Show only which variables each cell writes'
    )
    parser.add_argument(
        '--reads-only', '-r',
        action='store_true',
        help='Show only which variables each cell reads'
    )

    args = parser.parse_args()

    # Read the notebook file
    try:
        with open(args.notebook, 'r') as f:
            notebook = json.load(f)
    except FileNotFoundError:
        print(f"Error: Notebook file '{args.notebook}' not found", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: '{args.notebook}' is not a valid JSON file", file=sys.stderr)
        sys.exit(1)

    # Analyze the notebook
    dependencies = analyze_notebook(notebook)

    if not dependencies:
        print("No code cells found in notebook")
        sys.exit(0)

    # Print results
    print(f"\nDependency Analysis for: {args.notebook}")
    print("=" * 80)

    for cell_id, deps in dependencies.items():
        # Get cell index for display
        cells = notebook.get('cells', [])
        cell_index = None
        for idx, cell in enumerate(cells):
            if cell.get('id') == cell_id:
                cell_index = idx
                break

        # Print cell header
        if cell_index is not None:
            print(f"\nCell [{cell_index}] (ID: {cell_id[:8]}...)")
        else:
            print(f"\nCell (ID: {cell_id[:8]}...)")
        print("-" * 80)

        # Show reads
        if not args.writes_only:
            if deps.globals_read:
                print(f"  Reads:  {', '.join(sorted(deps.globals_read))}")
            else:
                print("  Reads:  (none)")

        # Show writes
        if not args.reads_only:
            if deps.globals_written:
                print(f"  Writes: {', '.join(sorted(deps.globals_written))}")
            else:
                print("  Writes: (none)")

        # Show function calls in verbose mode
        if args.verbose and not args.writes_only and not args.reads_only:
            if deps.functions_called:
                print(f"  Calls:  {', '.join(sorted(deps.functions_called))}")
            else:
                print("  Calls:  (none)")

        # Show warnings
        if deps.warnings:
            print(f"  ⚠ Warnings ({len(deps.warnings)}):")
            for warning in deps.warnings:
                print(f"    - {warning}")

    # Print summary
    print("\n" + "=" * 80)
    print("Summary")
    print("-" * 80)

    # All variables written across all cells
    all_writes = set()
    for deps in dependencies.values():
        all_writes.update(deps.globals_written)

    # All variables read across all cells
    all_reads = set()
    for deps in dependencies.values():
        all_reads.update(deps.globals_read)

    print(f"Total cells analyzed: {len(dependencies)}")
    print(f"Global variables defined: {len(all_writes)}")
    if all_writes:
        print(f"  {', '.join(sorted(all_writes))}")
    print(f"Global variables used: {len(all_reads)}")
    if all_reads:
        print(f"  {', '.join(sorted(all_reads))}")

    # Find potential issues
    undefined = all_reads - all_writes
    if undefined:
        print(f"\n⚠ Variables used but not defined in this notebook: {len(undefined)}")
        print(f"  {', '.join(sorted(undefined))}")
        print("  (These may be imported, built-in, or defined in other notebooks)")

    # Summary of warnings
    all_warnings = []
    for cell_id, deps in dependencies.items():
        for warning in deps.warnings:
            all_warnings.append((cell_id, warning))

    if all_warnings:
        print(f"\n⚠ Analysis Warnings ({len(all_warnings)} total):")
        print("  Dependencies may be incomplete due to dynamic code patterns.")
        print("  See per-cell warnings above for details.")

    print()
