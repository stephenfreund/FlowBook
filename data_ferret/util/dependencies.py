"""
Dependency analysis for Jupyter notebooks.

This module provides static analysis tools to track variable, function, and method
dependencies in notebook cells, creating a map from cell IDs to the global variables
they access.

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

Missing Cases and Under-Approximations
---------------------------------------
The following cases are NOT tracked (under-approximation - may miss dependencies):

1. DYNAMIC CODE EXECUTION:
   - eval(), exec(), compile() with string arguments
   - importlib.import_module() with dynamic names
   - Example: eval("x + y") → x, y not tracked

2. REFLECTION AND DYNAMIC ATTRIBUTE ACCESS:
   - getattr(obj, name), setattr(obj, name, value) with dynamic names
   - __dict__ manipulation
   - Example: getattr(obj, "method")() → method call not tracked

3. STAR IMPORTS:
   - "from module import *" → cannot determine which names are imported
   - These imported names won't be filtered from dependencies

4. ATTRIBUTE ASSIGNMENTS:
   - obj.attr = value → not tracked as a write
   - Only tracks module-level variable assignments

5. CONTAINER MODIFICATIONS:
   - list.append(), dict.update(), etc. → not tracked
   - Only tracks variable bindings, not mutations

6. GLOBAL/NONLOCAL KEYWORDS:
   - Effects of "global x" and "nonlocal x" declarations not fully modeled
   - May incorrectly treat as local in some nested scopes

7. CLASS/INSTANCE ATTRIBUTES:
   - self.attr reads/writes inside methods not tracked separately
   - Only method-level global dependencies are captured

8. INDIRECT CALLS:
   - Calls via function stored in data structures: funcs[0]()
   - Only direct calls and single-level attribute calls tracked

9. METACLASSES AND DESCRIPTORS:
   - Custom __getattribute__, __setattr__ behavior not modeled
   - Class decorators that modify behavior not tracked

Design Decisions
----------------
- CONSERVATIVE FOR SAFETY: Prefer false positives (extra dependencies) over
  false negatives (missing dependencies)

- NOTEBOOK-INTERNAL ONLY: Only tracks dependencies on variables/functions/classes
  defined within the notebook. Imported names and builtins are filtered out.

- TRANSITIVE CLOSURE: Dependencies propagate through function/method calls to
  ensure complete dependency graphs

- METHOD NAME-BASED DISPATCH: Without type information, uses method name
  matching across all classes as a sound over-approximation

Example Usage
-------------
    notebook = {"cells": [...]}
    dependencies = analyze_notebook(notebook)

    for cell_id, deps in dependencies.items():
        print(f"Cell {cell_id} reads: {deps.globals_read}")
        print(f"Cell {cell_id} writes: {deps.globals_written}")
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
            'methods_defined': {cls: sorted(list(methods)) for cls, methods in self.methods_defined.items()}
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
        self.function_defs: Dict[str, FunctionInfo] = {}  # Function definitions found
        self.written_at_module_level: Set[str] = set()  # Track module-level writes in order
        self.in_conditional: int = 0  # Track if we're inside conditional/loop (conservative)
        self.functions_defined: Set[str] = set()  # Function names defined at module level
        self.classes_defined: Set[str] = set()  # Class names defined at module level
        self.methods_called: Set[str] = set()  # Method names called (from obj.method())
        self.methods_defined: Dict[str, Set[str]] = {}  # class_name -> set of method names
        self.method_defs: Dict[str, FunctionInfo] = {}  # "ClassName.method_name" -> FunctionInfo
        self.current_class: Optional[str] = None  # Track which class we're currently analyzing

    def _current_scope(self) -> Set[str]:
        """Get the current local scope."""
        return self.scope_stack[-1]

    def _push_scope(self):
        """Enter a new scope."""
        self.scope_stack.append(set())

    def _pop_scope(self):
        """Exit the current scope."""
        if len(self.scope_stack) > 1:
            self.scope_stack.pop()

    def _is_local(self, name: str) -> bool:
        """Check if a name is local to any current scope."""
        return any(name in scope for scope in self.scope_stack)

    def _add_local(self, name: str):
        """Add a name to the current local scope."""
        self._current_scope().add(name)

    def _is_at_module_level(self) -> bool:
        """Check if we're currently at module level."""
        return len(self.scope_stack) == 1

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

        # Skip if it's a local variable
        if self._is_local(name):
            return

        # Determine if it's a read or write
        if isinstance(node.ctx, ast.Store):
            if self._is_at_module_level():
                self.globals_written.add(name)
                # Only add to written_at_module_level if not in conditional
                # (conservative: conditional writes might not execute)
                if self.in_conditional == 0:
                    self.written_at_module_level.add(name)
            else:
                self._add_local(name)
        elif isinstance(node.ctx, (ast.Load, ast.Del)):
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
                    if name not in self.written_at_module_level:
                        self.globals_read.add(name)
                else:
                    # In nested scope, add normally
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
            if not self._is_local(func_name):
                self.functions_called.add(func_name)
                self.globals_read.add(func_name)
        elif isinstance(node.func, ast.Attribute):
            # For method calls like obj.method(), track obj and the method name
            self.visit(node.func.value)
            # Track the method name - it might dispatch to a notebook-defined method
            method_name = node.func.attr
            self.methods_called.add(method_name)

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
                # Can't track star imports precisely
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

    def _extract_assignment_targets(self, node):
        """Extract names from assignment targets and add to appropriate scope."""
        if isinstance(node, ast.Name):
            if self._is_at_module_level():
                self.globals_written.add(node.id)
                # Track for flow-sensitivity
                if self.in_conditional == 0:
                    self.written_at_module_level.add(node.id)
            # Always add to local scope so it's not tracked as global read
            self._add_local(node.id)
        elif isinstance(node, (ast.Tuple, ast.List)):
            for elt in node.elts:
                self._extract_assignment_targets(elt)
        elif isinstance(node, ast.Starred):
            self._extract_assignment_targets(node.value)
        # For subscript, attribute, etc., visit normally
        elif isinstance(node, (ast.Subscript, ast.Attribute)):
            self.visit(node)

    def visit_Assign(self, node: ast.Assign):
        """Handle assignment statements.

        Important: Visit RHS before LHS to handle x = x + 1 correctly.
        """
        # Visit the RHS (value) first
        self.visit(node.value)

        # Now mark the LHS targets as written
        for target in node.targets:
            if isinstance(target, ast.Name) and self._is_at_module_level() and not self._is_local(target.id):
                self.globals_written.add(target.id)
                if self.in_conditional == 0:
                    self.written_at_module_level.add(target.id)
            # Visit the target for any nested structure
            self.visit(target)

    def visit_AugAssign(self, node: ast.AugAssign):
        """Handle augmented assignment (+=, -=, etc.).

        For x += 1, x is read before being written.
        """
        # Visit target first as Load (it's being read)
        if isinstance(node.target, ast.Name):
            name = node.target.id
            if not self._is_local(name):
                if self._is_at_module_level() and name not in self.written_at_module_level:
                    self.globals_read.add(name)

        # Visit the value
        self.visit(node.value)

        # Now mark as written
        if isinstance(node.target, ast.Name) and self._is_at_module_level() and not self._is_local(node.target.id):
            self.globals_written.add(node.target.id)
            if self.in_conditional == 0:
                self.written_at_module_level.add(node.target.id)

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

    # Remove imported items from notebook definitions - they're external
    all_notebook_definitions -= all_modules
    all_notebook_definitions -= all_imported_names

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

        # Remove all modules and imported names from globals_read and globals_written
        deps.globals_read -= all_modules
        deps.globals_written -= all_modules
        deps.globals_read -= all_imported_names
        deps.globals_written -= all_imported_names

        # KEEP ONLY notebook-defined variables in globals_read
        # (remove external dependencies - things not defined in the notebook)
        deps.globals_read &= all_notebook_definitions

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

    print()
