"""
Track the global reads of class bodies.

A class body looks names up with LOAD_NAME: first in the class namespace, then
in the globals, then in the builtins. The globals step reads the globals dict's
own storage directly, without the mapping protocol, so when cells run with a
TrackingDict as globals a class body's reads of notebook variables are never
seen by TrackingDict.__getitem__ (TrackingDict mirrors the namespace into that
storage while a cell runs, so the values are right; see tracking.py).

This module rewrites every name a class body loads into a call of a helper
that performs the same lookup through the mapping protocol:

    class CFG:
        years = df.year.unique()
  ->
    class CFG:
        years = (__flowbook_class_load__('df') if __flowbook_class_has__('df') else df).year.unique()

The helper looks in the class namespace first, then in the frame's globals
(TrackingDict.__getitem__: tracked, and subject to read blocking), then in the
builtins. When the name is unbound anywhere, the original load runs and raises
Python's own NameError from the class body (IPython 7 does not hide the frame
an exception is raised in, so raising it in the helper would show an extra frame).

What is rewritten: loads evaluated in the class scope itself, i.e. class-body
statements, method decorators and default values, the bases/keywords/decorators
of a nested class, and the first iterable of a comprehension. Function bodies,
lambda bodies and the rest of a comprehension are separate scopes whose global
reads already use LOAD_GLOBAL (tracked). Annotations are left alone (they may be
deferred strings under `from __future__ import annotations`), and so are names
the class body declares global/nonlocal.

In a class nested in a function, a name bound in an enclosing function is that
function's variable (LOAD_CLASSDEREF), not a global: it is left alone (the
helper could not see it, and function locals are not tracked locations). Every
other name such a class body loads is rewritten like a module-level class's.
"""
import ast
import builtins
import sys

CLASS_LOAD = "__flowbook_class_load__"
CLASS_HAS = "__flowbook_class_has__"


def _class_load(name):
    """Class-body name lookup (class namespace, globals, builtins) through the mapping protocol."""
    __tracebackhide__ = True  # noqa: F841 (IPython hides this frame in tracebacks)
    frame = sys._getframe(1)
    try:
        return frame.f_locals[name]
    except KeyError:
        pass
    try:
        return frame.f_globals[name]
    except KeyError:
        pass
    try:
        return frame.f_builtins[name]
    except KeyError:
        raise NameError(f"name '{name}' is not defined", name=name) from None


def _class_has(name) -> bool:
    """Whether a class-body load of name would succeed (untracked)."""
    frame = sys._getframe(1)
    try:
        frame.f_locals[name]
        return True
    except KeyError:
        pass
    return name in frame.f_globals or name in frame.f_builtins


def install() -> None:
    """Make the helpers reachable from every class body (idempotent)."""
    setattr(builtins, CLASS_LOAD, _class_load)
    setattr(builtins, CLASS_HAS, _class_has)


def _load_expr(node: ast.Name) -> ast.expr:
    """`(__flowbook_class_load__('x') if __flowbook_class_has__('x') else x)` for a load of x."""
    def call(helper):
        return ast.Call(func=ast.Name(id=helper, ctx=ast.Load()), args=[ast.Constant(value=node.id)], keywords=[])
    expr = ast.IfExp(test=call(CLASS_HAS), body=call(CLASS_LOAD), orelse=ast.Name(id=node.id, ctx=ast.Load()))
    for n in ast.walk(expr):
        ast.copy_location(n, node)
    return expr


class _ClassScopeLoads(ast.NodeTransformer):
    """Rewrites the name loads evaluated in one class body's scope."""

    def __init__(self, skip, outer):
        self.skip = skip | outer
        self.outer = outer  # names bound in enclosing functions

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load) and node.id not in self.skip and node.id not in (CLASS_LOAD, CLASS_HAS):
            return _load_expr(node)
        return node

    def _visit_list(self, nodes):
        return [self.visit(n) if n is not None else None for n in nodes]

    def _visit_args(self, args: ast.arguments):
        args.defaults = self._visit_list(args.defaults)
        args.kw_defaults = self._visit_list(args.kw_defaults)

    def visit_FunctionDef(self, node):
        node.decorator_list = self._visit_list(node.decorator_list)
        self._visit_args(node.args)
        return node  # body, annotations: other scopes / not evaluated here

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node):
        self._visit_args(node.args)
        return node

    def visit_ClassDef(self, node):
        node.decorator_list = self._visit_list(node.decorator_list)
        node.bases = self._visit_list(node.bases)
        for kw in node.keywords:
            kw.value = self.visit(kw.value)
        rewrite_class_body(node, self.outer)
        return node

    def _visit_comp(self, node):
        first = node.generators[0]
        first.iter = self.visit(first.iter)
        return node

    visit_ListComp = visit_SetComp = visit_DictComp = visit_GeneratorExp = _visit_comp

    def visit_AnnAssign(self, node):
        if node.value is not None:
            node.value = self.visit(node.value)
        if not isinstance(node.target, ast.Name):
            node.target = self.visit(node.target)  # x.attr: int / x[i]: int evaluate x
        return node


def _declared_nonlocal_names(body):
    names = set()
    stack = list(body)
    while stack:
        n = stack.pop()
        if isinstance(n, (ast.Global, ast.Nonlocal)):
            names.update(n.names)
        elif not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            stack.extend(ast.iter_child_nodes(n))
    return names


def _function_bindings(fn) -> set:
    """Names local to fn's scope (parameters and bindings), minus those it declares global."""
    a = fn.args
    names = {x.arg for x in a.posonlyargs + a.args + a.kwonlyargs}
    names.update(x.arg for x in (a.vararg, a.kwarg) if x is not None)
    declared_global = set()
    stack = list(fn.body)
    while stack:
        n = stack.pop()
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(n.name)  # its body is another scope
            continue
        if isinstance(n, ast.Lambda):
            continue
        if isinstance(n, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            # only an assignment expression binds in the enclosing function
            names.update(m.target.id for m in ast.walk(n) if isinstance(m, ast.NamedExpr))
            continue
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            names.add(n.id)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            names.update((al.asname or al.name).split(".")[0] for al in n.names if al.name != "*")
        elif isinstance(n, ast.ExceptHandler) and n.name:
            names.add(n.name)
        elif isinstance(n, (ast.MatchAs, ast.MatchStar)) and n.name:
            names.add(n.name)
        elif isinstance(n, ast.MatchMapping) and n.rest:
            names.add(n.rest)
        elif isinstance(n, ast.Global):
            declared_global.update(n.names)
        elif isinstance(n, ast.Nonlocal):
            names.update(n.names)  # bound in an enclosing function
        stack.extend(ast.iter_child_nodes(n))
    return names - declared_global


def rewrite_class_body(cls: ast.ClassDef, outer=frozenset()) -> None:
    """Rewrite the loads of cls's body in place (its bases/decorators belong to the
    enclosing scope). outer: names bound in the functions enclosing cls."""
    visitor = _ClassScopeLoads(_declared_nonlocal_names(cls.body), outer)
    cls.body = [visitor.visit(stmt) for stmt in cls.body]
    # classes defined inside the methods (also methods under if/try/with in the body)
    stack = list(cls.body)
    while stack:
        n = stack.pop()
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _Classes(outer | _function_bindings(n)).visit_body(n)
        elif not isinstance(n, (ast.ClassDef, ast.Lambda, ast.expr)):
            stack.extend(ast.iter_child_nodes(n))


class _Classes(ast.NodeTransformer):
    """Finds the classes in a scope; outer: names bound in the enclosing functions."""

    def __init__(self, outer=frozenset()):
        self.outer = frozenset(outer)

    def visit_body(self, fn):
        fn.body = [self.visit(stmt) for stmt in fn.body]

    def visit_FunctionDef(self, node):
        _Classes(self.outer | _function_bindings(node)).visit_body(node)
        return node

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node):
        return node  # a lambda cannot contain a class

    def visit_ClassDef(self, node):
        rewrite_class_body(node, self.outer)  # also handles classes nested in it
        return node


class ClassBodyReadTransformer(ast.NodeTransformer):
    """IPython AST transformer (shell.ast_transformers) applying the rewrite to a cell."""

    def visit_Module(self, node):
        return transform(node)

    visit_Interactive = visit_Module


def compile_cell(source: str, filename: str = "<string>"):
    """Compile a cell for exec with a TrackingDict as globals, class bodies rewritten."""
    install()
    return compile(transform(ast.parse(source, filename)), filename, "exec")


def transform(tree: ast.AST) -> ast.AST:
    """Rewrite the class bodies in tree (a parsed cell); returns tree."""
    if not any(isinstance(n, ast.ClassDef) for n in ast.walk(tree)):
        return tree
    tree = _Classes().visit(tree)
    return ast.fix_missing_locations(tree)
