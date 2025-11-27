"""AST manipulation utilities for code transformation."""

import ast


def wrap_last_expr_with_print_repr(src: str) -> str:
    """
    Given the code for a Jupyter cell as a string, if the final statement
    is an expression statement, replace it with:
        _val = <expr>
        if _val is not None:
            print(repr(_val))
    Otherwise, return the code unchanged.
    """
    try:
        tree = ast.parse(src, mode="exec", type_comments=True)
    except SyntaxError:
        return src

    if not tree.body:
        return src

    last = tree.body[-1]
    if isinstance(last, ast.Expr):
        # _val = <expr>
        assign = ast.Assign(
            targets=[ast.Name(id="_val", ctx=ast.Store())], value=last.value
        )

        # if _val is not None: print(repr(_val))
        cond = ast.Compare(
            left=ast.Name(id="_val", ctx=ast.Load()),
            ops=[ast.IsNot()],
            comparators=[ast.Constant(value=None)],
        )
        print_call = ast.Expr(
            value=ast.Call(
                func=ast.Name(id="print", ctx=ast.Load()),
                args=[
                    ast.Call(
                        func=ast.Name(id="repr", ctx=ast.Load()),
                        args=[ast.Name(id="_val", ctx=ast.Load())],
                        keywords=[],
                    )
                ],
                keywords=[],
            )
        )
        if_stmt = ast.If(test=cond, body=[print_call], orelse=[])

        # Replace the last expression with these two statements
        tree.body[-1:] = [assign, if_stmt]
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)

    return src
