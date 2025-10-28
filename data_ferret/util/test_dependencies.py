"""
Tests for the dependency analysis module.
"""

import pytest
from data_ferret.util.dependencies import (
    analyze_cell_dependencies,
    analyze_notebook,
    get_dependency_graph,
    get_cell_writes,
)


def test_simple_variable_access():
    """Test detection of simple global variable access."""
    source = """
x = 5
y = x + 10
"""
    deps, _ = analyze_cell_dependencies(source, "cell1")

    assert "x" in deps.globals_written
    assert "y" in deps.globals_written


def test_function_call_detection():
    """Test detection of function calls."""
    source = """
result = some_function(arg1, arg2)
"""
    deps, _ = analyze_cell_dependencies(source, "cell1")

    assert "some_function" in deps.functions_called
    assert "some_function" in deps.globals_read
    assert "result" in deps.globals_written


def test_function_passed_as_argument():
    """Test that functions passed as arguments are tracked as called."""
    source = """
result = map(my_func, data)
"""
    deps, _ = analyze_cell_dependencies(source, "cell1")

    assert "my_func" in deps.functions_called
    assert "map" in deps.functions_called
    assert "data" in deps.globals_read


def test_function_definition():
    """Test that function definitions create globals."""
    source = """
def my_function(x, y):
    return x + y
"""
    deps, _ = analyze_cell_dependencies(source, "cell1")

    assert "my_function" in deps.globals_written


def test_local_variables_not_global():
    """Test that local variables inside functions aren't tracked as globals."""
    source = """
def process_data(df):
    result = df.groupby('col').sum()
    return result

output = process_data(data)
"""
    deps, _ = analyze_cell_dependencies(source, "cell1")

    # Function definition is global
    assert "process_data" in deps.globals_written

    # Local variables inside function should not be in globals
    assert "result" not in deps.globals_read
    assert "df" not in deps.globals_read

    # But data and output should be
    assert "data" in deps.globals_read
    assert "output" in deps.globals_written


def test_class_definition():
    """Test class definitions."""
    source = """
class MyClass:
    def __init__(self, value):
        self.value = value
"""
    deps, _ = analyze_cell_dependencies(source, "cell1")

    assert "MyClass" in deps.globals_written


def test_imports():
    """Test import statement tracking."""
    source = """
import pandas as pd
from numpy import array
"""
    deps, _ = analyze_cell_dependencies(source, "cell1")

    # Modules should be tracked separately
    assert "pd" in deps.modules
    assert "pd" in deps.globals_written
    # Items from 'from' imports are tracked as imported_names
    assert "array" in deps.imported_names
    assert "array" in deps.globals_written
    assert "array" not in deps.modules


def test_method_calls():
    """Test method calls on objects."""
    source = """
result = df.groupby('column').mean()
"""
    deps, _ = analyze_cell_dependencies(source, "cell1")

    assert "df" in deps.globals_read
    assert "result" in deps.globals_written


def test_lambda_with_globals():
    """Test lambda expressions accessing globals."""
    source = """
f = lambda x: x + global_offset
"""
    deps, _ = analyze_cell_dependencies(source, "cell1")

    assert "global_offset" in deps.globals_read
    assert "f" in deps.globals_written


def test_comprehension_with_globals():
    """Test list comprehensions accessing globals."""
    source = """
result = [x * multiplier for x in data]
"""
    deps, _ = analyze_cell_dependencies(source, "cell1")

    assert "data" in deps.globals_read
    assert "multiplier" in deps.globals_read
    assert "result" in deps.globals_written


def test_for_loop():
    """Test for loops."""
    source = """
for item in items:
    process(item)
"""
    deps, _ = analyze_cell_dependencies(source, "cell1")

    assert "items" in deps.globals_read
    assert "process" in deps.functions_called


def test_with_statement():
    """Test with statements."""
    source = """
with open(filename) as f:
    data = f.read()
"""
    deps, _ = analyze_cell_dependencies(source, "cell1")

    assert "filename" in deps.globals_read
    assert "data" in deps.globals_written


def test_empty_cell():
    """Test that empty cells don't cause errors."""
    deps, _ = analyze_cell_dependencies("", "cell1")

    assert len(deps.globals_read) == 0
    assert len(deps.globals_written) == 0


def test_syntax_error():
    """Test that syntax errors are handled gracefully."""
    source = """
def broken(
    this is not valid python
"""
    deps, _ = analyze_cell_dependencies(source, "cell1")

    # Should return empty dependencies
    assert len(deps.globals_read) == 0
    assert len(deps.globals_written) == 0


def test_notebook_analysis():
    """Test analyzing a complete notebook."""
    notebook = {
        "cells": [
            {
                "id": "cell1",
                "cell_type": "code",
                "source": "import pandas as pd\ndf = pd.DataFrame()"
            },
            {
                "id": "cell2",
                "cell_type": "code",
                "source": "result = df.groupby('col').sum()"
            },
            {
                "id": "cell3",
                "cell_type": "markdown",
                "source": "# This is markdown"
            }
        ]
    }

    dependencies = analyze_notebook(notebook)

    # Check cell1
    assert "cell1" in dependencies
    # pd is a module, so it's tracked separately but excluded from globals_written
    assert "pd" in dependencies["cell1"].modules
    assert "pd" not in dependencies["cell1"].globals_written
    assert "df" in dependencies["cell1"].globals_written

    # Check cell2
    assert "cell2" in dependencies
    assert "df" in dependencies["cell2"].globals_read
    assert "result" in dependencies["cell2"].globals_written

    # Markdown cell should not be in dependencies
    assert "cell3" not in dependencies


def test_get_dependency_graph():
    """Test the simplified dependency graph."""
    notebook = {
        "cells": [
            {
                "id": "cell1",
                "cell_type": "code",
                "source": "x = 5"
            },
            {
                "id": "cell2",
                "cell_type": "code",
                "source": "y = x + 10"
            }
        ]
    }

    graph = get_dependency_graph(notebook)

    assert "cell1" in graph
    assert "cell2" in graph
    assert "x" in graph["cell2"]


def test_get_cell_writes():
    """Test getting cell writes."""
    notebook = {
        "cells": [
            {
                "id": "cell1",
                "cell_type": "code",
                "source": "x = 5\ny = 10"
            }
        ]
    }

    writes = get_cell_writes(notebook)

    assert "cell1" in writes
    assert "x" in writes["cell1"]
    assert "y" in writes["cell1"]


def test_nested_function_scopes():
    """Test that nested function scopes are handled correctly."""
    source = """
def outer():
    global_var1
    def inner():
        global_var2
        return x
    return inner()
"""
    deps, func_defs = analyze_cell_dependencies(source, "cell1")

    assert "outer" in deps.globals_written
    # Variables used inside the function are not in the cell's direct reads
    # They are tracked in the function's dependencies
    assert "outer" in func_defs
    assert "global_var1" in func_defs["outer"].globals_read
    # inner is local to outer, not a global
    assert "inner" not in deps.globals_read


def test_decorator_usage():
    """Test that decorators are tracked."""
    source = """
@my_decorator
def decorated_func():
    pass
"""
    deps, _ = analyze_cell_dependencies(source, "cell1")

    assert "my_decorator" in deps.globals_read
    assert "decorated_func" in deps.globals_written


def test_exception_handling():
    """Test exception handling."""
    source = """
try:
    risky_operation()
except CustomError as e:
    handle_error(e)
"""
    deps, _ = analyze_cell_dependencies(source, "cell1")

    assert "risky_operation" in deps.functions_called
    assert "handle_error" in deps.functions_called
    # Custom exception types should be tracked as they might be defined in other cells
    assert "CustomError" in deps.globals_read


def test_transitive_closure():
    """Test that dependencies are transitively closed over function calls."""
    notebook = {
        "cells": [
            {
                "id": "cell1",
                "cell_type": "code",
                "source": "x = 5"
            },
            {
                "id": "cell2",
                "cell_type": "code",
                "source": "def foo():\n    return x + 10"
            },
            {
                "id": "cell3",
                "cell_type": "code",
                "source": "result = foo()"
            }
        ]
    }

    dependencies = analyze_notebook(notebook)

    # Cell 3 calls foo, which reads x, so cell3 should depend on x transitively
    assert "foo" in dependencies["cell3"].functions_called
    assert "x" in dependencies["cell3"].globals_read


def test_transitive_closure_chained():
    """Test transitive closure through multiple function calls."""
    notebook = {
        "cells": [
            {
                "id": "cell1",
                "cell_type": "code",
                "source": "data = [1, 2, 3]"
            },
            {
                "id": "cell2",
                "cell_type": "code",
                "source": "def get_data():\n    return data"
            },
            {
                "id": "cell3",
                "cell_type": "code",
                "source": "def process():\n    return get_data()"
            },
            {
                "id": "cell4",
                "cell_type": "code",
                "source": "result = process()"
            }
        ]
    }

    dependencies = analyze_notebook(notebook)

    # Cell 4 calls process, which calls get_data, which reads data
    # So cell4 should transitively depend on data
    assert "process" in dependencies["cell4"].functions_called
    assert "data" in dependencies["cell4"].globals_read


def test_modules_excluded_from_dependencies():
    """Test that modules are excluded from dependency tracking."""
    notebook = {
        "cells": [
            {
                "id": "cell1",
                "cell_type": "code",
                "source": "import numpy as np"
            },
            {
                "id": "cell2",
                "cell_type": "code",
                "source": "result = np.array([1, 2, 3])"
            }
        ]
    }

    dependencies = analyze_notebook(notebook)

    # Cell 1 imports np, which should be tracked as a module
    assert "np" in dependencies["cell1"].modules

    # Cell 2 uses np, but it should be excluded from globals_read since it's a module
    assert "np" not in dependencies["cell2"].globals_read


def test_function_definition_tracking():
    """Test that function definitions are properly tracked."""
    source = """
def compute(x, y):
    return x + y + offset
"""
    deps, func_defs = analyze_cell_dependencies(source, "cell1")

    # Function should be in globals_written
    assert "compute" in deps.globals_written

    # Function definition should be captured
    assert "compute" in func_defs
    assert "offset" in func_defs["compute"].globals_read


def test_flow_sensitive_write_before_read():
    """Test that variables written before being read are not dependencies."""
    source = """
x = 5
y = x + 10
"""
    deps, _ = analyze_cell_dependencies(source, "cell1")

    # x is written before being read, so should NOT be in globals_read
    assert "x" not in deps.globals_read
    assert "x" in deps.globals_written
    assert "y" in deps.globals_written


def test_flow_sensitive_read_before_write():
    """Test that variables read before being written ARE dependencies."""
    source = """
y = x + 10
x = 5
"""
    deps, _ = analyze_cell_dependencies(source, "cell1")

    # x is read before being written, so SHOULD be in globals_read
    assert "x" in deps.globals_read
    assert "x" in deps.globals_written
    assert "y" in deps.globals_written


def test_flow_sensitive_self_reference():
    """Test x = x + 1 pattern (read before write in same statement)."""
    source = """
x = x + 1
"""
    deps, _ = analyze_cell_dependencies(source, "cell1")

    # x is read (RHS) before being written (LHS)
    assert "x" in deps.globals_read
    assert "x" in deps.globals_written


def test_flow_sensitive_conditional_write():
    """Test that conditional writes are treated conservatively."""
    source = """
if condition:
    x = 5
y = x + 10
"""
    deps, _ = analyze_cell_dependencies(source, "cell1")

    # x is conditionally written, so reading it later means it's a dependency
    assert "x" in deps.globals_read
    assert "condition" in deps.globals_read


def test_flow_sensitive_loop_write():
    """Test that loop writes are treated conservatively."""
    source = """
for i in range(10):
    x = i
y = x
"""
    deps, _ = analyze_cell_dependencies(source, "cell1")

    # x is written in a loop, so it's conditional - reading it is a dependency
    assert "x" in deps.globals_read


def test_flow_sensitive_multiple_writes():
    """Test multiple writes to same variable."""
    source = """
x = 5
x = x + 1
z = x + 10
"""
    deps, _ = analyze_cell_dependencies(source, "cell1")

    # First write: x not a dependency
    # Second statement: x read (from first write), then written
    # Since x was written unconditionally before being read, it's NOT a global dep
    assert "x" not in deps.globals_read
    assert "x" in deps.globals_written


def test_flow_sensitive_in_notebook():
    """Test flow-sensitivity in notebook context."""
    notebook = {
        "cells": [
            {
                "id": "cell1",
                "cell_type": "code",
                "source": "x = 5\ny = x + 10"
            },
            {
                "id": "cell2",
                "cell_type": "code",
                "source": "z = y + x"
            }
        ]
    }

    dependencies = analyze_notebook(notebook)

    # Cell 1: x written before read, not a dependency
    assert "x" not in dependencies["cell1"].globals_read
    assert "y" in dependencies["cell1"].globals_written

    # Cell 2: both x and y are dependencies (from previous cells)
    assert "x" in dependencies["cell2"].globals_read
    assert "y" in dependencies["cell2"].globals_read


def test_include_notebook_defined_functions():
    """Test that functions defined in the notebook ARE tracked as dependencies."""
    notebook = {
        "cells": [
            {
                "id": "cell1",
                "cell_type": "code",
                "source": "def foo():\n    return 42"
            },
            {
                "id": "cell2",
                "cell_type": "code",
                "source": "result = foo()"
            }
        ]
    }

    dependencies = analyze_notebook(notebook)

    # Cell 1 defines foo
    assert "foo" in dependencies["cell1"].functions_defined
    assert "foo" in dependencies["cell1"].globals_written

    # Cell 2 calls foo, and foo SHOULD be in globals_read
    # because it's defined in the notebook and used by this cell
    assert "foo" in dependencies["cell2"].functions_called
    assert "foo" in dependencies["cell2"].globals_read


def test_include_notebook_defined_classes():
    """Test that classes defined in the notebook ARE tracked as dependencies."""
    notebook = {
        "cells": [
            {
                "id": "cell1",
                "cell_type": "code",
                "source": "class MyClass:\n    def __init__(self):\n        pass"
            },
            {
                "id": "cell2",
                "cell_type": "code",
                "source": "obj = MyClass()"
            }
        ]
    }

    dependencies = analyze_notebook(notebook)

    # Cell 1 defines MyClass
    assert "MyClass" in dependencies["cell1"].classes_defined
    assert "MyClass" in dependencies["cell1"].globals_written

    # Cell 2 uses MyClass, and it SHOULD be in globals_read
    # because it's defined in the notebook
    assert "MyClass" in dependencies["cell2"].globals_read


def test_exclude_external_functions():
    """Test that externally defined functions are NOT tracked as dependencies."""
    notebook = {
        "cells": [
            {
                "id": "cell1",
                "cell_type": "code",
                "source": "result = external_function()"
            }
        ]
    }

    dependencies = analyze_notebook(notebook)

    # external_function is not defined in notebook, so it should NOT be a dependency
    assert "external_function" not in dependencies["cell1"].globals_read
    assert "external_function" in dependencies["cell1"].functions_called


def test_function_and_class_across_cells():
    """Test function/class tracking across multiple cells."""
    notebook = {
        "cells": [
            {
                "id": "cell1",
                "cell_type": "code",
                "source": "def helper():\n    return data"
            },
            {
                "id": "cell2",
                "cell_type": "code",
                "source": "class Processor:\n    pass"
            },
            {
                "id": "cell3",
                "cell_type": "code",
                "source": "data = 5"
            },
            {
                "id": "cell4",
                "cell_type": "code",
                "source": "result = helper()\nproc = Processor()"
            }
        ]
    }

    dependencies = analyze_notebook(notebook)

    # Cell 4 should have data, helper, and Processor as dependencies
    # (all defined in the notebook)
    assert "data" in dependencies["cell4"].globals_read
    assert "helper" in dependencies["cell4"].globals_read
    assert "Processor" in dependencies["cell4"].globals_read


def test_exclude_from_imports():
    """Test that names imported via 'from...import' are excluded from dependencies."""
    notebook = {
        "cells": [
            {
                "id": "cell1",
                "cell_type": "code",
                "source": "from sklearn.linear_model import LinearRegression"
            },
            {
                "id": "cell2",
                "cell_type": "code",
                "source": "model = LinearRegression()"
            }
        ]
    }

    dependencies = analyze_notebook(notebook)

    # Cell 1 imports LinearRegression
    assert "LinearRegression" in dependencies["cell1"].imported_names
    # LinearRegression is removed from globals_written because it's imported

    # Cell 2 uses LinearRegression, but it should NOT be in globals_read
    # because it's imported from an external module
    assert "LinearRegression" not in dependencies["cell2"].globals_read


def test_exclude_from_imports_with_alias():
    """Test that aliased imports are also excluded."""
    notebook = {
        "cells": [
            {
                "id": "cell1",
                "cell_type": "code",
                "source": "from numpy import array as np_array"
            },
            {
                "id": "cell2",
                "cell_type": "code",
                "source": "data = np_array([1, 2, 3])"
            }
        ]
    }

    dependencies = analyze_notebook(notebook)

    # Cell 1 imports array as np_array
    assert "np_array" in dependencies["cell1"].imported_names

    # Cell 2 uses np_array, but it should NOT be in globals_read
    assert "np_array" not in dependencies["cell2"].globals_read


def test_include_notebook_vs_imported():
    """Test distinction between notebook-defined and imported names."""
    notebook = {
        "cells": [
            {
                "id": "cell1",
                "cell_type": "code",
                "source": "from math import sqrt\ndef my_sqrt(x):\n    return sqrt(x)"
            },
            {
                "id": "cell2",
                "cell_type": "code",
                "source": "result = my_sqrt(4)"
            }
        ]
    }

    dependencies = analyze_notebook(notebook)

    # Cell 2 uses my_sqrt (notebook-defined) - should be a dependency
    assert "my_sqrt" in dependencies["cell2"].globals_read

    # Cell 2 transitively uses sqrt (imported) - should NOT be a dependency
    assert "sqrt" not in dependencies["cell2"].globals_read
