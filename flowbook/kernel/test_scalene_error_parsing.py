"""Test scalene error output parsing."""

from flowbook.kernel.scalene_runner import ScaleneRunner


def test_parse_scalene_error_output():
    """Test parsing of scalene error output."""
    # Create a mock shell and executed_cell_ids
    class MockShell:
        execution_count = 2

    shell = MockShell()
    executed_cell_ids = {1: "abc123"}
    runner = ScaleneRunner(shell, executed_cell_ids)

    # Example scalene error output
    scalene_output = """Error in program being profiled:
 division by zero
Traceback (most recent call last):
  File "/Users/freund/anaconda3/envs/flowbook/lib/python3.12/site-packages/scalene/scalene_profiler.py", line 1687, in profile_code
    exec(code, the_globals, the_locals)
  File "/Users/freund/other/FlowBook/examples/_ipython-input-1-profile", line 1, in <module>
    _val = print(N / 0)
                 ~~^~~
ZeroDivisionError: division by zero
Scalene: The specified code did not run for long enough to profile.
By default, Scalene only profiles code in the file executed and its subdirectories.
To track the time spent in all files, use the `--profile-all` option.
"""

    result = runner._parse_scalene_error_output(scalene_output)

    assert result is not None, "Should parse error output"
    ename, evalue, traceback_lines = result

    assert ename == "ZeroDivisionError"
    assert evalue == "division by zero"
    assert len(traceback_lines) > 0
    assert traceback_lines[0] == "Traceback (most recent call last):"
    assert traceback_lines[-1] == "ZeroDivisionError: division by zero"

    # Check that scalene frame was filtered out
    traceback_str = "\n".join(traceback_lines)
    assert "scalene_profiler.py" not in traceback_str
    assert "profile_code" not in traceback_str

    # Check that filename was replaced with cell ID
    assert "Cell abc123" in traceback_str
    assert "_ipython-input-1-profile" not in traceback_str

    print("Test passed!")
    print("\nParsed traceback:")
    print("\n".join(traceback_lines))


def test_parse_no_error():
    """Test that normal scalene output returns None."""
    class MockShell:
        execution_count = 2

    shell = MockShell()
    executed_cell_ids = {1: "abc123"}
    runner = ScaleneRunner(shell, executed_cell_ids)

    normal_output = """Scalene: The specified code did not run for long enough to profile.
By default, Scalene only profiles code in the file executed and its subdirectories.
To track the time spent in all files, use the `--profile-all` option.
"""

    result = runner._parse_scalene_error_output(normal_output)
    assert result is None, "Should return None for non-error output"
    print("No-error test passed!")


def test_parse_different_exception():
    """Test parsing of different exception types."""
    class MockShell:
        execution_count = 2

    shell = MockShell()
    executed_cell_ids = {1: "xyz789"}
    runner = ScaleneRunner(shell, executed_cell_ids)

    scalene_output = """Error in program being profiled:
 name 'foo' is not defined
Traceback (most recent call last):
  File "/Users/freund/anaconda3/envs/flowbook/lib/python3.12/site-packages/scalene/scalene_profiler.py", line 1687, in profile_code
    exec(code, the_globals, the_locals)
  File "/Users/freund/other/FlowBook/examples/_ipython-input-1-profile", line 2, in <module>
    print(foo)
          ^^^
NameError: name 'foo' is not defined
"""

    result = runner._parse_scalene_error_output(scalene_output)

    assert result is not None
    ename, evalue, traceback_lines = result

    assert ename == "NameError"
    assert evalue == "name 'foo' is not defined"

    traceback_str = "\n".join(traceback_lines)
    assert "Cell xyz789" in traceback_str
    assert "scalene_profiler.py" not in traceback_str

    print("Different exception test passed!")
    print("\nParsed traceback:")
    print("\n".join(traceback_lines))


if __name__ == "__main__":
    test_parse_scalene_error_output()
    print()
    test_parse_no_error()
    print()
    test_parse_different_exception()
