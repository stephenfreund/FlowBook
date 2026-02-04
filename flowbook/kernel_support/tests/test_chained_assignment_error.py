"""
Test that ChainedAssignmentError is raised as an error in FlowBook kernels.

This ensures that when user code in a notebook triggers pandas ChainedAssignmentError,
execution stops and the error is reported to the user.
"""

import time

import pytest
from jupyter_client import KernelManager

from flowbook import make_kernels
from flowbook.server.kernel_manager import FlowbookKernelClient


@pytest.fixture
def flowbook_kernel():
    """Start a FlowBook kernel and yield the client."""
    make_kernels()

    kernel_manager = KernelManager(kernel_name="flowbook_kernel")
    kernel_manager.start_kernel()

    kernel_client = FlowbookKernelClient(kernel_id=kernel_manager.kernel_id)
    kernel_client.load_connection_info(kernel_manager.get_connection_info())
    kernel_client.start_channels()

    # Wait for kernel to be ready
    time.sleep(2)
    kernel_client.wait_for_ready(timeout=30)

    yield kernel_client

    # Cleanup
    kernel_client.stop_channels()
    kernel_manager.shutdown_kernel(now=True)


def execute_and_get_result(kernel_client, code, cell_id="test"):
    """Execute code and collect all output messages."""
    msg_id = kernel_client.execute(code, cell_id=cell_id)

    outputs = []
    error_info = None

    # Collect messages until we get execute_reply
    while True:
        try:
            msg = kernel_client.iopub_channel.get_msg(timeout=10)
        except Exception:
            break

        msg_type = msg["msg_type"]
        parent_msg_id = msg["parent_header"].get("msg_id")

        if parent_msg_id != msg_id:
            continue

        if msg_type == "stream":
            outputs.append(("stream", msg["content"]))
        elif msg_type == "execute_result":
            outputs.append(("result", msg["content"]))
        elif msg_type == "error":
            error_info = msg["content"]
            outputs.append(("error", msg["content"]))
        elif msg_type == "status" and msg["content"]["execution_state"] == "idle":
            break

    # Get execute_reply
    reply = kernel_client.get_shell_msg(timeout=10)

    return {
        "outputs": outputs,
        "error": error_info,
        "reply_status": reply["content"]["status"],
    }


class TestChainedAssignmentError:
    """Test that ChainedAssignmentError stops execution in FlowBook kernels."""

    def test_chained_assignment_raises_error(self, flowbook_kernel):
        """Chained assignment on DataFrame should raise an error."""
        # First, create a DataFrame
        setup_code = """
import pandas as pd
df = pd.DataFrame({'col': ['a', 'b', 'c']})
"""
        result = execute_and_get_result(flowbook_kernel, setup_code, "setup")
        assert result["reply_status"] == "ok", f"Setup failed: {result}"

        # Now try chained assignment - this should error
        error_code = "df['col'].iloc[0] = 'z'"
        result = execute_and_get_result(flowbook_kernel, error_code, "error_cell")

        # Verify the error was raised
        assert result["reply_status"] == "error", (
            f"Expected error status but got {result['reply_status']}. "
            f"Chained assignment should raise ChainedAssignmentError."
        )
        assert result["error"] is not None, "Expected error output"
        assert "ChainedAssignmentError" in result["error"]["ename"], (
            f"Expected ChainedAssignmentError but got {result['error']['ename']}"
        )

    def test_proper_loc_assignment_works(self, flowbook_kernel):
        """Proper .loc assignment should work without error."""
        code = """
import pandas as pd
df = pd.DataFrame({'col': ['a', 'b', 'c']})
df.loc[0, 'col'] = 'z'
df['col'].iloc[0]
"""
        result = execute_and_get_result(flowbook_kernel, code, "proper")

        assert result["reply_status"] == "ok", f"Proper assignment failed: {result}"

    def test_chained_assignment_in_nested_object(self, flowbook_kernel):
        """Chained assignment in nested object should also error."""
        # Setup
        setup_code = """
import pandas as pd

class Container:
    def __init__(self, df):
        self.data = df

df = pd.DataFrame({'col': ['a', 'b', 'c']})
container = Container(df)
"""
        result = execute_and_get_result(flowbook_kernel, setup_code, "setup2")
        assert result["reply_status"] == "ok", f"Setup failed: {result}"

        # Chained assignment through nested object
        error_code = "container.data['col'].iloc[0] = 'z'"
        result = execute_and_get_result(flowbook_kernel, error_code, "nested_error")

        assert result["reply_status"] == "error", (
            f"Expected error status but got {result['reply_status']}. "
            f"Chained assignment through nested object should also error."
        )
        assert result["error"] is not None
        assert "ChainedAssignmentError" in result["error"]["ename"]

    def test_chained_assignment_with_loc_raises_error(self, flowbook_kernel):
        """Chained assignment with .loc should also raise an error."""
        setup_code = """
import pandas as pd
df = pd.DataFrame({'col': ['a', 'b', 'c']})
"""
        result = execute_and_get_result(flowbook_kernel, setup_code, "setup_loc")
        assert result["reply_status"] == "ok", f"Setup failed: {result}"

        # Chained assignment with .loc
        error_code = "df['col'].loc[0] = 'z'"
        result = execute_and_get_result(flowbook_kernel, error_code, "loc_error")

        assert result["reply_status"] == "error", (
            f"Expected error status but got {result['reply_status']}. "
            f"Chained assignment with .loc should raise ChainedAssignmentError."
        )
        assert result["error"] is not None
        assert "ChainedAssignmentError" in result["error"]["ename"]

    def test_error_message_contains_fix_suggestion(self, flowbook_kernel):
        """Error message should contain helpful fix suggestion."""
        setup_code = """
import pandas as pd
df = pd.DataFrame({'col': ['a', 'b', 'c']})
"""
        result = execute_and_get_result(flowbook_kernel, setup_code, "setup_msg")
        assert result["reply_status"] == "ok"

        error_code = "df['col'].iloc[0] = 'z'"
        result = execute_and_get_result(flowbook_kernel, error_code, "msg_error")

        assert result["error"] is not None
        error_value = result["error"]["evalue"]
        # Check that error message mentions the fix
        assert ".loc" in error_value, "Error message should suggest using .loc"
        assert "copy" in error_value.lower(), "Error message should mention copy behavior"

    def test_direct_iloc_assignment_works(self, flowbook_kernel):
        """Direct .iloc assignment (not chained) should work."""
        code = """
import pandas as pd
df = pd.DataFrame({'col': ['a', 'b', 'c']})
df.iloc[0, 0] = 'z'
df.iloc[0, 0]
"""
        result = execute_and_get_result(flowbook_kernel, code, "direct_iloc")

        assert result["reply_status"] == "ok", (
            f"Direct .iloc assignment should work but got error: {result}"
        )

    def test_direct_loc_assignment_works(self, flowbook_kernel):
        """Direct .loc assignment should work."""
        code = """
import pandas as pd
df = pd.DataFrame({'col': ['a', 'b', 'c']})
df.loc[0, 'col'] = 'z'
df.loc[0, 'col']
"""
        result = execute_and_get_result(flowbook_kernel, code, "direct_loc")

        assert result["reply_status"] == "ok", (
            f"Direct .loc assignment should work but got error: {result}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
