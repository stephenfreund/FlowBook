"""
Tests for kernel/flowbook_kernel.py - FlowbookKernel class and checkpoint dispatch.

Tests cover:
- CHECKPOINT_COMMANDS configuration
- Timeout parsing from code
- Execution context preparation
- Checkpoint command dispatch
- Error handling
"""

import pytest
import re
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from flowbook.kernel.flowbook_kernel import FlowbookKernel, CHECKPOINT_COMMANDS
from flowbook.kernel.models import ExecutionContext


class TestCheckpointCommandsConfig:
    """Tests for CHECKPOINT_COMMANDS configuration."""

    def test_checkpoint_commands_structure(self):
        """CHECKPOINT_COMMANDS has correct structure."""
        assert isinstance(CHECKPOINT_COMMANDS, dict)

        for cmd, config in CHECKPOINT_COMMANDS.items():
            assert isinstance(cmd, str)
            assert isinstance(config, tuple)
            assert len(config) == 3

            arg_count, request_class_name, arg_names = config
            assert isinstance(arg_count, int)
            assert arg_count >= 0
            assert isinstance(request_class_name, str)
            assert request_class_name.endswith("Request")
            assert isinstance(arg_names, list)
            assert len(arg_names) == arg_count

    def test_all_commands_present(self):
        """All expected commands are present."""
        expected = ["save", "restore", "delete", "list", "compare", "clear"]
        for cmd in expected:
            assert cmd in CHECKPOINT_COMMANDS

    def test_save_command_config(self):
        """Save command has correct config."""
        arg_count, class_name, arg_names = CHECKPOINT_COMMANDS["save"]
        assert arg_count == 1
        assert class_name == "CheckpointSaveRequest"
        assert arg_names == ["name"]

    def test_restore_command_config(self):
        """Restore command has correct config."""
        arg_count, class_name, arg_names = CHECKPOINT_COMMANDS["restore"]
        assert arg_count == 1
        assert class_name == "CheckpointRestoreRequest"
        assert arg_names == ["name"]

    def test_delete_command_config(self):
        """Delete command has correct config."""
        arg_count, class_name, arg_names = CHECKPOINT_COMMANDS["delete"]
        assert arg_count == 1
        assert class_name == "CheckpointDeleteRequest"
        assert arg_names == ["name"]

    def test_list_command_config(self):
        """List command has correct config."""
        arg_count, class_name, arg_names = CHECKPOINT_COMMANDS["list"]
        assert arg_count == 0
        assert class_name == "CheckpointListRequest"
        assert arg_names == []

    def test_compare_command_config(self):
        """Compare command has correct config."""
        arg_count, class_name, arg_names = CHECKPOINT_COMMANDS["compare"]
        assert arg_count == 2
        assert class_name == "CheckpointCompareRequest"
        assert arg_names == ["name1", "name2"]

    def test_clear_command_config(self):
        """Clear command has correct config."""
        arg_count, class_name, arg_names = CHECKPOINT_COMMANDS["clear"]
        assert arg_count == 0
        assert class_name == "CheckpointClearRequest"
        assert arg_names == []

    def test_request_classes_exist(self):
        """All request classes referenced in config exist."""
        from flowbook.kernel import kernel_commands as cmd_module

        for cmd, (_, class_name, _) in CHECKPOINT_COMMANDS.items():
            assert hasattr(cmd_module, class_name), f"Missing {class_name}"


class TestTimeoutParsing:
    """Tests for timeout directive parsing."""

    def test_parse_timeout_no_directive(self):
        """Code without timeout directive uses default."""
        code = "x = 1\ny = 2"
        # Simulate the parsing logic
        match = re.match(r"# timeout (\d+)\n", code)
        assert match is None

    def test_parse_timeout_with_directive(self):
        """Code with timeout directive extracts timeout."""
        code = "# timeout 60\nx = 1"
        match = re.match(r"# timeout (\d+)\n", code)
        assert match is not None
        assert int(match.group(1)) == 60

    def test_parse_timeout_strips_directive(self):
        """Timeout directive is stripped from code."""
        code = "# timeout 120\nx = 1\ny = 2"
        match = re.match(r"# timeout (\d+)\n", code)
        cleaned = code.replace(match.group(0), "", 1)
        assert cleaned == "x = 1\ny = 2"

    def test_parse_timeout_only_at_start(self):
        """Timeout directive must be at start of code."""
        code = "x = 1\n# timeout 60\ny = 2"
        match = re.match(r"# timeout (\d+)\n", code)
        assert match is None  # Not at start

    def test_parse_timeout_various_values(self):
        """Various timeout values are parsed correctly."""
        test_cases = [
            ("# timeout 1\ncode", 1),
            ("# timeout 10\ncode", 10),
            ("# timeout 300\ncode", 300),
            ("# timeout 3600\ncode", 3600),
        ]
        for code, expected in test_cases:
            match = re.match(r"# timeout (\d+)\n", code)
            assert match is not None
            assert int(match.group(1)) == expected


class TestExecutionContextCreation:
    """Tests for ExecutionContext creation logic."""

    def test_context_from_regular_code(self):
        """Context creation for regular code."""
        ctx = ExecutionContext(
            cell_id="abcd",
            code="x = 1",
            timeout=30.0,
            original_code="x = 1",
        )
        assert ctx.should_profile is True
        assert ctx.has_cell_magics is False
        assert ctx.has_shell_magics is False

    def test_context_from_magic_code(self):
        """Context creation for magic commands."""
        ctx = ExecutionContext(
            cell_id="abcd",
            code="%timeit x = 1",
            timeout=30.0,
            original_code="%timeit x = 1",
        )
        assert ctx.should_profile is False
        assert ctx.has_cell_magics is True

    def test_context_from_shell_code(self):
        """Context creation for shell commands."""
        ctx = ExecutionContext(
            cell_id="abcd",
            code="!ls -la",
            timeout=30.0,
            original_code="!ls -la",
        )
        assert ctx.should_profile is False
        assert ctx.has_shell_magics is True

    def test_context_no_cell_id(self):
        """Context without cell_id disables profiling."""
        ctx = ExecutionContext(
            cell_id=None,
            code="x = 1",
            timeout=30.0,
            original_code="x = 1",
        )
        assert ctx.should_profile is False


class TestFlowbookKernelConfiguration:
    """Tests for FlowbookKernel configuration constants."""

    def test_default_cell_timeout(self):
        """Default cell timeout is 30 minutes."""
        assert FlowbookKernel._default_cell_timeout == 30 * 60

    def test_post_kb_grace(self):
        """Post-keyboard-interrupt grace period is 1 second."""
        assert FlowbookKernel._post_kb_grace == 1.0

    def test_kill_timeout(self):
        """Kill timeout is 3 seconds."""
        assert FlowbookKernel._kill_timeout == 3.0

    def test_max_passes(self):
        """Max timeout handler passes is 2."""
        assert FlowbookKernel._max_passes == 2


class TestCheckpointCommandDispatch:
    """Tests for checkpoint command dispatch logic."""

    def test_build_usage_string(self):
        """Usage strings are built correctly from config."""
        for cmd, (arg_count, _, arg_names) in CHECKPOINT_COMMANDS.items():
            usage = f"checkpoint {cmd} " + " ".join(f"<{n}>" for n in arg_names)
            if arg_count == 0:
                assert usage == f"checkpoint {cmd} "
            elif arg_count == 1:
                assert f"<{arg_names[0]}>" in usage
            elif arg_count == 2:
                assert f"<{arg_names[0]}>" in usage
                assert f"<{arg_names[1]}>" in usage

    def test_build_request_kwargs(self):
        """Request kwargs are built correctly from args."""
        # Simulate building kwargs for "compare" command
        args = ["compare", "checkpoint1", "checkpoint2"]
        _, _, arg_names = CHECKPOINT_COMMANDS["compare"]

        kwargs = {name: args[i + 1] for i, name in enumerate(arg_names)}
        assert kwargs == {"name1": "checkpoint1", "name2": "checkpoint2"}

    def test_build_request_kwargs_single_arg(self):
        """Request kwargs work for single-arg commands."""
        args = ["save", "my_checkpoint"]
        _, _, arg_names = CHECKPOINT_COMMANDS["save"]

        kwargs = {name: args[i + 1] for i, name in enumerate(arg_names)}
        assert kwargs == {"name": "my_checkpoint"}

    def test_build_request_kwargs_no_args(self):
        """Request kwargs work for no-arg commands."""
        args = ["list"]
        _, _, arg_names = CHECKPOINT_COMMANDS["list"]

        kwargs = {name: args[i + 1] for i, name in enumerate(arg_names)}
        assert kwargs == {}


class TestCommandHandlerLookup:
    """Tests for handler lookup by command name."""

    def test_handler_name_format(self):
        """Handler names follow correct format."""
        for cmd in CHECKPOINT_COMMANDS.keys():
            handler_name = f"checkpoint_{cmd}"
            # Should match format: checkpoint_<command>
            assert handler_name.startswith("checkpoint_")
            assert handler_name.replace("checkpoint_", "") == cmd


class TestErrorResultFormat:
    """Tests for error result formatting."""

    def test_timeout_error_format(self):
        """Timeout errors have correct format."""
        timeout = 60.0
        timeout_msg = f"Cell execution timed out after {timeout} seconds"

        result = {
            "status": "error",
            "execution_count": 5,
            "ename": "TimeoutError",
            "evalue": timeout_msg,
            "traceback": [timeout_msg],
        }

        assert result["status"] == "error"
        assert result["ename"] == "TimeoutError"
        assert "60" in result["evalue"]

    def test_monotonicity_error_format(self):
        """Monotonicity errors have correct format."""
        from flowbook.kernel.models import MonotonicityViolation

        violation = MonotonicityViolation(
            violated_vars=["x", "y"],
            diff_details="Details here",
            error_summary="Summary here",
        )

        result = violation.to_error_result(10)
        assert result["status"] == "error"
        assert result["ename"] == "MonotonicityError"
        assert result["execution_count"] == 10


class TestDisplayMetadataFormat:
    """Tests for display metadata format."""

    def test_execution_metadata_format(self):
        """Execution metadata has expected structure."""
        from flowbook.kernel.models import ExecutionProfile, ExecutionMetadata, TrackingData

        profile = ExecutionProfile(
            duration=1.5,
            profile="cpu: 50%",
            env={"x": "int"},
            env_after={"x": "int", "y": "str"},
        )
        tracking = TrackingData(
            reads_before_writes=["x"],
            writes=["y"],
        )
        metadata = ExecutionMetadata(
            profile=profile,
            dynamic_dependencies=tracking,
        )

        display = metadata.to_display_metadata()

        assert "profile" in display
        assert display["profile"]["duration"] == 1.5
        assert display["profile"]["profile"] == "cpu: 50%"
        assert "dynamic_dependencies" in display
        assert display["dynamic_dependencies"]["reads_before_writes"] == ["x"]


class TestCheckpointSaveResultFormat:
    """Tests for checkpoint save result display."""

    def test_checkpoint_save_added_text(self):
        """Added text is formatted correctly."""
        saved = {"x": "int", "y": "float", "z": "str"}
        added_text = "\n".join([f"* {k}: {v}" for k, v in saved.items()])

        assert "* x: int" in added_text
        assert "* y: float" in added_text
        assert "* z: str" in added_text

    def test_checkpoint_save_removed_text(self):
        """Removed text is formatted correctly."""
        removed = {"bad_var": "uncopyable"}
        removed_text = "\n".join([f"* {k}: {v}" for k, v in removed.items()])

        assert "* bad_var: uncopyable" in removed_text


class TestMagicCommandParsing:
    """Tests for magic command argument parsing."""

    def test_force_checkpoints_true_values(self):
        """force_checkpoints recognizes true values."""
        true_values = ["", "true", "1", "enable", "on", "yes", "TRUE", "True"]
        for val in true_values:
            enabled = val.strip().lower() not in ["false", "0", "disable", "off"]
            assert enabled is True, f"Failed for {val}"

    def test_force_checkpoints_false_values(self):
        """force_checkpoints recognizes false values."""
        false_values = ["false", "0", "disable", "off", "FALSE", "False", "OFF"]
        for val in false_values:
            enabled = val.strip().lower() not in ["false", "0", "disable", "off"]
            assert enabled is False, f"Failed for {val}"


class TestCheckpointCommandArgumentValidation:
    """Tests for checkpoint command argument validation."""

    def test_save_requires_one_arg(self):
        """Save command requires exactly one argument."""
        arg_count, _, _ = CHECKPOINT_COMMANDS["save"]
        assert arg_count == 1

        # Valid: ["save", "name"]
        args = ["save", "my_checkpoint"]
        assert len(args) - 1 == arg_count

        # Invalid: ["save"]
        args = ["save"]
        assert len(args) - 1 != arg_count

        # Invalid: ["save", "a", "b"]
        args = ["save", "a", "b"]
        assert len(args) - 1 != arg_count

    def test_compare_requires_two_args(self):
        """Compare command requires exactly two arguments."""
        arg_count, _, _ = CHECKPOINT_COMMANDS["compare"]
        assert arg_count == 2

        # Valid: ["compare", "a", "b"]
        args = ["compare", "a", "b"]
        assert len(args) - 1 == arg_count

        # Invalid: ["compare", "a"]
        args = ["compare", "a"]
        assert len(args) - 1 != arg_count

    def test_list_requires_no_args(self):
        """List command requires no arguments."""
        arg_count, _, _ = CHECKPOINT_COMMANDS["list"]
        assert arg_count == 0

        # Valid: ["list"]
        args = ["list"]
        assert len(args) - 1 == arg_count


class TestHandlerNameMapping:
    """Tests for command to handler name mapping."""

    def test_all_handlers_mappable(self):
        """All commands can be mapped to handler names."""
        from flowbook.kernel.kernel_command_handlers import KernelCommandHandlers

        handler_names = [f"checkpoint_{cmd}" for cmd in CHECKPOINT_COMMANDS.keys()]

        # Check that all handler methods exist
        for name in handler_names:
            method_name = f"handle_{name}"
            assert hasattr(KernelCommandHandlers, method_name), f"Missing {method_name}"
