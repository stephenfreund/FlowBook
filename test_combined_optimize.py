"""
Comprehensive tests for combined optimization functionality.

This test suite mocks LLM calls with canned responses to test the optimization
flow end-to-end without requiring actual API calls.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Any, Dict, List, Optional
import nbformat
from nbformat.v4 import new_notebook, new_code_cell

from data_ferret.server.commands.optimize import (
    OptimizeCommand,
    CombinedOptimizedCodeResponse,
    OptimizationResultAndStats,
    CodeSnippet,
)
from data_ferret.util.ferret_metadata import (
    FerretMetadata,
    OptimizationPotential,
    OptimizationStep,
    ProfileData,
)
from data_ferret.agent.agent import FerretStats
from data_ferret.util.dependencies import CellDependencies
from data_ferret.kernel.types import TestCodeSuccess, DiffResult
from agents import Usage


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_notebook_with_optimization_plan():
    """Create a sample notebook with optimization metadata."""
    nb = new_notebook()

    # Cell 1: Helper function (will be optimized)
    cell1 = new_code_cell(
        source="""def compute_distance(x, y):
    result = []
    for i in range(len(x)):
        dist = 0
        for j in range(len(x[i])):
            dist += (x[i][j] - y[i][j]) ** 2
        result.append(dist ** 0.5)
    return result""",
        id="cell-1"
    )

    # Add optimization plan to cell1
    opt_plan = OptimizationPotential(
        potential=5,
        reasoning="Loop-based distance calculation can be vectorized",
        optimization_plan=[
            OptimizationStep(
                target_cell_id="cell-1",
                function_name="compute_distance",
                description=["Use numpy vectorization instead of loops"]
            )
        ]
    )
    profile = ProfileData(
        duration=1.5,
        profile="scalene profile data",
        env={"x": "numpy.ndarray", "y": "numpy.ndarray"},
        env_after={"x": "numpy.ndarray", "y": "numpy.ndarray", "result": "list"}
    )
    metadata = FerretMetadata(
        optimization_potential=opt_plan,
        profile=profile
    )
    cell1["metadata"]["ferret"] = metadata.model_dump()

    # Cell 2: Main code (uses the helper)
    cell2 = new_code_cell(
        source="""import numpy as np
x = np.random.rand(1000, 10)
y = np.random.rand(1000, 10)
distances = compute_distance(x, y)""",
        id="cell-2"
    )

    nb["cells"] = [cell1, cell2]
    return nb


@pytest.fixture
def sample_notebook_with_multiple_optimizations():
    """Create a notebook with multiple optimization steps."""
    nb = new_notebook()

    # Cell 1: Two functions to optimize
    cell1 = new_code_cell(
        source="""def pairwise_distance(x, y):
    result = []
    for i in range(len(x)):
        for j in range(len(y)):
            dist = sum((x[i][k] - y[j][k]) ** 2 for k in range(len(x[i])))
            result.append(dist ** 0.5)
    return result

def normalize_data(data):
    result = []
    for row in data:
        mean = sum(row) / len(row)
        std = (sum((x - mean) ** 2 for x in row) / len(row)) ** 0.5
        normalized = [(x - mean) / std for x in row]
        result.append(normalized)
    return result""",
        id="cell-1"
    )

    # Cell 2: Main computation
    cell2 = new_code_cell(
        source="""normalized = normalize_data(data)
distances = pairwise_distance(normalized, normalized)""",
        id="cell-2"
    )

    # Add optimization plan with multiple steps
    opt_plan = OptimizationPotential(
        potential=5,
        reasoning="Multiple functions with loop-based operations",
        optimization_plan=[
            OptimizationStep(
                target_cell_id="cell-1",
                function_name="pairwise_distance",
                description=["Use scipy.spatial.distance.cdist for pairwise distances"]
            ),
            OptimizationStep(
                target_cell_id="cell-1",
                function_name="normalize_data",
                description=["Use numpy operations for normalization"]
            )
        ]
    )
    profile = ProfileData(
        duration=2.0,
        profile="scalene profile data",
        env={"data": "numpy.ndarray"},
        env_after={"data": "numpy.ndarray", "normalized": "list", "distances": "list"}
    )
    metadata = FerretMetadata(
        optimization_potential=opt_plan,
        profile=profile
    )
    cell2["metadata"]["ferret"] = metadata.model_dump()

    nb["cells"] = [cell1, cell2]
    return nb


@pytest.fixture
def mock_dependencies_dict():
    """Create mock dependencies dictionary."""
    return {
        "cell-1": CellDependencies(
            cell_id="cell-1",
            globals_read=set(),
            globals_written={"compute_distance"},
        ),
        "cell-2": CellDependencies(
            cell_id="cell-2",
            globals_read={"compute_distance", "np"},
            globals_written={"x", "y", "distances"},
        )
    }


@pytest.fixture
def mock_ferret_stats():
    """Create mock FerretStats."""
    return FerretStats(
        model="gpt-4",
        time=2.5,
        usage=Usage(input_tokens=1000, output_tokens=500),
        log_path="/tmp/agent_log.txt"
    )


# ============================================================================
# Mock LLM Responses
# ============================================================================

def create_mock_optimization_response(optimized_code: str, explanation: str):
    """Create a mock CombinedOptimizedCodeResponse."""
    return CombinedOptimizedCodeResponse(
        optimized_code=optimized_code,
        explanation=explanation
    )


def get_single_function_optimization_response():
    """Canned response for single function optimization."""
    optimized_code = """### Cell cell-1 / compute_distance

import numpy as np

def compute_distance(x, y):
    # Vectorized implementation using numpy
    return np.sqrt(np.sum((x - y) ** 2, axis=1))"""

    explanation = "Applied numpy vectorization to replace nested loops with array operations"

    return create_mock_optimization_response(optimized_code, explanation)


def get_multiple_function_optimization_response():
    """Canned response for multiple function optimization."""
    optimized_code = """### Cell cell-1 / pairwise_distance

from scipy.spatial.distance import cdist

def pairwise_distance(x, y):
    # Use scipy's optimized pairwise distance
    return cdist(x, y, metric='euclidean').flatten()

### Cell cell-1 / normalize_data

import numpy as np

def normalize_data(data):
    # Vectorized normalization using numpy
    data = np.array(data)
    mean = np.mean(data, axis=1, keepdims=True)
    std = np.std(data, axis=1, keepdims=True)
    return (data - mean) / std"""

    explanation = "Applied scipy.spatial.distance.cdist for pairwise distances and numpy broadcasting for normalization"

    return create_mock_optimization_response(optimized_code, explanation)


def get_repair_response():
    """Canned response for repair operation."""
    repaired_code = """### Cell cell-1 / compute_distance

import numpy as np

def compute_distance(x, y):
    # Fixed: ensure return type matches original
    result = np.sqrt(np.sum((x - y) ** 2, axis=1))
    return result.tolist()  # Convert back to list for compatibility"""

    explanation = "Fixed type mismatch - original returned list, optimized returned ndarray. Added .tolist() conversion."

    return create_mock_optimization_response(repaired_code, explanation)


# ============================================================================
# Mock Kernel Client
# ============================================================================

@pytest.fixture
def mock_kernel_client_success():
    """Mock kernel client that returns successful validation."""
    client = Mock()

    # Mock successful test_code response
    test_result = TestCodeSuccess(
        diff=DiffResult(differences={}),
        original_duration=1.5,
        modified_duration=0.5,
        speedup=3.0
    )

    client.test_code = Mock(return_value=Mock(ok=True, result=test_result))

    return client


@pytest.fixture
def mock_kernel_client_validation_fail():
    """Mock kernel client that returns validation failure."""
    client = Mock()

    # Mock failed validation (different outputs)
    diff = DiffResult(
        differences={"distances": ("list of floats", "numpy array")}
    )
    test_result = TestCodeSuccess(
        diff=diff,
        original_duration=1.5,
        modified_duration=0.5,
        speedup=3.0
    )

    client.test_code = Mock(return_value=Mock(ok=True, result=test_result))

    return client


# ============================================================================
# Tests for Helper Methods
# ============================================================================

class TestHelperMethods:
    """Test helper methods for building combined inputs."""

    def test_build_combined_input_code_single_function(self, sample_notebook_with_optimization_plan):
        """Test building combined input code for a single function."""
        cmd = OptimizeCommand()
        nb = sample_notebook_with_optimization_plan
        cells = nb["cells"]
        cell_map = {c["id"]: c for c in cells}

        opt_plan = [
            OptimizationStep(
                target_cell_id="cell-1",
                function_name="compute_distance",
                description=["Use numpy vectorization"]
            )
        ]

        combined = cmd._build_combined_input_code(opt_plan, cell_map)

        assert "### Cell cell-1 / compute_distance" in combined
        assert "def compute_distance(x, y):" in combined
        assert "for i in range(len(x)):" in combined

    def test_build_combined_input_code_whole_cell(self, sample_notebook_with_optimization_plan):
        """Test building combined input code for whole cell."""
        cmd = OptimizeCommand()
        nb = sample_notebook_with_optimization_plan
        cells = nb["cells"]
        cell_map = {c["id"]: c for c in cells}

        opt_plan = [
            OptimizationStep(
                target_cell_id="cell-2",
                function_name=None,
                description=["Optimize imports"]
            )
        ]

        combined = cmd._build_combined_input_code(opt_plan, cell_map)

        assert "### Cell cell-2\n" in combined
        assert "import numpy as np" in combined
        assert "distances = compute_distance(x, y)" in combined

    def test_build_combined_input_code_multiple_segments(self, sample_notebook_with_multiple_optimizations):
        """Test building combined input code for multiple segments."""
        cmd = OptimizeCommand()
        nb = sample_notebook_with_multiple_optimizations
        cells = nb["cells"]
        cell_map = {c["id"]: c for c in cells}

        opt_plan = [
            OptimizationStep(
                target_cell_id="cell-1",
                function_name="pairwise_distance",
                description=["Use scipy"]
            ),
            OptimizationStep(
                target_cell_id="cell-1",
                function_name="normalize_data",
                description=["Use numpy"]
            )
        ]

        combined = cmd._build_combined_input_code(opt_plan, cell_map)

        assert "### Cell cell-1 / pairwise_distance" in combined
        assert "### Cell cell-1 / normalize_data" in combined
        assert "def pairwise_distance(x, y):" in combined
        assert "def normalize_data(data):" in combined

    def test_gather_optimization_suggestions(self):
        """Test gathering optimization suggestions."""
        cmd = OptimizeCommand()

        opt_plan = [
            OptimizationStep(
                target_cell_id="cell-1",
                function_name="func1",
                description=["Use numpy"]
            ),
            OptimizationStep(
                target_cell_id="cell-1",
                function_name="func2",
                description=["Cache results", "Use vectorization"]
            )
        ]

        suggestions = cmd._gather_optimization_suggestions(opt_plan)

        assert "- Use numpy" in suggestions
        assert "- Cache results" in suggestions
        assert "- Use vectorization" in suggestions

    def test_gather_environment_info(self, sample_notebook_with_optimization_plan):
        """Test gathering environment info from cells."""
        cmd = OptimizeCommand()
        nb = sample_notebook_with_optimization_plan
        cells = nb["cells"]
        cell_map = {c["id"]: c for c in cells}

        opt_plan = [
            OptimizationStep(
                target_cell_id="cell-1",
                function_name="compute_distance",
                description=["Optimize"]
            )
        ]

        env_info = cmd._gather_environment_info(opt_plan, cell_map)

        assert "x: numpy.ndarray" in env_info
        assert "y: numpy.ndarray" in env_info

    def test_get_output_variables(self, mock_dependencies_dict):
        """Test getting output variables."""
        cmd = OptimizeCommand()

        output_vars = cmd._get_output_variables("cell-2", mock_dependencies_dict)

        # Should get modified globals from cell-2
        assert "distances" in output_vars
        assert "x" in output_vars
        assert "y" in output_vars

    def test_build_combined_code_from_snippets(self):
        """Test building combined code from snippets."""
        cmd = OptimizeCommand()

        snippets = [
            CodeSnippet(
                cell_id="cell-1",
                function_name="func1",
                source="def func1():\n    return 1",
                optimizations_applied=["optimization 1"]
            ),
            CodeSnippet(
                cell_id="cell-2",
                function_name=None,
                source="result = func1()",
                optimizations_applied=["optimization 2"]
            )
        ]

        combined = cmd._build_combined_code_from_snippets(snippets)

        assert "### Cell cell-1 / func1" in combined
        assert "### Cell cell-2\n" in combined
        assert "def func1():" in combined
        assert "result = func1()" in combined


# ============================================================================
# Tests for Parsing
# ============================================================================

class TestParsing:
    """Test parsing of combined optimization responses."""

    def test_parse_single_function_response(self):
        """Test parsing response with single function."""
        cmd = OptimizeCommand()

        response = get_single_function_optimization_response()

        opt_plan = [
            OptimizationStep(
                target_cell_id="cell-1",
                function_name="compute_distance",
                description=["Optimize"]
            )
        ]

        snippets = cmd._parse_combined_optimization_response(response, opt_plan)

        assert len(snippets) == 1
        assert snippets[0].cell_id == "cell-1"
        assert snippets[0].function_name == "compute_distance"
        assert "numpy" in snippets[0].source
        assert "vectorization" in snippets[0].optimizations_applied[0]

    def test_parse_multiple_function_response(self):
        """Test parsing response with multiple functions."""
        cmd = OptimizeCommand()

        response = get_multiple_function_optimization_response()

        opt_plan = [
            OptimizationStep(
                target_cell_id="cell-1",
                function_name="pairwise_distance",
                description=["Use scipy"]
            ),
            OptimizationStep(
                target_cell_id="cell-1",
                function_name="normalize_data",
                description=["Use numpy"]
            )
        ]

        snippets = cmd._parse_combined_optimization_response(response, opt_plan)

        assert len(snippets) == 2

        # First snippet
        assert snippets[0].cell_id == "cell-1"
        assert snippets[0].function_name == "pairwise_distance"
        assert "cdist" in snippets[0].source

        # Second snippet
        assert snippets[1].cell_id == "cell-1"
        assert snippets[1].function_name == "normalize_data"
        assert "np.mean" in snippets[1].source

    def test_parse_whole_cell_response(self):
        """Test parsing response for whole cell."""
        cmd = OptimizeCommand()

        response = create_mock_optimization_response(
            optimized_code="""### Cell cell-2

import numpy as np
x = np.random.rand(1000, 10)
y = np.random.rand(1000, 10)
distances = compute_distance(x, y)""",
            explanation="No changes needed"
        )

        opt_plan = [
            OptimizationStep(
                target_cell_id="cell-2",
                function_name=None,
                description=["Review"]
            )
        ]

        snippets = cmd._parse_combined_optimization_response(response, opt_plan)

        assert len(snippets) == 1
        assert snippets[0].cell_id == "cell-2"
        assert snippets[0].function_name is None
        assert "import numpy as np" in snippets[0].source

    def test_validate_parsed_snippets_success(self):
        """Test validation of parsed snippets - success case."""
        cmd = OptimizeCommand()

        snippets = [
            CodeSnippet(cell_id="cell-1", function_name="func1", source="code"),
            CodeSnippet(cell_id="cell-1", function_name="func2", source="code")
        ]

        opt_plan = [
            OptimizationStep(target_cell_id="cell-1", function_name="func1", description=[""]),
            OptimizationStep(target_cell_id="cell-1", function_name="func2", description=[""])
        ]

        # Should not raise
        cmd._validate_parsed_snippets(snippets, opt_plan)

    def test_validate_parsed_snippets_missing(self):
        """Test validation of parsed snippets - missing snippet."""
        cmd = OptimizeCommand()

        snippets = [
            CodeSnippet(cell_id="cell-1", function_name="func1", source="code")
        ]

        opt_plan = [
            OptimizationStep(target_cell_id="cell-1", function_name="func1", description=[""]),
            OptimizationStep(target_cell_id="cell-1", function_name="func2", description=[""])
        ]

        with pytest.raises(ValueError, match="Missing expected snippets"):
            cmd._validate_parsed_snippets(snippets, opt_plan)

    def test_validate_parsed_snippets_wrong_count(self):
        """Test validation of parsed snippets - wrong count."""
        cmd = OptimizeCommand()

        snippets = [
            CodeSnippet(cell_id="cell-1", function_name="func1", source="code"),
            CodeSnippet(cell_id="cell-1", function_name="func2", source="code")
        ]

        opt_plan = [
            OptimizationStep(target_cell_id="cell-1", function_name="func1", description=[""])
        ]

        with pytest.raises(ValueError, match="Expected 1 snippets, got 2"):
            cmd._validate_parsed_snippets(snippets, opt_plan)


# ============================================================================
# Tests for Combined Optimization Flow
# ============================================================================

class TestCombinedOptimization:
    """Test the combined optimization flow end-to-end."""

    @pytest.mark.asyncio
    async def test_run_combined_llm_optimization_single_function(
        self,
        sample_notebook_with_optimization_plan,
        mock_dependencies_dict,
        mock_ferret_stats
    ):
        """Test combined LLM optimization with single function."""
        cmd = OptimizeCommand()
        nb = sample_notebook_with_optimization_plan
        cells = nb["cells"]

        opt_plan = [
            OptimizationStep(
                target_cell_id="cell-1",
                function_name="compute_distance",
                description=["Use numpy vectorization"]
            )
        ]

        # Mock the FerretAgent
        with patch('data_ferret.server.commands.optimize.FerretAgent') as MockAgent:
            # Create mock agent instance (regular Mock with AsyncMock run method)
            mock_agent_instance = Mock()
            mock_agent_instance.run = AsyncMock(
                return_value=(get_single_function_optimization_response(), mock_ferret_stats)
            )
            MockAgent.__getitem__.return_value = MockAgent
            MockAgent.return_value = mock_agent_instance

            # Mock NotebookTools
            with patch('data_ferret.server.commands.optimize.NotebookTools') as MockTools:
                mock_tools = Mock()
                mock_tools.tools = Mock(return_value=[])
                MockTools.return_value.__enter__ = Mock(return_value=mock_tools)
                MockTools.return_value.__exit__ = Mock(return_value=False)

                from data_ferret.server.commands.optimize import StatsAggregator
                stats_agg = StatsAggregator(model="gpt-4")

                original_snippets, optimized_snippets = await cmd._run_combined_llm_optimization(
                    opt_plan,
                    cells,
                    "cell-1",
                    "gpt-4",
                    mock_tools,
                    mock_dependencies_dict,
                    stats_agg
                )

        # Verify results
        assert len(original_snippets) == 1
        assert len(optimized_snippets) == 1

        assert original_snippets[0].cell_id == "cell-1"
        assert original_snippets[0].function_name == "compute_distance"

        assert optimized_snippets[0].cell_id == "cell-1"
        assert optimized_snippets[0].function_name == "compute_distance"
        assert "numpy" in optimized_snippets[0].source

    @pytest.mark.asyncio
    async def test_run_combined_llm_optimization_multiple_functions(
        self,
        sample_notebook_with_multiple_optimizations,
        mock_ferret_stats
    ):
        """Test combined LLM optimization with multiple functions."""
        cmd = OptimizeCommand()
        nb = sample_notebook_with_multiple_optimizations
        cells = nb["cells"]

        opt_plan = [
            OptimizationStep(
                target_cell_id="cell-1",
                function_name="pairwise_distance",
                description=["Use scipy"]
            ),
            OptimizationStep(
                target_cell_id="cell-1",
                function_name="normalize_data",
                description=["Use numpy"]
            )
        ]

        dependencies_dict = {
            "cell-1": CellDependencies(
                cell_id="cell-1",
                globals_read=set(),
                globals_written={"pairwise_distance", "normalize_data"},
            ),
            "cell-2": CellDependencies(
                cell_id="cell-2",
                globals_read={"pairwise_distance", "normalize_data", "data"},
                globals_written={"normalized", "distances"},
            )
        }

        # Mock the FerretAgent
        with patch('data_ferret.server.commands.optimize.FerretAgent') as MockAgent:
            mock_agent_instance = Mock()
            mock_agent_instance.run = AsyncMock(
                return_value=(get_multiple_function_optimization_response(), mock_ferret_stats)
            )
            MockAgent.__getitem__.return_value = MockAgent
            MockAgent.return_value = mock_agent_instance

            with patch('data_ferret.server.commands.optimize.NotebookTools') as MockTools:
                mock_tools = Mock()
                mock_tools.tools = Mock(return_value=[])
                MockTools.return_value.__enter__ = Mock(return_value=mock_tools)
                MockTools.return_value.__exit__ = Mock(return_value=False)

                from data_ferret.server.commands.optimize import StatsAggregator
                stats_agg = StatsAggregator(model="gpt-4")

                original_snippets, optimized_snippets = await cmd._run_combined_llm_optimization(
                    opt_plan,
                    cells,
                    "cell-2",
                    "gpt-4",
                    mock_tools,
                    dependencies_dict,
                    stats_agg
                )

        # Verify results
        assert len(original_snippets) == 2
        assert len(optimized_snippets) == 2

        # Check first optimization
        assert optimized_snippets[0].cell_id == "cell-1"
        assert optimized_snippets[0].function_name == "pairwise_distance"
        assert "cdist" in optimized_snippets[0].source

        # Check second optimization
        assert optimized_snippets[1].cell_id == "cell-1"
        assert optimized_snippets[1].function_name == "normalize_data"
        assert "np.mean" in optimized_snippets[1].source


# ============================================================================
# Tests for Validation and Retry
# ============================================================================

class TestValidationAndRetry:
    """Test validation and retry logic."""

    @pytest.mark.asyncio
    async def test_validation_success(
        self,
        sample_notebook_with_optimization_plan,
        mock_kernel_client_success,
        mock_dependencies_dict,
        mock_ferret_stats
    ):
        """Test successful validation."""
        cmd = OptimizeCommand()
        nb = sample_notebook_with_optimization_plan
        cells = nb["cells"]
        cell = cells[0]

        original_snippets = [
            CodeSnippet(
                cell_id="cell-1",
                function_name="compute_distance",
                source="def compute_distance(x, y):\n    return []"
            )
        ]

        optimized_snippets = [
            CodeSnippet(
                cell_id="cell-1",
                function_name="compute_distance",
                source="import numpy as np\ndef compute_distance(x, y):\n    return np.array([])"
            )
        ]

        with patch('data_ferret.server.commands.optimize.NotebookTools') as MockTools:
            mock_tools = Mock()
            mock_tools.tools = Mock(return_value=[])
            MockTools.return_value.__enter__ = Mock(return_value=mock_tools)
            MockTools.return_value.__exit__ = Mock(return_value=False)

            # Mock the test_code call
            with patch.object(cmd, '_send_test_code_comm') as mock_test:
                from data_ferret.kernel.types import TestCodeSuccess, DiffResult
                test_result = TestCodeSuccess(
                    diff=DiffResult(differences={}),
                    original_duration=1.0,
                    modified_duration=0.5,
                    speedup=2.0
                )
                from data_ferret.server.kernel_manager import TestCodeData
                mock_test.return_value = TestCodeData(ok=True, result=test_result)

                from data_ferret.server.commands.optimize import StatsAggregator
                stats_agg = StatsAggregator(model="gpt-4")

                is_valid, validated_snippets, timing_data = await cmd._validate_optimization_with_retry(
                    cell,
                    cells,
                    original_snippets,
                    optimized_snippets,
                    "gpt-4",
                    mock_tools,
                    mock_kernel_client_success,
                    mock_dependencies_dict,
                    stats_agg
                )

        assert is_valid is True
        assert len(validated_snippets) == 1
        assert timing_data is not None
        assert timing_data["speedup"] == 2.0

    @pytest.mark.asyncio
    async def test_validation_fail_then_repair_success(
        self,
        sample_notebook_with_optimization_plan,
        mock_dependencies_dict,
        mock_ferret_stats
    ):
        """Test validation fails, then repair succeeds."""
        cmd = OptimizeCommand()
        nb = sample_notebook_with_optimization_plan
        cells = nb["cells"]
        cell = cells[0]

        original_snippets = [
            CodeSnippet(
                cell_id="cell-1",
                function_name="compute_distance",
                source="def compute_distance(x, y):\n    return []"
            )
        ]

        optimized_snippets = [
            CodeSnippet(
                cell_id="cell-1",
                function_name="compute_distance",
                source="import numpy as np\ndef compute_distance(x, y):\n    return np.array([])"
            )
        ]

        # Track call count for test_code
        call_count = [0]

        def mock_test_code(*args, **kwargs):
            call_count[0] += 1
            from data_ferret.kernel.types import TestCodeSuccess, DiffResult
            from data_ferret.server.kernel_manager import TestCodeData

            if call_count[0] == 1:
                # First call: validation fails (type mismatch)
                diff = DiffResult(
                    differences={"result": ("list", "ndarray")}
                )
                test_result = TestCodeSuccess(
                    diff=diff,
                    original_duration=1.0,
                    modified_duration=0.5,
                    speedup=2.0
                )
                return TestCodeData(ok=True, result=test_result)
            else:
                # Second call: after repair, validation succeeds
                diff = DiffResult(differences={})
                test_result = TestCodeSuccess(
                    diff=diff,
                    original_duration=1.0,
                    modified_duration=0.5,
                    speedup=2.0
                )
                return TestCodeData(ok=True, result=test_result)

        with patch('data_ferret.server.commands.optimize.NotebookTools') as MockTools:
            mock_tools = Mock()
            mock_tools.tools = Mock(return_value=[])
            MockTools.return_value.__enter__ = Mock(return_value=mock_tools)
            MockTools.return_value.__exit__ = Mock(return_value=False)

            with patch.object(cmd, '_send_test_code_comm', side_effect=mock_test_code):
                # Mock the repair method
                with patch.object(cmd, '_repair_optimization') as mock_repair:
                    repaired_snippets = [
                        CodeSnippet(
                            cell_id="cell-1",
                            function_name="compute_distance",
                            source="import numpy as np\ndef compute_distance(x, y):\n    return np.array([]).tolist()"
                        )
                    ]
                    mock_repair.return_value = repaired_snippets

                    from data_ferret.server.commands.optimize import StatsAggregator
                    stats_agg = StatsAggregator(model="gpt-4")

                    mock_kernel = Mock()

                    is_valid, validated_snippets, timing_data = await cmd._validate_optimization_with_retry(
                        cell,
                        cells,
                        original_snippets,
                        optimized_snippets,
                        "gpt-4",
                        mock_tools,
                        mock_kernel,
                        mock_dependencies_dict,
                        stats_agg
                    )

        assert is_valid is True
        assert len(validated_snippets) == 1
        assert ".tolist()" in validated_snippets[0].source


# ============================================================================
# Tests for Repair Logic
# ============================================================================

class TestRepairLogic:
    """Test repair logic with combined format."""

    @pytest.mark.asyncio
    async def test_repair_optimization_combined_format(
        self,
        sample_notebook_with_optimization_plan,
        mock_ferret_stats
    ):
        """Test repair using combined format."""
        cmd = OptimizeCommand()
        nb = sample_notebook_with_optimization_plan
        cells = nb["cells"]
        cell = cells[0]

        original_snippets = [
            CodeSnippet(
                cell_id="cell-1",
                function_name="compute_distance",
                source="def compute_distance(x, y):\n    return []"
            )
        ]

        failed_snippets = [
            CodeSnippet(
                cell_id="cell-1",
                function_name="compute_distance",
                source="import numpy as np\ndef compute_distance(x, y):\n    return np.array([])"
            )
        ]

        validation_error = "Type mismatch: expected list, got ndarray"

        # Mock the FerretAgent for repair
        with patch('data_ferret.server.commands.optimize.FerretAgent') as MockAgent:
            mock_agent_instance = Mock()
            mock_agent_instance.run = AsyncMock(
                return_value=(get_repair_response(), mock_ferret_stats)
            )
            MockAgent.__getitem__.return_value = MockAgent
            MockAgent.return_value = mock_agent_instance

            with patch('data_ferret.server.commands.optimize.NotebookTools') as MockTools:
                mock_tools = Mock()
                mock_tools.tools = Mock(return_value=[])
                MockTools.return_value.__enter__ = Mock(return_value=mock_tools)
                MockTools.return_value.__exit__ = Mock(return_value=False)

                from data_ferret.server.commands.optimize import StatsAggregator
                stats_agg = StatsAggregator(model="gpt-4")

                repaired = await cmd._repair_optimization(
                    cells,
                    cell,
                    original_snippets,
                    failed_snippets,
                    validation_error,
                    "gpt-4",
                    mock_tools,
                    stats_agg
                )

        assert repaired is not None
        assert len(repaired) == 1
        assert ".tolist()" in repaired[0].source


# ============================================================================
# Tests for Full End-to-End Flow
# ============================================================================

class TestEndToEndOptimization:
    """Test complete optimization flow from start to finish."""

    @pytest.mark.asyncio
    async def test_optimize_cell_complete_flow(
        self,
        sample_notebook_with_optimization_plan,
        mock_dependencies_dict,
        mock_ferret_stats
    ):
        """Test complete optimize_cell flow with mocked LLM."""
        cmd = OptimizeCommand()
        nb = sample_notebook_with_optimization_plan

        # Mock the FerretAgent
        with patch('data_ferret.server.commands.optimize.FerretAgent') as MockAgent:
            mock_agent_instance = Mock()
            mock_agent_instance.run = AsyncMock(
                return_value=(get_single_function_optimization_response(), mock_ferret_stats)
            )
            MockAgent.__getitem__.return_value = MockAgent
            MockAgent.return_value = mock_agent_instance

            with patch('data_ferret.server.commands.optimize.NotebookTools') as MockTools:
                mock_tools = Mock()
                mock_tools.tools = Mock(return_value=[])
                MockTools.return_value.__enter__ = Mock(return_value=mock_tools)
                MockTools.return_value.__exit__ = Mock(return_value=False)

                # Mock successful validation
                with patch.object(cmd, '_send_test_code_comm') as mock_test:
                    from data_ferret.kernel.types import TestCodeSuccess, DiffResult
                    from data_ferret.server.kernel_manager import TestCodeData
                    test_result = TestCodeSuccess(
                        diff=DiffResult(differences={}),
                        original_duration=1.0,
                        modified_duration=0.3,
                        speedup=3.33
                    )
                    mock_test.return_value = TestCodeData(ok=True, result=test_result)

                    mock_kernel = Mock()

                    cell_id, result = await cmd.optimize_cell(
                        index=0,
                        nb=nb,
                        model="gpt-4",
                        kernel_client=mock_kernel,
                        dependencies_dict=mock_dependencies_dict
                    )

        # Verify results
        assert cell_id == "cell-1"
        assert len(result.optimized_code) == 1
        assert result.optimized_code[0].cell_id == "cell-1"
        assert result.optimized_code[0].function_name == "compute_distance"
        assert "numpy" in result.optimized_code[0].source
        assert result.speedup is not None
        assert result.speedup > 3.0

    @pytest.mark.asyncio
    async def test_optimize_cells_multiple_cells(
        self,
        sample_notebook_with_multiple_optimizations,
        mock_ferret_stats
    ):
        """Test optimizing multiple cells in a notebook."""
        cmd = OptimizeCommand()
        nb = sample_notebook_with_multiple_optimizations

        dependencies_dict = {
            "cell-1": CellDependencies(
                cell_id="cell-1",
                globals_read=set(),
                globals_written={"pairwise_distance", "normalize_data"},
            ),
            "cell-2": CellDependencies(
                cell_id="cell-2",
                globals_read={"pairwise_distance", "normalize_data"},
                globals_written={"normalized", "distances"},
            )
        }

        # Mock the FerretAgent
        with patch('data_ferret.server.commands.optimize.FerretAgent') as MockAgent:
            mock_agent_instance = Mock()
            mock_agent_instance.run = AsyncMock(
                return_value=(get_multiple_function_optimization_response(), mock_ferret_stats)
            )
            MockAgent.__getitem__.return_value = MockAgent
            MockAgent.return_value = mock_agent_instance

            with patch('data_ferret.server.commands.optimize.NotebookTools') as MockTools:
                mock_tools = Mock()
                mock_tools.tools = Mock(return_value=[])
                MockTools.return_value.__enter__ = Mock(return_value=mock_tools)
                MockTools.return_value.__exit__ = Mock(return_value=False)

                # Mock successful validation
                with patch.object(cmd, '_send_test_code_comm') as mock_test:
                    from data_ferret.kernel.types import TestCodeSuccess, DiffResult
                    from data_ferret.server.kernel_manager import TestCodeData
                    test_result = TestCodeSuccess(
                        diff=DiffResult(differences={}),
                        original_duration=2.0,
                        modified_duration=0.5,
                        speedup=4.0
                    )
                    mock_test.return_value = TestCodeData(ok=True, result=test_result)

                    mock_kernel = Mock()

                    # Mock analyze_notebook
                    with patch('data_ferret.server.commands.optimize.analyze_notebook', return_value=dependencies_dict):
                        new_nb, total_cost, cell_timing = await cmd.optimize_cells(
                            nb=nb,
                            model="gpt-4",
                            kernel_client=mock_kernel
                        )

        # Verify results
        assert new_nb is not None
        assert total_cost >= 0

        # Check that optimizations were applied
        cell1_source = new_nb["cells"][0]["source"]
        assert "cdist" in cell1_source or "np.mean" in cell1_source


# ============================================================================
# Tests for Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_parse_response_with_no_sections(self):
        """Test parsing response with missing section markers."""
        cmd = OptimizeCommand()

        response = create_mock_optimization_response(
            optimized_code="def func():\n    return 1",  # No section markers!
            explanation="Missing markers"
        )

        opt_plan = [
            OptimizationStep(
                target_cell_id="cell-1",
                function_name="func",
                description=[""]
            )
        ]

        # Should raise validation error
        with pytest.raises(ValueError):
            cmd._parse_combined_optimization_response(response, opt_plan)

    @pytest.mark.asyncio
    async def test_optimize_cell_with_no_optimization_plan(self, sample_notebook_with_optimization_plan):
        """Test optimizing a cell with no optimization plan."""
        cmd = OptimizeCommand()
        nb = new_notebook()
        cell = new_code_cell(source="print('hello')", id="cell-1")
        nb["cells"] = [cell]

        cell_id, result = await cmd.optimize_cell(
            index=0,
            nb=nb,
            model="gpt-4"
        )

        assert cell_id == "cell-1"
        assert len(result.optimized_code) == 0
        assert result.stats.cost == 0.0

    def test_build_combined_input_with_missing_cell(self):
        """Test building combined input when target cell doesn't exist."""
        cmd = OptimizeCommand()

        opt_plan = [
            OptimizationStep(
                target_cell_id="nonexistent-cell",
                function_name="func",
                description=["Optimize"]
            )
        ]

        cell_map = {}

        combined = cmd._build_combined_input_code(opt_plan, cell_map)

        # Should skip the missing cell
        assert combined == ""

    def test_gather_environment_info_with_no_profile(self, sample_notebook_with_optimization_plan):
        """Test gathering environment info when cell has no profile."""
        cmd = OptimizeCommand()
        nb = sample_notebook_with_optimization_plan

        # Remove profile metadata
        cell = nb["cells"][0]
        if "ferret" in cell["metadata"]:
            del cell["metadata"]["ferret"]["profile"]

        cells = nb["cells"]
        cell_map = {c["id"]: c for c in cells}

        opt_plan = [
            OptimizationStep(
                target_cell_id="cell-1",
                function_name="compute_distance",
                description=["Optimize"]
            )
        ]

        env_info = cmd._gather_environment_info(opt_plan, cell_map)

        # Should return empty string
        assert env_info == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
