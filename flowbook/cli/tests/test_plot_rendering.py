"""Tests for flowbook.cli.plot_rendering functions."""

import pytest

# Import the rendering functions
from flowbook.cli.plot_rendering import (
    render_plot1,
    render_plot2,
    render_plot3,
    render_plot4,
    render_plot5,
    render_plot6,
    render_combined_6panel,
)

# Import the plot data classes
from flowbook.cli.models import (
    Plot1Data,
    Plot2Data,
    Plot3Data,
    Plot4Data,
    Plot5Data,
    Plot6Data,
)


@pytest.fixture
def mock_axes():
    """Create a mock matplotlib axes for testing."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    yield ax
    plt.close(fig)


@pytest.fixture
def sample_plot1_data():
    """Sample Plot1Data for testing."""
    return Plot1Data(
        cells=[1, 2, 3],
        run_time_sec=[1.0, 2.0, 1.5],
        state_time_sec=[0.1, 0.2, 0.15],
        check_time_sec=[0.05, 0.1, 0.08],
        other_time_sec=[0.02, 0.03, 0.02],
        initial_count=3,
    )


@pytest.fixture
def sample_plot3_data():
    """Sample Plot3Data for testing."""
    return Plot3Data(
        cells=[1, 2, 3],
        user_ns_mb=[100, 150, 200],
        gpu_mb=[0, 0, 0],
        overhead_mb=[10, 20, 30],
        has_baseline=True,
        peak_overhead_mb=30,
        peak_overhead_pct=15.0,
        peak_cell=2,
        initial_count=3,
    )


@pytest.fixture
def sample_plot4_data():
    """Sample Plot4Data for testing."""
    return Plot4Data(
        cells=[1, 2, 3],
        namespace_mb=[100, 150, 200],
        gpu_mb=[0, 0, 0],
        var_series={"df": [10, 15, 20], "X": [5, 8, 10]},
        vars_ordered=["df", "X"],
        var_types={"df": "DataFrame", "X": "ndarray"},
        initial_count=3,
    )


@pytest.fixture
def sample_plot6_data():
    """Sample Plot6Data for testing."""
    return Plot6Data(
        cells=[1, 2, 3, 4, 5],
        ratios=[0.1, 0.2, 0.3, 0.4, 0.5],
        initial_count=5,
    )


class TestRenderPlot1:
    """Tests for render_plot1."""

    def test_renders_without_error(self, mock_axes, sample_plot1_data):
        """Rendering completes without error."""
        render_plot1(mock_axes, sample_plot1_data)
        # If we get here without exception, test passes

    def test_sets_labels(self, mock_axes, sample_plot1_data):
        """Labels are set correctly."""
        render_plot1(mock_axes, sample_plot1_data)
        assert mock_axes.get_xlabel() != ""
        assert mock_axes.get_ylabel() != ""


class TestRenderPlot3:
    """Tests for render_plot3."""

    def test_renders_without_error(self, mock_axes, sample_plot3_data):
        """Rendering completes without error."""
        render_plot3(mock_axes, sample_plot3_data)

    def test_shows_expected_labels(self, mock_axes, sample_plot3_data):
        """Shows correct labels for the three layers."""
        render_plot3(mock_axes, sample_plot3_data)
        legend = mock_axes.get_legend()
        labels = [t.get_text() for t in legend.get_texts()]
        assert "User Namespace" in labels
        assert "GPU Memory" in labels
        assert "FlowBook Overhead" in labels


class TestRenderPlot4:
    """Tests for render_plot4."""

    def test_renders_without_error(self, mock_axes, sample_plot4_data):
        """Rendering completes without error."""
        render_plot4(mock_axes, sample_plot4_data)

    def test_includes_namespace_layer(self, mock_axes, sample_plot4_data):
        """Namespace layer is included."""
        render_plot4(mock_axes, sample_plot4_data)
        legend = mock_axes.get_legend()
        labels = [t.get_text() for t in legend.get_texts()]
        assert "Namespace" in labels


class TestRenderPlot6:
    """Tests for render_plot6."""

    def test_renders_without_error(self, mock_axes, sample_plot6_data):
        """Rendering completes without error."""
        render_plot6(mock_axes, sample_plot6_data)

    def test_shows_bars(self, mock_axes, sample_plot6_data):
        """Bar chart is rendered with expected data."""
        render_plot6(mock_axes, sample_plot6_data)
        # Should have bar containers
        assert len(mock_axes.containers) > 0


class TestRenderCombined6Panel:
    """Tests for render_combined_6panel."""

    def test_renders_without_error_with_all_data(
        self, sample_plot1_data, sample_plot3_data, sample_plot4_data, sample_plot6_data
    ):
        """Rendering completes without error when all data provided."""
        import matplotlib.pyplot as plt

        fig, axes_2d = plt.subplots(3, 2, figsize=(14, 18))
        axes = [
            axes_2d[0, 0], axes_2d[0, 1],
            axes_2d[1, 0], axes_2d[1, 1],
            axes_2d[2, 0], axes_2d[2, 1],
        ]

        # Create minimal Plot2Data and Plot5Data
        p2 = Plot2Data(
            cells=[1, 2, 3],
            var_series={"df": [0.1, 0.2, 0.15]},
            vars_ordered=["df"],
            initial_count=3,
        )
        p5 = Plot5Data(
            cells=[1, 2, 3],
            state_sec=[0.1, 0.2, 0.15],
            check_sec=[0.05, 0.1, 0.08],
            other_sec=[0.02, 0.03, 0.02],
            initial_count=3,
        )

        render_combined_6panel(
            fig, axes,
            sample_plot1_data, p2,
            sample_plot3_data, sample_plot4_data,
            p5, sample_plot6_data,
        )
        plt.close(fig)

    def test_handles_none_data_gracefully(self):
        """Rendering handles None data without error."""
        import matplotlib.pyplot as plt

        fig, axes_2d = plt.subplots(3, 2, figsize=(14, 18))
        axes = [
            axes_2d[0, 0], axes_2d[0, 1],
            axes_2d[1, 0], axes_2d[1, 1],
            axes_2d[2, 0], axes_2d[2, 1],
        ]

        # All None should show "No data" messages
        render_combined_6panel(
            fig, axes,
            None, None, None, None, None, None,
        )
        plt.close(fig)
