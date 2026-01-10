"""
Tests for MultiIndex DataFrame handling across deepcopyable, deepcopy, diff, and checkpoint.
"""

import pytest
import pandas as pd
import numpy as np

from flowbook.kernel.deepcopyable import check_deepcopyable
from flowbook.kernel.deepcopy import deepcopy
from flowbook.kernel.diff import Diff, _get_column_as_series
from flowbook.kernel.checkpoint import (
    Checkpoints,
    _collect_reachable_ids,
    _collect_reachable_ids_with_paths,
)


# =============================================================================
# Fixtures for MultiIndex DataFrames
# =============================================================================


@pytest.fixture
def multiindex_df():
    """Create a DataFrame with MultiIndex columns."""
    arrays = [['A', 'A', 'B', 'B'], ['one', 'two', 'one', 'two']]
    tuples = list(zip(*arrays))
    index = pd.MultiIndex.from_tuples(tuples)
    df = pd.DataFrame([[1, 2, 3, 4], [5, 6, 7, 8]], columns=index)
    return df


@pytest.fixture
def multiindex_df_with_objects():
    """Create a DataFrame with MultiIndex columns and object dtype."""
    arrays = [['A', 'A', 'B', 'B'], ['one', 'two', 'one', 'two']]
    tuples = list(zip(*arrays))
    index = pd.MultiIndex.from_tuples(tuples)
    df = pd.DataFrame(
        [[{'a': 1}, 'x', [1, 2], 'y'],
         [{'b': 2}, 'z', [3, 4], 'w']],
        columns=index
    )
    return df


@pytest.fixture
def three_level_multiindex_df():
    """Create a DataFrame with 3-level MultiIndex columns."""
    arrays = [
        ['X', 'X', 'Y', 'Y'],
        ['A', 'A', 'B', 'B'],
        ['one', 'two', 'one', 'two']
    ]
    tuples = list(zip(*arrays))
    index = pd.MultiIndex.from_tuples(tuples)
    df = pd.DataFrame([[1, 2, 3, 4], [5, 6, 7, 8]], columns=index)
    return df


# =============================================================================
# Tests for check_deepcopyable
# =============================================================================


class TestIsDeepCopyableMultiIndex:
    """Tests for check_deepcopyable with MultiIndex DataFrames."""

    def test_multiindex_df_is_copyable(self, multiindex_df):
        """MultiIndex DataFrame with numeric data should be copyable."""
        assert check_deepcopyable(multiindex_df) is None

    def test_multiindex_df_with_objects_is_copyable(self, multiindex_df_with_objects):
        """MultiIndex DataFrame with object dtype should be copyable."""
        assert check_deepcopyable(multiindex_df_with_objects) is None

    def test_three_level_multiindex_is_copyable(self, three_level_multiindex_df):
        """3-level MultiIndex DataFrame should be copyable."""
        assert check_deepcopyable(three_level_multiindex_df) is None

    def test_multiindex_df_with_non_copyable_objects(self):
        """MultiIndex DataFrame with non-copyable objects should not be copyable."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()

        arrays = [['A', 'A'], ['one', 'two']]
        tuples = list(zip(*arrays))
        index = pd.MultiIndex.from_tuples(tuples)
        df = pd.DataFrame([[fig, ax]], columns=index)

        assert check_deepcopyable(df) is not None
        plt.close(fig)


# =============================================================================
# Tests for deepcopy
# =============================================================================


class TestDeepCopyMultiIndex:
    """Tests for deepcopy with MultiIndex DataFrames."""

    def test_deepcopy_multiindex_df(self, multiindex_df):
        """Deepcopy should work for MultiIndex DataFrames."""
        memo = {}
        copy = deepcopy(multiindex_df, memo)

        # Check structure preserved
        assert copy.columns.equals(multiindex_df.columns)
        assert copy.shape == multiindex_df.shape

        # Check values equal
        pd.testing.assert_frame_equal(copy, multiindex_df)

        # Check independence
        copy.iloc[0, 0] = 999
        assert multiindex_df.iloc[0, 0] != 999

    def test_deepcopy_multiindex_df_with_objects(self, multiindex_df_with_objects):
        """Deepcopy should handle object dtype with MultiIndex columns."""
        memo = {}
        copy = deepcopy(multiindex_df_with_objects, memo)

        # Check structure preserved
        assert copy.columns.equals(multiindex_df_with_objects.columns)

        # Check independence - modify mutable object in copy
        copy.iloc[0, 0]['a'] = 999
        assert multiindex_df_with_objects.iloc[0, 0]['a'] != 999

    def test_deepcopy_three_level_multiindex(self, three_level_multiindex_df):
        """Deepcopy should work for 3-level MultiIndex DataFrames."""
        memo = {}
        copy = deepcopy(three_level_multiindex_df, memo)

        pd.testing.assert_frame_equal(copy, three_level_multiindex_df)


# =============================================================================
# Tests for Diff
# =============================================================================


class TestDiffMultiIndex:
    """Tests for Diff with MultiIndex DataFrames."""

    def test_diff_identical_multiindex_dfs(self, multiindex_df):
        """Identical MultiIndex DataFrames should have no differences."""
        copy = multiindex_df.copy()

        differ = Diff(strict=False, report_close=False)
        result = differ.diff({'df': multiindex_df}, {'df': copy})

        assert not result.differences

    def test_diff_modified_multiindex_df(self, multiindex_df):
        """Modified MultiIndex DataFrame should be detected."""
        copy = multiindex_df.copy()
        copy.iloc[0, 0] = 999

        differ = Diff(strict=False, report_close=False)
        result = differ.diff({'df': multiindex_df}, {'df': copy})

        assert 'df' in result.differences

    def test_diff_three_level_multiindex(self, three_level_multiindex_df):
        """Diff should work with 3-level MultiIndex DataFrames."""
        copy = three_level_multiindex_df.copy()

        differ = Diff(strict=False, report_close=False)
        result = differ.diff({'df': three_level_multiindex_df}, {'df': copy})

        assert not result.differences

    def test_get_column_as_series_multiindex(self, multiindex_df):
        """Helper function should correctly extract Series from MultiIndex columns."""
        for col in multiindex_df.columns:
            series = _get_column_as_series(multiindex_df, col)
            assert isinstance(series, pd.Series)
            assert len(series) == len(multiindex_df)


# =============================================================================
# Tests for checkpoint alias detection
# =============================================================================


class TestCheckpointAliasMultiIndex:
    """Tests for checkpoint alias detection with MultiIndex DataFrames."""

    def test_collect_reachable_ids_multiindex(self, multiindex_df):
        """Should collect IDs from MultiIndex DataFrame without error."""
        visited = set()
        _collect_reachable_ids(multiindex_df, visited)

        # Should have collected some IDs
        assert len(visited) > 0

    def test_collect_reachable_ids_multiindex_with_objects(self, multiindex_df_with_objects):
        """Should collect IDs from MultiIndex DataFrame with objects."""
        visited = set()
        _collect_reachable_ids(multiindex_df_with_objects, visited)

        # Should have collected IDs for the mutable objects
        assert len(visited) > 0

    def test_collect_reachable_ids_with_paths_multiindex(self, multiindex_df_with_objects):
        """Should collect IDs with paths from MultiIndex DataFrame."""
        visited = set()
        id_to_path = {}
        _collect_reachable_ids_with_paths(
            multiindex_df_with_objects, 'df', visited, id_to_path
        )

        # Should have collected IDs
        assert len(visited) > 0


# =============================================================================
# Tests for Checkpoints
# =============================================================================


class TestCheckpointsMultiIndex:
    """Tests for Checkpoints with MultiIndex DataFrames."""

    def test_save_restore_multiindex_df(self, multiindex_df):
        """Should save and restore MultiIndex DataFrame."""
        user_ns = {'df': multiindex_df.copy()}

        cp = Checkpoints()
        cp.save('test', user_ns)

        # Modify
        user_ns['df'].iloc[0, 0] = 999
        assert user_ns['df'].iloc[0, 0] == 999

        # Restore
        cp.restore('test', user_ns)

        # Check restored correctly
        pd.testing.assert_frame_equal(user_ns['df'], multiindex_df)

    def test_save_restore_multiindex_df_with_objects(self, multiindex_df_with_objects):
        """Should save and restore MultiIndex DataFrame with objects."""
        original_value = multiindex_df_with_objects.iloc[0, 0].copy()
        user_ns = {'df': multiindex_df_with_objects.copy(deep=True)}
        # Need deep copy for object columns
        for i in range(len(user_ns['df'].columns)):
            if user_ns['df'].iloc[:, i].dtype == object:
                user_ns['df'].iloc[:, i] = user_ns['df'].iloc[:, i].apply(
                    lambda x: x.copy() if hasattr(x, 'copy') else x
                )

        cp = Checkpoints()
        cp.save('test', user_ns)

        # Modify
        user_ns['df'].iloc[0, 0]['a'] = 999
        assert user_ns['df'].iloc[0, 0]['a'] == 999

        # Restore
        cp.restore('test', user_ns)

        # Check restored correctly
        assert user_ns['df'].iloc[0, 0] == original_value

    def test_checkpoint_diff_multiindex(self, multiindex_df):
        """Checkpoint.diff should work with MultiIndex DataFrames."""
        from flowbook.kernel.checkpoint import Checkpoint

        user_ns_a = {'df': multiindex_df.copy()}
        user_ns_b = {'df': multiindex_df.copy()}
        user_ns_b['df'].iloc[0, 0] = 999

        cp = Checkpoints()

        cp.save('a', user_ns_a)
        cp.save('b', user_ns_b)

        checkpoint_a = cp.get('a')
        checkpoint_b = cp.get('b')

        diff_result = Checkpoint.diff(checkpoint_a, checkpoint_b)

        # Should detect the difference
        assert 'df' in diff_result.differences


# =============================================================================
# Tests for edge cases
# =============================================================================


class TestMultiIndexEdgeCases:
    """Edge case tests for MultiIndex handling."""

    def test_empty_multiindex_df(self):
        """Empty MultiIndex DataFrame should work."""
        arrays = [['A', 'B'], ['one', 'two']]
        tuples = list(zip(*arrays))
        index = pd.MultiIndex.from_tuples(tuples)
        df = pd.DataFrame(columns=index)

        assert check_deepcopyable(df) is None

        memo = {}
        copy = deepcopy(df, memo)
        assert copy.columns.equals(df.columns)

    def test_single_column_multiindex(self):
        """Single column MultiIndex DataFrame should work."""
        index = pd.MultiIndex.from_tuples([('A', 'one')])
        df = pd.DataFrame([[1], [2]], columns=index)

        assert check_deepcopyable(df) is None

        memo = {}
        copy = deepcopy(df, memo)
        pd.testing.assert_frame_equal(copy, df)

    def test_duplicate_multiindex_columns(self):
        """DataFrame with duplicate MultiIndex columns should work."""
        # Note: This is unusual but pandas allows it
        arrays = [['A', 'A'], ['one', 'one']]  # Duplicate columns
        tuples = list(zip(*arrays))
        index = pd.MultiIndex.from_tuples(tuples)
        df = pd.DataFrame([[1, 2], [3, 4]], columns=index)

        assert check_deepcopyable(df) is None

        memo = {}
        copy = deepcopy(df, memo)
        assert copy.shape == df.shape

    def test_multiindex_with_mixed_types(self):
        """MultiIndex with mixed level types should work."""
        arrays = [[1, 1, 2, 2], ['a', 'b', 'a', 'b']]
        tuples = list(zip(*arrays))
        index = pd.MultiIndex.from_tuples(tuples)
        df = pd.DataFrame([[1, 2, 3, 4]], columns=index)

        assert check_deepcopyable(df) is None

        memo = {}
        copy = deepcopy(df, memo)
        pd.testing.assert_frame_equal(copy, df)

    def test_multiindex_with_none_level(self):
        """MultiIndex with None in levels should work."""
        arrays = [['A', 'A', None], ['one', 'two', 'three']]
        tuples = list(zip(*arrays))
        index = pd.MultiIndex.from_tuples(tuples)
        df = pd.DataFrame([[1, 2, 3]], columns=index)

        assert check_deepcopyable(df) is None

        memo = {}
        copy = deepcopy(df, memo)
        assert copy.shape == df.shape
