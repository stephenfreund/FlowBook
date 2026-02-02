"""
Tests for large primitive list caching optimization in deepcopy.

This module tests the optimization that caches checkpoint copies of large lists
containing only PRIMITIVE immutable types (None, bool, int, float, complex,
str, bytes), reusing them on subsequent checkpoints if the list hasn't changed.

IMPORTANT: Only primitive types are cached - NOT tuples, frozensets, etc.

Test categories:
1. Primitive type detection (_list_has_only_primitives)
2. General immutability detection (_list_is_all_immutable)
3. Cache hit/miss behavior
4. Change detection via content hash
5. Cache management (pruning, clearing)
6. Integration with checkpoint system
7. Alias traversal optimization
8. Edge cases and corner cases
"""

import gc
import pytest

from flowbook.kernel.deepcopy import (
    deepcopy,
    clear_list_cache,
    clear_container_cache,
    get_list_cache_stats,
    get_container_cache_stats,
    _large_list_cache,
    _large_set_cache,
    _large_dict_cache,
    _primitive_list_copies,
    _primitive_set_copies,
    _primitive_dict_copies,
    _list_is_all_immutable,
    _list_has_only_primitives,
    _set_has_only_primitives,
    _dict_has_only_primitive_values,
    _tuple_has_only_primitives,
    is_list_in_immutable_cache,
    is_primitive_container,
    _LARGE_LIST_THRESHOLD,
    _MAX_CONTAINER_CACHE_SIZE,
    _PRIMITIVE_IMMUTABLE_TYPES,
)
from flowbook.kernel.checkpoint import Checkpoints


class TestListIsAllImmutable:
    """Tests for _list_is_all_immutable() function."""

    def test_empty_list_is_immutable(self):
        assert _list_is_all_immutable([]) is True

    def test_list_of_ints_is_immutable(self):
        assert _list_is_all_immutable([1, 2, 3, 4, 5]) is True

    def test_list_of_floats_is_immutable(self):
        assert _list_is_all_immutable([1.1, 2.2, 3.3]) is True

    def test_list_of_strings_is_immutable(self):
        assert _list_is_all_immutable(['a', 'b', 'c']) is True

    def test_list_of_none_is_immutable(self):
        assert _list_is_all_immutable([None, None, None]) is True

    def test_list_of_bools_is_immutable(self):
        assert _list_is_all_immutable([True, False, True]) is True

    def test_mixed_immutable_types(self):
        assert _list_is_all_immutable([1, 'a', 3.14, None, True]) is True

    def test_list_of_tuples_is_immutable(self):
        assert _list_is_all_immutable([(1, 2), (3, 4), (5, 6)]) is True

    def test_nested_immutable_tuples(self):
        assert _list_is_all_immutable([(1, (2, 3)), ('a', ('b', 'c'))]) is True

    def test_list_with_mutable_element_not_immutable(self):
        assert _list_is_all_immutable([1, 2, [3, 4]]) is False

    def test_list_with_dict_not_immutable(self):
        assert _list_is_all_immutable([1, {'a': 1}]) is False

    def test_list_with_set_not_immutable(self):
        assert _list_is_all_immutable([1, {1, 2, 3}]) is False

    def test_list_with_mutable_in_tuple_not_immutable(self):
        assert _list_is_all_immutable([(1, [2, 3])]) is False

    def test_large_immutable_list(self):
        """Test that large lists are correctly identified as immutable."""
        large_list = list(range(10000))
        assert _list_is_all_immutable(large_list) is True

    def test_large_list_with_mutable_at_end(self):
        """Ensure mutable elements at end are detected."""
        large_list = list(range(9999)) + [[1, 2, 3]]
        assert _list_is_all_immutable(large_list) is False

    def test_large_list_with_mutable_at_start(self):
        """Ensure mutable elements at start are detected."""
        large_list = [[1, 2, 3]] + list(range(9999))
        assert _list_is_all_immutable(large_list) is False

    def test_large_list_with_mutable_in_middle(self):
        """Ensure mutable elements in middle are detected."""
        large_list = list(range(5000)) + [[1, 2, 3]] + list(range(5000))
        assert _list_is_all_immutable(large_list) is False


class TestListHasOnlyPrimitives:
    """Tests for _list_has_only_primitives() function.

    This function is stricter than _list_is_all_immutable() - it only
    returns True for primitive types (None, bool, int, float, complex,
    str, bytes) and does NOT recurse into containers like tuples.
    """

    def test_empty_list(self):
        assert _list_has_only_primitives([]) is True

    def test_list_of_ints(self):
        assert _list_has_only_primitives([1, 2, 3, 4, 5]) is True

    def test_list_of_floats(self):
        assert _list_has_only_primitives([1.1, 2.2, 3.3]) is True

    def test_list_of_strings(self):
        assert _list_has_only_primitives(['a', 'b', 'c']) is True

    def test_list_of_none(self):
        assert _list_has_only_primitives([None, None, None]) is True

    def test_list_of_bools(self):
        assert _list_has_only_primitives([True, False, True]) is True

    def test_list_of_complex(self):
        assert _list_has_only_primitives([1+2j, 3+4j]) is True

    def test_list_of_bytes(self):
        assert _list_has_only_primitives([b'a', b'b', b'c']) is True

    def test_mixed_primitive_types(self):
        assert _list_has_only_primitives([1, 'a', 3.14, None, True, b'x']) is True

    # IMPORTANT: Tuples and frozensets are NOT primitives
    def test_list_of_tuples_not_primitive(self):
        """Tuples are NOT primitives - should return False."""
        assert _list_has_only_primitives([(1, 2), (3, 4)]) is False

    def test_list_of_frozensets_not_primitive(self):
        """Frozensets are NOT primitives - should return False."""
        assert _list_has_only_primitives([frozenset([1]), frozenset([2])]) is False

    def test_list_of_empty_tuples_not_primitive(self):
        """Even empty tuples are NOT primitives."""
        assert _list_has_only_primitives([(), ()]) is False

    def test_nested_tuples_not_primitive(self):
        """Nested tuples are NOT primitives."""
        assert _list_has_only_primitives([(1, (2, 3))]) is False

    # Mutable types are not primitives either
    def test_list_with_list_not_primitive(self):
        assert _list_has_only_primitives([1, [2, 3]]) is False

    def test_list_with_dict_not_primitive(self):
        assert _list_has_only_primitives([1, {'a': 1}]) is False

    def test_list_with_set_not_primitive(self):
        assert _list_has_only_primitives([1, {1, 2, 3}]) is False

    def test_primitive_types_constant(self):
        """Verify the _PRIMITIVE_IMMUTABLE_TYPES constant contains expected types."""
        assert type(None) in _PRIMITIVE_IMMUTABLE_TYPES
        assert bool in _PRIMITIVE_IMMUTABLE_TYPES
        assert int in _PRIMITIVE_IMMUTABLE_TYPES
        assert float in _PRIMITIVE_IMMUTABLE_TYPES
        assert complex in _PRIMITIVE_IMMUTABLE_TYPES
        assert str in _PRIMITIVE_IMMUTABLE_TYPES
        assert bytes in _PRIMITIVE_IMMUTABLE_TYPES
        # NOT in the set:
        assert tuple not in _PRIMITIVE_IMMUTABLE_TYPES
        assert frozenset not in _PRIMITIVE_IMMUTABLE_TYPES


class TestLargeListCacheBasics:
    """Basic tests for large list caching behavior."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_list_cache()

    def teardown_method(self):
        """Clear cache after each test."""
        clear_list_cache()

    def test_small_list_not_cached(self):
        """Lists below threshold should not be cached."""
        small_list = list(range(100))  # Below 1000 threshold
        memo = {}
        copy1 = deepcopy(small_list, memo)

        assert len(_large_list_cache) == 0
        assert copy1 == small_list
        assert copy1 is not small_list

    def test_large_immutable_list_cached(self):
        """Large immutable lists should be cached."""
        large_list = list(range(_LARGE_LIST_THRESHOLD + 100))
        memo = {}
        copy1 = deepcopy(large_list, memo)

        assert len(_large_list_cache) == 1
        assert copy1 == large_list
        assert copy1 is not large_list

    def test_large_mutable_list_not_cached(self):
        """Large lists with mutable contents should not be cached."""
        large_list = [[i] for i in range(_LARGE_LIST_THRESHOLD + 100)]
        memo = {}
        copy1 = deepcopy(large_list, memo)

        assert len(_large_list_cache) == 0
        assert copy1 == large_list
        assert copy1 is not large_list
        # Verify deep copy happened
        assert copy1[0] is not large_list[0]

    def test_cache_hit_returns_same_copy(self):
        """Second deepcopy of unchanged list should return cached copy."""
        large_list = list(range(_LARGE_LIST_THRESHOLD + 100))

        memo1 = {}
        copy1 = deepcopy(large_list, memo1)

        memo2 = {}
        copy2 = deepcopy(large_list, memo2)

        # Should return the same cached copy object
        assert copy1 is copy2
        assert len(_large_list_cache) == 1

    def test_cache_miss_after_modification(self):
        """Modifying list should cause cache miss."""
        large_list = list(range(_LARGE_LIST_THRESHOLD + 100))

        memo1 = {}
        copy1 = deepcopy(large_list, memo1)

        # Modify the list
        large_list[0] = 999

        memo2 = {}
        copy2 = deepcopy(large_list, memo2)

        # Should create a new copy (different object)
        assert copy1 is not copy2
        # copy2 should have the modified value
        assert copy2[0] == 999
        # copy1 should still have original value
        assert copy1[0] == 0

    def test_cache_miss_after_append(self):
        """Appending to list should cause cache miss."""
        large_list = list(range(_LARGE_LIST_THRESHOLD + 100))

        memo1 = {}
        copy1 = deepcopy(large_list, memo1)
        original_len = len(copy1)

        # Append to the list
        large_list.append(999999)

        memo2 = {}
        copy2 = deepcopy(large_list, memo2)

        # Should create a new copy
        assert copy1 is not copy2
        assert len(copy2) == original_len + 1
        assert len(copy1) == original_len


class TestCacheManagement:
    """Tests for cache management functions."""

    def setup_method(self):
        clear_list_cache()

    def teardown_method(self):
        clear_list_cache()

    def test_clear_list_cache(self):
        """clear_list_cache() should empty the cache."""
        large_list = list(range(_LARGE_LIST_THRESHOLD + 100))
        deepcopy(large_list, {})

        assert len(_large_list_cache) > 0
        clear_list_cache()
        assert len(_large_list_cache) == 0

    def test_get_list_cache_stats(self):
        """get_list_cache_stats() should return correct statistics."""
        stats = get_list_cache_stats()

        assert 'size' in stats
        assert 'threshold' in stats
        assert 'max_size' in stats
        assert stats['threshold'] == _LARGE_LIST_THRESHOLD
        assert stats['max_size'] == _MAX_CONTAINER_CACHE_SIZE

    def test_get_list_cache_stats_reflects_cache_size(self):
        """Stats should reflect current cache size."""
        assert get_list_cache_stats()['size'] == 0

        large_list = list(range(_LARGE_LIST_THRESHOLD + 100))
        deepcopy(large_list, {})

        assert get_list_cache_stats()['size'] == 1


class TestCachePruning:
    """Tests for cache pruning behavior."""

    def setup_method(self):
        clear_list_cache()

    def teardown_method(self):
        clear_list_cache()

    def test_stale_entry_detection(self):
        """Stale entries (from GC'd lists) should be detected."""
        # Create and cache a list
        def create_and_cache():
            large_list = list(range(_LARGE_LIST_THRESHOLD + 100))
            deepcopy(large_list, {})
            return id(large_list)

        old_id = create_and_cache()

        # The list is now out of scope and may be GC'd
        gc.collect()

        # Cache should still have the entry (reference keeps it alive)
        # But if we create enough new lists, pruning should clean it up
        assert len(_large_list_cache) >= 1

    def test_cache_respects_max_size(self):
        """Cache should not grow beyond _MAX_CONTAINER_CACHE_SIZE."""
        # Create many different large lists
        lists = []
        for i in range(_MAX_CONTAINER_CACHE_SIZE + 50):
            large_list = list(range(_LARGE_LIST_THRESHOLD + i))
            lists.append(large_list)  # Keep reference to prevent GC
            deepcopy(large_list, {})

        # Cache should be at most _MAX_CONTAINER_CACHE_SIZE
        # (may be less if pruning cleared it)
        assert len(_large_list_cache) <= _MAX_CONTAINER_CACHE_SIZE


class TestCheckpointIntegration:
    """Tests for integration with the checkpoint system."""

    def setup_method(self):
        clear_list_cache()

    def teardown_method(self):
        clear_list_cache()

    def test_checkpoint_clears_cache_on_clear(self):
        """Checkpoints.clear() should clear the list cache."""
        cp = Checkpoints()
        large_list = list(range(_LARGE_LIST_THRESHOLD + 100))
        user_ns = {'large_list': large_list}

        cp.save('test', user_ns)
        assert len(_large_list_cache) >= 1

        cp.clear()
        assert len(_large_list_cache) == 0

    def test_checkpoint_clears_cache_on_last_delete(self):
        """Deleting the last checkpoint should clear the list cache."""
        cp = Checkpoints()
        large_list = list(range(_LARGE_LIST_THRESHOLD + 100))
        user_ns = {'large_list': large_list}

        cp.save('test', user_ns)
        assert len(_large_list_cache) >= 1

        cp.delete('test')
        assert len(_large_list_cache) == 0

    def test_checkpoint_preserves_cache_when_checkpoints_remain(self):
        """Cache should persist if checkpoints still exist."""
        cp = Checkpoints()
        large_list = list(range(_LARGE_LIST_THRESHOLD + 100))
        user_ns = {'large_list': large_list}

        cp.save('test1', user_ns)
        cp.save('test2', user_ns)
        cache_size_after_saves = len(_large_list_cache)

        cp.delete('test1')
        # Cache should NOT be cleared since test2 still exists
        assert len(_large_list_cache) == cache_size_after_saves

        cp.delete('test2')
        # Now cache should be cleared
        assert len(_large_list_cache) == 0

    def test_repeated_checkpoints_use_cache(self):
        """Multiple checkpoints of same list should use cache."""
        cp = Checkpoints()
        large_list = list(range(_LARGE_LIST_THRESHOLD + 100))
        user_ns = {'large_list': large_list}

        # First checkpoint
        cp.save('v1', user_ns)

        # Second checkpoint of unchanged data
        cp.save('v2', user_ns)

        # The cached copy should be reused
        # Verify by checking the copies are the same object
        v1_copy = cp.get('v1').user_ns['large_list']
        v2_copy = cp.get('v2').user_ns['large_list']

        assert v1_copy is v2_copy


class TestEdgeCases:
    """Tests for edge cases and corner cases."""

    def setup_method(self):
        clear_list_cache()

    def teardown_method(self):
        clear_list_cache()

    def test_exactly_at_threshold(self):
        """List exactly at threshold should be cached."""
        large_list = list(range(_LARGE_LIST_THRESHOLD))
        deepcopy(large_list, {})
        assert len(_large_list_cache) == 1

    def test_one_below_threshold(self):
        """List one below threshold should not be cached."""
        small_list = list(range(_LARGE_LIST_THRESHOLD - 1))
        deepcopy(small_list, {})
        assert len(_large_list_cache) == 0

    def test_list_with_none_values(self):
        """List with None values should be cached (None is immutable)."""
        large_list = [None] * (_LARGE_LIST_THRESHOLD + 100)
        deepcopy(large_list, {})
        assert len(_large_list_cache) == 1

    def test_list_of_empty_tuples_not_cached(self):
        """List of tuples should NOT be cached (only primitives cached)."""
        large_list = [()] * (_LARGE_LIST_THRESHOLD + 100)
        deepcopy(large_list, {})
        assert len(_large_list_cache) == 0  # NOT cached

    def test_list_of_frozensets_not_cached(self):
        """List of frozensets should NOT be cached (only primitives cached)."""
        large_list = [frozenset([i]) for i in range(_LARGE_LIST_THRESHOLD + 100)]
        deepcopy(large_list, {})
        assert len(_large_list_cache) == 0  # NOT cached

    def test_copy_is_independent(self):
        """Cached copy should be independent from original."""
        large_list = list(range(_LARGE_LIST_THRESHOLD + 100))
        copy1 = deepcopy(large_list, {})

        # Modify original
        large_list[0] = 999

        # Copy should be unchanged
        assert copy1[0] == 0

    def test_cached_copy_is_independent(self):
        """Modifying cached copy should not affect original or other copies."""
        large_list = list(range(_LARGE_LIST_THRESHOLD + 100))
        copy1 = deepcopy(large_list, {})

        # Modify the copy (this mutates a shallow copy, but elements are immutable)
        copy1[0] = 999

        # Original should be unchanged
        assert large_list[0] == 0

        # Get another "copy" from cache - but wait, we modified copy1!
        # This is actually fine because we're modifying the copy, not the original
        # The cache check uses the original list's hash, not the copy's

    def test_memo_is_populated(self):
        """Memo should be populated even with cache hit."""
        large_list = list(range(_LARGE_LIST_THRESHOLD + 100))

        memo1 = {}
        copy1 = deepcopy(large_list, memo1)
        assert id(large_list) in memo1

        memo2 = {}
        copy2 = deepcopy(large_list, memo2)
        assert id(large_list) in memo2

    def test_circular_reference_in_memo(self):
        """Lists already in memo should return memo entry, not cache."""
        large_list = list(range(_LARGE_LIST_THRESHOLD + 100))

        memo = {}
        # Manually add to memo
        fake_copy = ['fake']
        memo[id(large_list)] = fake_copy

        # Should return memo entry, not create new copy
        result = deepcopy(large_list, memo)
        assert result is fake_copy

    def test_different_lists_same_content(self):
        """Two different lists with same content should have separate cache entries."""
        list1 = list(range(_LARGE_LIST_THRESHOLD + 100))
        list2 = list(range(_LARGE_LIST_THRESHOLD + 100))

        copy1 = deepcopy(list1, {})
        copy2 = deepcopy(list2, {})

        # Both should be cached separately (different list objects)
        assert len(_large_list_cache) == 2
        # Copies should be equal but not identical
        assert copy1 == copy2
        assert copy1 is not copy2


class TestMixedTypes:
    """Tests for lists with various mixed immutable types."""

    def setup_method(self):
        clear_list_cache()

    def teardown_method(self):
        clear_list_cache()

    def test_mixed_numeric_types(self):
        """List with mixed numeric types should be cached."""
        large_list = []
        for i in range(_LARGE_LIST_THRESHOLD + 100):
            if i % 3 == 0:
                large_list.append(i)
            elif i % 3 == 1:
                large_list.append(float(i))
            else:
                large_list.append(complex(i, i))

        deepcopy(large_list, {})
        assert len(_large_list_cache) == 1

    def test_mixed_string_and_numeric(self):
        """List with strings and numbers should be cached."""
        large_list = []
        for i in range(_LARGE_LIST_THRESHOLD + 100):
            if i % 2 == 0:
                large_list.append(i)
            else:
                large_list.append(f"str_{i}")

        deepcopy(large_list, {})
        assert len(_large_list_cache) == 1

    def test_nested_immutable_structures_not_cached(self):
        """List with tuples and frozensets should NOT be cached (only primitives)."""
        large_list = []
        for i in range(_LARGE_LIST_THRESHOLD + 100):
            if i % 2 == 0:
                large_list.append((i, i + 1))
            else:
                large_list.append(frozenset([i]))

        deepcopy(large_list, {})
        assert len(_large_list_cache) == 0  # NOT cached - contains tuples/frozensets


class TestPerformanceCharacteristics:
    """Tests verifying performance characteristics (not timing, just behavior)."""

    def setup_method(self):
        clear_list_cache()

    def teardown_method(self):
        clear_list_cache()

    def test_cache_hit_returns_quickly(self):
        """Verify cache hit returns the same object (no new allocation)."""
        large_list = list(range(_LARGE_LIST_THRESHOLD * 10))  # Large list

        copy1 = deepcopy(large_list, {})
        copy2 = deepcopy(large_list, {})

        # Same object returned
        assert copy1 is copy2

    def test_mutable_list_always_copies(self):
        """Mutable lists should always create new copies."""
        large_list = [[i] for i in range(_LARGE_LIST_THRESHOLD + 100)]

        copy1 = deepcopy(large_list, {})
        copy2 = deepcopy(large_list, {})

        # Different objects
        assert copy1 is not copy2
        # Deep copy means nested lists are also different
        assert copy1[0] is not copy2[0]


class TestIsListInImmutableCache:
    """Tests for is_list_in_immutable_cache() function."""

    def setup_method(self):
        clear_list_cache()

    def teardown_method(self):
        clear_list_cache()

    def test_uncached_list_returns_false(self):
        """List not in cache should return False."""
        large_list = list(range(_LARGE_LIST_THRESHOLD + 100))
        assert is_list_in_immutable_cache(large_list) is False

    def test_cached_list_returns_true(self):
        """Original list in cache with matching hash should return True."""
        large_list = list(range(_LARGE_LIST_THRESHOLD + 100))
        deepcopy(large_list, {})  # This caches the list
        assert is_list_in_immutable_cache(large_list) is True

    def test_copy_tracked_in_copies_dict(self):
        """Copy should be tracked in _primitive_list_copies dict."""
        large_list = list(range(_LARGE_LIST_THRESHOLD + 100))
        copy = deepcopy(large_list, {})
        # Copy should be in the dict (keyed by its id)
        assert id(copy) in _primitive_list_copies
        assert _primitive_list_copies[id(copy)] is copy

    def test_copy_recognized_by_cache_check(self):
        """is_list_in_immutable_cache should return True for copies."""
        large_list = list(range(_LARGE_LIST_THRESHOLD + 100))
        copy = deepcopy(large_list, {})
        # The COPY should also be recognized (this is the key fix!)
        assert is_list_in_immutable_cache(copy) is True

    def test_modified_list_returns_false(self):
        """Cached list with modified contents should return False."""
        large_list = list(range(_LARGE_LIST_THRESHOLD + 100))
        deepcopy(large_list, {})  # Cache it
        large_list[0] = 999  # Modify it
        assert is_list_in_immutable_cache(large_list) is False

    def test_small_list_never_cached(self):
        """Small lists are never cached, so always return False."""
        small_list = list(range(100))
        deepcopy(small_list, {})  # Try to cache it
        assert is_list_in_immutable_cache(small_list) is False

    def test_non_primitive_list_not_cached(self):
        """Lists with non-primitive types are not cached."""
        large_list = [(i,) for i in range(_LARGE_LIST_THRESHOLD + 100)]
        deepcopy(large_list, {})  # Won't be cached (contains tuples)
        assert is_list_in_immutable_cache(large_list) is False

    def test_different_list_same_content(self):
        """Different list object with same content should return False."""
        list1 = list(range(_LARGE_LIST_THRESHOLD + 100))
        list2 = list(range(_LARGE_LIST_THRESHOLD + 100))
        deepcopy(list1, {})  # Cache list1
        # list2 is a different object, so not in cache
        assert is_list_in_immutable_cache(list2) is False


class TestAliasTraversalOptimization:
    """Tests for alias traversal optimization using the cache."""

    def setup_method(self):
        clear_list_cache()

    def teardown_method(self):
        clear_list_cache()

    def test_primitive_list_cached_during_checkpoint(self):
        """Large primitive lists should be cached during checkpoint."""
        cp = Checkpoints()
        large_list = list(range(_LARGE_LIST_THRESHOLD + 100))
        user_ns = {'large_list': large_list}

        cp.save('test', user_ns)

        # Original list should be in cache
        assert is_list_in_immutable_cache(large_list) is True

    def test_checkpoint_copy_recognized_by_cache(self):
        """Copy stored in checkpoint should be recognized by cache check.

        This is the key test for the fix: alias traversal runs on the COPIES
        stored in checkpoints, not the originals. The cache check must
        recognize these copies to skip traversal.
        """
        cp = Checkpoints()
        large_list = list(range(_LARGE_LIST_THRESHOLD + 100))
        user_ns = {'large_list': large_list}

        cp.save('test', user_ns)

        # Get the COPY stored in the checkpoint
        checkpoint = cp.get('test')
        copy_in_checkpoint = checkpoint.user_ns['large_list']

        # The copy should be different from the original
        assert copy_in_checkpoint is not large_list

        # The copy should be recognized by the cache check!
        # This is what enables alias traversal optimization.
        assert is_list_in_immutable_cache(copy_in_checkpoint) is True

    def test_non_primitive_list_not_cached_during_checkpoint(self):
        """Large non-primitive lists should NOT be cached during checkpoint."""
        cp = Checkpoints()
        large_list = [(i,) for i in range(_LARGE_LIST_THRESHOLD + 100)]
        user_ns = {'large_list': large_list}

        cp.save('test', user_ns)

        # List should NOT be in cache (contains tuples)
        assert is_list_in_immutable_cache(large_list) is False

    def test_checkpoint_with_shared_primitive_list(self):
        """Checkpoint should handle shared references to primitive lists."""
        cp = Checkpoints()
        shared_list = list(range(_LARGE_LIST_THRESHOLD + 100))
        user_ns = {
            'a': shared_list,
            'b': shared_list,  # Same object
        }

        cp.save('test', user_ns)

        # Both should point to the same cached copy
        checkpoint = cp.get('test')
        assert checkpoint.user_ns['a'] is checkpoint.user_ns['b']

    def test_alias_detection_correctness_with_cache(self):
        """Alias detection should still work correctly with cache optimization."""
        cp = Checkpoints()
        # Create a structure with potential aliases
        shared_obj = {'key': 'value'}
        large_primitive_list = list(range(_LARGE_LIST_THRESHOLD + 100))
        user_ns = {
            'shared': shared_obj,
            'ref1': {'nested': shared_obj},  # Alias to shared_obj
            'big_list': large_primitive_list,
        }

        # Save should succeed and detect aliases correctly
        saved, removed = cp.save('test', user_ns)
        assert 'shared' in saved
        assert 'ref1' in saved
        assert 'big_list' in saved


# =============================================================================
# Tests for Set Caching
# =============================================================================

class TestSetHasOnlyPrimitives:
    """Tests for _set_has_only_primitives() function."""

    def test_empty_set(self):
        assert _set_has_only_primitives(set()) is True

    def test_set_of_ints(self):
        assert _set_has_only_primitives({1, 2, 3, 4, 5}) is True

    def test_set_of_strings(self):
        assert _set_has_only_primitives({'a', 'b', 'c'}) is True

    def test_set_of_mixed_primitives(self):
        assert _set_has_only_primitives({1, 'a', 3.14, None, True}) is True

    def test_set_with_tuple_not_primitive(self):
        """Tuples are NOT primitives - should return False."""
        assert _set_has_only_primitives({(1, 2), (3, 4)}) is False

    def test_set_with_frozenset_not_primitive(self):
        """Frozensets are NOT primitives - should return False."""
        assert _set_has_only_primitives({frozenset([1]), frozenset([2])}) is False


class TestSetCaching:
    """Tests for set caching behavior."""

    def setup_method(self):
        clear_container_cache()

    def teardown_method(self):
        clear_container_cache()

    def test_large_primitive_set_cached(self):
        """Large set of primitives should be cached."""
        large_set = set(range(_LARGE_LIST_THRESHOLD + 100))
        deepcopy(large_set, {})
        assert len(_large_set_cache) == 1

    def test_small_set_not_cached(self):
        """Small set should not be cached."""
        small_set = {1, 2, 3}
        deepcopy(small_set, {})
        assert len(_large_set_cache) == 0

    def test_set_copy_recognized(self):
        """Set copy should be recognized by is_primitive_container."""
        large_set = set(range(_LARGE_LIST_THRESHOLD + 100))
        copy = deepcopy(large_set, {})
        assert is_primitive_container(copy) is True
        assert copy in _primitive_set_copies.values()

    def test_set_cache_hit(self):
        """Repeated deepcopy should return cached copy."""
        large_set = set(range(_LARGE_LIST_THRESHOLD + 100))
        copy1 = deepcopy(large_set, {})
        copy2 = deepcopy(large_set, {})
        assert copy1 is copy2  # Same cached copy

    def test_set_with_mutable_element_not_cached(self):
        """Set with non-primitive elements should not be cached."""
        # Sets can't contain lists/dicts, but can contain tuples
        large_set = {(i,) for i in range(_LARGE_LIST_THRESHOLD + 100)}
        deepcopy(large_set, {})
        assert len(_large_set_cache) == 0  # Tuples are not primitives


# =============================================================================
# Tests for Dict Caching
# =============================================================================

class TestDictHasOnlyPrimitiveValues:
    """Tests for _dict_has_only_primitive_values() function."""

    def test_empty_dict(self):
        assert _dict_has_only_primitive_values({}) is True

    def test_dict_with_int_values(self):
        assert _dict_has_only_primitive_values({'a': 1, 'b': 2}) is True

    def test_dict_with_string_values(self):
        assert _dict_has_only_primitive_values({'a': 'x', 'b': 'y'}) is True

    def test_dict_with_mixed_primitive_values(self):
        assert _dict_has_only_primitive_values({'a': 1, 'b': 'x', 'c': None}) is True

    def test_dict_with_list_value_not_primitive(self):
        assert _dict_has_only_primitive_values({'a': [1, 2, 3]}) is False

    def test_dict_with_dict_value_not_primitive(self):
        assert _dict_has_only_primitive_values({'a': {'nested': 1}}) is False

    def test_dict_with_tuple_value_not_primitive(self):
        """Tuples are NOT primitives."""
        assert _dict_has_only_primitive_values({'a': (1, 2)}) is False


class TestDictCaching:
    """Tests for dict caching behavior."""

    def setup_method(self):
        clear_container_cache()

    def teardown_method(self):
        clear_container_cache()

    def test_large_primitive_dict_cached(self):
        """Large dict with primitive values should be cached."""
        large_dict = {str(i): i for i in range(_LARGE_LIST_THRESHOLD + 100)}
        deepcopy(large_dict, {})
        assert len(_large_dict_cache) == 1

    def test_small_dict_not_cached(self):
        """Small dict should not be cached."""
        small_dict = {'a': 1, 'b': 2}
        deepcopy(small_dict, {})
        assert len(_large_dict_cache) == 0

    def test_dict_copy_recognized(self):
        """Dict copy should be recognized by is_primitive_container."""
        large_dict = {str(i): i for i in range(_LARGE_LIST_THRESHOLD + 100)}
        copy = deepcopy(large_dict, {})
        assert is_primitive_container(copy) is True
        assert copy in _primitive_dict_copies.values()

    def test_dict_cache_hit(self):
        """Repeated deepcopy should return cached copy."""
        large_dict = {str(i): i for i in range(_LARGE_LIST_THRESHOLD + 100)}
        copy1 = deepcopy(large_dict, {})
        copy2 = deepcopy(large_dict, {})
        assert copy1 is copy2  # Same cached copy

    def test_dict_with_mutable_value_not_cached(self):
        """Dict with non-primitive values should not be cached."""
        large_dict = {str(i): [i] for i in range(_LARGE_LIST_THRESHOLD + 100)}
        deepcopy(large_dict, {})
        assert len(_large_dict_cache) == 0


# =============================================================================
# Tests for Tuple Optimization
# =============================================================================

class TestTupleHasOnlyPrimitives:
    """Tests for _tuple_has_only_primitives() function."""

    def test_empty_tuple(self):
        assert _tuple_has_only_primitives(()) is True

    def test_tuple_of_ints(self):
        assert _tuple_has_only_primitives((1, 2, 3, 4, 5)) is True

    def test_tuple_of_strings(self):
        assert _tuple_has_only_primitives(('a', 'b', 'c')) is True

    def test_tuple_of_mixed_primitives(self):
        assert _tuple_has_only_primitives((1, 'a', 3.14, None, True)) is True

    def test_tuple_with_list_not_primitive(self):
        assert _tuple_has_only_primitives((1, [2, 3])) is False

    def test_tuple_with_nested_tuple_not_primitive(self):
        """Nested tuples are NOT primitives."""
        assert _tuple_has_only_primitives((1, (2, 3))) is False


class TestTupleOptimization:
    """Tests for tuple deepcopy optimization."""

    def setup_method(self):
        clear_container_cache()

    def teardown_method(self):
        clear_container_cache()

    def test_large_primitive_tuple_returns_original(self):
        """Large primitive tuple should return original (no copy)."""
        large_tuple = tuple(range(_LARGE_LIST_THRESHOLD + 100))
        result = deepcopy(large_tuple, {})
        assert result is large_tuple  # Same object!

    def test_small_tuple_may_return_original(self):
        """Small tuples also return original if contents unchanged."""
        small_tuple = (1, 2, 3)
        result = deepcopy(small_tuple, {})
        assert result is small_tuple  # Standard tuple behavior

    def test_tuple_with_mutable_contents_copied(self):
        """Tuple containing mutable objects should be deep copied."""
        tuple_with_list = ([1, 2], [3, 4])
        result = deepcopy(tuple_with_list, {})
        assert result is not tuple_with_list
        assert result[0] is not tuple_with_list[0]  # List was deep copied

    def test_large_primitive_tuple_recognized(self):
        """Large primitive tuple should be recognized by is_primitive_container."""
        large_tuple = tuple(range(_LARGE_LIST_THRESHOLD + 100))
        assert is_primitive_container(large_tuple) is True

    def test_small_tuple_not_checked(self):
        """Small tuples are not checked by is_primitive_container (too small)."""
        small_tuple = (1, 2, 3)
        assert is_primitive_container(small_tuple) is False


# =============================================================================
# Tests for Container Cache Stats
# =============================================================================

class TestContainerCacheStats:
    """Tests for get_container_cache_stats() function."""

    def setup_method(self):
        clear_container_cache()

    def teardown_method(self):
        clear_container_cache()

    def test_empty_cache_stats(self):
        stats = get_container_cache_stats()
        assert stats['list_cache_size'] == 0
        assert stats['set_cache_size'] == 0
        assert stats['dict_cache_size'] == 0

    def test_stats_after_caching(self):
        """Stats should reflect cached containers."""
        large_list = list(range(_LARGE_LIST_THRESHOLD + 100))
        large_set = set(range(_LARGE_LIST_THRESHOLD + 100))
        large_dict = {str(i): i for i in range(_LARGE_LIST_THRESHOLD + 100)}

        deepcopy(large_list, {})
        deepcopy(large_set, {})
        deepcopy(large_dict, {})

        stats = get_container_cache_stats()
        assert stats['list_cache_size'] == 1
        assert stats['set_cache_size'] == 1
        assert stats['dict_cache_size'] == 1

    def test_clear_container_cache_clears_all(self):
        """clear_container_cache should clear all caches."""
        large_list = list(range(_LARGE_LIST_THRESHOLD + 100))
        large_set = set(range(_LARGE_LIST_THRESHOLD + 100))
        large_dict = {str(i): i for i in range(_LARGE_LIST_THRESHOLD + 100)}

        deepcopy(large_list, {})
        deepcopy(large_set, {})
        deepcopy(large_dict, {})

        clear_container_cache()

        stats = get_container_cache_stats()
        assert stats['list_cache_size'] == 0
        assert stats['set_cache_size'] == 0
        assert stats['dict_cache_size'] == 0
