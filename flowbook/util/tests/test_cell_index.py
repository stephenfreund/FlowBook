"""Tests for flowbook.util.cell_index."""

import pytest

from flowbook.util.cell_index import alpha_to_index, index_to_alpha, parse_cell_ref


class TestIndexToAlpha:
    def test_single_letters(self):
        assert index_to_alpha(0) == '@A'
        assert index_to_alpha(1) == '@B'
        assert index_to_alpha(25) == '@Z'

    def test_two_letters(self):
        assert index_to_alpha(26) == '@AA'
        assert index_to_alpha(27) == '@AB'
        assert index_to_alpha(51) == '@AZ'
        assert index_to_alpha(52) == '@BA'
        assert index_to_alpha(701) == '@ZZ'

    def test_three_letters(self):
        assert index_to_alpha(702) == '@AAA'
        assert index_to_alpha(703) == '@AAB'
        assert index_to_alpha(18277) == '@ZZZ'

    def test_negative_raises(self):
        with pytest.raises(ValueError, match='non-negative'):
            index_to_alpha(-1)

    def test_too_large_raises(self):
        with pytest.raises(ValueError, match='too large'):
            index_to_alpha(18278)


class TestAlphaToIndex:
    """alpha_to_index is strict: requires '@' prefix and uppercase."""

    def test_single_letters(self):
        assert alpha_to_index('@A') == 0
        assert alpha_to_index('@B') == 1
        assert alpha_to_index('@Z') == 25

    def test_two_letters(self):
        assert alpha_to_index('@AA') == 26
        assert alpha_to_index('@AB') == 27
        assert alpha_to_index('@AZ') == 51
        assert alpha_to_index('@BA') == 52
        assert alpha_to_index('@ZZ') == 701

    def test_three_letters(self):
        assert alpha_to_index('@AAA') == 702
        assert alpha_to_index('@AAB') == 703
        assert alpha_to_index('@ZZZ') == 18277

    def test_missing_prefix_raises(self):
        with pytest.raises(ValueError):
            alpha_to_index('A')

    def test_lowercase_raises(self):
        with pytest.raises(ValueError):
            alpha_to_index('@a')

    def test_empty_after_at_raises(self):
        with pytest.raises(ValueError):
            alpha_to_index('@')

    def test_invalid_chars_raises(self):
        with pytest.raises(ValueError):
            alpha_to_index('@1')

    def test_too_many_letters_raises(self):
        with pytest.raises(ValueError, match='too many letters'):
            alpha_to_index('@AAAA')


class TestParseCellRef:
    """parse_cell_ref is lenient: handles @-labels, plain letters, numbers."""

    def test_at_label(self):
        assert parse_cell_ref('@C') == 2
        assert parse_cell_ref('@AA') == 26

    def test_plain_letters(self):
        assert parse_cell_ref('C') == 2
        assert parse_cell_ref('AA') == 26

    def test_lowercase(self):
        assert parse_cell_ref('a') == 0
        assert parse_cell_ref('z') == 25
        assert parse_cell_ref('aa') == 26

    def test_numeric_string(self):
        assert parse_cell_ref('2') == 2
        assert parse_cell_ref('26') == 26
        assert parse_cell_ref('0') == 0

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            parse_cell_ref('')

    def test_whitespace_stripped(self):
        assert parse_cell_ref(' @B ') == 1
        assert parse_cell_ref(' 3 ') == 3


class TestRoundTrip:
    def test_round_trip_first_100(self):
        for i in range(100):
            assert alpha_to_index(index_to_alpha(i)) == i

    def test_round_trip_boundaries(self):
        for i in [0, 25, 26, 51, 52, 701, 702, 18277]:
            assert alpha_to_index(index_to_alpha(i)) == i

    def test_round_trip_full_single(self):
        for i in range(26):
            assert alpha_to_index(index_to_alpha(i)) == i

    def test_round_trip_sample_two_letter(self):
        import random
        rng = random.Random(42)
        for _ in range(50):
            i = rng.randint(26, 701)
            assert alpha_to_index(index_to_alpha(i)) == i

    def test_round_trip_sample_three_letter(self):
        import random
        rng = random.Random(42)
        for _ in range(50):
            i = rng.randint(702, 18277)
            assert alpha_to_index(index_to_alpha(i)) == i
