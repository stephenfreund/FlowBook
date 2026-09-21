"""Modules referenced from inside objects are shared by deepcopy, so objects that hold one
(scipy >= 1.15 result types keep their array namespace module in `_xp`) can be checkpointed.
Top-level module variables are still excluded from checkpoints."""

import copy
import types

import numpy as np
import pytest

from flowbook.kernel_support.deepcopy import deepcopy
from flowbook.kernel_support.deepcopyable import check_deepcopyable
from flowbook.kernel_support.memory_checkpoint import MemoryCheckpoints, filter_user_namespace

stats = pytest.importorskip("scipy.stats")


class Holder:
    def __init__(self):
        self.xp = np  # a module reference, as scipy's TtestResult._xp
        self.data = [1, 2, 3]


class TestModuleInsideObject:
    def test_module_is_shared_not_copied(self):
        h = Holder()
        c = deepcopy(h, {})
        assert c is not h
        assert c.xp is np  # same module object
        assert c.data == h.data and c.data is not h.data  # the rest is still a deep copy

    def test_module_in_containers(self):
        x = {"m": np, "l": [types, (np,)]}
        c = deepcopy(x, {})
        assert c is not x and c["m"] is np and c["l"][1][0] is np

    def test_stdlib_deepcopy_still_fails(self):
        """The reason this rule exists: standard deepcopy cannot handle it."""
        with pytest.raises(TypeError, match="cannot pickle 'module' object"):
            copy.deepcopy(Holder())


@pytest.fixture()
def samples():
    rng = np.random.default_rng(0)
    return rng.normal(10, 2, 40), rng.normal(12, 2, 40)


def _ttest_results(a, b):
    return {
        "ttest_ind": stats.ttest_ind(a, b, equal_var=False),
        "ttest_1samp": stats.ttest_1samp(a, 10.0),
        "ttest_rel": stats.ttest_rel(a, b),
    }


class TestScipyResultTypes:
    def test_ttest_results_carry_a_module(self, samples):
        """Guards the premise: if scipy stops storing the module, these tests become moot."""
        for res in _ttest_results(*samples).values():
            assert isinstance(getattr(res, "_xp", None), types.ModuleType)

    @pytest.mark.parametrize("name", ["ttest_ind", "ttest_1samp", "ttest_rel"])
    def test_ttest_result_deepcopies(self, samples, name):
        res = _ttest_results(*samples)[name]
        c = deepcopy(res, {})
        assert type(c) is type(res) and c is not res
        assert c.statistic == res.statistic and c.pvalue == res.pvalue and c.df == res.df
        assert tuple(c) == tuple(res)  # still a tuple of (statistic, pvalue)
        assert c._xp is res._xp  # the array namespace module is shared
        # the one method that uses _xp still works on the copy and agrees with the original
        lo, hi = c.confidence_interval(0.95)
        lo0, hi0 = res.confidence_interval(0.95)
        assert lo == lo0 and hi == hi0

    def test_other_results_unaffected(self, samples):
        a, b = samples
        for res in (stats.mannwhitneyu(a, b), stats.shapiro(a), stats.pearsonr(a, b), stats.describe(a)):
            c = deepcopy(res, {})
            assert tuple(c)[:2] == tuple(res)[:2]

    def test_check_deepcopyable(self, samples):
        res = _ttest_results(*samples)["ttest_ind"]
        assert check_deepcopyable(res) is None
        assert check_deepcopyable(Holder()) is None
        # a bare module is still reported as not copyable
        assert check_deepcopyable(np) is not None


class TestCheckpointing:
    def test_ttest_result_is_checkpointed(self, samples):
        a, b = samples
        result = stats.ttest_ind(a, b, equal_var=False)
        cp = MemoryCheckpoints()
        saved, removed = cp.save("t", {"a": a, "result": result})
        assert "result" in saved and removed == {}
        # round-trip: the restored result is a copy with the same values and the same array namespace
        ns = {"a": a, "result": None}
        cp.restore("t", ns)
        assert ns["result"] is not result and ns["result"].pvalue == result.pvalue
        assert ns["result"]._xp is result._xp
        assert ns["result"].confidence_interval() == result.confidence_interval()

    def test_top_level_modules_still_excluded(self):
        ns = filter_user_namespace({"np": np, "x": 1})
        assert "np" not in ns and ns["x"] == 1
