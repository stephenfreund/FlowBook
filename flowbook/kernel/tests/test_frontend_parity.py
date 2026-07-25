"""Cross-language parity tests (see FRONTEND_TESTING.md §6).

The ▷ conflict relation and the staleness-reason vocabulary exist in both
Python (flowbook/kernel/locations.py, models.py) and TypeScript
(src/flowbook/types.ts, reasonformat.ts), and each has drifted once
before. These tests consume the SAME JSON fixtures as the jest specs
(src/flowbook/tests/fixtures/), so either implementation drifting fails a
test on its own side.

Skipped when the src/ tree is absent (installed-package runs).
"""

import json
from pathlib import Path

import pytest

from flowbook.kernel.loc_ids import LocRef
from flowbook.kernel.locations import ReadLoc, WriteLoc, write_conflicts_read
from flowbook.kernel.models import ReasonType

_FIXTURES = (
    Path(__file__).resolve().parents[3] / "src" / "flowbook" / "tests" / "fixtures"
)

pytestmark = pytest.mark.skipif(
    not _FIXTURES.is_dir(),
    reason="frontend fixture tree (src/) not present — installed-package run",
)


def _qualifier(loc_dict):
    """Build the Python qualifier from a fixture loc dict.

    A numeric qualifier is a LocRef loc_id (with var_name); a string
    qualifier is used as-is — mirroring the TS IReadLoc/IWriteLoc shape.
    """
    q = loc_dict.get("qualifier")
    if isinstance(q, int):
        return LocRef(q, loc_dict.get("var_name", ""))
    return q


def _to_write_loc(d) -> WriteLoc:
    t = d["type"]
    if t == "var":
        return WriteLoc.var(d["name"])
    if t == "col":
        return WriteLoc.col(_qualifier(d), d["name"])
    if t == "cols":
        return WriteLoc.cols(d["name"], qualifier=_qualifier(d))
    if t == "rows":
        return WriteLoc.rows(d["name"], qualifier=_qualifier(d))
    if t == "file":
        return WriteLoc.file(d["name"])
    raise ValueError(f"unknown write loc type: {t}")


def _to_read_loc(d) -> ReadLoc:
    t = d["type"]
    if t == "var":
        return ReadLoc.var(d["name"])
    if t == "col":
        return ReadLoc.col(_qualifier(d), d["name"])
    if t == "cols":
        return ReadLoc.cols(d["name"], qualifier=_qualifier(d))
    if t == "rows":
        return ReadLoc.rows(d["name"], qualifier=_qualifier(d))
    if t == "file":
        return ReadLoc.file(d["name"])
    raise ValueError(f"unknown read loc type: {t}")


def _conflict_cases():
    with open(_FIXTURES / "conflict_cases.json", encoding="utf-8") as f:
        return json.load(f)["cases"]


class TestConflictRelationParity:
    """write_conflicts_read must agree with the shared fixture case-by-case
    (the jest side asserts the same for writeConflictsRead)."""

    @pytest.mark.parametrize(
        "case",
        _conflict_cases() if _FIXTURES.is_dir() else [],
        ids=lambda c: c["desc"],
    )
    def test_case(self, case):
        w = _to_write_loc(case["write"])
        r = _to_read_loc(case["read"])
        assert write_conflicts_read(w, r) is case["conflicts"], (
            f"▷ disagreement for {case['desc']}: "
            f"python says {write_conflicts_read(w, r)}, "
            f"fixture says {case['conflicts']}"
        )


class TestReasonVocabularyParity:
    """Every Python ReasonType value must appear in the shared vocabulary
    fixture (the jest side asserts reasonformat.ts formats each one
    specifically)."""

    def test_reason_types_subset_of_fixture(self):
        with open(_FIXTURES / "reason_types.json", encoding="utf-8") as f:
            fixture_types = set(json.load(f)["backend_reason_types"])
        python_types = {r.value for r in ReasonType}
        missing = python_types - fixture_types
        assert not missing, (
            f"ReasonType values missing from reason_types.json (add them "
            f"there AND to src/flowbook/reasonformat.ts): {sorted(missing)}"
        )
