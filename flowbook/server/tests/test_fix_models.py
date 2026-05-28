"""Tests for fix_models.validate_plan — the LLM-output guard rail."""

import pytest

from flowbook.server.fix_models import (
    FixPlan,
    FixSuggestion,
    PlanValidationError,
    ViolationContext,
    validate_plan,
)


def _ctx(
    cell_id: str = "B",
    cell_source: str = "train = pd.concat([train, extra])\n",
    cell_order=("A", "B", "C"),
    surrounding=None,
) -> ViolationContext:
    return ViolationContext(
        cell_id=cell_id,
        cell_alpha=cell_id,
        cell_source=cell_source,
        error_type="no_read_and_write",
        locations=["train"],
        causer_cells=[],
        cell_order=list(cell_order),
        surrounding_sources=surrounding or {},
    )


def _fix(tool: str, **args) -> FixSuggestion:
    return FixSuggestion(label="x", rationale="y", tool=tool, args=args)


class TestValidatePlanHappy:
    def test_valid_alpha_rename_passes(self):
        plan = FixPlan(fixes=[_fix("alpha_rename", cell_id="B", old_name="train", new_name="train_combined")])
        result = validate_plan(plan, _ctx())
        assert len(result.fixes) == 1
        assert result.fixes[0].tool == "alpha_rename"

    def test_valid_remove_inplace_passes(self):
        ctx = _ctx(cell_source="df.drop(columns=['a'], inplace=True)\n")
        plan = FixPlan(fixes=[_fix("remove_inplace", cell_id="B", variable="df")])
        result = validate_plan(plan, ctx)
        assert result.fixes[0].args["variable"] == "df"

    def test_valid_mark_diagnostic_passes(self):
        plan = FixPlan(fixes=[_fix("mark_diagnostic", cell_id="B")])
        result = validate_plan(plan, _ctx())
        assert result.fixes[0].tool == "mark_diagnostic"

    def test_valid_merge_cells_passes(self):
        plan = FixPlan(fixes=[_fix("merge_cells", cell_ids=["A", "B"])])
        result = validate_plan(plan, _ctx())
        assert result.fixes[0].args["cell_ids"] == ["A", "B"]

    def test_valid_move_cell_passes(self):
        plan = FixPlan(fixes=[_fix("move_cell", cell_id="B", after_cell_id="C")])
        result = validate_plan(plan, _ctx())
        assert result.fixes[0].args["after_cell_id"] == "C"


class TestValidatePlanRejections:
    def test_unknown_tool_dropped(self):
        # Pydantic itself enforces the Literal, so this is caught at FixPlan
        # construction time.
        with pytest.raises(Exception):
            FixPlan(fixes=[_fix("rm_rf", path="/")])

    def test_missing_arg_dropped(self):
        # alpha_rename without new_name
        plan = FixPlan(fixes=[_fix("alpha_rename", cell_id="B", old_name="train")])
        with pytest.raises(PlanValidationError, match="No valid"):
            validate_plan(plan, _ctx())

    def test_extra_arg_dropped(self):
        plan = FixPlan(fixes=[
            _fix("mark_diagnostic", cell_id="B", extra="surprise"),
        ])
        with pytest.raises(PlanValidationError):
            validate_plan(plan, _ctx())

    def test_unknown_cell_id_dropped(self):
        plan = FixPlan(fixes=[_fix("alpha_rename", cell_id="Z", old_name="train", new_name="train2")])
        with pytest.raises(PlanValidationError, match="No valid"):
            validate_plan(plan, _ctx())

    def test_alpha_rename_old_name_not_in_source_dropped(self):
        # 'train' is not in the source we provide
        ctx = _ctx(cell_source="x = 1\n")
        plan = FixPlan(fixes=[_fix("alpha_rename", cell_id="B", old_name="train", new_name="train2")])
        with pytest.raises(PlanValidationError, match="does not appear"):
            validate_plan(plan, ctx)

    def test_variable_not_in_source_dropped_for_remove_inplace(self):
        ctx = _ctx(cell_source="x = 1\n")
        plan = FixPlan(fixes=[_fix("remove_inplace", cell_id="B", variable="df")])
        with pytest.raises(PlanValidationError):
            validate_plan(plan, ctx)

    def test_merge_cells_too_few_dropped(self):
        plan = FixPlan(fixes=[_fix("merge_cells", cell_ids=["A"])])
        with pytest.raises(PlanValidationError):
            validate_plan(plan, _ctx())

    def test_merge_cells_unknown_member_dropped(self):
        plan = FixPlan(fixes=[_fix("merge_cells", cell_ids=["A", "ZZ"])])
        with pytest.raises(PlanValidationError):
            validate_plan(plan, _ctx())


class TestValidatePlanPartialKeep:
    def test_keeps_valid_drops_invalid(self):
        plan = FixPlan(fixes=[
            _fix("alpha_rename", cell_id="ZZ", old_name="x", new_name="y"),  # bad cell
            _fix("mark_diagnostic", cell_id="B"),  # good
        ])
        result = validate_plan(plan, _ctx())
        assert len(result.fixes) == 1
        assert result.fixes[0].tool == "mark_diagnostic"
