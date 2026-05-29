"""Pydantic models for the AI fix-suggestion feature.

These describe the wire format between the suggester and the apply handler,
and the validation contract that every LLM-produced plan must satisfy before
it can be dispatched to an AST tool.
"""

from typing import Any, Dict, List, Literal, Optional, Set, Union

from pydantic import BaseModel, Field, model_validator

from flowbook.tools.registry import REGISTRY as _REGISTRY


# The exact tool names the LLM is allowed to propose. Any value outside this
# allowlist is rejected by validate_plan() before dispatch. This stays a
# static Literal (it annotates pydantic fields); a test asserts it matches the
# registry, which is the runtime source of truth.
FixToolName = Literal[
    "alpha_rename",
    "remove_inplace",
    "insert_deepcopy",
    "mark_diagnostic",
    "merge_cells",
    "move_cell",
]


# Required arg keys per tool, derived from the unified registry so the
# validation contract cannot drift from the handlers that apply the fix.
TOOL_ARG_SCHEMAS: Dict[str, Set[str]] = {
    t.name: set(t.parameters.get("required", [])) for t in _REGISTRY
}


class FixSuggestion(BaseModel):
    """A single proposed fix the user can click to apply."""

    label: str = Field(
        ..., description="Short button label shown to the user (max ~60 chars)."
    )
    rationale: str = Field(
        ...,
        description="One-sentence explanation of why this fix addresses the violation.",
    )
    tool: FixToolName
    args: Dict[str, Any]


class FixPlan(BaseModel):
    """The full set of suggestions for one violation."""

    fixes: List[FixSuggestion] = Field(default_factory=list, max_length=3)


class ViolationContext(BaseModel):
    """Everything the suggester needs to diagnose one violation.

    Built by the handler from the request body + the notebook content + the
    flowbook metadata embedded on the violating cell.
    """

    cell_id: str = Field(..., description="The cell that triggered the violation.")
    cell_alpha: str = Field(..., description="The user-facing @-label, e.g. 'C'.")
    cell_source: str
    error_type: str = Field(
        ...,
        description=(
            "Violation predicate name: no_read_and_write, write_before_read, "
            "no_read_before_write, no_write_after_read, unrecoverable_mutation."
        ),
    )
    locations: List[str] = Field(
        default_factory=list,
        description="Location strings from the violation (e.g. 'train', 'df.age').",
    )
    causer_cells: List[str] = Field(
        default_factory=list,
        description="Other cells implicated by the violation, as @-labels.",
    )
    cell_order: List[str] = Field(
        ..., description="Full notebook cell order (cell_ids, in execution order)."
    )
    surrounding_sources: Dict[str, str] = Field(
        default_factory=dict,
        description="cell_id -> source for the ~3 cells above and below the violator.",
    )


class ApplyFixRequest(BaseModel):
    """POST /flowbook/apply-fix body."""

    notebook_path: str
    cell_id: str
    tool: FixToolName
    args: Dict[str, Any]
    notebook: Dict[str, Any] = Field(
        ..., description="Full notebook JSON content (cells, metadata, etc.)."
    )


class ApplyFixResponse(BaseModel):
    """POST /flowbook/apply-fix response."""

    ok: bool
    tool: FixToolName
    args: Dict[str, Any]
    modified_cells: List[str]
    pre_fix_sources: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "cell_id -> source as it was BEFORE the fix. The frontend stashes "
            "this so the Undo button can restore the prior state."
        ),
    )
    post_fix_sources: Dict[str, str] = Field(
        default_factory=dict,
        description="cell_id -> new source after the fix.",
    )
    cells_removed: List[str] = Field(
        default_factory=list,
        description="For merge_cells: the cell_ids no longer in the notebook.",
    )
    new_cell_order: Optional[List[str]] = Field(
        default=None,
        description="Set when the tool changed cell order (merge_cells, move_cell).",
    )
    error: Optional[str] = None


class CustomFixRequest(BaseModel):
    """POST /flowbook/custom-fix body."""

    notebook: Dict[str, Any]
    cell_id: str = Field(..., description="The violating cell the user invoked from.")
    instruction: str = Field(..., min_length=1, description="User's natural-language fix request.")


class CustomFixResponse(BaseModel):
    """POST /flowbook/custom-fix response."""

    ok: bool
    instruction: str
    summary: str = Field(
        default="",
        description="Free-text summary from the LLM of what it did.",
    )
    modified_cells: List[str] = Field(default_factory=list)
    cells_added: List[str] = Field(default_factory=list)
    cells_removed: List[str] = Field(default_factory=list)
    pre_fix_sources: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "cell_id -> source as it was BEFORE the custom fix. Includes "
            "every cell the LLM mutated or deleted; the frontend uses this "
            "for the same source-and-metadata Undo flow as the built-in fixes."
        ),
    )
    post_fix_sources: Dict[str, str] = Field(default_factory=dict)
    new_cell_order: Optional[List[str]] = Field(default=None)
    mutations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Ordered list of mutation entries (tool, args, summary).",
    )
    error: Optional[str] = None


class PlanValidationError(ValueError):
    """Raised when an LLM-produced FixPlan fails the validation contract.

    The handler converts this to a 'plan parsing failed' SSE event so the
    frontend can show the diagnosis text without offering broken buttons.
    """


def validate_plan(plan: FixPlan, context: ViolationContext) -> FixPlan:
    """Validate every suggestion in the plan against the allowlist + the context.

    Raises PlanValidationError if any suggestion is malformed. On success,
    returns the (possibly-pruned) plan with only the valid suggestions kept.
    Suggestions with minor issues (e.g. invalid cell_id) are dropped rather
    than the entire plan being rejected, so the user still sees the others.
    """
    valid_cell_ids: Set[str] = set(context.cell_order)
    surviving: List[FixSuggestion] = []
    errors: List[str] = []

    for i, fix in enumerate(plan.fixes):
        try:
            _validate_one(fix, context, valid_cell_ids)
        except PlanValidationError as e:
            errors.append(f"fix[{i}] ({fix.tool}): {e}")
            continue
        surviving.append(fix)

    if not surviving:
        raise PlanValidationError(
            "No valid fix suggestions in plan. Issues: " + "; ".join(errors)
        )

    return FixPlan(fixes=surviving)


def _validate_one(
    fix: FixSuggestion, context: ViolationContext, valid_cell_ids: Set[str]
) -> None:
    expected = TOOL_ARG_SCHEMAS.get(fix.tool)
    if expected is None:
        raise PlanValidationError(f"unknown tool '{fix.tool}'")

    missing = expected - set(fix.args.keys())
    if missing:
        raise PlanValidationError(f"missing args: {sorted(missing)}")

    extra = set(fix.args.keys()) - expected
    if extra:
        raise PlanValidationError(f"unexpected args: {sorted(extra)}")

    # cell_id (single) checks
    if "cell_id" in fix.args:
        cid = fix.args["cell_id"]
        if not isinstance(cid, str) or cid not in valid_cell_ids:
            raise PlanValidationError(
                f"cell_id '{cid}' is not a cell in this notebook"
            )

    # after_cell_id check (move_cell)
    if "after_cell_id" in fix.args:
        cid = fix.args["after_cell_id"]
        if not isinstance(cid, str) or cid not in valid_cell_ids:
            raise PlanValidationError(
                f"after_cell_id '{cid}' is not a cell in this notebook"
            )

    # cell_ids list check (merge_cells)
    if "cell_ids" in fix.args:
        cids = fix.args["cell_ids"]
        if not isinstance(cids, list) or len(cids) < 2:
            raise PlanValidationError("cell_ids must be a list of at least 2 cell ids")
        for cid in cids:
            if not isinstance(cid, str) or cid not in valid_cell_ids:
                raise PlanValidationError(
                    f"cell_ids contains unknown cell '{cid}'"
                )

    # variable/old_name must appear in the relevant cell's source
    if fix.tool == "alpha_rename":
        cid = fix.args["cell_id"]
        src = _resolve_source(context, cid)
        if src is not None and not _name_appears_in_source(
            fix.args["old_name"], src
        ):
            raise PlanValidationError(
                f"old_name '{fix.args['old_name']}' does not appear in cell {cid}"
            )
    if fix.tool in ("remove_inplace", "insert_deepcopy"):
        cid = fix.args["cell_id"]
        src = _resolve_source(context, cid)
        if src is not None and not _name_appears_in_source(fix.args["variable"], src):
            raise PlanValidationError(
                f"variable '{fix.args['variable']}' does not appear in cell {cid}"
            )


def _resolve_source(context: ViolationContext, cell_id: str) -> Optional[str]:
    if cell_id == context.cell_id:
        return context.cell_source
    return context.surrounding_sources.get(cell_id)


def _name_appears_in_source(name: str, source: str) -> bool:
    import re

    return re.search(rf"\b{re.escape(name)}\b", source) is not None
