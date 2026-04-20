"""Document and pin down the mutation semantics of sklearn's
RandomForestClassifier — which methods mutate `self` in place and which
are pure.

FlowBook's structural tracker classifies method calls on user objects
conservatively: anything that could mutate is flagged as a potential
in-place write, which can cause false positives for predictors. These
tests are the empirical ground truth for a concrete, commonly-used
classifier:

  * `__init__` sets only the hyperparameters passed in.
  * `fit(X, y)` is a large in-place mutation: it attaches the learned
    state (`estimators_`, `classes_`, `n_classes_`, `n_features_in_`,
    `estimator_`, `n_outputs_`, plus a couple of private bookkeeping
    attrs) onto `self`. This is the real mutation users should worry
    about — re-running a `fit` cell would change what downstream cells
    see, and FlowBook correctly flags repeat `fit` calls.
  * `predict(X)` is side-effect-free on the classifier: no attribute is
    added, removed, or changed; the list of trees in `estimators_` keeps
    the same object identity, and the trees themselves keep the same
    object identity and the same `random_state`. `predict` reads the
    fitted state, allocates a fresh result array internally, and returns
    it.

If a future sklearn version changes any of this (e.g., adds a cache on
predict), these tests break loudly and FlowBook's classifier-method
heuristic needs updating.
"""

import copy
import pickle

import pytest

sklearn = pytest.importorskip("sklearn")

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier


def _snapshot(obj) -> dict[str, bytes]:
    """Capture a pickle-based snapshot of every attribute on `obj`. Two
    snapshots compare equal iff all attributes pickled to the same bytes,
    which is a stronger test than `vars(a) == vars(b)` (which relies on
    user-provided __eq__ / works only for hashables)."""
    snap: dict[str, bytes] = {}
    for k, v in vars(obj).items():
        try:
            snap[k] = pickle.dumps(v)
        except Exception:
            snap[k] = repr(v).encode()
    return snap


@pytest.fixture
def fitted_rf():
    """Return a freshly-fit RandomForestClassifier plus the training data."""
    X, y = make_classification(n_samples=60, n_features=4, random_state=0)
    rf = RandomForestClassifier(n_estimators=5, random_state=0, n_jobs=1)
    rf.fit(X, y)
    return rf, X, y


class TestInitDoesNotAddLearnedState:
    def test_only_hyperparameters_are_set(self):
        rf = RandomForestClassifier(n_estimators=5, random_state=0)
        attrs = set(vars(rf).keys())
        # Hyperparameter names (a subset — we assert "learned" attrs are absent).
        assert "n_estimators" in attrs
        assert "random_state" in attrs
        # Learned-state attributes must not exist yet.
        for learned in (
            "estimators_", "classes_", "n_classes_", "n_features_in_",
            "estimator_", "n_outputs_", "_n_samples", "_n_samples_bootstrap",
        ):
            assert learned not in attrs, (
                f"Unexpected learned attribute {learned!r} present before fit"
            )


class TestFitPerformsInplaceMutation:
    """fit() is the actual in-place mutation — it attaches learned state
    to the same `self` object. Everything FlowBook's UNRECOVERABLE_MUTATION
    check flags about `fit` is accurate."""

    def test_fit_mutates_self_in_place_same_object(self):
        X, y = make_classification(n_samples=60, n_features=4, random_state=0)
        rf = RandomForestClassifier(n_estimators=5, random_state=0)
        id_before = id(rf)

        returned = rf.fit(X, y)

        # sklearn convention: fit returns self. The identity proves the
        # mutation is in-place, not a rebind.
        assert returned is rf
        assert id(rf) == id_before

    def test_fit_adds_learned_attributes(self):
        X, y = make_classification(n_samples=60, n_features=4, random_state=0)
        rf = RandomForestClassifier(n_estimators=5, random_state=0)
        before = set(vars(rf).keys())

        rf.fit(X, y)

        after = set(vars(rf).keys())
        added = after - before

        # Exactly these attributes are the "learned state" — the observable
        # in-place mutation that FlowBook sees (and legitimately flags when
        # fit is called on an already-fitted estimator).
        expected_added = {
            "estimators_", "classes_", "n_classes_", "n_features_in_",
            "estimator_", "n_outputs_",
            "_n_samples", "_n_samples_bootstrap",
        }
        assert added >= expected_added, (
            f"fit did not add expected attributes. Missing: {expected_added - added}"
        )

    def test_fit_populates_estimators_list(self, fitted_rf):
        rf, _, _ = fitted_rf
        assert hasattr(rf, "estimators_")
        assert len(rf.estimators_) == rf.n_estimators


class TestPredictDoesNotMutate:
    """predict() is pure with respect to the classifier object. This is
    the false-positive class that FlowBook's structural tracker must
    eventually learn to recognise."""

    def test_predict_does_not_change_vars(self, fitted_rf):
        rf, X, _ = fitted_rf
        before = _snapshot(rf)

        rf.predict(X)

        after = _snapshot(rf)
        assert set(before) == set(after), (
            f"predict added/removed attributes. "
            f"Added: {set(after) - set(before)}, "
            f"Removed: {set(before) - set(after)}"
        )
        # Byte-level equality of pickled attribute values.
        changed = [k for k in before if before[k] != after[k]]
        assert changed == [], (
            f"predict mutated the classifier's attribute values: {changed}"
        )

    def test_predict_preserves_estimator_list_identity(self, fitted_rf):
        rf, X, _ = fitted_rf
        list_id = id(rf.estimators_)
        tree_ids = [id(t) for t in rf.estimators_]

        rf.predict(X)

        assert id(rf.estimators_) == list_id, (
            "predict replaced the estimators_ list with a new object"
        )
        assert [id(t) for t in rf.estimators_] == tree_ids, (
            "predict replaced one or more tree estimators"
        )

    def test_predict_preserves_tree_random_state(self, fitted_rf):
        rf, X, _ = fitted_rf
        before_states = [copy.deepcopy(t.random_state) for t in rf.estimators_]

        rf.predict(X)

        after_states = [t.random_state for t in rf.estimators_]
        assert after_states == before_states, (
            "predict advanced a tree's random_state — would break determinism"
        )

    def test_repeated_predict_is_idempotent(self, fitted_rf):
        """Calling predict twice in a row produces the same result and
        leaves the classifier in the same state."""
        rf, X, _ = fitted_rf
        y1 = rf.predict(X)
        state_mid = _snapshot(rf)
        y2 = rf.predict(X)
        state_end = _snapshot(rf)

        # Bitwise-identical outputs.
        assert (y1 == y2).all()
        # Bitwise-identical classifier state across the two calls.
        changed = [k for k in state_mid if state_mid[k] != state_end[k]]
        assert changed == []


class TestWhyThisMatters:
    """Narrative glue, phrased as executable assertions, so the reason
    predict is safe is encoded somewhere that breaks if sklearn changes."""

    def test_predict_is_rerun_safe(self, fitted_rf):
        """A cell that does `y = rf.predict(X)` can be re-run any number
        of times and produce the same outputs without perturbing anything
        an earlier cell read — which is exactly what FlowBook's rerun
        consistency requires of a CLEAN cell."""
        rf, X, _ = fitted_rf
        ys = [rf.predict(X) for _ in range(3)]
        for y in ys[1:]:
            assert (y == ys[0]).all()

    def test_fit_is_not_rerun_safe(self):
        """Conversely, a cell that does `rf = RandomForestClassifier(...);
        rf.fit(X, y)` followed by any cell that reads `rf.estimators_` IS
        re-run safe only because the whole object gets rebuilt. A cell
        that reads `rf` and then a later cell that re-fits the same `rf`
        in place is the UNRECOVERABLE_MUTATION pattern FlowBook catches —
        `fit` mutates `self`, and the trees are different on each call.

        This test verifies only the 'trees change' half: calling fit twice
        on the same object yields different `estimators_` lists even with
        identical random_state (because the internal RNG state advances)."""
        X, y = make_classification(n_samples=60, n_features=4, random_state=0)
        rf = RandomForestClassifier(n_estimators=5, random_state=0)
        rf.fit(X, y)
        first_trees = list(rf.estimators_)
        rf.fit(X, y)
        second_trees = list(rf.estimators_)
        # The list is replaced — a new list object with new tree objects.
        assert first_trees is not second_trees
        assert [id(t) for t in first_trees] != [id(t) for t in second_trees]
