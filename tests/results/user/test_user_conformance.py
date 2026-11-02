"""The combination evaluator against the conformance fixture (ADR 0130 D4, A3, A5).

Three oracles, each naming the right answer:

* the goldens of ``tests/fixtures/user_results/conformance.json``, written by a
  reference in ``make_conformance.py`` that shares no code with the evaluator;
  the app's TypeScript evaluator (VU4) reads the same file;
* the kernels of ``apeGmsh.opensees.analysis.spectrum`` (``combine_modes`` with
  ``sa_periods``, ``combine_directions``, ``seismic_envelope``), an independent
  implementation of CQC, SRSS and 100/30;
* hand values, written as literals.
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pytest

from apeGmsh.opensees.analysis.spectrum import (
    combine_directions,
    combine_modes,
    seismic_envelope,
)
from apeGmsh.results.user import Combination, StateAddress, Term, UserResult
from apeGmsh.results.user._descriptor import resolve
from apeGmsh.results.user._evaluate import (
    SUPERPOSITION_BADGE,
    Case,
    CombinationRefused,
    Field,
    evaluate,
    is_signed,
)
from apeGmsh.results.user._states import _UserResultSource, node_keys

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "user_results"
DOC = json.loads((FIXTURE / "conformance.json").read_text(encoding="utf-8"))
RTOL = DOC["rtol"]


def _nan(v: Optional[float]) -> float:
    return math.nan if v is None else float(v)


def _field(f: dict[str, Any], scale: float = 1.0) -> Field:
    values = np.array([[_nan(v) for v in row] for row in f["values"]], dtype="f8")
    return Field(node_keys([k[0] for k in f["keys"]]), tuple(f["components"]), scale * values)


def _address(text: str) -> StateAddress:
    kind, source, step, mode = text.split("|")
    return StateAddress(kind, source, int(step), int(mode))


def _table(rows: Optional[list[dict[str, Any]]] = None):
    rows = DOC["combinations"] if rows is None else rows
    return resolve([
        Combination(
            c["name"], c["rule"],
            tuple(Term(t["kind"], t["source"], t["step"], t["mode"], t["factor"]) for t in c["terms"]),
            envelope=c["envelope"], fraction=c["fraction"], damping=c["damping"],
        )
        for c in rows
    ])


def _case(c: dict[str, Any]) -> Case:
    return Case(
        mode=np.array(c["mode"], dtype="i4"), lam=np.array(c["lambda"], dtype="f8"),
        sa_period=np.array(c["sa_period"], dtype="f8"),
        damping=np.array([_nan(v) for v in c["damping"]], dtype="f8"),
        increments=tuple(_field(f) for f in c["increments"]),
    )


class FixtureSource:
    """The recorded quantity of the fixture: every state, case and eigenvalue."""

    linear = True

    def __init__(self, doc: dict[str, Any] = DOC) -> None:
        self.doc = doc

    def stored(self, name: str) -> Optional[Field]:
        return None

    def field(self, address: StateAddress) -> Field:
        key = f"{address.kind}|{address.source}|{address.step}|{address.mode}"
        if key not in self.doc["states"]:
            raise CombinationRefused("", "unresolved", f"no state {key}")
        return _field(self.doc["states"][key])

    def eigenvalue(self, address: StateAddress) -> float:
        return float(self.doc["eigenvalues"][
            f"{address.kind}|{address.source}|{address.step}|{address.mode}"])

    def case(self, name: str) -> Case:
        return _case(self.doc["cases"][name])


def _user_result(name: str) -> UserResult:
    """A nodal user result holding every state of the fixture, scaled, plus its stored rows."""
    q = DOC["quantities"][name]
    states = [_address(a) for a in DOC["states"]]
    rows = [_field(DOC["states"][a], q["scale"]) for a in DOC["states"]]
    for combo, f in q["stored"].items():
        states.append(StateAddress("combination", combo, -1, 0))
        rows.append(_field(f))
    # Every state field of the fixture is on nodes 1-3, except W (1-2, ux).
    data = np.full((len(rows), 3, 2), np.nan)
    for i, f in enumerate(rows):
        for r, key in enumerate(f.keys["id"]):
            for c, label in enumerate(f.components):
                data[i, int(key) - 1, ("ux", "uy").index(label)] = f.values[r, c]
    return UserResult(
        name=name, target="nodes", basis="model", states=tuple(states), linear=q["linear"],
        components=("ux", "uy"), data=data, ids=np.array([1, 2, 3], dtype="i4"),
    )


class _Eigen:
    def eigenvalue(self, address: StateAddress) -> float:
        return FixtureSource().eigenvalue(address)


def _source(quantity: str) -> Any:
    if quantity == "recorded":
        return FixtureSource()
    return _UserResultSource(_user_result(quantity), _Eigen(), quantity)  # type: ignore[arg-type]


def _assert_golden(got, want: dict[str, Any]) -> None:
    assert got.signed is want["signed"]
    assert list(got.badges) == want["badges"]
    assert list(got.field.components) == want["components"]
    assert got.field.keys["id"].tolist() == [k[0] for k in want["keys"]]
    expected = np.array([[_nan(v) for v in row] for row in want["values"]])
    np.testing.assert_allclose(got.field.values, expected, rtol=RTOL, atol=0.0)


CASES = [(q, n) for q, goldens in DOC["goldens"].items() for n in goldens]


@pytest.mark.parametrize("quantity, name", CASES)
def test_every_rule_matches_the_conformance_golden(quantity: str, name: str) -> None:
    want = DOC["goldens"][quantity][name]
    if "refused" in want:
        with pytest.raises(CombinationRefused) as err:
            evaluate(_table(), name, _source(quantity))
        assert err.value.reason == want["refused"]
        return
    _assert_golden(evaluate(_table(), name, _source(quantity)), want)


def test_the_committed_fixture_is_what_the_script_writes() -> None:
    spec = importlib.util.spec_from_file_location("make_conformance", FIXTURE / "make_conformance.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.dumps(module.build()) == (FIXTURE / "conformance.json").read_text(encoding="utf-8")


def test_the_fixture_covers_every_rule_and_both_signs() -> None:
    rules = {c["rule"] for c in DOC["combinations"]}
    assert rules == {"linear", "srss", "cqc", "abs_sum", "directional", "envelope"}
    envelopes = {c["envelope"] for c in DOC["combinations"] if c["rule"] == "envelope"}
    assert envelopes == {"max", "min", "absmax"}
    goldens = DOC["goldens"]["recorded"].values()
    assert {g["signed"] for g in goldens if "signed" in g} == {True, False}
    assert {g["refused"] for g in DOC["goldens"]["wood_armer_top"].values() if "refused" in g} == {
        "not-linear"}


# ---- hand values -------------------------------------------------------------


def _value(name: str) -> np.ndarray:
    return evaluate(_table(), name, FixtureSource()).field.values


def test_hand_values() -> None:
    # ULS = 1.2 D + 1.6 L at node 1: 1.2 + 0.8 and 2.4 - 1.6.
    np.testing.assert_allclose(_value("ULS")[0], [2.0, 0.8], rtol=1e-15)
    # srss of D and -2 L at node 2: sqrt(3² + 4²) and sqrt(4² + 0²).
    np.testing.assert_allclose(_value("srss_list")[1], [5.0, 4.0], rtol=1e-15)
    # abs_sum at node 2: |3| + |-4| and |4| + |0|.
    np.testing.assert_allclose(_value("abs_list")[1], [7.0, 4.0], rtol=1e-15)
    # The join: W has no node 3 and no uy, so D + W is NaN there.
    got = _value("D+W")
    assert got[0, 0] == 11.0 and got[1, 0] == 23.0
    assert np.isnan(got[2, 0]) and np.all(np.isnan(got[:, 1]))
    # absmax of -D and 2 L at node 3: max(|-5|, |-6|) and max(|-6|, |2|).
    np.testing.assert_allclose(_value("env_absmax")[2], [6.0, 6.0], rtol=1e-15)


def test_nan_propagates_through_envelope() -> None:
    """E's NaN at node 1, uy: ``nanmax`` would read 2.0 there (the nanmax mutant)."""
    got = _value("env_max")
    assert np.isnan(got[0, 1])
    assert got[0, 0] == 1.0


def test_sign_is_derived_from_the_rule_and_terms() -> None:
    table = _table()
    assert is_signed(table, "ULS") and is_signed(table, "twice_ULS") and is_signed(table, "k_linear")
    assert not is_signed(table, "plus"), "linear over an unsigned term is unsigned"
    for name in ("srss_list", "modal_cqc", "dir", "env_max", "seismic_max"):
        assert not is_signed(table, name)


def test_the_superposition_badge_is_on_a_linear_over_a_stage() -> None:
    src = FixtureSource()
    assert evaluate(_table(), "ULS", src).badges == (SUPERPOSITION_BADGE,)
    assert evaluate(_table(), "k_linear", src).badges == ()
    assert evaluate(_table(), "twice_ULS", src).badges == ()


# ---- the spectrum kernels as a second oracle ---------------------------------


def _stack(fields: list[dict[str, Any]]) -> np.ndarray:
    return np.stack([_field(f).values for f in fields])


def _ex() -> dict[str, Any]:
    return DOC["cases"]["Ex"]


def test_whole_case_cqc_equals_combine_modes_with_sa_periods() -> None:
    ex = _ex()
    want = combine_modes(
        _stack(ex["increments"]), rule="CQC", omega=np.sqrt(ex["lambda"]),
        damping=np.array(ex["damping"]), sa_periods=np.array(ex["sa_period"]))
    np.testing.assert_allclose(_value("Ex_cqc"), want, rtol=RTOL)


def test_dropping_sa_period_changes_the_value() -> None:
    """The dropped-``sa_period`` mutant: the rigid pair as two modes is another number."""
    ex = _ex()
    ignored = combine_modes(
        _stack(ex["increments"]), rule="CQC", omega=np.sqrt(ex["lambda"]),
        damping=np.array(ex["damping"]))
    assert not np.allclose(_value("Ex_cqc"), ignored, rtol=1e-6)


def test_a_finite_row_damping_overrides_the_case() -> None:
    ex = _ex()
    want = combine_modes(
        _stack(ex["increments"]), rule="CQC", omega=np.sqrt(ex["lambda"]),
        damping=0.02, sa_periods=np.array(ex["sa_period"]))
    np.testing.assert_allclose(_value("Ex_cqc_d02"), want, rtol=RTOL)


def test_whole_case_srss_equals_combine_modes() -> None:
    ex = _ex()
    want = 2.0 * combine_modes(
        _stack(ex["increments"]), rule="SRSS", sa_periods=np.array(ex["sa_period"]))
    np.testing.assert_allclose(_value("Ex_srss"), want, rtol=RTOL)


def test_cqc_over_modes_equals_combine_modes() -> None:
    fields = [DOC["states"][f"mode|eig|-1|{k}"] for k in (1, 2, 3)]
    lam = [DOC["eigenvalues"][f"mode|eig|-1|{k}"] for k in (1, 2, 3)]
    want = combine_modes(_stack(fields), rule="CQC", omega=np.sqrt(lam), damping=0.05)
    np.testing.assert_allclose(_value("modal_cqc"), want, rtol=RTOL)
    np.testing.assert_allclose(
        _value("modal_srss"), combine_modes(_stack(fields), rule="SRSS"), rtol=RTOL)


def test_directional_equals_combine_directions() -> None:
    per_direction = np.stack([_value("Ex_cqc"), _value("Ey_cqc")])
    want = combine_directions(per_direction, method="100-30")
    np.testing.assert_allclose(_value("dir"), want, rtol=RTOL)


def test_the_gravity_plus_seismic_envelope_equals_seismic_envelope() -> None:
    """D5 as A1 corrects it: envelope over linear(S, +E) and linear(S, -E)."""
    up, lo = seismic_envelope(_value("ULS"), _value("dir"))
    np.testing.assert_allclose(_value("seismic_max"), up, rtol=RTOL)
    np.testing.assert_allclose(_value("seismic_min"), lo, rtol=RTOL)


# ---- per quantity: a combined force comes from per-term forces (ADR 0124 D6) --


def _forces_case() -> tuple[Case, np.ndarray, np.ndarray]:
    """Per-mode element forces N and per-mode displacements u, on one bar.

    N is not a linear map of the combined u: |N| is computed per mode and
    combined, while a force read off the combined, unsigned displacement has
    lost the modal signs.
    """
    u = np.array([[[0.5], [1.0]], [[0.6], [-0.8]]])        # (mode, node, ux)
    n_force = 100.0 * (u[:, 1, 0] - u[:, 0, 0])          # (mode,) k (u2 - u1)
    return Case(
        mode=np.array([1, 2]), lam=np.array([100.0, 110.0]), sa_period=np.zeros(2),
        damping=np.array([0.05, 0.05]),
        increments=tuple(Field(node_keys([7]), ("N",), np.array([[f]])) for f in n_force),
    ), u, n_force


def test_a_combined_force_comes_from_the_per_term_forces() -> None:
    case, u, n_force = _forces_case()

    class Forces(FixtureSource):
        def case(self, name: str) -> Case:
            return case

    table = resolve([Combination("E", "cqc", (Term("spectrum", "F"),))])
    got = evaluate(table, "E", Forces()).field.values[0, 0]
    want = float(combine_modes(n_force, rule="CQC", omega=np.sqrt(case.lam), damping=0.05))
    assert got == pytest.approx(want, rel=1e-12)
    # The force of the combined displacement: CQC of u at each node, then k (u2 - u1).
    u_cqc = combine_modes(u[:, :, 0], rule="CQC", omega=np.sqrt(case.lam), damping=0.05)
    assert abs(got - 100.0 * (u_cqc[1] - u_cqc[0])) > 1.0


# ---- user results under a combination (A5) -----------------------------------


def test_a_non_linear_user_result_refuses_every_summing_rule() -> None:
    """The summing-``wood_armer_*`` mutant: ULS of wood_armer would be read as a sum."""
    rows = [
        Combination("lin", "linear", (Term("stage", "D", factor=1.2), Term("stage", "L", factor=1.6))),
        Combination("srss", "srss", (Term("stage", "D"), Term("stage", "L"))),
        Combination("abs", "abs_sum", (Term("stage", "D"), Term("stage", "L"))),
        Combination("dir", "directional", (Term("stage", "D"), Term("stage", "L")), fraction=0.3),
        Combination("cqc", "cqc", (Term("mode", "eig", -1, 1), Term("mode", "eig", -1, 2)),
                    damping=0.05),
        Combination("env", "envelope", (Term("stage", "D"), Term("stage", "L")), envelope="max"),
    ]
    source = _source("wood_armer_top")
    for row in rows[:-1]:
        with pytest.raises(CombinationRefused) as err:
            evaluate(resolve(rows), row.name, source)
        assert err.value.reason == "not-linear", row.name
    got = evaluate(resolve(rows), "env", source).field.values
    np.testing.assert_allclose(got[1], [3.0 * 3.0, 3.0 * 4.0])  # max(3 D, 3 L) at node 2


def test_a_stored_row_wins_over_the_rule() -> None:
    got = evaluate(_table(), "ULS", _source("wood_armer_top"))
    np.testing.assert_array_equal(got.field.values, [[9.0, 8.0], [7.0, 6.0], [5.0, 4.0]])


def test_a_linear_user_result_sums() -> None:
    got = evaluate(_table(), "ULS", _source("m_xx")).field.values
    np.testing.assert_allclose(got[0], [0.5 * 2.0, 0.5 * 0.8], rtol=1e-15)


def test_a_state_free_user_result_is_not_combined() -> None:
    free = UserResult(name="R", target="nodes", basis="model", components=("R",),
                      data=np.ones((1, 1, 1)), ids=np.array([1], dtype="i4"))
    with pytest.raises(CombinationRefused) as err:
        _UserResultSource(free, None, "x")
    assert err.value.reason == "state-free"


# ---- refusals ------------------------------------------------------------------


def test_a_refused_row_is_refused_on_evaluation() -> None:
    table = resolve([Combination("bad", "median", (Term("stage", "D"),))])
    with pytest.raises(CombinationRefused) as err:
        evaluate(table, "bad", FixtureSource())
    assert err.value.reason == "refused"
    with pytest.raises(CombinationRefused) as err:
        evaluate(table, "nobody", FixtureSource())
    assert err.value.reason == "unresolved"


def test_terms_on_different_targets_are_refused() -> None:
    class Mixed(FixtureSource):
        def field(self, address: StateAddress) -> Field:
            if address.source == "L":
                from apeGmsh.results.user._states import sub_keys
                return Field(sub_keys([1], [0]), ("ux",), np.ones((1, 1)))
            return super().field(address)

    with pytest.raises(CombinationRefused) as err:
        evaluate(_table(), "ULS", Mixed())
    assert err.value.reason == "bad-term"


def test_entity_terms_join_on_kind_and_row_whatever_the_kind_width() -> None:
    from apeGmsh.results.user._states import entity_keys

    class Entities(FixtureSource):
        def field(self, address: StateAddress) -> Field:
            if address.source == "D":
                return Field(entity_keys(["cut"], [0]), ("V",), np.array([[2.0]]))
            return Field(entity_keys(["cut", "a_long_storey_kind"], [0, 3]), ("V",),
                         np.array([[1.0], [5.0]]))

    got = evaluate(_table(), "ULS", Entities()).field
    assert got.keys["kind"].tolist() == ["a_long_storey_kind", "cut"]
    assert got.keys["row"].tolist() == [3, 0]
    assert np.isnan(got.values[0, 0])  # D has no storey row
    assert got.values[1, 0] == pytest.approx(1.2 * 2.0 + 1.6 * 1.0)
