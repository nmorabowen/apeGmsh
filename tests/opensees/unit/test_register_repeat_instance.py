"""``apeSees.register`` is a no-op on an instance it already holds (#1409).

Before the fix a repeat ``register(t)`` appended ``t`` to the primitive
list a second time. The emitted deck itself stayed single, because
``topological_order`` dedupes by object, but every other reader of the
list saw the duplicate:

* ``element_tags="fem"`` refused a valid model, reading the element as
  fanned out twice;
* ``all_recorder_specs`` listed the recorder twice;
* the order of the list (read last-registered-wins by the strategy rung
  0 in ``analyze``) disagreed with the deck, which keeps first
  appearance.

The oracle in every case is the model registered once.
"""
from __future__ import annotations

from pathlib import Path

from apeGmsh.opensees import apeSees
from apeGmsh.opensees._internal.types import SolutionAlgorithm
from tests.opensees.fixtures import fem_stub


def _frame(mode: str, *, repeat: bool) -> apeSees:
    ops = apeSees(fem_stub.make_two_column_frame(), element_tags=mode)
    ops.model(ndm=3, ndf=6)
    tr = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    el = ops.element.elasticBeamColumn(
        pg="Cols", transf=tr,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4)
    ts = ops.timeSeries.Linear()
    pat = ops.pattern.Plain(series=ts)
    ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))
    if repeat:
        for prim in (tr, el, ts, pat, el, ts):
            assert ops.register(prim) is prim
    return ops


def test_a_repeat_register_keeps_one_entry_and_one_tag() -> None:
    once = _frame("sequential", repeat=False)
    again = _frame("sequential", repeat=True)
    assert len(again._primitives) == len(once._primitives) == 4
    assert [again.tag_for(p) for p in again._primitives] == [
        once.tag_for(p) for p in once._primitives
    ]


def test_registering_a_namespace_handle_keeps_one_entry() -> None:
    """The ``ops.register(ops.timeSeries.Linear())`` idiom of the
    staged-gravity-ssi example: the namespace already registered it."""
    ops = apeSees(fem_stub.make_two_column_frame())
    ops.model(ndm=3, ndf=6)
    ts = ops.register(ops.timeSeries.Linear())
    assert sum(p is ts for p in ops._primitives) == 1


def test_fem_tags_accept_a_repeat_registered_element(tmp_path: Path) -> None:
    for kind in ("tcl", "py"):
        ref, got = tmp_path / f"once.{kind}", tmp_path / f"again.{kind}"
        getattr(_frame("fem", repeat=False), kind)(str(ref))
        getattr(_frame("fem", repeat=True), kind)(str(got))
        assert got.read_text() == ref.read_text()


def test_a_repeat_registered_recorder_is_listed_once() -> None:
    ops = apeSees(fem_stub.make_two_column_frame())
    ops.model(ndm=3, ndf=6)
    rec = ops.recorder.Node(
        file="g.out", response="disp", nodes=(1,), dofs=(1, 2))
    ops.register(rec)
    assert ops.all_recorder_specs == (("global", rec),)


def test_the_last_algorithm_stays_the_deck_order(tmp_path: Path) -> None:
    """OpenSees runs the last ``algorithm`` the deck declares; the list
    order the strategy rung 0 reads must name the same one."""
    ops = apeSees(fem_stub.make_two_column_frame())
    ops.model(ndm=3, ndf=6)
    newton = ops.algorithm.Newton()
    ops.algorithm.ModifiedNewton()
    ops.register(newton)
    deck = tmp_path / "algo.tcl"
    ops.tcl(str(deck))
    in_deck = [
        ln.split()[1] for ln in deck.read_text().splitlines()
        if ln.startswith("algorithm ")
    ]
    in_list = [
        type(p).__name__ for p in ops._primitives
        if isinstance(p, SolutionAlgorithm)
    ]
    assert in_deck == in_list == ["Newton", "ModifiedNewton"]
