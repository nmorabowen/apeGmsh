"""ADR 0115 D2 (R2-b, #1695) without a solver: the read session's
refusals, the route refusals, ``ops.elements`` beside ``ops.element``,
and ``recorder.Element(label=)`` under both tag modes.

A fake openseespy module records every call, so each refusal is shown to
raise before any call into openseespy (the card's "a raise after
registration" seam). The live oracles are in
``tests/opensees/live/test_readback_live.py``.
"""
from __future__ import annotations

from typing import Any

import pytest

from apeGmsh.opensees import _readback
from apeGmsh.opensees._internal.build.errors import BridgeError
from apeGmsh.opensees._internal.ns.element import _ElementNS
from apeGmsh.opensees._internal.tag_plan import emit_mode
from apeGmsh.opensees._readback import (
    DroppedSession,
    ElementSet,
    _ElementAccessor,
    open_session,
)
from apeGmsh.opensees.apesees import apeSees
from apeGmsh.opensees.emitter import live as live_mod
from apeGmsh.opensees.emitter.tcl import TclEmitter

from tests.opensees.live.test_readback_live import (
    build_cantilever_fem,
    build_pratt_fem,
    cantilever_bridge,
    pratt_bridge,
)
from tests.opensees.unit.test_label_selector_fix_merge import (
    build_footing_fem,
)


class FakeOps:
    """Answers the read queries and records every call."""

    def __init__(self, ndf: int = 2) -> None:
        self.calls: list[str] = []
        self.ndf = ndf

    def getNDF(self, tag: int) -> list[int]:
        self.calls.append("getNDF")
        return [self.ndf]

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)

        def _call(*args: Any) -> Any:
            self.calls.append(name)
            if name == "eleResponse":
                return [7.0]
            return 3.0
        return _call


class FakeEmitter:
    def __init__(self, ops: FakeOps) -> None:
        self.ops = ops


@pytest.fixture(scope="module")
def pratt() -> Any:
    return build_pratt_fem()


@pytest.fixture(scope="module")
def cantilever() -> Any:
    return build_cantilever_fem()


def up_bridge(fem: Any) -> apeSees:
    """LadrunoUP over the footing plate, ndm=2, ndf=3 (u_x, u_y, p)."""
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=3)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e4, nu=0.3, rho=0.0)
    ops.element.LadrunoUP(pg="Plate", material=mat, Kf=2.2e6, poro=0.4,
                          rhoF=1.0, perm=(1e-4, 1e-4))
    return ops


@pytest.fixture(scope="module")
def footing_up() -> apeSees:
    return up_bridge(build_footing_fem())


def with_session(ops: apeSees, fake: FakeOps, *, ndm: int = 2) -> None:
    """Open a read session over ``fake`` with the plan the live emit
    would read (made here, outside any read)."""
    bm = ops.build()
    plan = bm._tag_plan(emit_mode(bm, split=False, supports_partitions=False))
    ops._read_state = open_session(
        FakeEmitter(fake),  # type: ignore[arg-type]
        ndm=ndm, element_plan=tuple(plan.elements.specs),
    )


# ---------------------------------------------------------------------------
# The session answers through the plan and the DOF layout
# ---------------------------------------------------------------------------


def test_reads_go_through_the_session(pratt: Any) -> None:
    ops = pratt_bridge(pratt)
    fake = FakeOps()
    with_session(ops, fake)
    assert ops.nodes.one(label="mid").read("displacement_y") == 3.0
    assert fake.calls == ["getNDF", "nodeDisp"]
    fake.calls.clear()
    ops.nodes.get(label="supports").read("reaction_force_y")
    assert fake.calls == ["getNDF", "getNDF", "reactions",
                          "nodeReaction", "nodeReaction"]
    fake.calls.clear()
    assert list(ops.elements.get(label="top2").read("axial_force")) == [7.0]
    assert fake.calls == ["eleResponse"]


def test_element_read_uses_the_planned_tag_not_the_fem_id(pratt: Any) -> None:
    """Invariant 3 at the seam: the tag sent to eleResponse is the plan's,
    which under sequential tags with reversed families is not the FEM id."""
    ops = pratt_bridge(pratt, reverse=True)
    sent: list[int] = []

    class Spy(FakeOps):
        def eleResponse(self, tag: int, token: str) -> list[float]:
            sent.append(int(tag))
            assert token == "axialForce"
            return [float(tag)]

    with_session(ops, Spy())
    eid = ops.elements.one(label="top2").id
    session = ops._read_state
    assert isinstance(session, _readback.ReadSession)
    [(spec, tag)] = session.element_rows((eid,), "t")
    assert tag != eid
    assert ops.elements.one(label="top2").read("axial_force") == float(tag)
    assert sent == [tag]


# ---------------------------------------------------------------------------
# Refusals before any call into openseespy
# ---------------------------------------------------------------------------


def test_no_session_names_ops_results(pratt: Any) -> None:
    ops = pratt_bridge(pratt)
    with pytest.raises(BridgeError, match=r"no live analyze.*ops\.results"):
        ops.nodes.one(label="mid").read("displacement_y")
    with pytest.raises(BridgeError, match=r"ops\.results\(path\)"):
        ops.elements.get().read("axial_force")


@pytest.mark.parametrize("verb", ["tcl", "py"])
def test_a_deck_run_closes_the_session_naming_ops_results(
    pratt: Any, tmp_path: Any, monkeypatch: pytest.MonkeyPatch, verb: str,
) -> None:
    from apeGmsh.opensees import apesees as apesees_mod

    monkeypatch.setattr(apesees_mod, "stream_run", lambda *a, **k: None)
    monkeypatch.setattr(
        apesees_mod, "_resolve_opensees_binary", lambda *a, **k: "x")
    monkeypatch.setattr(
        apesees_mod, "_resolve_python_binary", lambda *a, **k: "x")
    ops = pratt_bridge(pratt)
    fake = FakeOps()
    with_session(ops, fake)
    getattr(ops, verb)(str(tmp_path / f"deck.{verb}"), run=False)
    ops.nodes.one(label="mid").read("displacement_y")   # emit only: open
    getattr(ops, verb)(str(tmp_path / f"deck.{verb}"), run=True)
    fake.calls.clear()
    with pytest.raises(BridgeError, match=rf"ops\.{verb}\(\.\.\., run=True\)"
                       r".*ops\.results\(path\)"):
        ops.nodes.one(label="mid").read("displacement_y")
    assert fake.calls == []


def test_a_new_live_emitter_makes_the_session_stale(
    pratt: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invariant 4 through the epoch: any LiveOpsEmitter (another bridge,
    a procedure) rebuilds the domain."""
    ops = pratt_bridge(pratt)
    fake = FakeOps()
    with_session(ops, fake)
    before = live_mod.domain_epoch()
    monkeypatch.setattr(live_mod, "_get_ops", lambda: FakeOps())
    live_mod.LiveOpsEmitter(wipe=False)
    assert live_mod.domain_epoch() == before + 1
    fake.calls.clear()
    with pytest.raises(BridgeError, match="domain was rebuilt"):
        ops.nodes.one(label="mid").read("displacement_y")
    with pytest.raises(BridgeError, match="domain was rebuilt"):
        ops.elements.get().read("axial_force")
    assert fake.calls == []


def test_a_dropped_session_names_its_cause(pratt: Any) -> None:
    ops = pratt_bridge(pratt)
    ops._read_state = DroppedSession("the reason")
    with pytest.raises(BridgeError, match="closed: the reason"):
        ops.nodes.get().read("displacement_x")


@pytest.mark.parametrize("bad", ["displacement", "displacement_w",
                                 "stress_x", "pore_pressure", "force_x"])
def test_an_unknown_node_component_refuses_first(
    pratt: Any, bad: str,
) -> None:
    ops = pratt_bridge(pratt)
    fake = FakeOps()
    with_session(ops, fake)
    with pytest.raises(BridgeError, match="unknown node component"):
        ops.nodes.one(label="mid").read(bad)
    assert fake.calls == []


@pytest.mark.parametrize("bad", ["stress", "global_force", "axialForce"])
def test_a_non_scalar_element_component_points_to_results(
    pratt: Any, bad: str,
) -> None:
    ops = pratt_bridge(pratt)
    fake = FakeOps()
    with_session(ops, fake)
    with pytest.raises(BridgeError, match="Results"):
        ops.elements.get().read(bad)
    assert fake.calls == []


def test_selector_refusals_make_no_call(pratt: Any) -> None:
    ops = pratt_bridge(pratt)
    fake = FakeOps()
    with_session(ops, fake)
    with pytest.raises(BridgeError, match="Labels that exist"):
        ops.elements.get(label="nope")
    with pytest.raises(ValueError, match="empty"):
        ops.elements.get(label=[])
    with pytest.raises(ValueError, match="empty"):
        ops.elements.get(ids=[])
    with pytest.raises(KeyError):
        ops.elements.get(ids=[999999])
    with pytest.raises(KeyError):
        ops.elements.get(pg="Nope")
    with pytest.raises(ValueError, match="at most one"):
        ops.elements.get(pg="Webs", label="top2")
    with pytest.raises(ValueError, match="exactly one"):
        ops.elements.one()
    with pytest.raises(BridgeError, match=r"selects \d+ elements, not exactly"):
        ops.elements.one(pg="Chords")
    with pytest.raises(BridgeError, match="Labels that exist"):
        ops.nodes.get(label="nope")
    with pytest.raises(ValueError, match="empty"):
        ops.nodes.get(label=[])
    # A point-only label carries no elements.
    with pytest.raises(BridgeError, match="selects 0 elements"):
        ops.elements.get(label="mid")
    assert fake.calls == []


def test_an_element_with_no_planned_element_refuses(pratt: Any) -> None:
    """A FEM element no declaration fans out over has no tag to read."""
    ops = apeSees(pratt, default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.uniaxialMaterial.ElasticMaterial(E=1.0)
    ops.element.Truss(pg="Chords", A=1.0, material=mat)
    fake = FakeOps()
    with_session(ops, fake)
    with pytest.raises(BridgeError, match="no element was emitted"):
        ops.elements.get(pg="Webs").read("axial_force")
    assert fake.calls == []


def test_a_component_the_node_lacks_names_the_node_and_its_ndf(
    pratt: Any,
) -> None:
    """Invariant 5: a slot the node's ndf does not carry, and an ndf the
    layout does not name, both refuse."""
    ops = pratt_bridge(pratt)
    fake = FakeOps(ndf=3)
    with_session(ops, fake, ndm=2)
    mid = ops.nodes.one(label="mid")
    with pytest.raises(BridgeError,
                       match=rf"node {mid.tag} has no 'displacement_z': "
                             r"it has ndf=3 \(ndm=2\)"):
        mid.read("displacement_z")
    with_session(ops, FakeOps(ndf=4), ndm=3)
    with pytest.raises(BridgeError, match="ndf=4"):
        mid.read("displacement_x")


class DispSpy(FakeOps):
    """Records the ``(tag, dof)`` of every ``nodeDisp``."""

    def __init__(self, ndf: int) -> None:
        super().__init__(ndf=ndf)
        self.seen: list[tuple[int, int]] = []

    def nodeDisp(self, tag: int, dof: int) -> float:
        self.seen.append((int(tag), int(dof)))
        return 0.5


def test_a_beam_node_reads_rotation_z_at_dof_3(cantilever: Any) -> None:
    """The DOF layout (not the capture table's DOF 6) at ndf=3, ndm=2, on
    a node a beam-column establishes as rotational."""
    ops = cantilever_bridge(cantilever)
    spy = DispSpy(ndf=3)
    with_session(ops, spy, ndm=2)
    tip = ops.nodes.one(label="tip")
    assert tip.read("rotation_z") == 0.5
    assert spy.seen == [(tip.tag, 3)]


def test_a_up_node_refuses_every_rotational_component(footing_up: Any) -> None:
    """The R2-b review blocker (#1701): on a LadrunoUP node at ndm=2,
    ndf=3, DOF 3 is pore pressure; a rotational read refuses, naming the
    node, its ndf and the family, before any value is fetched. The
    translations still read."""
    ops = footing_up
    spy = DispSpy(ndf=3)
    with_session(ops, spy, ndm=2)
    node = ops.nodes.get()[0]
    for comp in ("rotation_z", "reaction_moment_z", "angular_velocity_z",
                 "angular_acceleration_z"):
        with pytest.raises(
                BridgeError,
                match=rf"node {node.tag} \(ndf=3, ndm=2\) has no "
                      rf"established '{comp}'.*LadrunoUP.*u-p"):
            node.read(comp)
    assert spy.seen == []
    assert "reactions" not in spy.calls
    assert node.read("displacement_y") == 0.5
    assert spy.seen == [(node.tag, 2)]


def test_a_truss_only_node_refuses_rotation(pratt: Any) -> None:
    """A truss uses the translations only, so it establishes no rotation
    in an ndf=3 slot (a future or unknown family is refused the same way)."""
    ops = pratt_bridge(pratt)
    spy = DispSpy(ndf=3)
    with_session(ops, spy, ndm=2)
    mid = ops.nodes.one(label="mid")
    with pytest.raises(BridgeError, match="no beam-column or shell"):
        mid.read("rotation_z")
    assert spy.seen == []


def test_an_element_the_domain_no_longer_holds_raises(pratt: Any) -> None:
    """A raw wipe outside the bridge leaves eleResponse empty: the read
    raises a BridgeError naming it, not an IndexError."""
    ops = pratt_bridge(pratt)

    class Wiped(FakeOps):
        def eleResponse(self, tag: int, token: str) -> list[float]:
            return []

    with_session(ops, Wiped())
    with pytest.raises(BridgeError, match="no longer holds it"):
        ops.elements.one(label="top2").read("axial_force")


# ---------------------------------------------------------------------------
# ops.elements beside ops.element (shared namespace)
# ---------------------------------------------------------------------------


def test_ops_elements_and_ops_element_do_not_shadow(pratt: Any) -> None:
    ops = apeSees(pratt, default_orientation=None)
    assert isinstance(ops.elements, _ElementAccessor)
    assert isinstance(ops.element, _ElementNS)
    assert hasattr(ops.element, "Truss")
    assert not hasattr(ops.elements, "Truss")
    assert isinstance(ops.elements.get(), ElementSet)
    assert len(ops.elements) == len(pratt.elements.ids)


# ---------------------------------------------------------------------------
# recorder.Element(label=) (ruling on #1695; full form, ADR 0115 amendment R2-f)
# ---------------------------------------------------------------------------


def _deck(ops: apeSees) -> list[str]:
    emitter = TclEmitter()
    ops.build().emit(emitter)
    return emitter.lines()


def test_recorder_element_label_resolves_under_fem_tags(pratt: Any) -> None:
    ops = pratt_bridge(pratt, element_tags="fem")
    eid = ops.elements.one(label="diag2").id
    rec = ops.recorder.Element(file="d.out", response=("axialForce",),
                               label="diag2")
    assert rec.elements == (eid,) and rec.pg is None
    line = next(ln for ln in _deck(ops) if ln.startswith("recorder Element"))
    assert f"-ele {eid} axialForce" in line
    # Two labels union, in snapshot order.
    both = ops.recorder.Element(file="b.out", response=("axialForce",),
                                label=["top2", "bot2"])
    index = pratt.elements.index
    assert both.elements == tuple(sorted(
        (ops.elements.one(label="top2").id,
         ops.elements.one(label="bot2").id), key=index))


def _truss_conn(deck: list[str]) -> dict[int, tuple[str, str]]:
    """``{tag: (iNode, jNode)}`` of every ``element Truss`` line."""
    return {
        int(ln.split()[2]): (ln.split()[3], ln.split()[4])
        for ln in deck if ln.startswith("element Truss ")
    }


def test_recorder_element_label_translates_under_sequential_tags(
    pratt: Any,
) -> None:
    """ADR 0115 amendment R2-f: under sequential tags (families declared
    in reverse, so tags are not FEM ids) ``label=`` resolves to FEM ids
    at the call and the build translates them. The oracle is the
    connectivity: the recorded tag is the one whose ``element`` line
    joins the nodes the FEM element joins in an ``element_tags="fem"``
    deck, where the tag is the FEM id."""
    ops = pratt_bridge(pratt, reverse=True)
    eid = ops.elements.one(label="diag2").id
    rec = ops.recorder.Element(file="d.out", response=("axialForce",),
                               label="diag2")
    assert rec.elements == (eid,) and rec.label == ("diag2",)
    deck = _deck(ops)
    fem_deck = _deck(pratt_bridge(pratt, element_tags="fem"))
    conn = _truss_conn(fem_deck)[eid]
    [tag] = [t for t, c in _truss_conn(deck).items() if c == conn]
    assert tag != eid  # the case the translation exists for
    line = next(ln for ln in deck if ln.startswith("recorder Element"))
    assert line.endswith(f"-ele {tag} axialForce")


@pytest.mark.parametrize("element_tags", ["sequential", "fem"])
def test_archived_element_recorder_ids_share_one_id_space(
    pratt: Any, tmp_path: Any, element_tags: str,
) -> None:
    """ADR 0115 amendment R2-f, the maintainer's condition on #1773: the
    archived ``decl_params`` ``elements`` of an ``Element`` recorder are
    OpenSees tags whether it was declared by ``label=`` or by explicit
    ids, under both tag modes. The ids-declared twin names the tag the
    deck's ``element`` line gives the labelled bar (the connectivity
    oracle above)."""
    from apeGmsh.opensees.emitter import h5_reader

    ops = pratt_bridge(pratt, element_tags=element_tags, reverse=True)
    eid = ops.elements.one(label="diag2").id
    conn = _truss_conn(_deck(pratt_bridge(pratt, element_tags="fem")))[eid]
    [tag] = [t for t, c in _truss_conn(_deck(ops)).items() if c == conn]
    ops.recorder.Element(file="l.out", response=("axialForce",),
                         label="diag2", name="by_label")
    ops.recorder.Element(file="i.out", response=("axialForce",),
                         elements=(tag,), name="by_ids")
    path = tmp_path / "ids.h5"
    ops.h5(str(path))
    with h5_reader.open(str(path)) as m:
        table = m.declarations()
    assert table is not None
    (by_label,) = table.params_for("opensees/recorder/by_label")
    (by_ids,) = table.params_for("opensees/recorder/by_ids")
    assert by_label.params["elements"] == by_ids.params["elements"] == (tag,)
    assert by_label.params["label"] == ("diag2",)
    assert by_ids.params["label"] is None
    if element_tags == "sequential":
        assert tag != eid  # the case where the id spaces differ


def test_recorder_element_label_needs_the_tag_map(pratt: Any) -> None:
    """A label recorder's ids are FEM ids: without the bridge's map,
    materialize refuses rather than emit them as tags."""
    ops = pratt_bridge(pratt)
    rec = ops.recorder.Element(file="d.out", response=("axialForce",),
                               label="diag2")
    with pytest.raises(BridgeError, match="FEM element ids"):
        rec.materialize(TclEmitter(), pratt, None, None)


def test_recorder_element_label_refusals(pratt: Any) -> None:
    ops = pratt_bridge(pratt, element_tags="fem")
    before = len(ops._primitives)
    with pytest.raises(ValueError, match="exactly one"):
        ops.recorder.Element(file="d.out", response=("axialForce",),
                             label="diag2", pg="Webs")
    with pytest.raises(BridgeError, match="Labels that exist"):
        ops.recorder.Element(file="d.out", response=("axialForce",),
                             label="Webs")
    assert len(ops._primitives) == before
