"""Unit tests for Phase SSI-2.B per-stage topology activation.

Covers:

1. Protocol contract — ``domain_change()``.
2. RecordingEmitter / Tcl / Py emit shape.
3. ``s.activate(pgs=[...])`` records PG names; validates non-empty
   strings; multiple ``activate()`` calls accumulate.
4. ``compute_stage_ownership`` — element ownership matches PG
   activation; node ownership is the lowest-index stage that
   references it; nodes shared with global elements stay global.
5. Bridge emit ordering: stage-bound nodes + elements emit inside
   the stage's block (after stage_open, before domain_change);
   global nodes + elements emit before any stage_open.
6. Same PG activated by two stages → BridgeError.
"""
from __future__ import annotations

import inspect
from typing import get_type_hints

import pytest

from apeGmsh.opensees.apesees import apeSees
from apeGmsh.opensees._internal.build import (
    BridgeError,
    StageRecord,
    compute_stage_ownership,
)
from apeGmsh.opensees.emitter.base import Emitter
from apeGmsh.opensees.emitter.py import PyEmitter
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.emitter.tcl import TclEmitter

from tests.opensees.fixtures.fem_stub import (
    FEMStub,
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
)


# ---------------------------------------------------------------------------
# 1-2. Protocol + emitter shape for domain_change
# ---------------------------------------------------------------------------


def test_emitter_protocol_has_domain_change() -> None:
    assert hasattr(Emitter, "domain_change")
    hints = get_type_hints(Emitter.domain_change)
    assert hints.get("return") is type(None)
    sig = inspect.signature(Emitter.domain_change)
    assert [p.name for p in sig.parameters.values()] == ["self"]


def test_recording_captures_domain_change() -> None:
    e = RecordingEmitter()
    e.domain_change()
    assert e.calls == [("domain_change", (), {})]


def test_tcl_domain_change_shape() -> None:
    e = TclEmitter()
    e.domain_change()
    assert e.lines()[-1] == "domainChange"


def test_py_domain_change_shape() -> None:
    e = PyEmitter()
    e.domain_change()
    assert e.lines()[-1] == "ops.domainChange()"


# ---------------------------------------------------------------------------
# 3. s.activate(pgs=) builder behavior
# ---------------------------------------------------------------------------


def _make_two_pg_fem() -> FEMStub:
    """Two quads side-by-side: rock (left) + cimbra (right) sharing
    nodes 2 and 3 along the interface."""
    return FEMStub(
        nodes=_NodesStub(
            ids=[1, 2, 3, 4, 5, 6],
            coords=[
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
                (2.0, 0.0, 0.0),
                (2.0, 1.0, 0.0),
            ],
            node_pgs={"Left": [1, 4], "Right": [5, 6]},
        ),
        elements=_ElementsStub(
            elem_pgs={
                "rock":   _ElementGroupView(
                    ids=(1,), connectivity=((1, 2, 3, 4),),
                ),
                "cimbra": _ElementGroupView(
                    ids=(2,), connectivity=((2, 5, 6, 3),),
                ),
            },
        ),
    )


def _full_chain(ops: apeSees) -> dict[str, object]:
    return {
        "test":        ops.test.NormDispIncr(tol=1e-4, max_iter=50),
        "algorithm":   ops.algorithm.Newton(),
        "integrator":  ops.integrator.LoadControl(dlam=0.1),
        "constraints": ops.constraints.Plain(),
        "numberer":    ops.numberer.RCM(),
        "system":      ops.system.UmfPack(),
        "analysis":    ops.analysis.Static(),
    }


def test_stage_activate_records_pgs() -> None:
    fem = _make_two_pg_fem()
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.FourNodeQuad(pg="rock", thickness=1.0, material=mat)
    ops.element.FourNodeQuad(pg="cimbra", thickness=1.0, material=mat)

    with ops.stage(name="A") as s:
        s.activate(pgs=["cimbra"])
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)

    assert ops._stage_records[0].activated_pgs == ("cimbra",)


def test_stage_activate_accumulates() -> None:
    fem = _make_two_pg_fem()
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=2)
    with ops.stage(name="A") as s:
        s.activate(pgs=["cimbra"])
        s.activate(pgs=["other"])
        s.activate(pgs=["cimbra"])  # duplicate — collapses
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    assert ops._stage_records[0].activated_pgs == ("cimbra", "other")


def test_stage_activate_rejects_empty_strings() -> None:
    fem = _make_two_pg_fem()
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=2)
    with pytest.raises(ValueError, match="iterable of non-empty strings"):
        with ops.stage(name="A") as s:
            s.activate(pgs=[""])


# ---------------------------------------------------------------------------
# 4. compute_stage_ownership
# ---------------------------------------------------------------------------


def _make_stage(name: str, pgs: tuple[str, ...]) -> StageRecord:
    """Build a minimal StageRecord for ownership-computation tests."""
    return StageRecord(
        name=name,
        initial_stress_records=(),
        test=None, algorithm=None, integrator=None,
        constraints=None, numberer=None, system=None, analysis=None,
        n_increments=1, dt=None,
        activated_pgs=pgs,
    )


def test_compute_ownership_global_when_no_activation() -> None:
    """Element whose pg isn't activated stays out of element_owner;
    its nodes stay out of node_owner."""
    fem = _make_two_pg_fem()
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    rock = ops.element.FourNodeQuad(pg="rock", thickness=1.0, material=mat)
    cim  = ops.element.FourNodeQuad(pg="cimbra", thickness=1.0, material=mat)

    # No stages declared at all.
    elem_owner, node_owner = compute_stage_ownership((), [rock, cim], fem)
    assert elem_owner == {}
    assert node_owner == {}


def test_compute_ownership_stage_bound_element() -> None:
    fem = _make_two_pg_fem()
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    rock = ops.element.FourNodeQuad(pg="rock", thickness=1.0, material=mat)
    cim  = ops.element.FourNodeQuad(pg="cimbra", thickness=1.0, material=mat)
    stages = (_make_stage("A", ("cimbra",)),)
    elem_owner, node_owner = compute_stage_ownership(
        stages, [rock, cim], fem,
    )
    # Only cimbra is stage-bound.
    assert elem_owner == {id(cim): 0}
    # Nodes 5 and 6 are exclusive to cimbra → stage-bound.
    # Nodes 2 and 3 are shared with rock (global) → stay global.
    assert node_owner == {5: 0, 6: 0}


def test_compute_ownership_shared_nodes_stay_global() -> None:
    """Nodes referenced by ANY global element stay global, even if
    they're ALSO referenced by stage-bound elements."""
    fem = _make_two_pg_fem()
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    rock = ops.element.FourNodeQuad(pg="rock", thickness=1.0, material=mat)
    cim  = ops.element.FourNodeQuad(pg="cimbra", thickness=1.0, material=mat)
    stages = (_make_stage("A", ("cimbra",)),)
    _elem_owner, node_owner = compute_stage_ownership(
        stages, [rock, cim], fem,
    )
    # Nodes 2 and 3 are on the interface (referenced by both rock
    # and cimbra) → global.
    assert 2 not in node_owner
    assert 3 not in node_owner


def test_compute_ownership_double_activation_errors() -> None:
    """A PG activated by more than one stage → BridgeError."""
    fem = _make_two_pg_fem()
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    rock = ops.element.FourNodeQuad(pg="rock", thickness=1.0, material=mat)
    cim  = ops.element.FourNodeQuad(pg="cimbra", thickness=1.0, material=mat)
    stages = (
        _make_stage("A", ("cimbra",)),
        _make_stage("B", ("cimbra",)),  # duplicate activation
    )
    with pytest.raises(BridgeError, match="activated by another stage"):
        compute_stage_ownership(stages, [rock, cim], fem)


def test_compute_ownership_lowest_stage_wins_for_shared_stage_bound_nodes() -> None:
    """If two stages each activate distinct PGs but share a node,
    the lowest-index stage owns the node."""
    # 3 quads: A (left), B (middle), C (right).
    # A shares nodes with B; B shares nodes with C; A and C don't touch.
    fem = FEMStub(
        nodes=_NodesStub(
            ids=[1, 2, 3, 4, 5, 6, 7, 8],
            coords=[(0,0,0),(1,0,0),(1,1,0),(0,1,0),
                    (2,0,0),(2,1,0),(3,0,0),(3,1,0)],
            node_pgs={},
        ),
        elements=_ElementsStub(elem_pgs={
            "A": _ElementGroupView(ids=(1,), connectivity=((1,2,3,4),)),
            "B": _ElementGroupView(ids=(2,), connectivity=((2,5,6,3),)),
            "C": _ElementGroupView(ids=(3,), connectivity=((5,7,8,6),)),
        }),
    )
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    a = ops.element.FourNodeQuad(pg="A", thickness=1.0, material=mat)
    b = ops.element.FourNodeQuad(pg="B", thickness=1.0, material=mat)
    c = ops.element.FourNodeQuad(pg="C", thickness=1.0, material=mat)
    stages = (
        _make_stage("s0", ("A",)),    # stage 0 owns A
        _make_stage("s1", ("B", "C")),  # stage 1 owns B and C
    )
    elem_owner, node_owner = compute_stage_ownership(stages, [a, b, c], fem)
    assert elem_owner == {id(a): 0, id(b): 1, id(c): 1}
    # Nodes 1, 4 — only in A → stage 0.
    # Nodes 2, 3 — in A and B → lowest stage = 0.
    # Nodes 5, 6 — in B and C → lowest stage = 1.
    # Nodes 7, 8 — only in C → stage 1.
    assert node_owner[1] == 0
    assert node_owner[4] == 0
    assert node_owner[2] == 0
    assert node_owner[3] == 0
    assert node_owner[5] == 1
    assert node_owner[6] == 1
    assert node_owner[7] == 1
    assert node_owner[8] == 1


# ---------------------------------------------------------------------------
# 5. Bridge emit ordering (full apeSees flow)
# ---------------------------------------------------------------------------


def _build_two_pg_two_stage(ops_factory, tmp_path):
    """Helper: build a 2-PG model with rock global + cimbra activated
    in stage 2.  Returns (ops, tcl_path)."""
    fem = _make_two_pg_fem()
    ops = ops_factory(fem)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.FourNodeQuad(pg="rock", thickness=1.0, material=mat)
    ops.element.FourNodeQuad(pg="cimbra", thickness=1.0, material=mat)
    ops.fix(pg="Left", dofs=(1, 1))  # nodes 1, 4 — global

    with ops.stage(name="rock_only") as s:
        s.analysis(**_full_chain(ops))
        s.run(n_increments=2)

    with ops.stage(name="install_cimbra") as s:
        s.activate(pgs=["cimbra"])
        s.analysis(**_full_chain(ops))
        s.run(n_increments=3)

    deck_path = tmp_path / "deck.tcl"
    ops.tcl(str(deck_path))
    return ops, deck_path


def test_global_nodes_emit_before_stages(tmp_path) -> None:
    ops, deck_path = _build_two_pg_two_stage(
        lambda fem: apeSees(fem, default_orientation=None),
        tmp_path,
    )
    text = deck_path.read_text()
    lines = text.splitlines()
    # Find the line numbers of node 1 and node 5 emit, and the first
    # stage banner.
    node_1_idx = next(
        i for i, ln in enumerate(lines) if ln.startswith("node 1 ")
    )
    node_5_idx = next(
        i for i, ln in enumerate(lines) if ln.startswith("node 5 ")
    )
    first_stage_idx = next(
        i for i, ln in enumerate(lines) if ln.startswith("# === Stage:")
    )
    # Node 1 (rock — global) emits BEFORE the first stage.
    assert node_1_idx < first_stage_idx
    # Node 5 (cimbra — stage-bound to stage 2) emits AFTER the first
    # stage banner.
    assert node_5_idx > first_stage_idx


def test_stage_bound_element_emits_inside_stage(tmp_path) -> None:
    ops, deck_path = _build_two_pg_two_stage(
        lambda fem: apeSees(fem, default_orientation=None),
        tmp_path,
    )
    text = deck_path.read_text()
    lines = text.splitlines()
    quad_lines = [
        (i, ln) for i, ln in enumerate(lines)
        if ln.startswith("element quad ")
    ]
    assert len(quad_lines) == 2
    # First quad (rock) is global → emits before any stage banner.
    rock_idx, rock_ln = quad_lines[0]
    install_idx = next(
        i for i, ln in enumerate(lines)
        if ln.startswith("# === Stage: install_cimbra")
    )
    assert rock_idx < install_idx
    # Second quad (cimbra) emits INSIDE the install_cimbra stage.
    cim_idx, cim_ln = quad_lines[1]
    assert cim_idx > install_idx
    # And before the second stage's wipeAnalysis (which closes it).
    install_close_idx = next(
        i for i, ln in enumerate(lines)
        if i > install_idx and ln == "wipeAnalysis"
    )
    assert cim_idx < install_close_idx


def test_domain_change_emits_after_activation(tmp_path) -> None:
    ops, deck_path = _build_two_pg_two_stage(
        lambda fem: apeSees(fem, default_orientation=None),
        tmp_path,
    )
    text = deck_path.read_text()
    lines = text.splitlines()
    install_idx = next(
        i for i, ln in enumerate(lines)
        if ln.startswith("# === Stage: install_cimbra")
    )
    domain_idx = next(
        i for i, ln in enumerate(lines)
        if i > install_idx and ln == "domainChange"
    )
    cim_quad_idx = next(
        i for i, ln in enumerate(lines)
        if i > install_idx and ln.startswith("element quad ")
    )
    # domainChange comes AFTER the cimbra element.
    assert domain_idx > cim_quad_idx
    # And BEFORE the analysis chain rebinds (test/algorithm/...).
    chain_idx = next(
        i for i, ln in enumerate(lines)
        if i > install_idx and ln.startswith("test ")
    )
    assert domain_idx < chain_idx


def test_no_domain_change_when_no_activation(tmp_path) -> None:
    """First stage doesn't activate anything → no domainChange."""
    ops, deck_path = _build_two_pg_two_stage(
        lambda fem: apeSees(fem, default_orientation=None),
        tmp_path,
    )
    text = deck_path.read_text()
    lines = text.splitlines()
    rock_stage_idx = next(
        i for i, ln in enumerate(lines)
        if ln.startswith("# === Stage: rock_only")
    )
    install_stage_idx = next(
        i for i, ln in enumerate(lines)
        if ln.startswith("# === Stage: install_cimbra")
    )
    # No domainChange between the two stage banners.
    between = lines[rock_stage_idx:install_stage_idx]
    assert "domainChange" not in between
    # But exactly one domainChange after the second stage banner.
    after_second = lines[install_stage_idx:]
    assert after_second.count("domainChange") == 1


# ---------------------------------------------------------------------------
# 6. Double-activation rejected at build time
# ---------------------------------------------------------------------------


def test_double_pg_activation_raises_on_build() -> None:
    fem = _make_two_pg_fem()
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.FourNodeQuad(pg="rock", thickness=1.0, material=mat)
    ops.element.FourNodeQuad(pg="cimbra", thickness=1.0, material=mat)

    with ops.stage(name="A") as s:
        s.activate(pgs=["cimbra"])
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    with ops.stage(name="B") as s:
        s.activate(pgs=["cimbra"])  # duplicate!
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)

    bm = ops.build()
    from apeGmsh.opensees.emitter.tcl import TclEmitter
    with pytest.raises(BridgeError, match="activated by another stage"):
        bm.emit(TclEmitter())
