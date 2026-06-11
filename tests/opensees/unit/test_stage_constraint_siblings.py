"""Smoke tests for the sibling stage-bound constraint claim methods.

One test per kind, all using the same claim-by-name contract proven
in :mod:`test_stage_embedded_claim`.  Coverage:

* ``s.equal_dof``           — ``NodePairRecord(kind="equal_dof")``
* ``s.rigid_link``          — ``NodePairRecord(kind="rigid_beam")``
* ``s.rigid_diaphragm``     — ``NodeGroupRecord(kind="rigid_diaphragm")``
* ``s.kinematic_coupling``  — ``NodeGroupRecord(kind="kinematic_coupling")``
* ``s.tie``                 — ``InterpolationRecord(kind="tie")``
* ``s.distributing``        — ``InterpolationRecord(kind="distributing")``

Each test verifies the record lands in the stage's pool, the bridge's
claim set picks it up, and the emitted deck routes it inside the
stage's block rather than the global pre-stage pass.
"""
from __future__ import annotations

import numpy as np

from apeGmsh._kernel.records._constraints import (
    InterpolationRecord, NodeGroupRecord, NodePairRecord,
)
from apeGmsh.opensees.apesees import apeSees

from tests.opensees.fixtures.fem_stub import (
    FEMStub,
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
)


def _make_two_node_fem() -> FEMStub:
    """Minimal node-couple fem: two nodes, no elements needed for
    node-side constraint emission."""
    return FEMStub(
        nodes=_NodesStub(
            ids=[1, 2, 3, 4, 5],
            coords=[
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.5, 0.5, 0.0),
            ],
            node_pgs={"Left": [1, 4]},
        ),
        elements=_ElementsStub(
            elem_pgs={
                "Rock": _ElementGroupView(
                    ids=(1,), connectivity=((1, 2, 3, 4),),
                ),
            },
        ),
    )


def _build_ops(fem):
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.FourNodeQuad(pg="Rock", thickness=1.0, material=mat)
    return ops


def _build_ops_3d_noelem(fem):
    """A 3D ndf=6 envelope with NO declared element — every node falls to the
    model ndf (6).  Used by the rigid-link / rigid-diaphragm routing tests,
    whose constraints reference rotational DOFs / a diaphragm master that the
    2D ndf=2 quad model would make invalid under ADR 0049 G2."""
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=3, ndf=6)
    return ops


def _full_chain(ops):
    return {
        "test":        ops.test.NormDispIncr(tol=1e-4, max_iter=50),
        "algorithm":   ops.algorithm.Newton(),
        "integrator":  ops.integrator.LoadControl(dlam=0.1),
        "constraints": ops.constraints.Transformation(),
        "numberer":    ops.numberer.RCM(),
        "system":      ops.system.UmfPack(),
        "analysis":    ops.analysis.Static(),
    }


def _assert_routed_inside_stage(deck_text: str, marker: str,
                                stage_name: str) -> None:
    """Assert ``marker`` appears in the deck AND lands after the stage
    open comment (so it's inside the stage block, not in the global
    pre-stage pass)."""
    assert marker in deck_text, f"{marker!r} not in deck"
    stage_open_idx = deck_text.index(f"# === Stage: {stage_name} ===")
    marker_idx = deck_text.index(marker)
    assert marker_idx > stage_open_idx, (
        f"{marker!r} emitted in the global pre-stage block; "
        "claimed records must emit inside the owning stage's block"
    )
    # And exactly once (no double-emission via global + stage).
    assert deck_text.count(marker) == 1


def test_equal_dof_claim_routes_to_stage(tmp_path) -> None:
    fem = _make_two_node_fem()
    rec = NodePairRecord(
        kind="equal_dof", name="my_eqdof",
        master_node=1, slave_node=2, dofs=[1, 2],
    )
    fem.add_node_constraints([rec])

    ops = _build_ops(fem)
    with ops.stage(name="bind") as s:
        claimed = s.equal_dof(name="my_eqdof")
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1, dt=1.0)

    assert claimed == (rec,)
    out = tmp_path / "deck.tcl"
    ops.tcl(str(out))
    _assert_routed_inside_stage(
        out.read_text(encoding="utf-8"),
        "equalDOF 1 2 1 2",
        "bind",
    )


def test_rigid_link_claim_routes_to_stage(tmp_path) -> None:
    fem = _make_two_node_fem()
    rec = NodePairRecord(
        kind="rigid_beam", name="my_rb",
        master_node=1, slave_node=2, dofs=[1, 2, 3, 4, 5, 6],
        offset=np.array([1.0, 0.0, 0.0]),
    )
    fem.add_node_constraints([rec])

    # A rigid_beam referencing DOFs 1-6 needs ndf=6 endpoints — model it in a
    # 3D envelope with no declared element so the nodes take the model ndf
    # (the 2D quad's ndf=2 would make DOFs 3-6 invalid, ADR 0049 G2).
    ops = _build_ops_3d_noelem(fem)
    with ops.stage(name="bind") as s:
        claimed = s.rigid_link(name="my_rb")
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1, dt=1.0)

    assert claimed == (rec,)
    out = tmp_path / "deck.tcl"
    ops.tcl(str(out))
    _assert_routed_inside_stage(
        out.read_text(encoding="utf-8"),
        "rigidLink beam 1 2",
        "bind",
    )


def test_rigid_diaphragm_claim_routes_to_stage(tmp_path) -> None:
    fem = _make_two_node_fem()
    rec = NodeGroupRecord(
        kind="rigid_diaphragm", name="my_dia",
        master_node=1, slave_nodes=[2, 3], dofs=[1, 2, 6],
        plane_normal=np.array([0.0, 0.0, 1.0]),
    )
    fem.add_node_constraints([rec])

    # A rigidDiaphragm retained node must carry ndf=6 (3D) / 3 (2D) exactly —
    # model it in a 3D envelope with no declared element so the master takes
    # the model ndf (the 2D quad's ndf=2 master would be invalid, ADR 0049 G2).
    ops = _build_ops_3d_noelem(fem)
    with ops.stage(name="bind") as s:
        claimed = s.rigid_diaphragm(name="my_dia")
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1, dt=1.0)

    assert claimed == (rec,)
    out = tmp_path / "deck.tcl"
    ops.tcl(str(out))
    _assert_routed_inside_stage(
        out.read_text(encoding="utf-8"),
        "rigidDiaphragm 3 1 2 3",
        "bind",
    )


def test_kinematic_coupling_claim_routes_to_stage(tmp_path) -> None:
    fem = _make_two_node_fem()
    rec = NodeGroupRecord(
        kind="kinematic_coupling", name="my_kc",
        master_node=1, slave_nodes=[2, 3], dofs=[1, 2],
    )
    fem.add_node_constraints([rec])

    ops = _build_ops(fem)
    with ops.stage(name="bind") as s:
        claimed = s.kinematic_coupling(name="my_kc")
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1, dt=1.0)

    assert claimed == (rec,)
    out = tmp_path / "deck.tcl"
    deck = (ops.tcl(str(out)), out.read_text(encoding="utf-8"))[1]
    # kinematic_coupling is now emitted as a single fork
    # ``element LadrunoKinematicCoupling $tag $ref $N $s1 $s2 -dof ...``
    # (RBE2), routed inside the claiming stage.
    _assert_routed_inside_stage(deck, "element LadrunoKinematicCoupling", "bind")
    # refNode 1, N=2, slaves 2 & 3, restricted to dofs 1 2.
    assert "1 2 2 3 -dof 1 2" in deck


def test_tie_claim_routes_to_stage(tmp_path) -> None:
    fem = _make_two_node_fem()
    rec = InterpolationRecord(
        kind="tie", name="my_tie",
        slave_node=5, master_nodes=[1, 2, 3], dofs=[1, 2, 3],
    )
    fem.add_surface_constraints([rec])

    ops = _build_ops(fem)
    with ops.stage(name="bind") as s:
        claimed = s.tie(name="my_tie")
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1, dt=1.0)

    assert claimed == (rec,)
    out = tmp_path / "deck.tcl"
    ops.tcl(str(out))
    text = out.read_text(encoding="utf-8")
    # Tie is emitted as ASDEmbeddedNodeElement (same Tcl shape).
    embed_lines = [
        ln for ln in text.splitlines()
        if "ASDEmbeddedNodeElement" in ln
    ]
    assert len(embed_lines) == 1
    stage_open_idx = text.index("# === Stage: bind ===")
    embed_idx = text.index("ASDEmbeddedNodeElement")
    assert embed_idx > stage_open_idx


def test_distributing_claim_routes_to_stage(tmp_path) -> None:
    fem = _make_two_node_fem()
    # RBE3: slave_node = reference (dependent) node R; master_nodes =
    # independents. Emits the fork element, not ASDEmbeddedNodeElement.
    rec = InterpolationRecord(
        kind="distributing", name="my_dist",
        slave_node=5, master_nodes=[1, 2, 4],
    )
    fem.add_surface_constraints([rec])

    ops = _build_ops(fem)
    with ops.stage(name="bind") as s:
        claimed = s.distributing(name="my_dist")
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1, dt=1.0)

    assert claimed == (rec,)
    out = tmp_path / "deck.tcl"
    ops.tcl(str(out))
    text = out.read_text(encoding="utf-8")
    dist_lines = [
        ln for ln in text.splitlines()
        if "LadrunoDistributingCoupling" in ln
    ]
    assert len(dist_lines) == 1
    # R=5, N=3, independents 1 2 4 (uniform ⇒ no -w).
    assert "5 3 1 2 4" in dist_lines[0]
    stage_open_idx = text.index("# === Stage: bind ===")
    dist_idx = text.index("LadrunoDistributingCoupling")
    assert dist_idx > stage_open_idx
