"""Shell-on-solid mixed-ndf modelling — guard, equilibrium, round-trip.

Three behaviours that together close the "2-D shell wall on a 3-D solid
footing" use case (the E7 docs example):

1.  **Fail-loud guard** — the broken idiom (a shell surface *fragmented
    onto* a solid volume so they share interface nodes, then
    ``g.node_ndf.set(shell, ndf=6)``) raises :class:`BridgeError` at
    build.  Sharing a node between an ``ndf=6`` shell element and an
    ``ndf=3`` solid element is unassemblable in OpenSees
    (``FE_Element::setID`` truncates the element's equation map →
    silent equilibrium loss).  Before the guard this produced a
    plausible deflection while ~half the applied load vanished.

2.  **Correct idiom + equilibrium** — the *separate coincident nodes*
    idiom (solid ``ndf=3`` node + shell ``ndf=6`` node at the same
    location, tied by ``equalDOF`` on the translational DOFs, with the
    shell-edge rotations clamped for the line hinge) transmits the full
    load: ``Σ reactions == Σ applied`` to numerical precision.

3.  **node_ndf round-trips through Results.from_native** — a real
    ``g.node_ndf`` model captured with ``ops.domain_capture`` writes the
    ndf envelope into the Composed file's ``/model/meta`` so
    ``OpenSeesModel.from_h5(path, fem_root="/model")`` recovers
    ``ndf=6`` (it read ``ndf=0`` and raised before the
    ``DomainCapture`` bridge-forwarding fix).

See ADR 0032 / 0033 (explicit-only per-node ndf) and the partitioned
companion ``test_emit_partitioned_mixed_ndf_shell_on_solid.py``.
"""
from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees._internal.build import BridgeError


# ---------------------------------------------------------------------------
# 1. Fail-loud guard — fragmented (shared-node) shell-on-solid.
# ---------------------------------------------------------------------------


def _build_fragmented_shell_on_solid(g):
    """Build the BROKEN idiom: a wall surface fragmented INTO a footing
    box so the wall-base line shares nodes with the footing tets, then
    declare the wall ``ndf=6`` and the rest ``ndf=3``.
    """
    import gmsh

    b, H = 1.0, 2.0
    G = g.model.geometry
    foot = G.add_box(-0.5, -0.5, -0.5, 2.0, 1.0, 0.5)
    p1 = G.add_point(0, 0, 0); p2 = G.add_point(b, 0, 0)
    p3 = G.add_point(b, 0, H); p4 = G.add_point(0, 0, H)
    l1 = G.add_line(p1, p2); l2 = G.add_line(p2, p3)
    l3 = G.add_line(p3, p4); l4 = G.add_line(p4, p1)
    loop = G.add_curve_loop([l1, l2, l3, l4])
    wall = G.add_plane_surface([loop])
    g.model.sync()
    # Fragment ties the wall base into the footing top face -> SHARED nodes.
    g.model.boolean.fragment([(3, foot), (2, wall)], [])
    g.model.sync()

    vols = [tg for (d, tg) in gmsh.model.getEntities(3)]
    surfs = [tg for (d, tg) in gmsh.model.getEntities(2)]
    g.physical.add(3, vols, name="Footing")
    wall_s = [
        s for s in surfs
        if abs(gmsh.model.occ.getCenterOfMass(2, s)[1]) < 1e-6
        and gmsh.model.occ.getCenterOfMass(2, s)[2] > 1e-6
    ]
    g.physical.add(2, wall_s, name="Wall")
    g.model.select(dim=2).on_plane((0, 0, -0.5), (0, 0, 1), tol=1e-3).to_physical("FootBase")
    # ADR 0048: ndf is inferred — the shared-node conflict (shell 6 vs
    # solid 3) is detected from the declared elements, no g.node_ndf.
    g.mesh.structured.set_recombine("Wall", dim=2)
    g.mesh.sizing.set_global_size(0.25)
    g.mesh.generation.generate(3)
    g.mesh.structured.recombine()
    g.mesh.partitioning.renumber(base=1)
    return g.mesh.queries.get_fem_data()


def test_fragmented_shell_on_solid_fails_loud(g):
    """A shell fragmented onto a solid (shared interface nodes) raises
    ``BridgeError`` at build — naming the offending node, both element
    types, and the fix — instead of silently losing half the load.
    """
    fem = _build_fragmented_shell_on_solid(g)
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=6)
    ops.element.FourNodeTetrahedron(
        pg="Footing", material=ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2),
    )
    ops.element.ShellMITC4(
        pg="Wall",
        section=ops.section.ElasticMembranePlateSection(E=30e9, nu=0.0, h=0.1),
    )
    ops.fix(pg="FootBase", dofs=(1, 1, 1, 1, 1, 1))

    with pytest.raises(BridgeError) as exc_info:
        ops.build().emit(_NullEmitter())
    msg = str(exc_info.value)
    assert "FourNodeTetrahedron" in msg and "ShellMITC4" in msg, (
        f"BridgeError must name both conflicting element types: {msg!r}"
    )
    assert "[3]" in msg and "[6]" in msg, (
        f"BridgeError must name the disjoint ndf sets: {msg!r}"
    )
    assert "equal_dof" in msg or "separate" in msg.lower(), (
        f"BridgeError must point at the separate-node fix: {msg!r}"
    )


class _NullEmitter:
    """Emitter that no-ops every Protocol call — lets us reach the build
    guard (which runs before any element is emitted) without needing a
    live OpenSees domain.  The guard raises before any of these fire.
    """

    def __getattr__(self, _name):
        def _noop(*_args, **_kwargs):
            return None
        return _noop


from apeGmsh._kernel.records._constraints import NodePairRecord
from apeGmsh._kernel.records._kinds import ConstraintKind

from tests.opensees.fixtures.fem_stub import (
    FEMStub,
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
)


# ---------------------------------------------------------------------------
# 1b. Guard coverage — every shell family is classified ndf={6}, and the
#     guard fires for the registry-gap shell (ASDShellT3) too.
# ---------------------------------------------------------------------------


def test_element_class_ndf_ok_classifies_all_shell_families():
    """All five shells resolve to ndf={6}, solids/quads to {3}/{2}, and
    multi-ndf beams to a superset — so the guard's disjoint-family test
    is correct for every shell-vs-solid pair, including ASDShellT3 (which
    has no _ELEM_REGISTRY entry and is covered via _EXTRA_CLASS_NDF_OK).
    """
    from apeGmsh.opensees._element_capabilities import element_class_ndf_ok

    for shell in ("ShellMITC3", "ShellMITC4", "ShellDKGQ",
                  "ASDShellQ4", "ASDShellT3"):
        assert element_class_ndf_ok(shell) == frozenset({6}), shell
    for solid in ("FourNodeTetrahedron", "stdBrick", "TenNodeTetrahedron"):
        assert element_class_ndf_ok(solid) == frozenset({3}), solid
    assert element_class_ndf_ok("FourNodeQuad") == frozenset({2})
    # Multi-ndf beam — intersects both solids and shells (never flagged).
    assert element_class_ndf_ok("elasticBeamColumn") == frozenset({3, 6})
    # Unknown class -> None (conservative skip).
    assert element_class_ndf_ok("NotARealElement") is None


def test_asdshellt3_on_solid_fails_loud():
    """A registry-gap shell (ASDShellT3, ndf=6) sharing a node with a
    solid tet (ndf=3) trips the guard — the coverage hole the
    _EXTRA_CLASS_NDF_OK fallback closes.  Build-time only (the guard
    raises before any element is emitted), so no openseespy needed.
    """
    nodes = _NodesStub(
        ids=[1, 2, 3, 4, 5, 6],
        coords=[
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),                      # 1-4 tet
            (1.0, 1.0, 0.0), (0.0, 1.0, 1.0),     # 5,6 extra shell corners
        ],
        node_pgs={"Solid": [1, 2, 3, 4], "Shell": [4, 5, 6]},
    )
    elements = _ElementsStub(elem_pgs={
        "Solid": _ElementGroupView(ids=(1,), connectivity=((1, 2, 3, 4),)),
        # ASDShellT3 (3-node shell) shares node 4 with the tet.
        "Shell": _ElementGroupView(ids=(2,), connectivity=((4, 5, 6),)),
    })
    fem = FEMStub(nodes=nodes, elements=elements)

    ops = apeSees(fem)
    ops.model(ndm=3, ndf=6)
    ops.element.FourNodeTetrahedron(
        pg="Solid", material=ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2),
    )
    ops.element.ASDShellT3(
        pg="Shell",
        section=ops.section.ElasticMembranePlateSection(E=30e9, nu=0.0, h=0.1),
    )
    with pytest.raises(BridgeError) as exc_info:
        ops.build().emit(_NullEmitter())
    msg = str(exc_info.value)
    assert "ASDShellT3" in msg and "FourNodeTetrahedron" in msg, msg


# ---------------------------------------------------------------------------
# 1c. domain_capture sidecar gate — only PARTITIONED staged builds keep
#     the pre-Composed behaviour (mirrors the ops.h5 guard; ADR 0055).
# ---------------------------------------------------------------------------


def test_domain_capture_forwards_sidecar_for_initial_stress_builds(tmp_path):
    """domain_capture forwards the bridge whenever ops.h5() (the sidecar
    writer) supports the build.  ADR 0055 Phase 1 made global
    initial-stress builds archivable, so they forward the bridge just
    like a plain build — only PARTITIONED staged builds gate it off
    (see test_h5_staged_domain_capture.py).
    """
    from apeGmsh.opensees._internal.build import InitialStressRecord
    from apeGmsh.results.capture.spec import DomainCaptureSpec

    nodes = _NodesStub(
        ids=[1, 2, 3, 4],
        coords=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],
        node_pgs={"Solid": [1, 2, 3, 4]},
    )
    elements = _ElementsStub(elem_pgs={
        "Solid": _ElementGroupView(ids=(1,), connectivity=((1, 2, 3, 4),)),
    })
    fem = FEMStub(nodes=nodes, elements=elements)
    fem.snapshot_id = "stub"  # spec resolve reads fem.snapshot_id
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)

    spec = DomainCaptureSpec(opensees=ops)
    spec.nodes(ids=[1], components=["displacement_x"], name="probe")

    # Plain build: sidecar forwarded so node_ndf can round-trip.
    cap_plain = ops.domain_capture(spec, path=str(tmp_path / "plain.h5"))
    assert cap_plain._bridge is ops

    # Initial-stress build: ops.h5 archives it (ADR 0055 Phase 1), so
    # the sidecar is forwarded too — the gate was lifted.
    ops._initial_stress_records.append(InitialStressRecord(
        name="insitu", pg="Solid", elements=None,
        sigma_xx=0.0, sigma_yy=0.0, sigma_zz=-1.0,
        ramp_steps=1, lambda_install=1.0,
    ))
    cap_is = ops.domain_capture(spec, path=str(tmp_path / "is.h5"))
    assert cap_is._bridge is ops


# ---------------------------------------------------------------------------
# 2. Correct idiom — separate coincident nodes + equalDOF + rotation clamp
#    transmits the full load (Σ reactions == Σ applied).
# ---------------------------------------------------------------------------


def _make_shell_on_solid_correct(*, b=1.0, H=2.0, depth=0.5, nz=4):
    """Build the CORRECT shell-on-solid idiom as a runnable FEM stub.

    A single solid brick (footing, ``ndf=3``) carries a vertical strip
    of ``nz`` shell quads (wall, ``ndf=6``).  The two wall-base nodes are
    SEPARATE tags coincident with the footing's top-front nodes (5, 6),
    tied by ``equalDOF(footing_node, wall_node, 1, 2, 3)``.  The wall-base
    rotations are clamped by the caller (line-hinge).  Returns
    ``(fem, wall_base, wall_top)``.
    """
    fb = [
        (0.0, 0.0, -depth), (b, 0.0, -depth), (b, depth, -depth), (0.0, depth, -depth),
        (0.0, 0.0, 0.0), (b, 0.0, 0.0), (b, depth, 0.0), (0.0, depth, 0.0),
    ]
    ids = list(range(1, 9))
    coords = list(fb)
    nid = 9
    grid: dict[tuple[int, int], int] = {}
    wall_ids: list[int] = []
    for r, z in enumerate(np.linspace(0, H, nz + 1)):
        for c, x in enumerate((0.0, b)):
            ids.append(nid)
            coords.append((x, 0.0, float(z)))
            grid[(c, r)] = nid
            wall_ids.append(nid)
            nid += 1
    wall_base = [grid[(0, 0)], grid[(1, 0)]]
    wall_top = [grid[(0, nz)], grid[(1, nz)]]

    nodes = _NodesStub(
        ids=ids, coords=coords,
        node_pgs={
            "Footing": list(range(1, 9)),
            "Wall": wall_ids,
            "FootBase": [1, 2, 3, 4],
            "Top": wall_top,
            "WallBase": wall_base,
        },
    )
    ndf_map = {n: 3 for n in range(1, 9)}
    ndf_map.update({n: 6 for n in wall_ids})
    nodes.set_per_node_ndf(ndf_map)

    shell_conn = []
    for r in range(nz):
        shell_conn.append(
            (grid[(0, r)], grid[(1, r)], grid[(1, r + 1)], grid[(0, r + 1)])
        )
    elements = _ElementsStub(elem_pgs={
        "Footing": _ElementGroupView(ids=(1,), connectivity=((1, 2, 3, 4, 5, 6, 7, 8),)),
        "Wall": _ElementGroupView(
            ids=tuple(range(100, 100 + nz)), connectivity=tuple(shell_conn),
        ),
    })
    fem = FEMStub(nodes=nodes, elements=elements)
    # equalDOF: footing-top-front (master, ndf3) <- wall-base (slave, ndf6).
    fem.add_node_constraints([
        NodePairRecord(
            kind=ConstraintKind.EQUAL_DOF, master_node=5,
            slave_node=wall_base[0], dofs=[1, 2, 3], name="iface_0",
        ),
        NodePairRecord(
            kind=ConstraintKind.EQUAL_DOF, master_node=6,
            slave_node=wall_base[1], dofs=[1, 2, 3], name="iface_1",
        ),
    ])
    return fem, wall_base, wall_top


@pytest.mark.live
def test_shell_on_solid_correct_idiom_satisfies_equilibrium():
    """The separate-coincident-node idiom transmits the full load:
    ``Σ y-reactions == applied`` to numerical precision, and the wall
    deflects out-of-plane like a clamped cantilever (non-trivial tip).

    Contrast with the fragmented (shared-node) model, which deflects
    plausibly but reacts only ~half the applied load.
    """
    pytest.importorskip("openseespy.opensees")
    from apeGmsh.opensees.emitter.live import LiveOpsEmitter

    P = 1800.0
    fem, wall_base, wall_top = _make_shell_on_solid_correct()

    ops = apeSees(cast("object", fem))
    ops.model(ndm=3, ndf=6)
    ops.element.stdBrick(
        pg="Footing", material=ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2),
    )
    ops.element.ShellMITC4(
        pg="Wall",
        section=ops.section.ElasticMembranePlateSection(E=30e9, nu=0.0, h=0.1),
    )
    ops.fix(pg="FootBase", dofs=(1, 1, 1))
    ops.fix(nodes=wall_base, dofs=(0, 0, 0, 1, 1, 1))  # clamp shell-base rotations

    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts) as pat:
        for n in wall_top:
            pat.load(node=n, forces=(0.0, P / len(wall_top), 0.0, 0.0, 0.0, 0.0))

    ops.constraints.Transformation()
    ops.numberer.RCM()
    ops.system.UmfPack()
    ops.test.NormDispIncr(tol=1e-9, max_iter=50)
    ops.algorithm.Newton()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()

    # wipe=True clears any domain a prior test left behind before build.
    emitter = LiveOpsEmitter(wipe=True)
    ops.build().emit(emitter)
    ret = emitter.analyze(steps=1)
    assert ret == 0, "openseespy.analyze returned non-zero"

    opspy = emitter.ops
    opspy.reactions()
    all_ids = [int(n) for n in fem.nodes.ids]
    total_ry = sum(opspy.nodeReaction(n, 2) for n in all_ids)
    tip = float(np.mean([opspy.nodeDisp(n, 2) for n in wall_top]))
    opspy.wipe()

    # Global y-equilibrium: reactions balance the applied load exactly.
    assert total_ry == pytest.approx(-P, abs=1e-3), (
        f"equilibrium violated: Σ y-reactions={total_ry:.4f} != applied "
        f"{-P} — the shell-on-solid interface dropped load."
    )
    # The wall actually deformed (not a rigid-body / mechanism solve).
    assert abs(tip) > 1e-6, (
        f"wall tip displacement is ~0 ({tip}); load did not deform the wall."
    )


# ---------------------------------------------------------------------------
# 3. node_ndf round-trips through ops.domain_capture -> Results.from_native.
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_node_ndf_roundtrips_through_domain_capture(g, tmp_path):
    """A real ``g.node_ndf`` model captured with ``ops.domain_capture``
    persists its ndf envelope into the Composed file's ``/model/meta`` so
    ``OpenSeesModel.from_h5(path, fem_root="/model")`` recovers ``ndf=6``.

    Before the ``DomainCapture(bridge=self)`` forwarding fix, the capture
    file carried only ``/model`` + ``/stages`` (no ``/opensees`` zone),
    the broker ``/model/meta`` had no bridge ``ndf``, and
    ``OpenSeesModel.from_h5`` read ``ndf=0`` —
    ``validate_envelope_covers_broker_ndf`` then rejected the model with
    ``BridgeError: OpenSeesModel(ndf=0) cannot host node ... ndf=6``.
    """
    pytest.importorskip("openseespy.opensees")
    import gmsh

    from apeGmsh.opensees import OpenSeesModel
    from apeGmsh.results import Results
    from apeGmsh.results.capture.spec import DomainCaptureSpec

    b, H = 1.0, 1.0
    G = g.model.geometry
    G.add_box(0, 0, -0.5, b, b, 0.5)
    # Detached shell plate ABOVE the footing — separate nodes, its own
    # supports (keeps the model trivially analyzable; the ndf round-trip
    # is independent of the interface idiom, which §2 covers).
    z0 = 0.5
    p1 = G.add_point(0, 0, z0); p2 = G.add_point(b, 0, z0)
    p3 = G.add_point(b, 0, z0 + H); p4 = G.add_point(0, 0, z0 + H)
    l1 = G.add_line(p1, p2); l2 = G.add_line(p2, p3)
    l3 = G.add_line(p3, p4); l4 = G.add_line(p4, p1)
    lp = G.add_curve_loop([l1, l2, l3, l4]); wall = G.add_plane_surface([lp])
    g.model.sync()

    vols = [tg for (d, tg) in gmsh.model.getEntities(3)]
    g.physical.add(3, vols, name="Footing")
    g.physical.add(2, [wall], name="Wall")
    g.model.select(dim=2).on_plane((0, 0, -0.5), (0, 0, 1), tol=1e-4).to_physical("FootBase")
    g.physical.add(1, [l1], name="WallBase")
    g.physical.add(1, [l3], name="WallTop")
    # ADR 0048: ndf inferred from the elements (tet→3, shell→6) on
    # separate nodes; the envelope is set via ops.model(ndf=6) below.
    g.mesh.structured.set_recombine("Wall", dim=2)
    g.mesh.sizing.set_global_size(0.34)
    g.mesh.generation.generate(3)
    g.mesh.structured.recombine()
    g.mesh.partitioning.renumber(base=1)
    fem = g.mesh.queries.get_fem_data()

    ops = apeSees(fem)
    ops.model(ndm=3, ndf=6)
    ops.element.FourNodeTetrahedron(
        pg="Footing", material=ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2),
    )
    ops.element.ShellMITC4(
        pg="Wall",
        section=ops.section.ElasticMembranePlateSection(E=30e9, nu=0.0, h=0.1),
    )
    ops.fix(pg="FootBase", dofs=(1, 1, 1))
    ops.fix(pg="WallBase", dofs=(1, 1, 1, 1, 1, 1))
    top = [int(n) for n in fem.nodes.select(pg="WallTop").ids]
    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts) as pat:
        for n in top:
            pat.load(node=n, forces=(0, 50.0, 0, 0, 0, 0))
    ops.constraints.Plain()
    ops.numberer.RCM()
    ops.system.UmfPack()
    ops.test.NormDispIncr(tol=1e-8, max_iter=50)
    ops.algorithm.Newton()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()

    path = str(tmp_path / "node_ndf_roundtrip.h5")
    spec = DomainCaptureSpec(opensees=ops)
    spec.nodes(pg="WallTop", components=["displacement_y"], name="tip")
    with ops.domain_capture(spec, path=path) as cap:
        cap.begin_stage("lat", kind="static")
        ops.analyze(steps=1)
        cap.step(t=1.0)
        cap.end_stage()

    # BUG 2: the ndf envelope must survive the Composed-file round-trip.
    om = OpenSeesModel.from_h5(path, fem_root="/model")
    assert om.ndf == 6, (
        f"node_ndf envelope lost across domain_capture: recovered "
        f"ndf={om.ndf}, expected 6.  The DomainCapture sidecar bridge "
        f"forwarding did not persist /model/meta ndf."
    )
    # And the full Results chain opens against the recovered model.
    with Results.from_native(path, model=om) as r:
        assert [s.name for s in r.stages] == ["lat"]
