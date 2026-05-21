"""Runnable-deck integration tests for Phase 7b (ADR 0022 INV-1).

These tests drive a full ``apeSees(fem)`` model that declares MP
constraints (``g.constraints.equal_dof(...)`` /
``g.constraints.rigid_diaphragm(...)`` / surface coupling) through
the bridge's emit pipeline, then run the result against an actual
:class:`LiveOpsEmitter` (in-process openseespy) and assert that
``analyze`` converges to a non-trivial answer.

Without these tests, a "syntactically valid" deck that silently
drops constraints would still pass — INV-1 is exactly the gate
that catches that.

Gated by the ``live`` marker — only runs when ``openseespy`` is
installed.
"""
from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from apeGmsh.opensees import apeSees

openseespy = pytest.importorskip("openseespy.opensees")

# Imports deferred until openseespy is confirmed present.
from apeGmsh._kernel.records._constraints import (  # noqa: E402
    InterpolationRecord,
    NodeGroupRecord,
    NodePairRecord,
    SurfaceCouplingRecord,
)
from apeGmsh._kernel.records._kinds import ConstraintKind  # noqa: E402
from apeGmsh.opensees.emitter.live import LiveOpsEmitter  # noqa: E402

from tests.opensees.fixtures.fem_stub import (  # noqa: E402
    FEMStub,
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
    make_two_column_frame,
)


# ---------------------------------------------------------------------------
# INV-1: runnable deck for each MP constraint kind
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_equalDOF_deck_analyzes() -> None:
    """A frame with an equalDOF constraint between the two column
    tops produces a runnable deck.

    Setup: two-column frame, base fully fixed, column tops 2 and 4
    coupled in all 6 DOFs via equalDOF.  A single load at node 2
    must produce non-trivial displacement at BOTH node 2 and node 4
    (the equalDOF makes them displace together).
    """
    fem = make_two_column_frame()
    fem.add_node_constraints([
        NodePairRecord(
            kind=ConstraintKind.EQUAL_DOF,
            master_node=2, slave_node=4,
            dofs=[1, 2, 3, 4, 5, 6],
            name="rigid_floor",
        ),
    ])
    ops = apeSees(cast("object", fem))
    ops.model(ndm=3, ndf=6)

    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols", transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )
    ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))

    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts) as p:
        p.load(node=2, forces=(1000.0, 0.0, 0.0, 0.0, 0.0, 0.0))

    # Use Transformation (not Plain) so equalDOF actually couples DOFs.
    ops.constraints.Transformation()
    ops.numberer.Plain()
    ops.system.BandGeneral()
    ops.test.NormDispIncr(tol=1e-9, max_iter=10)
    ops.algorithm.Linear()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()

    emitter = LiveOpsEmitter(wipe=True)
    bm = ops.build()
    bm.emit(emitter)
    ret = emitter.analyze(steps=1)
    assert ret == 0, "openseespy.analyze returned non-zero"

    # INV-1: non-trivial result.
    disp_2 = emitter.ops.nodeDisp(2, 1)
    disp_4 = emitter.ops.nodeDisp(4, 1)
    assert abs(disp_2) > 1e-9, (
        f"node 2 displacement is essentially zero ({disp_2}); "
        "the load did not translate into deformation."
    )
    # equalDOF makes node 4 displace identically to node 2.
    assert disp_2 == pytest.approx(disp_4, abs=1e-9)


@pytest.mark.live
def test_rigid_diaphragm_deck_analyzes() -> None:
    """A frame with a rigid_diaphragm constraint coupling the two
    column tops produces a runnable deck.

    Setup: two-column frame, base fully fixed, column tops 2 and 4
    rigidly coupled via rigidDiaphragm with master=2.  A load at
    node 2 must drive node 4 in lockstep (translation in the rigid
    plane, no relative rotation about perp_dir=3).
    """
    fem = make_two_column_frame()
    fem.add_node_constraints([
        NodeGroupRecord(
            kind=ConstraintKind.RIGID_DIAPHRAGM,
            master_node=2, slave_nodes=[4],
            dofs=[1, 2, 6],
            plane_normal=np.array([0.0, 0.0, 1.0]),
            name="floor_1",
        ),
    ])
    ops = apeSees(cast("object", fem))
    ops.model(ndm=3, ndf=6)

    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols", transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )
    ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))

    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts) as p:
        p.load(node=2, forces=(1000.0, 0.0, 0.0, 0.0, 0.0, 0.0))

    ops.constraints.Transformation()
    ops.numberer.Plain()
    ops.system.BandGeneral()
    ops.test.NormDispIncr(tol=1e-9, max_iter=10)
    ops.algorithm.Linear()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()

    emitter = LiveOpsEmitter(wipe=True)
    bm = ops.build()
    bm.emit(emitter)
    ret = emitter.analyze(steps=1)
    assert ret == 0, "openseespy.analyze returned non-zero"

    # INV-1: non-trivial result.  rigidDiaphragm with perp=3 couples
    # in-plane translation (DOFs 1, 2) and out-of-plane rotation
    # (DOF 6).  Node 4 sees the same X-translation as the master.
    disp_2_x = emitter.ops.nodeDisp(2, 1)
    disp_4_x = emitter.ops.nodeDisp(4, 1)
    assert abs(disp_2_x) > 1e-9, (
        f"master node 2 disp is essentially zero ({disp_2_x})"
    )
    assert disp_2_x == pytest.approx(disp_4_x, abs=1e-6)


@pytest.mark.live
def test_rigid_link_deck_analyzes() -> None:
    """A frame with a rigid_beam constraint between the two column
    tops produces a runnable deck.

    Setup: two-column frame, base fixed, master=2 / slave=4 rigid
    link (beam type — full 6-DOF rigid).  Load at node 2 propagates
    to node 4 via the rigid link.
    """
    fem = make_two_column_frame()
    fem.add_node_constraints([
        NodePairRecord(
            kind=ConstraintKind.RIGID_BEAM,
            master_node=2, slave_node=4,
            name="rigid_link_2_4",
        ),
    ])
    ops = apeSees(cast("object", fem))
    ops.model(ndm=3, ndf=6)

    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols", transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )
    ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))

    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts) as p:
        p.load(node=2, forces=(1000.0, 0.0, 0.0, 0.0, 0.0, 0.0))

    ops.constraints.Transformation()
    ops.numberer.Plain()
    ops.system.BandGeneral()
    ops.test.NormDispIncr(tol=1e-9, max_iter=10)
    ops.algorithm.Linear()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()

    emitter = LiveOpsEmitter(wipe=True)
    bm = ops.build()
    bm.emit(emitter)
    ret = emitter.analyze(steps=1)
    assert ret == 0, "openseespy.analyze returned non-zero"

    # INV-1: non-trivial result — node 2 displaces, node 4 follows
    # rigidly.
    disp_2 = emitter.ops.nodeDisp(2, 1)
    disp_4 = emitter.ops.nodeDisp(4, 1)
    assert abs(disp_2) > 1e-9
    assert disp_2 == pytest.approx(disp_4, abs=1e-6)


def _make_tied_contact_stub() -> FEMStub:
    """Single host tet + one slave node embedded in its interior.

    Geometry: a unit tet with vertices at (0,0,0), (1,0,0), (0,1,0),
    (0,0,1) is the "bottom plate" host element.  Node 5 sits at the
    barycentric centroid (0.25, 0.25, 0.25) — interior of the host
    tet — and stands in for a NON-MATCHING surface-mesh node from a
    top plate whose mesh shares no nodes with the bottom plate at
    their interface.

    The :class:`SurfaceCouplingRecord` ties node 5 to the host's
    four corner nodes via isoparametric interpolation (equal-weight
    barycentric, sum=1).  This mirrors what
    :func:`g.constraints.tied_contact` produces on a real two-plate
    fixture whose interface meshes are non-conforming: a single
    :class:`InterpolationRecord` per slave node, with master-node
    weights derived from the projection onto the master surface.

    Sharing **no** mesh nodes at the interface is exactly what makes
    this a tied-contact test (not an equalDOF / rigidLink test):
    without the ASDEmbeddedNodeElement emission, the slave node
    would be a free body and ``analyze`` would fail with a singular
    system.  With the §3.3 deferral closed, the bridge emits one
    ASDEmbeddedNodeElement per :class:`InterpolationRecord`, the
    slave is coupled to the host element, and the deck runs.

    PGs:
      * ``"Host"``: element 1 (the host tet).
      * ``"HostFixed"``: nodes 2, 3, 4 (3 corners — fix to ground).
      * ``"Apex"``: node 1 (free corner — receives the load).
      * ``"Slave"``: node 5 (the embedded slave).
    """
    nodes = _NodesStub(
        ids=[1, 2, 3, 4, 5],
        coords=[
            (0.0, 0.0, 0.0),     # 1 — host apex (loaded)
            (1.0, 0.0, 0.0),     # 2 — host base corner (fixed)
            (0.0, 1.0, 0.0),     # 3 — host base corner (fixed)
            (0.0, 0.0, 1.0),     # 4 — host base corner (fixed)
            (0.25, 0.25, 0.25),  # 5 — slave INSIDE the host tet
        ],
        node_pgs={
            "Host":      [1, 2, 3, 4],
            "HostFixed": [2, 3, 4],
            "Apex":      [1],
            "Slave":     [5],
        },
    )
    elements = _ElementsStub(
        elem_pgs={
            "Host": _ElementGroupView(
                ids=(1,), connectivity=((1, 2, 3, 4),),
            ),
        },
    )
    fem = FEMStub(nodes=nodes, elements=elements)
    fem.add_surface_constraints([
        SurfaceCouplingRecord(
            kind=ConstraintKind.TIED_CONTACT,
            name="interface",
            slave_nodes=[5],
            master_nodes=[1, 2, 3, 4],
            slave_records=[
                InterpolationRecord(
                    kind=ConstraintKind.TIED_CONTACT,
                    slave_node=5,
                    master_nodes=[1, 2, 3, 4],
                    # Barycentric weights for the centroid of the tet
                    # — sum to 1 (partition of unity).
                    weights=np.array([0.25, 0.25, 0.25, 0.25]),
                    dofs=[1, 2, 3],
                ),
            ],
            dofs=[1, 2, 3],
        ),
    ])
    return fem


@pytest.mark.live
def test_tied_contact_deck_analyzes() -> None:
    """A model with a tied_contact surface coupling produces a
    runnable deck — closes the last ADR 0022 INV-1 gap.

    Without ``tied_contact`` emission (the §3.3 deferral), the slave
    node would be a free body — its mesh shares no nodes with the
    host element's mesh, so the only way load reaches it is through
    the ASDEmbeddedNodeElement the bridge emits.  ``analyze`` would
    fail with a singular system if the constraint were silently
    dropped.

    Setup: one host tet (the "bottom plate") with three corners
    fixed and the apex free.  A single slave node sits at the
    barycentric centroid (interior of the host tet) — non-matching
    with respect to the host's nodes — and is tied via a
    :class:`SurfaceCouplingRecord` carrying one
    :class:`InterpolationRecord`.  A downward load on the host apex
    drives the host to deflect; the slave follows the host via the
    embedded-node coupling.

    Asserts:
      * ``ops.analyze(1) == 0``  (the deck converges).
      * Free host node deflects non-trivially under the load.
      * Slave node deflects non-trivially — the load reached it
        through the tied_contact coupling.
    """
    fem = _make_tied_contact_stub()
    ops = apeSees(cast("object", fem))
    ops.model(ndm=3, ndf=3)

    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.2, rho=2400)
    ops.element.FourNodeTetrahedron(pg="Host", material=mat)
    ops.fix(pg="HostFixed", dofs=(1, 1, 1))

    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts) as p:
        p.load(pg="Apex", forces=(0.0, 0.0, -1.0e3))

    # Transformation handler is required so ASDEmbeddedNodeElement
    # actually transfers DOFs.  (The Phase 8 auto-emit would default
    # to Transformation for us, but we declare it explicitly to
    # match the sibling tests in this file.)
    ops.constraints.Transformation()
    ops.numberer.Plain()
    ops.system.UmfPack()
    ops.test.NormDispIncr(tol=1e-6, max_iter=50)
    ops.algorithm.Newton()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()

    emitter = LiveOpsEmitter(wipe=True)
    bm = ops.build()
    bm.emit(emitter)
    ret = emitter.analyze(steps=1)
    assert ret == 0, f"openseespy.analyze returned non-zero ({ret})"

    # INV-1: the host's free apex deflects under the load.
    uz_apex = emitter.ops.nodeDisp(1, 3)
    assert abs(uz_apex) > 1e-9, (
        f"host apex displacement is essentially zero ({uz_apex}); "
        "the load did not translate into deformation."
    )

    # INV-1: the slave deflects too — the tied_contact coupling
    # transferred load from the host into the embedded slave node.
    # Without emission, the slave would be a free body and ``analyze``
    # would have failed with a singular system above; we still assert
    # non-trivial slave deflection to lock the load-transfer behaviour.
    uz_slave = emitter.ops.nodeDisp(5, 3)
    assert abs(uz_slave) > 1e-9, (
        f"slave node displacement is essentially zero ({uz_slave}); "
        "the tied_contact coupling did not transfer load."
    )


# ---------------------------------------------------------------------------
# Tcl-emit smoke — the Tcl deck has the constraint lines in the right place
# ---------------------------------------------------------------------------


def test_tcl_deck_contains_constraint_lines() -> None:
    """The Tcl emit path produces lines for every declared MP constraint
    in the correct order (after elements, before patterns)."""
    fem = make_two_column_frame()
    fem.add_node_constraints([
        NodeGroupRecord(
            kind=ConstraintKind.RIGID_DIAPHRAGM,
            master_node=1, slave_nodes=[2, 3, 4],
            dofs=[1, 2, 6],
            plane_normal=np.array([0.0, 0.0, 1.0]),
            name="floor_1",
        ),
    ])
    ops = apeSees(cast("object", fem))
    ops.model(ndm=3, ndf=6)

    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols", transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )
    ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))
    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts) as p:
        p.load(node=2, forces=(1000.0, 0.0, 0.0, 0.0, 0.0, 0.0))

    from apeGmsh.opensees.emitter.tcl import TclEmitter
    emitter = TclEmitter()
    bm = ops.build()
    bm.emit(emitter)

    lines = emitter.lines()
    # The diaphragm line is emitted with perp_dir=3.
    assert any("rigidDiaphragm 3 1 2 3 4" in line for line in lines)
    # The user's declaration label round-trips as a comment.
    assert "# floor_1" in lines
    # The comment immediately precedes the rigidDiaphragm line.
    idx = lines.index("# floor_1")
    assert lines[idx + 1].startswith("rigidDiaphragm")

    # INV-5: constraint line falls between the last element and the
    # first pattern.
    elem_indices = [
        i for i, line in enumerate(lines)
        if line.startswith("element ")
    ]
    constraint_indices = [
        i for i, line in enumerate(lines)
        if line.startswith("rigidDiaphragm")
    ]
    pattern_indices = [
        i for i, line in enumerate(lines)
        if line.startswith("pattern Plain")
    ]
    assert elem_indices, "no element lines found"
    assert constraint_indices, "no constraint lines found"
    assert pattern_indices, "no pattern lines found"
    assert (
        max(elem_indices) < min(constraint_indices) < min(pattern_indices)
    )
