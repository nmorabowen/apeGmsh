"""
Emission-iterator routing contract (deep-review PR-B).

The NodeConstraintSet iterators are the documented OpenSees emission
surface.  Two silent-wrong defects are locked out here:

* ``rigid_link_groups()`` must NOT collapse ``rigid_diaphragm`` (→
  double-constrain, since ``rigid_diaphragms()`` also emits it) or
  ``kinematic_coupling`` (→ silent 6-DOF over-constraint, ignoring
  its DOF subset) into a full ``rigidLink``.
* ``rigid_diaphragms()`` must surface the resolved plane normal as
  ``perpDirn`` so a non-horizontal diaphragm is not silently emitted
  with a hardcoded ``3``.
"""
import numpy as np
import pytest

from apeGmsh import apeGmsh
from apeGmsh._kernel.record_sets import NodeConstraintSet
from apeGmsh._kernel.records._constraints import NodeGroupRecord, NodePairRecord
from apeGmsh._kernel.records._kinds import ConstraintKind as K


# =====================================================================
# distributing_coupling (RBE3) now ships as the fork
# LadrunoDistributingCoupling element; mortar stays fail-loud.
# =====================================================================

def test_distributing_coupling_def_fields():
    # The factory validates part labels (like kinematic_coupling), so the
    # def shape is checked directly here; resolve/emit are covered by
    # tests/opensees/unit/test_distributing_coupling_emit.py.
    from apeGmsh._kernel.defs.constraints import DistributingCouplingDef
    d = DistributingCouplingDef(
        master_label="A", slave_label="B",
        master_point=(1.0, 0.0, 0.0), name="rbe3",
    )
    assert d.kind == "distributing"
    assert d.master_label == "A" and d.slave_label == "B"
    assert d.master_point == (1.0, 0.0, 0.0)
    assert d.weighting == "uniform"


def test_distributing_coupling_area_weighting_def():
    # weighting="area" is now supported (tributary areas computed by the
    # resolver — tests/test_constraint_resolver.py covers the math); the
    # def carries the mode through.
    from apeGmsh._kernel.defs.constraints import DistributingCouplingDef
    d = DistributingCouplingDef(
        master_label="A", slave_label="B", weighting="area",
    )
    assert d.weighting == "area"


def test_distributing_coupling_unknown_weighting_raises():
    with apeGmsh(model_name="pc_distrib_area", verbose=False) as g:
        with pytest.raises(ValueError, match="weighting"):
            g.constraints.distributing_coupling(
                "A", "B", weighting="tributary")


def test_mortar_factory_raises_not_implemented():
    with apeGmsh(model_name="pc_mortar", verbose=False) as g:
        with pytest.raises(NotImplementedError, match="mortar|tied_contact"):
            g.constraints.mortar("A", "B")


def test_embedded_factory_accepts_entity_scoping():
    """PR-D: host_entities/embedded_entities are now settable (the
    resolver already honoured them; the factory dropped them)."""
    with apeGmsh(model_name="pd_embed", verbose=False) as g:
        d = g.constraints.embedded(
            "concrete", "rebar",
            host_entities=[(3, 1)], embedded_entities=[(1, 7)],
        )
        assert d.host_entities == [(3, 1)]
        assert d.embedded_entities == [(1, 7)]


# ---------------------------------------------------------------------
# Host-decomposition contract (extends Phase 2)
# ---------------------------------------------------------------------
#
# ``_collect_host_subelements`` (renamed from ``_collect_host_elems``)
# now decomposes quad4 / hex8 / higher-order hosts into linear sub-tris
# / sub-tets using corner nodes only, so ``ASDEmbeddedNodeElement``
# (which only accepts 3-node or 4-node retained sets) can consume any
# supported host topology.  The coupling stays linear-over-corners
# regardless of the host's native interpolation order — see
# :class:`EmbeddedDef` ``host_coupling="linear"`` for the contract.
#
# Supported: tri3, tri6, quad4, quad8, quad9 (2D); tet4, tet10, hex8,
# hex20 (3D).  Prism (etype 6/18) and pyramid (etype 7) still raise.


def test_embedded_emit_rejects_unsupported_rnode_count():
    """C++ ASDEmbeddedNodeElement only accepts 3 or 4 Rnodes.  A
    hand-built record with 2 or 5+ masters must fail loud at emit
    rather than reaching OpenSees, which would either misread the
    extras as flags (5+) or abort in setDomain (2).
    """
    from apeGmsh._kernel.records._constraints import InterpolationRecord
    from apeGmsh._kernel.records._kinds import ConstraintKind
    from apeGmsh.opensees._internal.build import (
        _check_embedded_rnode_count,
    )

    bad = InterpolationRecord(
        kind=ConstraintKind.EMBEDDED,
        name="too_many_masters",
        slave_node=99,
        master_nodes=[1, 2, 3, 4, 5],
        dofs=[1, 2, 3],
    )
    with pytest.raises(ValueError, match="3 \\(tri3 host\\) or 4 \\(tet4 host\\)"):
        _check_embedded_rnode_count(bad)

    too_few = InterpolationRecord(
        kind=ConstraintKind.EMBEDDED,
        name="too_few_masters",
        slave_node=99,
        master_nodes=[1, 2],
        dofs=[1, 2, 3],
    )
    with pytest.raises(ValueError, match="3 \\(tri3 host\\) or 4 \\(tet4 host\\)"):
        _check_embedded_rnode_count(too_few)


def test_quad4_host_decomposed_to_triangles():
    """A quad-meshed (recombined) host surface decomposes into 2 tris
    per quad (split along the (0,2) diagonal), so
    ``ASDEmbeddedNodeElement`` (3-node retained set) can consume it
    via the linear corner-node coupling contract.

    The returned rows have shape (n_quads * 2, 3) and the unique node
    set equals the union of all quad corner nodes — no nodes
    fabricated, no real corners dropped.
    """
    import gmsh

    from apeGmsh.core.ConstraintsComposite import ConstraintsComposite

    with apeGmsh(model_name="phase2_mixed_host", verbose=False) as g:
        surf = g.model.geometry.add_rectangle(0.0, 0.0, 0.0, 1.0, 1.0)
        g.model.sync()
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        g.mesh.sizing.set_global_size(0.5)
        g.mesh.generation.generate(2)
        g.physical.add(2, [surf], name="concrete")

        # Sanity: the mesh actually contains quad4 (etype 3) elements.
        etypes, etags_by_type, enodes = gmsh.model.mesh.getElements(
            dim=2, tag=surf)
        codes = [int(e) for e in etypes]
        assert 3 in codes, (
            f"setup precondition: expected quad4 (etype 3) in the "
            f"recombined mesh; got etypes={codes}"
        )
        n_quads = len(etags_by_type[codes.index(3)])

        rows = ConstraintsComposite._collect_host_subelements(
            [(2, surf)])

        # Two tris per quad, all length-3.
        assert rows.shape == (2 * n_quads, 3), (
            f"expected {2 * n_quads} tri rows from {n_quads} quads, "
            f"got shape={rows.shape}"
        )

        # Every quad corner node appears in the returned rows.
        quad_nodes = np.unique(
            np.asarray(enodes[codes.index(3)], dtype=int))
        assert set(quad_nodes) == set(np.unique(rows)), (
            "decomposed tris must reference exactly the union of "
            "quad corner nodes — no fabrication, no drop"
        )


def test_prism_host_still_raises_actionably():
    """Prism (etype 6) is in the unsupported bucket per the linear-
    coupling scope.  The error must name prism explicitly and point
    at the remesh-or-issue escape hatch — not a generic 'unknown
    type' message.
    """
    import gmsh

    from apeGmsh.core.ConstraintsComposite import ConstraintsComposite

    with apeGmsh(model_name="prism_host_unsupported", verbose=False) as g:
        # A short extruded prism: triangulated bottom face,
        # extrude generates prism6 in the volume.
        surf = g.model.geometry.add_rectangle(0.0, 0.0, 0.0, 1.0, 1.0)
        g.model.sync()
        out = gmsh.model.occ.extrude(
            [(2, surf)], 0.0, 0.0, 0.5, numElements=[1], recombine=False)
        g.model.sync()
        vol_tag = next(t for d, t in out if d == 3)
        gmsh.option.setNumber("Mesh.RecombineAll", 0)
        g.mesh.sizing.set_global_size(0.5)
        g.mesh.generation.generate(3)

        etypes, _, _ = gmsh.model.mesh.getElements(dim=3, tag=vol_tag)
        codes = [int(e) for e in etypes]
        # Either prism (6) or hex (5) depending on Gmsh's extrude
        # decisions — skip cleanly if no prism produced.
        if 6 not in codes:
            pytest.skip(
                f"extrude produced {codes}, not prism6 — Gmsh's "
                f"extrude heuristic varies; skip rather than test "
                f"a different host type")

        with pytest.raises(ValueError, match="prism|not yet supported"):
            ConstraintsComposite._collect_host_subelements(
                [(3, vol_tag)])


def _diaphragm(master, slaves, normal):
    return NodeGroupRecord(
        kind=K.RIGID_DIAPHRAGM, master_node=master, slave_nodes=slaves,
        dofs=[1, 2, 3, 4, 5, 6], plane_normal=(None if normal is None
                                               else np.asarray(normal)),
    )


def _kinematic(master, slaves, dofs):
    return NodeGroupRecord(
        kind=K.KINEMATIC_COUPLING, master_node=master,
        slave_nodes=slaves, dofs=dofs,
    )


def _rigid_body(master, slaves):
    return NodeGroupRecord(
        kind=K.RIGID_BODY, master_node=master, slave_nodes=slaves,
        dofs=[1, 2, 3, 4, 5, 6],
    )


# =====================================================================
# rigid_link_groups() must exclude diaphragm + kinematic_coupling
# =====================================================================

def test_rigid_link_groups_excludes_diaphragm_and_kinematic():
    cs = NodeConstraintSet([
        _diaphragm(1, [2, 3], (0, 0, 1)),
        _kinematic(10, [11, 12], dofs=[1, 3, 5]),
        _rigid_body(20, [21]),
        NodePairRecord(kind=K.RIGID_BEAM, master_node=30, slave_node=31),
    ])
    masters = {m for m, _ in cs.rigid_link_groups()}
    # rigid_body + rigid_beam ARE rigid links; diaphragm (1) and
    # kinematic_coupling (10) are NOT.
    assert masters == {20, 30}
    assert 1 not in masters       # diaphragm not double-emitted
    assert 10 not in masters      # kinematic_coupling not 6-DOF-collapsed


def test_kinematic_coupling_still_reachable_via_pairs_with_dofs():
    cs = NodeConstraintSet([_kinematic(10, [11, 12], dofs=[1, 3, 5])])
    kp = [p for p in cs.pairs() if p.kind == K.KINEMATIC_COUPLING]
    assert len(kp) == 2
    assert all(p.master_node == 10 for p in kp)
    assert {p.slave_node for p in kp} == {11, 12}
    # The DOF subset is preserved (this is what the 6-DOF collapse
    # silently destroyed).
    assert all(p.dofs == [1, 3, 5] for p in kp)


# =====================================================================
# rigid_diaphragms() must surface perpDirn from the plane normal
# =====================================================================

def test_rigid_diaphragms_yields_perp_dirn_from_normal():
    cs = NodeConstraintSet([
        _diaphragm(1, [2, 3], (0, 0, 1)),     # XY plane  → perp 3
        _diaphragm(4, [5], (1, 0, 0)),        # YZ plane  → perp 1
        _diaphragm(6, [7], (0, 1, 0)),        # XZ plane  → perp 2
        _diaphragm(8, [9], None),             # missing   → 3 (legacy)
    ])
    got = {master: perp for perp, master, _ in cs.rigid_diaphragms()}
    assert got == {1: 3, 4: 1, 6: 2, 8: 3}


def test_diaphragm_not_double_emitted():
    """A rigid_diaphragm must appear in rigid_diaphragms() ONLY —
    never also in rigid_link_groups() (that double-constrains)."""
    cs = NodeConstraintSet([_diaphragm(1, [2, 3], (0, 0, 1))])
    diap_masters = {m for _, m, _ in cs.rigid_diaphragms()}
    link_masters = {m for m, _ in cs.rigid_link_groups()}
    assert diap_masters == {1}
    assert link_masters == set()          # NOT also here
