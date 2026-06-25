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


def test_mortar_delegates_to_fork_contact_tie():
    # mortar() is now a deprecated alias delegating to the fork ALM-penalty
    # mortar mesh-tie: contact(formulation="mortar", tie=True). It returns a
    # ContactDef (not the old MortarDef) and emits a DeprecationWarning.
    from apeGmsh._kernel.defs.constraints import ContactDef

    with apeGmsh(model_name="pc_mortar", verbose=False) as g:
        with pytest.warns(DeprecationWarning, match="mortar.*deprecated|contact"):
            d = g.constraints.mortar("A", "B", outward=(0.0, 0.0, 1.0))
        assert isinstance(d, ContactDef)
        assert d.formulation == "mortar" and d.tie is True
        assert d.outward == (0.0, 0.0, 1.0)
        assert d.eps_n == "auto"
        # the def is registered on the contact list (resolves to
        # fem.elements.contacts), not the MP-constraint list.
        assert d in g.constraints.contact_defs


def test_mortar_requires_outward():
    # tie=True mandates an explicit outward (a coincident-flat tie else binds
    # nothing). outward is a required keyword on mortar().
    with apeGmsh(model_name="pc_mortar_no_outward", verbose=False) as g:
        with pytest.raises(TypeError, match="outward"):
            g.constraints.mortar("A", "B")  # type: ignore[call-arg]


def test_mortar_drops_legacy_params():
    # the never-functional Lagrange-path params (dofs / integration_order) are
    # removed — passing them is a loud TypeError, not a silent no-op.
    import warnings

    with apeGmsh(model_name="pc_mortar_legacy", verbose=False) as g:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with pytest.raises(TypeError, match="dofs"):
                g.constraints.mortar(
                    "A", "B", outward=(0.0, 0.0, 1.0), dofs=[1, 2, 3])  # type: ignore[call-arg]
            with pytest.raises(TypeError, match="integration_order"):
                g.constraints.mortar(
                    "A", "B", outward=(0.0, 0.0, 1.0),
                    integration_order=3)  # type: ignore[call-arg]


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
# as_element rigid bodies: excluded from rigid_link_groups(), surfaced
# by rigid_body_elements() (adversarial-review fix for ADR 0071)
# =====================================================================

def test_rigid_link_groups_excludes_as_element_rigid_body():
    chain = _rigid_body(20, [21])
    elem = NodeGroupRecord(
        kind=K.RIGID_BODY, master_node=40, slave_nodes=[41, 42],
        dofs=[1, 2, 3, 4, 5, 6], as_element=True, mass=8.0,
    )
    cs = NodeConstraintSet([chain, elem])
    masters = {m for m, _ in cs.rigid_link_groups()}
    # The chain rigid_body (20) IS a rigidLink group; the as_element one
    # (40) is NOT — it goes out as the LadrunoRigidBody element, so
    # including it here too would double-constrain.
    assert masters == {20}
    assert 40 not in masters


def test_rigid_body_elements_yields_only_as_element_bodies():
    chain = _rigid_body(20, [21])
    elem = NodeGroupRecord(
        kind=K.RIGID_BODY, master_node=40, slave_nodes=[41, 42],
        dofs=[1, 2, 3, 4, 5, 6], as_element=True, mass=8.0,
    )
    spin = NodeGroupRecord(
        kind=K.RIGID_BODY, master_node=50, slave_nodes=[51],
        dofs=[1, 2, 3, 4, 5, 6], as_element=True, omega=(0.0, 0.0, 3.0),
    )
    cs = NodeConstraintSet([chain, elem, spin])
    rbe = list(cs.rigid_body_elements())
    # (master, slaves, mass, omega)
    assert rbe == [
        (40, [41, 42], 8.0, None),
        (50, [51], None, (0.0, 0.0, 3.0)),
    ]


# =====================================================================
# equal_dof_mixed: own accessor; not silently dropped (ADR 0069)
# =====================================================================

def test_equal_dofs_excludes_mixed_and_mixed_has_own_accessor():
    sym = NodePairRecord(kind=K.EQUAL_DOF, master_node=1, slave_node=2,
                         dofs=[1, 2, 3])
    mixed = NodePairRecord(kind=K.EQUAL_DOF_MIXED, master_node=3, slave_node=4,
                           dofs=[6, 1], master_dofs=[3, 1])
    cs = NodeConstraintSet([sym, mixed])
    # equal_dofs() yields only the symmetric tie (mixed is a different
    # OpenSees command); equal_dofs_mixed() yields the mixed one with both
    # DOF lists intact.
    eq = list(cs.equal_dofs())
    assert [r.slave_node for r in eq] == [2]
    em = list(cs.equal_dofs_mixed())
    assert len(em) == 1
    assert em[0].master_node == 3 and em[0].slave_node == 4
    assert em[0].master_dofs == [3, 1] and em[0].dofs == [6, 1]
    # pairs() still yields the mixed record whole (kind-dispatchable).
    mp = [p for p in cs.pairs() if p.kind == K.EQUAL_DOF_MIXED]
    assert len(mp) == 1 and mp[0].master_dofs == [3, 1]


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
