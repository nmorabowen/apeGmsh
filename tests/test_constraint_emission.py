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
# distributing_coupling / mortar are fail-loud at the API (PR-C):
# never silently emit a mechanically-wrong RBE3 / mortar record.
# =====================================================================

def test_distributing_coupling_factory_raises_not_implemented():
    with apeGmsh(model_name="pc_distrib", verbose=False) as g:
        with pytest.raises(NotImplementedError, match="RBE3|distributing"):
            g.constraints.distributing_coupling("A", "B")


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
# Phase 2 repro — mixed-host element types silently dropped
# ---------------------------------------------------------------------
#
# ``_collect_host_elems`` (ConstraintsComposite.py:1493) keeps only
# Gmsh element types 2 (tri3) and 4 (tet4); every other type — quad4
# (etype 3), hex8 (etype 5), prism (6), tet10 (11), … — is silently
# dropped.  A user with a mixed mesh (e.g. transition between hex and
# tet, or quad-recombined surface adjacent to a triangulated one) gets
# the supported subset and no warning; embedded nodes that fall into
# the dropped region project onto the nearest tri/tet (often distant)
# producing extrapolation weights, exactly the same wrong-physics
# failure as the off-host bug.
#
# Phase 2 fix: raise when the host PG contains any element type other
# than tri3/tet4, with a clear message naming the offending type.


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


def test_quad4_host_raises_with_clear_message():
    """A quad-meshed (recombined) host surface fails loud at
    ``_collect_host_elems``, naming quad4 as the unsupported type —
    not a misleading downstream 'host has no elements' message.
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
        etypes, _, _ = gmsh.model.mesh.getElements(dim=2, tag=surf)
        assert 3 in [int(e) for e in etypes], (
            f"setup precondition: expected quad4 (etype 3) in the "
            f"recombined mesh; got etypes={[int(e) for e in etypes]}"
        )

        # Direct call to the collector — currently returns empty,
        # which is the silent drop.  After Phase 2 fix it must raise.
        with pytest.raises(ValueError, match="quad4|element type 3|hex|unsupported"):
            ConstraintsComposite._collect_host_elems([(2, surf)])


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
