"""ADR 0050 P2 — g.displacements composite (prescribed motion).

Covers the new authoring surface, the `point` SP variant, and the
`surface` verb (the relocated `face_sp`) — including the unchanged
resolution path to `fem.nodes.sp`.
"""
from __future__ import annotations

import numpy as np
import pytest


def _build_box_with_top_face(g):
    """Box with the +z face named 'Top' as a physical group."""
    g.model.geometry.add_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, label='cube')
    faces = g.model.queries.boundary('cube', dim=2)
    top = None
    for d, t in faces:
        com = g.model.queries.center_of_mass(int(t), dim=int(d))
        if abs(com[2] - 1.0) < 1e-6:
            top = t
            break
    assert top is not None, "could not locate +z face"
    g.physical.add_volume('cube', name='Body')
    g.physical.add_surface([top], name='Top')
    g.mesh.sizing.set_global_size(0.5)
    g.mesh.generation.generate(dim=3)


# =====================================================================
# Composite wiring
# =====================================================================

def test_displacements_composite_is_wired(g):
    assert g.displacements is not None
    assert hasattr(g.displacements, "surface")
    assert hasattr(g.displacements, "point")


def test_face_sp_removed_from_loads(g):
    assert not hasattr(g.loads, "face_sp")


# =====================================================================
# Authoring
# =====================================================================

def test_point_builds_point_sp_def(g):
    d = g.displacements.point("X", dofs=[1, 1, 0], values=[0.01, 0.0, 0.0])
    assert d.kind == "point_sp"
    assert d.dofs == [1, 1, 0]
    assert d.values == (0.01, 0.0, 0.0)
    assert g.displacements.disp_defs[-1] is d


def test_point_homogeneous_when_values_none(g):
    d = g.displacements.point("X", dofs=[1, 1, 1])
    assert d.values is None


def test_surface_builds_face_sp_def(g):
    d = g.displacements.surface("Y", dofs=[1, 1, 0])
    assert d.kind == "face_sp"
    assert d.dofs == [1, 1, 0]


def test_surface_rejects_disp_with_magnitude(g):
    with pytest.raises(ValueError, match="not both"):
        g.displacements.surface("Y", disp_xyz=(0, 0, 0.1), magnitude=0.1,
                                normal=True)


def test_surface_magnitude_requires_normal_or_direction(g):
    with pytest.raises(ValueError, match="normal=True or direction"):
        g.displacements.surface("Y", magnitude=0.1)


def test_pattern_grouping(g):
    with g.displacements.pattern("settlement"):
        g.displacements.point("X", values=[0.0, 0.0, -0.02])
    assert g.displacements.disp_defs[-1].pattern == "settlement"


# =====================================================================
# End-to-end: defs -> SPRecords on fem.nodes.sp
# =====================================================================

def test_surface_normal_resolves_to_uniform_translation(g):
    """surface(magnitude=u, normal=True) on the +z face -> every node
    gets u_z = u (the relocated face_sp path, unchanged)."""
    _build_box_with_top_face(g)
    u = 0.05
    with g.displacements.pattern("Test"):
        g.displacements.surface('Top', magnitude=u, normal=True)

    fem = g.mesh.queries.get_fem_data(dim=3)
    by_node_dof = {(int(sp.node_id), int(sp.dof)): float(sp.value)
                   for sp in fem.nodes.sp}
    z_vals = [v for (n, d), v in by_node_dof.items() if d == 3]
    x_vals = [v for (n, d), v in by_node_dof.items() if d == 1]
    assert z_vals, "no z-component SPRecords emitted"
    np.testing.assert_allclose(z_vals, u, atol=1e-9)
    np.testing.assert_allclose(x_vals, 0.0, atol=1e-9)


def test_point_prescribed_value_resolves_to_sp(g):
    """point(dofs=[0,0,1], values=[0,0,-0.03]) on the +z face -> every
    node of the face gets a uz = -0.03 SPRecord, nothing on dof 1/2."""
    _build_box_with_top_face(g)
    with g.displacements.pattern("Settle"):
        g.displacements.point('Top', dofs=[0, 0, 1], values=[0, 0, -0.03])

    fem = g.mesh.queries.get_fem_data(dim=3)
    sp = list(fem.nodes.sp)
    assert sp, "no SPRecords emitted"
    assert all(int(r.dof) == 3 for r in sp), "expected only dof-3 records"
    np.testing.assert_allclose([float(r.value) for r in sp], -0.03, atol=1e-12)
    assert all(not r.is_homogeneous for r in sp)


def test_point_homogeneous_is_allowed(g):
    """A zero prescribed displacement is a legal pattern-bound hold."""
    _build_box_with_top_face(g)
    with g.displacements.pattern("Hold"):
        g.displacements.point('Top', dofs=[1, 1, 1])

    fem = g.mesh.queries.get_fem_data(dim=3)
    sp = list(fem.nodes.sp)
    assert sp, "no SPRecords emitted"
    assert all(r.is_homogeneous for r in sp)
    np.testing.assert_allclose([float(r.value) for r in sp], 0.0, atol=1e-12)
