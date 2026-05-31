"""
Public API tests for g.loads.surface.force_resultant_center_mass(... magnitude=, normal=, direction=).

Resolver-level math is covered in tests/test_recent_features.py.  These
tests exercise the public face_load() validation and the end-to-end
PG -> face elements -> NodalLoadRecord pipeline on a real session.
"""
from __future__ import annotations

import numpy as np
import pytest


def _build_box_with_top_face(g):
    """Box with the +z face named 'Top' as a physical group."""
    g.model.geometry.add_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, label='cube')
    # Identify the top face (z = 1) via boundary query
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
# Validation
# =====================================================================

def test_face_load_rejects_force_xyz_with_magnitude(g):
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label='cube')
    g.physical.add_surface(
        g.model.queries.boundary('cube', dim=2)[0][1], name='Some',
    )
    with pytest.raises(ValueError, match="not both"):
        g.loads.surface.force_resultant_center_mass(
            'Some', force=(0, 0, -10), magnitude=100.0, normal=True,
        )


def test_face_load_rejects_normal_and_direction_together(g):
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label='cube')
    g.physical.add_surface(
        g.model.queries.boundary('cube', dim=2)[0][1], name='Some',
    )
    with pytest.raises(ValueError, match="not both"):
        g.loads.surface.force_resultant_center_mass(
            'Some', magnitude=100.0, normal=True, direction=(0, 0, 1),
        )


def test_face_load_magnitude_requires_normal_or_direction(g):
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label='cube')
    g.physical.add_surface(
        g.model.queries.boundary('cube', dim=2)[0][1], name='Some',
    )
    with pytest.raises(ValueError, match="normal=True or direction"):
        g.loads.surface.force_resultant_center_mass('Some', magnitude=100.0)


def test_face_load_requires_some_input(g):
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label='cube')
    g.physical.add_surface(
        g.model.queries.boundary('cube', dim=2)[0][1], name='Some',
    )
    with pytest.raises(ValueError, match="force, moment, or magnitude"):
        g.loads.surface.force_resultant_center_mass('Some')


# =====================================================================
# End-to-end: PG -> NodalLoadRecord
# =====================================================================

def test_face_load_normal_resolves_to_total_force_along_normal(g):
    """End-to-end: face_load(magnitude=F, normal=True) on the cube's
    +z face produces records summing to (0, 0, +F) — magnitude acts
    along +n_avg.  Negative F gives the pressure-like into-face load."""
    _build_box_with_top_face(g)
    F = 200.0
    with g.loads.pattern("Test"):
        g.loads.surface.force_resultant_center_mass('Top', magnitude=F, normal=True)

    fem = g.mesh.queries.get_fem_data(dim=3)
    total = np.zeros(3)
    for nl in fem.nodes.loads:
        if nl.force_xyz is None:
            continue
        total += np.asarray(nl.force_xyz, dtype=float)
    np.testing.assert_allclose(total, [0.0, 0.0, +F], atol=1e-6)


def test_face_load_direction_resolves_to_total_force_along_direction(g):
    _build_box_with_top_face(g)
    F = 50.0
    with g.loads.pattern("Test"):
        g.loads.surface.force_resultant_center_mass('Top', magnitude=F, direction=(1.0, 0.0, 0.0))

    fem = g.mesh.queries.get_fem_data(dim=3)
    total = np.zeros(3)
    for nl in fem.nodes.loads:
        if nl.force_xyz is None:
            continue
        total += np.asarray(nl.force_xyz, dtype=float)
    np.testing.assert_allclose(total, [F, 0.0, 0.0], atol=1e-6)


# face_sp tests moved to tests/test_displacements.py (ADR 0050 P2 —
# face_sp relocated to g.displacements.surface).
