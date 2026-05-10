"""
Tests -- g.mesh.editing.crack(...) and the side_labels= behaviour.

Verifies:
  * Default ``side_labels=True`` attaches PGs ``<pg>_normal`` and
    ``<pg>_inverted`` to the two surviving face entities.
  * Each PG owns exactly one entity, and the two are disjoint.
  * The classification is correct: the entity tagged ``<pg>_normal``
    is bonded to volume tets on the +normal half-space of the
    original surface normal; ``<pg>_inverted`` to the -normal side.
  * ``side_labels=(a, b)`` overrides the default names.
  * ``side_labels=False`` skips the auto-labeling.
"""
from __future__ import annotations

import gmsh
import numpy as np
import pytest


# =====================================================================
# Helpers
# =====================================================================

def _build_box_with_embedded_plane(g):
    """Box (-50..50)^3 with an XY rectangle embedded at z=0."""
    g.model.geometry.add_box(-50, -50, -50, 100, 100, 100, label='box')
    g.model.geometry.add_rectangle(-10, -10, 0, 20, 20, label='plane')
    g.model.boolean.fragment(
        objects='box', tools='plane', cleanup_free=False,
    )
    g.physical.add_volume('box',    name='Body')
    g.physical.add_surface('plane', name='Crack')
    g.mesh.sizing.set_global_size(5.0)
    g.mesh.generation.generate(dim=3)


def _classify_entity_side(face_tag: int) -> int:
    """+1 if adjacent tet centroid lies in +normal half-space, -1 else."""
    etypes, _, enodes = gmsh.model.mesh.getElements(2, face_tag)
    tri = np.asarray(enodes[0], dtype=int).reshape(-1, 3)[0]
    p = np.array([gmsh.model.mesh.getNode(int(n))[0] for n in tri])
    n = np.cross(p[1] - p[0], p[2] - p[0])
    n = n / np.linalg.norm(n)
    ref = p.mean(axis=0)
    nodes_in_tri = {int(x) for x in tri}
    for vd, vt in gmsh.model.getEntities(3):
        _, _, vconn = gmsh.model.mesh.getElements(vd, vt)
        tets = np.asarray(vconn[0], dtype=int).reshape(-1, 4)
        for tet in tets:
            if sum(1 for k in tet if int(k) in nodes_in_tri) == 3:
                pts = np.array([
                    gmsh.model.mesh.getNode(int(k))[0] for k in tet
                ])
                centroid = pts.mean(axis=0)
                return 1 if np.dot(centroid - ref, n) > 0 else -1
    raise RuntimeError("no adjacent tet found")


# =====================================================================
# Tests
# =====================================================================

def test_crack_side_labels_default_creates_named_pgs(g):
    _build_box_with_embedded_plane(g)
    g.mesh.editing.crack('Crack', dim=2)   # side_labels=True is default

    n_tag = g.physical.get_tag(2, 'Crack_normal')
    i_tag = g.physical.get_tag(2, 'Crack_inverted')
    assert n_tag is not None, "Crack_normal PG not created"
    assert i_tag is not None, "Crack_inverted PG not created"


def test_crack_side_labels_each_pg_owns_one_disjoint_entity(g):
    _build_box_with_embedded_plane(g)
    g.mesh.editing.crack('Crack', dim=2)

    n_tag = g.physical.get_tag(2, 'Crack_normal')
    i_tag = g.physical.get_tag(2, 'Crack_inverted')
    n_ents = list(gmsh.model.getEntitiesForPhysicalGroup(2, n_tag))
    i_ents = list(gmsh.model.getEntitiesForPhysicalGroup(2, i_tag))

    assert len(n_ents) == 1
    assert len(i_ents) == 1
    assert set(n_ents).isdisjoint(set(i_ents))


def test_crack_side_labels_orientation_is_correct(g):
    """Original normal is +z (XY rectangle).  Crack_normal must be
    bonded to tets at z>0; Crack_inverted to tets at z<0."""
    _build_box_with_embedded_plane(g)
    g.mesh.editing.crack('Crack', dim=2)

    n_tag = g.physical.get_tag(2, 'Crack_normal')
    i_tag = g.physical.get_tag(2, 'Crack_inverted')
    (n_ent,) = gmsh.model.getEntitiesForPhysicalGroup(2, n_tag)
    (i_ent,) = gmsh.model.getEntitiesForPhysicalGroup(2, i_tag)

    assert _classify_entity_side(int(n_ent)) == +1, \
        "Crack_normal should be bonded to +normal-side tets"
    assert _classify_entity_side(int(i_ent)) == -1, \
        "Crack_inverted should be bonded to -normal-side tets"


def test_crack_side_labels_custom_names(g):
    _build_box_with_embedded_plane(g)
    g.mesh.editing.crack(
        'Crack', dim=2, side_labels=('topFace', 'botFace'),
    )

    assert g.physical.get_tag(2, 'topFace') is not None
    assert g.physical.get_tag(2, 'botFace') is not None
    assert g.physical.get_tag(2, 'Crack_normal')   is None
    assert g.physical.get_tag(2, 'Crack_inverted') is None

    (top_ent,) = gmsh.model.getEntitiesForPhysicalGroup(
        2, g.physical.get_tag(2, 'topFace'))
    (bot_ent,) = gmsh.model.getEntitiesForPhysicalGroup(
        2, g.physical.get_tag(2, 'botFace'))
    assert _classify_entity_side(int(top_ent)) == +1
    assert _classify_entity_side(int(bot_ent)) == -1


def test_crack_side_labels_disabled_creates_no_extras(g):
    _build_box_with_embedded_plane(g)

    pgs_before = {
        gmsh.model.getPhysicalName(d, t)
        for d, t in gmsh.model.getPhysicalGroups()
    }
    g.mesh.editing.crack('Crack', dim=2, side_labels=False)
    pgs_after = {
        gmsh.model.getPhysicalName(d, t)
        for d, t in gmsh.model.getPhysicalGroups()
    }

    assert pgs_after == pgs_before, \
        f"side_labels=False should not create PGs; got new: {pgs_after - pgs_before}"
