"""
Physical-outward normal for normal-pressure loads.

Verifies the per-face physical outward (computed via adjacent-tet
centroid) is used by ``surface(... normal=True)``,
``face_load(... normal=True)``, and ``face_sp(... normal=True)``
instead of the connectivity-derived face normal.

Why it matters:
    * Crack faces produced by ``g.mesh.editing.crack(...)`` have
      *the same* connectivity normal on the two coincident face
      entities (the Crack plugin does not flip triangle orientation),
      but the entities are bonded to opposite-side volume tets — so
      the physical outward is opposite for the two entities.  Using
      the connectivity normal silently flips the load direction on
      one of them.
    * Tilted faces created via ``add_rectangle(angles_deg=...)`` carry
      an unpredictable connectivity orientation.

The fix preserves backward compatibility on regular volume-boundary
surfaces, where the connectivity normal already agrees with the
physical outward — see ``test_backward_compat_cube_top_face_force``.

Sign convention recap (see the module docstring of
``apeGmsh/solvers/Loads.py`` for the authoritative version):

* ``surface(magnitude=+P, normal=True)``  →  ``f3 = -P * A * outward``
  (positive magnitude pushes the face into the body — compression).
* ``face_load(magnitude=+F, normal=True)`` →  ``f_total = +F * outward``
  (positive magnitude pulls the face along outward — tension).
* ``face_sp(magnitude=+u, normal=True)``   →  ``disp  = +u * outward``
  (positive magnitude displaces the face along outward).
"""
from __future__ import annotations

import gmsh
import numpy as np
import pytest


# =====================================================================
# Helpers
# =====================================================================

def _build_unit_cube_with_top(g):
    """Unit cube with the +z face named ``Top``."""
    g.model.geometry.add_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, label='cube')
    top_tag = None
    for d, t in g.model.queries.boundary('cube', dim=2):
        com = g.model.queries.center_of_mass(int(t), dim=int(d))
        if abs(com[2] - 1.0) < 1e-6:
            top_tag = int(t)
            break
    assert top_tag is not None
    g.physical.add_volume('cube', name='Body')
    g.physical.add_surface([top_tag], name='Top')
    g.mesh.sizing.set_global_size(0.5)
    g.mesh.generation.generate(dim=3)


def _build_box_with_embedded_crack(g):
    """100 m box with a 20 m embedded XY rectangle at z=0, then crack().

    Replicates the wave_propagation_crack.ipynb topology: ``Crack_normal``
    is the face entity bonded to the +z half-space, ``Crack_inverted``
    to the -z half-space.
    """
    g.model.geometry.add_box(-50, -50, -50, 100, 100, 100, label='box')
    g.model.geometry.add_rectangle(-10, -10, 0, 20, 20, label='plane')
    g.model.boolean.fragment(
        objects='box', tools='plane', cleanup_free=False,
    )
    g.physical.add_volume('box', name='Body')
    g.physical.add_surface('plane', name='Crack')
    g.mesh.sizing.set_global_size(20.0)
    g.mesh.generation.generate(dim=3)
    g.mesh.editing.crack('Crack', dim=2)


def _entity_node_set(physical_name: str) -> set[int]:
    """Set of node IDs belonging to the surface PG ``physical_name``."""
    pg_tag = None
    for d, t in gmsh.model.getPhysicalGroups(dim=2):
        if gmsh.model.getPhysicalName(d, t) == physical_name:
            pg_tag = int(t)
            break
    assert pg_tag is not None, f"no PG named {physical_name}"
    nodes: set[int] = set()
    for ent_tag in gmsh.model.getEntitiesForPhysicalGroup(2, pg_tag):
        ntags, _, _ = gmsh.model.mesh.getNodes(
            dim=2, tag=int(ent_tag),
            includeBoundary=True, returnParametricCoord=False,
        )
        nodes.update(int(n) for n in ntags)
    return nodes


def _sum_loads_for_nodes(fem, node_set: set[int]) -> np.ndarray:
    """Sum nodal force vectors over the given node set."""
    total = np.zeros(3, dtype=float)
    for nl in fem.nodes.loads:
        if nl.force_xyz is None:
            continue
        if int(nl.node_id) in node_set:
            total += np.asarray(nl.force_xyz, dtype=float)
    return total


# =====================================================================
# (a) Backward compat: regular volume-boundary face is unaffected
# =====================================================================

def test_backward_compat_cube_top_face_force(g):
    """``surface(+P, normal=True)`` on the cube's +z face gives a total
    z-force of ``-P * A`` per the convention — the connectivity normal
    and the physical outward agree on a regular boundary face, so the
    new physical-outward path produces the same result as the old
    connectivity-normal path."""
    _build_unit_cube_with_top(g)
    P = 7.0
    with g.loads.pattern("Test"):
        g.loads.surface('Top', magnitude=P, normal=True)

    fem = g.mesh.queries.get_fem_data(dim=3)
    total = np.zeros(3, dtype=float)
    for nl in fem.nodes.loads:
        if nl.force_xyz is None:
            continue
        total += np.asarray(nl.force_xyz, dtype=float)
    # Top face is 1x1 = 1.0 m^2; outward = +z; convention -P*A*outward.
    np.testing.assert_allclose(total, [0.0, 0.0, -P * 1.0], atol=1e-9)


def test_backward_compat_cube_top_face_load(g):
    """``face_load(+F, normal=True)`` on the cube's +z face still
    produces a total force of ``(0, 0, +F)`` — outward = connectivity
    normal = +z, so the new outward path matches the old one."""
    _build_unit_cube_with_top(g)
    F = 50.0
    with g.loads.pattern("Test"):
        g.loads.face_load('Top', magnitude=F, normal=True)

    fem = g.mesh.queries.get_fem_data(dim=3)
    total = np.zeros(3, dtype=float)
    for nl in fem.nodes.loads:
        if nl.force_xyz is None:
            continue
        total += np.asarray(nl.force_xyz, dtype=float)
    np.testing.assert_allclose(total, [0.0, 0.0, +F], atol=1e-9)


def test_backward_compat_cube_top_face_sp(g):
    """``face_sp(+u, normal=True)`` on the cube's +z face still
    prescribes ``u_z = +u`` on every node."""
    _build_unit_cube_with_top(g)
    u = 0.05
    with g.loads.pattern("Test"):
        g.loads.face_sp('Top', magnitude=u, normal=True)

    fem = g.mesh.queries.get_fem_data(dim=3)
    z_vals = [float(sp.value) for sp in fem.nodes.sp if int(sp.dof) == 3]
    xy_vals = [
        float(sp.value) for sp in fem.nodes.sp if int(sp.dof) in (1, 2)
    ]
    assert z_vals, "no z-component SPRecords emitted"
    np.testing.assert_allclose(z_vals, u, atol=1e-12)
    np.testing.assert_allclose(xy_vals, 0.0, atol=1e-12)


# =====================================================================
# (b) Embedded-crack symmetry: physical outward is opposite on the two
# coincident face entities, even though their connectivity normal is
# the same.
# =====================================================================

def test_crack_face_outward_normals_are_opposite(g):
    """``LoadsComposite._face_outward_normals`` returns opposite-pointing
    unit vectors for the two crack entities — even though their
    connectivity normal is identical (both +z)."""
    _build_box_with_embedded_crack(g)

    # Pull each entity's face elements as the composite would.
    cn_faces = g.loads._target_faces('Crack_normal',   source='auto')
    ci_faces = g.loads._target_faces('Crack_inverted', source='auto')
    assert cn_faces and ci_faces, "no face elements for crack PGs"

    cn_outwards = g.loads._face_outward_normals(cn_faces)
    ci_outwards = g.loads._face_outward_normals(ci_faces)
    assert cn_outwards is not None and ci_outwards is not None

    # Original crack plane = XY plane → connectivity normal is ±z.
    # Physical outward of CN (bonded to +z body) must be -z.
    # Physical outward of CI (bonded to -z body) must be +z.
    for n in cn_outwards:
        np.testing.assert_allclose(n, [0.0, 0.0, -1.0], atol=1e-9)
    for n in ci_outwards:
        np.testing.assert_allclose(n, [0.0, 0.0, +1.0], atol=1e-9)


def test_crack_surface_same_sign_gives_opposite_loads(g):
    """``surface(+P, normal=True)`` on BOTH crack faces.

    Per the convention ``f3 = -P * A * outward``:
      * CN outward = -z → CN nodal forces in +z direction (compression
        of the +z body, which mechanically OPENS the crack as the CN
        face is pushed up into the body interior).
      * CI outward = +z → CI nodal forces in -z direction (compression
        of the -z body — face pushed down into the body interior).
    The two face entities therefore receive opposite-direction loads
    even though the same magnitude was specified.  This is the
    same-sign-on-both idiom unlocked by the physical-outward fix.

    The mesh-rim nodes are shared between the two coincident face
    entities, so their loads cancel; the asymmetry is visible on the
    interior (disjoint) nodes plus in the global zero-sum check.
    """
    _build_box_with_embedded_crack(g)
    P = 1.0e3
    with g.loads.pattern("CrackPressure"):
        g.loads.surface('Crack_normal',   magnitude=+P, normal=True)
        g.loads.surface('Crack_inverted', magnitude=+P, normal=True)
    fem = g.mesh.queries.get_fem_data(dim=3)

    cn_nodes = _entity_node_set('Crack_normal')
    ci_nodes = _entity_node_set('Crack_inverted')
    assert cn_nodes and ci_nodes

    # Global Newton-3 check: equal-and-opposite loads on coincident
    # faces must sum to zero on the whole model.
    grand_total = np.zeros(3, dtype=float)
    for nl in fem.nodes.loads:
        if nl.force_xyz is not None:
            grand_total += np.asarray(nl.force_xyz, dtype=float)
    np.testing.assert_allclose(grand_total, 0.0, atol=1e-6)

    # Disjoint inner nodes show the per-side asymmetry directly.
    cn_only = cn_nodes - ci_nodes
    ci_only = ci_nodes - cn_nodes
    assert cn_only and ci_only
    cn_inner_total = _sum_loads_for_nodes(fem, cn_only)
    ci_inner_total = _sum_loads_for_nodes(fem, ci_only)
    np.testing.assert_allclose(cn_inner_total[:2], 0.0, atol=1e-6)
    np.testing.assert_allclose(ci_inner_total[:2], 0.0, atol=1e-6)
    # CN-only nodes feel +z, CI-only nodes feel -z, equal-and-opposite
    # in magnitude (because the loaded rectangle is the same on both
    # sides — symmetric mesh on the two coincident entities).
    assert cn_inner_total[2] > 0.0, \
        f"CN-only nodes should feel +z, got {cn_inner_total[2]}"
    assert ci_inner_total[2] < 0.0, \
        f"CI-only nodes should feel -z, got {ci_inner_total[2]}"
    np.testing.assert_allclose(
        cn_inner_total[2], -ci_inner_total[2], rtol=1e-6,
    )


def test_crack_surface_negative_sign_closes_crack(g):
    """``surface(-P, normal=True)`` on BOTH crack faces produces
    suction on each body — the sign-flipped twin of the opening case.

    Per the convention with ``magnitude = -P``:
      * CN outward = -z → CN nodal forces in -z direction (suction
        pulling CN face away from +z body interior).
      * CI outward = +z → CI nodal forces in +z direction.
    The faces are pushed *toward* each other (CN down, CI up) —
    closing the crack.  Used here as the sign-flip baseline confirming
    the convention is symmetric in the magnitude sign.
    """
    _build_box_with_embedded_crack(g)
    P = 1.0e3
    with g.loads.pattern("CrackSuction"):
        g.loads.surface('Crack_normal',   magnitude=-P, normal=True)
        g.loads.surface('Crack_inverted', magnitude=-P, normal=True)
    fem = g.mesh.queries.get_fem_data(dim=3)

    cn_only = _entity_node_set('Crack_normal') - _entity_node_set('Crack_inverted')
    ci_only = _entity_node_set('Crack_inverted') - _entity_node_set('Crack_normal')
    cn_inner_total = _sum_loads_for_nodes(fem, cn_only)
    ci_inner_total = _sum_loads_for_nodes(fem, ci_only)
    assert cn_inner_total[2] < 0.0, \
        f"CN-only nodes should feel -z with -P, got {cn_inner_total[2]}"
    assert ci_inner_total[2] > 0.0, \
        f"CI-only nodes should feel +z with -P, got {ci_inner_total[2]}"


def test_crack_face_load_same_sign_gives_opening(g):
    """``face_load(magnitude=-P, normal=True)`` on BOTH crack faces.

    The face_load convention is ``f_total = magnitude * outward``, so
    with magnitude = -P:
      * CN: -P * (-z) = +PA in +z → CN gets +z force (opens crack).
      * CI: -P * (+z) = -PA in -z → CI gets -z force (opens crack).
    Same-sign magnitude on both produces opening — previously this
    required opposite signs to compensate for the silent connectivity-
    normal flip.

    The two patterns are kept separate so we can verify each entity's
    total independently (the rim nodes are shared between the two
    coincident face entities, so a global per-node sum collapses
    them to zero).
    """
    _build_box_with_embedded_crack(g)
    P = 1.0e3
    with g.loads.pattern("CN"):
        g.loads.face_load('Crack_normal',   magnitude=-P, normal=True)
    with g.loads.pattern("CI"):
        g.loads.face_load('Crack_inverted', magnitude=-P, normal=True)
    fem = g.mesh.queries.get_fem_data(dim=3)

    # Per the face_load convention, each entity's TOTAL = magnitude *
    # outward.  With magnitude = -P:
    #   CN total = -P * (-z) = (0, 0, +P)
    #   CI total = -P * (+z) = (0, 0, -P)
    cn_total = np.zeros(3, dtype=float)
    ci_total = np.zeros(3, dtype=float)
    for nl in fem.nodes.loads:
        if nl.force_xyz is None:
            continue
        f = np.asarray(nl.force_xyz, dtype=float)
        if nl.pattern == "CN":
            cn_total += f
        elif nl.pattern == "CI":
            ci_total += f
    np.testing.assert_allclose(cn_total, [0.0, 0.0, +P], atol=1e-6)
    np.testing.assert_allclose(ci_total, [0.0, 0.0, -P], atol=1e-6)


def test_crack_face_sp_same_sign_gives_opening(g):
    """``face_sp(magnitude=-u, normal=True)`` on BOTH crack faces
    prescribes opposite-sign translations: CN moves +z (into +z body),
    CI moves -z (into -z body).  Mechanically, the prescribed displacement
    pulls each face *into* its bonded body interior.

    Two patterns keep CN and CI's prescriptions on disjoint records
    so we can verify each side independently — shared rim nodes would
    otherwise carry conflicting prescriptions from the two coincident
    entities (which is itself an oddity of using face_sp on a crack;
    in production code you would constrain only one side at a time).
    """
    _build_box_with_embedded_crack(g)
    u = 1.0e-3
    with g.loads.pattern("CN"):
        g.loads.face_sp('Crack_normal',   magnitude=-u, normal=True)
    with g.loads.pattern("CI"):
        g.loads.face_sp('Crack_inverted', magnitude=-u, normal=True)
    fem = g.mesh.queries.get_fem_data(dim=3)

    # Per the face_sp convention, disp = magnitude * outward.  With
    # magnitude = -u:  CN disp = -u * (-z) = +u*z;   CI disp = -u * (+z) = -u*z.
    cn_z = [
        float(sp.value) for sp in fem.nodes.sp
        if int(sp.dof) == 3 and sp.pattern == "CN"
    ]
    ci_z = [
        float(sp.value) for sp in fem.nodes.sp
        if int(sp.dof) == 3 and sp.pattern == "CI"
    ]
    assert cn_z and ci_z
    np.testing.assert_allclose(cn_z, +u, atol=1e-12)
    np.testing.assert_allclose(ci_z, -u, atol=1e-12)


# =====================================================================
# (c) The helper itself: outward agrees with adjacent-tet centroid
# direction.
# =====================================================================

def test_helper_outward_for_cube_top(g):
    """``_face_outward_normals`` must return ``+z`` for every face element
    of the cube's +z face."""
    _build_unit_cube_with_top(g)
    faces = g.loads._target_faces('Top', source='auto')
    assert faces, "no face elements for Top PG"
    outwards = g.loads._face_outward_normals(faces)
    assert outwards is not None and len(outwards) == len(faces)
    for n in outwards:
        np.testing.assert_allclose(n, [0.0, 0.0, +1.0], atol=1e-9)


def test_helper_returns_none_when_no_volumes(g):
    """No 3-D entities → helper returns ``None`` so the caller falls
    back to the connectivity normal (preserving 2-D-only semantics)."""
    g.model.geometry.add_rectangle(0.0, 0.0, 0.0, 1.0, 1.0, label='plate')
    g.physical.add_surface('plate', name='Plate')
    g.mesh.sizing.set_global_size(0.5)
    g.mesh.generation.generate(dim=2)

    faces = g.loads._target_faces('Plate', source='auto')
    assert faces
    assert g.loads._face_outward_normals(faces) is None
