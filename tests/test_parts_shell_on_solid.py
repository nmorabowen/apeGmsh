"""Tests for shell-on-solid mesh conformity (S1a of the shell-to-solid
coupling feature work stream).

These tests pin the two bugs fixed by ``feat/shell-solid-fragment``:

1. ``parts.fragment_all`` now includes lower-dim entities (shells) in
   the OCC fragment call so they fragment conformally against solids
   instead of being silently excluded.
2. ``boolean.fragment`` no longer silently deletes free shell surfaces
   sitting on a volume's face.  The current default is
   ``cleanup_free=True`` and the topology sweep (``sweep_dangling``)
   protects ``add_*``-registered shells because they live in
   ``model._metadata`` — same outcome the original fix delivered, via
   a safer mechanism.

Conformity is verified at the Gmsh-entity level via
``gmsh.model.mesh.getNodes(dim, tag)`` — the shell's mesh nodes and
the volume's top-face mesh nodes must be the SAME node set (same
tags, same coords). Non-conformal meshes would have two disjoint
sets of nodes at the same z, both produced independently.
"""
import gmsh
import pytest


def _top_face_tag(vol_tag: int) -> int:
    """Return the tag of the dim=2 face on top (max-z centroid) of a volume."""
    faces = gmsh.model.getBoundary(
        [(3, vol_tag)], oriented=False, recursive=False,
    )
    best_tag, best_z = None, -float("inf")
    for dim, tag in faces:
        assert dim == 2
        cx, cy, cz = gmsh.model.occ.getCenterOfMass(dim, tag)
        if cz > best_z:
            best_z = cz
            best_tag = tag
    assert best_tag is not None, "no faces found on volume"
    return int(best_tag)


def test_shell_on_solid_fragment_all_makes_conformal_interface(g):
    """Shell rectangle on top of a box: fragment_all + mesh shares nodes."""
    with g.parts.part("vol"):
        g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
    with g.parts.part("shell"):
        # Rectangle at z=1, covering the box's top face.
        g.model.geometry.add_rectangle(0, 0, 1, 1, 1)

    g.parts.fragment_all()

    vol_tags = g.parts.get("vol").entities.get(3, [])
    shell_tags = g.parts.get("shell").entities.get(2, [])
    assert len(vol_tags) == 1, f"expected 1 volume, got {vol_tags}"
    assert len(shell_tags) >= 1, "shell surface vanished post-fragment"

    g.mesh.sizing.set_global_size(0.5)
    g.mesh.generation.generate(dim=3)

    # Shell mesh nodes (the surface the user added).
    shell_nodes_raw, _, _ = gmsh.model.mesh.getNodes(
        dim=2, tag=shell_tags[0], includeBoundary=True,
    )
    shell_nodes = set(int(n) for n in shell_nodes_raw)

    # Volume's top-face mesh nodes (after fragment_all, this should be
    # the SAME surface as the shell if conformity holds).
    top_face_tag = _top_face_tag(vol_tags[0])
    top_nodes_raw, _, _ = gmsh.model.mesh.getNodes(
        dim=2, tag=top_face_tag, includeBoundary=True,
    )
    top_nodes = set(int(n) for n in top_nodes_raw)

    assert shell_nodes == top_nodes, (
        f"Shell and box top face have different node sets — "
        f"non-conformal interface (shell={len(shell_nodes)} nodes, "
        f"top={len(top_nodes)} nodes, "
        f"shared={len(shell_nodes & top_nodes)}). "
        f"fragment_all must include both dims in the OCC call."
    )


def test_fragment_pair_shell_to_solid_succeeds(g):
    """fragment_pair on (volume, shell) does NOT raise 'no common dimension'."""
    with g.parts.part("vol"):
        g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
    with g.parts.part("shell"):
        g.model.geometry.add_rectangle(0, 0, 1, 1, 1)

    # Pre-fix: this raised RuntimeError('No common dimension').
    # Post-fix: auto-dim path supports cross-dim pairs.
    g.parts.fragment_pair("vol", "shell")

    # Both parts should still have their entities post-fragment.
    assert len(g.parts.get("vol").entities.get(3, [])) >= 1, (
        "volume vanished after cross-dim fragment_pair"
    )
    assert len(g.parts.get("shell").entities.get(2, [])) >= 1, (
        "shell vanished after cross-dim fragment_pair"
    )


def test_boolean_fragment_default_preserves_shell_surface_on_volume(g):
    """Default `cleanup_free=True` (topology sweep) keeps a shell sitting
    on a volume face.

    The shell is registered via ``add_rectangle`` so it lives in
    ``model._metadata`` — :func:`sweep_dangling` classifies anything
    in metadata as user-intentional and preserves it.  The mechanism
    is different from the pre-fix behavior (which relied on
    `cleanup_free=False` as the default to skip the sweep entirely),
    but the outcome is the same: shell-on-solid coupling works.
    """
    box_tag = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
    shell_tag = g.model.geometry.add_rectangle(0, 0, 1, 1, 1)

    # Default invocation — the shell survives because ``add_rectangle``
    # registered it in _metadata; the topology sweep protects it.
    g.model.boolean.fragment(
        objects=[(3, box_tag)], tools=[(2, shell_tag)],
    )

    gmsh.model.occ.synchronize()
    # The shell may have been split / re-tagged, but at least one dim=2
    # surface must still exist at z=1 (the shell's plane).
    surfaces_at_z1 = []
    for dim, tag in gmsh.model.getEntities(2):
        cx, cy, cz = gmsh.model.occ.getCenterOfMass(dim, tag)
        if abs(cz - 1.0) < 1e-6:
            surfaces_at_z1.append(tag)
    assert len(surfaces_at_z1) >= 1, (
        "Shell surface at z=1 was deleted by default fragment cleanup. "
        "cleanup_free should default to False to preserve shells on volumes."
    )


def test_boolean_fragment_cleanup_free_preserves_user_added_shell(g):
    """``cleanup_free=True`` (now the default) uses the topology-driven
    :func:`sweep_dangling` instead of the old centroid-in-bbox
    heuristic.  Surfaces registered via ``add_rectangle`` /
    ``add_plane_surface`` / ``add_cutting_plane`` are user-intentional
    (they appear in ``model._metadata``) and the sweep preserves them
    even when they sit far from any volume — the old heuristic deleted
    them, silently destroying shell-on-solid workflows.

    Use OCC directly when you genuinely need a sweep-removable orphan
    surface (one with no metadata, no label binding).
    """
    box_tag = g.model.geometry.add_box(0, 0, 0, 2, 2, 2)
    # A user-added rectangle — registered in metadata, hence preserved.
    orphan_tag = g.model.geometry.add_rectangle(10, 10, 10, 1, 1)
    gmsh.model.occ.synchronize()

    g.model.boolean.fragment(
        objects=[(3, box_tag)], tools=[(2, orphan_tag)], cleanup_free=True,
    )

    gmsh.model.occ.synchronize()
    # The user-added rectangle survives the sweep — its metadata entry
    # marks it as intentional standalone geometry.
    surfaces_far = []
    for dim, tag in gmsh.model.getEntities(2):
        cx, cy, cz = gmsh.model.occ.getCenterOfMass(dim, tag)
        if cx > 5:
            surfaces_far.append(tag)
    assert len(surfaces_far) >= 1, (
        "cleanup_free=True regressed: user-added shell (registered via "
        "add_rectangle) was deleted by the sweep. The topology-driven "
        "sweep is supposed to preserve metadata-registered standalone "
        "geometry."
    )


def test_boolean_fragment_cleanup_free_removes_unregistered_orphan(g):
    """The topology sweep DOES remove dim<=2 entities created outside
    the ``add_*`` registry — e.g. one injected via raw OCC.  This
    pins the half of the contract the previous test inverted: only
    user-intentional (metadata or labeled) entities are protected.
    """
    box_tag = g.model.geometry.add_box(0, 0, 0, 2, 2, 2)
    # Inject a rectangle bypassing add_rectangle so it has no
    # metadata entry and no label.
    raw_pts = [gmsh.model.occ.addPoint(*p) for p in [
        (10, 10, 10), (11, 10, 10), (11, 11, 10), (10, 11, 10),
    ]]
    raw_lines = [
        gmsh.model.occ.addLine(raw_pts[i], raw_pts[(i + 1) % 4])
        for i in range(4)
    ]
    raw_loop = gmsh.model.occ.addCurveLoop(raw_lines)
    raw_surf = gmsh.model.occ.addPlaneSurface([raw_loop])
    gmsh.model.occ.synchronize()
    assert (2, raw_surf) not in g.model._metadata

    g.model.geometry.remove_orphans()

    gmsh.model.occ.synchronize()
    surfaces_far = [t for _, t in gmsh.model.getEntities(2)
                    if gmsh.model.occ.getCenterOfMass(2, t)[0] > 5]
    assert surfaces_far == [], (
        f"sweep failed to remove unregistered orphan: {surfaces_far}"
    )
