"""Tests for shell-on-solid mesh conformity (S1a of the shell-to-solid
coupling feature work stream).

These tests pin the two bugs fixed by ``feat/shell-solid-fragment``:

1. ``parts.fragment_all`` now includes lower-dim entities (shells) in
   the OCC fragment call so they fragment conformally against solids
   instead of being silently excluded.
2. ``boolean.fragment(cleanup_free=False)`` (the new default) no
   longer silently deletes free shell surfaces sitting on a volume's
   face.

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
    """Default cleanup_free=False keeps shells when they sit on a volume face."""
    box_tag = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
    shell_tag = g.model.geometry.add_rectangle(0, 0, 1, 1, 1)

    # Default invocation — cleanup_free should default to False so the
    # shell isn't auto-deleted just because it's a free dim=2 sitting
    # on top of a volume.
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


def test_boolean_fragment_explicit_cleanup_free_still_works(g):
    """Opt-in cleanup_free=True remains available for callers that want it."""
    # Two volumes: one inside another. cleanup_free=True should still
    # remove free dim=2 surfaces that aren't bounding any volume.
    box_tag = g.model.geometry.add_box(0, 0, 0, 2, 2, 2)
    # A free orphan rectangle far away (no volume bounds it).
    orphan_tag = g.model.geometry.add_rectangle(10, 10, 10, 1, 1)
    gmsh.model.occ.synchronize()

    g.model.boolean.fragment(
        objects=[(3, box_tag)], tools=[(2, orphan_tag)], cleanup_free=True,
    )

    gmsh.model.occ.synchronize()
    # The orphan rectangle (far from any volume) should be cleaned up
    # when cleanup_free=True is explicitly requested.
    surfaces_far = []
    for dim, tag in gmsh.model.getEntities(2):
        cx, cy, cz = gmsh.model.occ.getCenterOfMass(dim, tag)
        if cx > 5:  # the orphan was at x=10
            surfaces_far.append(tag)
    assert len(surfaces_far) == 0, (
        f"cleanup_free=True should have removed orphan rectangle "
        f"(found {len(surfaces_far)} surfaces with x>5)"
    )
