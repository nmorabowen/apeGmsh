"""Physical-group name index: O(1) lookups that never answer stale.

``g.physical.get_tag`` and ``g.labels.entities`` / ``has`` found a group
by listing every group and reading every name, so G named groups cost
O(G^2): 12.4 s of a 22.7 s mesh build for 2,622 groups, and 214.6 s
for a labelled fragment that raw gmsh does in 4.1 s, in the YOO tower.
Both composites now read through ``core/_pg_name_index.PGNameIndex``,
which must keep their semantics and stay correct after raw
``gmsh.model.*`` writes, which apeGmsh cannot see.
"""
from __future__ import annotations

import time

import gmsh
import pytest


def _lines(n: int) -> list[int]:
    occ = gmsh.model.occ
    pts = [occ.addPoint(float(i), 0.0, 0.0) for i in range(n + 1)]
    lines = [occ.addLine(pts[i], pts[i + 1]) for i in range(n)]
    occ.synchronize()
    return lines


# ---------------------------------------------------------------------
# Behaviour
# ---------------------------------------------------------------------


def test_get_tag_unknown_name_is_none(g):
    _lines(2)
    assert g.physical.get_tag(1, "nope") is None
    assert g.physical.get_tag(1, "") is None


def test_add_then_get_tag_and_upsert(g):
    a, b = _lines(2)
    tag = g.physical.add(1, [a], name="beam")
    assert g.physical.get_tag(1, "beam") == tag
    assert g.physical.add(1, [b], name="beam") == tag
    assert sorted(g.physical.get_entities(1, tag)) == sorted([a, b])


def test_cross_dimension_clash_still_raises(g):
    a, _ = _lines(2)
    g.physical.add(1, [a], name="X")
    p = gmsh.model.getBoundary([(1, a)], oriented=False)[0][1]
    with pytest.raises(ValueError, match="already exists at dim=1"):
        g.physical.add(0, [p], name="X")


def test_label_pgs_are_skipped(g):
    a, _ = _lines(2)
    g.labels.add(1, [a], name="lab")
    assert g.physical.get_tag(1, "lab") is None
    assert g.physical.get_tag(1, "_label:lab") is None


def test_rename_and_remove_are_reflected(g):
    a, b = _lines(2)
    t = g.physical.add(1, [a], name="old")
    # gmsh keeps a name the group already has: set_name alone is a no-op.
    g.physical.set_name(1, t, "new")
    assert g.physical.get_tag(1, "old") == t
    assert g.physical.get_tag(1, "new") is None
    g.physical.remove_name("old").set_name(1, t, "new")
    assert g.physical.get_tag(1, "old") is None
    assert g.physical.get_tag(1, "new") == t
    g.physical.remove([(1, t)])
    assert g.physical.get_tag(1, "new") is None
    t2 = g.physical.add(1, [b], name="kept")
    g.physical.remove_name("kept")
    assert g.physical.get_tag(1, "kept") is None
    assert g.physical.get_tag(1, "") == t2


def test_remove_all_is_reflected(g):
    # Kept apart from remove_name: in plain gmsh 4.15, renaming,
    # removePhysicalName and then removePhysicalGroups() corrupts the
    # heap at finalize (0xc0000374), with or without apeGmsh.
    a, _ = _lines(2)
    g.physical.add(1, [a], name="all")
    g.physical.remove_all()
    assert g.physical.get_tag(1, "all") is None


def test_raw_gmsh_create_and_rename_are_found(g):
    a, b, c = _lines(3)
    g.physical.add(1, [a], name="seen")          # index is warm
    t = gmsh.model.addPhysicalGroup(1, [b])
    gmsh.model.setPhysicalName(1, t, "raw")
    assert g.physical.get_tag(1, "raw") == t
    gmsh.model.removePhysicalName("raw")
    gmsh.model.setPhysicalName(1, t, "raw2")
    assert g.physical.get_tag(1, "raw") is None
    assert g.physical.get_tag(1, "raw2") == t
    # A raw-created name still blocks a clash at another dim.
    p = gmsh.model.getBoundary([(1, c)], oriented=False)[0][1]
    with pytest.raises(ValueError, match="already exists at dim=1"):
        g.physical.add(0, [p], name="raw2")


def test_raw_remove_and_tag_reuse_do_not_answer_stale(g):
    a, b = _lines(2)
    t = g.physical.add(1, [a], name="gone")
    gmsh.model.removePhysicalGroups([(1, t)])
    assert g.physical.get_tag(1, "gone") is None
    # The freed tag is reused for a different name.
    t2 = gmsh.model.addPhysicalGroup(1, [b], tag=t)
    gmsh.model.setPhysicalName(1, t2, "other")
    assert g.physical.get_tag(1, "gone") is None
    assert g.physical.get_tag(1, "other") == t


def test_raw_create_at_second_dim_is_seen_on_a_hit(g):
    a, _ = _lines(2)
    g.physical.add(1, [a], name="M")             # indexed at dim 1
    p = gmsh.model.getBoundary([(1, a)], oriented=False)[0][1]
    t0 = gmsh.model.addPhysicalGroup(0, [p])
    gmsh.model.setPhysicalName(0, t0, "M")
    with pytest.raises(ValueError, match="multiple dimensions"):
        g.physical.entities("M")
    assert g.physical.get_tag(0, "M") == t0


def test_entities_by_name_uses_the_index(g):
    a, _ = _lines(2)
    g.physical.add(1, [a], name="E")
    assert g.physical.entities("E") == [a]
    with pytest.raises(KeyError):
        g.physical.entities("missing")


# ---------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------


def test_named_add_cost_tracks_raw_gmsh(g):
    """G named adds cost a constant multiple of raw gmsh, not O(G^2).

    Raw ``addPhysicalGroup`` is itself ~0.25 ms per call on a desktop,
    so an absolute budget would measure the machine; the ratio to raw
    gmsh on the same model measures apeGmsh.  The old scan ran this at
    hundreds of times raw.
    """
    n = 4000
    lines = _lines(2 * n)
    raw, ours = lines[:n], lines[n:]

    t0 = time.perf_counter()
    for i, ln in enumerate(raw):
        t = gmsh.model.addPhysicalGroup(1, [ln])
        gmsh.model.setPhysicalName(1, t, f"R{i}")
    t_raw = time.perf_counter() - t0

    t0 = time.perf_counter()
    for i, ln in enumerate(ours):
        g.physical.add_curve([ln], name=f"FR_{i}")
    t_ours = time.perf_counter() - t0

    assert g.physical.get_tag(1, f"FR_{n - 1}") is not None
    assert t_ours < 3.0 * t_raw + 0.5, (t_ours, t_raw)
    # end() extracts every group for model.h5, which is itself O(G * E)
    # in gmsh; clearing them keeps this test about add().
    g.physical.remove_all()


# ---------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------


def test_label_lookup_add_merge_rename_remove(g):
    a, b = _lines(2)
    g.labels.add(1, [a], name="L")
    assert g.labels.has("L") and g.labels.has("L", dim=1)
    assert not g.labels.has("L", dim=0)
    assert not g.labels.has("nope")
    with pytest.warns(UserWarning, match="Merging"):
        g.labels.add(1, [b], name="L")
    assert sorted(g.labels.entities("L")) == sorted([a, b])
    g.labels.rename("L", "M")
    assert not g.labels.has("L")
    assert sorted(g.labels.entities("M", dim=1)) == sorted([a, b])
    g.labels.remove("M")
    assert not g.labels.has("M")
    with pytest.raises(KeyError, match="Available labels"):
        g.labels.entities("M")


def test_label_rename_to_its_own_name_returns(g):
    # The rename loop re-finds a group by its old name, so renaming
    # to the same name used to find the group it had just recreated,
    # forever.
    a, _ = _lines(2)
    g.labels.add(1, [a], name="A")
    g.labels.rename("A", "A")
    g.labels.rename("A", "A", dim=1)
    assert g.labels.entities("A") == [a]
    with pytest.raises(KeyError, match="no label 'A' found at dim=0"):
        g.labels.rename("A", "A", dim=0)
    with pytest.raises(KeyError, match="no label 'B' found"):
        g.labels.rename("B", "B")

def test_label_multi_dim_still_raises(g):
    a, _ = _lines(2)
    p = gmsh.model.getBoundary([(1, a)], oriented=False)[0][1]
    g.labels.add(1, [a], name="X")
    with pytest.warns(UserWarning, match="also being created"):
        g.labels.add(0, [p], name="X")
    with pytest.raises(ValueError, match="multiple dimensions"):
        g.labels.entities("X")
    with pytest.raises(ValueError, match="multiple dimensions"):
        g.labels.has("X")
    assert g.labels.entities("X", dim=0) == [p]


def test_label_raw_gmsh_edits_are_seen(g):
    a, b = _lines(2)
    g.labels.add(1, [a], name="seen")           # index is warm
    t = gmsh.model.addPhysicalGroup(1, [b])
    gmsh.model.setPhysicalName(1, t, "_label:raw")
    assert g.labels.entities("raw") == [b]
    gmsh.model.removePhysicalName("_label:raw")
    gmsh.model.setPhysicalName(1, t, "_label:raw2")
    assert not g.labels.has("raw")
    assert g.labels.entities("raw2") == [b]
    gmsh.model.removePhysicalGroups([(1, t)])
    assert not g.labels.has("raw2")
    # The label namespace and the PG namespace stay apart.
    assert g.physical.get_tag(1, "seen") is None
    assert g.physical.get_tag(1, "_label:seen") is None


def test_label_index_follows_a_fragment_remap(g):
    geo = g.model.geometry
    geo.add_rectangle(0, 0, 0, 2.0, 1.0, label="wall")
    p0 = geo.add_point(1.0, 0.5, -1.0)
    p1 = geo.add_point(1.0, 0.5, 1.0)
    geo.add_line(p0, p1, label="frame")
    assert g.labels.has("wall") and g.labels.has("frame")
    g.model.boolean.fragment(["wall"], ["frame"], dim=2)
    # The remap recreated every label PG behind the index's back.
    assert g.labels.entities("wall", dim=2)
    assert len(g.labels.entities("frame", dim=1)) == 2


def test_fragment_keeps_a_group_whose_entities_survive(g):
    # synchronize() keeps a group on surviving entities, so the remap
    # leaves it (and its tag) alone and recreates only changed groups.
    geo = g.model.geometry
    geo.add_rectangle(0, 0, 0, 2.0, 1.0, label="wall")
    p0 = geo.add_point(1.0, 0.5, -1.0)
    p1 = geo.add_point(1.0, 0.5, 1.0)
    geo.add_line(p0, p1, label="frame")
    geo.add_point(5.0, 5.0, 0.0, label="far")
    far_pg = g.labels._find("far")[0]
    t = gmsh.model.addPhysicalGroup(0, g.labels.entities("far"))
    gmsh.model.setPhysicalName(0, t, "far_pg")
    g.model.boolean.fragment(["wall"], ["frame", "far"], dim=2)
    assert g.labels._find("far") == {0: far_pg}
    assert g.physical.get_tag(0, "far_pg") == t
    assert len(g.labels.entities("frame", dim=1)) == 2
    tags = [t for d, t in gmsh.model.getPhysicalGroups(1)]
    assert len(tags) == 1                       # old frame group removed
    assert gmsh.model.getPhysicalName(1, tags[0]) == "_label:frame"


def test_list_resolution_reads_groups_once_and_stays_fresh(g, monkeypatch):
    from apeGmsh.core._helpers import resolve_to_dimtags
    a, b, c = _lines(3)
    g.labels.add(1, [a], name="A")
    g.labels.add(1, [b], name="B")
    # Raw edits the index has not seen: B moved to c under a new tag.
    old = g.labels._find("B")[1]
    gmsh.model.removePhysicalGroups([(1, old)])
    t = gmsh.model.addPhysicalGroup(1, [c])
    gmsh.model.setPhysicalName(1, t, "_label:B")
    per_group: list = []
    real = gmsh.model.getEntitiesForPhysicalGroup
    monkeypatch.setattr(
        gmsh.model, "getEntitiesForPhysicalGroup",
        lambda *a: per_group.append(a) or real(*a),
    )
    got = resolve_to_dimtags(["A", "B", (1, a)], default_dim=1, session=g)
    assert per_group == []                      # one bulk read instead
    assert got == [(1, a), (1, c), (1, a)]
    assert g.labels._bulk is None


def _labelled_fragment_case(n_walls: int, per_wall: int):
    """Walls in the xz plane, each crossed by ``per_wall`` frames."""
    occ = gmsh.model.occ
    walls, frames = [], []
    for i in range(n_walls):
        walls.append(occ.addRectangle(2.0 * i, 0.0, 0.0, 1.0, 3.0))
        occ.rotate([(2, walls[-1])], 2.0 * i, 0, 0, 1, 0, 0, 1.5707963267948966)
        for k in range(per_wall):
            x = 2.0 * i + 0.1 + 0.8 * k / per_wall
            a = occ.addPoint(x, -1.0, 1.0)
            b = occ.addPoint(x, 1.0, 1.0)
            frames.append(occ.addLine(a, b))
    occ.synchronize()
    return walls, frames


def test_labelled_fragment_cost_tracks_raw_gmsh(g):
    """A fragment over L labelled entities costs a multiple of raw gmsh.

    The old label lookups made it O(L^2): 103 s against 1.7 s raw for
    3,916 labels.  Then, read one group at a time and recreating every
    group, it held 2.4x raw here; reading all groups in one call and
    recreating only changed ones holds 1.55x.  Raw gmsh runs first on
    its own model, then the same geometry is labelled and fragmented
    through apeGmsh.
    """
    n_walls, per_wall = 150, 6

    gmsh.model.add("raw_reference")
    walls, frames = _labelled_fragment_case(n_walls, per_wall)
    t0 = time.perf_counter()
    gmsh.model.occ.fragment(
        [(2, w) for w in walls], [(1, f) for f in frames])
    gmsh.model.occ.synchronize()
    t_raw = time.perf_counter() - t0
    gmsh.model.remove()
    gmsh.model.setCurrent("test")

    walls, frames = _labelled_fragment_case(n_walls, per_wall)
    for i, w in enumerate(walls):
        g.labels.add(2, [w], name=f"PN_{i}")
    for j, f in enumerate(frames):
        g.labels.add(1, [f], name=f"FR_{j}")

    t0 = time.perf_counter()
    g.model.boolean.fragment(
        [f"PN_{i}" for i in range(n_walls)],
        [f"FR_{j}" for j in range(len(frames))],
        dim=2,
    )
    t_ours = time.perf_counter() - t0

    assert len(g.labels.entities(f"FR_{len(frames) - 1}", dim=1)) == 2
    assert t_ours < 2.0 * t_raw + 0.1, (t_ours, t_raw)


def test_remap_without_the_bulk_gmsh_call(g, monkeypatch):
    # A gmsh build without getPhysicalGroupsEntities reads group by group.
    monkeypatch.delattr(gmsh.model, "getPhysicalGroupsEntities")
    test_label_index_follows_a_fragment_remap(g)
