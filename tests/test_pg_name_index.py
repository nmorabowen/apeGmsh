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
    # #2594: gmsh keeps a name the group already has, so set_name was a
    # silent no-op.  It renames in place: same tag, same entities.
    assert g.physical.set_name(1, t, "new") is g.physical
    assert g.physical.get_tag(1, "old") is None
    assert g.physical.get_tag(1, "new") == t
    assert g.physical.get_name(1, t) == "new"
    assert list(g.physical.get_entities(1, t)) == [a]
    assert g.physical.get_all() == [(1, t)]
    # The freed name is free again, at any dim.
    p = gmsh.model.getBoundary([(1, b)], oriented=False)[0][1]
    assert g.physical.add(0, [p], name="old") != t
    g.physical.remove([(0, g.physical.get_tag(0, "old"))])
    g.physical.remove([(1, t)])
    assert g.physical.get_tag(1, "new") is None
    t2 = g.physical.add(1, [b], name="kept")
    g.physical.remove_name("kept")
    assert g.physical.get_tag(1, "kept") is None
    assert g.physical.get_tag(1, "") == t2


def test_set_name_onto_a_held_name_raises_and_changes_nothing(g):
    # #2588: gmsh keeps one group per (dim, name) and drops the call
    # silently, so both groups kept their names and nothing was raised.
    a, b, c = _lines(3)
    t_b = g.physical.add(1, [a], name="B")
    t_c = g.physical.add(1, [b], name="C")
    t_u = g.physical.add(1, [c])                  # unnamed
    for t, before in ((t_b, "B"), (t_u, "")):
        with pytest.raises(ValueError, match=r"'C' already exists at dim=1"):
            g.physical.set_name(1, t, "C")
        assert g.physical.get_name(1, t) == before
    assert g.physical.get_name(1, t_c) == "C"
    assert g.physical.get_tag(1, "C") == t_c
    assert g.physical.get_tag(1, "B") == t_b


def test_set_name_to_its_own_name_or_empty_is_a_no_op(g):
    a, b, c = _lines(3)
    t = g.physical.add(1, [a], name="C")
    assert g.physical.set_name(1, t, "C") is g.physical
    assert g.physical.get_tag(1, "C") == t
    # "" is no name: two unnamed groups do not clash.
    u1 = g.physical.add(1, [b])
    g.physical.add(1, [c])
    g.physical.set_name(1, u1, "")
    assert g.physical.get_name(1, u1) == ""


def test_set_name_refuses_a_name_held_at_another_dim(g):
    # #2594: add() refuses a name held at another dim; set_name let it
    # through, leaving _named("C") == {1: .., 0: ..}.
    a, b = _lines(2)
    t_c = g.physical.add(1, [a], name="C")
    p0, p1 = (gmsh.model.getBoundary([(1, b)], oriented=False)[i][1]
              for i in range(2))
    t_u = g.physical.add(0, [p0])                 # unnamed
    t_n = g.physical.add(0, [p1], name="P")       # named
    for t, before in ((t_u, ""), (t_n, "P")):
        with pytest.raises(ValueError, match=r"'C' already exists at dim=1"):
            g.physical.set_name(0, t, "C")
        assert g.physical.get_name(0, t) == before
    assert g.physical._named("C") == {1: t_c}
    assert g.physical.get_tag(0, "P") == t_n


def test_set_name_refuses_an_empty_name_for_a_named_group(g):
    # "" on a named group was a silent no-op; unnaming is remove_name's.
    a, _ = _lines(2)
    t = g.physical.add(1, [a], name="keep")
    with pytest.raises(ValueError, match=r"remove_name\('keep'\)"):
        g.physical.set_name(1, t, "")
    assert g.physical.get_tag(1, "keep") == t


def test_set_name_refuses_a_label_group(g):
    # A label's group renamed through g.physical was a silent no-op; a
    # real rename would take the group away from g.labels.
    a, _ = _lines(2)
    g.labels.add(1, [a], name="lab")
    (lt,) = [t for d, t in gmsh.model.getPhysicalGroups(1)
             if gmsh.model.getPhysicalName(d, t) == "_label:lab"]
    with pytest.raises(ValueError, match=r"is the label 'lab'.*labels\.rename"):
        g.physical.set_name(1, lt, "user")
    assert gmsh.model.getPhysicalName(1, lt) == "_label:lab"
    assert g.labels.entities("lab", dim=1) == [a]
    assert g.physical.get_tag(1, "user") is None


def test_rename_keeps_the_old_name_on_a_raw_other_dim_group(g):
    # removePhysicalName drops a name at every dim.  A raw-gmsh group at
    # another dim that shares the old name keeps it.
    a, _ = _lines(2)
    t = g.physical.add(1, [a], name="old")
    p = gmsh.model.getBoundary([(1, a)], oriented=False)[0][1]
    t0 = gmsh.model.addPhysicalGroup(0, [p])
    gmsh.model.setPhysicalName(0, t0, "old")
    g.physical.set_name(1, t, "new")
    assert gmsh.model.getPhysicalName(0, t0) == "old"
    assert g.physical._named("old") == {0: t0}
    assert g.physical._named("new") == {1: t}


def test_renamed_group_reaches_fem_and_the_h5_round_trip(g, tmp_path):
    from apeGmsh.mesh.FEMData import FEMData
    a, b = _lines(2)
    t = g.physical.add(1, [a, b], name="old")
    g.physical.set_name(1, t, "new")
    g.mesh.generation.generate(dim=1)
    fem = g.mesh.queries.get_fem_data(dim=1)
    out = tmp_path / "renamed.h5"
    fem.to_h5(str(out))
    for f in (fem, FEMData.from_h5(str(out))):
        assert "old" not in f.nodes.physical.names()
        assert f.nodes.physical.get_tag(1, "new") == t
        assert f.elements.physical.get_tag(1, "new") == t
        assert len(f.elements.physical.element_ids("new")) == len(
            fem.elements.physical.element_ids((1, t)))


def test_set_name_refuses_the_label_prefix(g):
    a, b = _lines(2)
    g.labels.add(1, [a], name="lab")
    t = g.physical.add(1, [b], name="user")
    for name in ("_label:lab", "_label:free"):
        with pytest.raises(ValueError, match="reserved for g.labels"):
            g.physical.set_name(1, t, name)
    assert g.physical.get_name(1, t) == "user"
    assert g.labels.entities("lab", dim=1) == [a]


def test_remove_all_is_reflected(g):
    # In plain gmsh 4.15.2 the argument-less removePhysicalGroups()
    # corrupts the heap at finalize (0xc0000374) once any targeted
    # removePhysicalGroups([(dim, tag)]) ran in the model; renaming
    # (removePhysicalName + setPhysicalName) is harmless.  remove_all()
    # lists the groups instead (#2608); the kill is pinned out of
    # process in tests/test_physical_upsert_remove_all.py.
    a, _ = _lines(2)
    g.physical.add(1, [a], name="all")
    g.physical.remove_all()
    assert g.physical.get_tag(1, "all") is None


def test_no_argument_less_remove_physical_groups(g, monkeypatch):
    # The subprocess pin in test_physical_upsert_remove_all.py needs
    # the heap corruption to kill the child, which Linux may not do.
    # This holds the contract itself on every OS: neither remove_all()
    # nor remove([]) ever hands gmsh no dimTags or an empty list, which
    # gmsh reads as "remove every group".
    calls: list[tuple[tuple, dict]] = []
    real = gmsh.model.removePhysicalGroups

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        sent = kwargs.get("dimTags", args[0] if args else None)
        # An empty or missing dimTags never reaches gmsh: after the
        # targeted remove below it would corrupt the heap and kill
        # the shared pytest process at the fixture's session.end().
        assert sent, f"argument-less removePhysicalGroups: {(args, kwargs)}"
        return real(*args, **kwargs)

    monkeypatch.setattr(gmsh.model, "removePhysicalGroups", spy)
    a, b = _lines(2)
    t = g.physical.add(1, [a], name="keep")
    t2 = g.physical.add(1, [b], name="other")
    # The empty selection is checked before any targeted removal: on
    # Windows that order is what lets a wrong implementation fail
    # this assertion instead of killing the process at teardown.
    g.physical.remove([])
    assert calls == [], calls
    assert gmsh.model.getPhysicalGroups() == [(1, t), (1, t2)]
    g.physical.remove([(1, t2)])
    assert g.physical.get_tag(1, "keep") == t
    g.physical.remove_all()
    assert gmsh.model.getPhysicalGroups() == []
    g.physical.remove_all()          # nothing left: no gmsh call at all
    sent = [kwargs.get("dimTags", args[0] if args else None)
            for args, kwargs in calls]
    assert sent == [[(1, t2)], [(1, t)]], sent


def test_remove_all_drops_the_names_too(g):
    # The argument-less form kept every name, so a group created later
    # at a freed tag inherited it and its own setPhysicalName was
    # dropped.  Listing the groups removes the names with them.
    a, b = _lines(2)
    t = g.physical.add(1, [a], name="all")
    g.physical.remove_all()
    assert gmsh.model.getPhysicalName(1, t) == ""
    t2 = g.physical.add(1, [b], name="other", tag=t)
    assert t2 == t
    assert g.physical.get_name(1, t2) == "other"
    assert g.physical.get_tag(1, "all") is None
    assert g.physical.get_tag(1, "other") == t2


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
    # Same stored name once prefixed: "_label:A" is "A".
    g.labels.rename("A", "_label:A")
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


def test_remap_replaces_a_group_whose_set_or_name_changed(g):
    # A group still held at its old tag is kept only when both its
    # entity set and its name match the remap; the same entity count
    # or the same entities under another name are not enough.
    from apeGmsh.core.Labels import (
        remap_physical_groups, snapshot_physical_groups,
    )
    a, b, c = _lines(3)
    ta = gmsh.model.addPhysicalGroup(1, [a])
    gmsh.model.setPhysicalName(1, ta, "SET")
    tc = gmsh.model.addPhysicalGroup(1, [c])
    gmsh.model.setPhysicalName(1, tc, "NAME")
    snap = snapshot_physical_groups()
    # Same count, other entity: the boolean maps a -> b.
    # Same entities, other name: a raw rename behind the snapshot.
    gmsh.model.removePhysicalName("NAME")
    gmsh.model.setPhysicalName(1, tc, "OTHER")
    remap_physical_groups(snap, [(1, a)], [[(1, b)]])
    by_name = {
        gmsh.model.getPhysicalName(1, t): sorted(
            gmsh.model.getEntitiesForPhysicalGroup(1, t))
        for _, t in gmsh.model.getPhysicalGroups(1)
    }
    assert by_name == {"SET": [b], "NAME": [c]}

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
