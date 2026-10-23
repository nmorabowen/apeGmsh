"""``add_*(..., sync=False, label=...)`` keeps its label.

The label was a physical group on entities gmsh did not know yet, so it
was dropped without a word and a later lookup raised ``KeyError``.  It is
now held until the next synchronising apeGmsh call: ``g.model.sync()``,
a label query, a boolean / transform, ``generate`` or ``get_fem_data``.
"""
import gmsh
import pytest


def _storey(g, *, sync):
    """Two slab panels, two beams and a column per level, labels shared."""
    geo = g.model.geometry
    for k in (1, 2):
        z = float(k)
        for x0 in (0.0, 1.0):
            geo.add_rectangle(x0, 0.0, z, 1.0, 1.0, label=f"Slab_{k}", sync=sync)
        for y in (0.0, 1.0):
            geo.add_polyline([(0.0, y, z), (2.0, y, z)], label=f"Beam_{k}", sync=sync)
        geo.add_polyline([(0.0, 0.0, z - 1.0), (0.0, 0.0, z)], label=f"Col_{k}", sync=sync)
        geo.add_rectangle(0.0, 0.0, z - 1.0, 2.0, 1.0, plane="xz", label=f"Wall_{k}", sync=sync)


def _label_map(g):
    return {
        (name, d): sorted(g.labels.entities(name, dim=d))
        for name in g.labels.get_all()
        for d in (1, 2)
        if _has(g, name, d)
    }


def _has(g, name, d):
    try:
        return bool(g.labels.entities(name, dim=d))
    except (KeyError, ValueError):
        return False


def test_sync_false_label_survives_model_sync(g):
    tag = g.model.geometry.add_rectangle(0, 0, 0, 1, 1, label="A", sync=False)
    curves = g.model.geometry.add_polyline([(0, 0, 0), (0, 0, 1)], label="B", sync=False)
    g.model.sync()
    assert g.labels.entities("A", dim=2) == [tag]
    assert g.labels.entities("B", dim=1) == curves


def test_fast_pattern_matches_sync_true(tmp_path):
    """sync=False + label= + one g.model.sync() == sync=True, label for label."""
    from apeGmsh import apeGmsh

    maps = {}
    for sync in (True, False):
        with apeGmsh(model_name=f"nosync_{sync}", verbose=False,
                     save_to=str(tmp_path / f"m_{sync}.h5")) as s:
            _storey(s, sync=sync)
            if not sync:
                s.model.sync()
            maps[sync] = _label_map(s)
    assert maps[False] == maps[True]
    assert len(maps[True]) == 8


def test_label_query_binds_without_explicit_sync(g):
    tag = g.model.geometry.add_rectangle(0, 0, 0, 1, 1, label="A", sync=False)
    assert g.labels.has("A")
    assert g.labels.entities("A", dim=2) == [tag]


def test_pending_label_follows_a_boolean(g):
    """A boolean on raw tags binds the held labels first, then remaps them."""
    s = g.model.geometry.add_rectangle(0, 0, 0, 2, 1, label="Slab", sync=False)
    c = g.model.geometry.add_polyline([(1, 0, 0), (1, 1, 0)], label="Cut", sync=False)
    g.model.boolean.fragment([(2, s)], [(1, t) for t in c], dim=2)
    assert len(g.labels.entities("Slab", dim=2)) == 2


def test_generate_binds_pending_labels(g):
    g.model.geometry.add_rectangle(0, 0, 0, 1, 1, label="A", sync=False)
    g.mesh.sizing.set_global_size(0.5)
    g.mesh.generation.generate(dim=2)
    fem = g.mesh.queries.get_fem_data(dim=2)
    assert len(fem.elements.select(label="A").ids) > 0


def test_public_add_on_unsynced_tags_is_held(g):
    t = gmsh.model.occ.addRectangle(0, 0, 0, 1, 1)
    g.labels.add(2, [t], name="raw")
    g.model.sync()
    assert g.labels.entities("raw", dim=2) == [t]


def test_label_of_an_entity_removed_before_binding_warns(g):
    t = g.model.geometry.add_rectangle(0, 0, 0, 1, 1, label="gone", sync=False)
    gmsh.model.occ.remove([(2, t)], recursive=True)
    with pytest.warns(UserWarning, match="gone"):
        g.model.sync()
    assert not g.labels.has("gone")


def test_bound_label_survives_a_later_unsynced_batch(g):
    a = g.model.geometry.add_rectangle(0, 0, 0, 1, 1, label="A")
    b = g.model.geometry.add_rectangle(2, 0, 0, 1, 1, label="A", sync=False)
    c = g.model.geometry.add_rectangle(4, 0, 0, 1, 1, label="C", sync=False)
    g.model.sync()
    assert sorted(g.labels.entities("A", dim=2)) == sorted([a, b])
    assert g.labels.entities("C", dim=2) == [c]


@pytest.fixture
def nosync_part():
    """A Part whose only label was asked for with sync=False (built
    before the assembly session: one gmsh kernel at a time)."""
    from apeGmsh import Part

    p = Part("nosync_part")
    with p:
        p.model.geometry.add_rectangle(0, 0, 0, 1, 1, label="slab", sync=False)
    yield p
    p.cleanup()


def test_part_carries_a_label_made_with_sync_false(nosync_part, g):
    """A Part's sidecar anchors include labels held at ``end()``."""
    inst = g.parts.add(nosync_part, label="s1")
    assert "s1.slab" in inst.label_names
    assert len(g.labels.entities("s1.slab", dim=2)) == 1


# ---------------------------------------------------------------------
# Review fixes (PR #2494): session ownership, make_conformal, bind sites
# ---------------------------------------------------------------------

def test_held_label_does_not_leak_into_another_session():
    """A label held in s1 neither binds into s2's model nor is lost."""
    from apeGmsh import apeGmsh

    s1 = apeGmsh(model_name="leak_s1", verbose=False)
    s1.begin()
    try:
        tag = s1.model.geometry.add_rectangle(0, 0, 0, 1, 1, label="LEAK", sync=False)
        with apeGmsh(model_name="leak_s2", verbose=False) as s2:
            s2.model.geometry.add_box(0, 0, 0, 1, 1, 1)
            s2.mesh.sizing.set_global_size(0.5)
            s2.mesh.generation.generate(dim=3)
            assert "LEAK" not in s2.labels.get_all()
            fem = s2.mesh.queries.get_fem_data(dim=3)
            with pytest.raises(KeyError, match="LEAK"):
                fem.elements.select(label="LEAK")
        # s2's end() leaves its model current; s1 binds in its own model.
        gmsh.model.setCurrent("leak_s1")
        s1.model.sync()
        assert s1.labels.entities("LEAK", dim=2) == [tag]
    finally:
        s1.end()


def test_label_still_held_when_its_model_is_not_current_warns_at_end():
    """end() cannot bind into a foreign model, so it says the label is lost."""
    from apeGmsh import apeGmsh

    s1 = apeGmsh(model_name="lost_s1", verbose=False)
    s1.begin()
    try:
        s1.model.geometry.add_rectangle(0, 0, 0, 1, 1, label="LOST", sync=False)
        with apeGmsh(model_name="lost_s2", verbose=False) as s2:
            s2.model.geometry.add_box(0, 0, 0, 1, 1, 1)
    finally:
        with pytest.warns(UserWarning, match="LOST"):
            s1.end()


@pytest.mark.parametrize("sync", [True, False])
def test_make_conformal_fragments_entities_made_with_sync_false(g, sync):
    """make_conformal binds held labels before it reads the entities, so
    sync=False fragments exactly like sync=True."""
    g.model.geometry.add_rectangle(0, 0, 0, 2, 1, label="A", sync=sync)
    g.model.geometry.add_polyline([(1, 0, 0), (1, 1, 0)], label="B", sync=sync)
    g.model.queries.make_conformal()
    assert len(g.labels.entities("A", dim=2)) == 2


def test_end_binds_held_labels(g):
    """A session's end() creates its held labels (here a session that
    writes no artifact, so nothing else on the end() path binds them)."""
    from apeGmsh import apeGmsh

    inner = apeGmsh(model_name="end_binds", verbose=False, _artifacts=False)
    inner.begin()
    tag = inner.model.geometry.add_rectangle(0, 0, 0, 1, 1, label="E", sync=False)
    inner.end()
    # g keeps the gmsh runtime alive; the inner model is still current.
    try:
        assert gmsh.model.getCurrent() == "end_binds"
        named = {
            gmsh.model.getPhysicalName(d, t): list(gmsh.model.getEntitiesForPhysicalGroup(d, t))
            for d, t in gmsh.model.getPhysicalGroups(2)
        }
    finally:
        gmsh.model.setCurrent("test")
    assert named == {"_label:E": [tag]}


def test_get_fem_data_binds_held_labels_without_generate(g):
    """get_fem_data binds on its own (the mesh here is a raw gmsh one)."""
    g.model.geometry.add_rectangle(0, 0, 0, 1, 1, label="A", sync=False)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 0.5)
    gmsh.model.mesh.generate(2)
    fem = g.mesh.queries.get_fem_data(dim=2)
    assert len(fem.elements.select(label="A").ids) > 0


def test_remove_binds_held_labels_first(g):
    g.model.geometry.add_rectangle(0, 0, 0, 1, 1, label="A", sync=False)
    g.labels.remove("A")
    g.model.sync()
    assert not g.labels.has("A")


def test_rename_binds_held_labels_first(g):
    tag = g.model.geometry.add_rectangle(0, 0, 0, 1, 1, label="A", sync=False)
    g.labels.rename("A", "B")
    assert g.labels.entities("B", dim=2) == [tag]
    assert not g.labels.has("A")


def test_a_failed_bind_keeps_the_labels_not_yet_bound(g, monkeypatch):
    """An error while binding one held label loses no other."""
    from apeGmsh.core.Labels import Labels

    a = g.model.geometry.add_rectangle(0, 0, 0, 1, 1, label="A", sync=False)
    b = g.model.geometry.add_rectangle(2, 0, 0, 1, 1, label="B", sync=False)
    real = Labels._add_bound

    def failing(self, dim, tags, name, **kw):
        if name == "A":
            raise RuntimeError("bind failed")
        return real(self, dim, tags, name, **kw)

    monkeypatch.setattr(Labels, "_add_bound", failing)
    with pytest.raises(RuntimeError, match="bind failed"):
        g.model.sync()
    monkeypatch.undo()
    g.model.sync()
    assert g.labels.entities("A", dim=2) == [a]
    assert g.labels.entities("B", dim=2) == [b]