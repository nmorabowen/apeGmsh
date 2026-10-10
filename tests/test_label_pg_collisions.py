"""Silent-empty and name-collision siblings of #1332 / #1335 (#1364).

Five sites, one oracle each:

1. ``g.labels.add(dim, [], name)`` raises (it registered nothing).
2. ``g.physical.add(..., name="_label:X")`` and
   ``promote_to_physical(..., pg_name="_label:X")`` refuse the reserved
   prefix before any gmsh write.
3. DXF layer PGs go through ``g.physical.add``: a second load of the
   same layer merges into its PG instead of leaving an unnamed one.
4. The node-target error lists group *names*, not the ``(dim, tag)``
   keys of the snapshot dict (the anchor sibling is covered by
   ``tests/assembly/test_anchor.py``).
5. ``promote_to_physical`` into a name held at another dim names the
   ``pg_name=`` knob.

Every refusal is checked to leave the PG count unchanged: the oracle
for "refused before registration" is ``gmsh.model.getPhysicalGroups()``
before and after.
"""
from __future__ import annotations

import ast
import re

import gmsh
import pytest

from apeGmsh._kernel._label_prefix import LABEL_PREFIX


def _pgs() -> list[tuple[int, int, str]]:
    return [
        (d, t, gmsh.model.getPhysicalName(d, t))
        for d, t in gmsh.model.getPhysicalGroups()
    ]


def _unit_line(g) -> None:
    geo = g.model.geometry
    geo.add_point(0, 0, 0, label="a")
    geo.add_point(1, 0, 0, label="b")
    geo.add_line("a", "b", label="beam")


# ---------------------------------------------------------------------
# 1. Labels.add with no entities / no name
# ---------------------------------------------------------------------


def test_labels_add_empty_tags_raises_and_registers_nothing(g):
    _unit_line(g)
    before = _pgs()
    with pytest.raises(ValueError, match=r"tags=\[\].*'ghost'"):
        g.labels.add(1, [], name="ghost")
    assert _pgs() == before
    assert not g.labels.has("ghost")


@pytest.mark.parametrize("name", ["", LABEL_PREFIX])
def test_labels_add_empty_name_raises(g, name):
    _unit_line(g)
    curve = g.labels.entities("beam")
    before = _pgs()
    with pytest.raises(ValueError, match="non-empty name"):
        g.labels.add(1, curve, name=name)
    assert _pgs() == before


def test_labels_add_still_creates_prefixed_pg(g):
    """The label system's own path keeps using the prefix."""
    _unit_line(g)
    curve = g.labels.entities("beam")
    g.labels.add(1, curve, name="span")
    assert (1, f"{LABEL_PREFIX}span") in {(d, n) for d, _, n in _pgs()}
    assert g.labels.entities("span") == curve


# ---------------------------------------------------------------------
# 2. reserved prefix in user PG names
# ---------------------------------------------------------------------


def test_physical_add_reserved_prefix_raises_before_gmsh(g):
    _unit_line(g)
    curve = g.labels.entities("beam")
    before = _pgs()
    with pytest.raises(ValueError, match="reserved"):
        g.physical.add(1, curve, name=f"{LABEL_PREFIX}beam")
    # Before #1364 gmsh refused the duplicate name and an unnamed PG
    # was left behind.
    assert _pgs() == before
    assert all(n for _, _, n in _pgs())


def test_promote_reserved_prefix_names_pg_name(g):
    _unit_line(g)
    before = _pgs()
    with pytest.raises(ValueError, match=r"pg_name="):
        g.labels.promote_to_physical("beam", pg_name=f"{LABEL_PREFIX}beam")
    assert _pgs() == before


# ---------------------------------------------------------------------
# 3. DXF layer PGs
# ---------------------------------------------------------------------


def _write_dxf(path, layer: str, x0: float) -> None:
    ezdxf = pytest.importorskip("ezdxf")
    doc = ezdxf.new()
    # ezdxf refuses ':' in a new layer name, but reads one from a file;
    # write a placeholder and substitute it in the text.
    doc.layers.add("LAYERX")
    doc.modelspace().add_line((x0, 0, 0), (x0 + 1.0, 0, 0),
                              dxfattribs={"layer": "LAYERX"})
    doc.saveas(str(path))
    path.write_text(path.read_text().replace("LAYERX", layer))


def test_dxf_layer_pg_merges_on_second_load(g, tmp_path):
    """Two DXFs with the same layer: one named PG per layer holding the
    curves of both loads.  Before #1364 the second raw
    addPhysicalGroup left an unnamed PG behind.

    The oracle is the importer's own return value, so the test holds
    whichever layer each curve lands in (``_unmatched`` included)."""
    a, b = tmp_path / "a.dxf", tmp_path / "b.dxf"
    _write_dxf(a, "Beams", 0.0)
    _write_dxf(b, "Beams", 5.0)
    first = g.model.io.load_dxf(a)
    second = g.model.io.load_dxf(b)

    assert all(n for _, _, n in _pgs()), _pgs()
    for layer in set(first) | set(second):
        expected = (set(first.get(layer, {}).get(1, []))
                    | set(second.get(layer, {}).get(1, [])))
        assert set(g.physical.entities(layer)) == expected, layer


def test_dxf_layer_held_at_other_dim_raises_before_import(g, tmp_path):
    _unit_line(g)
    g.physical.add(0, g.labels.entities("a"), name="Beams")
    path = tmp_path / "a.dxf"
    _write_dxf(path, "Beams", 3.0)
    before_pgs = _pgs()
    before_curves = gmsh.model.getEntities(1)
    with pytest.raises(ValueError, match=r"\('Beams', 0\)"):
        g.model.io.load_dxf(path)
    assert _pgs() == before_pgs
    assert gmsh.model.getEntities(1) == before_curves


def test_dxf_unmatched_held_at_other_dim_raises_before_import(g, tmp_path):
    """``_unmatched`` is a layer-PG name the importer adds itself, so it
    is refused before the import, like a file layer."""
    _unit_line(g)
    g.physical.add(0, g.labels.entities("a"), name="_unmatched")
    path = tmp_path / "a.dxf"
    _write_dxf(path, "Beams", 3.0)
    before_pgs = _pgs()
    before_curves = gmsh.model.getEntities(1)
    with pytest.raises(ValueError, match=r"\('_unmatched', 0\)"):
        g.model.io.load_dxf(path)
    assert _pgs() == before_pgs
    assert gmsh.model.getEntities(1) == before_curves


def test_dxf_layer_reserved_prefix_raises(g, tmp_path):
    path = tmp_path / "a.dxf"
    _write_dxf(path, f"{LABEL_PREFIX}x", 0.0)
    with pytest.raises(ValueError, match="reserved"):
        g.model.io.load_dxf(path)
    assert _pgs() == []
    assert gmsh.model.getEntities(1) == []


# ---------------------------------------------------------------------
# 4. errors list names, not (dim, tag) keys
# ---------------------------------------------------------------------


def _listed(message: str, after: str) -> list:
    """The Python list literal printed after *after* in *message*."""
    m = re.search(re.escape(after) + r"\s*(\[[^\]]*\])", message)
    assert m, message
    return ast.literal_eval(m.group(1))


@pytest.fixture
def fem(g):
    _unit_line(g)
    g.physical.add(1, g.labels.entities("beam"), name="Girder")
    g.mesh.generation.generate(dim=1)
    return g.mesh.queries.get_fem_data(dim=1)


def test_node_target_error_lists_names(fem):
    with pytest.raises(KeyError) as exc:
        fem.nodes.select(target="nope")
    msg = exc.value.args[0]
    assert _listed(msg, "Labels:") == fem.nodes.labels.names()
    assert _listed(msg, "physical groups:") == fem.nodes.physical.names()
    assert "beam" in fem.nodes.labels.names()


# ---------------------------------------------------------------------
# 5. promote into a name held at another dim names pg_name=
# ---------------------------------------------------------------------


def test_promote_cross_dim_names_pg_name(g):
    _unit_line(g)
    g.labels.promote_to_physical("beam", pg_name="Footing")
    before = _pgs()
    with pytest.raises(ValueError, match=r"pg_name=") as exc:
        g.labels.promote_to_physical("a", pg_name="Footing")
    assert "dim=1" in str(exc.value) and "dim=0" in str(exc.value)
    assert _pgs() == before
