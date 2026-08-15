"""Void-role tools, pre-mesh guard, apply_voids, sweep/loft (ADR 0097)."""
from __future__ import annotations

import gmsh
import pytest

from apeGmsh.core._geometry_errors import GeometryValidationError


def test_as_void_metadata_on_solid(g):
    tag = g.model.geometry.add_cylinder(
        0, 0, 0, 0, 0, 1, 0.2, as_void=True, label="duct",
    )
    assert g.model._metadata[(3, tag)]["kind"] == "cylinder"
    assert g.model._metadata[(3, tag)]["role"] == "void"


def test_as_void_metadata_on_rectangle(g):
    tag = g.model.geometry.add_rectangle(
        0, 0, 0, 1, 1, as_void=True, label="hole",
    )
    assert g.model._metadata[(2, tag)]["role"] == "void"


def test_unapplied_void_fails_generate(g):
    g.model.geometry.add_box(0, 0, 0, 2, 2, 2, label="wall")
    g.model.geometry.add_cylinder(
        1, 1, -0.1, 0, 0, 2.2, 0.3, as_void=True, label="duct",
    )
    g.mesh.sizing.set_global_size(1.0)
    with pytest.raises(GeometryValidationError, match="unapplied void"):
        g.mesh.generation.generate(dim=3)


def test_cut_then_generate_succeeds(g):
    g.model.geometry.add_box(0, 0, 0, 2, 2, 2, label="wall")
    g.model.geometry.add_cylinder(
        1, 1, -0.1, 0, 0, 2.2, 0.3, as_void=True, label="duct",
    )
    result = g.model.boolean.cut("wall", "duct")
    assert len(result) == 1
    assert not any(
        meta.get("role") == "void" for meta in g.model._metadata.values()
    )
    g.mesh.sizing.set_global_size(1.0)
    g.mesh.generation.generate(dim=3)
    assert gmsh.model.mesh.getNodes()[0].size > 0


def test_apply_voids_subtracts_all_tools(g):
    g.model.geometry.add_box(0, 0, 0, 4, 2, 2, label="wall")
    g.model.geometry.add_cylinder(
        1, 1, -0.1, 0, 0, 2.2, 0.25, as_void=True, label="d1",
    )
    g.model.geometry.add_cylinder(
        3, 1, -0.1, 0, 0, 2.2, 0.25, as_void=True, label="d2",
    )
    result = g.model.boolean.apply_voids("wall")
    assert len(result) == 1
    assert not any(
        meta.get("role") == "void" for meta in g.model._metadata.values()
    )
    vols = [t for _, t in gmsh.model.getEntities(3)]
    assert result[0] in vols
    assert len(vols) == 1


def test_apply_voids_no_tools_raises(g):
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="wall")
    with pytest.raises(ValueError, match="no unapplied void"):
        g.model.boolean.apply_voids("wall")


def test_2d_void_cut(g):
    g.model.geometry.add_rectangle(0, 0, 0, 2, 2, label="plate")
    g.model.geometry.add_rectangle(
        0.5, 0.5, 0, 0.4, 0.4, as_void=True, label="hole",
    )
    result = g.model.boolean.apply_voids("plate")
    assert len(result) >= 1
    assert not any(
        meta.get("role") == "void" for meta in g.model._metadata.values()
    )


def test_add_void_sweep_marks_role(g):
    path = g.model.geometry.add_polyline(
        [(0.0, 0.5, 0.5), (3.0, 0.5, 0.5)],
    )
    profile = g.model.geometry.add_polyline(
        [
            (0.0, 0.35, 0.35),
            (0.0, 0.65, 0.35),
            (0.0, 0.65, 0.65),
            (0.0, 0.35, 0.65),
        ],
        closed=True,
    )
    vol = g.model.geometry.add_void_sweep(profile, path, label="tunnel")
    assert g.model._metadata[(3, vol)]["role"] == "void"
    g.model.geometry.add_box(0, 0, 0, 3, 1, 1, label="host")
    g.model.boolean.apply_voids("host")
    g.mesh.sizing.set_global_size(0.5)
    g.mesh.generation.generate(dim=3)


def test_add_void_loft_matching_sections(g):
    a = g.model.geometry.add_polyline(
        [(0.4, 0.4, 0.0), (0.6, 0.4, 0.0), (0.6, 0.6, 0.0), (0.4, 0.6, 0.0)],
        closed=True,
    )
    b = g.model.geometry.add_polyline(
        [(0.3, 0.3, 2.0), (0.7, 0.3, 2.0), (0.7, 0.7, 2.0), (0.3, 0.7, 2.0)],
        closed=True,
    )
    vol = g.model.geometry.add_void_loft([a, b], label="taper")
    assert g.model._metadata[(3, vol)]["role"] == "void"
    g.model.geometry.add_box(0, 0, 0, 1, 1, 2, label="host")
    g.model.boolean.apply_voids("host")
    g.mesh.sizing.set_global_size(0.5)
    g.mesh.generation.generate(dim=3)


def test_add_void_loft_count_mismatch_raises(g):
    a = g.model.geometry.add_polyline(
        [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)],
        closed=True,
    )
    b = g.model.geometry.add_polyline(
        [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)],
        closed=True,
        fillet={1: 0.1},
    )
    with pytest.raises(ValueError, match="curve counts must match"):
        g.model.geometry.add_void_loft([a, b])
