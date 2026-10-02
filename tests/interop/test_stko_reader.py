"""STKO ``.scd`` reader: a synthetic document with every structure the
reader handles, plus the real San Ramon documents when available.

The synthetic geometry is a slab face on four edges with a free column
edge; the four slab edges also carry edge meshes (as STKO stores them)
that are not analysis elements.
"""
from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh.interop.stko import read_scd, write_brep

GEOM = 7


def _s(text: str) -> np.ndarray:
    return np.array([text.encode()])


def _i(value: int) -> np.ndarray:
    return np.array([value], dtype=np.int32)


def _xobj(group: h5py.Group, oid: int, name: str, xtype: str, params: list) -> h5py.Group:
    """``params``: ``(name, kind, value)`` with kind an attribute
    encoding (BOOL/INT/REAL/QNT_SCALAR/INDEX), a dataset encoding
    (STRING/QNT_VEC3/QNT_VECN/INDEX_VEC), CUSTOM_OBJECT or None."""
    g = group.create_group(f"{group.name.split('/')[-1].replace('PHYSICAL_PROPERTIES', 'PHYS_PROP').replace('ELEMENT_PROPERTIES', 'ELEM_PROP').replace('CONDITIONS', 'COND').replace('DEFINITIONS', 'DEF').replace('ANALYSIS_STEPS', 'STEP')}_{oid}")
    g.attrs["ID"] = _i(oid)
    g.attrs["NAME"] = _s(name)
    x = g.create_group("XOBJ")
    x.attrs["XOBJ_META"] = _s(xtype)
    attrs = x.create_group("ATTRIBUTES")
    for n, (pname, kind, value) in enumerate(params, start=1):
        a = attrs.create_group(f"ATTR_{n}")
        a.attrs["ATTR_META"] = _s(pname)
        a.attrs["VISIBLE"] = _i(1)
        if kind in ("BOOL", "INT", "INDEX"):
            a.attrs[kind] = np.array([value], dtype=np.int32)
        elif kind in ("REAL", "QNT_SCALAR"):
            a.attrs[kind] = np.array([value], dtype=np.float64)
        elif kind == "STRING":
            a.create_dataset("STRING", data=_s(value))
        elif kind in ("QNT_VEC3", "QNT_VECN"):
            a.create_dataset(kind, data=np.asarray(value, dtype=np.float64))
        elif kind == "INDEX_VEC":
            a.create_dataset(kind, data=np.asarray(value, dtype=np.int32))
        elif kind == "CUSTOM_OBJECT":
            c = a.create_group("CUSTOM_OBJECT")
            c.create_dataset("POINTS", data=np.asarray(value, dtype=np.float64))
            c.attrs["KIND"] = _s("rectangle")
    return g


def _write_scd(path: Path, *, duplicate_param: bool = False) -> Path:
    with h5py.File(path, "w") as f:
        f.create_dataset("FILE_INFO/VERSION", data=np.array([4, 1, 0], dtype=np.int32))

        g = f.create_group(f"GEOMETRIES/GEOM_{GEOM}")
        f["GEOMETRIES"].create_dataset("COMPOUND_SHAPE", data=np.zeros(4, dtype=np.int8))
        g.attrs.update(ID=_i(GEOM), NAME=_s("structure"), SHAPE_ID=_i(1))
        # 5 vertices (4 slab corners + column top), 5 edges (4 slab + column), 1 face
        for asn, values in {
            "ELEM_PROP_ASN": {"VER": [0] * 5, "EDG": [0, 0, 0, 0, 2], "FAC": [1]},
            "PHYS_PROP_ASN": {"VER": [0] * 5, "EDG": [0, 0, 0, 0, 4], "FAC": [3]},
            "LOCAX_ASN": {"VER": [0] * 5, "EDG": [0] * 5, "FAC": [1]},
        }.items():
            for key, v in values.items():
                g.create_dataset(f"{asn}/{key}", data=np.asarray(v, dtype=np.int32))

        m = f.create_group("MESH")
        # node IDs stored out of order, as STKO may
        m.create_dataset("NODE_IDS", data=np.array([5, 1, 2, 3, 4], dtype=np.int32))
        m.create_dataset("NODE_COORDINATES", data=np.array(
            [[0, 0, 1], [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64))
        m.create_dataset("NODE_FLAGS", data=np.zeros(5, dtype=np.int32))
        records = [
            [10, 303, 201, 0, 4, 0, 1, 2, 3, 4],    # slab quad
            [20, 102, 2, 0, 2, 0, 1, 5],            # column
            [30, 102, 2, 0, 2, 0, 1, 2],            # slab edge meshes
            [31, 102, 2, 0, 2, 0, 2, 3],
            [32, 102, 2, 0, 2, 0, 3, 4],
            [33, 102, 2, 0, 2, 0, 4, 1],
            [40, 600, 0, 0, 2, 4, 5, 3],            # interaction link
        ]
        m.create_dataset("ELEMENTS", data=np.concatenate(records).astype(np.int32))
        m.create_dataset("ELEMENT_ORIENTATION_IDS", data=np.array([10, 20], dtype=np.int32))
        m.create_dataset("ELEMENT_ORIENTATION_QUATERNIONS", data=np.array(
            [[0, 0, 0, 1], [0, -0.7071, 0, 0.7071]], dtype=np.float64))
        mog = m.create_group("MESH_OF_GEOMETRIES/MOG_1")
        mog.attrs["GEOM_ID"] = _i(GEOM)
        mog.create_dataset("VERTICES", data=np.array([1, 2, 3, 4, 5], dtype=np.int32))
        for i, elems in enumerate([[30], [31], [32], [33], [20]], start=1):
            mog.create_dataset(f"EDGES/DOM_{i}/ELEMENTS", data=np.asarray(elems, dtype=np.int32))
        mog.create_dataset("FACES/DOM_1/ELEMENTS", data=np.array([10], dtype=np.int32))
        moi = m.create_group("MESH_OF_INTERACTIONS/MOI_1")
        moi.attrs["INTERACTION_ID"] = _i(1)
        moi.create_dataset("ELEMENTS", data=np.array([40], dtype=np.int32))

        s = f.create_group("SELECTION_SETS/SET_1")
        s.attrs.update(ID=_i(1), NAME=_s("Columns"))
        item = s.create_group(f"ITEMS/ITEM_{GEOM}")
        item.attrs.update(ID=_i(GEOM), WHOLE_GEOM=_i(0))
        item.create_dataset("EDGES", data=np.array([4], dtype=np.int32))
        item.create_dataset("VERTICES", data=np.array([4], dtype=np.int32))
        s = f.create_group("SELECTION_SETS/SET_2")
        s.attrs.update(ID=_i(2), NAME=_s("All"))
        item = s.create_group(f"ITEMS/ITEM_{GEOM}")
        item.attrs.update(ID=_i(GEOM), WHOLE_GEOM=_i(1))

        phys = f.create_group("PHYSICAL_PROPERTIES")
        params = [
            ("$$(DATASTORE)$$", None, None),
            ("E", "QNT_SCALAR", 25000.0),
            ("nu", "REAL", 0.2),
            ("Shear Deformable", "BOOL", 1),
            ("version", "INT", 2),
            ("Dimension", "STRING", "3D"),
            ("Eb material", "INDEX", 4),
            ("mass", "QNT_VEC3", [1.0, 2.0, 3.0]),
            ("Section", "CUSTOM_OBJECT", [[0, 0], [700, 700]]),
        ]
        if duplicate_param:
            params.append(("E", "REAL", 1.0))
        _xobj(phys, 3, "Slab_Elastic", "sections.ElasticMembranePlateSection", params)
        _xobj(phys, 4, "C70x70", "sections.Elastic", [("E", "QNT_SCALAR", 25000.0)])
        elem = f.create_group("ELEMENT_PROPERTIES")
        _xobj(elem, 1, "ASDShellQ4_Slabs", "shell.ASDShellQ4", [("Drilling DOF Type", "STRING", "Elastic")])
        _xobj(elem, 2, "elasticBeamColumn", "beam_column_elements.elasticBeamColumn", [])

        cond = f.create_group("CONDITIONS")
        c = _xobj(cond, 1, "fix", "Constraints.sp.fix", [("Ux", "BOOL", 1)])
        # [geom, nV, nE, nF, nS, vertices..., edges..., faces..., solids...]
        c.create_dataset("ASN_GEOMETRY", data=np.array([GEOM, 2, 0, 1, 0, 0, 1, 0], dtype=np.int32))
        c = _xobj(cond, 2, "Embedded", "Constraints.mp.ASDEmbeddedNodeElement", [])
        c.create_dataset("ASN_INTERACTION", data=np.array([1], dtype=np.int32))
        _xobj(cond, 3, "DRM", "Loads.Generic.H5DRM", [("File Name", "STRING", "C:/x.h5drm")])

        inter = f.create_group("INTERACTIONS/INTER_1")
        inter.attrs.update(ID=_i(1), NAME=_s("structure_soil"), TYPE=_s("NE"),
                           ELEM_PROP_ID=_i(0), PHYS_PROP_ID=_i(0))
        inter.create_dataset("MASTERS", data=np.array([[GEOM, 3, 0]], dtype=np.int32))
        inter.create_dataset("SLAVES", data=np.array([[GEOM, 3, 0], [GEOM, 1, 4]], dtype=np.int32))

        ax = f.create_group("LOCAL_AXES/LOCAX_1")
        ax.attrs.update(ID=_i(1), NAME=_s("rotatedWalls"), TYPE=_s("Rec"))
        ax.create_dataset("DATA", data=np.array(
            [0, -30000, 0, 0, -1, 0, 1, 0, 0, 0, 0, 1], dtype=np.float64))

        _xobj(f.create_group("DEFINITIONS"), 2, "TH_X", "timeSeries.Path",
              [("list_of_values", "QNT_VECN", [0.0, 0.1, -0.2])])
        steps = f.create_group("ANALYSIS_STEPS")
        _xobj(steps, 11, "Analysis", "Analyses.AnalysesCommand", [])
        _xobj(steps, 2, "region", "Misc_commands.region", [("SelectionSets", "INDEX_VEC", [1, 2])])
    return path


@pytest.fixture
def scd(tmp_path: Path):
    return read_scd(_write_scd(tmp_path / "model.scd"))


# ── reading ──────────────────────────────────────────────────────────

def test_document_basics(scd) -> None:
    assert scd.version == (4, 1, 0)
    g = scd.geometries[GEOM]
    assert (g.name, g.shape_index) == ("structure", 1)
    assert g.counts == {"vertices": 5, "edges": 5, "faces": 1}
    assert g.element_property["edges"].tolist() == [0, 0, 0, 0, 2]


def test_mesh_nodes_are_sorted_with_their_coordinates(scd) -> None:
    assert scd.mesh.node_ids.tolist() == [1, 2, 3, 4, 5]
    assert scd.mesh.node_xyz(5).tolist() == [0.0, 0.0, 1.0]
    with pytest.raises(KeyError):
        scd.mesh.node_xyz(99)


def test_mesh_element_records(scd) -> None:
    quad = scd.mesh.elements[10]
    assert (quad.type_name, quad.rule, quad.nodes) == ("quad4", 201, (1, 2, 3, 4))
    assert scd.mesh.elements[40].type_name == "link"
    assert scd.mesh.domains[(GEOM, "edges")][4].tolist() == [20]
    assert scd.mesh.orientation[20] == (0.0, -0.7071, 0.0, 0.7071)


def test_parameter_encodings(scd) -> None:
    p = scd.physical_property("Slab_Elastic")
    assert p.type == "sections.ElasticMembranePlateSection"
    assert p["$$(DATASTORE)$$"] is None
    assert p["E"] == 25000.0 and isinstance(p["E"], float)
    assert p["nu"] == 0.2
    assert p["Shear Deformable"] is True
    assert p["version"] == 2 and isinstance(p["version"], int)
    assert p["Dimension"] == "3D"
    assert p["Eb material"] == 4
    assert p["mass"] == (1.0, 2.0, 3.0)
    assert p["Section"]["POINTS"].tolist() == [[0, 0], [700, 700]]
    assert p["Section"]["@attrs"]["KIND"][0] == b"rectangle"
    assert p.references == frozenset({"Eb material"})
    region = scd.analysis_steps[2]
    assert region["SelectionSets"] == (1, 2)
    assert "SelectionSets" in region.references
    assert scd.definitions[2]["list_of_values"] == (0.0, 0.1, -0.2)


def test_analysis_steps_in_id_order(scd) -> None:
    assert list(scd.analysis_steps) == [2, 11]


def test_duplicate_parameter_name_is_refused(tmp_path: Path) -> None:
    path = _write_scd(tmp_path / "dup.scd", duplicate_param=True)
    with pytest.raises(ValueError, match="two parameters named 'E'"):
        read_scd(path)


def test_conditions_and_their_assignments(scd) -> None:
    fix = scd.condition("fix")
    assert fix.type == "Constraints.sp.fix"
    assert fix.geometry[GEOM].vertices == (0, 1)
    assert fix.geometry[GEOM].faces == (0,)
    assert scd.condition("Embedded").interactions == (1,)
    drm = scd.condition("DRM")
    assert not drm.geometry and not drm.interactions
    assert drm.xobject["File Name"] == "C:/x.h5drm"


def test_interactions_and_local_axes(scd) -> None:
    inter = scd.interactions[1]
    assert (inter.name, inter.type, inter.elements) == ("structure_soil", "NE", (40,))
    assert inter.masters[0].geometry == GEOM and inter.masters[0].kind == 3
    assert len(inter.slaves) == 2
    ax = scd.local_axes[1]
    assert ax.origin == (0.0, -30000.0, 0.0) and ax.x == (0.0, -1.0, 0.0)


def test_lookup_by_unknown_name_lists_the_names(scd) -> None:
    with pytest.raises(KeyError, match="C70x70"):
        scd.physical_property("nope")


# ── selection sets and analysis elements ─────────────────────────────

def test_set_expansion(scd) -> None:
    assert scd.set_elements("Columns") == {20}
    assert scd.set_nodes("Columns") == {1, 5}


def test_whole_geometry_set_covers_everything(scd) -> None:
    assert scd.set_elements("All") == {10, 20, 30, 31, 32, 33}
    assert scd.set_nodes("All") == {1, 2, 3, 4, 5}


def test_unknown_set_lists_the_names(scd) -> None:
    with pytest.raises(KeyError, match="Columns"):
        scd.set_elements("nope")


def test_analysis_elements_skip_the_edge_meshes(scd) -> None:
    ae = scd.analysis_elements()
    assert set(ae) == {10, 20, 40}
    assert (ae[10].kind, ae[10].index, ae[10].element_property, ae[10].physical_property) == ("faces", 0, 1, 3)
    assert (ae[20].kind, ae[20].element_property, ae[20].physical_property) == ("edges", 2, 4)
    assert (ae[40].geometry, ae[40].interaction) == (None, 1)


# ── geometry ─────────────────────────────────────────────────────────

def test_write_brep_needs_ocp(scd, tmp_path: Path, monkeypatch) -> None:
    import builtins
    real_import = builtins.__import__

    def no_ocp(name, *args, **kwargs):
        if name.startswith("OCP"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_ocp)
    with pytest.raises(ImportError, match="cadquery-ocp"):
        write_brep(scd, tmp_path / "x.brep")


def _store_compound(path: Path, tmp_path: Path) -> None:
    """Replace the document's COMPOUND_SHAPE with a real binary BRep
    compound: child 1 a unit slab face, child 2 a column edge."""
    from OCP.BinTools import BinTools
    from OCP.BRep import BRep_Builder
    from OCP.BRepBuilderAPI import (
        BRepBuilderAPI_MakeEdge,
        BRepBuilderAPI_MakeFace,
        BRepBuilderAPI_MakePolygon,
    )
    from OCP.gp import gp_Pnt
    from OCP.TopoDS import TopoDS_Compound

    poly = BRepBuilderAPI_MakePolygon(
        gp_Pnt(0, 0, 0), gp_Pnt(1, 0, 0), gp_Pnt(1, 1, 0), gp_Pnt(0, 1, 0), True,
    )
    face = BRepBuilderAPI_MakeFace(poly.Wire()).Face()
    edge = BRepBuilderAPI_MakeEdge(gp_Pnt(0, 0, 0), gp_Pnt(0, 0, 1)).Edge()
    compound = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(compound)
    builder.Add(compound, face)
    builder.Add(compound, edge)
    binary = tmp_path / "compound.bin"
    BinTools.Write_s(compound, str(binary))
    with h5py.File(path, "a") as f:
        del f["GEOMETRIES/COMPOUND_SHAPE"]
        f["GEOMETRIES"].create_dataset(
            "COMPOUND_SHAPE", data=np.frombuffer(binary.read_bytes(), dtype=np.int8),
        )
        g = f.create_group("GEOMETRIES/GEOM_8")
        g.attrs.update(ID=_i(8), NAME=_s("column"), SHAPE_ID=_i(2))


def test_write_brep_extracts_the_compound_and_one_geometry(tmp_path: Path) -> None:
    pytest.importorskip("OCP")
    import gmsh

    path = _write_scd(tmp_path / "model.scd")
    _store_compound(path, tmp_path)

    def counts(brep: Path) -> list[int]:
        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.occ.importShapes(str(brep), highestDimOnly=False)
            gmsh.model.occ.synchronize()
            return [len(gmsh.model.getEntities(d)) for d in range(3)]
        finally:
            gmsh.finalize()

    whole = write_brep(path, tmp_path / "whole.brep")
    assert counts(whole) == [6, 5, 1]            # 4 + 2 points, 4 + 1 edges, 1 face
    column = write_brep(read_scd(path), tmp_path / "column.brep", geometry=8)
    assert counts(column) == [2, 1, 0]
    with pytest.raises(KeyError, match="no geometry 99"):
        write_brep(path, tmp_path / "x.brep", geometry=99)


# ── the real San Ramon documents (skipped unless available) ──────────

_SAMPLES = os.environ.get("APEGMSH_STKO_ORACLES")


@pytest.mark.skipif(not _SAMPLES, reason="set APEGMSH_STKO_ORACLES to the stko-rev0-tcl folder")
def test_san_ramon_1a_matches_the_tcl_export() -> None:
    m = read_scd(Path(_SAMPLES) / "Tier_1" / "1A" / "1A_TH_000.scd")
    assert len(m.mesh.node_ids) == 14003
    ae = m.analysis_elements()
    assert len(ae) == 13704                          # elements.tcl
    by_prop = {}
    for a in ae.values():
        name = m.element_properties[a.element_property].name
        by_prop[name] = by_prop.get(name, 0) + 1
    assert by_prop == {
        "elasticBeamColumn": 736, "ASDShellQ4_BasementsWalls": 936,
        "ASDShellQ4_ShearWalls": 1104, "ASDShellQ4_Slabs": 9456,
        "ASDShellQ4_Foundation": 1472,
    }
    assert (len(m.set_nodes("Columns_Set")), len(m.set_elements("Columns_Set"))) == (752, 736)


@pytest.mark.skipif(not _SAMPLES, reason="set APEGMSH_STKO_ORACLES to the stko-rev0-tcl folder")
def test_san_ramon_4d_matches_the_tcl_export() -> None:
    m = read_scd(Path(_SAMPLES) / "Tier_4" / "4D" / "4D_TH_000.scd")
    assert len(m.mesh.node_ids) == 55980
    ae = m.analysis_elements()
    assert len(ae) == 12968 + 192 + 37422 + 7324     # shells, columns, bricks, embedded
    assert sum(len(i.elements) for i in m.interactions.values()) == 7324
    assert m.condition("DRM").type == "Loads.Generic.H5DRM"


@pytest.mark.skipif(not _SAMPLES, reason="set APEGMSH_STKO_ORACLES to the stko-rev0-tcl folder")
def test_san_ramon_1a_geometry_keeps_face_and_edge_order(tmp_path: Path) -> None:
    """STKO face i is gmsh face i + 1 (edges likewise): the property
    assignments of the .scd land on the right gmsh entities."""
    pytest.importorskip("OCP")
    import gmsh

    m = read_scd(Path(_SAMPLES) / "Tier_1" / "1A" / "1A_TH_000.scd")
    brep = write_brep(m, tmp_path / "1a.brep", geometry=353)
    column_edges = np.flatnonzero(m.geometries[353].element_property["edges"])
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.occ.importShapes(str(brep), highestDimOnly=False)
        gmsh.model.occ.synchronize()
        assert [len(gmsh.model.getEntities(d)) for d in range(3)] == [1049, 2316, 1016]
        free = {t for _, t in gmsh.model.getEntities(1)
                if len(gmsh.model.getAdjacencies(1, t)[0]) == 0}
        assert free == {int(i) + 1 for i in column_edges}
    finally:
        gmsh.finalize()
