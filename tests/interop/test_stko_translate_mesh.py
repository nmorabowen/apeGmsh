"""STKO translator, mesh package (ADR 0111 D1; rules M1-M6).

A synthetic ``.scd`` (writer below, in the style of
``test_stko_reader.py``) covers every step of ``build_mesh`` with numbers
that can be checked by hand; the real San Ramon Tier-1 documents run
when ``APEGMSH_STKO_ORACLES`` points at the ``stko-rev0-tcl`` folder.

Synthetic document, geometry 1 ("structure"), in the plane z = 0 plus a
column::

    4 ---- 5 ---- 6        faces 0: quads 101 (1 2 5 4), 102 (2 3 6 5)
    |  101 |  102 |        edges 0: 201 (1 2), 202 (2 3)   slab edge, carrier
    1 ---- 2 ---- 3        edges 1: 203 (1 7)              column, analysis
                           edges 2: 204 (4 5), 205 (5 6)   slab edge, carrier
    vertices 0..4 -> nodes 1 3 6 4 7

Geometry 2 ("master") is one vertex on node 10. Interaction 1 (NN) holds
links 301 (10 2) and 302 (10 5) for the rigid diaphragm, condition 2.
Quad 102's quaternion (0, 0, 0.6, 0.8) puts its local x at
(0.28, 0.96, 0): rule M3 rotates its nodes to (3 6 5 2), and face 0
splits into two entities by local axis (rule M6).
"""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from apeGmsh import apeGmsh
from apeGmsh.interop.stko import read_scd
from apeGmsh.interop.stko.translate_mesh import (
    _group_names,
    build_mesh,
    check_supported,
    session_nodes,
    shell_node_order,
)
from apeGmsh.interop.stko.translate_types import UnsupportedSTKOTypes, local_axis

_PREFIX = {
    "PHYSICAL_PROPERTIES": "PHYS_PROP", "ELEMENT_PROPERTIES": "ELEM_PROP",
    "CONDITIONS": "COND", "DEFINITIONS": "DEF", "ANALYSIS_STEPS": "STEP",
}


def _s(text: str) -> np.ndarray:
    return np.array([text.encode()])


def _i(value: int) -> np.ndarray:
    return np.array([value], dtype=np.int32)


def _xobj(group: h5py.Group, oid: int, name: str, xtype: str, params: list) -> h5py.Group:
    g = group.create_group(f"{_PREFIX[group.name.split('/')[-1]]}_{oid}")
    g.attrs.update(ID=_i(oid), NAME=_s(name))
    x = g.create_group("XOBJ")
    x.attrs["XOBJ_META"] = _s(xtype)
    attrs = x.create_group("ATTRIBUTES")
    for n, (pname, kind, value) in enumerate(params, start=1):
        a = attrs.create_group(f"ATTR_{n}")
        a.attrs["ATTR_META"] = _s(pname)
        if kind in ("BOOL", "INT", "INDEX"):
            a.attrs[kind] = np.array([value], dtype=np.int32)
        elif kind in ("REAL", "QNT_SCALAR"):
            a.attrs[kind] = np.array([value], dtype=np.float64)
        elif kind == "QNT_VEC3":
            a.create_dataset(kind, data=np.asarray(value, dtype=np.float64))
        elif kind == "INDEX_VEC":
            a.create_dataset(kind, data=np.asarray(value, dtype=np.int32))
    return g


def _asn(values: dict[str, list[int]], g: h5py.Group, asn: str) -> None:
    for key, v in values.items():
        g.create_dataset(f"{asn}/{key}", data=np.asarray(v, dtype=np.int32))


def _write_scd(path: Path, *, unsupported: bool = False, diaphragm_referenced: bool = True,
               orphan_node: bool = False) -> Path:
    """The synthetic document; ``unsupported`` adds one of each thing the
    mesh slice refuses (see test_unsupported_items_are_all_listed)."""
    with h5py.File(path, "w") as f:
        f.create_dataset("FILE_INFO/VERSION", data=np.array([4, 1, 0], dtype=np.int32))

        g = f.create_group("GEOMETRIES/GEOM_1")
        g.attrs.update(ID=_i(1), NAME=_s("structure"), SHAPE_ID=_i(1))
        _asn({"VER": [0] * 5, "EDG": [0, 2, 0], "FAC": [1]}, g, "ELEM_PROP_ASN")
        _asn({"VER": [0] * 5, "EDG": [0, 4, 0], "FAC": [3]}, g, "PHYS_PROP_ASN")
        g = f.create_group("GEOMETRIES/GEOM_2")
        g.attrs.update(ID=_i(2), NAME=_s("master"), SHAPE_ID=_i(2))
        _asn({"VER": [0]}, g, "ELEM_PROP_ASN")
        _asn({"VER": [0]}, g, "PHYS_PROP_ASN")

        coords = {1: (0, 0, 0), 2: (1, 0, 0), 3: (2, 0, 0), 4: (0, 1, 0), 5: (1, 1, 0),
                  6: (2, 1, 0), 7: (0, 0, 1), 10: (1, 0.5, 0)}
        records = [
            [101, 303, 201, 0, 4, 0, 1, 2, 5, 4],
            [102, 303, 201, 0, 4, 0, 2, 3, 6, 5],
            [201, 102, 2, 0, 2, 0, 1, 2],
            [202, 102, 2, 0, 2, 0, 2, 3],
            [203, 102, 2, 0, 2, 0, 1, 7],
            [204, 102, 2, 0, 2, 0, 4, 5],
            [205, 102, 2, 0, 2, 0, 5, 6],
            [301, 600, 0, 0, 2, 0, 10, 2],
            [302, 600, 0, 0, 2, 0, 10, 5],
        ]
        orient = {101: (0, 0, 0, 1), 102: (0, 0, 0.6, 0.8), 203: (0, -0.6, 0, 0.8)}
        if orphan_node:
            coords |= {20: (5, 5, 5)}          # a mesh node no element and no vertex uses
        if unsupported:
            coords |= {11: (0, 0, -1), 12: (1, 0, -1), 13: (1, 1, -1), 14: (0, 1, -1)}
            records += [
                [401, 500, 0, 0, 8, 0, 1, 2, 5, 4, 11, 12, 13, 14],   # hex8 on a solid
                [402, 203, 0, 0, 3, 0, 4, 5, 14],                     # unknown type on face 1
            ]
            orient |= {401: (0, 0, 0, 1), 402: (0, 0, 0, 1)}
            g = f["GEOMETRIES/GEOM_1"]
            for asn, v in (("ELEM_PROP_ASN", [5]), ("PHYS_PROP_ASN", [3])):
                g.create_dataset(f"{asn}/SOL", data=np.asarray(v, dtype=np.int32))
            del g["ELEM_PROP_ASN/FAC"], g["PHYS_PROP_ASN/FAC"]
            _asn({"FAC": [1, 1]}, g, "ELEM_PROP_ASN")
            _asn({"FAC": [3, 3]}, g, "PHYS_PROP_ASN")

        m = f.create_group("MESH")
        ids = sorted(coords)
        m.create_dataset("NODE_IDS", data=np.asarray(ids[::-1], dtype=np.int32))
        m.create_dataset("NODE_COORDINATES", data=np.asarray([coords[n] for n in ids[::-1]], dtype=np.float64))
        m.create_dataset("NODE_FLAGS", data=np.zeros(len(ids), dtype=np.int32))
        m.create_dataset("ELEMENTS", data=np.concatenate(records).astype(np.int32))
        m.create_dataset("ELEMENT_ORIENTATION_IDS", data=np.asarray(list(orient), dtype=np.int32))
        m.create_dataset("ELEMENT_ORIENTATION_QUATERNIONS",
                         data=np.asarray(list(orient.values()), dtype=np.float64))
        mog = m.create_group("MESH_OF_GEOMETRIES/MOG_1")
        mog.attrs["GEOM_ID"] = _i(1)
        mog.create_dataset("VERTICES", data=np.array([1, 3, 6, 4, 7], dtype=np.int32))
        for i, elems in enumerate([[201, 202], [203], [204, 205]], start=1):
            mog.create_dataset(f"EDGES/DOM_{i}/ELEMENTS", data=np.asarray(elems, dtype=np.int32))
        mog.create_dataset("FACES/DOM_1/ELEMENTS", data=np.array([101, 102], dtype=np.int32))
        if unsupported:
            mog.create_dataset("FACES/DOM_2/ELEMENTS", data=np.array([402], dtype=np.int32))
            mog.create_dataset("SOLIDS/DOM_1/ELEMENTS", data=np.array([401], dtype=np.int32))
        mog = m.create_group("MESH_OF_GEOMETRIES/MOG_2")
        mog.attrs["GEOM_ID"] = _i(2)
        mog.create_dataset("VERTICES", data=np.array([10], dtype=np.int32))
        moi = m.create_group("MESH_OF_INTERACTIONS/MOI_1")
        moi.attrs["INTERACTION_ID"] = _i(1)
        moi.create_dataset("ELEMENTS", data=np.array([301, 302], dtype=np.int32))

        def sset(n: int, name: str, items: dict[int, dict[str, list[int]]], whole=()) -> None:
            s = f.create_group(f"SELECTION_SETS/SET_{n}")
            s.attrs.update(ID=_i(n), NAME=_s(name))
            for gid, kinds in items.items():
                item = s.create_group(f"ITEMS/ITEM_{gid}")
                item.attrs.update(ID=_i(gid), WHOLE_GEOM=_i(int(gid in whole)))
                for kind, v in kinds.items():
                    item.create_dataset(kind, data=np.asarray(v, dtype=np.int32))
        sset(1, "Columns", {1: {"EDGES": [1], "VERTICES": [4]}})
        sset(2, "All", {1: {}}, whole=(1,))
        sset(3, "Master", {2: {}}, whole=(2,))

        phys = f.create_group("PHYSICAL_PROPERTIES")
        _xobj(phys, 3, "Slab", "sections.ElasticMembranePlateSection", [("E", "QNT_SCALAR", 25000.0)])
        _xobj(phys, 4, "Col", "sections.Elastic", [("E", "QNT_SCALAR", 25000.0)])
        elem = f.create_group("ELEMENT_PROPERTIES")
        _xobj(elem, 1, "Shell", "shell.ASDShellQ4", [])
        _xobj(elem, 2, "Beam", "beam_column_elements.elasticBeamColumn", [])
        if unsupported:
            _xobj(elem, 5, "Brick", "brick_elements.stdBrick", [])

        cond = f.create_group("CONDITIONS")
        c = _xobj(cond, 1, "base", "Constraints.sp.fix", [("Ux", "BOOL", 1)])
        # [geom, nV, nE, nF, nS, vertices..., edges..., faces..., solids...]
        c.create_dataset("ASN_GEOMETRY", data=np.array([1, 1, 1, 0, 0, 0, 0], dtype=np.int32))
        c = _xobj(cond, 2, "diaphragm", "Constraints.mp.rigidDiaphragm", [("perpDirn", "INT", 3)])
        c.create_dataset("ASN_INTERACTION", data=np.array([1], dtype=np.int32))
        c = _xobj(cond, 3, "mass", "Mass.FaceMass", [("mass", "QNT_VEC3", [1.0, 1.0, 1.0])])
        c.create_dataset("ASN_GEOMETRY", data=np.array([1, 0, 0, 1, 0, 0], dtype=np.int32))

        def inter(n: int, name: str, typ: str) -> h5py.Group:
            i = f.create_group(f"INTERACTIONS/INTER_{n}")
            i.attrs.update(ID=_i(n), NAME=_s(name), TYPE=_s(typ), ELEM_PROP_ID=_i(0), PHYS_PROP_ID=_i(0))
            return i
        i = inter(1, "rd", "NN")
        i.create_dataset("MASTERS", data=np.array([[2, 1, 0]], dtype=np.int32))
        i.create_dataset("SLAVES", data=np.array([[1, 3, 0]], dtype=np.int32))
        if unsupported:
            inter(2, "embedded", "NE")
            inter(3, "links", "NN")          # NN not used by a rigid diaphragm
            inter(4, "odd", "XX")

        # STKO stores local axes as datasets with attributes (the reader fix)
        ax = f.create_group("LOCAL_AXES").create_dataset("LOCAX_1", data=np.array(
            [0, -30000, 0, 0, -1, 0, 1, 0, 0, 0, 0, 1], dtype=np.float64))
        ax.attrs.update(ID=_i(1), NAME=_s("rotatedWalls"), TYPE=_s("Rec"))
        f.create_group("DEFINITIONS")
        steps = f.create_group("ANALYSIS_STEPS")
        if diaphragm_referenced:
            # STKO writes a rigidDiaphragm only for a condition a constraint pattern lists
            _xobj(steps, 8, "constraints", "Patterns.addPattern.constraintPattern",
                  [("mp", "INDEX_VEC", [2])])
    return path


@pytest.fixture
def scd(tmp_path: Path):
    return read_scd(_write_scd(tmp_path / "model.scd"))


@pytest.fixture
def built(scd):
    """(scd, MeshMap, FEMData, {pg name: [entity tags]}, {entity tag: owned node ids})."""
    import gmsh
    with apeGmsh(model_name="stko_mesh", verbose=False) as g:
        mesh = build_mesh(g, scd)
        fem = g.mesh.queries.get_fem_data(dim=None)
        pg_entities = {
            gmsh.model.getPhysicalName(d, t): (d, list(gmsh.model.getEntitiesForPhysicalGroup(d, t)))
            for d, t in gmsh.model.getPhysicalGroups()
        }
        owned = {
            (d, t): sorted(int(n) for n in gmsh.model.mesh.getNodes(d, t)[0])
            for d, t in gmsh.model.getEntities()
        }
        yield scd, mesh, fem, pg_entities, owned


def _pg_nodes(fem, pg: str) -> set[int]:
    return {int(n) for n in fem.nodes.select(pg=pg).ids}


def _pg_elements(fem, pg: str) -> dict[int, tuple[int, ...]]:
    """Element id -> connectivity. ``.result().resolve()`` keeps ids and
    rows aligned; ``MeshSelection.ids`` is sorted while ``.connectivity``
    is in storage order, so the two do not pair up when ids are not
    stored ascending (STKO's are not)."""
    ids, conn = fem.elements.select(pg=pg).result().resolve()
    return {int(e): tuple(int(n) for n in row) for e, row in zip(ids, conn)}


# ── reader fixes ──────────────────────────────────────────────────────

def test_local_axes_stored_as_datasets_are_read(scd) -> None:
    ax = scd.local_axes[1]
    assert (ax.name, ax.type) == ("rotatedWalls", "Rec")
    assert ax.origin == (0.0, -30000.0, 0.0) and ax.x == (0.0, -1.0, 0.0)
    assert ax.z == (0.0, 0.0, 1.0)


# ── rules M3, M4/M5 ───────────────────────────────────────────────────

def test_shell_node_order_rule_m3(scd) -> None:
    # 101: local x (1, 0, 0) along p1 + p2 - p3 - p0 = (2, 0, 0): kept
    assert shell_node_order(scd, 101) == (1, 2, 5, 4)
    # 102: local x (0.28, 0.96, 0), |cos| = 0.28 <= 0.5: rotated by one
    assert shell_node_order(scd, 102) == (3, 6, 5, 2)
    assert shell_node_order(scd, 203) == (1, 7)          # not a shell: as stored


def test_local_axis_keys(scd) -> None:
    assert local_axis(scd, 101) == (1.0, 0.0, 0.0)
    assert local_axis(scd, 102) == (0.28, 0.96, 0.0)
    # beam: column 2 of the quaternion (0, -0.6, 0, 0.8) -> (-0.96, 0, 0.28)
    assert local_axis(scd, 203) == pytest.approx((-0.96, 0.0, 0.28))


# ── build_mesh ────────────────────────────────────────────────────────

def test_nodes_are_stko_nodes_with_stko_coordinates(built) -> None:
    scd, mesh, fem, _, _ = built
    assert mesh.node_ids == (1, 2, 3, 4, 5, 6, 7, 10)
    assert sorted(int(n) for n in fem.nodes.ids) == list(mesh.node_ids)
    xyz = {int(n): tuple(c) for n, c in zip(fem.nodes.ids, fem.nodes.coords)}
    for n in mesh.node_ids:
        assert xyz[n] == tuple(scd.mesh.node_xyz(n))


def test_element_groups_split_by_local_axis(built) -> None:
    _, mesh, fem, pg_entities, _ = built
    groups = {eg.pg: eg for eg in mesh.element_groups}
    assert list(groups) == ["Shell|Slab|L1", "Shell|Slab|L2", "Beam|Col"]
    l1, l2, beam = groups.values()
    assert (l1.dim, l1.element_property, l1.physical_property) == (2, 1, 3)
    assert (l1.local_axis, l1.element_ids) == ((0.28, 0.96, 0.0), (102,))
    assert (l2.local_axis, l2.element_ids) == ((1.0, 0.0, 0.0), (101,))
    assert (beam.dim, beam.element_ids) == (1, (203,))
    # STKO ids, rule-M3 node order, non-analysis edge meshes not in any group
    assert _pg_elements(fem, "Shell|Slab|L1") == {102: (3, 6, 5, 2)}
    assert _pg_elements(fem, "Shell|Slab|L2") == {101: (1, 2, 5, 4)}
    assert _pg_elements(fem, "Beam|Col") == {203: (1, 7)}
    # face 0 is two entities, one per axis
    assert len(mesh.subshape_entities[(1, "faces", 0)]) == 2
    assert pg_entities["Shell|Slab|L1"][0] == 2


def test_selection_set_pgs(built) -> None:
    scd, mesh, fem, _, _ = built
    assert mesh.selection_set_pgs == {
        "Columns": {"vertices": "Columns.vertices", "edges": "Columns.edges"},
        "All": {"vertices": "All.vertices", "edges": "All.edges", "faces": "All.faces"},
        "Master": {"vertices": "Master"},
    }
    for name, pgs in mesh.selection_set_pgs.items():
        nodes = set().union(*(_pg_nodes(fem, pg) for pg in pgs.values()))
        assert nodes == scd.set_nodes(name)
    # the slab-edge carriers (201, 202, 204, 205) carry the edge PG's nodes
    assert _pg_nodes(fem, "All.edges") == {1, 2, 3, 4, 5, 6, 7}


def test_condition_pgs(built) -> None:
    _, mesh, fem, _, _ = built
    assert mesh.condition_pgs == {
        1: {"vertices": "cond:1:base.vertices", "edges": "cond:1:base.edges"},
        2: {},                                     # interaction-only: no geometry
        3: {"faces": "cond:3:mass"},
    }
    assert _pg_nodes(fem, "cond:1:base.vertices") == {1}
    assert _pg_nodes(fem, "cond:1:base.edges") == {1, 2, 3}
    assert _pg_nodes(fem, "cond:3:mass") == {1, 2, 3, 4, 5, 6}


def test_diaphragm_carriers_own_their_nodes(built) -> None:
    _, mesh, fem, pg_entities, owned = built
    (d,) = mesh.diaphragms
    assert (d.condition, d.perp_dirn, d.master_node, d.slave_nodes) == (2, 3, 10, (2, 5))
    assert (d.master_pg, d.slave_pg) == ("rd:2:10:master", "rd:2:10:slaves")
    # g.constraints resolves nodes by ownership (getNodes), not connectivity
    (master_ent,) = pg_entities[d.master_pg][1]
    (slave_ent,) = pg_entities[d.slave_pg][1]
    assert owned[(0, master_ent)] == [10]
    assert owned[(0, slave_ent)] == [2, 5]
    # each node is owned by exactly one entity
    all_owned = [n for nodes in owned.values() for n in nodes]
    assert sorted(all_owned) == list(mesh.node_ids)


def test_diaphragm_resolves_through_g_constraints(scd) -> None:
    with apeGmsh(model_name="stko_rd", verbose=False) as g:
        mesh = build_mesh(g, scd)
        (d,) = mesh.diaphragms
        g.constraints.rigid_diaphragm(
            d.master_pg, d.slave_pg, master_point=tuple(scd.mesh.node_xyz(10)),
            plane_normal=(0.0, 0.0, 1.0), plane_tolerance=1e-3,
        )
        fem = g.mesh.queries.get_fem_data(dim=None)
    pairs = {(int(c.master_node), int(c.slave_node)) for c in fem.nodes.constraints.pairs()}
    assert pairs == {(10, 2), (10, 5)}


def test_carriers_are_synthetic_and_never_grouped(built) -> None:
    scd, mesh, fem, _, _ = built
    # 3 diaphragm points (master + 2 slaves) + 5 + 1 vertex points
    assert mesh.carrier_ids == range(303, 312)
    grouped = {e for eg in mesh.element_groups for e in eg.element_ids}
    assert grouped == {101, 102, 203}
    assert grouped.isdisjoint(mesh.carrier_ids)
    fem_ids = {int(e) for e in fem.elements.ids}
    assert fem_ids == {101, 102, 201, 202, 203, 204, 205, *mesh.carrier_ids}


def test_build_needs_an_empty_session(scd) -> None:
    with apeGmsh(model_name="stko_twice", verbose=False) as g:
        build_mesh(g, scd)
        with pytest.raises(ValueError, match="empty session"):
            build_mesh(g, scd)


def test_group_names_disambiguate_by_id() -> None:
    fake = SimpleNamespace(
        element_properties={1: SimpleNamespace(name="Shell"), 2: SimpleNamespace(name="Shell")},
        physical_properties={3: SimpleNamespace(name="Slab")},
    )
    x, y = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)
    assert _group_names(fake, [(1, 3, x), (2, 3, y), (2, 3, x)]) == {
        (1, 3, x): "Shell#1|Slab#3",
        (2, 3, x): "Shell#2|Slab#3|L2",
        (2, 3, y): "Shell#2|Slab#3|L1",
    }


# ── registry slice ────────────────────────────────────────────────────

def test_supported_document_has_nothing_to_report(scd) -> None:
    assert check_supported(scd) == []


def test_unsupported_items_are_all_listed(tmp_path: Path) -> None:
    import gmsh
    scd = read_scd(_write_scd(tmp_path / "bad.scd", unsupported=True))
    items = check_supported(scd)
    got = {(u.category, u.xobj_meta, u.ids, u.reason) for u in items}
    assert got == {
        ("interaction", "NE", (2,), "tier-only: not in this version"),
        ("interaction", "NN", (3,), "tier-only: not in this version"),
        ("interaction", "XX", (4,), "unknown STKO type"),
        ("mesh_element", "type(203)", (1,), "unknown STKO type"),
        ("mesh_element", "hex8(500)", (5,), "tier-only: not in this version"),
    }
    with apeGmsh(model_name="stko_bad", verbose=False) as g:
        with pytest.raises(UnsupportedSTKOTypes) as err:
            build_mesh(g, scd)
        assert g is not None and gmsh.model.getEntities() == []   # refused before touching the session
    assert len(err.value.items) == 5
    assert "5 STKO type(s) or option(s)" in str(err.value)


def test_session_nodes_predicts_the_built_session(built) -> None:
    scd, mesh, _, _, _ = built
    assert session_nodes(scd) == set(mesh.node_ids) == {1, 2, 3, 4, 5, 6, 7, 10}


def test_an_unreferenced_diaphragm_gets_no_carrier_and_is_refused_before_the_session(
        tmp_path: Path) -> None:
    """The carriers come from the same groups as the plan's pairs (referenced
    rigidDiaphragm conditions only), so the two cannot disagree after the session
    is built. Here no constraint pattern lists the diaphragm: STKO writes no
    rigidDiaphragm, its master (node 10, in selection set "Master") would be a
    free node, and the node guard refuses the document before the session."""
    import gmsh
    scd = read_scd(_write_scd(tmp_path / "unref.scd", diaphragm_referenced=False))
    items = check_supported(scd)
    assert [(u.category, u.xobj_meta, u.ids) for u in items] == [
        ("option", "mesh:session nodes outside the analysis model", (10,))]
    with apeGmsh(model_name="stko_unref", verbose=False) as g:
        with pytest.raises(UnsupportedSTKOTypes, match="free nodes"):
            build_mesh(g, scd)
        assert gmsh.model.getEntities() == []


def test_a_mesh_node_the_session_would_not_carry_is_refused(tmp_path: Path) -> None:
    """STKO writes every mesh node (write_node_not_assigned, ndf 3); a node no
    analysis element or referenced diaphragm uses would be missing from ours."""
    scd = read_scd(_write_scd(tmp_path / "orphan.scd", orphan_node=True))
    items = check_supported(scd)
    assert [(u.category, u.xobj_meta, u.ids) for u in items] == [
        ("option", "mesh:mesh nodes outside the session", (20,))]


# ── the real San Ramon documents (skipped unless available) ──────────

_SAMPLES = os.environ.get("APEGMSH_STKO_ORACLES")

#: case -> (nodes, analysis elements, element groups, diaphragm pairs)
_TIER1 = {
    "1A": (14003, 13704, 7, 0),
    "1B": (2547, 2456, 5, 360),
    "1C": (3979, 3704, 8, 840),
    "1D": (13459, 13160, 9, 0),
}


def _deck_elements(folder: Path) -> dict[int, tuple[int, ...]]:
    """``element`` lines of STKO's Tcl export (all partitions)."""
    nn = {"ASDShellQ4": 4, "elasticBeamColumn": 2, "forceBeamColumn": 2}
    out: dict[int, tuple[int, ...]] = {}
    for line in (folder / "elements.tcl").read_text().splitlines():
        t = line.split()
        if len(t) > 2 and t[0] == "element" and t[1] in nn:
            out[int(t[2])] = tuple(int(v) for v in t[3:3 + nn[t[1]]])
    return out


#: The files STKO's export writes (never a glob: a translated deck may sit next to them).
STKO_DECK_FILES = ("main.tcl", "nodes.tcl", "elements.tcl", "materials.tcl", "sections.tcl",
                   "definitions.tcl", "analysis_steps.tcl")


def _deck_rigid(folder: Path) -> set[tuple[int, int, int]]:
    out: set[tuple[int, int, int]] = set()
    for name in STKO_DECK_FILES:
        tcl = folder / name
        if not tcl.exists():
            continue
        for line in tcl.read_text().splitlines():
            t = line.split()
            if t and t[0] == "rigidDiaphragm":
                out |= {(int(t[1]), int(t[2]), int(s)) for s in t[3:]}
    return out


@pytest.mark.skipif(not _SAMPLES, reason="set APEGMSH_STKO_ORACLES to the stko-rev0-tcl folder")
@pytest.mark.parametrize("case", sorted(_TIER1))
def test_san_ramon_tier1_mesh(case: str) -> None:
    root = Path(_SAMPLES) / "Tier_1" / case
    scd = read_scd(root / f"{case}_TH_000.scd")
    n_nodes, n_elements, n_groups, n_pairs = _TIER1[case]
    assert check_supported(scd) == []
    with apeGmsh(model_name=f"stko_{case}", verbose=False) as g:
        mesh = build_mesh(g, scd)
        fem = g.mesh.queries.get_fem_data(dim=None)
        elements: dict[int, tuple[int, ...]] = {}
        for eg in mesh.element_groups:
            got = _pg_elements(fem, eg.pg)
            assert sorted(got) == list(eg.element_ids)
            elements |= got
        set_nodes = {
            name: set().union(*(_pg_nodes(fem, pg) for pg in pgs.values()))
            for name, pgs in mesh.selection_set_pgs.items()
        }
    analysis = scd.analysis_elements()
    assert set(mesh.node_ids) == session_nodes(scd)                       # the guard's prediction
    assert len(mesh.node_ids) == n_nodes                                  # rule M2
    assert set(mesh.node_ids) == {n for a in analysis.values() for n in a.element.nodes}
    assert len(elements) == n_elements                                    # rule M1
    assert set(elements) == {e for e, a in analysis.items() if a.interaction is None}
    assert len(mesh.element_groups) == n_groups                           # rule M6
    for e, conn in elements.items():                                      # rule M3
        assert conn == shell_node_order(scd, e)
    for name, nodes in set_nodes.items():
        assert nodes == scd.set_nodes(name), name
    pairs = {(d.perp_dirn, d.master_node, s) for d in mesh.diaphragms for s in d.slave_nodes}
    assert len(pairs) == n_pairs
    deck = root / ("input files" if case == "1A" else "campaign_sta2_rup1")
    if (deck / "elements.tcl").exists():                                 # STKO's own export
        assert elements == _deck_elements(deck)
        assert pairs == _deck_rigid(deck)


@pytest.mark.skipif(not _SAMPLES, reason="set APEGMSH_STKO_ORACLES to the stko-rev0-tcl folder")
def test_san_ramon_1b_reads_its_local_axes() -> None:
    scd = read_scd(Path(_SAMPLES) / "Tier_1" / "1B" / "1B_TH_000.scd")
    ax = scd.local_axes[1]
    assert ax.name == "rotatedWalls"
    assert (ax.origin, ax.x, ax.y) == ((0.0, -30000.0, 0.0), (0.0, -1.0, 0.0), (1.0, 0.0, 0.0))
    assert ax.type == "Rec"
