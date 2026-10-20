"""FEMData built from a results file's own MODEL group: element ``dim``.

Slice #2303. A ``.ladruno`` or ``.mpco`` MODEL group names each element
group by its OpenSees class. Before the fix the group's ``dim`` was a
class-name guess: ``LadrunoLST`` (``TOPOLOGY "custom"``, 6 nodes) became a
3-D wedge, and ``LadrunoCST`` in a 3-D model a line. That was a silent
wrong answer, not only a drawing one: ``select(dim=2)`` came back empty
and the static facet plot dropped the group.

The oracle is the response catalog's parent domain, read here on its own
(the width of each gauss-level class's ``natural_coords``; line families
are 1, layered shells 2), plus a hand table for the classes the guess got
wrong.
"""
from __future__ import annotations

import warnings

import h5py
import numpy as np
import pytest

from apeGmsh.mesh import _femdata_mpco_io
from apeGmsh.mesh._femdata_ladruno_io import read_fem_from_ladruno
from apeGmsh.mesh._femdata_mpco_io import (
    ElementDimContradictionWarning,
    ElementDimUnknownWarning,
    read_fem_from_mpco,
)
from apeGmsh.opensees import _response_catalog as rc


@pytest.fixture(autouse=True)
def _fresh_dim_warning_session(monkeypatch):
    """Each test starts a fresh warn-once session (#2312), and the
    process-wide registry is restored afterwards."""
    monkeypatch.setattr(_femdata_mpco_io, "_DIM_WARNED", set())


def _oracle_dims() -> dict[str, int]:
    dims: dict[str, int] = {}
    for (cls, _rule, _tok), layout in rc.RESPONSE_CATALOG.items():
        dims[cls] = int(layout.natural_coords.shape[1])
    for catalog in (rc.CUSTOM_RULE_CATALOG, rc.NODAL_FORCE_CATALOG,
                    rc.FIBER_CATALOG, rc.ZEROLENGTH_CATALOG):
        for cls, _tok in catalog:
            dims[cls] = 1
    for cls, _tok in rc.LAYER_CATALOG:
        dims[cls] = 2
    return dims


ORACLE = _oracle_dims()

# Hand pins: the parent domain of the classes the name guess got wrong,
# and of the self-describing simplices the card names.
HAND = {
    "LadrunoLST": 2, "LadrunoCST": 2, "SixNodeTri": 2, "BezierTri6": 2,
    "Tri31": 2, "BezierTet10": 3, "LadrunoBrick20": 3, "ShellDKGT": 2,
    "ZeroLength": 1, "ForceBeamColumn3d": 1,
}

# Node counts that reproduce what the file holds (and what fooled the guess).
_NPE = {
    "LadrunoLST": 6, "SixNodeTri": 6, "BezierTri6": 6,
    "LadrunoCST": 3, "Tri31": 3, "ShellDKGT": 3, "ShellNLDKGT": 3,
    "ASDShellT3": 3, "FourNodeTetrahedron": 4, "TenNodeTetrahedron": 10,
    "BezierTet10": 10, "LadrunoBrick20": 20, "Twenty_Node_Brick": 20,
    "EightNodeQuad": 8, "ShellMITC9": 9,
}
_NPE_BY_DIM = {1: 2, 2: 4, 3: 8}


def _npe(cls: str) -> int:
    return _NPE.get(cls, _NPE_BY_DIM[ORACLE[cls]])


def _write_model(f: h5py.File, classes, *, ladruno: bool, ndm: int,
                 topology: str = "custom") -> h5py.Group:
    model = f.create_group("MODEL")
    nodes = model.create_group("NODES")
    n_nodes = 27
    nodes["ID"] = np.arange(1, n_nodes + 1, dtype=np.int64).reshape(-1, 1)
    nodes["COORDINATES"] = np.random.default_rng(0).random((n_nodes, ndm))
    elements = model.create_group("ELEMENTS")
    for tag, (cls, npe) in enumerate(classes, start=1):
        conn = np.array([[tag, *range(1, npe + 1)]], dtype=np.int64)
        if ladruno:
            g = elements.create_group(f"{tag}-{cls}[0:0]")
            g["CONNECTIVITY"] = conn
            g.attrs["TOPOLOGY"] = topology
        else:
            elements[f"{tag}-{cls}[1:0]"] = conn
    return model


def _read(tmp_path, classes, *, ladruno: bool, ndm: int, **kw):
    path = tmp_path / ("m.ladruno" if ladruno else "m.mpco")
    with h5py.File(path, "w") as f:
        model = _write_model(f, classes, ladruno=ladruno, ndm=ndm, **kw)
        reader = read_fem_from_ladruno if ladruno else read_fem_from_mpco
        return reader(model)


def test_hand_pins_agree_with_catalog():
    for cls, dim in HAND.items():
        assert ORACLE[cls] == dim, cls


@pytest.mark.parametrize("ndm", [2, 3])
@pytest.mark.parametrize("ladruno", [True, False], ids=["ladruno", "mpco"])
def test_every_catalogued_class_gets_its_parent_dim(tmp_path, ladruno, ndm):
    classes = [(cls, _npe(cls)) for cls in sorted(ORACLE)]
    with warnings.catch_warnings():
        warnings.simplefilter("error", ElementDimUnknownWarning)
        fem = _read(tmp_path, classes, ladruno=ladruno, ndm=ndm)
    got = {g.element_type.gmsh_name: int(g.element_type.dim)
           for g in fem.elements}
    assert got.keys() == ORACLE.keys()
    wrong = {c: (got[c], ORACLE[c]) for c in ORACLE if got[c] != ORACLE[c]}
    assert not wrong, f"(got, parent dim): {wrong}"


@pytest.mark.parametrize("ladruno", [True, False], ids=["ladruno", "mpco"])
def test_lst_group_is_selected_and_drawn_as_a_surface(tmp_path, ladruno):
    """The downstream symptom: dim selection and the facet plot."""
    from apeGmsh.results.plot._facets import extract_facets

    fem = _read(tmp_path, [("LadrunoLST", 6), ("LadrunoCST", 3)],
                ladruno=ladruno, ndm=3)
    assert len(fem.elements.select(dim=2).ids) == 2
    assert len(fem.elements.select(dim=1).ids) == 0
    assert len(fem.elements.select(dim=3).ids) == 0
    tris, segs = extract_facets(fem)
    assert len(tris) == 2 and len(segs) == 0


@pytest.mark.parametrize("ladruno", [True, False], ids=["ladruno", "mpco"])
def test_viewer_substrate_draws_lst_and_cst_as_triangles(tmp_path, ladruno):
    """The substrate picks a non-Gmsh code's VTK cell by ``(dim, npe)``.

    With the dim=3 guess a LadrunoLST became a 6-point VTK_WEDGE, and the
    #2302 discrete contour left those cells unpainted. LadrunoCST in a 3-D
    model became a VTK_LINE. Both are VTK_TRIANGLE (5), drawn from the
    three corner nodes.
    """
    pytest.importorskip("pyvista")
    from apeGmsh.viewers.data import ViewerData
    from apeGmsh.viewers.scene.fem_scene import build_fem_scene

    fem = _read(tmp_path, [("LadrunoLST", 6), ("LadrunoCST", 3)],
                ladruno=ladruno, ndm=3)
    scene = build_fem_scene(ViewerData.from_fem(fem))
    assert not scene.skipped_types
    assert list(scene.grid.celltypes) == [5, 5]
    corners = scene.grid.cell_connectivity.reshape(2, 3)
    assert (scene.node_ids[corners] == [[1, 2, 3], [1, 2, 3]]).all()


@pytest.mark.parametrize("ladruno", [True, False], ids=["ladruno", "mpco"])
def test_uncatalogued_undeclared_class_is_named_in_a_warning(
        tmp_path, ladruno):
    with pytest.warns(ElementDimUnknownWarning, match="MadeUpPanel"):
        fem = _read(tmp_path, [("MadeUpPanel", 4), ("Truss", 2)],
                    ladruno=ladruno, ndm=3)
    assert {g.element_type.gmsh_name for g in fem.elements} == {
        "MadeUpPanel", "Truss"}


def test_uncatalogued_class_with_declared_topology_is_silent(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error", ElementDimUnknownWarning)
        fem = _read(tmp_path, [("MadeUpPanel", 4)], ladruno=True, ndm=3,
                    topology="quad")
    assert int(next(iter(fem.elements)).element_type.dim) == 2


def test_declared_topology_contradicting_catalog_warns_catalog_wins(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error", ElementDimUnknownWarning)
        with pytest.warns(ElementDimContradictionWarning,
                          match=r"LadrunoLST \(declared 3, catalog 2\)"):
            fem = _read(tmp_path, [("LadrunoLST", 6)], ladruno=True, ndm=3,
                        topology="wedge")
    assert int(next(iter(fem.elements)).element_type.dim) == 2


# ---------------------------------------------------------------------------
# #2312: one warning per fact per session, never one per read or per group
# ---------------------------------------------------------------------------


def _write(tmp_path, classes, *, ladruno: bool, ndm: int = 3, **kw):
    path = tmp_path / ("m.ladruno" if ladruno else "m.mpco")
    with h5py.File(path, "w") as f:
        _write_model(f, classes, ladruno=ladruno, ndm=ndm, **kw)
    return path


def _read_path(path, *, ladruno: bool):
    reader = read_fem_from_ladruno if ladruno else read_fem_from_mpco
    with h5py.File(path, "r") as f:
        return reader(f["MODEL"])


def _recorded(path, *, ladruno: bool, reads: int):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        for _ in range(reads):
            _read_path(path, ladruno=ladruno)
    return [w for w in rec if issubclass(
        w.category, (ElementDimUnknownWarning, ElementDimContradictionWarning))]


def test_contradiction_is_not_an_unknown_dim_warning():
    assert not issubclass(ElementDimContradictionWarning,
                          ElementDimUnknownWarning)


@pytest.mark.parametrize("ladruno", [True, False], ids=["ladruno", "mpco"])
def test_repeated_reads_warn_an_uncatalogued_class_once(tmp_path, ladruno):
    """ShellMITC3 (uncatalogued, guessed right) and a made-up class: three
    reads of one file give one warning naming both, not three."""
    path = _write(tmp_path, [("ShellMITC3", 3), ("MadeUpPanel", 4)],
                  ladruno=ladruno)
    got = _recorded(path, ladruno=ladruno, reads=3)
    assert [w.category for w in got] == [ElementDimUnknownWarning]
    msg = str(got[0].message)
    assert "ShellMITC3 (dim=2)" in msg and "MadeUpPanel" in msg


@pytest.mark.parametrize("ladruno", [True, False], ids=["ladruno", "mpco"])
def test_a_later_read_names_only_the_new_class(tmp_path, ladruno):
    first = _write(tmp_path, [("ShellMITC3", 3)], ladruno=ladruno)
    assert len(_recorded(first, ladruno=ladruno, reads=1)) == 1
    sub = tmp_path / "second"
    sub.mkdir()
    second = _write(sub, [("ShellMITC3", 3), ("MadeUpPanel", 4)],
                    ladruno=ladruno)
    got = _recorded(second, ladruno=ladruno, reads=2)
    assert len(got) == 1
    assert "MadeUpPanel" in str(got[0].message)
    assert "ShellMITC3" not in str(got[0].message)


def test_contradiction_warns_once_per_class_not_per_group_or_read(tmp_path):
    """Two LadrunoLST groups declaring ``wedge``, read twice: one warning."""
    path = _write(tmp_path, [("LadrunoLST", 6), ("LadrunoLST", 6)],
                  ladruno=True, topology="wedge")
    got = _recorded(path, ladruno=True, reads=2)
    assert [w.category for w in got] == [ElementDimContradictionWarning]
    assert str(got[0].message).count("LadrunoLST") == 1


@pytest.mark.parametrize("ladruno", [True, False], ids=["ladruno", "mpco"])
def test_two_groups_of_one_class_guessing_two_dims_are_both_named(
        tmp_path, ladruno):
    """FooThing with 2 and 4 nodes in a 3-D model guesses dims 1 and 3:
    both are named and both are spent, not only the last group's."""
    path = _write(tmp_path, [("FooThing", 2), ("FooThing", 4)],
                  ladruno=ladruno)
    got = _recorded(path, ladruno=ladruno, reads=2)
    assert len(got) == 1
    msg = str(got[0].message)
    assert "FooThing (dim=1)" in msg and "FooThing (dim=3)" in msg
    fem = _read_path(path, ladruno=ladruno)
    assert sorted(int(g.element_type.dim) for g in fem.elements) == [1, 3]


def test_two_groups_of_one_class_contradicting_two_ways_are_both_named(
        tmp_path):
    path = _write(tmp_path, [("LadrunoLST", 6), ("LadrunoLST", 6)],
                  ladruno=True)
    with h5py.File(path, "r+") as f:
        f["MODEL/ELEMENTS/1-LadrunoLST[0:0]"].attrs["TOPOLOGY"] = "wedge"
        f["MODEL/ELEMENTS/2-LadrunoLST[0:0]"].attrs["TOPOLOGY"] = "line"
    got = _recorded(path, ladruno=True, reads=2)
    assert [w.category for w in got] == [ElementDimContradictionWarning]
    msg = str(got[0].message)
    assert "LadrunoLST (declared 3, catalog 2)" in msg
    assert "LadrunoLST (declared 1, catalog 2)" in msg


@pytest.mark.parametrize("ladruno", [True, False], ids=["ladruno", "mpco"])
def test_a_raised_warning_is_not_spent(tmp_path, ladruno):
    """Under an ``error`` filter every read raises: a warning that became
    an exception was not delivered, so it is not marked as warned."""
    path = _write(tmp_path, [("MadeUpPanel", 4)], ladruno=ladruno)
    for _ in range(2):
        with warnings.catch_warnings():
            warnings.simplefilter("error", ElementDimUnknownWarning)
            with pytest.raises(ElementDimUnknownWarning, match="MadeUpPanel"):
                _read_path(path, ladruno=ladruno)
