"""Native files written before the simplex frame change read correct GPs.

The native writers copy the response catalog's ``natural_coords`` into
``_natural_coords``. The catalog used to hold the triangle and tet rules
node-0-first; it now holds them in gmsh reference coordinates, which is
what every consumer of ``GaussSlab.natural_coords`` evaluates. A native
file written before the change therefore stored the old rows, and read on
main with every multi-point simplex Gauss point misplaced: a Tet10's
extrapolated corner values came back rotated by one node, silently.

The fixtures in ``tests/fixtures/native_legacy_simplex/`` were written by
the writers as they stood before the change (``generate.py`` there says
how), through both native paths: ``DomainCapture`` of a live solve and the
recorder transcoder on the engine's own ``.out`` strains. Every element is
loaded with ``u_i = x_i^2 / 2``, so the strain it records at a Gauss point
IS that point's world position.

The oracles come from the engine, never from the catalog:

* the strain each file records at each Gauss point (``eps_ii = x_i``);
* ``GLOBAL_GP_COORDS`` that the fork's ``.ladruno`` recorder wrote for the
  same elements on the same geometry (``tests/fixtures/simplex_gp``).

The triangle shells (ShellDKGT/NLDKGT) have no engine oracle; their old
rows are checked against the closed-form frame change instead.
"""
from __future__ import annotations

import shutil
import warnings
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import h5py
import numpy as np
import pytest

from apeGmsh.fem._shape_functions import get_shape_functions
from apeGmsh.opensees._response_catalog import (
    ELE_TAG_BezierTri6,
    ELE_TAG_ShellDKGT,
    IntRule,
    lookup,
)
from apeGmsh.results._gauss_extrapolation import (
    extrapolate_gauss_slab_to_nodes,
)
from apeGmsh.results.readers._native import NativeReader

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
LEGACY = FIXTURES / "native_legacy_simplex"
SIMPLEX_GP = FIXTURES / "simplex_gp"
TOL = 1e-12

#: The geometry every fixture element sits on (generate.py).
CORNERS = {
    "tri": np.array([[0.1, 0.2], [2.3, 0.4], [0.6, 1.8]]),
    "tet": np.array([[0.1, 0.2, 0.0], [2.3, 0.1, 0.4],
                     [0.5, 1.7, 0.2], [0.3, 0.4, 1.9]]),
}
_AXES = {"tri": ("xx", "yy"), "tet": ("xx", "yy", "zz")}
_LINEAR_CODE = {"tri": 2, "tet": 4}

#: (fixture, stem, element id, class). ``global`` marks the classes whose
#: element in ``simplex_gp/<stem>.ladruno`` carries GLOBAL_GP_COORDS.
CASES = [
    ("capture_tri.h5", "tri", 1, "Tri31"),
    ("capture_tri.h5", "tri", 2, "SixNodeTri"),
    ("capture_tri.h5", "tri", 4, "LadrunoLST"),
    ("capture_tet.h5", "tet", 1, "FourNodeTetrahedron"),
    ("capture_tet.h5", "tet", 2, "TenNodeTetrahedron"),
    ("capture_tet.h5", "tet", 3, "BezierTet10"),
    ("transcoded_tri.h5", "tri", 2, "SixNodeTri"),
    ("transcoded_tri.h5", "tri", 4, "LadrunoLST"),
    ("transcoded_tet.h5", "tet", 2, "TenNodeTetrahedron"),
    ("transcoded_tet.h5", "tet", 3, "BezierTet10"),
]
_IDS = [f"{c[0].split('.')[0]}-{c[3]}" for c in CASES]
_GLOBAL = {"Tri31", "FourNodeTetrahedron", "TenNodeTetrahedron",
           "BezierTet10"}
#: The strain at a linear element's one Gauss point is the field's
#: gradient there only up to the element's own approximation.
_STRAIN_ORACLE = {"SixNodeTri", "LadrunoLST", "TenNodeTetrahedron",
                  "BezierTet10"}


def _read(path: Path, component: str, eid: int | None = None) -> Any:
    with NativeReader(path) as r:
        sid = r.stages()[0].id
        ids = None if eid is None else np.array([eid])
        return r.read_gauss(sid, component, element_ids=ids)


def _world(natural: np.ndarray, stem: str) -> np.ndarray:
    """Natural coords → world through the gmsh linear shape functions."""
    n_fn = get_shape_functions(_LINEAR_CODE[stem])[0]
    return n_fn(np.asarray(natural)) @ CORNERS[stem]


def _global_gp(stem: str, eid: int, n_gp: int) -> np.ndarray:
    """``GLOBAL_GP_COORDS`` of element ``eid`` in ``simplex_gp``."""
    ndim = len(_AXES[stem])
    with h5py.File(SIMPLEX_GP / f"{stem}.ladruno", "r") as f:
        for grp in f["MODEL_STAGE[1]/MODEL/ELEMENTS"].values():
            if int(grp["CONNECTIVITY"][0, 0]) == eid:
                g = np.asarray(grp["GLOBAL_GP_COORDS"][0], dtype=np.float64)
                return g.reshape(n_gp, -1)[:, :ndim]
    raise KeyError(eid)


# ---------------------------------------------------------------------------
# Older files read every Gauss point where the engine put it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,stem,eid,cls", CASES, ids=_IDS)
def test_legacy_file_places_gauss_points_where_the_engine_does(
    name: str, stem: str, eid: int, cls: str,
) -> None:
    path = LEGACY / name
    with warnings.catch_warnings():
        warnings.simplefilter("error")       # an exact match never warns
        slabs = [_read(path, f"strain_{a}", eid) for a in _AXES[stem]]
    world = _world(slabs[0].natural_coords, stem)
    checked = 0
    if cls in _STRAIN_ORACLE:
        recorded = np.stack([np.asarray(s.values)[-1] for s in slabs], axis=1)
        np.testing.assert_allclose(world, recorded, rtol=0, atol=TOL,
                                   err_msg=f"{name} {cls} vs its strain")
        checked += 1
    if cls in _GLOBAL:
        want = _global_gp(stem, eid, world.shape[0])
        np.testing.assert_allclose(world, want, rtol=0, atol=TOL,
                                   err_msg=f"{name} {cls} vs GLOBAL_GP_COORDS")
        checked += 1
    assert checked


def _gmsh_fem(stem: str, eid: int, code: int, npe: int,
              edges: "list[tuple[int, int]]") -> Any:
    """A FEMData stand-in holding one gmsh-coded quadratic element."""
    corners = CORNERS[stem]
    pts = np.vstack([corners] + [(corners[a] + corners[b]) / 2.0
                                 for a, b in edges])
    xyz = np.zeros((pts.shape[0], 3))
    xyz[:, :pts.shape[1]] = pts
    ids = np.arange(1, pts.shape[0] + 1, dtype=np.int64)

    class _Group:
        element_type = SimpleNamespace(
            code=code, dim=len(_AXES[stem]), npe=npe, order=2,
            gmsh_name="gmsh",
        )
        connectivity = ids[None, :]

        def __init__(self) -> None:
            self.ids = np.array([eid])

        def __len__(self) -> int:
            return 1

    return SimpleNamespace(
        nodes=SimpleNamespace(ids=ids, coords=xyz), elements=[_Group()],
    )


_TET10_EDGES = [(0, 1), (1, 2), (2, 0), (0, 3), (2, 3), (1, 3)]
_TRI6_EDGES = [(0, 1), (1, 2), (2, 0)]


@pytest.mark.parametrize("name,stem,eid,cls,code,npe,edges", [
    ("capture_tet.h5", "tet", 2, "TenNodeTetrahedron", 11, 10, _TET10_EDGES),
    ("transcoded_tet.h5", "tet", 2, "TenNodeTetrahedron", 11, 10,
     _TET10_EDGES),
    ("capture_tet.h5", "tet", 3, "BezierTet10", 11, 10, _TET10_EDGES),
    ("transcoded_tet.h5", "tet", 3, "BezierTet10", 11, 10, _TET10_EDGES),
    ("capture_tri.h5", "tri", 2, "SixNodeTri", 9, 6, _TRI6_EDGES),
    ("transcoded_tri.h5", "tri", 4, "LadrunoLST", 9, 6, _TRI6_EDGES),
], ids=lambda v: v if isinstance(v, str) and v.endswith(".h5") else None)
def test_legacy_file_extrapolates_to_its_own_corners(
    name: str, stem: str, eid: int, cls: str, code: int, npe: int,
    edges: "list[tuple[int, int]]",
) -> None:
    """The recorded ``strain_xx`` at each GP is the GP's x. Extrapolated to
    a gmsh-coded element it must give each corner its own x; read through
    the old rows it gave the corners' x rotated by one node."""
    slab = _read(LEGACY / name, "strain_xx", eid)
    fem = _gmsh_fem(stem, eid, code, npe, edges)
    node_ids, nodal = extrapolate_gauss_slab_to_nodes(slab, fem)
    assert node_ids.size == CORNERS[stem].shape[0]
    pos = {int(n): k for k, n in enumerate(fem.nodes.ids)}
    want = fem.nodes.coords[[pos[int(n)] for n in node_ids], 0]
    np.testing.assert_allclose(nodal[-1], want, rtol=0, atol=1e-10,
                               err_msg=f"{name} {cls}")


# ---------------------------------------------------------------------------
# The detection rule: older rows map, current rows and other classes pass
# ---------------------------------------------------------------------------


#: The classes whose stored rows changed: their catalog rule went from
#: node-0-first rows to gmsh rows (a catalog diff across the change).
#: Single-point centroids are listed too: Tri31's moved by one ulp.
CHANGED = {"Tri31", "LadrunoCST", "SixNodeTri", "LadrunoLST", "ShellDKGT",
           "ShellNLDKGT", "FourNodeTetrahedron", "TenNodeTetrahedron",
           "BezierTet10"}


def test_the_reader_maps_exactly_the_classes_whose_rows_changed() -> None:
    from apeGmsh.results.readers._native import _SIMPLEX_GP_ROWS

    assert {cls for cls, _, _ in _SIMPLEX_GP_ROWS.values()} == CHANGED
    for cls, legacy, current in _SIMPLEX_GP_ROWS.values():
        full = np.hstack([legacy, 1.0 - legacy.sum(axis=1, keepdims=True)])
        np.testing.assert_allclose(current, full[:, 1:], rtol=0, atol=1e-15,
                                   err_msg=cls)


def _group(f: h5py.File, cls: str) -> h5py.Group:
    root = f["stages/stage_0/partitions/partition_0/elements/gauss_points"]
    (grp,) = [g for k, g in root.items() if cls in k]
    return grp


def _rewrite_rows(tmp_path: Path, name: str, cls: str, rows: np.ndarray,
                  class_tag: int | None = None) -> Path:
    """A copy of fixture ``name`` with ``cls``'s group's rows replaced (and
    its components resized to the new GP count)."""
    target = tmp_path / name
    shutil.copyfile(LEGACY / name, target)
    with h5py.File(target, "r+") as f:
        grp = _group(f, cls)
        del grp["_natural_coords"]
        grp.create_dataset("_natural_coords", data=rows)
        for comp in [k for k in grp if not k.startswith("_")]:
            old = np.asarray(grp[comp])
            if old.shape[2] != rows.shape[0]:
                del grp[comp]
                grp.create_dataset(
                    comp, data=np.zeros(old.shape[:2] + (rows.shape[0],)),
                )
        if class_tag is not None:
            grp.attrs["class_tag"] = np.int64(class_tag)
    return target


@pytest.mark.parametrize("name,stem,eid,cls,rule", [
    ("capture_tri.h5", "tri", 2, "SixNodeTri", IntRule.Triangle_GL_2),
    ("transcoded_tet.h5", "tet", 2, "TenNodeTetrahedron", IntRule.Tet_GL_2),
    ("capture_tri.h5", "tri", 1, "Tri31", IntRule.Triangle_GL_1),
])
def test_current_rows_read_unchanged(
    tmp_path: Path, name: str, stem: str, eid: int, cls: str, rule: int,
) -> None:
    """A file written after the change holds the catalog's table, which the
    writers copy verbatim; it reads back bit for bit, with no warning."""
    rows = lookup(cls, rule, "strain").natural_coords
    path = _rewrite_rows(tmp_path, name, cls, rows)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        slab = _read(path, "strain_xx", eid)
    np.testing.assert_array_equal(slab.natural_coords, rows)


def test_other_classes_read_their_rows_unchanged(tmp_path: Path) -> None:
    """Only the classes older writers stored node-0-first are mapped. A
    BezierTri6 group holding the very rows a SixNodeTri group is mapped
    from reads them as they are, with no warning."""
    legacy = np.array([[2 / 3, 1 / 6], [1 / 6, 2 / 3], [1 / 6, 1 / 6]])
    path = _rewrite_rows(tmp_path, "capture_tri.h5", "SixNodeTri", legacy,
                         class_tag=ELE_TAG_BezierTri6)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        slab = _read(path, "strain_xx", 2)
    np.testing.assert_array_equal(slab.natural_coords, legacy)


def test_a_gmsh_frame_class_is_never_mapped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ``coord_system`` gate, not a coincidence of values, keeps a
    gmsh-frame class unmapped. Give BezierTri6 (``simplex_tri_gmsh``) a
    catalog table equal to an older table's frame change: the very shape
    a node-0-first class is mapped by. Its stored older-looking rows are
    still its own, and read as they are."""
    from apeGmsh.results.readers import _native

    current = _native._node0_rows_to_gmsh(_native._LEGACY_TRI_GL_2)
    key = ("BezierTri6", IntRule.Triangle_GL_2, "strain")
    layout = _native.RESPONSE_CATALOG[key]
    assert layout.coord_system == "simplex_tri_gmsh"
    catalog = dict(_native.RESPONSE_CATALOG)
    catalog[key] = replace(layout, natural_coords=current)
    monkeypatch.setattr(_native, "RESPONSE_CATALOG", catalog)
    rows = _native._simplex_gp_rows()
    assert ELE_TAG_BezierTri6 not in rows
    monkeypatch.setattr(_native, "_SIMPLEX_GP_ROWS", rows)

    legacy = _native._LEGACY_TRI_GL_2
    path = _rewrite_rows(tmp_path, "capture_tri.h5", "SixNodeTri", legacy,
                         class_tag=ELE_TAG_BezierTri6)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        slab = _read(path, "strain_xx", 2)
    np.testing.assert_array_equal(slab.natural_coords, legacy)


def test_triangle_shell_legacy_rows_map_by_the_frame_change(
    tmp_path: Path,
) -> None:
    """ShellDKGT has no engine oracle here. Its old rows were node-0-first
    (full row ``(L0, L1, 1 - L0 - L1)``); gmsh ``(u, v)`` is the weights of
    nodes 1 and 2, the closed form the reader must reproduce."""
    old = np.array([[1 / 3, 1 / 3], [1 / 5, 3 / 5], [3 / 5, 1 / 5],
                    [1 / 5, 1 / 5]])
    path = _rewrite_rows(tmp_path, "capture_tri.h5", "SixNodeTri", old,
                         class_tag=ELE_TAG_ShellDKGT)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        slab = _read(path, "strain_xx", 2)
    full = np.hstack([old, 1.0 - old.sum(axis=1, keepdims=True)])
    np.testing.assert_allclose(slab.natural_coords, full[:, 1:], rtol=0,
                               atol=1e-15)


@pytest.mark.parametrize("name,eid,cls", [
    ("capture_tet.h5", 2, "TenNodeTetrahedron"),
    ("transcoded_tri.h5", 2, "SixNodeTri"),
])
def test_unrecognised_rows_warn_once_and_read_unchanged(
    tmp_path: Path, name: str, eid: int, cls: str,
) -> None:
    """Rows that match neither table are not guessed at: they read as
    stored, and one warning names the class, the file and the fix."""
    from apeGmsh.results.readers._native import (
        LegacySimplexGaussCoordsWarning,
    )

    with h5py.File(LEGACY / name, "r") as f:
        stored = np.asarray(_group(f, cls)["_natural_coords"])
    odd = stored[::-1].copy()            # a permutation neither table has
    path = _rewrite_rows(tmp_path, name, cls, odd)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with NativeReader(path) as r:
            sid = r.stages()[0].id
            slabs = [r.read_gauss(sid, c, element_ids=np.array([eid]))
                     for c in ("strain_xx", "strain_yy", "strain_xx")]
    ours = [w for w in caught
            if issubclass(w.category, LegacySimplexGaussCoordsWarning)]
    assert len(ours) == 1, [str(w.message) for w in caught]
    msg = str(ours[0].message)
    assert cls in msg and path.name in msg
    assert "Re-capture" in msg and "re-transcode" in msg
    for slab in slabs:
        np.testing.assert_array_equal(slab.natural_coords, odd)


def test_a_group_the_read_filters_out_never_warns(tmp_path: Path) -> None:
    """The warning is about the Gauss points a read returns. A read whose
    ``element_ids`` exclude the unplaceable TenNodeTetrahedron group is
    silent, and leaves the once-per-class warning for the read that does
    return it."""
    from apeGmsh.results.readers._native import (
        LegacySimplexGaussCoordsWarning,
    )

    name, cls = "capture_tet.h5", "TenNodeTetrahedron"
    with h5py.File(LEGACY / name, "r") as f:
        stored = np.asarray(_group(f, cls)["_natural_coords"])
    path = _rewrite_rows(tmp_path, name, cls, stored[::-1].copy())
    with NativeReader(path) as r:
        sid = r.stages()[0].id
        with warnings.catch_warnings():
            warnings.simplefilter("error", LegacySimplexGaussCoordsWarning)
            other = r.read_gauss(sid, "strain_xx", element_ids=np.array([1]))
        assert set(other.element_index.tolist()) == {1}
        with pytest.warns(LegacySimplexGaussCoordsWarning, match=cls):
            r.read_gauss(sid, "strain_xx", element_ids=np.array([2]))
