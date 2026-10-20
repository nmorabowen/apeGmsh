"""GP world coordinates with a result file's own FEMData.

A FEMData read from a ``.ladruno`` or MPCO file's ``MODEL`` group keys
each class by ``-class_tag``, which names no topology. The placement
mapped only linear classes to a gmsh shape, so every P2 simplex
(SixNodeTri, LadrunoLST, TenNodeTetrahedron) took the centroid+bbox
approximation, about half an element off. On MPCO the Bezier classes did
too, because that FEMData calls them order 1.

The oracle is the engine's. The fixtures are one fork solve under
``u_i = x_i^2 / 2`` (``tests/results/test_gauss_simplex_coords.py``,
``SOLVE``), so a quadratic element's recorded strain ``eps_ii`` at a
Gauss point IS that point's coordinate ``x_i``. Where the fork also
records ``GLOBAL_GP_COORDS`` (the ``.ladruno`` tets and linear classes),
that is compared directly.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from apeGmsh.results._gauss_world_coords import (
    WarnGaussCoordsApproximate,
    compute_global_coords_from_arrays,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "simplex_gp"
AXES = {"tri": ("xx", "yy"), "tet": ("xx", "yy", "zz")}
TOL = 1e-12


def _reader(path: Path):
    from apeGmsh.results.readers._ladruno import LadrunoReader
    from apeGmsh.results.readers._mpco import MPCOReader

    return (LadrunoReader(path) if path.suffix == ".ladruno"
            else MPCOReader(path))


def _read(path: Path, component: str, eids=None):
    """The file's own FEMData and a Gauss slab of ``component``."""
    with warnings.catch_warnings():
        # The readers' own provenance warnings are not under test here.
        warnings.simplefilter("ignore")
        with _reader(path) as r:
            fem = r.fem()
            slab = r.read_gauss(
                r.stages()[0].id, component,
                element_ids=None if eids is None else np.asarray(eids),
            )
    return fem, slab


def _group_code(fem, eid: int) -> int:
    for g in fem.elements:
        if eid in {int(i) for i in g.ids}:
            return int(g.element_type.code)
    raise KeyError(eid)


def _place(slab, fem) -> np.ndarray:
    """``slab.global_coords(fem)``, failing on any degradation warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", WarnGaussCoordsApproximate)
        return slab.global_coords(fem)


@pytest.mark.parametrize("ext", ["ladruno", "mpco"])
@pytest.mark.parametrize("stem,eid,cls", [
    ("tri", 2, "SixNodeTri"),
    ("tri", 3, "BezierTri6"),
    ("tri", 4, "LadrunoLST"),
    ("tet", 2, "TenNodeTetrahedron"),
    ("tet", 3, "BezierTet10"),
])
def test_p2_simplex_gp_world_coords_are_the_engines(
    ext: str, stem: str, eid: int, cls: str,
) -> None:
    """Each P2 simplex GP sits where the engine integrated, with no
    degradation warning. Before the fix the Lagrange classes (and, on
    MPCO, the Bezier ones) were off by 0.47 to 0.63."""
    path = FIXTURES / f"{stem}.{ext}"
    world = None
    for axis, comp in enumerate(AXES[stem]):
        fem, slab = _read(path, f"strain_{comp}", [eid])
        assert _group_code(fem, eid) < 0, "not the file-coded FEMData"
        assert slab.element_index.size > 1, f"{cls} is not multi-point"
        if world is None:
            world = _place(slab, fem)
        np.testing.assert_allclose(
            world[:, axis], np.asarray(slab.values)[-1], rtol=0, atol=TOL,
            err_msg=f"{cls} {ext} axis {axis}",
        )


def _global_gp(path: Path, eid: int, n_gp: int) -> np.ndarray:
    """The fork's ``GLOBAL_GP_COORDS`` for one element, ``(n_gp, ndm)``."""
    with h5py.File(path, "r") as f:
        for grp in f["MODEL_STAGE[1]/MODEL/ELEMENTS"].values():
            if int(grp["CONNECTIVITY"][0, 0]) != eid:
                continue
            if "GLOBAL_GP_COORDS" not in grp:
                break
            return np.asarray(
                grp["GLOBAL_GP_COORDS"][0], dtype=np.float64,
            ).reshape(n_gp, -1)
    raise KeyError(f"{path.name}: no GLOBAL_GP_COORDS for element {eid}")


@pytest.mark.parametrize("stem,eid,cls", [
    ("tet", 2, "TenNodeTetrahedron"),
    ("tet", 3, "BezierTet10"),
    ("tet", 1, "FourNodeTetrahedron"),
    ("tet", 4, "Brick"),
    ("tri", 1, "Tri31"),
    ("tri", 5, "FourNodeQuad"),
])
def test_gp_world_coords_match_the_forks_recorded_positions(
    stem: str, eid: int, cls: str,
) -> None:
    """Against the GP positions the fork itself records. The linear
    classes guard the table this replaced."""
    path = FIXTURES / f"{stem}.ladruno"
    fem, slab = _read(path, "stress_xx", [eid])
    world = _place(slab, fem)
    want = _global_gp(path, eid, slab.element_index.size)
    np.testing.assert_allclose(world[:, :want.shape[1]], want, rtol=0,
                               atol=1e-10, err_msg=cls)


@pytest.mark.parametrize("ext", ["ladruno", "mpco"])
@pytest.mark.parametrize("stem", ["tri", "tet"])
def test_whole_file_slab_places_every_class_silently(
    ext: str, stem: str,
) -> None:
    """One slab over every class in the file: placed per element, no
    warning, and each P2 GP at its engine position."""
    path = FIXTURES / f"{stem}.{ext}"
    p2 = (2, 3, 4) if stem == "tri" else (2, 3)
    fem, slab = _read(path, f"strain_{AXES[stem][0]}")
    world = _place(slab, fem)
    keep = np.isin(np.asarray(slab.element_index), p2)
    assert np.unique(np.asarray(slab.element_index)[keep]).size == len(p2)
    np.testing.assert_allclose(world[keep, 0],
                               np.asarray(slab.values)[-1, keep],
                               rtol=0, atol=TOL)


# =====================================================================
# Synthetic: curved elements and the degraded paths
# =====================================================================

class _Group:
    def __init__(self, et, eids, conn) -> None:
        self.element_type = et
        self.ids = np.asarray(eids, dtype=np.int64)
        self.connectivity = np.asarray(conn, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.ids)


def _fem(coords, *, code, dim, npe, name="", order=1):
    coords = np.asarray(coords, dtype=np.float64)
    tags = np.arange(1, coords.shape[0] + 1, dtype=np.int64)
    et = SimpleNamespace(code=code, dim=dim, npe=npe, order=order,
                         gmsh_name=name)
    return SimpleNamespace(
        nodes=SimpleNamespace(ids=tags, coords=coords),
        elements=[_Group(et, [7], tags[None, :npe])],
    )


# A P2 tet with every mid-edge node pushed off its chord, in gmsh order
# (edges 0-1, 1-2, 2-0, 3-0, 3-2, 3-1).
_CORNERS = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2]], float)
_EDGES = [(0, 1), (1, 2), (2, 0), (3, 0), (3, 2), (3, 1)]
_BUMP = np.array([[0, -.2, .1], [.15, .15, 0], [-.2, 0, .1],
                  [.1, -.1, 0], [0, .2, .1], [.2, 0, .1]])
_TET10 = np.vstack([_CORNERS, [(_CORNERS[i] + _CORNERS[j]) / 2 + b
                               for (i, j), b in zip(_EDGES, _BUMP)]])
_A, _B = 0.1381966011250105, 0.5854101966249685
_TET_GP4 = np.array([[_A, _A, _A], [_B, _A, _A], [_A, _B, _A], [_A, _A, _B]])


def test_curved_p2_tet_matches_the_gmsh_coded_placement() -> None:
    """A curved TenNodeTetrahedron, file-coded, lands where the same
    element does as a gmsh Tet10: the full quadratic map, not its corners'
    affine one (which is off by the bumps)."""
    eids = np.full(4, 7)
    gmsh = compute_global_coords_from_arrays(
        eids, _TET_GP4, _fem(_TET10, code=11, dim=3, npe=10),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", WarnGaussCoordsApproximate)
        filed = compute_global_coords_from_arrays(
            eids, _TET_GP4,
            _fem(_TET10, code=-256, dim=3, npe=10,
                 name="TenNodeTetrahedron"),
        )
    np.testing.assert_allclose(filed, gmsh, rtol=0, atol=1e-14)
    lam = np.column_stack([1 - _TET_GP4.sum(axis=1), _TET_GP4])
    assert np.abs(filed - lam @ _CORNERS).max() > 1e-2


def test_class_without_shape_functions_warns_by_name() -> None:
    """A 13-node class with 3-D Gauss rows (a P2 pyramid): bbox, and the
    warning names the class and why."""
    fem = _fem(np.random.default_rng(0).random((13, 3)), code=-999, dim=3,
               npe=13, name="SomePyramid13")
    nat = np.array([[0.1, 0.1, 0.1], [0.2, 0.1, 0.3]])
    with pytest.warns(WarnGaussCoordsApproximate,
                      match=r"2 GP\(s\).*centroid\+bbox.*\[-999\].*"
                            r"SomePyramid13 \(code -999\).*13-node element "
                            r"with 3-D Gauss points"):
        compute_global_coords_from_arrays(np.full(2, 7), nat, fem)


def test_rows_without_natural_coords_are_nan_and_warn() -> None:
    """All-NaN rows name no position: the GPs are NaN, never the origin or
    an invented placement, and the class is named."""
    fem = _fem(_TET10, code=-256, dim=3, npe=10, name="TenNodeTetrahedron")
    with pytest.warns(WarnGaussCoordsApproximate,
                      match=r"4 GP\(s\) carry no natural coordinates and "
                            r"are NaN.*TenNodeTetrahedron \(code -256\)"):
        out = compute_global_coords_from_arrays(
            np.full(4, 7), np.full((4, 3), np.nan), fem,
        )
    assert np.isnan(out).all()


@pytest.mark.parametrize("code,name", [(-256, "TenNodeTetrahedron"),
                                       (11, "")])
def test_one_unplaceable_row_leaves_the_others_placed(
    code: int, name: str,
) -> None:
    """One GP without coordinates is NaN; its element's other GPs are still
    placed, on a file-coded and a gmsh-coded FEMData alike."""
    nat = _TET_GP4.copy()
    nat[2] = np.nan
    eids = np.full(4, 7)
    want = compute_global_coords_from_arrays(
        eids[[0, 1, 3]], _TET_GP4[[0, 1, 3]],
        _fem(_TET10, code=11, dim=3, npe=10),
    )
    with pytest.warns(WarnGaussCoordsApproximate,
                      match=r"1 GP\(s\) carry no natural coordinates"):
        out = compute_global_coords_from_arrays(
            eids, nat, _fem(_TET10, code=code, dim=3, npe=10, name=name),
        )
    assert np.isnan(out[2]).all()
    np.testing.assert_allclose(out[[0, 1, 3]], want, rtol=0, atol=1e-14)


# ---------------------------------------------------------------------
# 20-node hexes: the file stores them in serendipity order
# ---------------------------------------------------------------------
#
# ``brcshl`` (``SRC/element/UP-ucsd/shp3dv.cpp``), the shape functions of
# both LadrunoBrick20 and Twenty_Node_Brick, numbers the nodes: corners,
# lower-ring mid-edges, upper-ring mid-edges, verticals. Its RA/SA/TA table
# (x2) is transcribed here, independently of the bridge's permutation.
_SERENDIPITY_NAT = np.array([
    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
    [0, -1, -1], [1, 0, -1], [0, 1, -1], [-1, 0, -1],
    [0, -1, 1], [1, 0, 1], [0, 1, 1], [-1, 0, 1],
    [-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0],
], dtype=np.float64)
_G3 = np.sqrt(0.6) * np.array([-1.0, 0.0, 1.0])
_HEX_GP27 = np.array([[a, b, c] for c in _G3 for b in _G3 for a in _G3])


@pytest.mark.parametrize("cls,code", [("LadrunoBrick20", -33018),
                                      ("Twenty_Node_Brick", -49)])
def test_twenty_node_hex_in_serendipity_order_lands_on_the_cube(
    cls: str, code: int,
) -> None:
    """The cube [0, 2]^3, nodes in the file's serendipity order: each GP is
    at ``1 + xi``. Read as gmsh order it was off by 1.89, silently."""
    fem = _fem(1.0 + _SERENDIPITY_NAT, code=code, dim=3, npe=20, name=cls)
    with warnings.catch_warnings():
        warnings.simplefilter("error", WarnGaussCoordsApproximate)
        out = compute_global_coords_from_arrays(np.full(27, 7), _HEX_GP27,
                                                fem)
    np.testing.assert_allclose(out, 1.0 + _HEX_GP27, rtol=0, atol=1e-13)


def test_curved_twenty_node_hex_matches_the_gmsh_coded_placement() -> None:
    """A curved LadrunoBrick20 in serendipity order lands where the same
    element does as a gmsh Hex20 in gmsh order: the full map, mid-edge
    nodes included, through the right edges."""
    from apeGmsh.opensees._element_capabilities import (
        GMSH_HEX20_TO_SERENDIPITY,
    )

    rng = np.random.default_rng(3)
    ser = 1.0 + _SERENDIPITY_NAT
    ser[8:] += 0.15 * rng.standard_normal((12, 3))
    gmsh = np.empty_like(ser)
    gmsh[list(GMSH_HEX20_TO_SERENDIPITY)] = ser
    eids = np.full(27, 7)
    want = compute_global_coords_from_arrays(
        eids, _HEX_GP27, _fem(gmsh, code=17, dim=3, npe=20),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", WarnGaussCoordsApproximate)
        out = compute_global_coords_from_arrays(
            eids, _HEX_GP27,
            _fem(ser, code=-33018, dim=3, npe=20, name="LadrunoBrick20"),
        )
    np.testing.assert_allclose(out, want, rtol=0, atol=1e-13)


@pytest.mark.parametrize("npe,name", [(20, "SomeBrick20"),
                                      (27, "SomeBrick27"),
                                      (27, "TwentySevenNodeBrick")])
def test_quadratic_hex_of_unknown_node_order_warns_by_name(
    npe: int, name: str,
) -> None:
    """A 20- or 27-node class whose node order is not established here is
    not read as gmsh order: bbox, warned by name."""
    nodes = 1.0 + np.random.default_rng(1).random((npe, 3))
    fem = _fem(nodes, code=-999, dim=3, npe=npe, name=name)
    with pytest.warns(WarnGaussCoordsApproximate,
                      match=rf"{name} \(code -999\).*node order of a "
                            rf"{npe}-node 3-D class is not established"):
        compute_global_coords_from_arrays(np.full(27, 7), _HEX_GP27, fem)
