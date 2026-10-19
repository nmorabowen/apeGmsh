"""Simplex Gauss points are read in gmsh reference coordinates (#2228).

``GaussSlab.natural_coords`` is gmsh reference coordinates for every
family; on a triangle or tet that is ``(u, v[, w])`` with ``1 - Σ`` on node
0. The fork writes most simplex rules node-0-first instead (column ``k`` is
node ``k``'s weight), and the readers passed those rows through, so the
extrapolation and world-coordinate consumers placed every multi-point
simplex Gauss point at the wrong place: a Tet10's extrapolated corner
values came back rotated by one node, with no warning.

Every oracle here comes from the engine, never from the catalog:

* ``GLOBAL_GP_COORDS``, the world coordinates the fork's ``.ladruno``
  recorder writes for each element's Gauss points;
* the element's own strain under the prescribed field ``u_i = x_i^2 / 2``:
  ``eps_ii = x_i`` everywhere, and a quadratic element reproduces the
  field exactly, so the strain it records at a Gauss point IS that
  point's position (SixNodeTri, LadrunoLST, BezierTri6, TenNodeTetrahedron,
  BezierTet10). SixNodeTri and LadrunoLST write no GP_PARAM and no
  ``GLOBAL_GP_COORDS``; this is the only engine oracle they have.

The committed fixtures in ``tests/fixtures/simplex_gp/`` are one solve of
:data:`SOLVE` (fork build 79e062367). The ``fork`` parametrisation re-runs
it on the pinned build in the ``live-fork`` lane, so the fixtures cannot
drift from the engine unseen. Regenerate them with
``python tests/results/test_gauss_simplex_coords.py`` beside a fork build.

Not covered by an engine oracle: the triangle shells (ShellDKGT/NLDKGT,
ASDShellT3). The fork writes no GP_PARAM or ``GLOBAL_GP_COORDS`` for their
rules and their membrane strain is constant, so their frames rest on the
fork source the catalog cites.
"""
from __future__ import annotations

import subprocess
import sys
import warnings
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import h5py
import numpy as np
import pytest

from apeGmsh.fem._shape_functions import get_shape_functions
from apeGmsh.opensees._response_catalog import IntRule, flatten, lookup
from apeGmsh.results import Results
from apeGmsh.results._gauss_extrapolation import (
    extrapolate_gauss_slab_to_nodes,
)
from apeGmsh.results._gauss_world_coords import compute_global_coords
from apeGmsh.results.readers._ladruno import LadrunoReader
from apeGmsh.results.readers._mpco import MPCOReader

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "simplex_gp"
TOL = 1e-12

#: The solve behind the fixtures, run in a fresh process in the directory
#: it writes ``tri.{ladruno,mpco}`` and ``tet.{ladruno,mpco}`` into.
SOLVE = r'''
import numpy as np
from apeGmsh.opensees.emitter.live import get_ops

ops = get_ops()

# Skewed corners: no node at the reference origin, so a rotated frame
# cannot hide behind a symmetric element.
TRI = np.array([[0.1, 0.2], [2.3, 0.4], [0.6, 1.8]])
TET = np.array([[0.1, 0.2, 0.0], [2.3, 0.1, 0.4],
                [0.5, 1.7, 0.2], [0.3, 0.4, 1.9]])
QUAD = np.array([[0.0, 0.0], [2.0, 0.3], [2.4, 1.9], [0.2, 1.5]])
HEX = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float)
HEX = HEX * [1.0, 0.7, 0.5] + [[0.05, -0.02, 0.03], [0.1, 0.04, -0.06],
                                [-0.03, 0.08, 0.02], [0.02, -0.05, 0.07],
                                [-0.06, 0.03, -0.04], [0.04, 0.06, 0.01],
                                [0.07, -0.01, -0.05], [-0.02, 0.05, 0.06]]


def field(p):
    """u_i = x_i^2 / 2: the strain at any point is eps_ii = x_i."""
    return 0.5 * np.asarray(p, dtype=float) ** 2


def bezier_ctrl(corners, edges):
    """Bernstein control values of ``field`` on a straight-sided element
    whose mid-edge control points are the edge midpoints."""
    return ([field(p) for p in corners]
            + [2.0 * field((corners[a] + corners[b]) / 2.0)
               - (field(corners[a]) + field(corners[b])) / 2.0
               for a, b in edges])


def place(start, points, values):
    for i, p in enumerate(points):
        ops.node(start + i, *map(float, p))
    return [(start + i, v) for i, v in enumerate(values)]


def solve(stem, ndm, prescribed):
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    for n, v in prescribed:
        for d in range(ndm):
            ops.sp(n, d + 1, float(v[d]))
    ops.recorder("ladruno", f"{stem}.ladruno", "-E", "stresses", "strains")
    ops.recorder("mpco", f"{stem}.mpco", "-E", "stresses", "strains")
    ops.constraints("Transformation")
    ops.numberer("Plain")
    ops.system("FullGeneral")
    ops.integrator("LoadControl", 1.0)
    ops.algorithm("Linear")
    ops.analysis("Static")
    assert ops.analyze(1) == 0
    ops.wipe()


ops.wipe()
ops.model("basic", "-ndm", 2, "-ndf", 2)
ops.nDMaterial("ElasticIsotropic", 1, 1000.0, 0.25)
E3 = [(0, 1), (1, 2), (2, 0)]
P6 = list(TRI) + [(TRI[a] + TRI[b]) / 2.0 for a, b in E3]
pres = place(1, TRI, [field(p) for p in TRI])
ops.element("tri31", 1, 1, 2, 3, 1.0, "PlaneStress", 1)
pres += place(11, P6, [field(p) for p in P6])
ops.element("tri6n", 2, *range(11, 17), 1.0, "PlaneStress", 1)
pres += place(21, P6, bezier_ctrl(TRI, E3))
ops.element("BezierTri6", 3, *range(21, 27), 1.0, "PlaneStress", 1)
pres += place(31, P6, [field(p) for p in P6])
ops.element("LadrunoLST", 4, *range(31, 37), 1)
pres += place(41, QUAD, [field(p) for p in QUAD])
ops.element("quad", 5, 41, 42, 43, 44, 1.0, "PlaneStress", 1)
solve("tri", 2, pres)

ops.model("basic", "-ndm", 3, "-ndf", 3)
ops.nDMaterial("ElasticIsotropic", 1, 1000.0, 0.25)
# TenNodeTetrahedron mid-edges 4:0-1 5:1-2 6:2-0 7:0-3 8:2-3 9:1-3;
# BezierTet10 mid-edges (1-2, 2-3, 1-3, 1-4, 3-4, 2-4).
E_TEN = [(0, 1), (1, 2), (2, 0), (0, 3), (2, 3), (1, 3)]
E_BEZ = [(0, 1), (1, 2), (0, 2), (0, 3), (2, 3), (1, 3)]
pres = place(1, TET, [field(p) for p in TET])
ops.element("FourNodeTetrahedron", 1, 1, 2, 3, 4, 1)
P10 = list(TET) + [(TET[a] + TET[b]) / 2.0 for a, b in E_TEN]
pres += place(11, P10, [field(p) for p in P10])
ops.element("TenNodeTetrahedron", 2, *range(11, 21), 1)
P10B = list(TET) + [(TET[a] + TET[b]) / 2.0 for a, b in E_BEZ]
pres += place(21, P10B, bezier_ctrl(TET, E_BEZ))
ops.element("BezierTet10", 3, *range(21, 31), 1)
pres += place(31, HEX, [field(p) for p in HEX])
ops.element("stdBrick", 4, *range(31, 39), 1)
solve("tet", 3, pres)
'''

#: (file stem, element id, class, oracle). ``strain`` = the recorded
#: strain is the GP position; ``global`` = the file's GLOBAL_GP_COORDS.
#: The quad and the brick are the unchanged ``[-1, 1]`` controls.
CASES = [
    ("tri", 1, "Tri31", ("global",)),
    ("tri", 2, "SixNodeTri", ("strain",)),
    ("tri", 3, "BezierTri6", ("strain",)),
    ("tri", 4, "LadrunoLST", ("strain",)),
    ("tri", 5, "FourNodeQuad", ("global",)),
    ("tet", 1, "FourNodeTetrahedron", ("global",)),
    ("tet", 2, "TenNodeTetrahedron", ("strain", "global")),
    ("tet", 3, "BezierTet10", ("strain", "global")),
    ("tet", 4, "Brick", ("global",)),
]
_CASE_IDS = [c[2] for c in CASES]

#: gmsh linear shape-function code by (stem, n corners).
_LINEAR_CODE = {("tri", 3): 2, ("tri", 4): 3, ("tet", 4): 4, ("tet", 8): 5}
_N_CORNERS = {"Tri31": 3, "SixNodeTri": 3, "BezierTri6": 3, "LadrunoLST": 3,
              "FourNodeQuad": 4, "FourNodeTetrahedron": 4,
              "TenNodeTetrahedron": 4, "BezierTet10": 4, "Brick": 8}
_AXES = {"tri": ("xx", "yy"), "tet": ("xx", "yy", "zz")}


# ---------------------------------------------------------------------------
# Fixtures: the committed solve, and (fork lane) the same solve re-run
# ---------------------------------------------------------------------------


def _run_solve(directory: Path) -> None:
    from tests.opensees.integration_ladruno.test_mpco_modes_live import (
        _fresh_env,
    )
    script = directory / "solve.py"
    script.write_text(SOLVE, encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(script)], cwd=directory, env=_fresh_env(),
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]


@pytest.fixture(
    scope="module",
    params=["committed",
            pytest.param("fork", marks=pytest.mark.ladruno_fork)],
)
def solve_dir(request: Any, tmp_path_factory: Any) -> Path:
    if request.param == "committed":
        return FIXTURES
    directory = tmp_path_factory.mktemp("simplex_gp")
    _run_solve(directory)
    return directory


def _global_gp(ladruno: Path, eid: int, n_gp: int, ndim: int) -> np.ndarray:
    """The element's ``GLOBAL_GP_COORDS``, ``(n_gp, ndim)``, in GP order."""
    with h5py.File(ladruno, "r") as f:
        for grp in f["MODEL_STAGE[1]/MODEL/ELEMENTS"].values():
            if int(grp["CONNECTIVITY"][0, 0]) == eid:
                g = np.asarray(grp["GLOBAL_GP_COORDS"][0], dtype=np.float64)
                return g.reshape(n_gp, -1)[:, :ndim]
    raise KeyError(eid)


def _corners(ladruno: Path, eid: int, cls: str) -> np.ndarray:
    """The element's corner coordinates, in connectivity order."""
    with h5py.File(ladruno, "r") as f:
        model = f["MODEL_STAGE[1]/MODEL"]
        ids = model["NODES/ID"][...].ravel()
        xyz = model["NODES/COORDINATES"][...]
        for grp in model["ELEMENTS"].values():
            row = grp["CONNECTIVITY"][0]
            if int(row[0]) == eid:
                pos = {int(n): k for k, n in enumerate(ids)}
                return xyz[[pos[int(n)] for n in row[1:]]][:_N_CORNERS[cls]]
    raise KeyError(eid)


def _world(natural: np.ndarray, corners: np.ndarray, stem: str) -> np.ndarray:
    """Natural coords → world through the gmsh linear shape functions."""
    code = _LINEAR_CODE[(stem, corners.shape[0])]
    n_fn = get_shape_functions(code)[0]
    return n_fn(np.asarray(natural)[:, :corners.shape[1]]) @ corners


def _reader(path: Path) -> Any:
    return LadrunoReader(path) if path.suffix == ".ladruno" else MPCOReader(path)


def _slab(path: Path, component: str, eid: int) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with _reader(path) as r:
            slab = r.read_gauss(
                r.stages()[0].id, component, element_ids=np.array([eid]),
            )
    sel = np.asarray(slab.element_index) == eid
    assert sel.any(), f"{path.name}: no {component} Gauss points for {eid}"
    return replace(
        slab, values=np.asarray(slab.values)[:, sel],
        element_index=np.asarray(slab.element_index)[sel],
        natural_coords=np.asarray(slab.natural_coords)[sel],
    )


def _oracles(directory: Path, stem: str, eid: int,
             kinds: "tuple[str, ...]", reader_path: Path) -> list[np.ndarray]:
    """The engine's GP world positions for one element, by each oracle."""
    axes = _AXES[stem]
    out = []
    if "strain" in kinds:
        out.append(np.stack([
            _slab(reader_path, f"strain_{a}", eid).values[-1] for a in axes
        ], axis=1))
    if "global" in kinds:
        n_gp = _slab(reader_path, "stress_xx", eid).natural_coords.shape[0]
        out.append(_global_gp(directory / f"{stem}.ladruno", eid, n_gp,
                              len(axes)))
    return out


# ---------------------------------------------------------------------------
# Both readers place every Gauss point where the engine says it is
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ext", ["ladruno", "mpco"])
@pytest.mark.parametrize("stem,eid,cls,kinds", CASES, ids=_CASE_IDS)
def test_reader_places_gauss_points_where_the_engine_does(
    solve_dir: Path, ext: str, stem: str, eid: int, cls: str,
    kinds: "tuple[str, ...]",
) -> None:
    path = solve_dir / f"{stem}.{ext}"
    natural = _slab(path, "stress_xx", eid).natural_coords
    world = _world(natural, _corners(solve_dir / f"{stem}.ladruno", eid, cls),
                   stem)
    for want in _oracles(solve_dir, stem, eid, kinds, path):
        np.testing.assert_allclose(
            world, want, rtol=0, atol=TOL,
            err_msg=f"{cls} ({ext}): GP world coords vs the engine",
        )


@pytest.mark.parametrize("stem,eid,cls,kinds", CASES, ids=_CASE_IDS)
def test_mpco_and_ladruno_agree_on_the_same_solve(
    solve_dir: Path, stem: str, eid: int, cls: str, kinds: Any,
) -> None:
    lad = _slab(solve_dir / f"{stem}.ladruno", "stress_xx", eid)
    mpco = _slab(solve_dir / f"{stem}.mpco", "stress_xx", eid)
    np.testing.assert_allclose(
        lad.natural_coords, mpco.natural_coords, rtol=0, atol=TOL,
        err_msg=cls,
    )


# ---------------------------------------------------------------------------
# The consumers: Gauss -> node extrapolation, and GP world coordinates
# ---------------------------------------------------------------------------


def _gmsh_fem(ladruno: Path, eid: int, code: int, npe: int) -> Any:
    """A FEMData stand-in holding one gmsh-coded element (as an apeGmsh
    mesh would), on the file's own node ids and coordinates."""
    with h5py.File(ladruno, "r") as f:
        model = f["MODEL_STAGE[1]/MODEL"]
        ids = [int(n) for n in model["NODES/ID"][...].ravel()]
        raw = np.asarray(model["NODES/COORDINATES"][...], dtype=np.float64)
        xyz = np.zeros((raw.shape[0], 3))       # FEMData coords are 3-D
        xyz[:, :raw.shape[1]] = raw
        conn = next(
            np.asarray(g["CONNECTIVITY"][0, 1:], dtype=np.int64)
            for g in model["ELEMENTS"].values()
            if int(g["CONNECTIVITY"][0, 0]) == eid
        )

    class _Group:
        element_type = SimpleNamespace(
            code=code, dim=3 if code in (4, 11) else 2, npe=npe, order=2,
            gmsh_name="gmsh",
        )
        ids = np.array([eid])
        connectivity = conn[None, :]

        def __len__(self) -> int:
            return 1

    return SimpleNamespace(
        nodes=SimpleNamespace(ids=ids, coords=xyz), elements=[_Group()],
    )


@pytest.mark.parametrize("eid,cls", [(2, "TenNodeTetrahedron"),
                                     (3, "BezierTet10")])
def test_tet10_extrapolates_a_linear_field_to_its_own_corners(
    solve_dir: Path, eid: int, cls: str,
) -> None:
    """The #2228 reproducer. f = x at the true GP positions, extrapolated
    to the corners of a gmsh Tet10, must give each corner its own x. On
    main it gave the corners' x rotated by one node, silently."""
    lad = solve_dir / "tet.ladruno"
    slab = _slab(lad, "stress_xx", eid)
    gp_x = _global_gp(lad, eid, 4, 3)[:, 0]
    slab = replace(slab, values=gp_x[None, :])
    fem = _gmsh_fem(lad, eid, code=11, npe=10)
    node_ids, nodal = extrapolate_gauss_slab_to_nodes(slab, fem)
    pos = {n: k for k, n in enumerate(fem.nodes.ids)}
    want = fem.nodes.coords[[pos[int(n)] for n in node_ids], 0]
    np.testing.assert_allclose(nodal[0], want, rtol=0, atol=1e-10,
                               err_msg=cls)


@pytest.mark.parametrize("stem,eid,cls,code,npe", [
    ("tet", 2, "TenNodeTetrahedron", 11, 10),
    ("tet", 3, "BezierTet10", 11, 10),
    ("tri", 2, "SixNodeTri", 9, 6),
    ("tri", 4, "LadrunoLST", 9, 6),
    ("tri", 3, "BezierTri6", 9, 6),
])
def test_gp_world_coords_on_a_gmsh_mesh(
    solve_dir: Path, stem: str, eid: int, cls: str, code: int, npe: int,
) -> None:
    """``GaussSlab.global_coords`` against a gmsh-coded FEMData."""
    lad = solve_dir / f"{stem}.ladruno"
    slab = _slab(lad, "stress_xx", eid)
    with warnings.catch_warnings():
        warnings.simplefilter("error")   # no degraded (bbox) placement
        world = compute_global_coords(slab, _gmsh_fem(lad, eid, code, npe))
    want = _oracles(solve_dir, stem, eid, ("strain",), lad)[0]
    np.testing.assert_allclose(world[:, :want.shape[1]], want, rtol=0,
                               atol=TOL, err_msg=cls)


@pytest.mark.parametrize("stem,eid,cls", [("tet", 3, "BezierTet10"),
                                          ("tri", 3, "BezierTri6")])
def test_bezier_world_coords_on_the_files_own_fem(
    solve_dir: Path, stem: str, eid: int, cls: str,
) -> None:
    """The file's own FEMData names the Bezier classes, so their GP world
    coordinates go through the Bernstein basis, which takes full
    barycentric rows. They were right on main and must stay right."""
    lad = solve_dir / f"{stem}.ladruno"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with Results.from_ladruno(lad) as r:
            fem = r.fem
    slab = _slab(lad, "stress_xx", eid)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        world = compute_global_coords(slab, fem)
    want = _oracles(solve_dir, stem, eid, ("strain",), lad)[0]
    np.testing.assert_allclose(world[:, :want.shape[1]], want, rtol=0,
                               atol=TOL, err_msg=cls)


# ---------------------------------------------------------------------------
# The native paths: the recorder transcoder and DomainCapture
# ---------------------------------------------------------------------------


class _MinimalFem:
    def __init__(self, node_ids: "list[int]", coords: np.ndarray) -> None:
        coords3d = np.zeros((len(node_ids), 3), dtype=np.float64)
        coords3d[:, :coords.shape[1]] = coords
        self.nodes = SimpleNamespace(
            ids=np.asarray(node_ids, dtype=np.int64), coords=coords3d,
        )
        self.elements: list[Any] = []

    @property
    def snapshot_id(self) -> str:
        from apeGmsh.mesh._femdata_hash import compute_snapshot_id
        return compute_snapshot_id(self)

    def to_native_h5(self, group: Any) -> None:
        from apeGmsh.mesh._femdata_h5_io import write_neutral_zone_into_group
        write_neutral_zone_into_group(self, group, ndf=3)


def _native_world(target: Path, fem: Any, comps: "tuple[str, ...]",
                  corners: np.ndarray, stem: str) -> "tuple[Any, Any]":
    from tests.conftest import _open_model_from_h5
    with Results.from_native(target, fem=fem,
                             model=_open_model_from_h5(target)) as r:
        st = r.stage(r.stages[0].id)
        slabs = [st.elements.gauss.get(component=c) for c in comps]
    world = _world(slabs[0].natural_coords, corners, stem)
    recorded = np.stack([np.asarray(s.values)[-1] for s in slabs], axis=1)
    return world, recorded


@pytest.mark.parametrize("stem,eid,cls,rule", [
    ("tri", 2, "SixNodeTri", IntRule.Triangle_GL_2),
    ("tri", 4, "LadrunoLST", IntRule.Triangle_GL_2),
    ("tet", 2, "TenNodeTetrahedron", IntRule.Tet_GL_2),
])
def test_recorder_transcoder_places_gauss_points(
    solve_dir: Path, tmp_path: Path, stem: str, eid: int, cls: str,
    rule: int,
) -> None:
    """A ``.out`` file holding the engine's own strains (each GP's
    position) transcodes to a native file whose GPs sit there."""
    from apeGmsh.results.spec._resolved import (
        ResolvedRecorderRecord,
        ResolvedRecorderSpec,
    )
    from apeGmsh.results.transcoders._recorder import RecorderTranscoder

    layout = lookup(cls, rule, "strain")
    mpco = solve_dir / f"{stem}.mpco"
    per_comp = {
        c: _slab(mpco, c, eid).values[-1][None, None, :]
        for c in layout.component_layout
    }
    flat = flatten(per_comp, layout)                  # (1, 1, K)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    np.savetxt(out_dir / "rec_gauss.out",
               np.column_stack([[1.0], flat.reshape(1, -1)]))
    corners = _corners(solve_dir / f"{stem}.ladruno", eid, cls)
    fem = _MinimalFem([1], np.zeros((1, 3)))
    spec = ResolvedRecorderSpec(
        fem_snapshot_id=fem.snapshot_id,
        records=(ResolvedRecorderRecord(
            category="gauss", name="rec",
            components=tuple(layout.component_layout),
            dt=None, n_steps=None, element_ids=np.array([eid]),
            element_class_name=cls,
        ),),
    )
    target = tmp_path / "transcoded.h5"
    RecorderTranscoder(spec, out_dir, target, fem).run()
    comps = tuple(f"strain_{a}" for a in _AXES[stem])
    world, recorded = _native_world(target, fem, comps, corners, stem)
    np.testing.assert_allclose(world, recorded, rtol=0, atol=TOL,
                               err_msg=cls)


#: The penalty-imposed field is exact to ~1e-15 relative; the rotated
#: frame this guards against is off by order 1.
CAPTURE_TOL = 1e-9

_CAPTURE_MODEL = {
    # cls: (ndm, Tcl name, corners, mid-edges, trailing element args)
    "SixNodeTri": (2, "tri6n",
                   np.array([[0.1, 0.2], [2.3, 0.4], [0.6, 1.8]]),
                   [(0, 1), (1, 2), (2, 0)], (1.0, "PlaneStress", 1)),
    "TenNodeTetrahedron": (3, "TenNodeTetrahedron",
                           np.array([[0.1, 0.2, 0.0], [2.3, 0.1, 0.4],
                                     [0.5, 1.7, 0.2], [0.3, 0.4, 1.9]]),
                           [(0, 1), (1, 2), (2, 0), (0, 3), (2, 3), (1, 3)],
                           (1,)),
}


@pytest.mark.live
@pytest.mark.parametrize("cls", [
    "SixNodeTri",
    # TenNodeTetrahedron returns 24 strain values only on the fork build
    # (``732ab316d``); older engines return 6 and the capture refuses.
    pytest.param("TenNodeTetrahedron", marks=pytest.mark.ladruno_fork),
])
def test_domain_capture_places_gauss_points(tmp_path: Path, cls: str) -> None:
    """DomainCapture under the quadratic field: the captured strain at
    each GP is its position, and the native file's GP coords must map
    there."""
    from apeGmsh.opensees.emitter.live import get_ops
    from apeGmsh.results.capture._domain import DomainCapture
    from apeGmsh.results.capture.spec import (
        ResolvedDomainCaptureRecord,
        ResolvedDomainCaptureSpec,
    )

    from tests.fixtures.capture_tags import identity_tags

    ops = get_ops()
    ndm, tcl, corners, edges, extra = _CAPTURE_MODEL[cls]
    points = np.vstack([corners] + [(corners[a] + corners[b]) / 2.0
                                    for a, b in edges])
    ops.wipe()
    ops.model("basic", "-ndm", ndm, "-ndf", ndm)
    ops.nDMaterial("ElasticIsotropic", 1, 1000.0, 0.25)
    node_ids = list(range(1, len(points) + 1))
    for n, p in zip(node_ids, points):
        ops.node(n, *map(float, p))
    ops.element(tcl, 1, *node_ids, *extra)
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    for n, p in zip(node_ids, points):
        for d in range(ndm):
            ops.sp(n, d + 1, float(0.5 * p[d] ** 2))
    # Penalty, not Transformation: every DOF is prescribed, so the
    # Transformation handler leaves a zero-equation system, the likely
    # reason stock openseespy hard-exited the live-stock lane here (exit
    # 255). The penalty error is about K / alpha ~ 1e-15, far inside
    # CAPTURE_TOL.
    ops.constraints("Penalty", 1.0e18, 1.0e18)
    ops.numberer("Plain")
    ops.system("FullGeneral")
    ops.integrator("LoadControl", 1.0)
    ops.algorithm("Linear")
    ops.analysis("Static")
    assert ops.analyze(1) == 0
    assert ops.eleType(1) == cls

    rule = IntRule.Triangle_GL_2 if ndm == 2 else IntRule.Tet_GL_2
    layout = lookup(cls, rule, "strain")
    fem = _MinimalFem(node_ids, points)
    spec = ResolvedDomainCaptureSpec(
        fem_snapshot_id=fem.snapshot_id,
        records=(ResolvedDomainCaptureRecord(
            category="gauss", name="rec",
            components=tuple(layout.component_layout),
            dt=None, n_steps=None, element_ids=np.array([1]),
        ),),
    )
    target = tmp_path / "cap.h5"
    try:
        with DomainCapture(spec, target, fem, ops=ops,
                           tag_map=identity_tags(spec)) as cap:
            cap.begin_stage("static", kind="static")
            cap.step(t=ops.getTime())
            cap.end_stage()
    finally:
        ops.wipe()
    stem = "tri" if ndm == 2 else "tet"
    comps = tuple(f"strain_{a}" for a in _AXES[stem])
    world, recorded = _native_world(target, fem, comps, corners, stem)
    np.testing.assert_allclose(world, recorded, rtol=0, atol=CAPTURE_TOL,
                               err_msg=cls)


# ---------------------------------------------------------------------------
# The MPCO -> .ladruno converter writes the .ladruno frame and family
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stem,cls,mpco_key", [
    ("tri", "Tri31", "33-Tri31[100:0]"),
    ("tri", "BezierTri6", "33000-BezierTri6[101:0]"),
    ("tet", "TenNodeTetrahedron", "256-TenNodeTetrahedron[301:0]"),
    ("tet", "BezierTet10", "33001-BezierTet10[301:0]"),
])
def test_converter_writes_what_the_fork_writes(
    solve_dir: Path, tmp_path: Path, stem: str, cls: str, mpco_key: str,
) -> None:
    """The converter's GP_PARAM and FAMILY for a rule-tagged MPCO group
    equal the fork recorder's own, so a converted file reads back like a
    native one. (The fork's MPCO tags the Bezier classes rule 0, which the
    converter cannot describe; the rule-tagged keys are what other MPCO
    writers emit.)"""
    from apeGmsh.results.readers._ladruno_from_mpco import _basis

    with h5py.File(solve_dir / f"{stem}.ladruno", "r") as f:
        grp = next(g for n, g in f["MODEL_STAGE[1]/MODEL/ELEMENTS"].items()
                   if f"-{cls}[" in n)
        fork_gp = np.asarray(grp["QUADRATURE/GP_PARAM"][...])
        fork_family = grp.attrs["FAMILY"]
        fork_family = (fork_family[0] if np.ndim(fork_family) else
                       fork_family)
        fork_family = (fork_family.decode() if isinstance(fork_family, bytes)
                       else str(fork_family))
    with h5py.File(tmp_path / "m.h5", "w") as f:
        dset = f.create_dataset(mpco_key, data=np.zeros((1, 5), np.int32))
        basis = _basis(mpco_key, dset, mpco_key)
    assert basis.gp_param is not None, mpco_key
    assert (basis.param_domain, basis.family) == ("bary", fork_family)
    np.testing.assert_allclose(basis.gp_param, fork_gp, rtol=0, atol=TOL,
                               err_msg=cls)


# ---------------------------------------------------------------------------
# The conversion itself refuses what it cannot place
# ---------------------------------------------------------------------------


def test_bary_conversion_round_trips() -> None:
    from apeGmsh.results.readers._ladruno_element_io import (
        gmsh_to_gp_param,
        gp_param_to_gmsh,
    )

    rows = np.array([[0.6, 0.1, 0.2], [0.05, 0.15, 0.3]])
    for topo, fam, gp in (("tet", "lagrange", rows),
                          ("tet", "bernstein", rows),
                          ("tri", "lagrange", rows[:, :2]),
                          ("tri", "bernstein", rows[:, :2])):
        gm = gp_param_to_gmsh(gp, param_domain="bary", topology=topo,
                              family=fam, where="t")
        np.testing.assert_allclose(
            gmsh_to_gp_param(gm, topology=topo, family=fam), gp, atol=1e-15,
        )


@pytest.mark.parametrize("kw,match", [
    (dict(param_domain="[0,1]", topology="tri", family="lagrange"),
     "unknown PARAM_DOMAIN"),
    (dict(param_domain="[-1,1]", topology="tet", family="lagrange"),
     "simplex rule is barycentric"),
    (dict(param_domain="bary", topology="quad", family="lagrange"),
     "expected None free"),
    (dict(param_domain="bary", topology="tri", family="nurbs"),
     "no known barycentric frame"),
])
def test_bary_conversion_refuses_an_unknown_frame(kw: Any, match: str) -> None:
    from apeGmsh.results.readers._ladruno_element_io import gp_param_to_gmsh

    with pytest.raises(ValueError, match=match):
        gp_param_to_gmsh(np.array([[0.2, 0.3]]), where="t", **kw)


def test_gp_param_without_param_domain_refuses(tmp_path: Path) -> None:
    """A GP_PARAM whose frame the file does not state is not guessed."""
    import shutil
    path = tmp_path / "tet.ladruno"
    shutil.copy(FIXTURES / "tet.ladruno", path)
    with h5py.File(path, "r+") as f:
        name, _ = next(
            (n, g) for n, g in f["MODEL_STAGE[1]/MODEL/ELEMENTS"].items()
            if "TenNodeTetrahedron" in n
        )
        del f[f"MODEL_STAGE[1]/MODEL/ELEMENTS/{name}"].attrs["PARAM_DOMAIN"]
    with LadrunoReader(path) as r:
        with pytest.raises(ValueError, match="no PARAM_DOMAIN attribute"):
            r.read_gauss(r.stages()[0].id, "stress_xx")


# ---------------------------------------------------------------------------
# Converted files: older conversions are told by inference and refused
# ---------------------------------------------------------------------------

#: ASDShellT3.cpp:148-150 mid-edge (XI, ETA), with N0 = 1 - XI - ETA
#: (ASDShellT3.cpp:165): gmsh (u, v) as the element writes it.
ASDSHELLT3_GP = np.array([[0.5, 0.5], [0.0, 0.5], [0.5, 0.0]])
CONVERTED_WRITER = "apeGmsh mpco->ladruno 2.0.0"


def _converted_group_file(tmp_path: Path, mpco_key: str, conn: np.ndarray,
                          *, old_rows: "np.ndarray | None") -> Path:
    """A file holding one element group written by the converter, under the
    converter's PROVENANCE. ``old_rows`` rewrites it the way an older
    conversion did: the catalog's rows of the time, ``FAMILY="lagrange"``."""
    from apeGmsh.results.readers._ladruno_from_mpco import (
        _write_element_group,
    )

    path = tmp_path / "converted.ladruno"
    with h5py.File(path, "w") as f:
        f.create_group("INFO/PROVENANCE").attrs["writer"] = CONVERTED_WRITER
        src = f.create_group("_mpco").create_dataset(mpco_key, data=conn)
        grp = f.create_group(f"MODEL_STAGE[1]/MODEL/ELEMENTS/{mpco_key}")
        _write_element_group(mpco_key, src, grp, where=mpco_key)
        del f["_mpco"]
        if old_rows is not None:
            grp.attrs["FAMILY"] = np.array([b"lagrange"])
            grp["QUADRATURE/GP_PARAM"][...] = old_rows
    return path


def _read_gp_param(path: Path, key: str) -> np.ndarray:
    from apeGmsh.results.readers._ladruno_element_io import _gp_param_for

    with h5py.File(path, "r") as f:
        return _gp_param_for(f["MODEL_STAGE[1]/MODEL/ELEMENTS"], key)


def test_asdshellt3_new_conversion_reads_its_own_points(
    tmp_path: Path,
) -> None:
    path = _converted_group_file(
        tmp_path, "204-ASDShellT3[102:0]", np.array([[7, 1, 2, 3]]),
        old_rows=None,
    )
    np.testing.assert_allclose(
        _read_gp_param(path, "204-ASDShellT3[102:0:0]"), ASDSHELLT3_GP,
        rtol=0, atol=1e-15,
    )


def test_asdshellt3_old_conversion_refuses(tmp_path: Path) -> None:
    """The older converter stored ASDShellT3's gmsh rows as ``lagrange``;
    read node-0-first they come back rotated, so they are refused."""
    path = _converted_group_file(
        tmp_path, "204-ASDShellT3[102:0]", np.array([[7, 1, 2, 3]]),
        old_rows=ASDSHELLT3_GP,
    )
    with pytest.raises(ValueError, match="ASDShellT3 group was converted"):
        _read_gp_param(path, "204-ASDShellT3[102:0:0]")


def _bezier_tri6_converted(tmp_path: Path, *, old: bool) -> Path:
    """The fork's tri fixture with its BezierTri6 group replaced by the
    converter's (rule-tagged, as other MPCO writers key it)."""
    import shutil

    from apeGmsh.results.readers._ladruno_from_mpco import (
        _write_element_group,
    )

    path = tmp_path / "tri_converted.ladruno"
    shutil.copy(FIXTURES / "tri.ladruno", path)
    key = "33000-BezierTri6[101:0]"
    with h5py.File(path, "r+") as f:
        els = f["MODEL_STAGE[1]/MODEL/ELEMENTS"]
        fork_name = next(n for n in els if "-BezierTri6[" in n)
        conn = np.asarray(els[fork_name]["CONNECTIVITY"][...])
        fork_gp = np.asarray(els[fork_name]["QUADRATURE/GP_PARAM"][...])
        del els[fork_name]
        src = f.create_group("_mpco").create_dataset(key, data=conn)
        grp = els.create_group(key)
        _write_element_group(key, src, grp, where=key)
        del f["_mpco"]
        if old:
            grp.attrs["FAMILY"] = np.array([b"lagrange"])
            grp["QUADRATURE/GP_PARAM"][...] = fork_gp   # the gmsh rows
        f["INFO/PROVENANCE"].attrs["writer"] = CONVERTED_WRITER
    return path


def test_bezier_tri6_new_conversion_reads_correctly(tmp_path: Path) -> None:
    path = _bezier_tri6_converted(tmp_path, old=False)
    with h5py.File(path, "r") as f:
        grp = f["MODEL_STAGE[1]/MODEL/ELEMENTS/33000-BezierTri6[101:0]"]
        assert grp.attrs["FAMILY"][0] == b"bernstein"
    natural = _slab(path, "stress_xx", 3).natural_coords
    world = _world(natural, _corners(FIXTURES / "tri.ladruno", 3,
                                     "BezierTri6"), "tri")
    want = _oracles(FIXTURES, "tri", 3, ("strain",),
                    FIXTURES / "tri.ladruno")[0]
    np.testing.assert_allclose(world, want, rtol=0, atol=TOL)


def test_bezier_tri6_old_conversion_refuses(tmp_path: Path) -> None:
    path = _bezier_tri6_converted(tmp_path, old=True)
    with pytest.raises(ValueError, match="BezierTri6 group was converted"):
        _slab(path, "stress_xx", 3)


def test_fork_written_groups_are_not_taken_for_conversions() -> None:
    """The fork writes ``ORDER`` beside every barycentric GP_PARAM and no
    converter writer, so none of its groups goes through the refusal."""
    from apeGmsh.results.readers._ladruno_element_io import _converted_group

    for stem in ("tri", "tet"):
        with h5py.File(FIXTURES / f"{stem}.ladruno", "r") as f:
            checked = 0
            for name, grp in f["MODEL_STAGE[1]/MODEL/ELEMENTS"].items():
                if "QUADRATURE" not in grp:
                    continue       # no GP_PARAM: the refusal never runs
                checked += 1
                assert not _converted_group(grp), name
            assert checked >= 3, stem


if __name__ == "__main__":   # regenerate the committed fixtures (fork build)
    FIXTURES.mkdir(parents=True, exist_ok=True)
    for p in FIXTURES.glob("*"):
        p.unlink()
    _run_solve(FIXTURES)
    (FIXTURES / "solve.py").unlink()
    print(sorted(p.name for p in FIXTURES.iterdir()))
