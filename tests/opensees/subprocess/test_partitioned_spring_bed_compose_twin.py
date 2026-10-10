"""ADR 0120 numeric twin: RUN the routed and repartitioned decks.

The same model definition emits a serial deck (``OpenSees.exe``) and a
partitioned one (``mpiexec -n K OpenSeesMP.exe``, K = 2 and 4). The harness
appends the same driver to each: ``eigen`` of 6 modes, then a short Newmark
transient, then every rank prints ``nodeDisp`` of every node it holds. The
gates are the issue's: eigenvalues to 1e-9 (relative) and the final
displacements to round-off (1e-10 of the peak), on every node, and a node
held by two ranks must print the same numbers on both.

* **Gap 1 — spring bed.** The shell plate of
  ``test_emit_partitioned_node_pair_routing``: 35 dashpot springs from
  fixed decoupled ground nodes, half through 3-dof side nodes tied by
  ``equalDOF``, Gmsh-partitioned; a corner load drives the transient.
* **Gap 2 — hosted composition.** A shell box (mat, walls, roof) is the
  host; a tet soil box with a DRM layer is an instance; the mat nodes are
  embedded in the soil's top face (``ASDEmbeddedNodeElement``); fixed outer
  boundary, three Rayleigh damping regions, a named node region, and an
  ``H5DRM`` pattern over a synthetic ShakerMaker-layout file drive the
  transient. ``Assembly(host=...).fem().repartition(K)`` cuts it.

Requires ``APEGMSH_OPENSEES_BIN`` (a Ladruno-fork ``dist\\bin`` with
``OpenSees.exe`` and ``OpenSeesMP.exe``) and Intel MPI; skips otherwise.
"""
from __future__ import annotations

import os
import subprocess
import warnings
from pathlib import Path

import h5py
import numpy as np
import pytest

import gmsh
from apeGmsh import apeGmsh

EIG_TOL = 1.0e-9
DISP_TOL = 1.0e-10
RUN_TIMEOUT_S = 600


def _dist_bin() -> "Path | None":
    d = os.environ.get("APEGMSH_OPENSEES_BIN")
    if not d:
        return None
    p = Path(d)
    if (p / "OpenSees.exe").is_file() and (p / "OpenSeesMP.exe").is_file():
        return p
    return None


def _mpi_root() -> Path:
    oneapi = os.environ.get(
        "ONEAPI_ROOT", r"C:\Program Files (x86)\Intel\oneAPI")
    return Path(os.environ.get("I_MPI_ROOT") or Path(oneapi) / "mpi" / "latest")


def _mpiexec() -> "Path | None":
    c = _mpi_root() / "bin" / "mpiexec.exe"
    return c if c.is_file() else None


pytestmark = [
    pytest.mark.subprocess,
    pytest.mark.slow,
    pytest.mark.skipif(
        _dist_bin() is None,
        reason="APEGMSH_OPENSEES_BIN unset or lacks OpenSees.exe + "
               "OpenSeesMP.exe (a Ladruno-fork dist\\bin)"),
    pytest.mark.skipif(
        _mpiexec() is None,
        reason="Intel MPI mpiexec.exe not found (set I_MPI_ROOT)"),
    pytest.mark.filterwarnings("ignore:partitioned run"),
]

DRIVER = """
# ---- ADR 0120 twin driver (appended) ----
set _pid 0
catch {{set _pid [getPID]}}
wipeAnalysis
constraints Transformation
numberer {numberer}
system {system}
test NormDispIncr 1.0e-12 25
algorithm Linear
integrator Newmark 0.5 0.25
analysis Transient
set _lam [eigen {eig} 6]
if {{$_pid == 0}} {{
  set _f [open eig.txt w]
  puts $_f $_lam
  close $_f
}}
set _ok [analyze {nsteps} {dt}]
set _f [open disp_$_pid.txt w]
foreach _n [getNodeTags] {{ puts $_f "$_n [nodeDisp $_n]" }}
close $_f
puts "TWIN_DONE $_pid ok=$_ok"
"""


def _env(dist_bin: Path) -> "dict[str, str]":
    env = dict(os.environ)
    mpiroot = _mpi_root()
    dirs = [dist_bin, mpiroot / "bin", mpiroot / "opt" / "mpi" / "libfabric" / "bin"]
    env["PATH"] = os.pathsep.join(
        [str(d) for d in dirs if d.is_dir()] + [env.get("PATH", "")])
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    return env


def _run(model_tcl: Path, ranks: "int | None", *, nsteps: int, dt: float) -> dict:
    """Append the driver, run serial (``ranks=None``) or on ``ranks``."""
    dist, mpi = _dist_bin(), _mpiexec()
    assert dist is not None and mpi is not None
    deck = model_tcl.with_name("run_" + model_tcl.name)
    deck.write_text(model_tcl.read_text() + DRIVER.format(
        numberer="RCM" if ranks is None else "ParallelPlain",
        system="UmfPack" if ranks is None else "Mumps",
        eig="-genBandArpack" if ranks is None else "",
        nsteps=nsteps, dt=dt))
    cmd = ([str(dist / "OpenSees.exe"), deck.name] if ranks is None else
           [str(mpi), "-n", str(ranks), str(dist / "OpenSeesMP.exe"), deck.name])
    r = subprocess.run(cmd, cwd=deck.parent, env=_env(dist), capture_output=True,
                       text=True, timeout=RUN_TIMEOUT_S, stdin=subprocess.DEVNULL)
    out = r.stdout + r.stderr
    assert r.returncode == 0 and out.count("TWIN_DONE") == (ranks or 1), (
        f"{cmd} exited {r.returncode}:\n{out[-3000:]}")
    lam = np.array([float(v) for v in (deck.parent / "eig.txt").read_text().split()])
    disp: dict[int, np.ndarray] = {}
    for f in sorted(deck.parent.glob("disp_*.txt")):
        for line in f.read_text().splitlines():
            w = line.split()
            n, v = int(w[0]), np.array([float(x) for x in w[1:]])
            if n in disp:
                np.testing.assert_array_equal(disp[n], v, err_msg=f"node {n} across ranks")
            disp[n] = v
    return {"lam": lam, "disp": disp}


def _assert_twin(serial: dict, mp: dict) -> None:
    assert serial["lam"].size == 6 and mp["lam"].size == 6
    rel = np.abs(mp["lam"] - serial["lam"]) / np.abs(serial["lam"])
    assert rel.max() < EIG_TOL, (serial["lam"], mp["lam"])
    assert set(mp["disp"]) == set(serial["disp"])
    peak = max(float(np.abs(v).max()) for v in serial["disp"].values())
    assert peak > 0.0
    worst = max(float(np.abs(mp["disp"][n] - v).max()) for n, v in serial["disp"].items())
    assert worst <= DISP_TOL * peak, (worst, peak)


def _emit(ops, path: Path, *, flat: bool) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ops.tcl(str(path), flat=flat)
    return path


# ---------------------------------------------------------------------------
# Gap 1 — the spring bed
# ---------------------------------------------------------------------------


def _bed_ops(n_parts: int):
    from tests.opensees.integration.test_emit_partitioned_node_pair_routing import (
        GRID, LX, LY, declare, mesh_node_at, plate_on_springs,
    )

    fem, sides, grounds = plate_on_springs(n_parts)
    ops, fem = declare(fem, sides, grounds)
    ts = ops.timeSeries.Trig(t_start=0.0, t_end=1.0, period=0.25)
    with ops.pattern.Plain(series=ts) as p:
        p.load(node=mesh_node_at(fem, LX, LY), forces=(1e5, 5e4, -2e5, 0, 0, 0))
    assert len(GRID) == 35
    return ops


@pytest.fixture(scope="module")
def bed_serial(tmp_path_factory) -> dict:
    d = tmp_path_factory.mktemp("bed_serial")
    return _run(_emit(_bed_ops(1), d / "model.tcl", flat=True), None,
                nsteps=40, dt=0.01)


@pytest.mark.parametrize("ranks", [1, 2, 4])
def test_spring_bed_ranks_match_serial(bed_serial, tmp_path: Path, ranks: int):
    """1 rank: the serial deck under OpenSeesMP (ParallelPlain + Mumps)."""
    deck = _emit(_bed_ops(ranks), tmp_path / "model.tcl", flat=ranks == 1)
    _assert_twin(bed_serial, _run(deck, ranks, nsteps=40, dt=0.01))


# ---------------------------------------------------------------------------
# Gap 2 — the hosted composition
# ---------------------------------------------------------------------------

BX, BY, BZ, LAYER, SOIL_H = 8.0, 6.0, 8.0, 1.5, 1.5
DT, NT, VS = 0.005, 120, 300.0


def _in_box(dim, lo, hi, tol=1e-6):
    return [t for _d, t in gmsh.model.getEntitiesInBoundingBox(
        lo[0] - tol, lo[1] - tol, lo[2] - tol,
        hi[0] + tol, hi[1] + tol, hi[2] + tol, dim)]


def _soil_module(path: Path) -> None:
    with apeGmsh(model_name="twin_soil", verbose=False, save_to=str(path)) as g:
        occ = gmsh.model.occ
        outer = occ.addBox(-BX, -BY, -BZ, 2 * BX, 2 * BY, BZ)
        inner = occ.addBox(-BX + LAYER, -BY + LAYER, -BZ + LAYER,
                           2 * (BX - LAYER), 2 * (BY - LAYER), BZ - LAYER)
        out, _ = occ.fragment([(3, outer)], [(3, inner)])
        occ.synchronize()
        vols = [t for d, t in out if d == 3]
        interior = _in_box(3, (-BX + LAYER, -BY + LAYER, -BZ + LAYER),
                           (BX - LAYER, BY - LAYER, 0.0))
        g.physical.add_volume(interior, name="interior")
        g.physical.add_volume([v for v in vols if v not in interior], name="drm")
        g.physical.add_volume(vols, name="domain")
        top = _in_box(2, (-BX, -BY, 0.0), (BX, BY, 0.0))
        g.physical.add_surface(top, name="top")
        faces = [t for _d, t in gmsh.model.getBoundary(
            [(3, v) for v in vols], oriented=False, combined=True)]
        g.physical.add_surface([f for f in faces if f not in top], name="boundary")
        g.mesh.sizing.set_global_size(SOIL_H)
        g.mesh.generation.generate(dim=3)
        g.mesh.queries.get_fem_data(dim=None)


def _structure_module(path: Path) -> None:
    with apeGmsh(model_name="twin_structure", verbose=False, save_to=str(path)) as g:
        occ = gmsh.model.occ
        mat = occ.addRectangle(-3, -2, 0, 6, 4)
        roof = occ.addRectangle(-3, -2, 3, 6, 4)
        box = occ.addBox(-3, -2, 0, 6, 4, 3)
        occ.synchronize()
        faces = [t for _d, t in gmsh.model.getBoundary([(3, box)], oriented=False)]

        def vertical(f: int) -> bool:
            b = gmsh.model.getBoundingBox(2, f)
            return abs(b[5] - b[2]) > 1e-6

        walls = [f for f in faces if vertical(f)]
        occ.remove([(3, box)])
        occ.remove([(2, f) for f in faces if not vertical(f)], recursive=False)
        occ.fragment([(2, mat), (2, roof)], [(2, w) for w in walls])
        occ.synchronize()
        surf = [t for _d, t in gmsh.model.getEntities(2)]
        mats = _in_box(2, (-3, -2, 0), (3, 2, 0))
        roofs = _in_box(2, (-3, -2, 3), (3, 2, 3))
        g.physical.add_surface(mats, name="Mat")
        g.physical.add_surface(roofs, name="Roof")
        g.physical.add_surface([s for s in surf if s not in mats + roofs], name="Walls")
        g.mesh.sizing.set_global_size(0.75)
        for s in surf:
            gmsh.model.mesh.setRecombine(2, s)
        g.mesh.generation.generate(dim=2)
        g.mesh.queries.get_fem_data(dim=2)


def _drm_file(fem, path: Path) -> None:
    """A ShakerMaker-layout ``.h5drm`` over the DRM-layer nodes: a vertical
    S pulse, displacement and its second difference as acceleration."""
    layer = np.unique(np.asarray(
        fem.elements.select(pg="soil.drm", element_type="tet4").connectivity))
    inner = set(np.unique(np.asarray(
        fem.elements.select(pg="soil.interior", element_type="tet4").connectivity)).tolist())
    row = {int(n): i for i, n in enumerate(np.asarray(fem.nodes.ids))}
    xyz = np.asarray(fem.nodes.coords)[[row[int(n)] for n in layer]]
    t = np.arange(NT) * DT
    disp = np.zeros((3 * len(layer), NT))
    for i, p in enumerate(xyz):
        tau = t - (p[2] + BZ) / VS
        pulse = np.where(tau > 0, np.sin(2 * np.pi * 5.0 * tau) * np.exp(-3 * tau), 0.0)
        disp[3 * i] = 1e-3 * pulse
        disp[3 * i + 1] = 0.5e-3 * pulse
    acc = np.gradient(np.gradient(disp, DT, axis=1), DT, axis=1)
    with h5py.File(path, "w") as f:
        d = f.create_group("DRM_Data")
        d["xyz"] = xyz
        d["internal"] = np.array([int(int(n) in inner) for n in layer], dtype=np.int32)
        d["data_location"] = np.arange(len(layer), dtype=np.int32) * 3
        d["displacement"] = disp
        d["acceleration"] = acc
        m = f.create_group("DRM_Metadata")
        m["drmbox_x0"] = np.zeros(3)
        for k, v in (("drmbox_xmax", BX), ("drmbox_xmin", -BX), ("drmbox_ymax", BY),
                     ("drmbox_ymin", -BY), ("drmbox_zmax", 0.0), ("drmbox_zmin", -BZ),
                     ("dt", DT), ("tstart", 0.0), ("tend", (NT - 1) * DT)):
            m[k] = v


@pytest.fixture(scope="module")
def hosted_fem(tmp_path_factory):
    from apeGmsh.assembly import Assembly

    d = tmp_path_factory.mktemp("hosted_modules")
    _soil_module(d / "soil.h5")
    _structure_module(d / "structure.h5")
    asm = Assembly("twin", host=d / "structure.h5")
    asm.instance("soil", d / "soil.h5")
    asm.embedded("soil.top", "Mat", tolerance=1e-3, stiffness=1.0e10, name="mat_embed")
    fem = asm.fem()
    assert len(fem.partitions) == 0
    return fem


def _hosted_ops(fem, drm: Path):
    from apeGmsh.opensees import apeSees

    ops = apeSees(fem, element_tags="fem", _artifacts=False)
    ops.model(ndm=3, ndf=6)
    sec = ops.section.ElasticMembranePlateSection(E=30e9, nu=0.2, h=0.3, rho=2400.0)
    for pg in ("Mat", "Walls", "Roof"):
        ops.element.ASDShellQ4(pg=pg, section=sec)
    soil = ops.nDMaterial.ElasticIsotropic(E=200e6, nu=0.3, rho=2000.0)
    ops.element.FourNodeTetrahedron(pg="soil.domain", material=soil)
    ops.fix(pg="soil.boundary", dofs=(1, 1, 1))
    ops.damping.rayleigh(on=["Mat", "Walls", "Roof"], alpha_m=0.3, beta_k=0.002)
    ops.damping.rayleigh(on=["soil.interior"], alpha_m=0.2, beta_k_init=0.0006)
    ops.damping.rayleigh(on=["soil.drm"], alpha_m=2.0, beta_k_init=0.006)
    ops.region(name="roof_nodes", pg="Roof")
    ops.pattern.H5DRM(h5drm=drm.name, crd_scale=1.0, distance_tolerance=1e-3)
    return ops


def _hosted_deck(fem, d: Path, *, flat: bool) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    _drm_file(fem, d / "drm.h5drm")
    return _emit(_hosted_ops(fem, d / "drm.h5drm"), d / "model.tcl", flat=flat)


@pytest.fixture(scope="module")
def hosted_serial(hosted_fem, tmp_path_factory) -> dict:
    d = tmp_path_factory.mktemp("hosted_serial")
    return _run(_hosted_deck(hosted_fem, d, flat=True), None, nsteps=60, dt=DT)


@pytest.mark.parametrize("ranks", [1, 2, 4])
def test_hosted_composition_ranks_match_serial(hosted_fem, hosted_serial,
                                               tmp_path: Path, ranks: int):
    """1 rank: the serial deck under OpenSeesMP (ParallelPlain + Mumps)."""
    cut = hosted_fem.repartition(ranks)
    deck = _hosted_deck(cut, tmp_path, flat=ranks == 1)
    text = deck.read_text()
    assert text.count("if {[getPID] == ") >= (ranks if ranks > 1 else 0)
    _assert_twin(hosted_serial, _run(deck, ranks, nsteps=60, dt=DT))
