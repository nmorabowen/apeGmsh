"""The fork side of the ``mpm`` warrant rows (ADR 0121 D3, ADR 0134 D3, D7).

Two decks, emitted by ``apeSees.py`` with their particle sets, run on the
pinned fork in a fresh process whose working directory is not the deck's
folder (so the set is found only through the deck's anchored ``-file`` line).
Each run's ``.ladruno`` passes the fork's own validator
(``ladruno_format.validate``, vendored in ``tests/fixtures/mpm``) and is read
back through ``Results.particles`` (ADR 0134 D3, "after M2").

* Deck A, free fall under a ramped gravity: a 1.0 x 0.4 block lattice-seeded
  at h = 0.1, ppc = 2 (160 particles of mass rho (h/2)^2 = 2.5) falls with no
  contact, so it translates rigidly and its mean velocity is
  -g * integral(f) = -g t^2 / (2 T) while the ramp f = t / T rises (a
  constant factor would give -g t, four times larger at t = T / 2).
* Deck B, the block in the corner of the grid, pressed into ``xmin`` and
  ``ymin`` by an oblique gravity: the fork refuses a step that carries a
  particle centre out of the grid (guide, "One step"), so the run reaching
  its last step with every centre inside proves the face conditions held. Its
  step is 0.70 of the critical step, which the fork's default safety 0.5
  refuses and the declared 0.9 admits.

One test per member, each a row of ``tests/opensees/warrant_rows.py``.
"""
from __future__ import annotations

import importlib
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator

import h5py
import numpy as np
import pytest

import apeGmsh
from apeGmsh.opensees import apeSees
from apeGmsh.results import Results
from tests.opensees.mpm._meshes import grid_nodes, make_fem, quads

pytestmark = pytest.mark.ladruno_fork

_FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "mpm"
_FORK_IMPORT = (
    "from apeGmsh.opensees.emitter.live import get_ops\nops = get_ops()\n")

H, PPC, RHO, E, NU, G = 0.1, 2, 1000.0, 1.0e7, 0.3, 9.81
N_PARTICLES = 160
MASS = RHO * (H / PPC) ** 2               # per unit plane-strain thickness
RAMP_T = 0.2
A_STEPS, A_DT, A_EVERY = 500, 2.0e-4, 50
B_STEPS, B_EVERY = 300, 30
#: The critical step of an elastic particle at rest (fork guide,
#: ``criticalTimeStep``): h / sqrt(M / rho), M the P-wave modulus.
DT_CRIT = H / math.sqrt(E * (1 - NU) / ((1 + NU) * (1 - 2 * NU)) / RHO)
B_DT = 0.70 * DT_CRIT


def _env() -> "dict[str, str]":
    src = str(Path(apeGmsh.__file__).resolve().parents[1])
    env = dict(os.environ, LADRUNO_OPENSEES_QUIET="1")
    env["PYTHONPATH"] = os.pathsep.join(
        [src, *filter(None, [os.environ.get("PYTHONPATH")])])
    return env


def _bridge(lo: tuple[float, float], hi: tuple[float, float]) -> apeSees:
    nodes, ids = grid_nodes((4, 2), lo, hi)
    ops = apeSees(make_fem(nodes, {"Soil": quads(ids)}), units="N_m_s",
                  _artifacts=False)
    ops.model(ndm=2, ndf=2)
    ops.nDMaterial.ElasticIsotropic(E=E, nu=NU, name="soil")
    ops.fix(pg="Soil", dofs=[1, 1])          # no free FE DOF (fork v1)
    ops.mpm.grid(origin=(0.0, 0.0), h=H, cells=(20, 10))
    ops.mpm.body("Soil", material="soil", density=RHO, ppc=PPC)
    return ops


def _run(ops: apeSees, root: Path, name: str) -> "dict[str, Any]":
    deck_dir, cwd = root / f"{name}_deck", root / f"{name}_cwd"
    deck_dir.mkdir()
    cwd.mkdir()
    deck = deck_dir / f"{name}.py"
    ops.py(str(deck))
    text = deck.read_text(encoding="utf-8")
    stock = "import openseespy.opensees as ops"
    assert stock in text
    deck.write_text(text.replace(stock, _FORK_IMPORT), encoding="utf-8")
    proc = subprocess.run([sys.executable, str(deck)], cwd=cwd, env=_env(),
                          capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    with h5py.File(deck_dir / f"{name}.particles.h5", "r") as f:
        set_id = f.attrs["SET_ID"]
    (grid_tag,) = re.findall(r"ops\.mpm\('grid', (\d+),", text)
    return {"deck": text, "ladruno": deck_dir / f"{name}.ladruno",
            "set_id": set_id, "grid_tag": int(grid_tag),
            "stdout": proc.stdout + proc.stderr}


@pytest.fixture(scope="module")
def ladruno_format() -> Iterator[ModuleType]:
    """The vendored validator, with ``sys.path`` / ``sys.modules`` restored."""
    names = ("ladruno_format", "ladruno_basis", "mpm_contracts")
    saved_path = list(sys.path)
    saved = {n: sys.modules.pop(n, None) for n in names}
    sys.path.insert(0, str(_FIXTURES))
    try:
        yield importlib.import_module("ladruno_format")
    finally:
        sys.path[:] = saved_path
        for n, m in saved.items():
            if m is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = m


@pytest.fixture(scope="module")
def fall(tmp_path_factory: pytest.TempPathFactory) -> "dict[str, Any]":
    """Deck A: free fall under a ramped gravity."""
    root = tmp_path_factory.mktemp("mpm_fall")
    ops = _bridge((0.5, 0.5), (1.5, 0.9))
    ops.timeSeries.Path(time=[0.0, RAMP_T, 1.0e9], values=[0.0, 1.0, 1.0],
                        name="ramp")
    ops.mpm.gravity(0.0, -G)
    ops.mpm.timeSeries("ramp")
    ops.mpm.flip(0.99)
    ops.mpm.massCutoff(1.0e-10)
    ops.mpm.units("m", "N", "s", system="N_m_s")
    out = root / "fall_deck" / "fall.ladruno"
    ops.mpm.output(str(out).replace("\\", "/"), every=A_EVERY,
                   fields=("VELOCITY", "STRESS", "VOLUME"))
    ops.mpm.analyze(A_STEPS, A_DT)
    return _run(ops, root, "fall")


@pytest.fixture(scope="module")
def corner(tmp_path_factory: pytest.TempPathFactory) -> "dict[str, Any]":
    """Deck B: the block pressed into the xmin / ymin corner."""
    root = tmp_path_factory.mktemp("mpm_corner")
    ops = _bridge((0.0, 0.0), (1.0, 0.4))
    ops.mpm.fix("xmin", "x")
    ops.mpm.wall("ymin", 0.3)
    ops.mpm.gravity(-2.0, -G)
    ops.mpm.damping(0.05)
    ops.mpm.safety(0.9)
    out = root / "corner_deck" / "corner.ladruno"
    ops.mpm.output(str(out).replace("\\", "/"), every=B_EVERY)
    ops.mpm.analyze(B_STEPS, B_DT)
    return _run(ops, root, "corner")


def _open(run: "dict[str, Any]") -> Results:
    return Results.from_ladruno(str(run["ladruno"]))


def _positions(run: "dict[str, Any]") -> np.ndarray:
    r = _open(run)
    try:
        return np.asarray(r.particles.get(component="POSITION").values)
    finally:
        r.close()


def _completed(run: "dict[str, Any]", steps: int, dt: float, every: int) -> None:
    r = _open(run)
    try:
        s = r.particles.get(component="POSITION")
        assert s.n_outputs == steps // every + 1
        np.testing.assert_allclose(s.time[-1], steps * dt, rtol=1e-9)
    finally:
        r.close()


# -- one test per member ----------------------------------------------------

def test_mpm_grid_hosts_the_particle_set_on_fork(fall: "dict[str, Any]") -> None:
    r = _open(fall)
    try:
        g = r.particles.grid()
        np.testing.assert_array_equal(g.origin, [0.0, 0.0])
        np.testing.assert_array_equal(g.cell_size, [H, H])
        np.testing.assert_array_equal(g.num_cells, [20, 10])
        assert g.basis == "gimp"
        info = r.particles.info()
        assert info.pattern_tag == fall["grid_tag"]
        assert info.source_set_id == fall["set_id"]
    finally:
        r.close()


def test_mpm_body_seeds_the_lattice_on_fork(fall: "dict[str, Any]") -> None:
    r = _open(fall)
    try:
        assert r.particles.bodies() == {0: "Soil"}
        reg = r.particles.registry()
        assert reg.ids.size == N_PARTICLES
        assert set(reg.material_tags.tolist()) == {1}
        np.testing.assert_allclose(reg.mass, MASS, rtol=1e-12)
    finally:
        r.close()


def test_mpm_fix_holds_the_xmin_face_on_fork(corner: "dict[str, Any]") -> None:
    _completed(corner, B_STEPS, B_DT, B_EVERY)
    assert _positions(corner)[:, 0].min() >= 0.0


def test_mpm_wall_holds_the_ymin_face_on_fork(corner: "dict[str, Any]") -> None:
    _completed(corner, B_STEPS, B_DT, B_EVERY)
    assert _positions(corner)[:, 1].min() >= 0.0


def test_mpm_gravity_accelerates_the_free_body_on_fork(fall: "dict[str, Any]") -> None:
    r = _open(fall)
    try:
        v = r.particles.get(component="VELOCITY")
        ids, vals = v.output(-1)
        assert ids.size == N_PARTICLES
        np.testing.assert_allclose(vals[:, 0], 0.0, atol=1e-9)
        assert vals[:, 1].max() < 0.0
    finally:
        r.close()


def test_mpm_time_series_ramps_the_gravity_on_fork(fall: "dict[str, Any]") -> None:
    r = _open(fall)
    try:
        v = r.particles.get(component="VELOCITY")
        _ids, vals = v.output(-1)
        t = float(v.time[-1])
        assert t < RAMP_T
        ramped, constant = -G * t * t / (2 * RAMP_T), -G * t
        np.testing.assert_allclose(vals[:, 1].mean(), ramped, rtol=1e-2)
        assert abs(vals[:, 1].mean() - constant) > 0.5 * abs(constant)
    finally:
        r.close()


def test_mpm_damping_runs_to_the_last_step_on_fork(corner: "dict[str, Any]") -> None:
    assert "ops.mpm('damping', 'local', 0.05)" in corner["deck"]
    _completed(corner, B_STEPS, B_DT, B_EVERY)


def test_mpm_flip_runs_to_the_last_step_on_fork(fall: "dict[str, Any]") -> None:
    assert "ops.mpm('flip', 0.99)" in fall["deck"]
    _completed(fall, A_STEPS, A_DT, A_EVERY)


def test_mpm_mass_cutoff_runs_to_the_last_step_on_fork(fall: "dict[str, Any]") -> None:
    assert "ops.mpm('massCutoff', 1e-10)" in fall["deck"]
    _completed(fall, A_STEPS, A_DT, A_EVERY)


def test_mpm_safety_admits_a_step_the_default_refuses_on_fork(
    corner: "dict[str, Any]",
) -> None:
    assert 0.5 * DT_CRIT < B_DT < 0.9 * DT_CRIT
    _completed(corner, B_STEPS, B_DT, B_EVERY)


def test_mpm_units_reach_the_results_on_fork(fall: "dict[str, Any]") -> None:
    r = _open(fall)
    try:
        units = r.particles.info().units
        assert (units["UNIT_LENGTH"], units["UNIT_FORCE"], units["UNIT_TIME"]) \
            == ("m", "N", "s")
    finally:
        r.close()


def test_mpm_output_writes_a_valid_ladruno_on_fork(
    fall: "dict[str, Any]", corner: "dict[str, Any]", ladruno_format: ModuleType,
) -> None:
    for run in (fall, corner):
        assert run["ladruno"].is_file()
        assert ladruno_format.validate(str(run["ladruno"])) == []


def test_mpm_analyze_takes_every_step_on_fork(fall: "dict[str, Any]") -> None:
    _completed(fall, A_STEPS, A_DT, A_EVERY)
    r = _open(fall)
    try:
        pos = r.particles.get(component="POSITION")
        _i0, p0 = pos.output(0)
        _i1, p1 = pos.output(-1)
        t = float(pos.time[-1])
        # Rigid free fall under the ramp: the mean drop is g t^3 / (6 T).
        np.testing.assert_allclose((p1[:, 1] - p0[:, 1]).mean(),
                                   -G * t ** 3 / (6 * RAMP_T), rtol=2e-2)
    finally:
        r.close()
