"""Probe (#1973, ruling on #1961 Q4): does a plain ``.out`` element recorder
carry fibre and per-material responses on the build ``pip install
openseespy`` gives?

The question decides whether pip users need an in-process fibre/material
reader once ``DomainCapture`` is removed (#1767), or whether the spec
``.out`` emit can carry them (option D of #1961). Every recorder here is a
raw ``ops.recorder("Element", "-file", ..., "-precision", 16, ...)``: no
apeGmsh emit, reader or translator sits between the engine and the file.

The oracle is analytic, never the engine's own ``eleResponse``, which
shares ``setResponse`` with the recorder and so cannot stand alone; it is
compared too, as a secondary check (``.out`` == ``eleResponse``).

* **Fibres.** A cantilever of one ``dispBeamColumn`` or ``forceBeamColumn``
  with an elastic fibre section, under a tip axial force ``N`` and tip
  moment ``M``. Both elements are exact for constant ``N`` and ``M``, so at
  every section ``eps0 = N / EA``, ``kappa = M / EI`` (``EA = E sum A_i``,
  ``EI = E sum A_i y_i^2``; the fibres are symmetric, so ``yBar = 0``) and
  each fibre carries ``strain = eps0 - kappa y`` and ``stress = E strain``.
  ``section 1 fiberData`` is ``[y, z, A, stress, strain]`` per fibre;
  ``fiberData2`` is ``[y, z, A, matTag, stress, strain]``.
* **Per-material response.** One ``stdBrick`` (8 Gauss points) under
  uniaxial tension ``s``, read through ``material <gp> <resp>``. Every
  Gauss point carries ``sigma = (s, 0, 0, 0, 0, 0)`` (equilibrium, any
  material). On ``ElasticIsotropic`` it also carries ``eps = (s/E, -nu s/E,
  -nu s/E, 0, 0, 0)``. On ``ASDConcrete3D`` well below ``ft`` it carries
  ``damage = (d+, d-) = (0, 0)``. ASDConcrete3D's strain is not checked:
  its pre-crack secant is ``0.999991 E``, not ``E`` (measured on the fork
  build 79e0623670), a property of the material, not of the recorder.

Tolerance: rtol 1e-12 on every non-zero oracle value; a zero oracle value
is held to 1e-12 of the load's scale (the stress or strain magnitude).

Layers (layered shells) are refused off the fork by name and are not
probed here. Stock openseespy and the fork alike: no ``ladruno_fork``
marker, so this runs on CI's ``live-stock`` lane.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.live

# The gate: without a backend every case skips (the live-stock lane asserts
# the import before it runs, so it cannot skip there).
pytest.importorskip("openseespy.opensees")

from apeGmsh.opensees.emitter.live import get_ops  # noqa: E402

RTOL = 1e-12

# --- the elastic fibre section -------------------------------------------
E_FIB = 2.0e5
FIB_Y = (-0.3, -0.1, 0.1, 0.3)
FIB_A = 1.0e-2
MAT_TAG = 7            # not 1, so fiberData2's matTag column is not a default
L_BEAM = 2.0
N_TIP = 10.0 / 3.0   # no short decimal form: -precision 16 is load-bearing
M_TIP = 1.5 / 7.0
EA = E_FIB * FIB_A * len(FIB_Y)
EI = E_FIB * sum(FIB_A * y * y for y in FIB_Y)
EPS0 = N_TIP / EA
KAPPA = M_TIP / EI

# --- the ASDConcrete3D cube ----------------------------------------------
E_C = 30_000.0
NU_C = 0.2
FT = 3.0
FC = 30.0
SIG = 0.1 * FT / 7.0   # well inside the linear branch; no short decimal
N_GP = 8


def _ops() -> Any:
    ops = get_ops()
    ops.wipe()
    return ops


def _read_out(path: Path) -> np.ndarray:
    """The ``.out`` rows as a 2-D array; ``(0, 0)`` for an empty file."""
    text = path.read_text().strip()
    if not text:
        return np.zeros((0, 0))
    return np.atleast_2d(np.loadtxt(path))


def _assert_oracle(got: np.ndarray, exp: np.ndarray, scale: float,
                   what: str) -> None:
    got = np.asarray(got, dtype=float)
    exp = np.asarray(exp, dtype=float)
    assert got.shape == exp.shape, f"{what}: shape {got.shape} != {exp.shape}"
    tol = RTOL * np.maximum(np.abs(exp), scale * (exp == 0.0))
    bad = np.abs(got - exp) > tol
    assert not bad.any(), (
        f"{what}: {int(bad.sum())} value(s) off the analytic oracle at "
        f"rtol {RTOL:g}; worst |d| = {np.abs(got - exp).max():.3e}\n"
        f"got {got[bad]}\nexp {exp[bad]}"
    )


# =====================================================================
# Fibres: section 1 fiberData / fiberData2
# =====================================================================

def _run_fiber_cantilever(element: str, token: str, out: Path,
                          ) -> tuple[np.ndarray, np.ndarray]:
    """Run the cantilever; return (the ``.out`` rows, ``eleResponse``)."""
    ops = _ops()
    ops.model("basic", "-ndm", 2, "-ndf", 3)
    ops.node(1, 0.0, 0.0)
    ops.node(2, L_BEAM, 0.0)
    ops.fix(1, 1, 1, 1)
    ops.uniaxialMaterial("Elastic", MAT_TAG, E_FIB)
    ops.section("Fiber", 1)
    for y in FIB_Y:
        ops.fiber(y, 0.0, FIB_A, MAT_TAG)
    ops.geomTransf("Linear", 1)
    ops.beamIntegration("Lobatto", 1, 1, 3)
    ops.element(element, 1, 1, 2, 1, 1)
    ops.recorder("Element", "-file", str(out), "-precision", 16,
                 "-ele", 1, "section", "1", token)
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(2, N_TIP, 0.0, M_TIP)
    ops.constraints("Plain")
    ops.numberer("Plain")
    ops.system("FullGeneral")
    ops.integrator("LoadControl", 1.0)
    ops.algorithm("Linear")
    ops.analysis("Static")
    assert ops.analyze(1) == 0
    live = np.asarray(ops.eleResponse(1, "section", "1", token), dtype=float)
    ops.wipe()                 # flushes and closes the recorder
    return _read_out(out), live


def _fiber_oracle(token: str) -> np.ndarray:
    rows = []
    for y in FIB_Y:
        strain = EPS0 - KAPPA * y
        stress = E_FIB * strain
        head = [y, 0.0, FIB_A]
        if token == "fiberData2":
            head.append(float(MAT_TAG))
        rows.append(head + [stress, strain])
    return np.asarray(rows)


@pytest.mark.parametrize("token", ["fiberData", "fiberData2"])
@pytest.mark.parametrize("element", ["dispBeamColumn", "forceBeamColumn"])
def test_out_carries_fiber_data(element: str, token: str,
                                tmp_path: Path) -> None:
    rows, live = _run_fiber_cantilever(element, token,
                                       tmp_path / f"{token}.out")
    width = 6 if token == "fiberData2" else 5
    assert rows.shape == (1, width * len(FIB_Y)), (
        f"{element} section 1 {token}: the .out holds {rows.shape}, "
        f"expected one step of {width} x {len(FIB_Y)} fibre columns"
    )
    got = rows[0].reshape(len(FIB_Y), width)
    exp = _fiber_oracle(token)
    # Geometry columns: exactly what the section was built with.
    _assert_oracle(got[:, :width - 2], exp[:, :width - 2], 1.0,
                   f"{element} {token} geometry")
    # stress, strain: the analytic eps0 - kappa*y and E*strain.
    _assert_oracle(got[:, -1], exp[:, -1], abs(EPS0),
                   f"{element} {token} strain")
    _assert_oracle(got[:, -2], exp[:, -2], E_FIB * abs(EPS0),
                   f"{element} {token} stress")
    # Secondary: the file says what eleResponse says (same setResponse).
    np.testing.assert_allclose(rows[0], live, rtol=1e-15, atol=0.0)


# =====================================================================
# Per-material: material <gp> stress | strain | damage on stdBrick
# =====================================================================

_CUBE = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
         (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
_MAT_ORACLE = {
    "stress": ((SIG, 0.0, 0.0, 0.0, 0.0, 0.0), SIG),
    "strain": ((SIG / E_C, -NU_C * SIG / E_C, -NU_C * SIG / E_C,
                0.0, 0.0, 0.0), SIG / E_C),
    "damage": ((0.0, 0.0), 1.0),
}


def _run_cube(material: str, resp: str, out_dir: Path,
              ) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Run the cube; per Gauss point, (the ``.out`` rows, ``eleResponse``)."""
    ops = _ops()
    ops.model("basic", "-ndm", 3, "-ndf", 3)
    for i, (x, y, z) in enumerate(_CUBE, 1):
        ops.node(i, float(x), float(y), float(z))
        # Symmetry rollers: free lateral contraction, no rigid motion.
        ops.fix(i, int(x == 0), int(y == 0), int(z == 0))
    if material == "ElasticIsotropic":
        ops.nDMaterial("ElasticIsotropic", 1, E_C, NU_C)
    else:
        ops.nDMaterial("ASDConcrete3D", 1, E_C, NU_C,
                       "-Te", 0.0, FT / E_C, 10.0 * FT / E_C,
                       "-Ts", 0.0, FT, 0.1 * FT,
                       "-Ce", 0.0, FC / E_C, 10.0 * FC / E_C,
                       "-Cs", 0.0, FC, FC)
    ops.element("stdBrick", 1, *range(1, 9), 1)
    files = [out_dir / f"{resp}_gp{gp}.out" for gp in range(1, N_GP + 1)]
    for gp, f in enumerate(files, 1):
        ops.recorder("Element", "-file", str(f), "-precision", 16,
                     "-ele", 1, "material", str(gp), resp)
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    for i, (x, _y, _z) in enumerate(_CUBE, 1):
        if x == 1:
            ops.load(i, SIG / 4.0, 0.0, 0.0)
    ops.constraints("Plain")
    ops.numberer("Plain")
    ops.system("FullGeneral")
    ops.test("NormDispIncr", 1e-14, 10)
    ops.integrator("LoadControl", 1.0)
    ops.algorithm("Newton")
    ops.analysis("Static")
    assert ops.analyze(1) == 0
    live = [np.asarray(ops.eleResponse(1, "material", str(gp), resp),
                       dtype=float) for gp in range(1, N_GP + 1)]
    ops.wipe()
    return [_read_out(f) for f in files], live


@pytest.mark.parametrize(("material", "resp"), [
    ("ElasticIsotropic", "stress"),
    ("ElasticIsotropic", "strain"),
    ("ASDConcrete3D", "stress"),
    ("ASDConcrete3D", "damage"),
])
def test_out_carries_material_response(material: str, resp: str,
                                       tmp_path: Path) -> None:
    rows_per_gp, live_per_gp = _run_cube(material, resp, tmp_path)
    exp, scale = _MAT_ORACLE[resp]
    for gp, (rows, live) in enumerate(zip(rows_per_gp, live_per_gp), 1):
        assert rows.shape == (1, len(exp)), (
            f"stdBrick material {gp} {resp}: the .out holds {rows.shape}, "
            f"expected one step of {len(exp)} columns"
        )
        _assert_oracle(rows[0], np.asarray(exp), scale,
                       f"stdBrick material {gp} {resp}")
        np.testing.assert_allclose(rows[0], live, rtol=1e-15, atol=0.0)
