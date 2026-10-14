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

* **Fibres, 2-D.** A cantilever of one ``dispBeamColumn`` or
  ``forceBeamColumn`` with an elastic fibre section, under a tip axial
  force ``N`` and tip moment ``M``. Both elements are exact for constant
  ``N`` and ``M``, so at every section ``eps0 = N / EA``, ``kappa = M / EI``
  (the fibres are symmetric, so ``yBar = 0``) and each fibre carries
  ``strain = eps0 - kappa y`` and ``stress = E strain``. ``section 1
  fiberData`` is ``[y, z, A, stress, strain]`` per fibre; ``fiberData2`` is
  ``[y, z, A, matTag, stress, strain]``.
* **Fibres, 3-D.** The same in 3-D with fibres at ``(+-0.3, +-0.1)`` under
  ``N``, ``Mz`` and ``My``: ``strain = N/EA - y Mz/EIz + z My/EIy``. The
  ``z`` column is non-zero and ``|y| != |z|``, so a y/z swap fails.
* **Gauss-point index.** One ``stdBrick`` of ``ElasticIsotropic`` under the
  prescribed trilinear field ``u_x = a x (1 + y + 2 z)``, ``u_y = u_z = 0``,
  which the element reproduces exactly. Gauss point ``g`` (Brick.cpp order:
  ``g - 1 = 4 i + 2 j + k`` at ``(xi, eta, zeta) = (s_i, s_j, s_k)``,
  ``s = -+1/sqrt(3)``) carries ``eps_xx = a (1 + y + 2 z)``,
  ``gamma_xy = a x``, ``gamma_xz = 2 a x`` and ``sigma = C eps``: a distinct
  value at every point, so a file read at the wrong index fails.
* **Damage.** One ``stdBrick`` of ``ASDConcrete3D`` with the tension law
  ``Te = (0, et, 10 et)``, ``Td = (0, 0, 0.9)``, pulled by a prescribed
  ``u_x`` to ``4 et`` over 20 steps on symmetry rollers (uniaxial stress).
  The law interpolates damage linearly in the equivalent strain, so
  ``d+ = max(0, 0.1 (eps / et - 1))``: 0 up to step 5, then 0.02 .. 0.3;
  ``d- = 0`` (no ``-Cd``). The material's equivalent strain is not exactly
  the applied ``eps``: its pre-crack secant is ``0.999991 E`` (measured on
  the fork build 79e0623670), so this oracle holds to ``DAMAGE_ATOL``, not
  1e-12. The ``.out`` still matches ``eleResponse`` at rtol 1e-15.

Tolerance: rtol 1e-12 on every non-zero oracle value; a zero oracle value
is held to 1e-12 of the load's scale (the stress or strain magnitude).
The loads have no short decimal form, so ``-precision 16`` is load-bearing.

Layers (layered shells) are refused off the fork by name and are not
probed here. Stock openseespy and the fork alike: no ``ladruno_fork``
marker, so this runs on CI's ``live-stock`` lane.
"""
from __future__ import annotations

import math
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

# --- the elastic fibre sections ------------------------------------------
E_FIB = 2.0e5
FIB_A = 1.0e-2
MAT_TAG = 7            # not 1, so fiberData2's matTag column is not a default
L_BEAM = 2.0
N_TIP = 10.0 / 3.0     # no short decimal form: -precision 16 is load-bearing
MZ_TIP = 1.5 / 7.0
MY_TIP = 0.4 / 11.0
FIB_2D = tuple((y, 0.0) for y in (-0.3, -0.1, 0.1, 0.3))
FIB_3D = ((-0.3, -0.1), (-0.3, 0.1), (0.3, -0.1), (0.3, 0.1))

# --- the cubes -----------------------------------------------------------
E_C = 30_000.0
NU_C = 0.2
FT = 3.0
FC = 30.0
ET = FT / E_C
N_GP = 8
A_FIELD = 1.0e-5 / 3.0     # amplitude of u_x = a x (1 + y + 2 z)
N_STEPS = 20
U_MAX = 4.0 * ET
DAMAGE_ATOL = 1.0e-6       # the 0.999991 E secant; measured ~1e-7

_CUBE = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
         (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]


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
# Fibres: section 1 fiberData / fiberData2, 2-D and 3-D
# =====================================================================

def _run_fiber_cantilever(ndm: int, element: str, token: str, out: Path,
                          ) -> tuple[np.ndarray, np.ndarray]:
    """Run the cantilever; return (the ``.out`` rows, ``eleResponse``)."""
    ops = _ops()
    if ndm == 2:
        ops.model("basic", "-ndm", 2, "-ndf", 3)
        ops.node(1, 0.0, 0.0)
        ops.node(2, L_BEAM, 0.0)
        ops.fix(1, 1, 1, 1)
        ops.uniaxialMaterial("Elastic", MAT_TAG, E_FIB)
        ops.section("Fiber", 1)
        for y, _z in FIB_2D:
            ops.fiber(y, 0.0, FIB_A, MAT_TAG)
        ops.geomTransf("Linear", 1)
    else:
        ops.model("basic", "-ndm", 3, "-ndf", 6)
        ops.node(1, 0.0, 0.0, 0.0)
        ops.node(2, L_BEAM, 0.0, 0.0)
        ops.fix(1, 1, 1, 1, 1, 1, 1)
        ops.uniaxialMaterial("Elastic", MAT_TAG, E_FIB)
        ops.section("Fiber", 1, "-GJ", 1.0e6)
        for y, z in FIB_3D:
            ops.fiber(y, z, FIB_A, MAT_TAG)
        # vecxz = global Z: local y = global Y, local z = global Z.
        ops.geomTransf("Linear", 1, 0.0, 0.0, 1.0)
    ops.beamIntegration("Lobatto", 1, 1, 3)
    ops.element(element, 1, 1, 2, 1, 1)
    ops.recorder("Element", "-file", str(out), "-precision", 16,
                 "-ele", 1, "section", "1", token)
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    if ndm == 2:
        ops.load(2, N_TIP, 0.0, MZ_TIP)
    else:
        ops.load(2, N_TIP, 0.0, 0.0, 0.0, MY_TIP, MZ_TIP)
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


def _fiber_oracle(ndm: int, token: str) -> np.ndarray:
    fibres = FIB_2D if ndm == 2 else FIB_3D
    ea = E_FIB * FIB_A * len(fibres)
    eiz = E_FIB * sum(FIB_A * y * y for y, _z in fibres)
    eiy = E_FIB * sum(FIB_A * z * z for _y, z in fibres)
    my = MY_TIP if ndm == 3 else 0.0
    rows = []
    for y, z in fibres:
        strain = N_TIP / ea - y * MZ_TIP / eiz + (z * my / eiy if my else 0.0)
        head = [y, z, FIB_A]
        if token == "fiberData2":
            head.append(float(MAT_TAG))
        rows.append(head + [E_FIB * strain, strain])
    return np.asarray(rows)


@pytest.mark.parametrize("token", ["fiberData", "fiberData2"])
@pytest.mark.parametrize("element", ["dispBeamColumn", "forceBeamColumn"])
@pytest.mark.parametrize("ndm", [2, 3])
def test_out_carries_fiber_data(ndm: int, element: str, token: str,
                                tmp_path: Path) -> None:
    rows, live = _run_fiber_cantilever(ndm, element, token,
                                       tmp_path / f"{token}.out")
    exp = _fiber_oracle(ndm, token)
    n_fib, width = exp.shape
    assert rows.shape == (1, width * n_fib), (
        f"{ndm}-D {element} section 1 {token}: the .out holds "
        f"{rows.shape}, expected one step of {width} x {n_fib} columns"
    )
    got = rows[0].reshape(n_fib, width)
    what = f"{ndm}-D {element} {token}"
    eps_scale = float(np.abs(exp[:, -1]).max())
    # Geometry columns: exactly what the section was built with.
    _assert_oracle(got[:, :width - 2], exp[:, :width - 2], 1.0,
                   f"{what} geometry")
    # stress, strain: the analytic eps0 - kappa_z y + kappa_y z, E*strain.
    _assert_oracle(got[:, -1], exp[:, -1], eps_scale, f"{what} strain")
    _assert_oracle(got[:, -2], exp[:, -2], E_FIB * eps_scale,
                   f"{what} stress")
    # Secondary: the file says what eleResponse says (same setResponse).
    np.testing.assert_allclose(rows[0], live, rtol=1e-15, atol=0.0)


# =====================================================================
# Per-material: material <gp> <resp> on stdBrick
# =====================================================================

def _cube_nodes(ops: Any) -> None:
    ops.model("basic", "-ndm", 3, "-ndf", 3)
    for i, (x, y, z) in enumerate(_CUBE, 1):
        ops.node(i, float(x), float(y), float(z))


def _record_gps(ops: Any, resp: str, out_dir: Path) -> list[Path]:
    files = [out_dir / f"{resp}_gp{gp}.out" for gp in range(1, N_GP + 1)]
    for gp, f in enumerate(files, 1):
        ops.recorder("Element", "-file", str(f), "-precision", 16,
                     "-ele", 1, "material", str(gp), resp)
    return files


def _static(ops: Any, n_steps: int, resp: str) -> list[list[np.ndarray]]:
    """Run ``n_steps``; per step, ``eleResponse`` at every Gauss point.

    ``Transformation``, not ``Plain``: the Plain handler drops a
    non-homogeneous ``sp`` ("homogeneous constraint assumed").
    """
    ops.constraints("Transformation")
    ops.numberer("Plain")
    ops.system("FullGeneral")
    ops.test("NormDispIncr", 1e-14, 25)
    ops.integrator("LoadControl", 1.0 / n_steps)
    ops.algorithm("Newton")
    ops.analysis("Static")
    live = []
    for step in range(1, n_steps + 1):
        assert ops.analyze(1) == 0, f"step {step} did not converge"
        live.append([np.asarray(ops.eleResponse(1, "material", str(gp), resp),
                                dtype=float) for gp in range(1, N_GP + 1)])
    return live


def _gp_xyz(gp: int) -> tuple[float, float, float]:
    """Brick.cpp's Gauss point ``gp`` (1-based) mapped onto the unit cube."""
    s = (-1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0))
    i, j, k = (gp - 1) // 4, ((gp - 1) // 2) % 2, (gp - 1) % 2
    return tuple(0.5 * (1.0 + s[n]) for n in (i, j, k))  # type: ignore[return-value]


def _trilinear_oracle(resp: str, gp: int) -> np.ndarray:
    x, y, z = _gp_xyz(gp)
    exx = A_FIELD * (1.0 + y + 2.0 * z)
    gxy = A_FIELD * x
    gxz = 2.0 * A_FIELD * x
    if resp == "strain":
        return np.array([exx, 0.0, 0.0, gxy, 0.0, gxz])
    lam = E_C * NU_C / ((1.0 + NU_C) * (1.0 - 2.0 * NU_C))
    mu = E_C / (2.0 * (1.0 + NU_C))
    return np.array([(lam + 2.0 * mu) * exx, lam * exx, lam * exx,
                     mu * gxy, 0.0, mu * gxz])


@pytest.mark.parametrize("resp", ["stress", "strain"])
def test_out_material_gp_index_trilinear_field(resp: str,
                                               tmp_path: Path) -> None:
    ops = _ops()
    _cube_nodes(ops)
    ops.nDMaterial("ElasticIsotropic", 1, E_C, NU_C)
    ops.element("stdBrick", 1, *range(1, 9), 1)
    files = _record_gps(ops, resp, tmp_path)
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    for i, (x, y, z) in enumerate(_CUBE, 1):   # every dof prescribed
        ops.sp(i, 1, A_FIELD * x * (1.0 + y + 2.0 * z))
        ops.sp(i, 2, 0.0)
        ops.sp(i, 3, 0.0)
    live = _static(ops, 1, resp)[0]
    ops.wipe()
    for gp, f in enumerate(files, 1):
        rows = _read_out(f)
        exp = _trilinear_oracle(resp, gp)
        assert rows.shape == (1, 6), (
            f"stdBrick material {gp} {resp}: the .out holds {rows.shape}")
        _assert_oracle(rows[0], exp, float(np.abs(exp).max()),
                       f"stdBrick material {gp} {resp}")
        np.testing.assert_allclose(rows[0], live[gp - 1], rtol=1e-15,
                                   atol=0.0)
    # The field really is non-uniform: no two Gauss points share a state.
    states = {tuple(np.round(_trilinear_oracle(resp, g) / A_FIELD, 9))
              for g in range(1, N_GP + 1)}
    assert len(states) == N_GP


def test_out_material_damage_cracked(tmp_path: Path) -> None:
    ops = _ops()
    _cube_nodes(ops)
    for i, (x, y, z) in enumerate(_CUBE, 1):
        # Symmetry rollers: free lateral contraction, no rigid motion.
        ops.fix(i, int(x == 0), int(y == 0), int(z == 0))
    ops.nDMaterial("ASDConcrete3D", 1, E_C, NU_C,
                   "-Te", 0.0, ET, 10.0 * ET,
                   "-Ts", 0.0, FT, 0.1 * FT,
                   "-Td", 0.0, 0.0, 0.9,
                   "-Ce", 0.0, FC / E_C, 10.0 * FC / E_C,
                   "-Cs", 0.0, FC, FC)
    ops.element("stdBrick", 1, *range(1, 9), 1)
    files = _record_gps(ops, "damage", tmp_path)
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    for i, (x, _y, _z) in enumerate(_CUBE, 1):
        if x == 1:
            ops.sp(i, 1, U_MAX)
    live = _static(ops, N_STEPS, "damage")
    ops.wipe()
    eps = U_MAX * np.arange(1, N_STEPS + 1) / N_STEPS
    d_plus = np.maximum(0.0, 0.1 * (eps / ET - 1.0))
    exp = np.column_stack([d_plus, np.zeros(N_STEPS)])
    assert exp[-1, 0] > 0.25            # the oracle is far from zero
    for gp, f in enumerate(files, 1):
        rows = _read_out(f)
        assert rows.shape == (N_STEPS, 2), (
            f"stdBrick material {gp} damage: the .out holds {rows.shape}")
        np.testing.assert_allclose(
            rows, exp, rtol=0.0, atol=DAMAGE_ATOL,
            err_msg=f"stdBrick material {gp} damage vs d+ = 0.1 (e/et - 1)")
        np.testing.assert_allclose(
            rows, np.array([step[gp - 1] for step in live]),
            rtol=1e-15, atol=1e-17)
