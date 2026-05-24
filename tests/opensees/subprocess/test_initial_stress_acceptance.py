"""Phase SSI-1 acceptance test — empirical ramp-discriminator (full apeSees emit).

Runs the 1-element ASDPlasticMaterial3D MohrCoulomb patch test from
``C:\\Users\\nmora\\opensees_runs\\cerro_lindo\\ssi_test_stressctrl``
and verifies that the apeSees-emitted ramp produces the **FIXED**
intermediate stresses (linear interpolation 0 → target), not the
**BUGGY** STKO behaviour (single-step jump then plateau).

Phase SSI-1.5: the deck is now built **entirely via apeSees** —
material (``ops.nDMaterial.MohrCoulombSoil``), element
(``ops.element.FourNodeQuad``), boundary conditions, analysis chain,
and the ramp (``ops.initial_stress``).  No hand-written Tcl
preamble.  This is the full user-acceptance test from the original
Phase SSI-1 spec.

The discriminating step is step 5:

* FIXED (correct):  σxx ≈ -3024 kPa
* BUGGY (STKO):     σxx ≈ -5981 kPa

Tolerance: ±0.5 kPa per step (the test compares against the saved
``result_fixed.csv`` from the reference run on the same Ladruno
binary build hash 288f6d0f).

Gated on the Ladruno OpenSees binary being available.
"""
from __future__ import annotations

import csv
import os
import shutil
import subprocess
from pathlib import Path
from typing import cast

import pytest

from apeGmsh.opensees import apeSees

from tests.opensees.fixtures.fem_stub import (
    FEMStub,
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
)


REFERENCE_FIXED_CSV = Path(
    r"C:\Users\nmora\opensees_runs\cerro_lindo\ssi_test_stressctrl"
    r"\result_fixed.csv"
)


def _opensees_available() -> bool:
    return bool(os.environ.get("OPENSEES_BIN") or shutil.which("OpenSees"))


pytestmark = [
    pytest.mark.subprocess,
    pytest.mark.skipif(
        not _opensees_available(),
        reason="OpenSees binary not on PATH and OPENSEES_BIN not set",
    ),
    pytest.mark.skipif(
        not REFERENCE_FIXED_CSV.exists(),
        reason=f"Reference CSV not found at {REFERENCE_FIXED_CSV}",
    ),
]


def _make_single_quad_fem() -> FEMStub:
    """One 4-node quad, unit square in XY, mirrors the reference deck."""
    nodes = _NodesStub(
        ids=[1, 2, 3, 4],
        coords=[
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
        ],
        # PGs cover the boundary conditions of the reference deck:
        # nodes 1, 2, 4 are fully fixed; node 3 is fixed in X only.
        node_pgs={
            "Fixed_All": [1, 2, 4],
            "Fixed_X_only": [3],
        },
    )
    elements = _ElementsStub(
        elem_pgs={
            "Rock": _ElementGroupView(
                ids=(1,), connectivity=((1, 2, 3, 4),),
            ),
        },
    )
    return FEMStub(nodes=nodes, elements=elements)


def _build_full_apesees_deck(stress_recorder_path: str) -> apeSees:
    """Build the full SSI patch test via apeSees primitives.

    Mirrors the reference test_stressctrl.tcl::build_patch +
    setup_analysis + apeSees initial_stress.
    """
    fem = _make_single_quad_fem()
    ops = apeSees(cast("object", fem), default_orientation=None)  # type: ignore[arg-type]
    ops.model(ndm=2, ndf=2)

    # ASDPlasticMaterial3D MohrCoulomb — exact reference parameters.
    mat_3d = ops.nDMaterial.MohrCoulombSoil(
        c=1014.0,
        phi=45.95,
        psi=11.49,
        E=4080000.0,
        nu=0.18,
        rho=4.5,
    )
    # Plane-strain wrapping — ASDPlasticMaterial3D is strictly 3D,
    # so the 2D quad element needs the PlaneStrain wrapper to bridge
    # the constitutive interface.
    mat_2d = ops.nDMaterial.PlaneStrain(base=mat_3d)
    ops.element.FourNodeQuad(
        pg="Rock",
        thickness=1.0,
        material=mat_2d,
        plane_type="PlaneStrain",
    )

    # Boundary conditions matching the reference deck.
    ops.fix(pg="Fixed_All", dofs=(1, 1))      # nodes 1, 2, 4
    ops.fix(pg="Fixed_X_only", dofs=(1, 0))   # node 3 free in Y

    ops.constraints.Plain()
    ops.numberer.Plain()
    ops.system.FullGeneral()
    ops.test.NormDispIncr(tol=1e-8, max_iter=50)
    ops.algorithm.Newton()
    ops.integrator.LoadControl(dlam=0.1)
    ops.analysis.Static()

    # Element recorder for the per-material stress history.  The
    # bridge translates ``pg="Rock"`` through ``fem_eid_to_ops_tag``
    # so the emitted ``-ele <tag>`` references the actual OpenSees
    # element tag (2) the FourNodeQuad fan-out allocated, not the
    # FEM eid (1).
    ops.recorder.Element(
        file=stress_recorder_path,
        response=("material", "1", "stress"),
        pg="Rock",
        time_format="dt",
    )

    ops.initial_stress(
        name="rock_insitu",
        pg="Rock",
        sigma_xx=-6300.0,
        sigma_yy=-6300.0,
        sigma_zz=-6300.0,
        ramp_steps=10,
    )

    return ops


def _read_reference_fixed() -> dict[int, tuple[float, float, float]]:
    """Read result_fixed.csv → {step: (sxx, syy, szz)}."""
    out: dict[int, tuple[float, float, float]] = {}
    with REFERENCE_FIXED_CSV.open("r", newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            step = int(row["step"])
            out[step] = (
                float(row["sxx_committed"]),
                float(row["syy_committed"]),
                float(row["szz_committed"]),
            )
    return out


def _read_recorder_output(csv_path: Path) -> dict[int, tuple[float, float, float]]:
    """Element recorder ``-time -ele 1 material 1 stress`` writes rows
    ``time sxx syy szz sxy``.  ``dt=0.1`` → step k has time = 0.1*k."""
    out: dict[int, tuple[float, float, float]] = {}
    with csv_path.open("r") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            t = float(parts[0])
            step = int(round(t / 0.1))
            sxx = float(parts[1])
            syy = float(parts[2])
            szz = float(parts[3])
            out[step] = (sxx, syy, szz)
    return out


def test_initial_stress_ramp_matches_fixed_reference_full_apesees(
    tmp_path: Path,
) -> None:
    """Full apeSees emit produces the FIXED ramp values; step-5
    discriminator confirms the STKO bug was not replicated."""
    deck_path = tmp_path / "deck.tcl"
    stress_out = tmp_path / "stress.out"

    # The bridge now emits the Element recorder natively (the
    # ``Recorder.materialize`` chain translates ``pg="Rock"`` into the
    # OpenSees element tag via ``fem_eid_to_ops_tag``).  No raw-Tcl
    # post-pend.
    stress_out_posix = str(stress_out).replace("\\", "/")
    ops = _build_full_apesees_deck(stress_out_posix)
    ops.tcl(str(deck_path), analyze_steps=10, analyze_dt=0.1)

    binary = os.environ.get("OPENSEES_BIN") or shutil.which("OpenSees")
    assert binary is not None
    proc = subprocess.run(
        [binary, str(deck_path)],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, (
        f"OpenSees exit {proc.returncode}\n"
        f"--- stdout (last 100 lines) ---\n"
        f"{chr(10).join(proc.stdout.splitlines()[-100:])}\n"
        f"--- stderr (last 100 lines) ---\n"
        f"{chr(10).join(proc.stderr.splitlines()[-100:])}"
    )
    assert stress_out.exists(), (
        f"Recorder file not produced.\n"
        f"--- stdout ---\n{proc.stdout[-2000:]}\n"
        f"--- stderr ---\n{proc.stderr[-2000:]}\n"
        f"--- deck ---\n{deck_path.read_text()}"
    )

    actual = _read_recorder_output(stress_out)
    reference = _read_reference_fixed()

    TOL = 0.5  # kPa per spec

    failures: list[str] = []
    for step in sorted(reference.keys()):
        if step not in actual:
            failures.append(f"step {step}: missing from actual")
            continue
        a_sxx, a_syy, a_szz = actual[step]
        r_sxx, r_syy, r_szz = reference[step]
        for label, a_v, r_v in [
            ("sxx", a_sxx, r_sxx),
            ("syy", a_syy, r_syy),
            ("szz", a_szz, r_szz),
        ]:
            if abs(a_v - r_v) > TOL:
                failures.append(
                    f"step {step} {label}: actual={a_v:.4f} "
                    f"reference={r_v:.4f} delta={a_v-r_v:+.4f} "
                    f"(>{TOL} kPa)"
                )
    assert not failures, "\n".join(failures)

    # Explicit step-5 discriminator.
    step5_sxx = actual[5][0]
    assert step5_sxx > -4000.0, (
        f"step 5 sxx = {step5_sxx:.2f} kPa — STKO BUGGY pattern "
        "(single-step jump). The ramp divisor is /1.0 not /n_steps."
    )
    assert abs(step5_sxx - (-3023.87)) < 1.0, (
        f"step 5 sxx = {step5_sxx:.2f} kPa expected ≈ -3023.87 (FIXED)"
    )
