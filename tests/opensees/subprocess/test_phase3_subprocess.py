"""Subprocess acceptance for Phase SSI-3 convenience helpers.

Two scenarios:

1. ``ops.convergence_confinement(...)`` — drives the same 1-element
   MohrCoulomb patch test as the Phase SSI-1 acceptance, but with
   the convergence_confinement helper instead of ``initial_stress``.
   With ``lambda_target=1.0`` and ``n_steps=10`` the result must
   match the Phase SSI-1 fixed-variant reference within ±0.5 kPa
   per step (the helper is a thin delegation, so byte-equivalent
   numerical output is expected).

2. ``ops.imposed_displacement(...)`` — prescribes a displacement
   on one corner of an elastic quad; verifies the deck parses /
   runs on Ladruno and the prescribed-corner displacement actually
   reaches the prescribed value at the final analyze step.
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
]


def _make_single_quad_fem() -> FEMStub:
    return FEMStub(
        nodes=_NodesStub(
            ids=[1, 2, 3, 4],
            coords=[
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
            ],
            node_pgs={
                "Fixed_All": [1, 2, 4],
                "Fixed_X_only": [3],
            },
        ),
        elements=_ElementsStub(
            elem_pgs={
                "Rock": _ElementGroupView(
                    ids=(1,), connectivity=((1, 2, 3, 4),),
                ),
            },
        ),
    )


def _read_reference_fixed() -> dict[int, tuple[float, float, float]]:
    out: dict[int, tuple[float, float, float]] = {}
    if not REFERENCE_FIXED_CSV.exists():
        return out
    with REFERENCE_FIXED_CSV.open("r", newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            out[int(row["step"])] = (
                float(row["sxx_committed"]),
                float(row["syy_committed"]),
                float(row["szz_committed"]),
            )
    return out


@pytest.mark.skipif(
    not REFERENCE_FIXED_CSV.exists(),
    reason=f"Reference CSV not found at {REFERENCE_FIXED_CSV}",
)
def test_convergence_confinement_matches_initial_stress_reference(
    tmp_path: Path,
) -> None:
    """``ops.convergence_confinement(...)`` with ``lambda_target=1.0``
    produces the same ramp as ``ops.initial_stress(...)`` with
    ``lambda_install=1.0``.  We rerun the Phase SSI-1 patch test using
    the new helper and assert the same σ values per step (±0.5 kPa)."""
    fem = _make_single_quad_fem()
    ops = apeSees(cast("object", fem), default_orientation=None)  # type: ignore[arg-type]
    ops.model(ndm=2, ndf=2)
    mat_3d = ops.nDMaterial.MohrCoulombSoil(
        c=1014.0, phi=45.95, psi=11.49,
        E=4080000.0, nu=0.18, rho=4.5,
    )
    mat_2d = ops.nDMaterial.PlaneStrain(base=mat_3d)
    ops.element.FourNodeQuad(
        pg="Rock", thickness=1.0, material=mat_2d, plane_type="PlaneStrain",
    )
    ops.fix(pg="Fixed_All", dofs=(1, 1))
    ops.fix(pg="Fixed_X_only", dofs=(1, 0))
    ops.constraints.Plain()
    ops.numberer.Plain()
    ops.system.FullGeneral()
    ops.test.NormDispIncr(tol=1e-8, max_iter=50)
    ops.algorithm.Newton()
    ops.integrator.LoadControl(dlam=0.1)
    ops.analysis.Static()

    # The helper does the same thing as initial_stress with renamed
    # kwargs (lambda_target / n_steps).
    ops.convergence_confinement(
        name="rock_insitu",
        pg="Rock",
        sigma_xx=-6300.0,
        sigma_yy=-6300.0,
        sigma_zz=-6300.0,
        lambda_target=1.0,
        n_steps=10,
    )

    deck = tmp_path / "deck.tcl"
    stress_out = tmp_path / "stress.out"
    ops.tcl(str(deck), analyze_steps=10, analyze_dt=0.1)

    # Same Element-recorder workaround as the Phase SSI-1 acceptance.
    text = deck.read_text()
    quad_tag = next(
        int(ln.split()[2])
        for ln in text.splitlines() if ln.startswith("element quad ")
    )
    stress_out_posix = str(stress_out).replace("\\", "/")
    recorder = (
        f"recorder Element -file {stress_out_posix} -time -ele "
        f"{quad_tag} material 1 stress\n"
    )
    text = text.replace(
        "for {set _apesees_i",
        recorder + "for {set _apesees_i",
    )
    deck.write_text(text)

    binary = os.environ.get("OPENSEES_BIN") or shutil.which("OpenSees")
    assert binary is not None
    proc = subprocess.run(
        [binary, str(deck)],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, (
        f"OpenSees exit {proc.returncode}\n"
        f"--- stderr (tail) ---\n"
        f"{chr(10).join(proc.stderr.splitlines()[-30:])}"
    )
    assert stress_out.exists()

    reference = _read_reference_fixed()
    actual: dict[int, tuple[float, float, float]] = {}
    with stress_out.open() as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            t = float(parts[0])
            step = int(round(t / 0.1))
            actual[step] = (float(parts[1]), float(parts[2]), float(parts[3]))

    failures: list[str] = []
    for step in sorted(reference.keys()):
        if step not in actual:
            failures.append(f"step {step}: missing")
            continue
        for label, a_v, r_v in zip(
            ("sxx", "syy", "szz"), actual[step], reference[step],
        ):
            if abs(a_v - r_v) > 0.5:
                failures.append(
                    f"step {step} {label}: {a_v:.2f} vs ref {r_v:.2f}"
                )
    assert not failures, "\n".join(failures)


def test_imposed_displacement_runs_on_ladruno(tmp_path: Path) -> None:
    """An elastic quad with one corner free and the opposite corner
    pushed by ``imposed_displacement`` deforms — the deck parses,
    runs, and the prescribed-corner displacement reaches the
    prescribed value at the final step."""
    fem = FEMStub(
        nodes=_NodesStub(
            ids=[1, 2, 3, 4],
            coords=[
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
            ],
            node_pgs={
                "Anchor": [1, 4],   # left edge — fully fixed
                "Pusher": [2],      # bottom-right corner — push in -y
            },
        ),
        elements=_ElementsStub(
            elem_pgs={
                "Rock": _ElementGroupView(
                    ids=(1,), connectivity=((1, 2, 3, 4),),
                ),
            },
        ),
    )
    ops = apeSees(cast("object", fem), default_orientation=None)  # type: ignore[arg-type]
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1.0e6, nu=0.25, rho=0.0)
    ops.element.FourNodeQuad(
        pg="Rock", thickness=1.0, material=mat, plane_type="PlaneStrain",
    )
    ops.fix(pg="Anchor", dofs=(1, 1))
    # Push node 2 by uy = -0.05 at the final step.  Linear time
    # series with factor=1.0 plus dt=0.1 over 10 steps gives
    # t_final = 1.0; pattern_factor=0.05 ⇒ disp_at_end = 0.05·1 = 0.05.
    ops.imposed_displacement(
        nodes=[2], uy=-1.0, pattern_factor=0.05,
    )
    # ``constraints Plain`` only handles homogeneous SP constraints
    # (``fix``).  Non-zero SPs from ``imposed_displacement`` need
    # Transformation / Penalty / Lagrange.
    ops.constraints.Transformation()
    ops.numberer.Plain()
    ops.system.FullGeneral()
    ops.test.NormDispIncr(tol=1e-8, max_iter=50)
    ops.algorithm.Linear()  # elastic system — linear solver is enough.
    ops.integrator.LoadControl(dlam=0.1)
    ops.analysis.Static()

    deck = tmp_path / "deck.tcl"
    disp_out = tmp_path / "disp.out"

    ops.tcl(str(deck), analyze_steps=10)

    # Insert a node-displacement recorder at the top so it captures
    # node 2's uy across the analyze loop.  The Node recorder uses
    # node tags directly (no fan-out bug — that one only affects
    # Element recorders).
    text = deck.read_text()
    text = text.replace(
        "analyze 10",
        f"recorder Node -file {str(disp_out).replace(chr(92), '/')} "
        "-time -node 2 -dof 2 disp\nanalyze 10",
    )
    deck.write_text(text)

    binary = os.environ.get("OPENSEES_BIN") or shutil.which("OpenSees")
    assert binary is not None
    proc = subprocess.run(
        [binary, str(deck)],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, (
        f"OpenSees exit {proc.returncode}\n"
        f"--- stderr (tail) ---\n"
        f"{chr(10).join(proc.stderr.splitlines()[-30:])}"
    )
    assert disp_out.exists()

    rows = [
        ln.split()
        for ln in disp_out.read_text().splitlines() if ln.strip()
    ]
    # 10 analyze steps → 10 rows (t = 0.1, 0.2, ..., 1.0).
    assert len(rows) == 10
    final_uy = float(rows[-1][1])
    # Prescribed uy at t=1.0: -1.0 × pattern_factor × t = -0.05.
    assert abs(final_uy - (-0.05)) < 1e-6, (
        f"final uy = {final_uy} expected ≈ -0.05 "
        "(imposed_displacement didn't apply the prescribed value)"
    )
