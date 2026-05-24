"""Subprocess acceptance test for Phase SSI-2.A staged analysis.

Builds a 2-stage 1-element ASDPlasticMaterial3D MohrCoulomb patch
test via apeSees, emits the Tcl deck, runs it against the Ladruno
OpenSees binary, and verifies:

1. The deck **parses and runs** without OpenSees errors.
2. Per-stage stress accumulation works — after stage 1's ramp the
   committed σxx is ~-6300; after stage 2's incremental ramp the
   committed σxx jumps further by stage 2's installed increment.
3. The hook-list clear between stages prevents stage 1's ramp from
   firing in stage 2's analyze loop.

Gated on the Ladruno OpenSees binary being available.
"""
from __future__ import annotations

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


def _full_chain(ops: apeSees) -> dict[str, object]:
    return {
        "test":        ops.test.NormDispIncr(tol=1e-8, max_iter=50),
        "algorithm":   ops.algorithm.Newton(),
        "integrator":  ops.integrator.LoadControl(dlam=0.1),
        "constraints": ops.constraints.Plain(),
        "numberer":    ops.numberer.Plain(),
        "system":      ops.system.FullGeneral(),
        "analysis":    ops.analysis.Static(),
    }


def test_two_stage_deck_runs_on_ladruno(tmp_path: Path) -> None:
    """2-stage 1-element MohrCoulomb deck builds via apeSees, emits
    Tcl, runs on OpenSees.exe without errors, and produces non-empty
    stress recorder output for both stages."""
    fem = _make_single_quad_fem()
    ops = apeSees(cast("object", fem), default_orientation=None)  # type: ignore[arg-type]
    ops.model(ndm=2, ndf=2)
    mat_3d = ops.nDMaterial.MohrCoulombSoil(
        c=1014.0, phi=45.95, psi=11.49,
        E=4080000.0, nu=0.18, rho=4.5,
    )
    mat_2d = ops.nDMaterial.PlaneStrain(base=mat_3d)
    ops.element.FourNodeQuad(
        pg="Rock", thickness=1.0, material=mat_2d,
        plane_type="PlaneStrain",
    )
    ops.fix(pg="Fixed_All", dofs=(1, 1))
    ops.fix(pg="Fixed_X_only", dofs=(1, 0))

    # Stage 1: install -6300 kPa over 10 steps.
    with ops.stage(name="insitu") as s:
        s.add(ops.initial_stress(
            name="rock_in", pg="Rock",
            sigma_xx=-6300.0, sigma_yy=-6300.0, sigma_zz=-6300.0,
            ramp_steps=10,
        ))
        s.analysis(**_full_chain(ops))
        s.run(n_increments=10, dt=0.1)

    # Stage 2: install another -1000 kPa increment over 5 steps.
    with ops.stage(name="incr") as s:
        s.add(ops.initial_stress(
            name="rock_incr", pg="Rock",
            sigma_xx=-1000.0, sigma_yy=-1000.0, sigma_zz=-1000.0,
            ramp_steps=5,
        ))
        s.analysis(**_full_chain(ops))
        s.run(n_increments=5, dt=0.1)

    deck = tmp_path / "deck.tcl"
    ops.tcl(str(deck))

    # Post-pend a stress recorder.  Same workaround as the Phase
    # SSI-1 acceptance test (Element recorder pg= bug, see spawned
    # task).  The element's OpenSees tag is parsed from the
    # ``element quad`` line in the emitted deck.
    text = deck.read_text()
    quad_tag = next(
        int(ln.split()[2])
        for ln in text.splitlines() if ln.startswith("element quad ")
    )
    stress_out = tmp_path / "stress.out"
    stress_out_posix = str(stress_out).replace("\\", "/")
    # Insert recorder before the FIRST stage banner so it captures
    # both stages.
    text = text.replace(
        "# === Stage: insitu ===",
        f"recorder Element -file {stress_out_posix} -time -ele "
        f"{quad_tag} material 1 stress\n"
        "# === Stage: insitu ===",
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
        f"--- stdout (tail) ---\n"
        f"{chr(10).join(proc.stdout.splitlines()[-50:])}\n"
        f"--- stderr (tail) ---\n"
        f"{chr(10).join(proc.stderr.splitlines()[-50:])}"
    )

    assert stress_out.exists()
    rows = [
        ln.split()
        for ln in stress_out.read_text().splitlines()
        if ln.strip()
    ]
    # Stage 1 (10 steps) + Stage 2 (5 steps) = 15 rows.
    assert len(rows) == 15, (
        f"expected 15 stress rows (10 stage-1 + 5 stage-2), "
        f"got {len(rows)}.  Output:\n{stress_out.read_text()}"
    )
    # After stage 1's last step σxx ≈ -5980.76 (matches Phase SSI-1
    # reference — same single-element + 1-free-DOF configuration).
    stage_1_final = float(rows[9][1])
    assert abs(stage_1_final - (-5980.76)) < 5.0, (
        f"stage 1 final σxx = {stage_1_final:.2f} kPa, "
        "expected ≈ -5980.76 kPa (Phase SSI-1 reference)."
    )
    # After stage 2's last step σxx should be further negative
    # (stage 2 ramps an additional -1000 increment on top).  We
    # assert it's monotonically more compressive than stage 1's
    # final — the exact value depends on the material's nonlinear
    # response, so we don't lock a precise number.
    stage_2_final = float(rows[14][1])
    assert stage_2_final < stage_1_final, (
        f"stage 2 final σxx = {stage_2_final:.2f} kPa should be more "
        f"compressive than stage 1 final {stage_1_final:.2f} kPa "
        "(the stage-2 ramp added another -1000 kPa increment)."
    )
    # And the hook-list clear means stage 2 ONLY runs its own ramp,
    # not stage 1's.  If clearing failed, stage 2's analyze loop
    # would re-fire stage 1's proc — which has already saturated
    # its factor at 1.0 (no-op).  So the negative-finding test
    # would not catch that bug.  Instead, check that the deck's
    # textual structure has the explicit clear lines after the
    # first stage_close.
    deck_text = deck.read_text()
    first_close_idx = deck_text.find("loadConst -time 0.0")
    second_close_idx = deck_text.find("loadConst -time 0.0", first_close_idx + 1)
    assert first_close_idx >= 0 and second_close_idx > first_close_idx
    between = deck_text[first_close_idx:second_close_idx]
    assert "set _apesees_before_step_hooks {}" in between
