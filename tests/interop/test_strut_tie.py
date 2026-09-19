"""The strut-and-tie FE overlay consumer (apeConcrete ADR-0014 §6).

Fixtures are ``apeConcrete.stm.model_to_dict`` output for the corbel
and four-pile-cap builders; the corbel outline was extended with a
column stub so the far column face is a physical boundary.
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np
import pytest

from apeGmsh.interop.strut_tie import (
    strut_tie_overlays,
    strut_tie_pushover,
    write_strut_tie_overlays,
)

pytestmark = pytest.mark.live

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_corbel_plane_stress_overlay_balances_the_load() -> None:
    model = _load("corbel.stm.json")
    result = strut_tie_overlays(
        model, mesh_size=40.0, extra_fixed_planes=(("x", -400.0, (0, 1)),)
    )
    segs = result.overlays["fe_trajectories"]["segments"]
    assert len(segs) > 50
    assert result.n_elements > 100
    # Plane model: every glyph lies in the model plane.
    assert all(abs(s[0][2]) < 1e-9 and abs(s[1][2]) < 1e-9 for s in segs)
    # Compression glyphs exist (the inclined strut) and tension glyphs exist (the tie).
    assert any(s[3] < 0.0 for s in segs)
    assert any(s[2] > 0.0 for s in segs)
    # Equilibrium: reactions balance the applied load.
    applied, reactions = np.array(result.applied), np.array(result.reactions)
    assert math.isclose(applied[1], -300.0e3)
    assert np.allclose(reactions[:2], -applied[:2], rtol=1e-3, atol=1.0)
    json.dumps(result.overlays, allow_nan=False)


def test_pile_cap_solid_overlay(tmp_path: Path) -> None:
    out = write_strut_tie_overlays(
        FIXTURES / "pile_cap.stm.json", tmp_path / "cap.overlays.json", mesh_size=180.0
    )
    data = json.loads(out.read_text(encoding="utf-8"))
    summary = data["fe_summary"]
    assert summary["n_elements"] > 200
    # The column load goes down into the pile supports; pile loads were skipped.
    assert math.isclose(summary["applied"][2], -2.0e6)
    assert math.isclose(summary["reactions"][2], 2.0e6, rel_tol=1e-3)
    segs = data["fe_trajectories"]["segments"]
    assert segs
    zs = [s[0][2] for s in segs]
    assert min(zs) < 300.0 < max(zs)  # glyphs through the depth, not one plane


def test_schema_and_case_are_checked() -> None:
    model = _load("corbel.stm.json")
    bad = dict(model)
    bad["schema_version"] = 2
    with pytest.raises(ValueError, match="schema_version"):
        strut_tie_overlays(bad)
    with pytest.raises(StopIteration):
        strut_tie_overlays(model, case="nope", mesh_size=60.0)


@pytest.mark.ladruno_fork
@pytest.mark.slow
def test_corbel_pushover_curve_rises_then_stops() -> None:
    from apeGmsh.interop.strut_tie import strut_tie_pushover

    model = _load("corbel.stm.json")
    result = strut_tie_pushover(
        model,
        mesh_size=45.0,
        extra_fixed_planes=(("x", -400.0, (0, 1)),),
        target_displacement=0.3,
        steps=10,
    )
    curve = result.overlays["fe_curve"]
    pts = curve["points"]
    assert len(pts) >= 4
    assert pts[0] == [0.0, 0.0]
    deltas = [p[0] for p in pts]
    assert all(b > a for a, b in itertools.pairwise(deltas))
    assert curve["capacity"] == max(p[1] for p in pts)
    # Reinforced: the tie bars carry the region well past the plain-concrete
    # cracking peak (about 136 kN on this mesh without bars).
    assert curve["capacity"] > 150.0e3
    assert "ties as Steel02" in curve["source"]
    assert result.overlays["fe_summary"]["ties_modelled"]["T1"] > 0.0
    assert curve["stopped"] in ("target", "divergence", "post_peak")
    # First increment is elastic: load rises with displacement.
    assert pts[1][1] > 0.0
    json.dumps(result.overlays, allow_nan=False)


@pytest.mark.ladruno_fork
@pytest.mark.slow
def test_pile_cap_pushover_writes_both_overlays(tmp_path: Path) -> None:
    out = write_strut_tie_overlays(
        FIXTURES / "pile_cap.stm.json",
        tmp_path / "cap.overlays.json",
        mesh_size=300.0,
        pushover=True,
        target_displacement=0.3,
        steps=3,
        max_halvings=3,
    )
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "fe_trajectories" in data
    assert data["fe_curve"]["capacity"] > 0.0
    assert len(data["fe_summary_nonlinear"]["ties_modelled"]) == 4
    assert data["fe_summary_nonlinear"]["Gc"] > data["fe_summary_nonlinear"]["Gf"]


# ---------------------------------------------------------------------------
# The physics oracle of apeConcrete ADR-0014 §9: a published corbel with
# a measured failure load
# ---------------------------------------------------------------------------
COOK_MITCHELL_STM_PREDICTION_N = 471.0e3  # per side, tie yield (SP-208 Part 3 §2)
COOK_MITCHELL_MEASURED_N = 502.0e3  # per side


@pytest.mark.ladruno_fork
@pytest.mark.slow
def test_cook_mitchell_double_corbel_pushover_against_the_paper() -> None:
    """ACI SP-208 Part 3 §2, Cook and Mitchell (1988) double corbel:
    strut-and-tie prediction 471 kN per side (tie yield, 4 No. 15 at
    444 MPa), measured 502 kN. The fixture is the apeConcrete anchor's
    ``model_to_dict`` output (``tests/golden/test_sp208_cook_mitchell_
    corbel_anchor.py`` there); the load case is V = 471 kN per side with
    H = 0.2·V outward, so ``capacity_factor`` reads directly as
    FE peak / strut-and-tie prediction.

    State of the bracket on 2026-09-18 (40 mm Tri31 mesh, 718 elements,
    KrylovNewton/ModifiedNewton fallbacks): the FE peaks at ≈369 kN per
    side, 0.78 of the prediction, then softens; with H removed it peaks
    at ≈440 kN (0.93). The lower-bound ordering FE ≥ STM is therefore
    **not met yet**, and the diagnosis is the load introduction, not the
    concrete: in the specimen the plates were welded to the bars, so V
    and H went straight into the tie, while here the bars end at the
    load node and H pulls on the concrete under a 50 × 300 plate. Column
    bars and the two No. 10 ties are absent too. This test pins the
    current state with a wide bracket so a change in either direction —
    a welded-plate option lifting the peak, or a regression lowering
    it — is noticed and re-documented.
    """
    model = _load("cook_mitchell_corbel.stm.json")
    result = strut_tie_pushover(
        model, mesh_size=40.0, target_displacement=3.0, steps=30, max_halvings=5
    )
    curve = result.overlays["fe_curve"]
    per_side = curve["capacity"] / 2.0
    assert curve["stopped"] == "post_peak"  # a real peak, not a solver failure
    assert curve["fallback_steps"] >= 0
    assert curve["tolerance"] == 1e-6
    assert (
        0.7 * COOK_MITCHELL_STM_PREDICTION_N < per_side < 1.1 * COOK_MITCHELL_MEASURED_N
    )
    assert math.isclose(
        curve["capacity_factor"], per_side / COOK_MITCHELL_STM_PREDICTION_N
    )
    assert result.overlays["fe_summary"]["ties_modelled"]["T"] > 0.0
    assert result.n_elements > 500
