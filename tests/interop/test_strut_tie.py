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

from apeGmsh.interop.strut_tie import strut_tie_overlays, write_strut_tie_overlays

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
