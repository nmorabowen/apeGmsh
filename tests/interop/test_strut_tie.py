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
    _parse_fixed_plane,
    main,
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
    assert "fe_stress_paths" in data  # the bearing paths travel with the file
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


# ---------------------------------------------------------------------------
# The command line (what the office web app launches in the OpenSees venv)
# ---------------------------------------------------------------------------
def test_fixed_plane_syntax() -> None:
    assert _parse_fixed_plane("x=-400:01") == ("x", -400.0, (0, 1))
    assert _parse_fixed_plane("z=0:2") == ("z", 0.0, (2,))
    with pytest.raises(ValueError, match="AXIS=VALUE:DOFS"):
        _parse_fixed_plane("nonsense")


def test_cli_writes_the_linear_overlays(tmp_path: Path) -> None:
    out = tmp_path / "corbel.overlays.json"
    rc = main(
        [
            str(FIXTURES / "corbel.stm.json"),
            str(out),
            "--mesh-size",
            "60",
            "--fix-plane",
            "x=-400:01",
        ]
    )
    assert rc == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["fe_trajectories"]["segments"]
    assert "fe_curve" not in data
    assert math.isclose(data["fe_summary"]["applied"][1], -300.0e3)


def test_welded_plates_add_plate_chains_and_welds() -> None:
    """``weld_plates`` places each loaded plate as a stiff bar chain a quarter
    mesh size inside the concrete, welded to the tie node, and routes the
    tangential load into the tie node. Pushover numbers on the Cook and
    Mitchell corbel (40 mm, 2026-09-19): peak 412 kN per side at the 3 mm
    target, 0.87 × the prediction, against 369 kN without the weld — closer,
    still below, with the two No. 10 ties and the column bars still absent."""
    from apeGmsh.interop.strut_tie import _build

    model = _load("cook_mitchell_corbel.stm.json")
    fe = _build(
        model,
        case=None,
        mesh_size=60.0,
        extra_fixed_planes=(),
        verbose=False,
        reinforced=True,
        weld_plates=True,
    )
    assert set(fe.welds) == {"plate_L1", "weld_L1", "plate_L2", "weld_L2"}
    assert fe.welds["plate_L1"] == pytest.approx(25.0 * 300.0)
    assert fe.welds["weld_L1"] == pytest.approx(800.0)  # the tie's bars
    # the outward H goes to the tie node, the vertical V to the plate rows
    horizontal = [r for r, f in fe.nodal.items() if abs(f[0]) > 0.0]
    assert len(horizontal) == 2
    assert math.isclose(fe.applied[1], -2 * 471.0e3)
    with pytest.raises(ValueError, match="weld_plates needs"):
        _build(
            model,
            case=None,
            mesh_size=60.0,
            extra_fixed_planes=(),
            verbose=False,
            reinforced=False,
            weld_plates=True,
        )


def test_extra_bars_are_arranged_and_placed() -> None:
    """Hoops and column bars outside the strut-and-tie model, split where
    they cross the tie, the welds and each other, so gmsh can embed them.

    Cook and Mitchell corbel, 40 mm, welded plates + 2 hoops + 2 column
    bar lines (2026-09-19): implicit integration peaks at 410 kN per side
    (0.87 × the 471 kN prediction) and stops at 0.65 mm; IMPL-EX reaches a
    plateau of 683 kN at 0.1 mm steps and 614 kN at 0.025 mm steps (1.30 ×,
    1.22 × the measured 502 kN). The measurement sits inside that bracket:
    implicit under (premature numerical failure), IMPL-EX over (delayed
    damage by extrapolation)."""
    from apeGmsh.interop.strut_tie import _arrange, _build

    pts, segs = _arrange(
        {
            "a": [np.array([0.0, 0.0, 0.0]), np.array([10.0, 0.0, 0.0])],
            "b": [np.array([5.0, -5.0, 0.0]), np.array([5.0, 5.0, 0.0])],
            "c": [np.array([10.0, 0.0, 0.0]), np.array([10.0, 5.0, 0.0])],
        },
        tol=1e-6,
    )
    assert len(pts) == 6  # the crossing and the shared end are single points
    assert {k: len(v) for k, v in segs.items()} == {"a": 2, "b": 2, "c": 1}

    model = _load("cook_mitchell_corbel.stm.json")
    bars = _load("cook_mitchell_bars.json")
    fe = _build(
        model,
        case=None,
        mesh_size=60.0,
        extra_fixed_planes=(),
        verbose=False,
        reinforced=True,
        weld_plates=True,
        extra_bars=bars,
    )
    assert set(fe.bars) == {"hoop1", "hoop2", "col_left", "col_right"}
    assert fe.bars["col_left"] == pytest.approx(600.0)
    assert set(fe.welds) == {"plate_L1", "weld_L1", "plate_L2", "weld_L2"}


@pytest.mark.ladruno_fork
@pytest.mark.slow
def test_asd_material_reaches_a_plateau_and_records_the_bearing_stress_path() -> None:
    """The second concrete model, stock ``ASDConcrete3D``, on the welded and
    reinforced Cook and Mitchell corbel: it converges under plain Newton in
    seconds to a plateau of ≈600 kN per side (2026-09-19: 601 kN, 1.28 × the
    prediction, 1.20 × the measured 502 kN) where ``LadrunoConcrete3D``'s
    implicit return map stalls at 410 kN. The two materials therefore
    bracket the model; the bearing-element stress path recorded here is
    the data a fork investigation of the Ladruno return map starts from."""
    model = _load("cook_mitchell_corbel.stm.json")
    bars = _load("cook_mitchell_bars.json")
    result = strut_tie_pushover(
        model,
        mesh_size=40.0,
        target_displacement=2.0,
        steps=40,
        max_halvings=4,
        weld_plates=True,
        extra_bars=bars,
        material="asd",
    )
    curve = result.overlays["fe_curve"]
    per_side = curve["capacity"] / 2.0
    assert curve["material"] == "asd"
    assert "ASDConcrete3D" in curve["source"]
    assert (
        1.05 * COOK_MITCHELL_STM_PREDICTION_N
        < per_side
        < 1.5 * COOK_MITCHELL_STM_PREDICTION_N
    )
    paths = result.overlays["fe_stress_paths"]
    assert set(paths) == {"bearing_L1", "bearing_L2"}
    path = paths["bearing_L1"]["path"]
    assert len(path) >= 10
    assert paths["bearing_L1"]["components"] == ["s11", "s22", "s12"]
    # under the plate: vertical compression that builds up, then the
    # element unloads as it crushes — the path is the record of both
    s22 = [row[3] for row in path]
    assert s22[1] < 0.0
    assert min(s22) < s22[1]
    assert all(
        len(row) == 7 for row in path
    )  # delta, lambda, s11, s22, s12, sigma1, sigma3
    assert min(row[6] for row in path) <= min(
        s22
    )  # sigma3 is the compressive principal
