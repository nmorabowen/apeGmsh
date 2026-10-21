"""The site-response soil nDMaterials (#2386, #2387, #2390).

``PressureIndependMultiYield``, ``PressureDependMultiYield(02)``,
``PM4Sand`` and ``PM4Silt``: the argv each emits against its OpenSees
parser, the parser's own refusals made client-side, the
``ops.nDMaterial`` wrappers, the staged ``updateMaterialStage`` flip
(stage 2 for the pressure-dependent pair only) and the build refusal of
a multi-yield soil under ``algorithm Linear``.
"""
from __future__ import annotations

from typing import Any

import pytest

from apeGmsh.opensees.apesees import apeSees
from apeGmsh.opensees._internal.build.errors import BridgeError
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.material.nd import (
    PM4Sand,
    PM4Silt,
    PressureDependMultiYield,
    PressureDependMultiYield02,
    PressureIndependMultiYield,
)

from tests.opensees.unit.test_stage_ssi_2e_mutators import (
    _full_chain,
    _make_two_pg_fem,
)


_PIMY: dict[str, Any] = dict(
    nd=2, rho=1.8, ref_shear_modulus=6.0e4, ref_bulk_modulus=3.0e5,
    cohesion=40.0,
)
_PDMY: dict[str, Any] = dict(
    nd=2, rho=2.0, ref_shear_modulus=9.0e4, ref_bulk_modulus=2.2e5,
    friction_angle=32.0, peak_shear_strain=0.1, ref_pressure=80.0,
    press_depend_coe=0.5, pt_angle=26.0, contrac=0.21, dilat1=0.0,
    dilat2=0.0, liquefac1=10.0, liquefac2=0.02, liquefac4=1.0,
)
_PDMY02: dict[str, Any] = dict(
    nd=2, rho=2.0, ref_shear_modulus=9.0e4, ref_bulk_modulus=2.2e5,
    friction_angle=32.0, peak_shear_strain=0.1, ref_pressure=101.0,
    press_depend_coe=0.5, pt_angle=26.0, contrac1=0.067, contrac3=0.23,
    dilat1=0.06, dilat3=0.27,
)
_PM4SAND: dict[str, Any] = dict(Dr=0.55, G0=677.0, hpo=0.4, rho=1.7)
_PM4SILT: dict[str, Any] = dict(Su=0.0, Su_rate=0.25, G0=500.0, hpo=0.1,
                                rho=1.7)


def _argv(mat: object, tag: int = 3) -> tuple[Any, ...]:
    rec = RecordingEmitter()
    mat._emit(rec, tag)  # type: ignore[attr-defined]
    ((verb, args, _kw),) = rec.calls
    assert verb == "nDMaterial"
    return tuple(args)


# ---------------------------------------------------------------------------
# argv against the parsers
# ---------------------------------------------------------------------------

def test_pimy_emits_the_full_ten_value_block() -> None:
    # OPS_PressureIndependMultiYield reads the eight doubles and
    # noYieldSurf as one block: every value is always written.
    assert _argv(PressureIndependMultiYield(**_PIMY)) == (
        "PressureIndependMultiYield", 3, 2, 1.8, 6.0e4, 3.0e5, 40.0,
        0.1, 0.0, 100.0, 0.0, 20,
    )


def test_pimy_backbone_writes_minus_n_then_the_pairs() -> None:
    m = PressureIndependMultiYield(
        **_PIMY, backbone=((1e-4, 0.9), (1e-3, 0.6), (1e-2, 0.2)))
    assert _argv(m)[11:] == (-3, 1e-4, 0.9, 1e-3, 0.6, 1e-2, 0.2)


def test_pdmy_emits_required_block_then_full_optional_tail() -> None:
    argv = _argv(PressureDependMultiYield(**_PDMY))
    assert argv[:17] == (
        "PressureDependMultiYield", 3, 2, 2.0, 9.0e4, 2.2e5, 32.0, 0.1,
        80.0, 0.5, 26.0, 0.21, 0.0, 0.0, 10.0, 0.02, 1.0,
    )
    # noYieldSurf, e, cs1..cs3, pa, c, hv, pv — the parser's defaults.
    assert argv[17:] == (20, 0.6, 0.9, 0.02, 0.7, 101.0, 0.3, 0.0, 1.0)


def test_pdmy_backbone_sits_between_n_surf_and_the_void_ratio() -> None:
    argv = _argv(PressureDependMultiYield(
        **_PDMY, backbone=((1e-4, 0.95), (1e-2, 0.3))))
    assert argv[17:22] == (-2, 1e-4, 0.95, 1e-2, 0.3)
    assert argv[22] == 0.6


def test_pdmy02_emits_thirteen_required_then_the_tail() -> None:
    argv = _argv(PressureDependMultiYield02(**_PDMY02))
    assert argv[2:15] == (
        2, 2.0, 9.0e4, 2.2e5, 32.0, 0.1, 101.0, 0.5, 26.0, 0.067, 0.23,
        0.06, 0.27)
    assert argv[15:] == (
        20, 5.0, 3.0, 1.0, 0.0, 0.6, 0.9, 0.02, 0.7, 101.0, 0.1, 0.0, 1.0)


def test_pm4sand_emits_4_required_and_24_optional() -> None:
    argv = _argv(PM4Sand(**_PM4SAND))
    assert argv[:6] == ("PM4Sand", 3, 0.55, 677.0, 0.4, 1.7)
    assert len(argv) == 2 + 4 + 24
    assert argv[6:8] == (101.3, -1.0)          # P_atm, h0 (computed)
    assert argv[-4:] == (5, 0, 1e-8, 1e-8)     # IntScheme TanType TolF TolR


def test_pm4silt_emits_5_required_and_24_optional() -> None:
    argv = _argv(PM4Silt(**_PM4SILT))
    assert argv[:7] == ("PM4Silt", 3, 0.0, 0.25, 500.0, 0.1, 1.7)
    assert len(argv) == 2 + 5 + 24
    assert argv[-4:] == (5, 0, 1e-7, 1e-10)


# ---------------------------------------------------------------------------
# Refusals (OpenSees exits, or silently resets, on these)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls, base, bad, match", [
    (PressureIndependMultiYield, _PIMY, {"nd": 1}, "nd must be 2 or 3"),
    (PressureIndependMultiYield, _PIMY, {"cohesion": 0.0},
     "cohesion is the strength"),
    (PressureIndependMultiYield, _PIMY, {"ref_shear_modulus": 0.0},
     "ref_shear_modulus must be > 0"),
    (PressureIndependMultiYield, _PIMY, {"press_depend_coe": 0.5},
     "needs friction_angle > 0"),
    (PressureIndependMultiYield, _PIMY, {"n_surf": 101}, "n_surf must be"),
    (PressureIndependMultiYield, _PIMY,
     {"backbone": ((1e-3, 0.5), (1e-4, 0.9))}, "strictly increasing"),
    (PressureIndependMultiYield, _PIMY, {"backbone": ((1e-3, 1.5),)},
     r"G/Gmax must be in \(0, 1\]"),
    (PressureIndependMultiYield, _PIMY,
     {"backbone": tuple((1e-4 * (k + 1), 0.5) for k in range(40))},
     "1..39"),
    (PressureDependMultiYield, _PDMY, {"pt_angle": 35.0},
     "must not exceed"),
    (PressureDependMultiYield, _PDMY, {"friction_angle": 0.0},
     "friction_angle must be > 0"),
    (PressureDependMultiYield, _PDMY, {"n_surf": 41}, "n_surf must be"),
    (PressureDependMultiYield02, _PDMY02, {"n_surf": 0}, "n_surf must be"),
    (PM4Sand, _PM4SAND, {"e_min": 0.9}, "e_min"),
    (PM4Sand, _PM4SAND, {"h0": -2.0}, "or -1"),
    (PM4Sand, _PM4SAND, {"int_scheme": 1.5}, "int_scheme"),
    (PM4Silt, _PM4SILT, {"Su_rate": 0.0}, "give Su or Su_rate"),
])
def test_parser_refusals_are_client_side(cls, base, bad, match) -> None:
    with pytest.raises(ValueError, match=match):
        cls(**{**base, **bad})


def test_pdmy02_does_not_bound_pt_angle_by_friction_angle() -> None:
    # Its parser commented the check out (PressureDependMultiYield02.cpp).
    PressureDependMultiYield02(**{**_PDMY02, "pt_angle": 35.0})


# ---------------------------------------------------------------------------
# ops.nDMaterial wrappers, staged flip, Linear refusal
# ---------------------------------------------------------------------------

def _soil_ops(mat_kind: str = "PressureIndependMultiYield",
              kw: "dict[str, Any] | None" = None) -> tuple[apeSees, object]:
    ops = apeSees(_make_two_pg_fem(), default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = getattr(ops.nDMaterial, mat_kind)(**(kw or _PIMY))
    ops.element.FourNodeQuad(pg="rock", thickness=1.0, material=mat)
    ops.element.FourNodeQuad(pg="cimbra", thickness=1.0, material=mat)
    return ops, mat


@pytest.mark.parametrize("kind, kw", [
    ("PressureIndependMultiYield", _PIMY),
    ("PressureDependMultiYield", _PDMY),
    ("PressureDependMultiYield02", _PDMY02),
    ("PM4Sand", _PM4SAND),
    ("PM4Silt", _PM4SILT),
])
def test_namespace_registers_and_stage_flip_emits(kind, kw) -> None:
    ops, mat = _soil_ops(kind, kw)
    assert type(mat).__name__ == kind and ops.tag_for(mat) is not None
    with ops.stage(name="gravity") as s:
        s.update_material_stage(materials=[mat], stage=0)
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    with ops.stage(name="plastic") as s:
        s.update_material_stage(materials=[mat], stage=1)
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    rec = RecordingEmitter()
    ops.build().emit(rec)
    flips = [c[1] for c in rec.calls if c[0] == "update_material_stage"]
    assert flips == [(ops.tag_for(mat), 0), (ops.tag_for(mat), 1)]


def test_stage_2_is_for_the_pressure_dependent_sands_only() -> None:
    ops, pdmy = _soil_ops("PressureDependMultiYield", _PDMY)
    pimy = ops.nDMaterial.PressureIndependMultiYield(**_PIMY)
    with ops.stage(name="s") as s:
        s.update_material_stage(materials=[pdmy], stage=2)
        with pytest.raises(ValueError, match="takes 0 or 1"):
            s.update_material_stage(materials=[pimy], stage=2)
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)


def test_pm4sand_post_shake_rides_update_parameter() -> None:
    ops, sand = _soil_ops("PM4Sand", _PM4SAND)
    with ops.stage(name="post") as s:
        rec = s.update_parameter("PostShake", 1.0, pg="rock", material=sand)
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    assert rec.mat_tag == ops.tag_for(sand) and rec.name == "PostShake"


def _linear_chain(ops: apeSees) -> dict[str, object]:
    return {**_full_chain(ops), "algorithm": ops.algorithm.Linear()}


def test_multi_yield_soil_under_linear_is_refused_flat() -> None:
    ops, _mat = _soil_ops()
    _linear_chain(ops)  # the chain primitives register as they are made
    with pytest.raises(BridgeError, match="algorithm Linear"):
        ops.build().emit(RecordingEmitter())


def test_multi_yield_soil_under_linear_is_refused_in_a_stage() -> None:
    ops, mat = _soil_ops("PressureDependMultiYield", _PDMY)
    with ops.stage(name="gravity") as s:
        s.analysis(**_full_chain(ops))
        s.run(n_increments=1)
    with ops.stage(name="shake") as s:
        s.analysis(**_linear_chain(ops))
        s.run(n_increments=1)
    with pytest.raises(BridgeError, match="'shake'.*PressureDependMultiYield"):
        ops.build().emit(RecordingEmitter())


def test_newton_and_pm4sand_under_linear_pass() -> None:
    ops, _mat = _soil_ops()
    _full_chain(ops)
    ops.build().emit(RecordingEmitter())
    ops2, _sand = _soil_ops("PM4Sand", _PM4SAND)
    _linear_chain(ops2)
    ops2.build().emit(RecordingEmitter())
