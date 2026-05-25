"""Parity test for ``s.initial_stress(...)`` PUSH path.

Locks the contract that ``s.initial_stress(name=..., ...)`` (the PUSH
factory method on ``_StageBuilder``) produces a byte-identical Tcl
deck to the equivalent PULL pattern
``r = ops.initial_stress(name=..., ...); s.add(r)``.

Both paths land in the same per-stage emit hook; this test guarantees
that the only difference is which API surface the user typed at, not
the emitted deck.
"""
from __future__ import annotations

import re

from apeGmsh.opensees.apesees import apeSees

from tests.opensees.fixtures.fem_stub import (
    FEMStub,
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
)


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
            node_pgs={"Left": [1, 4], "Bottom": [1, 2]},
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
        "test":        ops.test.NormDispIncr(tol=1e-4, max_iter=50),
        "algorithm":   ops.algorithm.Newton(),
        "integrator":  ops.integrator.LoadControl(dlam=0.1),
        "constraints": ops.constraints.Plain(),
        "numberer":    ops.numberer.RCM(),
        "system":      ops.system.UmfPack(),
        "analysis":    ops.analysis.Static(),
    }


def _build_deck_pull(tmp_path) -> str:
    """PULL: ops.initial_stress registers globally; s.add binds it."""
    fem = _make_single_quad_fem()
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.FourNodeQuad(pg="Rock", thickness=1.0, material=mat)
    ops.fix(pg="Left", dofs=(1, 0))
    ops.fix(pg="Bottom", dofs=(0, 1))

    with ops.stage(name="insitu") as s:
        record = ops.initial_stress(
            name="rock_in", pg="Rock",
            sigma_xx=-100.0, sigma_yy=-100.0, sigma_zz=-100.0,
            ramp_steps=10,
        )
        s.add(record)
        s.analysis(**_full_chain(ops))
        s.run(n_increments=10, dt=0.1)

    out = tmp_path / "pull.tcl"
    ops.tcl(str(out))
    return out.read_text(encoding="utf-8")


def _build_deck_push(tmp_path) -> str:
    """PUSH: s.initial_stress creates the record in-place on the stage."""
    fem = _make_single_quad_fem()
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.FourNodeQuad(pg="Rock", thickness=1.0, material=mat)
    ops.fix(pg="Left", dofs=(1, 0))
    ops.fix(pg="Bottom", dofs=(0, 1))

    with ops.stage(name="insitu") as s:
        s.initial_stress(
            name="rock_in", pg="Rock",
            sigma_xx=-100.0, sigma_yy=-100.0, sigma_zz=-100.0,
            ramp_steps=10,
        )
        s.analysis(**_full_chain(ops))
        s.run(n_increments=10, dt=0.1)

    out = tmp_path / "push.tcl"
    ops.tcl(str(out))
    return out.read_text(encoding="utf-8")


def test_push_pull_parity_byte_identical(tmp_path) -> None:
    pull = _build_deck_pull(tmp_path)
    push = _build_deck_push(tmp_path)
    # The deck headers carry a timestamp on the comment line; strip
    # any "generated at ..." lines before comparing.
    strip = lambda txt: re.sub(r"^# (?:generated|written|created).*$", "", txt, flags=re.M)
    assert strip(push) == strip(pull), (
        "s.initial_stress PUSH must produce a byte-identical deck "
        "to the s.add(record) PULL path"
    )


def test_push_returns_the_record(tmp_path) -> None:
    """s.initial_stress returns the constructed record (mirrors PULL)."""
    fem = _make_single_quad_fem()
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.FourNodeQuad(pg="Rock", thickness=1.0, material=mat)

    with ops.stage(name="insitu") as s:
        rec = s.initial_stress(
            name="rock_in", pg="Rock",
            sigma_xx=-100.0, sigma_yy=-100.0, sigma_zz=-100.0,
            ramp_steps=10,
        )
        s.analysis(**_full_chain(ops))
        s.run(n_increments=10, dt=0.1)

    assert rec.name == "rock_in"
    assert rec.pg == "Rock"
    assert rec.ramp_steps == 10
    # The record is in the stage's pool, NOT the bridge's global pool.
    assert ops._initial_stress_records == []
    assert ops._stage_records[0].initial_stress_records == (rec,)


def test_push_validation_uses_stage_source_label(tmp_path) -> None:
    """Bad inputs to s.initial_stress raise with a stage-scoped error
    prefix (so users know which API surface they violated)."""
    import pytest

    fem = _make_single_quad_fem()
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.FourNodeQuad(pg="Rock", thickness=1.0, material=mat)

    with pytest.raises(ValueError, match=r"Stage 'insitu'\.initial_stress"):
        with ops.stage(name="insitu") as s:
            s.initial_stress(
                name="rock_in", pg=None, elements=None,
                sigma_xx=-100.0, sigma_yy=-100.0, sigma_zz=-100.0,
                ramp_steps=10,
            )
            s.analysis(**_full_chain(ops))
            s.run(n_increments=10, dt=0.1)
