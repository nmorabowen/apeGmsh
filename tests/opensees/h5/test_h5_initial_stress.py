"""ADR 0055 Phase 1 — global ``ops.initial_stress(...)`` H5 archival.

Round-trips the GLOBAL initial-stress bucket (the staged bucket stays
fail-loud — see ``test_h5_staged_fail_loud.py``).  Locks:

1. ``ops.h5`` writes ``/opensees/initial_stress/stress_NNN`` with the
   declarative field set (no parameter tags, no rendered ramp proc).
2. ``OpenSeesModel.from_h5(...).initial_stress()`` rehydrates the records.
3. The H5 → H5 round-trip is ``model_hash``-stable, and the group folds
   into ``model_hash`` (a model with initial stress hashes differently
   from the same model without).
4. A vanilla model (no initial stress) writes NO group — byte-stable.
5. ``_replay_into`` re-runs the emit helpers and — the load-bearing
   adversarial fix — registers the step hook BEFORE ``analyze`` so the
   trailing analyze re-wraps into the hook-driven loop (without this the
   ramp procs declare but never fire).
"""
from __future__ import annotations

from pathlib import Path

import h5py

from apeGmsh.opensees.apesees import apeSees
from apeGmsh.opensees.emitter.tcl import TclEmitter
from apeGmsh.opensees.opensees_model import OpenSeesModel
from apeGmsh.opensees.section.fiber import FiberPoint

from tests.opensees.h5._opensees_model_fixtures import build_simple_frame_fem


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------


def _build_frame(*, with_initial_stress: bool) -> apeSees:
    """One-column frame on the ``"Cols"`` PG, optionally + initial stress."""
    fem = build_simple_frame_fem()
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=6)
    steel = ops.uniaxialMaterial.Steel02(fy=420e6, E=200e9, b=0.01)
    sec = ops.section.Fiber(
        GJ=1.0e9,
        fibers=(FiberPoint(material=steel, y=0.0, z=0.0, area=0.01),),
    )
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    integ = ops.beamIntegration.Lobatto(section=sec, n_ip=5)
    ops.element.forceBeamColumn(pg="Cols", transf=transf, integration=integ)
    if with_initial_stress:
        ops.initial_stress(
            name="insitu", pg="Cols",
            sigma_xx=-100.0, sigma_yy=-200.0, sigma_zz=-300.0,
            ramp_steps=10, lambda_install=0.5,
        )
    return ops


# ---------------------------------------------------------------------------
# 1. Writer — the group lands with the declarative field set
# ---------------------------------------------------------------------------


def test_initial_stress_group_written(tmp_path: Path) -> None:
    out = tmp_path / "model.h5"
    _build_frame(with_initial_stress=True).h5(str(out))

    with h5py.File(str(out), "r") as f:
        assert "opensees/initial_stress" in f
        g = f["opensees/initial_stress/stress_000"]
        assert g.attrs["name"] == "insitu"
        assert float(g.attrs["sigma_xx"]) == -100.0
        assert float(g.attrs["sigma_yy"]) == -200.0
        assert float(g.attrs["sigma_zz"]) == -300.0
        assert int(g.attrs["ramp_steps"]) == 10
        assert float(g.attrs["lambda_install"]) == 0.5
        # PG-targeted → a ``pg`` attr, NOT an ``elements`` dataset.
        assert g.attrs["pg"] == "Cols"
        assert "elements" not in g


def test_initial_stress_elements_form_written(tmp_path: Path) -> None:
    """An explicit ``elements=`` record stores the int64 dataset (not pg)."""
    fem = build_simple_frame_fem()
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=6)
    steel = ops.uniaxialMaterial.Steel02(fy=420e6, E=200e9, b=0.01)
    sec = ops.section.Fiber(
        GJ=1.0e9,
        fibers=(FiberPoint(material=steel, y=0.0, z=0.0, area=0.01),),
    )
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    integ = ops.beamIntegration.Lobatto(section=sec, n_ip=5)
    ops.element.forceBeamColumn(pg="Cols", transf=transf, integration=integ)
    ops.initial_stress(
        name="byid", elements=[1],
        sigma_xx=-10.0, sigma_yy=-10.0, sigma_zz=-10.0, ramp_steps=5,
    )

    out = tmp_path / "model.h5"
    ops.h5(str(out))
    with h5py.File(str(out), "r") as f:
        g = f["opensees/initial_stress/stress_000"]
        assert "elements" in g
        assert list(g["elements"][:]) == [1]
        assert "pg" not in g.attrs


# ---------------------------------------------------------------------------
# 2. Reader — accessor rehydrates the declarative records
# ---------------------------------------------------------------------------


def test_initial_stress_accessor_roundtrips(tmp_path: Path) -> None:
    out = tmp_path / "model.h5"
    _build_frame(with_initial_stress=True).h5(str(out))

    model = OpenSeesModel.from_h5(str(out))
    recs = model.initial_stress()
    assert len(recs) == 1
    r = recs[0]
    assert r.name == "insitu"
    assert r.pg == "Cols"
    assert r.elements is None
    assert (r.sigma_xx, r.sigma_yy, r.sigma_zz) == (-100.0, -200.0, -300.0)
    assert r.ramp_steps == 10
    assert r.lambda_install == 0.5


def test_vanilla_model_has_no_initial_stress(tmp_path: Path) -> None:
    out = tmp_path / "model.h5"
    _build_frame(with_initial_stress=False).h5(str(out))

    with h5py.File(str(out), "r") as f:
        assert "opensees/initial_stress" not in f
    assert OpenSeesModel.from_h5(str(out)).initial_stress() == ()


# ---------------------------------------------------------------------------
# 3. model_hash — folds in, and is round-trip-stable
# ---------------------------------------------------------------------------


def test_initial_stress_folds_into_model_hash(tmp_path: Path) -> None:
    """The same model with vs without initial stress must hash differently
    (the group is authored state, not a regenerable carve-out)."""
    with_p = tmp_path / "with.h5"
    without_p = tmp_path / "without.h5"
    _build_frame(with_initial_stress=True).h5(str(with_p))
    _build_frame(with_initial_stress=False).h5(str(without_p))

    h_with = OpenSeesModel.from_h5(str(with_p)).lineage.model_hash
    h_without = OpenSeesModel.from_h5(str(without_p)).lineage.model_hash
    assert h_with and h_without
    assert h_with != h_without


def test_h5_roundtrip_model_hash_stable(tmp_path: Path) -> None:
    """from_h5 → to_h5 reproduces the same ``model_hash`` (the declarative
    store-and-echo guarantee — nothing non-deterministic enters the bytes)."""
    p1 = tmp_path / "model.h5"
    _build_frame(with_initial_stress=True).h5(str(p1))

    m1 = OpenSeesModel.from_h5(str(p1))
    p2 = tmp_path / "model_rt.h5"
    m1.to_h5(str(p2))
    m2 = OpenSeesModel.from_h5(str(p2))

    assert m1.lineage.model_hash == m2.lineage.model_hash
    # And the records survive the re-write.
    assert len(m2.initial_stress()) == 1
    assert m2.initial_stress()[0].name == "insitu"


# ---------------------------------------------------------------------------
# 4. Replay wiring — the hook-wrap (adversarial fix)
# ---------------------------------------------------------------------------


def test_replay_registers_hook_before_analyze() -> None:
    """``_replay_into`` with initial_stress + an analyze_call must emit the
    ramp ``step_hook_ramp`` BEFORE ``analyze`` so the analyze takes the
    hook-wrapped (for-loop) form — not the bare ``analyze N``.  Without the
    initial-stress wiring the ramp procs declare but never fire."""
    from apeGmsh.opensees._internal.build import InitialStressRecord
    from apeGmsh.opensees._internal.compose import _replay_into
    from apeGmsh.opensees._internal.typed_records import ElementRecord

    fem = build_simple_frame_fem()
    rec = InitialStressRecord(
        name="insitu", pg="Cols", elements=None,
        sigma_xx=-100.0, sigma_yy=-100.0, sigma_zz=-100.0,
        ramp_steps=10, lambda_install=1.0,
    )
    ele = ElementRecord(
        type_token="forceBeamColumn", tag=1, args=(1, 2),
        connectivity=(1, 2), fem_eid=1,
    )

    emitter = TclEmitter()
    _replay_into(
        emitter,
        ndm=3, ndf=6,
        nodes=((1, (0.0, 0.0, 0.0)), (2, (0.0, 0.0, 1.0))),
        elements=(ele,),
        fem=fem,
        initial_stress=(rec,),
        analyze_call=(10, None),
    )
    text = "\n".join(emitter.lines())

    # The ramp installed a per-step proc + addToParameter fan-out.
    assert "addToParameter" in text
    assert "commitStressIncrementXX" in text
    # The trailing analyze is the hook-wrapped form, NOT a bare ``analyze 10``.
    assert "analyze 10\n" not in text + "\n"
    # An addToParameter line precedes the analyze block.
    assert text.index("addToParameter") < text.rindex("analyze")
