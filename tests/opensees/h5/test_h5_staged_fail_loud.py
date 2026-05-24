"""Regression: ``apeSees.h5`` fails loud on staged + initial_stress builds.

Phase SSI-1 (initial_stress / parameter / addToParameter / step_hook_ramp)
and Phase SSI-2.A / 2.B (stage_open / stage_close / domain_change) added
new methods to the :class:`~apeGmsh.opensees.emitter.base.Emitter`
Protocol.  The Tcl, Py, Live, and Recording emitters implement them
fully; the H5 emitter ships them as no-ops (archival deferred).  Without
a guard at the bridge layer, a user who calls ``ops.h5(path)`` on a
staged model — or any model declaring ``ops.initial_stress(...)`` —
would get a file that round-trips through :meth:`OpenSeesModel.from_h5`
to a non-staged flat model, silently dropping every stage's analysis
chain rebinding, activated topology, per-stage initial-stress records,
and analyze loop.

This module pins the fail-loud contract added by the apeSees bridge:

1. Staged build  → ``NotImplementedError`` with a pointer to ``ops.tcl`` /
   ``ops.py`` (which support the full surface).
2. Global ``ops.initial_stress(...)`` build  → ditto.
3. Non-staged / non-initial-stress build  → still writes successfully
   (the guard is precise; it does not regress the vanilla path).

When the H5 schema bump (SCHEMA_VERSION 2.11.0 → 2.12.0) and the matching
``_compose_model_h5`` / :class:`OpenSeesModel` accessors land, drop the
two guards in :meth:`apeSees.h5` and replace this module with a
round-trip preservation test.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from apeGmsh.opensees.apesees import apeSees

from tests.opensees.fixtures.fem_stub import (
    FEMStub,
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
)
from tests.opensees.h5._opensees_model_fixtures import build_simple_frame_fem


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_quad_fem_stub() -> FEMStub:
    """Single-quad FEM stub with a ``"Rock"`` element PG.

    The bridge guard fires before any FEM access (no ``snapshot_id``
    read, no ``build()`` call), so the stub is sufficient for both
    fail-loud paths.
    """
    return FEMStub(
        nodes=_NodesStub(
            ids=[1, 2, 3, 4],
            coords=[
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
            ],
            node_pgs={"Rock": [1, 2, 3, 4]},
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
    """Build a complete analysis chain (for ``stage.analysis(**chain)``)."""
    return {
        "test":        ops.test.NormDispIncr(tol=1e-4, max_iter=50),
        "algorithm":   ops.algorithm.Newton(),
        "integrator":  ops.integrator.LoadControl(dlam=0.1),
        "constraints": ops.constraints.Plain(),
        "numberer":    ops.numberer.RCM(),
        "system":      ops.system.UmfPack(),
        "analysis":    ops.analysis.Static(),
    }


# ---------------------------------------------------------------------------
# 1. Staged build → NotImplementedError
# ---------------------------------------------------------------------------


def test_h5_staged_build_raises_notimplementederror(tmp_path: Path) -> None:
    """A bridge carrying ``_stage_records`` must fail loud on ``h5(...)``.

    The H5 emitter no-ops ``stage_open`` / ``stage_close`` /
    ``domain_change`` (and the per-stage ``addToParameter`` /
    ``step_hook_ramp`` calls inside the stage block).  Writing the file
    would produce a deck that round-trips to a non-staged flat model.
    """
    fem = _make_quad_fem_stub()
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.FourNodeQuad(pg="Rock", thickness=1.0, material=mat)
    with ops.stage(name="insitu") as s:
        s.analysis(**_full_chain(ops))
        s.run(n_increments=10, dt=0.1)

    out = tmp_path / "staged.h5"
    with pytest.raises(NotImplementedError) as excinfo:
        ops.h5(str(out))

    msg = str(excinfo.value)
    assert "staged builds is not yet supported" in msg
    assert "Phase SSI-2" in msg
    assert "ops.tcl" in msg
    assert "ops.py" in msg
    # File must not exist — the guard fires before _compose_model_h5
    # touches disk, so a half-written file is impossible.
    assert not out.exists()


# ---------------------------------------------------------------------------
# 2. Global ops.initial_stress → NotImplementedError
# ---------------------------------------------------------------------------


def test_h5_initial_stress_build_raises_notimplementederror(
    tmp_path: Path,
) -> None:
    """A bridge carrying global ``_initial_stress_records`` (no stage)
    must fail loud on ``h5(...)``.

    The H5 emitter no-ops ``addToParameter`` and ``step_hook_ramp``,
    so writing the file would drop every parameter declaration, the
    per-step ramp proc, and every addToParameter call.  The resulting
    deck would round-trip to a model with NO initial-stress ramp.
    """
    fem = _make_quad_fem_stub()
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.FourNodeQuad(pg="Rock", thickness=1.0, material=mat)
    ops.initial_stress(
        name="rock_in", pg="Rock",
        sigma_xx=-100.0, sigma_yy=-100.0, sigma_zz=-100.0,
        ramp_steps=10,
    )

    out = tmp_path / "initial_stress.h5"
    with pytest.raises(NotImplementedError) as excinfo:
        ops.h5(str(out))

    msg = str(excinfo.value)
    assert "ops.initial_stress" in msg
    assert "Phase SSI-1" in msg
    assert "ops.tcl" in msg
    assert "ops.py" in msg
    assert not out.exists()


# ---------------------------------------------------------------------------
# 3. Vanilla non-staged smoke — must still write
# ---------------------------------------------------------------------------


def test_h5_non_staged_smoke_still_writes(tmp_path: Path) -> None:
    """A vanilla build (no stages, no initial_stress) is unaffected by
    the guard and must still produce a valid H5 file.

    Uses :func:`build_simple_frame_fem` so the broker neutral zone
    actually writes (the fail-loud tests above use FEMStub because
    the guard fires before any FEM access; the smoke test needs a
    real :class:`FEMData` to exercise the full compose path).
    """
    from apeGmsh.opensees.section.fiber import FiberPoint

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
    ops.element.forceBeamColumn(
        pg="Cols", transf=transf, integration=integ,
    )

    out = tmp_path / "vanilla.h5"
    ops.h5(str(out))
    assert out.exists()
    assert out.stat().st_size > 0
