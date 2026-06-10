"""Staged-H5 guard contract after ADR 0055 Phase 2 (schema 2.18.0).

The historical contract (``apeSees.h5`` fails loud on ANY staged build)
was lifted for NON-PARTITIONED staged builds: the H5 emitter now
captures the per-stage emit stream into ``/opensees/stages`` (see
``test_h5_stages_writer.py`` for the group-shape coverage).  What
remains fail-loud, and what this module pins:

1. PARTITIONED staged build → ``NotImplementedError`` at ``ops.h5``
   (ADR 0055 Phase 5 — the per-rank emit shape has no capture/replay
   path; ``_emit_stages_partitioned`` itself never raises, so the
   ``h5()`` guard is the ONLY fail-loud boundary).
2. Reading a staged archive back through ``OpenSeesModel.from_h5`` →
   ``NotImplementedError`` (the staged read side has not landed; the
   flat path would silently FLATTEN the staged program — the exact
   hazard the old write-side guard existed to prevent).
3. Non-partitioned staged build → writes successfully.
4. Vanilla build → writes successfully, with NO ``/opensees/stages``
   key (the stages writer early-returns; vanilla files stay
   byte-identical to 2.17.x).

ADR 0055 Phase 1 (schema 2.16.0) lifted the *global*
``ops.initial_stress`` guard; see ``test_h5_initial_stress.py``.
"""
from __future__ import annotations

from pathlib import Path

import h5py
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
    """Single-quad FEM stub with a ``"Rock"`` element PG."""
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


def _staged_quad_bridge(fem: FEMStub) -> apeSees:
    """One-quad bridge with a single ``insitu`` stage (no activation)."""
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.FourNodeQuad(pg="Rock", thickness=1.0, material=mat)
    with ops.stage(name="insitu") as s:
        s.analysis(**_full_chain(ops))
        s.run(n_increments=10, dt=0.1)
    return ops


# ---------------------------------------------------------------------------
# 1. PARTITIONED staged build → NotImplementedError (Phase 5 stays loud)
# ---------------------------------------------------------------------------


def test_h5_partitioned_staged_build_raises(tmp_path: Path) -> None:
    """The lifted guard must positively test ``is_partitioned`` — a
    partitioned staged deck has no H5 capture/replay path (Phase 5),
    and the partitioned emit path itself never raises, so this guard
    is the only fail-loud boundary."""
    fem = _make_quad_fem_stub()
    fem.set_partitions([
        (0, [1, 2], [1]),
        (1, [3, 4], []),
    ])
    ops = _staged_quad_bridge(fem)

    out = tmp_path / "partitioned_staged.h5"
    with pytest.raises(NotImplementedError) as excinfo:
        ops.h5(str(out))

    msg = str(excinfo.value)
    assert "PARTITIONED staged" in msg
    assert "Phase 5" in msg
    assert "ops.tcl" in msg
    assert "ops.py" in msg
    # Guard fires before any disk write.
    assert not out.exists()


# ---------------------------------------------------------------------------
# 2. Staged archive → from_h5 fails loud (reader slice not landed)
# ---------------------------------------------------------------------------


def test_h5_staged_archive_reads_but_flat_replay_fails_loud(
    tmp_path: Path,
) -> None:
    """ADR 0055 P2.2: a staged archive LOADS (``.stages()`` exposes the
    program; the old read probe is lifted), but the flat tcl/py/live
    replay still fails loud — silently flattening the staged program
    is the hazard every guard generation has protected against.
    The full read contract lives in ``test_h5_stages_reader.py``.

    Uses a real :class:`FEMData` (not the stub) so the neutral zone
    is valid for ``FEMData.from_h5``.
    """
    from apeGmsh.opensees import OpenSeesModel
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
    ops.element.forceBeamColumn(pg="Cols", transf=transf, integration=integ)
    with ops.stage(name="insitu") as s:
        s.analysis(**_full_chain(ops))
        s.run(n_increments=10, dt=0.1)

    out = tmp_path / "staged.h5"
    ops.h5(str(out))
    assert out.exists()

    m = OpenSeesModel.from_h5(str(out))
    (stage,) = m.stages()
    assert stage.name == "insitu"
    assert stage.analyze_steps == 10

    with pytest.raises(NotImplementedError, match="P2.3"):
        m.build("tcl")


# ---------------------------------------------------------------------------
# 3. Non-partitioned staged build → writes successfully
# ---------------------------------------------------------------------------


def test_h5_non_partitioned_staged_build_writes(tmp_path: Path) -> None:
    """ADR 0055 Phase 2: the staged guard is lifted for non-partitioned
    builds; the stage persists under ``/opensees/stages`` and the
    stage's chain does NOT leak into a global ``/opensees/analysis``."""
    ops = _staged_quad_bridge(_make_quad_fem_stub())

    out = tmp_path / "staged.h5"
    ops.h5(str(out))
    assert out.exists()
    with h5py.File(str(out), "r") as f:
        stages = f["opensees"]["stages"]
        assert int(stages.attrs["n_stages"]) == 1
        g = stages["stage_000"]
        assert g.attrs["name"] == "insitu"
        assert int(g.attrs["analyze_steps"]) == 10
        assert float(g.attrs["analyze_dt"]) == pytest.approx(0.1)
        # The stage's chain is scoped to the stage group; a staged
        # file must carry NO global analysis group (phantom-leak
        # regression, ADR 0055 adversarial finding).
        assert "analysis" in g
        assert "analysis" not in f["opensees"]


# ---------------------------------------------------------------------------
# 4. Vanilla non-staged smoke — must still write, with no stages key
# ---------------------------------------------------------------------------


def test_h5_non_staged_smoke_still_writes(tmp_path: Path) -> None:
    """A vanilla build (no stages, no initial_stress) must produce a
    valid H5 file with NO ``/opensees/stages`` group (the stages
    writer early-returns; vanilla bytes are unchanged).

    Uses :func:`build_simple_frame_fem` so the broker neutral zone
    actually writes (the guard tests above use FEMStub; the smoke
    test needs a real :class:`FEMData` to exercise the full compose
    path).
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
    with h5py.File(str(out), "r") as f:
        assert "stages" not in f["opensees"]
