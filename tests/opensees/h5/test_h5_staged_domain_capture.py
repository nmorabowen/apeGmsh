"""ADR 0055 — staged builds keep their ``/opensees/`` zone through
``ops.domain_capture``.

The capture sidecar gate must mirror the ``ops.h5`` guard exactly:
only PARTITIONED staged builds (Phase 5 deferred) keep the
pre-Composed behaviour (``bridge=None`` — no sidecar, no
``/opensees/`` zone in the run file).  Non-partitioned staged builds
forward the bridge, so the Composed run file carries
``/opensees/stages`` plus the bridge envelope ndf and
``Results.from_native(...).model.stages()`` round-trips.
"""
from __future__ import annotations

from pathlib import Path

import h5py

from apeGmsh.opensees.apesees import apeSees
from apeGmsh.results import Results
from apeGmsh.results.capture.spec import DomainCaptureSpec

from tests.conftest import _open_model_from_h5
from tests.opensees.h5.test_h5_stages_writer import _build_two_stage_bridge
from tests.test_results_domain_capture import _FakeOps


def _probe_spec(ops: apeSees, **nodes_kwargs) -> DomainCaptureSpec:
    spec = DomainCaptureSpec(opensees=ops)
    spec.nodes(components=["displacement_x"], name="probe", **nodes_kwargs)
    return spec


# ---------------------------------------------------------------------------
# Gate — mirrors the ops.h5 guard (partitioned staged only)
# ---------------------------------------------------------------------------


def test_staged_nonpartitioned_forwards_bridge(tmp_path: Path) -> None:
    """Non-partitioned staged builds forward the bridge (gate lifted)."""
    ops = _build_two_stage_bridge()
    ops._fem.snapshot_id = "stub"  # spec resolve reads fem.snapshot_id
    cap = ops.domain_capture(
        _probe_spec(ops, ids=[1]), path=str(tmp_path / "run.h5"),
    )
    assert cap._bridge is ops


def test_staged_partitioned_keeps_bridge_none(tmp_path: Path) -> None:
    """PARTITIONED staged builds still gate the sidecar off — ops.h5
    raises NotImplementedError for them (ADR 0055 Phase 5 deferred), so
    forwarding the bridge would blow up ``with ops.domain_capture(...)``
    at __enter__."""
    ops = _build_two_stage_bridge()
    ops._fem.snapshot_id = "stub"
    ops._fem.set_partitions([
        (0, [1, 2, 3, 4], [1]),
        (1, [3, 4, 5, 6], [2]),
    ])
    cap = ops.domain_capture(
        _probe_spec(ops, ids=[1]), path=str(tmp_path / "run.h5"),
    )
    assert cap._bridge is None


# ---------------------------------------------------------------------------
# Round-trip — real session: capture run file carries /opensees/stages
# and the envelope ndf, readable through Results.from_native
# ---------------------------------------------------------------------------


def test_staged_capture_roundtrips_stages_and_ndf(g, tmp_path: Path) -> None:
    g.model.geometry.add_box(0, 0, 0, 1, 1, 1, label="box")
    g.physical.add_volume("box", name="Body")
    g.physical.add_surface(
        g.model.queries.boundary([(3, 1)]), name="Boundary",
    )
    g.mesh.sizing.set_global_size(2.0)
    g.mesh.generation.generate(dim=3)
    fem = g.mesh.queries.get_fem_data(dim=3)

    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3)
    ops.element.FourNodeTetrahedron(pg="Body", material=mat)
    ops.fix(pg="Boundary", dofs=(1, 1, 1))

    def chain() -> dict[str, object]:
        return {
            "test":        ops.test.NormDispIncr(tol=1e-4, max_iter=50),
            "algorithm":   ops.algorithm.Newton(),
            "integrator":  ops.integrator.LoadControl(dlam=0.1),
            "constraints": ops.constraints.Plain(),
            "numberer":    ops.numberer.RCM(),
            "system":      ops.system.UmfPack(),
            "analysis":    ops.analysis.Static(),
        }

    with ops.stage(name="construction") as s:
        s.analysis(**chain())
        s.run(n_increments=2)
    with ops.stage(name="loading") as s:
        s.analysis(**chain())
        s.run(n_increments=3, dt=0.01)

    out = tmp_path / "run.h5"
    cap = ops.domain_capture(
        _probe_spec(ops, pg="Boundary"), path=str(out), ops=_FakeOps(),
    )
    assert cap._bridge is ops
    with cap:
        cap.begin_stage("run", kind="static")
        cap.step(t=1.0)
        cap.end_stage()

    # Composed-file shape: the bridge zone (including the staged
    # bucket) rode the sidecar into the run file.
    with h5py.File(str(out), "r") as f:
        assert "opensees" in f
        assert "stages" in f["opensees"]

    # Read side: stages + envelope ndf round-trip through the broker.
    results = Results.from_native(out, model=_open_model_from_h5(out))
    model = results.model
    assert tuple(s.name for s in model.stages()) == (
        "construction", "loading",
    )
    assert int(model.ndf) == 3
