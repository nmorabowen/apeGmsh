"""ADR 0055 — staged builds keep their ``/opensees/`` zone through
``ops.domain_capture``.

Every build forwards the bridge (ADR 0055 Phase 5 / P5.3 retired the
last ``bridge=None`` gate — partitioned staged builds archive since
P5.1): the Composed run file carries ``/opensees/stages`` (plus
``/opensees/partitions`` for partitioned builds) and the bridge
envelope ndf, so ``Results.from_native(...).model.stages()``
round-trips.  The one remaining staged raise site (stage-claimed
phantom nodes, emitter gate-2) degrades sidecar-less at __enter__
with a warning.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh.opensees.apesees import apeSees
from apeGmsh.results import Results
from apeGmsh.results.capture.spec import DomainCaptureSpec

from tests.conftest import _open_model_from_h5
from tests.opensees.h5.test_h5_stages_writer import (
    _build_two_stage_bridge,
    _chain,
    _make_two_quad_fem_stub,
)
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


def test_staged_phantom_constraint_capture_degrades_sidecar_less(
    tmp_path: Path,
) -> None:
    """A stage-claimed ``node_to_surface`` emits phantom nodes, which
    ``ops.h5()`` still rejects (emitter gate-2 guard) even on a
    non-partitioned model.  The capture must NOT crash at __enter__ —
    it warns and degrades to the pre-Composed shape (no sidecar, no
    ``/opensees/`` zone), exactly what the old blanket ``bridge=None``
    gate produced for every staged build."""
    from apeGmsh._kernel.records._constraints import (
        NodePairRecord,
        NodeToSurfaceRecord,
    )
    from apeGmsh._kernel.records._kinds import ConstraintKind

    fem = _make_two_quad_fem_stub()
    fem.add_node_constraints([NodeToSurfaceRecord(
        kind=ConstraintKind.NODE_TO_SURFACE, name="hub",
        master_node=1, slave_nodes=[5, 6],
        phantom_nodes=[200, 201],
        phantom_coords=np.array([[1.0, 2.0, 0.0], [0.0, 2.0, 0.0]]),
        rigid_link_records=[
            NodePairRecord(
                kind=ConstraintKind.RIGID_BEAM,
                master_node=1, slave_node=200,
            ),
            NodePairRecord(
                kind=ConstraintKind.RIGID_BEAM,
                master_node=1, slave_node=201,
            ),
        ],
        equal_dof_records=[
            NodePairRecord(
                kind=ConstraintKind.EQUAL_DOF,
                master_node=200, slave_node=5, dofs=[1, 2],
            ),
            NodePairRecord(
                kind=ConstraintKind.EQUAL_DOF,
                master_node=201, slave_node=6, dofs=[1, 2],
            ),
        ],
        dofs=[1, 2],
    )])
    fem.snapshot_id = "stub"

    # FEMStub has no broker write surface; graft the same minimal
    # ``to_native_h5`` shim _MockFem uses so NativeWriter can embed a
    # ``/model/`` zone at capture __enter__ (nodes only — elements
    # aren't read back by this test).
    def _to_native_h5(group):
        from types import SimpleNamespace

        from apeGmsh.mesh._femdata_h5_io import (
            write_neutral_zone_into_group,
        )
        mini = SimpleNamespace(
            nodes=SimpleNamespace(
                ids=np.asarray(fem.nodes.ids),
                coords=np.asarray(fem.nodes.coords),
            ),
            elements=[],
            snapshot_id="stub",
        )
        write_neutral_zone_into_group(mini, group, ndf=2)

    fem.to_native_h5 = _to_native_h5
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.FourNodeQuad(pg="Rock", thickness=1.0, material=mat)
    ops.element.FourNodeQuad(pg="Fill", thickness=1.0, material=mat)
    ops.fix(pg="Base", dofs=(1, 1))
    with ops.stage(name="construction") as s:
        s.node_to_surface(name="hub")
        s.analysis(**_chain(ops))
        s.run(n_increments=5)

    out = tmp_path / "run.h5"
    cap = ops.domain_capture(
        _probe_spec(ops, ids=[1]), path=str(out), ops=_FakeOps(),
    )
    assert cap._bridge is ops  # gate forwards; degrade happens at enter
    with pytest.warns(UserWarning, match="could not archive"):
        with cap:
            cap.begin_stage("run", kind="static")
            cap.step(t=1.0)
            cap.end_stage()

    assert not out.with_suffix(".model.h5").exists()
    with h5py.File(str(out), "r") as f:
        assert "opensees" not in f  # pre-Composed shape
        assert "model" in f and "stages" in f


def test_staged_partitioned_forwards_bridge(tmp_path: Path) -> None:
    """PARTITIONED staged builds forward the bridge too (ADR 0055
    Phase 5 / P5.3) — ``ops.h5`` archives them since P5.1, so the
    sidecar write at __enter__ succeeds and the Composed run file
    carries the bridge zone.  The old ``bridge=None`` degrade is
    retired."""
    ops = _build_two_stage_bridge()
    ops._fem.snapshot_id = "stub"
    ops._fem.set_partitions([
        (0, [1, 2, 3, 4], [1]),
        (1, [3, 4, 5, 6], [2]),
    ])
    cap = ops.domain_capture(
        _probe_spec(ops, ids=[1]), path=str(tmp_path / "run.h5"),
    )
    assert cap._bridge is ops


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


# ---------------------------------------------------------------------------
# Round-trip — PARTITIONED staged capture (ADR 0055 Phase 5 / P5.3):
# the run file carries /opensees/stages AND /opensees/partitions —
# the stage-aware viewer's feedstock for partitioned SSI runs.
# ---------------------------------------------------------------------------


def test_partitioned_staged_capture_run_file_carries_stages_and_partitions(
    tmp_path: Path,
) -> None:
    from apeGmsh.opensees import OpenSeesModel

    from tests.opensees.h5.test_h5_partitioned_staged_capture import (
        _partitioned_bridge,
    )

    ops = _partitioned_bridge()
    out = tmp_path / "run.h5"
    cap = ops.domain_capture(
        _probe_spec(ops, pg="Base"), path=str(out), ops=_FakeOps(),
    )
    assert cap._bridge is ops
    with cap:
        cap.begin_stage("run", kind="static")
        cap.step(t=1.0)
        cap.end_stage()

    # Composed-file shape: staged bucket AND partition zone rode the
    # sidecar into the run file.
    with h5py.File(str(out), "r") as f:
        assert "opensees" in f
        assert "stages" in f["opensees"]
        assert "partitions" in f["opensees"]
        assert int(
            f["opensees"]["partitions"].attrs["n_partitions"]
        ) == 2

    # Read side: the staged program loads from the run file.
    model = OpenSeesModel.from_h5(str(out), fem_root="/model")
    assert tuple(s.name for s in model.stages()) == (
        "construction", "loading",
    )
    assert len(model.partitions()) == 2
    assert int(model.ndf) == 2
