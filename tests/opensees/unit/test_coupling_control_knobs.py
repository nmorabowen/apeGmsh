"""Explicit control knobs for the fork coupling elements (RBE2 / RBE3).

`CouplingControl` carries the `-k` / `-kr` / `-enforce` / `-bipenalty -dtcr`
/ `-absolute` knobs from `g.constraints.kinematic_coupling` /
`distributing_coupling` through to the emitted element line, and round-trips
through `model.h5`. These tests lock:

* validation (enforce domain, positive penalties, the al+bipenalty refusal),
* `emit_flags` ordering / default-eliding,
* the knobs reaching the emitted `LadrunoKinematicCoupling` /
  `LadrunoDistributingCoupling` line,
* H5 round-trip of the control (and `None` for non-coupling records).
"""
from __future__ import annotations

import numpy as np
import pytest

from apeGmsh._kernel._coupling_control import CouplingControl
from apeGmsh._kernel.records._constraints import (
    InterpolationRecord,
    NodeGroupRecord,
)
from apeGmsh.opensees._internal.build import (
    _emit_kinematic_couplings,
    _emit_one_interpolation,
)
from apeGmsh.opensees._internal.tag_allocator import TagAllocator
from apeGmsh.opensees.emitter.recording import RecordingEmitter


# ── CouplingControl: validation ──────────────────────────────────────────

def test_default_control_is_default_and_emits_nothing() -> None:
    c = CouplingControl()
    assert c.is_default
    assert c.emit_flags() == []


def test_emit_flags_order_and_eliding() -> None:
    c = CouplingControl(k=1e9, kr=2e9, enforce="al")
    # al is implicit → cannot pair with bipenalty; tested separately.
    assert c.emit_flags() == ["-k", 1e9, "-kr", 2e9, "-enforce", "al"]
    assert not c.is_default


def test_emit_flags_bipenalty_and_absolute() -> None:
    c = CouplingControl(k=1e8, bipenalty_dtcr=2e-6, absolute=True)
    assert c.emit_flags() == [
        "-k", 1e8, "-bipenalty", "-dtcr", 2e-6, "-absolute",
    ]


@pytest.mark.parametrize("bad", ["Penalty", "lagrange", "", "AL"])
def test_rejects_bad_enforce(bad: str) -> None:
    with pytest.raises(ValueError, match="enforce must be one of"):
        CouplingControl(enforce=bad)


@pytest.mark.parametrize("field", ["k", "kr", "bipenalty_dtcr"])
@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_rejects_non_positive_penalty(field: str, bad: float) -> None:
    with pytest.raises(ValueError, match=f"{field} must be > 0"):
        CouplingControl(**{field: bad})


def test_rejects_al_with_bipenalty() -> None:
    with pytest.raises(ValueError, match="enforce='al'.*bipenalty"):
        CouplingControl(enforce="al", bipenalty_dtcr=1e-6)


# ── Emit: knobs reach the element line ───────────────────────────────────

def test_kinematic_emit_appends_control_flags() -> None:
    rec = NodeGroupRecord(
        kind="kinematic_coupling", master_node=1, slave_nodes=[2, 3], dofs=[1, 2],
        control=CouplingControl(k=1e10, enforce="al"),
    )
    e = RecordingEmitter()
    _emit_kinematic_couplings(e, [rec], TagAllocator())
    flat = [c for c in e.calls if c[0] == "element"][0][1]
    assert flat[0] == "LadrunoKinematicCoupling"
    # ref=1, N=2, slaves 2 3, -dof 1 2, then the control flags.
    assert flat[2:] == (1, 2, 2, 3, "-dof", 1, 2, "-k", 1e10, "-enforce", "al")


def test_kinematic_emit_no_control_no_extra_flags() -> None:
    rec = NodeGroupRecord(
        kind="kinematic_coupling", master_node=1, slave_nodes=[2], dofs=[],
        control=None,
    )
    e = RecordingEmitter()
    _emit_kinematic_couplings(e, [rec], TagAllocator())
    flat = [c for c in e.calls if c[0] == "element"][0][1]
    assert flat[2:] == (1, 1, 2)          # no -dof, no control flags


def test_distributing_emit_appends_control_flags() -> None:
    rec = InterpolationRecord(
        kind="distributing", slave_node=1, master_nodes=[2, 3, 4],
        weights=None, control=CouplingControl(k=5e8, bipenalty_dtcr=1e-6),
    )
    e = RecordingEmitter()
    _emit_one_interpolation(e, rec, TagAllocator())
    flat = [c for c in e.calls if c[0] == "element"][0][1]
    assert flat[0] == "LadrunoDistributingCoupling"
    # ref=1, N=3, indep 2 3 4, (no -w), then control flags.
    assert flat[2:] == (
        1, 3, 2, 3, 4, "-k", 5e8, "-bipenalty", "-dtcr", 1e-6,
    )


# ── H5 round-trip ────────────────────────────────────────────────────────

def _roundtrip(rec):
    """Encode → decode one constraint record through the H5 payload layer."""
    from apeGmsh.mesh import _femdata_h5_io as io
    from apeGmsh.mesh._record_h5 import (
        interpolation_payload_dtype,
        node_group_payload_dtype,
    )
    if isinstance(rec, NodeGroupRecord):
        dt, enc, dec = (
            node_group_payload_dtype(), io._encode_node_group,
            io._decode_node_group,
        )
    else:
        dt, enc, dec = (
            interpolation_payload_dtype(), io._encode_interpolation,
            io._decode_interpolation,
        )
    payload = np.array([enc(rec)], dtype=dt)[0]
    row = {"payload": payload, "payload_kind": rec.kind}
    return dec(row, type(rec))


def test_h5_roundtrip_kinematic_control() -> None:
    rec = NodeGroupRecord(
        kind="kinematic_coupling", master_node=1, slave_nodes=[2, 3], dofs=[1, 2],
        control=CouplingControl(k=1e10, kr=3e10, enforce="al", absolute=True),
    )
    out = _roundtrip(rec)
    assert out.control == CouplingControl(
        k=1e10, kr=3e10, enforce="al", absolute=True,
    )


def test_h5_roundtrip_distributing_control_with_bipenalty() -> None:
    rec = InterpolationRecord(
        kind="distributing", slave_node=1, master_nodes=[2, 3, 4],
        control=CouplingControl(k=5e8, bipenalty_dtcr=2e-6),
    )
    out = _roundtrip(rec)
    assert out.control == CouplingControl(k=5e8, bipenalty_dtcr=2e-6)


def test_h5_roundtrip_none_control_stays_none() -> None:
    # A rigid_diaphragm (NodeGroupRecord) and a tie (InterpolationRecord)
    # carry no control → must decode back to None, not a default object.
    ng = NodeGroupRecord(
        kind="rigid_diaphragm", master_node=1, slave_nodes=[2, 3],
        dofs=[1, 2, 3], control=None,
    )
    ir = InterpolationRecord(
        kind="tie", slave_node=1, master_nodes=[2, 3, 4],
        weights=np.array([0.3, 0.3, 0.4]), control=None,
    )
    assert _roundtrip(ng).control is None
    assert _roundtrip(ir).control is None
