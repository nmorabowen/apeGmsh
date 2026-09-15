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


# ── Host auto-scalers (-k auto / -kAlpha / -host / -wcap) ────────────────

def test_rejects_auto_without_host() -> None:
    with pytest.raises(ValueError, match="k='auto' needs a representative"):
        CouplingControl(k="auto")


@pytest.mark.parametrize("bad", ["Auto", "AUTO", "", "automatic"])
def test_rejects_bad_k_string(bad: str) -> None:
    with pytest.raises(ValueError, match="positive number or 'auto'"):
        CouplingControl(k=bad, host=7)


def test_rejects_k_alpha_without_auto() -> None:
    with pytest.raises(ValueError, match="k_alpha only scales k='auto'"):
        CouplingControl(k=1e9, k_alpha=1e3)
    with pytest.raises(ValueError, match="k_alpha only scales k='auto'"):
        CouplingControl(k_alpha=1e3)


def test_rejects_wcap_without_host() -> None:
    with pytest.raises(ValueError, match="bipenalty_wcap needs the host"):
        CouplingControl(bipenalty_wcap=0.1)


def test_rejects_wcap_with_dtcr() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        CouplingControl(host=7, bipenalty_wcap=0.1, bipenalty_dtcr=1e-6)


def test_rejects_al_with_wcap() -> None:
    with pytest.raises(ValueError, match="enforce='al'.*bipenalty"):
        CouplingControl(enforce="al", host=7, bipenalty_wcap=0.1)


def test_rejects_dangling_host() -> None:
    with pytest.raises(ValueError, match="host has no consumer"):
        CouplingControl(k=1e9, host=7)
    with pytest.raises(ValueError, match="host has no consumer"):
        CouplingControl(host=7)


@pytest.mark.parametrize("bad", [0, -3, 2.5, True])
def test_rejects_bad_host(bad: object) -> None:
    with pytest.raises(ValueError, match="host must be a positive"):
        CouplingControl(k="auto", host=bad)


def test_host_control_is_not_default() -> None:
    assert not CouplingControl(k="auto", host=7).is_default
    assert not CouplingControl(host=7, bipenalty_wcap=0.1).is_default


def test_emit_flags_auto_host_order() -> None:
    c = CouplingControl(k="auto", k_alpha=5e2, host=7)
    assert c.emit_flags(host_ops_tag=1234) == [
        "-k", "auto", "-kAlpha", 5e2, "-host", 1234,
    ]


def test_emit_flags_wcap() -> None:
    c = CouplingControl(k=1e8, host=7, bipenalty_wcap=0.1)
    assert c.emit_flags(host_ops_tag=42) == [
        "-k", 1e8, "-host", 42, "-bipenalty", "-wcap", 0.1,
    ]


def test_emit_flags_hosted_requires_ops_tag() -> None:
    c = CouplingControl(k="auto", host=7)
    with pytest.raises(ValueError, match="host=7.*fem_eid_to_ops_tag"):
        c.emit_flags()


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


# ── Emit: hosted controls translate the FEM eid → ops tag ────────────────

def test_kinematic_emit_translates_host_eid() -> None:
    rec = NodeGroupRecord(
        kind="kinematic_coupling", master_node=1, slave_nodes=[2, 3], dofs=[],
        control=CouplingControl(k="auto", k_alpha=1e3, host=7),
    )
    e = RecordingEmitter()
    _emit_kinematic_couplings(
        e, [rec], TagAllocator(), fem_eid_to_ops_tag={7: 1234},
    )
    flat = [c for c in e.calls if c[0] == "element"][0][1]
    assert flat[2:] == (
        1, 2, 2, 3, "-k", "auto", "-kAlpha", 1e3, "-host", 1234,
    )


def test_distributing_emit_translates_host_eid_wcap() -> None:
    rec = InterpolationRecord(
        kind="distributing", slave_node=1, master_nodes=[2, 3, 4],
        weights=None,
        control=CouplingControl(host=9, bipenalty_wcap=0.1),
    )
    e = RecordingEmitter()
    _emit_one_interpolation(
        e, rec, TagAllocator(), fem_eid_to_ops_tag={9: 55},
    )
    flat = [c for c in e.calls if c[0] == "element"][0][1]
    assert flat[2:] == (
        1, 3, 2, 3, 4, "-host", 55, "-bipenalty", "-wcap", 0.1,
    )


def test_hosted_control_fails_loud_without_map() -> None:
    rec = NodeGroupRecord(
        kind="kinematic_coupling", master_node=1, slave_nodes=[2], dofs=[],
        control=CouplingControl(k="auto", host=7), name="lid",
    )
    e = RecordingEmitter()
    with pytest.raises(ValueError, match="'lid'.*host=7.*got none"):
        _emit_kinematic_couplings(e, [rec], TagAllocator())


def test_hosted_control_fails_loud_on_unknown_eid() -> None:
    rec = InterpolationRecord(
        kind="distributing", slave_node=1, master_nodes=[2, 3, 4],
        control=CouplingControl(k="auto", host=999),
    )
    e = RecordingEmitter()
    with pytest.raises(ValueError, match="host=999 is not an emitted"):
        _emit_one_interpolation(
            e, rec, TagAllocator(), fem_eid_to_ops_tag={7: 1234},
        )


def test_unhosted_control_ignores_missing_map() -> None:
    # A control without a host never touches the map — legacy direct
    # callers that pass no map keep working.
    rec = NodeGroupRecord(
        kind="kinematic_coupling", master_node=1, slave_nodes=[2], dofs=[],
        control=CouplingControl(k=1e10),
    )
    e = RecordingEmitter()
    _emit_kinematic_couplings(e, [rec], TagAllocator())
    flat = [c for c in e.calls if c[0] == "element"][0][1]
    assert flat[2:] == (1, 1, 2, "-k", 1e10)


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


def test_h5_roundtrip_host_autoscalers() -> None:
    rec = NodeGroupRecord(
        kind="kinematic_coupling", master_node=1, slave_nodes=[2, 3], dofs=[],
        control=CouplingControl(k="auto", k_alpha=5e2, host=7),
    )
    out = _roundtrip(rec)
    assert out.control == CouplingControl(k="auto", k_alpha=5e2, host=7)


def test_h5_roundtrip_wcap() -> None:
    rec = InterpolationRecord(
        kind="distributing", slave_node=1, master_nodes=[2, 3, 4],
        control=CouplingControl(k=1e8, host=9, bipenalty_wcap=0.1),
    )
    out = _roundtrip(rec)
    assert out.control == CouplingControl(k=1e8, host=9, bipenalty_wcap=0.1)


def test_h5_decode_pre_2_13_payload_lacks_autoscaler_columns() -> None:
    # A 2.12.0 file carries only the six v1 cpl_* columns — the reader
    # probes ``cpl_k_auto`` and decodes the v1 knobs with the
    # auto-scalers at their defaults.
    from apeGmsh.mesh import _femdata_h5_io as io

    old_dtype = np.dtype([
        ("master_node", np.int64),
        ("slave_nodes", np.dtype("O")),
        ("dofs", np.dtype("O")),
        ("offsets", np.dtype("O")),
        ("plane_normal", np.float64, (3,)),
        ("name", np.dtype("O")),
        ("cpl_has", np.uint8),
        ("cpl_k", np.float64),
        ("cpl_kr", np.float64),
        ("cpl_enforce", np.uint8),
        ("cpl_dtcr", np.float64),
        ("cpl_absolute", np.uint8),
    ])
    nan = float("nan")
    payload = np.array([(
        1, np.array([2, 3], dtype=np.int64), np.array([], dtype=np.int64),
        np.array([], dtype=np.float64), (nan, nan, nan), "",
        np.uint8(1), 1e10, nan, np.uint8(0), nan, np.uint8(0),
    )], dtype=old_dtype)[0]
    out = io._decode_node_group(
        {"payload": payload, "payload_kind": "kinematic_coupling"},
        NodeGroupRecord,
    )
    assert out.control == CouplingControl(k=1e10)


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


# ── -alUpdate (fork PR #839 / F9) ────────────────────────────────────────

def test_al_update_none_is_byte_identical() -> None:
    # The whole point of the default: a deck that never sets the knob is
    # byte-identical to one built before the token existed.
    assert CouplingControl(enforce="al").emit_flags() == ["-enforce", "al"]
    assert CouplingControl().al_update is None


@pytest.mark.parametrize("mode", ["commit", "iter"])
def test_al_update_emits_right_after_enforce(mode: str) -> None:
    c = CouplingControl(k=1e6, enforce="al", al_update=mode)
    assert c.emit_flags() == ["-k", 1e6, "-enforce", "al", "-alUpdate", mode]
    assert not c.is_default


@pytest.mark.parametrize("bad", ["Commit", "ITER", "", "always", "step"])
def test_rejects_bad_al_update(bad: str) -> None:
    with pytest.raises(ValueError, match="al_update must be one of"):
        CouplingControl(enforce="al", al_update=bad)


def test_h5_decode_refuses_an_out_of_range_al_update_code() -> None:
    # A corrupt / future ``cpl_al_update`` code must name the field and the
    # value, not surface as a bare IndexError from the code table.
    from apeGmsh.mesh import _femdata_h5_io as io
    from apeGmsh.opensees.emitter.h5_reader import MalformedH5Error

    with pytest.raises(MalformedH5Error, match="cpl_al_update carries 7"):
        io._control_from_values(
            1, 1e10, float("nan"), 0, float("nan"), 0, al_update=7,
        )


@pytest.mark.parametrize("mode", ["commit", "iter"])
def test_al_update_requires_enforce_al(mode: str) -> None:
    # The fork warns-and-ignores; apeGmsh refuses rather than emit a flag
    # the engine will silently drop.
    with pytest.raises(ValueError, match="no effect without enforce='al'"):
        CouplingControl(al_update=mode)


def test_embedded_control_refuses_al_update() -> None:
    from apeGmsh._kernel._coupling_control import EmbeddedNodeControl

    with pytest.raises(ValueError, match="LadrunoKinematicCoupling-only"):
        EmbeddedNodeControl(enforce="al", al_update="commit")


def test_kinematic_emit_carries_al_update() -> None:
    rec = NodeGroupRecord(
        kind="kinematic_coupling", master_node=1, slave_nodes=[2, 3], dofs=[],
        control=CouplingControl(k=1e6, enforce="al", al_update="iter"),
    )
    e = RecordingEmitter()
    _emit_kinematic_couplings(e, [rec], TagAllocator())
    flat = [c for c in e.calls if c[0] == "element"][0][1]
    assert flat[0] == "LadrunoKinematicCoupling"
    assert flat[2:] == (
        1, 2, 2, 3, "-k", 1e6, "-enforce", "al", "-alUpdate", "iter",
    )
    assert flat.count("-alUpdate") == 1


def test_distributing_emit_refuses_al_update() -> None:
    # -alUpdate is LadrunoKinematicCoupling-only; the RBE3 parser rejects it.
    rec = InterpolationRecord(
        kind="distributing", slave_node=1, master_nodes=[2, 3, 4],
        control=CouplingControl(enforce="al", al_update="commit"),
        name="ring",
    )
    e = RecordingEmitter()
    with pytest.raises(
        ValueError, match="'ring'.*LadrunoKinematicCoupling-only",
    ):
        _emit_one_interpolation(e, rec, TagAllocator())


def test_penalty_al_embedded_emit_refuses_al_update() -> None:
    # The LadrunoEmbeddedNode path shares CouplingControl too.
    rec = InterpolationRecord(
        kind="tie", slave_node=1, master_nodes=[2, 3, 4],
        weights=np.array([0.3, 0.3, 0.4]), enforce="penalty_al",
        control=CouplingControl(enforce="al", al_update="iter"),
    )
    e = RecordingEmitter()
    with pytest.raises(ValueError, match="LadrunoKinematicCoupling-only"):
        _emit_one_interpolation(e, rec, TagAllocator())


def test_public_knob_reaches_the_emitted_token() -> None:
    # The public g.constraints.kinematic_coupling signature carries the
    # knob, and a record built from it emits -alUpdate exactly once.
    import inspect

    from apeGmsh.core.ConstraintsComposite import ConstraintsComposite

    params = inspect.signature(
        ConstraintsComposite.kinematic_coupling).parameters
    assert "al_update" in params
    assert params["al_update"].default is None
    rec = NodeGroupRecord(
        kind="kinematic_coupling", master_node=1, slave_nodes=[2], dofs=[],
        control=CouplingControl(enforce="al", al_update="commit"),
    )
    e = RecordingEmitter()
    _emit_kinematic_couplings(e, [rec], TagAllocator())
    flat = [c for c in e.calls if c[0] == "element"][0][1]
    assert flat[-2:] == ("-alUpdate", "commit")


@pytest.mark.parametrize("mode", ["commit", "iter"])
def test_h5_roundtrip_al_update(mode: str) -> None:
    rec = NodeGroupRecord(
        kind="kinematic_coupling", master_node=1, slave_nodes=[2, 3], dofs=[],
        control=CouplingControl(k=1e6, enforce="al", al_update=mode),
    )
    out = _roundtrip(rec)
    assert out.control == CouplingControl(k=1e6, enforce="al", al_update=mode)


def test_h5_roundtrip_al_update_none_stays_none() -> None:
    rec = NodeGroupRecord(
        kind="kinematic_coupling", master_node=1, slave_nodes=[2], dofs=[],
        control=CouplingControl(k=1e6, enforce="al"),
    )
    assert _roundtrip(rec).control.al_update is None
