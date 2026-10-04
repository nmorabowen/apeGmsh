"""Conformance: ``Emitter.command()`` on the five emitters (ADR 0114 D2/D3).

The channel is fail-closed. A token is allowed only by a ``via="command"``
row in ``VERBS``, and none ships yet, so the round trip below injects one
row through ``base.VERBS`` (the mapping ``command_row`` reads). It also
pins the K1-2 ``H5Emitter`` helpers: ``_refuse`` raises
:class:`H5RefusedVerb` naming the verb and its row, ``_ledger`` counts a
dropped call silently, and a stage-only verb outside a stage bracket
raises instead of returning (ADR 0114 D2; a silent no-op before K1-2, so
this file fails on the pre-K1-2 emitters).
"""
from __future__ import annotations

from typing import Any

import pytest

from apeGmsh.opensees.emitter import base as base_mod
from apeGmsh.opensees.emitter.h5 import H5Emitter, H5RefusedVerb
from apeGmsh.opensees.emitter.live import LiveOpsEmitter
from apeGmsh.opensees.emitter.py import PyEmitter
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.emitter.tcl import TclEmitter
from apeGmsh.opensees.emitter.verbs import VERBS, Verb

ARGS = (1, 2.5, "x")


class _Ops:
    """A stand-in openseespy module with one command bound."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def probeCmd(self, *args: Any) -> None:
        self.calls.append(args)


def _live(ops: object) -> LiveOpsEmitter:
    """A ``LiveOpsEmitter`` bound to ``ops`` without resolving openseespy."""
    le = LiveOpsEmitter.__new__(LiveOpsEmitter)
    le._ops = ops  # type: ignore[assignment]
    le._in_partition = False
    return le


def _inject_row(monkeypatch: pytest.MonkeyPatch, **overrides: Any) -> Verb:
    fields: dict[str, Any] = dict(
        verb="probeCmd", via="command", family="model", scope="global",
        h5="archive", store="/opensees/commands", requires=frozenset(),
        returns="None", seq_what="", seq_op="", decl=False,
    )
    fields.update(overrides)
    row = Verb(**fields)
    monkeypatch.setattr(base_mod, "VERBS", {**VERBS, row.verb: row})
    return row


# -- the fail-closed allow-list ---------------------------------------------

def test_no_command_row_ships_in_k1_2() -> None:
    assert not [k for k, v in VERBS.items() if v.via == "command"]
    assert VERBS["command"].via == "protocol"
    assert VERBS["command"].h5 == "refuse"


@pytest.mark.parametrize("verb", ["noSuchVerb", "fix", "command"],
                         ids=["no-row", "protocol-row", "the-channel-itself"])
def test_unknown_or_protocol_verb_raises_on_every_emitter(verb: str) -> None:
    tcl, py, rec, h5 = TclEmitter(), PyEmitter(), RecordingEmitter(), H5Emitter()
    ops = _Ops()
    live = _live(ops)
    before = (tcl.lines(), py.lines())  # the deck preambles
    for emitter in (tcl, py, rec, h5, live):
        with pytest.raises(ValueError, match="ADR 0114 D3"):
            emitter.command(verb, *ARGS)
    assert (tcl.lines(), py.lines()) == before
    assert rec.calls == [] and ops.calls == []


# -- the round trip through an injected via='command' row ---------------------

def test_tcl_py_recording_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    _inject_row(monkeypatch)
    tcl, py, rec = TclEmitter(), PyEmitter(), RecordingEmitter()
    tcl.command("probeCmd", *ARGS)
    py.command("probeCmd", *ARGS)
    rec.command("probeCmd", *ARGS)
    assert tcl.lines()[-1] == "probeCmd 1 2.5 x"
    assert py.lines()[-1] == "ops.probeCmd(1, 2.5, 'x')"
    assert rec.calls == [("command", ("probeCmd", 1, 2.5, "x"), {})]


def test_live_calls_the_binding_by_name(monkeypatch: pytest.MonkeyPatch) -> None:
    _inject_row(monkeypatch)
    ops = _Ops()
    _live(ops).command("probeCmd", *ARGS)
    assert ops.calls == [ARGS]


def test_live_names_the_requires_when_the_binding_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _inject_row(monkeypatch, verb="forkOnlyCmd", requires=frozenset({"fork"}))
    with pytest.raises(RuntimeError, match=r"ops\.forkOnlyCmd.*requires fork"):
        _live(_Ops()).command("forkOnlyCmd", 1)


def test_h5_refuses_every_token_until_k1_4(monkeypatch: pytest.MonkeyPatch) -> None:
    _inject_row(monkeypatch)
    with pytest.raises(H5RefusedVerb) as info:
        H5Emitter().command("probeCmd", *ARGS)
    assert isinstance(info.value, NotImplementedError)
    assert info.value.verb == "command"
    assert info.value.row is VERBS["command"]
    assert "'probeCmd'" in str(info.value) and "/opensees/commands" in str(info.value)


# -- the H5 helpers ----------------------------------------------------------

def test_h5_refuse_names_the_verb_and_its_row() -> None:
    with pytest.raises(H5RefusedVerb) as info:
        H5Emitter().equalDOF_mixed(10, 20, [(3, 6)])
    assert info.value.verb == "equalDOF_mixed"
    assert info.value.row is VERBS["equalDOF_mixed"]
    assert "h5=refuse" in str(info.value) and "equalDOF_Mixed" in str(info.value)


def test_h5_ledger_counts_dropped_calls_silently(
    recwarn: pytest.WarningsRecorder,
) -> None:
    e = H5Emitter()
    e.profiler("start")
    e.profiler("report")
    e.modal_damping(0.05)
    e.contact(1, "a")
    e.contact_plane(2, "b")
    e.rayleigh(0.1, 0.2, 0.0, 0.0)
    assert e.eigen(3) == []
    assert e.modal_properties() == {}
    assert e.eigen_feast(0.0, 10.0) == []
    e.modal_response_history(1)
    e.response_spectrum_analysis(1, "-Tn", 0.1)
    assert dict(e._ledger_counts) == {
        "profiler": 2, "modal_damping": 1, "contact": 1, "contact_plane": 1,
        "rayleigh": 1, "eigen": 1, "modal_properties": 1, "eigen_feast": 1,
        "modal_response_history": 1, "response_spectrum_analysis": 1,
    }
    assert all(VERBS[v].h5 == "ledger" for v in e._ledger_counts)
    assert not recwarn.list


def test_h5_legacy_counters_read_the_ledger() -> None:
    e = H5Emitter()
    e.embedded_rebar(1, 2)
    e.embedded_node(3, 4)
    e.contact_surface(5, 6)
    e.contact_surface(7, 8)
    e.equationConstraint(1, 1, 1.0, [(2, 1, -1.0)])
    assert (e._skipped_reinforce_ties, e._skipped_embed_ties,
            e._skipped_contacts, e._skipped_equation_constraints) == (1, 1, 2, 1)


def test_h5_helpers_refuse_a_row_of_the_wrong_kind() -> None:
    e = H5Emitter()
    with pytest.raises(RuntimeError, match="not 'ledger'"):
        e._ledger("fix")
    with pytest.raises(RuntimeError, match="not 'refuse'"):
        e._refuse("fix", "planted")


# -- ADR 0114 D2: a stage-only verb outside a bracket raises -------------------

_STAGE_ONLY_CALLS: list[tuple[str, tuple[Any, ...]]] = [
    ("set_time", (2.5,)),
    ("set_creep", (True,)),
    ("reset", ()),
    ("domain_change", ()),
    ("remove_sp", (1, 1)),
    ("remove_element", (1,)),
    ("update_material_stage", (1, 1)),
    ("sp_hold", (1, 1)),
    ("update_parameter", (1, (1,), ("xPerm",), 1e-5)),
    ("set_node_vel", (1, 1, 0.0)),
    ("set_node_accel", (1, 1, 0.0)),
]


def test_stage_only_calls_cover_the_stage_rows() -> None:
    """Every ``scope == "stage"`` row is listed, except the bracket pair and
    ``flip_element_stage`` (archived by ``set_stage_records``; the lock pins
    its body trivial, so it cannot carry the raise)."""
    stage_rows = {k for k, v in VERBS.items() if v.scope == "stage"}
    exempt = {"stage_open", "stage_close", "flip_element_stage"}
    assert {name for name, _ in _STAGE_ONLY_CALLS} == stage_rows - exempt


@pytest.mark.parametrize(("verb", "args"), _STAGE_ONLY_CALLS,
                         ids=[v for v, _ in _STAGE_ONLY_CALLS])
def test_stage_only_verb_outside_a_bracket_raises(
    verb: str, args: tuple[Any, ...],
) -> None:
    e = H5Emitter()
    with pytest.raises(RuntimeError, match=f"H5Emitter.{verb}: .*outside a stage bracket"):
        getattr(e, verb)(*args)
    assert e._stage_blocks == []


def test_stage_only_verbs_inside_a_bracket_capture() -> None:
    e = H5Emitter()
    e.stage_open("s1")
    e.set_time(2.5)
    e.set_creep(True)
    e.reset()
    e.domain_change()
    e.remove_sp(1, 1)
    e.remove_element(7)
    e.update_material_stage(3, 1)
    e.stage_close()
    (blk,) = e._stage_blocks
    assert (blk.set_time, blk.set_creep_on, blk.pre_analyze_reset,
            blk.domain_changed) == (2.5, True, True, True)
    assert blk.remove_sps == [(1, 1)]
    assert blk.remove_elements == [7]
    assert blk.update_material_stages == [(3, 1)]


def test_refused_stage_verbs_inside_a_bracket_still_refuse() -> None:
    e = H5Emitter()
    e.stage_open("s1")
    with pytest.raises(H5RefusedVerb, match="updateParameter for 'xPerm'") as info:
        e.update_parameter(1, (1,), ("xPerm",), 1e-5)
    assert info.value.verb == "update_parameter"
    with pytest.raises(H5RefusedVerb, match="setNodeVel") as info:
        e.set_node_vel(1, 1, 0.0)
    assert info.value.verb == "set_node_vel"
    with pytest.raises(H5RefusedVerb, match="setNodeAccel") as info:
        e.set_node_accel(1, 1, 0.0)
    assert info.value.verb == "set_node_accel"
