"""ADR 0068 P4 — the ``enforce="penalty_al"`` tie route → fork
``LadrunoEmbeddedNode`` (penalty + augmented-Lagrange + bipenalty),
configured via the RBE2/RBE3 :class:`CouplingControl`.

Unlike the penalty ``ASDEmbeddedNodeElement`` route, the shape weights ARE
emitted (``-shape``) and any host arity is accepted. The knobs reuse
``CouplingControl`` (already H5-persisted via the ``cpl_*`` columns), so
``enforce``/``control`` round-trip with no new schema work.
"""
from __future__ import annotations

import numpy as np
import pytest

from apeGmsh._kernel._coupling_control import CouplingControl
from apeGmsh._kernel.defs.constraints import TieDef, TiedContactDef
from apeGmsh._kernel.records._constraints import InterpolationRecord
from apeGmsh._kernel.records._kinds import ConstraintKind as K
from apeGmsh.opensees._internal.build import _emit_one_interpolation
from apeGmsh.opensees._internal.tag_allocator import TagAllocator
from apeGmsh.opensees.emitter.recording import RecordingEmitter


# --------------------------------------------------------------------------
# def-level: control field + route compatibility
# --------------------------------------------------------------------------

def test_penalty_al_accepts_control():
    c = CouplingControl(k=1.0e9, enforce="al")
    d = TieDef(master_label="A", slave_label="B",
               enforce="penalty_al", control=c)
    assert d.control is c
    tc = TiedContactDef(master_label="A", slave_label="B",
                        enforce="penalty_al", control=c)
    assert tc.control is c


@pytest.mark.parametrize("enforce", ["penalty", "equation"])
def test_control_only_valid_with_penalty_al(enforce):
    c = CouplingControl(k=1.0e9)
    with pytest.raises(ValueError, match="penalty_al"):
        TieDef(master_label="A", slave_label="B", enforce=enforce, control=c)


@pytest.mark.parametrize("kw", [
    {"rotational": True},
    {"pressure": True},
    {"pressure": True, "stiffness_p": 1.0e12},
])
def test_penalty_al_rejects_asd_only_knobs(kw):
    with pytest.raises(ValueError,
                       match="ASDEmbeddedNodeElement-only|translations only"):
        TieDef(master_label="A", slave_label="B", enforce="penalty_al", **kw)


# --------------------------------------------------------------------------
# emission: LadrunoEmbeddedNode element line
# --------------------------------------------------------------------------

def _emit(rec):
    e = RecordingEmitter()
    _emit_one_interpolation(e, rec, TagAllocator())
    return e


def test_penalty_al_emits_ladruno_embedded_node_with_control_flags():
    c = CouplingControl(k=1.0e9, enforce="al", absolute=True)
    rec = InterpolationRecord(
        kind=K.TIE, slave_node=4, master_nodes=[1, 2, 3],
        weights=np.array([0.5, 0.3, 0.2]), dofs=[1, 2, 3],
        enforce="penalty_al", control=c,
    )
    calls = [x for x in _emit(rec).calls if x[0] == "element"]
    assert len(calls) == 1
    a = calls[0][1]
    assert a[0] == "LadrunoEmbeddedNode"
    assert a[2] == 4                        # cNode (a[1] is the ele tag)
    assert list(a[3:6]) == [1, 2, 3]        # host nodes (positional)
    assert a[6] == "-shape"
    assert list(a[7:10]) == [0.5, 0.3, 0.2]  # weights emitted (unlike penalty)
    assert list(a[10:]) == ["-k", 1.0e9, "-enforce", "al", "-absolute"]
    # not the penalty / equation routes
    assert not [x for x in _emit(rec).calls
                if x[0] in ("embeddedNode", "equationConstraint")]


def test_penalty_al_without_control_emits_bare_shape():
    rec = InterpolationRecord(
        kind=K.TIE, slave_node=5, master_nodes=[1, 2, 3, 4],
        weights=np.array([0.25, 0.25, 0.25, 0.25]), dofs=[1, 2, 3],
        enforce="penalty_al",
    )
    a = [x for x in _emit(rec).calls if x[0] == "element"][0][1]
    assert a[0] == "LadrunoEmbeddedNode"
    assert "-shape" in a                    # 4-node host accepted (no 3/4 guard)
    assert "-k" not in a and "-enforce" not in a   # fork defaults


def test_penalty_al_weight_mismatch_raises():
    rec = InterpolationRecord(
        kind=K.TIE, slave_node=4, master_nodes=[1, 2, 3],
        weights=np.array([0.5, 0.5]), dofs=[1, 2, 3], enforce="penalty_al",
    )
    with pytest.raises(ValueError, match="mismatch"):
        _emit(rec)


def test_penalty_al_is_fork_gated_in_live_emitter():
    # LadrunoEmbeddedNode must be in the live emitter's fork-only set so a
    # stock-OpenSees live run fails loud (not a cryptic parser error).
    from apeGmsh.opensees.emitter.live import _FORK_ONLY_ELEMENTS
    assert "LadrunoEmbeddedNode" in _FORK_ONLY_ELEMENTS


# --------------------------------------------------------------------------
# ADR 0069 follow-up — EmbeddedNodeControl pressure tie (-pressure / -kp)
# --------------------------------------------------------------------------

def test_embedded_node_control_emits_pressure_flags():
    from apeGmsh._kernel._coupling_control import EmbeddedNodeControl

    c = EmbeddedNodeControl(k=1.0e9, pressure=True, kp=2.0e12)
    rec = InterpolationRecord(
        kind=K.TIE, slave_node=4, master_nodes=[1, 2, 3],
        weights=np.array([0.5, 0.3, 0.2]), dofs=[1, 2, 3],
        enforce="penalty_al", control=c,
    )
    a = [x for x in _emit(rec).calls if x[0] == "element"][0][1]
    assert a[0] == "LadrunoEmbeddedNode"
    # base + pressure flags ride the same order-independent tail.
    assert list(a[10:]) == ["-k", 1.0e9, "-pressure", "-kp", 2.0e12]


def test_embedded_node_control_pressure_without_kp_omits_kp():
    from apeGmsh._kernel._coupling_control import EmbeddedNodeControl

    c = EmbeddedNodeControl(pressure=True)   # fork default kp
    rec = InterpolationRecord(
        kind=K.TIE, slave_node=4, master_nodes=[1, 2, 3],
        weights=np.array([0.5, 0.3, 0.2]), dofs=[1, 2, 3],
        enforce="penalty_al", control=c,
    )
    a = [x for x in _emit(rec).calls if x[0] == "element"][0][1]
    assert "-pressure" in a and "-kp" not in a


def test_embedded_node_control_accepted_on_tie_defs():
    from apeGmsh._kernel._coupling_control import EmbeddedNodeControl

    c = EmbeddedNodeControl(pressure=True, kp=1.0e12)
    # EmbeddedNodeControl is-a CouplingControl, so the penalty_al route
    # accepts it on every surface tie def.
    TieDef(master_label="A", slave_label="B", enforce="penalty_al", control=c)
    TiedContactDef(master_label="A", slave_label="B",
                   enforce="penalty_al", control=c)
