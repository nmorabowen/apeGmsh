"""R2b — bridge emit of g.reinforce LadrunoEmbeddedRebar ties.

Unit-level coverage of ``emit_reinforce_ties`` + the new ``embedded_rebar``
emitter method across backends, driven by hand-built
:class:`ReinforceTieRecord` rows (no Gmsh, no fork). The composite +
real-mesh end-to-end path is covered in ``tests/test_reinforce_composite``.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from apeGmsh._kernel.records._constraints import ReinforceTieRecord
from apeGmsh.opensees._internal.build import emit_reinforce_ties
from apeGmsh.opensees._internal.tag_allocator import TagAllocator
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.emitter.tcl import TclEmitter
from apeGmsh.opensees.emitter.h5 import H5Emitter


# --------------------------------------------------------------------------
# Minimal FEM stub exposing only ``elements.reinforce_ties``.
# --------------------------------------------------------------------------
class _Elems:
    def __init__(self, ties):
        self.reinforce_ties = ties


class _Fem:
    def __init__(self, ties):
        self.elements = _Elems(ties)


def _perfect_tie(name=None):
    return ReinforceTieRecord(
        kind="reinforce", name=name,
        rebar_node=9,
        host_nodes=[221, 288, 222, 320],
        weights=np.array([0.1, 0.6, 0.2, 0.1]),
        direction=np.array([0.0, 0.0, 1.0]),
        bond_scale=None, bond=None, perfect=1.0e12,
        kt=None, kt_alpha=None, enforce="penalty",
    )


def _bond_tie(name=None, bond="bond1"):
    return ReinforceTieRecord(
        kind="reinforce", name=name,
        rebar_node=12,
        host_nodes=[1, 2, 3, 4, 5, 6, 7, 8],
        weights=np.full(8, 0.125),
        direction=np.array([1.0, 0.0, 0.0]),
        bond_scale=0.0785, bond=bond, perfect=None,
        kt=1.0e3, kt_alpha=None, enforce="penalty",
    )


def test_perfect_tie_emits_shape_dir_perfect():
    """A perfect-bond tie emits the -shape weights, -dir axis and
    -perfect kAxial via the recording emitter."""
    em = RecordingEmitter()
    emit_reinforce_ties(em, _Fem([_perfect_tie()]), TagAllocator(),
                        name_to_tag={})
    calls = [c for c in em.calls if c[0] == "embedded_rebar"]
    assert len(calls) == 1
    args = calls[0][1]  # (ele_tag, rebar_node, nHost, *hosts, -shape, ...)
    assert args[1] == 9           # rebar node
    assert args[2] == 4           # nHost count
    assert list(args[3:7]) == [221, 288, 222, 320]
    assert "-shape" in args and "-dir" in args and "-perfect" in args
    # -dir axis follows the record direction
    di = args.index("-dir")
    assert list(args[di + 1:di + 4]) == [0.0, 0.0, 1.0]
    pi = args.index("-perfect")
    assert args[pi + 1] == 1.0e12


def test_bond_tie_resolves_name_to_tag_and_bondscale():
    """A bond tie resolves the bond material NAME to its bridge tag and
    emits -bond <tag> -bondScale."""
    em = RecordingEmitter()
    emit_reinforce_ties(em, _Fem([_bond_tie(bond="bond1")]), TagAllocator(),
                        name_to_tag={"bond1": 7})
    args = [c for c in em.calls if c[0] == "embedded_rebar"][0][1]
    assert "-bond" in args
    bi = args.index("-bond")
    assert args[bi + 1] == 7                      # resolved tag, not the name
    assert "-bondScale" in args
    assert args[args.index("-bondScale") + 1] == 0.0785
    assert "-perfect" not in args


def test_bond_unregistered_name_fails_loud():
    """A bond name that is not registered on the bridge raises, naming
    the offending material — never a dangling tag."""
    em = RecordingEmitter()
    with pytest.raises(ValueError, match="bond1"):
        emit_reinforce_ties(em, _Fem([_bond_tie(bond="bond1")]),
                            TagAllocator(), name_to_tag={"other": 3})


def test_fresh_element_tags_per_tie():
    """Each tie draws a fresh element tag from the allocator."""
    em = RecordingEmitter()
    tags = TagAllocator()
    emit_reinforce_ties(
        em, _Fem([_perfect_tie(), _perfect_tie()]), tags, name_to_tag={})
    ele_tags = [c[1][0] for c in em.calls if c[0] == "embedded_rebar"]
    assert len(ele_tags) == 2
    assert ele_tags[0] != ele_tags[1]


def test_name_round_trips_as_mp_comment():
    """A named tie precedes its element with an mp_constraint_comment."""
    em = RecordingEmitter()
    emit_reinforce_ties(em, _Fem([_perfect_tie(name="col_rebar")]),
                        TagAllocator(), name_to_tag={})
    kinds = [c[0] for c in em.calls]
    assert "mp_constraint_comment" in kinds
    assert kinds.index("mp_constraint_comment") < kinds.index("embedded_rebar")


def test_tcl_text_line():
    """The Tcl backend renders one ``element LadrunoEmbeddedRebar`` line."""
    em = TclEmitter()
    emit_reinforce_ties(em, _Fem([_perfect_tie()]), TagAllocator(),
                        name_to_tag={})
    line = next(l for l in em.lines() if "LadrunoEmbeddedRebar" in l)
    assert line.startswith("element LadrunoEmbeddedRebar")
    assert "-shape" in line and "-dir" in line and "-perfect" in line


def test_no_ties_is_noop():
    """A FEM with no reinforce_ties emits nothing."""
    em = RecordingEmitter()
    emit_reinforce_ties(em, _Fem([]), TagAllocator(), name_to_tag={})
    assert not any(c[0] == "embedded_rebar" for c in em.calls)


def test_h5_defers_deck_zone_without_warning():
    """The H5 backend does not write a dedicated tie record into the
    OpenSees DECK zone (``/opensees/...``) — that follow-on (ADR 0067
    P5.1 "A4 full") is deferred — and, since ADR 0067 P5.1 (#706) the
    NEUTRAL zone persists every tie in the same archive, it does so
    SILENTLY (no deviation warning; the round-trip is complete via the
    neutral zone + forward re-emit)."""
    em = H5Emitter(schema_version="x", model_name="m")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        emit_reinforce_ties(
            em, _Fem([_perfect_tie(), _perfect_tie()]), TagAllocator(),
            name_to_tag={})
    # Both deck-ties no-op'd (counted for observability) ...
    assert em._skipped_reinforce_ties == 2
    # ... but NO deviation warning fires (the retired
    # H5ReinforceDeviationWarning — ties persist via the neutral zone).
    assert not [
        x for x in w
        if "reinforc" in str(x.message).lower()
        and ("not persisted" in str(x.message).lower()
             or "deferred" in str(x.message).lower()
             or "missing" in str(x.message).lower())
    ]
    # The deck element store carries no reinforce record (deferred).
    assert not getattr(em, "_elements", [])


def test_h5_consumes_pending_mp_name():
    """The H5 no-op consumes a latched mp comment so it cannot leak onto
    the next real MP record (INV-2)."""
    em = H5Emitter(schema_version="x", model_name="m")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        emit_reinforce_ties(em, _Fem([_perfect_tie(name="r1")]),
                            TagAllocator(), name_to_tag={})
    assert em._pending_mp_name == ""
