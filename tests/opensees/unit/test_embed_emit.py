"""Emit + validation contract for ``g.embed`` → ``LadrunoEmbeddedNode``.

The isotropic node-to-host tie (ELE 33006), sibling of g.reinforce. These
text-only tests (no OpenSees backend) lock the emit grammar
(``embedded_node_args``), the def validation (``EmbedDef``), and the
record→emit path (``emit_embed_ties`` → ``embedded_node`` emitter call):

* U-only translational tie, g0 stress-free birth by default (no -absolute);
* ``-k`` numeric, ``-enforce al``, ``-bipenalty -dtcr`` legs;
* experimental modes (-rot / -pressure / -normal / -corot) are NEVER emitted;
* ``-k auto`` / ``-wcap`` are deferred (need the host-element-tag form).
"""
from __future__ import annotations

import numpy as np
import pytest

from apeGmsh._kernel.defs.constraints import EmbedDef
from apeGmsh._kernel.records._constraints import EmbedTieRecord
from apeGmsh.opensees._internal.build import emit_embed_ties
from apeGmsh.opensees._internal.tag_allocator import TagAllocator
from apeGmsh.opensees.element.embedded_node import embedded_node_args
from apeGmsh.opensees.emitter.recording import RecordingEmitter


# --------------------------------------------------------------------------
# Builder grammar (embedded_node_args)
# --------------------------------------------------------------------------
def test_penalty_default_g0_birth_no_absolute():
    a = embedded_node_args(cnode=9, host_nodes=[1, 2, 3, 4], shape=[0.25] * 4)
    # cNode, nHost+nodes, -shape weights; g0 default ⇒ no -absolute.
    assert a[0] == 9
    assert a[1:6] == [4, 1, 2, 3, 4]
    assert a[6] == "-shape"
    assert "-absolute" not in a
    # No experimental modes ever.
    for flag in ("-rot", "-pressure", "-normal", "-corot", "-kr", "-kp", "-matN"):
        assert flag not in a


def test_numeric_k_emits_k_flag():
    a = embedded_node_args(cnode=1, host_nodes=[1, 2, 3, 4], shape=[0.25] * 4, k=1e12)
    assert "-k" in a and a[a.index("-k") + 1] == 1e12


def test_enforce_al_leg():
    a = embedded_node_args(cnode=1, host_nodes=[1, 2], shape=[0.5, 0.5], enforce="al")
    assert a[-2:] == ["-enforce", "al"]


def test_bipenalty_dtcr_leg():
    a = embedded_node_args(
        cnode=1, host_nodes=[1, 2], shape=[0.5, 0.5], bipenalty=True, dtcr=2.5e-6)
    assert "-bipenalty" in a
    assert a[a.index("-dtcr") + 1] == 2.5e-6


def test_staged_false_emits_absolute():
    a = embedded_node_args(
        cnode=1, host_nodes=[1, 2], shape=[0.5, 0.5], staged=False)
    assert a[-1] == "-absolute"


def test_k_auto_needs_host_form():
    with pytest.raises(ValueError, match="host_ele"):
        embedded_node_args(cnode=1, host_nodes=[1, 2], shape=[0.5, 0.5], k="auto")


def test_bipenalty_penalty_gated():
    with pytest.raises(ValueError, match="penalty"):
        embedded_node_args(
            cnode=1, host_nodes=[1, 2], shape=[0.5, 0.5],
            enforce="al", bipenalty=True, dtcr=1e-5)


def test_shape_length_must_match_host_nodes():
    with pytest.raises(ValueError, match="weights"):
        embedded_node_args(cnode=1, host_nodes=[1, 2, 3], shape=[0.5, 0.5])


# --------------------------------------------------------------------------
# Def validation (EmbedDef)
# --------------------------------------------------------------------------
def test_def_enforce_must_be_penalty_or_al():
    with pytest.raises(ValueError, match="enforce"):
        EmbedDef(master_label="h", slave_label="n", enforce="rough")


def test_def_explicit_requires_dtcr():
    with pytest.raises(ValueError, match="dtcr"):
        EmbedDef(master_label="h", slave_label="n", explicit=True)


def test_def_dtcr_requires_explicit():
    with pytest.raises(ValueError, match="explicit"):
        EmbedDef(master_label="h", slave_label="n", dtcr=1e-5)


def test_def_explicit_gated_on_penalty():
    with pytest.raises(ValueError, match="penalty"):
        EmbedDef(master_label="h", slave_label="n",
                 enforce="al", explicit=True, dtcr=1e-5)


def test_def_k_auto_deferred():
    with pytest.raises(ValueError, match="auto"):
        EmbedDef(master_label="h", slave_label="n", k="auto")


# --------------------------------------------------------------------------
# Record → emit (emit_embed_ties)
# --------------------------------------------------------------------------
class _Fem:
    def __init__(self, ties):
        self.elements = type("E", (), {"embed_ties": ties})()


def _rec(**over):
    base = dict(
        kind="embed", node=9, host_nodes=[1, 2, 3, 4],
        weights=np.full(4, 0.25), k=1e12, enforce="penalty",
    )
    base.update(over)
    return EmbedTieRecord(**base)


def test_emit_routes_to_embedded_node_with_token():
    em = RecordingEmitter()
    emit_embed_ties(em, _Fem([_rec()]), TagAllocator())
    calls = [c for c in em.calls if c[0] == "embedded_node"]
    assert len(calls) == 1
    ele_tag, *args = calls[0][1]
    # cNode + nHost form + -shape + -k
    assert args[0] == 9
    assert args[1:6] == [4, 1, 2, 3, 4]
    assert "-shape" in args and "-k" in args


def test_emit_al_record_emits_enforce_al():
    em = RecordingEmitter()
    emit_embed_ties(em, _Fem([_rec(enforce="al", k=None)]), TagAllocator())
    args = [c for c in em.calls if c[0] == "embedded_node"][0][1]
    assert "-enforce" in args and args[args.index("-enforce") + 1] == "al"


def test_emit_noop_when_no_ties():
    em = RecordingEmitter()
    emit_embed_ties(em, _Fem([]), TagAllocator())
    assert [c for c in em.calls if c[0] == "embedded_node"] == []
