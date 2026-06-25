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


@pytest.mark.parametrize("kw, match", [
    (dict(k=-1.0), "k .penalty stiffness."),
    (dict(k=0.0), "k .penalty stiffness."),
    (dict(k_alpha=0.0), "k_alpha"),
    (dict(k_alpha=-2.0), "k_alpha"),
])
def test_def_range_validation(kw, match):
    with pytest.raises(ValueError, match=match):
        EmbedDef(master_label="h", slave_label="n", **kw)


def test_def_range_check_rejects_numpy_and_bool():
    with pytest.raises(ValueError, match="k .penalty stiffness."):
        EmbedDef(master_label="h", slave_label="n", k=np.float32(-1.0))
    with pytest.raises(ValueError, match="k .penalty stiffness."):
        EmbedDef(master_label="h", slave_label="n", k=True)
    EmbedDef(master_label="h", slave_label="n", k=np.float64(1.0e18))


# --------------------------------------------------------------------------
# Curved-host detector (#9) — edge-midpoint test, any-direction curvature
# --------------------------------------------------------------------------
@pytest.fixture
def _gmsh_session():
    import gmsh
    gmsh.initialize()
    try:
        yield gmsh
    finally:
        gmsh.finalize()


def test_curved_host_detector_quad8(_gmsh_session):
    """A straight quad8 is silent; an INWARD-bulged edge node (the bbox
    test's false-negative blind spot) is detected."""
    from apeGmsh.core.EmbedmentsComposite import _host_has_curved_edge
    row = list(range(10, 18))
    corners = {10: (0, 0, 0), 11: (1, 0, 0), 12: (1, 1, 0), 13: (0, 1, 0)}
    straight = {14: (0.5, 0, 0), 15: (1, 0.5, 0),
                16: (0.5, 1, 0), 17: (0, 0.5, 0)}
    cs = {k: np.array(v, float) for k, v in {**corners, **straight}.items()}
    assert _host_has_curved_edge(16, 8, row, cs) is False
    # bottom edge node pulled INWARD to y=0.3 — inside the corner bbox, so the
    # old bbox heuristic missed it; the midpoint test catches it.
    curved = dict(straight)
    curved[14] = (0.5, 0.3, 0)
    cc = {k: np.array(v, float) for k, v in {**corners, **curved}.items()}
    assert _host_has_curved_edge(16, 8, row, cc) is True


def test_curved_host_detector_straight_tet10(_gmsh_session):
    """A straight tet10 (edge nodes at true midpoints) is silent — generic
    across element families via gmsh reference param coords."""
    import gmsh

    from apeGmsh.core.EmbedmentsComposite import _host_has_curved_edge
    row = list(range(20, 30))
    cverts = [np.array(v, float) for v in
              ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))]
    loc = np.asarray(
        gmsh.model.mesh.getElementProperties(11)[4], float).reshape(10, 3)
    coords = {}
    for i in range(10):
        u, v, w = loc[i]
        coords[20 + i] = (1 - u - v - w) * cverts[0] + \
            u * cverts[1] + v * cverts[2] + w * cverts[3]
    assert _host_has_curved_edge(11, 10, row, coords) is False


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
