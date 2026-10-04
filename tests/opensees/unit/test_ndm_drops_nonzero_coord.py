"""#1337: ``ops.model(ndm=2)`` must not silently drop a varying z.

The emitters trim every node to ``ndm`` coordinates (ADR 0099). The trim
used to drop whatever lay beyond ``ndm``. A portal frame drawn in the
x-z plane (z up) under ``ops.model(ndm=2, ndf=3)`` then emitted
``node 1 0.0 0.0`` and ``node 2 0.0 0.0`` for its column ends. The
columns collapsed to zero length, and the run died inside OpenSees with
``LinearCrdTransf2d::computeElemtLengthAndOrien: 0 length``.

The oracle is the geometry. Dropping an axis is lossless exactly when
every node has the same value on it. A model in the plane z = z0 emits
the same 2-D deck for every z0, so the deck at z0 = 3 must equal the deck
at z0 = 0 line for line. Any spread in the dropped axis is lost geometry,
and the emit must refuse it and name the nodes.
"""
from __future__ import annotations

import pytest

from apeGmsh.opensees._internal.build import BridgeError
from apeGmsh.opensees.apesees import apeSees
from apeGmsh.opensees.emitter.base import DroppedAxisGuard
from apeGmsh.opensees.emitter.h5 import H5Emitter
from apeGmsh.opensees.emitter.py import PyEmitter
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.emitter.tcl import TclEmitter

from tests.opensees.fixtures.fem_stub import (
    FEMStub,
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
)


def _trim_all(ndm, nodes):
    g = DroppedAxisGuard(ndm)
    return [g.trim(c, tag) for tag, c in nodes]


# =====================================================================
# The guard: a constant dropped axis is padding, a varying one is refused
# =====================================================================

def test_varying_z_under_ndm2_raises_naming_both_nodes_and_values() -> None:
    with pytest.raises(BridgeError) as exc:
        _trim_all(2, [(1, (0.0, 0.0, 0.0)), (2, (0.0, 0.0, 3.0))])
    msg = str(exc.value)
    assert "ndm=2" in msg
    assert "node 1 has z = 0.0" in msg
    assert "node 2 has z = 3.0" in msg
    assert "constant z" in msg
    assert "ndm=3" in msg


def test_a_uniformly_offset_plane_is_padding() -> None:
    """The strut_tie case: a region meshed at z0 = the STM's z."""
    out = _trim_all(2, [
        (1, (0.0, 0.0, 3.0)), (2, (0.0, 4.0, 3.0)), (3, (6.0, 4.0, 3.0)),
    ])
    assert out == [(0.0, 0.0), (0.0, 4.0), (6.0, 4.0)]


@pytest.mark.parametrize(
    "z_ref, z_other",
    [
        (0.0, -0.0),            # signed zero
        (0.0, 1e-14),           # mesher round-off on z = 0
        (1.0e4, 1.0e4 + 1e-7),  # round-off relative to a far plane
    ],
)
def test_roundoff_spread_is_still_padding(z_ref, z_other) -> None:
    out = _trim_all(2, [(1, (6.0, 3.0, z_ref)), (2, (6.0, 0.0, z_other))])
    assert out == [(6.0, 3.0), (6.0, 0.0)]


def test_tolerance_is_relative_not_absolute() -> None:
    """A spread of 1e-6 is round-off at |x| = 1e4 but geometry at |x| = 1."""
    assert _trim_all(2, [(1, (1.0e4, 0.0, 0.0)), (2, (1.0e4, 1.0, 1e-6))])
    with pytest.raises(BridgeError, match="node 2 has z = 1e-06"):
        _trim_all(2, [(1, (1.0, 0.0, 0.0)), (2, (1.0, 1.0, 1e-6))])


def test_ndm1_refuses_a_varying_y() -> None:
    assert _trim_all(1, [(1, (0.0, 0.0, 0.0)), (2, (4.0, 0.0, 0.0))]) == [
        (0.0,), (4.0,),
    ]
    with pytest.raises(BridgeError, match="node 5 has y = 2.0"):
        _trim_all(1, [(1, (0.0, 0.0, 0.0)), (5, (1.0, 2.0, 0.0))])


def test_3d_and_before_model_never_inspect_the_tail() -> None:
    assert _trim_all(3, [(1, (0.0, 0.0, 0.0)), (2, (0.0, 0.0, 3.0))]) == [
        (0.0, 0.0, 0.0), (0.0, 0.0, 3.0),
    ]
    g = DroppedAxisGuard.BEFORE_MODEL
    assert g.trim((1.0, 2.0, 3.0), 1) == (1.0, 2.0, 3.0)
    assert g.trim((1.0, 2.0, 9.0), 2) == (1.0, 2.0, 9.0)


# =====================================================================
# Every emitter refuses (tcl / py / recording / h5; live below)
# =====================================================================

@pytest.mark.parametrize(
    "make", [TclEmitter, PyEmitter, RecordingEmitter, lambda: H5Emitter()],
)
def test_each_emitter_refuses_a_nonplanar_2d_node(make) -> None:
    e = make()
    e.model(ndm=2, ndf=3)
    e.node(1, 0.0, 0.0, 0.0)
    with pytest.raises(BridgeError, match="node 2 has z = 3.0"):
        e.node(2, 0.0, 0.0, 3.0)


def test_each_model_call_starts_a_fresh_reference() -> None:
    e = TclEmitter()
    e.model(ndm=2, ndf=3)
    e.node(1, 0.0, 0.0, 0.0)
    e.model(ndm=2, ndf=3)
    e.node(2, 0.0, 0.0, 3.0)
    assert "node 2 0.0 0.0" in e._lines


# =====================================================================
# End to end: the issue's portal frame
# =====================================================================

def _portal(*, vertical_axis: int, z0: float = 0.0) -> apeSees:
    """Columns A-B and D-C, beam B-C; 6 m bay, 3 m tall.

    ``vertical_axis`` 2 draws the frame in x-z (the #1337 script);
    1 draws it in x-y at ``z = z0``.
    """
    xyz = {}
    for tag, (x, v) in {1: (0.0, 0.0), 2: (0.0, 3.0),
                        3: (6.0, 3.0), 4: (6.0, 0.0)}.items():
        p = [x, 0.0, z0]
        p[vertical_axis] = v
        xyz[tag] = tuple(p)
    ids = sorted(xyz)
    fem = FEMStub(
        nodes=_NodesStub(ids=ids, coords=[xyz[i] for i in ids], node_pgs={}),
        elements=_ElementsStub(elem_pgs={
            "cols": _ElementGroupView(ids=(1, 3), connectivity=((1, 2), (4, 3))),
            "beam": _ElementGroupView(ids=(2,), connectivity=((2, 3),)),
        }),
    )
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=3)
    t = ops.geomTransf.Linear()
    ops.element.elasticBeamColumn(pg="cols", transf=t, A=0.25, E=25e9, Iz=0.005)
    ops.element.elasticBeamColumn(pg="beam", transf=t, A=0.25, E=25e9, Iz=0.005)
    return ops


@pytest.mark.parametrize("route, ext", [("tcl", "tcl"), ("py", "py"), ("h5", "h5")])
def test_xz_portal_under_ndm2_is_refused_before_any_file_exists(
    tmp_path, route, ext,
) -> None:
    out = tmp_path / f"model.{ext}"
    ops = _portal(vertical_axis=2)
    with pytest.raises(BridgeError, match=r"node 2 has z = 3\.0"):
        getattr(ops, route)(str(out))
    assert not out.exists(), "a refused emit must not leave a file behind"


def test_xy_portal_under_ndm2_emits_four_distinct_nodes(tmp_path) -> None:
    deck = tmp_path / "deck.tcl"
    _portal(vertical_axis=1).tcl(str(deck))
    nodes = [
        tuple(ln.split()[2:]) for ln in deck.read_text().splitlines()
        if ln.startswith("node ")
    ]
    assert sorted(nodes) == sorted([
        ("0.0", "0.0"), ("0.0", "3.0"), ("6.0", "3.0"), ("6.0", "0.0"),
    ])


def test_an_offset_plane_emits_the_same_deck_as_z0(tmp_path) -> None:
    """Dropping a constant z is lossless: the decks match line for line."""
    at0, at3 = tmp_path / "z0.tcl", tmp_path / "z3.tcl"
    _portal(vertical_axis=1, z0=0.0).tcl(str(at0))
    _portal(vertical_axis=1, z0=3.0).tcl(str(at3))

    def body(p):
        return [ln for ln in p.read_text().splitlines() if not ln.startswith("#")]

    assert body(at3) == body(at0)


@pytest.mark.live
def test_xz_portal_live_raises_bridge_error_not_an_opensees_abort() -> None:
    """The issue's live symptom was an OpenSees transformation error and a
    dead process. The live emitter trims through the same guard, so the
    refusal comes first, as a BridgeError the caller can catch."""
    ops = _portal(vertical_axis=2)
    with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
        p.load(node=3, forces=(1e3, 0.0, 0.0))
    ops.constraints.Plain()
    ops.numberer.RCM()
    ops.system.BandGeneral()
    ops.test.NormDispIncr(tol=1e-10, max_iter=10)
    ops.algorithm.Linear()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()
    with pytest.raises(BridgeError, match=r"node 2 has z = 3\.0"):
        ops.analyze(steps=1)
