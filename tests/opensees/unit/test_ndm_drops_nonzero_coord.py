"""#1337: ``ops.model(ndm=2)`` must not silently drop a non-zero z.

The emitters trim every node to ``ndm`` coordinates (ADR 0099,
``trim_coords_to_ndm``). The trim used to drop whatever sat beyond
``ndm``. A portal frame drawn in the x-z plane (z up) under
``ops.model(ndm=2, ndf=3)`` then emitted ``node 1 0.0 0.0`` and
``node 2 0.0 0.0`` for its column ends: the columns collapsed to zero
length and the run died inside OpenSees with
``LinearCrdTransf2d::computeElemtLengthAndOrien: 0 length``.

The oracle is the geometry itself. A coordinate beyond ``ndm`` is
padding only when it is zero (up to round-off). Anything else is lost
geometry, and the emit must refuse it by name. A planar model in the
z = 0 plane must emit exactly as before.
"""
from __future__ import annotations

import pytest

from apeGmsh.opensees._internal.build import BridgeError
from apeGmsh.opensees.apesees import apeSees
from apeGmsh.opensees.emitter.base import trim_coords_to_ndm
from apeGmsh.opensees.emitter.py import PyEmitter
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.emitter.tcl import TclEmitter

from tests.opensees.fixtures.fem_stub import (
    FEMStub,
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
)


# =====================================================================
# The trim helper: padding goes, geometry is refused
# =====================================================================

def test_nonzero_z_under_ndm2_raises_naming_node_axis_and_value() -> None:
    with pytest.raises(BridgeError) as exc:
        trim_coords_to_ndm((0.0, 0.0, 3.0), 2, tag=2)
    msg = str(exc.value)
    assert "ndm=2" in msg
    assert "node 2" in msg
    assert "z = 3.0" in msg
    assert "z = 0 plane" in msg
    assert "ndm=3" in msg


def test_negative_z_is_refused_too() -> None:
    with pytest.raises(BridgeError, match="z = -0.5"):
        trim_coords_to_ndm((1.0, 1.0, -0.5), 2, tag=7)


@pytest.mark.parametrize(
    "coords, expected",
    [
        ((6.0, 3.0, 0.0), (6.0, 3.0)),       # exact padding
        ((6.0, 3.0, -0.0), (6.0, 3.0)),      # signed zero is padding
        ((6.0, 3.0, 1e-14), (6.0, 3.0)),     # mesher round-off
        ((0.0, 0.0, 5e-10), (0.0, 0.0)),     # below the floor of 1 * rtol
        ((1.0e4, 0.0, 1.0e-6), (1.0e4, 0.0)),  # relative to the node's size
    ],
)
def test_roundoff_beyond_ndm_is_still_padding(coords, expected) -> None:
    assert trim_coords_to_ndm(coords, 2, tag=1) == expected


def test_tolerance_is_relative_not_absolute() -> None:
    """The same 1e-6 offset that is round-off at x = 1e4 is geometry at
    x = 1: the tolerance scales with the kept coordinates."""
    with pytest.raises(BridgeError, match="node 3"):
        trim_coords_to_ndm((1.0, 0.0, 1.0e-6), 2, tag=3)


def test_ndm1_refuses_a_nonzero_y() -> None:
    assert trim_coords_to_ndm((4.0, 0.0, 0.0), 1, tag=1) == (4.0,)
    with pytest.raises(BridgeError, match=r"node 5 has y = 2\.0"):
        trim_coords_to_ndm((1.0, 2.0, 0.0), 1, tag=5)


def test_3d_and_pre_model_paths_never_inspect_z() -> None:
    assert trim_coords_to_ndm((1.0, 2.0, 3.0), 3, tag=1) == (1.0, 2.0, 3.0)
    assert trim_coords_to_ndm((1.0, 2.0, 3.0), None, tag=1) == (1.0, 2.0, 3.0)


# =====================================================================
# Every emitter that trims refuses (tcl / py / recording; live below)
# =====================================================================

@pytest.mark.parametrize("emitter_cls", [TclEmitter, PyEmitter, RecordingEmitter])
def test_each_trimming_emitter_refuses_a_nonplanar_2d_node(emitter_cls) -> None:
    e = emitter_cls()
    e.model(ndm=2, ndf=3)
    e.node(1, 0.0, 0.0, 0.0)
    with pytest.raises(BridgeError, match="node 2 has z = 3.0"):
        e.node(2, 0.0, 0.0, 3.0)


# =====================================================================
# End to end: the issue's portal frame, x-z (refused) vs x-y (emits)
# =====================================================================

def _portal(*, vertical_axis: int) -> apeSees:
    """Columns A-B and D-C, beam B-C; 6 m bay, 3 m tall.

    ``vertical_axis`` 2 draws the frame in x-z (the #1337 script);
    1 draws the same frame in x-y, the plane a 2-D model must use.
    """
    xyz = {}
    for tag, (x, v) in {1: (0.0, 0.0), 2: (0.0, 3.0),
                        3: (6.0, 3.0), 4: (6.0, 0.0)}.items():
        p = [x, 0.0, 0.0]
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


@pytest.mark.parametrize("route", ["tcl", "py"])
def test_xz_portal_under_ndm2_is_refused_before_the_deck_exists(
    tmp_path, route,
) -> None:
    deck = tmp_path / f"deck.{route}"
    ops = _portal(vertical_axis=2)
    with pytest.raises(BridgeError, match=r"node 2 has z = 3\.0"):
        getattr(ops, route)(str(deck))
    assert not deck.exists(), "a refused emit must not leave a deck behind"


def test_xy_portal_under_ndm2_emits_four_distinct_nodes(tmp_path) -> None:
    deck = tmp_path / "deck.tcl"
    _portal(vertical_axis=1).tcl(str(deck))
    nodes = [
        tuple(ln.split()[2:4]) for ln in deck.read_text().splitlines()
        if ln.startswith("node ")
    ]
    assert sorted(nodes) == sorted([
        ("0.0", "0.0"), ("0.0", "3.0"), ("6.0", "3.0"), ("6.0", "0.0"),
    ])


@pytest.mark.live
def test_xz_portal_live_raises_bridge_error_not_an_opensees_abort() -> None:
    """The issue's live symptom was an OpenSees transformation error and a
    dead process. The live emitter trims through the same helper, so the
    refusal comes first, as a catchable BridgeError."""
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
