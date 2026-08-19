"""A ``node`` line carries exactly ``ndm`` coordinates.

The broker stores every node as ``(x, y, z)``.  OpenSees reads ``ndm``
coordinates off a ``node`` line and then scans what follows for optional
flags, so a padded third coordinate in a 2-D model desynchronises that
scan and the flag is silently never consumed.  Measured on Ladruno
``25a0647f``::

    model BasicBuilder -ndm 2 -ndf 3
    node 1 0.0 0.0 0.0 -ndf 2   ;# -> ndf 3, override DROPPED, no warning
    node 2 1.0 0.0     -ndf 2   ;# -> ndf 2
    node 5 4.0 0.0 0.0 -mass 1.0 1.0
                                ;# -> "incorrect number of nodal mass terms"

The dropped ``-ndf`` is the dangerous half: a gated continuum element
(``tri6n`` / ``LadrunoLST`` / ``quad``, which need node ndf 2) still
PARSES, because the builder-ndf bracket satisfies the parser gate — but
``setDomain`` then bails on the wrong node ndf without setting the
element's domain pointer, and the deck dies at analysis with
``FATAL FE_Element::FE_Element() - element has no domain``.

3-D decks are unaffected: trimming to ndm is the identity on a 3-tuple.
"""
from __future__ import annotations

import pytest

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
# The helper
# =====================================================================

@pytest.mark.parametrize(
    "coords, ndm, expected",
    [
        ((1.0, 2.0, 3.0), 3, (1.0, 2.0, 3.0)),   # 3-D: identity
        ((1.0, 2.0, 0.0), 2, (1.0, 2.0)),        # 2-D: padding dropped
        ((1.0, 0.0, 0.0), 1, (1.0,)),            # 1-D
        ((1.0, 2.0), 3, (1.0, 2.0)),             # already short: untouched
        ((1.0, 2.0, 3.0), None, (1.0, 2.0, 3.0)),  # before model(): untouched
    ],
)
def test_trim_coords_to_ndm(coords, ndm, expected) -> None:
    assert trim_coords_to_ndm(coords, ndm) == expected


# =====================================================================
# The emitters
# =====================================================================

def test_tcl_2d_node_drops_the_padded_coordinate() -> None:
    e = TclEmitter()
    e.model(ndm=2, ndf=3)
    e.node(1, 0.0, 0.0, 0.0, ndf=2)
    e.node(2, 1.0, 2.0, 0.0)
    assert "node 1 0.0 0.0 -ndf 2" in e._lines
    assert "node 2 1.0 2.0" in e._lines


def test_tcl_3d_node_is_unchanged() -> None:
    e = TclEmitter()
    e.model(ndm=3, ndf=6)
    e.node(9, 1.0, 2.0, 3.0, ndf=3)
    e.node(10, 4.0, 5.0, 6.0)
    assert "node 9 1.0 2.0 3.0 -ndf 3" in e._lines
    assert "node 10 4.0 5.0 6.0" in e._lines


def test_py_2d_node_drops_the_padded_coordinate() -> None:
    e = PyEmitter()
    e.model(ndm=2, ndf=3)
    e.node(1, 0.0, 0.0, 0.0, ndf=2)
    e.node(2, 1.0, 2.0, 0.0)
    assert "ops.node(1, 0.0, 0.0, '-ndf', 2)" in e._lines
    assert "ops.node(2, 1.0, 2.0)" in e._lines


def test_recording_mirrors_the_concrete_emitters() -> None:
    """Otherwise the parity sweep would compare unlike with unlike."""
    e = RecordingEmitter()
    e.model(ndm=2, ndf=3)
    e.node(1, 0.0, 0.0, 0.0, ndf=2)
    assert ("node", (1, 0.0, 0.0), {"ndf": 2}) in e.calls


def test_emitter_without_model_call_passes_coords_through() -> None:
    """``ndm`` is unknown until ``model()``; direct emitter use in tests
    must keep working."""
    e = TclEmitter()
    e.node(1, 0.0, 0.0, 0.0)
    assert "node 1 0.0 0.0 0.0" in e._lines


# =====================================================================
# End-to-end deck
# =====================================================================

def _mixed_ndf_2d() -> apeSees:
    """tri6 soil (inferred ndf 2) + a beam on separate nodes (ndf 3)."""
    soil = {
        1: (0.0, 0.0, 0.0), 2: (2.0, 0.0, 0.0), 3: (0.0, 1.0, 0.0),
        4: (1.0, 0.0, 0.0), 5: (1.0, 0.5, 0.0), 6: (0.0, 0.5, 0.0),
    }
    beam = {7: (5.0, 0.0, 0.0), 8: (5.0, 1.0, 0.0)}
    ids = sorted(soil | beam)
    fem = FEMStub(
        nodes=_NodesStub(
            ids=ids, coords=[(soil | beam)[i] for i in ids], node_pgs={},
        ),
        elements=_ElementsStub(elem_pgs={
            "Rock": _ElementGroupView(ids=(1,), connectivity=((1, 2, 3, 4, 5, 6),)),
            "Liner": _ElementGroupView(ids=(2,), connectivity=((7, 8),)),
        }),
    )
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=3)
    sec = ops.section.Elastic(E=2.0e8, A=0.01, Iz=1e-5)
    ops.element.dispBeamColumn(
        pg="Liner",
        transf=ops.geomTransf.Linear(),
        integration=ops.beamIntegration.Lobatto(section=sec, n_ip=5),
    )
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.SixNodeTri(pg="Rock", thickness=1.0, material=mat)
    return ops


def test_2d_deck_node_lines_carry_two_coordinates(tmp_path) -> None:
    deck = tmp_path / "deck.tcl"
    _mixed_ndf_2d().tcl(str(deck))
    node_lines = [
        ln for ln in deck.read_text().splitlines() if ln.startswith("node ")
    ]
    assert node_lines, "fixture emitted no node lines"
    for ln in node_lines:
        tok = ln.split()
        coords = tok[2:tok.index("-ndf")] if "-ndf" in tok else tok[2:]
        assert len(coords) == 2, f"{ln!r} carries {len(coords)} coordinates"


def test_2d_deck_ndf_override_follows_the_two_coordinates(tmp_path) -> None:
    """The parse-level property that actually matters: ``-ndf`` must sit
    immediately after the ndm coordinates, or OpenSees never reads it."""
    deck = tmp_path / "deck.tcl"
    _mixed_ndf_2d().tcl(str(deck))
    overridden = [
        ln.split() for ln in deck.read_text().splitlines()
        if ln.startswith("node ") and "-ndf" in ln
    ]
    assert overridden, "fixture emitted no per-node ndf override"
    for tok in overridden:
        # node <tag> <x> <y> -ndf <K>  ->  the flag sits at index 2 + ndm.
        assert tok[4] == "-ndf", (
            f"-ndf does not follow the ndm coordinates in {' '.join(tok)!r}"
        )
