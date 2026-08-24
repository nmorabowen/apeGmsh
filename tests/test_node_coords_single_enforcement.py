"""One invariant, one enforcement point (ADR 0099 reconciliation).

A 2-D ``node`` line must carry exactly ``ndm`` coordinates: a padded third
desynchronises OpenSees' optional-argument scan and the following ``-ndf K``
is silently swallowed. That rule used to be enforced TWICE — once in the
build layer (``node_coords_for_ndm``, with ``ndm`` threaded through ~7
function signatures to reach it) and again, last, at every text/live emitter
(``trim_coords_to_ndm``). The emitter copy always won, so the build-layer
copy was dead weight that still cost the threading.

What the build layer keeps is the *coercion*, which is load-bearing for a
different reason: the broker hands out numpy scalars, and the Tcl emitter
renders an unknown numeric via ``repr`` — under numpy 2.x that is the literal
text ``np.float64(0.0)``, which would emit a corrupt deck.
"""
from __future__ import annotations

import numpy as np
import pytest

from apeGmsh.opensees._internal.build import node_coords_as_floats
from apeGmsh.opensees.emitter.base import trim_coords_to_ndm


# ── the build layer coerces, and NOTHING else ────────────────────────

def test_returns_plain_floats_from_numpy():
    """The load-bearing half. `float` exactly — not np.float64, which is a
    `float` subclass and so would pass a naive isinstance check while still
    repr-ing as `np.float64(...)`."""
    out = node_coords_as_floats(np.array([1.5, 2.5, 3.5]))
    assert out == (1.5, 2.5, 3.5)
    assert all(v.__class__ is float for v in out)


def test_never_trims_regardless_of_dimension():
    """The whole point of the reconciliation: the build layer no longer has
    an opinion about model dimension, so it always hands over three."""
    assert node_coords_as_floats((1.0, 2.0, 3.0)) == (1.0, 2.0, 3.0)
    assert len(node_coords_as_floats(np.array([1.0, 2.0, 3.0]))) == 3


def test_a_numpy_coordinate_would_have_corrupted_the_deck():
    """Guards the reason the coercion survived. If this ever stops being
    true, the coercion is free to go — but until then it is not cosmetic."""
    from apeGmsh.opensees.emitter.tcl import _fmt_value

    assert _fmt_value(np.float64(1.5)) != "1.5"      # the hazard
    assert _fmt_value(node_coords_as_floats(np.array([1.5, 0.0, 0.0]))[0]) == "1.5"


# ── the emitter is the ONLY place dimension is applied ───────────────

@pytest.mark.parametrize("ndm,expected", [(2, (1.0, 2.0)), (3, (1.0, 2.0, 3.0))])
def test_the_emitter_owns_the_dimension_trim(ndm, expected):
    assert trim_coords_to_ndm((1.0, 2.0, 3.0), ndm) == expected


def test_a_2d_deck_still_carries_exactly_two_coordinates(tmp_path):
    """End-to-end: the build layer hands the emitter three coordinates and
    the emitted 2-D line still has two, with the `-ndf` token intact where
    it differs from the envelope. This is the behaviour the doubled
    enforcement existed to protect, now resting on the emitter alone."""
    from apeGmsh.opensees.emitter.tcl import TclEmitter

    em = TclEmitter()
    em.model(ndm=2, ndf=2)
    em.node(1, *node_coords_as_floats(np.array([1.0, 2.0, 0.0])))
    em.node(2, *node_coords_as_floats(np.array([3.0, 4.0, 0.0])), ndf=3)
    text = "\n".join(ln for ln in em.lines() if ln.startswith("node "))

    assert "np.float64" not in text
    assert "node 1 1.0 2.0" in text
    # the -ndf token must be the 4th field, not stranded behind a padded z
    assert "node 2 3.0 4.0 -ndf 3" in text
