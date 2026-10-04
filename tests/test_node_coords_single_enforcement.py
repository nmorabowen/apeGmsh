"""One invariant, one enforcement point (ADR 0099 reconciliation).

A 2-D ``node`` line must carry exactly ``ndm`` coordinates: a padded third
desynchronises OpenSees' optional-argument scan and the following ``-ndf K``
is silently swallowed. That rule used to be enforced TWICE — once in the
build layer (``node_coords_for_ndm``, with ``ndm`` threaded through ~7
function signatures to reach it) and again, last, at every text/live emitter
(``trim_coords_to_ndm``). The emitter copy always won, so the build-layer
copy was dead weight that still cost the threading.

What the build layer keeps is the *coercion*, which is load-bearing for a
different reason: the broker hands out numpy scalars. The deck formatters
used to render those via ``repr`` — under numpy 2.x the literal text
``np.float64(0.0)``, a corrupt deck — and since #1336 they render them as
plain numbers. The coercion still matters twice over: the emitters' ``node``
fast path takes exact ``float`` only (one f-string per mesh node, the
emit-cost gate's band), and the live route hands values straight to
openseespy, which accepts ``np.float64`` but rejects ``np.float32``.
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


def test_a_numpy_coordinate_renders_plain_and_the_coercion_keeps_the_fast_path():
    """The formatter no longer corrupts a numpy coordinate (#1336), so the
    deck no longer depends on the coercion for correctness. It still
    depends on it for the node fast path, which takes exact ``float``: a
    numpy coordinate falls through to the generic per-token route."""
    from apeGmsh.opensees.emitter.tcl import TclEmitter, _fmt_value

    assert _fmt_value(np.float64(1.5)) == "1.5"
    assert _fmt_value(node_coords_as_floats(np.array([1.5, 0.0, 0.0]))[0]) == "1.5"
    assert np.float64(1.5).__class__ is not float   # misses the fast path
    em = TclEmitter()
    em.node(1, *np.array([1.5, 0.0, 0.0]))
    em.node(2, *node_coords_as_floats(np.array([1.5, 0.0, 0.0])))
    nodes = [ln for ln in em.lines() if ln.startswith("node ")]
    assert nodes == ["node 1 1.5 0.0 0.0", "node 2 1.5 0.0 0.0"]


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
