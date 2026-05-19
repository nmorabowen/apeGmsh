"""P3-R — the SC-11 reviewed pin-FLIP: ``_mesh_filters`` silent-row-0
→ fail-loud (selection-unification v2).

This file was the P0-C spatial characterization battery; through P3-K it
pinned the **silent row-0 centroid corruption** on HEAD (the "wrong on
purpose" pin). P3-R is the reviewed same-commit production+assertion
diff (``selection-unification-v2.md`` §6.2 SC-11,
``selection-unification-v2-p3r-callers.md`` §5):

* **production half (P3-R / G6):** ``mesh/_mesh_filters.py:159``
  (``element_centroids``) and ``:215`` (``elements_on_plane``) no longer
  do ``node_id_to_idx.get(int(nid), 0)`` (silent row-0 substitution →
  corrupted result); they now raise ``KeyError`` fail-loud on a
  connectivity node id absent from the node set.
* **assertion half (this file):** the pin is FLIPPED — what was
  "no raise + corrupted output" is now "raises ``KeyError``". The
  byte-identical behaviour for *valid* input (every node present) is
  pinned too, so the flip is provably surgical (only the
  previously-silently-corrupted missing-node path changed).

Pin 1 (the legacy ``g.mesh_selection.add_nodes`` axis-aligned
observation) is **retired**: that surface is removed in P3-R. P3-S adds
the new-idiom spatial successors via
``g.mesh_selection.select(...).on_plane(point, normal, tol=...)`` /
element-level ``.coords`` against the unified ``_kernel/spatial.py``.

No ``openseespy`` / ``gmsh`` / ``apeGmsh`` session needed — pure
``_mesh_filters`` + numpy (curated no-openseespy CI gate); the
crafted-array characterization (a node id deliberately absent from the
id→row map) exercises the exact ``:159`` / ``:215`` code path directly.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from apeGmsh.mesh import _mesh_filters as _flt


# Shared crafted scene: id→row map missing id 777; element 1 references
# it (the previously row-0-contaminated case).  row 0 is deliberately
# (0,0,0) so the OLD silent blend was unmistakable.
_NODE_COORDS = np.array(
    [
        [0.0, 0.0, 0.0],     # row 0  (id 10) — the OLD silent fallback
        [10.0, 10.0, 10.0],  # row 1  (id 20)
        [20.0, 20.0, 20.0],  # row 2  (id 30)
    ],
    dtype=np.float64,
)
_NODE_ID_TO_IDX = {10: 0, 20: 1, 30: 2}
_CONN_WITH_MISSING = np.array(
    [
        [20, 30],    # both present
        [777, 30],   # 777 ABSENT from the map — the flipped path
    ],
    dtype=np.int64,
)
_CONN_ALL_PRESENT = np.array(
    [
        [20, 30],    # -> centroid (15,15,15)
        [10, 30],    # -> centroid (10,10,10)
    ],
    dtype=np.int64,
)


# =====================================================================
# Pin 2 — FLIPPED: element_centroids now FAILS LOUD (was silent row-0)
# =====================================================================

def test_pin_meshfilters_element_centroids_fail_loud():
    """``_mesh_filters.element_centroids`` raises on a missing node id.

    REVIEWED PIN-FLIP (SC-11). On HEAD-through-P3-K this returned a
    silently row-0-contaminated centroid with NO raise / NO warning.
    P3-R (``mesh/_mesh_filters.py:159``) replaces the
    ``node_id_to_idx.get(int(nid), 0)`` fallback with a fail-loud
    ``KeyError`` — the missing-node path now refuses to compute a
    corrupted centroid.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(KeyError) as ei:
            _flt.element_centroids(
                _CONN_WITH_MISSING, _NODE_ID_TO_IDX, _NODE_COORDS
            )
    msg = str(ei.value)
    assert "777" in msg, msg
    assert "not in the node set" in msg, msg
    assert "fail loud" in msg, msg
    # The flip is loud, not a deprecation warning.
    assert caught == [] or all(
        not issubclass(w.category, Warning) for w in caught
    )


def test_pin_meshfilters_element_centroids_valid_input_unchanged():
    """Surgical-flip proof: VALID input (every node present) is
    byte-identical to the pre-flip behaviour — only the
    previously-silently-corrupted missing-node path changed.
    """
    out = _flt.element_centroids(
        _CONN_ALL_PRESENT, _NODE_ID_TO_IDX, _NODE_COORDS
    )
    expected = np.array(
        [
            [15.0, 15.0, 15.0],   # mean of id20 (10,10,10) & id30 (20,20,20)
            [10.0, 10.0, 10.0],   # mean of id10 (0,0,0)   & id30 (20,20,20)
        ],
        dtype=np.float64,
    )
    np.testing.assert_array_equal(out, expected)


# =====================================================================
# :215 — FLIPPED: elements_on_plane now FAILS LOUD (was silent row-0)
# =====================================================================

def test_pin_meshfilters_elements_on_plane_fail_loud():
    """``_mesh_filters.elements_on_plane`` raises on a missing node id.

    The second SC-11 site (``mesh/_mesh_filters.py:215``): the same
    ``node_id_to_idx.get(int(nid), 0)`` silent row-0 substitution is
    flipped to a fail-loud ``KeyError`` (refusing a corrupted plane
    filter).
    """
    with pytest.raises(KeyError) as ei:
        _flt.elements_on_plane(
            _CONN_WITH_MISSING, _NODE_ID_TO_IDX, _NODE_COORDS,
            "z", 0.0, 1e-6,
        )
    msg = str(ei.value)
    assert "777" in msg, msg
    assert "not in the node set" in msg, msg
    assert "fail loud" in msg, msg


def test_pin_meshfilters_elements_on_plane_valid_input_unchanged():
    """Surgical-flip proof for ``:215``: VALID input is unchanged.

    All nodes present → the boolean ``ALL nodes on plane`` mask is the
    same array the pre-flip code produced.  Plane z=0.0: only element 0
    (ids 20,30 at z=10,20) and element 1 (ids 10,30 at z=0,20) — neither
    has *all* nodes on z=0, so both are False.
    """
    mask = _flt.elements_on_plane(
        _CONN_ALL_PRESENT, _NODE_ID_TO_IDX, _NODE_COORDS,
        "z", 0.0, 1e-6,
    )
    np.testing.assert_array_equal(mask, np.array([False, False]))
