"""S2 — box reconciliation lock (mesh ↔ results parity).

Phase S2 of the selection-unification work (see
``docs/plans/selection-unification.md`` §6 S2, ratified decision R4)
moved ``mesh/_mesh_filters.py`` ``nodes_in_box`` / ``elements_in_box``
from CLOSED-CLOSED to HALF-OPEN on the upper bound, to match the
already-canonical ``results/_composites.py`` ``_node_ids_in_box`` /
``_element_ids_in_box`` (``box_min <= xyz < box_max`` per axis).

This module LOCKS that reconciliation:

* Same coords + same box + a point exactly on the upper face ``hi``:
  ``_mesh_filters.nodes_in_box`` (default) membership ==
  ``results._composites._node_ids_in_box`` membership.
  This identity was conceptually RED on ``main`` pre-S2 (mesh CLOSED
  included the on-face point, results HALF-OPEN excluded it) and MUST
  be GREEN now.
* ``inclusive=True`` reproduces the OLD closed result (the escape
  hatch for point-family callers that need on-face inclusion).

Pure apeGmsh + numpy — no openseespy, no gmsh session, no results
file (the resolver only reads ``fem.nodes.coords`` / ``fem.nodes.ids``
and the centroid stub). Deterministic.
"""
from __future__ import annotations

import types

import numpy as np

from apeGmsh.mesh import _mesh_filters as _flt
from apeGmsh.results import _composites as _rc


# ---------------------------------------------------------------------
# A point exactly on the upper face is the discriminating case.
# ---------------------------------------------------------------------

_BOX = (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
_LO = (0.0, 0.0, 0.0)
_HI = (1.0, 1.0, 1.0)

# lower corner (kept), strict interior (kept), exactly-on-hi (the
# discriminator), one axis on hi (also on the upper face).
_COORDS = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
        [1.0, 1.0, 1.0],
        [1.0, 0.5, 0.5],
    ],
    dtype=np.float64,
)
_IDS = np.array([10, 11, 12, 13], dtype=np.int64)


def _results_membership(coords: np.ndarray, ids: np.ndarray) -> set[int]:
    """Run the canonical results box resolver, return the kept id set."""
    fem = types.SimpleNamespace()
    fem.nodes = types.SimpleNamespace(
        ids=np.asarray(ids, dtype=np.int64),
        coords=np.asarray(coords, dtype=np.float64),
    )
    out = _rc._node_ids_in_box(fem, _LO, _HI)
    return set(int(x) for x in out.tolist())


def _mesh_membership(coords: np.ndarray, ids: np.ndarray,
                     *, inclusive: bool = False) -> set[int]:
    """Run _mesh_filters.nodes_in_box, return the kept id set."""
    mask = _flt.nodes_in_box(coords, _BOX, inclusive=inclusive)
    return set(int(x) for x in np.asarray(ids)[mask].tolist())


# ---------------------------------------------------------------------
# The reconciliation, locked.
# ---------------------------------------------------------------------

def test_on_hi_point_mesh_default_excluded_like_main_pre_s2_red():
    """The exactly-on-hi point: mesh default now EXCLUDES it.

    Pre-S2 this was True (closed). It is the single fact S2 flipped.
    """
    on_hi = np.array([[1.0, 1.0, 1.0]], dtype=np.float64)
    assert bool(_flt.nodes_in_box(on_hi, _BOX)[0]) is False
    # results was already False (the canonical half-open reference).
    fem = types.SimpleNamespace()
    fem.nodes = types.SimpleNamespace(
        ids=np.array([1], dtype=np.int64), coords=on_hi,
    )
    assert _rc._node_ids_in_box(fem, _LO, _HI).size == 0


def test_mesh_default_equals_results_on_shared_hi_face():
    """mesh nodes_in_box (default) == results _node_ids_in_box.

    Conceptually RED on ``main`` pre-S2 (the divergence the whole
    plan exists to remove); GREEN after S2. The on-``hi`` rows {12,13}
    are excluded by BOTH; only the strictly-below rows {10,11} survive.
    """
    mesh_set = _mesh_membership(_COORDS, _IDS)
    results_set = _results_membership(_COORDS, _IDS)
    assert mesh_set == results_set                  # PARITY (was RED)
    assert mesh_set == {10, 11}                      # explicit anchor
    assert 12 not in mesh_set and 13 not in mesh_set


def test_inclusive_true_reproduces_old_closed_result():
    """The escape hatch: inclusive=True == the pre-S2 closed answer.

    Closed keeps every point, including the two on the upper face,
    and therefore DIVERGES from the half-open results reference —
    exactly the pre-S2 mesh behavior, now opt-in.
    """
    closed_set = _mesh_membership(_COORDS, _IDS, inclusive=True)
    results_set = _results_membership(_COORDS, _IDS)
    assert closed_set == {10, 11, 12, 13}            # old closed answer
    assert closed_set != results_set                 # the old divergence
    # And it is strictly a superset of the new half-open default.
    assert _mesh_membership(_COORDS, _IDS) < closed_set


def test_elements_in_box_inherits_parity_and_escape():
    """elements_in_box delegates to nodes_in_box -> same reconciliation.

    Centroid exactly on the upper face: excluded by default (parity
    with results' centroid box), included under inclusive=True.
    """
    cent = np.array(
        [[0.5, 0.5, 0.5], [1.0, 1.0, 1.0]], dtype=np.float64
    )
    # default half-open: on-face centroid dropped.
    assert _flt.elements_in_box(cent, _BOX).tolist() == [True, False]
    # results element box (canonical reference) on the same centroids.
    fem = types.SimpleNamespace()
    fem.nodes = types.SimpleNamespace(
        ids=np.array([1, 2], dtype=np.int64),
        coords=cent,
    )
    grp = (
        np.array([100, 200], dtype=np.int64),
        np.array([[1], [2]], dtype=np.int64),
    )
    # P3-R / §6.3 M-STOP-3 + disposition 4: ``_element_ids_in_box``
    # funnels through ``_element_centroids`` which now iterates
    # ``fem.elements._groups.values()`` directly — mirror ``grp``.
    fem.elements = types.SimpleNamespace(
        types=[types.SimpleNamespace(name="P1")],
        resolve=lambda *, element_type: grp,
        _groups={0: types.SimpleNamespace(
            ids=grp[0], connectivity=grp[1], type_name="P1",
        )},
    )
    res_ids = _rc._element_ids_in_box(fem, _LO, _HI).tolist()
    assert res_ids == [100]                          # on-face excluded
    # mesh default keeps element 100 only -> parity with results.
    mesh_keep = [
        int(e) for e, m in zip(
            grp[0], _flt.elements_in_box(cent, _BOX)
        ) if m
    ]
    assert mesh_keep == res_ids                       # PARITY
    # inclusive=True restores the on-face centroid (old closed).
    assert _flt.elements_in_box(
        cent, _BOX, inclusive=True
    ).tolist() == [True, True]
