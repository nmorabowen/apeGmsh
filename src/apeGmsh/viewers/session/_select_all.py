"""Outline select-all — the §8 selection's FOURTH writer (S4-2).

ADR 0098 §8 names four writers over one homogeneous set: click,
window, **outline "Select all nodes" / "Select all Gauss"**, and code.
§9 puts the third on the model-composition tree: "Check = this mesh
view's *scope*. Action = **select all nodes or Gauss** of that
category." Two knobs, one tree — scope is what is DRAWN, selection is
what is PLOTTED.

This module is only the resolution: a composition axis plus its
checked names → the node ids or the ``(element_id, gp_index)`` pairs
of that category. The write itself goes through
``session.selection.set_nodes`` / ``set_gauss`` — the same surface a
click uses, over ADR 0045's ONE store (INV-5), so the last-writer XOR
law applies unchanged and one select-all is one undoable ``SET``.

Three things it deliberately does NOT do:

* **It does not touch scope.** Selecting every node of "Web" leaves
  the view drawing whatever it drew. §8: "Scope (what is drawn) and
  selection (what is plotted) are different knobs."
* **It does not require the glyphs to be on.** §8: "Query-from-outline
  may fill the set with glyphs off; glyphs on is how you see it." The
  reconciler's selection term reads the set only when a glyph button
  is on, so a select-all with both off costs the panes nothing.
* **It does not resolve the ``materials`` axis.** No element→material
  index is published (see :mod:`._scope`), and the outline renders
  that axis disabled for the same reason.

§8 also names "or element" as a select-all subject. There is no
element row in the outline and no element pick in this window (an
element is a membership query, not a hit), so the v1 UI writer covers
the three axes the tree has; :func:`gauss_pairs` is the element case,
callable with any element ids a script already holds.

Cost: node membership is pure model topology and reads no results at
all. Gauss membership must ask the recorder how many integration
points each element wrote, which is one single-step probe slab — the
same read :mod:`._realize` makes for its pick targets, addressed
through the same :mod:`._gauss_addr` encoding so the set the outline
writes and the cloud a pane highlights cannot disagree.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Sequence

import numpy as np

from ._gauss_addr import gp_index_within_element, probe_component
from ._scope import resolve_scope

if TYPE_CHECKING:
    from apeGmsh.results.Results import Results
    from apeGmsh.results.session import MeshView, ResultsSession

#: The two things an outline row can select — the §8 kinds, and the
#: same two tokens ``SessionSelection.kind`` reports.
SELECT_ALL_KINDS = ("nodes", "gauss")


def select_all(
    session: "ResultsSession",
    kind: str,
    *,
    axis: str,
    names: "Sequence[str]",
    view: "Optional[MeshView]" = None,
) -> int:
    """Select every node / Gauss point of ``names`` on ``axis``.

    Returns how many targets the set now holds. ``view`` is only
    consulted for the Gauss half, to resolve WHICH stage's integration
    points are meant through the same §7 instant law realize applies —
    a pane and the set it highlights must not disagree about the
    stage. Raises rather than selecting nothing: an empty result means
    the names or the recorder are wrong, and a silently empty set
    reads as "nothing is there".
    """
    if kind not in SELECT_ALL_KINDS:
        raise ValueError(
            f"select_all kind must be one of {SELECT_ALL_KINDS}; got "
            f"{kind!r} (ADR 0098 §8 — selection is nodes XOR Gauss)."
        )
    results = _require_results(session)
    scoped = _membership(results, axis, names)
    if kind == "nodes":
        node_ids = [int(i) for i in scoped.node_ids]
        if not node_ids:
            raise ValueError(
                f"No nodes on {axis} {list(names)} — nothing to select."
            )
        session.selection.set_nodes(node_ids)
        return len(node_ids)
    pairs = gauss_pairs(results, scoped.element_ids, _stage_of(session, view))
    if not pairs:
        raise ValueError(
            f"No integration points recorded for {axis} {list(names)} in "
            f"this stage — nothing to select. (A stage that records no "
            f"Gauss composite has no integration points at all.)"
        )
    session.selection.set_gauss(pairs)
    return len(pairs)


def gauss_pairs(
    results: "Results",
    element_ids: "Sequence[int] | np.ndarray",
    stage_id: str,
) -> "list[tuple[int, int]]":
    """Every ``(element_id, gp_index)`` these elements recorded.

    This is §8's "all Gauss of this hex" as a function: the element is
    the membership query, and the answer is the integration points the
    recorder actually wrote for it — never a count guessed from the
    element type, which would address points that are not there.
    """
    probe = probe_component(results, stage_id)
    ids = np.asarray(element_ids, dtype=np.int64)
    if probe is None or ids.size == 0:
        return []
    slab = results.stage(stage_id).elements.gauss.get(
        ids=ids, component=probe, time=[0],
    )
    element_index = np.asarray(slab.element_index, dtype=np.int64)
    if element_index.size == 0:
        return []
    gp_indices = gp_index_within_element(element_index)
    return [
        (int(e), int(gp)) for e, gp in zip(element_index, gp_indices)
    ]


def counts(
    session: "ResultsSession",
    *,
    axis: str,
    names: "Sequence[str]",
    view: "Optional[MeshView]" = None,
) -> "tuple[int, int]":
    """``(node_count, gauss_count)`` this category would select.

    What the outline labels its two actions with, and how it decides
    whether they can act at all (0087 INV-2). Answers ``(0, 0)``
    rather than raising for anything the model cannot resolve — an
    unresolvable category is a disabled row, not an error dialog.
    """
    try:
        results = _require_results(session)
        scoped = _membership(results, axis, names)
    except Exception:
        return 0, 0
    try:
        gauss = len(gauss_pairs(
            results, scoped.element_ids, _stage_of(session, view),
        ))
    except Exception:
        gauss = 0
    return int(np.asarray(scoped.node_ids).size), gauss


# =====================================================================
# Internals
# =====================================================================


def _membership(results: "Results", axis: str, names: "Sequence[str]"):
    """The one cell set of a composition category (INV-MESH-1).

    The SAME resolver the view's scope uses — checking "Web" and
    selecting all of "Web" must mean the same elements, or the two
    knobs are describing different models.
    """
    from apeGmsh.results.session import Scope

    from ..data import ViewerData

    if results.fem is None:
        raise RuntimeError(
            "Select-all needs a bound FEMData to resolve a composition "
            "axis (construct with model= / model_h5= or call "
            "results.bind)."
        )
    return resolve_scope(
        Scope(axis=axis, names=tuple(names) or None),
        ViewerData.from_fem(results.fem),
    )


def _require_results(session: "ResultsSession") -> "Results":
    results = session.results
    if results is None:
        raise RuntimeError(
            "This session has no Results bound — construct it via "
            "results.session() to select from the outline."
        )
    return results


def _stage_of(
    session: "ResultsSession", view: "Optional[MeshView]",
) -> str:
    """The stage whose integration points a Gauss select-all means.

    Resolved through realize's own §7 instant law when a view is in
    hand, so the outline and the picture agree; the session's last
    stage otherwise (no pane selected — a code caller).
    """
    results = _require_results(session)
    if view is not None:
        from ._realize import _resolve_instant

        stage_id, _step = _resolve_instant(session, view, results)
        return stage_id
    stages = list(results.stages)
    if not stages:
        raise RuntimeError("This Results has no stages to select from.")
    return stages[-1].id


__all__ = ["SELECT_ALL_KINDS", "counts", "gauss_pairs", "select_all"]
