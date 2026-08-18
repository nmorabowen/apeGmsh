"""Plot-series resolution — a plot pane's live queries (ADR 0098 §6).

A plot is a pane, not a dock of a contour. Its series are **live
queries against Results at the cursor**, so this module resolves them
to arrays and nothing else: no matplotlib, no Qt. The matplotlib still
(``results.plot.history``) and the S4 Qt plot pane draw the same
numbers — the direct-equality oracle for this resolver is that a
``history`` series equals what ``results.plot.history`` plots.

Two rules settled in S4-2:

* **A membership source is one curve PER MEMBER** (plan decision 10b).
  ``add_plot_from_selection`` already expands a selection into one
  ``PlotSeries`` per node / Gauss point (§6: "select → New plot COPIES
  the membership"; "Several series on one chart are one plot view"), so
  a ``label`` / ``physical_group`` source that aggregated instead would
  make the same chart mean two different things depending on how it was
  authored. An aggregate curve also needs a reducer token (mean? max?
  sum?) that the IR has no field for, and defaulting to one silently is
  worse than not offering it. The expansion is ONE slab read per
  source, not one per member — ``pg=`` / ``label=`` are first-class
  filters on the query layer and a slab is already (T, members).
* **A plot has a curve cap** (:data:`MAX_SERIES`). It lives here
  because this is the one place both authoring paths funnel through,
  and because outline select-all (§8 writer 3) made
  ``add_plot_from_selection`` able to hand this module 100k sources for
  the first time. A chart with hundreds of curves is unreadable anyway;
  what the cap really prevents is reading them.

``path`` / ``xy`` plot kinds stay refusals (plan decision 10): neither
has an authoring surface in v1, and ``xy`` has nowhere in the IR to put
a second axis source.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

from ._gauss_addr import gp_index_within_element

if TYPE_CHECKING:
    from apeGmsh.results.Results import Results
    from apeGmsh.results.session import (
        Instant, PlotSeries, PlotView, ResultsSession,
    )


#: Most curves one plot pane resolves.
#:
#: Sized against what a named physical group actually holds — a base,
#: a floor, one wall face is hundreds of nodes, and §9's select-all
#: exists to plot exactly those — while still refusing the gesture
#: that motivates the cap: "select all nodes" on the whole model then
#: "New plot from selection", which arrives as one concrete source
#: PER NODE (measured 0.3 ms each, so 100k nodes is ~30 s of reads
#: for a chart of 100k overlaid lines). Past a few hundred curves the
#: spaghetti carries no information anyway. The refusal has to come
#: BEFORE the reads, not after them.
MAX_SERIES = 256


@dataclass(frozen=True, eq=False)
class RealizedSeries:
    """One resolved curve: the arrays a client draws.

    ``time`` and ``values`` are ``(T,)`` and row-aligned — the same two
    arrays ``results.plot.history`` passes to matplotlib.
    """

    label: str
    quantity: str
    time: np.ndarray
    values: np.ndarray


@dataclass(frozen=True, eq=False)
class RealizedPlot:
    """One plot pane, resolved (§6). ``cursor`` is the instant the
    §7 link resolved for this pane — the whole record is in ``series``;
    the cursor is where the playhead sits on it."""

    pane_id: str
    kind: str
    series: tuple[RealizedSeries, ...]
    cursor: "Optional[Instant]"


def realize_plot(
    session: "ResultsSession", plot: "PlotView",
) -> RealizedPlot:
    """Resolve every series of ``plot`` against the bound Results."""
    results = session.results
    if results is None:
        raise RuntimeError(
            "This session has no Results bound — construct it via "
            "results.session() to realize or render."
        )
    if plot.kind != "history":
        raise NotImplementedError(
            f"This resolver draws 'history' plots; {plot.kind!r} plots "
            f"have no v1 authoring surface. A 'path' plot needs an "
            f"ORDERED source and an arc-length abscissa — §8's "
            f"selection is a SET, and so is the outline's select-all — "
            f"and an 'xy' plot needs a second axis source, which "
            f"PlotSeries(source, quantity) has nowhere to hold. Both "
            f"are IR widenings, not resolver work."
        )
    # Cheap gate first: with N concrete sources the reads are N slabs,
    # and "select all nodes" → "New plot from selection" makes N as
    # large as the model (§8 writer 3). Refuse before reading anything.
    _require_under_cap(len(plot.series), "source")
    cursor = session.effective_instant(plot)
    stage_id = cursor.stage if cursor is not None else _last_stage(results)
    scoped = results.stage(stage_id)
    series: list[RealizedSeries] = []
    for spec in plot.series:
        series.extend(_resolve_series(spec, scoped, stage_id))
        _require_under_cap(len(series), "curve")
    return RealizedPlot(
        pane_id=plot.id, kind=plot.kind, series=tuple(series), cursor=cursor,
    )


def _require_under_cap(count: int, noun: str) -> None:
    if count <= MAX_SERIES:
        return
    raise ValueError(
        f"This plot resolves {count} {noun}s; the cap is "
        f"{MAX_SERIES}. A membership source is one curve per member "
        f"(ADR 0098 §6), so a whole physical group or a select-all "
        f"lands here as hundreds of reads for a chart no one can "
        f"read. Narrow the selection, or name the nodes / Gauss "
        f"points you mean."
    )


def _resolve_series(
    series: "PlotSeries", scoped: "Results", stage_id: str,
) -> "list[RealizedSeries]":
    """One spec → the curves it means. Concrete sources give exactly
    one; a membership source gives one per member (decision 10b)."""
    source = series.source
    if source.kind == "node":
        return [_node_series(series, scoped, stage_id)]
    if source.kind == "gauss":
        return [_gauss_series(series, scoped, stage_id)]
    return _membership_series(series, scoped, stage_id)


def _node_series(
    series: "PlotSeries", scoped: "Results", stage_id: str,
) -> RealizedSeries:
    """One node's whole record — the same query
    ``results.plot.history(node=, component=)`` runs."""
    node_id = int(series.source.key)  # type: ignore[arg-type]
    slab = scoped.nodes.get(ids=[node_id], component=series.quantity)
    values = np.asarray(slab.values)
    if values.size == 0 or values.shape[1] == 0:
        raise ValueError(
            f"No data for component {series.quantity!r} at node "
            f"{node_id} in stage {stage_id!r}."
        )
    if values.shape[1] != 1:
        # The same guard results.plot.history keeps before indexing
        # column 0 (_plot.py:465-468): more than one column back for a
        # single requested id means the read did not mean this node,
        # and drawing column 0 would silently plot a different one.
        raise ValueError(
            f"Expected one node's record for {series.quantity!r} at "
            f"node {node_id}; the read returned {values.shape[1]} "
            f"columns."
        )
    return RealizedSeries(
        label=f"node {node_id} — {series.quantity}",
        quantity=series.quantity,
        time=np.asarray(slab.time),
        values=values[:, 0],
    )


def _gauss_series(
    series: "PlotSeries", scoped: "Results", stage_id: str,
) -> RealizedSeries:
    """One integration point's whole record.

    The slab carries every GP row of the element; ``gp_index`` selects
    the row within that element, in the slab's own row order — the same
    ordering the §8 gauss selection target encodes.
    """
    element_id, gp_index = series.source.key  # type: ignore[misc]
    element_id, gp_index = int(element_id), int(gp_index)
    slab = scoped.elements.gauss.get(
        ids=[element_id], component=series.quantity,
    )
    values = np.asarray(slab.values)
    if values.size == 0:
        raise ValueError(
            f"No data for component {series.quantity!r} at element "
            f"{element_id} in stage {stage_id!r}."
        )
    rows = np.where(
        np.asarray(slab.element_index, dtype=np.int64) == element_id
    )[0]
    if gp_index >= rows.size:
        raise ValueError(
            f"Element {element_id} has {rows.size} integration point(s) "
            f"recorded for {series.quantity!r}; gp_index={gp_index} is "
            f"out of range."
        )
    return RealizedSeries(
        label=f"element {element_id} gp {gp_index} — {series.quantity}",
        quantity=series.quantity,
        time=np.asarray(slab.time),
        values=values[:, int(rows[gp_index])],
    )


def _membership_series(
    series: "PlotSeries", scoped: "Results", stage_id: str,
) -> "list[RealizedSeries]":
    """A ``label`` / ``physical_group`` source → one curve per member.

    ONE slab read, not one per member: ``pg=`` / ``label=`` are
    first-class filters on the query layer and a slab already comes
    back as ``(T, members)``. The membership is probed at a single
    step first, so an over-cap group is refused BEFORE its whole
    record is read — that is the difference between a loud refusal and
    an out-of-memory read on a select-all-sized group.

    Which topology the members are depends on the QUANTITY, not the
    source: a nodal component means the group's nodes, a Gauss
    component means its integration points. Nothing guesses — the
    stage's recorded vocabulary decides, and an ambiguous or unknown
    token refuses.
    """
    source = series.source
    key = {"physical_group": "pg", "label": "label"}[source.kind]
    where = {key: str(source.key)}
    topology = _membership_topology(series.quantity, scoped, stage_id)
    if topology == "nodes":
        return _member_node_series(series, scoped, stage_id, where)
    return _member_gauss_series(series, scoped, stage_id, where)


def _membership_topology(
    quantity: str, scoped: "Results", stage_id: str,
) -> str:
    """``"nodes"`` | ``"gauss"`` for a membership source's quantity."""
    try:
        recorded = scoped.inspect.components(stage=stage_id)
    except Exception:
        recorded = {}
    nodal = quantity in set(recorded.get("nodes", ()))
    gauss = quantity in set(recorded.get("gauss", ()))
    if nodal and not gauss:
        return "nodes"
    if gauss and not nodal:
        return "gauss"
    if nodal and gauss:
        raise ValueError(
            f"Component {quantity!r} is recorded at BOTH the node and "
            f"the Gauss level in stage {stage_id!r}, so a membership "
            f"source cannot say which one it means. Use node or gauss "
            f"sources to say it explicitly."
        )
    from apeGmsh.results import _derived

    if _derived.is_derived(quantity) or _derived.is_shell_derived(quantity):
        # Derived scalars (von Mises, principals, shell resultants) are
        # computed on read and never appear in the recorded vocabulary,
        # but they ARE Gauss quantities — refusing them here would make
        # a membership source weaker than the concrete gauss source
        # that resolves them today.
        return "gauss"
    raise ValueError(
        f"Component {quantity!r} is recorded at neither the node nor "
        f"the Gauss level in stage {stage_id!r} (nodes: "
        f"{sorted(recorded.get('nodes', ()))}; gauss: "
        f"{sorted(recorded.get('gauss', ()))})."
    )


def _member_node_series(
    series: "PlotSeries", scoped: "Results", stage_id: str, where: dict,
) -> "list[RealizedSeries]":
    probe = scoped.nodes.get(component=series.quantity, time=[0], **where)
    node_ids = np.asarray(probe.node_ids, dtype=np.int64)
    _require_membership(node_ids.size, series, stage_id)
    _require_under_cap(int(node_ids.size), "curve")
    slab = scoped.nodes.get(component=series.quantity, **where)
    values = np.asarray(slab.values)
    time = np.asarray(slab.time)
    return [
        RealizedSeries(
            label=f"node {int(node_id)} — {series.quantity}",
            quantity=series.quantity,
            time=time,
            values=values[:, column],
        )
        for column, node_id in enumerate(np.asarray(slab.node_ids))
    ]


def _member_gauss_series(
    series: "PlotSeries", scoped: "Results", stage_id: str, where: dict,
) -> "list[RealizedSeries]":
    probe = scoped.elements.gauss.get(
        component=series.quantity, time=[0], **where,
    )
    _require_membership(int(np.asarray(probe.element_index).size), series,
                        stage_id)
    _require_under_cap(int(np.asarray(probe.element_index).size), "curve")
    slab = scoped.elements.gauss.get(component=series.quantity, **where)
    values = np.asarray(slab.values)
    time = np.asarray(slab.time)
    element_index = np.asarray(slab.element_index, dtype=np.int64)
    gp_indices = gp_index_within_element(element_index)
    return [
        RealizedSeries(
            label=(
                f"element {int(element_index[column])} gp "
                f"{int(gp_indices[column])} — {series.quantity}"
            ),
            quantity=series.quantity,
            time=time,
            values=values[:, column],
        )
        for column in range(element_index.size)
    ]


def _require_membership(
    count: int, series: "PlotSeries", stage_id: str,
) -> None:
    """An empty membership is a wrong plot, not an empty one — the same
    reading ``_scope.resolve_scope`` gives an empty cell set."""
    if count:
        return
    raise ValueError(
        f"{series.source.kind} {series.source.key!r} has no members "
        f"recording {series.quantity!r} in stage {stage_id!r} — a plot "
        f"of nothing is a wrong query, not an empty chart."
    )


def _last_stage(results: "Results") -> str:
    stages = list(results.stages)
    if not stages:
        raise RuntimeError("session realize requires at least one stage.")
    return stages[-1].id


__all__ = ["RealizedPlot", "RealizedSeries", "realize_plot"]
