"""Plot-series resolution — a plot pane's live queries (ADR 0098 §6).

A plot is a pane, not a dock of a contour. Its series are **live
queries against Results at the cursor**, so this module resolves them
to arrays and nothing else: no matplotlib, no Qt. The matplotlib still
(``results.plot.history``) and the S4 Qt plot pane draw the same
numbers — the direct-equality oracle for this resolver is that a
``history`` series equals what ``results.plot.history`` plots.

v1 sources are the concrete ones the §8 selection produces (a node or
a Gauss point); ``label`` / ``physical_group`` sources need an
aggregation rule (one curve per member vs one aggregate curve) that
belongs to the S4-2 outline slice, so they refuse loudly here rather
than guessing. ``path`` / ``xy`` plot kinds likewise wait for S4-2.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from apeGmsh.results.Results import Results
    from apeGmsh.results.session import (
        Instant, PlotSeries, PlotView, ResultsSession,
    )


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
            f"S1-B resolves 'history' plots; {plot.kind!r} plots land "
            f"with the S4-2 plot pane (path is evaluated at an instant, "
            f"xy needs a second axis source — both are that slice's "
            f"decision)."
        )
    cursor = session.effective_instant(plot)
    stage_id = cursor.stage if cursor is not None else _last_stage(results)
    scoped = results.stage(stage_id)
    series = tuple(
        _resolve_series(s, scoped, stage_id) for s in plot.series
    )
    return RealizedPlot(
        pane_id=plot.id, kind=plot.kind, series=series, cursor=cursor,
    )


def _resolve_series(
    series: "PlotSeries", scoped: "Results", stage_id: str,
) -> RealizedSeries:
    source = series.source
    if source.kind == "node":
        return _node_series(series, scoped, stage_id)
    if source.kind == "gauss":
        return _gauss_series(series, scoped, stage_id)
    raise NotImplementedError(
        f"Plot source kind {source.kind!r} needs an aggregation rule "
        f"(one curve per member, or one aggregate curve) — that is the "
        f"S4-2 outline slice's decision. Use node or gauss sources, "
        f"which is what a selection produces (§8)."
    )


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


def _last_stage(results: "Results") -> str:
    stages = list(results.stages)
    if not stages:
        raise RuntimeError("session realize requires at least one stage.")
    return stages[-1].id


__all__ = ["RealizedPlot", "RealizedSeries", "realize_plot"]
