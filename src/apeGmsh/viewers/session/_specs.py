"""Slot record → ``DiagramSpec`` — the S1 mapping table (ADR 0098 §4).

The session IR stores opaque broker tokens (S0 decision 3); this module
interprets them against the data and the existing per-kind vocabulary:

* ``Contour(quantity, averaging)`` → the ``contour`` kind.
  ``averaging`` maps from the ADR vocabulary to the existing
  ``ContourStyle`` token (``"unaveraged"`` → ``"discrete"``); the
  nodal-vs-gauss topology is resolved from where the quantity is
  actually recorded in the realized stage. A quantity recorded at both
  levels resolves to ``nodes`` (the primary field family — Gauss values
  of the same field remain reachable through the ``gauss`` slot).

S1-B extends this table with the remaining six categories (vector's
``vector_glyph`` / ``principal_glyph`` resolver, line's
``LineForceStyle`` axis machinery, …).
"""
from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from ..diagrams._base import DiagramSpec
from ..diagrams._kinds import kind_def
from ..diagrams._selectors import normalize as normalize_selector

if TYPE_CHECKING:
    from apeGmsh.results.Results import Results
    from apeGmsh.results.session import Contour


def resolve_contour_topology(
    quantity: str, results: "Results", stage_id: str,
) -> str:
    """``"nodes"`` or ``"gauss"`` — where ``quantity`` is recorded.

    Loud on a quantity the realized stage does not record at either
    level: a still of a slot nobody recorded would be a lie, not an
    empty picture.
    """
    components = results.inspect.components(stage=stage_id)
    nodal = set(components.get("nodes", ()))
    gauss = set(components.get("gauss", ()))
    if quantity in nodal:
        return "nodes"
    if quantity in gauss:
        return "gauss"
    raise ValueError(
        f"Contour quantity {quantity!r} is not recorded in stage "
        f"{stage_id!r} (nodal: {sorted(nodal)}; gauss: {sorted(gauss)}). "
        f"Use results.inspect.diagnose({quantity!r}) for the routing "
        f"detail."
    )


def contour_spec(
    record: "Contour", results: "Results", stage_id: str,
) -> DiagramSpec:
    """The contour slot as a director-free ``DiagramSpec``.

    ``stage_id`` is pinned on the spec so the diagram's reads scope
    without a director (``_stage_pin_resolver`` stays unset).
    """
    topology = resolve_contour_topology(record.quantity, results, stage_id)
    averaging = (
        "discrete" if record.averaging == "unaveraged" else "averaged"
    )
    kdef = kind_def("contour")
    assert kdef is not None  # shipped kind — the registry always has it
    style = replace(
        kdef.make_default_style(record.quantity),
        topology=topology,
        averaging=averaging,
    )
    return DiagramSpec(
        kind="contour",
        selector=normalize_selector(component=record.quantity),
        style=style,
        stage_id=stage_id,
    )


__all__ = ["contour_spec", "resolve_contour_topology"]
