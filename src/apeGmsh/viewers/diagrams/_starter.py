"""Starter actions — programmatic default-diagram creation (R1.4).

The first-run empty-state HUD offers one-click starter actions when
the viewer boots with zero diagrams. The creation chain here is the
live GUI path (``DiagramSettingsTab._build_diagram`` +
``_on_creation_apply``) condensed to its programmatic core, kept as
module-level functions so it stays testable with just a
:class:`ResultsDirector` — no Qt, no viewer.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ._base import Diagram
    from ._director import ResultsDirector


def default_contour_component(
    director: "ResultsDirector",
) -> Optional[str]:
    """The component the starter contour should render.

    Mirrors the Add Diagram dialog's default heuristic
    (``_add_diagram_dialog``: nodes-leaning kinds prefer
    ``displacement_z``): a recorded displacement axis when available,
    else the first available nodal component in canonical-axis order.
    ``None`` when the file has no nodal components at all.
    """
    from ._kind_catalog import _component_sort_key, _union_across_stages

    nodal = _union_across_stages(director, "nodes")
    if not nodal:
        return None
    for comp in ("displacement_z", "displacement_x", "displacement_y"):
        if comp in nodal:
            return comp
    return sorted(nodal, key=_component_sort_key)[0]


def add_default_contour(director: "ResultsDirector") -> "Diagram":
    """Create + attach a contour on the default component.

    Builds the Diagram, tags composition membership FIRST (the
    registry's ``scene_resolver`` routes ``attach()`` through the
    owning geometry — same ordering as
    ``ResultsDirector.duplicate_geometry``), then ``registry.add`` and
    ``DIAGRAM_ATTACHED``.

    Raises
    ------
    ValueError
        When the file has no nodal components to contour.
    Exception
        Propagated from ``registry.add`` (e.g. ``NoDataError``) after
        rolling the membership tag back out — loud on purpose so the
        caller can surface it via status.
    """
    from ._base import DiagramSpec
    from ._dispatch import DIAGRAM_ATTACHED
    from ._kinds import kind_def
    from ._selectors import normalize as normalize_selector

    component = default_contour_component(director)
    if component is None:
        raise ValueError(
            "No nodal components recorded — nothing to contour."
        )
    kdef = kind_def("contour")
    spec = DiagramSpec(
        kind="contour",
        selector=normalize_selector(component=component),
        style=kdef.make_default_style(component),
        stage_id=director.stage_id,
    )
    diagram = kdef.diagram_class(spec, director.results)

    geoms = director.geometries
    geom = geoms.active or geoms.geometries[0]
    comp_mgr = geom.compositions
    if comp_mgr.active_accepts_layers:
        comp = comp_mgr.active
        created = None
    else:
        # Fresh boot: the geometry has no compositions yet.
        comp = comp_mgr.add(name="Diagram", make_active=True)
        created = comp
    comp_mgr.add_layer(comp.id, diagram)
    try:
        director.registry.add(diagram)
    except Exception:
        # registry.add rolled the diagram out of its list; drop the
        # membership tag too so the outline never shows a ghost layer
        # — and if we created the composition for this attempt, drop
        # it as well, or a failed click leaves a permanent empty
        # "Diagram" group in the outline.
        comp_mgr.remove_layer(diagram)
        if created is not None:
            comp_mgr.remove(created.id)
        raise
    director.dispatcher.fire(DIAGRAM_ATTACHED, layer=diagram)
    return diagram


__all__ = ["add_default_contour", "default_contour_component"]
