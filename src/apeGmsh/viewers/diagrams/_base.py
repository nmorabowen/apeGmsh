"""Diagram base + DiagramSpec record.

A Diagram is a renderable layer driven by one or more Results slabs.
Subclasses implement the four-method protocol:

* ``attach(plotter, fem)`` — build initial actors at step 0,
  resolve the selector to concrete IDs (once).
* ``update_to_step(step_index)`` — refresh actors for a new step.
  Must mutate VTK arrays in place; never re-add actors.
* ``detach()`` — remove actors and release cached arrays.
* ``settings_widget()`` — return a Qt widget for the diagram's
  per-diagram settings tab.

Phase 0 ships only the base. Concrete diagrams arrive in Phase 1+.

The ``DiagramSpec`` record is the persistable form of a Diagram —
``(kind, style, selector, stage_id)``. The Director can serialize a
list of specs to JSON; reconstructing diagrams from specs lets us
save / restore the active set of diagrams across sessions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from ._selectors import SlabSelector
from ._styles import DiagramStyle

if TYPE_CHECKING:
    from numpy import ndarray
    from apeGmsh.mesh.FEMData import FEMData
    from apeGmsh.results.Results import Results
    from ..scene.fem_scene import FEMSceneData


# =====================================================================
# Errors
# =====================================================================


class NoDataError(RuntimeError):
    """A diagram's ``attach()`` couldn't find any data to render.

    Raised when the slab read for the requested ``(component, stage,
    selector)`` returns empty — typically because the component name
    isn't recorded in this stage, or the selector resolved to elements
    that have no slab coverage. Caught by the Diagrams tab and surfaced
    as a status-bar message; without it the diagram would silently
    add a blank actor and look like a successful operation.
    """


# =====================================================================
# DiagramSpec — persistable form
# =====================================================================

@dataclass(frozen=True)
class DiagramSpec:
    """Persistable record describing a diagram instance.

    Frozen — saved/loaded by the Director, never mutated. The
    ``kind`` field is the registered diagram kind name (e.g.
    ``"contour"``, ``"deformed_shape"``); the registry maps it back
    to the concrete class.
    """
    kind: str
    selector: SlabSelector
    style: DiagramStyle
    stage_id: Optional[str] = None
    visible: bool = True
    label: Optional[str] = None     # user-supplied display name


# =====================================================================
# Diagram base class
# =====================================================================

class Diagram:
    """Base class for all renderable result diagrams.

    Subclasses set the class-level ``kind`` attribute (string,
    registry key) and override ``attach``/``update_to_step``/
    ``detach`` (mandatory) plus ``settings_widget`` (optional —
    returns ``None`` for a blank settings panel).

    Lifecycle::

        diagram = ContourDiagram(spec, results)
        diagram.attach(plotter, fem)         # builds actors
        diagram.update_to_step(0)            # initial render
        ...
        diagram.update_to_step(i)            # on time change
        ...
        diagram.detach()                     # on remove

    Performance contract (locked in Phase 0, enforced thereafter):

    * ``attach`` resolves the selector to concrete IDs **once**;
      subsequent ``update_to_step`` calls reuse them.
    * ``update_to_step`` mutates existing actor scalar arrays in
      place. Re-creating actors per step is forbidden — see the
      perf gates in ``plan_results_viewer.md``.
    * Slabs read for one step are released after the update; the
      diagram does not cache ``(T, …)`` arrays.
    """

    kind: str = ""        # subclasses must set
    topology: str = ""    # subclasses must set — names the Results
                          # composite that owns this diagram's data
                          # ("nodes", "line_stations", "fibers", "layers",
                          # "gauss", "springs"). Read by AddDiagramDialog
                          # to populate the Component combo from
                          # ``available_components()`` on that composite.

    def __init__(self, spec: DiagramSpec, results: "Results") -> None:
        if not self.kind:
            raise TypeError(
                f"{type(self).__name__} must declare a class-level "
                f"'kind' attribute (e.g. kind='contour')."
            )
        if spec.kind != self.kind:
            raise ValueError(
                f"{type(self).__name__} expects kind={self.kind!r}; "
                f"got spec.kind={spec.kind!r}."
            )
        self.spec = spec
        self._results = results

        # Attached state (populated by attach, cleared by detach)
        self._attached: bool = False
        self._plotter: Any = None
        self._fem: "FEMData | None" = None
        self._scene: "FEMSceneData | None" = None
        self._resolved_node_ids: "ndarray | None" = None
        self._resolved_element_ids: "ndarray | None" = None
        self._actors: list[Any] = []
        self._visible: bool = spec.visible

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def selector(self) -> SlabSelector:
        return self.spec.selector

    @property
    def style(self) -> DiagramStyle:
        return self.spec.style

    @property
    def stage_id(self) -> Optional[str]:
        return self.spec.stage_id

    @property
    def is_attached(self) -> bool:
        return self._attached

    @property
    def is_visible(self) -> bool:
        return self._visible

    def display_label(self) -> str:
        """User-facing label for the Diagrams tab list."""
        if self.spec.label:
            return self.spec.label
        return f"{self.kind} — {self.selector.short_label()}"

    # ------------------------------------------------------------------
    # Subclass hooks (override these)
    # ------------------------------------------------------------------

    def attach(
        self,
        plotter: Any,
        fem: "FEMData",
        scene: "FEMSceneData | None" = None,
    ) -> None:
        """Build initial actors. Subclass override.

        The default implementation stores ``plotter`` / ``fem`` /
        ``scene`` and resolves the selector to concrete IDs.
        Subclasses should ``super().attach(plotter, fem, scene)``
        first, then build their actors.

        ``scene`` is the substrate FEMSceneData built once at viewer
        open. Diagrams that paint on the substrate (Contour,
        DeformedShape, …) require it; stub diagrams may pass ``None``.
        """
        self._plotter = plotter
        self._fem = fem
        self._scene = scene
        # Subclasses decide which axis matters; resolving both is
        # cheap and avoids subclass boilerplate.
        try:
            self._resolved_node_ids = self.selector.resolve_node_ids(fem)
        except Exception:
            self._resolved_node_ids = None
        try:
            self._resolved_element_ids = self.selector.resolve_element_ids(fem)
        except Exception:
            self._resolved_element_ids = None
        self._attached = True

    def update_to_step(self, step_index: int) -> None:
        """Refresh actors for ``step_index``. Subclass override."""
        raise NotImplementedError(
            f"{type(self).__name__}.update_to_step must be overridden."
        )

    def detach(self) -> None:
        """Remove actors. Subclass override.

        The default implementation removes every actor in
        ``self._actors`` from the plotter and clears state.
        Subclasses should ``super().detach()`` last.
        """
        if self._plotter is not None:
            for actor in self._actors:
                try:
                    self._plotter.remove_actor(actor)
                except Exception:
                    pass
        self._actors = []
        self._plotter = None
        self._fem = None
        self._scene = None
        self._resolved_node_ids = None
        self._resolved_element_ids = None
        self._attached = False

    def settings_widget(self) -> Any:
        """Return a Qt widget for the per-diagram settings tab.

        Default: ``None`` — no per-diagram settings. Subclasses that
        expose styling controls return a ``QWidget``.
        """
        return None

    def make_side_panel(self, director: Any) -> Any:
        """Construct a dockable side-panel widget for this diagram.

        Default: ``None`` — no side panel. Diagrams with auxiliary
        2-D plots (fiber section scatter, layer through-thickness
        profile) override this to return a panel object whose
        ``.widget`` attribute is a ``QWidget``. The viewer docks the
        widget when the diagram is added and removes it on detach.
        """
        return None

    # ------------------------------------------------------------------
    # Visibility (default impl — subclasses can override for actor-
    # specific behaviour)
    # ------------------------------------------------------------------

    def set_visible(self, visible: bool) -> None:
        """Show / hide all actors without detaching."""
        self._visible = visible
        for actor in self._actors:
            try:
                actor.SetVisibility(bool(visible))
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        attached = "attached" if self._attached else "detached"
        return f"<{type(self).__name__} kind={self.kind} {attached}>"
