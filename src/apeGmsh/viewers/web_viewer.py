"""``WebViewer`` — the web / Jupyter results viewer shell (ADR 0042, R-C).

The web counterpart of the Qt :class:`~apeGmsh.viewers.results_viewer.ResultsViewer`.
It owns the same domain stack — a :class:`ResultsDirector`, a substrate
``FEMSceneData``, and the diagram registry — but renders through a
:class:`~apeGmsh.viewers.backends.trame.TrameBackend` (a plain
``pyvista.Plotter`` served via ``pyvista.trame``) instead of a
``QtInteractor``. That swap is the whole point of the render seam: the
director / diagrams / scene logic is reused verbatim; only the backend
and the windowing change.

**Slice 1 (this module) is view-only.** It renders the substrate plus
whatever diagrams the director holds, scoped to one stage/step, and
displays inline in Jupyter through pyvista's trame backend (the
kernel-safe path that replaces the blocking Qt viewer). Deferred to
later R-C slices: a trame time-slider / layer-toggle UI (needs trame
app state) and the hybrid client/server render-mode toggle; picking is
deferred to R-D.

Construction is fully headless — building a ``WebViewer`` binds the
backend and attaches diagrams without any render context, so it is
unit-testable. Only :meth:`WebViewer.show` needs a live browser/notebook
and is verified by eyeball.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from apeGmsh.results.Results import Results
    from .diagrams._director import ResultsDirector
    from .scene.fem_scene import FEMSceneData


class WebViewer:
    """Minimal view-only web/Jupyter shell around a :class:`ResultsDirector`.

    Parameters
    ----------
    results
        The :class:`~apeGmsh.results.Results` to render.
    stage
        Stage id or name to activate. Defaults to the first stage.
    substrate_color
        Solid colour for the FEM substrate mesh.
    plotter
        Inject a pyvista plotter (e.g. ``pv.Plotter(off_screen=True)`` in
        a render-capable test). Defaults to a fresh ``pv.Plotter()`` that
        the trame shell serves.
    """

    def __init__(
        self,
        results: "Results",
        *,
        stage: Optional[str] = None,
        substrate_color: str = "lightgray",
        plotter: Optional[Any] = None,
    ) -> None:
        import pyvista as pv

        from .backends import TrameBackend
        from .diagrams._director import ResultsDirector
        from .scene.fem_scene import build_fem_scene

        director = ResultsDirector(results)
        view = director.view
        if view is None:
            raise RuntimeError(
                "WebViewer requires a Results with bound FEMData. "
                "Construct Results with fem= or call results.bind(fem)."
            )
        scene = build_fem_scene(view)

        if plotter is None:
            plotter = pv.Plotter()
        plotter.add_mesh(
            scene.grid, color=substrate_color, show_edges=True,
            name="substrate", pickable=False,
        )

        backend = TrameBackend(plotter)
        director.bind_plotter(backend, scene=scene, render_callback=plotter.render)

        # Activate a stage so n_steps / set_step are live, then land on
        # step 0. Mirrors the Qt viewer's boot behaviour.
        stages = director.stages()
        if stages:
            director.set_stage(stage or stages[0].id)
            director.set_step(0)

        self._results = results
        self._director = director
        self._scene = scene
        self._plotter = plotter
        self._backend = backend

    # ------------------------------------------------------------------
    # Accessors (so callers can add diagrams, then re-show)
    # ------------------------------------------------------------------

    @property
    def director(self) -> "ResultsDirector":
        return self._director

    @property
    def scene(self) -> "FEMSceneData":
        return self._scene

    @property
    def plotter(self) -> Any:
        return self._plotter

    @property
    def backend(self) -> Any:
        return self._backend

    # ------------------------------------------------------------------
    # Stepping
    # ------------------------------------------------------------------

    def set_step(self, step_index: int) -> None:
        """Move the active step and re-render (programmatic scrub)."""
        self._director.set_step(int(step_index))
        self._plotter.render()

    @property
    def n_steps(self) -> int:
        return self._director.n_steps

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def show(self, *, jupyter_backend: str = "trame", **kwargs: Any) -> Any:
        """Display the scene inline (Jupyter) via pyvista's trame backend.

        Returns the trame/ipywidgets viewer object so a notebook renders
        it. Honours ``APEGMSH_SKIP_VIEWER`` (returns ``None``) so the same
        cell runs under ``nbconvert --execute`` / CI without a browser.
        """
        if os.environ.get("APEGMSH_SKIP_VIEWER"):
            print("[skip web viewer] APEGMSH_SKIP_VIEWER set")
            return None
        return self._plotter.show(
            jupyter_backend=jupyter_backend, return_viewer=True, **kwargs
        )


def show_web(
    results: "Results",
    *,
    stage: Optional[str] = None,
    show: bool = True,
) -> Any:
    """Open the view-only web/Jupyter results viewer (ADR 0042, R-C).

    Builds a :class:`WebViewer` around ``results`` and, when ``show`` is
    ``True`` (default), displays it inline. Returns the :class:`WebViewer`
    so callers can add diagrams via ``viewer.director`` and call
    ``viewer.show()`` again, or scrub with ``viewer.set_step(i)``.
    """
    viewer = WebViewer(results, stage=stage)
    if show:
        viewer.show()
    return viewer


__all__ = ["WebViewer", "show_web"]
