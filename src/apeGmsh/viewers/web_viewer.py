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


# Friendly render-mode names → pyvista ``jupyter_backend`` (ADR 0042 R-C,
# resolved Q2). ``client`` renders with VTK.js / WebGL in the browser, so
# camera interaction is local and instant — the fast default. ``server``
# renders on the kernel and streams images back (most VTK-feature-complete
# but laggy). ``hybrid`` is pyvista's ``trame`` backend: both, with a
# local/remote toggle button in the toolbar (boots remote → feels slow).
_RENDER_MODE_TO_BACKEND = {
    "client": "client",
    "server": "server",
    "hybrid": "trame",
}


def _resolve_jupyter_backend(render_mode: str) -> str:
    """Map a friendly ``render_mode`` to a pyvista ``jupyter_backend``."""
    try:
        return _RENDER_MODE_TO_BACKEND[render_mode]
    except KeyError:
        valid = ", ".join(repr(m) for m in _RENDER_MODE_TO_BACKEND)
        raise ValueError(
            f"Unknown render_mode {render_mode!r}; use one of {valid}."
        ) from None


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
        """Move the active step and re-render (programmatic scrub).

        ``plotter.render()`` is what propagates the change to the trame
        view: pyvista registers an on-render callback when the view is
        created (``_BasePyVistaView``), so a render pushes the new frame
        to the browser. The same call drives both the Qt and web paths.
        """
        self._director.set_step(int(step_index))
        self._plotter.render()

    def set_layer_visible(self, diagram: Any, visible: bool) -> None:
        """Show / hide one diagram layer and re-render."""
        self._director.registry.set_visible(diagram, bool(visible))
        self._plotter.render()

    @property
    def n_steps(self) -> int:
        return self._director.n_steps

    def layer_diagrams(self) -> list[Any]:
        """The diagrams currently in the registry (one per toggle)."""
        return list(self._director.registry.diagrams())

    # ------------------------------------------------------------------
    # Controls (ipywidgets — the Jupyter scrubbing / visibility UI)
    # ------------------------------------------------------------------

    def controls(self) -> Any:
        """Build an ``ipywidgets`` control panel for this viewer.

        A step :class:`~ipywidgets.IntSlider` (when the active stage has
        more than one step) plus one :class:`~ipywidgets.Checkbox` per
        diagram layer. The slider drives :meth:`set_step`; each checkbox
        drives :meth:`set_layer_visible`. Both re-render, which pushes to
        the trame view via pyvista's on-render callback.

        Returns a :class:`~ipywidgets.VBox`. Raises if ``ipywidgets`` is
        not installed — call :meth:`show` (which degrades gracefully to a
        bare view) if you only need the render.
        """
        try:
            import ipywidgets as W
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise RuntimeError(
                "WebViewer.controls() needs ipywidgets — install the "
                "[viewer] extra, or call show(controls=False)."
            ) from exc

        children: list[Any] = []
        n = self.n_steps
        if n > 1:
            slider = W.IntSlider(
                value=0, min=0, max=n - 1, step=1, description="Step",
                continuous_update=False,
            )
            slider.observe(
                lambda change: self.set_step(int(change["new"])),
                names="value",
            )
            children.append(slider)

        for diagram in self.layer_diagrams():
            checkbox = W.Checkbox(
                value=bool(getattr(diagram, "is_visible", True)),
                description=self._layer_label(diagram),
            )

            def _handler(diag: Any):
                return lambda change: self.set_layer_visible(
                    diag, bool(change["new"])
                )

            checkbox.observe(_handler(diagram), names="value")
            children.append(checkbox)

        return W.VBox(children)

    @staticmethod
    def _layer_label(diagram: Any) -> str:
        label = getattr(diagram, "display_label", None)
        if callable(label):
            try:
                return str(label())
            except Exception:
                pass
        return getattr(diagram, "kind", "layer")

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def show(
        self,
        *,
        controls: bool = True,
        render_mode: str = "client",
        jupyter_backend: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Display the scene inline (Jupyter) via pyvista's trame backend.

        ``render_mode`` picks how the scene is rendered (resolved Q2):

        * ``"client"`` (default) — VTK.js / WebGL in the browser; camera
          interaction is local and instant. Fast for typical models.
        * ``"server"`` — render on the kernel and stream images back; most
          VTK-feature-complete but laggy. Use for very large models.
        * ``"hybrid"`` — pyvista's ``trame`` backend (both, with a
          local/remote toggle in the toolbar).

        Pass ``jupyter_backend`` to override with a raw pyvista backend name
        (e.g. ``"html"``, ``"static"``); it takes precedence over
        ``render_mode``.

        Returns the trame view widget — or, when ``controls`` is ``True``
        and ``ipywidgets`` is available, a :class:`~ipywidgets.VBox` of the
        control panel stacked above the view. Honours
        ``APEGMSH_SKIP_VIEWER`` (returns ``None``) so the same cell runs
        under ``nbconvert --execute`` / CI without a browser.
        """
        if os.environ.get("APEGMSH_SKIP_VIEWER"):
            print("[skip web viewer] APEGMSH_SKIP_VIEWER set")
            return None
        backend = jupyter_backend or _resolve_jupyter_backend(render_mode)
        view = self._plotter.show(
            jupyter_backend=backend, return_viewer=True, **kwargs
        )
        if not controls:
            return view
        try:
            import ipywidgets as W
        except ImportError:  # pragma: no cover - env-dependent
            return view  # degrade: no controls, just the rendered view
        # The control panel can only be stacked when the trame view is an
        # ipywidget. pyvista returns a *non*-widget when the trame server
        # couldn't launch (a static-image fallback — usually a missing
        # ``nest_asyncio2``, which pyvista needs to start the server without
        # ``await``) or in some IFrame modes; wrapping that in a VBox raises
        # a TraitError. Degrade to the bare view with a clear pointer.
        if not isinstance(view, W.Widget):
            import warnings

            warnings.warn(
                "show_web fell back to a static image without the control "
                "panel — the trame server could not launch in-notebook. "
                "Install nest_asyncio2 (`pip install nest_asyncio2`) for the "
                "live interactive view plus the step / visibility controls.",
                RuntimeWarning,
                stacklevel=2,
            )
            return view
        return W.VBox([self.controls(), view])


def show_web(
    results: "Results",
    *,
    stage: Optional[str] = None,
    show: bool = True,
    controls: bool = True,
    render_mode: str = "client",
) -> Any:
    """Open the view-only web/Jupyter results viewer (ADR 0042, R-C).

    Builds a :class:`WebViewer` around ``results`` and, when ``show`` is
    ``True`` (default), displays it inline with an ``ipywidgets`` control
    panel (step slider + per-layer visibility) when ``controls`` is
    ``True``. ``render_mode`` (``"client"`` / ``"server"`` / ``"hybrid"``)
    selects the trame render mode — see :meth:`WebViewer.show`. Returns the
    :class:`WebViewer` so callers can add diagrams via ``viewer.director``
    and call ``viewer.show()`` again, or scrub with ``viewer.set_step(i)``.
    """
    viewer = WebViewer(results, stage=stage)
    if show:
        # ``viewer.show`` *returns* the trame widget (pyvista's
        # ``return_viewer=True``) — it does not display it. Because we
        # return the ``WebViewer`` (so callers can scrub / add diagrams),
        # that widget would never reach the notebook's display hook, so
        # nothing renders. Hand it to ``IPython.display`` explicitly.
        widget = viewer.show(controls=controls, render_mode=render_mode)
        if widget is not None:
            try:
                from IPython.display import display
            except ImportError:  # pragma: no cover - non-notebook env
                pass
            else:
                display(widget)
    return viewer


__all__ = ["WebViewer", "show_web"]
