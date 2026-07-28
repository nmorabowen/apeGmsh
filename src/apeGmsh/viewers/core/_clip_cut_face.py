"""ClipCutFaceRenderer — the contour on the cut (ADR 0083 S3).

Render-time clipping discards *fragments*, and a contour renders the
mesh's exterior skin, so a section plane through a solid opens into an
**empty box**: no field on the cut, nothing in the interior (the ADR's
*F1 revisited* evidence). This renderer closes that: for every active
plane it asks the backend for a real dataset slice of every contour in
the scene and adds the result as an ordinary scene-IR ``MeshLayer``.

Three properties make it honest rather than decorative:

* **It slices the contour's own dataset**, not the substrate. The
  substrate carries no scalars, and the contour's submesh is
  volumetric (``extract_points`` / ``extract_cells`` keep 3-D cells —
  only the *rendering* is a skin), so the cutter interpolates the
  contour's own values at the current step, with deformation already
  baked into the points.
* **It reuses the contour's ``ColorSpec``** verbatim. A cut face on a
  different scale from the contour it caps would be a lie about the
  same quantity.
* **``clip_exempt=True``.** The cap is coplanar with its own plane, so
  whether its fragments survive that plane's mapper test is a
  floating-point coin toss — it would flicker or vanish. Other planes
  still have to trim it, and they do: geometrically, inside
  ``slice_layer``'s ``trim``, where coplanarity is not at issue.

**Recompute policy.** On drag-release and on step change, never per
drag tick — slicing every dataset on every mouse-move is exactly the
cost render-time clipping exists to avoid (ADR 0083 alternative 2) and
it would undo S2's measured zero-rebuild drag. Mid-drag the caps are
*hidden* rather than left where the plane used to be: a stale cap
floating inside the model is worse than no cap. The in-flight question
is answered by the injected ``settled`` predicate — the gizmo
interactor's public :attr:`~._clip_gizmo_interactor.ClipGizmoInteractor.is_dragging`
— and the release is pushed back through its ``on_drag_end`` hook,
because a release fires no ``CLIP_PLANES_CHANGED`` of its own.

Like ``ClipGizmoRenderer`` this is a **projection** of
``ClipPlaneSetController`` state: it reads on :meth:`refresh` and
mutates nothing, so it fires no events of its own (ADR 0056).
"""
from __future__ import annotations

import dataclasses
from typing import Any, Callable, Optional, Sequence

import numpy as np

#: Layer-id prefix for every cut-face layer — namespaced so it can
#: never collide with a diagram's own layers.
CUT_FACE_LAYER_PREFIX = "clip_cut_face:"

#: How far off its plane the cap is drawn, as a fraction of its own
#: extent. The S2 gizmo quad sits on exactly the same plane, and two
#: coplanar surfaces fight fragment by fragment: measured live, the cap
#: came back speckled with patches of translucent quad. The cap steps a
#: hair into the **discarded** half — where render-time clipping has
#: left nothing else to draw — so it wins that contest deterministically
#: while the quad still frames it. Invisible at this size (7e-3 world
#: units on a 6-unit model) and, seen from the kept side, the cap is
#: behind solid material either way.
_CAP_NUDGE_FRAC = 1e-3


class ClipCutFaceRenderer:
    """Projects the plane set × the scene's contours onto cap layers."""

    def __init__(
        self,
        backend: Any,
        controller: Any,
        *,
        sources: Callable[[], Sequence[tuple]],
        settled: Optional[Callable[[], bool]] = None,
    ) -> None:
        """``sources()`` yields ``(key, layer_handle, color)`` per
        sliceable diagram — what ``ContourDiagram.cut_face_source``
        returns. ``settled()`` is ``False`` while a gizmo drag is in
        flight; ``None`` means "always settled" (headless / no gizmo).
        """
        self._backend = backend
        self._controller = controller
        self._sources = sources
        self._settled = settled
        # Strong refs, deliberately: the backend's clip registry holds
        # handles weakly, so whoever adds a layer keeps its handle alive
        # (the same contract diagrams and gizmos follow).
        self._handles: "dict[str, Any]" = {}
        #: Actor-visibility state of the caps as a group. A drag hides
        #: them without slicing; the next settled refresh shows them.
        self._visible = True

    def layer_ids(self) -> "set[str]":
        """The cut-face layers currently in the scene."""
        return set(self._handles)

    def refresh(self, *, force: bool = False) -> None:
        """Reconcile cut-face layers against the controller + sources.

        ``force`` recomputes even mid-drag; the drag's own release
        handler passes it, so the cap catches up the moment the user
        lets go rather than waiting for the next unrelated event.
        """
        if not force and self._settled is not None:
            try:
                settled = bool(self._settled())
            except Exception:
                settled = True
            if not settled:
                self._set_visible(False)
                return

        desired = self._build()
        for layer_id in list(self._handles):
            if layer_id not in desired:
                handle = self._handles.pop(layer_id)
                try:
                    self._backend.remove_layer(handle)
                except Exception:
                    pass
        for layer_id, layer in desired.items():
            try:
                if layer_id in self._handles:
                    self._backend.update_layer(self._handles[layer_id], layer)
                else:
                    self._handles[layer_id] = self._backend.add_layer(layer)
            except Exception:
                self._handles.pop(layer_id, None)
        self._set_visible(True)

    def clear(self) -> None:
        """Drop every cut-face layer. Idempotent."""
        for handle in self._handles.values():
            try:
                self._backend.remove_layer(handle)
            except Exception:
                pass
        self._handles.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build(self) -> "dict[str, Any]":
        """Slice every (active plane × contour) pair into a cap layer."""
        controller = self._controller
        if (
            controller is None
            or not controller.cut_face
            or not controller.apply_cuts
        ):
            return {}
        slicer = getattr(self._backend, "slice_layer", None)
        if slicer is None:
            return {}
        planes = [p for p in controller.planes() if p.active]
        specs = {p.plane_id: p.spec() for p in planes}
        try:
            sources = list(self._sources())
        except Exception:
            return {}

        layers: "dict[str, Any]" = {}
        for plane in planes:
            trim = tuple(
                spec for pid, spec in specs.items() if pid != plane.plane_id
            )
            for key, handle, color in sources:
                layer_id = f"{CUT_FACE_LAYER_PREFIX}{plane.plane_id}:{key}"
                try:
                    cut = slicer(
                        handle, specs[plane.plane_id],
                        layer_id=layer_id, trim=trim,
                    )
                except Exception:
                    cut = None
                if cut is None:
                    continue
                # The cap's every style flag is decided here, on the
                # domain side: the backend hands back geometry + fields
                # and makes no styling decisions.
                layers[layer_id] = dataclasses.replace(
                    cut,
                    points=_nudged(cut.points, specs[plane.plane_id]),
                    color=color,
                    clip_exempt=True,
                    pickable=False,
                )
        return layers

    def _set_visible(self, visible: bool) -> None:
        if bool(visible) == self._visible:
            return
        self._visible = bool(visible)
        for handle in self._handles.values():
            try:
                self._backend.set_layer_visible(handle, visible)
            except Exception:
                pass


def _nudged(points: Any, spec: Any) -> Any:
    """``points`` stepped a hair into ``spec``'s discarded half.

    See :data:`_CAP_NUDGE_FRAC` for why. Scaled off the cap's own
    extent so it is proportionate on a 6-unit model and on a
    600-metre one alike.
    """
    from ..scene_ir import PointSet

    coords = np.asarray(points.coords, dtype=np.float64)
    if coords.size == 0:
        return points
    span = float(np.linalg.norm(coords.max(axis=0) - coords.min(axis=0)))
    if span <= 0.0:
        return points
    normal = np.asarray(spec.normal, dtype=np.float64)
    return PointSet(coords - (_CAP_NUDGE_FRAC * span) * normal)


__all__ = ["CUT_FACE_LAYER_PREFIX", "ClipCutFaceRenderer"]
