"""``RenderBackend`` Protocol — the viewer-side render contract.

Defined by [ADR 0042](../../opensees/architecture/decisions/0042-render-backend-seam.md)
(§Decision, Part 2).  A backend consumes :mod:`SceneLayer
<apeGmsh.viewers.scene_ir._layers>` value types and produces pixels.
The domain layer (``diagrams/``, ``overlays/``, colour/visibility in
``core/``) calls only these methods — it never imports ``vtk`` /
``pyvista`` (INV-2).

Structural, not nominal: implementers do **not** subclass
``RenderBackend``; conformance is by :class:`typing.Protocol`, the
same discipline ADR 0026 used for ``H5ModelReader``.  The Protocols
are :func:`~typing.runtime_checkable` so tests can ``isinstance``-probe
a backend, but that only verifies method *presence*, not signature.

Picking is deliberately **off** this Protocol.  Ray-casting is the
most VTK-bound surface and the least essential for the first
web/Jupyter target; it gets its own ``PickBackend`` Protocol in
Phase R-D (a future ADR).  A backend reports its picking capability
via :meth:`RenderBackend.supports_picking`; ``False`` is legal and
means view-only.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from ._layers import ColorSpec, SceneLayer, ScalarBarSpec, VisibilityMask


@runtime_checkable
class LayerHandle(Protocol):
    """An opaque, backend-owned handle to an added layer.

    Returned by :meth:`RenderBackend.add_layer` and passed back to
    ``update_layer`` / ``remove_layer`` / ``set_visibility``.  The
    domain layer treats it as a token and never inspects it.
    """


@runtime_checkable
class RenderBackend(Protocol):
    """The render contract a backend implements.

    Reference implementer: ``PyVistaQtBackend`` (desktop).  Alternate:
    ``TrameBackend`` (web/Jupyter, hybrid local+remote).  Future:
    ``ParaViewExportBackend``.  A backend owns *all* VTK / pyvista /
    trame construction.
    """

    def add_layer(self, layer: SceneLayer) -> LayerHandle:
        """Add a layer to the scene and return its handle."""
        ...

    def update_layer(self, handle: LayerHandle, layer: SceneLayer) -> None:
        """Replace the data behind ``handle`` with ``layer``.

        Used for step-animation: the layer geometry/scalars change but
        the handle (and ideally the underlying actor) is reused so the
        camera and pick state survive.
        """
        ...

    def remove_layer(self, handle: LayerHandle) -> None:
        """Remove the layer behind ``handle``. Idempotent."""
        ...

    def set_visibility(
        self, handle: LayerHandle, mask: VisibilityMask
    ) -> None:
        """Apply a per-cell visibility mask to ``handle``."""
        ...

    def set_layer_visible(self, handle: LayerHandle, visible: bool) -> None:
        """Show or hide the whole layer behind ``handle``.

        Coarser than :meth:`set_visibility` (which hides individual
        cells) — this toggles the entire layer, the operation a
        diagram's show/hide checkbox drives.
        """
        ...

    def set_layer_color(self, handle: LayerHandle, color: "ColorSpec") -> None:
        """Re-apply colour to an existing layer without re-adding it.

        ``solid`` sets the actor colour; ``by_array`` rebinds the
        mapper's scalars + lookup table (preset / clim / log). This is
        the live-recolour path a diagram drives when the ColorMapEditor
        changes the cmap or range (the diagram-side LUT mirror
        translates its state into a plain :class:`ColorSpec` / ``LutSpec``
        and calls here — the backend never sees Qt).
        """
        ...

    def add_scalar_bar(self, handle: LayerHandle, spec: "ScalarBarSpec") -> None:
        """Show a scalar bar bound to ``handle``'s mapper.

        Keyed by ``spec.layer_id`` so multiple diagrams coexist even
        when they share a display title.
        """
        ...

    def remove_scalar_bar(self, layer_id: str) -> None:
        """Remove the scalar bar keyed by ``layer_id``. Idempotent."""
        ...

    def set_scalar_bar_format(self, layer_id: str, fmt: str) -> None:
        """Update the tick-label ``printf`` format of a layer's bar."""
        ...

    def reset_camera(self) -> None:
        """Fit the camera to the current scene bounds."""
        ...

    def render(self) -> None:
        """Flush pending changes to the display."""
        ...

    def screenshot(self, path: Path) -> None:
        """Write the current frame to ``path``."""
        ...

    def supports_picking(self) -> bool:
        """Whether this backend can resolve a screen pick to an entity.

        ``False`` is legal — the backend renders view-only.  Picking
        is restored on web in Phase R-D via a separate Protocol.
        """
        ...


__all__ = ["LayerHandle", "RenderBackend"]
