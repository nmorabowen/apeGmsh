"""``PyVistaQtBackend`` — the reference (desktop) render backend.

Implements :class:`~apeGmsh.viewers.scene_ir.RenderBackend` by
translating ``scene_ir`` value types into ``pyvista`` plotter calls.
This is where every VTK/pyvista concept lives that the domain layer
must not know about: the token → VTK-cell-type mapping, the
``vtkGhostType`` visibility bitmask, ``cell_data["colors"]`` RGB
arrays.

The IR is backend-neutral (string cell tokens, per-cell RGB as plain
arrays); this backend maps those to concrete VTK representations — the
seam working exactly as ADR 0042 intends.

**Testability split.** The data-only translation
(:func:`mesh_layer_to_grid`, :func:`apply_visibility_mask`) is pure —
it builds a ``pyvista.UnstructuredGrid`` and mutates its arrays without
ever touching an OpenGL context, so it is unit-tested headlessly.  The
plotter-driving methods on :class:`PyVistaQtBackend` (``add_layer`` and
friends) require a live render context and are verified by the desktop
viewer + the user's eyeball (this environment has no GPU; see the
project's viewer-verification note).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pyvista as pv

from apeGmsh.viewers.scene_ir import (
    CellBlocks,
    ColorSpec,
    GlyphLayer,
    LabelLayer,
    MeshLayer,
    PointSet,
    ScalarBarSpec,
    SceneLayer,
    VisibilityMask,
)

# VTK cell-type integer codes (vtkCellType.h). Kept as literals so the
# IR's neutral string tokens map here and nowhere else.
_VTK_VERTEX = 1
_VTK_LINE = 3
_VTK_TRIANGLE = 5
_VTK_QUAD = 9
_VTK_TETRA = 10
_VTK_HEXAHEDRON = 12
_VTK_WEDGE = 13
_VTK_PYRAMID = 14

#: The IR's neutral cell-type tokens -> concrete VTK cell-type codes.
#: A scene-builder emits these tokens on :class:`CellBlocks`; this is the
#: single place they become VTK integers.
TOKEN_TO_VTK: dict[str, int] = {
    "vertex": _VTK_VERTEX,
    "line": _VTK_LINE,
    "triangle": _VTK_TRIANGLE,
    "quad": _VTK_QUAD,
    "tetra": _VTK_TETRA,
    "hexahedron": _VTK_HEXAHEDRON,
    "wedge": _VTK_WEDGE,
    "pyramid": _VTK_PYRAMID,
}

#: Inverse — VTK cell-type code back to the neutral token, for
#: decomposing a pyvista grid into :class:`CellBlocks`.
VTK_TO_TOKEN: dict[int, str] = {v: k for k, v in TOKEN_TO_VTK.items()}

# vtkDataSetAttributes::HIDDENCELL — the bit a renderer honours to skip
# a cell. Matches viewers/core/element_visibility.py.
_GHOST_HIDDEN_CELL = 0x01


# =====================================================================
# Pure translation (no OpenGL context required)
# =====================================================================


def mesh_layer_to_grid(layer: MeshLayer) -> pv.UnstructuredGrid:
    """Build a ``pyvista.UnstructuredGrid`` from a :class:`MeshLayer`.

    Pure data construction — no plotter, no render context.  Attaches:

    * every :class:`ScalarField` as ``point_data`` / ``cell_data`` under
      its name;
    * for ``ColorSpec(per_entity_rgb)``, a ``cell_data["colors"]``
      ``uint8`` ``(n_cells, 3)`` array (the convention
      ``viewers/core/color_manager.py`` uses), aligned with the grid's
      cell order (= the iteration order of ``CellBlocks.blocks``);
    * the visibility mask as a ``cell_data["vtkGhostType"]`` bitmask.
    """
    cells_dict: dict[int, np.ndarray] = {}
    for token, conn in layer.cells.blocks.items():
        try:
            vtk_type = TOKEN_TO_VTK[token]
        except KeyError as exc:
            raise ValueError(
                f"MeshLayer {layer.layer_id!r}: unknown cell token {token!r}. "
                f"Known tokens: {sorted(TOKEN_TO_VTK)}."
            ) from exc
        cells_dict[vtk_type] = conn

    grid = pv.UnstructuredGrid(cells_dict, layer.points.coords)

    for sf in layer.fields:
        target = grid.point_data if sf.location == "point" else grid.cell_data
        target[sf.name] = sf.values

    if layer.color.mode == "per_entity_rgb" and layer.color.entity_rgb is not None:
        rgb = layer.color.entity_rgb
        # IR carries float RGB in [0, 1]; the colors convention is uint8.
        if np.issubdtype(rgb.dtype, np.floating):
            rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        else:
            rgb = rgb.astype(np.uint8)
        grid.cell_data["colors"] = rgb

    apply_visibility_mask(grid, layer.visibility)
    return grid


def apply_visibility_mask(
    grid: pv.UnstructuredGrid, mask: VisibilityMask
) -> None:
    """Write ``mask`` into ``grid.cell_data["vtkGhostType"]`` in place.

    Cells in ``mask.hidden_cells`` get the ``HIDDENCELL`` bit; all
    others are cleared.  Pure array mutation — no render context.
    """
    n = grid.n_cells
    ghost = np.zeros(n, dtype=np.uint8)
    if mask.hidden_cells:
        idx = np.fromiter(
            (c for c in mask.hidden_cells if 0 <= c < n),
            dtype=np.int64,
        )
        if idx.size:
            ghost[idx] = _GHOST_HIDDEN_CELL
    grid.cell_data["vtkGhostType"] = ghost


def cellblocks_from_grid(grid: pv.UnstructuredGrid) -> CellBlocks:
    """Decompose a pyvista grid into neutral :class:`CellBlocks`.

    The inverse of :func:`mesh_layer_to_grid`'s cell construction — uses
    pyvista's ``cells_dict`` and maps each VTK cell-type code back to the
    IR's neutral token. Cell types with no token mapping are dropped.
    Lets a diagram that still extracts a submesh via pyvista (transitional
    R-B) re-express the result as backend-neutral IR.
    """
    blocks: dict[str, np.ndarray] = {}
    for vtk_int, conn in grid.cells_dict.items():
        token = VTK_TO_TOKEN.get(int(vtk_int))
        if token is not None:
            blocks[token] = conn
    return CellBlocks(blocks)


def mesh_layer_from_grid(
    grid: pv.UnstructuredGrid,
    layer_id: str,
    *,
    color: Optional[ColorSpec] = None,
    opacity: float = 1.0,
    wireframe: bool = False,
) -> MeshLayer:
    """Build a :class:`MeshLayer` from a pyvista grid (points + cells)."""
    return MeshLayer(
        layer_id=layer_id,
        points=PointSet(np.asarray(grid.points)),
        cells=cellblocks_from_grid(grid),
        color=color if color is not None else ColorSpec(),
        opacity=opacity,
        wireframe=wireframe,
    )


# =====================================================================
# Layer handle
# =====================================================================


class _PvHandle:
    """Backend-owned handle to one added layer (a ``LayerHandle``).

    Opaque to the domain layer; holds the actor + the dataset so
    ``update_layer`` / ``set_visibility`` can mutate in place.
    """

    __slots__ = ("layer_id", "actor", "dataset", "kind")

    def __init__(self, layer_id: str, actor: Any, dataset: Any, kind: str) -> None:
        self.layer_id = layer_id
        self.actor = actor
        self.dataset = dataset
        self.kind = kind


# =====================================================================
# Backend
# =====================================================================


class PyVistaQtBackend:
    """Reference desktop backend over a ``pyvista`` plotter.

    Construct with any pyvista ``BasePlotter`` — the live
    ``pyvistaqt.QtInteractor`` plotter in the desktop viewer, or a
    ``pyvista.Plotter(off_screen=True)`` in a render-capable test.
    """

    def __init__(self, plotter: Any) -> None:
        self._plotter = plotter
        self._scalar_bars: dict[str, Any] = {}

    @property
    def plotter(self) -> Any:
        """The wrapped pyvista plotter.

        Escape hatch for the R-B transition: the diagram base derives
        ``self._plotter`` from this so un-migrated diagrams keep driving
        the raw plotter directly. Removed at R-B.final once every
        diagram emits through the backend.
        """
        return self._plotter

    # -- RenderBackend ------------------------------------------------

    def add_layer(self, layer: SceneLayer) -> _PvHandle:
        if isinstance(layer, MeshLayer):
            return self._add_mesh_layer(layer)
        if isinstance(layer, GlyphLayer):
            return self._add_glyph_layer(layer)
        if isinstance(layer, LabelLayer):
            return self._add_label_layer(layer)
        raise TypeError(f"Unsupported SceneLayer type: {type(layer).__name__}")

    def update_layer(self, handle: _PvHandle, layer: SceneLayer) -> None:
        # Reuse the actor when topology is unchanged (cheap animation
        # path): mutate point coords + scalar arrays on the bound
        # dataset. Otherwise rebuild from scratch.
        if (
            isinstance(layer, MeshLayer)
            and handle.kind == "mesh"
            and handle.dataset is not None
            and handle.dataset.n_points == layer.points.n_points
            and handle.dataset.n_cells == layer.cells.n_cells
        ):
            handle.dataset.points = layer.points.coords
            for sf in layer.fields:
                target = (
                    handle.dataset.point_data
                    if sf.location == "point"
                    else handle.dataset.cell_data
                )
                target[sf.name] = sf.values
            apply_visibility_mask(handle.dataset, layer.visibility)
            return
        self.remove_layer(handle)
        new = self.add_layer(layer)
        handle.actor, handle.dataset, handle.kind = (
            new.actor,
            new.dataset,
            new.kind,
        )

    def remove_layer(self, handle: _PvHandle) -> None:
        if handle.actor is not None:
            self._plotter.remove_actor(handle.actor)
            handle.actor = None

    def set_visibility(self, handle: _PvHandle, mask: VisibilityMask) -> None:
        if handle.dataset is not None and handle.kind == "mesh":
            apply_visibility_mask(handle.dataset, mask)

    def set_layer_visible(self, handle: _PvHandle, visible: bool) -> None:
        if handle.actor is not None:
            try:
                handle.actor.SetVisibility(bool(visible))
            except Exception:
                pass

    def set_layer_color(self, handle: _PvHandle, color: ColorSpec) -> None:
        actor = handle.actor
        if actor is None:
            return
        try:
            mapper = actor.GetMapper()
        except Exception:
            return
        if color.mode == "by_array" and color.lut is not None:
            try:
                if color.array_name and handle.dataset is not None:
                    handle.dataset.set_active_scalars(color.array_name)
            except Exception:
                pass
            try:
                table = _lookup_table_from_lutspec(color.lut)
                mapper.SetLookupTable(table)
                mapper.SetScalarRange(color.lut.vmin, color.lut.vmax)
            except Exception:
                pass
        elif color.mode == "solid":
            try:
                actor.prop.color = color.solid_rgb
            except Exception:
                pass

    def add_scalar_bar(self, handle: _PvHandle, spec: ScalarBarSpec) -> None:
        actor = handle.actor
        if actor is None:
            return
        try:
            mapper = actor.GetMapper()
        except Exception:
            return
        # Drop any prior bar for this layer before re-adding.
        self.remove_scalar_bar(spec.layer_id)
        try:
            bar = self._plotter.add_scalar_bar(
                title=spec.title, mapper=mapper, interactive=True,
            )
            self._scalar_bars[spec.layer_id] = (spec.title, bar)
        except Exception:
            pass

    def remove_scalar_bar(self, layer_id: str) -> None:
        entry = self._scalar_bars.pop(layer_id, None)
        if entry is not None:
            title, _bar = entry
            try:
                self._plotter.remove_scalar_bar(title)
            except Exception:
                pass

    def set_scalar_bar_format(self, layer_id: str, fmt: str) -> None:
        entry = self._scalar_bars.get(layer_id)
        if entry is not None:
            _title, bar = entry
            try:
                bar.SetLabelFormat(fmt)
            except Exception:
                pass

    def reset_camera(self) -> None:
        self._plotter.reset_camera()

    def render(self) -> None:
        self._plotter.render()

    def screenshot(self, path: Path) -> None:
        self._plotter.screenshot(str(path))

    def supports_picking(self) -> bool:
        return True

    # -- internals ----------------------------------------------------

    def _add_mesh_layer(self, layer: MeshLayer) -> _PvHandle:
        grid = mesh_layer_to_grid(layer)
        kwargs: dict[str, Any] = {
            "opacity": layer.opacity,
            "show_edges": layer.show_edges,
        }
        if layer.wireframe:
            kwargs["style"] = "wireframe"
        color = layer.color
        if color.mode == "solid":
            kwargs["color"] = color.solid_rgb
        elif color.mode == "by_array":
            kwargs["scalars"] = color.array_name
            if color.lut is not None:
                kwargs["cmap"] = color.lut.name
                kwargs["clim"] = (color.lut.vmin, color.lut.vmax)
        elif color.mode == "per_entity_rgb":
            kwargs["scalars"] = "colors"
            kwargs["rgb"] = True
        actor = self._plotter.add_mesh(grid, **kwargs)
        if layer.silhouette:
            try:
                self._plotter.add_silhouette(grid)
            except Exception:
                pass
        return _PvHandle(layer.layer_id, actor, grid, "mesh")

    def _add_glyph_layer(self, layer: GlyphLayer) -> _PvHandle:
        cloud = pv.PolyData(layer.positions.coords)
        if layer.orientations is not None:
            cloud["_vec"] = layer.orientations
        if layer.scales is not None:
            cloud["_size"] = layer.scales
        color = layer.color
        # Per-glyph colour scalar (by_array). Attached to the source so
        # vtkGlyph3D broadcasts it onto every glyph instance's points.
        if (
            color.mode == "by_array"
            and color.array_name
            and layer.color_scalar is not None
        ):
            cloud[color.array_name] = layer.color_scalar
        geom = _glyph_geometry(layer.kind)
        glyphed = cloud.glyph(
            geom=geom,
            orient="_vec" if layer.orientations is not None else False,
            scale="_size" if layer.scales is not None else False,
        )
        kwargs: dict[str, Any] = {}
        if color.mode == "by_array" and color.array_name:
            kwargs["scalars"] = color.array_name
            if color.lut is not None:
                kwargs["cmap"] = color.lut.name
                kwargs["clim"] = (color.lut.vmin, color.lut.vmax)
        else:
            kwargs["color"] = color.solid_rgb
        actor = self._plotter.add_mesh(glyphed, **kwargs)
        return _PvHandle(layer.layer_id, actor, glyphed, "glyph")

    def _add_label_layer(self, layer: LabelLayer) -> _PvHandle:
        actor = self._plotter.add_point_labels(
            layer.positions.coords, list(layer.texts)
        )
        return _PvHandle(layer.layer_id, actor, None, "label")


def _lookup_table_from_lutspec(lut: "Any") -> Any:
    """Build a ``pv.LookupTable`` from a :class:`LutSpec`.

    The re-homed counterpart of the diagram-side LUT mirror's
    ``to_pyvista_lookup_table`` — keeps all VTK/pyvista LUT construction
    inside the backend so the mirror stays Qt-only and trame-portable.
    """
    table = pv.LookupTable(lut.name)
    table.scalar_range = (lut.vmin, lut.vmax)
    if getattr(lut, "log_scale", False):
        try:
            table.log_scale = True
        except Exception:
            try:
                table.SetScaleToLog10()
            except Exception:
                pass
    return table


def _glyph_geometry(kind: str) -> Any:
    if kind == "arrow":
        return pv.Arrow()
    if kind == "cone":
        return pv.Cone()
    return pv.Sphere()


__all__ = [
    "PyVistaQtBackend",
    "mesh_layer_to_grid",
    "apply_visibility_mask",
    "cellblocks_from_grid",
    "mesh_layer_from_grid",
]
