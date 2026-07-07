"""``PyVistaBackend`` — the generic pyvista render backend (+ the
desktop ``PyVistaQtBackend`` subclass).

Implements :class:`~apeGmsh.viewers.scene_ir.RenderBackend` by
translating ``scene_ir`` value types into ``pyvista`` plotter calls.
``PyVistaBackend`` drives any ``pyvista.BasePlotter`` and is shared by
the desktop (:class:`PyVistaQtBackend`) and web/Jupyter
(``trame.TrameBackend``) backends; only the plotter's windowing/serving
differs, and that lives outside this generic core.
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
_VTK_POLYGON = 7
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
    "polygon": _VTK_POLYGON,
    "quad": _VTK_QUAD,
    "tetra": _VTK_TETRA,
    "hexahedron": _VTK_HEXAHEDRON,
    "wedge": _VTK_WEDGE,
    "pyramid": _VTK_PYRAMID,
}

#: Inverse — VTK cell-type code back to the neutral token, for
#: decomposing a pyvista grid into :class:`CellBlocks`.
VTK_TO_TOKEN: dict[int, str] = {v: k for k, v in TOKEN_TO_VTK.items()}

# vtkDataSetAttributes::HIDDENCELL. VTK's CellGhostTypes enum is
# DUPLICATECELL=0x01 ... HIDDENCELL=0x20 — the previous 0x01 here was
# DUPLICATECELL, which happens to hide 1/2/3-D cells (surface
# extraction drops duplicate ghosts) but leaves 0-D vertex cells fully
# visible, and even 0x21 fails for vertices (only the pure 0x20 byte
# hides them; render-verified 2026-07-07 on all cell classes).
_GHOST_HIDDEN_CELL = 0x20


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
    grid = _grid_from_cellblocks(layer)

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


def _grid_from_cellblocks(layer: MeshLayer) -> pv.UnstructuredGrid:
    """Build the bare ``UnstructuredGrid`` (points + cells) for a layer.

    Fixed-size cell types go through pyvista's fast ``cells_dict``
    constructor (cells grouped by ascending VTK type — the order
    ``cellblocks_from_grid`` round-trips and the ``group_to_orig``
    permutation in LayerStack / Contour relies on).

    The ``"polygon"`` token is **variable-length**: the ``cells_dict``
    constructor rejects it (a polygon block can't be a rectangular
    ``(n_cells, n_nodes)`` array in general), so any layer carrying a
    polygon block is built through the explicit ``(cells, celltypes)``
    VTK arrays instead. Block iteration order is preserved there.
    """
    blocks = layer.cells.blocks
    for token in blocks:
        if token not in TOKEN_TO_VTK:
            raise ValueError(
                f"MeshLayer {layer.layer_id!r}: unknown cell token {token!r}. "
                f"Known tokens: {sorted(TOKEN_TO_VTK)}."
            )

    if "polygon" not in blocks:
        cells_dict = {TOKEN_TO_VTK[t]: conn for t, conn in blocks.items()}
        return pv.UnstructuredGrid(cells_dict, layer.points.coords)

    cells: list[int] = []
    celltypes: list[int] = []
    for token, conn in blocks.items():
        vtk_type = TOKEN_TO_VTK[token]
        for row in conn:
            cells.append(int(row.shape[0]))
            cells.extend(int(x) for x in row)
            celltypes.append(vtk_type)
    return pv.UnstructuredGrid(
        np.asarray(cells, dtype=np.int64),
        np.asarray(celltypes, dtype=np.uint8),
        layer.points.coords,
    )


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


class PyVistaBackend:
    """Generic ``RenderBackend`` over any ``pyvista.BasePlotter``.

    Every method here drives a plain ``pyvista`` plotter — none of it is
    Qt- or web-specific, so both the desktop (:class:`PyVistaQtBackend`)
    and web/Jupyter (:class:`~apeGmsh.viewers.backends.trame.TrameBackend`)
    backends share it. The *windowing / serving* of the plotter is owned
    by the viewer layer (the Qt ``ResultsViewer`` or the trame shell), not
    by the backend, so it lives in the subclasses (or outside entirely).

    Construct with any pyvista ``BasePlotter`` — the live
    ``pyvistaqt.QtInteractor`` plotter in the desktop viewer, a
    ``pyvista.Plotter`` served via ``pyvista.trame``, or a
    ``pyvista.Plotter(off_screen=True)`` in a render-capable test.
    """

    def __init__(self, plotter: Any) -> None:
        self._plotter = plotter
        self._scalar_bars: dict[str, Any] = {}
        self._pick_backend: Any = None

    @property
    def plotter(self) -> Any:
        """The wrapped pyvista plotter.

        The single seam between the backend and its host: the Qt viewer
        adds substrate/label actors to it directly, the trame shell serves
        it, and the ``headless_plotter`` test fixture reads ``scalar_bars``
        off it. The domain layer (``diagrams/``) never touches it.
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
            # Point size lives on the actor property, not the dataset —
            # without this, a live size change on a point-cloud layer
            # (fiber / sand set_point_size) would be silently dropped
            # by the in-place path.
            if layer.point_size is not None and handle.actor is not None:
                try:
                    handle.actor.prop.point_size = float(layer.point_size)
                except Exception:
                    pass
            return
        # Rebuild path (e.g. GlyphLayer, which has no in-place fast path):
        # remove + re-add the actor. ``add_mesh`` would otherwise reset the
        # camera to refit the new bounds, so the model window appears to
        # rescale/zoom on every animation step. Preserve the camera across
        # the rebuild — an update is never a reason to reframe.
        camera = self._plotter.camera_position
        self.remove_layer(handle)
        new = self.add_layer(layer)
        self._plotter.camera_position = camera
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

    def set_layer_opacity(self, handle: _PvHandle, opacity: float) -> None:
        actor = handle.actor
        if actor is None:
            return
        try:
            actor.prop.opacity = float(opacity)
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
                fmt=spec.fmt,
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

    def picking(self) -> Any:
        """The ``PickBackend`` for this plotter (ADR 0047, Phase R-D).

        Lazily built and cached. Consumers probe ``supports_picking()``
        first, then narrow to this. Kept off the base ``RenderBackend``
        Protocol (ADR 0042 INV-3 / ADR 0047 INV-1) so view-only backends
        need not implement it."""
        if self._pick_backend is None:
            from ._pyvista_pick import PyVistaPickBackend

            self._pick_backend = PyVistaPickBackend(self._plotter)
        return self._pick_backend

    # -- internals ----------------------------------------------------

    def _add_mesh_layer(self, layer: MeshLayer) -> _PvHandle:
        grid = mesh_layer_to_grid(layer)
        kwargs: dict[str, Any] = {
            "opacity": layer.opacity,
            "show_edges": layer.show_edges,
            # Scalar bars are an explicit add_scalar_bar concern; never
            # let add_mesh auto-create one (it would collide with the
            # diagram's explicit bar and own the registry title).
            "show_scalar_bar": False,
        }
        if layer.wireframe:
            kwargs["style"] = "wireframe"
        if layer.point_size is not None:
            kwargs["point_size"] = layer.point_size
            kwargs["render_points_as_spheres"] = layer.render_points_as_spheres
        if layer.show_edges and layer.edge_color is not None:
            kwargs["edge_color"] = layer.edge_color
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
        if not layer.pickable and actor is not None:
            try:
                actor.SetPickable(False)
            except Exception:
                pass
        if layer.back_color is not None and actor is not None:
            _apply_backface_color(actor, layer.back_color, layer.opacity)
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
        geom = _glyph_geometry(layer)
        glyphed = cloud.glyph(
            geom=geom,
            orient="_vec" if layer.orientations is not None else False,
            scale="_size" if layer.scales is not None else False,
        )
        kwargs: dict[str, Any] = {
            "opacity": layer.opacity,
            "show_scalar_bar": False,
        }
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


class PyVistaQtBackend(PyVistaBackend):
    """Reference desktop backend — a :class:`PyVistaBackend` whose plotter
    is the live ``pyvistaqt.QtInteractor`` owned by the Qt ``ResultsViewer``
    (or a ``pyvista.Plotter(off_screen=True)`` in render-capable tests).

    Adds nothing over the base: desktop windowing lives in the viewer, and
    picking is supported (inherited ``supports_picking() -> True``). It
    stays a distinct type so the seam reads clearly and so future desktop-
    only tweaks have a home.
    """


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


def _apply_backface_color(actor: Any, back_color: Any, opacity: float) -> None:
    """Paint ``actor``'s back faces a distinct colour (two-tone mesh).

    Builds a ``vtkProperty`` cloned from the actor's front-face property,
    recolours it, and assigns it as the backface property. Disables
    backface culling so the back side renders at all. Non-fatal: on any
    failure the mesh degrades to single-tone (the front colour), which is
    still a legible cut face — the section-cut normal arrow remains as the
    side indicator.
    """
    try:
        prop = actor.GetProperty()
        prop.SetBackfaceCulling(False)
        from vtkmodules.vtkRenderingCore import vtkProperty
        backface = vtkProperty()
        backface.DeepCopy(prop)
        backface.SetColor(*pv.Color(back_color).float_rgb)
        backface.SetOpacity(float(opacity))
        actor.SetBackfaceProperty(backface)
    except Exception:
        pass


def _glyph_geometry(layer: GlyphLayer) -> Any:
    kind = layer.kind
    if kind == "arrow":
        return pv.Arrow()
    if kind == "cone":
        return pv.Cone()
    if kind == "moment":
        # Curved-arrow torque glyph. Geometry construction lives in the
        # backend (it builds a pyvista mesh) so the diagram stays
        # pyvista-free; the diagram only carries the arc spec on the IR.
        from apeGmsh.viewers.overlays.moment_glyph import make_moment_glyph
        arc = layer.arc_degrees if layer.arc_degrees is not None else 270.0
        return make_moment_glyph(arc_degrees=float(arc))
    return pv.Sphere()


__all__ = [
    "PyVistaBackend",
    "PyVistaQtBackend",
    "mesh_layer_to_grid",
    "apply_visibility_mask",
    "cellblocks_from_grid",
    "mesh_layer_from_grid",
]
