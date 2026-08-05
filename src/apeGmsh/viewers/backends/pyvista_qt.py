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

import weakref
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pyvista as pv
from vtkmodules.vtkCommonDataModel import vtkPlane

from apeGmsh.viewers.scene_ir import (
    CellBlocks,
    ClipPlaneSpec,
    ColorSpec,
    GlyphLayer,
    LabelLayer,
    MeshLayer,
    PointSet,
    ScalarBarSpec,
    ScalarField,
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

#: Cell-type tokens whose layers pay an O(volume) surface re-extraction
#: inside ``vtkDataSetMapper`` on every dataset MTime bump. Layers
#: carrying any of these render a pre-extracted surface instead (the
#: render-surface fast path); 0/1/2-D-only layers stay on the direct
#: mapper — their extraction is already O(n_cells), so the fast path
#: would add bookkeeping for nothing. Line/vertex layers in particular
#: bypass the surface path entirely.
_VOLUME_TOKENS = frozenset({"tetra", "hexahedron", "wedge", "pyramid"})

#: Cell-data key stamped onto every render surface built by
#: :func:`extract_render_surface`, mapping each surface cell back to its
#: source volumetric cell. Namespaced (not ``vtkOriginalCellIds``) so
#: the pick backend remaps ONLY surfaces this module created — the mesh
#: viewer's own extracted surfaces carry plain ``vtkOriginalCellIds``
#: and already resolve in surface-id space.
PICK_ORIG_CELL_IDS = "_apegmsh_orig_cell_ids"


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


def _carried_fields(grid: pv.UnstructuredGrid) -> "list[ScalarField]":
    """Every scalar a sliced grid should hand back as IR.

    VTK's own bookkeeping arrays (``vtkGhostType``, ``vtkOriginal*``)
    are dropped: they describe the *source* dataset's rows and mean
    nothing on the cross-section. Non-scalar (vector) arrays are
    dropped too — :class:`ScalarField` is 1-D by contract.
    """
    fields: list[ScalarField] = []
    for location, data in (
        ("point", grid.point_data), ("cell", grid.cell_data),
    ):
        for name in list(data.keys()):
            if str(name).startswith("vtk"):
                continue
            values = np.asarray(data[name])
            if values.ndim != 1:
                continue
            fields.append(ScalarField(str(name), values, location))
    return fields


def extract_render_surface(
    grid: Any,
) -> "Optional[tuple[Any, np.ndarray, np.ndarray]]":
    """Extract the boundary surface ``vtkDataSetMapper`` would render.

    The render-surface fast path's core: ``vtkDataSetMapper`` re-runs
    its internal surface extraction whenever its input grid's MTime
    changes — even for a scalars-only update — making every animation
    step O(volume) (measured 67 ms/step at 1M tets vs 1.1 ms on a
    pre-extracted surface). Rendering the extraction's output directly
    and scattering updates onto it makes the per-step cost O(surface).

    ``vtkDataSetSurfaceFilter`` is deliberate: it is the mapper's own
    internal filter, so its ghost-cell semantics (HIDDENCELL faces
    dropped, interior faces revealed, DUPLICATECELL kept) match the
    volumetric render pixel-for-pixel — and it measured ~1.8x faster
    than ``vtkGeometryFilter(FastMode)`` here (75 vs 137 ms at 1M
    tets). Pass-through ids give the two scatter maps:

    * ``vtkOriginalPointIds`` — surface point row -> volume point row;
    * ``vtkOriginalCellIds`` — surface cell -> volume cell, also
      stamped as :data:`PICK_ORIG_CELL_IDS` so the pick backend can
      translate picked surface cells back to volumetric cell ids
      before a ``PickHit`` crosses the seam.

    Returns ``(surface, point_ids, cell_ids)``, or ``None`` when the
    extraction cannot provide the maps (caller falls back to the
    direct volumetric path).
    """
    try:
        from vtkmodules.vtkFiltersGeometry import vtkDataSetSurfaceFilter

        f = vtkDataSetSurfaceFilter()
        f.SetInputData(grid)
        f.SetPassThroughCellIds(True)
        f.SetPassThroughPointIds(True)
        f.Update()
        surface = pv.wrap(f.GetOutput())
        point_ids = np.asarray(
            surface.point_data["vtkOriginalPointIds"], dtype=np.int64,
        )
        cell_ids = np.asarray(
            surface.cell_data["vtkOriginalCellIds"], dtype=np.int64,
        )
    except Exception:
        return None
    surface.cell_data[PICK_ORIG_CELL_IDS] = cell_ids
    return surface, point_ids, cell_ids


def _wants_render_surface(layer: MeshLayer) -> bool:
    """Whether a mesh layer takes the render-surface fast path.

    Only layers with at least one 3-D cell block benefit (see
    :data:`_VOLUME_TOKENS`); silhouette layers stay volumetric because
    ``add_silhouette`` binds its own pipeline to the grid.
    """
    return (
        not layer.silhouette
        and any(t in _VOLUME_TOKENS for t in layer.cells.blocks)
    )


class RenderSurface:
    """Off-seam render-surface bundle for the substrate actor pair.

    The results viewer's substrate fill / wireframe are added straight
    to the plotter (ADR 0083 Part 3), so they cannot ride the
    ``_PvHandle`` fast path — this bundle is their equivalent: the
    extracted surface both actors map, the two scatter maps, and a
    snapshot of the volume ghost bytes the surface was extracted under
    (so :func:`refresh_render_surface` can skip no-op refreshes).
    """

    __slots__ = (
        "surface", "point_ids", "cell_ids", "ghosts",
        "outline", "outline_rows", "outline_angle",
    )

    def __init__(
        self,
        surface: Any,
        point_ids: "np.ndarray",
        cell_ids: "np.ndarray",
        ghosts: "Optional[np.ndarray]",
    ) -> None:
        self.surface = surface
        self.point_ids = point_ids
        self.cell_ids = cell_ids
        self.ghosts = ghosts
        # ADR 0089 D1 — static feature-edge outline of ``surface``.
        # ``outline`` is the edge polydata the outline actor maps;
        # ``outline_rows`` maps each outline point to its GRID row so
        # the DEFORM lane's ``sync_render_surface_points`` can move the
        # outline with the surface (no new pump, no per-frame
        # extraction). ``None`` until ``build_outline_edges`` runs.
        self.outline: Any = None
        self.outline_rows: "Optional[np.ndarray]" = None
        self.outline_angle: float = 25.0


def _grid_ghosts(grid: Any) -> "Optional[np.ndarray]":
    try:
        return np.asarray(grid.cell_data["vtkGhostType"]).copy()
    except (KeyError, IndexError):
        return None


def build_render_surface(grid: Any) -> "Optional[RenderSurface]":
    """Build the substrate's :class:`RenderSurface` from its grid.

    ``None`` when extraction cannot provide the scatter maps — the
    caller renders the volumetric grid as before.
    """
    extracted = extract_render_surface(grid)
    if extracted is None:
        return None
    surface, point_ids, cell_ids = extracted
    return RenderSurface(surface, point_ids, cell_ids, _grid_ghosts(grid))


def refresh_render_surface(
    grid: Any, rs: "RenderSurface", actors: "Sequence[Any]",
) -> bool:
    """Re-extract ``rs`` from ``grid`` after a ghost change and swap it
    into ``actors``' mappers. Returns whether a refresh happened.

    Hidden cells change WHICH faces exist (dropped faces + revealed
    interior), so a ghost change is the one update that cannot be
    scattered onto the existing surface. Compares the ghost bytes
    first: per-cell visibility events are rare but may fire without an
    actual change, and the compare is a cheap memcmp next to the
    O(volume) re-extraction.
    """
    ghosts = _grid_ghosts(grid)
    if (
        (ghosts is None and rs.ghosts is None)
        or (
            ghosts is not None and rs.ghosts is not None
            and np.array_equal(ghosts, rs.ghosts)
        )
    ):
        return False
    extracted = extract_render_surface(grid)
    if extracted is None:
        return False
    rs.surface, rs.point_ids, rs.cell_ids = extracted
    rs.ghosts = ghosts
    for actor in actors:
        try:
            actor.GetMapper().SetInputData(rs.surface)
        except Exception:
            pass
    return True


_OUTLINE_ROW_ARRAY = "_ape_outline_row"


def _extract_outline(
    surface: Any, feature_angle: float,
) -> "Optional[tuple[Any, np.ndarray]]":
    """Static feature edges of ``surface`` + their surface-row map.

    ADR 0089 D1/D4 — boundary edges plus dihedral creases above
    ``feature_angle``, extracted ONCE (camera-independent; deliberately
    NOT ``vtkPolyDataSilhouette``, whose per-camera-tick recompute is
    why the model viewer's silhouettes need LOD hiding). Returns
    ``None`` when the surface has no such edges (e.g. a 1-D beam
    model, whose surface is lines).
    """
    try:
        if surface is None or surface.n_points == 0:
            return None
        surface.point_data[_OUTLINE_ROW_ARRAY] = np.arange(
            surface.n_points, dtype=np.int64,
        )
        try:
            edges = surface.extract_feature_edges(
                feature_angle=float(feature_angle),
                boundary_edges=True,
                feature_edges=True,
                manifold_edges=False,
                non_manifold_edges=False,
            )
        finally:
            # Keep the render surface clean — the tag array served the
            # extraction only.
            try:
                del surface.point_data[_OUTLINE_ROW_ARRAY]
            except KeyError:
                pass
        if edges is None or edges.n_points == 0 or edges.n_lines == 0:
            return None
        rows = np.asarray(
            edges.point_data[_OUTLINE_ROW_ARRAY], dtype=np.int64,
        ).copy()
        del edges.point_data[_OUTLINE_ROW_ARRAY]
    except Exception:
        return None
    return edges, rows


def build_outline_edges(
    rs: "RenderSurface", feature_angle: float,
) -> "Optional[Any]":
    """Build the substrate's feature-edge outline (ADR 0089 D1).

    Extracts the outline of ``rs.surface`` and stores it on ``rs``
    (``outline`` + ``outline_rows`` in GRID-row space) so the DEFORM
    lane's ``sync_render_surface_points`` scatters deformed points
    onto it alongside the surface, and ghost re-extractions can
    rebuild it in place (:func:`refresh_outline_edges`). Returns the
    outline polydata for the caller's actor, or ``None`` when the
    surface yields no feature edges.
    """
    extracted = _extract_outline(rs.surface, feature_angle)
    rs.outline_angle = float(feature_angle)
    if extracted is None:
        rs.outline = None
        rs.outline_rows = None
        return None
    edges, surface_rows = extracted
    rs.outline = edges
    rs.outline_rows = np.asarray(rs.point_ids, dtype=np.int64)[surface_rows]
    return edges


def refresh_outline_edges(rs: "RenderSurface") -> bool:
    """Re-extract ``rs.outline`` after a render-surface re-extraction.

    A ghost (per-cell visibility) change alters WHICH faces exist, so
    the outline — like the surface it wraps — cannot be scattered and
    must re-extract. In-place (``copy_from``) so the outline actor's
    mapper keeps its bound dataset. No-op when ``rs`` never grew an
    outline. Returns whether the outline changed.
    """
    if rs.outline is None:
        return False
    extracted = _extract_outline(rs.surface, rs.outline_angle)
    if extracted is None:
        # Every feature edge vanished (e.g. all cells hidden) — keep
        # the polydata object (the actor maps it) but empty it.
        try:
            rs.outline.copy_from(pv.PolyData())
        except Exception:
            return False
        rs.outline_rows = np.empty(0, dtype=np.int64)
        return True
    edges, surface_rows = extracted
    try:
        rs.outline.copy_from(edges)
    except Exception:
        return False
    rs.outline_rows = np.asarray(rs.point_ids, dtype=np.int64)[surface_rows]
    return True


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

    # ``__weakref__`` is explicit because of ``__slots__``: the backend's
    # clip-plane registry holds handles weakly (ADR 0083 Part 2), and a
    # slotted class without this slot cannot be weak-referenced at all.
    #
    # ``dataset`` is ALWAYS the volumetric grid — ``slice_layer`` cuts
    # it (a surface slice yields lines, not a filled cap) and picking
    # resolves against its cell ids. The render-surface fast path adds
    # its state as separate fields; it never repurposes ``dataset``:
    #
    # * ``render_surface`` — the extracted surface the actor actually
    #   maps (``None`` = direct volumetric path);
    # * ``surf_point_ids`` / ``surf_cell_ids`` — the scatter maps
    #   (surface row -> volume row);
    # * ``surf_hidden`` — the visibility mask the surface was
    #   extracted under, so updates re-extract only on mask change.
    __slots__ = (
        "layer_id", "actor", "dataset", "kind", "clip_exempt",
        "render_surface", "surf_point_ids", "surf_cell_ids", "surf_hidden",
        "__weakref__",
    )

    def __init__(
        self,
        layer_id: str,
        actor: Any,
        dataset: Any,
        kind: str,
        *,
        clip_exempt: bool = False,
    ) -> None:
        self.layer_id = layer_id
        self.actor = actor
        self.dataset = dataset
        self.kind = kind
        self.clip_exempt = clip_exempt
        self.render_surface: Any = None
        self.surf_point_ids: Any = None
        self.surf_cell_ids: Any = None
        self.surf_hidden: frozenset = frozenset()


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
        # ADR 0083 Part 2 — the handle registry the backend used to
        # lack, so ``set_clip_planes`` has a live set of mappers to
        # re-stamp. Weak by value: a handle the domain layer dropped
        # takes its entry with it, and the registry never keeps an
        # actor (or its dataset) alive past its owner.
        self._handles: "weakref.WeakValueDictionary[str, _PvHandle]" = (
            weakref.WeakValueDictionary()
        )
        self._clip_planes: tuple[ClipPlaneSpec, ...] = ()

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
            # Render-surface fast path: the volume write above is a
            # cheap memcpy (nothing maps the grid), the actor renders
            # the pre-extracted surface — scatter the update onto it.
            if handle.render_surface is not None:
                self._sync_render_surface(handle, layer)
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
        handle.clip_exempt = new.clip_exempt
        handle.render_surface = new.render_surface
        handle.surf_point_ids = new.surf_point_ids
        handle.surf_cell_ids = new.surf_cell_ids
        handle.surf_hidden = new.surf_hidden
        # ``new`` is discarded here, and the clip registry holds its
        # handles weakly — so the entry ``add_layer`` just made would
        # die with it, leaving the surviving handle unregistered and
        # the next ``set_clip_planes`` blind to this layer.
        self._register_handle(handle)

    def remove_layer(self, handle: _PvHandle) -> None:
        self._handles.pop(handle.layer_id, None)
        if handle.actor is not None:
            self._plotter.remove_actor(handle.actor)
            handle.actor = None

    def set_visibility(self, handle: _PvHandle, mask: VisibilityMask) -> None:
        if handle.dataset is not None and handle.kind == "mesh":
            apply_visibility_mask(handle.dataset, mask)
            if handle.render_surface is not None:
                hidden = frozenset(mask.hidden_cells)
                if hidden != handle.surf_hidden:
                    self._rebuild_render_surface(handle)
                    handle.surf_hidden = hidden

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
                if color.array_name and handle.render_surface is not None:
                    # The mapper reads from the render surface, so the
                    # active-scalars switch must land there too.
                    handle.render_surface.set_active_scalars(color.array_name)
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

    def set_clip_planes(self, planes: "Sequence[ClipPlaneSpec]") -> None:
        """Cut the scene with ``planes`` (ADR 0083 Part 2).

        Two halves, and both matter. The set is applied to every mesh /
        glyph mapper registered right now, and it is *remembered* so
        :meth:`_add_mesh_layer` / :meth:`_add_glyph_layer` can stamp it
        onto everything created later — including the actor
        :meth:`update_layer`'s rebuild path re-creates on every glyph
        step, which is where a one-shot attach silently loses the cut.

        Label handles are skipped: ``AddClippingPlane`` on a 2D text
        mapper is at best a no-op, and a half-clipped label would be a
        defect even if it worked. ``clip_exempt`` handles are skipped
        too (ADR 0083 S2): a plane gizmo sliced by its own plane — or
        by another plane — is useless.
        """
        self._clip_planes = tuple(planes or ())
        for handle in list(self._handles.values()):
            if handle.kind == "label" or handle.clip_exempt:
                continue
            apply_clip_planes(handle.actor, self._clip_planes)

    def viewport_size(self) -> tuple[int, int]:
        """Render-target size in pixels — the layout's only input.

        The ``LegendController`` derives legend boxes from font metrics
        in pixels (ADR 0081 Part 3), so it needs the viewport it is
        placing them in.
        """
        try:
            w, h = self._plotter.window_size
            return int(w), int(h)
        except Exception:
            return 1024, 768

    def add_scalar_bar(self, handle: _PvHandle, spec: ScalarBarSpec) -> None:
        """Project an already-resolved legend layout onto a bar actor.

        Every number here comes from the spec: placement, size and font
        sizes are the ``LegendController``'s job because only it can see
        all the legends at once (ADR 0081 Part 3). This method computes
        nothing and consults no theme.

        ``interactive=False`` is deliberate and load-bearing.
        ``interactive=True`` builds a ``vtkScalarBarWidget`` that
        observes the interactor at VTK priority 0.5, while the pick
        engine aborts every plain left-button press at priority 10 — so
        the widget could never receive a click, and its geometry was a
        second copy of the layout that nothing read back. Drag and
        resize live in the controller's own interactor instead.
        """
        actor = handle.actor
        if actor is None:
            return
        try:
            mapper = actor.GetMapper()
        except Exception:
            return
        # Drop any prior bar for this legend before re-adding.
        self.remove_scalar_bar(spec.key)
        try:
            # The title is always pyvista's registry key (the controller
            # keeps titles unique across legends); a spec carrying its
            # own title anchor merely stops VTK from *drawing* it.
            bar = self._plotter.add_scalar_bar(
                title=spec.title, mapper=mapper, interactive=False,
                fmt=spec.fmt, vertical=spec.vertical,
                width=spec.extent[0], height=spec.extent[1],
                position_x=spec.anchor[0], position_y=spec.anchor[1],
                title_font_size=spec.title_pt, label_font_size=spec.label_pt,
                n_labels=spec.n_labels,
            )
            title_actor = None
            if spec.title_anchor is not None:
                # ``vtkScalarBarActor`` has no draw-title flag; blanking
                # the actor's title is how you stop it rendering one.
                # pyvista's registry key is the title we passed above and
                # is unaffected, so removal still works by it.
                bar.SetTitle("")
                title_actor = self._add_bar_title(spec, bar)
            self._scalar_bars[spec.key] = (spec.title, bar, title_actor)
        except Exception:
            pass

    def _add_bar_title(self, spec: ScalarBarSpec, bar: Any) -> Any:
        """Draw a horizontal legend's title in its reserved band.

        The colour and font family are copied from the bar's own tick
        labels, which pyvista already themed — so the two always agree,
        and the render layer needs no import from ``ui/``.
        """
        import vtk

        actor = vtk.vtkTextActor()
        actor.SetInput(spec.title)
        actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        actor.SetPosition(*spec.title_anchor)
        prop = actor.GetTextProperty()
        prop.SetFontSize(int(spec.title_pt))
        prop.SetJustificationToCentered()
        prop.SetVerticalJustificationToBottom()
        try:
            labels = bar.GetLabelTextProperty()
            prop.SetColor(*labels.GetColor())
            prop.SetFontFamily(labels.GetFontFamily())
            prop.SetBold(labels.GetBold())
            prop.SetItalic(labels.GetItalic())
            prop.SetShadow(labels.GetShadow())
        except Exception:
            pass
        self._plotter.renderer.AddActor2D(actor)
        return actor

    def move_scalar_bar(self, bar_key: str, spec: ScalarBarSpec) -> bool:
        """Re-place an existing bar in place; ``False`` if not possible.

        A drag emits a mouse-move event per frame, and re-creating the
        bar actor on each one destroys and rebuilds a VTK actor 40+
        times a second — visible as flicker and felt as lag. Geometry-
        only changes (which is all a move or a resize is) therefore
        poke the existing actor instead.
        """
        entry = self._scalar_bars.get(bar_key)
        if entry is None:
            return False
        _title, bar, title_actor = entry
        try:
            bar.SetPosition(*spec.anchor)
            bar.SetWidth(spec.extent[0])
            bar.SetHeight(spec.extent[1])
            bar.GetTitleTextProperty().SetFontSize(int(spec.title_pt))
            bar.GetLabelTextProperty().SetFontSize(int(spec.label_pt))
            if title_actor is not None and spec.title_anchor is not None:
                title_actor.SetPosition(*spec.title_anchor)
                title_actor.GetTextProperty().SetFontSize(int(spec.title_pt))
        except Exception:
            return False
        return True

    def remove_scalar_bar(self, bar_key: str) -> None:
        entry = self._scalar_bars.pop(bar_key, None)
        if entry is None:
            return
        title, _bar, title_actor = entry
        try:
            self._plotter.remove_scalar_bar(title)
        except Exception:
            pass
        if title_actor is not None:
            try:
                self._plotter.renderer.RemoveActor2D(title_actor)
            except Exception:
                pass

    def set_scalar_bar_format(self, bar_key: str, fmt: str) -> None:
        entry = self._scalar_bars.get(bar_key)
        if entry is not None:
            try:
                entry[1].SetLabelFormat(fmt)
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

    def display_to_world_ray(
        self, x: float, y: float,
    ) -> "Optional[tuple[tuple[float, float, float], tuple[float, float, float]]]":
        """``(origin, unit direction)`` of the pick ray under a pixel.

        ``x`` / ``y`` are display coordinates with a bottom-left origin
        — exactly what ``GetEventPosition()`` reports, so an interactor
        can pass them straight through. Off the ``RenderBackend``
        Protocol (like ``picking()``): it is the gizmo interactor's
        hit-test service (ADR 0083 S2), not part of the render
        contract, and it lives here because unprojecting a pixel is a
        renderer/camera operation the domain layer must not reimplement
        (INV-2). ``None`` when the renderer cannot unproject.
        """
        try:
            renderer = self._plotter.renderer
            points = []
            for depth in (0.0, 1.0):
                renderer.SetDisplayPoint(float(x), float(y), depth)
                renderer.DisplayToWorld()
                wx, wy, wz, w = renderer.GetWorldPoint()
                if w == 0.0:
                    w = 1.0
                points.append((wx / w, wy / w, wz / w))
        except Exception:
            return None
        near, far = points
        direction = tuple(f - n for n, f in zip(near, far))
        mag = float(np.sqrt(sum(c * c for c in direction)))
        if mag <= 0.0:
            return None
        return near, tuple(c / mag for c in direction)

    def slice_layer(
        self,
        handle: _PvHandle,
        plane: ClipPlaneSpec,
        *,
        layer_id: str,
        trim: "Sequence[ClipPlaneSpec]" = (),
    ) -> "Optional[MeshLayer]":
        """``handle``'s cross-section at ``plane``, as neutral scene-IR.

        The cut-face pass (ADR 0083 S3). Render-time clipping discards
        *fragments*, so a section plane through a solid opens into an
        empty box — there is no geometry at the plane to colour. The
        only thing that puts the field on the cut is a real dataset
        slice, and a dataset slice is VTK work: it lives here, behind
        the seam (INV-2), and the renderer that drives it never sees a
        filter.

        The source is the layer's **own** dataset, which for a contour
        is its volumetric substrate submesh — so the scalars the cutter
        interpolates onto the polygon are the contour's own values at
        the current step, deformation included, and the caller can
        paint them with the contour's own LUT. Slicing anything else
        would be a differently-sampled picture of the same quantity.

        ``trim`` are the *other* active half-spaces. A cut face is
        exempt from mapper clipping (it is coplanar with its own plane,
        which makes the survival of its fragments a coin toss), so a
        corner cut has to trim it geometrically here instead.

        Off the ``RenderBackend`` Protocol, like
        :meth:`display_to_world_ray` (ADR 0042 INV-3): view-only
        backends need not implement it and callers probe for it. Every
        array is carried through except VTK's own bookkeeping
        (``vtkGhostType`` and friends). ``None`` when the plane misses
        the dataset entirely.
        """
        dataset = getattr(handle, "dataset", None)
        if dataset is None or int(getattr(dataset, "n_cells", 0)) == 0:
            return None
        try:
            cut = dataset.slice(
                normal=tuple(plane.normal),
                origin=tuple(plane.origin),
                generate_triangles=True,
            )
            for other in trim or ():
                if cut is None or cut.n_cells == 0:
                    break
                cut = cut.clip(
                    normal=tuple(other.normal),
                    origin=tuple(other.origin),
                    invert=False,
                )
            if cut is None or cut.n_cells == 0:
                return None
            grid = (
                cut.extract_surface().triangulate()
                .cast_to_unstructured_grid()
            )
        except Exception:
            return None
        if grid.n_cells == 0:
            return None
        return MeshLayer(
            layer_id=layer_id,
            points=PointSet(np.asarray(grid.points)),
            cells=cellblocks_from_grid(grid),
            fields=tuple(_carried_fields(grid)),
        )

    # -- internals ----------------------------------------------------

    def _register_handle(self, handle: _PvHandle) -> _PvHandle:
        """Track ``handle`` for :meth:`set_clip_planes`, and stamp it.

        Stamping happens at creation, not after: it is the only way an
        actor added while a cut is live — a diagram attached later, a
        second geometry materialized later, a glyph actor rebuilt on
        the next animation step — arrives already cut. ``clip_exempt``
        layers (ADR 0083 S2 — plane gizmos, slice 3's cut-face) arrive
        unstamped for the same reason in reverse: a gizmo created
        while its own plane cuts must not be sliced by it.
        """
        self._handles[handle.layer_id] = handle
        if (
            self._clip_planes
            and handle.kind != "label"
            and not handle.clip_exempt
        ):
            apply_clip_planes(handle.actor, self._clip_planes)
        return handle

    def _sync_render_surface(self, handle: _PvHandle, layer: MeshLayer) -> None:
        """Scatter an in-place mesh update onto the render surface.

        O(surface): points and fields are gathered through the two
        scatter maps. A visibility-mask change is the one update that
        cannot be scattered (hidden cells change WHICH faces exist —
        dropped faces, revealed interior), so it re-extracts instead;
        the volume grid already carries the new ghost array (the caller
        ran ``apply_visibility_mask`` first).
        """
        hidden = frozenset(layer.visibility.hidden_cells)
        if hidden != handle.surf_hidden:
            self._rebuild_render_surface(handle)
            handle.surf_hidden = hidden
            return
        surface = handle.render_surface
        pid = handle.surf_point_ids
        cid = handle.surf_cell_ids
        surface.points = layer.points.coords[pid]
        for sf in layer.fields:
            if sf.location == "point":
                surface.point_data[sf.name] = sf.values[pid]
            else:
                surface.cell_data[sf.name] = sf.values[cid]

    def _rebuild_render_surface(self, handle: _PvHandle) -> None:
        """Re-extract the render surface from the (updated) volume grid
        and swap it into the actor's mapper — same actor, new input.

        O(volume), paid per visibility-mask change rather than per
        frame. The extraction passes every point/cell array through, so
        no scatter is needed afterwards; the active-scalars selection is
        carried over so ``by_array`` colouring survives the swap.
        """
        extracted = extract_render_surface(handle.dataset)
        if extracted is None:
            return
        surface, point_ids, cell_ids = extracted
        old = handle.render_surface
        try:
            if old is not None:
                pa = old.point_data.active_scalars_name
                ca = old.cell_data.active_scalars_name
                if pa and pa in surface.point_data:
                    surface.point_data.active_scalars_name = pa
                if ca and ca in surface.cell_data:
                    surface.cell_data.active_scalars_name = ca
        except Exception:
            pass
        handle.render_surface = surface
        handle.surf_point_ids = point_ids
        handle.surf_cell_ids = cell_ids
        try:
            handle.actor.GetMapper().SetInputData(surface)
        except Exception:
            pass

    def _add_mesh_layer(self, layer: MeshLayer) -> _PvHandle:
        grid = mesh_layer_to_grid(layer)
        # Render-surface fast path: layers with 3-D cells render the
        # pre-extracted surface (O(surface) per animation step instead
        # of the mapper's O(volume) re-extraction). ``handle.dataset``
        # stays the volumetric grid regardless — slicing and picking
        # resolve against it.
        surface = point_ids = cell_ids = None
        if _wants_render_surface(layer):
            extracted = extract_render_surface(grid)
            if extracted is not None:
                surface, point_ids, cell_ids = extracted
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
        if layer.line_width is not None:
            kwargs["line_width"] = float(layer.line_width)
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
        actor = self._plotter.add_mesh(
            surface if surface is not None else grid, **kwargs,
        )
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
        handle = _PvHandle(
            layer.layer_id, actor, grid, "mesh",
            clip_exempt=layer.clip_exempt,
        )
        if surface is not None:
            handle.render_surface = surface
            handle.surf_point_ids = point_ids
            handle.surf_cell_ids = cell_ids
            handle.surf_hidden = frozenset(layer.visibility.hidden_cells)
        return self._register_handle(handle)

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
        return self._register_handle(
            _PvHandle(
                layer.layer_id, actor, glyphed, "glyph",
                clip_exempt=layer.clip_exempt,
            ),
        )

    def _add_label_layer(self, layer: LabelLayer) -> _PvHandle:
        actor = self._plotter.add_point_labels(
            layer.positions.coords, list(layer.texts)
        )
        # Registered like any other layer (so ``remove_layer`` stays
        # symmetric) but never stamped — see :meth:`_register_handle`.
        return self._register_handle(
            _PvHandle(layer.layer_id, actor, None, "label"),
        )


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


def apply_clip_planes(actor: Any, planes: "Sequence[ClipPlaneSpec]") -> None:
    """Replace ``actor``'s mapper clip set with ``planes`` (ADR 0083).

    The one place a :class:`ClipPlaneSpec` becomes a ``vtkPlane``.
    Public because the results viewer's **off-seam** actors — the
    substrate fill / wireframe / node cloud, which are added straight
    to the plotter and never pass through ``add_layer`` — have to be
    cut by the same set, and must not grow a second implementation.

    Always clears first: this is a *replace*, so a plane the user
    deleted stops cutting. Non-fatal throughout — an actor with no
    mapper (or a mapper that refuses planes) renders uncut rather than
    taking the viewer down.
    """
    if actor is None:
        return
    try:
        mapper = actor.GetMapper()
    except Exception:
        return
    if mapper is None:
        return
    try:
        mapper.RemoveAllClippingPlanes()
        for spec in planes or ():
            plane = vtkPlane()
            plane.SetOrigin(*spec.origin)
            plane.SetNormal(*spec.normal)
            mapper.AddClippingPlane(plane)
    except Exception:
        pass


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
    "RenderSurface",
    "apply_clip_planes",
    "build_outline_edges",
    "build_render_surface",
    "extract_render_surface",
    "mesh_layer_to_grid",
    "apply_visibility_mask",
    "cellblocks_from_grid",
    "mesh_layer_from_grid",
    "refresh_outline_edges",
    "refresh_render_surface",
    "PICK_ORIG_CELL_IDS",
]
