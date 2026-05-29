"""SectionCutDiagram — render a planned :class:`SectionCutDef` cut plane.

A static layer: the cut is geometry defined against the *input* model,
so ``update_to_step`` is a no-op and ``sync_substrate_points`` is too
(decision D6 — cuts ignore deformation).

Render seam (ADR 0042, R-B Wave 3 #3 — the final diagram). The cut
emits its geometry as :mod:`scene_ir <apeGmsh.viewers.scene_ir>` value
types through ``self._backend``; the diagram holds no VTK objects:

* the cut face is a single-polygon :class:`MeshLayer` (``"polygon"``
  cell token) carrying ``back_color`` for the two-tone front/back
  faces — the additive IR widening this migration introduced. The
  *kept* side (per :attr:`SectionCutDef.side`) shows ``kept_color``;
  the *discarded* side shows ``discarded_color`` via the backend's
  backface property;
* the side-aware normal arrow is a single-glyph :class:`GlyphLayer`
  (``kind="arrow"``) at the quad centroid, pointing into the kept
  half-space — the orbit-edge-on fallback so the side stays readable
  when both face colors collapse to a line;
* the filter-elements highlight (Phase 1b toggle) is a substrate
  submesh :class:`MeshLayer`.

Quad extent (decision D2)
-------------------------
* ``cut.bounding_polygon`` set → render the polygon vertices directly.
  That polygon **is** what the STKO kernel filters against, so the
  rendered region matches reality.
* ``cut.bounding_polygon`` is ``None`` → clip the cut plane to the
  AABB of the *filter elements*, **not** the full model AABB. Pulled
  from the FEM mesh via the OpenSees-tag → FEM-eid inverse on the
  director's :class:`apeGmsh.cuts.FemToOpsTagMap`. Showing a quad
  that spans the whole model when the cut actually integrates a
  handful of elements would be a visual fiction; a filter-AABB quad
  sits over the region the cut really touches.

Tag-map dependency
------------------
Attach requires a :class:`apeGmsh.cuts.FemToOpsTagMap` (built from
``model.h5``) to translate ``cut.element_ids`` (OpenSees tags) back
into FEM element ids. The director carries the map; the diagram
borrows a reference at construction. Without it, attach raises
``RuntimeError`` with a clear pointer to
``director.set_model(opensees_model)``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from ._base import Diagram, DiagramSpec, NoDataError
from ._styles import SectionCutStyle
from ..scene_ir import CellBlocks, ColorSpec, GlyphLayer, MeshLayer, PointSet

if TYPE_CHECKING:
    from numpy import ndarray
    from apeGmsh.cuts import FemToOpsTagMap, SectionCutDef
    from apeGmsh.results.Results import Results
    from apeGmsh.viewers.data import ViewerData
    from ..scene.fem_scene import FEMSceneData


class SectionCutDiagram(Diagram):
    """Render a planned section cut as a two-tone quad + normal arrow."""

    kind = "section_cut"
    # No Results-data dependency — cuts are FEM geometry, not solver
    # output. Catalog wiring is intentionally absent in Phase 1 (decision
    # D9: programmatic ingress only).
    topology = "section_cut"

    def __init__(
        self,
        spec: DiagramSpec,
        results: "Results",
        *,
        tag_map: "Optional[FemToOpsTagMap]" = None,
    ) -> None:
        if not isinstance(spec.style, SectionCutStyle):
            raise TypeError(
                "SectionCutDiagram requires a SectionCutStyle; "
                f"got {type(spec.style).__name__}."
            )
        super().__init__(spec, results)
        self._tag_map: "Optional[FemToOpsTagMap]" = tag_map

        # Attach-time products — emitted SceneLayers + backend handles.
        self._quad_layer: Optional[MeshLayer] = None
        self._quad_handle: Any = None
        self._quad_verts: "Optional[ndarray]" = None   # oriented (N, 3)
        self._arrow_layer: Optional[GlyphLayer] = None
        self._arrow_handle: Any = None
        # FEM element ids the cut filter resolves to — cached for the
        # filter highlight overlay. Cached at attach so the toggle is
        # a single extract_cells call.
        self._filter_fem_eids: "Optional[ndarray]" = None
        self._filter_layer: Optional[MeshLayer] = None
        self._filter_handle: Any = None
        # Runtime state for the highlight toggle — bootstraps from
        # ``style.show_filter_initially`` at attach.
        self._show_filter: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def attach(
        self,
        plotter: Any,
        view: "ViewerData",
        scene: "FEMSceneData | None" = None,
    ) -> None:
        if scene is None:
            raise RuntimeError(
                "SectionCutDiagram.attach requires a FEMSceneData (the "
                "viewer's substrate mesh)."
            )
        if self._tag_map is None:
            raise RuntimeError(
                "SectionCutDiagram.attach requires a FemToOpsTagMap. "
                "Bind the director to a Results that carries an "
                "/opensees/ orientation zone — typically via "
                "director.bind_results(results), then call "
                "director.add_section_cut(cut_def) which picks the "
                "tag-map up from the director's bound state."
            )
        super().attach(plotter, view, scene)

        style: SectionCutStyle = self.spec.style    # type: ignore[assignment]
        cut: "SectionCutDef" = style.cut

        # Resolve OpenSees tags → FEM eids. Failure here is a real
        # error: it means cut.element_ids doesn't match the loaded
        # model.h5 (different run, stale def, etc.).
        try:
            fem_eids = np.asarray(
                self._tag_map.fem_eids_for_ops_tags(cut.element_ids),
                dtype=np.int64,
            )
        except KeyError as exc:
            raise NoDataError(
                f"SectionCutDiagram: cut.element_ids do not match the "
                f"loaded model.h5 — {exc}. Check that the director's "
                f"bound Results points at the same run the cut was "
                f"built against (director.bind_results(results))."
            ) from exc
        self._filter_fem_eids = fem_eids

        # Build the quad vertices.
        if cut.bounding_polygon is not None:
            verts = np.asarray(cut.bounding_polygon, dtype=np.float64)
        else:
            verts = self._quad_from_filter_aabb(view, fem_eids, cut)

        # Orient the polygon so the front face points to the kept side.
        verts_oriented = self._orient_for_kept_side(verts, cut)
        self._quad_verts = verts_oriented

        self._quad_layer = self._build_quad_layer(verts_oriented, style)
        self._quad_handle = self._backend.add_layer(self._quad_layer)

        if style.show_normal_arrow:
            self._arrow_layer = self._build_arrow_layer(
                style, verts_oriented, cut, scene,
            )
            self._arrow_handle = self._backend.add_layer(self._arrow_layer)

        # Bootstrap the filter highlight from the style's initial flag.
        # Done after the main layers are emitted so the toggle's add-layer
        # call lands in a stable state.
        if style.show_filter_initially:
            self.set_show_filter(True)

    def update_to_step(self, step_index: int) -> None:
        # Cuts are static — no per-step refresh. Decision D6.
        return

    def sync_substrate_points(
        self,
        deformed_pts: "ndarray | None",
        scene: "FEMSceneData",
    ) -> None:
        # Cuts are defined against the reference configuration; they
        # don't follow deformation. Decision D6.
        return

    def detach(self) -> None:
        self._remove_filter_highlight_layer()
        if self._backend is not None:
            for handle in (self._quad_handle, self._arrow_handle):
                if handle is not None:
                    self._backend.remove_layer(handle)
        self._quad_layer = None
        self._quad_handle = None
        self._quad_verts = None
        self._arrow_layer = None
        self._arrow_handle = None
        self._filter_fem_eids = None
        self._show_filter = False
        super().detach()

    # ------------------------------------------------------------------
    # Filter highlight (runtime toggle — Phase 1b)
    # ------------------------------------------------------------------

    def set_show_filter(self, show: bool) -> None:
        """Toggle the filter-elements highlight overlay.

        Builds a uniform-color actor over the substrate cells that the
        cut's element filter resolves to, so the user can see which
        elements the integration will sweep. Cheap to toggle — the
        overlay reuses cached FEM eids resolved at attach.

        No-op when called before attach or after detach.
        """
        new = bool(show)
        if new == self._show_filter:
            return
        self._show_filter = new
        if not self._attached:
            return
        if new:
            self._add_filter_highlight_layer()
        else:
            self._remove_filter_highlight_layer()

    @property
    def show_filter(self) -> bool:
        """Whether the filter-highlight overlay is currently visible."""
        return self._show_filter

    @property
    def filter_highlight_handle(self) -> Any:
        """Backend handle to the filter-highlight layer, or ``None``."""
        return self._filter_handle

    @property
    def filter_highlight_layer(self) -> Optional[MeshLayer]:
        """The emitted filter-highlight :class:`MeshLayer`, or ``None``."""
        return self._filter_layer

    def _add_filter_highlight_layer(self) -> None:
        """Build and emit the filter-highlight layer.

        Walks ``scene.element_id_to_cell`` to translate cached FEM
        eids into substrate cell indices, then extracts those cells
        into a uniform-color submesh and emits it as a MeshLayer.
        Silent no-op if any required state is missing — the toggle
        stays "on" so a later attach cycle can pick it up.
        """
        if (
            self._backend is None
            or self._scene is None
            or self._filter_fem_eids is None
            or self._filter_fem_eids.size == 0
        ):
            return
        if self._filter_handle is not None:
            return     # already present
        scene = self._scene
        cell_indices = np.fromiter(
            (
                scene.element_id_to_cell.get(int(e), -1)
                for e in self._filter_fem_eids
            ),
            dtype=np.int64,
            count=int(self._filter_fem_eids.size),
        )
        cell_indices = cell_indices[cell_indices >= 0]
        if cell_indices.size == 0:
            return
        # ``extract_cells`` is a grid method (no pyvista import); the
        # cellblocks bridge re-expresses the submesh as backend-neutral
        # IR. ``.tolist()`` keeps the extract_cells stub happy across
        # pyvista versions without changing runtime semantics.
        from ..backends.pyvista_qt import cellblocks_from_grid
        submesh = scene.grid.extract_cells(cell_indices.tolist())
        if submesh.n_cells == 0:
            return
        style: SectionCutStyle = self.spec.style    # type: ignore[assignment]
        self._filter_layer = MeshLayer(
            layer_id=self._highlight_layer_id(),
            points=PointSet(np.asarray(submesh.points)),
            cells=cellblocks_from_grid(submesh),
            color=ColorSpec(mode="solid", solid_rgb=style.highlight_color),
            opacity=style.highlight_opacity,
            show_edges=False,
            pickable=False,
        )
        self._filter_handle = self._backend.add_layer(self._filter_layer)

    def _remove_filter_highlight_layer(self) -> None:
        if self._filter_handle is None:
            return
        if self._backend is not None:
            self._backend.remove_layer(self._filter_handle)
        self._filter_handle = None
        self._filter_layer = None

    def _highlight_layer_id(self) -> str:
        return f"section_cut_highlight_{id(self):x}"

    # ------------------------------------------------------------------
    # Visibility
    # ------------------------------------------------------------------

    def set_visible(self, visible: bool) -> None:
        self._visible = visible
        if self._backend is None:
            return
        for handle in (
            self._quad_handle, self._arrow_handle, self._filter_handle,
        ):
            if handle is not None:
                self._backend.set_layer_visible(handle, bool(visible))

    # ------------------------------------------------------------------
    # Introspection (used by Phase 1b filter highlight + tests)
    # ------------------------------------------------------------------

    @property
    def cut(self) -> "SectionCutDef":
        style: SectionCutStyle = self.spec.style    # type: ignore[assignment]
        return style.cut

    @property
    def filter_fem_eids(self) -> "Optional[ndarray]":
        """FEM eids the cut filter resolves to (after attach)."""
        return self._filter_fem_eids

    @property
    def quad_layer(self) -> Optional[MeshLayer]:
        """The emitted cut-face :class:`MeshLayer`, or ``None``."""
        return self._quad_layer

    @property
    def arrow_layer(self) -> Optional[GlyphLayer]:
        """The emitted normal-arrow :class:`GlyphLayer`, or ``None``."""
        return self._arrow_layer

    @property
    def quad_verts(self) -> "Optional[ndarray]":
        """Oriented cut-face vertices (kept-side winding), or ``None``."""
        return self._quad_verts

    # ------------------------------------------------------------------
    # Internal — geometry construction
    # ------------------------------------------------------------------

    def _quad_from_filter_aabb(
        self,
        view: "ViewerData",
        fem_eids: "ndarray",
        cut: "SectionCutDef",
    ) -> "ndarray":
        """Clip the cut plane to the AABB of the filter elements.

        Returns four corners of a rectangle on the plane, listed CCW
        around the plane normal. Falls back to the substrate AABB if
        the FEM connectivity lookup fails for any reason — the quad
        becomes a coarse hint rather than aborting the whole diagram.
        """
        coords = self._coords_of_filter_elements(view, fem_eids)
        if coords.size == 0:
            # Defensive: if we couldn't pull any coords (mesh / fem
            # mismatch the user didn't anticipate), fall back to the
            # scene's bounding box so the user at least sees a plane.
            if self._scene is not None:
                pts = np.asarray(self._scene.grid.points, dtype=np.float64)
                coords = pts
        if coords.size == 0:
            # Shouldn't happen — substrate is always non-empty — but
            # don't crash silently.
            raise NoDataError(
                "SectionCutDiagram: could not compute a quad — no "
                "FEM coordinates available for the cut filter."
            )

        return _plane_rectangle_from_points(
            point=cut.plane_point_arr,
            normal=cut.plane_normal_arr,
            points=coords,
        )

    def _coords_of_filter_elements(
        self,
        view: "ViewerData",
        fem_eids: "ndarray",
    ) -> "ndarray":
        """Gather the world coords of every node referenced by ``fem_eids``.

        Walks each :class:`ElementGroup` in :attr:`view.elements`,
        masks its rows against ``fem_eids``, and unions the resulting
        node ids. Then looks up positions in :attr:`view.nodes.coords`
        via a single id-to-row scatter. Robust to mixed-topology meshes.
        """
        if fem_eids.size == 0:
            return np.zeros((0, 3), dtype=np.float64)

        wanted = set(int(e) for e in fem_eids)
        node_ids_set: set[int] = set()
        for group in view.elements:
            g_ids = np.asarray(group.ids, dtype=np.int64)
            mask = np.isin(g_ids, list(wanted))
            if not mask.any():
                continue
            for row in np.asarray(group.connectivity, dtype=np.int64)[mask]:
                for nid in row:
                    if int(nid) > 0:
                        node_ids_set.add(int(nid))
        if not node_ids_set:
            return np.zeros((0, 3), dtype=np.float64)

        node_ids = np.array(sorted(node_ids_set), dtype=np.int64)
        all_node_ids = np.asarray(view.nodes.ids, dtype=np.int64)
        all_coords = np.asarray(view.nodes.coords, dtype=np.float64)
        max_nid = int(max(int(all_node_ids.max(initial=0)),
                          int(node_ids.max()))) + 1
        nid_to_row = np.full(max_nid + 1, -1, dtype=np.int64)
        nid_to_row[all_node_ids] = np.arange(
            all_node_ids.size, dtype=np.int64,
        )
        crows = nid_to_row[node_ids]
        crows = crows[crows >= 0]
        return all_coords[crows].astype(np.float64, copy=False)

    def _orient_for_kept_side(
        self,
        verts: "ndarray",
        cut: "SectionCutDef",
    ) -> "ndarray":
        """Reverse winding if needed so the polygon front face points to
        the kept side.

        VTK considers the face whose vertex winding is CCW from the
        camera the "front". By choosing the winding such that
        ``cross(v1 - v0, v2 - v0)`` aligns with the kept-side normal,
        a camera on the kept side sees the front face.

        ``side="positive"`` → kept normal = ``+plane_normal``
        ``side="negative"`` → kept normal = ``-plane_normal``
        """
        if verts.shape[0] < 3:
            return verts
        n = cut.plane_normal_arr
        kept_dir = n if cut.side == "positive" else -n
        v0, v1, v2 = verts[0], verts[1], verts[2]
        face_normal = np.cross(v1 - v0, v2 - v0)
        if float(np.dot(face_normal, kept_dir)) < 0.0:
            return verts[::-1].copy()
        return verts

    # ------------------------------------------------------------------
    # Internal — SceneLayer construction
    # ------------------------------------------------------------------

    def _quad_layer_id(self) -> str:
        return f"section_cut_quad_{id(self):x}"

    def _arrow_layer_id(self) -> str:
        return f"section_cut_arrow_{id(self):x}"

    def _build_quad_layer(
        self, verts: "ndarray", style: SectionCutStyle,
    ) -> MeshLayer:
        """Emit the cut face as a single-polygon two-tone MeshLayer.

        One ``"polygon"`` cell spans the N oriented vertices; winding is
        preserved so the backend's backface property paints the discarded
        side. Decorative — ``pickable=False`` so picks fall through to the
        substrate behind the cut.
        """
        n = int(verts.shape[0])
        poly = np.arange(n, dtype=np.int64).reshape(1, n)
        return MeshLayer(
            layer_id=self._quad_layer_id(),
            points=PointSet(verts),
            cells=CellBlocks({"polygon": poly}),
            color=ColorSpec(mode="solid", solid_rgb=style.kept_color),
            back_color=style.discarded_color,
            opacity=style.quad_opacity,
            show_edges=style.show_edges,
            edge_color=style.edge_color,
            pickable=False,
        )

    def _build_arrow_layer(
        self,
        style: SectionCutStyle,
        verts: "ndarray",
        cut: "SectionCutDef",
        scene: "FEMSceneData",
    ) -> GlyphLayer:
        """Emit the side-aware normal arrow as a single-glyph GlyphLayer.

        The arrow sits at the quad centroid, oriented into the kept
        half-space, sized to ``normal_arrow_fraction`` of the model
        diagonal.
        """
        centroid = verts.mean(axis=0)
        direction = (
            cut.plane_normal_arr
            if cut.side == "positive"
            else -cut.plane_normal_arr
        )
        diag = float(getattr(scene, "model_diagonal", 0.0) or 0.0)
        if diag <= 0.0:
            diag = float(np.linalg.norm(
                np.asarray(scene.grid.bounds[1::2]) -
                np.asarray(scene.grid.bounds[0::2])
            ))
            if diag <= 0.0:
                diag = 1.0
        scale = float(style.normal_arrow_fraction) * diag
        return GlyphLayer(
            layer_id=self._arrow_layer_id(),
            positions=PointSet(np.asarray(centroid, dtype=np.float64).reshape(1, 3)),
            kind="arrow",
            orientations=np.asarray(direction, dtype=np.float64).reshape(1, 3),
            scales=np.array([scale], dtype=np.float64),
            color=ColorSpec(mode="solid", solid_rgb=style.kept_color),
        )


# ---------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------

def _plane_basis(normal: "ndarray") -> "tuple[ndarray, ndarray]":
    """Orthonormal in-plane basis ``(e1, e2)`` — same convention as
    :func:`apeGmsh.cuts._polygons._plane_basis`."""
    n = np.asarray(normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    a = np.eye(3)[int(np.argmin(np.abs(n)))]
    e1 = np.cross(n, a)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    e2 /= np.linalg.norm(e2)
    return e1, e2


def _plane_rectangle_from_points(
    *,
    point: "ndarray",
    normal: "ndarray",
    points: "ndarray",
) -> "ndarray":
    """Return a 4-vertex rectangle on the plane covering the projection
    of ``points``.

    Algorithm: project every point to the plane, express in an in-plane
    (e1, e2) basis, take the 2-D AABB, re-embed the four corners back
    in 3-D. Vertices come out CCW around the plane normal.
    """
    p = np.asarray(point, dtype=np.float64)
    n = np.asarray(normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    e1, e2 = _plane_basis(n)
    rel = np.asarray(points, dtype=np.float64) - p
    # Orthogonal projection to the plane: drop the normal component.
    rel_proj = rel - np.outer(rel @ n, n)
    u = rel_proj @ e1
    v = rel_proj @ e2
    umin, umax = float(u.min()), float(u.max())
    vmin, vmax = float(v.min()), float(v.max())
    # Degenerate AABB (single point or collinear projection): pad to a
    # small square so the quad has area. 1e-6 of the projection span is
    # below any sane mesh tolerance.
    if umax <= umin:
        umax = umin + 1.0
    if vmax <= vmin:
        vmax = vmin + 1.0
    corners_2d = np.array(
        [
            [umin, vmin],
            [umax, vmin],
            [umax, vmax],
            [umin, vmax],
        ],
        dtype=np.float64,
    )
    return p + corners_2d[:, 0:1] * e1 + corners_2d[:, 1:2] * e2
