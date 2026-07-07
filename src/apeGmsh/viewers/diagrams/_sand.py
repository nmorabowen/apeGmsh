"""SandDiagram — field-colored grain cloud filling solid volumes.

Surface contours only show a solid's field on its skin; the interior
is invisible without clipping. This diagram fills every 3-D element
with small "sand" grains — random interior points colored by the
nodal component interpolated at each grain — so the 3-D distribution
(a stress bulb, a plastic zone, a propagating wave front) reads at a
glance through the volume.

How it works
------------
At attach, each solid element gets a grain count proportional to its
volume (uniform spatial density regardless of mesh grading), and each
grain a random position in the element's parent domain. Evaluating
the element's shape functions at those natural coordinates yields one
weight row per grain that serves three duties:

* **position** — ``weights @ node_coords`` (attach + deform-follow);
* **value** — ``weights @ nodal_field`` (every step);
* both stay consistent under deformation for free.

Rendered as a vertex-cell point-cloud :class:`MeshLayer` (flat GL
points, ``pickable=False``) — the same cheap representation as the
fiber dot cloud, so per-step updates hit the backend's in-place fast
path. ``occludes_substrate = True`` not for z-fighting (the
contour's reason) but for visibility: the grains live strictly inside
the volume, so behind an opaque substrate fill the diagram would be
invisible; hiding the fill leaves the wireframe as the volume outline.

Optionally (``style.weight_by_value``) grain *density* encodes the
field: each grain draws a fixed random threshold at attach and is
hidden at steps where the normalized ``|value|`` at its location
falls below it — dense sand where the field is strong.

Coverage: every 3-D type in the shape-function catalog (tet4/10,
hex8/20/27, wedge6). Elements of unsupported 3-D types are skipped
LOUDLY (one aggregated :class:`WarnSandUnsupportedElements` per
attach) — a silently half-filled volume would misread as a field
feature.
"""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Callable, Optional

import numpy as np
from numpy import ndarray

from apeGmsh.fem._shape_functions import (
    compute_jacobian_dets,
    get_shape_functions,
)

from ._base import Diagram, DiagramSpec, NoDataError
from ._kinds import register_diagram_kind
from ._scalar_color_support import ScalarColorSupport
from ._styles import SandStyle
from ..scene_ir import (
    CellBlocks,
    ColorSpec,
    LutSpec,
    MeshLayer,
    PointSet,
    ScalarField,
    VisibilityMask,
)

if TYPE_CHECKING:
    from apeGmsh.results.Results import Results
    from apeGmsh.viewers.data import ViewerData
    from ..scene.fem_scene import FEMSceneData


class WarnSandUnsupportedElements(UserWarning):
    """Some solid elements could not be seeded with grains.

    Their element type has no shape-function-catalog coverage (or
    their connectivity references nodes missing from the substrate),
    so that chunk of the volume renders empty. Loud per ADR 0056
    INV-6 — an unseeded region is indistinguishable from a zero-field
    region."""


# --------------------------------------------------------------------- #
# Parent-domain samplers — uniform random points per catalog family     #
# --------------------------------------------------------------------- #

def _sample_cube(rng: np.random.Generator, n: int) -> ndarray:
    """Uniform in ``[-1, +1]^3`` (hex8 / hex20 / hex27 parent)."""
    return rng.uniform(-1.0, 1.0, size=(n, 3))


def _sample_simplex(rng: np.random.Generator, n: int) -> ndarray:
    """Uniform in the unit tetrahedron (tet4 / tet10 parent) —
    Dirichlet(1,1,1,1) barycentrics; drop the origin weight."""
    return rng.dirichlet((1.0, 1.0, 1.0, 1.0), size=n)[:, 1:]


def _sample_wedge(rng: np.random.Generator, n: int) -> ndarray:
    """Uniform in tri-barycentric × ``ζ ∈ [-1, +1]`` (wedge6 parent)."""
    tri = rng.dirichlet((1.0, 1.0, 1.0), size=n)[:, 1:]
    z = rng.uniform(-1.0, 1.0, size=(n, 1))
    return np.concatenate([tri, z], axis=1)


# Gmsh type code → (sampler, parent_volume, parent_centroid).
# Higher-order codes share their linear parent domain; the catalog's
# ``n_corner`` for them is the FULL node count, so interpolation and
# jacobians are exact-quadratic there.
_FAMILY_BY_CODE: dict[
    int, tuple[Callable[[np.random.Generator, int], ndarray], float, tuple],
] = {
    4:  (_sample_simplex, 1.0 / 6.0, (0.25, 0.25, 0.25)),   # Tet4
    11: (_sample_simplex, 1.0 / 6.0, (0.25, 0.25, 0.25)),   # Tet10
    5:  (_sample_cube,    8.0,       (0.0, 0.0, 0.0)),      # Hex8
    12: (_sample_cube,    8.0,       (0.0, 0.0, 0.0)),      # Hex27
    17: (_sample_cube,    8.0,       (0.0, 0.0, 0.0)),      # Hex20
    6:  (_sample_wedge,   1.0,       (1.0 / 3, 1.0 / 3, 0.0)),  # Wedge6
}

# MPCO-derived FEMData carries synthetic negative codes; map to the
# equivalent Gmsh linear code by (dim, npe) — same trick as
# ``_gauss_world_coords``.
_GMSH_CODE_BY_DIM_NPE: dict[tuple[int, int], int] = {
    (3, 4): 4,   # Tet4
    (3, 8): 5,   # Hex8
    (3, 6): 6,   # Wedge6
}


@register_diagram_kind(label="Sand volume plot", style_class=SandStyle, order=75)
class SandDiagram(ScalarColorSupport, Diagram):
    """Grain cloud inside solid elements, colored by a nodal component."""

    kind = "sand"
    topology = "nodes"
    # Not for z-fighting (the contour's reason): the grains are strictly
    # interior, so an opaque substrate fill hides the entire diagram.
    # Hiding the fill leaves the wireframe as the volume outline.
    occludes_substrate = True

    def __init__(self, spec: DiagramSpec, results: "Results") -> None:
        if not isinstance(spec.style, SandStyle):
            raise TypeError(
                "SandDiagram requires a SandStyle; "
                f"got {type(spec.style).__name__}."
            )
        super().__init__(spec, results)

        self._layer: Optional[MeshLayer] = None
        self._handle: Any = None
        self._points: Optional[PointSet] = None
        self._cells: Optional[CellBlocks] = None
        self._values: Optional[ndarray] = None
        # Interpolation structure — one row per grain, zero-padded to
        # the widest element's node count. ``_node_pos`` indexes into
        # the needed-node arrays below.
        self._weights: Optional[ndarray] = None        # (P, C) shape-fn values
        self._node_pos: Optional[ndarray] = None       # (P, C) rows, pad → 0
        self._needed_ids: Optional[ndarray] = None     # (n,) FEM node ids
        self._pos_of_id: Optional[ndarray] = None      # id → needed row
        self._substrate_idx: Optional[ndarray] = None  # needed row → grid row
        # Fixed per-grain thresholds for ``weight_by_value`` density.
        self._density_u: Optional[ndarray] = None
        self._init_scalar_color_state()
        # Runtime grain-size override (None = style.point_size).
        self._runtime_point_size: Optional[float] = None

    # ------------------------------------------------------------------
    # Attach / update / deform / detach
    # ------------------------------------------------------------------

    def attach(
        self,
        plotter: Any,
        view: "ViewerData",
        scene: "FEMSceneData | None" = None,
    ) -> None:
        if scene is None:
            raise RuntimeError("SandDiagram.attach requires a FEMSceneData.")
        super().attach(plotter, view, scene)
        style: SandStyle = self.spec.style    # type: ignore[assignment]
        rng = np.random.default_rng(style.seed)

        weights, node_pos = self._seed_grains(view, scene, rng)
        if weights is None or node_pos is None or weights.shape[0] == 0:
            raise NoDataError(
                "Sand volume plot found no seedable 3-D solid elements "
                "in the selection."
            )
        self._weights = weights
        self._node_pos = node_pos

        n_grains = weights.shape[0]
        self._density_u = rng.uniform(size=n_grains)

        values = self._read_values(0)
        if values is None:
            raise NoDataError(
                f"Sand volume plot could not read nodal component "
                f"{self.spec.selector.component!r} at step 0."
            )
        self._values = values

        # Grain geometry — positions from the substrate points so the
        # cloud sits exactly on the displayed grid.
        assert self._substrate_idx is not None
        sub_pts = np.asarray(
            scene.grid.points, dtype=np.float64,
        )[self._substrate_idx]
        self._points = PointSet(self._grain_positions(sub_pts))
        self._cells = CellBlocks(
            {"vertex": np.arange(n_grains, dtype=np.int64).reshape(-1, 1)}
        )

        if style.clim is not None:
            self._initial_clim = (float(style.clim[0]), float(style.clim[1]))
        else:
            finite = values[np.isfinite(values)]
            if finite.size:
                lo, hi = float(finite.min()), float(finite.max())
                if lo == hi:
                    hi = lo + 1.0
                self._initial_clim = (lo, hi)
            else:
                self._initial_clim = (0.0, 1.0)

        self._layer = self._build_layer(values)
        self._handle = self._backend.add_layer(self._layer)

        self._init_lut()
        if self._effective_show_scalar_bar():
            self._backend.add_scalar_bar(
                self._handle, self._make_scalar_bar_spec(),
            )

    def update_to_step(self, step_index: int) -> None:
        if self._layer is None or self._handle is None:
            return
        values = self._read_values(int(step_index))
        if values is None or (
            self._values is not None and values.size != self._values.size
        ):
            return
        self._values = values
        # Same PointSet / CellBlocks objects → the backend's in-place
        # fast path recolours (and re-masks) without re-adding actors.
        self._layer = self._build_layer(values)
        self._backend.update_layer(self._handle, self._layer)

    def sync_substrate_points(
        self, deformed_pts: "ndarray | None", scene: "FEMSceneData",
    ) -> None:
        """Re-interpolate grain positions against the deformed substrate."""
        if (
            self._layer is None
            or self._handle is None
            or self._substrate_idx is None
            or self._values is None
        ):
            return
        target = (
            np.asarray(deformed_pts, dtype=np.float64)
            if deformed_pts is not None
            else np.asarray(scene.grid.points, dtype=np.float64)
        )
        sub_pts = target[self._substrate_idx]
        self._points = PointSet(self._grain_positions(sub_pts))
        self._layer = self._build_layer(self._values)
        self._backend.update_layer(self._handle, self._layer)

    def detach(self) -> None:
        self._remove_scalar_bar(self._scalar_bar_title())
        self._teardown_lut()
        if self._backend is not None and self._handle is not None:
            self._backend.remove_layer(self._handle)
        self._layer = None
        self._handle = None
        self._points = None
        self._cells = None
        self._values = None
        self._weights = None
        self._node_pos = None
        self._needed_ids = None
        self._pos_of_id = None
        self._substrate_idx = None
        self._density_u = None
        self._initial_clim = None
        super().detach()

    def set_visible(self, visible: bool) -> None:
        self._visible = visible
        if self._backend is not None and self._handle is not None:
            self._backend.set_layer_visible(self._handle, bool(visible))

    # ------------------------------------------------------------------
    # Runtime grain size (settings-tab spinner)
    # ------------------------------------------------------------------

    def current_point_size(self) -> float:
        if self._runtime_point_size is not None:
            return self._runtime_point_size
        style: SandStyle = self.spec.style    # type: ignore[assignment]
        return float(style.point_size)

    def set_point_size(self, size: float) -> None:
        """Live grain-size override; re-emits the layer when attached."""
        self._runtime_point_size = float(size)
        if (
            self._layer is not None
            and self._handle is not None
            and self._values is not None
        ):
            self._layer = self._build_layer(self._values)
            self._backend.update_layer(self._handle, self._layer)

    def _scalar_values_for_autofit(self) -> "ndarray | None":
        return self._values

    # ------------------------------------------------------------------
    # Seeding
    # ------------------------------------------------------------------

    def _seed_grains(
        self,
        view: "ViewerData",
        scene: "FEMSceneData",
        rng: np.random.Generator,
    ) -> tuple[Optional[ndarray], Optional[ndarray]]:
        """Sample grains in every selected 3-D element.

        Returns the zero-padded ``(P, C)`` weight and node-row arrays
        (``None`` when nothing is seedable) and populates the
        needed-node lookup state on ``self``.
        """
        style: SandStyle = self.spec.style    # type: ignore[assignment]
        wanted = (
            set(int(e) for e in self._resolved_element_ids)
            if self._resolved_element_ids is not None
            else None
        )

        grid_ids = np.asarray(scene.node_ids, dtype=np.int64)
        if grid_ids.size == 0:
            return None, None
        grid_lookup = np.full(int(grid_ids.max()) + 2, -1, dtype=np.int64)
        grid_lookup[grid_ids] = np.arange(grid_ids.size, dtype=np.int64)
        grid_pts = np.asarray(scene.grid.points, dtype=np.float64)

        # ── Pass 1: per-group corner connectivity + element volumes ──
        # Entries: (catalog_entry, sampler, conn_rows, vols)
        blocks: list[tuple[Any, Any, ndarray, ndarray]] = []
        skipped_codes: set[int] = set()
        n_skipped = 0
        n_dropped_nodes = 0
        for group in view.elements:
            et = group.element_type
            if int(et.dim) != 3:
                continue
            raw_code = int(et.code)
            code = (
                raw_code if raw_code >= 0
                else _GMSH_CODE_BY_DIM_NPE.get(
                    (int(et.dim), int(et.npe)), raw_code,
                )
            )
            fam = _FAMILY_BY_CODE.get(code)
            catalog = get_shape_functions(code)
            ids = np.asarray(group.ids, dtype=np.int64)
            if wanted is not None:
                mask = np.fromiter(
                    (int(e) in wanted for e in ids), dtype=bool, count=ids.size,
                )
                if not mask.any():
                    continue
            else:
                mask = np.ones(ids.size, dtype=bool)
            if fam is None or catalog is None:
                skipped_codes.add(raw_code)
                n_skipped += int(mask.sum())
                continue
            N_fn, dN_fn, _geom, n_corner = catalog
            conn = np.asarray(group.connectivity, dtype=np.int64)[mask][:, :n_corner]
            # Drop elements whose nodes are missing from the substrate.
            safe = conn.clip(min=0, max=grid_lookup.size - 1)
            rows = grid_lookup[safe]
            ok = (rows >= 0).all(axis=1) & (conn >= 0).all(axis=1)
            if not ok.all():
                n_dropped_nodes += int((~ok).sum())
                conn, rows = conn[ok], rows[ok]
            if conn.shape[0] == 0:
                continue
            sampler, parent_vol, centroid = fam
            nat_c = np.asarray(centroid, dtype=np.float64).reshape(1, 3)
            vols = compute_jacobian_dets(
                nat_c, grid_pts[rows], dN_fn, "solid",
            )[:, 0] * parent_vol
            blocks.append((catalog, sampler, conn, np.maximum(vols, 0.0)))

        if n_skipped or n_dropped_nodes:
            parts = []
            if n_skipped:
                parts.append(
                    f"{n_skipped} element(s) of unsupported type code(s) "
                    f"{sorted(skipped_codes)} (no shape-function coverage)"
                )
            if n_dropped_nodes:
                parts.append(
                    f"{n_dropped_nodes} element(s) referencing nodes "
                    f"missing from the substrate"
                )
            warnings.warn(
                "Sand volume plot skipped " + " and ".join(parts)
                + " — that region renders empty.",
                WarnSandUnsupportedElements,
                stacklevel=2,
            )
        if not blocks:
            return None, None

        # ── Pass 2: volume-proportional grain allocation ─────────────
        all_vols = np.concatenate([b[3] for b in blocks])
        total_vol = float(all_vols.sum())
        target = max(1, int(style.target_points))
        if total_vol <= 0.0:
            counts = np.ones(all_vols.size, dtype=np.int64)
        else:
            raw = target * all_vols / total_vol
            counts = np.floor(raw).astype(np.int64)
            remainder = target - int(counts.sum())
            if remainder > 0:
                top = np.argsort(raw - counts)[::-1][:remainder]
                counts[top] += 1
        if int(counts.sum()) == 0:
            counts[int(np.argmax(all_vols))] = 1

        # ── Pass 3: sample + evaluate shape functions per group ─────
        c_max = max(int(b[0][3]) for b in blocks)
        w_parts: list[ndarray] = []
        p_parts: list[ndarray] = []
        offset = 0
        for catalog, sampler, conn, _vols in blocks:
            N_fn, _dN, _geom, n_corner = catalog
            m = conn.shape[0]
            n_i = counts[offset:offset + m]
            offset += m
            total_g = int(n_i.sum())
            if total_g == 0:
                continue
            nat = sampler(rng, total_g)
            N = np.asarray(N_fn(nat), dtype=np.float64)       # (g, n_corner)
            elem_of_grain = np.repeat(np.arange(m), n_i)
            rows = grid_lookup[conn[elem_of_grain]]           # (g, n_corner)
            if n_corner < c_max:
                N = np.pad(N, ((0, 0), (0, c_max - n_corner)))
                rows = np.pad(rows, ((0, 0), (0, c_max - n_corner)))
            w_parts.append(N)
            p_parts.append(rows)
        if not w_parts:
            return None, None

        weights = np.concatenate(w_parts, axis=0)
        grain_grid_rows = np.concatenate(p_parts, axis=0)

        # Compress to the needed-node subset: per-step reads fetch only
        # these ids; positions index the same compact array.
        uniq_rows, inv = np.unique(grain_grid_rows, return_inverse=True)
        self._substrate_idx = uniq_rows.astype(np.int64)
        self._needed_ids = grid_ids[uniq_rows]
        pos_of_id = np.full(int(self._needed_ids.max()) + 2, -1, dtype=np.int64)
        pos_of_id[self._needed_ids] = np.arange(
            self._needed_ids.size, dtype=np.int64,
        )
        self._pos_of_id = pos_of_id
        node_pos = inv.reshape(grain_grid_rows.shape).astype(np.int64)
        return weights, node_pos

    # ------------------------------------------------------------------
    # Interpolation
    # ------------------------------------------------------------------

    def _grain_positions(self, needed_coords: ndarray) -> ndarray:
        """``(P, 3)`` world positions from needed-node coords."""
        assert self._weights is not None and self._node_pos is not None
        return np.einsum(
            "pc,pcx->px", self._weights, needed_coords[self._node_pos],
        )

    def _read_values(self, step_index: int) -> Optional[ndarray]:
        """Nodal component at ``step_index`` interpolated to each grain."""
        if (
            self._weights is None
            or self._node_pos is None
            or self._needed_ids is None
            or self._pos_of_id is None
        ):
            return None
        results = self._scoped_results()
        if results is None:
            return None
        try:
            slab = results.nodes.get(
                ids=self._needed_ids,
                component=self.spec.selector.component,
                time=[int(step_index)],
            )
        except Exception:
            return None
        if slab.values.size == 0:
            return None
        val_full = np.zeros(self._needed_ids.size, dtype=np.float64)
        slab_ids = np.asarray(slab.node_ids, dtype=np.int64)
        safe = slab_ids.clip(min=0, max=self._pos_of_id.size - 1)
        rows = self._pos_of_id[safe]
        ok = rows >= 0
        val_full[rows[ok]] = np.asarray(slab.values[0], dtype=np.float64)[ok]
        return (self._weights * val_full[self._node_pos]).sum(axis=1)

    # ------------------------------------------------------------------
    # Layer construction
    # ------------------------------------------------------------------

    def _layer_id(self) -> str:
        return f"sand_{id(self):x}"

    def _color_array_name(self) -> str:
        return self.spec.selector.component or "_sand_value"

    def _density_mask(self, values: ndarray) -> VisibilityMask:
        """Hide grains whose normalized ``|value|`` falls below their
        fixed random threshold (``weight_by_value`` mode)."""
        style: SandStyle = self.spec.style    # type: ignore[assignment]
        if not style.weight_by_value or self._density_u is None:
            return VisibilityMask()
        clim = self._runtime_clim or self._initial_clim or (0.0, 1.0)
        denom = max(abs(float(clim[0])), abs(float(clim[1]))) or 1.0
        floor = min(max(float(style.density_floor), 0.0), 1.0)
        w = floor + (1.0 - floor) * np.clip(np.abs(values) / denom, 0.0, 1.0)
        hidden = np.flatnonzero(self._density_u > w)
        return VisibilityMask(frozenset(int(i) for i in hidden))

    def _build_layer(self, values: ndarray) -> MeshLayer:
        style: SandStyle = self.spec.style    # type: ignore[assignment]
        assert self._points is not None and self._cells is not None
        clim = self._runtime_clim or self._initial_clim or (0.0, 1.0)
        cmap = self._runtime_cmap or style.cmap
        name = self._color_array_name()
        color = ColorSpec(
            mode="by_array",
            array_name=name,
            lut=LutSpec(name=cmap, vmin=float(clim[0]), vmax=float(clim[1])),
        )
        return MeshLayer(
            layer_id=self._layer_id(),
            points=self._points,
            cells=self._cells,
            fields=(ScalarField(name, values, "point"),),
            color=color,
            visibility=self._density_mask(values),
            opacity=style.opacity,
            point_size=self.current_point_size(),
            # Flat GL points, NOT sphere billboards: on some GL stacks
            # (verified 2026-07-07 on Windows, both off-screen and
            # on-screen) ``render_points_as_spheres`` draws nothing at
            # all, and an invisible diagram is worse than square grains.
            render_points_as_spheres=False,
            pickable=False,
        )
