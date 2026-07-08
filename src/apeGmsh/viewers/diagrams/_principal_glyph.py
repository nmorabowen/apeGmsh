"""PrincipalDirectionDiagram — principal-stress/strain arrows per Gauss point.

Reads the six continuum tensor components at each Gauss point, does a
per-GP eigendecomposition, and renders three arrows per point — one along
each principal direction, scaled by the principal magnitude and coloured
by the *signed* principal value (a diverging map, so compression and
tension read distinctly). Visualises stress flow: struts and ties, the
orientation of the principal field around openings and supports.

Render seam (ADR 0042): emits one arrow :class:`GlyphLayer` via the
backend (three glyph entries per GP, block-stacked p1|p2|p3); holds no
VTK objects. World placement reuses the ``GaussSlab.global_coords`` shape
functions (same as ``GaussPointDiagram``); the arrow layer + magnitude
LUT follow ``VectorGlyphDiagram``. Non-pickable (like the sand / gauss
clouds — GP picking is deferred to R-D).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from numpy import ndarray

from apeGmsh.results import _derived

from ._base import Diagram, DiagramSpec, NoDataError
from ._kinds import register_diagram_kind
from ._scalar_color_support import ScalarColorSupport
from ._styles import PrincipalGlyphStyle
from ..scene_ir import ColorSpec, GlyphLayer, LutSpec, PointSet, ScalarBarSpec

if TYPE_CHECKING:
    from apeGmsh.results.Results import Results
    from apeGmsh.viewers.data import ViewerData
    from ..scene.fem_scene import FEMSceneData

_STRESS_BASE = (
    "stress_xx", "stress_yy", "stress_zz",
    "stress_xy", "stress_yz", "stress_xz",
)
_STRAIN_BASE = tuple(n.replace("stress", "strain") for n in _STRESS_BASE)


def _default_style(component: str) -> PrincipalGlyphStyle:
    """Infer stress vs strain family from the picked component name."""
    family = "strain" if (component or "").startswith("strain") else "stress"
    return PrincipalGlyphStyle(family=family)


@register_diagram_kind(
    label="Principal directions (arrows)",
    style_class=PrincipalGlyphStyle,
    style_factory=_default_style,
    order=72,
)
class PrincipalDirectionDiagram(ScalarColorSupport, Diagram):
    """Three principal-direction arrows per Gauss point, signed-coloured."""

    kind = "principal_glyph"
    topology = "gauss"

    def __init__(self, spec: DiagramSpec, results: "Results") -> None:
        if not isinstance(spec.style, PrincipalGlyphStyle):
            raise TypeError(
                "PrincipalDirectionDiagram requires a PrincipalGlyphStyle; "
                f"got {type(spec.style).__name__}."
            )
        super().__init__(spec, results)
        self._layer: Optional[GlyphLayer] = None
        self._handle: Any = None
        self._coords: Optional[ndarray] = None          # GP world centers (n,3)
        self._present_base: tuple[str, ...] = ()
        self._element_ids_to_read: tuple[int, ...] = ()
        self._gp_element_index: Optional[ndarray] = None
        self._gp_natural_coords: Optional[ndarray] = None
        self._values: Optional[ndarray] = None          # (n,3) principals desc
        self._vectors: Optional[ndarray] = None         # (n,3,3) eigvecs
        self._color_values: Optional[ndarray] = None    # rendered signed scalars
        self._initial_scale: float = 1.0
        self._runtime_scale: Optional[float] = None
        self._init_scalar_color_state()

    # ------------------------------------------------------------------
    # Attach / update / detach
    # ------------------------------------------------------------------

    def attach(
        self, plotter: Any, view: "ViewerData",
        scene: "FEMSceneData | None" = None,
    ) -> None:
        if scene is None:
            raise RuntimeError(
                "PrincipalDirectionDiagram.attach requires a FEMSceneData."
            )
        super().attach(plotter, view, scene)
        style: PrincipalGlyphStyle = self.spec.style   # type: ignore[assignment]

        element_ids = self._resolved_element_ids
        if element_ids is None:
            element_ids = self._collect_continuum_element_ids(view)
        if element_ids.size == 0:
            return
        self._element_ids_to_read = tuple(int(e) for e in element_ids)

        results = self._scoped_results()
        if results is None:
            return
        base = _STRAIN_BASE if style.family == "strain" else _STRESS_BASE
        stored = set(results.elements.gauss.available_components())
        self._present_base = tuple(b for b in base if b in stored)
        if not self._present_base:
            raise NoDataError(
                f"PrincipalDirectionDiagram: no {style.family} tensor stored "
                f"(need any of {list(base)})."
            )

        frame = self._read_frame(0, view)
        if frame is None:
            return
        centers, values, vectors = frame
        self._coords, self._values, self._vectors = centers, values, vectors

        # Auto scale off the largest principal magnitude at step 0.
        max_abs = float(np.abs(values).max()) if values.size else 0.0
        if style.scale is None:
            diag = float(getattr(scene, "model_diagonal", 0.0)) or 0.0
            self._initial_scale = (
                style.auto_scale_fraction * diag / max_abs
                if max_abs > 0.0 and diag > 0.0 else 1.0
            )
        else:
            self._initial_scale = float(style.scale)

        # Symmetric clim about zero (diverging: compression ↔ tension).
        if style.clim is not None:
            self._initial_clim = (float(style.clim[0]), float(style.clim[1]))
        else:
            m = max_abs if max_abs > 0.0 else 1.0
            self._initial_clim = (-m, m)

        self._layer = self._build_layer(self.current_scale())
        self._handle = self._backend.add_layer(self._layer)
        self._init_lut()
        if self._effective_show_scalar_bar():
            self._backend.add_scalar_bar(
                self._handle,
                ScalarBarSpec(
                    layer_id=self._handle.layer_id,
                    title=self._scalar_bar_title(),
                    lut=self._current_lutspec(),
                ),
            )

    def update_to_step(self, step_index: int) -> None:
        if self._layer is None or self._handle is None or self._view is None:
            return
        frame = self._read_frame(int(step_index), self._view)
        if frame is None:
            return
        centers, values, vectors = frame
        self._coords, self._values, self._vectors = centers, values, vectors
        self._layer = self._build_layer(self.current_scale())
        self._backend.update_layer(self._handle, self._layer)

    def sync_substrate_points(
        self, deformed_pts: "ndarray | None", scene: "FEMSceneData",
    ) -> None:
        """Move arrow anchors to follow the deformed substrate."""
        if (
            self._layer is None or self._handle is None or self._view is None
            or self._gp_element_index is None
            or self._gp_natural_coords is None
        ):
            return
        try:
            from apeGmsh.results._gauss_world_coords import (
                compute_global_coords_from_arrays,
            )
            self._coords = np.asarray(compute_global_coords_from_arrays(
                self._gp_element_index, self._gp_natural_coords,
                self._view, node_coords_override=deformed_pts,  # type: ignore[arg-type]
            ), dtype=np.float64)
        except Exception:
            return
        self._layer = self._build_layer(self.current_scale())
        self._backend.update_layer(self._handle, self._layer)

    def detach(self) -> None:
        self._remove_scalar_bar(self._scalar_bar_title())
        self._teardown_lut()
        if self._backend is not None and self._handle is not None:
            self._backend.remove_layer(self._handle)
        self._layer = None
        self._handle = None
        self._coords = None
        self._values = None
        self._vectors = None
        self._color_values = None
        self._gp_element_index = None
        self._gp_natural_coords = None
        self._element_ids_to_read = ()
        self._present_base = ()
        self._initial_clim = None
        super().detach()

    def set_visible(self, visible: bool) -> None:
        self._visible = visible
        if self._backend is not None and self._handle is not None:
            self._backend.set_layer_visible(self._handle, bool(visible))

    # ------------------------------------------------------------------
    # Runtime scale (settings tab)
    # ------------------------------------------------------------------

    def set_scale(self, scale: float) -> None:
        self._runtime_scale = float(scale)
        if self._layer is not None and self._handle is not None:
            self._layer = self._build_layer(self.current_scale())
            self._backend.update_layer(self._handle, self._layer)

    def current_scale(self) -> float:
        if self._runtime_scale is not None:
            return self._runtime_scale
        return self._initial_scale

    # ------------------------------------------------------------------
    # ScalarColorSupport hooks
    # ------------------------------------------------------------------

    def _scalar_values_for_autofit(self) -> "ndarray | None":
        return self._color_values

    def _color_array_name(self) -> str:
        return self.spec.selector.component or f"principal_{self.spec.style.family}"

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _layer_id(self) -> str:
        return f"principal_{id(self):x}"

    def _read_frame(
        self, step_index: int, view: "ViewerData",
    ) -> Optional[tuple[ndarray, ndarray, ndarray]]:
        """Read the tensor columns at ``step_index`` → (centers, values, vecs)."""
        results = self._scoped_results()
        if results is None:
            return None
        style: PrincipalGlyphStyle = self.spec.style   # type: ignore[assignment]

        columns: dict[str, ndarray] = {}
        ref_slab = None
        for name in self._present_base:
            try:
                slab = results.elements.gauss.get(
                    ids=self._element_ids_to_read,
                    component=name, time=[step_index],
                )
            except Exception:
                continue
            if slab.values.size == 0:
                continue
            columns[name] = np.asarray(slab.values[0], dtype=np.float64)
            ref_slab = slab
        if ref_slab is None or not columns:
            return None

        # Cache per-GP arrays for deform-follow, then compute world coords.
        self._gp_element_index = np.asarray(
            ref_slab.element_index, dtype=np.int64).copy()
        self._gp_natural_coords = np.asarray(
            ref_slab.natural_coords, dtype=np.float64).copy()
        try:
            centers = np.asarray(ref_slab.global_coords(view), dtype=np.float64)
        except Exception:
            centers = np.zeros((ref_slab.element_index.size, 3), dtype=np.float64)

        prefix = "strain" if style.family == "strain" else "stress"
        eff_plane = style.plane
        if style.plane == "auto":
            from apeGmsh.results import _plane_recovery
            _plane_recovery.inject_out_of_plane(
                columns, self._gp_element_index, prefix=prefix,
                model=results.model,
            )
            eff_plane = None
        values, vectors = _derived.principal_frame(
            columns, prefix=prefix, plane=eff_plane, nu=style.nu,
        )
        return centers, values, vectors

    def _enabled_principals(self) -> list[int]:
        style: PrincipalGlyphStyle = self.spec.style   # type: ignore[assignment]
        flags = (style.show_p1, style.show_p2, style.show_p3)
        return [i for i, on in enumerate(flags) if on]

    def _build_layer(self, scale: float) -> GlyphLayer:
        """Arrow glyph: 3 arrows/GP, length = |principal|×scale, signed colour."""
        assert self._coords is not None
        assert self._values is not None and self._vectors is not None
        style: PrincipalGlyphStyle = self.spec.style   # type: ignore[assignment]
        enabled = self._enabled_principals() or [0]

        centers = np.concatenate([self._coords] * len(enabled), axis=0)
        oris = np.concatenate([self._vectors[:, :, i] for i in enabled], axis=0)
        vals = np.concatenate([self._values[:, i] for i in enabled], axis=0)
        self._color_values = vals

        clim = self._runtime_clim or self._initial_clim or (-1.0, 1.0)
        cmap = self._runtime_cmap or style.cmap
        color = ColorSpec(
            mode="by_array",
            array_name=self._color_array_name(),
            lut=LutSpec(name=cmap, vmin=float(clim[0]), vmax=float(clim[1])),
        )
        return GlyphLayer(
            layer_id=self._layer_id(),
            positions=PointSet(centers),
            kind="arrow",
            orientations=oris,
            scales=np.abs(vals) * float(scale),
            color_scalar=vals,
            color=color,
        )

    @staticmethod
    def _collect_continuum_element_ids(view: "ViewerData") -> ndarray:
        ids: list[int] = []
        for group in view.elements:
            if group.element_type.dim in (2, 3):
                ids.extend(int(x) for x in group.ids)
        return np.asarray(ids, dtype=np.int64)
