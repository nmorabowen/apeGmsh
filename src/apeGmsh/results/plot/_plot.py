"""ResultsPlot — matplotlib renderer attached to ``Results.plot``.

Mirrors the kind catalog the interactive viewer exposes, but produces
static matplotlib figures suitable for publication / headless CI.

Available methods: ``mesh``, ``contour``, ``deformed``, ``history``,
``vector_glyph``, ``reactions``, ``loads``, ``line_force``. Each
returns the matplotlib ``Axes`` for chaining or further
customization. Pass your own ``ax=`` to embed in a larger layout.

Reaction / load moments are not drawn — matplotlib has no curved-
arrow primitive; use ``results.viewer()`` for moment glyphs. Fiber-
section scatter plots (per-element σ vs ε) and animated step series
are likewise deferred to a future phase.

3-D limitation
--------------
matplotlib does not support Gouraud shading on ``Poly3DCollection``;
contour faces are coloured by the **mean of their three vertex
scalars** (per-face flat shading). With element edges drawn the
result is visually indistinguishable from smooth shading at the
densities typical of FE meshes.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from numpy import ndarray

from ._arrows import auto_arrow_scale, filter_significant, model_diagonal
from ._beams import (
    build_eid_to_endpoints,
    compute_local_axes,
    fill_axis_for,
    station_position,
)
from ._facets import coords_lookup, extract_facets

if TYPE_CHECKING:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers projection)
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
    from ..Results import Results


try:
    import matplotlib  # noqa: F811
    import matplotlib.pyplot as plt  # noqa: F811
    import matplotlib.colors as mcolors  # noqa: F811
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F811
    from mpl_toolkits.mplot3d.art3d import (
        Line3DCollection, Poly3DCollection,
    )    # noqa: F811
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


def _require_mpl() -> None:
    if not _HAS_MPL:
        raise ImportError(
            "apeGmsh.results.plot requires matplotlib. "
            "Install with: pip install apeGmsh[plot]"
        )


_DISPLACEMENT_COMPS: tuple[str, str, str] = (
    "displacement_x", "displacement_y", "displacement_z",
)
_REACTION_FORCE_COMPS: tuple[str, str, str] = (
    "reaction_force_x", "reaction_force_y", "reaction_force_z",
)
_REACTION_MOMENT_COMPS: tuple[str, str, str] = (
    "reaction_moment_x", "reaction_moment_y", "reaction_moment_z",
)

_DEFAULT_FIGSIZE: tuple[float, float] = (9, 7)
_COLOR_MESH = "#5B8DB8"
_COLOR_GHOST = "#bbbbbb"
_COLOR_REACTION = "#D62728"
_COLOR_LOAD = "#2CA02C"


class ResultsPlot:
    """``results.plot`` — static matplotlib renderer.

    The instance caches the triangulation/lookup tables on first use
    so subsequent calls (e.g. animating a sequence of frames) don't
    repeat the O(E) facet walk.
    """

    def __init__(self, results: "Results") -> None:
        self._r = results
        self._figsize: tuple[float, float] = _DEFAULT_FIGSIZE
        # (tris, segs, lookup, coords) — populated on first _facets() call
        self._facet_cache: Optional[
            tuple[ndarray, ndarray, ndarray, ndarray]
        ] = None

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def figsize(self, size: tuple[float, float]) -> "ResultsPlot":
        """Set the default figure size for axes this composite creates."""
        self._figsize = (float(size[0]), float(size[1]))
        return self

    # ------------------------------------------------------------------
    # mesh — undeformed wireframe / surface (no scalar)
    # ------------------------------------------------------------------

    def mesh(
        self,
        *,
        ax: Optional["Axes3D"] = None,
        color: str = _COLOR_MESH,
        edge_color: str = "white",
        alpha: float = 0.7,
        linewidth: float = 0.3,
    ) -> "Axes3D":
        """Render the undeformed mesh as a flat-colored surface."""
        _require_mpl()
        ax = self._ensure_ax(ax)
        tris, segs, lookup, coords = self._facets()

        if tris.size:
            verts = coords[lookup[tris]]
            ax.add_collection3d(Poly3DCollection(
                verts, facecolor=color,
                edgecolor=edge_color, linewidth=linewidth, alpha=alpha,
            ))

        if segs.size:
            seg_verts = coords[lookup[segs]]
            ax.add_collection3d(Line3DCollection(
                seg_verts, colors=color,
                linewidths=max(linewidth, 1.0),
            ))

        self._autoscale(ax, coords)
        return ax

    # ------------------------------------------------------------------
    # contour — paint a nodal scalar on the surface
    # ------------------------------------------------------------------

    def contour(
        self,
        component: str,
        *,
        step: int = -1,
        stage: Optional[str] = None,
        ax: Optional["Axes3D"] = None,
        cmap: str = "viridis",
        clim: Optional[tuple[float, float]] = None,
        edge_color: Optional[str] = "k",
        linewidth: float = 0.1,
        alpha: float = 1.0,
        scalar_bar: bool = True,
        deformed: bool = False,
        scale: float = 1.0,
        wireframe: bool = False,
        wireframe_color: str = "k",
        title: Optional[str] = None,
    ) -> "Axes3D":
        """Paint a nodal scalar on the mesh surface (and 1-D elements).

        Parameters
        ----------
        component
            Canonical nodal component name (``"displacement_z"``,
            ``"reaction_force_x"`` …). See
            ``results.nodes.available_components()``.
        step
            Step index. Negative indexing supported; ``-1`` (default)
            is the last step.
        stage
            Stage name / id. Required if the file has multiple stages.
        ax
            Existing 3-D axes to draw into. If ``None``, a new figure
            is created at ``self.figsize``.
        cmap, clim
            Colormap name and color limits. ``clim=None`` autoscales
            to the data range at this step.
        edge_color, linewidth
            Triangle edge styling. Pass ``edge_color=None`` to drop
            edges (smoother but harder to read at moderate density).
        deformed, scale
            If ``True``, warp the mesh by ``scale * displacement``
            before painting.
        wireframe, wireframe_color
            Overlay an undeformed wireframe in light grey when
            ``deformed=True`` (helps the viewer see the warp).
        """
        _require_mpl()
        ax = self._ensure_ax(ax)

        node_values = self._read_node_scalars(component, step=step, stage=stage)
        tris, segs, lookup, coords = self._facets()

        if deformed:
            plot_coords = self._deformed_coords(
                coords=coords, scale=scale, step=step, stage=stage,
            )
        else:
            plot_coords = coords

        if tris.size == 0 and segs.size == 0:
            raise RuntimeError(
                "contour: no renderable elements in mesh."
            )

        edge_kwargs: dict[str, Any] = {"linewidth": linewidth}
        if edge_color is None:
            edge_kwargs["edgecolor"] = "none"
        else:
            edge_kwargs["edgecolor"] = edge_color

        # Auto-clim across both surfaces and lines so a beam-+-shell
        # mesh shares one colour scale.
        if clim is None:
            sample_vals: list[ndarray] = []
            if tris.size:
                sample_vals.append(node_values[lookup[tris]].mean(axis=1))
            if segs.size:
                sample_vals.append(node_values[lookup[segs]].mean(axis=1))
            stacked = (
                np.concatenate(sample_vals) if sample_vals
                else np.array([0.0, 1.0])
            )
            finite = stacked[np.isfinite(stacked)]
            if finite.size == 0:
                clim = (0.0, 1.0)
            else:
                lo, hi = float(finite.min()), float(finite.max())
                if lo == hi:
                    hi = lo + 1.0
                clim = (lo, hi)

        norm = mcolors.Normalize(vmin=clim[0], vmax=clim[1])
        mappable: Any = None    # the artist we hand to colorbar

        if wireframe and deformed and tris.size:
            ghost_verts = coords[lookup[tris]]
            ax.add_collection3d(Poly3DCollection(
                ghost_verts, facecolor=(0.0, 0.0, 0.0, 0.0),
                edgecolor=wireframe_color,
                linewidth=max(linewidth * 0.6, 0.2),
                alpha=0.3,
            ))
        if wireframe and deformed and segs.size:
            ghost_segs = coords[lookup[segs]]
            ax.add_collection3d(Line3DCollection(
                ghost_segs, colors=wireframe_color,
                linewidths=max(linewidth * 0.6, 0.5), alpha=0.3,
            ))

        if tris.size:
            verts = plot_coords[lookup[tris]]
            face_vals = node_values[lookup[tris]].mean(axis=1)
            coll = Poly3DCollection(verts, alpha=alpha, **edge_kwargs)
            coll.set_array(face_vals)
            coll.set_cmap(cmap)
            coll.set_norm(norm)
            ax.add_collection3d(coll)
            mappable = coll

        if segs.size:
            seg_verts = plot_coords[lookup[segs]]
            seg_vals = node_values[lookup[segs]].mean(axis=1)
            line_coll = Line3DCollection(
                seg_verts, linewidths=max(linewidth * 5, 1.5),
            )
            line_coll.set_array(seg_vals)
            line_coll.set_cmap(cmap)
            line_coll.set_norm(norm)
            ax.add_collection3d(line_coll)
            if mappable is None:
                mappable = line_coll

        if scalar_bar and mappable is not None:
            cbar = ax.figure.colorbar(mappable, ax=ax, shrink=0.6, pad=0.05)
            cbar.set_label(component)

        if title is None:
            title = f"{component} @ step {step}"
            if deformed:
                title += f"  (deformed × {scale:g})"
        ax.set_title(title)
        self._autoscale(ax, plot_coords)
        return ax

    # ------------------------------------------------------------------
    # deformed — warp by displacement, optional scalar overlay
    # ------------------------------------------------------------------

    def deformed(
        self,
        *,
        step: int = -1,
        scale: float = 1.0,
        stage: Optional[str] = None,
        ax: Optional["Axes3D"] = None,
        component: Optional[str] = None,
        cmap: str = "viridis",
        clim: Optional[tuple[float, float]] = None,
        deformed_color: str = _COLOR_MESH,
        edge_color: Optional[str] = "k",
        linewidth: float = 0.3,
        ghost: bool = True,
    ) -> "Axes3D":
        """Plot the deformed shape, optionally painted with a scalar.

        ``ghost=True`` (default) overlays the undeformed mesh in light
        grey so the warp is visible. Pass ``component=`` to overlay a
        nodal scalar on the deformed mesh — equivalent to
        ``contour(component, deformed=True, scale=scale)`` plus the
        wireframe ghost.
        """
        _require_mpl()
        ax = self._ensure_ax(ax)

        if component is not None:
            return self.contour(
                component, step=step, stage=stage, ax=ax,
                cmap=cmap, clim=clim, edge_color=edge_color,
                linewidth=linewidth, alpha=1.0, scalar_bar=True,
                deformed=True, scale=scale,
                wireframe=ghost, wireframe_color=_COLOR_GHOST,
            )

        tris, segs, lookup, coords = self._facets()
        deformed_coords = self._deformed_coords(
            coords=coords, scale=scale, step=step, stage=stage,
        )

        if ghost:
            if tris.size:
                ghost_verts = coords[lookup[tris]]
                ax.add_collection3d(Poly3DCollection(
                    ghost_verts, facecolor=(0.0, 0.0, 0.0, 0.0),
                    edgecolor=_COLOR_GHOST,
                    linewidth=max(linewidth * 0.6, 0.2),
                    alpha=0.4,
                ))
            if segs.size:
                ghost_segs = coords[lookup[segs]]
                ax.add_collection3d(Line3DCollection(
                    ghost_segs, colors=_COLOR_GHOST,
                    linewidths=max(linewidth * 0.6, 0.5),
                    alpha=0.4,
                ))

        edge_kwargs: dict[str, Any] = {"linewidth": linewidth}
        edge_kwargs["edgecolor"] = "none" if edge_color is None else edge_color

        if tris.size:
            verts = deformed_coords[lookup[tris]]
            ax.add_collection3d(Poly3DCollection(
                verts, facecolor=deformed_color, alpha=0.85, **edge_kwargs,
            ))
        if segs.size:
            seg_verts = deformed_coords[lookup[segs]]
            ax.add_collection3d(Line3DCollection(
                seg_verts, colors=deformed_color,
                linewidths=max(linewidth * 5, 1.5),
            ))

        ax.set_title(f"deformed × {scale:g} @ step {step}")
        self._autoscale(ax, np.vstack([coords, deformed_coords]))
        return ax

    # ------------------------------------------------------------------
    # history — 2-D node component vs time
    # ------------------------------------------------------------------

    def history(
        self,
        node: int | None = None,
        *,
        component: str,
        point: tuple[float, float, float] | None = None,
        stage: Optional[str] = None,
        ax: Optional[Any] = None,
        label: Optional[str] = None,
        **plot_kwargs,
    ):
        """Plot ``component`` vs ``time`` at a single node (2-D axes).

        Provide either ``node=<id>`` or ``point=<xyz>`` (snaps to
        nearest node). Returns a 2-D ``Axes`` (not 3-D).
        """
        _require_mpl()
        if (node is None) == (point is None):
            raise ValueError(
                "history: provide exactly one of node= or point=."
            )

        if point is not None:
            slab = self._r.nodes.nearest_to(
                point, component=component, stage=stage,
            )
        else:
            slab = self._r.nodes.get(
                ids=[int(node)], component=component, stage=stage,
            )

        values = np.asarray(slab.values)
        if values.size == 0 or values.shape[1] == 0:
            raise RuntimeError(
                f"history: no data for component {component!r} at the "
                f"requested node."
            )
        if values.shape[1] != 1:
            raise RuntimeError(
                f"history: expected 1 node, got {values.shape[1]}."
            )
        series = values[:, 0]
        time = np.asarray(slab.time)

        if ax is None:
            fig, ax = plt.subplots(figsize=self._figsize)

        nid = int(slab.node_ids[0])
        if label is None:
            label = f"{component} @ N{nid}"
        ax.plot(time, series, label=label, **plot_kwargs)
        ax.set_xlabel("time")
        ax.set_ylabel(component)
        ax.legend()
        ax.grid(True, alpha=0.3)
        return ax

    # ------------------------------------------------------------------
    # vector_glyph — generic vector field arrows
    # ------------------------------------------------------------------

    def vector_glyph(
        self,
        prefix: str,
        *,
        step: int = -1,
        stage: Optional[str] = None,
        ax: Optional["Axes3D"] = None,
        color: str = _COLOR_REACTION,
        scale: Optional[float] = None,
        target_frac: float = 0.07,
        zero_tol: float = 1e-3,
        arrow_length_ratio: float = 0.3,
        linewidth: float = 1.0,
        with_mesh: bool = True,
        deformed: bool = False,
        deform_scale: float = 1.0,
        label: Optional[str] = None,
    ) -> "Axes3D":
        """Draw arrows at every node where the ``(x,y,z)`` triple is non-zero.

        Reads ``f"{prefix}_x"`` / ``f"{prefix}_y"`` / ``f"{prefix}_z"``
        as a vector field. Common prefixes:
        ``"displacement"``, ``"velocity"``, ``"reaction_force"``,
        ``"force"``, ``"acceleration"``.

        Parameters
        ----------
        prefix
            Component prefix (without the trailing ``_x``).
        scale
            Multiplier applied to raw vector lengths before drawing. If
            ``None``, auto-scaled so the longest arrow is
            ``target_frac × bbox_diagonal``.
        zero_tol
            Drop nodes whose magnitude is below ``zero_tol × max_mag``.
            Filters interior nodes from reaction-style fields where
            most rows are zero.
        with_mesh
            If ``True`` (default), overlays a faint undeformed mesh so
            the arrows have spatial context. Pass ``False`` to suppress.
        deformed, deform_scale
            Anchor the arrows at the deformed-position nodes.
        """
        _require_mpl()
        ax = self._ensure_ax(ax)

        fem = self._r._fem
        if fem is None:
            raise RuntimeError("vector_glyph requires a bound FEMData.")
        coords = np.asarray(fem.nodes.coords, dtype=np.float64)

        # Read vector triple component-wise. Missing components → zeros.
        vec = np.zeros((coords.shape[0], 3), dtype=np.float64)
        avail = self._r.nodes.available_components(stage=stage)
        comps = [f"{prefix}_x", f"{prefix}_y", f"{prefix}_z"]
        present = [c for c in comps if c in avail]
        if not present:
            raise RuntimeError(
                f"vector_glyph: no components matching {prefix!r} for "
                f"this stage. Available prefixes derive from: {avail!r}"
            )
        for axis, name in enumerate(comps):
            if name not in avail:
                continue
            comp_vals = self._read_node_scalars(name, step=step, stage=stage)
            comp_vals = np.where(np.isfinite(comp_vals), comp_vals, 0.0)
            vec[:, axis] = comp_vals

        # Anchor points: undeformed or deformed substrate.
        if deformed:
            anchors = self._deformed_coords(
                coords=coords, scale=deform_scale, step=step, stage=stage,
            )
        else:
            anchors = coords

        # Filter low-magnitude rows (otherwise mpl renders zero-length
        # quivers as ugly dots at every interior node).
        anchors_f, vec_f = filter_significant(
            anchors, vec, zero_tol=zero_tol,
        )

        if with_mesh:
            self._draw_ghost_mesh(ax, deformed=deformed,
                                  deform_scale=deform_scale,
                                  step=step, stage=stage)

        if vec_f.size == 0:
            ax.set_title(f"{prefix} (no significant vectors @ step {step})")
            self._autoscale(ax, anchors)
            return ax

        diag = model_diagonal(coords)
        if scale is None:
            scale = auto_arrow_scale(vec_f, diag, target_frac=target_frac)

        scaled = vec_f * float(scale)
        ax.quiver(
            anchors_f[:, 0], anchors_f[:, 1], anchors_f[:, 2],
            scaled[:, 0], scaled[:, 1], scaled[:, 2],
            length=1.0, normalize=False, color=color,
            arrow_length_ratio=arrow_length_ratio, linewidth=linewidth,
            label=label,
        )

        ax.set_title(label or f"{prefix} @ step {step}")
        # Autoscale on union of mesh + arrow tips so arrows aren't clipped.
        tips = anchors_f + scaled
        self._autoscale(ax, np.vstack([coords, tips]))
        return ax

    # ------------------------------------------------------------------
    # reactions — force arrows at supports (thin wrapper over vector_glyph)
    # ------------------------------------------------------------------

    def reactions(
        self,
        *,
        step: int = -1,
        stage: Optional[str] = None,
        ax: Optional["Axes3D"] = None,
        color: str = _COLOR_REACTION,
        scale: Optional[float] = None,
        target_frac: float = 0.07,
        zero_tol: float = 1e-3,
        arrow_length_ratio: float = 0.3,
        linewidth: float = 1.5,
        with_mesh: bool = True,
        deformed: bool = False,
        deform_scale: float = 1.0,
    ) -> "Axes3D":
        """Force arrows at constrained nodes.

        Reads the ``reaction_force_x/y/z`` triple. **Reaction moments
        are not drawn** — matplotlib has no curved-arrow primitive
        and a straight arrow would be physically misleading. Use the
        interactive viewer for moment glyphs.
        """
        return self.vector_glyph(
            "reaction_force",
            step=step, stage=stage, ax=ax, color=color, scale=scale,
            target_frac=target_frac, zero_tol=zero_tol,
            arrow_length_ratio=arrow_length_ratio, linewidth=linewidth,
            with_mesh=with_mesh, deformed=deformed,
            deform_scale=deform_scale,
            label=f"reactions @ step {step}",
        )

    # ------------------------------------------------------------------
    # loads — applied nodal force arrows from the broker (static)
    # ------------------------------------------------------------------

    def loads(
        self,
        *,
        pattern: Optional[str] = None,
        ax: Optional["Axes3D"] = None,
        color: str = _COLOR_LOAD,
        scale: Optional[float] = None,
        target_frac: float = 0.07,
        arrow_length_ratio: float = 0.3,
        linewidth: float = 1.5,
        with_mesh: bool = True,
    ) -> "Axes3D":
        """Applied nodal load arrows from ``fem.nodes.loads``.

        Reads broker-resolved nodal load records (no time evolution —
        the broker stores reference magnitudes only). If ``pattern``
        is ``None``, draws every pattern stacked with one color per
        pattern; pass a name to filter.

        Moments are skipped (matplotlib has no curved arrow). Records
        with no ``force_xyz`` are dropped silently.
        """
        _require_mpl()
        ax = self._ensure_ax(ax)

        fem = self._r._fem
        if fem is None:
            raise RuntimeError("loads requires a bound FEMData.")

        loads_composite = getattr(fem.nodes, "loads", None)
        if loads_composite is None or not hasattr(loads_composite, "patterns"):
            raise RuntimeError(
                "loads: bound FEMData has no nodes.loads composite. "
                "Loads are only available on session-side FEMData; "
                "the embedded MPCO snapshot omits them."
            )

        all_patterns = list(loads_composite.patterns())
        if pattern is not None:
            if pattern not in all_patterns:
                raise KeyError(
                    f"Pattern {pattern!r} not found. Available: "
                    f"{all_patterns!r}"
                )
            patterns_to_draw = [pattern]
        else:
            patterns_to_draw = all_patterns

        if with_mesh:
            self._draw_ghost_mesh(
                ax, deformed=False, deform_scale=1.0, step=0, stage=None,
            )

        coords = np.asarray(fem.nodes.coords, dtype=np.float64)
        all_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
        max_id = int(all_ids.max()) if all_ids.size else 0
        id_to_idx = np.full(max_id + 1, -1, dtype=np.int64)
        id_to_idx[all_ids] = np.arange(all_ids.size, dtype=np.int64)

        diag = model_diagonal(coords)

        # Pre-collect everything so auto-scale spans all patterns.
        per_pattern: list[tuple[str, ndarray, ndarray]] = []
        global_vecs: list[ndarray] = []
        for pat in patterns_to_draw:
            recs = list(loads_composite.by_pattern(pat))
            anchors = []
            vecs = []
            for r in recs:
                if r.force_xyz is None:
                    continue
                idx = id_to_idx[int(r.node_id)] if r.node_id <= max_id else -1
                if idx < 0:
                    continue
                anchors.append(coords[idx])
                vecs.append(np.asarray(r.force_xyz, dtype=np.float64))
            if not anchors:
                continue
            anchor_arr = np.vstack(anchors)
            vec_arr = np.vstack(vecs)
            per_pattern.append((pat, anchor_arr, vec_arr))
            global_vecs.append(vec_arr)

        if not per_pattern:
            ax.set_title("loads (no force vectors found)")
            self._autoscale(ax, coords)
            return ax

        if scale is None:
            stacked = np.vstack(global_vecs)
            scale = auto_arrow_scale(stacked, diag, target_frac=target_frac)

        # Color cycle when drawing >1 pattern. Single pattern uses the
        # explicit ``color=`` argument.
        if len(per_pattern) == 1:
            cycle = [color]
        else:
            cycle = [
                f"C{i}" for i in range(len(per_pattern))
            ]

        all_tips: list[ndarray] = []
        for (pat, anchor_arr, vec_arr), c in zip(per_pattern, cycle):
            scaled = vec_arr * float(scale)
            ax.quiver(
                anchor_arr[:, 0], anchor_arr[:, 1], anchor_arr[:, 2],
                scaled[:, 0], scaled[:, 1], scaled[:, 2],
                length=1.0, normalize=False, color=c,
                arrow_length_ratio=arrow_length_ratio, linewidth=linewidth,
                label=pat,
            )
            all_tips.append(anchor_arr + scaled)

        if len(per_pattern) > 1:
            ax.legend(loc="upper right", fontsize=8)
        title_pat = pattern if pattern else "all patterns"
        ax.set_title(f"loads — {title_pat}")
        self._autoscale(ax, np.vstack([coords, *all_tips]))
        return ax

    # ------------------------------------------------------------------
    # line_force — beam internal-force / strain diagrams along length
    # ------------------------------------------------------------------

    def line_force(
        self,
        component: str,
        *,
        step: int = -1,
        stage: Optional[str] = None,
        ax: Optional["Axes3D"] = None,
        ids: Optional[Any] = None,
        pg: Optional[str] = None,
        label: Optional[str] = None,
        scale: Optional[float] = None,
        target_frac: float = 0.10,
        axis: Optional[str] = None,
        color: str = "#2C4A6E",
        fill_alpha: float = 0.45,
        edge_linewidth: float = 1.2,
        with_mesh: bool = True,
    ) -> "Axes3D":
        """Beam internal-force / strain diagram along element length.

        Reads ``results.elements.line_stations`` for the selected
        elements (default: all beams in the file) and renders the
        classic envelope-plus-fill diagram in 3-D, oriented in each
        beam's local frame.

        Parameters
        ----------
        component
            Canonical line-station component:
            ``"axial_force"``, ``"shear_y"``, ``"shear_z"``,
            ``"torsion"``, ``"bending_moment_y"``, ``"bending_moment_z"``,
            or the conjugate strains (``"axial_strain"`` …).
        ids, pg, label
            Beam selection. At most one. Default: all beam elements
            with line-station data.
        scale
            Visual scale; maps the largest absolute station value to
            ``target_frac × bbox_diagonal``. Override for fixed scale
            across multiple figures.
        axis
            Force the fill axis: ``"y"`` / ``"z"`` (local-frame).
            ``None`` (default) chooses automatically per component
            (e.g. ``shear_y`` → local y, ``axial_force`` → local z).
        """
        _require_mpl()
        ax = self._ensure_ax(ax)

        fem = self._r._fem
        if fem is None:
            raise RuntimeError("line_force requires a bound FEMData.")
        coords = np.asarray(fem.nodes.coords, dtype=np.float64)
        all_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
        max_id = int(all_ids.max()) if all_ids.size else 0
        id_to_idx = np.full(max_id + 1, -1, dtype=np.int64)
        id_to_idx[all_ids] = np.arange(all_ids.size, dtype=np.int64)

        composite = self._r.elements.line_stations
        slab = composite.get(
            component=component, time=step, stage=stage,
            ids=ids, pg=pg, label=label,
        )
        if slab.values.size == 0 or slab.element_index.size == 0:
            raise RuntimeError(
                f"line_force: no line-station data for component "
                f"{component!r} (selector empty?)."
            )

        slab_eids = np.asarray(slab.element_index, dtype=np.int64)
        slab_xi = np.asarray(slab.station_natural_coord, dtype=np.float64)
        slab_vals = np.asarray(slab.values[0], dtype=np.float64)

        endpoints = build_eid_to_endpoints(fem)
        axis_name = fill_axis_for(component, axis)

        # Auto-scale across all stations.
        max_abs = float(np.max(np.abs(slab_vals))) if slab_vals.size else 0.0
        if max_abs <= 0.0:
            ax.set_title(f"{component} (all zero @ step {step})")
            if with_mesh:
                self._draw_ghost_mesh(
                    ax, deformed=False, deform_scale=1.0,
                    step=step, stage=stage,
                )
            self._autoscale(ax, coords)
            return ax

        diag = model_diagonal(coords)
        if scale is None:
            scale = target_frac * diag / max_abs

        if with_mesh:
            self._draw_ghost_mesh(
                ax, deformed=False, deform_scale=1.0,
                step=step, stage=stage,
            )

        # Per-beam render: envelope polyline + filled trapezoidal patches.
        unique_eids = np.unique(slab_eids)
        env_segments: list[ndarray] = []
        fill_polys: list[ndarray] = []
        all_tips: list[ndarray] = []

        for eid in unique_eids:
            eid_int = int(eid)
            if eid_int not in endpoints:
                continue
            i_node, j_node = endpoints[eid_int]
            i_idx = id_to_idx[i_node] if i_node <= max_id else -1
            j_idx = id_to_idx[j_node] if j_node <= max_id else -1
            if i_idx < 0 or j_idx < 0:
                continue
            ci, cj = coords[i_idx], coords[j_idx]
            try:
                x_local, y_local, z_local, _L = compute_local_axes(ci, cj)
            except ValueError:
                continue
            fill_dir = y_local if axis_name == "y" else z_local

            sel = np.where(slab_eids == eid_int)[0]
            order = np.argsort(slab_xi[sel], kind="stable")
            sorted_sel = sel[order]
            sorted_xi = slab_xi[sorted_sel]
            sorted_vals = slab_vals[sorted_sel]
            n = sorted_sel.size
            if n == 0:
                continue

            # Base + top points on this beam.
            base = np.array([
                station_position(ci, cj, float(xi)) for xi in sorted_xi
            ])
            top = base + (sorted_vals * float(scale))[:, None] * fill_dir[None, :]
            all_tips.append(top)

            # Envelope: draw connecting segments along the top points,
            # plus closing segments at each end down to the beam axis.
            if n >= 2:
                for k in range(n - 1):
                    env_segments.append(np.array([top[k], top[k + 1]]))
                    fill_polys.append(np.array([
                        base[k], base[k + 1], top[k + 1], top[k],
                    ]))
            # End closure lines (top → axis at the first/last station).
            env_segments.append(np.array([base[0], top[0]]))
            env_segments.append(np.array([base[-1], top[-1]]))

        if env_segments:
            ax.add_collection3d(Line3DCollection(
                env_segments, colors=color, linewidths=edge_linewidth,
            ))
        if fill_polys:
            poly = Poly3DCollection(
                fill_polys, facecolor=color, alpha=fill_alpha,
                edgecolor="none",
            )
            ax.add_collection3d(poly)

        ax.set_title(f"{component} @ step {step} (scale × {scale:g})")
        union = (
            np.vstack([coords, *all_tips]) if all_tips else coords
        )
        self._autoscale(ax, union)
        return ax

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ensure_ax(self, ax: Optional["Axes3D"]) -> "Axes3D":
        if ax is not None:
            return ax
        fig = plt.figure(figsize=self._figsize)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        return ax    # type: ignore[return-value]

    def _draw_ghost_mesh(
        self,
        ax: "Axes3D",
        *,
        deformed: bool,
        deform_scale: float,
        step: int,
        stage: Optional[str],
    ) -> None:
        """Faint wireframe-style mesh under arrow / beam diagrams."""
        try:
            tris, segs, lookup, coords = self._facets()
        except RuntimeError:
            return
        plot_coords = (
            self._deformed_coords(
                coords=coords, scale=deform_scale, step=step, stage=stage,
            )
            if deformed else coords
        )
        if tris.size:
            ax.add_collection3d(Poly3DCollection(
                plot_coords[lookup[tris]],
                facecolor=(0.0, 0.0, 0.0, 0.0),
                edgecolor=_COLOR_GHOST,
                linewidth=0.3, alpha=0.4,
            ))
        if segs.size:
            ax.add_collection3d(Line3DCollection(
                plot_coords[lookup[segs]],
                colors=_COLOR_GHOST, linewidths=0.7, alpha=0.5,
            ))

    def _facets(self) -> tuple[ndarray, ndarray, ndarray, ndarray]:
        if self._facet_cache is None:
            fem = self._r._fem
            if fem is None:
                raise RuntimeError(
                    "results.plot.* requires a bound FEMData. Open "
                    "with Results.from_native(path) or call "
                    "results.bind(fem)."
                )
            tris, segs = extract_facets(fem)
            lookup, coords = coords_lookup(fem)
            self._facet_cache = (tris, segs, lookup, coords)
        return self._facet_cache

    def _read_node_scalars(
        self,
        component: str,
        *,
        step: int,
        stage: Optional[str],
    ) -> ndarray:
        """Read a nodal scalar at one step, scattered into a full-mesh array.

        Returns a ``(N,)`` array parallel to ``fem.nodes.ids`` with NaN
        for nodes the slab didn't carry.
        """
        slab = self._r.nodes.get(component=component, time=step, stage=stage)
        fem = self._r._fem
        all_ids = np.asarray(fem.nodes.ids, dtype=np.int64)    # type: ignore[union-attr]
        out = np.full(all_ids.size, np.nan, dtype=np.float64)
        slab_ids = np.asarray(slab.node_ids, dtype=np.int64)
        if slab_ids.size == 0:
            return out
        max_id = int(max(int(all_ids.max()), int(slab_ids.max())))
        idx_lookup = np.full(max_id + 1, -1, dtype=np.int64)
        idx_lookup[all_ids] = np.arange(all_ids.size, dtype=np.int64)
        positions = idx_lookup[slab_ids]
        valid = positions >= 0
        out[positions[valid]] = np.asarray(slab.values[0])[valid]
        return out

    def _deformed_coords(
        self,
        *,
        coords: ndarray,
        scale: float,
        step: int,
        stage: Optional[str],
    ) -> ndarray:
        """Return ``coords + scale * displacement`` at the given step."""
        avail = self._r.nodes.available_components(stage=stage)
        comps = [c for c in _DISPLACEMENT_COMPS if c in avail]
        if not comps:
            raise RuntimeError(
                "deformed: no displacement_x/y/z components available "
                f"for this stage. Available: {avail!r}"
            )
        warp = np.zeros_like(coords)
        for axis, name in enumerate(_DISPLACEMENT_COMPS):
            if name not in comps:
                continue
            comp_vals = self._read_node_scalars(name, step=step, stage=stage)
            # NaN → 0 so nodes without disp data simply don't move.
            comp_vals = np.where(np.isfinite(comp_vals), comp_vals, 0.0)
            warp[:, axis] = comp_vals
        return coords + float(scale) * warp

    @staticmethod
    def _autoscale(ax: "Axes3D", coords: ndarray) -> None:
        if coords.size == 0:
            return
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
        max_span = float(np.max(maxs - mins))
        if max_span == 0:
            max_span = 1.0
        center = (mins + maxs) / 2.0
        half = max_span / 2.0 * 1.05
        ax.set_xlim(float(center[0] - half), float(center[0] + half))
        ax.set_ylim(float(center[1] - half), float(center[1] + half))
        ax.set_zlim(float(center[2] - half), float(center[2] + half))
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass
