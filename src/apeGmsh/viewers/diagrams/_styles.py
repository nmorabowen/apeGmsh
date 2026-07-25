"""DiagramStyle — frozen records of per-diagram render parameters.

Each Diagram subclass declares its own ``DiagramStyle`` subclass with
the fields it needs (colormap, clim, scale, …). The base class is
empty — kinds share no fields.

Style records are frozen for stability across save / load. Within a
session, a Diagram may carry mutable runtime state (current scale,
auto-fit clim) that overrides the style; persistence captures the
runtime values into a fresh frozen ``DiagramSpec`` when needed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from apeGmsh.cuts import SectionCutDef


@dataclass(frozen=True)
class DiagramStyle:
    """Base for all diagram style records.

    Subclasses are frozen dataclasses carrying render parameters
    (colormap, clim, opacity, scale, glyph size, …). The base is
    empty so that style records do not share accidentally — each kind
    declares only what it uses.
    """
    pass


@dataclass(frozen=True)
class ContourStyle(DiagramStyle):
    """Render parameters for ``ContourDiagram``.

    Attributes
    ----------
    cmap
        Matplotlib / VTK colormap name.
    clim
        Explicit ``(vmin, vmax)`` for the scalar range. ``None`` means
        auto-fit from step 0 at attach time (fixed thereafter; user can
        re-fit via the settings tab).
    opacity
        Surface opacity in ``[0, 1]``.
    show_edges
        Whether to draw element edges over the contour fill.
    show_scalar_bar
        Whether to render the colorbar overlay. Also live-toggleable
        via ``ContourDiagram.set_show_scalar_bar(bool)`` from the
        details panel.
    fmt
        ``printf``-style format string for tick labels on the scalar
        bar (e.g. ``"%.3g"``, ``"%.2e"``, ``"%.4f"``). Live-editable
        via ``ContourDiagram.set_fmt(str)``.
    scalar_bar_vertical
        Bar orientation — ``True`` vertical, ``False`` horizontal,
        ``None`` (default) = the viewer theme's default. Live-editable
        via ``set_scalar_bar_vertical(bool)``.
    scalar_bar_scale
        On-screen size multiplier for the bar (``1.0`` = pyvista's
        default size). Live-editable via ``set_scalar_bar_scale(float)``;
        the bar can also be dragged / resized directly in the scene.
    topology
        ``"nodes"`` (default) forces the nodal-scalar path — values
        come from ``results.nodes.get`` and are painted as point data.
        ``"gauss"`` reads from ``results.elements.gauss.get``; the
        rendering then depends on ``averaging`` and the per-element
        Gauss-point count.
    averaging
        Only honored when ``topology == "gauss"``. ``"averaged"``
        (default) extrapolates GP values to corners and averages
        across elements that share a node — smooth contours that
        cross element boundaries. ``"discrete"`` keeps each element's
        own corner values (no cross-element averaging) and renders on
        a shattered submesh, so element-boundary jumps are visible.
        For elements with one Gauss point, ``"discrete"`` paints flat
        cell data; ``"averaged"`` smears the cell value to corners and
        averages across neighbours.
    """
    cmap: str = "jet"
    clim: Optional[tuple[float, float]] = None
    opacity: float = 1.0
    show_edges: bool = False
    show_scalar_bar: bool = True
    fmt: str = "%.3g"
    scalar_bar_vertical: Optional[bool] = None
    scalar_bar_scale: float = 1.0
    topology: str = "nodes"
    averaging: str = "averaged"


@dataclass(frozen=True)
class IsochroneMapStyle(DiagramStyle):
    """Render parameters for ``IsochroneMapDiagram`` (arrival-time map).

    The painted scalar is a **time**, not a response value: one number
    per node, derived from that node's whole history. The colour scale
    is therefore in the stage's time units and the map does not change
    as the user scrubs.

    Attributes
    ----------
    mode
        How the arrival time is derived from each node's history.

        * ``"first_crossing"`` — the first time the tracked value
          reaches ``threshold``. Nodes that never cross are dropped
          from the painted submesh (the wavefront hasn't reached them),
          leaving the substrate wireframe showing through as "not yet
          arrived".
        * ``"time_to_peak"`` — the time of the largest tracked value.
          Always defined, so every selected node is painted.
    threshold
        Absolute crossing level for ``"first_crossing"``. ``None``
        (default) derives it as ``threshold_fraction`` × the largest
        tracked value over the whole selection and history — the
        unit-agnostic default, so the diagram is usable without knowing
        the field's magnitude up front.
    threshold_fraction
        Fraction used for the derived threshold. Ignored when
        ``threshold`` is set explicitly.
    use_abs
        Track ``|value|`` rather than the signed value. ``True``
        (default) is what a wavefront wants — a first arrival is a
        departure from zero in either direction. Set ``False`` to
        detect a signed level crossing (e.g. "when did this node first
        go above +0.02", ignoring negative excursions).
    interpolate
        Linearly interpolate the crossing time between the two
        bracketing steps instead of snapping to the later step. Costs
        nothing and removes the step-quantized banding that otherwise
        dominates a coarsely-sampled wavefront.
    cmap
        Matplotlib / VTK colormap name. Defaults to ``"turbo"`` —
        arrival time is a sequential quantity read across its whole
        range, which a high-contrast rainbow serves better than the
        response-value default.
    clim
        Explicit ``(vmin, vmax)`` time range. ``None`` fits to the
        computed arrival times at attach.
    opacity, show_edges
        Standard surface controls.
    show_scalar_bar, fmt, scalar_bar_vertical, scalar_bar_scale
        Scalar-bar controls, as on :class:`ContourStyle`. The bar is
        titled after the time quantity, not the tracked component.
    max_history_samples
        Safety budget on ``steps × selected nodes``. An arrival time is
        a reduction over the WHOLE history, so this diagram reads a
        ``(T, N)`` float64 block at attach — 1 M nodes × 1 000 steps is
        7.5 GiB, which would take the viewer down rather than draw
        anything. Exceeding the budget raises with a message naming the
        fix (restrict with a selector, or raise this). The default
        ~50 M samples is about 400 MiB.
    """
    mode: str = "first_crossing"
    threshold: Optional[float] = None
    threshold_fraction: float = 0.1
    use_abs: bool = True
    interpolate: bool = True
    max_history_samples: int = 50_000_000
    cmap: str = "turbo"
    clim: Optional[tuple[float, float]] = None
    opacity: float = 1.0
    show_edges: bool = False
    show_scalar_bar: bool = True
    fmt: str = "%.3g"
    scalar_bar_vertical: Optional[bool] = None
    scalar_bar_scale: float = 1.0


@dataclass(frozen=True)
class IsochroneProfileStyle(DiagramStyle):
    """Render parameters for ``IsochroneProfileDiagram``.

    The diagram's product is a 2-D side panel: the selected nodes'
    values plotted against position along a path, one curve per time
    instant (the consolidation / soil-column "isochrone" plot). In 3-D
    it draws only the sampled path so the user can see which nodes feed
    the curves.

    Attributes
    ----------
    path_axis
        Which coordinate orders the nodes along the path and supplies
        the position abscissa. ``"auto"`` (default) picks the axis with
        the largest spatial extent across the selected nodes — the
        right answer for a soil column (z) or a horizontal line of
        nodes (x / y). Set ``"x"`` / ``"y"`` / ``"z"`` to force it.

        Position is the **coordinate on that axis**, not arc length
        along a general curve; a path that doubles back on the chosen
        axis is not a function of position and will read as a
        zig-zag. Pick a selection that is monotone along one axis.
    value_axis
        Which screen axis carries the response value. ``"auto"``
        (default) puts the value on the horizontal axis whenever
        ``path_axis`` resolves to ``z``, so a depth profile is drawn
        upright in the geotechnical convention (depth vertical); every
        other axis puts position horizontal. ``"horizontal"`` /
        ``"vertical"`` force it.
    n_curves
        How many time instants to draw, spread evenly over the stage's
        steps. The first and last steps are always included.
    cmap
        Colormap sampled along time to colour the curves — early
        instants at one end, late at the other.
    line_width
        Curve line width.
    show_markers
        Draw a marker at each sampled node on every curve. Useful on a
        coarse path; noisy on a fine one.
    mark_current_step
        Overdraw the curve at the viewer's current step in a heavier
        highlight so scrubbing shows where "now" sits in the family.
    show_path
        Draw the sampled path as a polyline in the 3-D scene.
    path_color, path_line_width
        Appearance of that 3-D polyline.
    """
    path_axis: str = "auto"
    value_axis: str = "auto"
    n_curves: int = 8
    cmap: str = "viridis"
    line_width: float = 1.4
    show_markers: bool = False
    mark_current_step: bool = True
    show_path: bool = True
    path_color: str = "#FFB000"
    path_line_width: float = 4.0


@dataclass(frozen=True)
class IsochroneStrobeStyle(DiagramStyle):
    """Render parameters for ``IsochroneStrobeDiagram``.

    Superimposes the deformed shape at several instants at once — a
    strobe / motion-trail — as one wireframe layer whose points carry
    their own frame time, so a single colour scale reads as "early →
    late" and a single scalar bar reports it in time units.

    The frames are built on the substrate points the deformation pump
    hands the diagram, so a spatial offset on the owning geometry moves
    the whole strobe with it. Intended for a **deform-off** geometry —
    the strobe *is* the deformation display. On a deform-on geometry
    each frame stacks on top of that geometry's current warp, which
    double-counts the motion.

    Attributes
    ----------
    field
        Nodal vector prefix driving the warp (``"displacement"``,
        ``"velocity"``, ``"acceleration"``) — the diagram reads
        ``<field>_x/_y/_z`` and tolerates missing axes (2-D models).
    scale
        Warp amplification. ``None`` auto-fits at attach so the largest
        frame displacement reaches ``auto_scale_fraction`` of the model
        diagonal.
    auto_scale_fraction
        Used when ``scale`` is ``None``.
    n_frames
        How many instants to draw, spread evenly over the stage's
        steps (first and last always included).
    cmap
        Colormap sampled along time to colour the frames.
    line_width
        Wireframe line width.
    opacity
        Frame opacity. Below 1.0 the overlapping frames read as a
        trail rather than a thicket.
    max_points
        Safety budget on ``n_frames × selected points``. The strobe
        replicates its submesh once per frame, so an unrestricted
        selection on a large solid mesh would build a huge layer;
        exceeding this raises with a message naming the two knobs
        (fewer frames, or a selector).
    show_scalar_bar, fmt, scalar_bar_vertical, scalar_bar_scale
        Scalar-bar controls, as on :class:`ContourStyle`. The bar
        reports frame time.
    """
    field: str = "displacement"
    scale: Optional[float] = None
    auto_scale_fraction: float = 0.10
    n_frames: int = 6
    cmap: str = "plasma"
    line_width: float = 1.5
    opacity: float = 0.9
    max_points: int = 2_000_000
    show_scalar_bar: bool = True
    fmt: str = "%.3g"
    scalar_bar_vertical: Optional[bool] = None
    scalar_bar_scale: float = 1.0


@dataclass(frozen=True)
class LineForceStyle(DiagramStyle):
    """Render parameters for ``LineForceDiagram``.

    Attributes
    ----------
    scale
        Amplification factor — fill height = ``scale * value``. Auto-
        selected at attach if left ``None`` so the largest fill is a
        configurable fraction of the model diagonal.
    fill_axis
        Direction the fill extends from each beam, perpendicular to
        the beam axis. Accepts:

        * ``None`` — select via ``component`` (axial / shear / bending
          defaults from ``_beam_geometry.COMPONENT_TO_LOCAL_AXIS``).
        * ``"y"`` / ``"z"`` — local-frame axis.
        * ``"global_x"`` / ``"global_y"`` / ``"global_z"`` — global axis,
          projected perpendicular to each beam's axis.
        * ``(dx, dy, dz)`` — explicit world-frame direction, projected
          per beam.

        World-frame overrides are useful for 2-D models you want to
        view obliquely: pick ``"global_z"`` to extrude the diagram out
        of the model plane so it stays visible from any camera angle.
    fill_color
        Solid fill color.
    edge_color
        Color of the fill outline (top + connecting lines to base).
    show_edges
        Whether to draw the fill outline.
    show_axis_line
        Whether to draw the beam axis (the line from node i to node j)
        as a darker baseline. Useful when the substrate edges aren't
        visible enough.
    opacity
        Fill opacity in ``[0, 1]``.
    flip_sign
        If ``True``, flips the sign of every value before rendering —
        useful when the user's sign convention disagrees with
        OpenSees' (e.g., sagging-positive vs hogging-positive).
    auto_scale_fraction
        When ``scale`` is ``None``, the auto-fit at attach time picks
        ``scale`` so the maximum-magnitude fill at step 0 reaches this
        fraction of the model's bounding-box diagonal.
    """
    scale: Optional[float] = None
    fill_axis: Optional["str | tuple[float, float, float]"] = None
    fill_color: str = "#3CB371"          # medium sea green — readable on dark themes
    edge_color: str = "#1F5C39"
    show_edges: bool = True
    show_axis_line: bool = True
    opacity: float = 0.85
    flip_sign: bool = False
    auto_scale_fraction: float = 0.10


@dataclass(frozen=True)
class VectorGlyphStyle(DiagramStyle):
    """Render parameters for ``VectorGlyphDiagram``.

    Attributes
    ----------
    components
        Two or three canonical component names that drive the arrow
        direction. Default: 3-D translational displacement.
    scale
        Amplification factor — arrow length = ``scale * |vector|``.
        ``None`` selects auto-fit at attach so the largest arrow at
        step 0 reaches a fraction of the model diagonal.
    auto_scale_fraction
        Used when ``scale`` is ``None``.
    cmap, clim
        Color the arrows by magnitude. ``clim=None`` auto-fits at attach.
    color
        Solid arrow color when ``cmap`` is ``None`` (or when scalars
        are off).
    use_magnitude_colors
        If ``True``, arrows colored by ``|vector|``; otherwise solid
        ``color``.
    arrow_tip_fraction
        Tip-cone length as a fraction of total arrow length.
    scalar_bar_vertical
        Bar orientation — ``True`` vertical, ``False`` horizontal,
        ``None`` (default) = the viewer theme's default.
    scalar_bar_scale
        On-screen size multiplier for the bar (``1.0`` = default).
    """
    components: tuple[str, ...] = (
        "displacement_x", "displacement_y", "displacement_z",
    )
    scale: Optional[float] = None
    auto_scale_fraction: float = 0.10
    cmap: str = "viridis"
    clim: Optional[tuple[float, float]] = None
    color: str = "#FFB000"
    use_magnitude_colors: bool = True
    arrow_tip_fraction: float = 0.25
    show_scalar_bar: bool = True
    fmt: str = "%.3g"
    scalar_bar_vertical: Optional[bool] = None
    scalar_bar_scale: float = 1.0


@dataclass(frozen=True)
class PrincipalGlyphStyle(DiagramStyle):
    """Render parameters for ``PrincipalDirectionDiagram``.

    Three arrows per Gauss point along the principal directions of the
    stress (or strain) tensor, each scaled by its principal magnitude and
    coloured by the signed principal value (diverging map: compression ↔
    tension). Reads the six tensor components regardless of the picked
    component; ``family`` selects stress vs strain.

    Attributes
    ----------
    family
        ``"stress"`` or ``"strain"`` — which tensor to diagonalise.
    scale
        Arrow-length amplification (length = ``scale × |principal|``).
        ``None`` auto-fits at attach so the largest arrow reaches
        ``auto_scale_fraction`` of the model diagonal.
    auto_scale_fraction
        Used when ``scale`` is ``None``.
    cmap, clim
        Colour by signed principal value; ``clim=None`` auto-fits to a
        symmetric range about zero so compression / tension read
        distinctly on a diverging map.
    arrow_tip_fraction
        Tip-cone length as a fraction of arrow length.
    show_p1, show_p2, show_p3
        Toggle each principal direction independently.
    plane, nu
        2-D out-of-plane recovery (see the derived-scalar layer):
        ``plane="strain"`` + ``nu`` fills σ_zz = ν(σ_xx+σ_yy).
    scalar_bar_vertical
        Bar orientation — ``True`` vertical, ``False`` horizontal,
        ``None`` (default) = the viewer theme's default.
    scalar_bar_scale
        On-screen size multiplier for the bar (``1.0`` = default).
    """
    family: str = "stress"
    scale: Optional[float] = None
    auto_scale_fraction: float = 0.06
    cmap: str = "coolwarm"
    clim: Optional[tuple[float, float]] = None
    arrow_tip_fraction: float = 0.25
    show_p1: bool = True
    show_p2: bool = True
    show_p3: bool = True
    plane: Optional[str] = "auto"
    nu: Optional[float] = None
    show_scalar_bar: bool = True
    fmt: str = "%.3g"
    scalar_bar_vertical: Optional[bool] = None
    scalar_bar_scale: float = 1.0


@dataclass(frozen=True)
class LoadsStyle(DiagramStyle):
    """Render parameters for ``LoadsDiagram`` (applied nodal force arrows).

    Constant-magnitude arrows — the broker carries reference force
    magnitudes only (no timeSeries function), so the diagram does not
    scale with steps. Auto-fit picks ``scale`` at attach so the
    largest force arrow reaches ``auto_scale_fraction`` of the model
    diagonal, matching the VectorGlyph convention.

    Attributes
    ----------
    scale
        Amplification factor — arrow length = ``scale * |force|``.
        ``None`` selects auto-fit at attach.
    auto_scale_fraction
        Used when ``scale`` is ``None``.
    color
        Solid arrow color.
    arrow_tip_fraction
        Tip-cone length as a fraction of total arrow length.
    """
    scale: Optional[float] = None
    auto_scale_fraction: float = 0.10
    color: str = "#FF3030"
    arrow_tip_fraction: float = 0.25


@dataclass(frozen=True)
class ReactionsStyle(DiagramStyle):
    """Render parameters for ``ReactionsDiagram`` (recorded reactions).

    Reactions are step-resolved (the recorder writes them per step),
    so the arrows scale with the active time step. Forces and moments
    are rendered as two separate actors with independent auto-fit
    scales — they have different units (force vs torque), so coupling
    them would skew one or the other.

    Attributes
    ----------
    show_forces, show_moments
        Toggle each glyph family. Both default ``True``; flip to
        ``False`` to render only one.
    force_scale, moment_scale
        Amplification factors. ``None`` selects auto-fit at attach so
        the largest arrow over the time-history reaches
        ``auto_scale_fraction`` of the model diagonal.
    auto_scale_fraction
        Used when either ``*_scale`` is ``None``.
    force_color, moment_color
        Solid arrow colors. Defaults are visually distinct.
    arrow_tip_fraction
        Tip-cone length as a fraction of total straight-arrow length
        (force glyph only).
    moment_arc_degrees
        Sweep of the moment glyph's curved tube. Larger sweeps read
        more clearly as torques but crowd dense supports.
    zero_tol
        Auto-filter threshold for noise. Nodes whose reaction
        magnitude max-over-time is below ``zero_tol × global_max``
        are dropped (so free-interior nodes that the recorder writes
        as effective zero don't pollute the scene).
    """
    show_forces: bool = True
    show_moments: bool = True
    force_scale: Optional[float] = None
    moment_scale: Optional[float] = None
    auto_scale_fraction: float = 0.10
    force_color: str = "#1F8FFF"
    moment_color: str = "#00C8B4"
    arrow_tip_fraction: float = 0.25
    moment_arc_degrees: float = 270.0
    zero_tol: float = 1e-9


@dataclass(frozen=True)
class GaussMarkerStyle(DiagramStyle):
    """Render parameters for ``GaussPointDiagram``.

    Attributes
    ----------
    cmap, clim
        Color GP markers by component value.
    opacity
        Marker opacity.
    point_size
        Screen-space marker size in pixels.
    show_scalar_bar
        Whether to render a colorbar.
    scalar_bar_vertical
        Bar orientation — ``True`` vertical, ``False`` horizontal,
        ``None`` (default) = the viewer theme's default.
    scalar_bar_scale
        On-screen size multiplier for the bar (``1.0`` = default).
    """
    cmap: str = "viridis"
    clim: Optional[tuple[float, float]] = None
    opacity: float = 1.0
    point_size: float = 12.0
    show_scalar_bar: bool = True
    fmt: str = "%.3g"
    scalar_bar_vertical: Optional[bool] = None
    scalar_bar_scale: float = 1.0


@dataclass(frozen=True)
class SandStyle(DiagramStyle):
    """Render parameters for ``SandDiagram``.

    Attributes
    ----------
    cmap, clim
        Color the grains by the nodal component value interpolated at
        each grain. ``clim=None`` auto-fits from step 0 at attach.
    opacity
        Grain opacity in ``[0, 1]``.
    point_size
        Screen-space grain size in pixels. Live-adjustable via
        ``SandDiagram.set_point_size``.
    target_points
        Total grain budget, distributed across the solid elements
        proportionally to element volume — uniform spatial density
        regardless of mesh grading.
    weight_by_value
        If ``True``, grain *density* encodes the field: each grain
        draws a fixed random threshold at attach and is shown at a
        step only when the normalized ``|value|`` at its location
        exceeds it — dense sand where the field is strong, sparse
        where it is weak. Thresholds are fixed per grain so animation
        reads as growth/decay, not flicker.
    density_floor
        Minimum show probability in ``weight_by_value`` mode, so the
        volume outline never disappears entirely.
    seed
        RNG seed for grain placement (and thresholds) — the same spec
        reproduces the same cloud.
    scalar_bar_vertical
        Bar orientation — ``True`` vertical, ``False`` horizontal,
        ``None`` (default) = the viewer theme's default.
    scalar_bar_scale
        On-screen size multiplier for the bar (``1.0`` = default).
    """
    cmap: str = "viridis"
    clim: Optional[tuple[float, float]] = None
    opacity: float = 1.0
    point_size: float = 5.0
    show_scalar_bar: bool = True
    fmt: str = "%.3g"
    target_points: int = 5000
    weight_by_value: bool = False
    density_floor: float = 0.05
    seed: int = 0
    scalar_bar_vertical: Optional[bool] = None
    scalar_bar_scale: float = 1.0


@dataclass(frozen=True)
class SpringForceStyle(DiagramStyle):
    """Render parameters for ``SpringForceDiagram``.

    Each spring is rendered as a single arrow at the spring's nodes
    pair midpoint, oriented along the spring's configured direction
    (deduced from the canonical component name suffix — ``_0`` ->
    direction 0, etc.). Magnitude scales arrow length.

    Attributes
    ----------
    direction
        Override the (x, y, z) direction for the arrow (unit vector).
        ``None`` selects a default per spring component index.
    scale
        Length amplification factor. ``None`` auto-fits at attach.
    auto_scale_fraction
        Fraction of model diagonal that the largest arrow reaches.
    color
        Solid arrow color.
    arrow_tip_fraction
        Cone tip length as fraction of arrow length.
    """
    direction: Optional[tuple[float, float, float]] = None
    scale: Optional[float] = None
    auto_scale_fraction: float = 0.05
    color: str = "#E15050"
    arrow_tip_fraction: float = 0.30


@dataclass(frozen=True)
class FiberSectionStyle(DiagramStyle):
    """Render parameters for ``FiberSectionDiagram``.

    Attributes
    ----------
    cmap, clim, opacity
        Standard contour controls applied to the 3-D fiber dot cloud.
    point_size
        Screen-space dot size in pixels for the 3-D fiber cloud.
        Live-adjustable via ``FiberSectionDiagram.set_point_size``.
    point_size_fraction
        DEPRECATED — never consumed (leftover from a world-sized-
        sphere design). Kept only so sessions saved before
        ``point_size`` existed still deserialize; use ``point_size``.
    show_scalar_bar
        Whether to render the colorbar.
    panel_marker_scale
        2-D side panel marker size relative to the unit-area circle.
        Larger fibers (by ``area``) draw bigger markers.
    panel_show_areas
        If ``True``, the 2-D scatter scales markers by fiber ``area``.
        If ``False``, all markers are drawn the same size.
    scalar_bar_vertical
        Bar orientation — ``True`` vertical, ``False`` horizontal,
        ``None`` (default) = the viewer theme's default.
    scalar_bar_scale
        On-screen size multiplier for the bar (``1.0`` = default).
    """
    cmap: str = "coolwarm"
    clim: Optional[tuple[float, float]] = None
    opacity: float = 1.0
    point_size: float = 10.0
    point_size_fraction: float = 0.005
    show_scalar_bar: bool = True
    fmt: str = "%.3g"
    panel_marker_scale: float = 60.0
    panel_show_areas: bool = True
    scalar_bar_vertical: Optional[bool] = None
    scalar_bar_scale: float = 1.0


@dataclass(frozen=True)
class LayerStackStyle(DiagramStyle):
    """Render parameters for ``LayerStackDiagram``.

    Attributes
    ----------
    cmap, clim, opacity
        Standard contour controls for the 3-D shell mid-surface fill.
    show_edges
        Draw shell-cell edges over the contour fill.
    show_scalar_bar
        Whether to render the colorbar.
    aggregation
        How to reduce the per-(elem, gp, layer, sub_gp) values to one
        per shell cell for the 3-D contour: ``"mid_layer"`` picks the
        sub-GP nearest the mid-thickness; ``"mean"`` averages all
        layers and sub-GPs of the cell's GPs; ``"max_abs"`` picks the
        signed value of largest magnitude.
    scalar_bar_vertical
        Bar orientation — ``True`` vertical, ``False`` horizontal,
        ``None`` (default) = the viewer theme's default.
    scalar_bar_scale
        On-screen size multiplier for the bar (``1.0`` = default).
    """
    cmap: str = "coolwarm"
    clim: Optional[tuple[float, float]] = None
    opacity: float = 1.0
    show_edges: bool = False
    show_scalar_bar: bool = True
    fmt: str = "%.3g"
    aggregation: str = "mid_layer"
    scalar_bar_vertical: Optional[bool] = None
    scalar_bar_scale: float = 1.0


@dataclass(frozen=True)
class SectionCutStyle(DiagramStyle):
    """Render parameters for ``SectionCutDiagram``.

    Carries the :class:`apeGmsh.cuts.SectionCutDef` as the ``cut`` field
    — frozen-in-frozen, fully picklable. The quad is drawn two-tone:
    the kept side (per ``cut.side``) shows ``kept_color``; the discarded
    side shows ``discarded_color`` via the VTK backface property. A
    small normal arrow at the quad centroid points into the kept
    half-space.

    Attributes
    ----------
    cut
        The :class:`apeGmsh.cuts.SectionCutDef` instance being rendered.
    kept_color
        Front-face color — visible when looking at the cut from the kept
        side of the plane.
    discarded_color
        Back-face color — visible from the discarded side.
    quad_opacity
        Opacity of the cut quad in ``[0, 1]``. Default ``0.45`` is
        semi-transparent so the structure behind the cut stays legible.
    edge_color
        Outline color of the cut quad.
    show_edges
        Whether to draw the outline.
    show_normal_arrow
        Whether to draw the side-aware normal arrow at the quad
        centroid.
    normal_arrow_fraction
        Arrow length as a fraction of the model bounding-box diagonal.
        Default ``0.05`` is readable on most structures without
        dominating the scene.
    """
    cut: "SectionCutDef"
    kept_color: str = "#3CB371"           # medium sea green
    discarded_color: str = "#B33C3C"      # warm red
    quad_opacity: float = 0.45
    edge_color: str = "#404040"
    show_edges: bool = True
    show_normal_arrow: bool = True
    normal_arrow_fraction: float = 0.05
    # Filter-highlight controls — drawn as a separate cell-extracted
    # overlay actor when ``show_filter_initially`` is True or
    # ``SectionCutDiagram.set_show_filter(True)`` is called at runtime.
    highlight_color: str = "#FFC107"      # amber
    highlight_opacity: float = 0.85
    show_filter_initially: bool = False
