"""DiagramStyle — frozen records of per-diagram render parameters.

Each Diagram subclass declares its own ``DiagramStyle`` subclass with
the fields it needs (colormap, clim, scale, …). The base class is
empty — kinds share no fields. Phase 1 adds Contour and DeformedShape
styles.

Style records are frozen for stability across save / load. Within a
session, a Diagram may carry mutable runtime state (current scale,
auto-fit clim) that overrides the style; persistence captures the
runtime values into a fresh frozen ``DiagramSpec`` when needed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


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
        Whether to render the colorbar overlay.
    """
    cmap: str = "viridis"
    clim: Optional[tuple[float, float]] = None
    opacity: float = 1.0
    show_edges: bool = False
    show_scalar_bar: bool = True


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
        ``"y"`` or ``"z"`` — local axis along which the fill extends.
        ``None`` selects via ``component`` (axial / shear / bending
        defaults from ``_beam_geometry.COMPONENT_TO_LOCAL_AXIS``).
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
    fill_axis: Optional[str] = None
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
    """
    cmap: str = "viridis"
    clim: Optional[tuple[float, float]] = None
    opacity: float = 1.0
    point_size: float = 12.0
    show_scalar_bar: bool = True


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
    point_size_fraction
        3-D dot radius as a fraction of the model diagonal. ``0.005``
        is small enough to keep crowded sections readable.
    show_scalar_bar
        Whether to render the colorbar.
    panel_marker_scale
        2-D side panel marker size relative to the unit-area circle.
        Larger fibers (by ``area``) draw bigger markers.
    panel_show_areas
        If ``True``, the 2-D scatter scales markers by fiber ``area``.
        If ``False``, all markers are drawn the same size.
    """
    cmap: str = "coolwarm"
    clim: Optional[tuple[float, float]] = None
    opacity: float = 1.0
    point_size_fraction: float = 0.005
    show_scalar_bar: bool = True
    panel_marker_scale: float = 60.0
    panel_show_areas: bool = True


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
    """
    cmap: str = "coolwarm"
    clim: Optional[tuple[float, float]] = None
    opacity: float = 1.0
    show_edges: bool = False
    show_scalar_bar: bool = True
    aggregation: str = "mid_layer"


@dataclass(frozen=True)
class DeformedShapeStyle(DiagramStyle):
    """Render parameters for ``DeformedShapeDiagram``.

    Attributes
    ----------
    components
        The 1, 2, or 3 canonical component names that drive the warp.
        Default: 3-D translational displacement (``displacement_x``,
        ``displacement_y``, ``displacement_z``).
    scale
        Initial amplification factor applied to the displacement
        vector. The diagram's runtime API (``set_scale``) lets the
        user adjust this without re-attaching.
    show_undeformed
        Whether to render a wireframe ghost of the undeformed mesh
        behind the warped one.
    color
        Solid color for the deformed mesh when no contour overlay is
        active.
    undeformed_color
        Color of the ghost reference.
    undeformed_opacity
        Opacity of the ghost reference (low values keep it visible
        without dominating the deformed shape).
    """
    components: tuple[str, ...] = (
        "displacement_x", "displacement_y", "displacement_z",
    )
    scale: float = 1.0
    show_undeformed: bool = True
    color: str = "#E05C00"            # warm orange — readable on dark themes
    undeformed_color: str = "#888888"
    undeformed_opacity: float = 0.25
