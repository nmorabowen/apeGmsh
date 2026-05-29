"""SceneLayer value types — the backend-agnostic scene description.

These are the frozen value types defined by
[ADR 0042](../../opensees/architecture/decisions/0042-render-backend-seam.md)
(§Decision, Part 1).  A viewer's domain logic (``diagrams/``,
``overlays/``, the colour/visibility logic in ``core/``) *emits* these
and never touches a render backend's API directly.

INV-1 (ADR 0042): this module imports **neither ``vtk`` nor
``pyvista``**.  Array data is carried in light typed bundles
(:class:`PointSet`, :class:`CellBlocks`, :class:`ScalarField`) that
pin dtype + contiguity at construction and validate shape at emit
time.  Pinned dtype lets a backend hand arrays to VTK zero-copy, which
is what keeps step-animation free of per-frame casts; validated shape
makes a malformed diagram fail loud at emit, not as a cryptic backend
error.

Equality: every type that carries a ``numpy`` array sets
``eq=False``.  The auto-generated ``__eq__`` would compare array
fields with ``==``, which returns an array and raises on the
ambiguous truth value.  Identity equality is the correct default for a
scene layer; parity tests assert on the array contents explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping, Optional, Sequence, Union

import numpy as np


# =====================================================================
# Array bundles
# =====================================================================


@dataclass(frozen=True, eq=False)
class PointSet:
    """``(n, 3)`` coordinates, pinned C-contiguous ``float32``.

    Coordinates are coerced at construction; pass any array-like that
    ``numpy`` can turn into an ``(n, 3)`` float array.
    """

    coords: np.ndarray

    def __post_init__(self) -> None:
        arr = np.ascontiguousarray(self.coords, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(
                f"PointSet.coords must be (n, 3); got shape {arr.shape}."
            )
        object.__setattr__(self, "coords", arr)

    @property
    def n_points(self) -> int:
        return int(self.coords.shape[0])


@dataclass(frozen=True, eq=False)
class CellBlocks:
    """Connectivity keyed by VTK cell-type token.

    ``blocks`` maps a cell-type token (e.g. ``"triangle"``,
    ``"tetra"``, ``"hexahedron"`` — the backend owns the mapping to a
    concrete VTK cell type) to an ``(n_cells, n_nodes)`` ``int64``
    connectivity array indexing into the owning layer's
    :class:`PointSet`.  Each block is coerced to C-contiguous
    ``int64`` at construction.
    """

    blocks: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        coerced: dict[str, np.ndarray] = {}
        for token, conn in self.blocks.items():
            arr = np.ascontiguousarray(conn, dtype=np.int64)
            if arr.ndim != 2:
                raise ValueError(
                    f"CellBlocks block {token!r} must be 2-D "
                    f"(n_cells, n_nodes); got shape {arr.shape}."
                )
            coerced[token] = arr
        object.__setattr__(self, "blocks", coerced)

    @property
    def n_cells(self) -> int:
        return sum(int(a.shape[0]) for a in self.blocks.values())


@dataclass(frozen=True, eq=False)
class ScalarField:
    """A named scalar field bound to a domain location.

    Carries its own contiguity guarantee so a backend never re-casts
    mid-animation.  ``location`` removes the point-vs-cell ambiguity
    that a raw dict-of-arrays would leave implicit.  ``values`` is 1-D
    (one value per point or per cell); vector quantities belong on
    :class:`GlyphLayer.orientations`, not here.
    """

    name: str
    values: np.ndarray
    location: Literal["point", "cell"]

    def __post_init__(self) -> None:
        arr = np.ascontiguousarray(self.values)
        if arr.ndim != 1:
            raise ValueError(
                f"ScalarField {self.name!r}.values must be 1-D; "
                f"got shape {arr.shape}."
            )
        object.__setattr__(self, "values", arr)


# =====================================================================
# Style specs
# =====================================================================


@dataclass(frozen=True)
class LutSpec:
    """A lookup table for ``ColorSpec.mode == "by_array"``."""

    name: str = "viridis"
    vmin: float = 0.0
    vmax: float = 1.0
    n_colors: int = 256
    log_scale: bool = False


@dataclass(frozen=True, eq=False)
class ColorSpec:
    """How a layer is coloured. Exactly one mode is active.

    * ``"solid"`` — a single ``solid_rgb`` triple.
    * ``"by_array"`` — map ``array_name`` (a :class:`ScalarField` on
      the layer) through ``lut``.
    * ``"per_entity_rgb"`` — explicit ``(n_cells, 3)`` RGB, the mode
      the shipped Partition / PhysicalGroup colouring uses
      (ADR 0027).
    """

    mode: Literal["solid", "by_array", "per_entity_rgb"] = "solid"
    # An (r, g, b) float triple in [0, 1], OR a backend-resolvable colour
    # string (named like "red" or hex "#RRGGBB"). The string form lets
    # style-driven diagrams pass their existing colour through unchanged;
    # a backend resolves it to RGB.
    solid_rgb: Union[tuple[float, float, float], str] = (1.0, 1.0, 1.0)
    array_name: Optional[str] = None
    lut: Optional[LutSpec] = None
    entity_rgb: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.mode == "by_array":
            if not self.array_name:
                raise ValueError("ColorSpec(by_array) requires array_name.")
            if self.lut is None:
                object.__setattr__(self, "lut", LutSpec())
        elif self.mode == "per_entity_rgb":
            if self.entity_rgb is None:
                raise ValueError(
                    "ColorSpec(per_entity_rgb) requires entity_rgb."
                )
            arr = np.ascontiguousarray(self.entity_rgb, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[1] != 3:
                raise ValueError(
                    f"ColorSpec.entity_rgb must be (n, 3); got {arr.shape}."
                )
            object.__setattr__(self, "entity_rgb", arr)


@dataclass(frozen=True)
class VisibilityMask:
    """Per-cell visibility — the IR-level expression of
    ``VisibilityManager`` / ``ElementVisibility`` /
    ``OverlayVisibilityModel`` (INV-5, ADR 0042).

    A backend applies it however it likes (``vtkGhostType`` bitmask,
    ``extract_cells``, per-actor toggles); the *semantics* live here,
    so every backend hides the same cells.
    """

    hidden_cells: frozenset[int] = field(default_factory=frozenset)


# =====================================================================
# Layers
# =====================================================================


@dataclass(frozen=True, eq=False)
class MeshLayer:
    """An unstructured mesh layer: points + cells + optional fields."""

    layer_id: str
    points: PointSet
    cells: CellBlocks
    fields: Sequence[ScalarField] = ()
    color: ColorSpec = field(default_factory=ColorSpec)
    visibility: VisibilityMask = field(default_factory=VisibilityMask)
    opacity: float = 1.0
    show_edges: bool = False
    silhouette: bool = False
    # Render the mesh as edges only (no filled faces) — the undeformed
    # "ghost" reference and similar overlays. A backend maps this to its
    # wireframe representation.
    wireframe: bool = False
    # Point-cloud rendering (vertex-cell meshes, e.g. the fiber-section
    # dot cloud). When ``point_size`` is set, a backend renders the
    # layer's points at that **screen-space** size; ``render_points_as_
    # spheres`` makes them GPU sphere billboards rather than flat dots.
    # ``None`` leaves point size at the backend default.
    point_size: Optional[float] = None
    render_points_as_spheres: bool = False
    # Whether the rendered actor participates in picking. Decorative
    # overlays (the fiber cloud) set ``False`` so clicks pass through to
    # the substrate behind them.
    pickable: bool = True

    def field_named(self, name: str) -> Optional[ScalarField]:
        for f in self.fields:
            if f.name == name:
                return f
        return None


@dataclass(frozen=True, eq=False)
class GlyphLayer:
    """A glyph layer — loads / masses / constraints / reactions /
    vector fields rendered as oriented, scaled markers."""

    layer_id: str
    positions: PointSet
    kind: Literal["arrow", "sphere", "cone", "axes"] = "sphere"
    orientations: Optional[np.ndarray] = None
    scales: Optional[np.ndarray] = None
    # Per-glyph scalar used ONLY for ``ColorSpec(by_array)`` colouring —
    # distinct from ``scales`` (glyph size). E.g. a vector diagram sizes
    # by magnitude×factor but colours by the raw magnitude.
    color_scalar: Optional[np.ndarray] = None
    color: ColorSpec = field(default_factory=ColorSpec)
    visibility: VisibilityMask = field(default_factory=VisibilityMask)
    opacity: float = 1.0

    def __post_init__(self) -> None:
        if self.orientations is not None:
            arr = np.ascontiguousarray(self.orientations, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[1] != 3:
                raise ValueError(
                    f"GlyphLayer.orientations must be (n, 3); got {arr.shape}."
                )
            object.__setattr__(self, "orientations", arr)
        if self.scales is not None:
            arr = np.ascontiguousarray(self.scales, dtype=np.float32)
            object.__setattr__(self, "scales", arr)
        if self.color_scalar is not None:
            arr = np.ascontiguousarray(self.color_scalar)
            object.__setattr__(self, "color_scalar", arr)


@dataclass(frozen=True, eq=False)
class LabelLayer:
    """Text annotations anchored at points."""

    layer_id: str
    positions: PointSet
    texts: Sequence[str] = ()

    def __post_init__(self) -> None:
        if len(self.texts) != self.positions.n_points:
            raise ValueError(
                f"LabelLayer {self.layer_id!r}: {len(self.texts)} texts vs "
                f"{self.positions.n_points} positions — must match."
            )


@dataclass(frozen=True)
class ScalarBarSpec:
    """A scalar bar (colour legend) bound to a layer's LUT.

    Not a layer — it is passed to ``RenderBackend.add_scalar_bar`` and
    keyed by ``layer_id`` so ``remove_scalar_bar`` can target it.
    """

    layer_id: str
    title: str
    lut: LutSpec


#: The union a ``RenderBackend.add_layer`` accepts.
SceneLayer = Union[MeshLayer, GlyphLayer, LabelLayer]


__all__ = [
    "PointSet",
    "CellBlocks",
    "ScalarField",
    "LutSpec",
    "ColorSpec",
    "VisibilityMask",
    "MeshLayer",
    "GlyphLayer",
    "LabelLayer",
    "ScalarBarSpec",
    "SceneLayer",
]
