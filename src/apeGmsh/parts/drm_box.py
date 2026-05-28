"""
DRMBox — Domain-Reduction-Method box geometry primitive.

A DRM box is a structured layered solid used in seismic
soil-structure-interaction modelling: three concentric regions per
lateral axis (inner core, transition layer, outer absorbing layer)
and a downward-only Z stack (top / mid / bottom layers).  The
classic symmetric case has 5 segments along X, 5 along Y, 3 along
Z, giving ``5 * 5 * 3 = 75`` axis-aligned hex sub-volumes that need
structured-hex meshing with per-region element counts.

This module owns:

* :class:`DRMBox` — a :class:`~apeGmsh.core.Part.Part` subclass that
  builds the sliced geometry in its own Gmsh session.  Geometry
  only — no labels, no physical groups, no mesh settings.  Persists
  to STEP exactly like any other Part.
* :class:`DRMBoxResult` — frozen dataclass returned by the assembly
  helper ``g.parts.add_DRM_box(...)`` summarising the named PGs and
  Axis1D descriptors so the user can refer to them later.

The assembly-side wiring (PG tagging + transfinite cascade) lives
in :func:`apeGmsh.core._parts_registry.PartsRegistry.add_DRM_box`,
not here — transfinite directives don't survive STEP, so they must
be applied post-import on the assembly's session.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..core.Part import Part
from ._axis1d import Axis1D

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class DRMBoxResult:
    """Summary of a DRM-box placement.

    Returned by :func:`PartsRegistry.add_DRM_box`.  The user keeps it
    for downstream references — PG names to feed into recorders or
    constraints, Axis1D descriptors to drive auxiliary mesh sizing.
    """

    inner_pg: str
    transition_pg: str
    outer_pg: str
    line_pgs: dict[str, str] = field(default_factory=dict)
    axes: dict[str, Axis1D] = field(default_factory=dict)
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_z: float = 0.0


class DRMBox(Part):
    """Layered DRM-box geometry, built in its own Gmsh session.

    The box is centred laterally on ``(0, 0)`` and descends from
    ``z = 0`` (top of the inner box, free-surface convention).  No
    labels or physical groups are attached — the assembly-side helper
    re-classifies sub-volumes by centroid + Axis1D lookup after
    import, which is robust to STEP renumbering and to the
    placement transform.

    Parameters
    ----------
    x_inner, x_layer, x_outer, y_inner, y_layer, y_outer :
        ``(size, n_elements)`` tuples — symmetric layered axes
        (outer | layer | inner | layer | outer) along X and Y.
    z_top, z_mid, z_bottom :
        ``(size, n_elements)`` tuples — downward Z stack
        (bottom | mid | top, ``hi = 0``).
    name :
        Gmsh model name and default Part / instance name.
    """

    def __init__(
        self,
        *,
        x_inner: tuple[float, int],
        x_layer: tuple[float, int],
        x_outer: tuple[float, int],
        y_inner: tuple[float, int],
        y_layer: tuple[float, int],
        y_outer: tuple[float, int],
        z_top: tuple[float, int],
        z_mid: tuple[float, int],
        z_bottom: tuple[float, int],
        name: str = "drm_box",
    ) -> None:
        super().__init__(name=name)
        self.axis_x = Axis1D.symmetric_layered(
            "x", inner=x_inner, layer=x_layer, outer=x_outer,
        )
        self.axis_y = Axis1D.symmetric_layered(
            "y", inner=y_inner, layer=y_layer, outer=y_outer,
        )
        self.axis_z = Axis1D.downward_layered(
            "z", top=z_top, mid=z_mid, bottom=z_bottom,
        )
        # Stored so the assembly-side helper can re-create them via
        # ``result.axes`` if it skipped the live-Part path.
        self.properties.update({
            "drm_box": {
                "x_inner": tuple(x_inner),
                "x_layer": tuple(x_layer),
                "x_outer": tuple(x_outer),
                "y_inner": tuple(y_inner),
                "y_layer": tuple(y_layer),
                "y_outer": tuple(y_outer),
                "z_top": tuple(z_top),
                "z_mid": tuple(z_mid),
                "z_bottom": tuple(z_bottom),
            },
        })

    # ------------------------------------------------------------------
    # Geometry build
    # ------------------------------------------------------------------

    def build(self) -> "DRMBox":
        """Build the 75-volume sliced box inside the Part's session.

        Must be called inside ``with drm_box:``.  Returns ``self`` so
        the caller can chain ``with DRMBox(...) as d: d.build()`` if
        desired.  Idempotent within a session — repeated calls slice
        nothing on the already-fully-sliced model.
        """
        if not self._active:
            raise RuntimeError(
                f"DRMBox({self.name!r}).build(): Part session is not "
                f"active.  Call build() inside a `with` block."
            )

        x_breaks = self.axis_x.breaks
        y_breaks = self.axis_y.breaks
        z_breaks = self.axis_z.breaks

        x0, x_end = x_breaks[0], x_breaks[-1]
        y0, y_end = y_breaks[0], y_breaks[-1]
        z0, z_end = z_breaks[0], z_breaks[-1]

        self.model.geometry.add_box(
            x0, y0, z0,
            x_end - x0, y_end - y0, z_end - z0,
        )

        for x in self.axis_x.slice_offsets():
            self.model.geometry.slice(axis="x", offset=float(x))
        for y in self.axis_y.slice_offsets():
            self.model.geometry.slice(axis="y", offset=float(y))
        for z in self.axis_z.slice_offsets():
            self.model.geometry.slice(axis="z", offset=float(z))

        return self
