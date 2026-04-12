"""
FieldHelper — Fluent wrapper around ``gmsh.model.mesh.field``.

Extracted from Mesh.py to reduce file size.  Accessed via ``g.mesh.field``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import gmsh

if TYPE_CHECKING:
    from .Mesh import Mesh


class FieldHelper:
    """
    Fluent wrapper around ``gmsh.model.mesh.field``.

    Accessed via ``g.mesh.field``.  Two usage levels:

    **Raw control** (full flexibility)::

        f = g.mesh.field.add("Distance")
        g.mesh.field.set_numbers(f, "CurvesList", [1, 2, 3])
        g.mesh.field.set_background(f)

    **Convenience builders** (common fields with named parameters)::

        dist  = g.mesh.field.distance(curves=[1, 2])
        thr   = g.mesh.field.threshold(dist, size_min=0.05, size_max=0.5,
                                         dist_min=0.1, dist_max=1.0)
        g.mesh.field.set_background(thr)
    """

    def __init__(self, parent_mesh: "Mesh") -> None:
        self._mesh = parent_mesh

    def _log(self, msg: str) -> None:
        if self._mesh._parent._verbose:
            print(f"[Field] {msg}")

    # ------------------------------------------------------------------
    # Raw control
    # ------------------------------------------------------------------

    def add(self, field_type: str) -> int:
        """Create a new field of the given type and return its tag."""
        tag = gmsh.model.mesh.field.add(field_type)
        self._mesh._directives.append({
            'kind': 'field_add', 'field_type': field_type, 'field_tag': tag,
        })
        self._log(f"add({field_type!r}) -> field tag {tag}")
        return tag

    def set_number(self, tag: int, name: str, value: float) -> "FieldHelper":
        """Set a scalar parameter on a field."""
        gmsh.model.mesh.field.setNumber(tag, name, value)
        return self

    def set_numbers(self, tag: int, name: str, values: list[float]) -> "FieldHelper":
        """Set a list parameter on a field."""
        gmsh.model.mesh.field.setNumbers(tag, name, values)
        return self

    def set_string(self, tag: int, name: str, value: str) -> "FieldHelper":
        """Set a string parameter on a field."""
        gmsh.model.mesh.field.setString(tag, name, value)
        return self

    def set_background(self, tag: int) -> "FieldHelper":
        """Register a field as the global background mesh size."""
        gmsh.model.mesh.field.setAsBackgroundMesh(tag)
        self._mesh._directives.append({
            'kind': 'field_background', 'field_tag': tag,
        })
        self._log(f"set_background(field={tag})")
        return self

    def set_boundary_layer_field(self, tag: int) -> "FieldHelper":
        """Register a BoundaryLayer field to be applied during meshing."""
        gmsh.model.mesh.field.setAsBoundaryLayer(tag)
        self._log(f"set_boundary_layer_field(field={tag})")
        return self

    # ------------------------------------------------------------------
    # Convenience builders
    # ------------------------------------------------------------------

    def distance(
        self,
        *,
        curves  : list[int] | None = None,
        surfaces: list[int] | None = None,
        points  : list[int] | None = None,
        sampling: int              = 100,
    ) -> int:
        """Create a ``Distance`` field measuring shortest distance to entities."""
        tag = gmsh.model.mesh.field.add("Distance")
        if curves:
            gmsh.model.mesh.field.setNumbers(tag, "CurvesList",   curves)
        if surfaces:
            gmsh.model.mesh.field.setNumbers(tag, "SurfacesList", surfaces)
        if points:
            gmsh.model.mesh.field.setNumbers(tag, "PointsList",   points)
        gmsh.model.mesh.field.setNumber(tag, "Sampling", sampling)
        self._log(
            f"distance(curves={curves!r}, surfaces={surfaces!r}, "
            f"points={points!r}) -> field {tag}"
        )
        return tag

    def threshold(
        self,
        distance_field : int,
        *,
        size_min       : float,
        size_max       : float,
        dist_min       : float,
        dist_max       : float,
        sigmoid        : bool = False,
        stop_at_dist_max: bool = False,
    ) -> int:
        """Create a ``Threshold`` field ramping size from size_min to size_max."""
        tag = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(tag, "InField",        distance_field)
        gmsh.model.mesh.field.setNumber(tag, "SizeMin",        size_min)
        gmsh.model.mesh.field.setNumber(tag, "SizeMax",        size_max)
        gmsh.model.mesh.field.setNumber(tag, "DistMin",        dist_min)
        gmsh.model.mesh.field.setNumber(tag, "DistMax",        dist_max)
        gmsh.model.mesh.field.setNumber(tag, "Sigmoid",        int(sigmoid))
        gmsh.model.mesh.field.setNumber(tag, "StopAtDistMax",  int(stop_at_dist_max))
        self._log(
            f"threshold(in={distance_field}, "
            f"size=[{size_min},{size_max}], "
            f"dist=[{dist_min},{dist_max}]) -> field {tag}"
        )
        return tag

    def math_eval(self, expression: str) -> int:
        """Create a ``MathEval`` field using an expression in x, y, z."""
        tag = gmsh.model.mesh.field.add("MathEval")
        gmsh.model.mesh.field.setString(tag, "F", expression)
        self._log(f"math_eval({expression!r}) -> field {tag}")
        return tag

    def box(
        self,
        *,
        x_min    : float,
        y_min    : float,
        z_min    : float,
        x_max    : float,
        y_max    : float,
        z_max    : float,
        size_in  : float,
        size_out : float,
        thickness: float = 0.0,
    ) -> int:
        """Create a ``Box`` field: size_in inside, size_out outside."""
        tag = gmsh.model.mesh.field.add("Box")
        gmsh.model.mesh.field.setNumber(tag, "VIn",  size_in)
        gmsh.model.mesh.field.setNumber(tag, "VOut", size_out)
        gmsh.model.mesh.field.setNumber(tag, "XMin", x_min)
        gmsh.model.mesh.field.setNumber(tag, "YMin", y_min)
        gmsh.model.mesh.field.setNumber(tag, "ZMin", z_min)
        gmsh.model.mesh.field.setNumber(tag, "XMax", x_max)
        gmsh.model.mesh.field.setNumber(tag, "YMax", y_max)
        gmsh.model.mesh.field.setNumber(tag, "ZMax", z_max)
        if thickness > 0.0:
            gmsh.model.mesh.field.setNumber(tag, "Thickness", thickness)
        self._log(
            f"box(size_in={size_in}, size_out={size_out}, "
            f"x=[{x_min},{x_max}], y=[{y_min},{y_max}], "
            f"z=[{z_min},{z_max}]) -> field {tag}"
        )
        return tag

    def minimum(self, field_tags: list[int]) -> int:
        """Create a ``Min`` field — element-wise minimum of several fields."""
        tag = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(tag, "FieldsList", field_tags)
        self._log(f"minimum({field_tags}) -> field {tag}")
        return tag

    def boundary_layer(
        self,
        *,
        curves     : list[int] | None = None,
        points     : list[int] | None = None,
        size_near  : float,
        ratio      : float            = 1.2,
        n_layers   : int              = 5,
        thickness  : float | None     = None,
        fan_points : list[int] | None = None,
    ) -> int:
        """Create a ``BoundaryLayer`` field for wall-resolved meshes."""
        tag = gmsh.model.mesh.field.add("BoundaryLayer")
        if curves:
            gmsh.model.mesh.field.setNumbers(tag, "CurvesList",    curves)
        if points:
            gmsh.model.mesh.field.setNumbers(tag, "PointsList",    points)
        if fan_points:
            gmsh.model.mesh.field.setNumbers(tag, "FanPointsList", fan_points)
        gmsh.model.mesh.field.setNumber(tag, "Size",     size_near)
        gmsh.model.mesh.field.setNumber(tag, "Ratio",    ratio)
        gmsh.model.mesh.field.setNumber(tag, "NbLayers", n_layers)
        if thickness is not None:
            gmsh.model.mesh.field.setNumber(tag, "Thickness", thickness)
        self._log(
            f"boundary_layer(size={size_near}, ratio={ratio}, "
            f"layers={n_layers}) -> field {tag}"
        )
        return tag
