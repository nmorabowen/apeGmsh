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

    def _resolve_at_dim(self, refs, expected_dim: int, what: str) -> list[int]:
        """Resolve refs and require every hit to live at *expected_dim*.

        Used by field builders where the parameter (``curves`` / ``surfaces``
        / ``points``) is dim-restricted by Gmsh.  Accepts int, label/PG
        string, ``(d, t)`` tuple, or list of any mix.

        Bare int refs are trusted at *expected_dim* — the parameter
        name disambiguates them, so a tag value that collides with an
        entity at another dim does not silently route to the wrong
        dim.  Strings and ``(d, t)`` tuples are validated and raise on
        dim mismatch.
        """
        if refs is None:
            return []

        # Normalise to a list of refs.  Treat a (dim, tag) tuple as a
        # single ref, not a 2-element list.
        is_dimtag_tuple = (
            isinstance(refs, tuple) and len(refs) == 2
            and isinstance(refs[0], int) and isinstance(refs[1], int)
        )
        if is_dimtag_tuple or not isinstance(refs, (list, tuple)):
            refs_list = [refs]
        else:
            refs_list = list(refs)

        out: list[int] = []
        for r in refs_list:
            if isinstance(r, bool):
                raise TypeError(
                    f"{what}: bool refs are not supported (got {r!r})."
                )
            if isinstance(r, int):
                out.append(int(r))
                continue
            if (
                isinstance(r, tuple) and len(r) == 2
                and isinstance(r[0], int) and isinstance(r[1], int)
            ):
                d, t = r
                if d != expected_dim:
                    raise ValueError(
                        f"{what}: expected dim={expected_dim} but got "
                        f"({d}, {t}) in ref list."
                    )
                out.append(int(t))
                continue
            if isinstance(r, str):
                from apeGmsh.core._helpers import resolve_to_dimtags
                dimtags = resolve_to_dimtags(
                    r, default_dim=expected_dim, session=self._mesh._parent,
                )
                bad = [(d, t) for d, t in dimtags if d != expected_dim]
                if bad:
                    raise ValueError(
                        f"{what}: label/PG {r!r} resolved to {bad}; "
                        f"expected dim={expected_dim}."
                    )
                out.extend(int(t) for _, t in dimtags)
                continue
            raise TypeError(
                f"{what}: unsupported ref type {type(r).__name__!r} "
                f"(value {r!r})."
            )
        return out

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
        curves  = None,
        surfaces = None,
        points   = None,
        sampling: int = 100,
    ) -> int:
        """Create a ``Distance`` field measuring shortest distance to entities.

        Each of ``curves``, ``surfaces``, ``points`` accepts an int, a
        label or PG name, a ``(dim, tag)`` tuple, or a list of any mix.
        Refs are validated against the expected dimension.
        """
        curve_tags = self._resolve_at_dim(curves, 1, "distance(curves=)")
        surf_tags  = self._resolve_at_dim(surfaces, 2, "distance(surfaces=)")
        point_tags = self._resolve_at_dim(points, 0, "distance(points=)")

        tag = gmsh.model.mesh.field.add("Distance")
        if curve_tags:
            gmsh.model.mesh.field.setNumbers(tag, "CurvesList",   curve_tags)
        if surf_tags:
            gmsh.model.mesh.field.setNumbers(tag, "SurfacesList", surf_tags)
        if point_tags:
            gmsh.model.mesh.field.setNumbers(tag, "PointsList",   point_tags)
        gmsh.model.mesh.field.setNumber(tag, "Sampling", sampling)
        self._log(
            f"distance(curves={curve_tags!r}, surfaces={surf_tags!r}, "
            f"points={point_tags!r}) -> field {tag}"
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
        curves     = None,
        points     = None,
        size_near  : float,
        ratio      : float        = 1.2,
        n_layers   : int          = 5,
        thickness  : float | None = None,
        fan_points = None,
    ) -> int:
        """Create a ``BoundaryLayer`` field for wall-resolved meshes.

        ``curves``, ``points``, and ``fan_points`` accept any flexible
        reference shape (int / label / PG / ``(dim, tag)`` / list).
        """
        curve_tags = self._resolve_at_dim(curves, 1, "boundary_layer(curves=)")
        point_tags = self._resolve_at_dim(points, 0, "boundary_layer(points=)")
        fan_tags   = self._resolve_at_dim(fan_points, 0,
                                          "boundary_layer(fan_points=)")

        tag = gmsh.model.mesh.field.add("BoundaryLayer")
        if curve_tags:
            gmsh.model.mesh.field.setNumbers(tag, "CurvesList",    curve_tags)
        if point_tags:
            gmsh.model.mesh.field.setNumbers(tag, "PointsList",    point_tags)
        if fan_tags:
            gmsh.model.mesh.field.setNumbers(tag, "FanPointsList", fan_tags)
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
