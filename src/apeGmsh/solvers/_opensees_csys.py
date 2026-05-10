"""
Coordinate systems for beam frame orientation.

Each CS class evaluates an orthonormal triad ``(e1, e2, e3)`` at a point
``p``.  The third axis ``e3`` is the *reference axis* the orientation
rule uses to place the section's local-z in the plane of (beam-axis,
e3).

Used by :meth:`g.opensees.elements.add_geom_transf(csys=...)`.

Three flavors:

* :class:`Cartesian`   — constant triad everywhere (default ``e3 = +Z``)
* :class:`Cylindrical` — radial / circumferential / axial triad about
                          a user-defined axis of revolution
* :class:`Spherical`   — outward-radial / latitude / longitude triad
                          about a user-defined origin

The orientation rule applied at each beam element:

::

    t = unit beam tangent (gmsh-given)
    if |t · e3| < 1 - tol:
        local_y = unit(e3 × t)
    else:                          # tangent parallel to reference axis
        local_y = e2               # fall back to CS in-plane axis
    vecxz   = t × local_y          # local-z = vector in x-z plane
    if roll_deg != 0:
        vecxz = Rodrigues(vecxz, axis=t, angle=roll_deg)
"""
from __future__ import annotations

import math

import numpy as np


__all__ = [
    "Cartesian",
    "Cylindrical",
    "Spherical",
    "resolve_vecxz",
]


_TOL = 1e-9


def _unit(v) -> np.ndarray:
    a = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(a))
    if n < 1e-12:
        raise ValueError(f"zero-length vector: {tuple(a)}")
    return a / n


def _rodrigues(v: np.ndarray, axis: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate ``v`` about unit ``axis`` by ``angle_deg`` (right-hand rule)."""
    th = math.radians(angle_deg)
    c, s = math.cos(th), math.sin(th)
    k = axis
    return v * c + np.cross(k, v) * s + k * float(np.dot(k, v)) * (1.0 - c)


# ---------------------------------------------------------------------------
# Cartesian
# ---------------------------------------------------------------------------

class Cartesian:
    """
    Constant Cartesian triad.  ``reference_axis`` defines ``e3``;
    ``e1`` and ``e2`` are picked deterministically from the global
    axis least aligned with ``e3``.

    The default ``reference_axis = (0, 0, 1)`` reproduces the legacy
    "Z up" convention: horizontal beams get ``vecxz = (0, 0, 1)`` and
    vertical columns fall back to ``vecxz = (-1, 0, 0)`` (the sign
    follows the tangent direction; see :ref:`shoebuckle`).

    Parameters
    ----------
    reference_axis : 3-vector
        The axis ``e3``.  Need not be unit length.

    Example
    -------
    ::

        from apeGmsh.solvers import Cartesian

        # Standard structural convention: Z is vertical
        cs = Cartesian()                          # reference_axis = +Z

        # Mechanical CAD convention: Y is vertical
        cs = Cartesian(reference_axis=(0, 1, 0))
    """

    def __init__(self, reference_axis=(0.0, 0.0, 1.0)) -> None:
        e3 = _unit(reference_axis)
        # Pick the global axis least aligned with e3, project it
        # perpendicular to e3, normalise -> e1.
        candidates = (
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        )
        idx = int(np.argmin([abs(float(np.dot(c, e3))) for c in candidates]))
        c0 = candidates[idx]
        e1 = c0 - float(np.dot(c0, e3)) * e3
        e1 /= float(np.linalg.norm(e1))
        e2 = np.cross(e3, e1)
        self._e1 = e1
        self._e2 = e2
        self._e3 = e3

    def triad_at(self, p) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._e1, self._e2, self._e3

    def __repr__(self) -> str:
        return f"Cartesian(reference_axis={tuple(self._e3)})"


# ---------------------------------------------------------------------------
# Cylindrical
# ---------------------------------------------------------------------------

class Cylindrical:
    """
    Cylindrical CS about an axis of revolution.

    At a point ``p``:

    * ``e1`` = radial outward, perpendicular to ``axis``
    * ``e2`` = circumferential, ``axis × e1``
    * ``e3`` = ``axis`` (constant)  ← reference axis for the rule

    Use this for ring beams, tank stiffeners, and any beam set whose
    natural "vertical" is the axis of revolution.

    Parameters
    ----------
    origin : 3-vector
        Any point on the axis of revolution.
    axis   : 3-vector
        Direction of the axis of revolution.  Need not be unit length.

    Example
    -------
    ::

        from apeGmsh.solvers import Cylindrical

        # Vertical tank
        cs = Cylindrical(origin=(0, 0, 0), axis=(0, 0, 1))
    """

    def __init__(self, origin=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0)) -> None:
        self._origin = np.asarray(origin, dtype=float)
        self._axis = _unit(axis)

    def triad_at(self, p) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        d = np.asarray(p, dtype=float) - self._origin
        radial = d - float(np.dot(d, self._axis)) * self._axis
        r_mag = float(np.linalg.norm(radial))
        if r_mag < 1e-12:
            # On the axis — pick any vector perpendicular to axis.
            fallback = np.array([1.0, 0.0, 0.0])
            radial = fallback - float(np.dot(fallback, self._axis)) * self._axis
            if float(np.linalg.norm(radial)) < 1e-9:
                fallback = np.array([0.0, 1.0, 0.0])
                radial = fallback - float(np.dot(fallback, self._axis)) * self._axis
            r_mag = float(np.linalg.norm(radial))
        e_r = radial / r_mag
        e_theta = np.cross(self._axis, e_r)
        return e_r, e_theta, self._axis

    def __repr__(self) -> str:
        return (
            f"Cylindrical(origin={tuple(self._origin)}, "
            f"axis={tuple(self._axis)})"
        )


# ---------------------------------------------------------------------------
# Spherical
# ---------------------------------------------------------------------------

class Spherical:
    """
    Spherical CS about a fixed origin.  Polar axis is global ``+Z``.

    At a point ``p`` (with ``r = |p − origin|``):

    * ``e1`` = ``e_θ`` — along the meridian (south at the equator)
    * ``e2`` = ``e_φ`` — along the parallel (east)
    * ``e3`` = ``e_r`` — outward radial  ← reference axis for the rule

    Useful for fan vaults, geodesic ribs, and any beam network with
    natural radial structure.  Note: for a *planar* curved beam (e.g.
    a vertical-plane arch), :class:`Cartesian` with ``reference_axis``
    in the plane gives the same answer with less ceremony.

    Parameters
    ----------
    origin : 3-vector
        Centre of the sphere.

    Example
    -------
    ::

        from apeGmsh.solvers import Spherical

        cs = Spherical(origin=(0, 0, 0))
    """

    def __init__(self, origin=(0.0, 0.0, 0.0)) -> None:
        self._origin = np.asarray(origin, dtype=float)

    def triad_at(self, p) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        d = np.asarray(p, dtype=float) - self._origin
        r_mag = float(np.linalg.norm(d))
        if r_mag < 1e-12:
            # At origin — degenerate; return global axes.
            return (
                np.array([1.0, 0.0, 0.0]),
                np.array([0.0, 1.0, 0.0]),
                np.array([0.0, 0.0, 1.0]),
            )
        e_r = d / r_mag
        z_axis = np.array([0.0, 0.0, 1.0])
        cross_zr = np.cross(z_axis, e_r)
        cross_mag = float(np.linalg.norm(cross_zr))
        if cross_mag < 1e-9:
            # At a pole of the sphere — pick an arbitrary tangent.
            e_phi = np.array([0.0, 1.0, 0.0])
            e_theta = np.cross(e_phi, e_r)
            e_theta /= float(np.linalg.norm(e_theta))
        else:
            e_phi = cross_zr / cross_mag
            e_theta = np.cross(e_phi, e_r)
        return e_theta, e_phi, e_r

    def __repr__(self) -> str:
        return f"Spherical(origin={tuple(self._origin)})"


# ---------------------------------------------------------------------------
# Orientation rule
# ---------------------------------------------------------------------------

def resolve_vecxz(
    tangent: np.ndarray,
    e1     : np.ndarray,
    e2     : np.ndarray,
    e3     : np.ndarray,
    roll_deg: float = 0.0,
) -> tuple[float, float, float]:
    """
    Compute ``vecxz`` for a beam element from its tangent and a CS
    triad.  See module docstring for the rule.

    ``tangent`` must be unit-length.  ``(e1, e2, e3)`` must be an
    orthonormal triad.  ``roll_deg`` rotates the result about
    ``tangent`` (right-hand rule).
    """
    t = np.asarray(tangent, dtype=float)
    e2 = np.asarray(e2, dtype=float)
    e3 = np.asarray(e3, dtype=float)

    if abs(float(np.dot(t, e3))) < 1.0 - _TOL:
        ly = np.cross(e3, t)
        ly /= float(np.linalg.norm(ly))
    else:
        # Beam is parallel to e3 — fall back to e2 (perpendicular by
        # construction of the triad).
        ly = e2

    lz = np.cross(t, ly)
    lz_mag = float(np.linalg.norm(lz))
    if lz_mag < 1e-12:
        raise ValueError(
            "resolve_vecxz: degenerate triad — local_y is collinear "
            "with tangent."
        )
    lz = lz / lz_mag

    if roll_deg != 0.0:
        lz = _rodrigues(lz, t, roll_deg)

    return (float(lz[0]), float(lz[1]), float(lz[2]))
