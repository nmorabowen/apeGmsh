"""
Orientation fields for beam frame local-axis derivation.

Each orientation class evaluates an orthonormal triad ``(e1, e2, e3)``
at a point ``p``.  The third axis ``e3`` is the *reference axis* the
orientation rule uses to place the section's local-z in the plane of
(beam-axis, e3).

Used by the typed :class:`~apeGmsh.opensees.transform.Linear`,
:class:`~apeGmsh.opensees.transform.PDelta`, and
:class:`~apeGmsh.opensees.transform.Corotational` geom_transf
primitives when constructed with ``orientation=``.

Four flavors:

* :class:`Cartesian`   — constant triad everywhere (default ``e3 = +Z``)
* :class:`Cylindrical` — radial / circumferential / axial triad about
                          a user-defined axis of revolution
* :class:`Spherical`   — outward-radial / latitude / longitude triad
                          about a user-defined origin
* :class:`AlongBeam`   — tangent-derived: ``e3`` follows the tangent
                          of a reference physical group (for stirrups
                          along curved longitudinal bars, transverse
                          diaphragms along girders)

The orientation rule applied at each beam element:

::

    t = unit beam tangent (gmsh-given)
    if |t · e3| < 1 - tol:
        local_y = unit(e3 × t)
    else:                          # tangent parallel to reference axis
        local_y = e2               # fall back to in-plane axis
    vecxz   = t × local_y          # local-z = vector in x-z plane
    if roll_deg != 0:
        vecxz = Rodrigues(vecxz, axis=t, angle=roll_deg)
"""
from __future__ import annotations

import math
from typing import cast

import numpy as np
from numpy.typing import ArrayLike


__all__ = [
    "AlongBeam",
    "Cartesian",
    "Cylindrical",
    "Spherical",
    "resolve_vecxz",
]


_TOL = 1e-9
# ``ArrayLike`` is the broad numpy alias for anything ``np.asarray``
# accepts; the orientation classes use it so callers can pass tuples,
# lists, or arrays interchangeably for the 3-vector inputs.


def _unit(v: ArrayLike) -> np.ndarray:
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
    return cast(np.ndarray, v * c + np.cross(k, v) * s
                + k * float(np.dot(k, v)) * (1.0 - c))


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

        from apeGmsh.opensees import Cartesian

        # Standard structural convention: Z is vertical
        orientation = Cartesian()                          # reference_axis = +Z

        # Mechanical CAD convention: Y is vertical
        orientation = Cartesian(reference_axis=(0, 1, 0))
    """

    def __init__(self, reference_axis: ArrayLike = (0.0, 0.0, 1.0)) -> None:
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

    def triad_at(self, p: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._e1, self._e2, self._e3

    def __repr__(self) -> str:
        return f"Cartesian(reference_axis={tuple(self._e3)})"


# ---------------------------------------------------------------------------
# Cylindrical
# ---------------------------------------------------------------------------

class Cylindrical:
    """
    Cylindrical orientation about an axis of revolution.

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

        from apeGmsh.opensees import Cylindrical

        # Vertical tank
        orientation = Cylindrical(origin=(0, 0, 0), axis=(0, 0, 1))
    """

    def __init__(
        self,
        origin: ArrayLike = (0.0, 0.0, 0.0),
        axis: ArrayLike = (0.0, 0.0, 1.0),
    ) -> None:
        self._origin = np.asarray(origin, dtype=float)
        self._axis = _unit(axis)

    def triad_at(self, p: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    Spherical orientation about a fixed origin.  Polar axis is global ``+Z``.

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

        from apeGmsh.opensees import Spherical

        orientation = Spherical(origin=(0, 0, 0))
    """

    def __init__(self, origin: ArrayLike = (0.0, 0.0, 0.0)) -> None:
        self._origin = np.asarray(origin, dtype=float)

    def triad_at(self, p: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
# AlongBeam — tangent-derived orientation from a reference PG
# ---------------------------------------------------------------------------

def _perp_triad_from_e3(e3: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build an orthonormal triad with the given e3.

    e1 is the global axis least aligned with e3, projected
    perpendicular to e3 and normalized; e2 = e3 × e1. Deterministic
    in the choice of e1 — same algorithm as :class:`Cartesian`.
    """
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
    return e1, e2, e3


class AlongBeam:
    """
    Tangent-derived orientation field.

    At a query point ``p``, the orientation's ``e3`` is the unit
    tangent of the **nearest point** on the reference PG's chain of
    line elements (true projection, not nearest-midpoint — stirrups
    offset axially from a reference bar still get the right tangent).
    ``e1`` and ``e2`` are derived deterministically perpendicular to
    ``e3``.

    Use cases:

    * Stirrups along a curved longitudinal rebar (each stirrup's
      ``reference_axis`` = local tangent of the rebar).
    * Transverse diaphragms perpendicular to a curved girder.
    * Any orientation whose "up" direction is set by another beam,
      not by a fixed geometric construction.

    The reference PG must contain line elements (2-node elements).
    The bridge calls :meth:`bind_fem` once before per-element
    fan-out; that materializes the reference segments and their
    tangents into numpy arrays for fast projection queries.

    Parameters
    ----------
    reference_pg : str
        Name of a physical group containing the reference line
        elements. Must be present on the FEM snapshot when the
        bridge builds.

    Example
    -------
    ::

        from apeGmsh.opensees import AlongBeam

        orientation = AlongBeam(reference_pg="MainBar")
        ops.geomTransf.Linear(orientation=orientation)

    Notes
    -----
    Projection cost is ``O(M)`` per query (where ``M`` is the
    reference PG's element count), vectorized via numpy. For
    ``M = 500`` and ``500`` stirrups the full per-stirrup loop is
    fractions of a second; if profiling ever shows it's a hotspot,
    a KD-tree on segment midpoints + local search would bring it
    down further.
    """

    def __init__(self, reference_pg: str) -> None:
        self._reference_pg = reference_pg
        # Populated by bind_fem(); triad_at raises until then.
        self._p_a: np.ndarray | None = None
        self._p_b: np.ndarray | None = None
        self._seg: np.ndarray | None = None
        self._seg_len2: np.ndarray | None = None

    @property
    def reference_pg(self) -> str:
        """Name of the reference physical group."""
        return self._reference_pg

    def bind_fem(self, fem: object) -> None:
        """Cache the reference-PG segments + tangents from ``fem``.

        Called by the bridge once before the per-element vecxz
        fan-out. Idempotent — safe to call multiple times.
        """
        try:
            # selection-unification v2 P3-R / §6.3 §2 #7 (P-GROUPRESULT;
            # m3 — resolution raises at .select()).
            result = fem.elements.select(pg=self._reference_pg).groups()  # type: ignore[attr-defined]
        except (KeyError, ValueError) as e:
            raise ValueError(
                f"AlongBeam(reference_pg={self._reference_pg!r}): "
                f"reference PG not found in FEM snapshot ({e})."
            ) from e

        p_a_list: list[np.ndarray] = []
        p_b_list: list[np.ndarray] = []
        for group in result:
            for eid, conn in group:
                if len(conn) != 2:
                    raise ValueError(
                        f"AlongBeam(reference_pg={self._reference_pg!r}): "
                        f"reference element {int(eid)} has "
                        f"{len(conn)} nodes; AlongBeam requires line "
                        "elements (2 nodes per element)."
                    )
                idx_a = fem.nodes.index(int(conn[0]))  # type: ignore[attr-defined]
                idx_b = fem.nodes.index(int(conn[1]))  # type: ignore[attr-defined]
                p_a_list.append(np.asarray(fem.nodes.coords[idx_a], dtype=float))  # type: ignore[attr-defined]
                p_b_list.append(np.asarray(fem.nodes.coords[idx_b], dtype=float))  # type: ignore[attr-defined]

        if not p_a_list:
            raise ValueError(
                f"AlongBeam(reference_pg={self._reference_pg!r}): "
                "reference PG contains no elements. Did you forget "
                "to assign elements to it before snapshotting the FEM?"
            )

        self._p_a = np.stack(p_a_list)
        self._p_b = np.stack(p_b_list)
        self._seg = self._p_b - self._p_a
        self._seg_len2 = np.einsum("ij,ij->i", self._seg, self._seg)

        # Guard against zero-length reference segments (degenerate
        # mesh): the projection denominator would blow up.
        if float(self._seg_len2.min()) < 1e-24:
            raise ValueError(
                f"AlongBeam(reference_pg={self._reference_pg!r}): "
                "reference PG contains a zero-length element. "
                "Tangent is undefined; the AlongBeam projection "
                "cannot proceed."
            )

    def triad_at(self, p: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if (
            self._p_a is None
            or self._seg is None
            or self._seg_len2 is None
        ):
            raise RuntimeError(
                f"AlongBeam(reference_pg={self._reference_pg!r}).triad_at "
                "called before bind_fem(). The bridge build pipeline is "
                "responsible for the bind call; this error indicates a "
                "direct call to triad_at outside the bridge."
            )

        p_arr = np.asarray(p, dtype=float)
        # Vectorized projection onto every segment, clipped to the
        # segment endpoints.
        to_p = p_arr - self._p_a                                     # (M, 3)
        t = np.einsum("ij,ij->i", to_p, self._seg) / self._seg_len2  # (M,)
        t = np.clip(t, 0.0, 1.0)
        proj = self._p_a + t[:, None] * self._seg                    # (M, 3)
        diff = proj - p_arr                                          # (M, 3)
        d2 = np.einsum("ij,ij->i", diff, diff)                       # (M,)
        i = int(np.argmin(d2))
        e3 = self._seg[i] / float(np.sqrt(self._seg_len2[i]))
        return _perp_triad_from_e3(e3)

    def __repr__(self) -> str:
        return f"AlongBeam(reference_pg={self._reference_pg!r})"


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
    Compute ``vecxz`` for a beam element from its tangent and an
    orientation triad.  See module docstring for the rule.

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
