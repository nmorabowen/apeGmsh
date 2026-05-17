"""Per-beam local-axis math — OpenSees ``vecxz`` convention.

Pure functions; no Qt, no PyVista. Used by ``LineForceDiagram`` to
compute the directions in which beam internal-force diagrams are
drawn.

Convention (OpenSees ``Linear`` / ``PDelta`` / ``Corotational``
geomTransf):

* ``x_local = (j - i) / |j - i|``
* ``z_local = (vecxz - (vecxz · x_local) * x_local) / |...|``
  (Gram-Schmidt: the part of ``vecxz`` perpendicular to ``x_local``).
* ``y_local = z_local × x_local`` (right-hand rule).

The user-supplied ``vecxz`` is a vector lying in the local x-z plane.
When omitted, ``default_vecxz(x_local)`` mimics the typical structural
default — global Z for non-vertical beams, global X for vertical.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Iterator

import numpy as np
from numpy import ndarray

if TYPE_CHECKING:
    from ..data import ViewerData


_VERTICAL_TOL = 0.99      # |dot(x_local, +Z)| above this -> beam is vertical
_DEGENERATE_EPS = 1e-12


def default_vecxz(x_local: ndarray) -> ndarray:
    """Sensible fallback ``vecxz`` when the user provides none.

    For a beam that is not (nearly) parallel to global Z, returns
    ``[0, 0, 1]``. For a vertical beam, returns ``[1, 0, 0]`` so the
    Gram-Schmidt step produces a non-degenerate ``z_local``.
    """
    x = np.asarray(x_local, dtype=np.float64)
    if abs(np.dot(x, np.array([0.0, 0.0, 1.0]))) > _VERTICAL_TOL:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return np.array([0.0, 0.0, 1.0], dtype=np.float64)


def compute_local_axes(
    coord_i: ndarray,
    coord_j: ndarray,
    vecxz: ndarray | None = None,
) -> tuple[ndarray, ndarray, ndarray, float]:
    """Return ``(x_local, y_local, z_local, length)`` for a beam.

    Parameters
    ----------
    coord_i, coord_j
        Endpoint coordinates (3,).
    vecxz
        Reference vector for the local x-z plane. ``None`` selects
        ``default_vecxz(x_local)``.

    Raises
    ------
    ValueError
        If ``i`` and ``j`` coincide, or ``vecxz`` is parallel to
        ``x_local`` and we have no fallback.
    """
    ci = np.asarray(coord_i, dtype=np.float64)
    cj = np.asarray(coord_j, dtype=np.float64)
    raw_x = cj - ci
    L = float(np.linalg.norm(raw_x))
    if L <= _DEGENERATE_EPS:
        raise ValueError(
            "Beam endpoints coincide — cannot build a local frame."
        )
    x_local = raw_x / L

    if vecxz is None:
        vecxz = default_vecxz(x_local)
    v = np.asarray(vecxz, dtype=np.float64)

    # z_local: Gram-Schmidt — part of vecxz perpendicular to x_local
    proj = float(np.dot(v, x_local)) * x_local
    z_raw = v - proj
    z_norm = float(np.linalg.norm(z_raw))
    if z_norm < _DEGENERATE_EPS:
        # vecxz parallel to x_local — pick a perpendicular fallback.
        fallback = default_vecxz(x_local)
        proj = float(np.dot(fallback, x_local)) * x_local
        z_raw = fallback - proj
        z_norm = float(np.linalg.norm(z_raw))
        if z_norm < _DEGENERATE_EPS:
            raise ValueError(
                "Could not derive a non-degenerate z_local; "
                f"x_local={x_local}, vecxz={v}."
            )
    z_local = z_raw / z_norm

    # y_local: right-hand rule
    y_local = np.cross(z_local, x_local)
    y_local /= float(np.linalg.norm(y_local))

    return x_local, y_local, z_local, L


def station_position(
    coord_i: ndarray,
    coord_j: ndarray,
    natural_coord: float,
) -> ndarray:
    """Position along the beam at natural coord ``xi`` in ``[-1, +1]``.

    ``xi = -1`` -> node i; ``xi = +1`` -> node j. Linear interpolation
    matches OpenSees integration-point placement on a 1-D parent
    element.
    """
    ci = np.asarray(coord_i, dtype=np.float64)
    cj = np.asarray(coord_j, dtype=np.float64)
    t = (1.0 + float(natural_coord)) / 2.0
    return ci + t * (cj - ci)


# ======================================================================
# Component -> fill axis selection
# ======================================================================
#
# Maps the canonical component name to the local axis the diagram
# should fill *along*. The convention pairs each force/strain with the
# direction perpendicular to the beam axis that geometrically reflects
# the action:
#
#  * ``shear_y`` / ``bending_moment_z`` -> y_local  (in-plane y-bending)
#  * ``shear_z`` / ``bending_moment_y`` -> z_local  (in-plane z-bending)
#  * ``torsion`` -> y_local (visualisation choice; user can override)
#  * ``axial_force`` -> z_local (offset along z so the diagram is
#    visible regardless of orientation; user can override)

COMPONENT_TO_LOCAL_AXIS: dict[str, str] = {
    # Forces
    "axial_force":          "z",
    "shear_y":              "y",
    "shear_z":              "z",
    "torsion":              "y",
    "bending_moment_y":     "z",
    "bending_moment_z":     "y",
    # Conjugate strains
    "axial_strain":         "z",
    "shear_strain_y":       "y",
    "shear_strain_z":       "z",
    "torsional_strain":     "y",
    "curvature_y":          "z",
    "curvature_z":          "y",
}


def fill_axis_for(component: str, override: str | None = None) -> str:
    """Pick the fill axis name (``"y"`` or ``"z"``) for a component.

    User override wins. Falls back to ``COMPONENT_TO_LOCAL_AXIS``,
    then to ``"y"`` for unknown components.

    Note: this only handles the *local-frame* names. World-frame
    overrides (``"global_x"``, ``"global_y"``, ``"global_z"``, or
    a length-3 tuple) are resolved by :func:`resolve_fill_direction`.
    """
    if override is not None:
        if override not in ("y", "z"):
            raise ValueError(
                f"override must be 'y' or 'z' (got {override!r})."
            )
        return override
    return COMPONENT_TO_LOCAL_AXIS.get(component, "y")


_GLOBAL_AXIS_TO_VEC: dict[str, ndarray] = {
    "global_x": np.array([1.0, 0.0, 0.0], dtype=np.float64),
    "global_y": np.array([0.0, 1.0, 0.0], dtype=np.float64),
    "global_z": np.array([0.0, 0.0, 1.0], dtype=np.float64),
}


FillAxisSpec = "str | tuple[float, float, float] | ndarray | None"


def resolve_fill_direction(
    component: str,
    override: "FillAxisSpec",
    x_local: ndarray,
    y_local: ndarray,
    z_local: ndarray,
) -> ndarray:
    """Resolve the unit fill direction for one beam.

    Supported ``override`` forms:

    * ``None`` — fall back to the component default in
      ``COMPONENT_TO_LOCAL_AXIS`` (``"y"`` or ``"z"``).
    * ``"y"`` / ``"z"`` — local-frame axis.
    * ``"global_x" | "global_y" | "global_z"`` — global axis.
    * ``(dx, dy, dz)`` — explicit world-frame direction.

    For global / explicit overrides the result is the user direction
    projected perpendicular to ``x_local`` and renormalised. If that
    projection is degenerate (override parallel to the beam axis), the
    function falls back to ``y_local`` so the diagram still renders.
    """
    if override is None or (isinstance(override, str) and override in ("y", "z")):
        name = fill_axis_for(component, override if override in ("y", "z") else None)
        return y_local if name == "y" else z_local

    if isinstance(override, str):
        vec = _GLOBAL_AXIS_TO_VEC.get(override)
        if vec is None:
            raise ValueError(
                f"Unknown fill axis spec {override!r}. Must be one of "
                f"'y', 'z', 'global_x', 'global_y', 'global_z' or a "
                f"length-3 tuple."
            )
        d = vec
    else:
        d = np.asarray(override, dtype=np.float64).reshape(-1)
        if d.size != 3:
            raise ValueError(
                f"Custom fill axis must have 3 components (got {d.size})."
            )
        norm = float(np.linalg.norm(d))
        if norm < _DEGENERATE_EPS:
            return y_local
        d = d / norm

    # Project perpendicular to x_local, then renormalise.
    perp = d - float(np.dot(d, x_local)) * x_local
    perp_norm = float(np.linalg.norm(perp))
    if perp_norm < _DEGENERATE_EPS:
        # Override is (almost) parallel to the beam axis — nothing
        # sensible to render along; fall back to local y.
        return y_local
    return perp / perp_norm


def normalize_fill_axis_spec(spec: "FillAxisSpec") -> "FillAxisSpec":
    """Validate and canonicalise a user-supplied fill-axis spec.

    Returns the spec unchanged if valid; raises ``ValueError`` otherwise.
    Tuples are coerced to ``tuple[float, float, float]``.
    """
    if spec is None:
        return None
    if isinstance(spec, str):
        if spec in ("y", "z") or spec in _GLOBAL_AXIS_TO_VEC:
            return spec
        raise ValueError(
            f"Unknown fill axis name {spec!r}. Must be one of "
            f"'y', 'z', 'global_x', 'global_y', 'global_z' or a "
            f"length-3 tuple."
        )
    arr = np.asarray(spec, dtype=np.float64).reshape(-1)
    if arr.size != 3:
        raise ValueError(
            f"Custom fill axis must have 3 components (got {arr.size})."
        )
    return (float(arr[0]), float(arr[1]), float(arr[2]))


# ======================================================================
# Per-element local frame — the reusable orientation seam
# ======================================================================
#
# ``LocalFrame`` is the single resolved orientation record for one beam
# element: its midpoint, the orthonormal triad in OpenSees geomTransf
# convention, and its length.  Today it feeds the local-axis overlay and
# (with the real ``vecxz``) the line-force fill direction.  It is also
# the deliberate seam for *frame section extrusion*: once the bridge
# carries section profiles, an extruder sweeps the section polygon —
# expressed in the local (y, z) plane — along ``x`` between the element
# ends, reusing exactly this triad.  Keep it solver-agnostic and free
# of Qt / PyVista so both the overlay and a future extruder can consume
# it unchanged.


@dataclass(frozen=True)
class LocalFrame:
    """Resolved local coordinate frame for one beam element.

    ``x`` is the element axis (node i → node j); ``y``/``z`` complete
    the right-handed triad per the OpenSees ``vecxz`` convention (see
    :func:`compute_local_axes`).  ``origin`` is the element midpoint —
    the natural glyph anchor and the centroid an extruded section is
    swept about.
    """
    element_id: int
    origin: ndarray      # (3,)
    x: ndarray           # (3,) unit
    y: ndarray           # (3,) unit
    z: ndarray           # (3,) unit
    length: float


def iter_local_frames(
    view: "ViewerData",
    node_coord: Callable[[int], "ndarray | None"],
) -> Iterator[LocalFrame]:
    """Yield a :class:`LocalFrame` for every 1-D (line) element.

    Parameters
    ----------
    view
        The viewer's structural snapshot.  ``view.elements`` supplies
        connectivity; ``view.elements.vecxz_for(eid)`` supplies the
        real geomTransf reference vector when the source carried
        OpenSees enrichment (h5 path).  When it returns ``None`` the
        frame falls back to :func:`default_vecxz` — same behaviour as a
        model with no explicit orientation.
    node_coord
        Maps a FEM node id to its current ``(3,)`` coordinate, or
        ``None`` if absent.  Pass a lookup over the *deformed* substrate
        points to make the frames follow a deformed shape; pass a
        reference-coordinate lookup for the undeformed frame.

    Degenerate elements (coincident endpoints, missing nodes) are
    skipped silently — a single bad element never aborts the sweep.
    """
    for group in view.elements:
        if group.element_type.dim != 1:
            continue
        for eid, conn in group:
            if len(conn) < 2:
                continue
            ci = node_coord(int(conn[0]))
            cj = node_coord(int(conn[1]))
            if ci is None or cj is None:
                continue
            vecxz = view.elements.vecxz_for(int(eid))
            try:
                x_local, y_local, z_local, length = compute_local_axes(
                    ci, cj, vecxz,
                )
            except ValueError:
                continue
            origin = 0.5 * (
                np.asarray(ci, dtype=np.float64)
                + np.asarray(cj, dtype=np.float64)
            )
            yield LocalFrame(
                element_id=int(eid),
                origin=origin,
                x=x_local, y=y_local, z=z_local,
                length=float(length),
            )
