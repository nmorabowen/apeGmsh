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

import numpy as np
from numpy import ndarray


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
    """
    if override is not None:
        if override not in ("y", "z"):
            raise ValueError(
                f"override must be 'y' or 'z' (got {override!r})."
            )
        return override
    return COMPONENT_TO_LOCAL_AXIS.get(component, "y")
