"""Beam local-axis math for line-force diagrams.

Pure-numpy copy of the helpers in
``viewers/diagrams/_beam_geometry.py`` that this matplotlib module
needs — kept local to avoid a results→viewers package dependency
(viewers pulls Qt + PyVista at import).

Convention (OpenSees ``Linear`` / ``PDelta`` / ``Corotational``
geomTransf):

* ``x_local = (j - i) / |j - i|``
* ``z_local`` = part of ``vecxz`` perpendicular to ``x_local``,
  normalised
* ``y_local = z_local × x_local``

The user-supplied ``vecxz`` is a vector lying in the local x-z plane.
When omitted, ``default_vecxz(x_local)`` returns the typical structural
default — global Z for non-vertical beams, global X for vertical.
"""
from __future__ import annotations

import numpy as np
from numpy import ndarray


_VERTICAL_TOL = 0.99
_DEGENERATE_EPS = 1e-12


# Maps canonical line-station component name to the local axis the
# diagram fills along (``"y"`` or ``"z"``). User can override via the
# ``axis=`` kwarg on ``line_force``.
COMPONENT_TO_LOCAL_AXIS: dict[str, str] = {
    "axial_force":      "z",
    "shear_y":          "y",
    "shear_z":          "z",
    "torsion":          "y",
    "bending_moment_y": "z",
    "bending_moment_z": "y",
    "axial_strain":     "z",
    "shear_strain_y":   "y",
    "shear_strain_z":   "z",
    "torsional_strain": "y",
    "curvature_y":      "z",
    "curvature_z":      "y",
}


def default_vecxz(x_local: ndarray) -> ndarray:
    x = np.asarray(x_local, dtype=np.float64)
    if abs(np.dot(x, np.array([0.0, 0.0, 1.0]))) > _VERTICAL_TOL:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return np.array([0.0, 0.0, 1.0], dtype=np.float64)


def compute_local_axes(
    coord_i: ndarray,
    coord_j: ndarray,
    vecxz: ndarray | None = None,
) -> tuple[ndarray, ndarray, ndarray, float]:
    """Return ``(x_local, y_local, z_local, length)`` for a beam."""
    ci = np.asarray(coord_i, dtype=np.float64)
    cj = np.asarray(coord_j, dtype=np.float64)
    raw_x = cj - ci
    L = float(np.linalg.norm(raw_x))
    if L <= _DEGENERATE_EPS:
        raise ValueError("Beam endpoints coincide.")
    x_local = raw_x / L

    if vecxz is None:
        vecxz = default_vecxz(x_local)
    v = np.asarray(vecxz, dtype=np.float64)

    proj = float(np.dot(v, x_local)) * x_local
    z_raw = v - proj
    z_norm = float(np.linalg.norm(z_raw))
    if z_norm < _DEGENERATE_EPS:
        fallback = default_vecxz(x_local)
        proj = float(np.dot(fallback, x_local)) * x_local
        z_raw = fallback - proj
        z_norm = float(np.linalg.norm(z_raw))
        if z_norm < _DEGENERATE_EPS:
            raise ValueError("Could not derive a non-degenerate z_local.")
    z_local = z_raw / z_norm

    y_local = np.cross(z_local, x_local)
    y_local /= float(np.linalg.norm(y_local))
    return x_local, y_local, z_local, L


def station_position(
    coord_i: ndarray,
    coord_j: ndarray,
    natural_coord: float,
) -> ndarray:
    """Position along the beam at natural coord ``xi`` in ``[-1, +1]``."""
    ci = np.asarray(coord_i, dtype=np.float64)
    cj = np.asarray(coord_j, dtype=np.float64)
    t = (1.0 + float(natural_coord)) / 2.0
    return ci + t * (cj - ci)


def fill_axis_for(component: str, override: str | None = None) -> str:
    """Pick local-frame fill axis (``"y"`` or ``"z"``) for a component."""
    if override is not None:
        if override not in ("y", "z"):
            raise ValueError(
                f"axis override must be 'y' or 'z' (got {override!r})."
            )
        return override
    return COMPONENT_TO_LOCAL_AXIS.get(component, "y")


def build_eid_to_endpoints(fem) -> dict[int, tuple[int, int]]:
    """Return ``{element_id: (i_node_id, j_node_id)}`` for every line element.

    Walks all element groups; only 2-node groups (line2, line3 ignoring
    midnodes) are included. Useful for correlating a
    ``LineStationSlab.element_index`` row with its parent beam.
    """
    out: dict[int, tuple[int, int]] = {}
    for group in fem.elements:
        if group.element_type.dim != 1:
            continue
        if group.npe < 2:
            continue
        ids = np.asarray(group.ids, dtype=np.int64)
        conn = np.asarray(group.connectivity, dtype=np.int64)
        for k in range(ids.size):
            out[int(ids[k])] = (int(conn[k, 0]), int(conn[k, 1]))
    return out
