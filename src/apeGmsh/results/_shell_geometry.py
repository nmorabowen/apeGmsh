"""Shell-element local-axes + quaternion utilities.

Layered-shell DomainCapture needs per-element local-axes quaternions
to populate the :class:`~apeGmsh.results._slabs.LayerSlab` schema.
The convention follows OpenSees' shell-element formulation
(``ASDShellQ4``, ``ShellMITC4``, etc.):

- ``x_local`` = unit vector along the first edge ``(n2 - n1)``
- ``z_local`` = unit normal to the element plane
- ``y_local`` = ``z_local × x_local`` (right-handed)

The quaternion is scalar-first ``(w, x, y, z)``, matching MPCO's
``LOCAL_AXES/QUATERNIONS`` storage.

Conversion from rotation matrix to quaternion uses the Shepperd
trace-pivot method for numerical stability across the four
quadrant cases.
"""
from __future__ import annotations

import numpy as np
from numpy import ndarray


# =====================================================================
# Element class → number of corner nodes used for local-axes derivation
# =====================================================================

# Quad shells use the first 4 nodes; tri shells use 3. 9-node shells
# (ShellMITC9) use the 4 corner nodes only — internal nodes don't
# affect the element-local frame.
_QUAD_SHELL_CLASSES = frozenset({
    "ShellMITC4", "ShellDKGQ", "ShellNLDKGQ", "ASDShellQ4",
})
_TRI_SHELL_CLASSES = frozenset({
    "ShellDKGT", "ShellNLDKGT", "ASDShellT3",
})
_QUAD9_SHELL_CLASSES = frozenset({
    "ShellMITC9",
})


def _frame_edges(node_coords: ndarray, class_name: str) -> tuple[ndarray, ndarray]:
    """Return the two edge vectors used to define the local frame.

    For quads: ``(n2 - n1)``, ``(n4 - n1)``.
    For triangles: ``(n2 - n1)``, ``(n3 - n1)``.
    For 9-node shells: same as quads (corner nodes 1..4).
    """
    if class_name in _QUAD_SHELL_CLASSES or class_name in _QUAD9_SHELL_CLASSES:
        if node_coords.shape[0] < 4:
            raise ValueError(
                f"{class_name} expects ≥4 corner nodes; got "
                f"node_coords.shape={node_coords.shape}."
            )
        n1, n2, _n3, n4 = (node_coords[i] for i in range(4))
        return (n2 - n1, n4 - n1)
    if class_name in _TRI_SHELL_CLASSES:
        if node_coords.shape[0] < 3:
            raise ValueError(
                f"{class_name} expects ≥3 corner nodes; got "
                f"node_coords.shape={node_coords.shape}."
            )
        n1, n2, n3 = (node_coords[i] for i in range(3))
        return (n2 - n1, n3 - n1)
    raise ValueError(f"Unsupported shell class for local axes: {class_name!r}")


# =====================================================================
# Local-axes rotation matrix
# =====================================================================

def shell_local_axes(node_coords: ndarray, class_name: str) -> ndarray:
    """Compute the (3, 3) rotation matrix mapping global → element-local.

    Rows of the returned matrix are ``[x_local; y_local; z_local]``,
    each a unit vector in global coordinates. The convention matches
    OpenSees shell elements: x_local along the first edge, z_local
    along the element normal (right-handed), y_local = z × x.

    Parameters
    ----------
    node_coords : (n_nodes, 3) float
        Global node coordinates in element connectivity order.
    class_name : str
        OpenSees shell class name (e.g. ``"ASDShellQ4"``).

    Returns
    -------
    R : (3, 3) float64
        Rotation matrix with rows = (x_local, y_local, z_local).

    Raises
    ------
    ValueError
        If the class is not a recognised shell, or if the geometry
        is degenerate (zero-length edge or collinear nodes).
    """
    coords = np.asarray(node_coords, dtype=np.float64)
    e1, e2 = _frame_edges(coords, class_name)

    e1_norm = np.linalg.norm(e1)
    if e1_norm < 1e-14:
        raise ValueError(
            f"Degenerate shell element ({class_name}): first edge has "
            f"zero length."
        )
    x_local = e1 / e1_norm

    z_raw = np.cross(e1, e2)
    z_norm = np.linalg.norm(z_raw)
    if z_norm < 1e-14:
        raise ValueError(
            f"Degenerate shell element ({class_name}): edges are "
            f"collinear (zero normal)."
        )
    z_local = z_raw / z_norm

    y_local = np.cross(z_local, x_local)
    # y_local is unit by construction (z and x are orthonormal).

    return np.stack([x_local, y_local, z_local], axis=0)


# =====================================================================
# Rotation matrix → quaternion (scalar-first)
# =====================================================================

def rotation_matrix_to_quaternion(R: ndarray) -> ndarray:
    """Convert a (3, 3) rotation matrix to a scalar-first quaternion.

    Returns ``(w, x, y, z)`` matching MPCO's ``LOCAL_AXES/QUATERNIONS``
    convention. Uses the Shepperd trace-pivot algorithm for stability
    across all four quadrants.

    The input rows are interpreted as the body-frame basis vectors
    expressed in the world frame, which is what
    :func:`shell_local_axes` returns.
    """
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3):
        raise ValueError(f"R must be (3, 3); got {R.shape}.")

    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z], dtype=np.float64)


def shell_quaternion(node_coords: ndarray, class_name: str) -> ndarray:
    """Compose ``shell_local_axes`` + ``rotation_matrix_to_quaternion``.

    Returns a (4,) quaternion ``(w, x, y, z)``.
    """
    R = shell_local_axes(node_coords, class_name)
    return rotation_matrix_to_quaternion(R)
