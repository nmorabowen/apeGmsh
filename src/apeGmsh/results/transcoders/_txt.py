"""TXT recorder file parser (OpenSees ``.out`` format).

OpenSees Tcl/Py text recorders write space-separated columns with
``-time`` prepended. The column layout depends on the recorder
command — this module decodes node files using the metadata from
the spec's emitted ``LogicalRecorder``.

For a node recorder emitted with ``-node n1 n2 ... -dof d1 d2 ...
ops_type``, the file looks like::

    t0  v(n1,d1)  v(n1,d2)  ...  v(n1,dK)  v(n2,d1)  ...  v(nM,dK)
    t1  v(n1,d1)  ...
    ...

Total columns = 1 (time) + M nodes × K DOFs. The order is
**node-major**: node n1 first with all its DOFs, then n2, etc.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy import ndarray

if TYPE_CHECKING:
    from ...solvers._recorder_emit import LogicalRecorder


def parse_node_file(
    path: str | Path,
    logical: "LogicalRecorder",
) -> tuple[ndarray, dict[int, ndarray]]:
    """Parse a node recorder ``.out`` file into time + per-DOF arrays.

    Parameters
    ----------
    path : str or Path
        File path (e.g. emitted by ``g.opensees.export.tcl``).
    logical : LogicalRecorder
        The emitted recorder spec — gives us node order, DOFs, and
        ops_type. Must have ``kind="Node"`` and ``dofs`` populated.

    Returns
    -------
    time : ndarray (T,)
        Time vector from column 0.
    per_dof : dict[int, ndarray]
        Maps each DOF index (1-based, OpenSees convention) to a
        ``(T, N)`` array of values across the recorder's nodes (in
        the order ``logical.target_ids``).
    """
    if logical.kind != "Node":
        raise ValueError(
            f"parse_node_file expects kind='Node', got {logical.kind!r}."
        )
    if logical.dofs is None or not logical.dofs:
        raise ValueError(
            f"LogicalRecorder for {logical.record_name!r} has no DOFs."
        )

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Recorder output file not found: {path}. "
            f"Did the OpenSees analysis run successfully?"
        )

    raw = np.loadtxt(path, dtype=np.float64)
    if raw.ndim == 1:
        # Single-step files come back as 1-D; reshape to (1, n_cols).
        raw = raw[None, :]

    n_nodes = len(logical.target_ids)
    dofs = list(logical.dofs)
    n_dofs = len(dofs)
    expected_cols = 1 + n_nodes * n_dofs
    if raw.shape[1] != expected_cols:
        raise ValueError(
            f"{path}: expected {expected_cols} columns "
            f"(1 time + {n_nodes} nodes × {n_dofs} dofs), "
            f"got {raw.shape[1]}."
        )

    time = raw[:, 0]
    body = raw[:, 1:]                                   # (T, N*K)
    # Reshape to (T, N, K) — node-major, then dof
    reshaped = body.reshape(raw.shape[0], n_nodes, n_dofs)
    per_dof: dict[int, ndarray] = {}
    for k, dof in enumerate(dofs):
        # (T, N) for this DOF across all nodes in record order
        per_dof[int(dof)] = reshaped[:, :, k]
    return time, per_dof
