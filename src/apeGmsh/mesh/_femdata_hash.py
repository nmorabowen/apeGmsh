"""Deterministic FEMData content hash for snapshot identity.

The ``snapshot_id`` is computed once at construction time over a
canonical representation of:

- Node IDs and coordinates (sorted by ID)
- Element IDs and connectivity, per element-type group (sorted by
  type name, then by element ID within each type)
- Physical-group membership (sorted by (dim, tag), with member node
  IDs sorted within each group)

Two FEMData objects with byte-identical canonical representations
produce the same hash; any change (re-mesh, coord edit, connectivity
edit, PG rename) produces a different hash.

This hash is the contract that ties recorder specs, results files,
and bound FEMData together — see ``Results_architecture.md`` §
"FEMData embedding & binding".
"""
from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .FEMData import FEMData


_DIGEST_SIZE = 16  # 16 bytes → 32 hex chars


def compute_snapshot_id(fem: "FEMData") -> str:
    """Return a hex digest identifying this FEMData's contents.

    Length: 32 hex characters (128-bit blake2b digest).
    """
    h = hashlib.blake2b(digest_size=_DIGEST_SIZE)

    _hash_nodes(h, fem)
    _hash_elements(h, fem)
    _hash_physical_groups(h, fem)

    return h.hexdigest()


# ---------------------------------------------------------------------
# Section: nodes
# ---------------------------------------------------------------------

def _hash_nodes(h: "hashlib._Hash", fem: "FEMData") -> None:
    h.update(b"NODES|")
    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    coords = np.asarray(fem.nodes.coords, dtype=np.float64)
    if node_ids.size == 0:
        return
    sort_idx = np.argsort(node_ids, kind="stable")
    h.update(node_ids[sort_idx].tobytes())
    h.update(coords[sort_idx].tobytes())


# ---------------------------------------------------------------------
# Section: elements
# ---------------------------------------------------------------------

def _hash_elements(h: "hashlib._Hash", fem: "FEMData") -> None:
    h.update(b"ELEMENTS|")
    # Iterate in type-name order so the result is permutation-stable.
    groups = list(fem.elements)
    groups.sort(key=lambda g: g.type_name)
    for group in groups:
        eids = np.asarray(group.ids, dtype=np.int64)
        if eids.size == 0:
            continue
        h.update(group.type_name.encode("utf-8"))
        h.update(b"|")
        conn = np.asarray(group.connectivity, dtype=np.int64)
        sort_idx = np.argsort(eids, kind="stable")
        h.update(eids[sort_idx].tobytes())
        h.update(conn[sort_idx].tobytes())


# ---------------------------------------------------------------------
# Section: physical groups
# ---------------------------------------------------------------------

def _hash_physical_groups(h: "hashlib._Hash", fem: "FEMData") -> None:
    h.update(b"PGS|")
    physical = getattr(fem.nodes, "physical", None)
    if physical is None:
        return
    try:
        all_pgs = sorted(physical.get_all())
    except Exception:
        return
    for (dim, tag) in all_pgs:
        try:
            name = physical.get_name(dim, tag)
        except Exception:
            name = ""
        h.update(f"{dim}|{tag}|{name}|".encode("utf-8"))
        try:
            nids = np.asarray(physical.node_ids((dim, tag)), dtype=np.int64)
        except Exception:
            continue
        if nids.size:
            h.update(np.sort(nids).tobytes())
