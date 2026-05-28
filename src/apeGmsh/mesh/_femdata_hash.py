"""Deterministic FEMData content hash for snapshot identity.

The ``snapshot_id`` is computed once at construction time over a
canonical representation of:

- Node IDs and coordinates (sorted by ID)
- Element IDs and connectivity, per element-type group (sorted by
  type name, then by element ID within each type)
- Node-side physical-group membership (sorted by (dim, tag), with
  member node IDs sorted within each group)
- Element-side physical-group membership (sorted by (dim, tag), with
  member element IDs + node IDs sorted within each group) — added in
  neutral schema 2.10.0 (B2) so element-side-only PGs influence the
  snapshot identity, closing the drift hole that the prior reader
  heuristic masked
- Node-side + element-side label membership (same canonical walk as
  PGs) — also added in 2.10.0

Two FEMData objects with byte-identical canonical representations
produce the same hash; any change (re-mesh, coord edit, connectivity
edit, PG rename, label add) produces a different hash.

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
    # Schema 2.10.0 (B2) — fold the previously un-hashed taxonomies
    # so element-side PGs + labels participate in the snapshot id.
    # Order is locked: element-side PGs, then labels (node + element).
    # Changing this order rotates every existing snapshot_id.
    _hash_element_side_pgs(h, fem)
    _hash_labels(h, fem)
    _hash_composed_from(h, fem)

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

    # Per-node ndf (shell-to-solid coupling, S1b → S2).  Skip the
    # fold when the channel is empty: ``_ndf is None`` (legacy /
    # direct-test FEMs and from_msh which has no NodeNDFComposite)
    # OR all-sentinel (from_gmsh with no ``g.node_ndf`` calls).  Both
    # cases mean "the user declared no per-node ndf", so the digest
    # must be identical — otherwise from_msh and from_gmsh of the
    # same uniform-ndf geometry would hash differently.  ``getattr``
    # tolerates ``SimpleNamespace`` mocks in
    # ``test_results_femdata_hash`` that omit ``_ndf`` entirely.
    ndf = getattr(fem.nodes, "_ndf", None)
    if ndf is not None:
        ndf_arr = np.asarray(ndf, dtype=np.int8)
        if np.any(ndf_arr != 0):
            h.update(b"NDF|")
            # Use the same node-id sort order so a permutation of the
            # node array (same ids, same ndf-per-id) produces the same
            # digest.
            h.update(ndf_arr[sort_idx].tobytes())


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

def _hash_composed_from(h: "hashlib._Hash", fem: "FEMData") -> None:
    """Fold the ``composed_from`` provenance into the digest.

    Phase 3A.1 (ADR 0038).  Skips the channel entirely when
    ``composed_from`` is empty — the uncomposed case must hash
    identically to pre-2.9.0 broker objects so existing pin tests stay
    green.  Records are folded in ascending-label order so a
    permutation at compose time (different ``Compose.add(...)``
    sequence) produces the same digest.
    """
    composed = getattr(fem, "composed_from", None)
    if not composed:
        return
    h.update(b"COMPOSED|")
    for rec in composed:
        h.update(rec.label.encode("utf-8"))
        h.update(b"|")
        h.update(rec.source_fem_hash.encode("utf-8"))
        h.update(b"|")
        h.update(rec.source_neutral_schema_version.encode("utf-8"))
        h.update(b"|")
        h.update(np.asarray(rec.translate, dtype=np.float64).tobytes())
        if rec.rotate is not None:
            h.update(b"R")
            h.update(np.asarray(rec.rotate, dtype=np.float64).tobytes())
        if rec.partition_rank is not None:
            h.update(b"P")
            h.update(int(rec.partition_rank).to_bytes(
                8, "little", signed=True))
        h.update(b"|")


def _hash_physical_groups(h: "hashlib._Hash", fem: "FEMData") -> None:
    """Fold node-side PG membership (``fem.nodes.physical``) into the digest.

    Canonical walk: sort by ``(dim, tag)``, emit ``dim|tag|name|`` then
    the sorted member node ids.  Skips the channel entirely when the
    composite has no node-side PG set (legacy / direct-test fixtures).
    """
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


def _hash_element_side_pgs(h: "hashlib._Hash", fem: "FEMData") -> None:
    """Fold element-side PG membership (``fem.elements.physical``) into
    the digest.

    Schema 2.10.0 (B2).  Mirrors :func:`_hash_physical_groups` but walks
    the element composite's PG set; emits ``dim|tag|name|`` then sorted
    member element ids then sorted member node ids (when present on
    the entry).  Skips the channel entirely when the composite has no
    element-side PG set or it is empty — the channel-empty case must
    hash identically to pre-2.10.0 broker objects that had no element-
    side PG awareness, so the tag preserves the empty hash.
    """
    h.update(b"EPGS|")
    physical = getattr(fem.elements, "physical", None)
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
            eids = np.asarray(
                physical.element_ids((dim, tag)), dtype=np.int64,
            )
        except Exception:
            eids = np.array([], dtype=np.int64)
        if eids.size:
            h.update(b"E")
            h.update(np.sort(eids).tobytes())
        try:
            nids = np.asarray(physical.node_ids((dim, tag)), dtype=np.int64)
        except Exception:
            nids = np.array([], dtype=np.int64)
        if nids.size:
            h.update(b"N")
            h.update(np.sort(nids).tobytes())


def _hash_labels(h: "hashlib._Hash", fem: "FEMData") -> None:
    """Fold node-side + element-side label membership into the digest.

    Schema 2.10.0 (B2).  Folds ``fem.nodes.labels`` then
    ``fem.elements.labels`` — each with the same canonical walk used
    by the PG folds: sort by ``(dim, tag)``, emit ``dim|tag|name|``,
    then sorted member node ids (and for the element side, sorted
    member element ids).  Skips the channel entirely when both sides
    are absent / empty.
    """
    h.update(b"LBL|")
    node_side = getattr(fem.nodes, "labels", None)
    if node_side is not None:
        try:
            keys = sorted(node_side.get_all())
        except Exception:
            keys = []
        for (dim, tag) in keys:
            try:
                name = node_side.get_name(dim, tag)
            except Exception:
                name = ""
            h.update(f"N|{dim}|{tag}|{name}|".encode("utf-8"))
            try:
                nids = np.asarray(
                    node_side.node_ids((dim, tag)), dtype=np.int64,
                )
            except Exception:
                continue
            if nids.size:
                h.update(np.sort(nids).tobytes())

    elem_side = getattr(fem.elements, "labels", None)
    if elem_side is not None:
        try:
            keys = sorted(elem_side.get_all())
        except Exception:
            keys = []
        for (dim, tag) in keys:
            try:
                name = elem_side.get_name(dim, tag)
            except Exception:
                name = ""
            h.update(f"E|{dim}|{tag}|{name}|".encode("utf-8"))
            try:
                eids = np.asarray(
                    elem_side.element_ids((dim, tag)), dtype=np.int64,
                )
            except Exception:
                eids = np.array([], dtype=np.int64)
            if eids.size:
                h.update(b"E")
                h.update(np.sort(eids).tobytes())
            try:
                nids = np.asarray(
                    elem_side.node_ids((dim, tag)), dtype=np.int64,
                )
            except Exception:
                nids = np.array([], dtype=np.int64)
            if nids.size:
                h.update(b"N")
                h.update(np.sort(nids).tobytes())
