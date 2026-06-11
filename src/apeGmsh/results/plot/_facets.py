"""Facet extraction for static plots.

Walks ``FEMData.elements`` once and returns the renderable primitives
used by ``ResultsPlot``: triangles for surfaces (and the boundary of
solids) and line segments for 1-D elements.

Why boundary-only for solids: drawing every interior tet/hex face
produces a visually opaque mass with overlapping polygons; only the
outer hull carries useful information for a static figure. A face is
on the boundary when it appears in exactly one element.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy import ndarray

if TYPE_CHECKING:
    from apeGmsh.mesh.FEMData import FEMData


# Local-node-index sequences for the faces of each volume element type.
# Winding is outward-facing (Gmsh convention) but only the *set* matters
# for boundary detection; the recorded ordering is reused when emitting
# the triangle.
_VOLUME_FACES: dict[str, list[tuple[int, ...]]] = {
    "tet4": [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)],
    "hex8": [
        (0, 3, 2, 1), (4, 5, 6, 7),
        (0, 1, 5, 4), (1, 2, 6, 5),
        (2, 3, 7, 6), (3, 0, 4, 7),
    ],
}

_SURFACE_TYPES = {"tri3", "quad4"}


def _quad_to_tris(q: tuple[int, int, int, int]) -> list[tuple[int, int, int]]:
    return [(q[0], q[1], q[2]), (q[0], q[2], q[3])]


def extract_facets(
    fem: "FEMData",
) -> tuple[ndarray, ndarray]:
    """Return ``(triangles, segments)`` of node IDs.

    ``triangles`` shape ``(M, 3)``: the node IDs of every renderable
    triangle. Volume elements contribute only their boundary faces;
    surface elements contribute themselves; quads split on the
    ``(0, 1, 2)`` + ``(0, 2, 3)`` diagonal.

    ``segments`` shape ``(S, 2)``: 1-D element endpoints (``line3``
    midnodes are dropped — visually equivalent for static figures).
    """
    tris, segs, _, _ = extract_facets_owned(fem)
    return tris, segs


def extract_facets_owned(
    fem: "FEMData",
) -> tuple[ndarray, ndarray, ndarray, ndarray]:
    """:func:`extract_facets` plus per-facet element ownership.

    Returns ``(triangles, segments, tri_owner, seg_owner)`` where
    ``tri_owner[m]`` / ``seg_owner[s]`` is the FEM element ID the facet
    belongs to. Ownership is unambiguous: a boundary face appears in
    exactly one volume element, surface/line facets are their element.
    The discrete (unaveraged) gauss contour needs it to paint each
    facet from its *own* element's corner values.
    """
    tris: list[tuple[int, int, int]] = []
    segs: list[tuple[int, int]] = []
    tri_owner: list[int] = []
    seg_owner: list[int] = []

    # ── Volume elements: boundary-face extraction ─────────────────
    face_count: dict[frozenset, int] = {}
    face_first: dict[frozenset, tuple[int, ...]] = {}
    face_eid: dict[frozenset, int] = {}

    for group in fem.elements:
        tname = group.type_name
        if tname not in _VOLUME_FACES:
            continue
        face_defs = _VOLUME_FACES[tname]
        conn = np.asarray(group.connectivity, dtype=np.int64)
        eids = np.asarray(group.ids, dtype=np.int64)
        for k, row in enumerate(conn):
            for face in face_defs:
                ids = tuple(int(row[i]) for i in face)
                key = frozenset(ids)
                face_count[key] = face_count.get(key, 0) + 1
                if key not in face_first:
                    face_first[key] = ids
                    face_eid[key] = int(eids[k])

    for key, count in face_count.items():
        if count != 1:
            continue
        ids = face_first[key]
        owner = face_eid[key]
        if len(ids) == 3:
            tris.append(ids)  # type: ignore[arg-type]
            tri_owner.append(owner)
        elif len(ids) == 4:
            tris.extend(_quad_to_tris(ids))  # type: ignore[arg-type]
            tri_owner.extend((owner, owner))

    # ── Surface elements: drawn directly ──────────────────────────
    for group in fem.elements:
        tname = group.type_name
        if tname not in _SURFACE_TYPES:
            continue
        conn = np.asarray(group.connectivity, dtype=np.int64)
        eids = np.asarray(group.ids, dtype=np.int64)
        if tname == "tri3":
            for k, row in enumerate(conn):
                tris.append((int(row[0]), int(row[1]), int(row[2])))
                tri_owner.append(int(eids[k]))
        else:    # quad4
            for k, row in enumerate(conn):
                tris.extend(_quad_to_tris(
                    (int(row[0]), int(row[1]), int(row[2]), int(row[3])),
                ))
                tri_owner.extend((int(eids[k]), int(eids[k])))

    # ── 1-D elements: line segments ───────────────────────────────
    # By dimension, not type name: a .ladruno-synthesized FEMData carries
    # solver-flavoured 1-D groups ("truss", "beam", …) whose endpoints are
    # the first two nodes — same convention as line2/line3 (midnodes are
    # dropped, visually equivalent for static figures).
    for group in fem.elements:
        if group.element_type.dim != 1 or group.element_type.npe < 2:
            continue
        conn = np.asarray(group.connectivity, dtype=np.int64)
        eids = np.asarray(group.ids, dtype=np.int64)
        for k, row in enumerate(conn):
            segs.append((int(row[0]), int(row[1])))
            seg_owner.append(int(eids[k]))

    tri_arr = (
        np.asarray(tris, dtype=np.int64).reshape(-1, 3)
        if tris else np.empty((0, 3), dtype=np.int64)
    )
    seg_arr = (
        np.asarray(segs, dtype=np.int64).reshape(-1, 2)
        if segs else np.empty((0, 2), dtype=np.int64)
    )
    return (
        tri_arr,
        seg_arr,
        np.asarray(tri_owner, dtype=np.int64),
        np.asarray(seg_owner, dtype=np.int64),
    )


def coords_lookup(fem: "FEMData") -> tuple[ndarray, ndarray]:
    """Return ``(id_to_idx, coords)`` for O(1) node-ID → coordinate lookup.

    ``id_to_idx[node_id]`` gives the row in ``coords`` (or -1 for missing).
    """
    ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    coords = np.asarray(fem.nodes.coords, dtype=np.float64)
    if ids.size == 0:
        return np.empty(0, dtype=np.int64), coords
    max_id = int(ids.max())
    lookup = np.full(max_id + 1, -1, dtype=np.int64)
    lookup[ids] = np.arange(ids.size, dtype=np.int64)
    return lookup, coords
