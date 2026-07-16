"""Constructor-time mesh snapshot for the section analyzer (ADR 0078).

Extracts everything :class:`SectionProperties` needs from a ``FEMData``
into flat numpy arrays, applies every input gate, and — per the ADR 0001
doctrine — leaves the analyzer independent of the gmsh session.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping

import numpy as np
from numpy import ndarray

from ._errors import SectionMeshError
from ._materials import GEOMETRIC_ONLY, SectionMaterial

if TYPE_CHECKING:  # pragma: no cover
    from apeGmsh.mesh.FEMData import FEMData

# Supported gmsh 2-D element-type codes and their corner counts.
_SUPPORTED_2D: dict[int, tuple[str, int]] = {
    2: ("tri3", 3),
    9: ("tri6", 3),
    3: ("quad4", 4),
    16: ("quad8", 4),
    10: ("quad9", 4),
}

#: Codes whose shape functions are linear (accuracy warning in warping).
LINEAR_2D_CODES: frozenset[int] = frozenset({2, 3})

# Relative tolerance for the one-plane (z = const) gate.
_PLANAR_RTOL = 1e-9


@dataclass(frozen=True, slots=True)
class _Block:
    """One homogeneous element block, row-indexed into the snapshot."""

    code: int                # gmsh element-type code
    type_name: str           # 'tri6', ...
    n_corners: int           # 3 for triangles, 4 for quads
    eids: ndarray            # (E,) int64 — original element IDs
    conn: ndarray            # (E, npe) int64 — ROW indices into coords
    mat_idx: ndarray         # (E,) int64 — index into materials tuple


@dataclass(frozen=True, slots=True)
class SectionSnapshot:
    """Frozen solver input: coordinates, connectivity, material map."""

    coords: ndarray                        # (N, 2) float64, authoring axes
    node_ids: ndarray                      # (N,) int64 — original node IDs
    blocks: tuple[_Block, ...]
    materials: tuple[SectionMaterial, ...]
    material_names: tuple[str, ...]        # PG names, aligned with materials
    geometric_only: bool
    node_component: ndarray                # (N,) int64 — connected-component id
    n_components: int

    @property
    def n_elements(self) -> int:
        return sum(len(b.eids) for b in self.blocks)

    @property
    def single_moduli(self) -> tuple[float, float] | None:
        """``(E, G)`` when every material shares one modulus pair, else
        ``None`` (composite)."""
        pairs = {(m.E, m.shear_modulus) for m in self.materials}
        if len(pairs) == 1:
            return next(iter(pairs))
        return None


def build_snapshot(
    fem: "FEMData",
    materials: Mapping[str, SectionMaterial] | None,
    *,
    name: str | None,
) -> SectionSnapshot:
    """Extract + gate.  Raises :class:`SectionMeshError` on any violation."""
    handle = name or "section"

    # ── collect 2-D groups; reject solids ─────────────────────────────
    groups_2d = []
    bad_types: list[str] = []
    for group in fem.elements:
        dim = group.element_type.dim
        if dim == 3:
            bad_types.append(group.type_name)
        elif dim == 2:
            if group.element_type.code not in _SUPPORTED_2D:
                bad_types.append(group.type_name)
            else:
                groups_2d.append(group)
        # dim 0/1 (auxiliary points / boundary lines) carry no area and
        # are ignored.
    if bad_types:
        raise SectionMeshError(
            f"{handle}: cross-section meshes must contain only planar "
            f"tri3/tri6/quad4/quad8/quad9 elements; got {sorted(set(bad_types))}. "
            f"Mesh the face with g.mesh.generation.generate(dim=2) and pass "
            f"g.mesh.queries.get_fem_data(dim=2)."
        )
    if not groups_2d:
        raise SectionMeshError(
            f"{handle}: no 2-D elements found — mesh the section face with "
            f"g.mesh.generation.generate(dim=2) first."
        )

    # ── referenced nodes → row indexing ───────────────────────────────
    all_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    all_xyz = np.asarray(fem.nodes.coords, dtype=np.float64)
    used = np.unique(np.concatenate([g.connectivity.ravel() for g in groups_2d]))
    id_to_row = {int(nid): i for i, nid in enumerate(used)}
    # rows of `used` inside the full node table
    full_row = {int(nid): i for i, nid in enumerate(all_ids)}
    try:
        take = np.fromiter(
            (full_row[int(nid)] for nid in used), dtype=np.int64, count=len(used)
        )
    except KeyError as exc:  # connectivity references a missing node
        raise SectionMeshError(
            f"{handle}: element connectivity references node {exc} not present "
            f"in the FEMData node table."
        ) from exc
    xyz = all_xyz[take]

    # ── one-plane gate (authoring plane = global XY) ──────────────────
    extents = np.ptp(xyz, axis=0) if len(xyz) else np.zeros(3)
    span = float(max(extents[0], extents[1]))
    z_extent = float(extents[2])
    if z_extent > _PLANAR_RTOL * max(1.0, span):
        raise SectionMeshError(
            f"{handle}: section nodes span Δz = {z_extent:.3e} — the analyzer "
            f"requires the face authored in the global XY plane (ADR 0078 "
            f"authoring-axes contract)."
        )
    coords = np.ascontiguousarray(xyz[:, :2])

    # ── material map (PG exact-cover) ─────────────────────────────────
    if materials is None:
        mats: tuple[SectionMaterial, ...] = (GEOMETRIC_ONLY,)
        mat_names: tuple[str, ...] = ("geometric-only",)
        geometric_only = True
        eid_to_mat: dict[int, int] | None = None
    else:
        if not materials:
            raise SectionMeshError(
                f"{handle}: materials= was given but is empty — pass a "
                f"non-empty PG-name → SectionMaterial mapping, or omit it "
                f"for geometric-only mode."
            )
        mats = tuple(materials.values())
        mat_names = tuple(materials.keys())
        geometric_only = False
        eid_to_mat = {}
        pgs = fem.elements.physical
        for m_i, pg_name in enumerate(mat_names):
            try:
                eids = pgs.element_ids(pg_name, dim=2)
            except (KeyError, ValueError) as exc:
                raise SectionMeshError(
                    f"{handle}: materials= names physical group "
                    f"'{pg_name}' which has no 2-D elements ({exc})."
                ) from exc
            for eid in np.asarray(eids).ravel():
                eid_i = int(eid)
                prev = eid_to_mat.setdefault(eid_i, m_i)
                if prev != m_i:
                    raise SectionMeshError(
                        f"{handle}: element {eid_i} is claimed by both "
                        f"'{mat_names[prev]}' and '{pg_name}' — material "
                        f"regions must partition the section (fragment "
                        f"overlapping faces and assign disjoint PGs)."
                    )

    # ── per-block arrays ──────────────────────────────────────────────
    blocks: list[_Block] = []
    uncovered: list[int] = []
    for group in groups_2d:
        type_name, n_corners = _SUPPORTED_2D[group.element_type.code]
        flat = group.connectivity.ravel()
        conn_flat = np.fromiter(
            (id_to_row[int(n)] for n in flat), dtype=np.int64, count=flat.size
        )
        conn = conn_flat.reshape(group.connectivity.shape)
        eids = np.asarray(group.ids, dtype=np.int64)
        if eid_to_mat is None:
            mat_idx = np.zeros(len(eids), dtype=np.int64)
        else:
            mat_idx = np.empty(len(eids), dtype=np.int64)
            for row, eid in enumerate(eids):
                m = eid_to_mat.get(int(eid))
                if m is None:
                    uncovered.append(int(eid))
                    m = -1
                mat_idx[row] = m
        blocks.append(
            _Block(
                code=group.element_type.code,
                type_name=type_name,
                n_corners=n_corners,
                eids=eids,
                conn=conn,
                mat_idx=mat_idx,
            )
        )
    if uncovered:
        shown = ", ".join(str(e) for e in uncovered[:8])
        more = f" (+{len(uncovered) - 8} more)" if len(uncovered) > 8 else ""
        raise SectionMeshError(
            f"{handle}: {len(uncovered)} element(s) are covered by no "
            f"materials= physical group: {shown}{more}. Every 2-D element "
            f"must belong to exactly one named PG."
        )

    # ── connected components (node-adjacency star per element) ────────
    node_component, n_components = _connected_components(len(coords), blocks)

    return SectionSnapshot(
        coords=coords,
        node_ids=used,
        blocks=tuple(blocks),
        materials=mats,
        material_names=mat_names,
        geometric_only=geometric_only,
        node_component=node_component,
        n_components=n_components,
    )


def _connected_components(
    n_nodes: int, blocks: list[_Block]
) -> tuple[ndarray, int]:
    """Label mesh-connectivity components (scipy.sparse.csgraph)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    rows: list[ndarray] = []
    cols: list[ndarray] = []
    for b in blocks:
        # star coupling: node 0 of each element to every other node —
        # sufficient for component labeling, far sparser than all-pairs.
        first = np.repeat(b.conn[:, 0], b.conn.shape[1] - 1)
        rest = b.conn[:, 1:].ravel()
        rows.append(first)
        cols.append(rest)
    r = np.concatenate(rows)
    c = np.concatenate(cols)
    graph = coo_matrix(
        (np.ones(len(r), dtype=np.int8), (r, c)), shape=(n_nodes, n_nodes)
    )
    n_comp, labels = connected_components(graph, directed=False)
    return labels.astype(np.int64), int(n_comp)
