"""Partial FEMData synthesis from MPCO ``MODEL/`` group.

MPCO files carry the geometry that produced the results. We build a
*partial* FEMData from this — enough for the viewer to render and
for ID-based queries — but the synthesized object lacks:

- apeGmsh-specific ``labels`` (Tier-1 internal labels)
- Pre-mesh declarations (loads / masses / constraints)
- STKO selection-set names (those live in the ``.cdata`` sidecar,
  not the ``.mpco`` HDF5)

Element type metadata is reconstructed using **synthetic codes**:
``code = -<class_tag>`` (negative to avoid colliding with Gmsh codes
that native FEMData uses). The element type *name* mirrors the
OpenSees class name. ``snapshot_id`` therefore differs from any
native FEMData of the same mesh — that's expected; bind() will
refuse such mismatches loudly.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
from numpy import ndarray

if TYPE_CHECKING:
    import h5py
    from .FEMData import FEMData


# ``5-ElasticBeam3d[1:0]`` style names
_ELEM_NAME_RE = re.compile(
    r"^(?P<tag>\d+)-(?P<class>[^\[]+)"
    r"\[(?P<rule>\d+):(?P<custom>\d+)\]$"
)


def read_fem_from_mpco(group: "h5py.Group") -> "FEMData":
    """Reconstruct a partial FEMData from an MPCO ``MODEL/`` group.

    The argument is the ``MODEL`` group inside one ``MODEL_STAGE[…]``
    container — typically the **last** stage's MODEL (the most
    up-to-date geometry).
    """
    from ._element_types import ElementGroup, ElementTypeInfo, make_type_info
    from ._group_set import LabelSet, PhysicalGroupSet
    from .FEMData import ElementComposite, FEMData, MeshInfo, NodeComposite

    nodes_grp = group["NODES"]
    node_ids = np.asarray(nodes_grp["ID"][...], dtype=np.int64).flatten()
    coords = np.asarray(nodes_grp["COORDINATES"][...], dtype=np.float64)
    # Pad to (N, 3) if 2D — apeGmsh always stores 3D coords.
    if coords.ndim == 2 and coords.shape[1] == 2:
        coords = np.hstack(
            [coords, np.zeros((coords.shape[0], 1), dtype=np.float64)]
        )

    # Per-class element groups
    element_groups: dict[int, ElementGroup] = {}
    types_meta: list[ElementTypeInfo] = []
    elements_grp = group.get("ELEMENTS")
    if elements_grp is not None:
        for ds_name in elements_grp:
            parsed = _parse_element_name(ds_name)
            if parsed is None:
                continue
            class_tag, class_name, _rule, _custom = parsed
            data = np.asarray(elements_grp[ds_name][...], dtype=np.int64)
            if data.ndim != 2 or data.shape[1] < 2:
                continue
            ids = data[:, 0].copy()
            connectivity = data[:, 1:].copy()
            npe = connectivity.shape[1]

            # Synthetic ElementTypeInfo — code is negated class_tag
            # so it never collides with Gmsh-numbered types.
            info = make_type_info(
                code=-class_tag,
                gmsh_name=class_name,
                dim=_guess_dim_from_npe(npe, _coords_dim(coords)),
                order=1,           # MPCO doesn't expose order; assume linear
                npe=npe,
                count=ids.size,
            )
            types_meta.append(info)
            element_groups[info.code] = ElementGroup(
                element_type=info, ids=ids, connectivity=connectivity,
            )

    # Physical groups from MPCO regions (MODEL/SETS/SET_<tag>).
    # STKO selection sets live in the .cdata sidecar (not parsed here).
    pg_dict: dict[tuple[int, int], dict] = {}
    sets_grp = group.get("SETS")
    if sets_grp is not None:
        for tag_idx, set_name in enumerate(sets_grp.keys()):
            sub = sets_grp[set_name]
            if not _is_set_group(sub):
                continue
            pg_node_ids = (
                np.asarray(sub["NODES"][...], dtype=np.int64).flatten()
                if "NODES" in sub else np.array([], dtype=np.int64)
            )
            # Use coords of those nodes (lookup against full node set)
            node_id_to_idx = {int(n): i for i, n in enumerate(node_ids)}
            sel_idx = np.array(
                [node_id_to_idx[int(n)] for n in pg_node_ids
                 if int(n) in node_id_to_idx], dtype=np.int64,
            )
            pg_coords = (
                coords[sel_idx] if sel_idx.size
                else np.zeros((0, 3), dtype=np.float64)
            )
            info = {
                "name": _strip_set_prefix(set_name),
                "node_ids": pg_node_ids,
                "node_coords": pg_coords,
            }
            if "ELEMENTS" in sub:
                info["element_ids"] = np.asarray(
                    sub["ELEMENTS"][...], dtype=np.int64,
                ).flatten()
            # Use 3 as a placeholder dim (volume); MPCO doesn't carry dim.
            pg_dict[(3, tag_idx + 1)] = info

    nodes = NodeComposite(
        node_ids=node_ids, node_coords=coords,
        physical=PhysicalGroupSet(pg_dict),
        labels=LabelSet({}),       # MPCO has no apeGmsh labels
    )
    elements = ElementComposite(
        groups=element_groups,
        physical=PhysicalGroupSet(pg_dict),
        labels=LabelSet({}),
    )

    n_elems = sum(len(g) for g in element_groups.values())
    info = MeshInfo(
        n_nodes=len(node_ids),
        n_elems=n_elems,
        bandwidth=0,
        types=types_meta,
    )
    return FEMData(nodes=nodes, elements=elements, info=info)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _parse_element_name(name: str) -> tuple[int, str, int, int] | None:
    """Parse ``5-ElasticBeam3d[1:0]`` → ``(5, "ElasticBeam3d", 1, 0)``."""
    m = _ELEM_NAME_RE.match(name)
    if m is None:
        return None
    return (
        int(m["tag"]),
        m["class"],
        int(m["rule"]),
        int(m["custom"]),
    )


def _coords_dim(coords: ndarray) -> int:
    """Spatial dim implied by the coords array (2 or 3)."""
    if coords.ndim != 2:
        return 3
    # We pad 2D to 3D, so the original dim was 2 if z is all zeros.
    # That heuristic is unreliable; default to 3.
    return 3 if coords.shape[1] >= 3 else 2


def _guess_dim_from_npe(npe: int, ndm: int) -> int:
    """Best-effort element topological dim from node-per-element count."""
    # Lines: 2 or 3 nodes
    if npe in (2, 3):
        return 1
    # Triangles / quads
    if npe in (4, 8) and ndm == 2:
        return 2
    if npe == 4:
        # Could be tet (3D) or quad (2D). Default to 3D in 3D models.
        return 3 if ndm == 3 else 2
    # Solids
    if npe in (6, 8, 10, 20, 27):
        return 3
    return ndm


def _is_set_group(obj) -> bool:
    """Crude check that ``obj`` looks like a region SET group."""
    try:
        return "NODES" in obj or "ELEMENTS" in obj
    except Exception:
        return False


def _strip_set_prefix(name: str) -> str:
    """``"SET_3"`` → ``"3"``; pass through anything else."""
    return name[4:] if name.startswith("SET_") else name
