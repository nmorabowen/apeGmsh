"""Partial FEMData synthesis from a ``.ladruno`` ``MODEL/`` group.

The Ladruno recorder is **self-describing**: each element group declares
its own basis (``TOPOLOGY``/``ORDER``/``NUM_CTRL``) so we can reconstruct
a partial FEMData without an element-class table. This mirrors
:func:`apeGmsh.mesh._femdata_mpco_io.read_fem_from_mpco` but for the
``.ladruno`` layout, which differs in two ways:

* ``MODEL/ELEMENTS/<classTag>-<ClassName>[..]`` is a **group** carrying a
  ``CONNECTIVITY`` dataset (rows ``(elemTag, c1..cK)``) plus the BASIS
  descriptor as attributes — *not* a flat dataset like MPCO.
* element topological dim/order come from the BASIS ``TOPOLOGY``/``ORDER``
  attrs when present (falling back to the MPCO class-name heuristic).

Like the MPCO synthesizer this is a *partial* FEMData (no apeGmsh labels,
no pre-mesh declarations); element type codes are negated class tags so
they never collide with Gmsh codes.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import h5py
    from .FEMData import FEMData


# ``12-Truss[1:0]`` / ``5-ElasticBeam3d`` / ``33000-BezierTri6[201:0]`` —
# classTag-ClassName with an optional, variable-length bracketed suffix.
_ELEM_NAME_RE = re.compile(
    r"^(?P<tag>\d+)-(?P<class>[^\[]+)(?:\[(?P<suffix>[^\]]*)\])?$"
)

# Ladruno BASIS ``TOPOLOGY`` → topological dimension. Falls back to the
# class-name heuristic when TOPOLOGY is absent or "custom".
_DIM_BY_TOPOLOGY: dict[str, int] = {
    "line": 1,
    "tri": 2,
    "quad": 2,
    "tet": 3,
    "hex": 3,
    "wedge": 3,
    "pyramid": 3,
}


def read_fem_from_ladruno(group: "h5py.Group") -> "FEMData":
    """Reconstruct a partial FEMData from a ``.ladruno`` ``MODEL/`` group.

    The argument is the ``MODEL`` group inside one ``MODEL_STAGE[..]``
    container — typically the **last** stage's MODEL (the most
    up-to-date geometry).
    """
    from ._element_types import ElementGroup, ElementTypeInfo, make_type_info
    from ._group_set import LabelSet, PhysicalGroupSet
    from .FEMData import ElementComposite, FEMData, MeshInfo, NodeComposite

    nodes_grp = group["NODES"]
    node_ids = np.asarray(nodes_grp["ID"][...], dtype=np.int64).flatten()
    coords = np.asarray(nodes_grp["COORDINATES"][...], dtype=np.float64)
    # Capture spatial dim BEFORE padding (element-dim guessing needs it).
    ndm = (
        int(coords.shape[1])
        if coords.ndim == 2 and coords.shape[1] in (2, 3) else 3
    )
    if coords.ndim == 2 and coords.shape[1] == 2:
        coords = np.hstack(
            [coords, np.zeros((coords.shape[0], 1), dtype=np.float64)]
        )

    element_groups: dict[int, ElementGroup] = {}
    types_meta: list[ElementTypeInfo] = []
    elements_grp = group.get("ELEMENTS")
    if elements_grp is not None:
        for grp_name in elements_grp:
            parsed = _parse_element_name(grp_name)
            if parsed is None:
                continue
            class_tag, class_name = parsed
            elem_grp = elements_grp[grp_name]
            if "CONNECTIVITY" not in elem_grp:
                continue
            data = np.asarray(elem_grp["CONNECTIVITY"][...], dtype=np.int64)
            if data.ndim != 2 or data.shape[1] < 2:
                continue
            ids = data[:, 0].copy()
            connectivity = data[:, 1:].copy()
            npe = connectivity.shape[1]

            dim = _dim_from_basis(elem_grp, class_name, npe, ndm)
            order = _attr_int(elem_grp, "ORDER", default=1)

            info = make_type_info(
                code=-class_tag,            # negated → never collides with Gmsh
                gmsh_name=class_name,
                dim=dim,
                order=order,
                npe=npe,
                count=ids.size,
            )
            types_meta.append(info)
            element_groups[info.code] = ElementGroup(
                element_type=info, ids=ids, connectivity=connectivity,
            )

    # Physical groups from ladruno SETS (MODEL/SETS/SET_<tag>).
    pg_dict: dict[tuple[int, int], dict] = {}
    sets_grp = group.get("SETS")
    if sets_grp is not None:
        node_id_to_idx = {int(n): i for i, n in enumerate(node_ids)}
        for tag_idx, set_name in enumerate(sets_grp.keys()):
            sub = sets_grp[set_name]
            if not _is_set_group(sub):
                continue
            pg_node_ids = (
                np.asarray(sub["NODES"][...], dtype=np.int64).flatten()
                if "NODES" in sub else np.array([], dtype=np.int64)
            )
            sel_idx = np.array(
                [node_id_to_idx[int(n)] for n in pg_node_ids
                 if int(n) in node_id_to_idx], dtype=np.int64,
            )
            pg_coords = (
                coords[sel_idx] if sel_idx.size
                else np.zeros((0, 3), dtype=np.float64)
            )
            info_d: dict = {
                "name": _strip_set_prefix(set_name),
                "node_ids": pg_node_ids,
                "node_coords": pg_coords,
            }
            if "ELEMENTS" in sub:
                info_d["element_ids"] = np.asarray(
                    sub["ELEMENTS"][...], dtype=np.int64,
                ).flatten()
            pg_dict[(3, tag_idx + 1)] = info_d

    nodes = NodeComposite(
        node_ids=node_ids, node_coords=coords,
        physical=PhysicalGroupSet(pg_dict),
        labels=LabelSet({}),
    )
    elements = ElementComposite(
        groups=element_groups,
        physical=PhysicalGroupSet(pg_dict),
        labels=LabelSet({}),
    )
    n_elems = sum(len(g) for g in element_groups.values())
    mesh_info = MeshInfo(
        n_nodes=len(node_ids), n_elems=n_elems, bandwidth=0, types=types_meta,
    )
    return FEMData(nodes=nodes, elements=elements, info=mesh_info)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _parse_element_name(name: str) -> tuple[int, str] | None:
    """``12-Truss[1:0]`` → ``(12, "Truss")``; ``5-ElasticBeam3d`` → ``(5, ...)``."""
    m = _ELEM_NAME_RE.match(name)
    if m is None:
        return None
    return int(m["tag"]), m["class"]


def _decode_attr(value) -> str:
    """Coerce an h5py string attr (bytes / 1-elem array / str) to str."""
    if isinstance(value, np.ndarray):
        value = value.flat[0] if value.size else b""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


def _attr_int(grp, name: str, *, default: int) -> int:
    """Read an int attr (scalar or 1-elem array); ``default`` if absent."""
    if name not in grp.attrs:
        return default
    v = grp.attrs[name]
    if isinstance(v, np.ndarray):
        return int(v.flat[0]) if v.size else default
    return int(v)


def _dim_from_basis(elem_grp, class_name: str, npe: int, ndm: int) -> int:
    """Prefer the self-declared ``TOPOLOGY`` attr; fall back to the
    MPCO class-name + npe/ndm heuristic."""
    if "TOPOLOGY" in elem_grp.attrs:
        topo = _decode_attr(elem_grp.attrs["TOPOLOGY"]).lower()
        if topo in _DIM_BY_TOPOLOGY:
            return _DIM_BY_TOPOLOGY[topo]
    from ._femdata_mpco_io import _guess_dim_from_class
    return _guess_dim_from_class(class_name, npe, ndm)


def _is_set_group(obj) -> bool:
    try:
        return "NODES" in obj or "ELEMENTS" in obj
    except Exception:
        return False


def _strip_set_prefix(name: str) -> str:
    return name[4:] if name.startswith("SET_") else name
