"""FEMData ↔ apeGmsh native HDF5 ``/model/`` group.

Phase 1 scope: nodes, elements (per type), physical groups, labels.
Loads/masses/constraints are deferred — they don't affect
``snapshot_id`` and the viewer rebuild does not require them.

The HDF5 layout under ``/model/`` is::

    /model/
    ├── attrs:
    │   ├── snapshot_id      str
    │   ├── ndm              int (best-effort, derived from element dims)
    │   ├── ndf              int (currently always 0 — see note)
    │   ├── model_name       str
    │   └── units            str
    ├── nodes/
    │   ├── ids              (N,) int64
    │   ├── coords           (N, 3) float64
    │   └── physical_groups/<name>/
    │       ├── attrs: dim, tag
    │       ├── node_ids       (Np,) int64
    │       └── node_coords    (Np, 3) float64
    │   └── labels/<name>/  (same shape as physical_groups)
    └── elements/
        ├── per-type subgroups: <type_name>/
        │   ├── attrs: dim, npe, code, gmsh_name, alias, order
        │   ├── ids            (Eg,) int64
        │   └── connectivity   (Eg, npe) int64
        ├── physical_groups/<name>/
        │   ├── attrs: dim, tag
        │   ├── node_ids        (Np,) int64
        │   ├── node_coords     (Np, 3) float64
        │   └── element_ids     (Ep,) int64
        └── labels/<name>/   (same shape as elements/physical_groups)

Note on ``ndf``: FEMData itself does not currently carry an ``ndf``
attribute (it lives on the OpenSees bridge). Phase 1 stores 0; later
phases that thread ``ndf`` through ``get_fem_data()`` will populate it.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import h5py
    from .FEMData import FEMData


# =====================================================================
# Public API
# =====================================================================

def write_fem_to_h5(fem: "FEMData", group: "h5py.Group") -> None:
    """Embed a FEMData snapshot into an open HDF5 group (``/model/``)."""
    _write_attrs(fem, group)
    _write_nodes(fem, group.require_group("nodes"))
    _write_elements(fem, group.require_group("elements"))
    _write_mesh_selection(fem, group)


def read_fem_from_h5(group: "h5py.Group") -> "FEMData":
    """Reconstruct a FEMData from its embedded ``/model/`` group.

    The reconstructed FEMData carries: nodes, elements (per type),
    physical groups, labels. Loads/masses/constraints are empty.

    The reconstructed FEMData's ``snapshot_id`` matches the embedded
    one — this is the linking contract used by ``Results.bind()``.
    """
    from ._element_types import ElementGroup, ElementTypeInfo, make_type_info
    from ._group_set import LabelSet, PhysicalGroupSet
    from .FEMData import ElementComposite, FEMData, MeshInfo, NodeComposite

    nodes_grp = group["nodes"]
    elem_grp = group["elements"]

    node_ids = np.asarray(nodes_grp["ids"][...], dtype=np.int64)
    node_coords = np.asarray(nodes_grp["coords"][...], dtype=np.float64)

    # Per-type element groups
    element_groups: dict[int, ElementGroup] = {}
    types_meta: list[ElementTypeInfo] = []
    for type_name in sorted(_subgroup_names(elem_grp)):
        if type_name in ("physical_groups", "labels"):
            continue
        sub = elem_grp[type_name]
        ids = np.asarray(sub["ids"][...], dtype=np.int64)
        conn = np.asarray(sub["connectivity"][...], dtype=np.int64)
        attrs = sub.attrs
        npe = int(attrs.get("npe", conn.shape[1] if conn.ndim == 2 else 0))
        info = make_type_info(
            code=int(attrs.get("code", 0)),
            gmsh_name=str(attrs.get("gmsh_name", type_name)),
            dim=int(attrs.get("dim", 0)),
            order=int(attrs.get("order", 1)),
            npe=npe,
            count=ids.shape[0],
        )
        types_meta.append(info)
        element_groups[info.code] = ElementGroup(
            element_type=info, ids=ids, connectivity=conn,
        )

    # Physical groups & labels — separate dicts for nodes and elements,
    # but they share the same source so we read once per side.
    node_pgs = _read_named_groups(
        nodes_grp.get("physical_groups"),
        node_ids=node_ids, node_coords=node_coords,
        is_element_side=False,
    )
    node_labels = _read_named_groups(
        nodes_grp.get("labels"),
        node_ids=node_ids, node_coords=node_coords,
        is_element_side=False,
    )
    elem_pgs = _read_named_groups(
        elem_grp.get("physical_groups"),
        node_ids=node_ids, node_coords=node_coords,
        is_element_side=True,
    )
    elem_labels = _read_named_groups(
        elem_grp.get("labels"),
        node_ids=node_ids, node_coords=node_coords,
        is_element_side=True,
    )

    nodes = NodeComposite(
        node_ids=node_ids, node_coords=node_coords,
        physical=PhysicalGroupSet(node_pgs),
        labels=LabelSet(node_labels),
    )
    elements = ElementComposite(
        groups=element_groups,
        physical=PhysicalGroupSet(elem_pgs),
        labels=LabelSet(elem_labels),
    )

    n_elems = sum(len(g) for g in element_groups.values())
    info = MeshInfo(
        n_nodes=len(node_ids),
        n_elems=n_elems,
        bandwidth=0,        # not tracked across the embed boundary
        types=types_meta,
    )

    mesh_selection = _read_mesh_selection(group.get("mesh_selection"))

    return FEMData(
        nodes=nodes, elements=elements, info=info,
        mesh_selection=mesh_selection,
    )


# =====================================================================
# Write helpers
# =====================================================================

def _write_attrs(fem: "FEMData", group: "h5py.Group") -> None:
    group.attrs["snapshot_id"] = fem.snapshot_id

    # Best-effort ndm: max element dim, falling back to 3.
    ndm = 3
    try:
        dims = [g.dim for g in fem.elements if hasattr(g, "dim")]
        if dims:
            ndm = max(dims)
    except Exception:
        pass
    group.attrs["ndm"] = int(ndm)
    group.attrs["ndf"] = 0

    # Optional metadata
    group.attrs["model_name"] = ""
    group.attrs["units"] = ""


def _write_nodes(fem: "FEMData", group: "h5py.Group") -> None:
    node_ids = np.asarray(fem.nodes.ids, dtype=np.int64)
    coords = np.asarray(fem.nodes.coords, dtype=np.float64)
    group.create_dataset("ids", data=node_ids)
    group.create_dataset("coords", data=coords)

    physical = getattr(fem.nodes, "physical", None)
    if physical is not None:
        _write_named_groups(group.require_group("physical_groups"), physical)
    labels = getattr(fem.nodes, "labels", None)
    if labels is not None:
        _write_named_groups(group.require_group("labels"), labels)


def _write_elements(fem: "FEMData", group: "h5py.Group") -> None:
    for elem_group in fem.elements:
        if elem_group.ids.size == 0:
            continue
        # Use type_name as subgroup name; sanitize to avoid '/'.
        type_name = elem_group.type_name.replace("/", "_")
        sub = group.create_group(type_name)
        et = elem_group.element_type
        sub.attrs["code"] = int(et.code)
        sub.attrs["gmsh_name"] = str(et.gmsh_name)
        sub.attrs["npe"] = int(et.npe)
        sub.attrs["dim"] = int(et.dim)
        sub.attrs["order"] = int(et.order)
        sub.create_dataset("ids", data=np.asarray(elem_group.ids, dtype=np.int64))
        sub.create_dataset(
            "connectivity",
            data=np.asarray(elem_group.connectivity, dtype=np.int64),
        )

    physical = getattr(fem.elements, "physical", None)
    if physical is not None:
        _write_named_groups(group.require_group("physical_groups"), physical)
    labels = getattr(fem.elements, "labels", None)
    if labels is not None:
        _write_named_groups(group.require_group("labels"), labels)


def _write_named_groups(parent: "h5py.Group", group_set) -> None:
    """Write a NamedGroupSet (PhysicalGroupSet or LabelSet) under ``parent``.

    Each named group becomes ``parent/<sanitized_name>/`` with attrs
    ``dim``, ``tag`` and datasets ``node_ids``, ``node_coords``, and
    optionally ``element_ids``.
    """
    seen: set[str] = set()
    for (dim, tag) in group_set.get_all():
        try:
            name = group_set.get_name(dim, tag) or f"_unnamed_{dim}_{tag}"
        except Exception:
            name = f"_unnamed_{dim}_{tag}"

        # Sanitize for HDF5 path; disambiguate name collisions across dims.
        safe = name.replace("/", "_")
        if safe in seen:
            safe = f"{safe}__{dim}_{tag}"
        seen.add(safe)

        sub = parent.create_group(safe)
        sub.attrs["dim"] = int(dim)
        sub.attrs["tag"] = int(tag)

        try:
            nids = np.asarray(
                group_set.node_ids((dim, tag)), dtype=np.int64,
            )
            ncoords = np.asarray(
                group_set.node_coords((dim, tag)), dtype=np.float64,
            )
            sub.create_dataset("node_ids", data=nids)
            sub.create_dataset("node_coords", data=ncoords)
        except Exception:
            sub.create_dataset("node_ids", data=np.array([], dtype=np.int64))
            sub.create_dataset(
                "node_coords",
                data=np.zeros((0, 3), dtype=np.float64),
            )

        try:
            eids = np.asarray(
                group_set.element_ids((dim, tag)), dtype=np.int64,
            )
            sub.create_dataset("element_ids", data=eids)
        except Exception:
            pass     # dim=0 PGs have no element data; that's fine


# =====================================================================
# Read helpers
# =====================================================================

def _subgroup_names(group) -> list[str]:
    if group is None:
        return []
    return [k for k in group.keys() if hasattr(group[k], "keys")]


def _read_named_groups(
    parent,
    *,
    node_ids: np.ndarray,
    node_coords: np.ndarray,
    is_element_side: bool,
) -> dict[tuple[int, int], dict]:
    """Reconstruct the dict accepted by NamedGroupSet.__init__."""
    out: dict[tuple[int, int], dict] = {}
    if parent is None:
        return out
    for name in parent.keys():
        sub = parent[name]
        attrs = sub.attrs
        dim = int(attrs.get("dim", 0))
        tag = int(attrs.get("tag", 0))
        info: dict = {"name": name}

        if "node_ids" in sub:
            info["node_ids"] = np.asarray(sub["node_ids"][...], dtype=np.int64)
        if "node_coords" in sub:
            info["node_coords"] = np.asarray(
                sub["node_coords"][...], dtype=np.float64,
            )

        if is_element_side and "element_ids" in sub:
            info["element_ids"] = np.asarray(
                sub["element_ids"][...], dtype=np.int64,
            )

        out[(dim, tag)] = info
    return out


# =====================================================================
# Mesh selection round-trip
# =====================================================================
#
# MeshSelectionStore is structurally similar to PhysicalGroupSet — a
# dict keyed by ``(dim, tag)`` with ``name`` / ``node_ids`` /
# ``node_coords`` and optional ``element_ids`` / ``connectivity``.
# We give it its own subgroup ``/model/mesh_selection/`` so the
# embedded FEMData round-trips post-mesh selections (created via
# ``g.mesh_selection.add_nodes(...)`` / ``add_elements(...)``).


def _write_mesh_selection(fem: "FEMData", model_group: "h5py.Group") -> None:
    store = getattr(fem, "mesh_selection", None)
    if store is None or len(store) == 0:
        return

    parent = model_group.create_group("mesh_selection")
    seen: set[str] = set()
    for (dim, tag) in store.get_all():
        try:
            name = store.get_name(dim, tag) or f"_unnamed_{dim}_{tag}"
        except Exception:
            name = f"_unnamed_{dim}_{tag}"

        safe = name.replace("/", "_")
        if safe in seen:
            safe = f"{safe}__{dim}_{tag}"
        seen.add(safe)

        sub = parent.create_group(safe)
        sub.attrs["dim"] = int(dim)
        sub.attrs["tag"] = int(tag)

        # Node-side data is always present (even for dim>=1 element
        # selections, the implied unique node IDs are stored).
        try:
            nodes_data = store.get_nodes(dim, tag)
            sub.create_dataset(
                "node_ids",
                data=np.asarray(nodes_data["tags"], dtype=np.int64),
            )
            sub.create_dataset(
                "node_coords",
                data=np.asarray(nodes_data["coords"], dtype=np.float64),
            )
        except Exception:
            sub.create_dataset("node_ids", data=np.array([], dtype=np.int64))
            sub.create_dataset(
                "node_coords",
                data=np.zeros((0, 3), dtype=np.float64),
            )

        # Element-side data only for dim>=1 selections.
        if dim >= 1:
            try:
                elem_data = store.get_elements(dim, tag)
                sub.create_dataset(
                    "element_ids",
                    data=np.asarray(
                        elem_data["element_ids"], dtype=np.int64,
                    ),
                )
                sub.create_dataset(
                    "connectivity",
                    data=np.asarray(elem_data["connectivity"], dtype=np.int64),
                )
            except Exception:
                pass


def _read_mesh_selection(parent):
    """Reconstruct a MeshSelectionStore from its embedded subgroup.

    Returns ``None`` if no mesh_selection group is present (matches
    the FEMData convention of ``mesh_selection=None`` for fems without
    any selections).
    """
    if parent is None:
        return None
    from .MeshSelectionSet import MeshSelectionStore

    sets: dict[tuple[int, int], dict] = {}
    for name in parent.keys():
        sub = parent[name]
        attrs = sub.attrs
        dim = int(attrs.get("dim", 0))
        tag = int(attrs.get("tag", 0))
        info: dict = {"name": name}

        if "node_ids" in sub:
            info["node_ids"] = np.asarray(sub["node_ids"][...], dtype=np.int64)
            info["node_coords"] = np.asarray(
                sub["node_coords"][...], dtype=np.float64,
            )
        if "element_ids" in sub:
            info["element_ids"] = np.asarray(
                sub["element_ids"][...], dtype=np.int64,
            )
        if "connectivity" in sub:
            info["connectivity"] = np.asarray(
                sub["connectivity"][...], dtype=np.int64,
            )

        sets[(dim, tag)] = info

    if not sets:
        return None
    return MeshSelectionStore(sets)
