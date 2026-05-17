"""ViewerData — read-only structural snapshot consumed by the viewer.

Phase 8.7 commit 3 introduces this adapter as the viewer package's
single point of consumption.  Two builders, one shape:

* :meth:`ViewerData.from_fem` — live :class:`apeGmsh.mesh.FEMData`
  path (g.mesh.viewer / pre-results).
* :meth:`ViewerData.from_h5` — ``model.h5`` path (results viewer
  fixtures + post-solve sidecar discovery).

Downstream viewer code does not learn which source the snapshot came
from — that's the point of the adapter, and the reason Phase 8.7
ships its acceptance test as an AST-walk forbidding ``from
apeGmsh.mesh ...`` imports in ``viewers/``.

See [phase-8.7-scope.md](../../opensees/architecture/phase-8.7-scope.md)
and [ADR 0014](../../opensees/architecture/decisions/0014-viewer-is-pure-h5-consumer.md).
"""
from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy import ndarray

from ._elements import (
    ElementLoadView,
    SurfaceConstraintView,
    ViewerElementGroup,
    ViewerElements,
    ViewerElementType,
    viewer_elements_from_fem,
)
from ._nodes import (
    MassView,
    NodalLoadView,
    NodeConstraintView,
    SPView,
    ViewerNodes,
    _NamedNodeSelection,
    viewer_nodes_from_fem,
)
from ._records import (
    ConstraintRow,
    ElementLoadRow,
    InterpolationRow,
    MassRow,
    NodalLoadRow,
    SPRow,
    SurfaceCouplingRow,
    decode_constraint_row,
    decode_element_load_row,
    decode_mass_row,
    decode_nodal_load_row,
    decode_sp_row,
)
from ._elements import _SurfaceConstraintRow

# ---------------------------------------------------------------------------
# GMSH alias → element-type fallback (for /elements/{alias} attrs)
# ---------------------------------------------------------------------------
#
# The /elements/{alias}/ writer (mesh/_femdata_h5_io.py::_write_elements)
# already stores ``code``, ``gmsh_name``, ``npe``, ``dim``, ``order``
# attrs.  Reader-side we just read them back.  No fallback table needed.


class ViewerData:
    """Read-only structural snapshot consumed by the viewer package.

    Construct via :meth:`from_fem` or :meth:`from_h5`; do not
    instantiate directly.

    Attributes
    ----------
    snapshot_id : str
        Content hash identifying the FEMData this snapshot was
        captured from.  Stable across the two builders (h5 round-trip
        of a FEMData produces the same id, as long as the broker's
        :func:`apeGmsh.mesh._femdata_hash.compute_snapshot_id` is
        deterministic).  May be empty when neither source could
        compute one (rare).
    source_kind : Literal["fem", "h5"]
        Which builder produced this instance.  Mostly informational —
        downstream code should treat both sources interchangeably.
    nodes : :class:`ViewerNodes`
    elements : :class:`ViewerElements`
    """

    __slots__ = ("_snapshot_id", "_source_kind", "nodes", "elements")

    def __init__(
        self,
        *,
        snapshot_id: str,
        source_kind: Literal["fem", "h5"],
        nodes: ViewerNodes,
        elements: ViewerElements,
    ) -> None:
        self._snapshot_id = str(snapshot_id)
        self._source_kind = source_kind
        self.nodes = nodes
        self.elements = elements

    @property
    def snapshot_id(self) -> str:
        return self._snapshot_id

    @property
    def source_kind(self) -> Literal["fem", "h5"]:
        return self._source_kind

    # ------------------------------------------------------------------
    # Builders
    # ------------------------------------------------------------------

    @classmethod
    def from_fem(cls, fem: Any) -> "ViewerData":
        """Wrap a live :class:`apeGmsh.mesh.FEMData` for viewer consumption.

        Eagerly snapshots every named index and every record set so the
        resulting :class:`ViewerData` is fully decoupled from mutation
        of the FEMData.
        """
        try:
            snapshot_id = str(fem.snapshot_id)
        except Exception:
            snapshot_id = ""
        return cls(
            snapshot_id=snapshot_id,
            source_kind="fem",
            nodes=viewer_nodes_from_fem(fem),
            elements=viewer_elements_from_fem(fem),
        )

    @classmethod
    def from_h5(cls, path: str) -> "ViewerData":
        """Open ``path`` via the reference h5_reader and build the snapshot.

        Performs the reader's schema version check on open.  The file
        is closed before this method returns — :class:`ViewerData` holds
        only decoded arrays / row tuples, no live h5py handle.
        """
        from apeGmsh.opensees.emitter import h5_reader
        with h5_reader.open(path) as model:
            return cls.from_h5_model(model)

    @classmethod
    def from_h5_model(cls, model: Any) -> "ViewerData":
        """Build a :class:`ViewerData` from an already-open
        :class:`apeGmsh.opensees.emitter.h5_reader.H5Model`.

        For callers that already opened the file (e.g. to read
        ``/opensees/`` enrichment alongside).  Does not close the model.
        """
        meta = model.meta()
        snapshot_id = str(meta.get("snapshot_id", "") or "")
        nodes_dict = model.nodes()
        if not nodes_dict:
            ids = np.array([], dtype=np.int64)
            coords = np.zeros((0, 3), dtype=np.float64)
        else:
            ids = np.asarray(nodes_dict["ids"], dtype=np.int64)
            coords = np.asarray(nodes_dict["coords"], dtype=np.float64)

        # Named indexes — physical_groups, labels, mesh_selections all
        # share the same {name → {dim, tag, name, node_ids,
        # node_coords, element_ids?}} shape (Phase 8.5 / 8.7).
        node_physical = _named_view_from_h5_index(
            model.physical_groups(), side="node",
            raise_on_missing=True, label="physical group",
        )
        node_labels = _named_view_from_h5_index(
            model.labels(), side="node",
            raise_on_missing=True, label="label",
        )
        node_selection = _named_view_from_h5_index(
            model.mesh_selections(), side="node",
            raise_on_missing=False, label="selection",
        )
        elem_physical = _named_view_from_h5_index(
            model.physical_groups(), side="element",
            raise_on_missing=True, label="physical group",
        )
        elem_labels = _named_view_from_h5_index(
            model.labels(), side="element",
            raise_on_missing=True, label="label",
        )
        elem_selection = _named_view_from_h5_index(
            model.mesh_selections(), side="element",
            raise_on_missing=False, label="selection",
        )

        # Record sets (loads / masses / constraints).
        nodal_loads, sp_rows = _decode_loads(model.loads())
        masses_rows = _decode_masses(model.masses())
        node_cs_rows, elem_cs_rows = _decode_constraints(model.constraints())

        nodes = ViewerNodes(
            ids=ids, coords=coords,
            physical=node_physical, labels=node_labels,
            selection=node_selection,
            loads=NodalLoadView(nodal_loads),
            sp=SPView(sp_rows),
            masses=MassView(masses_rows),
            constraints=NodeConstraintView(node_cs_rows),
        )

        # Per-element geomTransf vecxz — the schema-aware transforms ↔
        # element_meta ↔ vocabulary join lives in h5_reader (the one
        # apeGmsh.opensees module viewers may import). Absent for
        # mesh-only / pre-bridge archives → empty dict, no error.
        try:
            vecxz = model.element_local_axes_vecxz()
        except Exception:
            vecxz = {}

        elements = ViewerElements(
            groups=_decode_element_groups(model),
            physical=elem_physical, labels=elem_labels,
            selection=elem_selection,
            loads=ElementLoadView(_decode_element_loads(model.loads())),
            constraints=SurfaceConstraintView(elem_cs_rows),
            vecxz=vecxz,
        )

        return cls(
            snapshot_id=snapshot_id,
            source_kind="h5",
            nodes=nodes,
            elements=elements,
        )


# =====================================================================
# H5-side decoders
# =====================================================================


def _named_view_from_h5_index(
    index: dict[str, dict[str, Any]],
    *,
    side: str,
    raise_on_missing: bool,
    label: str,
) -> _NamedNodeSelection:
    by_name: dict[str, ndarray] = {}
    for name, entry in index.items():
        if side == "node":
            arr = entry.get("node_ids")
            if arr is None:
                continue
            by_name[name] = np.asarray(arr, dtype=np.int64)
        else:
            arr = entry.get("element_ids")
            if arr is None:
                continue
            by_name[name] = np.asarray(arr, dtype=np.int64)
    return _NamedNodeSelection(
        by_name, raise_on_missing=raise_on_missing, label=label,
    )


def _decode_element_groups(model: Any) -> list[ViewerElementGroup]:
    out: list[ViewerElementGroup] = []
    for alias, attrs in model.elements().items():
        arrays = model.element_arrays(alias)
        ids = np.asarray(arrays["ids"], dtype=np.int64)
        if ids.size == 0:
            continue
        connectivity = np.asarray(
            arrays.get("connectivity", np.zeros((ids.size, 0), dtype=np.int64)),
            dtype=np.int64,
        )
        et = ViewerElementType(
            code=int(attrs.get("code", 0)),
            name=str(alias),
            gmsh_name=str(attrs.get("gmsh_name", "")),
            dim=int(attrs.get("dim", 0)),
            npe=int(attrs.get("npe", connectivity.shape[1] if connectivity.ndim == 2 else 0)),
            order=int(attrs.get("order", 1)),
        )
        out.append(ViewerElementGroup(
            element_type=et, ids=ids, connectivity=connectivity,
        ))
    return out


def _decode_loads(
    loads: dict[str, dict[str, Any]],
) -> tuple[list[NodalLoadRow], list[SPRow]]:
    """Decode ``/loads`` into (nodal_load_rows, sp_rows).

    Element-side loads are split out separately via
    :func:`_decode_element_loads` (kept on a parallel path for clarity
    — the broker's writer / reader treats them as siblings under
    ``/loads`` but the viewer routes them to different composites).
    """
    nodal: list[NodalLoadRow] = []
    sp: list[SPRow] = []
    for pattern, rows in (loads.get("nodal") or {}).items():
        for row in rows:
            nodal.append(decode_nodal_load_row(row, pattern))
    for pattern, rows in (loads.get("sp") or {}).items():
        for row in rows:
            sp.append(decode_sp_row(row, pattern))
    return nodal, sp


def _decode_element_loads(
    loads: dict[str, dict[str, Any]],
) -> list[ElementLoadRow]:
    out: list[ElementLoadRow] = []
    for pattern, rows in (loads.get("element") or {}).items():
        for row in rows:
            out.append(decode_element_load_row(row, pattern))
    return out


def _decode_masses(masses: Any) -> list[MassRow]:
    if masses is None:
        return []
    return [decode_mass_row(row) for row in masses]


def _decode_constraints(
    constraints: dict[str, Any],
) -> tuple[list[ConstraintRow], list[_SurfaceConstraintRow]]:
    """Split ``/constraints/{kind}`` rows into (node-side, element-side).

    Routing follows the record's *kind* (which determines its record
    class on the writer side) rather than the symmetric compound's
    ``target_kind`` field — the latter describes the *target column*
    (``"node"`` vs ``"element"`` tag), not which FEMData composite
    the record originated from.  Mirrors FEMData's distribution:

    - NodePair / NodeGroup / NodeToSurface → ``fem.nodes.constraints``
    - Interpolation / SurfaceCoupling      → ``fem.elements.constraints``
    """
    from ._records import (
        _INTERPOLATION_KINDS,
        _NODE_GROUP_KINDS,
        _NODE_PAIR_KINDS,
        _NODE_TO_SURFACE_KINDS,
        _SURFACE_COUPLING_KINDS,
    )
    _NODE_SIDE = _NODE_PAIR_KINDS | _NODE_GROUP_KINDS | _NODE_TO_SURFACE_KINDS
    _ELEM_SIDE = _INTERPOLATION_KINDS | _SURFACE_COUPLING_KINDS
    node_rows: list[ConstraintRow] = []
    elem_rows: list[_SurfaceConstraintRow] = []
    for kind, arr in constraints.items():
        for row in arr:
            decoded = decode_constraint_row(row)
            if kind in _NODE_SIDE:
                node_rows.append(decoded)
            elif kind in _ELEM_SIDE:
                assert isinstance(decoded, (InterpolationRow, SurfaceCouplingRow))
                elem_rows.append(decoded)
            else:
                # Forward-compat: an unrecognised kind would have
                # already raised ViewerDataDecodeError inside
                # decode_constraint_row; we'll never reach this branch
                # unless that path softens.
                continue
    return node_rows, elem_rows


__all__ = ["ViewerData"]
