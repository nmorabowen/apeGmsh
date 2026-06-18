"""ViewerElements — element-side composite of :class:`ViewerData`.

Mirrors the FEMData element-side accessors the viewer exercises:
``__iter__`` over typed groups, :attr:`types` listing, the named
``physical`` / ``labels`` / ``selection`` views, and surface-side
constraint records.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Union

import numpy as np
from numpy import ndarray

from ._nodes import (
    _NamedNodeSelection,
    _named_selection_from_groupset,
    _named_selection_from_store,
)
from ._records import (
    ElementLoadRow,
    InterpolationRow,
    SurfaceCouplingRow,
    constraint_row_from_record,
    element_load_row_from_record,
)


@dataclass(frozen=True)
class ViewerElementType:
    """Lightweight type-info view consumed by ``fem_scene.py`` and
    diagrams.  Mirrors only the attributes the viewer reads off
    :class:`apeGmsh.mesh._element_types.ElementTypeInfo`.
    """
    code: int
    name: str
    gmsh_name: str
    dim: int
    npe: int
    order: int


class ViewerElementGroup:
    """One homogeneous element block — matches the viewer's read of
    :class:`apeGmsh.mesh._element_types.ElementGroup`.

    Attributes
    ----------
    element_type : :class:`ViewerElementType`
        Type metadata (``code``, ``dim``, ``npe`` are the load-bearing
        fields for ``viewers/scene/fem_scene.py``).
    ids : ndarray(N,) int64
    connectivity : ndarray(N, npe) int64
    """

    __slots__ = ("element_type", "ids", "connectivity")

    def __init__(
        self,
        *,
        element_type: ViewerElementType,
        ids: ndarray,
        connectivity: ndarray,
    ) -> None:
        self.element_type = element_type
        self.ids = np.asarray(ids, dtype=np.int64)
        self.connectivity = np.asarray(connectivity, dtype=np.int64)

    @property
    def type_name(self) -> str:
        return self.element_type.name

    @property
    def type_code(self) -> int:
        return self.element_type.code

    @property
    def dim(self) -> int:
        return self.element_type.dim

    @property
    def npe(self) -> int:
        return self.element_type.npe

    def __len__(self) -> int:
        return int(self.ids.size)

    def __iter__(self) -> Iterator[tuple[int, tuple[int, ...]]]:
        for i in range(len(self.ids)):
            yield int(self.ids[i]), tuple(int(n) for n in self.connectivity[i])


class ElementLoadView:
    """Iterable view over element-side load rows."""

    __slots__ = ("_rows",)

    def __init__(self, rows: list[ElementLoadRow]) -> None:
        self._rows = rows

    def __iter__(self) -> Iterator[ElementLoadRow]:
        return iter(self._rows)

    def __len__(self) -> int:
        return len(self._rows)

    def __bool__(self) -> bool:
        return bool(self._rows)

    def patterns(self) -> list[str]:
        seen: list[str] = []
        for r in self._rows:
            if r.pattern not in seen:
                seen.append(r.pattern)
        return seen

    def by_pattern(self, name: str) -> list[ElementLoadRow]:
        return [r for r in self._rows if r.pattern == name]


_SurfaceConstraintRow = Union[InterpolationRow, SurfaceCouplingRow]


class SurfaceConstraintView:
    """Iterable view over element-side (surface) constraint rows.

    Mirrors :class:`apeGmsh.mesh._record_set.SurfaceConstraintSet`:
    ``__iter__`` over every row plus typed iterators
    :meth:`interpolations` and :meth:`couplings`.
    """

    __slots__ = ("_rows",)

    def __init__(self, rows: list[_SurfaceConstraintRow]) -> None:
        self._rows = rows

    def __iter__(self) -> Iterator[_SurfaceConstraintRow]:
        return iter(self._rows)

    def __len__(self) -> int:
        return len(self._rows)

    def __bool__(self) -> bool:
        return bool(self._rows)

    def interpolations(self) -> Iterator[InterpolationRow]:
        """Yield every InterpolationRow + the slave_records inside
        every SurfaceCouplingRow (mirrors the live method's behavior)."""
        for row in self._rows:
            if isinstance(row, InterpolationRow):
                yield row
            elif isinstance(row, SurfaceCouplingRow):
                yield from row.slave_records

    def couplings(self) -> Iterator[SurfaceCouplingRow]:
        for row in self._rows:
            if isinstance(row, SurfaceCouplingRow):
                yield row


class ViewerElements:
    """Read-only element-side composite consumed by the viewer."""

    __slots__ = (
        "_groups",
        "_types",
        "_vecxz_by_eid",
        "_partition_by_eid",
        "_module_by_eid",
        "physical",
        "labels",
        "selection",
        "loads",
        "constraints",
    )

    def __init__(
        self,
        *,
        groups: list[ViewerElementGroup],
        physical: _NamedNodeSelection,
        labels: _NamedNodeSelection,
        selection: _NamedNodeSelection,
        loads: ElementLoadView,
        constraints: SurfaceConstraintView,
        vecxz: dict[int, ndarray] | None = None,
        partition_by_eid: dict[int, int] | None = None,
        module_by_eid: dict[int, str] | None = None,
    ) -> None:
        self._groups = list(groups)
        self._types = [g.element_type for g in self._groups]
        # FEM element id -> geomTransf vecxz (3,). Populated only on the
        # h5 path (OpenSees enrichment); empty for from_fem (FEMData is
        # the solver-agnostic neutral zone and has no geomTransf).
        self._vecxz_by_eid: dict[int, ndarray] = {
            int(k): np.asarray(v, dtype=np.float64).reshape(3)
            for k, v in (vecxz or {}).items()
        }
        # FEM element id -> OpenSeesMP rank (schema 2.10.0 / ADR 0027).
        # Joined from /opensees/element_meta/{token}/{fem_eids,
        # partition_ids}. Empty for single-partition models, pre-2.10.0
        # files, and the from_fem path (FEMData has no rank labelling).
        self._partition_by_eid: dict[int, int] = {
            int(k): int(v) for k, v in (partition_by_eid or {}).items()
        }
        # FEM element id -> compose-module label (schema 2.9.0 /
        # ADR 0038).  Populated from the broker's per-element
        # ``_module_label`` dict (from_fem path) or from each
        # ``/elements/{type}/module_label`` parallel dataset (h5 path
        # via :meth:`H5Model.bulk_module_labels_for_elements`).
        # Host-owned rows carry the empty-string label on the wire and
        # are excluded from this mapping, so ``module_for(host_eid)``
        # returns ``None``.  Empty for uncomposed FEMData, pre-2.9.0
        # archives, and any source that has no module-label metadata
        # at all.  For nested-compose models the label is the full
        # joined label produced by
        # :func:`apeGmsh.mesh._compose._join_module_label` (e.g.
        # ``"bayP/frameA"``).
        self._module_by_eid: dict[int, str] = {
            int(k): str(v) for k, v in (module_by_eid or {}).items()
            if str(v) != ""
        }
        self.physical = physical
        self.labels = labels
        self.selection = selection
        self.loads = loads
        self.constraints = constraints

    def vecxz_for(self, element_id: int) -> "ndarray | None":
        """Return the geomTransf ``vecxz`` (3,) for a beam element, or
        ``None`` when unknown (non-beam, or no OpenSees enrichment in
        the source — e.g. the live ``from_fem`` path)."""
        return self._vecxz_by_eid.get(int(element_id))

    @property
    def has_vecxz(self) -> bool:
        """True when per-element orientation is available (h5 path with
        a ``/opensees/transforms`` zone)."""
        return bool(self._vecxz_by_eid)

    def partition_for(self, element_id: int) -> "int | None":
        """Return the OpenSeesMP rank for a FEM element id, or ``None``
        when the element has no rank labelling (single-partition model,
        pre-2.10.0 archive, or the live ``from_fem`` path)."""
        return self._partition_by_eid.get(int(element_id))

    @property
    def has_partitions(self) -> bool:
        """True when at least one element carries an OpenSeesMP rank
        label (schema 2.10.0 partition zone present and non-trivial)."""
        return bool(self._partition_by_eid)

    def module_for(self, element_id: int) -> "str | None":
        """Return the compose-module label that owns a FEM element id, or
        ``None`` when the element is host-owned (no compose origin) or
        the source carries no module-label metadata at all (uncomposed
        FEMData, pre-2.9.0 archive).

        For nested-compose models the label is the full joined label
        (e.g. ``"bayP/frameA"``) — host-rows always read as ``None``."""
        return self._module_by_eid.get(int(element_id))

    @property
    def has_modules(self) -> bool:
        """True when at least one element carries a compose-module label
        (schema 2.9.0 / ADR 0038 — composed FEMData or composed
        ``model.h5``)."""
        return bool(self._module_by_eid)

    def __iter__(self) -> Iterator[ViewerElementGroup]:
        return iter(self._groups)

    def __len__(self) -> int:
        return sum(len(g) for g in self._groups)

    def __bool__(self) -> bool:
        return len(self) > 0

    @property
    def types(self) -> list[ViewerElementType]:
        return list(self._types)


# =====================================================================
# Builders — from FEMData
# =====================================================================


def viewer_elements_from_fem(fem: Any) -> ViewerElements:
    groups: list[ViewerElementGroup] = []
    for g in fem.elements:
        if g.ids.size == 0:
            continue
        et = g.element_type
        groups.append(ViewerElementGroup(
            element_type=ViewerElementType(
                code=int(et.code),
                name=str(et.name),
                gmsh_name=str(et.gmsh_name),
                dim=int(et.dim),
                npe=int(et.npe),
                order=int(et.order),
            ),
            ids=np.asarray(g.ids, dtype=np.int64),
            connectivity=np.asarray(g.connectivity, dtype=np.int64),
        ))

    physical = _named_selection_from_groupset(
        fem.elements.physical, label="physical group", side="element",
        raise_on_missing=True,
    )
    labels = _named_selection_from_groupset(
        fem.elements.labels, label="label", side="element",
        raise_on_missing=True,
    )
    selection = _named_selection_from_store(
        getattr(fem, "mesh_selection", None), side="element",
    )

    loads = ElementLoadView([
        element_load_row_from_record(r) for r in (fem.elements.loads or [])
    ])

    cs_rows: list[_SurfaceConstraintRow] = []
    if fem.elements.constraints is not None:
        for rec in fem.elements.constraints:
            row = constraint_row_from_record(rec)
            # The writer-side classifier only ever puts Interpolation /
            # SurfaceCoupling rows onto fem.elements.constraints; assert
            # for mypy and to catch writer-side drift early.
            assert isinstance(row, (InterpolationRow, SurfaceCouplingRow))
            cs_rows.append(row)
    constraints = SurfaceConstraintView(cs_rows)

    # Per-element compose-module label (schema 2.9.0 / ADR 0038).
    # ``_module_label`` on the broker is ``dict[type_code, ndarray(N,)
    # object dtype]`` aligned 1:1 with each ``ElementGroup.ids``.  Host
    # rows are empty strings and are filtered out by the
    # ``ViewerElements`` ctor.  Falls back to an empty mapping for
    # uncomposed FEMData (``_module_label is None``) — ``has_modules``
    # will then be ``False``.
    module_by_eid: dict[int, str] = {}
    ml_dict = getattr(fem.elements, "_module_label", None)
    if ml_dict:
        for g in fem.elements:
            arr = ml_dict.get(int(g.element_type.code))
            if arr is None:
                continue
            ids = np.asarray(g.ids, dtype=np.int64)
            for i in range(min(len(ids), len(arr))):
                lbl = arr[i]
                if lbl is None:
                    continue
                s = str(lbl)
                if s == "":
                    continue
                module_by_eid[int(ids[i])] = s

    # Per-element partition rank (schema 2.10.0 / ADR 0027).
    # ElementComposite._partitions is dict[rank, {'element_ids': ndarray}]
    # populated by extract_partitions() in the live FEM path.
    partition_by_eid: dict[int, int] = {}
    elem_parts: dict = getattr(fem.elements, "_partitions", {}) or {}
    for rank, pdata in elem_parts.items():
        eids = pdata.get("element_ids")
        if eids is None:
            continue
        for eid in np.asarray(eids, dtype=np.int64).flat:
            partition_by_eid[int(eid)] = int(rank)

    return ViewerElements(
        groups=groups,
        physical=physical, labels=labels, selection=selection,
        loads=loads, constraints=constraints,
        partition_by_eid=partition_by_eid or None,
        module_by_eid=module_by_eid or None,
    )
