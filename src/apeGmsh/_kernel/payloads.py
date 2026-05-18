"""apeGmsh._kernel.payloads — dependency-free numpy result payloads.

THE KEYSTONE (selection-unification-v2 P1-K).  These four symbols are a
closed, coupled triad relocated **verbatim** (class identity unchanged
— only the module path moved) so that the lightweight numpy payloads
sit in the root-leaf ``_kernel`` layer instead of straddling ``mesh``:

* :class:`NodeResult`   — pair-iterating ``(node_id, xyz)`` view
  (was ``apeGmsh.mesh.FEMData.NodeResult``).
* :class:`ElementGroup` — one homogeneous element block
  (was ``apeGmsh.mesh._element_types.ElementGroup``).
* :class:`GroupResult`  — iterable collection of ``ElementGroup``
  (was ``apeGmsh.mesh._element_types.GroupResult``).
* :func:`resolve_type_filter` — type-filter helper that
  ``GroupResult.get`` calls and that reads ``ElementGroup`` attributes
  (was ``apeGmsh.mesh._element_types.resolve_type_filter``).

The old ``apeGmsh.mesh.FEMData`` / ``apeGmsh.mesh._element_types`` /
``apeGmsh.mesh`` paths keep these names importable via thin downward
re-exports (Option-i — keeps the byte-unchanged contract/pin tests
working).  Relocating this triad closes HT1 (``_record_set`` straddle),
HT8 (iteration contract) and R3-B (element payload) simultaneously.

Pure: stdlib ``typing`` + numpy only (``NodeResult.to_dataframe`` has a
deferred ``import pandas``).  Zero ``apeGmsh.*`` imports.
``ElementTypeInfo`` / ``make_type_info`` / the alias machinery stay in
:mod:`apeGmsh.mesh._element_types` (they never call this trio, so no
back-edge).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy import ndarray

if TYPE_CHECKING:
    import pandas as pd

# NOTE: ``ElementTypeInfo`` lives in ``apeGmsh.mesh._element_types`` and
# is referenced ONLY in string annotations below (PEP 563 — never
# evaluated at runtime).  It is deliberately NOT imported here: an
# import of ``apeGmsh.mesh`` from this root-leaf module — even under
# ``TYPE_CHECKING`` — would re-form the cycle P1-K deletes.


# =====================================================================
# NodeResult — pair-iterating node view
# =====================================================================

class NodeResult:
    """Iterable pair-view of node IDs and coordinates.

    Iteration yields ``(node_id, xyz)`` pairs — clean one-liner for
    solver node emission::

        for nid, xyz in fem.nodes.get(pg="Base"):
            ops.node(nid, *xyz)

    Array access is still available::

        result = fem.nodes.get(pg="Base")
        result.ids       # ndarray(N,) object dtype — iterates as Python int
        result.coords    # ndarray(N, 3) float64
        result.to_dataframe()

    The ID array uses ``dtype=object`` so iterating yields plain
    Python ``int`` values, which OpenSees and other C-extension
    solvers accept without ``int()`` casts.
    """

    __slots__ = ('_ids', '_coords')

    def __init__(self, ids: ndarray, coords: ndarray) -> None:
        ids_arr = np.asarray(ids)
        if ids_arr.dtype != object:
            ids_arr = ids_arr.astype(object)
        self._ids = ids_arr
        self._coords = np.asarray(coords, dtype=np.float64)

    @property
    def ids(self) -> ndarray:
        return self._ids

    @property
    def coords(self) -> ndarray:
        return self._coords

    def __iter__(self):
        for nid, xyz in zip(self._ids, self._coords):
            yield nid, xyz

    def __len__(self) -> int:
        return len(self._ids)

    def __bool__(self) -> bool:
        return len(self._ids) > 0

    def __repr__(self) -> str:
        return f"NodeResult({len(self)} nodes)"

    def to_dataframe(self) -> "pd.DataFrame":
        import pandas as pd
        return pd.DataFrame(
            self._coords,
            index=pd.Index(
                [int(x) for x in self._ids], name='node_id'),
            columns=['x', 'y', 'z'],
        )


# =====================================================================
# ElementGroup
# =====================================================================

class ElementGroup:
    """One homogeneous block — single element type, rectangular connectivity.

    This is the atomic unit of element storage.  Every element in the
    group has the same type, same number of nodes per element, and the
    connectivity is a rectangular ``ndarray(N, npe)``.

    Iterable: ``for eid, conn_row in group:`` yields ``(int, ndarray)``
    pairs for solver loops.

    Attributes
    ----------
    element_type : ElementTypeInfo
        Type metadata.
    ids : ndarray
        Element IDs, shape ``(N,)``.
    connectivity : ndarray
        Node connectivity, shape ``(N, npe)``.
    """

    __slots__ = ('element_type', 'ids', 'connectivity')

    def __init__(
        self,
        element_type: "ElementTypeInfo",
        ids: ndarray,
        connectivity: ndarray,
    ) -> None:
        self.element_type = element_type
        self.ids = np.asarray(ids, dtype=np.int64)
        self.connectivity = np.asarray(connectivity, dtype=np.int64)

    # ── Convenience shortcuts ───────────────────────────────

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

    # ── Iteration / sizing ──────────────────────────────────

    def __len__(self) -> int:
        return len(self.ids)

    def __iter__(self):
        """Yield ``(eid, conn_row)`` pairs for solver loops.

        Both ``eid`` and each node in ``conn_row`` are plain Python
        ``int`` — safe for OpenSees and other C-backed APIs that
        reject ``numpy.int64``.
        """
        for i in range(len(self.ids)):
            yield int(self.ids[i]), tuple(int(n) for n in self.connectivity[i])

    def __repr__(self) -> str:
        return (
            f"ElementGroup({self.type_name!r}, "
            f"n={len(self)}, npe={self.npe})"
        )


# =====================================================================
# GroupResult
# =====================================================================

class GroupResult:
    """Iterable collection of ``ElementGroup`` objects.

    Returned by ``ElementComposite.get()`` and chainable via
    ``.get()`` for further filtering.

    Usage
    -----
    ::

        # Iterate groups
        for group in result:
            for eid, conn_row in group:
                ops.element(etype, eid, *conn_row, mat)

        # Flat access (single-type mesh)
        ids, conn = result.resolve()

        # Flat access (pick one type)
        ids, conn = result.resolve(element_type='tet4')
    """

    __slots__ = ('_groups',)

    def __init__(self, groups: list[ElementGroup]) -> None:
        self._groups = list(groups)

    # ── Iteration ───────────────────────────────────────────

    def __iter__(self):
        return iter(self._groups)

    def __len__(self) -> int:
        return len(self._groups)

    def __bool__(self) -> bool:
        return len(self._groups) > 0

    # ── Aggregate properties ────────────────────────────────

    @property
    def ids(self) -> ndarray:
        """All element IDs concatenated across groups."""
        if not self._groups:
            return np.array([], dtype=np.int64)
        return np.concatenate([g.ids for g in self._groups])

    @property
    def n_elements(self) -> int:
        """Total element count across all groups."""
        return sum(len(g) for g in self._groups)

    @property
    def types(self) -> "list[ElementTypeInfo]":
        """Unique element types present."""
        return [g.element_type for g in self._groups]

    @property
    def is_homogeneous(self) -> bool:
        """True if all elements are the same type."""
        return len(self._groups) <= 1

    @property
    def connectivity(self) -> ndarray:
        """Connectivity array — only if homogeneous.

        Raises
        ------
        TypeError
            If multiple element types are present.
        """
        if not self._groups:
            return np.empty((0, 0), dtype=np.int64)
        if not self.is_homogeneous:
            names = [g.type_name for g in self._groups]
            raise TypeError(
                f"Cannot return flat connectivity: {len(self._groups)} "
                f"element types present ({', '.join(names)}). "
                f"Use .resolve(element_type='...') to pick one, "
                f"or iterate groups with: for group in result: ..."
            )
        return self._groups[0].connectivity

    # ── Chainable filter ────────────────────────────────────

    def get(
        self,
        *,
        dim: int | None = None,
        element_type: str | int | None = None,
    ) -> "GroupResult":
        """Re-filter this result (AND intersection).

        Parameters
        ----------
        dim : int, optional
            Keep only groups at this dimension.
        element_type : str or int, optional
            Keep only groups matching this type (alias, code, or Gmsh name).
        """
        filtered = self._groups

        if dim is not None:
            filtered = [g for g in filtered if g.dim == dim]

        if element_type is not None:
            codes = resolve_type_filter(element_type, self._groups)
            filtered = [g for g in filtered if g.type_code in codes]

        return GroupResult(filtered)

    # ── Resolve to flat arrays ──────────────────────────────

    def resolve(
        self,
        element_type: str | int | None = None,
    ) -> tuple[ndarray, ndarray]:
        """Flatten to ``(ids, connectivity)`` arrays.

        Parameters
        ----------
        element_type : str or int, optional
            If given, filter to this type first.
            If not given, must be homogeneous (single type).

        Returns
        -------
        (ndarray, ndarray)
            ``(ids, connectivity)`` — shape ``(N,)`` and ``(N, npe)``.

        Raises
        ------
        TypeError
            If multiple types present and *element_type* not specified.
        """
        target = self
        if element_type is not None:
            target = self.get(element_type=element_type)

        if not target._groups:
            return np.array([], dtype=np.int64), np.empty((0, 0), dtype=np.int64)

        if not target.is_homogeneous:
            names = [g.type_name for g in target._groups]
            raise TypeError(
                f"Cannot resolve: {len(target._groups)} element types "
                f"present ({', '.join(names)}). "
                f"Use .resolve(element_type='...') to pick one, "
                f"or iterate: for group in result: ..."
            )

        group = target._groups[0]
        return group.ids, group.connectivity

    # ── Display ─────────────────────────────────────────────

    def __repr__(self) -> str:
        if not self._groups:
            return "GroupResult(empty)"
        parts = [f"{g.type_name}:{len(g)}" for g in self._groups]
        return f"GroupResult({', '.join(parts)})"


# =====================================================================
# Type filter resolution
# =====================================================================

def resolve_type_filter(
    type_key: str | int,
    groups: list[ElementGroup],
) -> set[int]:
    """Resolve a type filter to a set of Gmsh type codes.

    Parameters
    ----------
    type_key : str or int
        Alias name (``'tet4'``), Gmsh code (``4``),
        or Gmsh name (``'Tetrahedron 4'``).
    groups : list[ElementGroup]
        Available groups to search.

    Returns
    -------
    set[int]
        Matching type codes.

    Raises
    ------
    KeyError
        If no match found.
    """
    if isinstance(type_key, int):
        return {type_key}

    # Try alias match
    for g in groups:
        if g.type_name == type_key:
            return {g.type_code}

    # Try Gmsh name match
    for g in groups:
        if g.element_type.gmsh_name == type_key:
            return {g.type_code}

    available = [g.type_name for g in groups]
    raise KeyError(
        f"Unknown element type {type_key!r}. "
        f"Available: {available}"
    )
