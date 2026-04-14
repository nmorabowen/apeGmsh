"""
_element_types — Per-type element storage for the FEM broker.
==============================================================

Provides three classes that replace the old flat ``ndarray(N, npe)``
connectivity storage:

    ElementTypeInfo   — metadata for one Gmsh element type
    ElementGroup      — one homogeneous block (single type, rectangular conn)
    GroupResult       — iterable collection returned by ``.get()``

Also provides the alias system that maps Gmsh type codes to short names
(``'tet4'``, ``'hex8'``, etc.) and a filter resolution helper.

This module is a **leaf** — zero dependencies on the rest of apeGmsh.
"""
from __future__ import annotations

import numpy as np
from numpy import ndarray


# =====================================================================
# Curated alias table
# =====================================================================

_KNOWN_ALIASES: dict[int, str] = {
    1:  'line2',      2:  'tri3',       3:  'quad4',
    4:  'tet4',       5:  'hex8',       6:  'prism6',     7:  'pyramid5',
    8:  'line3',      9:  'tri6',      10:  'quad9',
    11: 'tet10',     12:  'hex27',     15:  'point1',
    16: 'quad8',     17:  'hex20',     18:  'prism15',   19:  'pyramid13',
    20: 'tri9',      21:  'tri10',     26:  'line4',
    29: 'tet20',     36:  'quad16',    92:  'hex64',
}

_SHAPE_PREFIXES: dict[str, str] = {
    'Point':          'point',
    'Line':           'line',
    'Triangle':       'tri',
    'Quadrilateral':  'quad',
    'Tetrahedron':    'tet',
    'Hexahedron':     'hex',
    'Prism':          'prism',
    'Pyramid':        'pyramid',
}


def _auto_alias(gmsh_name: str, npe: int) -> str:
    """Generate a short alias from a Gmsh element name.

    Examples: ``'Tetrahedron 4'`` → ``'tet4'``,
    ``'Hexahedron 64'`` → ``'hex64'``.
    """
    for key, prefix in _SHAPE_PREFIXES.items():
        if key in gmsh_name:
            return f"{prefix}{npe}"
    # Ultimate fallback: clean the gmsh name
    return gmsh_name.lower().replace(' ', '')


def _alias_for(code: int, gmsh_name: str, npe: int) -> str:
    """Resolve the short alias for a Gmsh type code."""
    if code in _KNOWN_ALIASES:
        return _KNOWN_ALIASES[code]
    return _auto_alias(gmsh_name, npe)


# =====================================================================
# ElementTypeInfo
# =====================================================================

class ElementTypeInfo:
    """Metadata for one Gmsh element type.

    Attributes
    ----------
    code : int
        Gmsh element type code (4 = tet4, 5 = hex8, ...).
        This is the primary key — always unique, always works.
    name : str
        Short alias (``'tet4'``, ``'hex8'``).  Curated for common
        types, auto-generated for exotic ones.
    gmsh_name : str
        Gmsh's own name (``'Tetrahedron 4'``, ``'Hexahedron 8'``).
    dim : int
        Topological dimension (0–3).
    order : int
        Polynomial order (1 = linear, 2 = quadratic, ...).
    npe : int
        Nodes per element.
    count : int
        Number of elements of this type in the mesh.
    """

    __slots__ = ('code', 'name', 'gmsh_name', 'dim', 'order', 'npe', 'count')

    def __init__(
        self,
        code: int,
        name: str,
        gmsh_name: str,
        dim: int,
        order: int,
        npe: int,
        count: int = 0,
    ) -> None:
        self.code = code
        self.name = name
        self.gmsh_name = gmsh_name
        self.dim = dim
        self.order = order
        self.npe = npe
        self.count = count

    def __repr__(self) -> str:
        return (
            f"ElementTypeInfo({self.name!r}, code={self.code}, "
            f"dim={self.dim}, order={self.order}, "
            f"npe={self.npe}, count={self.count})"
        )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ElementTypeInfo):
            return self.code == other.code
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.code)


def make_type_info(
    code: int,
    gmsh_name: str,
    dim: int,
    order: int,
    npe: int,
    count: int = 0,
) -> ElementTypeInfo:
    """Create an ``ElementTypeInfo`` with auto-resolved alias."""
    return ElementTypeInfo(
        code=code,
        name=_alias_for(code, gmsh_name, npe),
        gmsh_name=gmsh_name,
        dim=dim,
        order=order,
        npe=npe,
        count=count,
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
        element_type: ElementTypeInfo,
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
    def types(self) -> list[ElementTypeInfo]:
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
