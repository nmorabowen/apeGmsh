"""
_element_types — Per-type element storage for the FEM broker.
==============================================================

Provides three classes that replace the old flat ``ndarray(N, npe)``
connectivity storage:

    ElementTypeInfo   — metadata for one Gmsh element type
    ElementGroup      — one homogeneous block (single type, rectangular conn)
    GroupResult       — iterable collection returned by ``.get()``

Also provides the alias system that maps Gmsh type codes to short names
(``'tet4'``, ``'hex8'``, etc.), a filter resolution helper, and the
element-topology table (``TOPOLOGY_BY_CODE`` / :func:`element_topology`)
that every ``gmsh.model.mesh.getElements`` walk consults.

This module is a **leaf** — no module-level dependencies on the rest of
apeGmsh (``element_topology(..., integrable=True)`` defers one import
of the leaf-pure ``apeGmsh.fem`` shape-function catalog).
"""
from __future__ import annotations

from typing import NamedTuple

from .._kernel.payloads import (  # noqa: F401  (Option-i downward re-export)
    ElementGroup,
    GroupResult,
    resolve_type_filter,
)


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
# Element topology — the one table for walking gmsh connectivity
# =====================================================================
#
# A walk over ``gmsh.model.mesh.getElements`` output needs, per element
# type, the connectivity width (to reshape the flat node list) and the
# corner count: gmsh lists the corner nodes first, so ``row[:n_corner]``
# is the straight-sided corner element and ``row[0], row[1]`` are the
# two ends of a line.  Both follow from the curated alias (``'quad9'``
# is shape ``quad`` with 9 nodes), so the table is derived from
# ``_KNOWN_ALIASES`` rather than copied.  Consult it instead of writing
# an inline ``{2: 3, 3: 4, ...}`` map: inline copies in the load and
# mass composites omitted quad9 and hex27 (silently dropping their
# masses and loads) and took a line3's mid node for its end node.

class ElementTopology(NamedTuple):
    """Connectivity shape of one Gmsh element type."""

    dim: int
    """Topological dimension (0–3)."""
    npe: int
    """Nodes per element — the width of one connectivity row."""
    n_corner: int
    """Corner (vertex) nodes, which gmsh lists first in each row."""


# shape prefix -> (topological dim, corner-node count)
_SHAPE_TOPOLOGY: dict[str, tuple[int, int]] = {
    'point':   (0, 1),
    'line':    (1, 2),
    'tri':     (2, 3),
    'quad':    (2, 4),
    'tet':     (3, 4),
    'hex':     (3, 8),
    'prism':   (3, 6),
    'pyramid': (3, 5),
}


def _topology_from_alias(alias: str) -> ElementTopology:
    """``'quad9'`` → ``ElementTopology(dim=2, npe=9, n_corner=4)``."""
    shape = alias.rstrip('0123456789')
    dim, n_corner = _SHAPE_TOPOLOGY[shape]
    return ElementTopology(dim=dim, npe=int(alias[len(shape):]),
                           n_corner=n_corner)


TOPOLOGY_BY_CODE: dict[int, ElementTopology] = {
    code: _topology_from_alias(alias)
    for code, alias in _KNOWN_ALIASES.items()
}


def element_topology(
    code: int,
    dim: int,
    *,
    integrable: bool = False,
    context: str = "",
) -> ElementTopology:
    """Topology of gmsh element type *code*, which must be a *dim*-D type.

    The lookup for code that walks ``getElements`` output.  An element
    type the walk cannot handle raises instead of being skipped: a
    skipped type silently drops every mass or load on its elements.

    Parameters
    ----------
    code : int
        Gmsh element-type code.
    dim : int
        The dimension the caller walks (``getElements(dim, tag)``
        returns only elements of that dimension).
    integrable : bool
        Set when the caller hands the **full** connectivity row of a
        face or cell to a resolver.  The resolvers identify an element
        by its node count and integrate it with the :mod:`apeGmsh.fem`
        shape functions, so only a type with a shape-function catalog
        entry is accepted (a 9-node tri9 would otherwise be integrated
        as a quad9).  Corner-only walks leave it unset (any known type
        has corners), and so do edge walks: the edge integrator carries
        its own line2 / line3 shapes and refuses other widths itself.
    context : str
        Prefix for the error message, e.g. ``"load target 'slab'"``.

    Raises
    ------
    ValueError
        If *code* is not a known *dim*-D type, or, with
        ``integrable=True``, has no shape functions.
    """
    code = int(code)
    where = f"{context}: " if context else ""
    topo = TOPOLOGY_BY_CODE.get(code)
    if topo is None or topo.dim != dim:
        known = ", ".join(_KNOWN_ALIASES[c]
                          for c, t in TOPOLOGY_BY_CODE.items() if t.dim == dim)
        raise ValueError(
            f"{where}gmsh element type {code} is not a known {dim}-D "
            f"element type (known: {known}). Refusing to skip its "
            f"elements silently."
        )
    if integrable:
        from apeGmsh.fem._shape_functions import get_shape_functions

        if get_shape_functions(code) is None:
            ok = ", ".join(_KNOWN_ALIASES[c]
                           for c, t in TOPOLOGY_BY_CODE.items()
                           if t.dim == dim and get_shape_functions(c) is not None)
            raise ValueError(
                f"{where}gmsh element type {code} ({_KNOWN_ALIASES[code]}) "
                f"has no shape functions in apeGmsh, so its elements cannot "
                f"be integrated. Mesh this target with one of: {ok}."
            )
    return topo


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
# ElementGroup / GroupResult / resolve_type_filter
# =====================================================================
#
# RELOCATED to apeGmsh._kernel.payloads (selection-unification-v2
# P1-K, THE KEYSTONE — closes HT1/HT8/R3-B).  Class identity is
# unchanged; only the module path moved.  These three names are
# re-exported above via
#     from .._kernel.payloads import ElementGroup, GroupResult, resolve_type_filter
# (a downward mesh -> _kernel edge — the intended layering
# direction) so from apeGmsh.mesh._element_types import ElementGroup
# (and GroupResult / resolve_type_filter) and the byte-unchanged
# contract/pin tests keep resolving.  ElementTypeInfo / make_type_info
# / the alias machinery stay HERE (they never call the moved trio, so
# no back-edge).  Flagged as a P3/P4 internal-cleanup candidate.
