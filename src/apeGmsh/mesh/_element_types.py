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
