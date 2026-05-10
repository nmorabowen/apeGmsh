"""
Coordinate-system primitives for ``geomTransf`` orientation.

This module re-exports the CS classes shipped in
:mod:`apeGmsh.solvers._opensees_csys` per ADR 0010. That cross-package
import is the **only** sanctioned dependency from
:mod:`apeGmsh.opensees` into :mod:`apeGmsh.solvers` — see ADR 0009.

The actual GeomTransf primitives (``Linear``, ``PDelta``,
``Corotational``) land in Phase 1D and will live in this module
alongside the re-exports.
"""
from __future__ import annotations

from apeGmsh.solvers._opensees_csys import (
    Cartesian,
    Cylindrical,
    Spherical,
    resolve_vecxz,
)

__all__ = [
    "Cartesian",
    "Cylindrical",
    "Spherical",
    "resolve_vecxz",
]
