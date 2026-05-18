"""apeGmsh.core.loads.defs — backward-compatible import shim.

The load-definition dataclasses were relocated to
:mod:`apeGmsh._kernel.defs.loads` (selection-unification-v2 P1-K, the
keystone cycle-break).  Class identity is unchanged — only the module
path moved.

This module is a thin **downward** re-export (``core`` → ``_kernel``,
the intended layering direction) so that the public
``apeGmsh.core.loads.defs`` path and the byte-unchanged contract tests
keep resolving.  Flagged as a P3/P4 internal-cleanup candidate (sweep
with the legacy surface).
"""

from __future__ import annotations

from apeGmsh._kernel.defs.loads import (  # noqa: F401
    BodyLoadDef,
    FaceLoadDef,
    FaceSPDef,
    GravityLoadDef,
    LineLoadDef,
    LoadDef,
    PointClosestLoadDef,
    PointLoadDef,
    SurfaceLoadDef,
)
from apeGmsh._kernel.defs.loads import __all__ as __all__  # noqa: F401
