"""apeGmsh._kernel.defs — pre-mesh definition dataclasses.

Relocated verbatim from ``apeGmsh.core.{loads,masses,constraints}.defs``
(plan P1-K).  These describe *intent* at the geometry/PG/part level
before meshing; the resolved counterparts live in
:mod:`apeGmsh._kernel.records`.

Pure: ``dataclasses`` only, zero ``apeGmsh.*`` imports.  The old
``apeGmsh.core.*.defs`` module paths remain importable via thin
downward re-export stubs (Option-i — keeps the byte-unchanged
contract tests working).
"""

from __future__ import annotations
