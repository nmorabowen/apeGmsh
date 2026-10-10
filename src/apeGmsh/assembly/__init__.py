"""``apeGmsh.assembly`` — instances of model files, tied by label (ADR 0117).

``Assembly`` is imported from this sub-package, not the top-level package
(``from apeGmsh.assembly import Assembly``; the v1.0 contract in
``tests/test_library_contracts.py``). The v2 API is
``instance`` / ``tie`` (and the coupling verbs) / ``bridge``. The v1
``add`` / ``couple(part_a, part_b, ports=)`` / ``materialize`` and
``g.compose`` were removed in AS5-c (ADR 0117 D7).
"""
from __future__ import annotations

from ._assembly import Assembly
from ._instances import Instance, Tie
from ._errors import AssemblyError

__all__ = ["Assembly", "AssemblyError", "Instance", "Tie"]
