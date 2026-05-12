"""apeGmsh.solvers — Legacy compatibility shim.

This package is being retired in stages by the Phase-8 untangle:

* Phase 8.1 moved the broker-side content (record dataclasses, the
  constraint / load / mass resolvers, the Numberer) into the mesh
  and core layers.
* Phase 8.2 moved the coordinate-system helpers
  (``Cartesian`` / ``Cylindrical`` / ``Spherical`` / ``resolve_vecxz``)
  into ``apeGmsh.opensees``.
* Phase 8 PR γ (this revision) deleted the legacy ``OpenSees``
  bridge class and the ``_opensees_*`` cluster.  ``g.opensees`` is
  no longer a session attribute; construct decks explicitly via
  ``apeGmsh.opensees.apeSees(fem)``.
* Phase 8.3b will move the recorder cluster
  (``Recorders.py``, ``_recorder_emit.py``, ``_recorder_specs.py``,
  ``_element_specs.py``) and rewire the results layer onto the
  typed ``apeGmsh.opensees.recorder`` primitives.

Until 8.3b finishes, this package keeps the old import paths alive
for legacy attribute access.  The per-module shim files
(``Numberer.py``, ``Constraints.py``, ``Loads.py``, ``Masses.py``,
``_kinds.py``, ``_opensees_csys.py``, ``_element_response.py``)
warn on import; the package-level ``__getattr__`` below catches the
``from apeGmsh.solvers import X`` shape so that path also emits a
one-shot :class:`DeprecationWarning`.
"""

from __future__ import annotations

import warnings
from typing import Any

# Names that moved out of apeGmsh.solvers during the Phase-8 untangle.
# Listed as ``{attr: (canonical_module, canonical_attr)}`` so
# __getattr__ can point users at the new home.
#
# Phase 8.1 — broker-side records & resolvers (mesh/ + core/).
# Phase 8.2 — bridge-side coordinate-system helpers (opensees/).
_RELOCATED: dict[str, tuple[str, str]] = {
    # Phase 8.1
    "Numberer":     ("apeGmsh.mesh._numberer", "Numberer"),
    "NumberedMesh": ("apeGmsh.mesh._numberer", "NumberedMesh"),
    # Phase 8.2
    "Cartesian":    ("apeGmsh.opensees", "Cartesian"),
    "Cylindrical":  ("apeGmsh.opensees", "Cylindrical"),
    "Spherical":    ("apeGmsh.opensees", "Spherical"),
}


def __getattr__(name: str) -> Any:
    """Lazy attribute hook for Phase-8 relocated names.

    Fires a one-shot :class:`DeprecationWarning` and returns the
    canonical object so existing ``from apeGmsh.solvers import X``
    code keeps working for one release cycle.
    """
    target = _RELOCATED.get(name)
    if target is not None:
        mod_path, attr = target
        warnings.warn(
            f"apeGmsh.solvers.{name} is deprecated; import {attr} from "
            f"{mod_path} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from importlib import import_module
        return getattr(import_module(mod_path), attr)
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )


__all__ = [
    # Accessible via __getattr__ (one-shot DeprecationWarning):
    "Numberer",
    "NumberedMesh",
    "Cartesian",
    "Cylindrical",
    "Spherical",
]
