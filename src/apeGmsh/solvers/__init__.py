"""apeGmsh.solvers — Legacy compatibility shim.

This package is being retired in stages by the Phase-8 untangle:

* Phase 8.1 moved the broker-side content (record dataclasses, the
  constraint / load / mass resolvers, the Numberer) into the mesh
  and core layers.
* Phase 8.2 moved the coordinate-system helpers
  (``Cartesian`` / ``Cylindrical`` / ``Spherical`` / ``resolve_vecxz``)
  into ``apeGmsh.opensees``.
* Phase 8.3a moved the OpenSees element response catalog
  (``_element_response.py``) into
  :mod:`apeGmsh.opensees._response_catalog`.
* Phase 8 PR γ deleted the legacy ``OpenSees`` bridge class and the
  ``_opensees_*`` cluster.  ``g.opensees`` is no longer a session
  attribute; construct decks explicitly via
  ``apeGmsh.opensees.apeSees(fem)``.
* Phase 8.3b moved the recorder cluster:

  - ``Recorders.py`` → :mod:`apeGmsh.results.spec.declaration`
    (re-exported as ``apeGmsh.results.spec.Recorders``)
  - ``_recorder_specs.py`` → :mod:`apeGmsh.results.spec._resolved`
    (``ResolvedRecorderSpec`` / ``ResolvedRecorderRecord``
    re-exported under ``apeGmsh.results.spec``)
  - ``_recorder_emit.py`` → :mod:`apeGmsh.results.spec._emit`
  - ``_element_specs.py`` → :mod:`apeGmsh.opensees._element_capabilities`

After Phase 8.3b this package is purely a deprecation envelope.  The
per-module shim files (``Numberer.py``, ``Constraints.py``, ``Loads.py``,
``Masses.py``, ``_kinds.py``, ``_opensees_csys.py``, ``_element_response.py``,
``Recorders.py``, ``_recorder_specs.py``, ``_recorder_emit.py``,
``_element_specs.py``) warn on import; the package-level ``__getattr__``
below catches the ``from apeGmsh.solvers import X`` shape so that path
also emits a one-shot :class:`DeprecationWarning`.

Phase 8.8 will delete this package after one release cycle of
deprecation.
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
# Phase 8.3b — recorder cluster (results/spec/ + opensees/).
_RELOCATED: dict[str, tuple[str, str]] = {
    # Phase 8.1
    "Numberer":              ("apeGmsh.mesh._numberer", "Numberer"),
    "NumberedMesh":          ("apeGmsh.mesh._numberer", "NumberedMesh"),
    # Phase 8.2
    "Cartesian":             ("apeGmsh.opensees", "Cartesian"),
    "Cylindrical":           ("apeGmsh.opensees", "Cylindrical"),
    "Spherical":             ("apeGmsh.opensees", "Spherical"),
    # Phase 8.3b (Recorders subsequently relocated again in Phase 9 commit 4)
    "Recorders":             ("apeGmsh.opensees.recorder", "Recorders"),
    "ResolvedRecorderSpec":  ("apeGmsh.results.spec", "ResolvedRecorderSpec"),
    "ResolvedRecorderRecord":("apeGmsh.results.spec", "ResolvedRecorderRecord"),
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
    "Recorders",
    "ResolvedRecorderSpec",
    "ResolvedRecorderRecord",
]
