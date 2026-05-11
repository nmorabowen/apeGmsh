"""apeGmsh.solvers — Legacy compatibility shim (Phase 8.1).

This package is being retired.  Phase 8.1 moved the broker-side
content (record dataclasses, the constraint / load / mass resolvers,
and the Numberer) out of ``apeGmsh.solvers`` into the mesh and core
layers where they belong; the OpenSees-side helpers
(``OpenSees``, ``Cartesian``/``Cylindrical``/``Spherical``, the
recorder / response catalog, …) move in subsequent sub-phases.

Until those finish, this module keeps the old import paths alive.
The per-module shim files (``Numberer.py``, ``Constraints.py``,
``Loads.py``, ``Masses.py``, ``_kinds.py``, …) warn on import;
the package-level ``__getattr__`` below catches the
``from apeGmsh.solvers import Numberer`` shape so that path also
emits a one-shot :class:`DeprecationWarning`.

OpenSees-side re-exports (``OpenSees``, the coordinate-system
helpers) stay eager and silent — they relocate in Phase 8.2.
"""

from __future__ import annotations

import warnings
from typing import Any

from .OpenSees import OpenSees
from ._opensees_csys import Cartesian, Cylindrical, Spherical

# Phase 8.1 names that moved out of apeGmsh.solvers.  Listed as
# ``{attr: (canonical_module, canonical_attr)}`` so __getattr__ can
# point users at the new home.
_RELOCATED: dict[str, tuple[str, str]] = {
    "Numberer":     ("apeGmsh.mesh._numberer", "Numberer"),
    "NumberedMesh": ("apeGmsh.mesh._numberer", "NumberedMesh"),
}


def __getattr__(name: str) -> Any:
    """Lazy attribute hook for Phase 8.1-relocated names.

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
    "OpenSees",
    "Cartesian",
    "Cylindrical",
    "Spherical",
    # Numberer / NumberedMesh accessible via __getattr__
    "Numberer",
    "NumberedMesh",
]
