"""apeGmsh.results.spec — recorder declaration + resolved spec + emit helpers.

Phase 8.3b relocated the recorder cluster out of ``apeGmsh.solvers``
into this sub-package:

- :mod:`apeGmsh.results.spec._resolved` — resolved containers
  (:class:`ResolvedRecorderSpec`, :class:`ResolvedRecorderRecord`)
  consumed by the results-side capture / live / transcoder / writer
  machinery.
- :mod:`apeGmsh.results.spec._emit` — emission helpers
  (:func:`emit_logical`, :func:`to_ops_args`, :func:`mpco_ops_args`,
  :func:`line_station_gpx_path`, …) that translate a resolved spec
  into OpenSees-native recorder commands.

Phase 9 commit 4 relocates the :class:`Recorders` fluent helper to
the bridge package (``apeGmsh.opensees.recorder``). This package
continues to re-export ``Recorders`` for one release cycle via a
lazy ``__getattr__`` (avoiding the cycle that an eager import would
create with the bridge-side
:mod:`apeGmsh.opensees._recorders_builder` module's transitional
``_resolved`` dependency).

Canonical user import after commit 4::

    from apeGmsh.opensees.recorder import Recorders

The ``apeGmsh.results.spec.Recorders`` and
``apeGmsh.results.spec.declaration.Recorders`` paths continue to
work; the latter fires a one-shot :class:`DeprecationWarning`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._resolved import ResolvedRecorderRecord, ResolvedRecorderSpec

if TYPE_CHECKING:
    from ...opensees.recorder import Recorders  # noqa: F401

# ``Recorders`` is exposed via ``__getattr__`` below (lazy re-export
# from the bridge). Ruff's F822 check doesn't follow module-level
# ``__getattr__``, so the entry is annotated explicitly.
__all__ = [  # noqa: F822
    "Recorders",
    "ResolvedRecorderRecord",
    "ResolvedRecorderSpec",
]


def __getattr__(name: str) -> Any:
    """Lazy re-export of ``Recorders`` from the bridge.

    Avoids the eager import cycle between ``apeGmsh.opensees.recorder``
    (which transitively imports
    :mod:`apeGmsh.results.spec._resolved`) and this package's
    ``__init__``. No DeprecationWarning is fired for
    ``apeGmsh.results.spec.Recorders`` — the deprecation envelope is
    only attached to the legacy module path
    :mod:`apeGmsh.results.spec.declaration`.
    """
    if name == "Recorders":
        from ...opensees.recorder import Recorders
        return Recorders
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )
