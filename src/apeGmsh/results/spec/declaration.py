"""apeGmsh.results.spec.declaration — Phase 9 commit 4 deprecation shim.

The :class:`Recorders` fluent helper relocated from this module to
:mod:`apeGmsh.opensees.recorder` (the bridge package) in Phase 9
commit 4. Importing from the legacy path continues to work for one
release cycle but fires a one-shot :class:`DeprecationWarning` per
attribute access.

New code should prefer :func:`ops.recorder.declare`, which returns a
typed :class:`apeGmsh.opensees.recorder.RecorderDeclaration` directly
on the bridge without going through ``Recorders.resolve``.
"""
from __future__ import annotations

import warnings
from typing import Any

# Names that moved from this module to the bridge in Phase 9 commit 4.
# Listed as ``{attr: (canonical_module, canonical_attr)}``.
_RELOCATED: dict[str, tuple[str, str]] = {
    "Recorders": ("apeGmsh.opensees.recorder", "Recorders"),
}


def __getattr__(name: str) -> Any:
    """Lazy attribute hook for Phase-9 relocated names.

    Fires a one-shot :class:`DeprecationWarning` and returns the
    canonical object so existing ``from apeGmsh.results.spec.declaration
    import X`` code keeps working for one release cycle.
    """
    target = _RELOCATED.get(name)
    if target is not None:
        mod_path, attr = target
        warnings.warn(
            f"apeGmsh.results.spec.declaration.{name} is deprecated; "
            f"import {attr} from {mod_path} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from importlib import import_module
        return getattr(import_module(mod_path), attr)
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )


# Names resolved via module ``__getattr__`` above (lazy with a
# one-shot ``DeprecationWarning``). Static checkers can't follow
# ``__getattr__``, so we annotate the ``__all__`` entries explicitly.
__all__ = ["Recorders"]  # noqa: F822
