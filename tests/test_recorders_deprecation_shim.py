"""Phase 9 commit 4 — Recorders deprecation envelope tests.

After the relocation to ``apeGmsh.opensees.recorder``, the legacy
import paths continue to work for one release cycle:

- ``apeGmsh.results.spec.declaration.Recorders`` — fires a one-shot
  :class:`DeprecationWarning` per attribute access.
- ``apeGmsh.solvers.Recorders`` — fires its own ``DeprecationWarning``
  (the Phase 8.3b envelope, now pointing at the new canonical home).
- ``apeGmsh.results.spec.Recorders`` — package re-export, no
  ``DeprecationWarning`` (the package-level access path is the
  recommended migration target for callers who don't want to depend
  on the bridge package directly).

In all three cases the object returned is the canonical
``apeGmsh.opensees.recorder.Recorders`` class.
"""
from __future__ import annotations

import warnings


CANONICAL_RECORDERS_MOD = "apeGmsh.opensees.recorder"


def test_canonical_path_imports_cleanly() -> None:
    """No warnings when importing from the canonical bridge path."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning fails the test
        from apeGmsh.opensees.recorder import Recorders  # noqa: F401


def test_legacy_declaration_module_fires_deprecation_warning() -> None:
    """The legacy ``...spec.declaration.Recorders`` access warns."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        from apeGmsh.results.spec.declaration import Recorders as Legacy
    deprec = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert deprec, (
        "Expected DeprecationWarning when importing Recorders from "
        "apeGmsh.results.spec.declaration"
    )
    # The warning message should point at the canonical bridge path.
    assert any(CANONICAL_RECORDERS_MOD in str(w.message) for w in deprec)
    # And the returned object IS the canonical class.
    from apeGmsh.opensees.recorder import Recorders as Canonical
    assert Legacy is Canonical


def test_solvers_path_fires_deprecation_warning() -> None:
    """``apeGmsh.solvers.Recorders`` still works (Phase 8.3b envelope)."""
    import apeGmsh.solvers as solvers
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Legacy = solvers.Recorders
    deprec = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert deprec, (
        "Expected DeprecationWarning when accessing "
        "apeGmsh.solvers.Recorders"
    )
    # Target message should now point at the bridge path (post commit 4).
    assert any(CANONICAL_RECORDERS_MOD in str(w.message) for w in deprec)
    # Resolves to the canonical class.
    from apeGmsh.opensees.recorder import Recorders as Canonical
    assert Legacy is Canonical


def test_results_spec_package_path_resolves_without_warning() -> None:
    """``apeGmsh.results.spec.Recorders`` is a quiet pivoted re-export."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        from apeGmsh.results.spec import Recorders as Pivoted
    from apeGmsh.opensees.recorder import Recorders as Canonical
    assert Pivoted is Canonical
