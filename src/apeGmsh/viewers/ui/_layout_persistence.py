"""Layout persistence — ``QSettings``-backed save/restore for QMainWindow.

Persists window geometry, dock positions/sizes, visibility, floating
state, and tabification across viewer launches. Schema-versioned so
incompatible changes can be detected and fall back to defaults rather
than producing a corrupted layout.

Usage::

    persist = LayoutPersistence(window_key="results")
    # On show:
    if not persist.restore(window):
        window.resize(1600, 1000)        # default geometry
    # On close:
    persist.save(window)

QSettings location is platform-standard (registry on Windows,
``~/.config`` on Linux, ``~/Library/Preferences`` on macOS) — Qt
handles the path. The org/app combo is fixed: ``"apeGmsh"`` /
``"viewers.{window_key}"``.

Schema discipline
-----------------
The :attr:`SCHEMA_VERSION` constant is the persistence invariant.
Bump it whenever you make a *non-additive* change to the dock layout
(rename a ``dock_id``, change tabify groupings significantly, etc.).
Old saved state with a mismatched schema is silently ignored on
restore — users fall back to defaults, not a broken layout.

Additive changes (a new dock_id appears in the registry) don't need a
bump; Qt's ``restoreState`` just leaves the new dock at its default
position.
"""
from __future__ import annotations

from typing import Any


class LayoutPersistence:
    """Save/restore window + dock layout via ``QSettings``.

    Parameters
    ----------
    window_key
        Stable identifier for this window's settings group. Different
        viewers use different keys; same key means same persistent
        layout. Conventional values: ``"results"``, ``"mesh"``,
        ``"model"``.

    Notes
    -----
    The class is stateless beyond ``window_key`` — every call opens
    a fresh ``QSettings`` handle. Safe to construct cheaply.
    """

    SCHEMA_VERSION: int = 1

    _ORG: str = "apeGmsh"
    _APP_PREFIX: str = "viewers"

    def __init__(self, window_key: str) -> None:
        if not window_key:
            raise ValueError("LayoutPersistence requires a non-empty window_key")
        # Reject characters that would split QSettings paths or
        # confuse the key namespace.
        if "/" in window_key or "\\" in window_key:
            raise ValueError(
                f"LayoutPersistence.window_key={window_key!r} must not "
                f"contain path separators"
            )
        self._window_key = window_key

    @property
    def window_key(self) -> str:
        return self._window_key

    def _settings(self) -> Any:
        from qtpy import QtCore
        return QtCore.QSettings(
            self._ORG, f"{self._APP_PREFIX}.{self._window_key}",
        )

    # ------------------------------------------------------------------
    # save / restore
    # ------------------------------------------------------------------

    def save(self, window: Any) -> None:
        """Capture window geometry + dock state to ``QSettings``.

        Call from ``QMainWindow.closeEvent`` (after the user's own
        on-close handlers, before ``super().closeEvent``).
        """
        s = self._settings()
        s.setValue("schema", self.SCHEMA_VERSION)
        s.setValue("geometry", window.saveGeometry())
        s.setValue("state", window.saveState(self.SCHEMA_VERSION))
        s.sync()

    def restore(self, window: Any) -> bool:
        """Apply saved geometry + dock state to ``window``.

        Call before showing the window (and before the registry
        constructs / mounts docks isn't required — Qt's restoreState
        matches docks by ``objectName`` after they exist, but applying
        geometry first avoids a visible resize flash).

        Returns
        -------
        bool
            ``True`` if a saved state existed, was for the current
            schema, and was applied. ``False`` if no state was found,
            the schema mismatched, or Qt rejected the restore (corrupt
            blob). On ``False`` the caller should apply default
            geometry — the window state is unchanged.
        """
        s = self._settings()
        schema = s.value("schema")
        if schema is None:
            return False
        try:
            schema_int = int(schema)
        except (TypeError, ValueError):
            return False
        if schema_int != self.SCHEMA_VERSION:
            return False
        geom = s.value("geometry")
        state = s.value("state")
        if geom is None or state is None:
            return False
        if not window.restoreGeometry(geom):
            return False
        if not window.restoreState(state, self.SCHEMA_VERSION):
            return False
        return True

    def reset(self) -> None:
        """Delete saved state for this ``window_key``.

        Used by a "Reset Layout" action in the View menu. The next
        call to :meth:`restore` will return ``False`` until the next
        :meth:`save`.
        """
        s = self._settings()
        s.clear()
        s.sync()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def has_saved_state(self) -> bool:
        """``True`` if a (possibly stale-schema) saved state exists.

        Useful for the View menu — disable "Reset Layout" when there's
        nothing to reset.
        """
        s = self._settings()
        return s.value("schema") is not None
