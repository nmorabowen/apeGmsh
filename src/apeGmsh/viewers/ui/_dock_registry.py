"""Dock registry — decouple dock construction from per-viewer code.

Replaces the inline ``add_right_bottom_dock`` / inline ``addDockWidget``
pattern in :class:`ViewerWindow`. Each viewer builds a
:class:`DockRegistry` of :class:`DockSpec` entries; the window walks
the registry once at mount time, creating ``QDockWidget`` instances
with stable ``objectName`` values.

Stable ``objectName`` is what lets Qt's ``saveState`` / ``restoreState``
roundtrip dock positions, sizes, visibility, floating state, and
tabification across sessions. See :class:`LayoutPersistence` for the
state persistence side.

Usage::

    reg = DockRegistry()
    reg.register(DockSpec(
        dock_id="outline",
        title="Outline",
        factory=lambda parent: OutlineTree(parent),
        default_area="left",
    ))
    reg.register(DockSpec(
        dock_id="diagrams",
        title="Diagrams",
        factory=lambda parent: DiagramsTab(parent, director),
        tabify_with="outline",          # ← grouped with outline tab
    ))
    docks = reg.mount(window)   # dict[dock_id, QDockWidget]

Design notes
------------
* The factory takes ``parent`` and returns the *content* widget. The
  ``QDockWidget`` itself is constructed by :meth:`DockRegistry.mount`.
  Factories run in registration order — keep them cheap.
* ``dock_id`` is used directly as the Qt ``objectName``. Renaming an
  existing ``dock_id`` orphans its persisted state silently; bump
  ``LayoutPersistence.SCHEMA_VERSION`` if you need a hard reset.
* ``tabify_with`` requires the referenced dock to already be registered
  — forward references are rejected at :meth:`register` time so layout
  bugs surface early.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional


_AREAS = ("left", "right", "top", "bottom")


@dataclass(frozen=True)
class DockSpec:
    """One dock's registration — opaque to the registry, applied by mount.

    Attributes
    ----------
    dock_id
        Stable identifier. Becomes the QDockWidget's ``objectName``.
        Must be unique within a registry. Stable across launches —
        renaming orphans the persisted state.
    title
        User-facing label (dock title bar + View menu toggle action).
    factory
        ``factory(parent: QWidget) -> QWidget`` — builds the dock's
        content widget. Called at mount time.
    default_area
        ``"left" | "right" | "top" | "bottom"`` — initial dock area
        when no saved state exists. Ignored on restored launches.
    default_visible
        Initial visibility when no saved state exists.
    default_floating
        Initial floating state when no saved state exists.
    tabify_with
        ``dock_id`` of a previously-registered dock to tabify with.
        Forward references are rejected; register the parent dock
        first.
    """
    dock_id: str
    title: str
    factory: Callable[[Any], Any]
    default_area: str = "right"
    default_visible: bool = True
    default_floating: bool = False
    tabify_with: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.dock_id:
            raise ValueError("DockSpec.dock_id must be non-empty")
        # objectName is used in QSettings keys — keep it boring.
        bare = self.dock_id.replace("_", "").replace("-", "").replace(".", "")
        if not bare.isalnum():
            raise ValueError(
                f"DockSpec.dock_id={self.dock_id!r} must be alphanumeric "
                f"with optional '_' / '-' / '.' separators (used as Qt "
                f"objectName + QSettings key)"
            )
        if self.default_area not in _AREAS:
            raise ValueError(
                f"DockSpec.default_area={self.default_area!r} must be "
                f"one of {_AREAS}"
            )


class DockRegistry:
    """Holds :class:`DockSpec` entries; mounts them onto a QMainWindow.

    The registry is a passive container — it doesn't import Qt until
    :meth:`mount` is called. Construct freely in headless / test
    contexts.
    """

    def __init__(self) -> None:
        self._specs: list[DockSpec] = []
        self._ids: set[str] = set()

    def __len__(self) -> int:
        return len(self._specs)

    def __contains__(self, dock_id: str) -> bool:
        return dock_id in self._ids

    def register(self, spec: DockSpec) -> None:
        """Add ``spec``. Validates uniqueness + tabify_with backref.

        Raises
        ------
        ValueError
            If ``spec.dock_id`` is already registered, or
            ``spec.tabify_with`` references an unregistered id.
        """
        if spec.dock_id in self._ids:
            raise ValueError(
                f"Duplicate dock_id={spec.dock_id!r} in registry"
            )
        if spec.tabify_with is not None and spec.tabify_with not in self._ids:
            raise ValueError(
                f"DockSpec(dock_id={spec.dock_id!r}).tabify_with="
                f"{spec.tabify_with!r} references an unregistered dock — "
                f"register the parent dock first"
            )
        self._specs.append(spec)
        self._ids.add(spec.dock_id)

    def specs(self) -> list[DockSpec]:
        """Read-only snapshot of registered specs (registration order)."""
        return list(self._specs)

    def mount(self, window: Any) -> dict[str, Any]:
        """Instantiate every registered dock onto ``window``.

        Walks specs in registration order. For each:

        1. Calls the spec's factory with ``window`` as parent.
        2. Wraps the result in a ``QDockWidget`` whose ``objectName``
           is the spec's ``dock_id``.
        3. Calls ``window.addDockWidget(default_area, dock)``.
        4. If ``tabify_with`` is set, calls
           ``window.tabifyDockWidget(parent, dock)`` to group them.
        5. Applies ``default_visible`` / ``default_floating``.

        Returns
        -------
        dict[str, QDockWidget]
            Map from ``dock_id`` → mounted ``QDockWidget``. The window
            owns the docks via Qt parentage; this dict is a
            convenience for callers that need to address individual
            docks (toggle actions, programmatic show/hide).
        """
        from qtpy import QtCore, QtWidgets

        area_map = {
            "left":   QtCore.Qt.DockWidgetArea.LeftDockWidgetArea,
            "right":  QtCore.Qt.DockWidgetArea.RightDockWidgetArea,
            "top":    QtCore.Qt.DockWidgetArea.TopDockWidgetArea,
            "bottom": QtCore.Qt.DockWidgetArea.BottomDockWidgetArea,
        }

        docks: dict[str, Any] = {}
        for spec in self._specs:
            content = spec.factory(window)
            dock = QtWidgets.QDockWidget(spec.title, window)
            # objectName MUST be set for Qt's saveState/restoreState to
            # match the dock across sessions. Without it Qt warns at
            # save time and silently skips at restore time.
            dock.setObjectName(spec.dock_id)
            dock.setWidget(content)
            window.addDockWidget(area_map[spec.default_area], dock)
            if spec.tabify_with is not None:
                window.tabifyDockWidget(docks[spec.tabify_with], dock)
            dock.setFloating(spec.default_floating)
            dock.setVisible(spec.default_visible)
            docks[spec.dock_id] = dock
        return docks
