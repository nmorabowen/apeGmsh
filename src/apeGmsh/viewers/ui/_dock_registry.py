"""Dock registry + helpers — decouple dock construction from per-viewer code.

Replaces the inline ``addDockWidget`` pattern in :class:`ViewerWindow`
and :class:`ResultsWindow`. Each viewer builds a
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
from typing import Any, Callable, Iterable, Optional, Tuple


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
    split_below
        ``objectName`` of an already-mounted dock to split UNDER
        (vertical split — this dock lands full-width below the target
        in the same dock area). Mutually exclusive with
        ``tabify_with``. The mechanism behind ADR 0088 D3's
        Output-console-below-the-scrubber placement.
    sanitize
        Opt-in to per-launch degenerate-placement healing via
        :func:`sanitize_dock_placement`. Set ``True`` for primary
        navigation docks (Outline / Selection) so a corrupt persisted
        layout can never trap them floating, in the Top/Bottom area, or
        crushed below the size floors. Right-side tool docks leave this
        ``False`` so their own size policy is untouched. When ``True``,
        supply ``min_width`` / ``min_height`` (the sticky floors) and
        ``initial_width`` / ``initial_height`` (applied only when the
        dock had to be re-docked).
    min_width, min_height, initial_width, initial_height
        Size floors + initial extents used by the sanitizer. Only read
        when ``sanitize`` is ``True``; ``None`` skips that axis.
    """
    dock_id: str
    title: str
    factory: Callable[[Any], Any]
    default_area: str = "right"
    default_visible: bool = True
    default_floating: bool = False
    tabify_with: Optional[str] = None
    split_below: Optional[str] = None
    sanitize: bool = False
    min_width: Optional[int] = None
    min_height: Optional[int] = None
    initial_width: Optional[int] = None
    initial_height: Optional[int] = None

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
        if self.tabify_with is not None and self.split_below is not None:
            raise ValueError(
                f"DockSpec(dock_id={self.dock_id!r}) sets both "
                f"tabify_with and split_below — they are mutually "
                f"exclusive placements"
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

        Walks specs in registration order. Each spec is mounted by
        :func:`mount_dock_spec` — see that function for the per-spec
        mount steps. Returns a dict mapping ``dock_id`` → mounted
        ``QDockWidget``. Qt parentage keeps the docks alive; this dict
        is just a convenience for callers that need to address
        individual docks.
        """
        docks: dict[str, Any] = {}
        reserved: set[str] = set()
        for spec in self._specs:
            dock = mount_dock_spec(
                window, spec, reserved_ids=reserved,
            )
            docks[spec.dock_id] = dock
            reserved.add(spec.dock_id)
        return docks


def mount_dock_spec(
    window: Any,
    spec: DockSpec,
    *,
    reserved_ids: set[str] = frozenset(),
) -> Any:
    """Mount a single :class:`DockSpec` onto ``window`` — module-level helper.

    Used by :meth:`DockRegistry.mount` to iterate its specs, and by
    :class:`ResultsWindow` to mount extension docks alongside its own
    pre-built dock set. Module-level for direct unit-testing against a
    vanilla ``QMainWindow`` without the VTK overhead of a full viewer
    shell.

    Steps:

    1. Validate ``spec.dock_id`` is not in ``reserved_ids``.
    2. Call ``spec.factory(window)`` to build the content widget.
    3. Wrap in a ``QDockWidget`` whose ``objectName`` is ``spec.dock_id``.
    4. ``window.addDockWidget(default_area, dock)``.
    5. If ``spec.tabify_with`` is set, resolve via
       ``window.findChild(QDockWidget, spec.tabify_with)`` and call
       ``window.tabifyDockWidget(target, dock)``.
    6. Apply ``default_visible`` / ``default_floating``.

    Parameters
    ----------
    window
        The ``QMainWindow`` to mount onto.
    spec
        The :class:`DockSpec` to mount.
    reserved_ids
        ``dock_id``s the caller wants to forbid as collisions — e.g.
        the seven built-in objectNames in :class:`ResultsWindow`.
        Defaults to empty.

    Returns
    -------
    QDockWidget
        The newly-mounted dock.

    Raises
    ------
    ValueError
        If ``spec.dock_id`` is in ``reserved_ids``, or
        ``spec.tabify_with`` doesn't resolve to a child ``QDockWidget``.
    """
    from qtpy import QtCore, QtWidgets

    if spec.dock_id in reserved_ids:
        raise ValueError(
            f"Duplicate dock_id={spec.dock_id!r} (already reserved)"
        )

    area_map = {
        "left":   QtCore.Qt.DockWidgetArea.LeftDockWidgetArea,
        "right":  QtCore.Qt.DockWidgetArea.RightDockWidgetArea,
        "top":    QtCore.Qt.DockWidgetArea.TopDockWidgetArea,
        "bottom": QtCore.Qt.DockWidgetArea.BottomDockWidgetArea,
    }

    content = spec.factory(window)
    dock = QtWidgets.QDockWidget(spec.title, window)
    # objectName MUST be set for Qt's saveState/restoreState to match
    # the dock across sessions. Without it Qt warns at save time and
    # silently skips at restore time.
    dock.setObjectName(spec.dock_id)
    dock.setWidget(content)
    window.addDockWidget(area_map[spec.default_area], dock)

    if spec.tabify_with is not None:
        target = window.findChild(QtWidgets.QDockWidget, spec.tabify_with)
        if target is None:
            raise ValueError(
                f"DockSpec(dock_id={spec.dock_id!r}).tabify_with="
                f"{spec.tabify_with!r} not found in window — must "
                f"reference an already-mounted QDockWidget objectName"
            )
        window.tabifyDockWidget(target, dock)

    if spec.split_below is not None:
        target = window.findChild(QtWidgets.QDockWidget, spec.split_below)
        if target is None:
            raise ValueError(
                f"DockSpec(dock_id={spec.dock_id!r}).split_below="
                f"{spec.split_below!r} not found in window — must "
                f"reference an already-mounted QDockWidget objectName"
            )
        window.splitDockWidget(
            target, dock, QtCore.Qt.Orientation.Vertical,
        )
        # Size floors / initial extent for the split — read here even
        # without ``sanitize`` (the sanitizer serves nav docks; a
        # split-below console just wants its height honoured).
        if spec.min_height is not None:
            dock.setMinimumHeight(spec.min_height)
        if spec.initial_height is not None:
            try:
                window.resizeDocks(
                    [dock], [spec.initial_height],
                    QtCore.Qt.Orientation.Vertical,
                )
            except Exception:
                pass

    dock.setFloating(spec.default_floating)
    dock.setVisible(spec.default_visible)
    return dock


def wire_tabified_dock_action(action: Any, dock: Any) -> None:
    """Bind a checkable toolbar ``action`` to a possibly-tabified ``dock``.

    The naive pairing — ``toggled → setVisible(checked)`` plus
    ``visibilityChanged → setChecked(visible)`` — works for a
    free-standing dock (the colour-map editor) but is a feedback loop
    for a **tabified** one: Qt emits ``visibilityChanged(False)`` when
    a dock lands *behind* the current tab, which unchecks the action,
    which closes the dock on the very click that opened it (ADR 0083
    S1 review finding B1). Two changes break the loop:

    * opening uses ``show()`` **and** ``raise_()``, so the dock becomes
      the current tab (genuinely visible) instead of a background one;
    * check-state written back from ``visibilityChanged`` is guarded so
      it can never drive visibility — only a user click closes the dock.

    Net behaviour: click opens and fronts the dock; clicking again (or
    the dock's own close button) hides it; stacking it behind another
    tab merely unchecks the action, ready to re-front on the next click.
    """
    state = {"sync": False}

    def _on_toggled(checked: bool) -> None:
        if state["sync"]:
            return
        if checked:
            dock.show()
            dock.raise_()
        else:
            dock.hide()

    def _on_visibility(visible: bool) -> None:
        state["sync"] = True
        try:
            action.setChecked(bool(visible))
        finally:
            state["sync"] = False

    action.toggled.connect(_on_toggled)
    dock.visibilityChanged.connect(_on_visibility)


def sanitize_dock_placement(
    window: Any,
    dock: Any,
    *,
    min_width: Optional[int] = None,
    min_height: Optional[int] = None,
    initial_width: Optional[int] = None,
    initial_height: Optional[int] = None,
) -> None:
    """Idempotent per-launch heal for a *degenerate* dock placement.

    Run on every launch for opt-in navigation docks (``DockSpec.sanitize``)
    so a corrupt persisted layout can never trap them. The design is
    deliberately conservative — it heals only genuinely-degenerate
    states and leaves every healthy user choice (Left/Right area,
    persisted size, tabification) untouched, so this composes with the
    Plan-08 ``saveState`` / ``restoreState`` persistence rather than
    overriding it.

    Two independent mechanisms:

    1. **Sticky size floors.** ``setMinimumWidth`` / ``setMinimumHeight``
       are *properties* Qt enforces at layout time (including after the
       window is shown — these run before ``show()``). The **height**
       floor is the load-bearing fix for the recurring bug: the dock
       was collapsing to a title-bar-height sliver on the window's upper
       edge, which a width-only floor never caught. ``setMinimumHeight``
       reclaims the crushed space from the *sibling* dock without
       tearing this dock out of its split — verified headless.

    2. **Re-dock on true area-degeneracy only.** If the dock came back
       *floating* or pinned into the *Top/Bottom* area (where a vertical
       tree is useless), pull it back into the Left area and apply the
       initial extents. We intentionally do **not** key off
       ``dock.width()`` / ``dock.height()`` here: before ``show()`` those
       read 0, which would re-dock everything every launch and tear
       splits apart. Size-crush is the floors' job (mechanism 1); this
       branch is purely about a lost/nonsensical area.

    The one deliberate policy call: a dock the user *floated* that then
    loses its area on restore is snapped back to Left — that state is
    indistinguishable from the historical floating corruption, and a
    navigation dock floating-by-default isn't worth preserving. Left /
    Right docking and tabification ARE preserved.

    Parameters
    ----------
    window
        The host ``QMainWindow``.
    dock
        The ``QDockWidget`` to heal (already created + added).
    min_width, min_height
        Sticky size floors. ``None`` skips that axis.
    initial_width, initial_height
        Extents applied via ``resizeDocks`` only when the dock had to be
        re-docked. ``None`` skips that axis.
    """
    from qtpy import QtCore, QtWidgets

    Q = QtCore.Qt
    # Re-assert the standard feature set so no stale state can strip the
    # drag handles (the "can't move it" half of the bug).
    try:
        dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetClosable
            | QtWidgets.QDockWidget.DockWidgetMovable
            | QtWidgets.QDockWidget.DockWidgetFloatable
        )
    except Exception:
        pass

    # Mechanism 1 — sticky floors (heal size-crush, incl. the upper-edge
    # height sliver) without disturbing area or split.
    if min_width is not None:
        dock.setMinimumWidth(min_width)
    if min_height is not None:
        dock.setMinimumHeight(min_height)

    # Mechanism 2 — re-dock only on a lost / nonsensical area.
    area = window.dockWidgetArea(dock)
    area_degenerate = (
        dock.isFloating()
        or area == Q.TopDockWidgetArea
        or area == Q.BottomDockWidgetArea
    )
    if area_degenerate:
        dock.setFloating(False)
        window.addDockWidget(Q.LeftDockWidgetArea, dock)
        if initial_width is not None:
            try:
                window.resizeDocks([dock], [initial_width], Q.Horizontal)
            except Exception:
                pass
        if initial_height is not None:
            try:
                window.resizeDocks([dock], [initial_height], Q.Vertical)
            except Exception:
                pass


def build_view_menu(
    menu_bar: Any,
    *,
    docks: Iterable[Any],
    on_reset_layout: Optional[Callable[[], None]] = None,
    title: str = "View",
) -> Tuple[Any, Any]:
    """Build a ``QMenu`` of toggle actions — one per dock — plus a
    "Reset Layout" entry. Module-level for unit-testing against any
    ``QMenuBar`` + plain ``QDockWidget`` instances.

    Each dock's ``toggleViewAction()`` is added (checkable; text =
    dock's windowTitle). A separator is inserted before the Reset
    Layout entry so callers can later
    ``menu.insertAction(separator, new_toggle)`` to grow the toggle
    section without touching Reset Layout.

    Parameters
    ----------
    menu_bar
        The ``QMenuBar`` to attach to. Any pre-existing menu titled
        ``title`` is removed first (idempotent — callers can rebuild).
    docks
        Iterable of ``QDockWidget`` instances. Toggle actions appear
        in iteration order.
    on_reset_layout
        Optional callback bound to the "Reset Layout" action. If
        ``None``, the entry is still added but disabled.
    title
        Menu title. Defaults to ``"View"``.

    Returns
    -------
    (QMenu, QAction)
        The created menu and the separator action above "Reset
        Layout" — pass the separator to
        :func:`add_view_menu_toggle` when appending later toggles to
        keep "Reset Layout" pinned at the bottom.
    """
    from qtpy import QtWidgets

    # Drop any pre-existing menu with the same title.
    for action in list(menu_bar.actions()):
        if action.text() == title:
            menu_bar.removeAction(action)

    menu = menu_bar.addMenu(title)

    for dock in docks:
        toggle = dock.toggleViewAction()
        toggle.setCheckable(True)
        menu.addAction(toggle)

    separator = menu.addSeparator()

    # Sentence case per ADR 0087 INV-5 / Appendix B.
    reset_action = QtWidgets.QAction("Reset layout", menu)
    if on_reset_layout is not None:
        reset_action.triggered.connect(on_reset_layout)
    else:
        reset_action.setEnabled(False)
    menu.addAction(reset_action)

    return menu, separator


def add_view_menu_toggle(
    menu: Any,
    separator_before_reset: Any,
    dock: Any,
) -> Any:
    """Append a toggle action for ``dock`` to a View menu, above the
    Reset Layout separator. Returns the toggle action."""
    toggle = dock.toggleViewAction()
    toggle.setCheckable(True)
    if separator_before_reset is not None:
        menu.insertAction(separator_before_reset, toggle)
    else:
        menu.addAction(toggle)
    return toggle
