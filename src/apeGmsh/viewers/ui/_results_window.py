"""ResultsWindow — Qt shell for the post-solve viewer.

Composes :class:`ViewerWindow` and arranges its central widget +
side docks as the ADR 0088 D1 four-region partition:

::

    ┌──────────────────────────────────────────────────────────────┐
    │ (OS native title bar — filename / minimise / maximise / ×)   │
    ├────────────┬─────────────────────────────┬───────────────────┤
    │ Outline    │                             │ Plots (hidden     │
    │ (dock,     │   3D viewport               │  until 1st plot)  │
    │  left)     │   (central widget)          ├───────────────────┤
    │            │                             │ Inspector         │
    ├────────────┴─────────────────────────────┴───────────────────┤
    │ Time scrubber (dock, bottom — Movable | Floatable, no close) │
    │ (Output console — hidden; full-width UNDER the scrubber)     │
    └──────────────────────────────────────────────────────────────┘

Visible at boot (fresh profile): **Outline, Inspector, Time
scrubber** — three docks. Everything else is on-demand (ADR 0088 D3):
Plots auto-shows on first plot creation; Output is summoned by its
status-bar badge; Display and Definitions live in the View menu;
Section planes rides the toolbar ``section`` glyph. The docks are
movable / floatable / tabifiable :class:`QDockWidget` instances with
stable object names so layout state round-trips through
``QSettings``. The viewport is the immovable central widget.

Layout persists across launches under ``QSettings('apeGmsh',
'ResultsViewer')`` keys ``layout/state`` and ``layout/geometry``.
``reset_layout()`` restores the ADR 0088 D1 boot state.

Public API consumed by :class:`ResultsViewer`:
``plotter``, ``window``, ``set_status``, ``exec``,
``set_left_widget``, ``set_right_widget``, ``set_inspector_widget``,
``set_session_widget``, ``set_bottom_widget``, ``raise_inspector_dock``,
``show_plots_dock``, ``reset_layout``.
"""
from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

from ._dock_registry import (
    DockSpec,
    add_view_menu_toggle,
    build_view_menu,
    mount_dock_spec,
)
from ._layout_metrics import LAYOUT
from .viewer_window import ViewerWindow


class ResultsWindow:
    """Results-viewer-specific window shell.

    Wraps a :class:`ViewerWindow` and arranges its central widget +
    side docks: viewport as the central widget, with Outline (left),
    Plots / Details / Display (right), and Time Scrubber (bottom) as
    movable :class:`QDockWidget` instances.

    Parameters
    ----------
    title
        Window title (shown in the OS title bar).
    on_close
        Optional callback invoked when the window is closed.
    extension_docks
        Optional sequence of :class:`DockSpec` registrations mounted
        alongside the five built-in docks. Useful for the Output
        dock (plan 01) and other panels that want to ride the same
        persistence + View-menu machinery without editing this file.
        ``tabify_with`` / ``split_below`` may reference any built-in
        dock by its ``objectName`` (``dock_results_outline``,
        ``dock_results_right``, ``dock_results_inspector``,
        ``dock_results_session``, ``dock_results_scrubber``) or
        another extension dock_id.
    """

    # Layout schema — bumped whenever the dock set changes (added /
    # removed / renamed). Saved layout state is only restored if the
    # stored version matches; mismatched state is discarded so users
    # don't get a half-broken arrangement after a structural change.
    #
    # v1–v5: pre-audit dock churn, ending in the "stuck outline dock"
    # fixes (see ViewerWindow's history for the sibling bumps).
    # v6 (2026-07): the seven-dock partition — Outline / Plots /
    # Diagram / Geometry / Details / Display / Time scrubber.
    # v7 (2026-08-04): ADR 0088 window architecture — the tab spine
    # consolidates into ONE Inspector dock (``dock_results_inspector``)
    # driven by outline selection; the Color Mapping dock dissolves
    # into the Inspector's diagram context; Plots / Display / Output /
    # Definitions / Section planes go on-demand; Output stacks
    # full-width BELOW the scrubber. Retired objectNames — NEVER to be
    # reused for different content (a stale saved geometry would
    # half-apply): ``dock_results_diagram``, ``dock_results_geometry``,
    # ``dock_results_details``, ``dock_color_map_editor``.
    _LAYOUT_SCHEMA_VERSION = 7

    # objectNames of the five built-in docks — exposed so extension
    # specs can tabify with them by name without reaching into private
    # attributes.
    DOCK_OUTLINE   = "dock_results_outline"
    DOCK_PLOTS     = "dock_results_right"
    DOCK_INSPECTOR = "dock_results_inspector"
    DOCK_SESSION   = "dock_results_session"
    DOCK_SCRUBBER  = "dock_results_scrubber"

    def __init__(
        self,
        *,
        title: str = "Results",
        on_close: Optional[Callable[[], None]] = None,
        extension_docks: Optional[Sequence[DockSpec]] = None,
        central_interactor: bool = True,
    ) -> None:
        # Wrap user-supplied on_close so we always persist layout on shutdown.
        # ViewerWindow's _MainWindow.closeEvent calls on_close *before* tearing
        # down the VTK interactor, so saveState() still has a live window.
        user_on_close = on_close

        def _wrapped_on_close() -> None:
            try:
                self._save_layout()
            except Exception:
                pass
            if user_on_close is not None:
                user_on_close()

        self._vw = ViewerWindow(
            title=title,
            on_close=_wrapped_on_close,
            central_interactor=central_interactor,
        )

        # Populated by _build_layout()
        self._dock_left: Any = None
        self._dock_right: Any = None
        self._dock_inspector: Any = None
        self._dock_session: Any = None
        self._dock_bottom: Any = None
        # Host widgets inside each dock's QScrollArea — content swap target.
        self._left_host: Any = None
        self._right_host: Any = None
        self._inspector_host: Any = None
        self._session_host: Any = None
        self._bottom_host: Any = None
        # Default state captured after layout is built — target for reset_layout.
        self._default_layout_state: Any = None
        self._default_layout_geometry: Any = None
        # Focus-mode snapshot — populated when Ctrl+H hides everything,
        # consumed when Ctrl+H restores. None = currently NOT in focus mode.
        self._focus_state: Any = None
        # Window-level QActions owning the Ctrl+H / Q shortcuts. Menu
        # items reuse the SAME actions so the shortcut renders in the
        # menu without double-registering (ADR 0087 Appendix B).
        self._act_focus_mode: Any = None
        self._act_close_window: Any = None
        # Marker separator before the View → Theme section so later
        # submenu additions (Orbit axis) land between Camera and Theme.
        self._view_menu_theme_separator: Any = None
        # Extension docks: registered via __init__ or add_extension_dock.
        # Keyed by spec.dock_id (== Qt objectName). Stored so
        # add_extension_dock can validate uniqueness and so the View
        # menu can iterate them.
        self._extension_specs: list[DockSpec] = (
            list(extension_docks) if extension_docks else []
        )
        self._extension_docks: dict[str, Any] = {}
        # View menu + handle to the extensions-section separator so
        # add_extension_dock can insert above "Reset Layout".
        self._view_menu: Any = None
        self._view_menu_reset_separator: Any = None

        self._build_layout()
        self._build_view_menu()

    # ------------------------------------------------------------------
    # Public API (forwards / new)
    # ------------------------------------------------------------------

    @property
    def plotter(self):
        """The PyVista QtInteractor (plotter) the toolbar acts on.

        With a pane host installed (ADR 0098 Amendment 1) this follows
        the active pane — see :attr:`ViewerWindow.plotter`.
        """
        return self._vw.plotter

    def set_plotter_provider(self, provider) -> None:
        """Route :attr:`plotter` through ``provider`` (forwarded)."""
        self._vw.set_plotter_provider(provider)

    def set_central_widget(self, widget) -> None:
        """Mount ``widget`` as the central widget (forwarded).

        The ADR 0098 Amendment 1 A1.1 seam: a shell built with
        ``central_interactor=False`` mounts the pane host here.
        """
        self._vw.set_central_widget(widget)

    @property
    def window(self):
        """The underlying QMainWindow."""
        return self._vw.window

    def set_status(self, text: str, timeout: int = 0) -> None:
        self._vw.set_status(text, timeout)

    def exec(self) -> int:
        return self._vw.exec()

    def present(self) -> None:
        """Show + raise + render without entering the event loop.

        Forwards :meth:`ViewerWindow.present` — the ``enter_loop=False``
        path (File → Open results… spawning a second window on an
        already-running loop) calls this on the ResultsWindow shell.
        """
        self._vw.present()

    def on_theme_changed(self, callback) -> None:
        """Register a callback fired with the new ``Palette`` on theme change.

        Forwards to the wrapped :class:`ViewerWindow` so consumers
        (e.g. :class:`ResultsViewer` updating substrate colors) don't
        need to reach into private state.
        """
        self._vw.on_theme_changed(callback)

    def add_toolbar_action(
        self,
        tooltip: str,
        icon_text: str,
        callback,
        *,
        checkable: bool = False,
        triggered_signal: str = "triggered",
        icon: "str | None" = None,
    ):
        """Plan 02 — forward the toolbar extensibility hook.

        Mirrors :meth:`ViewerWindow.add_toolbar_action` so diagrams /
        overlays in ``results.viewer`` can register their own buttons
        without reaching past ``ResultsWindow`` into the wrapped
        ``ViewerWindow``. Returns the ``QAction`` for later removal
        or checked-state updates. ``icon`` names an icon-factory glyph
        (ADR 0087 INV-4) and overrides ``icon_text``.
        """
        return self._vw.add_toolbar_action(
            tooltip, icon_text, callback,
            checkable=checkable,
            triggered_signal=triggered_signal,
            icon=icon,
        )

    def remove_toolbar_action(self, action) -> None:
        """Remove a previously-added toolbar action (forwarded)."""
        self._vw.remove_toolbar_action(action)

    def add_view_menu_submenu(self, title: str):
        """Append a titled submenu to ResultsWindow's OWN View menu.

        NOT forwarded to the wrapped :class:`ViewerWindow`: its lazy
        ``_ensure_view_menu`` sees no menu of its own and would rebuild
        a fresh, empty View menu at the END of the menu bar — dropping
        every dock toggle and landing after Help (ADR 0087 INV-5 wants
        File / View / Help). A separator keeps the submenu reading as
        its own section below the toggles + Reset layout.

        When the built-in Theme section exists, the submenu is inserted
        ABOVE it so viewer-supplied submenus (Orbit axis) land between
        Camera and Theme — the Appendix B ordering.
        """
        if self._view_menu is None:
            return self._vw.add_view_menu_submenu(title)
        from qtpy import QtWidgets
        marker = self._view_menu_theme_separator
        if marker is not None:
            sub = QtWidgets.QMenu(title, self._view_menu)
            self._view_menu.insertMenu(marker, sub)
            self._view_menu.insertSeparator(sub.menuAction())
            return sub
        self._view_menu.addSeparator()
        return self._view_menu.addMenu(title)

    def save_screenshot(self) -> None:
        """File → Save screenshot… — mirrors the toolbar ``save`` action."""
        self._vw._save_screenshot()  # noqa: SLF001 — wrapped window

    @property
    def close_window_action(self):
        """The window-level "Close window" ``QAction`` (shortcut Q).

        Exposed so the viewer's File menu can list the SAME action —
        one action, one shortcut registration, no double-fire.
        """
        return self._act_close_window

    def set_bottom_widget(self, widget) -> None:
        """Mount a widget in the bottom (time-scrubber) dock."""
        self._set_host_widget(self._bottom_host, widget)

    def set_left_widget(self, widget) -> None:
        """Mount a widget in the left (Outline) dock."""
        self._set_host_widget(self._left_host, widget)
        # Hide the dock while empty so it doesn't reserve dead space.
        self._dock_left.setVisible(widget is not None)

    def set_right_widget(self, widget) -> None:
        """Mount a widget in the right-top (Plots) dock.

        Mounting does NOT show the dock — Plots is on-demand (ADR 0088
        D3): it auto-shows on first plot creation via
        :meth:`show_plots_dock`, or through its View-menu toggle.
        """
        self._set_host_widget(self._right_host, widget)

    def set_inspector_widget(self, widget) -> None:
        """Mount the Inspector content (right side, visible at boot)."""
        self._set_host_widget(self._inspector_host, widget)
        self._dock_inspector.setVisible(widget is not None)

    def set_session_widget(self, widget) -> None:
        """Mount a widget in the Display dock.

        Mounting does NOT show the dock — Display is on-demand (ADR
        0088 D3): the View menu summons it; View → Theme covers the
        common case without it.
        """
        self._set_host_widget(self._session_host, widget)

    def raise_inspector_dock(self) -> None:
        """Show + front the Inspector dock (e.g. from the ``colormap``
        toolbar action or a selection that must surface its context)."""
        if self._dock_inspector is not None:
            try:
                self._dock_inspector.show()
                self._dock_inspector.raise_()
            except Exception:
                pass

    def show_plots_dock(self) -> None:
        """Show + front the Plots dock — the first-plot auto-show path
        (ADR 0088 D3). Closing the dock never discards plot data; this
        just makes the pane visible again."""
        if self._dock_right is not None:
            try:
                self._dock_right.show()
                self._dock_right.raise_()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_layout(self) -> None:
        """Replace the wrapped ViewerWindow's central+dock layout.

        - VTK interactor stays where ``ViewerWindow`` put it: as the
          window's central widget.
        - Outline / Plots / Inspector / Display / Time scrubber become
          :class:`QDockWidget` instances on the left, right, and bottom
          dock areas — the ADR 0088 D1 partition.
        - The legacy ``_tabs_dock`` from :class:`ViewerWindow` is removed
          from the layout (its widget stays alive for any external reader).
        """
        from qtpy import QtWidgets, QtCore

        win = self._vw.window

        # ── Bottom area spans the full window width ─────────────────
        # ViewerWindow gives the right column both right corners; the
        # ADR 0088 D1 partition instead runs the Time scrubber (and the
        # Output console beneath it) edge-to-edge, so both bottom
        # corners belong to the bottom area.
        win.setCorner(
            QtCore.Qt.Corner.BottomLeftCorner,
            QtCore.Qt.DockWidgetArea.BottomDockWidgetArea,
        )
        win.setCorner(
            QtCore.Qt.Corner.BottomRightCorner,
            QtCore.Qt.DockWidgetArea.BottomDockWidgetArea,
        )

        # ── Retire the legacy right-side tabs dock ──────────────────
        # ViewerWindow installs its own right-side QDockWidget for tabs.
        # We're claiming that area for the Right Rail dock, so remove
        # the legacy one from the layout. Do not delete — the underlying
        # QTabWidget may still be referenced by other code paths.
        legacy_dock = getattr(self._vw, "_tabs_dock", None)
        if legacy_dock is not None:
            try:
                win.removeDockWidget(legacy_dock)
                legacy_dock.hide()
            except Exception:
                pass

        # ── Tabified-dock UX is owned by ViewerWindow ───────────────
        # ``ViewerWindow.__init__`` (the window we wrap) already sets
        # ``setDockNestingEnabled`` / ``setDockOptions`` /
        # ``setTabPosition(West)`` on this same ``win`` and installs
        # the horizontal-text proxy style + a ChildAdded filter that
        # restyles any QTabBar Qt creates later. Since ViewerWindow's
        # __init__ runs before this ``_build_layout``, the filter is
        # live by the time we tabify docks below — they get the
        # sidebar tab strip automatically. The duplicate block
        # that used to live here was removed (2026-05-16) once
        # ViewerWindow owned the machinery; see PR #179.

        # ── Five docks (ADR 0088 D1) ────────────────────────────────
        QDW = QtWidgets.QDockWidget
        movable_floatable = (
            QDW.DockWidgetFeature.DockWidgetMovable
            | QDW.DockWidgetFeature.DockWidgetFloatable
        )
        with_close = movable_floatable | QDW.DockWidgetFeature.DockWidgetClosable

        self._dock_left, self._left_host = self._make_dock(
            "Outline", "dock_results_outline",
            min_width=LAYOUT.outline_min_width,
            features=with_close,
        )
        self._dock_left.setVisible(False)  # empty until ResultsViewer mounts it

        # Plots — on-demand: hidden until the first plot is created
        # (auto-show via ``show_plots_dock``) or the View menu summons
        # it (ADR 0088 D3).
        self._dock_right, self._right_host = self._make_dock(
            "Plots", "dock_results_right",
            min_width=LAYOUT.right_min_width,
            features=with_close,
        )
        self._dock_right.setVisible(False)

        # Inspector — the outline-selection-driven context host that
        # replaced the Diagram / Geometry / Details tab spine (ADR
        # 0088 D2). Visible at boot once ResultsViewer mounts it.
        self._dock_inspector, self._inspector_host = self._make_dock(
            "Inspector", "dock_results_inspector",
            min_width=LAYOUT.right_min_width,
            features=with_close,
        )
        self._dock_inspector.setVisible(False)  # until mounted

        # Display preferences (theme picker, point size, ...). Titled
        # "Display" per ADR 0087 INV-1/INV-5 — the objectName stays
        # ``dock_results_session`` so persisted QSettings layouts keyed
        # by it keep round-tripping. On-demand: hidden until the View
        # menu summons it (app-level preferences are not properties of
        # a selection — ADR 0088 D2). Tabified behind the Inspector so
        # a summon lands in the right column.
        self._dock_session, self._session_host = self._make_dock(
            "Display", "dock_results_session",
            min_width=LAYOUT.right_min_width,
            features=with_close,
        )
        self._dock_session.setVisible(False)

        # Bottom: time scrubber. NOT closable — losing the playhead is a
        # usability footgun. Floatable so power users can pop it out.
        # "Time scrubber" — sentence case per ADR 0087 INV-5; the
        # objectName stays ``dock_results_scrubber`` so persisted
        # layouts keep round-tripping.
        self._dock_bottom, self._bottom_host = self._make_dock(
            "Time scrubber", "dock_results_scrubber",
            min_height=LAYOUT.scrubber_min_height,
            features=movable_floatable,
        )

        win.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self._dock_left)
        # Right column: Plots on top, Inspector below (the probe→plot
        # workflow wants both visible at once — ADR 0088 D7). Display
        # tabifies behind the Inspector; Section planes (an extension
        # dock) joins the same tab group at mount time.
        win.addDockWidget(QtCore.Qt.RightDockWidgetArea, self._dock_right)
        win.addDockWidget(QtCore.Qt.RightDockWidgetArea, self._dock_inspector)
        win.splitDockWidget(
            self._dock_right, self._dock_inspector, QtCore.Qt.Vertical,
        )
        win.addDockWidget(QtCore.Qt.RightDockWidgetArea, self._dock_session)
        win.tabifyDockWidget(self._dock_inspector, self._dock_session)
        # Keep Inspector as the front tab on first launch.
        self._dock_inspector.raise_()

        win.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self._dock_bottom)

        # ── Extension docks ─────────────────────────────────────────
        # Mounted BEFORE the default-layout snapshot so they participate
        # in saveState/restoreState. Specs are validated for uniqueness
        # against built-in objectNames + each other.
        for spec in self._extension_specs:
            self._mount_extension_dock(spec)

        # Initial extents from LayoutMetrics. At 1280 px the viewport
        # keeps ≥ 50 % of the width with both side columns at initial
        # sizes (1280 − 260 − 380 = 640 — the ADR 0088 D1 invariant).
        try:
            win.resizeDocks(
                [self._dock_left, self._dock_inspector],
                [LAYOUT.outline_initial_width, LAYOUT.right_initial_width],
                QtCore.Qt.Horizontal,
            )
            win.resizeDocks(
                [self._dock_bottom],
                [LAYOUT.scrubber_initial_height],
                QtCore.Qt.Vertical,
            )
        except Exception:
            pass

        # ── Capture default layout, then restore prior user layout ──
        try:
            self._default_layout_state = win.saveState()
            self._default_layout_geometry = win.saveGeometry()
        except Exception:
            self._default_layout_state = None
            self._default_layout_geometry = None
        self._restore_layout()

        # ── Focus-mode shortcut (Ctrl+H) ────────────────────────────
        self._install_focus_shortcut()

    # ------------------------------------------------------------------
    # Extension docks (plan 08 — registry-driven extras alongside the
    # seven built-in docks)
    # ------------------------------------------------------------------

    _BUILTIN_DOCK_IDS = frozenset({
        DOCK_OUTLINE, DOCK_PLOTS, DOCK_INSPECTOR,
        DOCK_SESSION, DOCK_SCRUBBER,
    })

    def add_extension_dock(self, spec: DockSpec) -> Any:
        """Register and mount an additional dock at runtime.

        Used by overlays / plugins / plan-01 output dock to add a
        panel without editing this file. Returns the mounted
        ``QDockWidget``.

        Docks registered this way participate in :meth:`_save_layout`
        on close. To round-trip through :meth:`_restore_layout` on a
        cold start, register the spec via the ``extension_docks``
        constructor argument instead — anything added post-construction
        is restored at its default position on the next launch (the
        saved-state entry for it is still emitted; the dock just
        hasn't been built yet by the time restoreState runs).

        Raises
        ------
        ValueError
            If ``spec.dock_id`` collides with a built-in dock or a
            previously-registered extension, or if
            ``spec.tabify_with`` doesn't resolve.
        """
        dock = self._mount_extension_dock(spec)
        self._extension_specs.append(spec)
        # If the View menu already exists, append a toggle for the new
        # dock. (Will be a no-op if __init__ hasn't finished yet.)
        if self._view_menu is not None:
            self._add_view_menu_toggle(spec.dock_id, dock)
        return dock

    def _mount_extension_dock(self, spec: DockSpec) -> Any:
        """Build a QDockWidget from ``spec`` and dock it onto the window.

        Internal: shared by the constructor's extension-mount loop and
        :meth:`add_extension_dock`. Validates against built-in
        ``objectName`` collisions, then delegates to
        :func:`mount_dock_spec`. Does NOT touch ``_extension_specs``
        or the View menu — callers handle those.
        """
        # Reserve all known ids (built-ins + previously-mounted
        # extensions) so mount_dock_spec rejects collisions with a
        # message that points at this method.
        reserved = self._BUILTIN_DOCK_IDS | set(self._extension_docks)
        if spec.dock_id in self._BUILTIN_DOCK_IDS:
            raise ValueError(
                f"DockSpec.dock_id={spec.dock_id!r} collides with a "
                f"built-in ResultsWindow dock — pick a different id"
            )
        dock = mount_dock_spec(
            self._vw.window, spec, reserved_ids=reserved,
        )
        self._extension_docks[spec.dock_id] = dock
        return dock

    def extension_dock(self, dock_id: str) -> Any:
        """Return the QDockWidget for the given extension ``dock_id``.

        Raises ``KeyError`` if no extension is registered under that id.
        """
        return self._extension_docks[dock_id]

    # ------------------------------------------------------------------
    # View menu (auto-populated toggle actions for every dock + Reset Layout)
    # ------------------------------------------------------------------

    def _all_docks(self) -> list[tuple[str, Any]]:
        """Iterate ``(dock_id, QDockWidget)`` for every dock — built-in
        first in their built-in order, then extensions in registration
        order."""
        builtins = [
            (self.DOCK_OUTLINE,   self._dock_left),
            (self.DOCK_PLOTS,     self._dock_right),
            (self.DOCK_INSPECTOR, self._dock_inspector),
            (self.DOCK_SESSION,   self._dock_session),
            (self.DOCK_SCRUBBER,  self._dock_bottom),
        ]
        out: list[tuple[str, Any]] = [
            (i, d) for (i, d) in builtins if d is not None
        ]
        for spec in self._extension_specs:
            dock = self._extension_docks.get(spec.dock_id)
            if dock is not None:
                out.append((spec.dock_id, dock))
        return out

    def _build_view_menu(self) -> None:
        """Build the View menu — toggle action per dock + Reset Layout.

        Delegates to :func:`build_view_menu`. Idempotent: replaces any
        prior View menu on the window's menu bar. ViewerWindow's own
        conditional view-menu population is not triggered for
        ResultsWindow (no console_dock, no extra_docks), so this method
        owns the View menu exclusively.
        """
        menu_bar = self._vw.window.menuBar()
        if menu_bar is None:
            return
        docks_in_order = [dock for (_id, dock) in self._all_docks()]
        self._view_menu, self._view_menu_reset_separator = build_view_menu(
            menu_bar,
            docks=docks_in_order,
            on_reset_layout=self.reset_layout,
        )

        # ── ADR 0087 Appendix B — Focus mode / Camera / Theme ───────
        # Focus mode reuses the window-level QAction so Ctrl+H renders
        # in the item without a second registration.
        menu = self._view_menu
        menu.addSeparator()
        if self._act_focus_mode is not None:
            menu.addAction(self._act_focus_mode)
        navigation = menu.addMenu("Navigation")
        self._vw.populate_navigation_menu(navigation)
        camera = menu.addMenu("Camera")
        self._vw.populate_camera_menu(camera)
        # Marker separator: viewer-added submenus (Orbit axis) insert
        # above it, keeping Theme the last section.
        self._view_menu_theme_separator = menu.addSeparator()
        theme = menu.addMenu("Theme")
        self._vw.populate_theme_menu(theme)

    def _add_view_menu_toggle(self, dock_id: str, dock: Any) -> None:
        """Append a toggle action for ``dock`` to the View menu.

        Delegates to :func:`add_view_menu_toggle`. Inserts above the
        Reset Layout separator (if present) so Reset Layout stays
        pinned at the bottom.
        """
        if self._view_menu is None:
            return
        add_view_menu_toggle(
            self._view_menu,
            self._view_menu_reset_separator,
            dock,
        )

    def _make_dock(
        self,
        title: str,
        object_name: str,
        *,
        min_width: int = 0,
        min_height: int = 0,
        features,
    ):
        """Build a ``QDockWidget`` whose content is a scrollable host widget.

        Returns ``(dock, host)``. Mount content with
        ``self._set_host_widget(host, widget)``.
        """
        from qtpy import QtWidgets, QtCore

        host = QtWidgets.QWidget()
        host.setObjectName(f"{object_name}_host")
        host_lay = QtWidgets.QVBoxLayout(host)
        host_lay.setContentsMargins(0, 0, 0, 0)
        host_lay.setSpacing(0)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(host)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded,
        )
        scroll.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded,
        )

        dock = QtWidgets.QDockWidget(title)
        dock.setObjectName(object_name)
        dock.setWidget(scroll)
        dock.setAllowedAreas(QtCore.Qt.DockWidgetArea.AllDockWidgetAreas)
        dock.setFeatures(features)
        if min_width:
            dock.setMinimumWidth(min_width)
        if min_height:
            dock.setMinimumHeight(min_height)
        return dock, host

    @staticmethod
    def _set_host_widget(host, widget) -> None:
        """Replace whatever's currently inside ``host`` with ``widget``.

        The widget is added with stretch=1 and an Expanding size policy so
        it grows to fill the dock — without this, panels with default
        ``Preferred`` policy sit at the top with empty space below.
        """
        from qtpy import QtWidgets
        layout = host.layout()
        while layout.count():
            item = layout.takeAt(0)
            old = item.widget()
            if old is not None:
                old.setParent(None)
        if widget is not None:
            try:
                widget.setSizePolicy(
                    QtWidgets.QSizePolicy.Policy.Expanding,
                    QtWidgets.QSizePolicy.Policy.Expanding,
                )
            except Exception:
                pass
            layout.addWidget(widget, stretch=1)

    # ------------------------------------------------------------------
    # Layout persistence
    # ------------------------------------------------------------------

    @staticmethod
    def _layout_settings():
        """Return the QSettings handle used for dock layout persistence."""
        from qtpy.QtCore import QSettings
        return QSettings("apeGmsh", "ResultsViewer")

    def _save_layout(self) -> None:
        """Persist the current dock arrangement and window geometry."""
        win = self._vw.window
        s = self._layout_settings()
        try:
            s.setValue("layout/schema_version", self._LAYOUT_SCHEMA_VERSION)
            s.setValue("layout/geometry", win.saveGeometry())
            s.setValue("layout/state", win.saveState())
        except Exception:
            pass

    def _restore_layout(self) -> None:
        """Restore the user's last dock arrangement, if any.

        Skips restoration if the stored schema version doesn't match the
        current build — applying a v1 state to a v2 layout produces
        broken arrangements (e.g. tabified docks getting un-tabbed).
        """
        win = self._vw.window
        s = self._layout_settings()

        stored_version = s.value("layout/schema_version")
        try:
            stored_version_int = int(stored_version) if stored_version is not None else None
        except (TypeError, ValueError):
            stored_version_int = None
        if stored_version_int != self._LAYOUT_SCHEMA_VERSION:
            # Stored state is from a different layout schema — discard.
            return

        geom = s.value("layout/geometry")
        state = s.value("layout/state")
        if geom is not None:
            try:
                win.restoreGeometry(geom)
            except Exception:
                pass
        if state is not None:
            try:
                win.restoreState(state)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Focus mode (Ctrl+H — hide every dock + left toolbar, then restore)
    # ------------------------------------------------------------------

    def _install_focus_shortcut(self) -> None:
        """Build the Ctrl+H focus-mode and Q close-window ``QAction``\\ s.

        QActions (not bare ``QShortcut``\\ s) so the menu items that list
        them render the shortcut right-aligned without registering the
        key twice (ADR 0087 Appendix B — a coexisting QShortcut would
        make the sequence ambiguous and fire neither). Both are added
        to the window so the shortcuts work even before the menus are
        built. ``Qt.ApplicationShortcut`` because the VTK
        ``QtInteractor`` intercepts keyboard events at native level —
        Qt's default ``WindowShortcut`` would never see the keys while
        the viewport has focus.
        """
        from qtpy import QtGui, QtCore
        win = self._vw.window

        act_focus = QtGui.QAction("Focus mode", win)
        act_focus.setShortcut(QtGui.QKeySequence("Ctrl+H"))
        act_focus.setShortcutContext(
            QtCore.Qt.ShortcutContext.ApplicationShortcut,
        )
        act_focus.triggered.connect(self.toggle_focus_mode)
        win.addAction(act_focus)
        self._act_focus_mode = act_focus

        # Bare-Q closes the window. Bare keys are aggressive — a Spin/
        # LineEdit being focused would still receive the key as input
        # under ApplicationShortcut, so the handler guards with a
        # focus check.
        act_close = QtGui.QAction("Close window", win)
        act_close.setShortcut(QtGui.QKeySequence("Q"))
        act_close.setShortcutContext(
            QtCore.Qt.ShortcutContext.ApplicationShortcut,
        )
        act_close.triggered.connect(self._on_q_pressed)
        win.addAction(act_close)
        self._act_close_window = act_close

    def _on_q_pressed(self) -> None:
        """Close the window — but skip if the user is typing in a field."""
        from qtpy import QtWidgets
        fw = QtWidgets.QApplication.focusWidget()
        if isinstance(fw, (
            QtWidgets.QLineEdit,
            QtWidgets.QSpinBox,
            QtWidgets.QDoubleSpinBox,
            QtWidgets.QPlainTextEdit,
            QtWidgets.QTextEdit,
            QtWidgets.QComboBox,
        )):
            return
        try:
            self._vw.window.close()
        except Exception:
            pass

    def toggle_focus_mode(self) -> None:
        """Focus mode: viewport + Time scrubber + HUDs, or restore.

        ADR 0088 D6 — reviewing an animation is the focus-mode use
        case, so the Time scrubber STAYS (the everything-hidden
        variant lost the playhead). Every other dock — built-in and
        extension — plus the left toolbar hides. Viewport HUDs are
        viewport-owned and unaffected. On first call, snapshots which
        docks / toolbar were visible and hides them. On second call,
        restores exactly that snapshot — so a user who manually hid
        (e.g.) the Display dock before entering focus mode comes back
        to the same arrangement.
        """
        docks = [
            d for (dock_id, d) in self._all_docks()
            if dock_id != self.DOCK_SCRUBBER
        ]
        toolbar = getattr(self._vw, "_toolbar", None)

        if self._focus_state is None:
            # Currently in normal mode — capture state, hide everything
            # except the scrubber.
            self._focus_state = {
                "docks": [d for d in docks if d is not None and d.isVisible()],
                "toolbar_visible": (
                    toolbar.isVisible() if toolbar is not None else False
                ),
            }
            for d in docks:
                if d is not None:
                    d.setVisible(False)
            if toolbar is not None:
                toolbar.setVisible(False)
            try:
                self.set_status("Focus mode — Ctrl+H to restore", 3000)
            except Exception:
                pass
        else:
            # Restore the snapshot.
            for d in self._focus_state["docks"]:
                if d is not None:
                    d.setVisible(True)
            if toolbar is not None:
                toolbar.setVisible(self._focus_state["toolbar_visible"])
            self._focus_state = None
            try:
                self.set_status("Focus mode off", 2000)
            except Exception:
                pass

    def reset_layout(self) -> None:
        """Restore the ADR 0088 D1 boot state from any arrangement.

        Visible set exactly {Outline, Inspector, Time scrubber} at
        initial sizes; the on-demand set (Plots, Display, every
        extension dock) hidden — not whatever the user's first-launch
        restore produced (ADR 0088 D6).
        """
        from qtpy import QtCore
        win = self._vw.window
        # Leaving focus mode implicitly — a stale snapshot would
        # otherwise re-hide docks on the next Ctrl+H.
        self._focus_state = None
        if self._default_layout_state is not None:
            try:
                win.restoreState(self._default_layout_state)
            except Exception:
                pass
        boot_visible = {
            self.DOCK_OUTLINE, self.DOCK_INSPECTOR, self.DOCK_SCRUBBER,
        }
        for dock_id, dock in self._all_docks():
            if dock is not None:
                dock.setVisible(dock_id in boot_visible)
        toolbar = getattr(self._vw, "_toolbar", None)
        if toolbar is not None:
            toolbar.setVisible(True)
        # Re-assert initial extents — the captured default state may
        # predate the ResultsViewer mounting content into the docks.
        try:
            win.resizeDocks(
                [self._dock_left, self._dock_inspector],
                [LAYOUT.outline_initial_width, LAYOUT.right_initial_width],
                QtCore.Qt.Horizontal,
            )
            win.resizeDocks(
                [self._dock_bottom],
                [LAYOUT.scrubber_initial_height],
                QtCore.Qt.Vertical,
            )
        except Exception:
            pass
        try:
            self.set_status("Layout reset to default", 3000)
        except Exception:
            pass

