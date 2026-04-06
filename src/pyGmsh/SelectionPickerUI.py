"""
SelectionPickerUI
=================

Qt (pyvistaqt) front-end for :class:`SelectionPicker`.

Provides the interactive 3D GUI:

* **Center**: ``pyvistaqt.QtInteractor`` hosting the VTK scene (same
  scene the headless ``SelectionPicker`` core builds — same picking,
  same hover, same box-select).

* **Right dock — Model tree**: a ``QTreeWidget`` with two roots:
  *Physical groups* (each group's members listed as leaves) and
  *Unassigned* (entities not in any group, bucketed by dim).  Clicking
  an entity leaf toggles its pick; clicking a group header activates
  that group so subsequent picks commit to it.

* **Right dock — Preferences**: session-only sliders / checkboxes for
  point size, line width, surface opacity, edges, anti-aliasing and
  theme.  Every change applies live; nothing is persisted.

* **Toolbar**: New / Rename / Delete group, ⊥ Parallel ↔ perspective
  toggle, Fit view.

The window is blocking — ``SelectionPicker.show()`` calls
``SelectionPickerWindow(...).exec()`` and waits.  On close, the picker
core flushes every staged group to Gmsh.

Dependencies (all optional, lazy-imported): ``pyvistaqt`` + one of
``PyQt5`` / ``PyQt6`` / ``PySide2`` / ``PySide6`` (qtpy resolves the
binding).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import gmsh

if TYPE_CHECKING:
    from pyGmsh.SelectionPicker import SelectionPicker


DimTag = tuple[int, int]


def _lazy_qt():
    """Import Qt + pyvistaqt on first use.  Raises a clear ImportError
    if the deps aren't installed."""
    try:
        from qtpy import QtWidgets, QtCore, QtGui    # noqa: F401
        from pyvistaqt import QtInteractor           # noqa: F401
    except ImportError as err:
        raise ImportError(
            "SelectionPicker's Qt UI requires pyvistaqt + a Qt binding "
            "(PyQt5 / PyQt6 / PySide2 / PySide6). Install with "
            "`pip install pyvistaqt PyQt5`."
        ) from err
    return QtWidgets, QtCore, QtGui, QtInteractor


_DIM_NAMES = {0: "Points", 1: "Curves", 2: "Surfaces", 3: "Volumes"}
_DIM_ABBR  = {0: "P", 1: "C", 2: "S", 3: "V"}


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class SelectionPickerWindow:
    """QMainWindow hosting the 3D viewport + model tree + prefs dock.

    Not a direct ``QMainWindow`` subclass because Qt classes must be
    defined after the ``QApplication`` exists.  Instead, this wrapper
    holds an instance of a runtime-created ``_PickerMainWindow``
    subclass, and forwards ``.exec()``.
    """

    def __init__(
        self,
        picker: "SelectionPicker",
        *,
        title: str = "SelectionPicker",
        maximized: bool = True,
    ) -> None:
        QtWidgets, QtCore, QtGui, QtInteractor = _lazy_qt()

        # Ensure a QApplication exists
        self._app = QtWidgets.QApplication.instance()
        if self._app is None:
            self._app = QtWidgets.QApplication([])

        self._picker = picker
        self._QtWidgets = QtWidgets
        self._QtCore = QtCore
        self._QtGui = QtGui
        self._title = title
        self._maximized = maximized

        # Fast lookups from DimTag / group-name → tree item
        self._tree_items_by_dt: dict[DimTag, QtWidgets.QTreeWidgetItem] = {}
        self._tree_items_by_group: dict[str, QtWidgets.QTreeWidgetItem] = {}

        # Parallel projection state (mirrors QtInteractor.camera)
        self._parallel = False

        # Re-entry guard to avoid pick ↔ tree feedback loops
        self._syncing = False

        # Build the actual QMainWindow
        self._window = self._build_window(QtInteractor)

        # Wire picker → UI callbacks
        picker._on_pick_changed.append(self._refresh_tree_picks)
        picker._on_pick_changed.append(self._log_pick_changed)
        picker._on_visibility_changed.append(self._refresh_tree_visibility)
        picker._on_visibility_changed.append(self._sync_dim_checkboxes)
        picker._on_hover_changed.append(self._refresh_info)

    # ------------------------------------------------------------------
    # Exec / teardown
    # ------------------------------------------------------------------

    def exec(self) -> int:
        """Show the window and run the Qt event loop until closed."""
        # Re-apply the dark gradient background right before the window
        # becomes visible — on second-open, pyvista's QtInteractor init
        # can paint a frame with the default theme before _setup_on
        # runs, and the GL context may not honour set_background until
        # the widget is actually shown.
        from pyGmsh.SelectionPicker import _BG_TOP, _BG_BOTTOM
        try:
            self._qt_interactor.set_background(_BG_TOP, top=_BG_BOTTOM)
        except Exception:
            pass
        if self._maximized:
            self._window.showMaximized()
        else:
            self._window.show()
        # One more render after the window is visible to guarantee
        # the GL context has the correct background.
        try:
            self._qt_interactor.render()
        except Exception:
            pass
        return self._app.exec_()

    # ------------------------------------------------------------------
    # QMainWindow construction
    # ------------------------------------------------------------------

    def _build_window(self, QtInteractor):
        QtWidgets = self._QtWidgets
        QtCore = self._QtCore

        ui_self = self

        class _PickerMainWindow(QtWidgets.QMainWindow):
            def closeEvent(self, event):
                # Commit the active group's picks to staging BEFORE the
                # window closes — so _flush_staged_to_gmsh() sees them.
                ui_self._picker._commit_active_group()
                # Properly shut down the QtInteractor:
                #   - stops the auto-render timer (prevents wglMakeCurrent
                #     errors from firing against a destroyed GL context)
                #   - calls BasePlotter.close (releases VTK resources)
                #   - nulls _RenderWindow and renderer
                # This MUST happen while the window is still alive —
                # doing it after super().closeEvent destroys the widget
                # first and leaves VTK dangling.
                try:
                    ui_self._qt_interactor.close()
                except Exception:
                    pass
                # Let picker clean up any VTK references it holds
                ui_self._picker._plotter = None
                super().closeEvent(event)

        window = _PickerMainWindow()
        window.setWindowTitle(self._title)
        window.resize(1600, 1000)

        # Central: QtInteractor (wraps a VTK render window)
        self._qt_interactor = QtInteractor(parent=window)
        window.setCentralWidget(self._qt_interactor.interactor)

        # Picker core configures the VTK plotter (scene, picking, hover)
        self._picker._setup_on(self._qt_interactor)

        # ---- Right panel: tabs (Preferences last) ----
        self._tabs = QtWidgets.QTabWidget()
        self._tabs.addTab(self._build_browser_tab(), "Browser")
        self._tabs.addTab(self._build_view_tab(), "View")
        self._tabs.addTab(self._build_filter_tab(), "Filter")
        self._tabs.addTab(self._build_prefs_tab(), "Preferences")

        tabs_dock = QtWidgets.QDockWidget("Panel")
        tabs_dock.setMinimumWidth(320)
        tabs_dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetMovable
            | QtWidgets.QDockWidget.DockWidgetFloatable
        )
        tabs_dock.setWidget(self._tabs)
        window.addDockWidget(QtCore.Qt.RightDockWidgetArea, tabs_dock)

        # Physical Groups dock — stacked below tabs in the right area.
        # Contains the groups list + name input + action buttons
        # (merged from the old "Groups" tab).
        pg_dock = self._build_committed_groups_dock()
        window.addDockWidget(QtCore.Qt.RightDockWidgetArea, pg_dock)

        # ---- Bottom docks: Console + Entity Info (collapsible) ----
        console_dock = QtWidgets.QDockWidget("Console")
        console_dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetMovable
            | QtWidgets.QDockWidget.DockWidgetFloatable
            | QtWidgets.QDockWidget.DockWidgetClosable
        )
        console_dock.setWidget(self._build_console_tab())
        window.addDockWidget(QtCore.Qt.BottomDockWidgetArea, console_dock)
        self._console_dock = console_dock

        info_dock = QtWidgets.QDockWidget("Entity Info")
        info_dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetMovable
            | QtWidgets.QDockWidget.DockWidgetFloatable
            | QtWidgets.QDockWidget.DockWidgetClosable
        )
        info_dock.setWidget(self._build_info_tab())
        window.addDockWidget(QtCore.Qt.BottomDockWidgetArea, info_dock)
        self._info_dock = info_dock

        # Tabify the bottom docks so they share space with a tab bar
        window.tabifyDockWidget(console_dock, info_dock)
        # Start collapsed (hidden) — user opens via View menu
        console_dock.hide()
        info_dock.hide()

        # Right-side docks take priority over bottom docks —
        # the right column spans the full window height, and the
        # bottom docks only span under the viewport.
        window.setCorner(
            QtCore.Qt.BottomRightCorner, QtCore.Qt.RightDockWidgetArea,
        )
        window.setCorner(
            QtCore.Qt.TopRightCorner, QtCore.Qt.RightDockWidgetArea,
        )

        # Floating toolbar — vertical, on the left, unattached to any
        # dock area so it floats over the viewport (Blender-style).
        toolbar = self._build_toolbar()
        toolbar.setOrientation(QtCore.Qt.Vertical)
        toolbar.setFloatable(True)
        window.addToolBar(QtCore.Qt.LeftToolBarArea, toolbar)

        # View menu — toggle bottom docks back on after closing
        menu_bar = window.menuBar()
        view_menu = menu_bar.addMenu("View")
        view_menu.addAction(console_dock.toggleViewAction())
        view_menu.addAction(info_dock.toggleViewAction())

        # Status bar
        self._statusbar = window.statusBar()
        self._refresh_statusbar()

        # Initial tree populate
        self._populate_tree()

        # Keyboard shortcuts (Qt-level, so they work even when VTK
        # doesn't have focus — e.g. Tab, which Qt normally eats for
        # widget focus-cycling).
        QtGui = self._QtGui
        # Q → close window
        sc_q = QtWidgets.QShortcut(QtGui.QKeySequence("Q"), window)
        sc_q.activated.connect(window.close)
        # Esc → deselect all picks (clear working set)
        sc_esc = QtWidgets.QShortcut(QtGui.QKeySequence("Esc"), window)
        sc_esc.activated.connect(self._action_deselect_all)
        # Tab → cycle overlapping entities (Revit-style).
        # Must be a Qt shortcut — Qt intercepts Tab for focus-cycling
        # before VTK's key-event observer ever sees it.
        sc_tab = QtWidgets.QShortcut(
            QtGui.QKeySequence(QtCore.Qt.Key_Tab), window,
        )
        sc_tab.setContext(QtCore.Qt.ApplicationShortcut)
        sc_tab.activated.connect(self._picker._cycle_pick)

        return window

    # ------------------------------------------------------------------
    # Model tree dock
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Bottom dock — committed physical groups
    # ------------------------------------------------------------------

    def _build_committed_groups_dock(self):
        """Right-side dock showing physical groups + name input + action
        buttons.  Combines the old 'Groups' tab and the committed-list
        into one always-visible panel below the tab widget."""
        QtWidgets = self._QtWidgets
        QtCore = self._QtCore

        dock = QtWidgets.QDockWidget("Physical Groups")
        dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetMovable
            | QtWidgets.QDockWidget.DockWidgetFloatable
        )

        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Group list
        self._committed_list = QtWidgets.QListWidget()
        self._committed_list.setAlternatingRowColors(True)
        self._committed_list.currentTextChanged.connect(
            self._on_committed_list_selected,
        )
        layout.addWidget(self._committed_list)

        # Name input
        name_row = QtWidgets.QHBoxLayout()
        name_row.addWidget(QtWidgets.QLabel("Name:"))
        self._group_name_edit = QtWidgets.QLineEdit()
        self._group_name_edit.setPlaceholderText("group name…")
        name_row.addWidget(self._group_name_edit)
        layout.addLayout(name_row)

        # Action buttons — two rows
        row1 = QtWidgets.QHBoxLayout()
        row1.setSpacing(4)

        btn_apply = QtWidgets.QPushButton("Apply")
        btn_apply.setToolTip("Create / overwrite group with current picks")
        btn_apply.clicked.connect(self._grp_apply)
        row1.addWidget(btn_apply)

        btn_modify = QtWidgets.QPushButton("Modify")
        btn_modify.setToolTip("Load group members into picks for editing")
        btn_modify.clicked.connect(self._committed_modify)
        row1.addWidget(btn_modify)
        layout.addLayout(row1)

        row2 = QtWidgets.QHBoxLayout()
        row2.setSpacing(4)

        btn_select = QtWidgets.QPushButton("Select")
        btn_select.setToolTip("Add group members to current picks")
        btn_select.clicked.connect(self._committed_select)
        row2.addWidget(btn_select)

        btn_deselect = QtWidgets.QPushButton("Deselect")
        btn_deselect.setToolTip("Clear all current picks")
        btn_deselect.clicked.connect(self._action_deselect_all)
        row2.addWidget(btn_deselect)

        btn_delete = QtWidgets.QPushButton("Delete")
        btn_delete.setToolTip("Delete this group")
        btn_delete.clicked.connect(self._committed_delete)
        row2.addWidget(btn_delete)
        layout.addLayout(row2)

        dock.setWidget(container)
        return dock

    def _on_committed_list_selected(self, text: str) -> None:
        """Populate name field when clicking a group in the list."""
        if not text:
            return
        # Strip the " (N Surfaces)" suffix
        name = text.split("  (")[0]
        self._group_name_edit.setText(name)

    def _refresh_committed_table(self) -> None:
        """Rebuild the physical-groups list from Gmsh + staged."""
        if not hasattr(self, "_committed_list"):
            return
        lst = self._committed_list
        cur = lst.currentItem()
        cur_name = cur.data(self._QtCore.Qt.UserRole) if cur else ""
        lst.clear()
        groups = self._collect_groups()
        for gname in sorted(groups.keys()):
            members = groups[gname]
            if not members:
                continue
            n = len(members)
            dims = sorted(set(d for d, _ in members))
            dim_str = ", ".join(
                _DIM_NAMES.get(d, str(d)) for d in dims
            )
            item = self._QtWidgets.QListWidgetItem(
                f"{gname}  ({n} {dim_str})"
            )
            item.setData(self._QtCore.Qt.UserRole, gname)
            lst.addItem(item)
        # Restore selection
        for i in range(lst.count()):
            if lst.item(i).data(self._QtCore.Qt.UserRole) == cur_name:
                lst.setCurrentRow(i)
                break

    def _committed_selected_name(self) -> str | None:
        """Return the group name from the currently-selected list item."""
        lst = self._committed_list
        cur = lst.currentItem()
        if cur is None:
            self._statusbar.showMessage(
                "Select a group in the Physical Groups panel first.", 3000,
            )
            return None
        return cur.data(self._QtCore.Qt.UserRole)

    def _committed_select(self) -> None:
        name = self._committed_selected_name()
        if not name:
            return
        from pyGmsh.SelectionPicker import _load_physical_group_members
        members = self._picker._staged_groups.get(name)
        if members is None:
            members = _load_physical_group_members(name)
        if members:
            self._picker.select_dimtags(members, replace=False)
            self.log(f"Selected {len(members)} entities from '{name}'")

    def _committed_modify(self) -> None:
        name = self._committed_selected_name()
        if not name:
            return
        self._picker.set_active_group(name)
        self._group_name_edit.setText(name)
        self._populate_tree()
        self._refresh_statusbar()
        self.log(f"Editing group '{name}'")

    def _committed_delete(self) -> None:
        name = self._committed_selected_name()
        if not name:
            return
        self._delete_group(name)
        self._refresh_committed_table()

    # ------------------------------------------------------------------
    # Tabs
    # ------------------------------------------------------------------

    def _build_browser_tab(self):
        """Tab 1 — Project Browser (model tree)."""
        QtWidgets = self._QtWidgets
        QtCore = self._QtCore

        tree = QtWidgets.QTreeWidget()
        tree.setHeaderLabels(["Entity / group", "Info"])
        tree.setColumnWidth(0, 200)
        tree.setAlternatingRowColors(True)
        tree.setRootIsDecorated(True)
        tree.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)

        tree.itemClicked.connect(self._on_tree_item_clicked)
        tree.customContextMenuRequested.connect(self._on_tree_context_menu)

        self._tree = tree
        return tree

    def _build_view_tab(self):
        """View tab — toggle entity labels/tags on screen."""
        QtWidgets = self._QtWidgets
        QtCore = self._QtCore

        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        group = QtWidgets.QGroupBox("Show entity labels on screen")
        group_layout = QtWidgets.QVBoxLayout(group)

        self._view_label_cbs: dict[int, QtWidgets.QCheckBox] = {}
        for dim, name in _DIM_NAMES.items():
            cb = QtWidgets.QCheckBox(f"{name} tags")
            cb.setChecked(False)
            cb.toggled.connect(self._on_view_labels_changed)
            group_layout.addWidget(cb)
            self._view_label_cbs[dim] = cb

        layout.addWidget(group)

        # Label style options
        style_group = QtWidgets.QGroupBox("Label style")
        style_layout = QtWidgets.QFormLayout(style_group)
        style_layout.setSpacing(4)

        self._label_size_spin = QtWidgets.QSpinBox()
        self._label_size_spin.setRange(6, 24)
        self._label_size_spin.setValue(10)
        self._label_size_spin.valueChanged.connect(self._on_view_labels_changed)
        style_layout.addRow("Font size", self._label_size_spin)

        self._label_use_names = QtWidgets.QCheckBox("Show names instead of tags")
        self._label_use_names.setChecked(False)
        self._label_use_names.toggled.connect(self._on_view_labels_changed)
        style_layout.addRow(self._label_use_names)

        layout.addWidget(style_group)
        layout.addStretch(1)

        # Track label actors so we can remove them
        self._view_label_actors: list = []

        return panel

    def _on_view_labels_changed(self) -> None:
        """Add or remove entity label overlays in the 3D viewport."""
        import gmsh
        import numpy as np
        import pyvista as pv

        plotter = self._picker._plotter
        if plotter is None:
            return

        # Remove existing label actors
        for actor in self._view_label_actors:
            try:
                plotter.remove_actor(actor)
            except Exception:
                pass
        self._view_label_actors.clear()

        registry = self._picker._model._registry
        use_names = self._label_use_names.isChecked()
        font_size = self._label_size_spin.value()

        # Read font prefs (available after prefs tab is built)
        if hasattr(self, "_font_combo"):
            font_family = self._font_combo.currentFont().family()
        else:
            font_family = "Arial"
        if hasattr(self, "_font_size_spin"):
            font_size = self._font_size_spin.value()
        if hasattr(self, "_font_color"):
            text_color = self._font_color
        else:
            text_color = "white"

        for dim, cb in self._view_label_cbs.items():
            if not cb.isChecked():
                continue
            points = []
            labels = []
            for _, tag in gmsh.model.getEntities(dim=dim):
                try:
                    bb = gmsh.model.getBoundingBox(dim, tag)
                    cx = (bb[0] + bb[3]) * 0.5
                    cy = (bb[1] + bb[4]) * 0.5
                    cz = (bb[2] + bb[5]) * 0.5
                    points.append([cx, cy, cz])
                except Exception:
                    continue
                if use_names:
                    info = registry.get((dim, tag), {})
                    lbl = info.get("label", f"{_DIM_ABBR[dim]}{tag}")
                else:
                    lbl = f"{_DIM_ABBR[dim]}{tag}"
                labels.append(lbl)

            if not points:
                continue

            pts = np.array(points)
            try:
                actor = plotter.add_point_labels(
                    pts, labels,
                    font_size=font_size,
                    font_family=font_family,
                    text_color=text_color,
                    shape_color="#333333",
                    shape_opacity=0.6,
                    show_points=False,
                    always_visible=True,
                    name=f"_view_labels_dim{dim}",
                )
                self._view_label_actors.append(
                    f"_view_labels_dim{dim}"
                )
            except Exception:
                pass

        plotter.render()

    def _build_groups_tab(self):
        """Tab 2 — Physical Groups management.

        Layout:
          ┌─────────────────────────┐
          │ Existing groups (list)  │
          │  ▸ base_supports  (3)   │
          │  ▸ columns       (12)   │
          ├─────────────────────────┤
          │ Group name: [________]  │
          │ [Apply] [Modify]        │
          │ [Select] [Deselect]     │
          │ [Delete]                │
          └─────────────────────────┘
        """
        QtWidgets = self._QtWidgets
        QtCore = self._QtCore

        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # ---- Existing groups list ----
        layout.addWidget(QtWidgets.QLabel("Existing groups:"))
        self._group_list = QtWidgets.QListWidget()
        self._group_list.setAlternatingRowColors(True)
        self._group_list.currentTextChanged.connect(
            self._on_group_list_selected,
        )
        layout.addWidget(self._group_list)

        # ---- Group name input ----
        name_row = QtWidgets.QHBoxLayout()
        name_row.addWidget(QtWidgets.QLabel("Name:"))
        self._group_name_edit = QtWidgets.QLineEdit()
        self._group_name_edit.setPlaceholderText("physical group name…")
        name_row.addWidget(self._group_name_edit)
        layout.addLayout(name_row)

        # ---- Action buttons ----
        btn_grid = QtWidgets.QGridLayout()
        btn_grid.setSpacing(4)

        btn_apply = QtWidgets.QPushButton("Apply")
        btn_apply.setToolTip(
            "Create / overwrite the group with current picks"
        )
        btn_apply.clicked.connect(self._grp_apply)
        btn_grid.addWidget(btn_apply, 0, 0)

        btn_modify = QtWidgets.QPushButton("Modify")
        btn_modify.setToolTip(
            "Load the group's members into picks for editing"
        )
        btn_modify.clicked.connect(self._grp_modify)
        btn_grid.addWidget(btn_modify, 0, 1)

        btn_select = QtWidgets.QPushButton("Select")
        btn_select.setToolTip(
            "Add all members of the group to the current picks"
        )
        btn_select.clicked.connect(self._grp_select)
        btn_grid.addWidget(btn_select, 1, 0)

        btn_deselect = QtWidgets.QPushButton("Deselect")
        btn_deselect.setToolTip("Clear all current picks")
        btn_deselect.clicked.connect(self._action_deselect_all)
        btn_grid.addWidget(btn_deselect, 1, 1)

        btn_delete = QtWidgets.QPushButton("Delete")
        btn_delete.setToolTip("Delete the named group")
        btn_delete.clicked.connect(self._grp_delete)
        btn_grid.addWidget(btn_delete, 2, 0, 1, 2)

        layout.addLayout(btn_grid)
        layout.addStretch(1)
        return panel

    # ---- Groups-tab helpers ----

    def _refresh_group_list(self) -> None:
        """Repopulate the group list widget from Gmsh + staged groups."""
        lst = self._committed_list
        lst.blockSignals(True)
        cur = lst.currentItem()
        cur_text = cur.text().split("  (")[0] if cur else ""
        lst.clear()
        groups = self._collect_groups()
        for gname in sorted(groups.keys()):
            n = len(groups[gname])
            lst.addItem(f"{gname}  ({n})")
        # Restore selection
        for i in range(lst.count()):
            if lst.item(i).text().startswith(cur_text + "  ("):
                lst.setCurrentRow(i)
                break
        lst.blockSignals(False)

    def _on_group_list_selected(self, text: str) -> None:
        """When user clicks a group in the list, populate the name field."""
        if not text:
            return
        name = text.split("  (")[0]
        self._group_name_edit.setText(name)

    def _grp_name(self) -> str:
        return self._group_name_edit.text().strip()

    def _grp_apply(self) -> None:
        """Apply: stage current picks as the named group."""
        name = self._grp_name()
        if not name:
            self._statusbar.showMessage("Enter a group name first.", 3000)
            return
        if self._group_name_taken(name):
            if not self._confirm_overwrite(name):
                return
        self._picker._staged_groups[name] = list(self._picker._picks)
        self._picker._active_group = name
        self._populate_tree()
        self._refresh_group_list()
        self._refresh_statusbar()
        self.log(f"Applied group '{name}' ({len(self._picker._picks)} entities)")

    def _grp_modify(self) -> None:
        """Modify: load the named group's members into picks."""
        name = self._grp_name()
        if not name:
            self._statusbar.showMessage("Enter a group name first.", 3000)
            return
        self._picker.set_active_group(name)
        self._populate_tree()
        self._refresh_group_list()
        self._refresh_statusbar()
        self.log(f"Editing group '{name}' ({len(self._picker._picks)} members loaded)")

    def _grp_select(self) -> None:
        """Select: add all members of the named group to picks."""
        name = self._grp_name()
        if not name:
            return
        from pyGmsh.SelectionPicker import _load_physical_group_members
        # Check staged first, then Gmsh
        members = self._picker._staged_groups.get(name)
        if members is None:
            members = _load_physical_group_members(name)
        if not members:
            self._statusbar.showMessage(
                f"Group '{name}' is empty or does not exist.", 3000,
            )
            return
        self._picker.select_dimtags(members, replace=False)
        self.log(f"Selected {len(members)} entities from '{name}'")

    def _grp_delete(self) -> None:
        """Delete: remove the named group."""
        name = self._grp_name()
        if not name:
            return
        self._delete_group(name)
        self._refresh_group_list()

    def _collect_groups(self) -> dict[str, list[DimTag]]:
        """Union of Gmsh physical groups and session-staged groups.
        Staged entries override Gmsh entries on name collision.  The
        active group is always shown with its current working picks."""
        groups: dict[str, list[DimTag]] = {}

        # From Gmsh
        for d, pg_tag in gmsh.model.getPhysicalGroups():
            try:
                name = gmsh.model.getPhysicalName(d, pg_tag)
            except Exception:
                name = ""
            if not name:
                name = f"(unnamed:dim={d},tag={pg_tag})"
            bucket = groups.setdefault(name, [])
            try:
                for t in gmsh.model.getEntitiesForPhysicalGroup(d, pg_tag):
                    bucket.append((d, int(t)))
            except Exception:
                pass

        # Overlay staged groups (explicit edits this session)
        for name, members in self._picker._staged_groups.items():
            groups[name] = list(members)

        # Active group always reflects the *current* working picks
        if self._picker._active_group is not None:
            groups[self._picker._active_group] = list(self._picker._picks)

        return groups

    def _populate_tree(self) -> None:
        QtWidgets = self._QtWidgets
        QtCore = self._QtCore

        self._tree.clear()
        self._tree_items_by_dt.clear()
        self._tree_items_by_group.clear()

        groups = self._collect_groups()
        assigned: set[DimTag] = {dt for mems in groups.values() for dt in mems}
        active = self._picker._active_group
        registry = self._picker._model._registry

        # ---- Physical groups root ----
        pg_root = QtWidgets.QTreeWidgetItem(["Physical groups", f"{len(groups)}"])
        pg_root.setFont(0, _bold(QtWidgets))
        self._tree.addTopLevelItem(pg_root)
        pg_root.setExpanded(True)

        for gname in sorted(groups.keys()):
            members = groups[gname]
            header = QtWidgets.QTreeWidgetItem(
                [_group_label(gname, active, len(members)),
                 f"{len(members)} entities"]
            )
            header.setData(0, QtCore.Qt.UserRole, ("group", gname))
            if gname == active:
                header.setFont(0, _bold(QtWidgets))
            pg_root.addChild(header)
            self._tree_items_by_group[gname] = header

            for dt in members:
                leaf = self._make_entity_leaf(dt, registry)
                header.addChild(leaf)

        # ---- Unassigned root ----
        unassigned: dict[int, list[DimTag]] = {0: [], 1: [], 2: [], 3: []}
        for dim in (0, 1, 2, 3):
            for _, tag in gmsh.model.getEntities(dim=dim):
                dt = (dim, int(tag))
                if dt not in assigned:
                    unassigned[dim].append(dt)
        total_unassigned = sum(len(v) for v in unassigned.values())

        u_root = QtWidgets.QTreeWidgetItem(
            ["Unassigned", f"{total_unassigned}"]
        )
        u_root.setFont(0, _bold(QtWidgets))
        self._tree.addTopLevelItem(u_root)
        u_root.setExpanded(False)

        for dim in (0, 1, 2, 3):
            members = unassigned[dim]
            if not members:
                continue
            header = QtWidgets.QTreeWidgetItem(
                [_DIM_NAMES[dim], f"{len(members)}"]
            )
            header.setData(0, QtCore.Qt.UserRole, ("dim-bucket", dim))
            u_root.addChild(header)
            for dt in members:
                leaf = self._make_entity_leaf(dt, registry)
                header.addChild(leaf)

        # ---- Instances root (Assembly context only) ----
        self._build_instances_root()

        # Apply pick coloring in the fresh tree
        self._refresh_tree_picks()

        # Keep the Groups tab list in sync
        if hasattr(self, "_group_list"):
            self._refresh_group_list()
        # Keep the bottom-dock table in sync
        self._refresh_committed_table()

    def _build_instances_root(self) -> None:
        """Append an 'Instances' root showing each Assembly instance and
        its entities bucketed by dim.  No-op when the picker's parent is
        not an Assembly (Part / standalone pyGmsh have no ``instances``
        attribute).  Read-only view — the tree offers single-click batch
        selection and a right-click "create group from instance" action.
        """
        parent = self._picker._parent
        if not hasattr(parent, "instances"):
            return
        instances = parent.instances
        if not instances:
            return

        QtWidgets = self._QtWidgets
        QtCore = self._QtCore

        i_root = QtWidgets.QTreeWidgetItem(
            ["Instances", f"{len(instances)}"]
        )
        i_root.setFont(0, _bold(QtWidgets))
        self._tree.addTopLevelItem(i_root)
        i_root.setExpanded(True)

        for label, inst in instances.items():
            total = sum(len(v) for v in inst.entities.values())
            header = QtWidgets.QTreeWidgetItem(
                [f"{label}  ← Part '{inst.part_name}'",
                 f"{total} entities"]
            )
            header.setData(
                0, QtCore.Qt.UserRole, ("instance", label),
            )
            i_root.addChild(header)
            for dim in (0, 1, 2, 3):
                tags = inst.entities.get(dim) or []
                if not tags:
                    continue
                bucket = QtWidgets.QTreeWidgetItem(
                    [_DIM_NAMES[dim], f"{len(tags)}"]
                )
                bucket.setData(
                    0, QtCore.Qt.UserRole,
                    ("instance-dim", label, dim),
                )
                header.addChild(bucket)

    def _make_entity_leaf(self, dt: DimTag, registry: dict):
        QtWidgets = self._QtWidgets
        QtCore = self._QtCore
        dim, tag = dt
        info = registry.get(dt, {})
        label = info.get("label", f"{_DIM_ABBR[dim]}{tag}")
        kind = info.get("kind", "")
        leaf = QtWidgets.QTreeWidgetItem(
            [label, f"dim={dim} tag={tag}  {kind}"]
        )
        leaf.setData(0, QtCore.Qt.UserRole, ("entity", dim, tag))
        self._tree_items_by_dt[dt] = leaf
        return leaf

    # ------------------------------------------------------------------
    # Tree interactions
    # ------------------------------------------------------------------

    def _on_tree_item_clicked(self, item, _column: int) -> None:
        QtCore = self._QtCore
        if self._syncing:
            return
        data = item.data(0, QtCore.Qt.UserRole)
        if data is None:
            return
        kind = data[0]
        if kind == "entity":
            dt = (int(data[1]), int(data[2]))
            # Entities are only toggleable when there's an active group
            if self._picker._active_group is None:
                self._statusbar.showMessage(
                    "No active group — click or create a group first.", 4000,
                )
                return
            self._picker._toggle_pick(dt)
        elif kind == "group":
            gname = data[1]
            self._picker.set_active_group(gname)
            self._populate_tree()
            self._refresh_statusbar()
        elif kind == "instance":
            label = data[1]
            dts = self._instance_dimtags(label)
            self._picker.select_dimtags(dts, replace=True)
        elif kind == "instance-dim":
            label, dim = data[1], int(data[2])
            dts = self._instance_dimtags(label, dim=dim)
            self._picker.select_dimtags(dts, replace=True)

    def _instance_dimtags(
        self, label: str, *, dim: int | None = None,
    ) -> list[DimTag]:
        """Return the list of DimTags for one Assembly instance.
        Restricts to a single dim if *dim* is given."""
        parent = self._picker._parent
        if not hasattr(parent, "instances"):
            return []
        inst = parent.instances.get(label)
        if inst is None:
            return []
        out: list[DimTag] = []
        for d, tags in inst.entities.items():
            if dim is not None and d != dim:
                continue
            for t in tags:
                out.append((int(d), int(t)))
        return out

    def _on_tree_context_menu(self, pos) -> None:
        QtWidgets = self._QtWidgets
        QtCore = self._QtCore
        item = self._tree.itemAt(pos)
        if item is None:
            return
        data = item.data(0, QtCore.Qt.UserRole)
        if data is None:
            return
        kind = data[0]

        if kind == "group":
            gname = data[1]
            menu = QtWidgets.QMenu(self._tree)
            act_active = menu.addAction("Set as active")
            act_rename = menu.addAction("Rename…")
            act_delete = menu.addAction("Delete")
            chosen = menu.exec_(self._tree.mapToGlobal(pos))
            if chosen == act_active:
                self._picker.set_active_group(gname)
                self._populate_tree()
                self._refresh_statusbar()
            elif chosen == act_rename:
                self._rename_group(gname)
            elif chosen == act_delete:
                self._delete_group(gname)
            return

        if kind in ("instance", "instance-dim"):
            if kind == "instance":
                label = data[1]
                dts = self._instance_dimtags(label)
                suggested = label
            else:
                label, dim = data[1], int(data[2])
                dts = self._instance_dimtags(label, dim=dim)
                suggested = f"{label}_{_DIM_NAMES[dim].lower()}"
            menu = QtWidgets.QMenu(self._tree)
            act_select = menu.addAction("Select entities")
            act_group = menu.addAction("Create physical group from instance…")
            chosen = menu.exec_(self._tree.mapToGlobal(pos))
            if chosen == act_select:
                self._picker.select_dimtags(dts, replace=True)
            elif chosen == act_group:
                self._create_group_from_dimtags(dts, suggested=suggested)
            return

    def _create_group_from_dimtags(
        self, dts: list[DimTag], *, suggested: str,
    ) -> None:
        """Prompt for a group name, stage it, and refresh the tree."""
        QtWidgets = self._QtWidgets
        if not dts:
            self._statusbar.showMessage(
                "Instance has no entities in the current scene.", 3000,
            )
            return
        name, ok = QtWidgets.QInputDialog.getText(
            self._window, "New physical group", "Group name:",
            text=suggested,
        )
        if not ok or not name.strip():
            return
        name = name.strip()
        if self._group_name_taken(name):
            if not self._confirm_overwrite(name):
                return
        # Stage the group and make it active
        self._picker._staged_groups[name] = list(dts)
        self._picker.set_active_group(name)
        self._populate_tree()
        self._refresh_statusbar()

    # ------------------------------------------------------------------
    # Tree refresh (bidirectional sync)
    # ------------------------------------------------------------------

    def _refresh_tree_picks(self) -> None:
        """Highlight leaves whose DimTag is in the current pick set.
        Fired by picker when picks change."""
        QtGui = self._QtGui
        self._syncing = True
        try:
            picks = set(self._picker._picks)
            pick_brush = QtGui.QBrush(QtGui.QColor(self._picker._pick_color))
            idle_brush = QtGui.QBrush()  # default
            active = self._picker._active_group
            # Update the active group's leaves (the ones that can toggle)
            if active is not None and active in self._tree_items_by_group:
                header = self._tree_items_by_group[active]
                # Rebuild the active group's visible member count
                header.setText(1, f"{len(picks)} entities")
                # Note: the leaves inside the header only reflect the
                # *initial* membership; after a pick change, the new
                # entity may live in the Unassigned tree.  Repopulating
                # the tree on every pick would be noisy, so we just
                # recolor any known leaves and defer full refresh.
            for dt, leaf in self._tree_items_by_dt.items():
                leaf.setForeground(
                    0, pick_brush if dt in picks else idle_brush,
                )
            self._refresh_statusbar()
        finally:
            self._syncing = False

    def _refresh_tree_visibility(self) -> None:
        """Dim leaves whose DimTag is in the hidden set.  Fired by the
        picker's H / I / R keybindings."""
        QtGui = self._QtGui
        hidden = self._picker._hidden
        dim_brush = QtGui.QBrush(QtGui.QColor("#777777"))
        normal_brush = QtGui.QBrush()
        for dt, leaf in self._tree_items_by_dt.items():
            leaf.setForeground(
                1, dim_brush if dt in hidden else normal_brush,
            )

    # ------------------------------------------------------------------
    # Toolbar actions
    # ------------------------------------------------------------------

    def _make_icon(self, char: str, color: str = "#cccccc") -> "QtGui.QIcon":
        """Paint a single Unicode character into a 24x24 QPixmap and
        return it as a QIcon.  Guarantees the symbol renders even if
        the platform font lacks the glyph (falls back to Segoe UI
        Symbol / Symbola / system emoji font)."""
        QtGui = self._QtGui
        QtCore = self._QtCore
        size = 24
        pix = QtGui.QPixmap(size, size)
        pix.fill(QtGui.QColor(0, 0, 0, 0))  # transparent bg
        painter = QtGui.QPainter(pix)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        # Try fonts likely to have good symbol coverage.
        # inFont() only accepts a single char — for multi-char labels
        # (e.g. "Bo", "Bk") just use the default font directly.
        font = QtGui.QFont("Arial", 14)
        if len(char) == 1:
            for family in ("Segoe UI Symbol", "Segoe UI Emoji",
                           "Symbola", "Noto Sans Symbols", "Arial"):
                f = QtGui.QFont(family, 14)
                if QtGui.QFontMetrics(f).inFont(char):
                    font = f
                    break
        # Shrink font for multi-char labels so they fit in the icon
        if len(char) > 1:
            font.setPointSize(10)
        painter.setFont(font)
        painter.setPen(QtGui.QColor(color))
        painter.drawText(
            QtCore.QRect(0, 0, size, size),
            QtCore.Qt.AlignCenter, char,
        )
        painter.end()
        return QtGui.QIcon(pix)

    def _build_toolbar(self):
        QtWidgets = self._QtWidgets
        QtCore = self._QtCore

        bar = QtWidgets.QToolBar("Tools")
        bar.setMovable(True)
        bar.setIconSize(QtCore.QSize(28, 28))
        bar.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        bar.setStyleSheet(
            "QToolBar { spacing: 2px; }"
            "QToolButton { border: none; border-radius: 4px;"
            "  padding: 4px; margin: 1px; }"
            "QToolButton:hover { background: rgba(255,255,255,30); }"
            "QToolButton:pressed { background: rgba(255,255,255,50); }"
            "QToolButton:checked { background: rgba(100,180,255,60);"
            "  border: 1px solid rgba(100,180,255,120); }"
        )

        # ---- Group operations ----
        act_new = bar.addAction(self._make_icon("+", "#66bb6a"), "")
        act_new.setToolTip("New physical group")
        act_new.triggered.connect(self._action_new_group)

        act_rename = bar.addAction(self._make_icon("\u270F", "#ffa726"), "")
        act_rename.setToolTip("Rename active group")
        act_rename.triggered.connect(self._action_rename_active)

        act_delete = bar.addAction(self._make_icon("\u2212", "#ef5350"), "")
        act_delete.setToolTip("Delete active group")
        act_delete.triggered.connect(self._action_delete_active)

        bar.addSeparator()

        # ---- Visibility ----
        act_hide = bar.addAction(self._make_icon("\u25CB", "#90a4ae"), "")
        act_hide.setToolTip("Hide selected  [H]")
        act_hide.triggered.connect(self._picker._hide_selected)

        act_isolate = bar.addAction(self._make_icon("\u25CE", "#42a5f5"), "")
        act_isolate.setToolTip("Isolate selected  [I]")
        act_isolate.triggered.connect(self._picker._isolate_selected)

        act_show = bar.addAction(self._make_icon("\u25C9", "#66bb6a"), "")
        act_show.setToolTip("Show all  [R]")
        act_show.triggered.connect(self._picker._show_all)

        bar.addSeparator()

        # ---- Camera ----
        self._act_parallel = bar.addAction(
            self._make_icon("\u2316", "#b0bec5"), "")
        self._act_parallel.setToolTip("Ortho / perspective  toggle")
        self._act_parallel.setCheckable(True)
        self._act_parallel.toggled.connect(self._action_toggle_parallel)

        act_fit = bar.addAction(self._make_icon("\u2922", "#b0bec5"), "")
        act_fit.setToolTip("Fit view")
        act_fit.triggered.connect(self._action_fit_view)

        bar.addSeparator()

        # ---- Standard views ----
        act_top = bar.addAction(self._make_icon("T", "#42a5f5"), "")
        act_top.setToolTip("Top view  (Z+)")
        act_top.triggered.connect(lambda: self._snap_view("top"))

        act_bottom = bar.addAction(self._make_icon("Bo", "#42a5f5"), "")
        act_bottom.setToolTip("Bottom view  (Z-)")
        act_bottom.triggered.connect(lambda: self._snap_view("bottom"))

        act_front = bar.addAction(self._make_icon("F", "#66bb6a"), "")
        act_front.setToolTip("Front view  (Y-)")
        act_front.triggered.connect(lambda: self._snap_view("front"))

        act_back = bar.addAction(self._make_icon("Bk", "#66bb6a"), "")
        act_back.setToolTip("Back view  (Y+)")
        act_back.triggered.connect(lambda: self._snap_view("back"))

        act_left = bar.addAction(self._make_icon("L", "#ef5350"), "")
        act_left.setToolTip("Left view  (X-)")
        act_left.triggered.connect(lambda: self._snap_view("left"))

        act_right = bar.addAction(self._make_icon("R", "#ef5350"), "")
        act_right.setToolTip("Right view  (X+)")
        act_right.triggered.connect(lambda: self._snap_view("right"))

        act_iso = bar.addAction(self._make_icon("\u25E3", "#ffa726"), "")
        act_iso.setToolTip("Isometric view")
        act_iso.triggered.connect(lambda: self._snap_view("iso"))

        bar.addSeparator()

        # ---- Screenshot ----
        act_screenshot = bar.addAction(
            self._make_icon("\u2399", "#ce93d8"), "")
        act_screenshot.setToolTip("Save screenshot")
        act_screenshot.triggered.connect(self._action_screenshot)

        bar.addSeparator()

        act_help = bar.addAction(self._make_icon("?", "#ffffff"), "")
        act_help.setToolTip("Shortcuts help")
        act_help.triggered.connect(self._action_show_help)

        return bar

    def _group_name_taken(self, name: str) -> bool:
        """True if *name* already refers to a group — either staged in
        this session with non-empty members, or present in Gmsh."""
        staged = self._picker._staged_groups
        # Staged groups with empty members are "scheduled for deletion"
        # and do not count as taken.
        if name in staged and staged[name]:
            return True
        from pyGmsh.SelectionPicker import _load_physical_group_members
        if _load_physical_group_members(name):
            return True
        return False

    def _confirm_overwrite(self, name: str) -> bool:
        """Prompt before overwriting an existing group.  Returns True if
        the user approved, False if they cancelled."""
        QtWidgets = self._QtWidgets
        reply = QtWidgets.QMessageBox.question(
            self._window,
            "Group name already exists",
            f"A physical group called '{name}' already exists.\n"
            f"Overwrite its members?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel,
            QtWidgets.QMessageBox.Cancel,
        )
        return reply == QtWidgets.QMessageBox.Yes

    def _action_new_group(self) -> None:
        QtWidgets = self._QtWidgets
        name, ok = QtWidgets.QInputDialog.getText(
            self._window, "New physical group", "Group name:",
        )
        if not ok or not name.strip():
            return
        name = name.strip()
        if self._group_name_taken(name):
            if not self._confirm_overwrite(name):
                return
        self._picker.set_active_group(name)
        self._populate_tree()
        self._refresh_statusbar()
        self.log(f"Created group '{name}'")

    def _action_rename_active(self) -> None:
        active = self._picker._active_group
        if not active:
            self._statusbar.showMessage("No active group to rename.", 3000)
            return
        self._rename_group(active)

    def _action_delete_active(self) -> None:
        active = self._picker._active_group
        if not active:
            self._statusbar.showMessage("No active group to delete.", 3000)
            return
        self._delete_group(active)

    def _rename_group(self, old_name: str) -> None:
        QtWidgets = self._QtWidgets
        new_name, ok = QtWidgets.QInputDialog.getText(
            self._window, "Rename group", "New name:", text=old_name,
        )
        if not ok:
            return
        new_name = new_name.strip()
        if not new_name or new_name == old_name:
            return
        # Guard: refuse to silently overwrite an existing group
        if self._group_name_taken(new_name):
            if not self._confirm_overwrite(new_name):
                return
        # Move staged members; update active pointer
        staged = self._picker._staged_groups
        members = staged.pop(old_name, None)
        if self._picker._active_group == old_name:
            # Stage current picks under new name
            staged[new_name] = list(self._picker._picks)
            self._picker._active_group = new_name
        elif members is not None:
            staged[new_name] = members
        else:
            # Pull members from Gmsh into staging under the new name
            from pyGmsh.SelectionPicker import _load_physical_group_members
            staged[new_name] = _load_physical_group_members(old_name)
        # Mark old name as empty so _flush_staged deletes it on close
        staged[old_name] = []
        self._populate_tree()
        self._refresh_statusbar()
        self.log(f"Renamed '{old_name}' -> '{new_name}'")

    def _delete_group(self, name: str) -> None:
        # Stage empty → deleted on close
        self._picker._staged_groups[name] = []
        if self._picker._active_group == name:
            self._picker._active_group = None
            self._picker._picks = []
            self._picker._pick_history = []
            self._picker._recolor_all()
        self._populate_tree()
        self.log(f"Deleted group '{name}'")
        self._refresh_statusbar()

    def _action_toggle_parallel(self, checked: bool) -> None:
        self._parallel = bool(checked)
        try:
            if checked:
                self._qt_interactor.enable_parallel_projection()
            else:
                self._qt_interactor.disable_parallel_projection()
        except Exception:
            # Fallback via camera property
            try:
                self._qt_interactor.camera.parallel_projection = bool(checked)
            except Exception:
                pass
        self._qt_interactor.render()

    def _action_fit_view(self) -> None:
        try:
            self._qt_interactor.reset_camera()
        except Exception:
            pass
        self._qt_interactor.render()

    def _log_pick_changed(self) -> None:
        """Console-log the pick state after each change."""
        n = len(self._picker._picks)
        if n == 0:
            self.log("Selection cleared")
        else:
            last = self._picker._picks[-1] if self._picker._picks else None
            self.log(f"Picks: {n} total (last: {last})")

    def _action_deselect_all(self) -> None:
        """Esc — clear the entire working set (deselect all picks)."""
        if not self._picker._picks:
            return
        old = list(self._picker._picks)
        self._picker._picks.clear()
        self._picker._pick_history.clear()
        for dt in old:
            self._picker._recolor(dt)
        self._picker._update_status()
        self._picker._fire_pick_changed()

    def _snap_view(self, direction: str) -> None:
        """Snap the camera to a standard orthogonal or isometric view."""
        try:
            p = self._qt_interactor
            views = {
                "top":    (lambda: p.view_xy(negative=False)),
                "bottom": (lambda: p.view_xy(negative=True)),
                "front":  (lambda: p.view_xz(negative=False)),
                "back":   (lambda: p.view_xz(negative=True)),
                "right":  (lambda: p.view_yz(negative=False)),
                "left":   (lambda: p.view_yz(negative=True)),
                "iso":    (lambda: p.view_isometric()),
            }
            views[direction]()
            p.reset_camera()
            p.render()
        except Exception:
            pass

    def _action_screenshot(self) -> None:
        """Save a screenshot of the 3D viewport to a user-chosen file."""
        QtWidgets = self._QtWidgets
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self._window, "Save screenshot", "screenshot.png",
            "PNG (*.png);;JPEG (*.jpg);;All Files (*)",
        )
        if not path:
            return
        try:
            self._qt_interactor.screenshot(path)
            self.log(f"Screenshot saved: {path}")
            self._statusbar.showMessage(f"Screenshot saved: {path}", 4000)
        except Exception as exc:
            self._statusbar.showMessage(f"Screenshot failed: {exc}", 4000)

    def _action_show_help(self) -> None:
        """Show a modeless dialog listing all keyboard / mouse shortcuts."""
        QtWidgets = self._QtWidgets
        QtCore = self._QtCore
        dlg = QtWidgets.QDialog(self._window)
        dlg.setWindowTitle("SelectionPicker — shortcuts")
        dlg.setModal(False)
        dlg.resize(520, 460)

        layout = QtWidgets.QVBoxLayout(dlg)
        text = QtWidgets.QTextBrowser()
        text.setOpenExternalLinks(False)
        text.setHtml(_SHORTCUTS_HTML)
        layout.addWidget(text)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch(1)
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(dlg.close)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

        dlg.show()
        # Keep a reference so the dialog isn't garbage-collected
        self._help_dialog = dlg

    # ------------------------------------------------------------------
    # Prefs dock
    # ------------------------------------------------------------------

    def _build_prefs_tab(self):
        """Tab 2 — Preferences (visual settings)."""
        QtWidgets = self._QtWidgets
        QtCore = self._QtCore
        QtGui = self._QtGui

        panel = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(panel)
        form.setContentsMargins(8, 8, 8, 8)
        form.setSpacing(6)

        picker = self._picker

        # Point size
        self._s_point = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._s_point.setRange(1, 50)
        self._s_point.setValue(int(picker._point_size))
        self._s_point.valueChanged.connect(self._on_point_size_changed)
        form.addRow("Point size", self._s_point)

        # Line width
        self._s_line = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._s_line.setRange(1, 20)
        self._s_line.setValue(int(picker._line_width))
        self._s_line.valueChanged.connect(self._on_line_width_changed)
        form.addRow("Line width", self._s_line)

        # Surface opacity (0..100 mapping 0.0..1.0)
        self._s_alpha = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._s_alpha.setRange(0, 100)
        self._s_alpha.setValue(int(picker._surface_opacity * 100))
        self._s_alpha.valueChanged.connect(self._on_opacity_changed)
        form.addRow("Surface α", self._s_alpha)

        # Show edges
        self._cb_edges = QtWidgets.QCheckBox("Show surface edges")
        self._cb_edges.setChecked(picker._show_surface_edges)
        self._cb_edges.toggled.connect(self._on_edges_toggled)
        form.addRow(self._cb_edges)

        # Anti-aliasing
        self._cb_aa = QtWidgets.QCheckBox("Anti-aliasing (SSAA)")
        self._cb_aa.setChecked(True)
        self._cb_aa.toggled.connect(self._on_aa_toggled)
        form.addRow(self._cb_aa)

        # Drag threshold (pixels before click becomes a box-select)
        self._s_drag_thresh = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._s_drag_thresh.setRange(1, 20)
        self._s_drag_thresh.setValue(picker._drag_threshold)
        self._s_drag_thresh.valueChanged.connect(self._on_drag_thresh_changed)
        form.addRow("Drag threshold (px)", self._s_drag_thresh)

        # Selection color
        self._color_btn = QtWidgets.QPushButton()
        self._color_btn.setFixedSize(60, 24)
        self._update_color_btn_style(picker._pick_color)
        self._color_btn.clicked.connect(self._on_pick_color_clicked)
        form.addRow("Selection color", self._color_btn)

        # ---- Font settings ----
        form.addRow(QtWidgets.QLabel(""))  # spacer

        # Font family
        self._font_combo = QtWidgets.QFontComboBox()
        self._font_combo.setCurrentFont(QtGui.QFont("Arial"))
        self._font_combo.currentFontChanged.connect(self._on_font_changed)
        form.addRow("Label font", self._font_combo)

        # Font size
        self._font_size_spin = QtWidgets.QSpinBox()
        self._font_size_spin.setRange(6, 30)
        self._font_size_spin.setValue(10)
        self._font_size_spin.valueChanged.connect(self._on_font_changed)
        form.addRow("Label font size", self._font_size_spin)

        # Font color
        self._font_color_btn = QtWidgets.QPushButton()
        self._font_color_btn.setFixedSize(60, 24)
        self._font_color = "#ffffff"
        self._font_color_btn.setStyleSheet(
            f"background-color: {self._font_color}; border: 1px solid #888;"
        )
        self._font_color_btn.clicked.connect(self._on_font_color_clicked)
        form.addRow("Label font color", self._font_color_btn)

        # ---- Theme ----
        form.addRow(QtWidgets.QLabel(""))  # spacer

        self._theme_combo = QtWidgets.QComboBox()
        self._theme_combo.addItems(["Dark", "Light"])
        self._theme_combo.currentTextChanged.connect(self._on_theme_changed)
        form.addRow("Theme", self._theme_combo)

        return panel

    def _on_point_size_changed(self, value: int) -> None:
        """Point-spheres are built at unit ``base_r`` with the initial
        ``point_size`` applied via ``SetScale`` — so the slider value is
        used directly here, keeping the displayed size in sync with the
        slider position regardless of the initial ``point_size``."""
        scale = float(value)
        self._picker._point_size = scale
        for (d, _), actor in self._picker._dimtag_to_actor.items():
            if d != 0:
                continue
            try:
                actor.SetScale(scale, scale, scale)
            except Exception:
                pass
        self._qt_interactor.render()

    def _on_line_width_changed(self, value: int) -> None:
        self._picker._line_width = float(value)
        for (d, _), actor in self._picker._dimtag_to_actor.items():
            if d != 1:
                continue
            try:
                actor.GetProperty().SetLineWidth(float(value))
            except Exception:
                pass
        self._qt_interactor.render()

    def _on_opacity_changed(self, value: int) -> None:
        """Apply the slider alpha to dim=2 / dim=3 actors — but skip any
        actor whose dim is currently filtered out (so the 'dimmed for
        pick filter' visual is preserved).  Uses the ``_pickable_dims``
        state directly instead of a magic-number opacity sentinel."""
        alpha = float(value) / 100.0
        self._picker._surface_opacity = alpha
        vol_alpha = max(0.05, alpha * 0.6)
        pickable_dims = self._picker._pickable_dims
        for (d, _), actor in self._picker._dimtag_to_actor.items():
            if d not in (2, 3):
                continue
            if d not in pickable_dims:
                continue   # dim is filtered out — keep dimmed
            try:
                actor.GetProperty().SetOpacity(
                    alpha if d == 2 else vol_alpha,
                )
            except Exception:
                pass
        self._qt_interactor.render()

    def _on_edges_toggled(self, checked: bool) -> None:
        self._picker._show_surface_edges = bool(checked)
        for (d, _), actor in self._picker._dimtag_to_actor.items():
            if d != 2:
                continue
            try:
                actor.GetProperty().SetEdgeVisibility(bool(checked))
            except Exception:
                pass
        self._qt_interactor.render()

    def _on_aa_toggled(self, checked: bool) -> None:
        try:
            if checked:
                self._qt_interactor.enable_anti_aliasing("ssaa")
            else:
                self._qt_interactor.disable_anti_aliasing()
        except Exception:
            pass
        self._qt_interactor.render()

    def _on_drag_thresh_changed(self, value: int) -> None:
        self._picker._drag_threshold = value

    def _update_color_btn_style(self, hex_color: str) -> None:
        """Set the button's background to the current pick colour."""
        self._color_btn.setStyleSheet(
            f"background-color: {hex_color}; border: 1px solid #888;"
        )
        self._color_btn.setToolTip(hex_color)

    def _on_pick_color_clicked(self) -> None:
        """Open a QColorDialog and apply the chosen colour to all
        currently-picked entities + future picks."""
        QtWidgets = self._QtWidgets
        QtGui = self._QtGui
        cur = QtGui.QColor(self._picker._pick_color)
        color = QtWidgets.QColorDialog.getColor(
            cur, self._window, "Selection highlight color",
        )
        if not color.isValid():
            return
        hex_color = color.name()   # "#RRGGBB"
        self._picker._pick_color = hex_color
        self._update_color_btn_style(hex_color)
        # Re-colour every currently-picked actor with the new colour
        self._picker._recolor_all()
        # Refresh tree pick-brush too
        self._refresh_tree_picks()

    def _on_theme_changed(self, value: str) -> None:
        if value == "Light":
            self._qt_interactor.set_background("#eeeeee", top="#ffffff")
        else:
            # Use the dark palette from SelectionPicker's module constants
            from pyGmsh.SelectionPicker import _BG_TOP, _BG_BOTTOM
            self._qt_interactor.set_background(_BG_TOP, top=_BG_BOTTOM)
        self._qt_interactor.render()

    def _on_font_changed(self) -> None:
        """Re-render entity labels in the viewport when font changes."""
        if hasattr(self, "_view_label_cbs"):
            self._on_view_labels_changed()

    def _on_font_color_clicked(self) -> None:
        """Open a color dialog for the label font color."""
        QtWidgets = self._QtWidgets
        QtGui = self._QtGui
        cur = QtGui.QColor(self._font_color)
        color = QtWidgets.QColorDialog.getColor(
            cur, self._window, "Label font color",
        )
        if not color.isValid():
            return
        self._font_color = color.name()
        self._font_color_btn.setStyleSheet(
            f"background-color: {self._font_color}; border: 1px solid #888;"
        )
        # Re-render labels with new color
        if hasattr(self, "_view_label_cbs"):
            self._on_view_labels_changed()

    # ------------------------------------------------------------------
    # Tab 3 — Console
    # ------------------------------------------------------------------

    def _build_console_tab(self):
        """Tab 3 — Console (read-only event log)."""
        QtWidgets = self._QtWidgets
        QtGui = self._QtGui

        console = QtWidgets.QTextEdit()
        console.setReadOnly(True)
        console.setFont(QtGui.QFont("Consolas", 9))
        console.setStyleSheet(
            "QTextEdit { background: #1e1e1e; color: #d4d4d4; }"
        )
        self._console = console
        return console

    def log(self, msg: str) -> None:
        """Append a timestamped line to the Console tab."""
        import datetime
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self._console.append(f"[{ts}] {msg}")
        # Auto-scroll to bottom
        sb = self._console.verticalScrollBar()
        sb.setValue(sb.maximum())

    # ------------------------------------------------------------------
    # Tab 4 — Entity Info
    # ------------------------------------------------------------------

    def _build_info_tab(self):
        """Entity Info — BRep topology tree for the hovered entity.
        Shows the entity's topological hierarchy: its boundary children
        recursively (Volume→Surfaces→Curves→Points)."""
        QtWidgets = self._QtWidgets

        self._info_tree = QtWidgets.QTreeWidget()
        self._info_tree.setHeaderLabels(["Entity", "Details"])
        self._info_tree.setColumnWidth(0, 180)
        self._info_tree.setAlternatingRowColors(True)
        self._info_tree.setRootIsDecorated(True)
        return self._info_tree

    def _refresh_info(self) -> None:
        """Rebuild the BRep tree for the hovered entity."""
        import gmsh
        QtWidgets = self._QtWidgets
        QtCore = self._QtCore

        self._info_tree.clear()
        dt = self._picker._hover_dt
        if dt is None:
            item = QtWidgets.QTreeWidgetItem(["(hover over an entity)"])
            self._info_tree.addTopLevelItem(item)
            return

        registry = self._picker._model._registry

        def _make_node(dim: int, tag: int, parent_item=None):
            """Create a tree item for (dim, tag) and recurse into
            its boundary children."""
            info = registry.get((dim, tag), {})
            label = info.get("label", f"{_DIM_ABBR[dim]}{tag}")
            kind = info.get("kind", "")

            # Details column
            details = f"dim={dim}  tag={tag}"
            try:
                bb = gmsh.model.getBoundingBox(dim, tag)
                details += (
                    f"  bbox=({bb[0]:.1f},{bb[1]:.1f},{bb[2]:.1f})"
                    f"→({bb[3]:.1f},{bb[4]:.1f},{bb[5]:.1f})"
                )
            except Exception:
                pass
            if kind:
                details += f"  {kind}"

            # Pick state badge
            if (dim, tag) in self._picker._picks:
                label += "  [PICKED]"
            elif (dim, tag) in self._picker._hidden:
                label += "  [hidden]"

            node = QtWidgets.QTreeWidgetItem([label, details])
            if parent_item is None:
                self._info_tree.addTopLevelItem(node)
            else:
                parent_item.addChild(node)

            # Recurse into boundary (one level down)
            if dim > 0:
                try:
                    boundary = gmsh.model.getBoundary(
                        [(dim, tag)], combined=False,
                        oriented=False, recursive=False,
                    )
                    seen = set()
                    for bd, bt in boundary:
                        if (bd, bt) not in seen:
                            seen.add((bd, bt))
                            _make_node(bd, bt, parent_item=node)
                except Exception:
                    pass
            return node

        dim, tag = dt
        root = _make_node(dim, tag)

        # Physical groups this entity belongs to
        try:
            groups = []
            for d, pg in gmsh.model.getPhysicalGroups():
                if d != dim:
                    continue
                ents = gmsh.model.getEntitiesForPhysicalGroup(d, pg)
                if tag in ents:
                    name = gmsh.model.getPhysicalName(d, pg) or f"pg={pg}"
                    groups.append(name)
            if groups:
                pg_node = QtWidgets.QTreeWidgetItem(
                    ["Groups", ", ".join(groups)]
                )
                root.addChild(pg_node)
        except Exception:
            pass

        # Instance info
        parent = self._picker._parent
        if hasattr(parent, "instances"):
            for lbl, inst in parent.instances.items():
                if tag in inst.entities.get(dim, []):
                    inst_node = QtWidgets.QTreeWidgetItem(
                        ["Instance", f"{lbl} (Part '{inst.part_name}')"]
                    )
                    root.addChild(inst_node)
                    break

        root.setExpanded(True)

        self._info_panel.setPlainText("\n".join(lines))

    # ------------------------------------------------------------------
    # Tab 5 — Selection Filter
    # ------------------------------------------------------------------

    def _build_filter_tab(self):
        """Tab 5 — Selection Filter (dim checkboxes + name filter)."""
        QtWidgets = self._QtWidgets
        QtCore = self._QtCore

        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Dimension filter
        dim_group = QtWidgets.QGroupBox("Dimension filter")
        dim_layout = QtWidgets.QVBoxLayout(dim_group)
        self._dim_cbs = {}
        pickable = self._picker._pickable_dims
        for d, name in _DIM_NAMES.items():
            cb = QtWidgets.QCheckBox(name)
            cb.setChecked(d in pickable)
            cb.toggled.connect(self._on_dim_filter_changed)
            dim_layout.addWidget(cb)
            self._dim_cbs[d] = cb
        layout.addWidget(dim_group)

        # Entity labels filter (checkboxes for each label found in registry)
        label_group = QtWidgets.QGroupBox("Entity labels")
        label_layout = QtWidgets.QVBoxLayout(label_group)
        self._label_cbs: dict[str, QtWidgets.QCheckBox] = {}
        # Collect unique labels from the model registry
        registry = self._picker._model._registry
        labels: set[str] = set()
        for dt, info in registry.items():
            lbl = info.get("label", "")
            if lbl:
                labels.add(lbl)
        if labels:
            for lbl in sorted(labels):
                cb = QtWidgets.QCheckBox(lbl)
                cb.setChecked(True)
                cb.toggled.connect(self._on_label_filter_changed)
                label_layout.addWidget(cb)
                self._label_cbs[lbl] = cb
        else:
            label_layout.addWidget(
                QtWidgets.QLabel("(no labelled entities)")
            )
        # Scroll area in case there are many labels
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(label_group)
        scroll.setMaximumHeight(200)
        layout.addWidget(scroll)

        # Name / text filter
        name_group = QtWidgets.QGroupBox("Name filter")
        name_layout = QtWidgets.QVBoxLayout(name_group)
        self._name_filter = QtWidgets.QLineEdit()
        self._name_filter.setPlaceholderText("Type to filter entities…")
        self._name_filter.textChanged.connect(self._on_name_filter_changed)
        name_layout.addWidget(self._name_filter)
        layout.addWidget(name_group)

        layout.addStretch(1)
        return panel

    def _on_dim_filter_changed(self) -> None:
        """Sync dim-filter checkboxes → picker._set_pickable_dims."""
        dims = set()
        for d, cb in self._dim_cbs.items():
            if cb.isChecked():
                dims.add(d)
        if not dims:
            dims = set(self._picker._dims)  # at least one
        self._picker._set_pickable_dims(dims)
        self._populate_tree()

    def _on_label_filter_changed(self) -> None:
        """Show/hide entities in the 3D view and tree based on which
        label checkboxes are checked."""
        if not hasattr(self, "_label_cbs"):
            return
        checked_labels = {
            lbl for lbl, cb in self._label_cbs.items() if cb.isChecked()
        }
        registry = self._picker._model._registry
        for dt, actor in self._picker._dimtag_to_actor.items():
            info = registry.get(dt, {})
            lbl = info.get("label", "")
            # If no label, always show; if label exists, check filter
            if lbl and lbl not in checked_labels:
                actor.VisibilityOff()
                actor.SetPickable(False)
            else:
                if dt not in self._picker._hidden:
                    actor.VisibilityOn()
                    if dt[0] in self._picker._pickable_dims:
                        actor.SetPickable(True)
        self._picker._plotter.render()
        # Also filter tree leaves
        for dt, leaf in self._tree_items_by_dt.items():
            info = registry.get(dt, {})
            lbl = info.get("label", "")
            if lbl and lbl not in checked_labels:
                leaf.setHidden(True)
            else:
                leaf.setHidden(False)

    def _on_name_filter_changed(self, text: str) -> None:
        """Hide tree leaves whose label doesn't match the text filter."""
        text = text.strip().lower()
        for dt, leaf in self._tree_items_by_dt.items():
            if not text:
                leaf.setHidden(False)
            else:
                label = leaf.text(0).lower()
                leaf.setHidden(text not in label)

    def _sync_dim_checkboxes(self) -> None:
        """Update the Filter tab checkboxes to match the current
        ``_pickable_dims`` (called after keyboard shortcuts 1-4/0)."""
        if not hasattr(self, "_dim_cbs"):
            return
        pickable = self._picker._pickable_dims
        for d, cb in self._dim_cbs.items():
            cb.blockSignals(True)
            cb.setChecked(d in pickable)
            cb.blockSignals(False)

    # ------------------------------------------------------------------
    # Status bar
    # ------------------------------------------------------------------

    def _refresh_statusbar(self) -> None:
        picker = self._picker
        active = picker._active_group or "(none)"
        counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for d, _ in picker._picks:
            counts[d] = counts.get(d, 0) + 1
        bits = [
            f"{n} {_DIM_NAMES[d].lower()}"
            for d, n in counts.items() if n > 0
        ]
        breakdown = ", ".join(bits) or "no picks"
        self._statusbar.showMessage(
            f"active={active}   picks: {len(picker._picks)} ({breakdown})"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _group_label(name: str, active: str | None, count: int) -> str:
    marker = " ★" if name == active else ""
    return f"{name}{marker}  ({count})"


def _bold(QtWidgets):
    """Return a bold QFont for tree headers."""
    from qtpy import QtGui
    f = QtGui.QFont()
    f.setBold(True)
    return f


_SHORTCUTS_HTML = """
<style>
  h3   { margin-top: 12px; margin-bottom: 4px; }
  table { border-spacing: 0; width: 100%; }
  td.key { font-family: Consolas, monospace; color: #bbb;
           padding: 2px 10px 2px 2px; white-space: nowrap;
           vertical-align: top; }
  td.desc { padding: 2px 0; }
</style>

<h3>Mouse — 3D viewport</h3>
<table>
  <tr><td class="key">Left click</td>
      <td class="desc">Pick entity under cursor (pixel-accurate).</td></tr>
  <tr><td class="key">Left drag</td>
      <td class="desc">Rubber-band box-select (additive).
            <b>L→R</b> = window (entity fully enclosed),
            <b>R→L</b> = crossing (entity overlaps).</td></tr>
  <tr><td class="key">Ctrl + Left click</td>
      <td class="desc">Unpick (deselect) entity under cursor.</td></tr>
  <tr><td class="key">Ctrl + Left drag</td>
      <td class="desc">Rubber-band box-UNselect (removes from picks).</td></tr>
  <tr><td class="key">Middle drag</td>
      <td class="desc">Pan camera.</td></tr>
  <tr><td class="key">Shift + Middle drag</td>
      <td class="desc">Rotate camera (orbit).</td></tr>
  <tr><td class="key">Right drag</td>
      <td class="desc">Pan camera.</td></tr>
  <tr><td class="key">Wheel</td>
      <td class="desc">Zoom in / out.</td></tr>
  <tr><td class="key">Hover</td>
      <td class="desc">Highlight the entity under the cursor (gold).</td></tr>
</table>

<h3>Keyboard — pick filter</h3>
<table>
  <tr><td class="key">1</td><td class="desc">Points only</td></tr>
  <tr><td class="key">2</td><td class="desc">Curves only</td></tr>
  <tr><td class="key">3</td><td class="desc">Surfaces only</td></tr>
  <tr><td class="key">4</td><td class="desc">Volumes only</td></tr>
  <tr><td class="key">0</td><td class="desc">All dims (reset filter)</td></tr>
</table>

<h3>Keyboard — visibility &amp; edit</h3>
<table>
  <tr><td class="key">H</td><td class="desc">Hide currently picked entities</td></tr>
  <tr><td class="key">I</td><td class="desc">Isolate picks (hide everything else)</td></tr>
  <tr><td class="key">R</td><td class="desc">Reveal all hidden entities</td></tr>
  <tr><td class="key">U</td><td class="desc">Undo last pick</td></tr>
  <tr><td class="key">Tab</td><td class="desc">Cycle through overlapping entities at the
            last click position (Revit-style). Unpicks the current,
            picks the next candidate in the ring.</td></tr>
  <tr><td class="key">Esc</td><td class="desc">Deselect all (clear the working set).</td></tr>
  <tr><td class="key">Q</td><td class="desc">Close window.</td></tr>
</table>

<h3>Model tree (right panel)</h3>
<table>
  <tr><td class="key">Click group header</td>
      <td class="desc">Activate that physical group — subsequent picks
            commit to it on close. ★ marks the active group.</td></tr>
  <tr><td class="key">Click entity leaf</td>
      <td class="desc">Toggle its pick in the active group's working set.</td></tr>
  <tr><td class="key">Right-click group</td>
      <td class="desc">Rename / delete / set-active menu.</td></tr>
  <tr><td class="key">Click instance</td>
      <td class="desc">Select all entities of that Assembly instance.</td></tr>
  <tr><td class="key">Right-click instance</td>
      <td class="desc">Create a physical group directly from an instance.</td></tr>
</table>
""".strip()
