"""
MeshViewerUI
============

Qt (pyvistaqt) front-end for :class:`MeshViewer`.

Inherits from :class:`SelectionPickerWindow` (shared window shell, toolbar,
console, prefs, camera controls, Browser tab with tree + group buttons,
View tab, Filter tab, Preferences tab, Entity Info dock, keyboard shortcuts)
and adds mesh-specific UI:

* **Info tab** (replaces Browser): picked element/node details tree.
* **Display tab**: color mode, node/element labels, wireframe, edges.
* **Mesh Filter tab** (replaces Filter): nodes visibility, mesh dims,
  element types, physical groups.
* **Toolbar extras**: E/N buttons for element/node picking.
* **Status bar**: pick mode, pick counts, mesh stats, color mode.

The window is blocking -- ``MeshViewer.show()`` calls
``MeshViewerWindow(...).exec()`` and waits.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import gmsh

from .SelectionPickerUI import SelectionPickerWindow

if TYPE_CHECKING:
    from .MeshViewer import MeshViewer


# Mapping from Gmsh element type integer to human-readable name.
# Only the most common types are listed; unknown types fall back to
# "Type <N>".
_ELEM_TYPE_NAMES: dict[int, str] = {
    1: "Line2",
    2: "Tri3",
    3: "Quad4",
    4: "Tet4",
    5: "Hex8",
    6: "Prism6",
    7: "Pyramid5",
    8: "Line3",
    9: "Tri6",
    10: "Quad9",
    11: "Tet10",
    15: "Point1",
    16: "Quad8",
    17: "Hex20",
    18: "Prism15",
    21: "Tri10",
    26: "Line4",
    29: "Tet20",
    36: "Quad16",
    92: "Hex27",
    93: "Hex64",
}

_DIM_NAMES = {0: "Points", 1: "Curves", 2: "Surfaces", 3: "Volumes"}

_IC = "#2d2d2d"


# ======================================================================
# MeshViewerWindow
# ======================================================================

class MeshViewerWindow(SelectionPickerWindow):
    """QMainWindow hosting the 3D viewport + mesh info + display controls.

    Inherits from :class:`SelectionPickerWindow` which provides the Qt
    shell, console, toolbar (camera buttons + group/visibility actions),
    Browser tab, View tab, Filter tab, Preferences tab, Entity Info dock,
    and keyboard shortcuts.  This subclass replaces/adds mesh-specific UI.
    """

    def __init__(
        self,
        viewer: "MeshViewer",
        *,
        title: str = "MeshViewer",
        maximized: bool = True,
    ) -> None:
        self._show_node_labels = False
        self._show_elem_labels = False
        self._picker: "MeshViewer" = viewer  # typed reference

        super().__init__(viewer, title=title, maximized=maximized)

        # Reconfigure point-size slider for mesh nodes
        if hasattr(self, "_s_point"):
            self._s_point.setRange(1, 5)
            self._s_point.setValue(int(viewer._point_size))

        # Wire pick-changed -> refresh info panel
        viewer._on_pick_changed.append(self._refresh_mesh_info)

    # ==================================================================
    # Override inherited methods that reference the Browser tree
    # ==================================================================
    # MeshViewerUI replaces Browser with Info tab, so the tree widget
    # from SelectionPickerUI gets deleted.  Override as no-ops.

    def _populate_tree(self) -> None:
        pass

    def _refresh_info(self) -> None:
        self._refresh_mesh_info()

    def _refresh_tree_picks(self) -> None:
        pass

    def _refresh_tree_visibility(self) -> None:
        pass

    # ==================================================================
    # BaseViewerWindow hook overrides
    # ==================================================================

    def _build_tabs(self):
        """Return tabs: Info, Display, Mesh Filter, Preferences."""
        return [
            ("Info", self._build_info_tab()),
            ("Display", self._build_display_tab()),
            ("Filter", self._build_mesh_filter_tab()),
            ("Preferences", self._build_prefs_tab()),  # from base
        ]

    def _build_docks(self):
        """No Entity Info dock -- Info is now a tab."""
        return []

    # ------------------------------------------------------------------
    # Toolbar
    # ------------------------------------------------------------------

    def _build_toolbar_extra(self, bar) -> None:
        """Add E/N buttons for element/node picking before inherited
        buttons."""
        self._act_elem_pick = bar.addAction(self._make_icon("E", _IC), "")
        self._act_elem_pick.setToolTip("Element picking mode")
        self._act_elem_pick.setCheckable(True)
        self._act_elem_pick.setChecked(True)
        self._act_elem_pick.toggled.connect(
            lambda c: self._set_mesh_pick_ui("elem") if c else None,
        )

        self._act_node_pick = bar.addAction(self._make_icon("N", _IC), "")
        self._act_node_pick.setToolTip("Node picking mode")
        self._act_node_pick.setCheckable(True)
        self._act_node_pick.toggled.connect(
            lambda c: self._set_mesh_pick_ui("node") if c else None,
        )

        bar.addSeparator()
        super()._build_toolbar_extra(bar)

    # ------------------------------------------------------------------
    # Status bar
    # ------------------------------------------------------------------

    def _refresh_statusbar(self) -> None:
        """Show mode (ELEM/NODE/BRep), pick counts, mesh stats,
        color mode."""
        v = self._picker
        pick_mode = getattr(v, "_mesh_pick_mode", "off").upper()
        color_mode = getattr(v, "_color_mode", "Default")

        nn = len(getattr(v, "_picked_nodes", []))
        ne = len(getattr(v, "_picked_elems", []))

        total_nodes = 0
        total_elems = 0
        node_tags = getattr(v, "_node_tags", None)
        elem_data = getattr(v, "_elem_data", {})
        if node_tags is not None:
            total_nodes = len(node_tags)
        total_elems = len(elem_data)

        self._statusbar.showMessage(
            f"mode={pick_mode}  "
            f"sel: {nn} nodes, {ne} elems  |  "
            f"mesh: {total_nodes} nodes, {total_elems} elems  |  "
            f"color={color_mode}"
        )

    # ------------------------------------------------------------------
    # Visual changes
    # ------------------------------------------------------------------

    def _apply_visual_changes(self) -> None:
        """Rebuild node cloud + recolor after pref changes."""
        self._picker._apply_coloring()

    # ------------------------------------------------------------------
    # Point size override (sphere glyph approach)
    # ------------------------------------------------------------------

    def _on_point_size_changed(self, value: int) -> None:
        """Rebuild node glyphs with new radius."""
        v = self._picker
        v._point_size = float(value)
        try:
            v._highlight_picked_nodes()
        except Exception:
            pass
        try:
            self._qt_interactor.render()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Hover: no expensive info refresh on hover
    # ------------------------------------------------------------------

    def _on_hover_changed_ui(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Help rows
    # ------------------------------------------------------------------

    def _help_extra_rows(self):
        """Additional shortcut rows for the mesh viewer."""
        return [
            ("E", "Switch to element picking mode"),
            ("N", "Switch to node picking mode"),
            ("Esc", "Deselect all / return to BRep mode"),
            ("H", "Hide selected"),
            ("I", "Isolate selected"),
            ("R", "Show all"),
        ]

    # ==================================================================
    # Mesh pick mode UI synchronisation
    # ==================================================================

    def _set_mesh_pick_ui(self, mode: str) -> None:
        """Sync toolbar toggle state and call viewer pick mode setter.

        Parameters
        ----------
        mode : str
            ``"elem"``, ``"node"``, or ``"brep"``.
        """
        # Map to viewer's mesh_pick_mode API
        if mode == "elem":
            self._picker._set_mesh_pick_mode("element")
        elif mode == "node":
            self._picker._set_mesh_pick_mode("node")
        else:
            self._picker._set_mesh_pick_mode("off")

        # Block signals to avoid re-entrant toggling
        self._act_elem_pick.blockSignals(True)
        self._act_node_pick.blockSignals(True)
        self._act_elem_pick.setChecked(mode == "elem")
        self._act_node_pick.setChecked(mode == "node")
        self._act_elem_pick.blockSignals(False)
        self._act_node_pick.blockSignals(False)

        self._refresh_statusbar()

    # ==================================================================
    # Deselect-all helper
    # ==================================================================

    def _action_deselect_all(self) -> None:
        """Clear all picks and refresh the UI."""
        try:
            self._picker.clear()
        except Exception:
            pass
        self._refresh_statusbar()

    # ==================================================================
    # Info tab (replaces Browser)
    # ==================================================================

    def _build_info_tab(self):
        """QTreeWidget for picked element/node details."""
        QtWidgets = self._QtWidgets

        self._mesh_info_tree = QtWidgets.QTreeWidget()
        self._mesh_info_tree.setHeaderLabels(["Property", "Value"])
        self._mesh_info_tree.setColumnCount(2)
        self._mesh_info_tree.setAlternatingRowColors(True)
        self._mesh_info_tree.setRootIsDecorated(True)
        return self._mesh_info_tree

    def _refresh_mesh_info(self) -> None:
        """Populate the info tree with picked elements and nodes."""
        tree = self._mesh_info_tree
        tree.clear()

        v = self._picker
        pick_mode = getattr(v, "_mesh_pick_mode", "off")

        if pick_mode in ("element", "node"):
            self._refresh_info_mesh_level(tree, v)
        else:
            self._refresh_info_brep_level(tree, v)

        tree.expandAll()
        tree.resizeColumnToContents(0)
        tree.resizeColumnToContents(1)

    def _refresh_info_mesh_level(self, tree, v) -> None:
        """Populate info tree for mesh-level picking."""
        QtWidgets = self._QtWidgets

        # Show selected elements first
        selected_elems = getattr(v, "_picked_elems", [])
        if selected_elems:
            elem_tag = selected_elems[-1]
            elem_data = getattr(v, "_elem_data", {})
            info = elem_data.get(elem_tag, {})
            QtWidgets.QTreeWidgetItem(tree, ["Element tag", str(elem_tag)])
            QtWidgets.QTreeWidgetItem(
                tree, ["Type", info.get("type_name", "unknown")],
            )
            brep_dt = info.get("brep_dt")
            if brep_dt:
                QtWidgets.QTreeWidgetItem(
                    tree, ["BRep entity", f"({brep_dt[0]}, {brep_dt[1]})"],
                )
                gname = getattr(v, "_brep_to_group", {}).get(brep_dt, "")
                if gname:
                    QtWidgets.QTreeWidgetItem(tree, ["Physical group", gname])
            nodes = info.get("nodes", [])
            if nodes:
                nodes_item = QtWidgets.QTreeWidgetItem(
                    tree, ["Nodes", f"{len(nodes)} node(s)"],
                )
                for nt in nodes:
                    node_coords = getattr(v, "_node_coords", None)
                    node_tag_to_idx = getattr(v, "_node_tag_to_idx", {})
                    idx = node_tag_to_idx.get(nt)
                    if idx is not None and node_coords is not None:
                        c = node_coords[idx]
                        QtWidgets.QTreeWidgetItem(
                            nodes_item,
                            [f"N{nt}", f"({c[0]:.4g}, {c[1]:.4g}, {c[2]:.4g})"],
                        )
                    else:
                        QtWidgets.QTreeWidgetItem(nodes_item, [f"N{nt}", ""])
            return

        # Fall back to node info
        selected_nodes = getattr(v, "_picked_nodes", [])
        if not selected_nodes:
            QtWidgets.QTreeWidgetItem(tree, ["(no selection)", ""])
            return

        node_tag = selected_nodes[-1]
        QtWidgets.QTreeWidgetItem(tree, ["Node tag", str(node_tag)])

        # Coordinates
        try:
            coords, _, _, _ = gmsh.model.mesh.getNode(node_tag)
            x, y, z = coords[0], coords[1], coords[2]
            coord_item = QtWidgets.QTreeWidgetItem(
                tree, ["Coordinates", ""],
            )
            QtWidgets.QTreeWidgetItem(coord_item, ["x", f"{x:.6g}"])
            QtWidgets.QTreeWidgetItem(coord_item, ["y", f"{y:.6g}"])
            QtWidgets.QTreeWidgetItem(coord_item, ["z", f"{z:.6g}"])
        except Exception:
            QtWidgets.QTreeWidgetItem(
                tree, ["Coordinates", "(unavailable)"],
            )

        # Connected elements -- use cached _elem_data (fast lookup)
        elem_data = getattr(v, "_elem_data", {})
        connected = [
            etag for etag, info in elem_data.items()
            if node_tag in info.get("nodes", [])
        ]
        if connected:
            conn_item = QtWidgets.QTreeWidgetItem(
                tree,
                ["Connected elements", f"{len(connected)} element(s)"],
            )
            for eid in connected[:20]:
                info = elem_data.get(eid, {})
                etype = info.get("type_name", "")
                QtWidgets.QTreeWidgetItem(
                    conn_item, [f"Elem {eid}", etype],
                )
            if len(connected) > 20:
                QtWidgets.QTreeWidgetItem(
                    conn_item, [f"... +{len(connected) - 20} more", ""],
                )
        else:
            QtWidgets.QTreeWidgetItem(
                tree, ["Connected elements", "none"],
            )

        # Physical group membership -- use cached mappings
        brep_to_group = getattr(v, "_brep_to_group", {})
        elem_to_brep = getattr(v, "_elem_to_brep", {})
        memberships: list[str] = []
        seen_groups: set[str] = set()
        for etag in connected:
            brep_dt = elem_to_brep.get(etag)
            if brep_dt is not None:
                gname = brep_to_group.get(brep_dt)
                if gname and gname not in seen_groups:
                    seen_groups.add(gname)
                    memberships.append(gname)
        if memberships:
            pg_item = QtWidgets.QTreeWidgetItem(
                tree,
                ["Physical groups", f"{len(memberships)} group(s)"],
            )
            for m in memberships:
                QtWidgets.QTreeWidgetItem(pg_item, [m, ""])
        else:
            QtWidgets.QTreeWidgetItem(
                tree, ["Physical groups", "none"],
            )

    def _refresh_info_brep_level(self, tree, v) -> None:
        """Populate info tree for BRep-level (patch) picking."""
        QtWidgets = self._QtWidgets

        picks = getattr(v, "_picks", [])
        if not picks:
            QtWidgets.QTreeWidgetItem(tree, ["(no patch selected)", ""])
            return

        # Show the last picked entity
        dim, tag = picks[-1]
        dim_name = _DIM_NAMES.get(dim, f"Dim{dim}")

        QtWidgets.QTreeWidgetItem(
            tree, ["BRep entity", f"{dim_name} {tag}"],
        )
        QtWidgets.QTreeWidgetItem(tree, ["Dimension", str(dim)])
        QtWidgets.QTreeWidgetItem(tree, ["Tag", str(tag)])

        # Element count and types
        try:
            elem_types, elem_tags_list, _ = gmsh.model.mesh.getElements(
                dim, tag,
            )
            total_elems = sum(len(et) for et in elem_tags_list)
            QtWidgets.QTreeWidgetItem(
                tree, ["Element count", str(total_elems)],
            )

            types_item = QtWidgets.QTreeWidgetItem(
                tree, ["Element types", ""],
            )
            for i, et in enumerate(elem_types):
                type_name = _ELEM_TYPE_NAMES.get(et, f"Type{et}")
                n_elems = len(elem_tags_list[i])
                QtWidgets.QTreeWidgetItem(
                    types_item, [type_name, str(n_elems)],
                )
        except Exception:
            QtWidgets.QTreeWidgetItem(
                tree, ["Element count", "(unavailable)"],
            )

        # Node count
        try:
            node_tags_ent, _, _ = gmsh.model.mesh.getNodes(dim, tag)
            QtWidgets.QTreeWidgetItem(
                tree, ["Node count", str(len(node_tags_ent))],
            )
        except Exception:
            QtWidgets.QTreeWidgetItem(
                tree, ["Node count", "(unavailable)"],
            )

    # ==================================================================
    # Display tab
    # ==================================================================

    def _build_display_tab(self):
        """Build the Display tab with visualization controls.

        Controls:
        - Color mode combo (Default/Partition/Quality/Element Type/Physical Group)
        - Show node labels checkbox
        - Show element labels checkbox
        - Wireframe checkbox
        - Show edges checkbox
        """
        QtWidgets = self._QtWidgets

        panel = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(panel)
        form.setContentsMargins(8, 8, 8, 8)
        form.setSpacing(6)

        # ---- Color mode ----
        self._combo_color_mode = QtWidgets.QComboBox()
        self._combo_color_mode.addItems([
            "Default", "Partition", "Quality", "Element Type", "Physical Group",
        ])
        self._combo_color_mode.currentTextChanged.connect(
            self._on_color_mode_changed,
        )
        form.addRow("Color mode", self._combo_color_mode)

        # ---- Show node labels ----
        self._cb_node_labels = QtWidgets.QCheckBox("Show node labels")
        self._cb_node_labels.setChecked(False)
        self._cb_node_labels.toggled.connect(self._on_node_labels_toggled)
        form.addRow(self._cb_node_labels)

        # ---- Show element labels ----
        self._cb_elem_labels = QtWidgets.QCheckBox("Show element labels")
        self._cb_elem_labels.setChecked(False)
        self._cb_elem_labels.toggled.connect(self._on_elem_labels_toggled)
        form.addRow(self._cb_elem_labels)

        # ---- Wireframe ----
        self._cb_wireframe = QtWidgets.QCheckBox("Wireframe")
        self._cb_wireframe.setChecked(False)
        self._cb_wireframe.toggled.connect(self._on_wireframe_toggled)
        form.addRow(self._cb_wireframe)

        # ---- Show edges ----
        self._cb_show_edges = QtWidgets.QCheckBox("Show edges")
        self._cb_show_edges.setChecked(True)
        self._cb_show_edges.toggled.connect(self._on_show_edges_toggled)
        form.addRow(self._cb_show_edges)

        return panel

    # ---- Display callbacks ----

    def _on_color_mode_changed(self, mode: str) -> None:
        """Forward color mode change to the viewer."""
        try:
            self._picker._set_color_mode(mode)
        except Exception:
            pass
        self._refresh_statusbar()
        try:
            self._qt_interactor.render()
        except Exception:
            pass

    def _on_node_labels_toggled(self, checked: bool) -> None:
        """Toggle node label overlays."""
        self._show_node_labels = checked
        try:
            self._picker._toggle_node_labels(checked)
        except Exception:
            pass
        try:
            self._qt_interactor.render()
        except Exception:
            pass

    def _on_elem_labels_toggled(self, checked: bool) -> None:
        """Toggle element label overlays."""
        self._show_elem_labels = checked
        try:
            self._picker._toggle_elem_labels(checked)
        except Exception:
            pass
        try:
            self._qt_interactor.render()
        except Exception:
            pass

    def _on_wireframe_toggled(self, checked: bool) -> None:
        """Toggle wireframe / solid rendering on all mesh actors."""
        try:
            plotter = self._qt_interactor
            for actor in plotter.renderer.actors.values():
                try:
                    prop = actor.GetProperty()
                    if prop is not None:
                        if checked:
                            prop.SetRepresentationToWireframe()
                        else:
                            prop.SetRepresentationToSurface()
                except Exception:
                    pass
            plotter.render()
        except Exception:
            pass

    def _on_show_edges_toggled(self, checked: bool) -> None:
        """Toggle mesh edge visibility on all actors."""
        try:
            plotter = self._qt_interactor
            for actor in plotter.renderer.actors.values():
                try:
                    prop = actor.GetProperty()
                    if prop is not None:
                        prop.SetEdgeVisibility(checked)
                except Exception:
                    pass
            plotter.render()
        except Exception:
            pass

    # ==================================================================
    # Mesh Filter tab (replaces SelectionPicker Filter)
    # ==================================================================

    def _build_mesh_filter_tab(self):
        """Build the mesh-specific Filter tab.

        Sections:
        - Show nodes checkbox
        - Mesh dimensions (1D Lines / 2D Surfaces / 3D Volumes)
        - Element types (checkboxes from _brep_dominant_type categories)
        - Physical groups (checkboxes, including "(ungrouped)")
        """
        QtWidgets = self._QtWidgets

        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # ---- Show nodes ----
        self._cb_show_nodes = QtWidgets.QCheckBox("Show nodes")
        self._cb_show_nodes.setChecked(True)
        self._cb_show_nodes.toggled.connect(self._on_show_nodes_toggled)
        layout.addWidget(self._cb_show_nodes)

        # ---- Mesh dimensions ----
        dim_group = QtWidgets.QGroupBox("Mesh dimensions")
        dim_layout = QtWidgets.QVBoxLayout(dim_group)
        self._mesh_dim_cbs: dict[int, object] = {}
        dim_labels = {1: "1D Lines", 2: "2D Surfaces", 3: "3D Volumes"}
        for d, label in dim_labels.items():
            cb = QtWidgets.QCheckBox(label)
            cb.setChecked(True)
            cb.toggled.connect(self._on_mesh_dim_filter_changed)
            dim_layout.addWidget(cb)
            self._mesh_dim_cbs[d] = cb
        layout.addWidget(dim_group)

        # ---- Element types ----
        elem_group = QtWidgets.QGroupBox("Element types")
        elem_layout = QtWidgets.QVBoxLayout(elem_group)
        self._elem_type_cbs: dict[str, object] = {}

        # Collect unique element type categories from brep_dominant_type
        v = self._picker
        categories: set[str] = set()
        brep_dom = getattr(v, "_brep_dominant_type", {})
        for cat in brep_dom.values():
            categories.add(cat)

        if categories:
            for cat in sorted(categories):
                cb = QtWidgets.QCheckBox(cat)
                cb.setChecked(True)
                cb.toggled.connect(self._on_elem_type_filter_changed)
                elem_layout.addWidget(cb)
                self._elem_type_cbs[cat] = cb
        else:
            elem_layout.addWidget(QtWidgets.QLabel("(no elements)"))

        scroll_elem = QtWidgets.QScrollArea()
        scroll_elem.setWidgetResizable(True)
        scroll_elem.setWidget(elem_group)
        scroll_elem.setMaximumHeight(160)
        layout.addWidget(scroll_elem)

        # ---- Physical groups ----
        pg_group = QtWidgets.QGroupBox("Physical groups")
        pg_layout = QtWidgets.QVBoxLayout(pg_group)
        self._pg_filter_cbs: dict[str, object] = {}

        try:
            phys_groups = gmsh.model.getPhysicalGroups()
        except Exception:
            phys_groups = []

        pg_names: list[str] = []
        for dim, tag in phys_groups:
            try:
                name = gmsh.model.getPhysicalName(dim, tag)
            except Exception:
                name = ""
            if not name:
                name = f"PhysGrp({dim},{tag})"
            if name not in pg_names:
                pg_names.append(name)

        # Always add "(ungrouped)" for entities not in any group
        pg_names.append("(ungrouped)")

        for name in pg_names:
            cb = QtWidgets.QCheckBox(name)
            cb.setChecked(True)
            cb.toggled.connect(self._on_pg_filter_changed)
            pg_layout.addWidget(cb)
            self._pg_filter_cbs[name] = cb

        scroll_pg = QtWidgets.QScrollArea()
        scroll_pg.setWidgetResizable(True)
        scroll_pg.setWidget(pg_group)
        scroll_pg.setMaximumHeight(200)
        layout.addWidget(scroll_pg)

        layout.addStretch(1)
        return panel

    # ---- Filter callbacks ----

    def _on_show_nodes_toggled(self, checked: bool) -> None:
        """Toggle node actor visibility."""
        v = self._picker
        node_actor = getattr(v, "_node_actor", None)
        if node_actor is not None:
            try:
                if checked:
                    node_actor.VisibilityOn()
                else:
                    node_actor.VisibilityOff()
            except Exception:
                pass
        try:
            self._qt_interactor.render()
        except Exception:
            pass

    def _on_mesh_dim_filter_changed(self) -> None:
        """Dimension filter changed -- apply combined filter."""
        self._apply_mesh_filters()

    def _on_elem_type_filter_changed(self) -> None:
        """Element type filter changed -- apply combined filter."""
        self._apply_mesh_filters()

    def _on_pg_filter_changed(self) -> None:
        """Physical group filter changed -- apply combined filter."""
        self._apply_mesh_filters()

    def _apply_mesh_filters(self) -> None:
        """Single pass: combine dim filter + element type filter +
        physical group filter.  Each actor is visible only if ALL pass."""
        v = self._picker

        # Collect enabled dims
        enabled_dims: set[int] = set()
        for d, cb in self._mesh_dim_cbs.items():
            try:
                if cb.isChecked():
                    enabled_dims.add(d)
            except RuntimeError:
                enabled_dims.add(d)

        # Collect enabled element type categories
        enabled_types: set[str] = set()
        for cat, cb in self._elem_type_cbs.items():
            try:
                if cb.isChecked():
                    enabled_types.add(cat)
            except RuntimeError:
                enabled_types.add(cat)

        # Collect enabled physical group names
        enabled_pgs: set[str] = set()
        for name, cb in self._pg_filter_cbs.items():
            try:
                if cb.isChecked():
                    enabled_pgs.add(name)
            except RuntimeError:
                enabled_pgs.add(name)

        id_to_actor = getattr(v, "_id_to_actor", {})
        brep_dom = getattr(v, "_brep_dominant_type", {})
        brep_to_group = getattr(v, "_brep_to_group", {})

        for dt, actor in id_to_actor.items():
            dim = dt[0]

            # 1) Dimension filter
            dim_pass = dim in enabled_dims

            # 2) Element type filter
            cat = brep_dom.get(dt, "")
            type_pass = (not self._elem_type_cbs) or (cat in enabled_types)

            # 3) Physical group filter
            gname = brep_to_group.get(dt)
            if gname:
                pg_pass = gname in enabled_pgs
            else:
                pg_pass = "(ungrouped)" in enabled_pgs

            visible = dim_pass and type_pass and pg_pass

            try:
                if visible:
                    actor.VisibilityOn()
                    actor.SetPickable(True)
                else:
                    actor.VisibilityOff()
                    actor.SetPickable(False)
            except Exception:
                pass

        try:
            self._qt_interactor.render()
        except Exception:
            pass

    # ==================================================================
    # Preferences extras
    # ==================================================================

    def _build_prefs_extra(self, form) -> None:
        """Extend the Preferences tab with mesh-specific controls.

        Adds:
        - Node marker size slider
        - Edge color picker button
        """
        QtWidgets = self._QtWidgets
        QtCore = self._QtCore

        form.addRow(QtWidgets.QLabel(""))  # visual spacer
        form.addRow(QtWidgets.QLabel("--- Mesh-specific ---"))

        # ---- Node marker size ----
        self._s_node_marker = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._s_node_marker.setRange(1, 30)
        self._s_node_marker.setValue(
            int(getattr(self._picker, "_node_marker_size", 6)),
        )
        self._s_node_marker.valueChanged.connect(self._on_node_marker_changed)
        form.addRow("Node marker size", self._s_node_marker)

        # ---- Edge color picker ----
        self._btn_edge_color = QtWidgets.QPushButton()
        self._btn_edge_color.setFixedSize(60, 24)
        edge_color = getattr(self._picker, "_edge_color", "#000000")
        self._btn_edge_color.setStyleSheet(
            f"background-color: {edge_color}; border: 1px solid #999;",
        )
        self._btn_edge_color.clicked.connect(self._on_edge_color_pick)
        form.addRow("Edge color", self._btn_edge_color)

    def _on_node_marker_changed(self, value: int) -> None:
        """Update node marker size on the viewer and re-render."""
        try:
            self._picker._node_marker_size = float(value)
        except Exception:
            pass
        self._apply_visual_changes()
        try:
            self._qt_interactor.render()
        except Exception:
            pass

    def _on_edge_color_pick(self) -> None:
        """Open a color dialog and apply the chosen edge color."""
        QtWidgets = self._QtWidgets
        QtGui = self._QtGui

        current = getattr(self._picker, "_edge_color", "#000000")
        color = QtWidgets.QColorDialog.getColor(
            QtGui.QColor(current), self._window, "Edge Color",
        )
        if color.isValid():
            hex_color = color.name()
            self._picker._edge_color = hex_color
            self._btn_edge_color.setStyleSheet(
                f"background-color: {hex_color}; border: 1px solid #999;",
            )
            self._apply_visual_changes()
            try:
                self._qt_interactor.render()
            except Exception:
                pass
