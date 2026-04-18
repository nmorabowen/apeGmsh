"""
ConstraintsTabPanel — read-only constraint display for the mesh viewer.

Renders a tree of ConstraintDef objects grouped by kind, with one
checkbox per kind.  Toggling a checkbox triggers an
``on_kinds_changed(set[str])`` callback so the viewer can show / hide
the corresponding line / glyph overlays in the 3-D viewport.

This panel never modifies state — it reads from
``g.constraints.constraint_defs`` and (optionally) ``fem.nodes.constraints``
/ ``fem.elements.constraints`` for stats.
"""
from __future__ import annotations

from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from apeGmsh.core.ConstraintsComposite import ConstraintsComposite
    from apeGmsh.mesh.FEMData import FEMData


def _qt():
    from qtpy import QtWidgets, QtCore, QtGui
    return QtWidgets, QtCore, QtGui


# ── Catppuccin Mocha palette for constraint kinds ────────────
_CONSTRAINT_COLORS: dict[str, str] = {
    "equal_dof":          "#89b4fa",  # blue
    "rigid_beam":         "#f38ba8",  # red
    "rigid_rod":          "#fab387",  # peach
    "rigid_diaphragm":    "#a6e3a1",  # green
    "rigid_body":         "#cba6f7",  # mauve
    "kinematic_coupling": "#f9e2af",  # yellow
    "penalty":            "#94e2d5",  # teal
    "node_to_surface":    "#f5c2e7",  # pink
    "tie":                "#74c7ec",  # sapphire
    "distributing":       "#b4befe",  # lavender
    "embedded":           "#eba0ac",  # maroon
    "tied_contact":       "#89dceb",  # sky
    "mortar":             "#a6adc8",  # overlay
}

_FALLBACK_COLOR = "#a6adc8"


def constraint_color(kind: str) -> str:
    """Return the hex color for a constraint kind."""
    return _CONSTRAINT_COLORS.get(kind, _FALLBACK_COLOR)


def _def_kind_key(d) -> str:
    """Resolve the grouping key for a constraint definition.

    ``RigidLinkDef`` has ``kind="rigid_link"`` but the resolved records
    use ``"rigid_beam"`` or ``"rigid_rod"`` depending on ``link_type``.
    We use the resolved form so the tree matches the callback filter.
    """
    from apeGmsh.solvers.Constraints import RigidLinkDef
    if isinstance(d, RigidLinkDef):
        return f"rigid_{d.link_type}"
    return d.kind


class ConstraintsTabPanel:
    """Read-only constraints display: kind checkboxes + def tree + stats."""

    _DT_ROLE = 0x0100

    def __init__(
        self,
        constraints_composite: "ConstraintsComposite",
        fem: "FEMData | None" = None,
        *,
        on_kinds_changed: Callable[[set[str]], None] | None = None,
    ) -> None:
        QtWidgets, QtCore, QtGui = _qt()
        self._QtGui = QtGui
        self._constraints = constraints_composite
        self._fem = fem
        self._on_kinds_changed = on_kinds_changed

        self.widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(self.widget)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        # ── header ────────────────────────────────────────────
        self._header = QtWidgets.QLabel("Constraints")
        self._header.setStyleSheet("font-weight: bold; padding: 2px;")
        layout.addWidget(self._header)

        # ── empty state ───────────────────────────────────────
        self._empty_label = QtWidgets.QLabel(
            "No constraints defined.\n\n"
            "Use g.constraints.equal_dof(),\n"
            "g.constraints.tie(), etc."
        )
        from .theme import THEME as _THEME
        self._empty_label.setStyleSheet(
            f"color: {_THEME.current.overlay}; padding: 12px;"
        )
        self._empty_label.setWordWrap(True)
        layout.addWidget(self._empty_label)

        # ── needs-fem warning ─────────────────────────────────
        self._fem_warning = QtWidgets.QLabel(
            "No resolved mesh available.\n"
            "Generate a mesh first to display\n"
            "constraint overlays in the 3-D viewport."
        )
        _warn = _THEME.current.warning
        self._fem_warning.setStyleSheet(
            f"color: {_warn}; padding: 6px; "
            f"border: 1px solid {_warn}; border-radius: 3px;"
        )
        self._fem_warning.setWordWrap(True)
        layout.addWidget(self._fem_warning)

        # ── tree ──────────────────────────────────────────────
        self._tree = QtWidgets.QTreeWidget()
        self._tree.setHeaderLabels(["Constraint", "Detail"])
        self._tree.setColumnCount(2)
        self._tree.setAlternatingRowColors(True)
        self._tree.setRootIsDecorated(True)
        self._tree.setIndentation(16)
        self._tree.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self._tree)

        self._kind_items: dict[str, Any] = {}
        self._suppress_signal = False

        # ── stats ─────────────────────────────────────────────
        self._stats_box = QtWidgets.QGroupBox("Stats")
        stats_layout = QtWidgets.QFormLayout(self._stats_box)
        self._lbl_node_pairs = QtWidgets.QLabel("—")
        self._lbl_interp = QtWidgets.QLabel("—")
        self._lbl_phantoms = QtWidgets.QLabel("—")
        stats_layout.addRow("Node pairs:", self._lbl_node_pairs)
        stats_layout.addRow("Interpolations:", self._lbl_interp)
        stats_layout.addRow("Phantom nodes:", self._lbl_phantoms)
        layout.addWidget(self._stats_box)

        # ── buttons ───────────────────────────────────────────
        btn_row = QtWidgets.QHBoxLayout()
        btn_show = QtWidgets.QPushButton("Show all")
        btn_show.clicked.connect(self._show_all)
        btn_row.addWidget(btn_show)
        btn_hide = QtWidgets.QPushButton("Hide all")
        btn_hide.clicked.connect(self._hide_all)
        btn_row.addWidget(btn_hide)
        btn_refresh = QtWidgets.QPushButton("Refresh")
        btn_refresh.clicked.connect(self.refresh)
        btn_row.addWidget(btn_refresh)
        layout.addLayout(btn_row)

        self.refresh()

    # ── Build tree ──────────────────────────────────────────────

    def refresh(self) -> None:
        """Re-read constraint_defs and rebuild the tree + stats."""
        QtGui = self._QtGui
        from qtpy.QtWidgets import QTreeWidgetItem
        from qtpy.QtCore import Qt

        self._suppress_signal = True
        self._tree.clear()
        self._kind_items.clear()

        defs = getattr(self._constraints, 'constraint_defs', None) or []

        has_defs = bool(defs)
        self._empty_label.setVisible(not has_defs)
        self._tree.setVisible(has_defs)
        self._fem_warning.setVisible(self._fem is None and has_defs)

        # Group by resolved kind key
        by_kind: dict[str, list] = {}
        for d in defs:
            key = _def_kind_key(d)
            by_kind.setdefault(key, []).append(d)

        n = len(defs)
        n_kinds = len(by_kind)
        self._header.setText(
            f"Constraints ({n} def{'s' if n != 1 else ''}, "
            f"{n_kinds} kind{'s' if n_kinds != 1 else ''})"
            if has_defs else "Constraints"
        )

        if not has_defs:
            self._update_stats()
            self._suppress_signal = False
            return

        for kind_name, kind_defs in by_kind.items():
            root = QTreeWidgetItem(self._tree)
            root.setText(0, kind_name)
            root.setText(1, f"{len(kind_defs)} def(s)")
            root.setData(0, self._DT_ROLE, ("kind", kind_name))
            root.setFlags(root.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            root.setCheckState(0, Qt.CheckState.Unchecked)

            color = QtGui.QColor(constraint_color(kind_name))
            root.setForeground(0, QtGui.QBrush(color))
            font = root.font(0)
            font.setBold(True)
            root.setFont(0, font)

            self._kind_items[kind_name] = root

            for d in kind_defs:
                child = QTreeWidgetItem(root)
                # Prefer the human-readable name if set, otherwise
                # fall back to master_label → slave_label.
                display = getattr(d, 'name', None)
                if not display:
                    master = getattr(d, 'master_label', '?')
                    slave = getattr(d, 'slave_label', '?')
                    display = f"{master} \u2192 {slave}"
                child.setText(0, display)
                child.setText(1, self._format_def_detail(d))
                child.setData(0, self._DT_ROLE, ("def", id(d)))

            root.setExpanded(False)

        self._tree.resizeColumnToContents(0)
        self._update_stats()
        self._suppress_signal = False

    def _update_stats(self) -> None:
        """Populate the stats box from resolved FEM data."""
        if self._fem is None:
            self._lbl_node_pairs.setText("—")
            self._lbl_interp.setText("—")
            self._lbl_phantoms.setText("—")
            self._stats_box.setVisible(False)
            return

        self._stats_box.setVisible(True)
        nc = self._fem.nodes.constraints
        sc = self._fem.elements.constraints

        n_pairs = sum(1 for _ in nc.pairs())
        n_interp = sum(1 for _ in sc.interpolations())
        n_phantom = len(nc.phantom_nodes())

        self._lbl_node_pairs.setText(str(n_pairs))
        self._lbl_interp.setText(str(n_interp))
        self._lbl_phantoms.setText(str(n_phantom))

    @staticmethod
    def _format_def_detail(d) -> str:
        """One-line summary of a ConstraintDef for the Detail column.

        Extracts the most informative fields from whatever def type
        is passed, without importing every concrete class.
        """
        parts: list[str] = []
        # Most defs have these — show whichever exists
        for attr, fmt in [
            ('link_type',         'type={}'),
            ('dofs',              'dofs={}'),
            ('tolerance',         'tol={}'),
            ('stiffness',         'k={:.3g}'),
            ('plane_normal',      'normal={}'),
            ('master_point',      'pt={}'),
            ('weighting',         'weighting={}'),
            ('integration_order', 'order={}'),
        ]:
            val = getattr(d, attr, None)
            if val is None:
                continue
            # dofs=None means "all" by convention
            if attr == 'dofs' and val is None:
                continue
            try:
                parts.append(fmt.format(val))
            except (TypeError, ValueError):
                parts.append(f"{attr}={val}")
            if len(parts) >= 2:
                break  # two fields is enough for a one-liner
        return ", ".join(parts) or d.kind

    # ── Kind state queries ────────────────────────────────────

    def active_kinds(self) -> set[str]:
        """Kinds whose checkboxes are currently checked."""
        from qtpy.QtCore import Qt
        active: set[str] = set()
        for name, item in self._kind_items.items():
            if item.checkState(0) == Qt.CheckState.Checked:
                active.add(name)
        return active

    # ── Button handlers ───────────────────────────────────────

    def _show_all(self) -> None:
        from qtpy.QtCore import Qt
        self._suppress_signal = True
        for item in self._kind_items.values():
            item.setCheckState(0, Qt.CheckState.Checked)
        self._suppress_signal = False
        if self._on_kinds_changed:
            self._on_kinds_changed(self.active_kinds())

    def _hide_all(self) -> None:
        from qtpy.QtCore import Qt
        self._suppress_signal = True
        for item in self._kind_items.values():
            item.setCheckState(0, Qt.CheckState.Unchecked)
        self._suppress_signal = False
        if self._on_kinds_changed:
            self._on_kinds_changed(self.active_kinds())

    def _on_item_changed(self, item, _column) -> None:
        if self._suppress_signal:
            return
        if self._on_kinds_changed:
            self._on_kinds_changed(self.active_kinds())

    # ── External fem update ───────────────────────────────────

    def set_fem(self, fem) -> None:
        """Update the FEM reference (e.g. after re-meshing)."""
        self._fem = fem
        self._fem_warning.setVisible(
            self._fem is None
            and bool(getattr(self._constraints, 'constraint_defs', None))
        )
        self._update_stats()
