"""
LoadsTabPanel — read-only loads display for the model viewer.

Renders a tree of LoadDef objects grouped by pattern, with one
checkbox per pattern.  Toggling a checkbox triggers an
``on_patterns_changed(set[str])`` callback so the viewer can show /
hide the corresponding arrow glyphs in the 3-D viewport.

This panel never modifies state — it reads from
``g.loads.load_defs`` and (optionally) ``fem.loads`` for stats.
"""
from __future__ import annotations

from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from apeGmsh.core.LoadsComposite import LoadsComposite
    from apeGmsh.mesh.FEMData import FEMData


def _qt():
    from qtpy import QtWidgets, QtCore, QtGui
    return QtWidgets, QtCore, QtGui


# Catppuccin Mocha palette — same as PartsTreePanel
_PATTERN_PALETTE = [
    "#a6e3a1",  # green
    "#fab387",  # peach
    "#f9e2af",  # yellow
    "#89b4fa",  # blue
    "#cba6f7",  # mauve
    "#f38ba8",  # red
    "#94e2d5",  # teal
]


def pattern_color(name: str) -> str:
    """Stable color from a pattern name via hash."""
    return _PATTERN_PALETTE[abs(hash(name)) % len(_PATTERN_PALETTE)]


class LoadsTabPanel:
    """Read-only loads display: pattern checkboxes + def tree + refresh."""

    _DT_ROLE = 0x0100

    def __init__(
        self,
        loads_composite: "LoadsComposite",
        fem: "FEMData | None" = None,
        *,
        on_patterns_changed: Callable[[set[str]], None] | None = None,
    ) -> None:
        QtWidgets, QtCore, QtGui = _qt()
        self._QtGui = QtGui
        self._loads = loads_composite
        self._fem = fem
        self._on_patterns_changed = on_patterns_changed

        self.widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(self.widget)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        self._header = QtWidgets.QLabel("Loads")
        self._header.setStyleSheet("font-weight: bold; padding: 2px;")
        layout.addWidget(self._header)

        # ── empty state ───────────────────────────────────────
        self._empty_label = QtWidgets.QLabel(
            "No loads defined.\n\n"
            "Use g.loads.point(), g.loads.gravity(), etc.\n"
            "inside a g.loads.pattern(name) block."
        )
        self._empty_label.setStyleSheet("color: #6c7086; padding: 12px;")
        self._empty_label.setWordWrap(True)
        layout.addWidget(self._empty_label)

        # ── needs-fem warning ─────────────────────────────────
        self._fem_warning = QtWidgets.QLabel(
            "Pass fem= to g.model.viewer() to display\n"
            "load arrow overlays in the 3-D viewport."
        )
        self._fem_warning.setStyleSheet(
            "color: #f9e2af; padding: 6px; "
            "border: 1px solid #f9e2af; border-radius: 3px;"
        )
        self._fem_warning.setWordWrap(True)
        layout.addWidget(self._fem_warning)

        # ── tree ──────────────────────────────────────────────
        self._tree = QtWidgets.QTreeWidget()
        self._tree.setHeaderLabels(["Load", "Detail"])
        self._tree.setColumnCount(2)
        self._tree.setAlternatingRowColors(True)
        self._tree.setRootIsDecorated(True)
        self._tree.setIndentation(16)
        self._tree.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self._tree)

        # Map pattern_name -> root QTreeWidgetItem (for state queries)
        self._pattern_items: dict[str, Any] = {}  # QTreeWidgetItem (lazy Qt import)
        # Suppress itemChanged during programmatic edits
        self._suppress_signal = False

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
        """Re-read load_defs and rebuild the tree."""
        QtGui = self._QtGui
        from qtpy.QtWidgets import QTreeWidgetItem
        from qtpy.QtCore import Qt

        self._suppress_signal = True
        self._tree.clear()
        self._pattern_items.clear()

        defs = self._loads.load_defs

        # Empty state visibility
        has_defs = bool(defs)
        self._empty_label.setVisible(not has_defs)
        self._tree.setVisible(has_defs)
        self._fem_warning.setVisible(self._fem is None and has_defs)

        n = len(defs)
        n_pats = len(self._loads.patterns())
        self._header.setText(
            f"Loads ({n} def{'s' if n != 1 else ''}, "
            f"{n_pats} pattern{'s' if n_pats != 1 else ''})"
            if has_defs else "Loads"
        )

        if not has_defs:
            self._suppress_signal = False
            return

        # Group defs by pattern (preserve insertion order)
        by_pat: dict[str, list] = {}
        for d in defs:
            by_pat.setdefault(d.pattern, []).append(d)

        for pat_name, pat_defs in by_pat.items():
            # Root: pattern with checkbox
            root = QTreeWidgetItem(self._tree)
            root.setText(0, pat_name)
            root.setText(1, f"{len(pat_defs)} load(s)")
            root.setData(0, self._DT_ROLE, ("pattern", pat_name))
            root.setFlags(root.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            root.setCheckState(0, Qt.CheckState.Unchecked)

            color = QtGui.QColor(pattern_color(pat_name))
            root.setForeground(0, QtGui.QBrush(color))
            font = root.font(0)
            font.setBold(True)
            root.setFont(0, font)

            self._pattern_items[pat_name] = root

            for d in pat_defs:
                child = QTreeWidgetItem(root)
                child.setText(0, f"{d.kind} -> {d.target}")
                child.setText(1, self._format_def_detail(d))
                child.setData(0, self._DT_ROLE, ("def", id(d)))

            root.setExpanded(False)

        self._tree.resizeColumnToContents(0)
        self._suppress_signal = False

    def _format_def_detail(self, d) -> str:
        """One-line summary of a LoadDef for the Detail column."""
        from apeGmsh.solvers.Loads import (
            PointLoadDef, LineLoadDef, SurfaceLoadDef,
            GravityLoadDef, BodyLoadDef,
        )
        if isinstance(d, PointLoadDef):
            parts = []
            if d.force_xyz:
                parts.append(f"F={tuple(d.force_xyz)}")
            if d.moment_xyz:
                parts.append(f"M={tuple(d.moment_xyz)}")
            return ", ".join(parts) or "(empty)"
        if isinstance(d, LineLoadDef):
            if d.q_xyz is not None:
                return f"q={tuple(d.q_xyz)} N/m"
            return f"{d.magnitude:.4g} N/m, dir={d.direction}"
        if isinstance(d, SurfaceLoadDef):
            kind = "pressure" if d.normal else "traction"
            return f"{d.magnitude:.4g} Pa ({kind})"
        if isinstance(d, GravityLoadDef):
            rho = f", ρ={d.density}" if d.density else ""
            return f"g={tuple(d.g)}{rho}"
        if isinstance(d, BodyLoadDef):
            return f"bf={tuple(d.force_per_volume)} N/m³"
        return "(unknown)"

    # ── Pattern state queries ──────────────────────────────────

    def active_patterns(self) -> set[str]:
        """Patterns whose checkboxes are currently checked."""
        from qtpy.QtCore import Qt
        active: set[str] = set()
        for name, item in self._pattern_items.items():
            if item.checkState(0) == Qt.CheckState.Checked:
                active.add(name)
        return active

    # ── Button handlers ────────────────────────────────────────

    def _show_all(self) -> None:
        from qtpy.QtCore import Qt
        self._suppress_signal = True
        for item in self._pattern_items.values():
            item.setCheckState(0, Qt.CheckState.Checked)
        self._suppress_signal = False
        if self._on_patterns_changed:
            self._on_patterns_changed(self.active_patterns())

    def _hide_all(self) -> None:
        from qtpy.QtCore import Qt
        self._suppress_signal = True
        for item in self._pattern_items.values():
            item.setCheckState(0, Qt.CheckState.Unchecked)
        self._suppress_signal = False
        if self._on_patterns_changed:
            self._on_patterns_changed(self.active_patterns())

    def _on_item_changed(self, item, _column) -> None:
        if self._suppress_signal:
            return
        if self._on_patterns_changed:
            self._on_patterns_changed(self.active_patterns())

    # ── External fem update ────────────────────────────────────

    def set_fem(self, fem) -> None:
        """Update the FEM reference (e.g. after re-meshing)."""
        self._fem = fem
        self._fem_warning.setVisible(
            self._fem is None and bool(self._loads.load_defs)
        )
