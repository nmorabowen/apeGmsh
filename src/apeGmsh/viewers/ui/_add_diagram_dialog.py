"""Modal dialog: pick diagram kind, stage, component, selector.

Phase 1 surface — Contour and Deformed Shape only. Each diagram kind
lives in a small spec entry so adding new kinds in later phases is a
one-line registration.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Iterable, Optional

from ..diagrams._base import DiagramSpec
from ..diagrams._contour import ContourDiagram
from ..diagrams._deformed_shape import DeformedShapeDiagram
from ..diagrams._fiber_section import FiberSectionDiagram
from ..diagrams._gauss_marker import GaussPointDiagram
from ..diagrams._layer_stack import LayerStackDiagram
from ..diagrams._line_force import LineForceDiagram
from ..diagrams._selectors import normalize_selector
from ..diagrams._spring_force import SpringForceDiagram
from ..diagrams._styles import (
    ContourStyle, DeformedShapeStyle, FiberSectionStyle,
    GaussMarkerStyle, LayerStackStyle, LineForceStyle,
    SpringForceStyle, VectorGlyphStyle,
)
from ..diagrams._vector_glyph import VectorGlyphDiagram

if TYPE_CHECKING:
    from ..diagrams._director import ResultsDirector


def _qt():
    from qtpy import QtWidgets, QtCore
    return QtWidgets, QtCore


# =====================================================================
# Available kinds (Phase 1)
# =====================================================================

@dataclass(frozen=True)
class _KindEntry:
    label: str
    kind_id: str
    diagram_class: type
    style_factory: Callable[[str], Any]

    def make_default_style(self, component: str) -> Any:
        return self.style_factory(component)


def _contour_default_style(_component: str) -> ContourStyle:
    return ContourStyle()


def _deformed_default_style(_component: str) -> DeformedShapeStyle:
    return DeformedShapeStyle()


def _line_force_default_style(_component: str) -> LineForceStyle:
    return LineForceStyle()


def _fiber_default_style(_component: str) -> FiberSectionStyle:
    return FiberSectionStyle()


def _layer_default_style(_component: str) -> LayerStackStyle:
    return LayerStackStyle()


def _vector_default_style(_component: str) -> VectorGlyphStyle:
    return VectorGlyphStyle()


def _gauss_default_style(_component: str) -> GaussMarkerStyle:
    return GaussMarkerStyle()


def _spring_default_style(_component: str) -> SpringForceStyle:
    return SpringForceStyle()


_KINDS: list[_KindEntry] = [
    _KindEntry(
        label="Contour",
        kind_id="contour",
        diagram_class=ContourDiagram,
        style_factory=_contour_default_style,
    ),
    _KindEntry(
        label="Deformed shape",
        kind_id="deformed_shape",
        diagram_class=DeformedShapeDiagram,
        style_factory=_deformed_default_style,
    ),
    _KindEntry(
        label="Line force diagram",
        kind_id="line_force",
        diagram_class=LineForceDiagram,
        style_factory=_line_force_default_style,
    ),
    _KindEntry(
        label="Fiber section",
        kind_id="fiber_section",
        diagram_class=FiberSectionDiagram,
        style_factory=_fiber_default_style,
    ),
    _KindEntry(
        label="Layer stack (shell)",
        kind_id="layer_stack",
        diagram_class=LayerStackDiagram,
        style_factory=_layer_default_style,
    ),
    _KindEntry(
        label="Vector glyph (arrows)",
        kind_id="vector_glyph",
        diagram_class=VectorGlyphDiagram,
        style_factory=_vector_default_style,
    ),
    _KindEntry(
        label="Gauss point markers",
        kind_id="gauss_marker",
        diagram_class=GaussPointDiagram,
        style_factory=_gauss_default_style,
    ),
    _KindEntry(
        label="Spring force",
        kind_id="spring_force",
        diagram_class=SpringForceDiagram,
        style_factory=_spring_default_style,
    ),
]


def kinds_available() -> list[_KindEntry]:
    """Return the registered kinds — used by the Diagrams tab to enable Add."""
    return list(_KINDS)


# =====================================================================
# Dialog
# =====================================================================

class AddDiagramDialog:
    """Construct a new diagram via a modal Qt dialog.

    Returns ``True`` from ``run()`` if the user clicked OK and a diagram
    was added to the registry.
    """

    def __init__(self, director: "ResultsDirector", parent: Any = None) -> None:
        QtWidgets, QtCore = _qt()
        self._director = director

        dlg = QtWidgets.QDialog(parent)
        dlg.setWindowTitle("Add Diagram")
        dlg.setModal(True)
        dlg.setMinimumWidth(440)
        self._dlg = dlg

        form = QtWidgets.QFormLayout(dlg)
        form.setContentsMargins(12, 12, 12, 12)
        form.setSpacing(8)

        # Kind
        self._kind_combo = QtWidgets.QComboBox()
        for k in _KINDS:
            self._kind_combo.addItem(k.label, k)
        form.addRow("Kind:", self._kind_combo)

        # Stage
        self._stage_combo = QtWidgets.QComboBox()
        for s in director.stages():
            label = (
                f"{getattr(s, 'name', s.id)} ({getattr(s, 'kind', '')})"
            )
            self._stage_combo.addItem(label, getattr(s, "id", str(s)))
        active_id = director.stage_id
        if active_id is not None:
            for i in range(self._stage_combo.count()):
                if self._stage_combo.itemData(i) == active_id:
                    self._stage_combo.setCurrentIndex(i)
                    break
        form.addRow("Stage:", self._stage_combo)

        # Component
        self._component_edit = QtWidgets.QLineEdit("displacement_x")
        self._component_edit.setToolTip(
            "Canonical component name, e.g. 'displacement_x', 'velocity_z'. "
            "For deformed shape, this is the diagram label; the warp uses "
            "displacement_x/y/z by default."
        )
        form.addRow("Component:", self._component_edit)

        # Selector
        self._selector_kind = QtWidgets.QComboBox()
        self._selector_kind.addItem("All nodes", "all")
        self._selector_kind.addItem("Physical group", "pg")
        self._selector_kind.addItem("Label", "label")
        self._selector_kind.currentIndexChanged.connect(self._on_selector_change)
        form.addRow("Selector:", self._selector_kind)

        self._selector_name = QtWidgets.QLineEdit()
        self._selector_name.setEnabled(False)
        self._selector_name.setPlaceholderText("(unused for All)")
        form.addRow("Selector name:", self._selector_name)

        # Optional label override
        self._label_edit = QtWidgets.QLineEdit()
        self._label_edit.setPlaceholderText("(optional)")
        form.addRow("Display label:", self._label_edit)

        # Buttons
        bb = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok
            | QtWidgets.QDialogButtonBox.Cancel
        )
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        form.addRow(bb)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_selector_change(self, _index: int) -> None:
        kind = self._selector_kind.currentData()
        if kind == "all":
            self._selector_name.setEnabled(False)
            self._selector_name.clear()
            self._selector_name.setPlaceholderText("(unused for All)")
        elif kind == "pg":
            self._selector_name.setEnabled(True)
            self._selector_name.setPlaceholderText("Physical group name")
        elif kind == "label":
            self._selector_name.setEnabled(True)
            self._selector_name.setPlaceholderText("Label name")

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self) -> bool:
        QtWidgets, _ = _qt()
        result = self._dlg.exec_()
        if result != QtWidgets.QDialog.Accepted:
            return False

        kind_entry: _KindEntry = self._kind_combo.currentData()
        component = self._component_edit.text().strip()
        if not component:
            return False

        stage_id = self._stage_combo.currentData()
        sel_kind = self._selector_kind.currentData()
        sel_name = self._selector_name.text().strip() or None

        selector_kwargs: dict[str, Any] = {"component": component}
        if sel_kind == "pg" and sel_name:
            selector_kwargs["pg"] = sel_name
        elif sel_kind == "label" and sel_name:
            selector_kwargs["label"] = sel_name

        try:
            selector = normalize_selector(**selector_kwargs)
        except Exception as exc:
            self._show_error(f"Invalid selector: {exc}")
            return False

        style = kind_entry.make_default_style(component)
        label = self._label_edit.text().strip() or None
        spec = DiagramSpec(
            kind=kind_entry.kind_id,
            selector=selector,
            style=style,
            stage_id=stage_id,
            label=label,
        )

        try:
            diagram = kind_entry.diagram_class(spec, self._director.results)
        except Exception as exc:
            self._show_error(f"Could not construct diagram: {exc}")
            return False

        # Switch to the spec's stage if the director isn't already there
        if stage_id and self._director.stage_id != stage_id:
            try:
                self._director.set_stage(stage_id)
            except Exception:
                pass

        self._director.registry.add(diagram)
        return True

    def _show_error(self, msg: str) -> None:
        QtWidgets, _ = _qt()
        box = QtWidgets.QMessageBox(self._dlg)
        box.setIcon(QtWidgets.QMessageBox.Warning)
        box.setWindowTitle("Add Diagram")
        box.setText(msg)
        box.exec_()
