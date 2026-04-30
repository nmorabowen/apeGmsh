"""Modal dialog: pick diagram kind, stage, component, selector.

Phase 1 surface — Contour and Deformed Shape only. Each diagram kind
lives in a small spec entry so adding new kinds in later phases is a
one-line registration.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Iterable, Optional

from ..diagrams._base import DiagramSpec, NoDataError
from ..diagrams._contour import ContourDiagram
from ..diagrams._deformed_shape import DeformedShapeDiagram
from ..diagrams._fiber_section import FiberSectionDiagram
from ..diagrams._gauss_marker import GaussPointDiagram
from ..diagrams._layer_stack import LayerStackDiagram
from ..diagrams._line_force import LineForceDiagram
from ..diagrams._selectors import normalize as normalize_selector
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


def _line_force_default_style(component: str) -> LineForceStyle:
    """Default style for a line-force diagram.

    Bending moments default to ``flip_sign=True`` so the diagram
    renders on the tension side of the beam (sagging-positive
    convention universally used by structural engineers). Axial
    force, shear, and torsion keep the natural sign — those have no
    "tension side" tradition to follow.
    """
    is_bending_moment = component.startswith("bending_moment")
    return LineForceStyle(flip_sign=is_bending_moment)


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


# Maps each diagram kind to the Results-composite path whose
# ``available_components()`` should populate the Component combo.
# Derived from the subclass-level ``topology`` attribute so the dialog
# and the diagram cannot drift apart.
_KIND_TO_TOPOLOGY: dict[str, str] = {
    entry.kind_id: entry.diagram_class.topology for entry in _KINDS
}


def _components_for(results: Any, topology: str) -> list[str]:
    """Resolve ``available_components()`` against a stage-scoped Results."""
    try:
        if topology == "nodes":
            return sorted(results.nodes.available_components())
        if topology == "line_stations":
            return sorted(results.elements.line_stations.available_components())
        if topology == "fibers":
            return sorted(results.elements.fibers.available_components())
        if topology == "layers":
            return sorted(results.elements.layers.available_components())
        if topology == "gauss":
            return sorted(results.elements.gauss.available_components())
        if topology == "springs":
            return sorted(results.elements.springs.available_components())
    except Exception:
        return []
    return []


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

        # Topology — only relevant for Contour, where a
        # ContourStyle.topology field selects between nodal-scalar
        # rendering (point data) and element-constant Gauss
        # rendering (cell data, n_gp == 1). Shown for Contour, hidden
        # for every other kind so the form stays uncluttered.
        self._topology_label = QtWidgets.QLabel("Topology:")
        self._topology_combo = QtWidgets.QComboBox()
        self._topology_combo.addItem("Auto",  "auto")
        self._topology_combo.addItem("Nodes", "nodes")
        self._topology_combo.addItem("Gauss", "gauss")
        self._topology_combo.setToolTip(
            "Auto: prefer nodal data when both composites have the\n"
            "component; fall through to Gauss otherwise.\n"
            "Nodes: force the nodal-scalar path (point data).\n"
            "Gauss: force the element-constant path (cell data;\n"
            "requires n_gp == 1 per element)."
        )
        self._topology_combo.currentIndexChanged.connect(
            self._populate_components,
        )
        form.addRow(self._topology_label, self._topology_combo)

        # Component — editable combo populated from the chosen
        # (kind, stage) pair. Editable so the user can also type a
        # component name not yet present in the file.
        self._component_combo = QtWidgets.QComboBox()
        self._component_combo.setEditable(True)
        self._component_combo.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        self._component_combo.lineEdit().setPlaceholderText(
            "displacement_z, bending_moment_y, fiber_stress, …"
        )
        self._component_combo.setToolTip(
            "Component name. The list shows what's available for the\n"
            "selected kind + stage; you can also type a custom name."
        )
        form.addRow("Component:", self._component_combo)

        # Repopulate the combo whenever kind or stage changes.
        self._kind_combo.currentIndexChanged.connect(
            self._on_kind_changed,
        )
        self._stage_combo.currentIndexChanged.connect(
            self._populate_components,
        )

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

        # Initial visibility + component list for the default kind.
        self._update_topology_row_visibility()
        self._populate_components()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_kind_changed(self, *_args: Any) -> None:
        self._update_topology_row_visibility()
        self._populate_components()

    def _update_topology_row_visibility(self) -> None:
        """Show the Topology row only when Contour is the kind."""
        kind_entry: _KindEntry = self._kind_combo.currentData()
        is_contour = kind_entry is not None and kind_entry.kind_id == "contour"
        self._topology_label.setVisible(is_contour)
        self._topology_combo.setVisible(is_contour)

    def _populate_components(self, *_args: Any) -> None:
        """Refresh the Component combo from the current (kind, stage)."""
        kind_entry: _KindEntry = self._kind_combo.currentData()
        stage_id = self._stage_combo.currentData()
        if kind_entry is None or stage_id is None:
            return

        # Resolve which composite(s) to enumerate. Contour is special:
        # the user can override the class-default ("nodes") via the
        # Topology sub-combo. ``"auto"`` lists the union of nodes and
        # gauss components so the user sees everything reachable.
        topology = _KIND_TO_TOPOLOGY.get(kind_entry.kind_id)
        contour_topology: str | None = None
        if kind_entry.kind_id == "contour":
            contour_topology = self._topology_combo.currentData() or "auto"

        components: list[str] = []
        try:
            scoped = self._director.results.stage(stage_id)
        except Exception:
            scoped = None

        if scoped is not None:
            if contour_topology == "auto":
                # Union of nodes + gauss, sorted, deduplicated.
                union = set(_components_for(scoped, "nodes"))
                union.update(_components_for(scoped, "gauss"))
                components = sorted(union)
            elif contour_topology in ("nodes", "gauss"):
                components = _components_for(scoped, contour_topology)
            elif topology is not None:
                components = _components_for(scoped, topology)

        # Preserve the user's prior text only when the new list contains
        # it — otherwise it was a default from a different kind and we
        # must not carry it over (e.g. ``displacement_z`` showing up in
        # the field after the user picked Gauss point markers).
        prior = self._component_combo.currentText().strip()

        # Default heuristic: nodes-leaning kinds prefer displacement_z;
        # everything else takes the first available. Contour with
        # explicit gauss topology falls into the "first available"
        # bucket since displacement_z isn't a gauss quantity.
        prefers_disp_z = (
            topology == "nodes" and contour_topology in (None, "nodes", "auto")
        )

        self._component_combo.blockSignals(True)
        try:
            self._component_combo.clear()
            self._component_combo.addItems(components)
            preferred_default = (
                "displacement_z" if prefers_disp_z else
                (components[0] if components else "")
            )
            if prior and prior in components:
                self._component_combo.setCurrentText(prior)
            elif preferred_default in components:
                self._component_combo.setCurrentText(preferred_default)
            elif components:
                self._component_combo.setCurrentIndex(0)
            else:
                # Empty list for this (kind, stage) — clear the field
                # rather than leaving stale text from the previous kind.
                # The placeholder still hints at typing a custom name.
                self._component_combo.setEditText("")
        finally:
            self._component_combo.blockSignals(False)

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
        component = self._component_combo.currentText().strip()
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
        # Contour exposes a per-instance ``topology`` field; thread the
        # dialog's sub-combo selection into the constructed style.
        # Other kinds have no equivalent control to thread.
        if kind_entry.kind_id == "contour":
            chosen = self._topology_combo.currentData() or "auto"
            style = ContourStyle(
                cmap=style.cmap,
                clim=style.clim,
                opacity=style.opacity,
                show_edges=style.show_edges,
                show_scalar_bar=style.show_scalar_bar,
                topology=chosen,
            )
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

        try:
            self._director.registry.add(diagram)
        except NoDataError as exc:
            # The diagram's attach() raised because there's no slab
            # data for this (component, stage, selector). Surface the
            # specific message so the user knows what to change.
            self._show_error(f"No data to render: {exc}")
            return False
        except Exception as exc:
            self._show_error(f"Could not attach diagram: {exc}")
            return False
        return True

    def _show_error(self, msg: str) -> None:
        QtWidgets, _ = _qt()
        box = QtWidgets.QMessageBox(self._dlg)
        box.setIcon(QtWidgets.QMessageBox.Warning)
        box.setWindowTitle("Add Diagram")
        box.setText(msg)
        box.exec_()
