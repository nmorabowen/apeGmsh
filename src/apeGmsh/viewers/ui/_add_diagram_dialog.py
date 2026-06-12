"""Modal dialog: pick diagram kind, stage, component, selector.

The available kinds, their labels, classes, and default-style
factories come from the declarative kind registry
(``..diagrams._kinds``, ADR 0058 S0) — each diagram's module registers
itself, so adding a kind needs no edit here.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from ..diagrams._base import DiagramSpec, NoDataError
from ..diagrams._kinds import DiagramKindDef, all_kinds
from ..diagrams._selectors import normalize as normalize_selector
from ..diagrams._styles import ContourStyle

if TYPE_CHECKING:
    from apeGmsh.cuts import (
        PreflightReport, SectionCutDef, SectionSweepDef,
    )

    from ..diagrams._director import ResultsDirector


SECTION_CUT_KIND_ID = "section_cut"


def _qt():
    from qtpy import QtWidgets, QtCore
    return QtWidgets, QtCore


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

    def __init__(
        self,
        director: "ResultsDirector",
        parent: Any = None,
        *,
        initial_kind: Optional[str] = None,
    ) -> None:
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

        # Kind. Pre-flight every kind against every stage so we can mark
        # kinds whose topology has no data anywhere in the file. The
        # picker stays selectable (editable component combo lets users
        # type custom names regardless), but the suffix tells them up
        # front not to expect a populated dropdown.
        self._kinds_without_data: set[str] = self._compute_kinds_without_data(
            director,
        )
        self._kind_combo = QtWidgets.QComboBox()
        for k in all_kinds():
            label = k.label
            if k.kind_id in self._kinds_without_data:
                label = f"{k.label} — no data"
            self._kind_combo.addItem(label, k)
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
        # rendering (point data) and Gauss-extrapolated rendering.
        # Shown for Contour, hidden for every other kind so the form
        # stays uncluttered.
        self._topology_label = QtWidgets.QLabel("Topology:")
        self._topology_combo = QtWidgets.QComboBox()
        self._topology_combo.addItem("Nodes", "nodes")
        self._topology_combo.addItem("Gauss", "gauss")
        self._topology_combo.setToolTip(
            "Nodes: nodal-scalar path (point data; one value per\n"
            "global node — e.g. displacement, reaction).\n"
            "Gauss: read GP values from the elements composite. The\n"
            "Averaging row controls whether jumps at element\n"
            "boundaries are smoothed."
        )
        self._topology_combo.currentIndexChanged.connect(
            self._populate_components,
        )
        self._topology_combo.currentIndexChanged.connect(
            self._update_averaging_row_visibility,
        )
        form.addRow(self._topology_label, self._topology_combo)

        # Averaging — only relevant for Contour + Gauss topology.
        self._averaging_label = QtWidgets.QLabel("Averaging:")
        self._averaging_combo = QtWidgets.QComboBox()
        self._averaging_combo.addItem("Averaged", "averaged")
        self._averaging_combo.addItem("Discrete", "discrete")
        self._averaging_combo.setToolTip(
            "Averaged: extrapolate GP values to corners and average\n"
            "across elements that share a node — smooth contours.\n"
            "Discrete: each element keeps its own corner values\n"
            "(no cross-element averaging) — boundary jumps visible.\n"
            "For one-GP elements, Discrete paints flat cell colour."
        )
        form.addRow(self._averaging_label, self._averaging_combo)

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

        # Preset — filters to the current kind. The "(default)" entry
        # leaves the dialog's existing default-style behaviour intact.
        self._preset_combo = QtWidgets.QComboBox()
        self._preset_combo.setToolTip(
            "Reuse a saved style preset for this kind. (default) keeps "
            "the dialog's built-in defaults."
        )
        form.addRow("Preset:", self._preset_combo)
        self._populate_presets()

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

        # ===== Section-cut rows (visible only when kind = section_cut) =====
        # Loaded state — populated by the file picker / textChanged, or
        # by the h5 dropdown when Source=h5. ``_cut_loaded`` is either
        # None, a ``SectionCutDef``, or a ``SectionSweepDef``; the OK
        # handler branches on type.
        self._cut_loaded: "SectionCutDef | SectionSweepDef | None" = None
        self._cut_load_error: Optional[str] = None

        # Source — v4: pick between the Phase-4 file picker and the new
        # "browse cuts persisted in model.h5" dropdown. File picker is
        # the default so the Phase-4 flow remains the dialog's first
        # impression on existing users.
        self._cut_source_combo = QtWidgets.QComboBox()
        self._cut_source_combo.addItem("File (.pkl)", "file")
        self._cut_source_combo.addItem("In model.h5", "h5")
        self._cut_source_combo.setToolTip(
            "Pick the cut's source: a pickled .pkl/.pkl.gz file, or a\n"
            "cut/sweep persisted inside the model.h5 below."
        )
        self._cut_source_combo.currentIndexChanged.connect(
            self._on_cut_source_changed,
        )
        form.addRow("Source:", self._cut_source_combo)

        self._cut_file_edit = QtWidgets.QLineEdit()
        self._cut_file_edit.setPlaceholderText(
            "/path/to/cut.pkl or .pkl.gz"
        )
        self._cut_file_edit.setToolTip(
            "Pickled SectionCutDef or SectionSweepDef "
            "(.pkl or .pkl.gz)."
        )
        self._cut_file_browse = QtWidgets.QPushButton("Browse…")
        self._cut_file_browse.clicked.connect(self._on_cut_file_browse)
        self._cut_file_edit.textChanged.connect(
            self._on_cut_file_text_changed,
        )
        self._cut_file_row = self._make_path_row(
            self._cut_file_edit, self._cut_file_browse,
        )
        form.addRow("File:", self._cut_file_row)

        # h5-source dropdown — populated by ``_populate_h5_cut_dropdown``
        # whenever model.h5 changes (or source switches to h5). Each
        # entry's ``itemData`` carries the live ``SectionCutDef`` /
        # ``SectionSweepDef`` instance, so dropdown selection alone is
        # enough to populate ``_cut_loaded``.
        self._cut_h5_dropdown = QtWidgets.QComboBox()
        self._cut_h5_dropdown.setToolTip(
            "Cut or sweep persisted under /opensees/cuts/ or\n"
            "/opensees/sweeps/ in the model.h5 above."
        )
        self._cut_h5_dropdown.currentIndexChanged.connect(
            self._on_cut_h5_dropdown_changed,
        )
        form.addRow("Cut:", self._cut_h5_dropdown)

        self._cut_model_h5_edit = QtWidgets.QLineEdit()
        # ADR 0026 PR-stretch — director no longer exposes `.model_h5`
        # as a stored field.  Prefill the dialog from the bound
        # Results' path via the canonical orientation probe; the
        # dialog gives the same UX as before without the field.
        from ..data._h5_probe import resolve_orientation_source
        prefill = resolve_orientation_source(director.results)
        if prefill is not None:
            self._cut_model_h5_edit.setText(str(prefill))
        self._cut_model_h5_edit.setPlaceholderText(
            "/path/to/model.h5 (defaults to viewer's bound model.h5)"
        )
        self._cut_model_h5_edit.setToolTip(
            "Phase 8.6+ model.h5 carrying the FEM↔OpenSees-tag bridge "
            "the cut was built against."
        )
        self._cut_model_h5_browse = QtWidgets.QPushButton("Browse…")
        self._cut_model_h5_browse.clicked.connect(
            self._on_cut_model_h5_browse,
        )
        self._cut_model_h5_edit.textChanged.connect(
            self._on_cut_model_h5_text_changed,
        )
        self._cut_model_h5_row = self._make_path_row(
            self._cut_model_h5_edit, self._cut_model_h5_browse,
        )
        form.addRow("Model.h5:", self._cut_model_h5_row)

        # Preflight status — colored dot + short label
        self._cut_preflight_status = QtWidgets.QLabel(
            "(load a file to preflight)"
        )
        self._cut_preflight_status.setTextFormat(QtCore.Qt.RichText)
        form.addRow("Preflight:", self._cut_preflight_status)

        # Preflight full report — multi-line, monospaced
        self._cut_preflight_summary = QtWidgets.QPlainTextEdit()
        self._cut_preflight_summary.setReadOnly(True)
        self._cut_preflight_summary.setFixedHeight(130)
        self._cut_preflight_summary.setStyleSheet(
            "font-family: Consolas, 'Courier New', monospace; "
            "font-size: 9pt;"
        )
        form.addRow("", self._cut_preflight_summary)
        # Track the form for later row-show/hide via ``labelForField``.
        self._form = form

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
        self._ok_button = bb.button(QtWidgets.QDialogButtonBox.Ok)

        # Pre-select the requested kind (used by the inline 2×4 picker
        # in OutlineTree which jumps straight from "click kind" to the
        # configuration dialog).
        if initial_kind is not None:
            for i in range(self._kind_combo.count()):
                entry = self._kind_combo.itemData(i)
                if entry is not None and entry.kind_id == initial_kind:
                    self._kind_combo.setCurrentIndex(i)
                    break

        # Initial visibility + component list for the default kind.
        self._update_section_cut_visibility()
        self._update_topology_row_visibility()
        self._update_averaging_row_visibility()
        self._populate_components()
        self._update_ok_enabled()

    # ------------------------------------------------------------------
    # Pre-flight: which kinds have no recorded data anywhere?
    # ------------------------------------------------------------------

    def _compute_kinds_without_data(
        self, director: "ResultsDirector",
    ) -> set[str]:
        """Return ``kind_id`` of every kind whose topology is empty in
        every stage of this Results file.

        Contour is treated specially: it can read either composite, so
        we mark it without-data only if both ``nodes`` and ``gauss``
        composites are empty in every stage.
        """
        out: set[str] = set()
        try:
            stages = list(director.stages())
        except Exception:
            return out
        if not stages:
            return out

        for entry in all_kinds():
            topology = entry.data_topology
            if topology is None:
                continue
            has_any = False
            for s in stages:
                sid = getattr(s, "id", None)
                if sid is None:
                    continue
                try:
                    scoped = director.results.stage(sid)
                except Exception:
                    continue
                if entry.kind_id == "contour":
                    found = (
                        _components_for(scoped, "nodes")
                        or _components_for(scoped, "gauss")
                    )
                elif entry.kind_id == "reactions":
                    nodal = set(_components_for(scoped, "nodes"))
                    found = any(
                        c in nodal for c in (
                            "reaction_force_x", "reaction_force_y",
                            "reaction_force_z", "reaction_moment_x",
                            "reaction_moment_y", "reaction_moment_z",
                        )
                    )
                else:
                    found = _components_for(scoped, topology)
                if found:
                    has_any = True
                    break
            if not has_any:
                out.add(entry.kind_id)
        return out

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_kind_changed(self, *_args: Any) -> None:
        self._update_section_cut_visibility()
        self._update_topology_row_visibility()
        self._update_averaging_row_visibility()
        self._populate_components()
        self._populate_presets()
        self._update_ok_enabled()

    def _populate_presets(self) -> None:
        """Refresh the Preset combo against the current kind.

        Inserts a leading ``(default)`` entry so users can pick "no
        preset" without an extra control. Preset names that fail to
        load (corrupt JSON / removed kind) are skipped silently.
        """
        from ..diagrams._style_presets import default_store
        kind_entry: DiagramKindDef = self._kind_combo.currentData()
        kind_id = kind_entry.kind_id if kind_entry is not None else None
        self._preset_combo.blockSignals(True)
        try:
            self._preset_combo.clear()
            self._preset_combo.addItem("(default)", None)
            if kind_id is not None:
                try:
                    names = default_store().list_for_kind(kind_id)
                except Exception:
                    names = []
                for name in names:
                    self._preset_combo.addItem(name, name)
        finally:
            self._preset_combo.blockSignals(False)

    def _update_topology_row_visibility(self) -> None:
        """Show the Topology row only when Contour is the kind."""
        kind_entry: DiagramKindDef = self._kind_combo.currentData()
        is_contour = kind_entry is not None and kind_entry.kind_id == "contour"
        self._topology_label.setVisible(is_contour)
        self._topology_combo.setVisible(is_contour)

    def _update_averaging_row_visibility(self) -> None:
        """Show the Averaging row only for Contour + Gauss topology."""
        kind_entry: DiagramKindDef = self._kind_combo.currentData()
        is_contour = kind_entry is not None and kind_entry.kind_id == "contour"
        is_gauss = (
            self._topology_combo.currentData() == "gauss"
        )
        show = is_contour and is_gauss
        self._averaging_label.setVisible(show)
        self._averaging_combo.setVisible(show)

    def _populate_components(self, *_args: Any) -> None:
        """Refresh the Component combo from the current (kind, stage)."""
        kind_entry: DiagramKindDef = self._kind_combo.currentData()
        stage_id = self._stage_combo.currentData()
        if kind_entry is None or stage_id is None:
            return

        # Resolve which composite to enumerate. Contour is special:
        # the user picks "nodes" or "gauss" via the Topology sub-combo.
        topology = kind_entry.data_topology
        contour_topology: str | None = None
        if kind_entry.kind_id == "contour":
            contour_topology = self._topology_combo.currentData() or "nodes"

        components: list[str] = []
        try:
            scoped = self._director.results.stage(stage_id)
        except Exception:
            scoped = None

        is_reactions = kind_entry.kind_id == "reactions"
        if scoped is not None:
            if is_reactions:
                # Reactions Data list: resultant + per-axis force
                # options, filtered to axes the file actually records.
                nodal = set(_components_for(scoped, "nodes"))
                if any(
                    c in nodal for c in (
                        "reaction_force_x", "reaction_force_y",
                        "reaction_force_z", "reaction_moment_x",
                        "reaction_moment_y", "reaction_moment_z",
                    )
                ):
                    components.append("reactions")
                for opt, force_comp in (
                    ("reaction_x", "reaction_force_x"),
                    ("reaction_y", "reaction_force_y"),
                    ("reaction_z", "reaction_force_z"),
                ):
                    if force_comp in nodal:
                        components.append(opt)
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
        # Reactions opts out — its first entry ("reactions" → resultant)
        # is the desired default.
        prefers_disp_z = (
            topology == "nodes"
            and contour_topology in (None, "nodes")
            and not is_reactions
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
                self._component_combo.setEditText("")
                # Replace the generic example placeholder with a
                # specific reason so the user knows whether the file
                # has no such data anywhere or just not in this stage.
                self._update_empty_placeholder(kind_entry, stage_id)
        finally:
            self._component_combo.blockSignals(False)
        # Restore the generic example placeholder when we did populate
        # something — otherwise the empty-state hint persists across
        # subsequent populations.
        if components:
            self._component_combo.lineEdit().setPlaceholderText(
                "displacement_z, bending_moment_y, fiber_stress, …"
            )

    def _update_empty_placeholder(
        self, kind_entry: DiagramKindDef, stage_id: Any,
    ) -> None:
        """Set a placeholder explaining why the Component combo is empty.

        Distinguishes "no data of this topology anywhere in the file"
        from "no data in this particular stage" — same empty combo, very
        different fix on the user's side (record a different recorder vs
        switch stages).
        """
        topology = kind_entry.data_topology or ""
        if kind_entry.kind_id in self._kinds_without_data:
            text = f"(no {topology} data in file)"
        else:
            text = f"(no {topology} data in selected stage)"
        self._component_combo.lineEdit().setPlaceholderText(text)

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
    # Section-cut layout + handlers
    # ------------------------------------------------------------------

    def _make_path_row(self, line_edit: Any, browse_button: Any) -> Any:
        """Pack a QLineEdit + Browse button into a horizontal container."""
        QtWidgets, _ = _qt()
        container = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(container)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(line_edit, 1)
        h.addWidget(browse_button, 0)
        return container

    def _set_row_visible(self, field_widget: Any, visible: bool) -> None:
        """Hide or show a QFormLayout row by toggling label + field."""
        label = self._form.labelForField(field_widget)
        if label is not None:
            label.setVisible(visible)
        field_widget.setVisible(visible)

    def _is_section_cut_kind(self) -> bool:
        entry = self._kind_combo.currentData()
        return entry is not None and entry.kind_id == SECTION_CUT_KIND_ID

    def _is_h5_source(self) -> bool:
        """True iff section_cut is the kind AND Source=h5."""
        return (
            self._is_section_cut_kind()
            and self._cut_source_combo.currentData() == "h5"
        )

    def _update_section_cut_visibility(self) -> None:
        """Toggle rows between the Results-data flow and the section-cut flow."""
        is_cut = self._is_section_cut_kind()
        # Hide every Results-data row when section_cut is picked.
        for field in (
            self._stage_combo,
            self._component_combo,
            self._preset_combo,
            self._selector_kind,
            self._selector_name,
        ):
            self._set_row_visible(field, not is_cut)
        # Topology / averaging have their own conditional visibility on
        # the Results path; for section_cut they must be hidden regardless.
        if is_cut:
            self._topology_label.setVisible(False)
            self._topology_combo.setVisible(False)
            self._averaging_label.setVisible(False)
            self._averaging_combo.setVisible(False)
        # Source combo visible whenever section_cut is picked.
        self._set_row_visible(self._cut_source_combo, is_cut)
        # File row vs h5 dropdown — mutually exclusive within the
        # section-cut flow. _is_h5_source already guards on is_cut.
        h5_mode = self._is_h5_source()
        self._set_row_visible(self._cut_file_row, is_cut and not h5_mode)
        self._set_row_visible(self._cut_h5_dropdown, h5_mode)
        # Model.h5 + preflight always visible in the section-cut flow —
        # they're needed for both load paths.
        for field in (
            self._cut_model_h5_row,
            self._cut_preflight_status,
            self._cut_preflight_summary,
        ):
            self._set_row_visible(field, is_cut)

    # ── File / model.h5 pickers ─────────────────────────────────────

    def _on_cut_file_browse(self) -> None:
        QtWidgets, _ = _qt()
        path, _filter = QtWidgets.QFileDialog.getOpenFileName(
            self._dlg,
            "Select pickled section cut",
            self._cut_file_edit.text() or "",
            "Pickled cut (*.pkl *.pkl.gz);;All files (*.*)",
        )
        if path:
            self._cut_file_edit.setText(path)

    def _on_cut_model_h5_browse(self) -> None:
        QtWidgets, _ = _qt()
        path, _filter = QtWidgets.QFileDialog.getOpenFileName(
            self._dlg,
            "Select model.h5",
            self._cut_model_h5_edit.text() or "",
            "model.h5 (*.h5);;All files (*.*)",
        )
        if path:
            self._cut_model_h5_edit.setText(path)

    def _on_cut_file_text_changed(self, _text: str) -> None:
        self._load_cut_from_path()
        self._run_dialog_preflight()
        self._update_ok_enabled()

    def _on_cut_model_h5_text_changed(self, _text: str) -> None:
        # When Source=h5, the dropdown is sourced from this path —
        # re-enumerate /opensees/cuts/ + /opensees/sweeps/ whenever the
        # path text changes. In file mode this is a no-op aside from
        # the preflight rerun (the file picker drives _cut_loaded).
        if self._is_h5_source():
            self._populate_h5_cut_dropdown()
        self._run_dialog_preflight()
        self._update_ok_enabled()

    # ── Source toggle / h5 dropdown ─────────────────────────────────

    def _on_cut_source_changed(self, *_args: Any) -> None:
        """Swap row visibility, reset loaded state, re-run preflight."""
        # Switching sources invalidates whatever was loaded before —
        # the new source must drive a fresh selection.
        self._cut_loaded = None
        self._cut_load_error = None
        self._update_section_cut_visibility()
        if self._is_h5_source():
            self._populate_h5_cut_dropdown()
        else:
            # File mode — reload from the current path text (may be
            # empty, in which case _load_cut_from_path is a no-op).
            self._load_cut_from_path()
        self._run_dialog_preflight()
        self._update_ok_enabled()

    def _on_cut_h5_dropdown_changed(self, *_args: Any) -> None:
        """User picked a different entry in the h5 cut dropdown."""
        if not self._is_h5_source():
            return
        data = self._cut_h5_dropdown.currentData()
        self._cut_load_error = None
        # ``data`` is either a SectionCutDef, a SectionSweepDef, or
        # ``None`` (placeholder entry: "(no cuts in this h5)", "(set
        # Model.h5 first)", or a load-failure marker).
        self._cut_loaded = data if data is not None else None
        self._run_dialog_preflight()
        self._update_ok_enabled()

    def _populate_h5_cut_dropdown(self) -> None:
        """Re-enumerate /opensees/cuts/ + /opensees/sweeps/ from model_h5.

        Blocks signals during the rebuild so per-item ``addItem`` calls
        don't fire the change slot mid-populate. After the rebuild, we
        explicitly call ``_on_cut_h5_dropdown_changed`` once so the
        current selection's data lands in ``_cut_loaded``.
        """
        self._cut_h5_dropdown.blockSignals(True)
        try:
            self._cut_h5_dropdown.clear()
            model_h5_text = self._cut_model_h5_edit.text().strip()
            if not model_h5_text:
                self._cut_h5_dropdown.addItem(
                    "(set Model.h5 first)", None,
                )
                return
            try:
                from apeGmsh.cuts import read_cuts_and_sweeps
                cuts, sweeps = read_cuts_and_sweeps(model_h5_text)
            except Exception as exc:
                self._cut_h5_dropdown.addItem(
                    f"(load failed: {exc})", None,
                )
                return
            if not cuts and not sweeps:
                self._cut_h5_dropdown.addItem(
                    "(no cuts persisted in this model.h5)", None,
                )
                return
            for i, cut in enumerate(cuts):
                cut_name = f"cut_{i}"
                display_label = cut.label or cut_name
                self._cut_h5_dropdown.addItem(
                    f"{display_label} ({cut_name})", cut,
                )
            for i, sweep in enumerate(sweeps):
                sweep_name = f"sweep_{i}"
                self._cut_h5_dropdown.addItem(
                    f"{sweep_name} (sweep, {len(sweep)} cuts)", sweep,
                )
        finally:
            self._cut_h5_dropdown.blockSignals(False)
        # Sync loaded state to whatever is now selected.
        self._on_cut_h5_dropdown_changed()

    # ── Load + preflight ────────────────────────────────────────────

    def _load_cut_from_path(self) -> None:
        """Attempt to load the pickle at the current path.

        Accepts both ``SectionCutDef`` and ``SectionSweepDef`` pickles —
        tries the cut loader first, falls back to the sweep loader on
        a type-mismatch. Stores either the loaded object or a short
        error string for the preflight panel to surface.
        """
        from pathlib import Path

        from apeGmsh.cuts import SectionCutDef, SectionSweepDef

        self._cut_loaded = None
        self._cut_load_error = None
        path_text = self._cut_file_edit.text().strip()
        if not path_text:
            return
        path = Path(path_text)
        if not path.exists():
            self._cut_load_error = f"File not found: {path}"
            return
        try:
            self._cut_loaded = SectionCutDef.load_pickle(path)
            return
        except TypeError:
            pass
        except Exception as exc:
            self._cut_load_error = f"Failed to load pickle: {exc}"
            return
        try:
            self._cut_loaded = SectionSweepDef.load_pickle(path)
        except Exception as exc:
            self._cut_load_error = (
                f"Not a SectionCutDef or SectionSweepDef pickle: {exc}"
            )

    def _run_dialog_preflight(self) -> None:
        """Run preflight against the current (cut, fem, model_h5).

        Updates the colored status label and the multi-line summary
        widget. Called from text-change signals on either path field
        — debouncing isn't needed (load+preflight is microseconds for
        typical cuts).
        """
        if self._cut_load_error is not None:
            self._set_preflight_state(
                "error",
                self._cut_load_error,
                "",
            )
            return
        if self._cut_loaded is None:
            self._set_preflight_state(
                "neutral",
                "(load a file to preflight)",
                "",
            )
            return

        fem = self._director.results.fem
        if fem is None:
            self._set_preflight_state(
                "error",
                "Director has no FEMData bound — cannot preflight.",
                "",
            )
            return

        model_h5_text = self._cut_model_h5_edit.text().strip() or None
        try:
            reports = self._preflight_dispatch(model_h5=model_h5_text)
        except Exception as exc:
            self._set_preflight_state(
                "error",
                f"Preflight failed: {exc}",
                "",
            )
            return

        if not reports:
            self._set_preflight_state(
                "error",
                "Preflight returned no reports.",
                "",
            )
            return

        n_errs = sum(len(r.errors) for r in reports)
        n_warns = sum(len(r.warnings) for r in reports)
        n_cuts = len(reports)
        summary = "\n\n".join(str(r) for r in reports)
        if n_errs:
            status_text = (
                f"ERRORS ({n_errs})" if n_cuts == 1
                else f"{n_cuts} cuts: "
                     f"{n_cuts - sum(1 for r in reports if not r.ok)} with errors"
            )
            self._set_preflight_state("error", status_text, summary)
        elif n_warns:
            status_text = (
                f"WARNINGS ({n_warns})" if n_cuts == 1
                else f"{n_cuts} cuts: "
                     f"{sum(1 for r in reports if r.warnings)} with warnings"
            )
            self._set_preflight_state("warning", status_text, summary)
        else:
            status_text = "OK" if n_cuts == 1 else f"{n_cuts} cuts: all ok"
            self._set_preflight_state("ok", status_text, summary)

    def _preflight_dispatch(
        self, *, model_h5: Optional[str],
    ) -> "tuple[PreflightReport, ...]":
        """Run preflight on the loaded cut or sweep, returning a tuple."""
        from apeGmsh.cuts import SectionSweepDef

        cut = self._cut_loaded
        assert cut is not None
        fem = self._director.results.fem
        # Caller (`_run_dialog_preflight`) already shows an error and
        # returns when fem is None; the dispatch path only fires once
        # that guard passes. Assert to make the narrowing visible to
        # mypy.
        assert fem is not None
        if isinstance(cut, SectionSweepDef):
            return cut.preflight(fem, model_h5=model_h5)
        return (cut.preflight(fem, model_h5=model_h5),)

    def _set_preflight_state(
        self, severity: str, status_text: str, summary: str,
    ) -> None:
        """Apply colored dot + text to the status label and summary widget."""
        dot = {
            "ok":      "<span style='color:#3cb44b;'>●</span>",
            "warning": "<span style='color:#f58231;'>●</span>",
            "error":   "<span style='color:#e6194b;'>●</span>",
            "neutral": "<span style='color:#888888;'>○</span>",
        }.get(severity, "<span style='color:#888888;'>○</span>")
        self._cut_preflight_status.setText(f"{dot}&nbsp;{status_text}")
        self._cut_preflight_summary.setPlainText(summary)
        self._cut_preflight_severity = severity

    # ── OK gating ───────────────────────────────────────────────────

    def _update_ok_enabled(self) -> None:
        """Section-cut path: enable OK only when a clean preflight loaded."""
        if not self._is_section_cut_kind():
            self._ok_button.setEnabled(True)
            return
        ok = (
            self._cut_loaded is not None
            and self._cut_load_error is None
            and getattr(self, "_cut_preflight_severity", "neutral")
                in ("ok", "warning")
        )
        self._ok_button.setEnabled(ok)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self) -> bool:
        QtWidgets, _ = _qt()
        result = self._dlg.exec_()
        if result != QtWidgets.QDialog.Accepted:
            return False

        kind_entry: DiagramKindDef = self._kind_combo.currentData()
        if kind_entry is not None and kind_entry.kind_id == SECTION_CUT_KIND_ID:
            return self._run_section_cut()

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

        # Style: prefer a chosen preset; otherwise fall back to the
        # kind's per-component default. Preset takes priority over the
        # Topology sub-combo for Contour — saving a preset captures
        # the topology field too, so loading restores it intact.
        preset_name = self._preset_combo.currentData()
        style: Any = None
        if preset_name is not None:
            from ..diagrams._style_presets import default_store
            try:
                _kind_id, style = default_store().load(preset_name)
            except Exception as exc:
                self._show_error(f"Failed to load preset: {exc}")
                return False
        if style is None:
            style = kind_entry.make_default_style(component)
            if kind_entry.kind_id == "contour":
                chosen_topology = (
                    self._topology_combo.currentData() or "nodes"
                )
                chosen_averaging = (
                    self._averaging_combo.currentData() or "averaged"
                )
                style = ContourStyle(
                    cmap=style.cmap,
                    clim=style.clim,
                    opacity=style.opacity,
                    show_edges=style.show_edges,
                    show_scalar_bar=style.show_scalar_bar,
                    fmt=style.fmt,
                    topology=chosen_topology,
                    averaging=chosen_averaging,
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

    def _run_section_cut(self) -> bool:
        """OK-handler branch for the section_cut kind.

        Re-runs preflight as belt-and-braces (catches any user edit
        between pick and click), then dispatches to
        ``director.add_section_cut`` or ``add_section_cut_sweep``.
        """
        from apeGmsh.cuts import SectionSweepDef

        if self._cut_loaded is None:
            self._show_error("No cut loaded — pick a .pkl file first.")
            return False

        # Re-preflight; the file or model.h5 path may have been edited
        # after the last text-change signal fired (paste, IME compose, …).
        self._run_dialog_preflight()
        if getattr(self, "_cut_preflight_severity", "neutral") not in (
            "ok", "warning",
        ):
            self._show_error(
                "Preflight reports errors — see the dialog summary. "
                "Re-pickle from a fresh from_planar_pg(...) and retry."
            )
            return False

        model_h5_text = self._cut_model_h5_edit.text().strip() or None
        layer_label = self._label_edit.text().strip() or None

        try:
            if isinstance(self._cut_loaded, SectionSweepDef):
                self._director.add_section_cut_sweep(
                    self._cut_loaded,
                    model_h5=model_h5_text,
                    label_prefix=layer_label,
                )
            else:
                self._director.add_section_cut(
                    self._cut_loaded,
                    model_h5=model_h5_text,
                    label=layer_label,
                )
        except Exception as exc:
            self._show_error(f"Could not add section cut: {exc}")
            return False
        return True

    def _show_error(self, msg: str) -> None:
        QtWidgets, _ = _qt()
        box = QtWidgets.QMessageBox(self._dlg)
        box.setIcon(QtWidgets.QMessageBox.Warning)
        box.setWindowTitle("Add Diagram")
        box.setText(msg)
        box.exec_()
