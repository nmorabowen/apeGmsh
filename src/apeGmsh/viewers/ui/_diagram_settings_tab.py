"""Per-diagram settings tab — styling controls for the active selection.

Reads the currently-selected diagram from the Diagrams tab and shows a
kind-specific control set:

* ``contour`` — clim min/max, auto-fit button, opacity slider, cmap combo
* ``deformed_shape`` — scale slider, show-undeformed toggle

When no diagram is selected, an empty-state message is shown.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional

from ..diagrams._base import Diagram

if TYPE_CHECKING:
    from ..diagrams._director import ResultsDirector


def _qt():
    from qtpy import QtWidgets, QtCore
    return QtWidgets, QtCore


_CMAP_PRESETS = [
    "viridis", "plasma", "cividis", "magma", "inferno",
    "coolwarm", "RdBu", "Spectral", "turbo", "jet",
]


class DiagramSettingsTab:
    """Settings panel for the diagram selected in the Diagrams tab.

    Subscribes to the Director's ``on_diagrams_changed`` so the
    panel refreshes when the selected diagram is removed or replaced.
    The selection itself is set by ``set_selected(diagram)`` from the
    Diagrams tab when the user clicks a row.
    """

    def __init__(self, director: "ResultsDirector") -> None:
        QtWidgets, _ = _qt()
        self._director = director
        # ── Mode state — exactly one of three is active ─────────────
        # ``_show_stack``: render every registry diagram as a stack
        # of QGroupBox cards (the v2 pivot — clicking the outline's
        # Diagram 1 row enters this mode).
        # ``_create_new``: render the + Add layer creation form.
        # ``_selected``  : legacy single-layer edit mode (kept for
        # callers that still call ``set_selected``).
        self._show_stack: bool = False
        self._create_new: bool = False
        self._selected: Optional[Diagram] = None
        # Cached kind catalog — built lazily from the bound Results
        # the first time creation mode renders. Catalog is per-file
        # so safe to memoize.
        self._kind_catalog: Any = None

        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(6, 6, 6, 6)

        title_row = QtWidgets.QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title_row.setSpacing(6)
        self._title = QtWidgets.QLabel("No diagram selected.")
        font = self._title.font()
        font.setBold(True)
        self._title.setFont(font)
        title_row.addWidget(self._title)
        title_row.addStretch(1)
        self._btn_add_layer = QtWidgets.QPushButton("+ Add layer")
        self._btn_add_layer.setFlat(True)
        self._btn_add_layer.setToolTip("Add a new diagram layer")
        self._btn_add_layer.clicked.connect(self._on_add_layer_clicked)
        title_row.addWidget(self._btn_add_layer)
        layout.addLayout(title_row)

        self._content = QtWidgets.QWidget()
        self._content_layout = QtWidgets.QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(0, 6, 0, 0)
        layout.addWidget(self._content, stretch=1)

        empty_hint = QtWidgets.QLabel(
            "Click a Catalog entry to configure a new diagram, or "
            "select an active diagram to edit it."
        )
        empty_hint.setWordWrap(True)
        # Color + italic come from the theme stylesheet via the
        # DiagramSettingsEmptyHint objectName so the hint recolors
        # with the active palette.
        empty_hint.setObjectName("DiagramSettingsEmptyHint")
        layout.addWidget(empty_hint)
        self._empty_hint = empty_hint

        self._widget = widget

        director.subscribe_diagrams(self._on_diagrams_changed)
        # Geometry events fire on any state change (geometry list,
        # active geometry, per-geometry composition list, active
        # composition, rename, layer membership) — one subscription
        # rebuilds the stack when relevant.
        self._unsub_compositions = director.geometries.subscribe(
            self._on_compositions_changed,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def widget(self):
        return self._widget

    def set_selected(self, diagram: Optional[Diagram]) -> None:
        """Legacy single-layer edit mode. v2 prefers :meth:`show_stack`."""
        self._selected = diagram
        if diagram is not None:
            self._create_new = False
            self._show_stack = False
        self._rebuild()

    def set_create_new(self, active: bool) -> None:
        """Toggle the new-layer creation card.

        When active *and* stack mode is on, the creation form renders
        as an extra card at the bottom of the stack (preserving the
        existing layer cards above). When active without stack mode
        (e.g. from idle), a standalone creation form is shown.
        """
        self._create_new = bool(active)
        if active:
            self._selected = None
        self._rebuild()

    def show_stack(self) -> None:
        """Render every active diagram as a stacked card (v2 default mode)."""
        self._show_stack = True
        self._selected = None
        # Note: don't toggle ``_create_new`` here — caller may have
        # arrived via ``+ Add layer`` and want the pending card kept.
        self._rebuild()

    def set_idle(self) -> None:
        """Drop into the empty-hint state (no mode active)."""
        self._show_stack = False
        self._create_new = False
        self._selected = None
        self._rebuild()

    def _on_add_layer_clicked(self) -> None:
        """+ Add layer → ensure a composition exists in the active
        Geometry, then enter stack mode with a pending creation card.
        """
        geom_mgr = self._director.geometries
        geom = geom_mgr.active
        if geom is None:
            return
        comp_mgr = geom.compositions
        if not comp_mgr.active_accepts_layers:
            comp_mgr.add(name="Diagram", make_active=True)
        self.show_stack()
        self.set_create_new(True)

    def _ensure_catalog(self) -> Any:
        if self._kind_catalog is None:
            from ..diagrams._kind_catalog import build_catalog
            try:
                self._kind_catalog = build_catalog(self._director)
            except Exception:
                self._kind_catalog = []
        return self._kind_catalog

    # ------------------------------------------------------------------
    # Internal — rebuild content for the selection
    # ------------------------------------------------------------------

    def _on_diagrams_changed(self) -> None:
        # Stack mode: rebuild on every registry change so cards
        # appear/disappear as layers are added/removed.
        if self._show_stack:
            self._rebuild()
            return
        # Legacy single-select: drop the reference if removed.
        if self._selected is None:
            return
        active = self._director.registry.diagrams()
        if self._selected not in active:
            self._selected = None
            self._rebuild()

    def _on_compositions_changed(self) -> None:
        # Active composition changed (or one was renamed) — refresh
        # the title bar / cards if we're in stack mode.
        if self._show_stack:
            self._rebuild()

    def _rebuild(self) -> None:
        QtWidgets, _ = _qt()
        # Clear content layout — recursive teardown so nested
        # QFormLayout / QHBoxLayout children don't leak across modes.
        self._clear_layout(self._content_layout)

        # ── Stack mode (with optional pending creation card) ────────
        if self._show_stack:
            active = self._director.compositions.active
            self._title.setText(active.name if active is not None else "Diagram")
            self._empty_hint.setVisible(False)
            self._content.setVisible(True)
            self._build_stack_view(include_pending=self._create_new)
            return

        # ── Standalone creation mode (no stack context) ─────────────
        if self._create_new:
            self._title.setText("New layer")
            self._empty_hint.setVisible(False)
            self._content.setVisible(True)
            self._build_creation_panel()
            self._content_layout.addStretch(1)
            return

        # ── Legacy single-layer edit mode ───────────────────────────
        d = self._selected
        if d is None:
            self._title.setText("Nothing selected.")
            self._empty_hint.setVisible(True)
            self._content.setVisible(False)
            return

        self._title.setText(d.display_label())
        self._empty_hint.setVisible(False)
        self._content.setVisible(True)

        self._build_data_swap_row(d)
        self._dispatch_kind_panel(d)
        self._build_preset_row(d)
        self._build_delete_row(d)
        self._content_layout.addStretch(1)

    # ------------------------------------------------------------------
    # Stack mode
    # ------------------------------------------------------------------

    def _build_stack_view(self, *, include_pending: bool = False) -> None:
        """Render one QGroupBox per *active-composition* layer.

        Filters the registry's flat diagram list to just the layers
        belonging to the active composition. Each card wraps the
        same per-kind controls used in legacy edit mode. Cards are
        added to a scroll area so a 5+ layer composition doesn't blow
        up the dock height.

        When ``include_pending`` is True, a *creation card* is
        appended at the bottom — Kind ▾ + Data ▾ + Apply / Cancel —
        so the user can configure a new layer while still seeing
        every existing one.
        """
        QtWidgets, _ = _qt()
        active = self._director.compositions.active
        diagrams = list(active.layers) if active is not None else []

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QtWidgets.QWidget()
        inner_lay = QtWidgets.QVBoxLayout(inner)
        inner_lay.setContentsMargins(0, 0, 0, 0)
        inner_lay.setSpacing(8)

        if not diagrams and not include_pending:
            msg = "No layers yet. Click + Add layer to add one."
            empty = QtWidgets.QLabel(msg)
            empty.setWordWrap(True)
            empty.setObjectName("DiagramSettingsEmptyHint")
            inner_lay.addWidget(empty)

        for d in diagrams:
            inner_lay.addWidget(self._build_layer_card(d))

        if include_pending:
            inner_lay.addWidget(self._build_pending_creation_card())

        inner_lay.addStretch(1)
        scroll.setWidget(inner)
        self._content_layout.addWidget(scroll, stretch=1)

    def _build_pending_creation_card(self) -> Any:
        """A QGroupBox card holding the Kind+Data+Apply/Cancel form.

        Same form as :meth:`_build_creation_panel` but rendered inside
        a card so it stacks naturally below the existing layers.
        """
        QtWidgets, _ = _qt()
        card = QtWidgets.QGroupBox("New layer")
        card_lay = QtWidgets.QVBoxLayout(card)
        card_lay.setContentsMargins(8, 4, 8, 6)
        card_lay.setSpacing(4)
        # Swap the target layout so the existing creation builder
        # writes into this card instead of the outer content layout.
        saved = self._content_layout
        self._content_layout = card_lay
        try:
            self._build_creation_panel()
        finally:
            self._content_layout = saved
        return card

    def _build_layer_card(self, d: "Diagram") -> Any:
        """Build one QGroupBox card for ``d`` containing all its controls.

        Reuses every existing per-kind builder by temporarily swapping
        ``self._content_layout`` to the card's own layout for the
        duration of the build.
        """
        QtWidgets, _ = _qt()
        from ._add_diagram_dialog import _KINDS as KIND_ENTRIES
        id_to_label = {k.kind_id: k.label for k in KIND_ENTRIES}
        title = id_to_label.get(d.kind, d.kind)
        comp = getattr(d.selector, "component", "")
        if comp:
            title = f"{title} · {comp}"

        card = QtWidgets.QGroupBox(title)
        # Checkable groupbox = built-in visibility checkbox in the
        # title row — wired to set_visible on the underlying diagram.
        card.setCheckable(True)
        card.setChecked(bool(d.is_visible))
        card.toggled.connect(
            lambda checked, d=d: self._director.registry.set_visible(
                d, bool(checked),
            ),
        )
        card_lay = QtWidgets.QVBoxLayout(card)
        card_lay.setContentsMargins(8, 4, 8, 6)
        card_lay.setSpacing(4)

        # Swap the target layout so the existing builders write here.
        saved = self._content_layout
        self._content_layout = card_lay
        try:
            self._build_reorder_row(d)
            self._build_data_swap_row(d)
            self._dispatch_kind_panel(d)
            self._build_delete_row(d)
        finally:
            self._content_layout = saved
        return card

    def _build_reorder_row(self, d: "Diagram") -> None:
        """↑ / ↓ buttons that move ``d`` within the active composition.

        Order in the composition drives stack-view rendering order;
        we mirror the move into the registry so paint order
        (``z-order``) matches what the user sees.
        """
        QtWidgets, _ = _qt()
        comp_mgr = self._director.compositions
        comp = (
            comp_mgr.composition_for_layer(d) if comp_mgr is not None
            else None
        )
        if comp is None:
            return
        try:
            idx = comp.layers.index(d)
        except ValueError:
            return
        is_first = idx == 0
        is_last = idx == len(comp.layers) - 1

        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        btn_up = QtWidgets.QPushButton("↑")
        btn_up.setFixedWidth(28)
        btn_up.setEnabled(not is_first)
        btn_up.setToolTip("Move layer up (paint earlier)")
        btn_up.clicked.connect(lambda _=False, d=d: self._on_move(d, -1))
        btn_down = QtWidgets.QPushButton("↓")
        btn_down.setFixedWidth(28)
        btn_down.setEnabled(not is_last)
        btn_down.setToolTip("Move layer down (paint later)")
        btn_down.clicked.connect(lambda _=False, d=d: self._on_move(d, +1))
        row.addWidget(btn_up)
        row.addWidget(btn_down)
        self._content_layout.addLayout(row)

    def _on_move(self, d: "Diagram", delta: int) -> None:
        """Slide ``d`` up (delta=-1) or down (delta=+1) within its
        composition; mirror the move in the flat registry list."""
        comp_mgr = self._director.compositions
        comp = (
            comp_mgr.composition_for_layer(d) if comp_mgr is not None
            else None
        )
        if comp is None:
            return
        try:
            idx = comp.layers.index(d)
        except ValueError:
            return
        new_idx = max(0, min(idx + int(delta), len(comp.layers) - 1))
        if new_idx == idx:
            return
        comp.layers.pop(idx)
        comp.layers.insert(new_idx, d)
        # Mirror to the registry so VTK paint order tracks the UI.
        try:
            registry = self._director.registry
            r_idx = registry.index_of(d)
            if r_idx is not None:
                registry.move(r_idx, r_idx + int(delta))
        except Exception:
            pass
        # Manager observers don't fire on direct layers-list mutation;
        # nudge them so the stack rebuilds in the new order.
        if comp_mgr is not None:
            comp_mgr._notify()  # noqa: SLF001

    def _dispatch_kind_panel(self, d: "Diagram") -> None:
        """Dispatch to the right ``_build_*_panel`` for the kind."""
        QtWidgets, _ = _qt()
        kind = d.kind
        if kind == "contour":
            self._build_contour_panel(d)
        elif kind == "deformed_shape":
            self._build_deformed_panel(d)
        elif kind == "line_force":
            self._build_line_force_panel(d)
        elif kind in ("fiber_section", "layer_stack", "gauss_marker"):
            self._build_color_panel(d)
        elif kind == "vector_glyph":
            self._build_vector_panel(d)
        elif kind == "spring_force":
            self._build_spring_panel(d)
        else:
            self._content_layout.addWidget(QtWidgets.QLabel(
                f"No settings UI for kind {kind!r} yet."
            ))

    # ------------------------------------------------------------------
    # Layout teardown helper
    # ------------------------------------------------------------------

    def _clear_layout(self, layout) -> None:
        """Recursively remove every widget *and* sub-layout from ``layout``.

        Qt's ``takeAt(0)`` returns a ``QLayoutItem``. Three cases:

        - widget item → ``deleteLater()`` it.
        - layout item (e.g. nested ``QFormLayout`` / ``QHBoxLayout``)
          → recurse, then ``deleteLater()`` the layout itself.
        - spacer item → nothing to delete; just dropped on the floor
          when the loop pops it.

        Without the recursion, sub-layout widgets stay parented to the
        outer container and reappear painted on top of the next
        rebuild's widgets — visible as the overlapping rows reported
        in PR #54's punch list.
        """
        while layout.count():
            item = layout.takeAt(0)
            if item is None:
                continue
            w = item.widget()
            if w is not None:
                w.deleteLater()
                continue
            sub = item.layout()
            if sub is not None:
                self._clear_layout(sub)
                sub.deleteLater()
            # spacer items have neither widget() nor layout() — drop.

    # ------------------------------------------------------------------
    # Creation mode (+ Add layer) — Kind + Data + Apply / Cancel
    # ------------------------------------------------------------------

    def _build_creation_panel(self) -> None:
        """Render Kind + Data + style preview + Apply/Cancel.

        Uses the kind catalog to populate the dropdowns. The Kind combo
        lists every kind; disabled kinds (no data feeds them in this
        file) are still shown but tagged "(no data)" and unselectable.
        Changing Kind repopulates the Data combo. Apply builds a spec,
        instantiates the diagram, and adds it to the registry.
        """
        QtWidgets, _ = _qt()
        from ._add_diagram_dialog import _KINDS as KIND_ENTRIES

        catalog = self._ensure_catalog()
        id_to_kind_entry = {k.kind_id: k for k in KIND_ENTRIES}

        form = QtWidgets.QFormLayout()
        self._content_layout.addLayout(form)

        # ── Kind combo ─────────────────────────────────────────────
        kind_combo = QtWidgets.QComboBox()
        # Sort: enabled first (catalog order), disabled last.
        enabled = [k for k in catalog if k.enabled]
        disabled = [k for k in catalog if not k.enabled]
        for k in enabled + disabled:
            label = k.label if k.enabled else f"{k.label} — no data"
            kind_combo.addItem(label, k.kind_id)
            idx = kind_combo.count() - 1
            if not k.enabled:
                # Grey out + uncheckable item flag (Qt::NoItemFlags).
                kind_combo.model().item(idx).setEnabled(False)
        form.addRow("Kind:", kind_combo)

        # ── Data combo (populated when Kind changes) ───────────────
        data_combo = QtWidgets.QComboBox()
        data_label = QtWidgets.QLabel("Data:")
        form.addRow(data_label, data_combo)

        def _populate_data_for(kind_id: str) -> None:
            entry = next((k for k in catalog if k.kind_id == kind_id), None)
            data_combo.blockSignals(True)
            data_combo.clear()
            if entry is None:
                data_combo.blockSignals(False)
                return
            if not entry.requires_data:
                data_combo.setEnabled(False)
                data_combo.setEditText("(no data needed)")
                data_label.setVisible(False)
                data_combo.setVisible(False)
            else:
                data_label.setVisible(True)
                data_combo.setVisible(True)
                data_combo.setEnabled(True)
                data_combo.addItems(list(entry.data_options))
                if entry.default_data is not None:
                    data_combo.setCurrentText(entry.default_data)
            data_combo.blockSignals(False)

        kind_combo.currentIndexChanged.connect(
            lambda _i: _populate_data_for(kind_combo.currentData()),
        )
        # Initial population — pick first enabled kind.
        if enabled:
            kind_combo.setCurrentIndex(0)
            _populate_data_for(enabled[0].kind_id)

        # ── Apply / Cancel row ─────────────────────────────────────
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch(1)
        btn_cancel = QtWidgets.QPushButton("Cancel")
        btn_apply = QtWidgets.QPushButton("Apply")
        btn_apply.setDefault(True)
        btn_row.addWidget(btn_cancel)
        btn_row.addWidget(btn_apply)
        self._content_layout.addLayout(btn_row)

        btn_cancel.clicked.connect(lambda: self.set_create_new(False))
        btn_apply.clicked.connect(
            lambda: self._on_creation_apply(
                kind_combo.currentData(),
                data_combo.currentText().strip()
                if data_combo.isEnabled() else "",
            ),
        )

    def _on_creation_apply(self, kind_id: str, data: str) -> None:
        """Build a Diagram from the creation form and add it to the registry."""
        if not kind_id:
            return
        diagram = self._build_diagram(kind_id, data)
        if diagram is None:
            return
        try:
            self._director.registry.add(diagram)
        except Exception as exc:
            from .._failures import report
            report("DiagramSettingsTab._on_creation_apply", exc)
            return
        # Tag the new layer with the active composition so the
        # outline + stack view group it correctly.
        comp_mgr = self._director.compositions
        active = comp_mgr.active if comp_mgr is not None else None
        if active is not None and comp_mgr is not None:
            comp_mgr.add_layer(active.id, diagram)
        # Drop the pending-creation card; keep stack mode so the
        # newly-added layer appears as a real card alongside the
        # existing ones (registry.add fires the observer which
        # triggers _rebuild via _on_diagrams_changed).
        self._create_new = False
        if not self._show_stack:
            # Caller arrived without stack mode (e.g. legacy path) —
            # fall through to the standalone edit mode.
            self.set_selected(diagram)
        else:
            self._rebuild()

    # ------------------------------------------------------------------
    # Edit mode helpers: Data swap + Delete
    # ------------------------------------------------------------------

    def _build_data_swap_row(self, d: "Diagram") -> None:
        """Component dropdown that live-swaps the diagram in place.

        Same kind, different selector. Uses ``registry.replace()`` so
        the layer keeps its z-position. Kind itself is shown as a
        read-only label — to switch kind, the user deletes + re-adds.
        """
        QtWidgets, _ = _qt()
        catalog = self._ensure_catalog()
        kind_entry = next(
            (k for k in catalog if k.kind_id == d.kind), None,
        )

        form = QtWidgets.QFormLayout()
        self._content_layout.addLayout(form)

        # Kind: read-only label.
        kind_label = QtWidgets.QLabel(
            kind_entry.label if kind_entry else d.kind,
        )
        form.addRow("Kind:", kind_label)

        # Data combo (only if kind requires data).
        if kind_entry is not None and not kind_entry.requires_data:
            return
        data_combo = QtWidgets.QComboBox()
        if kind_entry is not None:
            data_combo.addItems(list(kind_entry.data_options))
        # Set current to the diagram's selector component.
        current_comp = getattr(d.selector, "component", "")
        if current_comp:
            idx = data_combo.findText(current_comp)
            if idx >= 0:
                data_combo.setCurrentIndex(idx)
            else:
                # Component not in catalog list (e.g. user typed a
                # custom name in a previous version); add it so we
                # don't clobber it on selection.
                data_combo.addItem(current_comp)
                data_combo.setCurrentText(current_comp)
        # Live swap on change.
        data_combo.currentTextChanged.connect(
            lambda new_data: self._on_data_swap(d, new_data),
        )
        form.addRow("Data:", data_combo)

    def _on_data_swap(self, old: "Diagram", new_data: str) -> None:
        if not new_data or new_data == getattr(old.selector, "component", ""):
            return
        new_diagram = self._build_diagram(old.kind, new_data)
        if new_diagram is None:
            return
        # Locate the composition holding ``old`` so we can swap the
        # membership too (preserve z-position within the comp).
        comp_mgr = self._director.compositions
        comp = (
            comp_mgr.composition_for_layer(old) if comp_mgr is not None
            else None
        )
        try:
            self._director.registry.replace(old, new_diagram)
        except Exception as exc:
            from .._failures import report
            report("DiagramSettingsTab._on_data_swap", exc)
            return
        if comp is not None and comp_mgr is not None:
            try:
                idx = comp.layers.index(old)
                comp.layers[idx] = new_diagram
                # Manager observers don't fire on direct list mutation;
                # nudge them so the outline label refreshes if needed.
                comp_mgr._notify()  # noqa: SLF001
            except ValueError:
                pass
        # Selected reference is now the new diagram; rebuild form.
        self._selected = new_diagram
        self._rebuild()

    def _build_delete_row(self, d: "Diagram") -> None:
        QtWidgets, _ = _qt()
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch(1)
        btn_delete = QtWidgets.QPushButton("Delete layer")
        btn_delete.clicked.connect(lambda: self._on_delete(d))
        btn_row.addWidget(btn_delete)
        self._content_layout.addLayout(btn_row)

    def _on_delete(self, d: "Diagram") -> None:
        # Untag from whichever geometry's composition owns it, then
        # remove from the registry.
        try:
            owner = self._director.geometries.geometry_for_layer(d)
            if owner is not None:
                owner.compositions.remove_layer(d)
        except Exception:
            pass
        try:
            self._director.registry.remove(d)
        except Exception as exc:
            from .._failures import report
            report("DiagramSettingsTab._on_delete", exc)
            return
        # _on_diagrams_changed clears self._selected if needed.

    # ------------------------------------------------------------------
    # Diagram constructor — shared by creation + data-swap
    # ------------------------------------------------------------------

    def _build_diagram(self, kind_id: str, data: str) -> Optional["Diagram"]:
        """Instantiate a Diagram for ``kind_id`` + ``data``.

        Builds a default style from the kind entry (mirrors
        AddDiagramDialog). Returns None on failure (errors land in
        the slot-failure pipeline so the user sees a status toast).
        """
        from ..diagrams._base import DiagramSpec
        from ..diagrams._selectors import normalize as normalize_selector
        from ..diagrams._styles import ContourStyle
        from ._add_diagram_dialog import _KINDS as KIND_ENTRIES
        from .._failures import report

        kind_entry = next(
            (k for k in KIND_ENTRIES if k.kind_id == kind_id), None,
        )
        if kind_entry is None:
            return None

        catalog_entry = next(
            (k for k in self._ensure_catalog() if k.kind_id == kind_id), None,
        )
        component = data or ""
        try:
            selector = normalize_selector(component=component)
        except Exception as exc:
            report("DiagramSettingsTab._build_diagram", exc)
            return None

        style = kind_entry.make_default_style(component)
        # Contour against gauss data: pin topology so the diagram
        # reads from the gauss composite rather than nodes.
        if kind_id == "contour":
            comp_in_gauss = (
                catalog_entry is not None
                and component in catalog_entry.data_options
            )
            # Heuristic: if the component is a tensor (xx/yy/...) it's
            # a gauss reading; vector axes (x/y/z) are nodal. The
            # catalog already mixes both, so we route by suffix.
            from ...solvers._element_response import split_canonical_component
            parts = split_canonical_component(component)
            is_tensor = parts is not None and parts[1] in {
                "xx", "yy", "zz", "xy", "yz", "xz",
            }
            if comp_in_gauss and is_tensor:
                style = ContourStyle(
                    cmap=style.cmap, clim=style.clim, opacity=style.opacity,
                    show_edges=style.show_edges,
                    show_scalar_bar=style.show_scalar_bar,
                    fmt=style.fmt,
                    topology="gauss",
                )

        spec = DiagramSpec(
            kind=kind_id,
            selector=selector,
            style=style,
            stage_id=self._director.stage_id,
        )
        try:
            return kind_entry.diagram_class(spec, self._director.results)
        except Exception as exc:
            report("DiagramSettingsTab._build_diagram", exc)
            return None

    # ------------------------------------------------------------------
    # Contour panel
    # ------------------------------------------------------------------

    def _build_contour_panel(self, d: Diagram) -> None:
        QtWidgets, QtCore = _qt()
        form = QtWidgets.QFormLayout()
        self._content_layout.addLayout(form)

        # Cmap combo
        cmap_combo = QtWidgets.QComboBox()
        cmap_combo.setEditable(True)
        for name in _CMAP_PRESETS:
            cmap_combo.addItem(name)
        current_cmap = getattr(d, "_runtime_cmap", None) or d.spec.style.cmap
        cmap_combo.setCurrentText(current_cmap)
        cmap_combo.currentTextChanged.connect(
            lambda txt: self._safe_call(d.set_cmap, txt)
        )
        form.addRow("Colormap:", cmap_combo)

        # Clim
        clim = d.current_clim() if hasattr(d, "current_clim") else None
        lo_default, hi_default = clim if clim else (0.0, 1.0)

        lo_spin = QtWidgets.QDoubleSpinBox()
        lo_spin.setRange(-1e30, 1e30)
        lo_spin.setDecimals(6)
        lo_spin.setValue(float(lo_default))
        hi_spin = QtWidgets.QDoubleSpinBox()
        hi_spin.setRange(-1e30, 1e30)
        hi_spin.setDecimals(6)
        hi_spin.setValue(float(hi_default))

        def _commit_clim() -> None:
            self._safe_call(
                d.set_clim, float(lo_spin.value()), float(hi_spin.value()),
            )

        lo_spin.editingFinished.connect(_commit_clim)
        hi_spin.editingFinished.connect(_commit_clim)
        form.addRow("Clim min:", lo_spin)
        form.addRow("Clim max:", hi_spin)

        autofit_btn = QtWidgets.QPushButton("Auto-fit at current step")

        def _autofit() -> None:
            new_clim = self._safe_call(d.autofit_clim_at_current_step)
            if new_clim is not None:
                lo_spin.setValue(float(new_clim[0]))
                hi_spin.setValue(float(new_clim[1]))

        autofit_btn.clicked.connect(_autofit)
        self._content_layout.addWidget(autofit_btn)

        # Opacity slider
        opacity_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        opacity_slider.setRange(0, 100)
        current_opacity = (
            getattr(d, "_runtime_opacity", None)
            if getattr(d, "_runtime_opacity", None) is not None
            else d.spec.style.opacity
        )
        opacity_slider.setValue(int(round(float(current_opacity) * 100)))
        opacity_slider.valueChanged.connect(
            lambda v: self._safe_call(d.set_opacity, v / 100.0)
        )
        form.addRow("Opacity:", opacity_slider)

        # Scalar-bar live controls (Show + Format) — shared with the
        # generic color panel used by fiber / layer / gauss_marker /
        # vector_glyph so every contour-bearing diagram gets the same
        # UI surface.
        self._add_scalar_bar_controls(d, form)

    # ------------------------------------------------------------------
    # Deformed shape panel
    # ------------------------------------------------------------------

    def _build_deformed_panel(self, d: Diagram) -> None:
        QtWidgets, QtCore = _qt()
        form = QtWidgets.QFormLayout()
        self._content_layout.addLayout(form)

        # Scale slider — log-ish via spinbox for ergonomics
        scale_spin = QtWidgets.QDoubleSpinBox()
        scale_spin.setRange(0.0, 1e9)
        scale_spin.setDecimals(4)
        scale_spin.setSingleStep(0.1)
        scale_spin.setValue(float(d.current_scale()))
        scale_spin.editingFinished.connect(
            lambda: self._safe_call(d.set_scale, float(scale_spin.value()))
        )
        form.addRow("Scale:", scale_spin)

        # Show-undeformed checkbox
        chk = QtWidgets.QCheckBox("Show undeformed reference")
        style = d.spec.style
        current = (
            getattr(d, "_runtime_show_undeformed", None)
            if getattr(d, "_runtime_show_undeformed", None) is not None
            else getattr(style, "show_undeformed", True)
        )
        chk.setChecked(bool(current))
        chk.toggled.connect(
            lambda v: self._safe_call(d.set_show_undeformed, bool(v))
        )
        form.addRow("", chk)

    # ------------------------------------------------------------------
    # Line force panel
    # ------------------------------------------------------------------

    def _build_line_force_panel(self, d: Diagram) -> None:
        QtWidgets, QtCore = _qt()
        form = QtWidgets.QFormLayout()
        self._content_layout.addLayout(form)

        # Scale
        scale_spin = QtWidgets.QDoubleSpinBox()
        scale_spin.setRange(0.0, 1e9)
        scale_spin.setDecimals(6)
        scale_spin.setSingleStep(0.1)
        scale_spin.setValue(float(d.current_scale()))
        scale_spin.editingFinished.connect(
            lambda: self._safe_call(d.set_scale, float(scale_spin.value()))
        )
        form.addRow("Scale:", scale_spin)

        # Fill axis — local frame, world axes, or auto.
        # World-axis options are useful for 2-D models you want to view
        # obliquely: pick ``Global Z`` to extrude the diagram out of
        # the model plane so it stays visible from any camera angle.
        axis_combo = QtWidgets.QComboBox()
        axis_combo.addItem("Auto (component default)", None)
        axis_combo.addItem("y_local", "y")
        axis_combo.addItem("z_local", "z")
        axis_combo.addItem("Global X", "global_x")
        axis_combo.addItem("Global Y", "global_y")
        axis_combo.addItem("Global Z", "global_z")
        # Reflect runtime override (or, if absent, the spec's style).
        runtime_axis = getattr(d, "_runtime_axis", None)
        current = (
            runtime_axis if runtime_axis is not None
            else getattr(d.spec.style, "fill_axis", None)
        )
        for i in range(axis_combo.count()):
            if axis_combo.itemData(i) == current:
                axis_combo.setCurrentIndex(i)
                break

        def _on_axis_change(idx: int) -> None:
            self._safe_call(d.set_fill_axis, axis_combo.itemData(idx))

        axis_combo.currentIndexChanged.connect(_on_axis_change)
        form.addRow("Fill axis:", axis_combo)

        # Flip sign
        flip_chk = QtWidgets.QCheckBox("Flip sign")
        runtime_flip = getattr(d, "_runtime_flip", None)
        style_flip = getattr(d.spec.style, "flip_sign", False)
        flip_chk.setChecked(bool(
            runtime_flip if runtime_flip is not None else style_flip
        ))
        flip_chk.toggled.connect(
            lambda v: self._safe_call(d.set_flip_sign, bool(v))
        )
        form.addRow("", flip_chk)

    # ------------------------------------------------------------------
    # Vector-glyph panel (scale + cmap/clim)
    # ------------------------------------------------------------------

    def _build_vector_panel(self, d: Diagram) -> None:
        QtWidgets, _ = _qt()
        form = QtWidgets.QFormLayout()
        self._content_layout.addLayout(form)

        scale_spin = QtWidgets.QDoubleSpinBox()
        scale_spin.setRange(0.0, 1e9)
        scale_spin.setDecimals(6)
        scale_spin.setSingleStep(0.1)
        scale_spin.setValue(float(d.current_scale()))
        scale_spin.editingFinished.connect(
            lambda: self._safe_call(d.set_scale, float(scale_spin.value()))
        )
        form.addRow("Scale:", scale_spin)

        # Color settings reuse the shared color panel
        self._build_color_panel(d)

    # ------------------------------------------------------------------
    # Spring-force panel (scale + direction)
    # ------------------------------------------------------------------

    def _build_spring_panel(self, d: Diagram) -> None:
        QtWidgets, _ = _qt()
        form = QtWidgets.QFormLayout()
        self._content_layout.addLayout(form)

        scale_spin = QtWidgets.QDoubleSpinBox()
        scale_spin.setRange(0.0, 1e9)
        scale_spin.setDecimals(6)
        scale_spin.setSingleStep(0.1)
        scale_spin.setValue(float(d.current_scale()))
        scale_spin.editingFinished.connect(
            lambda: self._safe_call(d.set_scale, float(scale_spin.value()))
        )
        form.addRow("Scale:", scale_spin)

        # Direction overrides — three spinboxes for (dx, dy, dz)
        dir_row = QtWidgets.QHBoxLayout()
        spins = []
        cur = getattr(d, "_direction", None)
        defaults = (
            (1.0, 0.0, 0.0) if cur is None
            else (float(cur[0]), float(cur[1]), float(cur[2]))
        )
        for axis_label, default in zip(("dx", "dy", "dz"), defaults):
            sp = QtWidgets.QDoubleSpinBox()
            sp.setRange(-1e9, 1e9)
            sp.setDecimals(4)
            sp.setSingleStep(0.1)
            sp.setValue(default)
            spins.append(sp)
            dir_row.addWidget(QtWidgets.QLabel(axis_label))
            dir_row.addWidget(sp)

        def _commit_dir() -> None:
            self._safe_call(
                d.set_direction,
                (
                    float(spins[0].value()),
                    float(spins[1].value()),
                    float(spins[2].value()),
                ),
            )

        for sp in spins:
            sp.editingFinished.connect(_commit_dir)
        self._content_layout.addLayout(dir_row)

    # ------------------------------------------------------------------
    # Generic color/clim panel — used by fiber & layer diagrams which
    # share a common surface (cmap + clim + auto-fit).
    # ------------------------------------------------------------------

    def _build_color_panel(self, d: Diagram) -> None:
        QtWidgets, _ = _qt()
        form = QtWidgets.QFormLayout()
        self._content_layout.addLayout(form)

        cmap_combo = QtWidgets.QComboBox()
        cmap_combo.setEditable(True)
        for name in _CMAP_PRESETS:
            cmap_combo.addItem(name)
        current_cmap = (
            getattr(d, "_runtime_cmap", None)
            or getattr(d.spec.style, "cmap", "viridis")
        )
        cmap_combo.setCurrentText(current_cmap)
        cmap_combo.currentTextChanged.connect(
            lambda txt: self._safe_call(d.set_cmap, txt)
        )
        form.addRow("Colormap:", cmap_combo)

        clim = d.current_clim() if hasattr(d, "current_clim") else None
        lo_default, hi_default = clim if clim else (0.0, 1.0)
        lo_spin = QtWidgets.QDoubleSpinBox()
        lo_spin.setRange(-1e30, 1e30)
        lo_spin.setDecimals(6)
        lo_spin.setValue(float(lo_default))
        hi_spin = QtWidgets.QDoubleSpinBox()
        hi_spin.setRange(-1e30, 1e30)
        hi_spin.setDecimals(6)
        hi_spin.setValue(float(hi_default))

        def _commit() -> None:
            self._safe_call(
                d.set_clim, float(lo_spin.value()), float(hi_spin.value()),
            )

        lo_spin.editingFinished.connect(_commit)
        hi_spin.editingFinished.connect(_commit)
        form.addRow("Clim min:", lo_spin)
        form.addRow("Clim max:", hi_spin)

        autofit = QtWidgets.QPushButton("Auto-fit at current step")

        def _autofit() -> None:
            new_clim = self._safe_call(d.autofit_clim_at_current_step)
            if new_clim is not None:
                lo_spin.setValue(float(new_clim[0]))
                hi_spin.setValue(float(new_clim[1]))

        autofit.clicked.connect(_autofit)
        self._content_layout.addWidget(autofit)

        # Scalar-bar live controls — every diagram routed to this
        # panel inherits ScalarBarSupport, so the setters exist.
        self._add_scalar_bar_controls(d, form)

    # ------------------------------------------------------------------
    # Scalar-bar controls — Show checkbox + Format line edit
    # ------------------------------------------------------------------

    def _add_scalar_bar_controls(self, d: Diagram, form: Any) -> None:
        """Append a Show-scale checkbox + Format field to ``form``.

        Skips silently when the diagram doesn't support the live API
        (older diagrams not yet on ``ScalarBarSupport``). The form
        owner — the caller — keeps its existing layout.
        """
        if not hasattr(d, "set_show_scalar_bar") or not hasattr(d, "set_fmt"):
            return
        QtWidgets, _ = _qt()

        show_chk = QtWidgets.QCheckBox("Show scale")
        runtime_show = getattr(d, "_runtime_show_scalar_bar", None)
        current_show = (
            getattr(d.spec.style, "show_scalar_bar", True)
            if runtime_show is None else bool(runtime_show)
        )
        show_chk.setChecked(bool(current_show))
        show_chk.toggled.connect(
            lambda v: self._safe_call(d.set_show_scalar_bar, bool(v))
        )
        form.addRow(show_chk)

        fmt_edit = QtWidgets.QLineEdit()
        fmt_edit.setPlaceholderText("%.3g")
        current_fmt = (
            getattr(d, "_runtime_fmt", None)
            or getattr(d.spec.style, "fmt", "%.3g")
        )
        fmt_edit.setText(current_fmt)
        fmt_edit.setToolTip(
            "printf-style format for scalar-bar tick labels.\n"
            "Examples: %.3g (general, 3 digits), %.2e (exponent),\n"
            "%.4f (fixed, 4 decimals)."
        )
        fmt_edit.editingFinished.connect(
            lambda: self._safe_call(d.set_fmt, fmt_edit.text() or "%.3g")
        )
        form.addRow("Format:", fmt_edit)

    # ------------------------------------------------------------------
    # Preset row
    # ------------------------------------------------------------------

    def _build_preset_row(self, d: Diagram) -> None:
        """Render Save / Apply preset controls for the selected diagram.

        Save button captures the diagram's current ``spec.style`` to a
        named JSON in the user's preset directory. Apply combo lists
        every preset of the same kind and re-applies it to the live
        diagram by replacing its style and re-attaching.
        """
        QtWidgets, _ = _qt()
        from ..diagrams._style_presets import default_store

        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.HLine)
        self._content_layout.addWidget(sep)

        title = QtWidgets.QLabel("PRESETS")
        f = title.font()
        f.setBold(True)
        title.setFont(f)
        self._content_layout.addWidget(title)

        row = QtWidgets.QHBoxLayout()
        self._content_layout.addLayout(row)

        # Apply combo
        combo = QtWidgets.QComboBox()
        combo.addItem("(choose preset…)", None)
        try:
            names = default_store().list_for_kind(d.kind)
        except Exception:
            names = []
        for name in names:
            combo.addItem(name, name)
        row.addWidget(combo, stretch=1)

        apply_btn = QtWidgets.QPushButton("Apply")
        apply_btn.setEnabled(False)
        row.addWidget(apply_btn)

        save_btn = QtWidgets.QPushButton("Save…")
        row.addWidget(save_btn)

        def _on_combo_change(_idx: int) -> None:
            apply_btn.setEnabled(combo.currentData() is not None)
        combo.currentIndexChanged.connect(_on_combo_change)

        def _on_apply() -> None:
            name = combo.currentData()
            if name is None:
                return
            try:
                _kind_id, style = default_store().load(name)
            except Exception as exc:
                QtWidgets.QMessageBox.warning(
                    self._widget, "Apply preset",
                    f"Could not load preset {name!r}: {exc}",
                )
                return
            self._apply_style_to_diagram(d, style)
        apply_btn.clicked.connect(_on_apply)

        def _on_save() -> None:
            name, ok = QtWidgets.QInputDialog.getText(
                self._widget, "Save preset",
                "Preset name:",
            )
            if not ok or not name.strip():
                return
            try:
                default_store().save(name.strip(), d.kind, d.spec.style)
            except Exception as exc:
                QtWidgets.QMessageBox.warning(
                    self._widget, "Save preset",
                    f"Could not save preset: {exc}",
                )
                return
            # Refresh the combo so the new entry is visible without
            # reselecting the diagram.
            combo.addItem(name.strip(), name.strip())
        save_btn.clicked.connect(_on_save)

    def _apply_style_to_diagram(self, d: Diagram, style: Any) -> None:
        """Swap the diagram's spec.style and re-attach.

        Mirrors the runtime style-mutation pattern other settings
        panels use: rebuild ``DiagramSpec`` with the new style and
        ask the registry to re-attach the diagram so the rendered
        output reflects the change immediately.
        """
        from ..diagrams._base import DiagramSpec
        new_spec = DiagramSpec(
            kind=d.spec.kind,
            selector=d.spec.selector,
            style=style,
            stage_id=d.spec.stage_id,
            label=d.spec.label,
        )
        # Diagrams hold spec via a public attribute on Diagram; mutate
        # the attached instance in place and ask the registry to re-
        # attach everything so the new style is rendered. The registry
        # has no per-diagram reattach today; reattach_all() is cheap
        # enough for the panel-driven path.
        d.spec = new_spec
        try:
            self._director.registry.reattach_all()
        except Exception:
            pass
        self._rebuild()

    def _safe_call(self, fn: Callable, *args, **kwargs) -> Any:
        try:
            result = fn(*args, **kwargs)
            self._fire_render()
            return result
        except Exception as exc:
            import sys
            print(
                f"[DiagramSettingsTab] {fn.__qualname__} raised: {exc}",
                file=sys.stderr,
            )
            return None

    def _fire_render(self) -> None:
        # The Director's render_callback isn't directly accessible; rely
        # on the registry's update path via a no-op step set when present.
        # For Phase 1 we just call the plotter's render via the Director's
        # registry-bound plotter.
        plotter = getattr(self._director.registry, "_plotter", None)
        if plotter is not None:
            try:
                plotter.render()
            except Exception:
                pass
