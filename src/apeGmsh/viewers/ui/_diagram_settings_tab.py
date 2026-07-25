"""Per-diagram settings tab — styling controls for the active selection.

Reads the currently-selected diagram from the Diagrams tab and shows a
kind-specific control set:

* ``contour`` — clim min/max, auto-fit button, opacity slider, cmap combo

When no diagram is selected, an empty-state message is shown.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional

from ..core._legend import MAX_FONT_SCALE as _MAX_FONT_SCALE
from ..core._legend import MIN_FONT_SCALE as _MIN_FONT_SCALE
from ..diagrams._base import Diagram

if TYPE_CHECKING:
    from ..diagrams._director import ResultsDirector


def _qt():
    from qtpy import QtWidgets, QtCore
    return QtWidgets, QtCore


_CMAP_PRESETS = [
    "viridis", "plasma", "cividis", "magma", "inferno",
    "coolwarm", "RdBu", "RdBu_r", "bwr", "seismic", "BrBG",
    "Spectral", "turbo", "jet",
]


class DiagramSettingsTab:
    """Settings panel for the diagram selected in the Diagrams tab.

    Subscribes to the Director's ``on_diagrams_changed`` so the
    panel refreshes when the selected diagram is removed or replaced.
    The selection itself is set by ``set_selected(diagram)`` from the
    Diagrams tab when the user clicks a row.
    """

    # ──────────────────────────────────────────────────────────────────
    # Plan 05 — Auto-Apply / Reset
    # ──────────────────────────────────────────────────────────────────
    # ``_auto_apply_enabled`` is persisted at ``apeGmsh / ResultsViewer``
    # under the key below so the toggle survives launches.
    _AUTO_APPLY_SETTINGS_KEY = "settings_tab/auto_apply"
    # Debounce interval (ms) for live commits when Auto-Apply is on —
    # coalesces slider drags / rapid spinbox edits so we don't run
    # update_to_step on every micro-change.
    _AUTO_APPLY_DEBOUNCE_MS = 150

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

        # ── Plan 05 state ───────────────────────────────────────────
        # Auto-Apply toggle persisted across launches. Default OFF —
        # the explicit Apply-button workflow is the baseline; users
        # opt into live preview.
        self._auto_apply_enabled: bool = self._load_auto_apply_pref()
        # Per-rebuild list of {layer-commit closures} so the debounce
        # timer can flush every card's pending edits in one shot.
        # Reset at the start of every _rebuild().
        self._card_commits: list[Any] = []
        # Lazily-built single-shot QTimer that fires
        # _AUTO_APPLY_DEBOUNCE_MS after the most recent staged widget
        # change. Restarting before timeout coalesces rapid edits.
        self._auto_commit_timer: Any = None

        # Plan 04 step 2 cont. — broadcast which layer card the user
        # focused (clicked / tabbed into). Owners (ResultsViewer) wire
        # this to ``ActiveObjects.set_active_layer`` so the Color Map
        # Editor and any future panel that "follows the active layer"
        # tracks card focus, not just composition selection.
        self._layer_focus_callback: Optional[
            Callable[[Optional[Diagram]], None]
        ] = None
        # Held references to per-card focus filters so they don't get
        # garbage-collected while the cards are alive. Cleared at the
        # start of every _rebuild().
        self._card_focus_filters: list[Any] = []

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
        # Auto-Apply toggle — when on, widget edits live-commit after
        # _AUTO_APPLY_DEBOUNCE_MS. When off (default), users hit the
        # per-card Apply button. Persisted via QSettings.
        self._auto_apply_cb = QtWidgets.QCheckBox("Auto-Apply")
        self._auto_apply_cb.setToolTip(
            "When on, edits live-commit after a short debounce.\n"
            "When off (default), use the per-card Apply button."
        )
        self._auto_apply_cb.setChecked(self._auto_apply_enabled)
        self._auto_apply_cb.toggled.connect(self._on_auto_apply_toggled)
        title_row.addWidget(self._auto_apply_cb)
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
        # rebuilds the stack when relevant. ``attach_dispatcher``
        # swaps this for the UI-lane coalesced dispatcher subscription
        # once the viewer's event bus is constructed; the legacy path
        # stays for headless tests that never wire a dispatcher.
        self._unsub_compositions: Optional[Callable[[], None]] = (
            director.geometries.subscribe(self._on_compositions_changed)
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

    # ------------------------------------------------------------------
    # Plan 04 step 2 cont. — layer focus callback
    # ------------------------------------------------------------------

    def on_layer_focused(
        self,
        callback: Optional[Callable[[Optional[Diagram]], None]],
    ) -> None:
        """Register a callback fired when a layer card is focused.

        The callback receives the underlying :class:`Diagram` (or
        ``None`` if focus left a card). Fired on mouse-down anywhere
        in a card's bounds — covers the title bar, body, and any
        descendant widget. Idempotent on the same card (callback only
        fires when the focused diagram actually changes).

        Owners typically wire this to ``ActiveObjects.set_active_layer``
        so the Color Map Editor (and future "follows the active layer"
        panels) react to card-level navigation.
        """
        self._layer_focus_callback = callback

    def _fire_layer_focused(self, diagram: Optional[Diagram]) -> None:
        cb = self._layer_focus_callback
        if cb is None:
            return
        try:
            cb(diagram)
        except Exception as exc:
            # Listener failures must never propagate into the Qt event
            # loop from inside an event filter — the filter's outer
            # call would otherwise mark the event handled and freeze
            # the click that triggered it.
            from .._failures import report
            report("DiagramSettingsTab._fire_layer_focused", exc)

    def attach_dispatcher(self, dispatcher: Any) -> None:
        """Migrate the geometry-changed subscription onto the dispatcher.

        Called by :class:`ResultsViewer` once the :class:`Dispatcher`
        is constructed (after this panel's ``__init__``). Replaces the
        raw ``director.geometries.subscribe(self._on_compositions_changed)``
        wiring with a UI-lane coalesced subscription over the granular
        geometry kinds + the omnibus fallback.

        Same `attach_dispatcher` pattern as :class:`OutlineTree`:
        legacy unsub fires before the new subscribe, idempotent on
        repeated calls, and a None dispatcher is a no-op so the
        legacy path stays alive in headless / test contexts.
        """
        if dispatcher is None:
            return
        from ..diagrams._dispatch import (
            COMPOSITION_CHANGED,
            GEOMETRIES_CHANGED,
            GEOMETRY_ACTIVE_CHANGED,
            GEOMETRY_ADDED,
            GEOMETRY_DEFORM_CHANGED,
            GEOMETRY_REMOVED,
            GEOMETRY_RENAMED,
            Lane,
        )
        if self._unsub_compositions is not None:
            try:
                self._unsub_compositions()
            except Exception:
                pass
            self._unsub_compositions = None
        self._unsub_compositions = dispatcher.subscribe(
            (
                GEOMETRIES_CHANGED,
                GEOMETRY_ACTIVE_CHANGED,
                GEOMETRY_DEFORM_CHANGED,
                GEOMETRY_ADDED,
                GEOMETRY_REMOVED,
                GEOMETRY_RENAMED,
                COMPOSITION_CHANGED,
            ),
            lambda _kind, _payload: self._on_compositions_changed(),
            lane=Lane.UI,
            coalesce=True,
        )

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
        # Plan 05 — clear card-commit closures captured by the
        # previous render. Each card re-populates this list when its
        # Apply/Reset row is built. Stale closures would otherwise
        # reference orphaned widgets after Reset.
        self._card_commits = []
        # Plan 04 step 2 cont. — drop per-card focus filters too. Each
        # card's _build_layer_card re-installs a fresh filter on the
        # rebuilt widget tree.
        self._card_focus_filters = []
        # Cancel any pending debounce — outstanding flushes would
        # reference the (now-cleared) closures.
        if self._auto_commit_timer is not None:
            try:
                self._auto_commit_timer.stop()
            except Exception:
                pass

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
        from ..diagrams._kinds import kind_def
        entry = kind_def(d.kind)
        title = entry.label if entry is not None else d.kind
        comp = getattr(d.selector, "component", "")
        if comp:
            title = f"{title} · {comp}"

        card = QtWidgets.QGroupBox(title)
        # Checkable groupbox = built-in visibility checkbox in the
        # title row — wired to set_visible on the underlying diagram.
        card.setCheckable(True)
        card.setChecked(bool(d.is_visible))
        def _on_card_toggled(checked: bool, _d=d) -> None:
            from .._log import log_action
            log_action(
                "ui.settings", "visibility_toggled",
                layer=_d, visible=bool(checked),
            )
            # Owner-fired (ADR 0056 Part 2): the registry mutator fires
            # LAYER_VISIBILITY_CHANGED itself — no call-site fire.
            self._director.registry.set_visible(_d, bool(checked))

        card.toggled.connect(_on_card_toggled)
        card_lay = QtWidgets.QVBoxLayout(card)
        card_lay.setContentsMargins(8, 4, 8, 6)
        card_lay.setSpacing(4)

        # Swap the target layout so the existing builders write here.
        # ``_pending_appliers`` is per-card: each panel builder appends
        # zero-arg closures that read their numeric spinboxes and call
        # the right setter on ``d``. After dispatch, the Apply button
        # below fires them as a batch.
        saved = self._content_layout
        saved_appliers = getattr(self, "_pending_appliers", None)
        self._content_layout = card_lay
        self._pending_appliers = []
        try:
            self._build_reorder_row(d)
            self._build_data_swap_row(d)
            self._dispatch_kind_panel(d)
            self._build_apply_button(d)
            self._build_delete_row(d)
        finally:
            self._content_layout = saved
            self._pending_appliers = saved_appliers
        # Plan 04 step 2 cont. — install a mouse-press filter on the
        # card so any click inside it broadcasts ``d`` as the active
        # layer. Done last (after all child widgets exist) so the
        # filter covers the full subtree.
        self._install_card_focus_filter(card, d)
        return card

    # ------------------------------------------------------------------
    # Card focus filter (plan 04 step 2 cont.)
    # ------------------------------------------------------------------

    def _install_card_focus_filter(
        self, card: Any, diagram: "Diagram",
    ) -> None:
        """Install a mouse-press / focus-in filter on ``card`` and every
        descendant widget so that any user interaction with this card
        broadcasts ``diagram`` as the newly-focused layer."""
        _, QtCore = _qt()
        cls = _resolve_card_focus_filter_class()
        filter_obj = cls(self, diagram)
        self._card_focus_filters.append(filter_obj)
        # Install on the card itself, then recursively on every widget
        # inside it. Children consume their own mouse events, so the
        # filter has to be present at every level.
        for w in (card, *card.findChildren(QtCore.QObject)):
            try:
                w.installEventFilter(filter_obj)
            except Exception:
                pass

    def _build_apply_button(self, d: "Diagram") -> None:
        """Bottom-of-card Apply / Reset row.

        Snapshots the per-card appliers list so future cards' edits
        don't leak into this card's button. After the setters fire,
        dispatches ``DIAGRAM_MODIFIED`` so RENDER coalesces and the
        viewport actually paints.

        When Auto-Apply is on, the Apply button is shown disabled with
        an "(auto)" suffix — changes are committed via the debounce
        timer instead. Reset rebuilds the entire stack from current
        diagram state (effectively discarding any staged widget values).

        Records the card's commit closure in ``self._card_commits`` so
        the debounce timer can flush every card's pending edits in one
        shot when Auto-Apply is on.
        """
        QtWidgets, _ = _qt()
        appliers = list(self._pending_appliers or [])
        if not appliers:
            return

        def _commit() -> None:
            from .._log import log_action
            log_action(
                "ui.settings", "apply_clicked",
                layer=d, n_appliers=len(appliers),
            )
            for fn in appliers:
                self._safe_call(fn)
            from ..diagrams._dispatch import DIAGRAM_MODIFIED
            self._director.dispatcher.fire(DIAGRAM_MODIFIED, layer=d)

        # Remember this card's commit closure for the debounce-timer
        # flush when Auto-Apply is on. Cleared at the start of every
        # _rebuild(), so stale closures from previous renders don't
        # leak in.
        self._card_commits.append(_commit)

        # Apply + Reset row.
        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        apply_btn = QtWidgets.QPushButton("Apply")
        if self._auto_apply_enabled:
            apply_btn.setText("Apply (auto)")
            apply_btn.setEnabled(False)
            apply_btn.setToolTip(
                "Auto-Apply is on — edits commit live with a short debounce."
            )
        else:
            apply_btn.setToolTip(
                "Apply staged value edits in this layer"
            )
        apply_btn.clicked.connect(_commit)
        row.addWidget(apply_btn)

        reset_btn = QtWidgets.QPushButton("Reset")
        reset_btn.setToolTip(
            "Discard staged edits — reload widget values from the "
            "diagram's current state"
        )
        reset_btn.clicked.connect(self._rebuild)
        row.addWidget(reset_btn)
        row.addStretch(1)

        self._content_layout.addLayout(row)

    # ------------------------------------------------------------------
    # Plan 05 — Auto-Apply machinery
    # ------------------------------------------------------------------

    def _load_auto_apply_pref(self) -> bool:
        """Read the persisted Auto-Apply toggle state. Defaults to OFF."""
        try:
            from qtpy.QtCore import QSettings
            settings = QSettings("apeGmsh", "ResultsViewer")
            val = settings.value(self._AUTO_APPLY_SETTINGS_KEY)
            if val is None:
                return False
            # QSettings returns a string under some bindings.
            if isinstance(val, bool):
                return val
            return str(val).strip().lower() in ("true", "1", "yes")
        except Exception:
            return False

    def _save_auto_apply_pref(self, enabled: bool) -> None:
        try:
            from qtpy.QtCore import QSettings
            settings = QSettings("apeGmsh", "ResultsViewer")
            settings.setValue(self._AUTO_APPLY_SETTINGS_KEY, bool(enabled))
            settings.sync()
        except Exception:
            pass

    def _on_auto_apply_toggled(self, checked: bool) -> None:
        """Slot for the Auto-Apply checkbox."""
        self._auto_apply_enabled = bool(checked)
        self._save_auto_apply_pref(self._auto_apply_enabled)
        # Rebuild the stack so every card's Apply button reflects the
        # new mode (disabled "(auto)" when on, enabled otherwise).
        # Also rewires the per-widget signals for live-commit.
        self._rebuild()

    def _ensure_auto_commit_timer(self) -> None:
        """Lazily construct the single-shot debounce timer."""
        if self._auto_commit_timer is not None:
            return
        from qtpy.QtCore import QTimer
        timer = QTimer(self._widget)
        timer.setSingleShot(True)
        timer.setInterval(self._AUTO_APPLY_DEBOUNCE_MS)
        timer.timeout.connect(self._flush_auto_commits)
        self._auto_commit_timer = timer

    def _kick_debounce(self) -> None:
        """Called by staged-widget signals — restart the debounce timer.

        No-op when Auto-Apply is off. When on, the timer fires
        :meth:`_flush_auto_commits` after the configured debounce.
        """
        if not self._auto_apply_enabled:
            return
        self._ensure_auto_commit_timer()
        # Restart — coalesces rapid widget signals into one commit.
        self._auto_commit_timer.start()

    def _flush_auto_commits(self) -> None:
        """Run every visible card's pending commit closure.

        Closures captured during card build (in
        :meth:`_build_apply_button`) push widget values to their
        diagrams + dispatch DIAGRAM_MODIFIED for render coalescing.
        Running them all is safe — unchanged widgets push identical
        values (no-op setters) and the dispatcher coalesces renders.
        """
        for commit_fn in list(self._card_commits):
            self._safe_call(commit_fn)

    def _stage_with_signal(
        self,
        widget: Any,
        signal_name: str,
        applier: Any,
    ) -> None:
        """Register an applier with the Apply button AND wire its
        widget signal for live commits when Auto-Apply is on.

        Drop-in replacement for the ``self._pending_appliers.append(...)``
        idiom — adds the auto-commit wiring on top so the same call
        site supports both modes.

        ``signal_name`` is the widget signal to subscribe to
        (e.g. ``"valueChanged"`` for QSpinBox, ``"currentTextChanged"``
        for QComboBox). Connection silently no-ops if the signal
        doesn't exist on the widget.
        """
        self._pending_appliers.append(applier)
        if not self._auto_apply_enabled:
            return
        signal = getattr(widget, signal_name, None)
        if signal is None:
            return
        # The slot ignores all signal args — we just want to know
        # something changed.
        def _on_widget_changed(*_args, **_kwargs) -> None:
            self._kick_debounce()
        try:
            signal.connect(_on_widget_changed)
        except Exception:
            pass

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
        from .._log import log_action
        log_action(
            "ui.settings", "reorder_layer",
            layer=d, delta=int(delta), new_idx=new_idx,
        )
        # Manager observers don't fire on direct layers-list mutation;
        # nudge them so the stack rebuilds in the new order.
        if comp_mgr is not None:
            comp_mgr._notify()  # noqa: SLF001
        # Re-stack actors so VTK paint order matches the new layer
        # order. Without this, the registry list updates but the
        # actors paint in their original add-order.
        from ..diagrams._dispatch import LAYER_REORDERED
        self._director.dispatcher.fire(LAYER_REORDERED)

    def _dispatch_kind_panel(self, d: "Diagram") -> None:
        """Dispatch to the right ``_build_*_panel`` for the kind."""
        QtWidgets, _ = _qt()
        kind = d.kind
        if kind == "contour":
            self._build_contour_panel(d)
        elif kind == "line_force":
            self._build_line_force_panel(d)
        elif kind in ("fiber_section", "sand"):
            self._build_point_cloud_panel(d)
        elif kind in ("layer_stack", "gauss_marker"):
            self._build_color_panel(d)
        elif kind == "vector_glyph":
            self._build_vector_panel(d)
        elif kind == "spring_force":
            self._build_spring_panel(d)
        elif kind == "section_cut":
            self._build_section_cut_panel(d)
        elif kind == "isochrone_map":
            self._build_isochrone_map_panel(d)
        elif kind == "isochrone_profile":
            self._build_isochrone_profile_panel(d)
        elif kind == "isochrone_strobe":
            self._build_isochrone_strobe_panel(d)
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
        from ..diagrams._kinds import all_kinds

        catalog = self._ensure_catalog()
        id_to_kind_entry = {k.kind_id: k for k in all_kinds()}

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
        from .._log import log_action
        log_action(
            "ui.settings", "add_layer",
            kind=kind_id, component=data, layer=diagram,
        )
        # Granular dispatch — push the new layer's step + deformation
        # state and re-fire the gate now that it's tagged. Replaces the
        # blanket _refresh_new_layers callback we previously hung off
        # subscribe_diagrams.
        from ..diagrams._dispatch import DIAGRAM_ATTACHED
        self._director.dispatcher.fire(DIAGRAM_ATTACHED, layer=diagram)
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
        from .._log import log_action
        log_action("ui.settings", "delete_layer", layer=d)
        try:
            self._director.registry.remove(d)
        except Exception as exc:
            from .._failures import report
            report("DiagramSettingsTab._on_delete", exc)
            return
        # Granular dispatch — gate refresh + render.
        from ..diagrams._dispatch import DIAGRAM_DETACHED
        self._director.dispatcher.fire(DIAGRAM_DETACHED)
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
        from ..diagrams._kinds import kind_def
        from ..diagrams._selectors import normalize as normalize_selector
        from ..diagrams._styles import ContourStyle
        from .._failures import report

        kind_entry = kind_def(kind_id)
        if kind_entry is None:
            return None

        catalog_entry = next(
            (k for k in self._ensure_catalog() if k.kind_id == kind_id), None,
        )
        component = data or ""
        # SlabSelector requires a non-empty component string. Kinds
        # whose creation form has no Data combo (reactions today) get
        # a synthetic placeholder — the diagram reads its components
        # off a fixed list, so the selector's component value is
        # never consulted.
        if not component and catalog_entry is not None and not catalog_entry.requires_data:
            component = kind_id
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
            from ...opensees._response_catalog import split_canonical_component
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

        # Cmap combo — staged via Apply.
        cmap_combo = QtWidgets.QComboBox()
        cmap_combo.setEditable(True)
        for name in _CMAP_PRESETS:
            cmap_combo.addItem(name)
        current_cmap = getattr(d, "_runtime_cmap", None) or d.spec.style.cmap
        cmap_combo.setCurrentText(current_cmap)
        self._stage_with_signal(
            cmap_combo,
            "currentTextChanged",
            lambda: self._safe_call(d.set_cmap, cmap_combo.currentText()),
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

        _clim_applier = lambda: self._safe_call(
            d.set_clim, float(lo_spin.value()), float(hi_spin.value()),
        )
        self._stage_with_signal(lo_spin, "valueChanged", _clim_applier)
        self._stage_with_signal(hi_spin, "valueChanged", _clim_applier)
        form.addRow("Clim min:", lo_spin)
        form.addRow("Clim max:", hi_spin)

        autofit_btn = QtWidgets.QPushButton("Auto-fit at current step")

        def _autofit() -> None:
            # Auto-fit reads the data and rewrites the spinboxes —
            # the user still has to click Apply to commit, matching
            # the staged-edit contract for every other control.
            new_clim = self._safe_call(d.autofit_clim_at_current_step)
            if new_clim is not None:
                lo_spin.setValue(float(new_clim[0]))
                hi_spin.setValue(float(new_clim[1]))

        autofit_btn.clicked.connect(_autofit)
        self._content_layout.addWidget(autofit_btn)

        # Opacity slider — staged via Apply.
        opacity_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        opacity_slider.setRange(0, 100)
        current_opacity = (
            getattr(d, "_runtime_opacity", None)
            if getattr(d, "_runtime_opacity", None) is not None
            else d.spec.style.opacity
        )
        opacity_slider.setValue(int(round(float(current_opacity) * 100)))
        self._stage_with_signal(
            opacity_slider,
            "valueChanged",
            lambda: self._safe_call(d.set_opacity, opacity_slider.value() / 100.0),
        )
        form.addRow("Opacity:", opacity_slider)

        # Scalar-bar live controls (Show + Format) — shared with the
        # generic color panel used by fiber / layer / gauss_marker /
        # vector_glyph so every contour-bearing diagram gets the same
        # UI surface.
        self._add_scalar_bar_controls(d, form)

    # ------------------------------------------------------------------
    # Section cut panel
    # ------------------------------------------------------------------

    def _build_section_cut_panel(self, d: Diagram) -> None:
        """Section-cut layer settings — side, label, and filter-highlight
        toggle.

        Side and label are committed immediately via the rebuild-and-
        replace path (same pattern as ``_on_data_swap``): a new
        ``SectionCutDef`` + ``SectionCutDiagram`` is constructed and
        swapped into the registry / composition. The plane geometry
        and the element filter remain immutable in this phase — those
        edits land in v2.3 once a clean preflight check exists.

        Filter-highlight is the one staged Apply control because it
        only mutates the runtime overlay (no rebuild required).
        """
        QtWidgets, _ = _qt()
        form = QtWidgets.QFormLayout()
        self._content_layout.addLayout(form)

        cut = self._section_cut_def(d)
        current_side = cut.side if cut is not None else "positive"
        current_label = (
            (cut.label if cut is not None else None)
            or d.spec.label or ""
        )

        # Side combobox — fires on commit, rebuilds the diagram.
        side_combo = QtWidgets.QComboBox()
        side_combo.addItem("positive")
        side_combo.addItem("negative")
        side_combo.setCurrentText(current_side)
        side_combo.currentTextChanged.connect(
            lambda new_side: self._on_section_cut_rebuild(d, side=new_side)
        )
        form.addRow("Kept side:", side_combo)

        # Label edit — commits on Enter / focus loss.
        label_edit = QtWidgets.QLineEdit()
        label_edit.setText(current_label)
        label_edit.editingFinished.connect(
            lambda: self._on_section_cut_rebuild(
                d, label=label_edit.text(),
            )
        )
        form.addRow("Label:", label_edit)

        # Show-filter checkbox — staged via the per-card Apply button.
        chk = QtWidgets.QCheckBox("Show filter elements")
        current = bool(getattr(d, "show_filter", False))
        chk.setChecked(current)
        self._pending_appliers.append(
            lambda: self._safe_call(d.set_show_filter, bool(chk.isChecked()))
        )
        form.addRow("", chk)

    @staticmethod
    def _section_cut_def(d: "Diagram") -> Any:
        """Pull the :class:`SectionCutDef` carried on a SectionCutStyle."""
        style = getattr(d.spec, "style", None)
        return getattr(style, "cut", None) if style is not None else None

    def _on_section_cut_rebuild(
        self,
        old: "Diagram",
        *,
        side: Optional[str] = None,
        label: Optional[str] = None,
    ) -> None:
        """Rebuild a SectionCutDiagram with an edited side or label and
        swap it into the registry.

        Mirrors :meth:`_on_data_swap` — preserves z-position by going
        through ``registry.replace``; updates the owning composition's
        layer list so the outline / settings panel re-bind to the new
        instance.

        Runtime state worth preserving across the swap:

        * ``show_filter`` — the user's filter-highlight toggle. We thread
          it into the new style as ``show_filter_initially`` so the new
          diagram attaches with the overlay on.

        No-op when nothing actually changed (avoids spurious rebuilds
        on focus-loss with an unchanged label).
        """
        from dataclasses import replace as dc_replace
        from ..diagrams._base import DiagramSpec
        from ..diagrams._section_cut import SectionCutDiagram
        from ..diagrams._selectors import SlabSelector
        from ..diagrams._styles import SectionCutStyle
        from .._failures import report

        cut = self._section_cut_def(old)
        old_style = getattr(old.spec, "style", None)
        if cut is None or not isinstance(old_style, SectionCutStyle):
            return

        new_side = side if side is not None else cut.side
        if side is not None and new_side not in ("positive", "negative"):
            return
        new_label_raw = (
            label if label is not None
            else (cut.label or "")
        )
        new_label: Optional[str] = (
            (new_label_raw.strip() or None)
            if isinstance(new_label_raw, str) else None
        )

        if new_side == cut.side and new_label == cut.label:
            return     # no-op

        try:
            new_cut = dc_replace(cut, side=new_side, label=new_label)
        except Exception as exc:
            report("DiagramSettingsTab._on_section_cut_rebuild", exc)
            return

        carry_runtime = bool(getattr(old, "show_filter", False))
        new_style = SectionCutStyle(
            cut=new_cut,
            kept_color=old_style.kept_color,
            discarded_color=old_style.discarded_color,
            quad_opacity=old_style.quad_opacity,
            edge_color=old_style.edge_color,
            show_edges=old_style.show_edges,
            show_normal_arrow=old_style.show_normal_arrow,
            normal_arrow_fraction=old_style.normal_arrow_fraction,
            highlight_color=old_style.highlight_color,
            highlight_opacity=old_style.highlight_opacity,
            show_filter_initially=carry_runtime,
        )
        display_label = new_label or "section cut"
        new_spec = DiagramSpec(
            kind=old.spec.kind,
            selector=SlabSelector(component=display_label),
            style=new_style,
            stage_id=old.spec.stage_id,
            visible=old.spec.visible,
            label=display_label,
        )
        tag_map = getattr(old, "_tag_map", None)
        try:
            new_diagram = SectionCutDiagram(
                new_spec, old._results, tag_map=tag_map,  # noqa: SLF001
            )
        except Exception as exc:
            report("DiagramSettingsTab._on_section_cut_rebuild", exc)
            return

        # Composition swap mirrors _on_data_swap.
        comp_mgr = self._director.compositions
        comp = (
            comp_mgr.composition_for_layer(old) if comp_mgr is not None
            else None
        )
        try:
            self._director.registry.replace(old, new_diagram)
        except Exception as exc:
            report("DiagramSettingsTab._on_section_cut_rebuild", exc)
            return
        if comp is not None and comp_mgr is not None:
            try:
                idx = comp.layers.index(old)
                comp.layers[idx] = new_diagram
                comp_mgr._notify()    # noqa: SLF001
            except ValueError:
                pass

        if self._selected is old:
            self._selected = new_diagram
        self._rebuild()

    # ------------------------------------------------------------------
    # Line force panel
    # ------------------------------------------------------------------

    def _build_line_force_panel(self, d: Diagram) -> None:
        QtWidgets, QtCore = _qt()
        form = QtWidgets.QFormLayout()
        self._content_layout.addLayout(form)

        # Scale — committed by the per-card Apply button.
        scale_spin = QtWidgets.QDoubleSpinBox()
        scale_spin.setRange(0.0, 1e9)
        scale_spin.setDecimals(6)
        scale_spin.setSingleStep(0.1)
        scale_spin.setValue(float(d.current_scale()))
        self._stage_with_signal(
            scale_spin,
            "valueChanged",
            lambda: self._safe_call(d.set_scale, float(scale_spin.value())),
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

        def _apply_axis() -> None:
            from .._log import log_action
            new_axis = axis_combo.itemData(axis_combo.currentIndex())
            log_action(
                "ui.settings", "fill_axis_changed",
                layer=d, axis=str(new_axis),
            )
            self._safe_call(d.set_fill_axis, new_axis)

        # Fill-axis change forces a full re-attach (the per-station
        # fill directions are baked at attach), so it has to commit
        # via Apply alongside scale / flip — clicking the combo no
        # longer rebuilds the diagram on its own.
        self._pending_appliers.append(_apply_axis)
        form.addRow("Fill axis:", axis_combo)

        # Flip sign — staged via Apply.
        flip_chk = QtWidgets.QCheckBox("Flip sign")
        runtime_flip = getattr(d, "_runtime_flip", None)
        style_flip = getattr(d.spec.style, "flip_sign", False)
        flip_chk.setChecked(bool(
            runtime_flip if runtime_flip is not None else style_flip
        ))
        self._pending_appliers.append(
            lambda: self._safe_call(d.set_flip_sign, bool(flip_chk.isChecked()))
        )
        form.addRow("", flip_chk)

    # ------------------------------------------------------------------
    # Point-cloud panel (dot size + cmap/clim) — fiber section & sand
    # ------------------------------------------------------------------

    def _build_point_cloud_panel(self, d: Diagram) -> None:
        QtWidgets, _ = _qt()
        form = QtWidgets.QFormLayout()
        self._content_layout.addLayout(form)

        size_spin = QtWidgets.QDoubleSpinBox()
        size_spin.setRange(0.5, 100.0)
        size_spin.setDecimals(1)
        size_spin.setSingleStep(1.0)
        size_spin.setSuffix(" px")
        size_spin.setValue(float(d.current_point_size()))
        self._stage_with_signal(
            size_spin,
            "valueChanged",
            lambda: self._safe_call(d.set_point_size, float(size_spin.value())),
        )
        form.addRow("Point size:", size_spin)

        # Color settings reuse the shared color panel
        self._build_color_panel(d)

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
        self._stage_with_signal(
            scale_spin,
            "valueChanged",
            lambda: self._safe_call(d.set_scale, float(scale_spin.value())),
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
        self._stage_with_signal(
            scale_spin,
            "valueChanged",
            lambda: self._safe_call(d.set_scale, float(scale_spin.value())),
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

        self._pending_appliers.append(
            lambda: self._safe_call(
                d.set_direction,
                (
                    float(spins[0].value()),
                    float(spins[1].value()),
                    float(spins[2].value()),
                ),
            )
        )
        self._content_layout.addLayout(dir_row)

    # ------------------------------------------------------------------
    # Isochrone panels
    #
    # The isochrone kinds derive their whole content at attach time (an
    # arrival field, a sampled path, a frame set), so their defining
    # knobs cannot be pushed onto a live layer the way a colormap can —
    # editing one means rebuilding the diagram. ``_rebuild_with_style``
    # is the shared commit path; the purely visual controls (cmap, clim,
    # opacity, scalar bar) still go through the normal staged setters.
    # ------------------------------------------------------------------

    def _rebuild_with_style(self, old: "Diagram", **style_changes: Any) -> None:
        """Replace ``old`` with the same diagram carrying an edited style.

        Mirrors :meth:`_on_data_swap`: ``registry.replace`` keeps the
        layer's z-position and the composition membership is patched in
        place, so a threshold tweak doesn't send the layer to the top of
        the stack. A construction / attach failure leaves the original
        layer untouched and reports through the slot-failure pipeline.
        """
        import dataclasses

        from ..diagrams._base import DiagramSpec
        from ..diagrams._kinds import kind_def
        from .._failures import report

        entry = kind_def(old.kind)
        if entry is None:
            return
        try:
            new_style = dataclasses.replace(old.spec.style, **style_changes)
        except Exception as exc:
            report("DiagramSettingsTab._rebuild_with_style", exc)
            return
        spec = DiagramSpec(
            kind=old.kind,
            selector=old.spec.selector,
            style=new_style,
            stage_id=old.spec.stage_id,
            visible=old.spec.visible,
            label=old.spec.label,
        )
        try:
            new_diagram = entry.diagram_class(spec, self._director.results)
        except Exception as exc:
            report("DiagramSettingsTab._rebuild_with_style", exc)
            return

        comp_mgr = self._director.compositions
        comp = (
            comp_mgr.composition_for_layer(old) if comp_mgr is not None
            else None
        )
        try:
            self._director.registry.replace(old, new_diagram)
        except Exception as exc:
            report("DiagramSettingsTab._rebuild_with_style", exc)
            return
        if comp is not None and comp_mgr is not None:
            try:
                idx = comp.layers.index(old)
                comp.layers[idx] = new_diagram
                comp_mgr._notify()  # noqa: SLF001
            except ValueError:
                pass
        self._selected = new_diagram
        self._rebuild()

    def _add_readout(self, text: str, tooltip: str = "") -> None:
        """A wrapping, dimmed, read-only line in the current card."""
        QtWidgets, _ = _qt()
        label = QtWidgets.QLabel(text)
        label.setWordWrap(True)
        label.setStyleSheet("color: gray; font-style: italic;")
        if tooltip:
            label.setToolTip(tooltip)
        self._content_layout.addWidget(label)

    def _build_isochrone_map_panel(self, d: Diagram) -> None:
        """Arrival-criterion controls (rebuild) + colour controls (live)."""
        QtWidgets, QtCore = _qt()
        style = d.spec.style

        # What the map is actually showing — with an auto threshold the
        # user cannot otherwise know what level was applied.
        describe = getattr(d, "describe_criterion", None)
        if callable(describe):
            self._add_readout(
                self._safe_call(describe) or "",
                "The criterion this map applied, and how many of the "
                "selected nodes satisfied it.",
            )

        form = QtWidgets.QFormLayout()
        self._content_layout.addLayout(form)

        mode_combo = QtWidgets.QComboBox()
        mode_combo.addItem("First crossing", "first_crossing")
        mode_combo.addItem("Time to peak", "time_to_peak")
        mode_combo.setToolTip(
            "First crossing: the time the tracked value first reaches\n"
            "the threshold — nodes that never do stay unpainted.\n"
            "Time to peak: the time of each node's maximum. Always\n"
            "defined, so every selected node is painted."
        )
        mode_idx = mode_combo.findData(style.mode)
        if mode_idx >= 0:
            mode_combo.setCurrentIndex(mode_idx)
        mode_combo.currentIndexChanged.connect(
            lambda _i: self._rebuild_with_style(
                d, mode=mode_combo.currentData(),
            ),
        )
        form.addRow("Criterion:", mode_combo)

        is_crossing = style.mode == "first_crossing"

        # Threshold: an explicit level, or the auto fraction of the peak.
        auto_chk = QtWidgets.QCheckBox("Auto (fraction of peak)")
        auto_chk.setChecked(style.threshold is None)
        auto_chk.setEnabled(is_crossing)
        form.addRow("Threshold:", auto_chk)

        frac_spin = QtWidgets.QDoubleSpinBox()
        frac_spin.setRange(1e-6, 1.0)
        frac_spin.setDecimals(4)
        frac_spin.setSingleStep(0.05)
        frac_spin.setValue(float(style.threshold_fraction))
        frac_spin.setEnabled(is_crossing and style.threshold is None)
        form.addRow("Fraction:", frac_spin)

        level_spin = QtWidgets.QDoubleSpinBox()
        level_spin.setRange(-1e30, 1e30)
        level_spin.setDecimals(6)
        used = self._safe_call(lambda: float(d.threshold_used))
        level_spin.setValue(
            float(style.threshold) if style.threshold is not None
            else float(used or 0.0)
        )
        level_spin.setEnabled(is_crossing and style.threshold is not None)
        form.addRow("Level:", level_spin)

        def _sync_threshold_enabled(checked: bool) -> None:
            frac_spin.setEnabled(is_crossing and checked)
            level_spin.setEnabled(is_crossing and not checked)

        auto_chk.toggled.connect(_sync_threshold_enabled)

        apply_btn = QtWidgets.QPushButton("Recompute arrival")
        apply_btn.setEnabled(is_crossing)
        apply_btn.setToolTip(
            "Rebuild the map with the criterion above. The arrival field "
            "is a whole-history reduction, so it cannot be re-coloured "
            "in place."
        )
        apply_btn.clicked.connect(
            lambda: self._rebuild_with_style(
                d,
                threshold=(
                    None if auto_chk.isChecked()
                    else float(level_spin.value())
                ),
                threshold_fraction=float(frac_spin.value()),
            ),
        )
        self._content_layout.addWidget(apply_btn)

        abs_chk = QtWidgets.QCheckBox("Track |value|")
        abs_chk.setChecked(bool(style.use_abs))
        abs_chk.setToolTip(
            "On: an arrival is a departure from zero in either "
            "direction (what a wavefront wants).\n"
            "Off: a signed level crossing."
        )
        abs_chk.toggled.connect(
            lambda checked: self._rebuild_with_style(d, use_abs=bool(checked)),
        )
        form.addRow("", abs_chk)

        interp_chk = QtWidgets.QCheckBox("Interpolate crossing time")
        interp_chk.setChecked(bool(style.interpolate))
        interp_chk.setEnabled(is_crossing)
        interp_chk.setToolTip(
            "Interpolate between the bracketing steps instead of "
            "snapping to the later one — removes step-quantized banding."
        )
        interp_chk.toggled.connect(
            lambda checked: self._rebuild_with_style(
                d, interpolate=bool(checked),
            ),
        )
        form.addRow("", interp_chk)

        # Opacity is a live runtime override, like the contour's.
        opacity_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        opacity_slider.setRange(0, 100)
        current_opacity = (
            getattr(d, "_runtime_opacity", None)
            if getattr(d, "_runtime_opacity", None) is not None
            else style.opacity
        )
        opacity_slider.setValue(int(round(float(current_opacity) * 100)))
        self._stage_with_signal(
            opacity_slider,
            "valueChanged",
            lambda: self._safe_call(
                d.set_opacity, opacity_slider.value() / 100.0,
            ),
        )
        form.addRow("Opacity:", opacity_slider)

        self._build_color_panel(d)

    def _build_isochrone_profile_panel(self, d: Diagram) -> None:
        """Path / curve-family controls — all rebuild-on-change."""
        QtWidgets, _ = _qt()
        style = d.spec.style

        resolved = self._safe_call(lambda: d.path_axis_name)
        n_nodes = self._safe_call(
            lambda: 0 if d.path_node_ids is None else int(d.path_node_ids.size)
        )
        self._add_readout(
            f"Path: {n_nodes or 0} node(s) ordered along {resolved or '?'}. "
            f"The chart is in the plot pane.",
            "Position is the coordinate on this axis — not arc length. "
            "A selection that doubles back on it will read as a zig-zag.",
        )

        form = QtWidgets.QFormLayout()
        self._content_layout.addLayout(form)

        axis_combo = QtWidgets.QComboBox()
        axis_combo.addItem("Auto (largest extent)", "auto")
        for name in ("x", "y", "z"):
            axis_combo.addItem(name, name)
        idx = axis_combo.findData(style.path_axis)
        if idx >= 0:
            axis_combo.setCurrentIndex(idx)
        axis_combo.currentIndexChanged.connect(
            lambda _i: self._rebuild_with_style(
                d, path_axis=axis_combo.currentData(),
            ),
        )
        form.addRow("Path axis:", axis_combo)

        value_combo = QtWidgets.QComboBox()
        value_combo.addItem("Auto", "auto")
        value_combo.addItem("Value horizontal", "horizontal")
        value_combo.addItem("Value vertical", "vertical")
        value_combo.setToolTip(
            "Auto draws a z-path upright (depth vertical), the "
            "geotechnical convention, and everything else with position "
            "on the x-axis."
        )
        idx = value_combo.findData(style.value_axis)
        if idx >= 0:
            value_combo.setCurrentIndex(idx)
        value_combo.currentIndexChanged.connect(
            lambda _i: self._rebuild_with_style(
                d, value_axis=value_combo.currentData(),
            ),
        )
        form.addRow("Chart layout:", value_combo)

        curves_spin = QtWidgets.QSpinBox()
        curves_spin.setRange(1, 200)
        curves_spin.setValue(int(style.n_curves))
        curves_spin.setToolTip(
            "How many instants to draw, spread evenly over the stage. "
            "The first and last steps are always included."
        )
        curves_spin.editingFinished.connect(
            lambda: self._rebuild_with_style(
                d, n_curves=int(curves_spin.value()),
            ),
        )
        form.addRow("Curves:", curves_spin)

        cmap_combo = QtWidgets.QComboBox()
        cmap_combo.setEditable(True)
        for name in _CMAP_PRESETS:
            cmap_combo.addItem(name)
        cmap_combo.setCurrentText(style.cmap)
        cmap_combo.setToolTip("Colormap sampled along time for the curves.")
        cmap_combo.activated.connect(
            lambda _i: self._rebuild_with_style(
                d, cmap=cmap_combo.currentText(),
            ),
        )
        form.addRow("Curve colors:", cmap_combo)

        markers_chk = QtWidgets.QCheckBox("Markers at sampled nodes")
        markers_chk.setChecked(bool(style.show_markers))
        markers_chk.toggled.connect(
            lambda checked: self._rebuild_with_style(
                d, show_markers=bool(checked),
            ),
        )
        form.addRow("", markers_chk)

        current_chk = QtWidgets.QCheckBox("Highlight current step")
        current_chk.setChecked(bool(style.mark_current_step))
        current_chk.toggled.connect(
            lambda checked: self._rebuild_with_style(
                d, mark_current_step=bool(checked),
            ),
        )
        form.addRow("", current_chk)

        path_chk = QtWidgets.QCheckBox("Show path in 3-D")
        path_chk.setChecked(bool(style.show_path))
        path_chk.toggled.connect(
            lambda checked: self._rebuild_with_style(
                d, show_path=bool(checked),
            ),
        )
        form.addRow("", path_chk)

    def _build_isochrone_strobe_panel(self, d: Diagram) -> None:
        """Frame-set controls (rebuild) + scale / colour controls (live)."""
        QtWidgets, QtCore = _qt()
        style = d.spec.style

        times = self._safe_call(lambda: d.frame_times)
        if times is not None and len(times):
            self._add_readout(
                f"{len(times)} frames, t = {float(times[0]):.4g} … "
                f"{float(times[-1]):.4g}.",
                "The instants drawn. Fixed at attach — scrubbing the "
                "time cursor does not change them.",
            )

        form = QtWidgets.QFormLayout()
        self._content_layout.addLayout(form)

        field_combo = QtWidgets.QComboBox()
        field_combo.setEditable(True)
        for name in ("displacement", "velocity", "acceleration"):
            field_combo.addItem(name)
        field_combo.setCurrentText(style.field)
        field_combo.setToolTip(
            "Nodal vector prefix driving the warp; the diagram reads "
            "<field>_x/_y/_z."
        )
        field_combo.activated.connect(
            lambda _i: self._rebuild_with_style(
                d, field=field_combo.currentText().strip(),
            ),
        )
        form.addRow("Field:", field_combo)

        frames_spin = QtWidgets.QSpinBox()
        frames_spin.setRange(1, 100)
        frames_spin.setValue(int(style.n_frames))
        frames_spin.setToolTip(
            "How many instants to superimpose, spread evenly over the "
            "stage. The first and last steps are always included."
        )
        frames_spin.editingFinished.connect(
            lambda: self._rebuild_with_style(
                d, n_frames=int(frames_spin.value()),
            ),
        )
        form.addRow("Frames:", frames_spin)

        # Scale re-warps the cached frames in place — no rebuild.
        scale_spin = QtWidgets.QDoubleSpinBox()
        scale_spin.setRange(0.0, 1e9)
        scale_spin.setDecimals(6)
        scale_spin.setValue(float(self._safe_call(lambda: d.scale_used) or 1.0))
        scale_spin.setToolTip(
            "Warp amplification. Auto-fitted at attach so the largest "
            "frame reaches a fraction of the model diagonal."
        )
        self._stage_with_signal(
            scale_spin,
            "valueChanged",
            lambda: self._safe_call(d.set_scale, float(scale_spin.value())),
        )
        form.addRow("Scale:", scale_spin)

        opacity_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        opacity_slider.setRange(0, 100)
        current_opacity = (
            getattr(d, "_runtime_opacity", None)
            if getattr(d, "_runtime_opacity", None) is not None
            else style.opacity
        )
        opacity_slider.setValue(int(round(float(current_opacity) * 100)))
        self._stage_with_signal(
            opacity_slider,
            "valueChanged",
            lambda: self._safe_call(
                d.set_opacity, opacity_slider.value() / 100.0,
            ),
        )
        form.addRow("Opacity:", opacity_slider)

        self._build_color_panel(d)

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
        self._stage_with_signal(
            cmap_combo,
            "currentTextChanged",
            lambda: self._safe_call(d.set_cmap, cmap_combo.currentText()),
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

        _clim_applier = lambda: self._safe_call(
            d.set_clim, float(lo_spin.value()), float(hi_spin.value()),
        )
        self._stage_with_signal(lo_spin, "valueChanged", _clim_applier)
        self._stage_with_signal(hi_spin, "valueChanged", _clim_applier)
        form.addRow("Clim min:", lo_spin)
        form.addRow("Clim max:", hi_spin)

        autofit = QtWidgets.QPushButton("Auto-fit at current step")

        def _autofit() -> None:
            # Auto-fit reads the data and rewrites the spinboxes —
            # the user still has to click Apply to commit, matching
            # the staged-edit contract for every other control.
            new_clim = self._safe_call(d.autofit_clim_at_current_step)
            if new_clim is not None:
                lo_spin.setValue(float(new_clim[0]))
                hi_spin.setValue(float(new_clim[1]))

        autofit.clicked.connect(_autofit)
        self._content_layout.addWidget(autofit)

        # Scalar-bar controls — every diagram routed to this
        # panel inherits ScalarBarSupport, so the setters exist.
        # Show / Format are staged via Apply alongside the rest.
        self._add_scalar_bar_controls(d, form)

    # ------------------------------------------------------------------
    # Scalar-bar controls — Show / Format / Orientation / Size
    # ------------------------------------------------------------------

    def _add_scalar_bar_controls(self, d: Diagram, form: Any) -> None:
        """Append Show-scale, Format, Orientation and Size controls.

        Skips silently when the diagram doesn't support the live API
        (older diagrams not yet on ``ScalarBarSupport``). The form
        owner — the caller — keeps its existing layout. All controls
        are staged via the per-card Apply button alongside cmap / clim.
        """
        if not hasattr(d, "set_show_scalar_bar") or not hasattr(d, "set_fmt"):
            return
        QtWidgets, _ = _qt()

        # Widgets are projections of the LegendController (ADR 0081):
        # every current value is read back through the diagram's
        # _effective_* accessors, which resolve against the owning
        # legend entry and fall back to the style only before attach.
        show_chk = QtWidgets.QCheckBox("Show scale")
        show_chk.setChecked(bool(self._safe_call(d._effective_show_scalar_bar)))
        self._pending_appliers.append(
            lambda: self._safe_call(d.set_show_scalar_bar, bool(show_chk.isChecked()))
        )
        form.addRow(show_chk)

        fmt_edit = QtWidgets.QLineEdit()
        fmt_edit.setPlaceholderText("%.3g")
        fmt_edit.setText(self._safe_call(d._effective_fmt) or "%.3g")
        fmt_edit.setToolTip(
            "printf-style format for scale tick labels.\n"
            "Examples: %.3g (general, 3 digits), %.2e (exponent),\n"
            "%.4f (fixed, 4 decimals).\n"
            "The scale resizes to fit the labels this produces."
        )
        self._pending_appliers.append(
            lambda: self._safe_call(d.set_fmt, fmt_edit.text() or "%.3g")
        )
        form.addRow("Format:", fmt_edit)

        if not hasattr(d, "set_scalar_bar_vertical"):
            return

        orient_combo = QtWidgets.QComboBox()
        orient_combo.addItems(["Horizontal", "Vertical"])
        orient_combo.setCurrentIndex(
            1 if self._safe_call(d._effective_bar_vertical) else 0
        )
        self._pending_appliers.append(
            lambda: self._safe_call(
                d.set_scalar_bar_vertical,
                orient_combo.currentText() == "Vertical",
            )
        )
        form.addRow("Orientation:", orient_combo)

        size_spin = QtWidgets.QDoubleSpinBox()
        size_spin.setRange(_MIN_FONT_SCALE, _MAX_FONT_SCALE)
        size_spin.setSingleStep(0.1)
        size_spin.setDecimals(2)
        size_spin.setSuffix(" ×")
        size_spin.setValue(float(self._safe_call(d._effective_bar_scale) or 1.0))
        size_spin.setToolTip(
            "Text size of the scale (1.0 = default).\n"
            "The box is sized to fit the text, so this scales the whole\n"
            "legend without ever clipping its labels."
        )
        self._pending_appliers.append(
            lambda: self._safe_call(d.set_scalar_bar_scale, float(size_spin.value()))
        )
        form.addRow("Size:", size_spin)

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
            # No render here (ADR 0056 Part 4): appliers run inside a
            # commit that ends with a DIAGRAM_MODIFIED fire — the
            # dispatcher's coalesced render is the only render path.
            # (The old per-applier plotter.render() made every commit
            # render once per applier.)
            return fn(*args, **kwargs)
        except Exception as exc:
            import sys
            print(
                f"[DiagramSettingsTab] {fn.__qualname__} raised: {exc}",
                file=sys.stderr,
            )
            return None


# =========================================================================
# Card focus filter (plan 04 step 2 cont.)
# =========================================================================


def _make_card_focus_filter_class():
    """Build the :class:`_CardFocusFilter` class on demand.

    Lazy Qt import keeps the module body importable in headless
    contexts (mirrors the pattern used by ``_qt()`` at the top).
    """
    from qtpy import QtCore

    class _CardFocusFilter(QtCore.QObject):
        """Event filter that broadcasts card focus on mouse-down.

        Installed by :meth:`DiagramSettingsTab._install_card_focus_filter`
        on every widget inside one layer card. When any descendant
        widget receives ``QEvent.MouseButtonPress``, the filter notifies
        the parent tab that this card's diagram should become the
        active layer. The filter never consumes the event — it just
        observes — so normal widget behaviour (spinbox focus, button
        clicks, etc.) is unaffected.
        """
        def __init__(self, tab: Any, diagram: Any) -> None:
            # Tab is not a QObject (composes one), so we can't parent
            # to it directly. Lifetime is managed by the tab's
            # ``_card_focus_filters`` list — clear that on rebuild and
            # the filter goes away.
            super().__init__(None)
            self._tab = tab
            self._diagram = diagram

        def eventFilter(self, obj: Any, event: Any) -> bool:    # noqa: ARG002
            if event.type() == QtCore.QEvent.MouseButtonPress:
                self._tab._fire_layer_focused(self._diagram)
            return False    # never consume

    return _CardFocusFilter


_CardFocusFilterClass: Any = None


def _resolve_card_focus_filter_class() -> Any:
    """Build the focus-filter class on first call; cache thereafter."""
    global _CardFocusFilterClass
    if _CardFocusFilterClass is None:
        _CardFocusFilterClass = _make_card_focus_filter_class()
    return _CardFocusFilterClass
