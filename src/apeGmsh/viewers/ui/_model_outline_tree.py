"""ModelOutlineTree — left-rail navigator for ``model.viewer``.

ParaView-style outline tree showing what's in the model: physical
groups + parts. Sits in the left dock area as primary navigation,
parallel to :class:`OutlineTree` in ``results.viewer``.

The right-side ``BrowserTab`` continues to coexist during this
transition — same data, different surface. Once the outline is the
preferred navigator, the Browser tab can be removed in a follow-up.

Top-level groups
----------------
* **Physical Groups** — user-facing physical groups (skips internal
  ``_label:`` prefixed ones). Click activates the group (same as
  the Browser tab); the eye toggles visibility for every member via
  :class:`VisibilityManager`.
* **Labels** — the Tier-1 internal labels (the ``_label:*`` PGs,
  prefix stripped). Hidden entirely when the model carries none.
* **Parts** — the session's :class:`PartsRegistry` instances, when
  present. Click a part to select all its entities; the eye toggles
  visibility for the union of its entities.

Each leaf row (entity DimTag) gets its own eye icon for per-entity
control. Right-click menus handle rename/delete on groups and
Hide/Isolate/Reveal-all on any visibility-bearing row.

Layout is ParaView-faithful: the eye is a real DecorationRole icon
in its own fixed-width column (logical column 1), moved to render
*before* the name — exactly like ParaView's ``pqPipelineModel``
(eye = column 1, header ``moveSection(1, 0)``). It can never paint
over the label the way an over-painted delegate did.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional

import gmsh

from ._eye_icon_delegate import ROLE_VISIBLE

if TYPE_CHECKING:
    from apeGmsh._types import DimTag
    from ..core.selection import SelectionState
    from ..core.visibility import VisibilityManager


def _qt():
    from qtpy import QtCore, QtGui, QtWidgets
    return QtCore, QtGui, QtWidgets


def _theme():
    from .theme import THEME
    return THEME


# Role constants — distinct from the Browser tab's so the two trees
# can coexist without subtle confusion if a future refactor pulls
# items between them.
_ROLE_KIND = int(0x0200)        # "group" | "label" | "entity" | "part" | "header"
_ROLE_PAYLOAD = int(0x0201)     # name (group/label/part) | DimTag (entity)

# ParaView-faithful eye column. Column 0 = name (+ branch arrows +
# theme-coloured text); column 1 = the visibility eye, rendered as a
# real DecorationRole icon in its own fixed-width column. The eye
# section is moved to visual position 0 so it draws *before* the
# label (ParaView: pqPipelineModel eye = column 1, moveSection(1, 0)).
_EYE_COL = 1
_EYE_COL_WIDTH = 24
_EYE_GLYPH_SIZE = 16


class ModelOutlineTree:
    """Left-rail outline tree for ``model.viewer``.

    Parameters
    ----------
    selection
        The viewer's :class:`SelectionState` — drives the active-group
        highlight + click-to-toggle on entity rows.
    vis_mgr
        The viewer's :class:`VisibilityManager` — read for eye state,
        mutated on click.
    parts_registry
        Optional :class:`PartsRegistry` (``g.parts``). When ``None``,
        the Parts group is hidden.
    on_group_activated
        Callback fired when a Physical Group row is clicked — same
        contract as :class:`BrowserTab.on_group_activated`.
    on_entity_toggled
        Callback fired when an entity leaf is clicked — same contract
        as :class:`BrowserTab.on_entity_toggled`.
    on_rename_group / on_delete_group / on_new_group
        Optional handlers wired into the group-row context menu.
    on_new_label
        Optional handler for creating a (multi-dimensional) label
        from the current selection. When both this and
        ``on_new_group`` are given, the header ``+`` becomes a small
        menu offering "New physical group" / "New label".
    on_rename_label / on_delete_label
        Optional handlers wired into the label-row context menu
        (Rename / Delete), the multi-dim counterpart of
        ``on_rename_group`` / ``on_delete_group``.
    """

    def __init__(
        self,
        selection: "SelectionState",
        vis_mgr: "VisibilityManager",
        parts_registry: Any = None,
        *,
        on_group_activated: Optional[Callable[[str], None]] = None,
        on_entity_toggled: Optional[Callable[["DimTag"], None]] = None,
        on_rename_group: Optional[Callable[[str], None]] = None,
        on_delete_group: Optional[Callable[[str], None]] = None,
        on_new_group: Optional[Callable[[], None]] = None,
        on_new_label: Optional[Callable[[], None]] = None,
        on_rename_label: Optional[Callable[[str], None]] = None,
        on_delete_label: Optional[Callable[[str], None]] = None,
        on_row_focused: Optional[Callable[[str, Any], None]] = None,
    ) -> None:
        QtCore, QtGui, QtWidgets = _qt()
        self._selection = selection
        self._vis_mgr = vis_mgr
        self._parts = parts_registry
        self._on_group_activated = on_group_activated
        self._on_entity_toggled = on_entity_toggled
        self._on_rename_group = on_rename_group
        self._on_delete_group = on_delete_group
        self._on_new_group = on_new_group
        self._on_new_label = on_new_label
        self._on_rename_label = on_rename_label
        self._on_delete_label = on_delete_label
        # Generic row-focused signal — fires for every selectable row
        # with ``(kind, payload)``. Viewers map kinds to tab names and
        # call ``win.focus_tab(...)`` to reveal the property editor.
        self._on_row_focused = on_row_focused

        # ── Outer container + header ────────────────────────────────
        widget = QtWidgets.QWidget()
        widget.setObjectName("ModelOutlineTree")
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QtWidgets.QFrame()
        header.setObjectName("OutlineHeader")
        header_lay = QtWidgets.QHBoxLayout(header)
        header_lay.setContentsMargins(10, 4, 6, 4)
        label = QtWidgets.QLabel("OUTLINE")
        label.setObjectName("OutlineHeaderLabel")
        header_lay.addWidget(label)
        header_lay.addStretch(1)
        if self._on_new_group is not None or self._on_new_label is not None:
            btn_new = QtWidgets.QPushButton("+")
            btn_new.setFlat(True)
            btn_new.setFixedWidth(24)
            if (self._on_new_group is not None
                    and self._on_new_label is not None):
                btn_new.setToolTip(
                    "New physical group / label from the current "
                    "selection"
                )

                def _show_new_menu() -> None:
                    _QtC, _, _QtW = _qt()
                    m = _QtW.QMenu(self._widget)
                    a_pg = m.addAction("New physical group")
                    a_lb = m.addAction("New label")
                    chosen = m.exec_(
                        btn_new.mapToGlobal(
                            btn_new.rect().bottomLeft()
                        )
                    )
                    if chosen == a_pg:
                        self._on_new_group()
                    elif chosen == a_lb:
                        self._on_new_label()

                btn_new.clicked.connect(_show_new_menu)
            elif self._on_new_group is not None:
                btn_new.setToolTip("New physical group")
                btn_new.clicked.connect(lambda: self._on_new_group())
            else:
                btn_new.setToolTip("New label")
                btn_new.clicked.connect(lambda: self._on_new_label())
            header_lay.addWidget(btn_new)
        layout.addWidget(header)

        # ── Tree ────────────────────────────────────────────────────
        tree = QtWidgets.QTreeWidget()
        tree.setObjectName("ModelOutlineTreeWidget")
        tree.setColumnCount(2)              # 0 = name (+ branch), 1 = eye
        tree.setHeaderHidden(True)
        tree.setRootIsDecorated(True)
        tree.setIndentation(14)
        tree.setUniformRowHeights(True)
        tree.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        tree.itemClicked.connect(self._on_item_clicked)
        tree.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        tree.customContextMenuRequested.connect(self._on_context_menu)
        layout.addWidget(tree, stretch=1)
        self._tree = tree

        # ── ParaView-faithful eye column ────────────────────────────
        # The eye is a real DecorationRole icon on column 1 (not an
        # over-painted delegate), so Qt lays it out inside its own
        # fixed-width column and it can never collide with the label.
        # moveSection(1, 0) makes that column render first — the eye
        # sits to the *left* of the name. Expand arrows stay on the
        # tree column (logical 0 = name), matching ParaView.
        header = tree.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(_EYE_COL, QtWidgets.QHeaderView.Fixed)
        tree.setColumnWidth(_EYE_COL, _EYE_COL_WIDTH)
        header.moveSection(_EYE_COL, 0)
        # Pin the decoration size to the pixmap size so Qt never
        # rescales the eye icon (that rescale is what made the dot
        # read as a fuzzy oval).
        tree.setIconSize(QtCore.QSize(_EYE_GLYPH_SIZE, _EYE_GLYPH_SIZE))
        self._eye_icons: dict[bool, Any] = {}
        self._rebuild_eye_icons()

        # ── Top-level groups ────────────────────────────────────────
        self._group_groups = self._make_header_item("Physical Groups")
        self._group_labels = self._make_header_item("Labels")
        self._group_parts = self._make_header_item("Parts")
        tree.addTopLevelItem(self._group_groups)
        tree.addTopLevelItem(self._group_labels)
        tree.addTopLevelItem(self._group_parts)
        self._group_groups.setExpanded(True)
        self._group_labels.setExpanded(True)
        self._group_parts.setExpanded(True)

        self._widget = widget

        # ── Subscribe + populate ────────────────────────────────────
        self.refresh()
        vis_mgr.on_changed.append(self._refresh_eye_states)
        # Eye icons are real pixmaps (not palette-recoloured at paint
        # time like the old delegate), so they must be rebuilt on a
        # theme switch. Mirrors OutlineTree's THEME subscription.
        from .theme import THEME
        self._unsub_theme = THEME.subscribe(
            lambda _p: self._on_theme_changed()
        )

    @property
    def widget(self) -> Any:
        return self._widget

    # ------------------------------------------------------------------
    # Eye-column icons (ParaView-style decoration, theme-following)
    # ------------------------------------------------------------------

    def _rebuild_eye_icons(self) -> None:
        """(Re)build the visibility dot icons in the theme text colour.

        Drawn with ``QPainter.drawEllipse`` (not a font glyph) at the
        view's device-pixel-ratio so the pixmap is never rescaled —
        that rescale is what made the old ●/○ read as a fuzzy oval.
        Filled = visible, hollow ring = hidden. Rebuilt on theme
        change since a real icon can't recolour itself."""
        QtCore, QtGui, _ = _qt()
        color = QtGui.QColor(_theme().current.text)
        s = _EYE_GLYPH_SIZE
        try:
            dpr = float(self._tree.devicePixelRatioF()) or 1.0
        except Exception:
            dpr = 1.0
        d = s * 0.55                       # dot diameter (logical px)
        ring_w = max(1.0, s * 0.09)        # ring stroke — same size-
                                           # proportional logic as the dot
        off = (s - d) / 2.0
        rect = QtCore.QRectF(off, off, d, d)
        icons: dict[bool, Any] = {}
        for visible in (True, False):
            pix = QtGui.QPixmap(round(s * dpr), round(s * dpr))
            pix.setDevicePixelRatio(dpr)
            pix.fill(QtGui.QColor(0, 0, 0, 0))
            painter = QtGui.QPainter(pix)
            painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
            if visible:
                painter.setPen(QtCore.Qt.NoPen)
                painter.setBrush(color)            # solid dot
            else:
                pen = QtGui.QPen(color)
                pen.setWidthF(ring_w)
                painter.setPen(pen)
                painter.setBrush(QtCore.Qt.NoBrush)  # hollow ring
            painter.drawEllipse(rect)
            painter.end()
            icons[visible] = QtGui.QIcon(pix)
        self._eye_icons = icons

    def _set_eye(self, item: Any, visible: bool) -> None:
        """Stamp the visibility state + paint the column-1 eye icon.

        ``ROLE_VISIBLE`` on column 0 stays the source of truth (read
        by :meth:`_group_is_visible`, the ``vis_mgr`` sync path, and
        the tests); the icon is purely derived from it."""
        item.setData(0, ROLE_VISIBLE, bool(visible))
        item.setIcon(_EYE_COL, self._eye_icons[bool(visible)])

    def _on_theme_changed(self) -> None:
        self._rebuild_eye_icons()
        self.refresh()

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        """Full rebuild — call after groups added / renamed / deleted."""
        self._refresh_groups()
        self._refresh_labels()
        self._refresh_parts()

    def update_active(self) -> None:
        """Lightweight refresh after pick changes — re-bold the active
        group, update its child count. Doesn't rebuild the tree."""
        _, QtGui, _ = _qt()
        active = self._selection.active_group
        active_count = len(self._selection.picks)
        for i in range(self._group_groups.childCount()):
            item = self._group_groups.child(i)
            if item.data(0, _ROLE_KIND) != "group":
                continue
            name = item.data(0, _ROLE_PAYLOAD)
            font = item.font(0)
            if name == active:
                font.setBold(True)
                item.setForeground(
                    0, QtGui.QBrush(QtGui.QColor(_theme().current.success)),
                )
                # Column 1 is now the eye; show the staged/persisted
                # pick count as a suffix on the name instead.
                base_count = item.data(0, int(0x0202)) or 0
                item.setText(0, f"{name}   {active_count}/{base_count}")
            else:
                font.setBold(False)
                item.setForeground(
                    0, QtGui.QBrush(QtGui.QColor(_theme().current.info)),
                )
                item.setText(0, name)
            item.setFont(0, font)

    def _refresh_groups(self) -> None:
        QtCore, QtGui, QtWidgets = _qt()
        self._group_groups.takeChildren()
        groups = self._collect_groups()
        if not groups:
            empty = QtWidgets.QTreeWidgetItem(self._group_groups)
            empty.setText(0, "(no groups)")
            flags = empty.flags() & ~QtCore.Qt.ItemIsSelectable
            empty.setFlags(flags)
            empty.setForeground(
                0, QtGui.QBrush(QtGui.QColor(_theme().current.overlay)),
            )
            return

        dim_labels = {0: "pt", 1: "crv", 2: "srf", 3: "vol"}
        active = self._selection.active_group
        for name, _dim, _pg_tag, members in groups:
            item = QtWidgets.QTreeWidgetItem(self._group_groups)
            item.setText(0, name)
            item.setData(0, _ROLE_KIND, "group")
            item.setData(0, _ROLE_PAYLOAD, name)
            item.setData(0, int(0x0202), len(members))    # base count
            self._set_eye(item, self._group_is_visible(members))
            if name == active:
                font = item.font(0)
                font.setBold(True)
                item.setFont(0, font)
                item.setForeground(
                    0, QtGui.QBrush(QtGui.QColor(_theme().current.success)),
                )
            else:
                item.setForeground(
                    0, QtGui.QBrush(QtGui.QColor(_theme().current.info)),
                )
            for dim, tag in members:
                child = QtWidgets.QTreeWidgetItem(item)
                child.setText(0, f"{dim_labels.get(dim, '?')} {tag}")
                child.setData(0, _ROLE_KIND, "entity")
                child.setData(0, _ROLE_PAYLOAD, (dim, tag))
                self._set_eye(
                    child, not self._vis_mgr.is_hidden((dim, tag)),
                )

    def _refresh_labels(self) -> None:
        """Populate the Labels group from internal ``_label:*`` PGs.

        Same shape as :meth:`_refresh_groups` but reads the Tier-1
        label PGs (prefix stripped). The header hides itself when the
        model has no labels — the way Parts does — since most models
        carry none and an empty section is just noise."""
        _, QtGui, QtWidgets = _qt()
        self._group_labels.takeChildren()
        labels = self._collect_labels()
        if not labels:
            self._group_labels.setHidden(True)
            return
        self._group_labels.setHidden(False)

        dim_labels = {0: "pt", 1: "crv", 2: "srf", 3: "vol"}
        for name, _dim, _pg_tag, members in labels:
            item = QtWidgets.QTreeWidgetItem(self._group_labels)
            item.setText(0, name)
            item.setData(0, _ROLE_KIND, "label")
            item.setData(0, _ROLE_PAYLOAD, name)
            item.setForeground(
                0, QtGui.QBrush(QtGui.QColor(_theme().current.info)),
            )
            self._set_eye(item, self._group_is_visible(members))
            for dim, tag in members:
                child = QtWidgets.QTreeWidgetItem(item)
                child.setText(0, f"{dim_labels.get(dim, '?')} {tag}")
                child.setData(0, _ROLE_KIND, "entity")
                child.setData(0, _ROLE_PAYLOAD, (dim, tag))
                self._set_eye(
                    child, not self._vis_mgr.is_hidden((dim, tag)),
                )

    def _refresh_parts(self) -> None:
        _, _, QtWidgets = _qt()
        self._group_parts.takeChildren()
        if self._parts is None or not getattr(self._parts, "instances", None):
            self._group_parts.setHidden(True)
            return
        self._group_parts.setHidden(False)

        dim_labels = {0: "pt", 1: "crv", 2: "srf", 3: "vol"}
        for name in sorted(self._parts.instances.keys()):
            inst = self._parts.instances[name]
            entities = getattr(inst, "entities", {}) or {}
            flat: list[tuple[int, int]] = []
            for dim, tags in entities.items():
                flat.extend((int(dim), int(t)) for t in tags)
            item = QtWidgets.QTreeWidgetItem(self._group_parts)
            item.setText(0, name)
            item.setData(0, _ROLE_KIND, "part")
            item.setData(0, _ROLE_PAYLOAD, name)
            self._set_eye(item, self._group_is_visible(flat))
            for dim, tag in flat:
                child = QtWidgets.QTreeWidgetItem(item)
                child.setText(0, f"{dim_labels.get(dim, '?')} {tag}")
                child.setData(0, _ROLE_KIND, "entity")
                child.setData(0, _ROLE_PAYLOAD, (dim, tag))
                self._set_eye(
                    child, not self._vis_mgr.is_hidden((dim, tag)),
                )

    # ------------------------------------------------------------------
    # Eye toggle
    # ------------------------------------------------------------------

    def _group_is_visible(self, dts: list[tuple[int, int]]) -> bool:
        if not dts:
            return True
        return any(not self._vis_mgr.is_hidden(dt) for dt in dts)

    def _on_eye_clicked(self, item: Any) -> None:
        if item is None:
            return
        kind = item.data(0, _ROLE_KIND)
        current_hidden = set(self._vis_mgr.hidden)
        if kind in ("group", "label", "part"):
            members = self._collect_item_dts(item)
            if not members:
                return
            visible_now = self._group_is_visible(members)
            if visible_now:
                current_hidden.update(members)
            else:
                current_hidden.difference_update(members)
            self._vis_mgr.set_hidden(current_hidden)
        elif kind == "entity":
            dt = item.data(0, _ROLE_PAYLOAD)
            if dt in current_hidden:
                current_hidden.discard(dt)
            else:
                current_hidden.add(dt)
            self._vis_mgr.set_hidden(current_hidden)

    def _collect_item_dts(self, item: Any) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        for i in range(item.childCount()):
            child = item.child(i)
            if child.data(0, _ROLE_KIND) == "entity":
                dt = child.data(0, _ROLE_PAYLOAD)
                if dt is not None:
                    out.append(tuple(dt))
        return out

    def _refresh_eye_states(self) -> None:
        """Repaint eyes after a programmatic visibility change."""
        for i in range(self._tree.topLevelItemCount()):
            top = self._tree.topLevelItem(i)
            for j in range(top.childCount()):
                row = top.child(j)
                kind = row.data(0, _ROLE_KIND)
                if kind in ("group", "label", "part"):
                    members = self._collect_item_dts(row)
                    self._set_eye(row, self._group_is_visible(members))
                    for k in range(row.childCount()):
                        leaf = row.child(k)
                        if leaf.data(0, _ROLE_KIND) == "entity":
                            dt = leaf.data(0, _ROLE_PAYLOAD)
                            self._set_eye(
                                leaf,
                                not self._vis_mgr.is_hidden(tuple(dt)),
                            )
        try:
            self._tree.viewport().update()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Row selection
    # ------------------------------------------------------------------

    def _on_item_clicked(self, item: Any, column: int) -> None:
        if item is None:
            return
        # ParaView parity: a click in the eye column toggles
        # visibility *only* — pqPipelineBrowserWidget::handleIndexClicked
        # responds solely to column == 1. Any eye-bearing row carries
        # ROLE_VISIBLE; route it there and stop (no activation).
        if column == _EYE_COL and item.data(0, ROLE_VISIBLE) is not None:
            self._on_eye_clicked(item)
            return
        kind = item.data(0, _ROLE_KIND)
        payload = item.data(0, _ROLE_PAYLOAD)
        if kind == "group" and self._on_group_activated is not None:
            self._on_group_activated(payload)
        elif kind == "entity" and self._on_entity_toggled is not None:
            self._on_entity_toggled(payload)
        # Generic row-focus signal — fires for every kind so the
        # viewer can raise the matching property tab. Header rows
        # (``kind == "header"``) are non-selectable and never reach
        # here, so no guard needed.
        if (kind in ("group", "label", "entity", "part")
                and self._on_row_focused):
            self._on_row_focused(kind, payload)

    # ------------------------------------------------------------------
    # Right-click context menu
    # ------------------------------------------------------------------

    def _on_context_menu(self, pos: Any) -> None:
        QtCore, _, QtWidgets = _qt()
        item = self._tree.itemAt(pos)
        if item is None:
            return
        kind = item.data(0, _ROLE_KIND)
        if kind not in ("group", "label", "entity", "part"):
            return

        # Resolve target DimTags for visibility actions.
        if kind == "entity":
            dts = [tuple(item.data(0, _ROLE_PAYLOAD))]
        else:
            dts = self._collect_item_dts(item)

        menu = QtWidgets.QMenu(self._widget)
        act_rename = act_delete = None
        if kind in ("group", "label"):
            act_rename = menu.addAction("Rename")
            act_delete = menu.addAction("Delete")
            menu.addSeparator()

        n = len(dts)
        act_hide = menu.addAction(f"Hide ({n})") if dts else None
        act_isolate = menu.addAction(f"Isolate ({n})") if dts else None
        act_reveal = menu.addAction("Reveal all")

        chosen = menu.exec_(self._tree.viewport().mapToGlobal(pos))
        if chosen is None:
            return
        name = (
            item.data(0, _ROLE_PAYLOAD)
            if kind in ("group", "label") else None
        )
        if chosen == act_rename and name:
            cb = (
                self._on_rename_group if kind == "group"
                else self._on_rename_label
            )
            if cb is not None:
                cb(name)
        elif chosen == act_delete and name:
            cb = (
                self._on_delete_group if kind == "group"
                else self._on_delete_label
            )
            if cb is not None:
                cb(name)
        elif chosen == act_hide and dts:
            current = set(self._vis_mgr.hidden)
            current.update(dts)
            self._vis_mgr.set_hidden(current)
        elif chosen == act_isolate and dts:
            self._vis_mgr.isolate_dts(dts)
        elif chosen == act_reveal:
            self._vis_mgr.reveal_all()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_header_item(self, label: str) -> Any:
        QtCore, _, QtWidgets = _qt()
        item = QtWidgets.QTreeWidgetItem([label])
        item.setData(0, _ROLE_KIND, "header")
        flags = item.flags() & ~QtCore.Qt.ItemIsSelectable
        item.setFlags(flags)
        font = item.font(0)
        font.setBold(True)
        item.setFont(0, font)
        return item

    @staticmethod
    def _merge_by_name(
        raw: list[tuple[str, int, int, list[tuple[int, int]]]],
    ) -> list[tuple[str, int, int, list[tuple[int, int]]]]:
        """Collapse same-name entries into one multi-dimensional row.

        This is the **Label** grouping mechanism. A label is a
        multi-dimensional grouping: it is backed by one ``_label:``
        Gmsh PG *per dimension* (PGs are dimension-scoped), all under
        the same name. Merging by name presents the label as a single
        row whose members span every dimension it covers. Members are
        unioned; the first dim/pg_tag is kept for the tuple shape;
        rows are ordered by lowest pg_tag.

        Physical groups do **not** use this — a PG is dimension-scoped
        and listed per-PG (mixed-dim PG creation is rejected at the
        source, see ``model_viewer._on_new_group``).
        """
        merged: dict[str, list] = {}    # name -> [dim, tag, members, seen, min_tag]
        for name, dim, tag, members in raw:
            slot = merged.get(name)
            if slot is None:
                merged[name] = [dim, tag, list(members), set(members), tag]
            else:
                slot[4] = min(slot[4], tag)
                for dt in members:
                    if dt not in slot[3]:
                        slot[3].add(dt)
                        slot[2].append(dt)
        out = sorted(
            ((n, s[0], s[1], s[2], s[4]) for n, s in merged.items()),
            key=lambda x: x[4],
        )
        return [(n, d, t, m) for n, d, t, m, _ in out]

    def _collect_groups(self) -> list[tuple[str, int, int, list[tuple[int, int]]]]:
        """User-facing physical groups, read from staging.

        ADR 0045 S3c: staging is the single source of truth — every group
        (pre-existing, seeded at viewer init + newly staged) lives in
        ``SelectionState.staged_groups``; gmsh is written only at flush.
        Returns ``[(name, dim, order_index, members), ...]`` in creation
        order. Members are SelectionTargets → DimTags via ``.dimtag``.
        Labels (``_label:`` PGs) are a separate concept — see
        :meth:`_collect_labels`, which still reads them from gmsh.
        """
        staged = self._selection.staged_groups
        order = self._selection.group_order
        names = [n for n in order if n in staged]
        for n in staged:
            if n not in names:
                names.append(n)
        out: list[tuple[str, int, int, list[tuple[int, int]]]] = []
        for idx, name in enumerate(names):
            members = [t.dimtag for t in staged[name]]
            dim = members[0][0] if members else 0
            out.append((name, dim, idx, members))
        return out

    @staticmethod
    def _collect_labels() -> list[tuple[str, int, int, list[tuple[int, int]]]]:
        """Internal Tier-1 labels — the ``_label:*`` PGs, prefix stripped.

        A label is a **multi-dimensional grouping**: its backing
        ``_label:`` PGs (one per dimension, since PGs are
        dimension-scoped) are merged by name into a single row whose
        members span every dimension the label covers. Same
        ``[(name, dim, pg_tag, members), ...]`` shape as
        :meth:`_collect_groups`; the inverse filter (keep the label
        PGs, drop user-facing ones)."""
        from apeGmsh.core.Labels import is_label_pg, strip_prefix
        raw = []
        for pg_dim, pg_tag in gmsh.model.getPhysicalGroups():
            try:
                name = gmsh.model.getPhysicalName(pg_dim, pg_tag)
            except Exception:
                continue
            if not is_label_pg(name):
                continue
            ents = gmsh.model.getEntitiesForPhysicalGroup(pg_dim, pg_tag)
            members = [(pg_dim, int(t)) for t in ents]
            raw.append((strip_prefix(name), pg_dim, pg_tag, members))
        return ModelOutlineTree._merge_by_name(raw)


__all__ = ["ModelOutlineTree"]
