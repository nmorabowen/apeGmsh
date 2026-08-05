"""DefinitionsPanel — read-only list of bridge-side named primitives.

Surfaces the ``/opensees/names`` aliases a model was authored with
(``ops.<family>.<Type>(..., name="...")``) so the engineer sees their
own names — "rebar", "col_sec" — next to the OpenSees kind + tag,
instead of bare integers.

Fed exclusively from :attr:`ViewerData.names`, which arrives through
the ``H5Model.names()`` read channel (ADR 0026 — the viewers' single
legal import surface).  A live-FEM snapshot carries no bridge names,
so the panel renders an idle hint in that case.
"""
from __future__ import annotations

from typing import Any


def _qt():
    from qtpy import QtCore, QtWidgets
    return QtWidgets, QtCore


# Display order + human headers for the kinds the bridge can register.
_KIND_LABEL: dict[str, str] = {
    "uniaxialMaterial": "Uniaxial materials",
    "nDMaterial": "nD materials",
    "section": "Sections",
    "geomTransf": "Geometric transforms",
    "beamIntegration": "Beam integrations",
    "timeSeries": "Time series",
    "pattern": "Patterns",
}


class DefinitionsPanel:
    """Read-only tree of named primitives, grouped by OpenSees kind.

    Construct once at window build time; call :meth:`set_data` whenever
    the active :class:`ViewerData` changes (or once, after load).
    """

    def __init__(self) -> None:
        QtWidgets, _ = _qt()
        self.widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(self.widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # One muted hint line (ADR 0087 INV-2) — the API signature is
        # implementation detail, so it rides the tooltip, not the panel.
        self._hint = QtWidgets.QLabel("No named definitions.")
        self._hint.setObjectName("DefinitionsEmptyHint")
        self._hint.setWordWrap(True)
        self._hint.setToolTip(
            "Names come from ops.<family>.<Type>(..., name=…) and load "
            "from a model.h5 / results.h5."
        )
        layout.addWidget(self._hint)

        self._tree = QtWidgets.QTreeWidget()
        self._tree.setColumnCount(2)
        self._tree.setHeaderLabels(["Name", "Tag"])
        self._tree.setRootIsDecorated(True)
        self._tree.setAlternatingRowColors(True)
        layout.addWidget(self._tree)

        self._set_visible(has_rows=False)

    def set_data(self, viewer_data: Any) -> None:
        """Populate from ``viewer_data.names`` (``(name, kind, tag)`` rows).

        Tolerates ``None`` / a snapshot without ``names`` (older
        ViewerData) by rendering the idle hint.
        """
        names = tuple(getattr(viewer_data, "names", ()) or ())
        self._populate(names)

    # -- internals ------------------------------------------------------

    def _populate(self, names: "tuple[tuple[str, str, int], ...]") -> None:
        QtWidgets, _ = _qt()
        self._tree.clear()
        if not names:
            self._set_visible(has_rows=False)
            return

        # Group by kind, preserving the canonical display order; any
        # unrecognised kind sorts last under its raw token.
        by_kind: dict[str, list[tuple[str, int]]] = {}
        for nm, kind, tag in names:
            by_kind.setdefault(kind, []).append((nm, int(tag)))

        ordered = list(_KIND_LABEL) + [
            k for k in sorted(by_kind) if k not in _KIND_LABEL
        ]
        for kind in ordered:
            rows = by_kind.get(kind)
            if not rows:
                continue
            header = _KIND_LABEL.get(kind, kind)
            parent = QtWidgets.QTreeWidgetItem([f"{header} ({len(rows)})", ""])
            for nm, tag in sorted(rows):
                parent.addChild(QtWidgets.QTreeWidgetItem([nm, str(tag)]))
            self._tree.addTopLevelItem(parent)
            parent.setExpanded(True)

        self._set_visible(has_rows=True)

    def _set_visible(self, *, has_rows: bool) -> None:
        self._tree.setVisible(has_rows)
        self._hint.setVisible(not has_rows)


def make_definitions_dock(viewer_data: Any) -> "tuple[DefinitionsPanel, Any]":
    """Build a :class:`DefinitionsPanel` + its :class:`DockSpec`.

    Mirrors ``make_output_dock`` / ``make_clip_planes_dock``: the
    panel is populated from ``viewer_data`` up front, and the returned
    spec mounts it as a right-side extension dock, hidden by default
    and surfaced via the View menu (lowest layout risk — it joins the
    standard extension-dock persistence machinery).
    """
    from ._dock_registry import DockSpec

    panel = DefinitionsPanel()
    panel.set_data(viewer_data)
    spec = DockSpec(
        dock_id="dock_results_definitions",
        title="Definitions",
        factory=lambda parent: panel.widget,
        default_area="right",
        default_visible=False,
    )
    return panel, spec
