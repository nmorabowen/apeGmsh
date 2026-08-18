"""The ``restore_session="prompt"`` dialog (ADR 0098 S6a).

The Qt half of :func:`apeGmsh.results.session._boot.boot_session`. It
is a separate module because the policy is Qt-free and this is not: a
headless caller passes ``confirm=None`` and never imports this at all.

Two things make the new dialog worth keeping rather than dropping to a
silent restore. It names what the file actually contains — panes, not
"N diagrams" — and it is the one place a **degraded** restore can be
seen BEFORE the picture is: a snapshot whose stage was renamed restores
with notices (plan decision 15), and a human staring at a window with
one pane fewer than they remember deserves the sentence explaining it,
not a console line they scrolled past.

The No button states its consequence. Declining leaves auto-save armed
— the caller asked for it and the human chose the fresh start — so the
saved session is replaced when the window closes. That is a footgun
only if it is unsaid.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

_MAX_LISTED = 5


def describe(restored: Any) -> str:
    """One line naming what is in the snapshot: panes, then extras."""
    from apeGmsh.results.session import MeshView

    panes = list(getattr(restored.session, "panes", ()) or ())
    meshes = [p for p in panes if isinstance(p, MeshView)]
    plots = [p for p in panes if not isinstance(p, MeshView)]

    def _plural(n: int, one: str) -> str:
        return f"{n} {one}" if n == 1 else f"{n} {one}s"

    parts = []
    if meshes:
        parts.append(_plural(len(meshes), "mesh view"))
    if plots:
        parts.append(_plural(len(plots), "plot"))
    if not parts:
        return "no panes"

    filled = sorted({
        category
        for view in meshes
        for category in view.slots
    })
    if filled:
        shown = ", ".join(filled[:_MAX_LISTED])
        more = "" if len(filled) <= _MAX_LISTED else ", …"
        parts.append(f"slots: {shown}{more}")
    return " · ".join(parts)


def confirm_restore(path: "str | Path", restored: Any) -> bool:
    """Ask whether to restore ``path``. ``False`` when nobody can be asked.

    Headless is a decline, not a silent yes: ``"prompt"`` means the
    human decides, and a build with no dialog to show has not asked
    anyone.
    """
    try:
        from qtpy.QtWidgets import QMessageBox
    except Exception:
        return False

    name = Path(path).name
    notices = tuple(getattr(restored, "notices", ()) or ())
    body = (
        f"A saved session is available:\n"
        f"  {name}\n"
        f"  {describe(restored)}\n"
    )
    if notices:
        listed = "\n".join(f"  • {n}" for n in notices[:_MAX_LISTED])
        more = (
            ""
            if len(notices) <= _MAX_LISTED
            else f"\n  • (+{len(notices) - _MAX_LISTED} more)"
        )
        body += (
            f"\nIt does not fit these results exactly:\n{listed}{more}\n"
        )
    body += (
        "\nRestore it?\n\n"
        "No opens the default picture — and replaces the saved session "
        "when this window closes."
    )

    reply = QMessageBox.question(
        None,
        "Restore previous session?",
        body,
        QMessageBox.Yes | QMessageBox.No,
        QMessageBox.Yes,
    )
    return reply == QMessageBox.Yes


__all__ = ["confirm_restore", "describe"]
