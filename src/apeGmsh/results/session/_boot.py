"""How a session boots for a window — the S6a open policy (ADR 0098 §11).

§11 puts "the restore/save snapshot semantics" in the S6a flip, and §1
says what they are: ``restore_session`` / ``save_session`` load and save
a **session snapshot**, and an old ``schema_version`` "skips with a
one-line notice (no prompt) and renames the file aside (``.legacy``) so
save-on-close cannot destroy the optional importer's input".

This module is that policy, and it is deliberately Qt-free: the same
three lines decide what happens whether the caller is
:meth:`Results.viewer`, the ``python -m apeGmsh.viewers`` child, or a
test with no display. The window is handed a session and a save target;
it does not read or write disk itself, so a bare
``results.session().show()`` stays exactly as pure as it was at S2.

**INV-SESSION-OPEN — the window always opens.** A session file that
cannot be honoured never blocks the window and is never destroyed by
it. Three things happen together, and the third is the one that is easy
to forget:

1. the failure is said out loud, naming the file;
2. the default picture boots, so the human sees their results;
3. **auto-save is disarmed for that window**, so the file that could
   not be read is still there when they go looking.

Without (3) the guarantee is a lie: the save target IS the file we
declined to read (plan decision 11 adopted ``<results>.viewer-session
.json`` at this flip), so booting fresh with auto-save armed would
overwrite on close exactly what the refusal was protecting. That is the
whole reason :func:`~._snapshot.rename_legacy_aside` refuses to
overwrite an existing ``.legacy`` — and the reason that refusal has to
land here as a notice rather than as a traceback out of ``viewer()``.

The upgrade → downgrade → upgrade sequence is what makes (3) real
rather than theoretical. First flipped open: a v13 file is renamed to
``.legacy`` and a fresh session boots. Downgrade: the old window
overwrites ``<results>.viewer-session.json`` with v13 again. Second
flipped open: the aside is taken, ``rename_legacy_aside`` refuses, and
a human who only wanted their window back gets it — with both files
intact and a notice explaining which one to move.

Degradation is NOT failure. A snapshot that restores with notices (a
stage that was renamed, a step past the end of a shortened run — plan
decision 15) is an honoured file: the session restores, the notices are
surfaced, and auto-save stays armed, because the window is showing what
the file said as faithfully as the results allow.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

from ._snapshot import (
    SnapshotError,
    default_snapshot_path,
    load_snapshot,
    save_snapshot,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from apeGmsh.results.Results import Results

    from ._session import ResultsSession

#: ``confirm(path, restored) -> bool`` — the ``restore_session="prompt"``
#: hook. Qt supplies one; ``None`` means nobody can be asked.
ConfirmRestore = Callable[[Path, Any], bool]


@dataclass(frozen=True)
class SessionBoot:
    """What a window should open with, and where it may save.

    Attributes
    ----------
    session
        The session to project. Restored from disk, or the default
        picture (one empty mesh view) when there was nothing to restore
        or the file could not be honoured.
    save_path
        Where to write on close, or ``None`` for **auto-save disarmed**
        — the caller must not invent a path of its own. ``None`` means
        one of: ``save=False``, an in-memory Results with no file to sit
        beside, or a refusal that this window must not overwrite.
    notices
        Everything the human is owed about the open, in order. Both
        families (plan decision 15) land here — a degraded restore and a
        refused one — because the caller decides how loud to be.
    restored
        Whether :attr:`session` came off disk. ``False`` for the default
        picture, including when a restore was offered and declined.
    source
        The file :attr:`session` was restored from, when it was.
    """

    session: "ResultsSession"
    save_path: Optional[Path]
    notices: tuple[str, ...] = ()
    restored: bool = False
    source: Optional[Path] = None

    @property
    def autosaves(self) -> bool:
        """Whether :func:`save_on_close` will write anything."""
        return self.save_path is not None


def viewer_snapshot_path(results: "Results") -> Optional[Path]:
    """``<results>.viewer-session.json``, or ``None`` for in-memory.

    The name the old window used, adopted at this flip (plan decision
    11) now that nothing else writes it.
    """
    path = getattr(results, "_path", None)
    if path is None:
        return None
    return default_snapshot_path(Path(path).resolve())


def boot_session(
    results: "Results",
    *,
    restore: "bool | str" = "prompt",
    save: bool = True,
    confirm: Optional[ConfirmRestore] = None,
) -> SessionBoot:
    """Decide what the window opens with (INV-SESSION-OPEN).

    Parameters
    ----------
    results
        The broker the session presents. Its ``_path`` is what makes a
        snapshot possible at all — an in-memory Results has no file to
        sit beside, so it neither restores nor saves.
    restore
        ``True`` restores silently, ``False`` ignores any saved session,
        ``"prompt"`` asks through ``confirm`` first.
    save
        Arm auto-save on close. Only ever a ceiling: a refusal below
        disarms it regardless.
    confirm
        Asked once, with the path and the :class:`RestoredSession`, when
        ``restore="prompt"`` finds a readable snapshot. ``None`` (no Qt,
        no human) declines — "ask first" cannot be honoured by guessing.

    Never raises for a bad file. Reading a snapshot is the one step here
    that touches somebody else's data, and every way it can go wrong is
    a notice, because the alternative is a human who cannot open their
    own results until they delete a JSON file.
    """
    if restore is not True and restore is not False and restore != "prompt":
        raise ValueError(
            f"restore_session must be True, False or 'prompt'; got "
            f"{restore!r}. An unrecognised token would silently mean "
            f"'restore', which is the one outcome nobody asked for."
        )

    path = viewer_snapshot_path(results)
    armed = bool(save) and path is not None
    notices: list[str] = []

    if path is None or restore is False or not path.exists():
        return SessionBoot(
            session=results.session(),
            save_path=path if armed else None,
            notices=tuple(notices),
        )

    restored: Optional[Any] = None
    try:
        restored = load_snapshot(path, results)
    except FileExistsError as exc:
        # rename_legacy_aside refused: an older aside is already there.
        # Both files stay; this window stops being able to overwrite
        # either of them.
        notices.append(str(exc))
        armed = False
    except SnapshotError as exc:
        # A schema / ontology violation. restore_snapshot is RIGHT to
        # refuse loudly (plan decision 15) — the loudness lands here as
        # a notice, and the file survives to be looked at.
        notices.append(
            f"{path.name} is not a session this build can read, so a "
            f"fresh session was opened and auto-save is off for this "
            f"window (nothing was overwritten): {exc}"
        )
        armed = False
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        notices.append(
            f"{path.name} could not be read, so a fresh session was "
            f"opened and auto-save is off for this window (nothing was "
            f"overwritten): {exc}"
        )
        armed = False

    if restored is None:
        # Either a failure above, or the legacy gate fired and already
        # printed its own line while renaming the v13 file aside. In the
        # gate's case the path is now FREE, so auto-save stays armed —
        # there is nothing left there to destroy.
        return SessionBoot(
            session=results.session(),
            save_path=path if armed else None,
            notices=tuple(notices),
        )

    notices.extend(restored.notices)

    if restore == "prompt" and not (
        confirm is not None and confirm(path, restored)
    ):
        # Declined (or nobody to ask). The human chose the fresh start
        # with the consequence stated in the dialog, so auto-save is
        # left exactly as the caller asked for it.
        return SessionBoot(
            session=results.session(),
            save_path=path if armed else None,
            notices=tuple(notices),
        )

    return SessionBoot(
        session=restored.session,
        save_path=path if armed else None,
        notices=tuple(notices),
        restored=True,
        source=path,
    )


def save_on_close(boot: SessionBoot) -> Optional[Path]:
    """Write the session if this window is allowed to. Never raises.

    Called after the window closes — the widgets are disposed by then,
    but the session IR is untouched, so this writes what was on screen.
    A write that fails is a printed line, not an exception: the window
    is already gone, and there is nothing left for a traceback to
    protect.
    """
    if boot.save_path is None:
        return None
    try:
        return save_snapshot(boot.session, boot.save_path)
    except Exception as exc:  # noqa: BLE001 - the window is already closed
        print(
            f"[session] could not save {boot.save_path.name}: "
            f"{type(exc).__name__}: {exc}"
        )
        return None


def announce(boot: SessionBoot) -> None:
    """Print the open's notices, one ``[session]`` line each."""
    for notice in boot.notices:
        print(f"[session] {notice}")


__all__ = [
    "ConfirmRestore",
    "SessionBoot",
    "announce",
    "boot_session",
    "save_on_close",
    "viewer_snapshot_path",
]
