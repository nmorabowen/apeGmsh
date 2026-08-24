"""``.apegmsh/progress.json`` — the live solve's step counter (ADR 0095
Amendment 11).

The analyze loop already emits one ``APEGMSH_PROGRESS i=.. n=.. t=..``
marker per increment, and :mod:`apeGmsh.opensees._run` already parses it
to draw a console counter. Until now that was the *only* consumer: an
out-of-process reader wanting "how far along is this solve" had to tail
and re-grep the solver log, i.e. reimplement a private regex against a
file whose format is the solver's, not ours. This module promotes the
sample it already has to a published habitat file.

Three properties make it safe to write from inside a running solve:

* **Habitat-only.** The root is the ordinary INV-15 resolution, and a
  resolved root only counts when it already holds a ``.apegmsh/``
  directory. A plain ``python model.py`` outside a habitat writes
  nothing and grows no dot-dir — ``resolve_root`` falls back to cwd by
  design, so without that check every script run anywhere would deposit
  one.
* **Atomic replace** (INV-16), same single-writer discipline as
  ``names.json``: a reader polling mid-solve sees the previous complete
  sample or the next one, never half of either.
* **Silent on failure.** Every public method swallows its own errors.
  A read-only directory, a full disk, or an antivirus lock is a reason
  to lose a progress sample — never a reason to kill a solve that may
  have been running for hours.

Throttling is inherited from the marker cadence (the emitters aim for
~20 samples per analyze) — no timer of our own, so the file's write rate
is whatever the deck already prints.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ._contract import CONTRACT_VERSION
from ._paths import DEFAULT_PROGRESS_REL, atomic_write_text, display_path, resolve_root

PROGRESS_SCHEMA = 1

__all__ = ["PROGRESS_SCHEMA", "ProgressSidecar", "habitat_root"]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def habitat_root(
    root: Path | str | None = None,
    *,
    start: Path | str | None = None,
) -> Path | None:
    """The INV-15 root when a habitat really lives there; ``None`` otherwise.

    :func:`~apeGmsh.studio._paths.resolve_root` always answers — cwd is
    its last fallback — which is right for a studio verb the user
    invoked on purpose and wrong for a sidecar that writes itself. So
    the answer only counts here when ``.apegmsh/`` already exists: the
    sidecar joins a habitat, it never founds one.
    """
    try:
        base = resolve_root(root, start=start)
    except OSError:
        return None
    return base if (base / ".apegmsh").is_dir() else None


class ProgressSidecar:
    """Mirror an in-flight solve into ``.apegmsh/progress.json``.

    Construct once per run, call :meth:`sample` per parsed marker, and
    :meth:`finish` exactly once when the child exits.

    Nothing is written until the first sample. A run that emits no
    marker at all (a deck with no ``analyze``) leaves the file alone
    rather than publishing a fabricated ``0/0`` — "no progress to
    report" and "an analysis that finished at step zero" are different
    claims, and a consumer polling for a progress bar must not be told
    the second when the first is true.
    """

    def __init__(
        self,
        *,
        deck: Path | str | None = None,
        log: Path | str | None = None,
        root: Path | str | None = None,
        start: Path | str | None = None,
    ) -> None:
        self._root = habitat_root(root, start=start)
        self._deck = self._show(deck)
        self._log = self._show(log)
        self._last: dict[str, Any] | None = None

    # -- introspection -------------------------------------------------

    @property
    def enabled(self) -> bool:
        """``True`` when a habitat resolved and writes will be attempted."""
        return self._root is not None

    @property
    def path(self) -> Path | None:
        """Destination file, or ``None`` outside a habitat."""
        return None if self._root is None else self._root / DEFAULT_PROGRESS_REL

    # -- writing -------------------------------------------------------

    def sample(self, *, i: int, n: int, t: str, warnings: int = 0) -> None:
        """Publish one in-flight sample (``done: false``)."""
        self._last = {
            "i": int(i),
            "n": int(n),
            "t": str(t),
            "warnings": int(warnings),
        }
        self._write(done=False, ok=None)

    def finish(self, *, ok: bool, warnings: int | None = None) -> None:
        """Publish the terminal record (``done: true`` plus ``ok``).

        A no-op when no sample was ever seen — see the class docstring.
        *warnings* refreshes the cumulative tally with the run's final
        count (lines can still arrive after the last marker).
        """
        if self._last is None:
            return
        if warnings is not None:
            self._last["warnings"] = int(warnings)
        self._write(done=True, ok=bool(ok))

    # -- internals -----------------------------------------------------

    def _show(self, path: Path | str | None) -> str | None:
        """Root-relative posix under the habitat, absolute outside it (S5i)."""
        if path is None or self._root is None:
            return None if path is None else str(path)
        try:
            return display_path(path, self._root)
        except Exception:
            return str(path)

    def _write(self, *, done: bool, ok: bool | None) -> None:
        dest = self.path
        if dest is None or self._last is None:
            return
        payload: dict[str, Any] = {
            "schema": PROGRESS_SCHEMA,
            "contract_version": CONTRACT_VERSION,
            "deck": self._deck,
            "log": self._log,
            "i": self._last["i"],
            "n": self._last["n"],
            "t": self._last["t"],
            "ts": _utc_now(),
            "done": done,
            "warnings": self._last["warnings"],
        }
        if ok is not None:
            payload["ok"] = ok
        try:
            atomic_write_text(dest, json.dumps(payload, indent=2) + "\n")
        except Exception:
            # A lost sample is not worth a dead solve. Deliberately broad:
            # anything this writer can raise (OSError, encoding, a torn
            # temp file) is strictly less important than the run.
            return
