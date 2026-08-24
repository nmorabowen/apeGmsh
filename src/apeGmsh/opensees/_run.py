"""Stream an OpenSees / openseespy subprocess to a log file + console.

Used by :meth:`apeSees.tcl` / :meth:`apeSees.py` when ``run=True``.
Replaces the old ``subprocess.run(capture_output=True)`` calls, which
swallowed all output on success and dumped the whole buffer into an
exception on failure.

Two guarantees:

* **The full raw output is always tee'd to ``log_path``** — verbatim,
  including the ``APEGMSH_PROGRESS`` markers the analyze loop emits, so
  the on-disk log is exactly what the solver printed and can be
  ``tail -f``'d live.
* **Console output is opt-in via ``verbose``.** ``verbose=False`` (the
  default) prints three lines only — begin, what op, end. ``verbose=True``
  additionally renders a live step counter parsed from the progress
  markers and echoes warning / convergence lines as they stream.

Inside a studio habitat the same parsed samples are also mirrored to
``.apegmsh/progress.json`` (ADR 0095 Amendment 11) so an out-of-process
consumer can draw a progress bar without tailing this log and
reimplementing :data:`_PROGRESS_RE`. That writer lives in
``studio._progress`` — the habitat contract belongs to ``studio``, and
this module reaches it through a **deferred** import inside
:func:`stream_run`, the same idiom ``studio`` already uses in the
opposite direction (``studio._replay`` imports ``apeSees`` in a function
body). So the module-level import set here stays stdlib-only and
``import apeGmsh.opensees`` gains no eager edge onto the habitat.
Outside a habitat, and on any failure of the writer itself, the solve
is unaffected.

On a non-zero exit the raised :class:`RuntimeError` carries only the
last few lines plus the log path — never the whole buffer (it is on
disk already).

ASCII-only console markers (``>>`` / ``[OK]`` / ``[FAIL]`` / ``[warn]``);
Windows cp1252 terminals choke on unicode.
"""
from __future__ import annotations

import os
import re
import subprocess
import time
from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # type-only — no runtime edge onto the habitat
    from ..studio._progress import ProgressSidecar

__all__ = ["stream_run", "resolve_log_path", "run_label"]

#: One machine-readable marker per emitted analyze increment sample
#: (``puts "APEGMSH_PROGRESS i=.. n=.. t=.."`` — see the Tcl / Py
#: emitters' ``analyze``). Parsed for the ``verbose`` live counter.
_PROGRESS_RE = re.compile(r"APEGMSH_PROGRESS i=(\d+) n=(\d+) t=(\S+)")

#: Lines worth surfacing live in ``verbose`` mode and tallying always.
_WARN_RE = re.compile(r"WARNING|FATAL|FAILED|failed to converge")

#: How many trailing raw lines the failure exception / tail carries.
_TAIL = 15


def _open_progress(
    deck_path: str | None, log_path: str
) -> "ProgressSidecar | None":
    """The habitat progress sidecar, or ``None``.

    Deferred import (see the module docstring): ``opensees`` must not
    grow an eager edge onto ``studio``, and a solve must survive a
    habitat that cannot be imported or resolved at all.
    """
    try:
        from ..studio._progress import ProgressSidecar

        sidecar = ProgressSidecar(deck=deck_path, log=log_path)
    except Exception:
        return None
    return sidecar if sidecar.enabled else None


def resolve_log_path(log: str | None, deck_path: str) -> str:
    """Resolve the log path. ``str`` overrides; else ``<deck>.log``.

    The log is always written (not optional) — a value of ``None``
    derives the default next to the deck (``out/model.tcl`` ->
    ``out/model.log``).
    """
    if isinstance(log, str):
        return log
    return os.path.splitext(deck_path)[0] + ".log"


def run_label(deck_path: str, steps: int | None, dt: float | None) -> str:
    """A one-line description of the op being run, for the begin banner."""
    if steps is None:
        return f"run  ->  {deck_path}"
    if dt is None:
        return f"analyze {int(steps)}  ->  {deck_path}"
    return f"analyze {int(steps)} x dt={dt}  ->  {deck_path}"


def stream_run(
    argv: list[str],
    *,
    log_path: str,
    verbose: bool,
    label: str,
    header: str = "OpenSees",
    env: dict[str, str] | None = None,
    deck_path: str | None = None,
) -> None:
    """Run ``argv``, tee stdout+stderr to ``log_path``, report to console.

    Raises :class:`RuntimeError` on a non-zero exit, with the last
    ``_TAIL`` lines and the log path.

    *deck_path* is recorded in the habitat progress sidecar so a
    consumer knows which deck the counter belongs to; omitting it costs
    that field only.
    """
    started = time.perf_counter()
    n_lines = 0
    n_warn = 0
    warn_at: list[int] = []
    tail: deque[str] = deque(maxlen=_TAIL)
    progress = _open_progress(deck_path, log_path)

    print(f">> {header}  |  {label}")
    print(f"   log: {log_path}")
    if not verbose:
        print("   running ...")

    # errors='replace' so a stray cp1252 byte never crashes the tee;
    # bufsize=1 line-buffers the parent's read side.
    proc = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
    )
    assert proc.stdout is not None
    with open(log_path, "w", encoding="utf-8", errors="replace") as logf:
        for line in proc.stdout:
            logf.write(line)
            logf.flush()
            n_lines += 1
            raw = line.rstrip("\n")

            m = _PROGRESS_RE.search(raw)
            if m is not None:
                i, n = int(m.group(1)), int(m.group(2))
                if verbose:
                    pct = (100 * i) // n if n else 0
                    el = time.perf_counter() - started
                    print(
                        f"   step {i:>7}/{n}  {pct:>3d}%   "
                        f"t={m.group(3)}   elapsed {el:5.1f}s"
                    )
                if progress is not None:
                    progress.sample(i=i, n=n, t=m.group(3), warnings=n_warn)
                continue

            if _WARN_RE.search(raw):
                n_warn += 1
                warn_at.append(n_lines)
                if verbose:
                    print(f"   [warn] {raw.strip()}  (line {n_lines})")
            tail.append(raw)
        proc.wait()

    elapsed = time.perf_counter() - started
    rc = proc.returncode
    if progress is not None:
        # Before the raise below — a failed solve must still close its
        # sidecar, or a consumer polls a ``done: false`` file forever.
        progress.finish(ok=rc == 0, warnings=n_warn)
    if rc == 0:
        print(
            f"[OK] finished in {elapsed:.1f}s  (exit 0)  |  "
            f"{n_lines:,} log lines, {n_warn} warnings"
        )
        if verbose and n_warn:
            head = ", ".join(str(x) for x in warn_at[:8])
            more = " ..." if len(warn_at) > 8 else ""
            print(f"     {n_warn} warnings at log lines {head}{more}")
        return

    print(f"[FAIL] exited {rc} after {elapsed:.1f}s  |  see {log_path}")
    tail_lines = list(tail)
    if verbose and tail_lines:
        print(f"     ---- last {len(tail_lines)} log lines ----")
        for ln in tail_lines:
            print(f"     {ln}")
    tail_txt = "\n".join(tail_lines)
    raise RuntimeError(
        f"{header} subprocess returned {rc}. Full log: {log_path}\n"
        f"---- last {len(tail_lines)} lines ----\n{tail_txt}"
    )
