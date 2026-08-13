"""Script replay with last-good frame (ADR 0095 S2 / INV-4).

``run_until`` execs the current file with the session held open
(``end()`` / ``__exit__`` become no-ops) and stubs ``viewer()`` so the
script cannot nest a Qt window. A failed exec keeps the previous
successful result — the host must not blank the viewport.

v0 is one-shot in the studio process (the agent's own runs do not
attach to this kernel — INV-5). Refresh is re-running the CLI.
"""

from __future__ import annotations

import hashlib
import os
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class ReplayResult:
    """Outcome of one ``run_until`` call."""

    ok: bool
    phase: str
    geometry_hash: str | None
    error: str | None
    session: Any = None
    skipped: bool = False


ExecFn = Callable[[Path, str], ReplayResult]


def _source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _exec_hold_open(script: Path, phase: str) -> ReplayResult:
    """Exec *script* with sessions held open. Raises on failure."""
    from apeGmsh._session import _SessionBase
    from apeGmsh.core.Model import Model
    from apeGmsh.mesh.Mesh import Mesh

    script = Path(script).resolve()
    opened: list[Any] = []
    original_begin = _SessionBase.begin
    original_end = _SessionBase.end
    original_exit = _SessionBase.__exit__
    original_model_viewer = Model.viewer
    original_mesh_viewer = Mesh.viewer

    def begin(self):
        result = original_begin(self)
        opened.append(self)
        return result

    def end(self) -> None:
        # Hold the kernel for the host. Autosave still runs if configured
        # — the snapshot is useful; finalize is what we skip.
        if getattr(self, "_active", False):
            save_to = getattr(self, "_save_to", None)
            if save_to is not None:
                try:
                    resolve = getattr(self, "_resolve_save_target", None)
                    target = resolve(None) if resolve is not None else save_to
                    do_save = getattr(self, "_do_save", None)
                    if do_save is not None:
                        do_save(target)
                except Exception:
                    pass
        opened.append(self)

    def exit_(self, exc_type, exc_val, exc_tb) -> None:
        end(self)

    def _noop_viewer(self, **kwargs):
        return None

    _SessionBase.begin = begin  # type: ignore[method-assign]
    _SessionBase.end = end  # type: ignore[method-assign]
    _SessionBase.__exit__ = exit_  # type: ignore[method-assign]
    Model.viewer = _noop_viewer  # type: ignore[method-assign]
    Mesh.viewer = _noop_viewer  # type: ignore[method-assign]

    ns: dict[str, Any] = {
        "__name__": "__studio_replay__",
        "__file__": str(script),
        "__package__": None,
    }
    source = script.read_text(encoding="utf-8")
    code = compile(source, str(script), "exec")
    old_cwd = os.getcwd()
    try:
        os.chdir(script.parent)
        exec(code, ns, ns)  # noqa: S102 — replay of the caller's file
    finally:
        os.chdir(old_cwd)
        _SessionBase.begin = original_begin  # type: ignore[method-assign]
        _SessionBase.end = original_end  # type: ignore[method-assign]
        _SessionBase.__exit__ = original_exit  # type: ignore[method-assign]
        Model.viewer = original_model_viewer  # type: ignore[method-assign]
        Mesh.viewer = original_mesh_viewer  # type: ignore[method-assign]

    mains = [s for s in opened if type(s).__name__ == "apeGmsh"]
    session = mains[-1] if mains else (opened[-1] if opened else None)
    return ReplayResult(
        ok=True,
        phase=phase,
        geometry_hash=_source_hash(script),
        error=None,
        session=session,
    )


class ReplayRunner:
    """Replay a script; keep the last successful result on failure (INV-4)."""

    def __init__(self, *, exec_fn: ExecFn | None = None) -> None:
        self._exec = exec_fn if exec_fn is not None else _exec_hold_open
        self._last_good: ReplayResult | None = None

    @property
    def last_good(self) -> ReplayResult | None:
        return self._last_good

    def run_until(self, script: Path, *, phase: str = "model") -> ReplayResult:
        script = Path(script)
        digest = _source_hash(script) if script.is_file() else None
        if (
            self._last_good is not None
            and self._last_good.ok
            and digest is not None
            and digest == self._last_good.geometry_hash
        ):
            return ReplayResult(
                ok=True,
                phase=self._last_good.phase,
                geometry_hash=digest,
                error=None,
                session=self._last_good.session,
                skipped=True,
            )
        try:
            result = self._exec(script, phase)
        except Exception:
            err = traceback.format_exc()
            if self._last_good is not None:
                return ReplayResult(
                    ok=False,
                    phase=self._last_good.phase,
                    geometry_hash=self._last_good.geometry_hash,
                    error=err,
                    session=self._last_good.session,
                )
            return ReplayResult(
                ok=False,
                phase=phase,
                geometry_hash=digest,
                error=err,
                session=None,
            )
        if result.ok:
            stamped = ReplayResult(
                ok=True,
                phase=result.phase,
                geometry_hash=digest,
                error=None,
                session=result.session,
            )
            self._last_good = stamped
            return stamped
        return result
