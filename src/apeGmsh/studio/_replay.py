"""Script replay with last-good frame (ADR 0095 S2 / INV-4).

``run_until`` execs the current file with the session held open
(``end()`` / ``__exit__`` become no-ops) and stubs ``viewer()`` so the
script cannot nest a Qt window. A failed exec keeps the previous
successful result — the host must not blank the viewport.

``phase`` is a real gate: ``model`` stops before ``generate()``,
``mesh`` stops before ``apeSees`` / ``Results``, ``results`` runs to
completion. The stop is ``StopAtPhase`` (a ``BaseException`` so a
script ``except Exception`` cannot swallow it). At every successful
stop the runner writes ``.apegmsh/names.json`` and appends one
line to ``.apegmsh/runs.jsonl``.

v0 is one-shot in the studio process (the agent's own runs do not
attach to this kernel — INV-5). Refresh is re-running the CLI.
"""

from __future__ import annotations

import hashlib
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

PHASES = ("model", "mesh", "results")
TRIGGERS = ("open", "refresh", "watch")


class StopAtPhase(BaseException):
    """Replay reached the requested phase gate. Success, not an error."""

    def __init__(self, phase: str, gate: str) -> None:
        super().__init__(phase, gate)
        self.phase = phase
        self.gate = gate


@dataclass(frozen=True)
class ReplayResult:
    """Outcome of one ``run_until`` call."""

    ok: bool
    phase: str
    geometry_hash: str | None
    error: str | None
    session: Any = None
    skipped: bool = False
    stopped_at: str | None = None


ExecFn = Callable[[Path, str], ReplayResult]


def _source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stopper(phase: str, gate: str):
    def _raise(*args, **kwargs):
        raise StopAtPhase(phase, gate)

    return _raise


def _cls_stopper(phase: str, gate: str):
    def _raise(cls, *args, **kwargs):
        raise StopAtPhase(phase, gate)

    return classmethod(_raise)


def _install(patches: list[tuple[Any, str, Any]], obj: Any, name: str, replacement: Any) -> None:
    patches.append((obj, name, getattr(obj, name)))
    setattr(obj, name, replacement)


def _restore(patches: list[tuple[Any, str, Any]]) -> None:
    for obj, name, original in reversed(patches):
        setattr(obj, name, original)


def _install_phase_gates(patches: list[tuple[Any, str, Any]], phase: str) -> None:
    if phase == "model":
        from apeGmsh.mesh._mesh_generation import _Generation

        _install(
            patches,
            _Generation,
            "generate",
            _stopper(phase, "g.mesh.generation.generate"),
        )
    if phase in ("model", "mesh"):
        from apeGmsh.opensees.apesees import apeSees
        from apeGmsh.results.Results import Results

        _install(patches, apeSees, "__init__", _stopper(phase, "apeSees"))
        for meth in ("from_native", "from_mpco", "from_recorders"):
            _install(
                patches,
                Results,
                meth,
                _cls_stopper(phase, f"Results.{meth}"),
            )


def _dump_names(phase: str, *, root: Path) -> dict[str, Any]:
    from ._names import collect_manifest, write_names
    from ._paths import names_path

    manifest = collect_manifest(phase=phase)
    write_names(names_path(root), manifest)
    return manifest


def _record_run(
    script: Path,
    *,
    phase: str,
    ok: bool,
    stopped_at: str | None,
    session: Any,
    geometry_hash: str | None,
    manifest: dict[str, Any] | None,
    error: str | None,
    root: Path,
    cwd: Path | None,
    trigger: str = "open",
) -> None:
    from ._ledger import append_run, make_record
    from ._paths import ledger_path

    name = getattr(session, "name", None) if session is not None else None
    if name is not None:
        name = str(name)
    append_run(
        ledger_path(root),
        make_record(
            script=script,
            phase=phase,
            ok=ok,
            hash=geometry_hash,
            stopped_at=stopped_at,
            session=name,
            manifest=manifest,
            error=error,
            root=root,
            cwd=cwd,
            trigger=trigger,
        ),
    )
    if ok and session is not None:
        from ._project import write_project

        try:
            write_project(root, script=script)
        except OSError:
            pass


def _exec_hold_open(
    script: Path,
    phase: str,
    *,
    root: Path | str | None = None,
    trigger: str = "open",
) -> ReplayResult:
    """Exec *script* with sessions held open. Raises on failure."""
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}; got {phase!r}")

    from apeGmsh._session import _SessionBase
    from apeGmsh.core.Model import Model
    from apeGmsh.mesh.Mesh import Mesh

    from ._paths import resolve_root

    script = Path(script).resolve()
    habitat = resolve_root(root)
    exec_cwd = script.parent
    opened: list[Any] = []
    patches: list[tuple[Any, str, Any]] = []
    original_begin = _SessionBase.begin

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

    _install(patches, _SessionBase, "begin", begin)
    _install(patches, _SessionBase, "end", end)
    _install(patches, _SessionBase, "__exit__", exit_)
    _install(patches, Model, "viewer", _noop_viewer)
    _install(patches, Mesh, "viewer", _noop_viewer)
    _install_phase_gates(patches, phase)

    ns: dict[str, Any] = {
        "__name__": "__main__",
        "__file__": str(script),
        "__package__": None,
    }
    old_cwd = os.getcwd()
    script_dir = str(script.parent)
    path_inserted = script_dir not in sys.path
    stopped_at: str | None = None
    error_text: str | None = None
    caught: BaseException | None = None
    try:
        os.chdir(script.parent)
        if path_inserted:
            sys.path.insert(0, script_dir)
        source = script.read_text(encoding="utf-8")
        code = compile(source, str(script), "exec")
        try:
            exec(code, ns, ns)  # noqa: S102 — replay of the caller's file
        except StopAtPhase as stop:
            stopped_at = stop.gate
        except Exception as exc:
            error_text = traceback.format_exc()
            caught = exc
    except Exception as exc:
        if caught is None:
            error_text = traceback.format_exc()
            caught = exc
    finally:
        if path_inserted:
            try:
                sys.path.remove(script_dir)
            except ValueError:
                pass
        os.chdir(old_cwd)
        _restore(patches)

    digest = _source_hash(script)
    if caught is not None:
        _record_run(
            script,
            phase=phase,
            ok=False,
            stopped_at=None,
            session=None,
            geometry_hash=digest,
            manifest=None,
            error=error_text,
            root=habitat,
            cwd=exec_cwd,
            trigger=trigger,
        )
        raise caught

    mains = [s for s in opened if type(s).__name__ == "apeGmsh"]
    session = mains[-1] if mains else (opened[-1] if opened else None)
    manifest = _dump_names(phase, root=habitat) if session is not None else None
    _record_run(
        script,
        phase=phase,
        ok=True,
        stopped_at=stopped_at,
        session=session,
        geometry_hash=digest,
        manifest=manifest,
        error=None,
        root=habitat,
        cwd=exec_cwd,
        trigger=trigger,
    )
    return ReplayResult(
        ok=True,
        phase=phase,
        geometry_hash=digest,
        error=None,
        session=session,
        stopped_at=stopped_at,
    )


class ReplayRunner:
    """Replay a script; keep the last successful result on failure (INV-4)."""

    def __init__(
        self,
        *,
        exec_fn: ExecFn | None = None,
        root: Path | str | None = None,
    ) -> None:
        self._exec = exec_fn
        self._root = root
        self._last_good: ReplayResult | None = None

    @property
    def last_good(self) -> ReplayResult | None:
        return self._last_good

    def run_until(
        self,
        script: Path,
        *,
        phase: str = "model",
        root: Path | str | None = None,
        trigger: str = "open",
    ) -> ReplayResult:
        if phase not in PHASES:
            raise ValueError(f"phase must be one of {PHASES}; got {phase!r}")
        if trigger not in TRIGGERS:
            raise ValueError(f"trigger must be one of {TRIGGERS}; got {trigger!r}")
        from ._paths import display_path, resolve_root

        script = Path(script)
        habitat = resolve_root(root if root is not None else self._root)
        digest = _source_hash(script) if script.is_file() else None
        if (
            self._last_good is not None
            and self._last_good.ok
            and digest is not None
            and digest == self._last_good.geometry_hash
            and phase == self._last_good.phase
        ):
            return ReplayResult(
                ok=True,
                phase=self._last_good.phase,
                geometry_hash=digest,
                error=None,
                session=self._last_good.session,
                skipped=True,
                stopped_at=self._last_good.stopped_at,
            )
        from ._busy import read_busy, release_busy, try_acquire_busy

        if not try_acquire_busy(
            habitat,
            op="run_until",
            phase=phase,
            script=display_path(script, habitat),
        ):
            info = read_busy(habitat)
            return ReplayResult(
                ok=False,
                phase=phase,
                geometry_hash=digest,
                error=f"BUSY: habitat locked by pid={info.get('pid')} "
                f"op={info.get('op')}",
                session=None,
            )
        try:
            if self._exec is None:
                result = _exec_hold_open(script, phase, root=habitat, trigger=trigger)
            else:
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
                    stopped_at=self._last_good.stopped_at,
                )
            return ReplayResult(
                ok=False,
                phase=phase,
                geometry_hash=digest,
                error=err,
                session=None,
            )
        finally:
            release_busy(habitat)
        if result.ok:
            stamped = ReplayResult(
                ok=True,
                phase=result.phase,
                geometry_hash=digest,
                error=None,
                session=result.session,
                stopped_at=result.stopped_at,
            )
            self._last_good = stamped
            return stamped
        return result
