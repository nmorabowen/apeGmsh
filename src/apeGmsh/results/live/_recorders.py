"""LiveRecorders — in-process classic-recorder emission.

Pushes ``ops.recorder("Node", ...)`` / ``ops.recorder("Element", ...)``
calls into the running openseespy domain on stage boundaries
(``begin_stage`` / ``end_stage``) and removes those recorders when
each stage closes (which is what flushes the output files). The
user keeps driving the analysis loop themselves.

Categories
----------
- ``nodes``, ``elements``, ``gauss``, ``line_stations`` — emitted.
- ``fibers``, ``layers`` — warn-and-skip (use MPCO or capture).
- ``modal`` — raises at ``__enter__`` (needs ``ops.eigen()``
  driving, which lives on :meth:`ResolvedRecorderSpec.capture`).

Multi-stage support
-------------------
Each ``begin_stage(name, kind=...)`` issues a fresh set of recorders
whose output filenames are prefixed with ``<name>__`` so the per-stage
files don't collide. Read each stage with::

    Results.from_recorders(spec, output_dir, fem=fem, stage_id=name)

Usage
-----
::

    spec = g.opensees.recorders.resolve(fem, ndm=3, ndf=3)

    with spec.emit_recorders("out/") as live:
        live.begin_stage("gravity", kind="static")
        for _ in range(n_grav):
            ops.analyze(1, 1.0)
        live.end_stage()

        live.begin_stage("dynamic", kind="transient")
        for _ in range(n_dyn):
            ops.analyze(1, dt)
        live.end_stage()

    grav = Results.from_recorders(spec, "out/", fem=fem, stage_id="gravity")
    dyn  = Results.from_recorders(spec, "out/", fem=fem, stage_id="dynamic")
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from ...solvers._recorder_emit import emit_logical, to_ops_args

if TYPE_CHECKING:
    from ...solvers._recorder_specs import ResolvedRecorderSpec


# Categories the live recorder path can emit today.
_SUPPORTED_CATEGORIES = frozenset(
    {"nodes", "elements", "gauss", "line_stations"}
)

# Categories that don't fit the classic recorder format and stay on
# the MPCO / capture paths. Warn-and-skip with a pointer message.
_NON_RECORDER_CATEGORIES = frozenset({"fibers", "layers"})

# Modal records need ``ops.eigen`` driving — out of scope for the
# classic recorder path. Raises at ``__enter__``.
_MODAL_CATEGORIES = frozenset({"modal"})


@dataclass(frozen=True)
class StageRecord:
    """Bookkeeping for one stage that ran inside a LiveRecorders context."""
    name: str
    kind: str
    tags: tuple[int, ...] = field(default_factory=tuple)


class LiveRecorders:
    """Context manager that owns stage-scoped OpenSees recorders.

    Parameters
    ----------
    spec
        The :class:`ResolvedRecorderSpec` whose records to emit.
    output_dir
        Directory the recorder ``.out`` / ``.xml`` files land in.
        Created on ``__enter__`` if missing.
    file_format
        ``"out"`` (text) or ``"xml"``. Defaults to ``"out"``.
    ops
        The openseespy module (or a stand-in for testing). Defaults
        to ``openseespy.opensees`` resolved lazily on ``__enter__``.

    Raises
    ------
    RuntimeError
        On ``__enter__`` if the spec contains any modal records.
    """

    def __init__(
        self,
        spec: "ResolvedRecorderSpec",
        output_dir: "str | Path",
        *,
        file_format: str = "out",
        ops=None,
    ) -> None:
        self._spec = spec
        self._output_dir = str(output_dir) if output_dir else ""
        self._file_format = file_format
        self._ops = ops

        self._opened = False
        self._exited = False
        self._current_stage: Optional[StageRecord] = None
        self._current_tags: list[int] = []
        self._stages: list[StageRecord] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __enter__(self) -> "LiveRecorders":
        if self._opened:
            raise RuntimeError(
                "LiveRecorders is single-use; create a new instance."
            )
        self._opened = True

        # Modal records can't be emitted on the classic recorder path —
        # they need ops.eigen() driving. Fail fast so the user knows
        # to use spec.capture(...) for that portion.
        for record in self._spec.records:
            if record.category in _MODAL_CATEGORIES:
                raise RuntimeError(
                    f"LiveRecorders cannot emit modal record "
                    f"{record.name!r}: modal capture requires "
                    f"ops.eigen() driving, which lives on "
                    f"spec.capture(...). Remove the modal records "
                    f"from the spec before calling emit_recorders, "
                    f"or use spec.capture(...) instead."
                )

        if self._ops is None:
            import openseespy.opensees as ops_module
            self._ops = ops_module

        if self._output_dir:
            Path(self._output_dir).mkdir(parents=True, exist_ok=True)

        return self

    def begin_stage(self, name: str, kind: str = "transient") -> None:
        """Issue recorders for a new stage. Files are prefixed ``<name>__``.

        ``kind`` is forwarded to ``Results.from_recorders(...,
        stage_kind=kind)`` when the stage is read back; valid values
        are ``"transient"`` or ``"static"``.
        """
        self._require_opened()
        if self._current_stage is not None:
            raise RuntimeError(
                f"begin_stage({name!r}) called while stage "
                f"{self._current_stage.name!r} is still open. Call "
                f"end_stage() first."
            )
        if not name:
            raise ValueError("Stage name must be a non-empty string.")
        if "__" in name:
            raise ValueError(
                f"Stage name {name!r} contains '__', which collides "
                f"with the stage/record filename separator."
            )

        self._current_stage = StageRecord(name=name, kind=kind)
        self._current_tags = []

        for record in self._spec.records:
            if record.category in _SUPPORTED_CATEGORIES:
                for logical in emit_logical(
                    record,
                    output_dir=self._output_dir,
                    file_format=self._file_format,
                    stage_id=name,
                ):
                    args = to_ops_args(logical)
                    tag = self._ops.recorder(*args)
                    if isinstance(tag, int):
                        self._current_tags.append(tag)
                continue

            if record.category in _NON_RECORDER_CATEGORIES:
                warnings.warn(
                    f"LiveRecorders: skipping record {record.name!r} "
                    f"(category={record.category!r}); fiber/layer "
                    f"data can't be emitted via classic recorders. "
                    f"Use spec.capture(...) for in-process capture "
                    f"or spec.emit_mpco(...) for the STKO MPCO "
                    f"recorder.",
                    stacklevel=2,
                )
                continue

            # Modal already raised in __enter__, but defend in depth.
            if record.category in _MODAL_CATEGORIES:
                continue

            warnings.warn(
                f"LiveRecorders: skipping record {record.name!r} "
                f"with unrecognised category={record.category!r}.",
                stacklevel=2,
            )

    def end_stage(self) -> None:
        """Remove the current stage's recorders and flush their files."""
        self._require_opened()
        if self._current_stage is None:
            raise RuntimeError(
                "end_stage() called without a matching begin_stage()."
            )

        for tag in self._current_tags:
            try:
                self._ops.remove("recorder", tag)
            except Exception:  # noqa: BLE001 — best-effort flush
                warnings.warn(
                    f"LiveRecorders: failed to remove recorder tag "
                    f"{tag}; output file may not be flushed.",
                    stacklevel=2,
                )

        completed = StageRecord(
            name=self._current_stage.name,
            kind=self._current_stage.kind,
            tags=tuple(self._current_tags),
        )
        self._stages.append(completed)
        self._current_stage = None
        self._current_tags = []

    def __exit__(self, exc_type, exc, tb) -> None:
        # Auto-close a stage that the user forgot to end_stage(). Don't
        # raise here — we don't want to mask the user's exception if
        # one is already in flight.
        if self._current_stage is not None:
            self.end_stage()

        # Friendly diagnostic when the context manager exits without
        # any stage having been opened — almost certainly a bug.
        if exc_type is None and not self._stages:
            warnings.warn(
                "LiveRecorders exited without any begin_stage()/"
                "end_stage() pair. No recorders were issued, so no "
                "output files were produced. Did you forget to call "
                "live.begin_stage(...) inside the with-block?",
                stacklevel=2,
            )

        # Mark closed *after* auto-close so end_stage() above can run.
        self._exited = True

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    @property
    def stages(self) -> tuple[StageRecord, ...]:
        """All completed stages, in the order they ran."""
        return tuple(self._stages)

    @property
    def tags(self) -> tuple[int, ...]:
        """Recorder tags issued so far across all stages (read-only)."""
        out: list[int] = []
        for s in self._stages:
            out.extend(s.tags)
        out.extend(self._current_tags)
        return tuple(out)

    @property
    def output_dir(self) -> str:
        return self._output_dir

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _require_opened(self) -> None:
        if not self._opened:
            raise RuntimeError(
                "LiveRecorders methods must be called from inside the "
                "``with`` block."
            )
        if self._exited:
            raise RuntimeError(
                "LiveRecorders has already exited; create a new instance."
            )
