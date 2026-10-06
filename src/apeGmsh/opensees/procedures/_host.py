"""Typing scaffold for the procedure mixins (ADR 0114 S1-b).

``_ProcedureHost`` declares, for the type checker only, the ``apeSees``
attributes and methods the moved procedure bodies use. The class body is
empty at runtime: nothing here shadows or replaces an ``apeSees`` member, so
the MRO and attribute access of ``apeSees`` are unchanged.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from .._internal.build import StageRecord
    from .._internal.types import Primitive
    from ..pattern.pattern import Plain
    from .._target import OpenSeesCapabilities
    from ..apesees import BuiltModel
    from ..emitter.live import LiveOpsEmitter

    _P = TypeVar("_P", bound=Primitive)


class _ProcedureHost:
    if TYPE_CHECKING:
        _live_emitter: "LiveOpsEmitter | None"
        _primitives: list[Primitive]
        _stage_records: list[StageRecord]

        def capabilities(self) -> "OpenSeesCapabilities": ...
        def _assert_fork_if_required(self) -> None: ...
        def _resolve(
            self,
            ref: "_P | str",
            *,
            base: type[Primitive] = ...,
        ) -> "_P": ...
        def tag_for(self, prim: Primitive) -> int | None: ...
        def build(self) -> BuiltModel: ...
        def _check_explicit_solver_compat(self) -> None: ...
        def _warn_if_unguarded_explicit_run(self) -> None: ...
        def _check_analysis_chain_for_analyze(self) -> None: ...
        def _modal_prereqs_and_guards(
            self, num_modes: int, *, context: str,
        ) -> None: ...
        def _resolve_load_pattern_tag(
            self,
            load: "Plain | str",
            *,
            context: str,
            plain_cls: type,
        ) -> int: ...
