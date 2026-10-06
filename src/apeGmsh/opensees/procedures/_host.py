"""Typing scaffold for the procedure mixins (ADR 0114 S1-b).

``_ProcedureHost`` declares, for the type checker only, the ``apeSees``
attributes and methods the moved procedure bodies use. The class body is
empty at runtime: nothing here shadows or replaces an ``apeSees`` member, so
the MRO and attribute access of ``apeSees`` are unchanged.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..._internal.provenance import ProvenanceStore
    from ...mesh.FEMData import FEMData
    from .._internal.artifact_write import BridgeArtifactWriter
    from .._internal.build import (
        DampingAttachRecord,
        EquationConstraintRecord,
        FixRecord,
        InitialStressRecord,
        MassRecord,
        ModalDampingRecord,
        NdfRecord,
        RayleighRecord,
        RegionAssignmentRecord,
        StageRecord,
    )
    from .._internal.types import Primitive
    from ..pattern.pattern import Plain
    from .._target import OpenSeesCapabilities
    from ..apesees import BuiltModel
    from ..emitter.live import LiveOpsEmitter
    from ...cuts import SectionCutDef, SectionSweepDef

    _P = TypeVar("_P", bound=Primitive)


class _ProcedureHost:
    if TYPE_CHECKING:
        # The declared state the automatic artifact write keys on
        # (``_internal/artifact_write.py``, P5) and the writer it calls.
        _artifacts: "BridgeArtifactWriter"
        _fem: "FEMData"
        _provenance: "ProvenanceStore"
        _live_emitter: "LiveOpsEmitter | None"
        _primitives: list[Primitive]
        _fix_records: list[FixRecord]
        _equation_constraint_records: list[EquationConstraintRecord]
        _mass_records: list[MassRecord]
        _ndf_records: list[NdfRecord]
        _region_records: list[RegionAssignmentRecord]
        _rayleigh_records: list[RayleighRecord]
        _damping_attach_records: list[DampingAttachRecord]
        _modal_damping_records: list[ModalDampingRecord]
        _initial_stress_records: list[InitialStressRecord]
        _stage_records: list[StageRecord]
        _mass_from_model: bool
        _fix_from_model: bool
        _ndm: int | None
        _ndf: int | None

        def h5(
            self,
            path: str,
            *,
            model_name: str | None = ...,
            cuts: "Sequence[SectionCutDef]" = ...,
            sweeps: "Sequence[SectionSweepDef]" = ...,
        ) -> None: ...

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
