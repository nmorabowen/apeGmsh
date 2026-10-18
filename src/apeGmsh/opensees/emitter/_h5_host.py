"""Typing scaffold for the H5Emitter mixins (ADR 0114 S5-b).

``_H5Host`` declares, for the type checker only, the ``H5Emitter`` state and the
cross-mixin methods the moved bodies use. The class body is empty at runtime:
nothing here shadows or replaces an ``H5Emitter`` member, so the MRO and attribute
access of ``H5Emitter`` are unchanged. Precedent:
``opensees/procedures/_host.py::_ProcedureHost``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, NoReturn, Sequence
    from .base import DroppedAxisGuard
    from .._internal.typed_records import (
        BeamIntegrationRecord as _BeamIntegrationRecord,
        DampingObjectRecord as _DampingObjectRecord,
        DeclContext as _DeclContext,
        ElementRecord as _ElementRecord,
        EmbeddedNodeRecord as _EmbeddedNodeRecord,
        EqualDOFRecord as _EqualDOFRecord,
        FixRecord as _FixRecord,
        MassRecord as _MassRecord,
        MaterialRecord as _MaterialRecord,
        PatternRecord as _PatternRecord,
        RecorderRecord as _RecorderRecord,
        RegionRecord as _RegionRecord,
        RigidDiaphragmRecord as _RigidDiaphragmRecord,
        RigidLinkRecord as _RigidLinkRecord,
        SectionComplexRecord as _SectionComplexRecord,
        SectionSimpleRecord as _SectionSimpleRecord,
        TimeSeriesRecord as _TimeSeriesRecord,
        TransformRecord as _TransformRecord,
    )
    from ._h5_blocks import _PartitionEmitBlock, _StageEmitBlock
    from ._h5_program import _NO_ROW, _CommandRecord, _EmbeddedRebarRecord, _ProgramTape
    from .h5 import Declaration, DeclTagRun, DeclParamsRow


if TYPE_CHECKING:
    from .base import Emitter as _HostBase
else:
    _HostBase = object


class _H5Host(_HostBase):
    if TYPE_CHECKING:
        _analysis_attrs: dict[str, Any]
        _analyze_call: tuple[int, float | None] | None
        _beam_integrations: list[_BeamIntegrationRecord]
        _command_decls: list[int]
        _commands: list[_CommandRecord]
        _dampings: list[_DampingObjectRecord]
        _decl_context: _DeclContext | None
        _decl_dup_rows: 'dict[tuple[Any, ...], tuple[list[int], int]]'
        _decl_open: int
        _decl_params: 'list[tuple[int, str, dict[str, Any]]] | None'
        _decl_params_restored: 'list[DeclParamsRow] | None'
        _decl_rows_restored: 'dict[str, list[int]] | None'
        _decl_side_rows: 'dict[str, list[int]]'
        _decl_tag_index: 'dict[str, tuple[list[int], list[int], list[int]]]'
        _decl_tags: 'tuple[DeclTagRun, ...]'
        _declarations: 'tuple[Declaration, ...] | None'
        _dropped_axes: DroppedAxisGuard
        _element_decl_run: 'tuple[int, int, int]'
        _element_ranks: list[int]
        _elements: list[_ElementRecord]
        _embedded_nodes: list[_EmbeddedNodeRecord]
        _embedded_rebars: list[_EmbeddedRebarRecord]
        _equal_dofs: list[_EqualDOFRecord]
        _fix_decls: list[int]
        _fixes: list[_FixRecord]
        _initial_stress_param_tags: tuple[tuple[int, int, int], ...]
        _initial_stress_records: list[Any]
        _mass_decls: list[int]
        _mass_from_model: bool
        _masses: list[_MassRecord]
        _nd: list[_MaterialRecord]
        _ndf: int | None
        _ndm: int | None
        _node_coords: list[tuple[float, float, float]]
        _node_tags: list[int]
        _open_pattern: _PatternRecord | None
        _open_pattern_resumed: bool
        _open_section: _SectionComplexRecord | None
        _orientation_tag_counter: int
        _partition_block_by_rank: dict[int, _PartitionEmitBlock]
        _partition_blocks: list[_PartitionEmitBlock]
        _partition_current: _PartitionEmitBlock | None
        _partition_patterns_by_tag: dict[int, _PatternRecord]
        _partition_resumed: bool
        _partition_seen: set[tuple[Any, ...]]
        _patterns_complete: list[_PatternRecord]
        _pending_mp_name: str
        _phantom_node_tags: list[int]
        _program: _ProgramTape
        _program_decl_restored: 'str | None'
        _program_restored: 'tuple[tuple[tuple[int, ...], ...], tuple[str, ...], tuple[str, ...], int] | None'
        _ramp_tags: dict[str, tuple[int, int, int]]
        _recorder_decls: list[int]
        _recorders: list[_RecorderRecord]
        _regions: list[_RegionRecord]
        _rigid_diaphragms: list[_RigidDiaphragmRecord]
        _rigid_links: list[_RigidLinkRecord]
        _sections_complex: list[_SectionComplexRecord]
        _sections_simple: list[_SectionSimpleRecord]
        _solve_stamp_input: 'tuple[bool, str, tuple[str, ...], tuple[str, ...]] | None'
        _stage_blocks: list[_StageEmitBlock]
        _stage_current: _StageEmitBlock | None
        _stage_open_pattern_resumed: bool
        _stage_partition_patterns: 'dict[tuple[str, int], _PatternRecord]'
        _stage_partition_regions: 'dict[tuple[str, int], int]'
        _stage_records_attached: bool
        _time_series: list[_TimeSeriesRecord]
        _transforms: list[_TransformRecord]
        _uniaxial: list[_MaterialRecord]

        def _refuse(self, verb: str, detail: str) -> NoReturn:
            ...

        def _ledger(self, verb: str) -> None:
            ...

        def _stage_block(self, verb: str) -> _StageEmitBlock:
            ...

        def _program_stage(self) -> int:
            ...

        def _command(self, method: str, args: 'tuple[int | float | str, ...]', *, names: 'tuple[str, ...] | None'=None, token: str='', stage: int=_NO_ROW) -> None:
            ...

        def _partition_dup(self, key: 'tuple[Any, ...]', *, own_row: bool=True) -> bool:
            ...

        def _decl_note(self, key: 'tuple[Any, ...]', decls: list[int]) -> None:
            ...

        def _decl_claim(self, key: 'tuple[Any, ...]') -> None:
            ...

        def _flush_open_pattern(self) -> None:
            ...

        def _flush_stage_open_pattern(self, blk: '_StageEmitBlock') -> None:
            ...

        def _decl_params_rows(self) -> 'list[DeclParamsRow]':
            ...

        def _ops_group(self, f: Any) -> Any:
            ...

        def _write_bcs_fix(self, bcs_group: Any, fixes: 'list[_FixRecord] | None'=None) -> None:
            ...

        def _write_bcs_mass(self, bcs_group: Any, masses: 'list[_MassRecord] | None'=None) -> None:
            ...

        def _write_pattern_record(self, parent: Any, rec: _PatternRecord) -> None:
            ...

        def _write_initial_stress_records_into(self, grp: Any, records: 'Sequence[Any]', param_tags: 'Sequence[tuple[int, int, int]]') -> None:
            ...

        def set_declarations(self, decls: 'Sequence[tuple[str, str, str, bool]]', tags: 'Sequence[DeclTagRun]') -> None:
            ...

        def _decl_run(self, kind: str, tag: int) -> 'tuple[int, int, int] | None':
            ...

        @staticmethod
        def _drop_absent_stores(f: Any, rows: 'tuple[tuple[int, ...], ...]', stores: 'tuple[str, ...]') -> 'tuple[tuple[tuple[int, ...], ...], tuple[str, ...]]':
            ...

        def _write_recorder_record(self, parent: Any, rec: _RecorderRecord, idx: int) -> None:
            ...

        def _write_constraints_equal_dof(self, parent: Any, records: 'list[_EqualDOFRecord] | None'=None) -> None:
            ...

        def _write_constraints_rigid_link(self, parent: Any, records: 'list[_RigidLinkRecord] | None'=None) -> None:
            ...

        def _write_constraints_rigid_diaphragm(self, parent: Any, records: 'list[_RigidDiaphragmRecord] | None'=None) -> None:
            ...

        def _write_constraints_embedded_node(self, parent: Any, records: 'list[_EmbeddedNodeRecord] | None'=None) -> None:
            ...
