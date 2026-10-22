"""The partition, stage, parameter, domain and analysis-chain verbs of
:class:`~apeGmsh.opensees.emitter.h5.H5Emitter`, with the stage writer and the
stage and partition restores (ADR 0114 S5-b). A pure move out of ``h5.py``; the mixin holds no state.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Sequence
from .base import command_row
from .._internal.typed_records import DeclContext as _DeclContext
from ._h5_blocks import (
    _PartitionEmitBlock,
    _StageEmitBlock,
    _ro_to_stage_block,
    _with_param_rows,
    _write_param_rows,
)
from ._h5_program import _NO_ROW
from ._h5_write import _set_attr, _write_param_array
from ._h5_write_model import pattern_name
from ._h5_host import _H5Host

if TYPE_CHECKING:
    from .base import StrategySpec


class _H5StagesMixin(_H5Host):
    """Mixin: holds no state, no ``__init__``; see the module docstring."""

    def recorder_declaration_begin(
        self,
        *,
        declaration_name: str,
        record_name: str | None,
        category: str,
        components: tuple[str, ...],
        raw: tuple[str, ...] = (),
        pg: tuple[str, ...] = (),
        label: tuple[str, ...] = (),
        selection: tuple[str, ...] = (),
        ids: tuple[int, ...] | None = None,
        dt: float | None = None,
        n_steps: int | None = None,
        file_root: str = ".",
    ) -> None:
        """Open a recorder-declaration archival context.

        Each subsequent :meth:`recorder` call (until the matching
        :meth:`recorder_declaration_end`) is tagged with this
        metadata, persisted in the unified ``/opensees/recorders/``
        group as schema 2.3.0 ``kind="declared"`` entries.
        """
        if self._decl_context is not None:
            raise RuntimeError(
                "H5Emitter.recorder_declaration_begin: a declaration "
                "context is already open. Nested declarations are not "
                "supported — call recorder_declaration_end() first."
            )
        self._decl_context = _DeclContext(
            declaration_name=declaration_name,
            record_name=record_name,
            category=category,
            components=components,
            raw=raw,
            pg=pg,
            label=label,
            selection=selection,
            ids=ids,
            dt=dt,
            n_steps=n_steps,
            file_root=file_root,
        )

    def recorder_declaration_end(self) -> None:
        """Close the active recorder-declaration archival context."""
        if self._decl_context is None:
            raise RuntimeError(
                "H5Emitter.recorder_declaration_end: no active "
                "declaration context (begin/end mismatch)."
            )
        self._decl_context = None

    # =====================================================================
    # Protocol — Partition emission (ADR 0027, schema 2.10.0)
    # =====================================================================

    def partition_open(self, rank: int) -> None:
        """Enter rank-scoped emit tracking.

        Subsequent :meth:`node` and :meth:`element` calls — until the
        matching :meth:`partition_close` — are mirrored onto an in-memory
        accumulator for the rank.  Each accumulator becomes one
        ``/opensees/partitions/partition_NN`` sub-group at
        :meth:`write` time.

        Unlike the streaming emitters (Tcl / Py), H5 has no notion of a
        "current rank block" in the file format; the call is purely a
        bookkeeping mode switch on the in-memory buffer.

        Raises
        ------
        RuntimeError
            If another partition block is already open.  Nested
            partition brackets are not supported — the bridge fan-out
            in ``_internal/build/`` calls
            ``partition_open(rank) ... partition_close()`` in strict
            sequence, one rank at a time (ADR 0027 §"Implementation
            pointer").
        """
        if self._partition_current is not None:
            raise RuntimeError(
                "H5Emitter.partition_open: a partition block is "
                f"already open (rank={self._partition_current.rank}); "
                "call partition_close() first."
            )
        # P5.1: re-opening a rank RESUMES its accumulator (the
        # partitioned staged emit brackets each rank once per stage) —
        # one ``partition_NN`` group per rank, ever.
        existing = self._partition_block_by_rank.get(int(rank))
        if existing is not None:
            self._partition_current = existing
            self._partition_resumed = True
            return
        self._partition_current = _PartitionEmitBlock(rank=int(rank))

    def partition_close(self) -> None:
        """Close the active partition block and stash it for write-time.

        The accumulated ``node_ids`` / ``element_ids`` are frozen into
        the block; :meth:`_write_partitions` later writes them out and
        computes ``boundary_node_ids`` per-rank as the intersection
        against every other rank's node set (ADR 0027 §"Decision" —
        the boundary-node computation is symmetric across ranks and
        does not require a separate pass).

        Calling :meth:`partition_close` with no open block is a no-op
        — mirrors the tolerance :meth:`pattern_close` already provides.
        """
        if self._partition_current is None:
            return
        if self._partition_resumed:
            # Resumed block is already in ``_partition_blocks``.
            self._partition_resumed = False
            self._partition_current = None
            return
        self._partition_blocks.append(self._partition_current)
        self._partition_block_by_rank[self._partition_current.rank] = (
            self._partition_current
        )
        self._partition_current = None

    # -- Partition runtime-conditional fallback (ADR 0027 INV-5) ----------

    def parallel_runtime_fallback_numberer(
        self, primary: str, fallback: str,
    ) -> None:
        """Record the primary numberer as canonical; stash the fallback
        in ``numberer_runtime_fallback`` so re-emit can reconstruct the
        runtime conditional.

        Minimal additive change to the analysis attrs (ADR 0027 INV-5
        amendment 2026-05-23).
        """
        # Stage-aware sink (ADR 0055): behaviour-identical today —
        # partitioned staged is fail-loud — but routes through the
        # bucket once the Phase-5 lift lands, so the partitioned chain
        # can't reintroduce the global-analysis leak.
        attrs = self._chain_attrs
        attrs["numberer"] = primary
        attrs["numberer_runtime_fallback"] = fallback

    def parallel_runtime_fallback_system(
        self, primary: str, fallback: str,
    ) -> None:
        """Record the primary system as canonical; stash the fallback
        in ``system_runtime_fallback`` so re-emit can reconstruct the
        runtime conditional.

        Mirror of :meth:`parallel_runtime_fallback_numberer`.
        """
        attrs = self._chain_attrs
        attrs["system"] = primary
        attrs["system_runtime_fallback"] = fallback

    # -- Command channel (ADR 0114 D2/D3) ---------------------------------

    def command(self, verb: str, *args: int | float | str) -> None:
        command_row(verb)
        self._command(
            "command", args, token=verb, stage=self._program_stage(),
        )

    # =====================================================================
    # Protocol — Stress control (Phase SSI-1) + Staged analysis (SSI-2)
    # =====================================================================
    #
    # ADR 0055 Phase 2 (schema 2.18.0): the staged-analysis bracket
    # methods CAPTURE into per-stage :class:`_StageEmitBlock` buckets;
    # :meth:`_write_stages` persists them under ``/opensees/stages``.
    # The parameter verbs' information persists declaratively — the
    # global ``set_initial_stress_records`` side-channel (Phase 1) and the
    # per-stage ``initial_stress`` / ``activate_absorbing`` /
    # ``update_parameter`` sub-tables attached by
    # :meth:`set_stage_records` (Phase 2).  Since schema 2.27.0 (K1-8)
    # ``step_hook_ramp`` / ``flip_element_stage`` / ``update_parameter``
    # also capture the parameter tags the build planned, which each record
    # stores as ``param_tags``: the replay reads them and mints none.
    # ``addToParameter`` names a ramp tag ``step_hook_ramp`` already
    # captured, so it stays a no-op.

    def _call_rank(self) -> int:
        """The runtime rank of the open partition bracket, ``-1`` outside
        one (the ``rank`` column of a stored ``(rank, pid)`` row)."""
        part = self._partition_current
        return _NO_ROW if part is None else int(part.rank)

    def addToParameter(
        self, tag: int, ele_tag: int, response: str,
    ) -> None:
        """No-op — resolved form; persists declaratively (ADR 0055).

        The call is its initial stress's declaration's: the bridge opens
        it around the record's calls (opensees 2.32.0).
        """
        del tag, ele_tag, response

    def flip_element_stage(
        self, pid: int, ele_tags: tuple[int, ...],
    ) -> None:
        """Capture the flip's parameter tag (schema 2.27.0, K1-8).

        The record itself persists declaratively via the per-stage
        ``activate_absorbing`` sub-table (ADR 0055 Phase 2);
        :meth:`set_stage_records` gives each record the ``(rank, pid)``
        rows of its calls. The call is its record's declaration's, which
        the bridge opens around it (opensees 2.32.0).
        """
        del ele_tags
        blk = self._stage_block("flip_element_stage")
        blk.flip_calls.append((self._call_rank(), int(pid)))

    def update_parameter(
        self,
        pid: int,
        ele_tags: tuple[int, ...],
        args: tuple[str | int, ...],
        value: float,
    ) -> None:
        """Capture an ``s.update_parameter`` call (schema 2.27.0, K1-8).

        The record persists declaratively under the stage's
        ``update_parameter`` sub-table (:meth:`set_stage_records` attaches
        it and gives it the ``(rank, pid)`` rows of its calls, after
        checking each call's ``args`` and ``value`` against it). The call
        is the record's declaration's, which the bridge opens around it
        (opensees 2.32.0).
        """
        del ele_tags
        blk = self._stage_block("update_parameter")
        blk.update_calls.append(
            (self._call_rank(), int(pid), tuple(args), float(value)))

    def step_hook_ramp(
        self,
        name: str,
        *,
        targets: tuple[tuple[int, float], ...],
        n_steps_to_full: float,
        phase: Literal["before", "after"] = "before",
    ) -> None:
        """Capture the ramp's three parameter tags (schema 2.27.0, K1-8).

        The record persists declaratively via
        ``set_initial_stress_records`` / the per-stage ``initial_stress``
        sub-table (ADR 0055); its ``param_tags`` are these, by name. A
        second ramp of one name in one scope with other tags refuses: the
        archive could not tell which record they belong to. The call is
        the record's declaration's, which the bridge opens around it
        (opensees 2.32.0).
        """
        del n_steps_to_full, phase
        if len(targets) != 3:
            raise RuntimeError(
                f"H5Emitter.step_hook_ramp: ramp {name!r} has "
                f"{len(targets)} targets; an initial-stress ramp has three "
                "(XX, YY, ZZ)."
            )
        xx, yy, zz = (int(tag) for tag, _value in targets)
        tags = (xx, yy, zz)
        blk = self._stage_current
        held = self._ramp_tags if blk is None else blk.ramp_tags
        if held.setdefault(str(name), tags) != tags:
            raise RuntimeError(
                f"H5Emitter.step_hook_ramp: initial stress {name!r} ramps "
                f"twice in one scope, with tags {held[str(name)]} and {tags}."
            )

    def stage_open(self, name: str) -> None:
        """Open a per-stage capture bucket (ADR 0055 Phase 2)."""
        if self._stage_current is not None:
            raise RuntimeError(
                "H5Emitter.stage_open: a stage block is already open "
                f"({self._stage_current.name!r}); call stage_close() "
                "first — stage brackets do not nest."
            )
        if self._open_pattern is not None:
            raise RuntimeError(
                "H5Emitter.stage_open: a global pattern is still open; "
                "the bridge closes every pattern before the staged "
                "sequence begins."
            )
        # A dangling global MP-constraint label must not leak into the
        # first stage constraint's persisted name column (gate-2).
        self._pending_mp_name = ""
        self._stage_current = _StageEmitBlock(name=str(name))

    def stage_close(self) -> None:
        """Close the active stage bucket and stash it for write-time.

        Tolerates a bare call with no open bucket (mirrors
        ``pattern_close`` — Protocol-conformance tests drive the
        bracket methods outside a bridge emit).
        """
        if self._stage_current is None:
            return
        blk = self._stage_current
        # Defensive flush; the bridge always pairs pattern brackets.
        self._flush_stage_open_pattern(blk)
        self._stage_blocks.append(blk)
        self._stage_current = None

    def domain_change(self) -> None:
        """Record the stage's ``domainChange`` barrier (ADR 0055 Phase 2)."""
        self._stage_block("domain_change").domain_changed = True

    # -- Staged-analysis mutators (Phase SSI-2.E) ---------------------------
    # Capture into the active stage bucket (ADR 0055 Phase 2).  All five
    # are only emitted inside a stage bracket by bridge-driven call
    # sites; bare calls (Protocol-conformance tests) stay no-ops.
    # Presence semantics: a mutator never called leaves its bucket
    # field at the sentinel and the writer omits the attribute —
    # never-set is structurally distinct from value-0.

    def set_time(self, t: float) -> None:
        self._stage_block("set_time").set_time = float(t)

    def set_creep(self, on: bool) -> None:
        self._stage_block("set_creep").set_creep_on = bool(on)

    def reset(self) -> None:
        self._stage_block("reset").pre_analyze_reset = True

    def set_node_vel(self, node: int, dof: int, value: float) -> None:
        self._count_node_kinematics("set_node_vel", value)

    def set_node_accel(self, node: int, dof: int, value: float) -> None:
        self._count_node_kinematics("set_node_accel", value)

    def _count_node_kinematics(self, verb: str, value: float) -> None:
        """Count one ``s.zero_velocities`` call (schema 2.27.0, K1-8).

        The stage's ``ZeroVelocityRecord``s persist declaratively under
        its ``zero_velocities`` sub-table (:meth:`set_stage_records`
        attaches them); replay re-expands them over the archived nodes
        and their ndf.  The count cross-checks that a stage which zeroed
        anything declared a record, and a nonzero value refuses: the
        record can only say "zero".
        """
        blk = self._stage_block(verb)
        if float(value) != 0.0:
            raise RuntimeError(
                f"H5Emitter.{verb}: stage {blk.name!r} sets a nonzero "
                f"value ({value!r}); the archive stores s.zero_velocities "
                "records, which zero the state."
            )
        blk.node_kinematics_calls += 1

    def remove_sp(self, node: int, dof: int) -> None:
        blk = self._stage_block("remove_sp")
        # P5.1: remove_sp replicates on every rank owning the node
        # (mirrors fix's INV-4 fan-out) — capture once.
        key = ("stage_remove_sp", blk.name, int(node), int(dof))
        if self._partition_dup(key):
            self._decl_claim(key)
            return
        self._decl_note(key, blk.remove_sp_decls)
        blk.remove_sps.append((int(node), int(dof)))

    def remove_element(self, tag: int) -> None:
        blk = self._stage_block("remove_element")
        # A removal fires on the element's one owner rank: no replica.
        blk.remove_element_decls.append(self._decl_open)
        self._program.decl = self._decl_open  # A7
        blk.remove_elements.append(int(tag))

    def update_material_stage(self, mat_tag: int, stage: int) -> None:
        blk = self._stage_block("update_material_stage")
        # A stage flip is model-global: the partitioned emit path
        # replicates the line on EVERY rank (each rank is its own
        # process with its own static mElastFlag), so the capture
        # dedupes exactly like remove_sp's INV-4 fan-out.  Without
        # this the archive would carry N copies per material under
        # partitioning and one copy without it — an unstable
        # model_hash for the same authored model.
        key = ("stage_update_material_stage", blk.name,
               int(mat_tag), int(stage))
        if self._partition_dup(key):
            self._decl_claim(key)
            return
        self._decl_note(key, blk.update_material_stage_decls)
        blk.update_material_stages.append((int(mat_tag), int(stage)))

    # =====================================================================
    # Protocol — Analysis chain
    # =====================================================================

    @property
    def _chain_attrs(self) -> "dict[str, Any]":
        """The analysis-chain attr sink — stage-aware (ADR 0055 Phase 2).

        Inside a stage bracket the chain directives belong to THAT
        stage; routing them here (instead of ``_analysis_attrs``) is
        what prevents the last stage's chain from leaking into a
        phantom global ``/opensees/analysis`` group.
        """
        if self._stage_current is not None:
            return self._stage_current.chain_attrs
        return self._analysis_attrs

    def constraints(self, c_type: str, *args: int | float | str) -> None:
        # Every handler is recorded, ``LadrunoContact`` included (#2430). The
        # OpenSees *deck* zone (``/opensees/...``) carries no contactSurface /
        # contact records (the contact persists in the NEUTRAL zone, schema
        # 2.21.0; ``FEMData.from_h5`` → ``apeSees(fem)`` re-runs
        # emit_contacts), so the replayed deck declares the handler with no
        # contact to handle; LadrunoContact is Plain-style for MP, the
        # default handler's behaviour. Recording it is what puts ``"fork"``
        # in the stamp (``caps.FORK_CHAIN``): the replay names a fork-only
        # handler, so a stock backend must refuse it.
        attrs = self._chain_attrs
        attrs["handler"] = c_type
        if args:
            attrs["handler_args"] = tuple(args)

    def numberer(self, n_type: str) -> None:
        self._chain_attrs["numberer"] = n_type

    def system(self, s_type: str, *args: int | float | str) -> None:
        attrs = self._chain_attrs
        attrs["system"] = s_type
        if args:
            attrs["system_args"] = tuple(args)

    def test(self, t_type: str, *args: int | float | str) -> None:
        attrs = self._chain_attrs
        attrs["test"] = t_type
        if args:
            attrs["test_args"] = tuple(args)

    def algorithm(self, a_type: str, *args: int | float | str) -> None:
        attrs = self._chain_attrs
        attrs["algorithm"] = a_type
        if args:
            attrs["algorithm_args"] = tuple(args)

    def integrator(self, i_type: str, *args: int | float | str) -> None:
        attrs = self._chain_attrs
        attrs["integrator"] = i_type
        if args:
            attrs["integrator_args"] = tuple(args)

    def analysis(self, a_type: str) -> None:
        self._chain_attrs["analysis"] = a_type

    def analyze(
        self, *, steps: int, dt: float | None = None,
        label: str | None = None,
        strategy: "StrategySpec | None" = None,
    ) -> int:
        # ``label`` is a deck-banner concern (py/tcl fail-loud loops);
        # the archive stores the declarative (steps, dt) only — the
        # stage name is already archived on the stage record itself.
        # ``strategy`` (ADR 0057) is accepted and NOT archived in
        # Phase A — declaration persistence is Phase C (schema bump).
        del strategy
        call = (int(steps), None if dt is None else float(dt))
        if self._stage_current is not None:
            self._stage_current.analyze_call = call
        else:
            self._analyze_call = call
        return 0

    def eigen(
        self, num_modes: int, *, solver: str = "-genBandArpack",
    ) -> list[float]:
        # The bridge emits ``eigen`` into a deck only ahead of
        # ``modalDamping`` (ADR 0053 D4), so it is part of the deck's
        # program: a ``/opensees/commands`` row (ADR 0114 R3a).
        # ``apeSees.eigen(...)`` drives a LiveOpsEmitter, never H5.
        self._command(
            "eigen", (num_modes, solver), names=("", "solver"),
        )
        return []

    def modal_properties(
        self, *, unorm: bool = False, out: str | None = None,
    ) -> dict[str, list[float]]:
        # Runtime one-shot retrieval — nothing in the model definition
        # to archive.  No-op, mirroring ``eigen`` above.
        del unorm, out
        self._ledger("modal_properties")
        return {}

    def eigen_feast(
        self, f_min: float, f_max: float, *, certify: bool = False,
    ) -> list[float]:
        # Runtime one-shot retrieval — no-op, mirroring ``eigen`` above
        # (ADR 0075 INV-2).
        del f_min, f_max, certify
        self._ledger("eigen_feast")
        return []

    def modal_response_history(
        self, *args: int | float | str,
    ) -> None:
        # Runtime analysis command — not model definition.  No-op,
        # mirroring ``eigen`` above (ADR 0075 INV-2).
        del args
        self._ledger("modal_response_history")

    def response_spectrum_analysis(
        self, direction: int, *args: int | float | str,
    ) -> None:
        # Runtime analysis command — not model definition.  No-op,
        # mirroring ``eigen`` above (ADR 0075 INV-2).
        del direction, args
        self._ledger("response_spectrum_analysis")

    def profiler(self, *args: int | float | str) -> None:
        # A stage's ``s.profile`` bracket (TIMs A8): ``profiler start``
        # before the stage's analyze, ``stop`` / ``report`` after. Each
        # call is a stage ``/opensees/commands`` row (ADR 0114 R3a); the
        # program's emit order places it around ``analyze`` on replay.
        self._stage_block("profiler")
        self._command("profiler", args, stage=self._program_stage())

    def restore_partition_blocks(
        self, partitions_ro: "Sequence[Any]",
    ) -> None:
        """Re-install partition blocks from read-side
        ``PartitionEmittedRecord`` values (ADR 0055 Phase 5 / P5.0b —
        the ``from_h5 → to_h5`` echo path, mirroring
        :meth:`restore_stage_blocks`).

        Before this echo existed the re-write path was partition-blind:
        ``OpenSeesModel._populate_emitter_h5`` re-drives the element
        pool with no rank brackets, so a re-written partitioned archive
        silently dropped ``/opensees/partitions`` and degraded every
        ``element_meta/*/partition_ids`` column to ``-1`` — drifting
        ``model_hash`` (both zones fold in).

        Trusts the archive: per-rank ``node_ids`` / ``element_ids``
        re-populate ``_PartitionEmitBlock`` accumulators verbatim;
        ``boundary_node_ids`` is NOT restored — :meth:`_write_partitions`
        recomputes it from the restored node sets (one-pass symmetric
        intersection), reproducing the stored dataset byte-for-byte.
        Also re-stamps ``_element_ranks`` by tag (the flat populate
        pass left every entry at the ``-1`` sentinel), so call this
        AFTER the element pool is populated.

        Refuses to overwrite in-flight or already-captured partition
        state — the restore path is only ever driven on a fresh
        emitter (``OpenSeesModel._compose_h5``).
        """
        if self._partition_current is not None or self._partition_blocks:
            raise RuntimeError(
                "H5Emitter.restore_partition_blocks: emitter already "
                "carries captured partition state; restore is only "
                "valid on a fresh emitter."
            )
        tag_to_rank: dict[int, int] = {}
        blocks: list[_PartitionEmitBlock] = []
        for ro in partitions_ro:
            blk = _PartitionEmitBlock(rank=int(ro.rank))
            for n in ro.node_ids:
                blk.add_node(int(n))
            for e in ro.element_ids:
                blk.add_element(int(e))
                tag_to_rank[int(e)] = int(ro.rank)
            blocks.append(blk)
        self._partition_blocks = blocks
        self._partition_block_by_rank = {blk.rank: blk for blk in blocks}
        self._element_ranks = [
            tag_to_rank.get(int(rec.tag), -1) for rec in self._elements
        ]

    def restore_stage_blocks(self, stages_ro: "Sequence[Any]") -> None:
        """Re-install captured stage buckets from read-side
        ``StageRecordRO`` values (ADR 0055 Phase 2, the
        ``from_h5 → to_h5`` echo path).

        Unlike :meth:`set_stage_records` (which cross-checks live
        capture against bridge ``StageRecord``s), this trusts the
        archive: the RO records ARE the persisted truth, thawed
        field-for-field by :func:`_ro_to_stage_block`.  Marks the
        declarative complement as attached so :meth:`_write_stages`
        accepts the blocks.  Refuses to overwrite an in-flight or
        already-captured stage state — the restore path is only ever
        driven on a fresh emitter (``OpenSeesModel._compose_h5``).
        """
        if self._stage_current is not None or self._stage_blocks:
            raise RuntimeError(
                "H5Emitter.restore_stage_blocks: emitter already "
                "carries captured stage state; restore is only valid "
                "on a fresh emitter."
            )
        self._stage_blocks = [_ro_to_stage_block(ro) for ro in stages_ro]
        self._stage_records_attached = True

    def _write_stages(self, f: Any) -> None:
        """Persist ``/opensees/stages/stage_NNN`` groups (ADR 0055 Phase 2).

        One group per captured stage bucket, zero-padded so name-sorted
        order == registration order == replay order.  Sub-tables reuse
        the global per-group writers (same compound dtypes — the
        global-ndf width pinning keeps ``model_hash`` stable across a
        round-trip).  Early-return when no stage exists: vanilla files
        stay byte-identical and their ``model_hash`` is unchanged.
        Folds into ``model_hash`` (authored model state).
        """
        import numpy as np

        if not self._stage_blocks:
            if self._stage_current is not None:
                raise RuntimeError(
                    "H5Emitter._write_stages: stage block "
                    f"{self._stage_current.name!r} is still open at "
                    "write time — unbalanced stage_open/stage_close."
                )
            return
        if self._stage_current is not None:
            raise RuntimeError(
                "H5Emitter._write_stages: stage block "
                f"{self._stage_current.name!r} is still open at write "
                "time — unbalanced stage_open/stage_close."
            )
        if not self._stage_records_attached:
            raise RuntimeError(
                "H5Emitter._write_stages: captured "
                f"{len(self._stage_blocks)} stage bracket(s) but "
                "set_stage_records() was never called — writing now "
                "would silently drop the declarative complement "
                "(activated_pgs / per-stage initial_stress / "
                "activate_absorbing).  Drive the emitter through "
                "apeSees.h5, or call set_stage_records first."
            )
        stages = self._ops_group(f).create_group("stages")
        _set_attr(stages, "n_stages", len(self._stage_blocks))
        for idx, blk in enumerate(self._stage_blocks):
            g = stages.create_group(f"stage_{idx:03d}")
            _set_attr(g, "name", blk.name)
            # Analyze loop — steps always present (cross-checked against
            # StageRecord.n_increments); dt only when the stage is
            # transient (attr presence == tri-state).
            if blk.analyze_call is None:
                raise RuntimeError(
                    f"H5Emitter._write_stages: stage {blk.name!r} "
                    "captured no analyze call — malformed bracket."
                )
            steps, dt_val = blk.analyze_call
            _set_attr(g, "analyze_steps", steps)
            if dt_val is not None:
                _set_attr(g, "analyze_dt", dt_val)
            # Presence-encoded SSI-2.E time-state mutators.
            if blk.set_time is not None:
                _set_attr(g, "set_time", float(blk.set_time))
            if blk.set_creep_on is not None:
                _set_attr(g, "set_creep_on", int(blk.set_creep_on))
            if blk.pre_analyze_reset:
                _set_attr(g, "pre_analyze_reset", 1)
            _set_attr(g, "domain_change", int(blk.domain_changed))
            # Activation manifest — verbatim StageRecord tuple order
            # (never sorted-from-a-set; ordering folds into the hash).
            if blk.activated_pgs:
                import h5py
                g.create_dataset(
                    "activated_pgs",
                    data=np.asarray(list(blk.activated_pgs), dtype=object),
                    dtype=h5py.string_dtype(encoding="utf-8"),
                )
            # Owned topology — resolved tags in emit order (== replay
            # order; ADR 0055 "persist, do NOT re-derive ownership").
            if blk.owned_node_ids:
                g.create_dataset(
                    "owned_node_ids",
                    data=np.asarray(blk.owned_node_ids, dtype=np.int64),
                )
            if blk.owned_element_ids:
                g.create_dataset(
                    "owned_element_ids",
                    data=np.asarray(blk.owned_element_ids, dtype=np.int64),
                )
            # SSI-2.E removals (resolved targets, emit order).
            if blk.remove_sps:
                g.create_dataset(
                    "remove_sp",
                    data=np.asarray(
                        [[n, d] for n, d in blk.remove_sps], dtype=np.int64,
                    ).reshape(len(blk.remove_sps), 2),
                )
            if blk.remove_elements:
                g.create_dataset(
                    "remove_element",
                    data=np.asarray(blk.remove_elements, dtype=np.int64),
                )
            # SSI-2.E SANISAND stage flips — (mat_tag, stage) rows.
            if blk.update_material_stages:
                g.create_dataset(
                    "update_material_stage",
                    data=np.asarray(
                        [[m, v] for m, v in blk.update_material_stages],
                        dtype=np.int64,
                    ).reshape(len(blk.update_material_stages), 2),
                )
            # Stage-bound BCs — same compound writers as the global
            # ``/opensees/bcs`` zone.
            if blk.fixes or blk.masses:
                bcs = g.create_group("bcs")
                if blk.fixes:
                    self._write_bcs_fix(bcs, blk.fixes)
                if blk.masses:
                    self._write_bcs_mass(bcs, blk.masses)
            # Stage regions (resolved echo — region tags freeze per the
            # ADR 0055 identity model; replay re-emits them verbatim).
            if blk.regions:
                regions = g.create_group("regions")
                for r_idx, region_rec in enumerate(blk.regions):
                    rg = regions.create_group(f"region_{r_idx:03d}")
                    _set_attr(rg, "tag", region_rec.tag)
                    # Provenance (gate-2): four producers share this
                    # pool (stage regions, region-scoped rayleigh,
                    # damping attach, recorder pg-filter); the kind is
                    # derived from the flag tail and emit_index pins
                    # the original interleaving (OpenSees overwrite
                    # semantics are order-sensitive).
                    str_toks = {
                        a for a in region_rec.args if isinstance(a, str)
                    }
                    if "-rayleigh" in str_toks:
                        kind = "rayleigh"
                    elif "-damp" in str_toks:
                        kind = "damping_attach"
                    else:
                        kind = "node_or_filter"
                    _set_attr(rg, "kind", kind)
                    _set_attr(rg, "emit_index", blk.region_seq[r_idx])
                    _write_param_array(rg, "params", region_rec.args)
            # Stage MP constraints — same compound writers as the
            # global ``/opensees/constraints`` zone.
            if (
                blk.equal_dofs or blk.rigid_links
                or blk.rigid_diaphragms or blk.embedded_nodes
            ):
                cons = g.create_group("constraints")
                if blk.equal_dofs:
                    self._write_constraints_equal_dof(cons, blk.equal_dofs)
                if blk.rigid_links:
                    self._write_constraints_rigid_link(cons, blk.rigid_links)
                if blk.rigid_diaphragms:
                    self._write_constraints_rigid_diaphragm(
                        cons, blk.rigid_diaphragms,
                    )
                if blk.embedded_nodes:
                    self._write_constraints_embedded_node(
                        cons, blk.embedded_nodes,
                    )
                # ADR 0055 P2.3: per-MP-kind emit_index so replay
                # reproduces the genuine-vs-kinematic equalDOF order
                # straddling rigidDiaphragm.  Parallel int64 datasets
                # (one per kind, row-aligned with the compound above).
                for nm, seq in (
                    ("equalDOF_emit_index", blk.equal_dof_seq),
                    ("rigidLink_emit_index", blk.rigid_link_seq),
                    ("rigidDiaphragm_emit_index", blk.rigid_diaphragm_seq),
                    ("embeddedNode_emit_index", blk.embedded_node_seq),
                ):
                    if seq:
                        cons.create_dataset(
                            nm, data=np.asarray(seq, dtype=np.int64),
                        )
            # Stage rayleigh (global ``on=()`` form — four coefficients
            # per call; region-scoped forms ride the regions echo).
            if blk.rayleighs:
                g.create_dataset(
                    "rayleigh",
                    data=np.asarray(blk.rayleighs, dtype=np.float64).reshape(
                        len(blk.rayleighs), 4,
                    ),
                )
                g.create_dataset(
                    "rayleigh_emit_index",
                    data=np.asarray(blk.rayleigh_seq, dtype=np.int64),
                )
            # Per-stage declarative initial stress (Phase-1 field set).
            if blk.initial_stress_records:
                self._write_initial_stress_records_into(
                    g.create_group("initial_stress"),
                    blk.initial_stress_records,
                    blk.initial_stress_param_tags,
                )
            # Absorbing-boundary stage flip (ADR 0054 AB-3) — declarative
            # pg XOR elements, mirroring the initial-stress discriminant,
            # plus the ``param_tags`` (rank, pid) rows (schema 2.27.0).
            if blk.activate_absorbing_records:
                absorb = g.create_group("activate_absorbing")
                for a_idx, (ab_rec, rows) in enumerate(_with_param_rows(
                    blk.name, "activate_absorbing",
                    blk.activate_absorbing_records,
                    blk.absorbing_param_tags,
                )):
                    ag = absorb.create_group(f"absorb_{a_idx:03d}")
                    if ab_rec.elements is not None:
                        ag.create_dataset(
                            "elements",
                            data=np.asarray(
                                [int(e) for e in ab_rec.elements],
                                dtype=np.int64,
                            ),
                        )
                    else:
                        _set_attr(ag, "pg", ab_rec.pg)
                    _write_param_rows(ag, rows)
            # ``s.update_parameter`` (schema 2.27.0, K1-8) — the same
            # declarative target discriminant, the parameter name, value
            # and optional material tag, and the (rank, pid) rows.
            if blk.update_parameter_records:
                upd = g.create_group("update_parameter")
                for u_idx, (up_rec, rows) in enumerate(_with_param_rows(
                    blk.name, "update_parameter",
                    blk.update_parameter_records,
                    blk.update_parameter_param_tags,
                )):
                    ug = upd.create_group(f"update_{u_idx:03d}")
                    _set_attr(ug, "name", str(up_rec.name))
                    _set_attr(ug, "value", float(up_rec.value))
                    if up_rec.mat_tag is not None:
                        _set_attr(ug, "mat_tag", int(up_rec.mat_tag))
                    if up_rec.elements is not None:
                        ug.create_dataset(
                            "elements",
                            data=np.asarray(
                                [int(e) for e in up_rec.elements],
                                dtype=np.int64,
                            ),
                        )
                    else:
                        _set_attr(ug, "pg", up_rec.pg)
                    _write_param_rows(ug, rows)
            # ``s.zero_velocities`` (schema 2.27.0, K1-8) — one group per
            # record; a ``nodes`` dataset when it names nodes, none when it
            # zeroes the whole domain (``nodes=None``).
            if blk.zero_velocity_records:
                zero = g.create_group("zero_velocities")
                for z_idx, z_rec in enumerate(blk.zero_velocity_records):
                    zg = zero.create_group(f"zero_{z_idx:03d}")
                    if z_rec.nodes is not None:
                        zg.create_dataset(
                            "nodes",
                            data=np.asarray(
                                [int(n) for n in z_rec.nodes],
                                dtype=np.int64,
                            ),
                        )
            # Per-stage analysis chain + patterns + recorders.
            if blk.chain_attrs:
                chain = g.create_group("analysis")
                for key, value in blk.chain_attrs.items():
                    _set_attr(chain, key, value)
            if blk.patterns_complete:
                patterns = g.create_group("patterns")
                for p_idx, pat_rec in enumerate(blk.patterns_complete):
                    self._write_pattern_record(patterns, pat_rec)
                    pgrp = patterns[pattern_name(pat_rec)]
                    _set_attr(pgrp, "emit_index", blk.pattern_seq[p_idx])
                    if pat_rec.sp_holds:
                        # Explicit role attr — the ADR 0052 HOLD
                        # support pattern, not a stage load pattern
                        # (sp_holds-presence stays as data, the role
                        # is the contract; gate-2 finding).
                        _set_attr(pgrp, "role", "hold")
            if blk.recorders:
                recorders = g.create_group("recorders")
                for r_idx, recorder_rec in enumerate(blk.recorders):
                    self._write_recorder_record(
                        recorders, recorder_rec, r_idx,
                    )
