"""The declaration, claim, program and command methods of
:class:`~apeGmsh.opensees.emitter.h5.H5Emitter` (ADR 0114 S5-b). A pure move out of
``h5.py``; the mixin holds no state.
"""
from __future__ import annotations

import bisect
from collections.abc import Mapping
from typing import Any, Iterable, Sequence, cast
from .caps import SolveStamp
from .verbs import VERBS
from ._h5_blocks import _attribute_param_calls, _check_update_calls, _ramp_tags_by_name
from ._h5_program import _KIND_STR, _NO_DECL, _NO_ROW, _CommandRecord, _command_kind
from ._h5_write_model import pattern_name
from ._h5_host import _H5Host


def _json_text(value: Any) -> str:
    import json

    return json.dumps(value, allow_nan=False, separators=(",", ":"))


def _decl_refs(value: Any) -> "Iterable[str]":
    """Every declaration key an encoded value references, at any depth."""
    if isinstance(value, dict):
        if set(value) == {"$decl"}:
            yield value["$decl"]
        else:
            for v in value.values():
                yield from _decl_refs(v)
    elif isinstance(value, list):
        for v in value:
            yield from _decl_refs(v)


class _H5DeclsMixin(_H5Host):
    """Mixin: holds no state, no ``__init__``; see the module docstring."""

    def _decl_note(self, key: "tuple[Any, ...]", decls: list[int]) -> None:
        """Note the open declaration for the record about to be appended
        to the store whose declaration column is ``decls`` (K1-6).

        Inside a partition bracket the row is remembered under its dedupe
        ``key``, so a replica that arrives with a declaration open can
        give it one (:meth:`_decl_claim`). The open declaration is also
        the call's ``/opensees/program`` ``decl`` (A7).
        """
        decls.append(self._decl_open)
        self._program.decl = self._decl_open
        if self._partition_current is not None:
            self._decl_dup_rows[key] = (decls, len(decls) - 1)

    def _decl_claim(self, key: "tuple[Any, ...]") -> None:
        """A partition replica of ``key`` arrived: its declaration fills
        the first capture's row when that one was captured with none open
        (a ghost replay before the owner's fix, ADR 0027 INV-2). The
        replica's own call is the open declaration's (A7)."""
        self._program.decl = self._decl_open
        if self._decl_open == _NO_DECL:
            return
        decls, row = self._decl_dup_rows[key]
        if decls[row] == _NO_DECL:
            decls[row] = self._decl_open

    # -- Per-group writers (split out so each step adds one) -------------

    def _argv_by_tag(self) -> "dict[str, dict[int, tuple[Any, ...]]]":
        """``kind -> tag -> argv`` of the tagged stores whose rows carry
        one flat argv per tag (:data:`_ARGV_STORE_KINDS`): the argv a
        declaration's ``params_names`` is checked against."""
        return {
            "uniaxialMaterial": {r.tag: r.params for r in self._uniaxial},
            "nDMaterial": {r.tag: r.params for r in self._nd},
            "section": {r.tag: r.params for r in self._sections_simple},
            "beamIntegration": {r.tag: r.args for r in self._beam_integrations},
            "timeSeries": {r.tag: r.args for r in self._time_series},
            "damping": {r.tag: r.args for r in self._dampings},
        }

    def _write_decl_params(self, f: Any) -> None:
        """Persist ``/opensees/decl_params`` (ADR 0114 A6/Q4, K1-7, schema
        2.26.0): one row per handed-in primitive, parallel columns
        ``decl``, ``type``, ``params``, ``params_names``, ``transf_ref``,
        ``integration_ref`` and ``section_ref``. Written only once
        :meth:`set_decl_params` ran (or a rewrite echoed the source's);
        a referenced key the declarations lack refuses. Hash-excluded:
        ``decl_params`` is in ``MODEL_HASH_EXCLUDED_CHILDREN``.
        """
        if self._decl_params_restored is not None:
            rows = self._decl_params_restored
        elif self._decl_params is not None:
            if self._declarations is None:
                raise RuntimeError(
                    "H5Emitter._write_decl_params: parameters without "
                    "declarations; set_declarations runs first.")
            keys = {d.key for d in self._declarations}
            for decl, type_name, params in self._decl_params:
                for ref in _decl_refs(params):
                    if ref not in keys:
                        raise RuntimeError(
                            f"H5Emitter._write_decl_params: {type_name} "
                            f"(declaration {decl}) references {ref!r}, "
                            "which /opensees/decls does not declare.")
            rows = self._decl_params_rows()
        else:
            return
        import h5py
        import numpy as np

        str_dt = h5py.string_dtype(encoding="utf-8")
        g = self._ops_group(f).create_group("decl_params")
        g.create_dataset(
            "decl", data=np.array([r[0] for r in rows], dtype=np.int64))
        for col, name in ((1, "type"), (2, "params"), (3, "params_names")):
            g.create_dataset(name, data=[r[col] for r in rows], dtype=str_dt)
        # Each ``*_ref`` is an index run: row i's keys are
        # ``key[first[i] : first[i] + count[i]]``, in field order.
        for col, name in ((4, "transf_ref"), (5, "integration_ref"),
                          (6, "section_ref")):
            run = g.create_group(name)
            run_keys: list[str] = []
            first: list[int] = []
            count: list[int] = []
            for r in rows:
                refs = cast("tuple[str, ...]", r[col])
                first.append(len(run_keys))
                count.append(len(refs))
                run_keys.extend(refs)
            run.create_dataset("first", data=np.array(first, dtype=np.int64))
            run.create_dataset("count", data=np.array(count, dtype=np.int64))
            run.create_dataset("key", data=run_keys, dtype=str_dt)

    def _write_decls(self, f: Any) -> None:
        """Persist ``/opensees/decls`` (ADR 0114 R5, K1-6, schema 2.25.0).

        Written only once :meth:`set_declarations` ran. One row per
        declaration (``key``, ``family``, ``name``, ``synth``); ``tags``
        holds the ``(kind, first, count, decl)`` runs of the tagged ones;
        ``rows/<store>`` holds one declaration index per record of each
        tagless store this archive wrote, aligned to its rows:
        ``bcs/fix``, ``bcs/mass``, ``recorders`` (in group index order)
        and the same three under ``stages/stage_NNN``. A tagless record
        with no declaration open refuses: the archive would carry a row
        no declaration claims. Hash-excluded: ``decls`` is in
        ``MODEL_HASH_EXCLUDED_CHILDREN``.
        """
        if self._declarations is None:
            return
        import h5py
        import numpy as np

        str_dt = h5py.string_dtype(encoding="utf-8")
        decls = self._declarations
        g = self._ops_group(f).create_group("decls")
        # A7: whether /opensees/program's decl column records each call's
        # declaration. On the hash-excluded group, not on /opensees (whose
        # attributes fold into model_hash), so a rewrite keeps the hash.
        from .h5_reader import PROGRAM_DECL_COMPLETE, PROGRAM_DECL_MARKERS
        marker = (PROGRAM_DECL_COMPLETE if self._program_decl_restored is None
                  else self._program_decl_restored)
        if marker not in PROGRAM_DECL_MARKERS:
            raise RuntimeError(
                f"H5Emitter._write_decls: program_decl marker {marker!r} is "
                f"not one of {sorted(PROGRAM_DECL_MARKERS)}.")
        g.attrs["program_decl"] = marker
        g.create_dataset("key", data=[d.key for d in decls], dtype=str_dt)
        g.create_dataset(
            "family", data=[d.family for d in decls], dtype=str_dt)
        g.create_dataset("name", data=[d.name for d in decls], dtype=str_dt)
        g.create_dataset(
            "synth", data=np.array([int(d.synth) for d in decls],
                                   dtype=np.int8))
        tags = g.create_group("tags")
        runs = self._decl_tags
        tags.create_dataset("kind", data=[r[0] for r in runs], dtype=str_dt)
        for col, name in ((1, "first"), (2, "count"), (3, "decl")):
            tags.create_dataset(
                name, data=np.array([r[col] for r in runs], dtype=np.int64))

        side = self._decl_side_rows
        stores: list[tuple[str, list[int], Sequence[Any]]] = [
            ("bcs/fix", self._fix_decls, self._fixes),
            ("bcs/mass", self._mass_decls, self._masses),
            ("recorders", self._recorder_decls, self._recorders),
            ("initial_stress", side.get("initial_stress", []),
             self._initial_stress_records),
            # One row per ``/opensees/commands`` row, global and stage.
            ("commands", self._command_decls, self._commands),
        ]
        for idx, blk in enumerate(self._stage_blocks):
            base = f"stages/stage_{idx:03d}"
            stores += [
                (f"{base}/bcs/fix", blk.fix_decls, blk.fixes),
                (f"{base}/bcs/mass", blk.mass_decls, blk.masses),
                (f"{base}/recorders", blk.recorder_decls, blk.recorders),
                (f"{base}/remove_sp", blk.remove_sp_decls, blk.remove_sps),
                (f"{base}/remove_element", blk.remove_element_decls,
                 blk.remove_elements),
                (f"{base}/update_material_stage",
                 blk.update_material_stage_decls, blk.update_material_stages),
                (f"{base}/initial_stress",
                 side.get(f"{base}/initial_stress", []),
                 blk.initial_stress_records),
                (f"{base}/activate_absorbing",
                 side.get(f"{base}/activate_absorbing", []),
                 blk.activate_absorbing_records),
                (f"{base}/update_parameter",
                 side.get(f"{base}/update_parameter", []),
                 blk.update_parameter_records),
                (f"{base}/zero_velocities",
                 side.get(f"{base}/zero_velocities", []),
                 blk.zero_velocity_records),
                (f"{base}/rayleigh", blk.rayleigh_decls, blk.rayleighs),
            ]
            stores += [
                (f"{base}/patterns/{pattern_name(pat)}/sp_holds",
                 blk.sp_hold_decls.get(pat.tag, []), pat.sp_holds)
                for pat in blk.patterns_complete if pat.sp_holds
            ]
        restored = self._decl_rows_restored
        if restored is not None:
            unknown = sorted(set(restored) - {s for s, _c, r in stores if r})
            if unknown:
                raise RuntimeError(
                    "H5Emitter._write_decls: the echoed /opensees/decls rows "
                    f"cover {unknown}, which this rewrite does not write."
                )
        rows = g.create_group("rows")
        for store, column, records in stores:
            n_records = len(records)
            if not n_records:
                continue
            if restored is not None:
                if store not in restored:
                    raise RuntimeError(
                        f"H5Emitter._write_decls: the rewrite writes "
                        f"/opensees/{store}, but the echoed /opensees/decls "
                        "holds no rows column for it."
                    )
                column = restored[store]
            if len(column) != n_records:
                raise RuntimeError(
                    f"H5Emitter._write_decls: /opensees/{store} has "
                    f"{n_records} rows but {len(column)} declaration "
                    "entries; a record reached the store without its "
                    "declaration (a Protocol call, or the bridge's "
                    "set_declaration_rows for a side-channel store)."
                )
            if _NO_DECL in column:
                bad = column.index(_NO_DECL)
                raise RuntimeError(
                    f"H5Emitter._write_decls: /opensees/{store} row {bad} "
                    f"({records[bad]!r}) was written with no "
                    "declaration open; the bridge opens one "
                    "(set_declaration) at the head of every fix, mass and "
                    "recorder loop."
                )
            rows.create_dataset(store, data=np.array(column, dtype=np.int64))

    def _write_solve_stamp(self, f: Any) -> None:
        """Persist ``/opensees@will_solve``, ``@solve_mode``,
        ``@solve_refusals``, ``@solve_refusals_flat`` and ``@requires``
        (ADR 0114 D6, schema 2.24.0).

        Written only once :meth:`set_solve_stamp` ran: ``will_solve`` is
        an ``int8`` 0/1, ``solve_mode`` a string, the other three
        vlen-string arrays, empty when there is nothing to name. All
        five fold into ``model_hash``.
        """
        stamp = self.solve_stamp()
        if stamp is None:
            return
        import h5py
        import numpy as np

        ops = self._ops_group(f)
        str_dt = h5py.string_dtype(encoding="utf-8")
        ops.attrs["will_solve"] = np.int8(1 if stamp.will_solve else 0)
        ops.attrs.create(
            "solve_mode", np.array(stamp.solve_mode, dtype=object),
            dtype=str_dt)
        assert stamp.solve_refusals_flat is not None
        for name, tokens in (
            ("solve_refusals", stamp.solve_refusals),
            ("solve_refusals_flat", stamp.solve_refusals_flat),
            ("requires", stamp.requires),
        ):
            ops.attrs.create(
                name, np.array(list(tokens), dtype=object), dtype=str_dt)

    # -- Stages (ADR 0055 Phase 2, schema 2.18.0) -------------------------

    def set_stage_records(self, stage_records: "Sequence[Any]") -> None:
        """Attach the declarative complement to the captured stage buckets.

        Side-channel (the Phase-1 ``set_initial_stress_records``
        pattern): :meth:`apeGmsh.opensees.apeSees.h5` calls this after
        ``bm.emit(self)`` drove the staged sequence.  The resolved
        per-stage emit stream was already captured in-band by the
        ``stage_open`` … ``stage_close`` bracket; ``StageRecord``
        supplies only what never flows through the resolved Protocol
        stream — ``activated_pgs``, the per-stage declarative
        ``initial_stress`` / ``activate_absorbing`` records — plus the
        cross-check fields used to fail loud on a capture/record drift.

        Raises
        ------
        RuntimeError
            On any structural mismatch between the captured buckets and
            the bridge's ``StageRecord``s (count, name order, analyze
            params, a still-open bracket, or a chain leak into the
            global ``_analysis_attrs``).  A malformed capture must
            never persist silently as a different staged program.
        """
        records = list(stage_records)
        if self._stage_current is not None:
            raise RuntimeError(
                "H5Emitter.set_stage_records: stage block "
                f"{self._stage_current.name!r} is still open — "
                "BuiltModel.emit pairs every stage_open with a "
                "stage_close before the writer runs."
            )
        if len(records) != len(self._stage_blocks):
            raise RuntimeError(
                "H5Emitter.set_stage_records: captured "
                f"{len(self._stage_blocks)} stage bracket(s) but the "
                f"bridge declared {len(records)} StageRecord(s) — the "
                "emit stream and the build diverged."
            )
        if records and (
            self._analysis_attrs or self._analyze_call is not None
        ):
            raise RuntimeError(
                "H5Emitter.set_stage_records: global analysis state is "
                "non-empty on a staged build — a stage's chain or "
                "analyze leaked outside its bracket (phantom "
                "/opensees/analysis)."
            )
        for idx, (blk, rec) in enumerate(zip(self._stage_blocks, records)):
            if blk.name != rec.name:
                raise RuntimeError(
                    "H5Emitter.set_stage_records: stage order drift — "
                    f"captured {blk.name!r} where the bridge declared "
                    f"{rec.name!r}."
                )
            expected_dt = None if rec.dt is None else float(rec.dt)
            if blk.analyze_call is None or blk.analyze_call != (
                int(rec.n_increments), expected_dt,
            ):
                raise RuntimeError(
                    f"H5Emitter.set_stage_records: stage {rec.name!r} "
                    "analyze capture mismatch (got "
                    f"{blk.analyze_call!r}, expected (steps, dt)=("
                    f"{rec.n_increments}, {expected_dt!r}))."
                )
            if blk.phantom_node_tags:
                raise NotImplementedError(
                    f"H5Emitter.set_stage_records: stage {rec.name!r} "
                    f"emitted {len(blk.phantom_node_tags)} phantom "
                    "node(s) (stage-claimed node_to_surface).  Their "
                    "coordinates have no per-stage store yet, so the "
                    "archive would be irreplayable — H5 archival of "
                    "stage-claimed phantom-node constraints is "
                    "deferred.  Use ops.tcl(path) / ops.py(path)."
                )
            # TIMs A8: ``s.profile`` reaches the archive as this stage's
            # ``profiler`` command rows; a declared bracket with none
            # captured (or rows with no bracket) is capture drift.
            profiled = any(
                c.method == "profiler" and c.stage == idx
                for c in self._commands
            )
            declared = getattr(rec, "profile", None) is not None
            if profiled != declared:
                raise RuntimeError(
                    f"H5Emitter.set_stage_records: stage {rec.name!r} "
                    f"{'declares' if declared else 'has no'} "
                    "s.profile bracket but the capture holds "
                    f"{'no' if not profiled else 'its'} profiler rows."
                )
            blk.activated_pgs = tuple(rec.activated_pgs)
            blk.initial_stress_records = tuple(rec.initial_stress_records)
            blk.activate_absorbing_records = tuple(
                rec.activate_absorbing_records
            )
            # K1-8 (schema 2.27.0): the two formerly refused verbs'
            # records, and every record's parameter tags from its calls.
            blk.update_parameter_records = tuple(
                rec.update_parameter_records)
            blk.zero_velocity_records = tuple(rec.zero_velocity_records)
            where = f"stage {rec.name!r}"
            blk.initial_stress_param_tags = _ramp_tags_by_name(
                blk.initial_stress_records, blk.ramp_tags, where)
            blk.absorbing_param_tags = _attribute_param_calls(
                where, "activate_absorbing", blk.activate_absorbing_records,
                blk.flip_calls)
            blk.update_parameter_param_tags = _attribute_param_calls(
                where, "update_parameter", blk.update_parameter_records,
                [(rank, pid) for rank, pid, _a, _v in blk.update_calls])
            _check_update_calls(
                where, blk.update_parameter_records,
                blk.update_parameter_param_tags, blk.update_calls)
            if blk.node_kinematics_calls and not blk.zero_velocity_records:
                raise RuntimeError(
                    f"H5Emitter.set_stage_records: {where} emitted "
                    f"{blk.node_kinematics_calls} setNodeVel / setNodeAccel "
                    "call(s) but declares no s.zero_velocities record — the "
                    "emit stream and the build diverged."
                )
        self._stage_records_attached = True

    def set_solve_stamp(
        self, *, will_solve: bool, solve_refusals: Sequence[str] = (),
        solve_mode: str = "flat",
        solve_refusals_flat: "Sequence[str] | None" = None,
    ) -> None:
        """Hand in the archive's solve stamp (ADR 0114 D6, schema 2.24.0).

        Side channel, not a Protocol call: ``BuiltModel.emit`` passes
        ``will_solve`` (``staged or any(Analysis)``), the archive's own
        partition mode and the ids of the solve-time gates that would
        refuse under it (``solve_refusals``) and under the flat mode
        (``solve_refusals_flat``; defaults to ``solve_refusals`` for a
        flat archive); the H5 -> H5 rewrite echoes the source's values.
        ``@requires`` is not an argument: the writer derives it from the
        verbs its program holds (:meth:`solve_stamp`). Refuses a second
        call, which would silently overwrite the first.
        """
        if self._solve_stamp_input is not None:
            raise RuntimeError(
                "H5Emitter.set_solve_stamp: the stamp is already set "
                f"({self._solve_stamp_input!r}); it is written once per "
                "emit."
            )
        if not isinstance(will_solve, bool):
            raise TypeError(
                "H5Emitter.set_solve_stamp: will_solve must be a bool, got "
                f"{type(will_solve).__name__}"
            )

        def _ids(name: str, value: "Sequence[str]") -> tuple[str, ...]:
            ids = tuple(value)
            if isinstance(value, str) or not all(
                    isinstance(i, str) and i for i in ids):
                raise TypeError(
                    f"H5Emitter.set_solve_stamp: {name} must be a sequence "
                    f"of non-empty strings, got {value!r}"
                )
            return ids

        own = _ids("solve_refusals", solve_refusals)
        flat = (
            None if solve_refusals_flat is None
            else _ids("solve_refusals_flat", solve_refusals_flat)
        )
        # Validate the mode / verdict pair once, the way the reader will:
        # a flat stamp copies its verdict, a partitioned one must bring
        # the flat verdict along.
        stamp = SolveStamp(will_solve=will_solve, solve_refusals=own,
                           solve_mode=solve_mode, solve_refusals_flat=flat)
        assert stamp.solve_refusals_flat is not None
        self._solve_stamp_input = (
            will_solve, solve_mode, own, stamp.solve_refusals_flat)

    def _decl_run(self, kind: str, tag: int) -> "tuple[int, int, int] | None":
        """The ``/opensees/decls/tags`` run that holds ``(kind, tag)``, as
        ``(first, end, decl)``, or ``None`` when no run holds it (A7)."""
        found = self._decl_tag_index.get(kind)
        if found is None:
            return None
        firsts, ends, decls = found
        i = bisect.bisect_right(firsts, tag) - 1
        if i >= 0 and tag < ends[i]:
            return firsts[i], ends[i], decls[i]
        return None

    def _decl_of_tag(self, kind: str, tag: int) -> int:
        """The declaration that owns ``(kind, tag)``, or :data:`_NO_DECL`
        when no run holds it (A7): a tagged call's ``/opensees/program``
        ``decl``, proved by the bridge's join, never guessed."""
        run = self._decl_run(kind, tag)
        return _NO_DECL if run is None else run[2]

    def restore_declarations(self, table: Any) -> None:
        """Echo a source archive's ``/opensees/decls`` and its
        ``decl_params`` (the ``from_h5 -> to_h5`` path, K1-6 / K1-7).

        ``table`` is the reader's ``DeclarationTable``. The rewrite
        replays every store in its source order, so the source's
        ``rows`` columns still describe it; the writer checks each against
        its store's row count and refuses a mismatch. The ``params`` rows
        are echoed verbatim (their JSON text included): a rewrite has no
        primitives to re-encode, and the columns are a derived view the
        source already derived.
        """
        self.set_declarations(
            [(d.key, d.family, d.name, d.synth) for d in table.decls],
            table.tags,
        )
        self._decl_rows_restored = {
            str(store): [int(v) for v in column]
            for store, column in table.rows.items()
        }
        # A7: the rewrite echoes the source's program, so it echoes the
        # source's word on whether that program's decl joins are recorded.
        self._program_decl_restored = str(table.program_decl)
        if table.params:
            self._decl_params_restored = [
                (int(decl), p.type, p.params_json,
                 "" if p.params_names is None else _json_text(list(p.params_names)),
                 p.transf_ref, p.integration_ref, p.section_ref)
                for decl, entries in sorted(table.params.items())
                for p in entries
            ]

    #: The stores a side channel fills (no Protocol call notes a row), by
    #: their path below a scope: their rows columns come from the bridge.
    _SIDE_ROW_STORES: "frozenset[str]" = frozenset(
        {"initial_stress", "activate_absorbing", "update_parameter",
         "zero_velocities"})

    def set_declaration_rows(
        self, store: str, column: "Sequence[int]",
    ) -> None:
        """Hand in the rows column of a side-channel store (K1-6):
        ``initial_stress`` or ``stages/stage_NNN/{initial_stress,
        activate_absorbing, update_parameter, zero_velocities}``, one declaration per record the side channel
        attached, in its order. The writer checks the length; a second
        call for one store, or another store, refuses."""
        if self._declarations is None:
            raise RuntimeError(
                "H5Emitter.set_declaration_rows: call set_declarations first.")
        leaf = store.rsplit("/", 1)[-1]
        if leaf not in self._SIDE_ROW_STORES or store in self._decl_side_rows:
            raise ValueError(
                f"H5Emitter.set_declaration_rows: {store!r} is not a "
                "side-channel store, or its column is already set.")
        values = [int(v) for v in column]
        if any(not 0 <= v < len(self._declarations) for v in values):
            raise IndexError(
                f"H5Emitter.set_declaration_rows: {store!r} points past the "
                "declarations.")
        self._decl_side_rows[store] = values

    def unclaimed_fix_rows(self) -> "tuple[tuple[int, int, tuple[int, ...]], ...]":
        """The ``fix`` rows no declaration claims yet, as ``(stage, tag,
        dofs)`` (``stage`` ``-1`` global, else the ``stage_NNN`` ordinal).

        A ghost replay (ADR 0027 INV-2) writes a fix outside any loop the
        bridge opens; it is claimed by its owner's replica when one
        follows, and otherwise by :meth:`claim_fix_rows` after the emit.
        """
        out: list[tuple[int, int, tuple[int, ...]]] = []
        pools = [(-1, self._fixes, self._fix_decls)] + [
            (i, blk.fixes, blk.fix_decls)
            for i, blk in enumerate(self._stage_blocks)]
        for stage, records, column in pools:
            out.extend(
                (stage, rec.tag, rec.dofs)
                for rec, decl in zip(records, column) if decl == _NO_DECL)
        return tuple(out)

    def claim_fix_rows(
        self, claims: "Mapping[tuple[int, int, tuple[int, ...]], int]",
    ) -> None:
        """Give each unclaimed ``fix`` row the declaration ``claims`` maps
        its ``(stage, tag, dofs)`` to (K1-6). A row ``claims`` lacks stays
        unclaimed, so the write still refuses it."""
        if self._declarations is None:
            raise RuntimeError(
                "H5Emitter.claim_fix_rows: call set_declarations first.")
        pools = [(-1, self._fixes, self._fix_decls)] + [
            (i, blk.fixes, blk.fix_decls)
            for i, blk in enumerate(self._stage_blocks)]
        for stage, records, column in pools:
            for row, rec in enumerate(records):
                if column[row] != _NO_DECL:
                    continue
                decl = claims.get((stage, rec.tag, rec.dofs))
                if decl is not None:
                    if not 0 <= decl < len(self._declarations):
                        raise IndexError(
                            f"H5Emitter.claim_fix_rows: declaration {decl} "
                            "is out of range.")
                    column[row] = decl

    def unclaimed_remove_sp_rows(self) -> "tuple[tuple[int, int, int], ...]":
        """The stage ``remove_sp`` rows no declaration claims yet, as
        ``(stage, node, dof)``: a ghost's replay of its owner's release
        (ADR 0027 INV-2), claimed after the emit like a ghost fix."""
        return tuple(
            (i, node, dof)
            for i, blk in enumerate(self._stage_blocks)
            for (node, dof), decl in zip(blk.remove_sps, blk.remove_sp_decls)
            if decl == _NO_DECL
        )

    def claim_remove_sp_rows(
        self, claims: "Mapping[tuple[int, int, int], int]",
    ) -> None:
        """Give each unclaimed stage ``remove_sp`` row the declaration
        ``claims`` maps its ``(stage, node, dof)`` to; a row it lacks
        stays unclaimed, so the write refuses it."""
        if self._declarations is None:
            raise RuntimeError(
                "H5Emitter.claim_remove_sp_rows: call set_declarations first.")
        for i, blk in enumerate(self._stage_blocks):
            column = blk.remove_sp_decls
            for row, (node, dof) in enumerate(blk.remove_sps):
                if column[row] != _NO_DECL:
                    continue
                decl = claims.get((i, node, dof))
                if decl is not None:
                    if not 0 <= decl < len(self._declarations):
                        raise IndexError(
                            "H5Emitter.claim_remove_sp_rows: declaration "
                            f"{decl} is out of range.")
                    column[row] = decl

    def set_declaration(self, decl: int) -> None:
        """Open declaration ``decl`` for the tagless calls that follow (K1-6).

        Called by the bridge at the head of each tagless loop (one fix,
        mass or recorder declaration's fan-out); every ``fix``, ``mass``
        and ``recorder`` call until the next one is that declaration's.
        :data:`_NO_DECL` closes it, so a replicated call the bridge does
        not attribute (a ghost replay, ADR 0027 INV-2) opens none.
        Refuses before :meth:`set_declarations` and an index out of range.
        """
        if self._declarations is None:
            raise RuntimeError(
                "H5Emitter.set_declaration: call set_declarations first; "
                "there are no declarations to open."
            )
        decl = int(decl)
        if decl != _NO_DECL and not 0 <= decl < len(self._declarations):
            raise IndexError(
                f"H5Emitter.set_declaration: declaration {decl} is out of "
                f"range (there are {len(self._declarations)})."
            )
        self._decl_open = decl

    def solve_stamp(self) -> SolveStamp | None:
        """The stamp :meth:`write` will put on ``/opensees``, or ``None``
        before :meth:`set_solve_stamp` ran.

        ``requires`` is the sorted union of ``VERBS[method].requires``
        over the methods the program holds: this emitter's own tape, or
        the echoed source program after :meth:`restore_program`. Ledger
        rows count too (the verb was emitted, even if it stored nothing),
        so the union names what the model as emitted needed.
        """
        if self._solve_stamp_input is None:
            return None
        will_solve, mode, refusals, refusals_flat = self._solve_stamp_input
        methods: "Sequence[str]" = (
            self._program_restored[1] if self._program_restored is not None
            else self._program.methods
        )
        requires = sorted({t for m in methods for t in VERBS[m].requires})
        return SolveStamp(
            will_solve=will_solve, solve_refusals=refusals,
            requires=tuple(requires), solve_mode=mode,
            solve_refusals_flat=refusals_flat,
        )

    def restore_program(
        self, runs: "Sequence[Any]", commands: "Sequence[Any]",
    ) -> None:
        """Echo a source archive's ``/opensees/program`` and
        ``/opensees/commands`` (the ``from_h5 -> to_h5`` path).

        The rewrite re-emits category-major and restores stages without
        emitting, so its own calls cannot regenerate the source's order;
        the stores it rewrites are the source's, row for row, so the
        source's runs still address them. ``runs`` and ``commands`` are
        the reader's ``ProgramRun`` / ``CommandRecordRO`` values. Refuses
        when this emitter already captured command rows, which the echo
        would duplicate.
        """
        if self._commands:
            raise RuntimeError(
                "H5Emitter.restore_program: emitter already holds "
                f"{len(self._commands)} command row(s); the echo would "
                "duplicate them. Call it on an emitter whose replay "
                "passed no commands."
            )
        methods: list[str] = []
        stores: list[str] = []
        rows: list[tuple[int, ...]] = []
        # K2b: a source below 2.29.0 wrote its reinforce ties to no store
        # (a ``ledger`` verb); the rewrite stores them, so their runs link
        # to the store as a 2.29 writer's would, in call order.
        rebar_store = VERBS["embedded_rebar"].store
        rebar_row = 0
        for run in runs:
            method, store = str(run.method), str(run.store)
            row = int(run.row)
            if method == "embedded_rebar" and not store:
                store, row = rebar_store, rebar_row
                rebar_row += int(run.count)
            if method not in methods:
                methods.append(method)
            if store and store not in stores:
                stores.append(store)
            rows.append((
                int(run.first), int(run.count), methods.index(method),
                stores.index(store) if store else _NO_ROW,
                row, int(run.stage), int(run.decl),
            ))
        self._program_restored = (
            tuple(rows), tuple(methods), tuple(stores),
            sum(r[1] for r in rows),
        )
        for rec in commands:
            self._commands.append(_CommandRecord(
                method=str(rec.method), token=str(rec.token),
                stage=int(rec.stage), args=tuple(rec.args),
                names=tuple(str(n) for n in rec.names),
            ))

    def _write_commands(self, f: Any) -> None:
        """Persist ``/opensees/commands`` (ADR 0114 R3a, schema 2.23.0).

        One row per call: ``method``, ``token`` (the OpenSees verb of a
        ``command`` row, else ``""``), ``stage`` (``-1`` global), and the
        ragged arguments ``arg_offsets`` (n + 1) over ``args`` (f8, NaN
        for a string), ``args_str`` (``""`` for a number), ``arg_kinds``
        (0 int, 1 float, 2 str) and ``arg_names`` (``""`` positional).
        Written only when a row exists, so other files keep their bytes.
        """
        if not self._commands:
            return
        import h5py
        import numpy as np

        str_dt = h5py.string_dtype(encoding="utf-8")
        g = self._ops_group(f).create_group("commands")
        recs = self._commands
        offsets = [0]
        nums: list[float] = []
        strs: list[str] = []
        kinds: list[int] = []
        names: list[str] = []
        for rec in recs:
            for value, name in zip(rec.args, rec.names):
                kind = _command_kind(rec.method, value)
                kinds.append(kind)
                names.append(name)
                if kind == _KIND_STR:
                    nums.append(float("nan"))
                    strs.append(str(value))
                else:
                    nums.append(float(value))
                    strs.append("")
            offsets.append(len(kinds))
        def _strs(values: "list[str]") -> Any:
            return np.array(values, dtype=object)

        g.create_dataset(
            "method", data=_strs([r.method for r in recs]), dtype=str_dt)
        g.create_dataset(
            "token", data=_strs([r.token for r in recs]), dtype=str_dt)
        g.create_dataset(
            "stage", data=np.asarray([r.stage for r in recs], dtype=np.int32))
        g.create_dataset(
            "arg_offsets", data=np.asarray(offsets, dtype=np.int32))
        g.create_dataset("args", data=np.asarray(nums, dtype=np.float64))
        g.create_dataset("args_str", data=_strs(strs), dtype=str_dt)
        g.create_dataset("arg_kinds", data=np.asarray(kinds, dtype=np.int8))
        g.create_dataset("arg_names", data=_strs(names), dtype=str_dt)

    def _write_program(self, f: Any) -> None:
        """Persist ``/opensees/program`` (ADR 0114 R2, schema 2.23.0).

        A compound table, one row per run (see :class:`_ProgramTape`):
        ``first`` (1-based emit index), ``count``, ``method`` (into
        ``@methods``), ``store`` (into ``@stores``, ``-1`` when the call
        wrote nothing), ``row``, ``stage`` (``-1`` global) and ``decl``
        (the ``/opensees/decls`` row that made the calls, ``-1`` when none
        is proved at emit time; A7, 2.31.0; outside ``{-1} ∪ [0, n_decls)``
        it refuses). ``@emit_count`` is the number
        of calls; the runs tile ``[1, emit_count]``. Hashed.
        """
        if self._program_restored is not None:
            rows, methods, stores, count = self._program_restored
            if count:
                rows, stores = self._drop_absent_stores(f, rows, stores)
        else:
            tape = self._program
            rows = tuple(tuple(r) for r in tape.runs)
            methods, stores = tuple(tape.methods), tuple(tape.stores)
            count = tape.count
        if not count:
            return
        n_decls = len(self._declarations) if self._declarations else 0
        for r in rows:
            if r[6] != _NO_DECL and not 0 <= r[6] < n_decls:
                raise RuntimeError(
                    f"H5Emitter._write_program: run {r!r} names declaration "
                    f"{r[6]}; /opensees/decls holds {n_decls}.")
        import h5py
        import numpy as np

        dt = np.dtype([
            ("first", np.int32), ("count", np.int32), ("method", np.int16),
            ("store", np.int16), ("row", np.int32), ("stage", np.int32),
            ("decl", np.int32),
        ])
        if len(methods) > np.iinfo(np.int16).max or (
                len(stores) > np.iinfo(np.int16).max):
            raise ValueError(
                "H5Emitter._write_program: more methods or stores than an "
                "i2 index holds."
            )
        ds = self._ops_group(f).create_dataset(
            "program", data=np.array(list(rows), dtype=dt))
        str_dt = h5py.string_dtype(encoding="utf-8")
        ds.attrs.create(
            "methods", np.array(list(methods), dtype=object), dtype=str_dt)
        ds.attrs.create(
            "stores", np.array(list(stores), dtype=object), dtype=str_dt)
        ds.attrs["emit_count"] = np.int32(count)
