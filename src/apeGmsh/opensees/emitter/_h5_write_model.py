"""The model-section writers of :class:`~apeGmsh.opensees.emitter.h5.H5Emitter`:
bcs, materials, sections, patches and fibers, transforms, element meta, patterns,
constraints, recorders, partitions (ADR 0114 S5-b). A pure move out of ``h5.py``;
the mixin holds no state.
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence
from .._internal.typed_records import (
    BeamIntegrationRecord as _BeamIntegrationRecord,
    EleLoadRecord as _EleLoadRecord,
    ElementRecord as _ElementRecord,
    EmbeddedNodeRecord as _EmbeddedNodeRecord,
    EqualDOFRecord as _EqualDOFRecord,
    FiberRecord as _FiberRecord,
    FixRecord as _FixRecord,
    LayerRecord as _LayerRecord,
    LoadRecord as _LoadRecord,
    MassRecord as _MassRecord,
    MaterialRecord as _MaterialRecord,
    PatchRecord as _PatchRecord,
    PatternRecord as _PatternRecord,
    RecorderRecord as _RecorderRecord,
    RigidDiaphragmRecord as _RigidDiaphragmRecord,
    RigidLinkRecord as _RigidLinkRecord,
    SPRecord as _SPRecord,
    SectionComplexRecord as _SectionComplexRecord,
    SectionSimpleRecord as _SectionSimpleRecord,
    TimeSeriesRecord as _TimeSeriesRecord,
    TransformRecord as _TransformRecord,
)
from ._h5_blocks import _ramp_tags_by_name
from ._h5_program import _KIND_STR, _command_kind
from ._h5_write import (
    _scan_flag,
    _set_attr,
    _set_attr_or_dataset,
    _write_param_array,
)
from ._h5_host import _H5Host


def material_name(rec: _MaterialRecord) -> str:
    """Generate the H5 group name for a material record (``Steel02_1``)."""
    return f"{rec.type_token}_{rec.tag}"


def section_name_simple(rec: _SectionSimpleRecord) -> str:
    """Generate the H5 group name for a simple section record."""
    return f"{rec.type_token}_{rec.tag}"


def section_name_complex(rec: _SectionComplexRecord) -> str:
    """Generate the H5 group name for a fiber / aggregator section."""
    return f"{rec.type_token}_{rec.tag}"


def transform_name(rec: _TransformRecord) -> str:
    """Generate the H5 group name for a transform record."""
    return f"{rec.type_token}_{rec.tag}"


def beam_integration_name(rec: _BeamIntegrationRecord) -> str:
    """Generate the H5 group name for a beam-integration rule."""
    return f"{rec.type_token}_{rec.tag}"


def element_group_name(type_token: str) -> str:
    """Group name under ``/elements``: keyed by element type token."""
    return type_token


def time_series_name(rec: _TimeSeriesRecord) -> str:
    """Generate the H5 group name for a time-series record."""
    return f"{rec.type_token}_{rec.tag}"


def pattern_name(rec: _PatternRecord) -> str:
    """Generate the H5 group name for a pattern record."""
    return f"{rec.type_token}_{rec.tag}"


def recorder_name(rec: _RecorderRecord, idx: int) -> str:
    """Generate the H5 group name for a recorder record.

    Recorders don't carry a tag in the Protocol (they're side-effects
    on the OpenSees domain); we name them positionally by their order
    in the emit stream as ``{kind}_{idx}``.
    """
    return f"{rec.kind}_{idx}"


class _H5WriteModelMixin(_H5Host):
    """Mixin: holds no state, no ``__init__``; see the module docstring."""

    def _write_bcs(self, f: Any) -> None:
        """Write ``/opensees/bcs/fix`` and ``/opensees/bcs/mass`` compound datasets.

        Both are emitted only if at least one record exists. The bridge's
        ``fix`` / ``mass`` fan-out has already resolved any ``pg=``
        targets into per-node calls, so every record's ``target_kind``
        is ``"node"`` and ``target`` is the integer tag rendered as a
        string (per the schema's compound-dataset convention).
        """
        if not self._fixes and not self._masses and not self._mass_from_model:
            return
        bcs = self._ops_group(f).create_group("bcs")
        if self._mass_from_model:
            import numpy as np
            bcs.attrs["mass_from_model"] = np.int8(1)
        if self._fixes:
            self._write_bcs_fix(bcs)
        if self._masses:
            self._write_bcs_mass(bcs)

    def _write_bcs_fix(
        self, bcs_group: Any, fixes: "list[_FixRecord] | None" = None,
    ) -> None:
        import h5py
        import numpy as np

        records = self._fixes if fixes is None else fixes
        # Compound width pinned to the GLOBAL ndf envelope (never a
        # records-local max alone) so per-stage datasets (ADR 0055
        # Phase 2) keep the same dtype across a from_h5 → to_h5
        # round-trip — a narrower stage would otherwise drift
        # ``model_hash`` via the hashed dtype tag.
        ndf = max(int(self._ndf or 0), max(len(r.dofs) for r in records))
        dt = np.dtype(
            [
                ("target_kind", h5py.string_dtype(encoding="utf-8")),
                ("target", h5py.string_dtype(encoding="utf-8")),
                ("dofs", np.int64, (ndf,)),
            ]
        )
        rows = np.empty(len(records), dtype=dt)
        for i, rec in enumerate(records):
            padded = list(rec.dofs) + [0] * (ndf - len(rec.dofs))
            rows[i] = ("node", str(rec.tag), tuple(padded))
        bcs_group.create_dataset("fix", data=rows)

    def _write_bcs_mass(
        self, bcs_group: Any, masses: "list[_MassRecord] | None" = None,
    ) -> None:
        import h5py
        import numpy as np

        records = self._masses if masses is None else masses
        ndf = max(int(self._ndf or 0), max(len(r.values) for r in records))
        dt = np.dtype(
            [
                ("target_kind", h5py.string_dtype(encoding="utf-8")),
                ("target", h5py.string_dtype(encoding="utf-8")),
                ("values", np.float64, (ndf,)),
            ]
        )
        rows = np.empty(len(records), dtype=dt)
        for i, rec in enumerate(records):
            padded = list(rec.values) + [0.0] * (ndf - len(rec.values))
            rows[i] = ("node", str(rec.tag), tuple(padded))
        bcs_group.create_dataset("mass", data=rows)

    # -- Materials -------------------------------------------------------

    def _write_materials(self, f: Any) -> None:
        if not self._uniaxial and not self._nd:
            return
        materials = self._ops_group(f).create_group("materials")
        if self._uniaxial:
            uni = materials.create_group("uniaxial")
            for rec in self._uniaxial:
                self._write_material_record(uni, rec)
        if self._nd:
            nd = materials.create_group("nd")
            for rec in self._nd:
                self._write_material_record(nd, rec)

    def _write_material_record(
        self, parent: Any, rec: _MaterialRecord,
    ) -> None:
        g = parent.create_group(material_name(rec))
        _set_attr(g, "type", rec.type_token)
        _set_attr(g, "tag", rec.tag)
        _write_param_array(g, "params", rec.params)

    # -- Sections --------------------------------------------------------

    def _write_sections(self, f: Any) -> None:
        if not self._sections_simple and not self._sections_complex:
            return
        sections = self._ops_group(f).create_group("sections")
        for rec_simple in self._sections_simple:
            self._write_section_simple(sections, rec_simple)
        for rec_complex in self._sections_complex:
            self._write_section_complex(sections, rec_complex)

    def _write_section_simple(
        self, parent: Any, rec: _SectionSimpleRecord,
    ) -> None:
        g = parent.create_group(section_name_simple(rec))
        _set_attr(g, "type", rec.type_token)
        _set_attr(g, "tag", rec.tag)
        _write_param_array(g, "params", rec.params)

    def _write_section_complex(
        self, parent: Any, rec: _SectionComplexRecord,
    ) -> None:
        g = parent.create_group(section_name_complex(rec))
        _set_attr(g, "type", rec.type_token)
        _set_attr(g, "tag", rec.tag)
        _write_param_array(g, "params", rec.params)
        if rec.patches:
            self._write_patches(g, rec.patches)
        if rec.fibers:
            self._write_fibers(g, rec.fibers)
        if rec.layers:
            self._write_layers(g, rec.layers)

    def _write_patches(
        self, sec_group: Any, patches: list[_PatchRecord],
    ) -> None:
        """Write the ``patches`` compound dataset.

        Patch arg layout (per OpenSees Tcl manual):

        * ``rect``: ``matTag numSubdivY numSubdivZ yI zI yJ zJ`` → 4 coords
        * ``quad``: ``matTag numSubdivIJ numSubdivJK yI zI yJ zJ yK zK yL zL`` → 8 coords
        * ``circ``: ``matTag numSubdivCirc numSubdivRad yC zC intRad extRad startAng endAng`` → 6 coords (padded to 8)

        Unknown kinds: emit row with all-NaN coords and a
        ``__deviation__`` sibling attr noting the kind.
        """
        import h5py
        import numpy as np

        dt = np.dtype(
            [
                ("kind", h5py.string_dtype(encoding="utf-8")),
                ("material_ref", h5py.string_dtype(encoding="utf-8")),
                ("ny", np.int64),
                ("nz", np.int64),
                ("coords", np.float64, (8,)),
            ]
        )
        rows = np.empty(len(patches), dtype=dt)
        unknown_kinds: list[str] = []
        for i, p in enumerate(patches):
            mat_tag, ny, nz, coords = self._decode_patch(p, unknown_kinds)
            rows[i] = (
                p.kind, self._material_ref(mat_tag),
                ny, nz, coords,
            )
        sec_group.create_dataset("patches", data=rows)
        if unknown_kinds:
            _set_attr(
                sec_group, "__deviation_patches__",
                f"unknown patch kinds with truncated coords: "
                f"{','.join(sorted(set(unknown_kinds)))}",
            )

    def _decode_patch(
        self, p: _PatchRecord, unknown_kinds: list[str],
    ) -> tuple[int, int, int, tuple[float, ...]]:
        """Return ``(mat_tag, ny, nz, coords_padded_to_8)`` for one patch."""
        args = list(p.args)
        # First three args after kind are: matTag, n1, n2 — for all
        # standard kinds (rect / quad / circ).
        if len(args) < 3:
            unknown_kinds.append(p.kind)
            return (0, 0, 0, (float("nan"),) * 8)
        mat_tag = int(args[0])
        ny = int(args[1])
        nz = int(args[2])
        coord_args = [float(x) for x in args[3:]]
        if p.kind not in ("rect", "quad", "circ"):
            unknown_kinds.append(p.kind)
        # Pad with NaN to 8.
        padded = coord_args + [float("nan")] * (8 - len(coord_args))
        return (mat_tag, ny, nz, tuple(padded[:8]))

    def _write_fibers(
        self, sec_group: Any, fibers: list[_FiberRecord],
    ) -> None:
        import h5py
        import numpy as np

        dt = np.dtype(
            [
                ("y", np.float64),
                ("z", np.float64),
                ("area", np.float64),
                ("material_ref", h5py.string_dtype(encoding="utf-8")),
            ]
        )
        rows = np.empty(len(fibers), dtype=dt)
        for i, fiber in enumerate(fibers):
            rows[i] = (
                fiber.y, fiber.z, fiber.area,
                self._material_ref(fiber.mat_tag),
            )
        sec_group.create_dataset("fibers", data=rows)

    def _write_layers(
        self, sec_group: Any, layers: list[_LayerRecord],
    ) -> None:
        """Write the ``layers`` compound dataset.

        Layer arg layout (per OpenSees Tcl manual):

        * ``straight``: ``matTag numBars area yStart zStart yEnd zEnd`` → 4 line floats
        * ``circ``: ``matTag numBars area yC zC radius startAng endAng`` → 6 line floats

        ``line`` is sized to 6 (schema 1.1 widening from the v1.0 [4]
        spec). Straight layers pad the trailing 2 with NaN.
        """
        import h5py
        import numpy as np

        dt = np.dtype(
            [
                ("kind", h5py.string_dtype(encoding="utf-8")),
                ("material_ref", h5py.string_dtype(encoding="utf-8")),
                ("n_bars", np.int64),
                ("area", np.float64),
                ("line", np.float64, (6,)),
            ]
        )
        rows = np.empty(len(layers), dtype=dt)
        unknown_kinds: list[str] = []
        for i, lyr in enumerate(layers):
            mat_tag, n_bars, area, line = self._decode_layer(
                lyr, unknown_kinds,
            )
            rows[i] = (
                lyr.kind, self._material_ref(mat_tag),
                n_bars, area, line,
            )
        sec_group.create_dataset("layers", data=rows)
        if unknown_kinds:
            _set_attr(
                sec_group, "__deviation_layers__",
                f"unknown layer kinds: {','.join(sorted(set(unknown_kinds)))}",
            )

    def _decode_layer(
        self, lyr: _LayerRecord, unknown_kinds: list[str],
    ) -> tuple[int, int, float, tuple[float, ...]]:
        args = list(lyr.args)
        if len(args) < 3:
            unknown_kinds.append(lyr.kind)
            return (0, 0, 0.0, (float("nan"),) * 6)
        mat_tag = int(args[0])
        n_bars = int(args[1])
        area = float(args[2])
        line_args = [float(x) for x in args[3:]]
        if lyr.kind not in ("straight", "circ"):
            unknown_kinds.append(lyr.kind)
        padded = line_args + [float("nan")] * (6 - len(line_args))
        return (mat_tag, n_bars, area, tuple(padded[:6]))

    def _material_ref(self, mat_tag: int) -> str:
        """Resolve a material tag to its ``/opensees/materials/...`` HDF5 path.

        Uniaxial tags shadow nd tags in the OpenSees domain (separate
        namespaces), so we check uniaxial first, then nd. Returns the
        empty string if the tag is unknown — viewers should treat this
        as a missing reference and degrade.
        """
        for rec in self._uniaxial:
            if rec.tag == mat_tag:
                return f"/opensees/materials/uniaxial/{material_name(rec)}"
        for rec in self._nd:
            if rec.tag == mat_tag:
                return f"/opensees/materials/nd/{material_name(rec)}"
        return ""

    # -- Transforms ------------------------------------------------------

    def _write_transforms(self, f: Any) -> None:
        """Write one ``/opensees/transforms/{type}_{tag}/`` group per geomTransf call.

        See module docstring for the schema deviation rationale: the
        H5 emitter sees one call per emitted ``geomTransf`` line — for
        orientation-driven transforms the bridge fans these out across
        distinct per-element vecxz, but the streaming Protocol does
        not surface the spec boundary that would let us aggregate them
        into the schema's per-spec grouping.
        """
        if not self._transforms:
            return
        import numpy as np

        transforms = self._ops_group(f).create_group("transforms")
        for rec in self._transforms:
            g = transforms.create_group(transform_name(rec))
            _set_attr(g, "type", rec.type_token)
            _set_attr(g, "tag", rec.tag)
            # Each emitted geomTransf line corresponds to a single
            # vecxz; per_element_vecxz is therefore (1, 3).
            vec_arr = np.asarray([list(rec.vec)], dtype=np.float64)
            g.create_dataset("per_element_vecxz", data=vec_arr)
            if rec.jnt_offset:
                # opensees 2.35.0: the line's -jntOffset tail, (1, 6) in
                # 3-D, (1, 4) in 2-D; absent when the line has none.
                g.create_dataset(
                    "jnt_offset",
                    data=np.asarray([list(rec.jnt_offset)], dtype=np.float64))
            tag_arr = np.asarray([rec.tag], dtype=np.int64)
            g.create_dataset("per_element_emitted_tag", data=tag_arr)
            _set_attr(g, "__deviation__", "per-emitted-call grouping")

    # -- Beam integration -----------------------------------------------

    def _write_beam_integration(self, f: Any) -> None:
        """Write ``/opensees/beam_integration/{type}_{tag}/`` groups."""
        if not self._beam_integrations:
            return
        bi = self._ops_group(f).create_group("beam_integration")
        for rec in self._beam_integrations:
            g = bi.create_group(beam_integration_name(rec))
            _set_attr(g, "type", rec.type_token)
            _set_attr(g, "tag", rec.tag)
            _write_param_array(g, "params", rec.args)

    # -- Element metadata (OpenSees-specific args) ----------------------

    def _write_element_meta(self, f: Any) -> None:
        """Write ``/opensees/element_meta/{type_token}/`` groups.

        Phase 8.5 split element data across two zones: the broker
        owns the neutral ``/elements/{gmsh_alias}/`` group (ids +
        connectivity), and the bridge owns the OpenSees-specific
        parameter tail (positional args, string flags,
        cross-references) under ``/opensees/element_meta/{type_token}/``.
        Phase 8.6 added ``fem_eids`` parallel to ``ids`` so a consumer
        can map an OpenSees tag back to its FEM element id.

        For each OpenSees element type token (``forceBeamColumn``,
        ``FourNodeTetrahedron``, …) seen during emit, this method
        creates one group with:

        * ``ids`` — int64 ``(N,)`` of OpenSees element tags.
        * ``fem_eids`` — int64 ``(N,)`` of FEM element ids the
          bridge fanned each OpenSees tag out from.  Entries are
          ``-1`` (:data:`MISSING_FEM_ELEMENT_ID`) for records emitted
          outside a PG fan-out: test scenarios and the bridge-minted
          rows (node-pair / interface ``zeroLength``, auto-emitted
          rebar ``CorotTruss``, couplings).  ``architecture/h5-schema.md``
          "Bridge-minted rows" documents the sentinel and the rebar
          join key (#1290).
        * ``args`` — float64 ``(N, max_tail)`` of positional args
          after the connectivity prefix; NaN in slots that hold a
          string token.
        * ``args_str`` — vlen-utf8 ``(N, max_tail)`` of string
          tokens at the matching slot; empty string elsewhere.
          Present only if at least one slot is a string.
        """
        if not self._elements:
            return
        import numpy as np

        # Bin by type token; preserve per-type insertion order.  Each
        # bin entry pairs a record with its index in ``self._elements``
        # so the parallel ``self._element_ranks`` row (schema 2.10.0
        # ``partition_ids`` column) can be looked up by index.
        bins: dict[str, list[tuple[int, _ElementRecord]]] = {}
        for i, rec in enumerate(self._elements):
            bins.setdefault(rec.type_token, []).append((i, rec))

        # Safety net: if the partition tracker fell out of sync with
        # the element list, fall back to all-sentinel.  The element()
        # method is the only writer for both lists, so they should be
        # the same length — this guards against unrelated bridge
        # refactors that might bypass element() (e.g. a future
        # declarative front-door that forgets to append).
        ranks_ok = len(self._element_ranks) == len(self._elements)

        parent = self._ops_group(f).create_group("element_meta")
        for type_token, indexed_recs in bins.items():
            recs = [rec for _, rec in indexed_recs]
            g = parent.create_group(element_group_name(type_token))
            _set_attr(g, "type", type_token)
            g.create_dataset(
                "ids",
                data=np.asarray([r.tag for r in recs], dtype=np.int64),
            )
            # Phase 8.6: parallel array of FEM element ids — the
            # (fem_eid, ops_tag) mapping the master plan put under
            # `/opensees/tag_map/`.  Entries are
            # `MISSING_FEM_ELEMENT_ID` (`-1`) for records emitted
            # outside a PG fan-out (standalone test calls and the
            # bridge-minted rows, h5-schema.md "Bridge-minted rows").
            g.create_dataset(
                "fem_eids",
                data=np.asarray([r.fem_eid for r in recs], dtype=np.int64),
            )
            # Schema 2.10.0 (ADR 0027): parallel array of partition
            # ranks — one int64 per row of ``ids`` carrying the rank
            # the element was emitted under, or ``-1`` for elements
            # emitted outside a ``partition_open`` / ``partition_close``
            # bracket.  Always emitted (additive minor — old 2.9.x
            # readers ignore the new column).
            if ranks_ok:
                rank_row = [self._element_ranks[i] for i, _ in indexed_recs]
            else:
                rank_row = [-1] * len(indexed_recs)
            g.create_dataset(
                "partition_ids",
                data=np.asarray(rank_row, dtype=np.int64),
            )
            self._write_inline_connectivity(g, recs)
            self._write_element_argstack(g, recs)

    def _write_inline_connectivity(
        self, g: Any, recs: list[_ElementRecord],
    ) -> None:
        """Write ``inline_connectivity`` for node-pair rows (ADR 0049).

        A node-pair element (``ops.element.ZeroLength(nodes=...)`` etc.)
        carries ``fem_eid == MISSING_FEM_ELEMENT_ID`` (``-1``) and has no
        gmsh cell in the neutral ``/elements`` zone, so its endpoint tags
        cannot be sourced from there on re-emit.  Persist them here, one
        ragged row per element (empty for ordinary PG-fanned rows whose
        connectivity is recoverable from the FEM).  Written **only** when a
        type group has ≥1 node-pair row, so PG-only models stay
        byte-identical and their ``model_hash`` is unperturbed.
        """
        import h5py
        import numpy as np

        if not any(r.fem_eid < 0 and r.connectivity for r in recs):
            return
        vlen = h5py.vlen_dtype(np.int64)
        rows = np.empty(len(recs), dtype=object)
        for i, r in enumerate(recs):
            if r.fem_eid < 0 and r.connectivity:
                rows[i] = np.asarray(
                    [int(c) for c in r.connectivity], dtype=np.int64,
                )
            else:
                rows[i] = np.empty(0, dtype=np.int64)
        g.create_dataset("inline_connectivity", data=rows, dtype=vlen)

    def _write_element_argstack(
        self, g: Any, recs: list[_ElementRecord],
    ) -> None:
        """Write the element ``args`` / ``args_str`` array pair.

        The first ``len(connectivity)`` positional args of every element
        are the node tags (recoverable from the broker's
        ``/elements/{gmsh_alias}/connectivity`` zone for PG-fanned
        elements, or from ``inline_connectivity`` for node-pair elements);
        we drop them and record only the tail (parameter / cross-reference
        payload).  The drop is **per record** (``args[len(r.connectivity):]``)
        rather than a single max-arity over the type group, so a node-pair
        row whose connectivity length differs from its PG siblings still
        slices its own tail correctly (ADR 0049).
        """
        import h5py
        import numpy as np

        max_tail = max(len(r.args) - len(r.connectivity) for r in recs)
        if max_tail <= 0:
            return

        arg_nums = np.full((len(recs), max_tail), float("nan"), dtype=np.float64)
        arg_strs: list[list[str]] = []
        any_str = False
        for i, r in enumerate(recs):
            tail = r.args[len(r.connectivity):]
            row_strs: list[str] = []
            for j, v in enumerate(tail):
                if isinstance(v, str):
                    row_strs.append(v)
                    any_str = True
                else:
                    arg_nums[i, j] = float(v)
                    row_strs.append("")
            row_strs.extend([""] * (max_tail - len(row_strs)))
            arg_strs.append(row_strs)

        g.create_dataset("args", data=arg_nums)
        if any_str:
            g.create_dataset(
                "args_str",
                data=np.asarray(arg_strs, dtype=object),
                dtype=h5py.string_dtype(encoding="utf-8"),
            )

    # -- Time series -----------------------------------------------------

    def _write_time_series(self, f: Any) -> None:
        if not self._time_series:
            return
        ts = self._ops_group(f).create_group("time_series")
        for rec in self._time_series:
            g = ts.create_group(time_series_name(rec))
            _set_attr(g, "type", rec.type_token)
            _set_attr(g, "tag", rec.tag)
            _write_param_array(g, "params", rec.args)
            # Schema's ``time`` and ``values`` arrays would require
            # vocabulary-aware decoding of args (e.g. -dt / -filePath /
            # inline values). The H5 emitter records the raw args; a
            # vocabulary-aware reader (or a future schema-bumped
            # primitive-side _emit) can populate time/values.

    # -- Patterns --------------------------------------------------------

    def _write_patterns(self, f: Any) -> None:
        # Flush any still-open pattern (defensive; bridge always closes).
        self._flush_open_pattern()
        if not self._patterns_complete:
            return
        patterns = self._ops_group(f).create_group("patterns")
        for rec in self._patterns_complete:
            self._write_pattern_record(patterns, rec)

    def _write_pattern_record(self, parent: Any, rec: _PatternRecord) -> None:
        """Write one pattern group under ``parent`` (global zone or a
        ``stage_NNN`` group's ``patterns`` sub-group — ADR 0055 Phase 2)."""
        g = parent.create_group(pattern_name(rec))
        _set_attr(g, "type", rec.type_token)
        _set_attr(g, "tag", rec.tag)
        series_ref = self._pattern_series_ref(rec)
        if series_ref is not None:
            _set_attr(g, "series_ref", series_ref)
        # Direction is a meaningful pattern-level attr for
        # UniformExcitation; surface it explicitly.
        if rec.type_token == "UniformExcitation" and rec.args:
            _set_attr(g, "direction", int(rec.args[0]))
        _write_param_array(g, "params", rec.args)

        if rec.loads:
            self._write_pattern_loads(g, rec.loads)
        if rec.sps:
            self._write_pattern_sps(g, rec.sps)
        if rec.ele_loads:
            self._write_pattern_ele_loads(g, rec.ele_loads)
        if rec.sp_holds:
            # ADR 0055 Phase 2 / ADR 0052: stage HOLD lines — (node,
            # dof) pairs; the ``sp <node> <dof> [nodeDisp …] -const``
            # form re-renders on the deck side at replay.
            import numpy as np
            g.create_dataset(
                "sp_holds",
                data=np.asarray(
                    [[int(n), int(d)] for n, d in rec.sp_holds],
                    dtype=np.int64,
                ).reshape(len(rec.sp_holds), 2),
            )

    def _pattern_series_ref(self, rec: _PatternRecord) -> str | None:
        """Resolve the time-series tag a pattern references → an HDF5 path.

        For ``Plain``: ``args = (ts_tag, ...)``.
        For ``UniformExcitation``: ``args = (direction, "-accel", ts_tag)``
        (per the typed primitive's emit shape). Search ``args`` for an
        int that matches a known time-series tag; the first match wins.
        Returns ``None`` if no match.
        """
        if rec.type_token == "Plain" and rec.args:
            ts_tag = rec.args[0]
            return self._time_series_ref_for_tag(ts_tag)
        if rec.type_token == "UniformExcitation":
            for v in rec.args:
                if isinstance(v, int) and not isinstance(v, bool):
                    candidate = self._time_series_ref_for_tag(v)
                    if candidate is not None and candidate.endswith(
                        f"_{v}"
                    ) and v != rec.args[0]:
                        return candidate
            # Fallback: scan integer-coercible args other than
            # direction (args[0]) for a known time-series tag.
            for v in rec.args[1:]:
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    ref = self._time_series_ref_for_tag(int(v))
                    if ref is not None:
                        return ref
        return None

    def _time_series_ref_for_tag(self, tag: int | float | str) -> str | None:
        try:
            tag_int = int(tag)
        except (TypeError, ValueError):
            return None
        for rec in self._time_series:
            if rec.tag == tag_int:
                return f"/opensees/time_series/{time_series_name(rec)}"
        return None

    def _write_pattern_loads(
        self, g: Any, loads: list[_LoadRecord],
    ) -> None:
        import h5py
        import numpy as np

        ndf = max(int(self._ndf or 0), max(len(r.forces) for r in loads))
        dt = np.dtype(
            [
                ("target_kind", h5py.string_dtype(encoding="utf-8")),
                ("target", h5py.string_dtype(encoding="utf-8")),
                ("forces", np.float64, (ndf,)),
            ]
        )
        rows = np.empty(len(loads), dtype=dt)
        for i, r in enumerate(loads):
            padded = list(r.forces) + [float("nan")] * (ndf - len(r.forces))
            rows[i] = ("node", str(r.target), tuple(padded))
        g.create_dataset("loads", data=rows)

    def _write_pattern_sps(
        self, g: Any, sps: list[_SPRecord],
    ) -> None:
        import h5py
        import numpy as np

        dt = np.dtype(
            [
                ("target_kind", h5py.string_dtype(encoding="utf-8")),
                ("target", h5py.string_dtype(encoding="utf-8")),
                ("dof", np.int64),
                ("value", np.float64),
            ]
        )
        rows = np.empty(len(sps), dtype=dt)
        for i, r in enumerate(sps):
            rows[i] = ("node", str(r.target), r.dof, r.value)
        g.create_dataset("sps", data=rows)

    # -- Recorders -------------------------------------------------------

    def _write_regions(self, f: Any) -> None:
        """Persist ``/opensees/regions/region_NN`` groups (one per region call).

        Each group carries the integer ``tag`` attribute plus a
        ``params`` dataset capturing the raw OpenSees flag tail
        (``-node n1 n2 ...``, ``-ele e1 e2 ...``, ``-eleOnly``, etc.).
        Empty when no region was emitted — auto-emit from the recorder
        fan-out (MPCO ``pg=`` form) is the only producer today.
        """
        if not self._regions:
            return
        regions = self._ops_group(f).create_group("regions")
        for idx, rec in enumerate(self._regions):
            g = regions.create_group(f"region_{idx:03d}")
            _set_attr(g, "tag", rec.tag)
            _write_param_array(g, "params", rec.args)

    def _write_dampings(self, f: Any) -> None:
        """Persist ``/opensees/dampings/damping_NNN`` groups (ADR 0053 D3b).

        One group per :meth:`damping` call carrying the ``type`` token, the
        integer ``tag`` attribute, and a ``params`` dataset with the raw
        OpenSees argument tail (ζ / freq / β, ``-activateTime`` /
        ``-deactivateTime`` / ``-factor $tsTag``). Empty when no damping
        object was emitted. Folds into ``model_hash`` (authored state).
        """
        if not self._dampings:
            return
        dampings = self._ops_group(f).create_group("dampings")
        for idx, rec in enumerate(self._dampings):
            g = dampings.create_group(f"damping_{idx:03d}")
            _set_attr(g, "type", rec.type_token)
            _set_attr(g, "tag", rec.tag)
            _write_param_array(g, "params", rec.args)

    def set_initial_stress_records(
        self,
        records: "Iterable[Any]",
        *,
        param_tags: "Sequence[tuple[int, int, int]] | None" = None,
    ) -> None:
        """Buffer the bridge's global ``InitialStressRecord``s for archival.

        Side-channel (ADR 0055 Phase 1): :meth:`apeGmsh.opensees.apeSees.h5`
        calls this after ``bm.emit(self)`` and before the compose write, so
        :meth:`_write_initial_stress` can persist the declarative records.
        Each record's ``param_tags`` (schema 2.27.0, K1-8) are the tags its
        ``step_hook_ramp`` call handed in, by name; a record no ramp named
        refuses. The echo path (``OpenSeesModel._compose_h5``) has no
        ramps and passes the stored ``param_tags`` instead, one per record.
        """
        self._initial_stress_records = list(records)
        if param_tags is None:
            self._initial_stress_param_tags = _ramp_tags_by_name(
                self._initial_stress_records, self._ramp_tags,
                "the global initial stress")
            return
        tags = tuple(
            (int(xx), int(yy), int(zz)) for xx, yy, zz in param_tags)
        if len(tags) != len(self._initial_stress_records):
            raise RuntimeError(
                "H5Emitter.set_initial_stress_records: "
                f"{len(self._initial_stress_records)} record(s) but "
                f"{len(tags)} param_tags entries; one per record.")
        self._initial_stress_param_tags = tags

    def _write_initial_stress(self, f: Any) -> None:
        """Persist ``/opensees/initial_stress/stress_NNN`` groups (ADR 0055).

        One group per global ``ops.initial_stress(...)`` record carrying the
        pre-resolve declarative field set: ``name`` + ``sigma_xx/yy/zz`` +
        ``ramp_steps`` + ``lambda_install`` scalar attrs, and EITHER a ``pg``
        attr (PG-targeted) XOR an ``elements`` int64 dataset (explicit element
        list), plus the ``param_tags`` int64 (3,) attr (schema 2.27.0, K1-8):
        the XX / YY / ZZ parameter tags the build planned.  No rendered
        ramp proc — replay re-runs :func:`write_planned_ramps` /
        :func:`emit_initial_stress_addtoparameter` over the stored tags to
        regenerate the deck.  Empty when no global initial-stress record
        exists (vanilla files stay byte-identical).  Folds into
        ``model_hash`` (authored state).
        """
        if not self._initial_stress_records:
            return
        grp = self._ops_group(f).create_group("initial_stress")
        self._write_initial_stress_records_into(
            grp, self._initial_stress_records,
            self._initial_stress_param_tags,
        )

    def _write_initial_stress_records_into(
        self, grp: Any, records: "Sequence[Any]",
        param_tags: "Sequence[tuple[int, int, int]]",
    ) -> None:
        """Write ``stress_NNN`` declarative sub-groups under ``grp``.

        Shared by the global ``/opensees/initial_stress`` zone (Phase 1)
        and each stage's ``initial_stress`` sub-group (Phase 2) — same
        pre-resolve field set, same pg-XOR-elements discriminant, and one
        ``param_tags`` entry per record (a missing one refuses: a 2.27.0
        archive never leaves its replay a tag to mint).
        """
        import numpy as np

        if len(param_tags) != len(records):
            raise RuntimeError(
                f"H5Emitter: {grp.name} holds {len(records)} initial "
                f"stress record(s) but {len(param_tags)} param_tags "
                "entries; every record stores its ramp tags (schema "
                "2.27.0)."
            )
        for idx, (rec, tags) in enumerate(zip(records, param_tags)):
            g = grp.create_group(f"stress_{idx:03d}")
            _set_attr(g, "name", rec.name)
            _set_attr(g, "sigma_xx", float(rec.sigma_xx))
            _set_attr(g, "sigma_yy", float(rec.sigma_yy))
            _set_attr(g, "sigma_zz", float(rec.sigma_zz))
            _set_attr(g, "ramp_steps", int(rec.ramp_steps))
            _set_attr(g, "lambda_install", float(rec.lambda_install))
            # pg XOR elements — store whichever is present so the read side
            # reconstructs the same target discriminant (never both).
            if rec.elements is not None:
                g.create_dataset(
                    "elements",
                    data=np.asarray([int(e) for e in rec.elements], dtype=np.int64),
                )
            else:
                _set_attr(g, "pg", rec.pg)
            _set_attr(g, "param_tags", [int(t) for t in tags])

    def _write_recorders(self, f: Any) -> None:
        if not self._recorders:
            return
        recorders = self._ops_group(f).create_group("recorders")
        for idx, rec in enumerate(self._recorders):
            self._write_recorder_record(recorders, rec, idx)

    def _write_recorder_record(
        self, parent: Any, rec: _RecorderRecord, idx: int,
    ) -> None:
        """Write one recorder group under ``parent`` (global zone or a
        ``stage_NNN`` group's ``recorders`` sub-group — ADR 0055 Phase 2)."""
        g = parent.create_group(recorder_name(rec, idx))
        # Schema 2.3.0: every record carries a ``kind`` attr.
        # ``typed`` — Node / Element / MPCO primitive (1:1 OpenSees).
        # ``declared`` — fan-out call from a ``RecorderDeclaration``
        # (carries the original declaration metadata alongside).
        _set_attr(g, "kind", "declared" if rec.decl_context else "typed")
        _set_attr(g, "type", rec.kind)
        # Surface the -file flag's value as an explicit attr; it's
        # the most-used identifier across recorders.
        file_path = _scan_flag(rec.args, "-file")
        if file_path is not None:
            _set_attr(g, "file", file_path)
        _write_param_array(g, "params", rec.args)
        # Schema 2.3.0: declaration metadata for ``declared`` records.
        ctx = rec.decl_context
        if ctx is not None:
            _set_attr(g, "declaration_name", ctx.declaration_name)
            _set_attr(g, "record_name", ctx.record_name)
            _set_attr(g, "category", ctx.category)
            _set_attr(g, "components", ctx.components)
            _set_attr(g, "raw", ctx.raw)
            _set_attr(g, "pg", ctx.pg)
            _set_attr(g, "label", ctx.label)
            _set_attr(g, "selection", ctx.selection)
            if ctx.ids is not None:
                # Past the attribute limit, an int64 dataset of the
                # same name (opensees 2.34.0).
                _set_attr_or_dataset(g, "ids", ctx.ids)
            _set_attr(g, "dt", ctx.dt)
            _set_attr(g, "n_steps", ctx.n_steps)
            _set_attr(g, "file_root", ctx.file_root)

    # -- MP constraints (Phase 7b, ADR 0022, schema 2.7.0) ---------------

    def _write_constraints(self, f: Any) -> None:
        """Write ``/opensees/constraints/`` with one compound dataset per
        MP constraint kind, plus a flat ``phantom_node_tags`` array.

        Empty groups are skipped — the group is created only if at least
        one MP constraint or phantom node was emitted.  Per ADR 0023
        the schema bump from 2.6 → 2.7 is additive: 2.6.x readers
        ignore the new group.
        """
        has_any = (
            self._equal_dofs or self._rigid_links
            or self._rigid_diaphragms or self._embedded_nodes
            or self._phantom_node_tags or self._embedded_rebars
        )
        if not has_any:
            return

        import numpy as np

        constraints = self._ops_group(f).create_group("constraints")

        if self._phantom_node_tags:
            constraints.create_dataset(
                "phantom_node_tags",
                data=np.asarray(self._phantom_node_tags, dtype=np.int64),
            )

        if self._equal_dofs:
            self._write_constraints_equal_dof(constraints)
        if self._rigid_links:
            self._write_constraints_rigid_link(constraints)
        if self._rigid_diaphragms:
            self._write_constraints_rigid_diaphragm(constraints)
        if self._embedded_nodes:
            self._write_constraints_embedded_node(constraints)
        if self._embedded_rebars:
            self._write_constraints_embedded_rebar(constraints)

    def _write_constraints_embedded_rebar(self, parent: Any) -> None:
        """Persist ``/opensees/constraints/embeddedRebar`` (K2b, 2.29.0).

        One row per tie: ``ele_tag`` (int64), ``name`` and the ragged
        arguments ``arg_offsets`` (n + 1) over ``args`` (f8, NaN for a
        string), ``args_str`` (``""`` for a number) and ``arg_kinds``
        (0 int, 1 float, 2 str), the ``/opensees/commands`` encoding.
        Written only when a tie exists, so other files keep their bytes.
        """
        import h5py
        import numpy as np

        str_dt = h5py.string_dtype(encoding="utf-8")
        recs = self._embedded_rebars
        offsets = [0]
        nums: list[float] = []
        strs: list[str] = []
        kinds: list[int] = []
        for rec in recs:
            for value in rec.args:
                kind = _command_kind("embedded_rebar", value)
                kinds.append(kind)
                if kind == _KIND_STR:
                    nums.append(float("nan"))
                    strs.append(str(value))
                else:
                    nums.append(float(value))
                    strs.append("")
            offsets.append(len(kinds))
        g = parent.create_group("embeddedRebar")
        g.create_dataset(
            "ele_tag",
            data=np.asarray([r.ele_tag for r in recs], dtype=np.int64))
        g.create_dataset(
            "name", data=np.array([r.name for r in recs], dtype=object),
            dtype=str_dt)
        g.create_dataset(
            "arg_offsets", data=np.asarray(offsets, dtype=np.int32))
        g.create_dataset("args", data=np.asarray(nums, dtype=np.float64))
        g.create_dataset(
            "args_str", data=np.array(strs, dtype=object), dtype=str_dt)
        g.create_dataset("arg_kinds", data=np.asarray(kinds, dtype=np.int8))

    def _write_constraints_equal_dof(
        self, parent: Any,
        records: "list[_EqualDOFRecord] | None" = None,
    ) -> None:
        import h5py
        import numpy as np

        recs = self._equal_dofs if records is None else records
        ndf = max(
            int(self._ndf or 0),
            max(len(r.dofs) for r in recs),
        )
        # ndf must be at least 1 — even a 1-DOF equalDOF needs a slot.
        ndf = max(ndf, 1)
        dt = np.dtype(
            [
                ("master", np.int64),
                ("slave", np.int64),
                ("dofs", np.int64, (ndf,)),
                ("name", h5py.string_dtype(encoding="utf-8")),
            ]
        )
        rows = np.empty(len(recs), dtype=dt)
        for i, rec in enumerate(recs):
            padded = list(rec.dofs) + [0] * (ndf - len(rec.dofs))
            rows[i] = (rec.master, rec.slave, tuple(padded), rec.name)
        parent.create_dataset("equalDOF", data=rows)

    def _write_constraints_rigid_link(
        self, parent: Any,
        records: "list[_RigidLinkRecord] | None" = None,
    ) -> None:
        import h5py
        import numpy as np

        recs = self._rigid_links if records is None else records
        dt = np.dtype(
            [
                ("kind", h5py.string_dtype(encoding="utf-8")),
                ("master", np.int64),
                ("slave", np.int64),
                ("name", h5py.string_dtype(encoding="utf-8")),
            ]
        )
        rows = np.empty(len(recs), dtype=dt)
        for i, rec in enumerate(recs):
            rows[i] = (rec.kind, rec.master, rec.slave, rec.name)
        parent.create_dataset("rigidLink", data=rows)

    def _write_constraints_rigid_diaphragm(
        self, parent: Any,
        records: "list[_RigidDiaphragmRecord] | None" = None,
    ) -> None:
        import h5py
        import numpy as np

        recs = self._rigid_diaphragms if records is None else records
        max_slaves = max(len(r.slaves) for r in recs)
        max_slaves = max(max_slaves, 1)
        dt = np.dtype(
            [
                ("perp_dir", np.int32),
                ("master", np.int64),
                ("slaves", np.int64, (max_slaves,)),
                ("n_slaves", np.int32),
                ("name", h5py.string_dtype(encoding="utf-8")),
            ]
        )
        rows = np.empty(len(recs), dtype=dt)
        for i, rec in enumerate(recs):
            padded = (
                list(rec.slaves)
                + [0] * (max_slaves - len(rec.slaves))
            )
            rows[i] = (
                rec.perp_dir, rec.master,
                tuple(padded), len(rec.slaves), rec.name,
            )
        parent.create_dataset("rigidDiaphragm", data=rows)

    def _write_constraints_embedded_node(
        self, parent: Any,
        records: "list[_EmbeddedNodeRecord] | None" = None,
    ) -> None:
        import h5py
        import numpy as np

        recs = self._embedded_nodes if records is None else records
        max_args = max(len(r.args) for r in recs)
        max_args = max(max_args, 1)
        # Schema 2.11.0 (ADR 0035): four typed columns added to round-
        # trip the ASDEmbeddedNodeElement optional flags.  Defaults
        # match the C++ parser so 2.10.x readers that ignore the new
        # columns observe the same on-deck behaviour as before.
        dt = np.dtype(
            [
                ("ele_tag", np.int64),
                ("cnode", np.int64),
                ("args", np.float64, (max_args,)),
                ("n_args", np.int32),
                ("stiffness", np.float64),
                ("stiffness_p", np.float64),
                ("has_stiffness_p", np.uint8),
                ("rotational", np.uint8),
                ("pressure", np.uint8),
                ("name", h5py.string_dtype(encoding="utf-8")),
            ]
        )
        rows = np.empty(len(recs), dtype=dt)
        for i, rec in enumerate(recs):
            padded = (
                [float(v) for v in rec.args]
                + [float("nan")] * (max_args - len(rec.args))
            )
            kp = rec.stiffness_p
            rows[i] = (
                rec.ele_tag, rec.cnode,
                tuple(padded), len(rec.args),
                float(rec.stiffness),
                float(kp) if kp is not None else float("nan"),
                1 if kp is not None else 0,
                1 if rec.rotational else 0,
                1 if rec.pressure else 0,
                rec.name,
            )
        parent.create_dataset("embeddedNode", data=rows)

    # -- Partition emission (ADR 0027, schema 2.10.0) -------------------

    def _write_partitions(self, f: Any) -> None:
        """Persist ``/opensees/partitions/`` per ADR 0027.

        One ``partition_NN`` sub-group per closed
        :meth:`partition_open` / :meth:`partition_close` block; each
        sub-group carries:

        * ``rank`` (int64 attr) — the OpenSeesMP rank id.
        * ``n_elements`` / ``n_nodes`` (int64 attrs) — convenience
          counters (``len`` of the respective dataset).
        * ``element_ids`` (int64 1-D dataset) — OpenSees element tags
          owned by this rank (insertion order).
        * ``node_ids`` (int64 1-D dataset) — OpenSees node tags
          declared on this rank's block, deduped within block.
          Includes foreign-declared nodes from cross-partition MP
          constraints (ADR 0027 §"Decision" INV-2 / phantom-node
          policy).
        * ``boundary_node_ids`` (int64 1-D dataset) — nodes shared
          with at least one other rank (the per-rank set intersected
          against the union of every other rank's node set).  May be
          empty (e.g. a model with one partition).

        The parent group carries the ``n_partitions`` (int64 attr) so
        a reader can quickly probe the count without enumerating
        children.  When no partition block was ever opened, the group
        is not created at all — old 2.9.x readers see no change to
        the file shape.

        Boundary-node strategy: **one-pass intersection on write** —
        symmetric across ranks because the per-rank node sets are
        all available when this method runs (the bridge has finished
        emitting before :meth:`write_opensees_into` is called).  See
        ADR 0027 §"Decision" INV-3 for the determinism contract.
        """
        # If a caller forgot to close the final partition block, flush
        # it defensively (mirrors ``_write_patterns``'s open-pattern
        # auto-close).  The bridge always pairs open/close; this is a
        # safety net for direct tests / interactive use.
        if self._partition_current is not None:
            if not self._partition_resumed:
                self._partition_blocks.append(self._partition_current)
            self._partition_resumed = False
            self._partition_current = None

        if not self._partition_blocks:
            return

        import numpy as np

        # Precompute each block's node set so the per-rank intersection
        # is O(N_blocks * mean_block_size) total.  A second loop then
        # subtracts the per-rank set from the global union to identify
        # the boundary slice for each rank.
        node_sets = [set(blk.node_ids) for blk in self._partition_blocks]

        parent = self._ops_group(f).create_group("partitions")
        _set_attr(parent, "n_partitions", len(self._partition_blocks))

        for idx, blk in enumerate(self._partition_blocks):
            g = parent.create_group(f"partition_{idx:02d}")
            _set_attr(g, "rank", blk.rank)
            _set_attr(g, "n_elements", len(blk.element_ids))
            _set_attr(g, "n_nodes", len(blk.node_ids))
            g.create_dataset(
                "element_ids",
                data=np.asarray(blk.element_ids, dtype=np.int64),
            )
            g.create_dataset(
                "node_ids",
                data=np.asarray(blk.node_ids, dtype=np.int64),
            )
            # Boundary nodes: intersect this rank's node set with the
            # union of every other rank's node set.  A node that lives
            # on rank K and rank L appears in both ranks'
            # ``boundary_node_ids`` (symmetric).  Preserve emission
            # order from this rank's ``node_ids`` so the dataset is
            # deterministic and easy to diff.
            others: set[int] = set()
            for other_idx, other_set in enumerate(node_sets):
                if other_idx == idx:
                    continue
                others.update(other_set)
            boundary = [n for n in blk.node_ids if n in others]
            g.create_dataset(
                "boundary_node_ids",
                data=np.asarray(boundary, dtype=np.int64),
            )

    # -- Analysis chain --------------------------------------------------

    def _write_analysis(self, f: Any) -> None:
        if not self._analysis_attrs and self._analyze_call is None:
            return
        analysis = self._ops_group(f).create_group("analysis")
        for key, value in self._analysis_attrs.items():
            _set_attr(analysis, key, value)
        if self._analyze_call is not None:
            steps, dt = self._analyze_call
            _set_attr(analysis, "analyze_steps", steps)
            if dt is not None:
                _set_attr(analysis, "analyze_dt", dt)

    def _write_pattern_ele_loads(
        self, g: Any, ele_loads: list[_EleLoadRecord],
    ) -> None:
        """Element-load records carry vocabulary-rich ``*args``; we
        store them as a single string array per row (since the
        ``-type``/``-ele``/``-eleRange`` flag tokens make positional
        decoding without vocabulary impractical). One row per call."""
        import h5py
        import numpy as np

        max_len = max(len(r.args) for r in ele_loads)
        rows = np.empty((len(ele_loads), max_len), dtype=object)
        for i, r in enumerate(ele_loads):
            row = [
                str(v) if isinstance(v, str) else repr(v)
                for v in r.args
            ]
            row.extend([""] * (max_len - len(row)))
            rows[i] = row
        g.create_dataset(
            "element_loads", data=rows,
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
