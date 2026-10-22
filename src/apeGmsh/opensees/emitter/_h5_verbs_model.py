"""The model verbs of :class:`~apeGmsh.opensees.emitter.h5.H5Emitter`: node, fix, mass,
constraints, contact, materials, sections, element, patterns, loads, recorder, region
and damping (ADR 0114 S5-b). A pure move out of ``h5.py``; the mixin holds no state.
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence, cast
from .base import DroppedAxisGuard, plain_scalar
from .verbs import VERBS
from .._internal.tag_resolution import (
    ATTR_ELEMENT_NODES,
    clear_element_emit_context,
    current_fem_element_id,
)
from .._internal.typed_records import (
    BeamIntegrationRecord as _BeamIntegrationRecord,
    DampingObjectRecord as _DampingObjectRecord,
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
    RegionRecord as _RegionRecord,
    RigidDiaphragmRecord as _RigidDiaphragmRecord,
    RigidLinkRecord as _RigidLinkRecord,
    SPRecord as _SPRecord,
    SectionComplexRecord as _SectionComplexRecord,
    SectionSimpleRecord as _SectionSimpleRecord,
    TimeSeriesRecord as _TimeSeriesRecord,
    TransformRecord as _TransformRecord,
)
from ._h5_blocks import _StageEmitBlock, _merge_node_region_args
from ._h5_program import _NO_DECL, _NO_ROW, _EmbeddedRebarRecord, _command_kind
from ._h5_host import _H5Host


_NODE_STORE = VERBS["node"].store
_ELEMENT_STORE, _BAR, _ELEMENT_STAGE_STORE = VERBS["element"].store.partition("|")
del _BAR


class _H5ModelVerbsMixin(_H5Host):
    """Mixin: holds no state, no ``__init__``; see the module docstring."""

    # =====================================================================
    # Protocol — Model
    # =====================================================================

    def model(self, *, ndm: int, ndf: int) -> None:
        self._ndm = ndm
        self._ndf = ndf
        self._dropped_axes = DroppedAxisGuard(ndm)

    def node(
        self, tag: int, *coords: float, ndf: int | None = None,
    ) -> None:
        # Normalize to 3-tuple by zero-padding 2-D models.
        cs = tuple(float(c) for c in coords)
        if len(cs) == 2:
            x, y = cs
            triple = (x, y, 0.0)
        elif len(cs) == 3:
            self._dropped_axes.trim(cs, int(tag))
            x, y, z = cs
            triple = (x, y, z)
        else:
            raise ValueError(
                f"H5Emitter.node: expected 2 or 3 coordinates, got "
                f"{len(cs)}: {cs!r}"
            )
        self._node_tags.append(int(tag))
        self._node_coords.append(triple)
        # Phantom nodes (ADR 0022 INV-3) — track tags so a reader can
        # identify which nodes were synthesized by the phantom-emit
        # pre-step.  Per S2 (ADR 0033) the per-node ``ndf`` kwarg is
        # now legal on real broker nodes too, so the discriminator is
        # the explicit phantom-tag predicate set on the emitter ONCE
        # by :func:`emit_mp_constraints` (and the partitioned variant)
        # before any node emission begins.  Phantom tags are disjoint
        # from real broker tags (the resolver allocates above
        # ``max(broker_node_tag)``), so the per-call lookup is
        # unambiguous and order-independent.
        from .._internal.tag_resolution import is_phantom_node
        if is_phantom_node(self, int(tag)):
            self._phantom_node_tags.append(int(tag))
        # Partition emission (ADR 0027) — while a per-rank block is
        # open, also record the tag on the active block so the
        # ``/opensees/partitions/partition_NN/node_ids`` dataset
        # reflects every node declared on that rank (native + foreign
        # MP-constraint declarations both count, per INV-2).
        if self._partition_current is not None:
            self._partition_current.add_node(int(tag))
        # Stage emission (ADR 0055 Phase 2) — dual-append: the global
        # buffers above stay complete (phantom bookkeeping spans all
        # nodes); the stage bucket additionally records the owned tag
        # in emit order.  Phantom tags are tracked separately — their
        # coordinates have no persisted home yet, so set_stage_records
        # fails loud on them rather than archiving an irreplayable
        # staged program.
        if self._stage_current is not None:
            blk = self._stage_current
            if self._partition_current is None:
                # Flat staged path — byte-identical to Phase 2.
                blk.owned_node_ids.append(int(tag))
                if is_phantom_node(self, int(tag)):
                    blk.phantom_node_tags.append(int(tag))
            else:
                # Partitioned staged capture (ADR 0055 Phase 5 / P5.1):
                # a stage's rank brackets declare three node kinds —
                # phantoms (stage-claimed node_to_surface; keep the
                # Phase-2 fail-loud signal, deduped across ranks),
                # stage-OWNED topology (a boundary stage node declares
                # on every owning rank — capture once), and FOREIGN
                # ghost declarations preceding replicated MP
                # constraints (ADR 0027 INV-2 — NOT stage topology;
                # the bridge surfaces the owned set via
                # ``set_stage_owned_node_tags`` because an
                # already-declared heuristic mis-classifies a foreign
                # decl that precedes its owning rank's bracket).
                from .._internal.tag_resolution import is_stage_owned_node
                if is_phantom_node(self, int(tag)):
                    if not self._partition_dup(
                        ("stage_phantom", blk.name, int(tag)),
                        own_row=False,
                    ):
                        blk.phantom_node_tags.append(int(tag))
                elif is_stage_owned_node(self, int(tag)):
                    if not self._partition_dup(
                        ("stage_node", blk.name, int(tag)), own_row=False,
                    ):
                        blk.owned_node_ids.append(int(tag))
                # else: foreign decl — partition-block mirror only.
        # ADR 0114 R2: the hot path notes inline, not through ``_noted``.
        # A7: a node is neutral-declared (the FEM snapshot), never a
        # bridge declaration's.
        tape = self._program
        if not tape.depth:
            tape.note_plain(
                "node", _NODE_STORE,
                _NO_ROW if self._stage_current is None
                else len(self._stage_blocks), _NO_DECL)

    def _partition_dup(
        self, key: "tuple[Any, ...]", *, own_row: bool = True,
    ) -> bool:
        """True iff ``key`` was already captured under a partition bracket.

        P5.0a (ADR 0055 Phase 5 groundwork): the partitioned global
        pass replicates fix / mass / MP-constraint / pattern-line
        emission across owning ranks (ADR 0027 INV-1/INV-4); the flat
        H5 capture must record each logical record once.  Outside a
        partition bracket this never reports a duplicate — flat-build
        capture (including genuine user duplicates) is unchanged.

        ``own_row``: ``key`` names the record the call writes to its own
        store, so a replica's ``/opensees/program`` row repeats the first
        capture's. ``False`` for a side list (``node``'s stage-owned ids).
        """
        if self._partition_current is None:
            return False
        if key in self._partition_seen:
            if own_row:
                self._program.repeat(key)
            return True
        self._partition_seen.add(key)
        if own_row:
            self._program.remember(key)
        return False

    def fix(self, tag: int, *dofs: int) -> None:
        # Stage emission (ADR 0055 Phase 2) — redirect: a stage-bound
        # fix must NOT land in the global ``/opensees/bcs`` zone (it
        # would double-apply as a t=0 BC on replay).
        rec = _FixRecord(tag=int(tag), dofs=tuple(int(d) for d in dofs))
        blk = self._stage_current
        if blk is not None:
            # P5.1: a stage-bound fix on a cross-rank shared node
            # replicates per owning rank (INV-4) — capture once.
            # ``_partition_dup`` is inert outside a partition bracket,
            # so the flat staged path is unchanged.
            key: "tuple[Any, ...]" = ("stage_fix", blk.name, rec.tag, rec.dofs)
            if self._partition_dup(key):
                self._decl_claim(key)
                return
            self._decl_note(key, blk.fix_decls)
            blk.fixes.append(rec)
            return
        key = ("fix", rec.tag, rec.dofs)
        if self._partition_dup(key):
            self._decl_claim(key)
            return
        self._decl_note(key, self._fix_decls)
        self._fixes.append(rec)

    def mark_mass_from_model(self) -> None:
        """Record that the model's nodal masses come from the snapshot.

        Called by the bridge instead of streaming one ``mass`` per node
        under ``mass_from_model()`` (ADR 0112 amendment 5): the masses
        already persist in the neutral zone (``/masses``).  Written as
        ``/opensees/bcs@mass_from_model = 1`` so
        :meth:`OpenSeesModel.build` streams ``fem.nodes.masses`` on
        replay.  A staged bracket cannot carry it: the declaration is
        model-level.
        """
        if self._stage_current is not None:
            raise RuntimeError(
                "H5Emitter.mark_mass_from_model: called inside stage "
                f"{self._stage_current.name!r}; mass_from_model() is "
                "model-level."
            )
        self._mass_from_model = True

    def mass(self, tag: int, *values: float) -> None:
        rec = _MassRecord(
            tag=int(tag), values=tuple(float(v) for v in values),
        )
        blk = self._stage_current
        if blk is not None:
            key: "tuple[Any, ...]" = (
                "stage_mass", blk.name, rec.tag, rec.values)
            if self._partition_dup(key):
                self._decl_claim(key)
                return
            self._decl_note(key, blk.mass_decls)
            blk.masses.append(rec)
            return
        key = ("mass", rec.tag, rec.values)
        if self._partition_dup(key):
            self._decl_claim(key)
            return
        self._decl_note(key, self._mass_decls)
        self._masses.append(rec)

    # =====================================================================
    # Protocol — MP constraints (ADR 0022, Phase 7b, schema 2.7.0)
    # =====================================================================

    def equalDOF(self, master: int, slave: int, *dofs: int) -> None:
        name = self._consume_pending_mp_name()
        rec = _EqualDOFRecord(
            master=int(master), slave=int(slave),
            dofs=tuple(int(d) for d in dofs),
            name=name,
        )
        if self._stage_current is not None:
            blk = self._stage_current
            # P5.1: stage-claimed constraint replicated per rank —
            # capture once, and skip the emit_index stamp with it.
            if self._partition_dup(
                ("stage_equalDOF", blk.name,
                 rec.master, rec.slave, rec.dofs, rec.name),
            ):
                return
            blk.equal_dof_seq.append(blk.next_emit_index())
            blk.equal_dofs.append(rec)
        else:
            # P5.0a: cross-rank replication (ADR 0027 INV-1) captures once.
            if self._partition_dup(
                ("equalDOF", rec.master, rec.slave, rec.dofs, rec.name),
            ):
                return
            self._equal_dofs.append(rec)

    def equalDOF_mixed(
        self, master: int, slave: int,
        dof_pairs: "Sequence[tuple[int, int]]",
    ) -> None:
        # ADR 0069 — OpenSees-deck archival of equalDOF_Mixed to H5 is
        # deferred (the intricate stage-block / partition-dedup / emit-
        # index machinery the other four MP kinds use). Fail loud here,
        # exactly like the staged-mutator H5 path, rather than silently
        # dropping the constraint from the archived deck. NB this is the
        # *deck* archival only — the canonical FEMData snapshot
        # (_femdata_h5_io) DOES round-trip equal_dof_mixed records.
        _ = (master, slave, dof_pairs)
        self._refuse(
            "equalDOF_mixed",
            "equalDOF_Mixed archival to the OpenSees H5 deck is deferred "
            "(ADR 0069). Emit the model with .tcl() / .py() / live run, or "
            "persist it via the FEMData .h5 snapshot (get_fem_data), which "
            "round-trips equal_dof_mixed records. Deck archival is tracked "
            "as a follow-up.",
        )

    def rigidLink(self, kind: str, master: int, slave: int) -> None:
        name = self._consume_pending_mp_name()
        rec = _RigidLinkRecord(
            kind=str(kind),
            master=int(master), slave=int(slave),
            name=name,
        )
        if self._stage_current is not None:
            blk = self._stage_current
            if self._partition_dup(
                ("stage_rigidLink", blk.name,
                 rec.kind, rec.master, rec.slave, rec.name),
            ):
                return
            blk.rigid_link_seq.append(blk.next_emit_index())
            blk.rigid_links.append(rec)
        else:
            # P5.0a: cross-rank replication (ADR 0027 INV-1) captures once.
            if self._partition_dup(
                ("rigidLink", rec.kind, rec.master, rec.slave, rec.name),
            ):
                return
            self._rigid_links.append(rec)

    def rigidDiaphragm(
        self, perp_dir: int, master: int, *slaves: int,
    ) -> None:
        name = self._consume_pending_mp_name()
        rec = _RigidDiaphragmRecord(
            perp_dir=int(perp_dir),
            master=int(master),
            slaves=tuple(int(s) for s in slaves),
            name=name,
        )
        if self._stage_current is not None:
            blk = self._stage_current
            if self._partition_dup(
                ("stage_rigidDiaphragm", blk.name,
                 rec.perp_dir, rec.master, rec.slaves, rec.name),
            ):
                return
            blk.rigid_diaphragm_seq.append(blk.next_emit_index())
            blk.rigid_diaphragms.append(rec)
        else:
            # P5.0a: a diaphragm replicates on every rank owning any
            # slave node (ADR 0027); capture the full line once.
            if self._partition_dup(
                (
                    "rigidDiaphragm", rec.perp_dir, rec.master,
                    rec.slaves, rec.name,
                ),
            ):
                return
            self._rigid_diaphragms.append(rec)

    def embeddedNode(
        self, ele_tag: int, cnode: int, *master_nodes: int,
        stiffness: float = 1.0e18,
        stiffness_p: float | None = None,
        rotational: bool = False,
        pressure: bool = False,
    ) -> None:
        name = self._consume_pending_mp_name()
        rec = _EmbeddedNodeRecord(
            ele_tag=int(ele_tag),
            cnode=int(cnode),
            args=tuple(int(m) for m in master_nodes),
            stiffness=float(stiffness),
            stiffness_p=(
                None if stiffness_p is None else float(stiffness_p)
            ),
            rotational=bool(rotational),
            pressure=bool(pressure),
            name=name,
        )
        if self._stage_current is not None:
            blk = self._stage_current
            blk.embedded_node_seq.append(blk.next_emit_index())
            blk.embedded_nodes.append(rec)
        else:
            self._embedded_nodes.append(rec)

    def equationConstraint(
        self, cnode: int, cdof: int, ccoef: float,
        retained: "Sequence[tuple[int, int, float]]",
    ) -> None:
        # Counted, not stored: the deck zone has no equationConstraint store.
        # A bridge-level ``apeSees.equation_constraint`` row is archived
        # anyway, as its ``EquationConstraintRecord`` ``/opensees/decl_params``
        # row, which replay re-emits; ``apeSees._write_h5`` takes
        # those rows off this count before warning. The rest come
        # from an ``enforce="equation"`` tie (ADR 0068, Open item 4,
        # RESOLVED), which deck replay skips. It is NOT data loss: the tie
        # is a resolved ``InterpolationRecord``, and the NEUTRAL zone
        # already persists its ``enforce`` route AND the projection
        # ``weights`` (schema 2.14.0).
        # ``apeSees(fem).h5(path)`` writes that neutral zone into the same
        # archive — so an equation-tied model.h5 round-trips its interface via
        # ``FEMData.from_h5`` → ``apeSees(fem).tcl()/py()/run()`` (the forward
        # emit re-runs ``_emit_one_interpolation`` → ``_emit_equation_tie``).
        # Hence no deviation warning here (mirroring embed / contact /
        # reinforce ties); just no-op the deck record and consume any latched
        # mp comment so it can't leak onto the next real MP record.
        del cnode, cdof, ccoef, retained
        self._consume_pending_mp_name()
        self._ledger("equationConstraint")

    def embedded_rebar(
        self, ele_tag: int, *args: int | float | str,
    ) -> None:
        # g.reinforce (ADR 20 / R2b): one ``/opensees/constraints/
        # embeddedRebar`` row per tie (K2b, schema 2.29.0), its planned
        # tag and the arguments ``embedded_rebar_args`` built, so a replay
        # writes the tie from the archive and mints no tag. Global only:
        # the bridge writes every tie before the stages, and the archive
        # has no stage store for one. Every argument is checked here, so a
        # refusal happens at emit, before any group is written.
        if self._stage_current is not None:
            raise RuntimeError(
                "H5Emitter.embedded_rebar: a global-only verb (VERBS scope "
                "'global') was emitted inside stage "
                f"{self._stage_current.name!r}; the archive has no stage "
                "store for a reinforce tie."
            )
        values = tuple(args)
        for value in values:
            _command_kind("embedded_rebar", value)
        name = self._consume_pending_mp_name()
        rec = _EmbeddedRebarRecord(
            ele_tag=int(ele_tag), args=values, name=name)
        # A partitioned emit writes each tie on the rank that owns it; a
        # replica of the same line is captured once (P5.0a).
        if self._partition_dup(
            ("embedded_rebar", rec.ele_tag, rec.args, rec.name),
        ):
            return
        self._embedded_rebars.append(rec)

    def embedded_node(
        self, ele_tag: int, *args: int | float | str,
    ) -> None:
        # g.embed: the OpenSees *deck* zone (``/opensees/...``) does not carry a
        # dedicated LadrunoEmbeddedNode record (a deck-replay follow-on, like
        # reinforceTie). This is NOT data loss: as of neutral schema 2.22.0
        # (ADR 0073) the NEUTRAL zone persists every EmbedTieRecord, and
        # ``apeSees(fem).h5(path)`` writes it into the same archive — so an
        # embedded model.h5 round-trips its embedment via ``FEMData.from_h5`` →
        # ``apeSees(fem).tcl()/py()/run()`` (the forward emit re-runs
        # ``emit_embed_ties``). Hence no deviation warning here (mirroring
        # reinforce / contact ties); just no-op the deck record and consume any
        # latched mp comment so it can't leak onto the next real MP record.
        del ele_tag, args
        self._consume_pending_mp_name()
        self._ledger("embedded_node")

    def contact_surface(
        self, tag: int, *args: int | float | str,
    ) -> None:
        # g.constraints.contact: the OpenSees *deck* zone (``/opensees/...``)
        # does not carry a dedicated contactSurface/contact record (a
        # deck-replay follow-on, like reinforceTie). This is NOT data loss: as
        # of neutral schema 2.21.0 (ADR 0073) the NEUTRAL zone persists every
        # ContactRecord, and ``apeSees(fem).h5(path)`` writes that neutral zone
        # into the same archive — so a contact model.h5 round-trips its contact
        # via ``FEMData.from_h5`` → ``apeSees(fem).tcl()/py()/run()`` (the
        # forward emit re-runs ``emit_contacts``). Hence no deviation warning
        # here (mirroring reinforce ties); just no-op the deck record and
        # consume any latched mp comment so it can't leak onto the next real MP
        # record.
        del tag, args
        self._consume_pending_mp_name()
        self._ledger("contact_surface")

    def contact(
        self, tag: int, *args: int | float | str,
    ) -> None:
        # Companion to contact_surface — the warning fires once on the first
        # contactSurface; just consume the call here.
        del tag, args
        self._ledger("contact")

    def contact_plane(
        self, tag: int, *args: int | float | str,
    ) -> None:
        # g.constraints.contact_plane: deck-zone no-op, recovered via the
        # neutral /contact_planes group (ContactPlaneRecord round-trips through
        # FEMData.to_h5/from_h5 → emit_contact_planes re-emits). The slave-set
        # contactSurface call is consumed by ``contact_surface`` above; this
        # just consumes the contactPlane verb.
        del tag, args
        self._ledger("contact_plane")

    def mp_constraint_comment(self, name: str) -> None:
        # Latch the declaration label; the next MP-constraint call will
        # consume it via ``_consume_pending_mp_name`` (INV-2).  This is
        # the H5 emitter's analogue of the Tcl/Py ``# {name}`` line.
        self._pending_mp_name = str(name)

    def _consume_pending_mp_name(self) -> str:
        """Return-and-clear the pending mp_constraint_comment label."""
        name = self._pending_mp_name
        self._pending_mp_name = ""
        return name

    # =====================================================================
    # Protocol — Constitutive
    # =====================================================================

    def uniaxialMaterial(
        self, mat_type: str, tag: int, *params: float | str,
    ) -> None:
        self._uniaxial.append(
            _MaterialRecord(type_token=mat_type, tag=int(tag), params=tuple(params))
        )

    def nDMaterial(
        self, mat_type: str, tag: int, *params: float | str,
    ) -> None:
        self._nd.append(
            _MaterialRecord(type_token=mat_type, tag=int(tag), params=tuple(params))
        )

    def section(
        self, sec_type: str, tag: int, *params: float | str,
    ) -> None:
        self._sections_simple.append(
            _SectionSimpleRecord(
                type_token=sec_type, tag=int(tag), params=tuple(params),
            )
        )

    def geomTransf(
        self, t_type: str, tag: int, *vec: float,
        jnt_offset: tuple[float, ...] = (),
    ) -> None:
        self._transforms.append(
            _TransformRecord(
                type_token=t_type, tag=int(tag),
                vec=tuple(float(v) for v in vec),
                jnt_offset=tuple(float(v) for v in jnt_offset),
            )
        )

    # =====================================================================
    # Protocol — Sections that take blocks (Fiber)
    # =====================================================================

    def section_open(
        self, sec_type: str, tag: int, *params: float | str,
    ) -> None:
        if self._open_section is not None:
            raise RuntimeError(
                "H5Emitter.section_open: a section is already open "
                f"({self._open_section.type_token} tag={self._open_section.tag}); "
                "call section_close first."
            )
        self._open_section = _SectionComplexRecord(
            type_token=sec_type, tag=int(tag), params=tuple(params),
        )

    def section_close(self) -> None:
        if self._open_section is None:
            raise RuntimeError(
                "H5Emitter.section_close called with no open section."
            )
        self._sections_complex.append(self._open_section)
        self._open_section = None

    def patch(self, kind: str, *args: int | float) -> None:
        if self._open_section is None:
            raise RuntimeError(
                "H5Emitter.patch called outside a section_open / "
                "section_close block."
            )
        self._open_section.patches.append(
            _PatchRecord(kind=kind, args=tuple(args))
        )

    def fiber(
        self, y: float, z: float, area: float, mat_tag: int,
    ) -> None:
        if self._open_section is None:
            raise RuntimeError(
                "H5Emitter.fiber called outside a section_open / "
                "section_close block."
            )
        self._open_section.fibers.append(
            _FiberRecord(
                y=float(y), z=float(z),
                area=float(area), mat_tag=int(mat_tag),
            )
        )

    def layer(self, kind: str, *args: int | float) -> None:
        if self._open_section is None:
            raise RuntimeError(
                "H5Emitter.layer called outside a section_open / "
                "section_close block."
            )
        self._open_section.layers.append(
            _LayerRecord(kind=kind, args=tuple(args))
        )

    # =====================================================================
    # Protocol — Beam integration rules
    # =====================================================================

    def beamIntegration(
        self, rule_type: str, tag: int, *args: int | float | str,
    ) -> None:
        self._beam_integrations.append(
            _BeamIntegrationRecord(
                type_token=rule_type, tag=int(tag), args=tuple(args),
            )
        )

    # =====================================================================
    # Protocol — Topology
    # =====================================================================

    def element(
        self, ele_type: str, tag: int, *args: int | float | str,
    ) -> None:
        # The bridge sets _current_element_nodes via
        # set_element_nodes(emitter, ...) right before each element call
        # (see e.g. _internal/build/element_emit.py::
        # emit_element_spec_partitioned). Reading it here
        # gives us the connectivity for the schema's
        # /elements/{type}/connectivity dataset.
        connectivity = getattr(self, ATTR_ELEMENT_NODES, None)
        if connectivity is None:
            # Allow direct test driving without the bridge wrapper —
            # fall back to extracting node tags from args (the first
            # `n` integer args before the first non-int arg are the
            # nodes). We don't try to be clever here; tests that need a
            # specific connectivity install set_element_nodes first.
            connectivity = tuple()
        # Phase 8.6: capture the FEM element id from the side channel
        # (sentinel ``-1`` when called outside a bridge fan-out — see
        # `_internal/tag_resolution.MISSING_FEM_ELEMENT_ID`).
        fem_eid = current_fem_element_id(self)
        # Consume both side channels: the bridge re-installs them per
        # fan-out row, and a sticky value would otherwise leak the LAST
        # mesh row's fem_eid + connectivity onto elements minted by the
        # side-list passes (interfaces / rebar / couplings — the
        # ADR 0093 S10 mislabeled-springs finding).
        clear_element_emit_context(self)
        self._elements.append(
            _ElementRecord(
                type_token=ele_type, tag=int(tag),
                args=tuple(args),
                connectivity=tuple(int(c) for c in connectivity),
                fem_eid=fem_eid,
            )
        )
        # Partition emission (ADR 0027) — record the per-element rank
        # parallel to ``self._elements`` so the per-type
        # ``partition_ids`` column on ``/opensees/element_meta/`` can
        # be materialised at write time.  Sentinel ``-1`` for elements
        # emitted outside any ``partition_open`` / ``partition_close``
        # bracket.  Also append to the active block's ``element_ids``
        # list so the per-rank dataset reflects emission order.
        if self._partition_current is not None:
            self._element_ranks.append(self._partition_current.rank)
            self._partition_current.add_element(int(tag))
        else:
            self._element_ranks.append(-1)
        # Stage emission (ADR 0055 Phase 2) — dual-append: the global
        # ``element_meta`` zone must carry EVERY element (the reader
        # rebuilds the full pool from it; the ``fem_eids`` column is
        # the persisted fem_eid→ops_tag map); the stage bucket records
        # the owned tag so replay re-emits it inside the stage block.
        if self._stage_current is not None:
            self._stage_current.owned_element_ids.append(int(tag))
        # ADR 0114 R2: the hot path notes inline, not through ``_noted``.
        tape = self._program
        if not tape.depth:
            # A7: the declaration whose ``element`` tag run holds ``tag``;
            # a fan-out walks one run in order, so the last hit answers.
            tag = int(tag)
            first, end, decl = self._element_decl_run
            if not first <= tag < end:
                run = self._decl_run("element", tag)
                if run is None:
                    decl = _NO_DECL
                else:
                    self._element_decl_run = run
                    decl = run[2]
            if self._stage_current is None:
                tape.note_plain("element", _ELEMENT_STORE, _NO_ROW, decl)
            else:
                tape.note_plain(
                    "element", _ELEMENT_STAGE_STORE, len(self._stage_blocks),
                    decl)

    # =====================================================================
    # Public — declarative orientation inject (ADR 0018 / ModelData)
    # =====================================================================

    def add_oriented_elements(
        self,
        *,
        type_token: str,
        vecxz: "tuple[float, float, float]",
        elements: "Iterable[tuple[int, Sequence[int]]]",
        ndm: int,
    ) -> None:
        """Append one ``geomTransf`` + N elements as one orientation fact.

        Public schema-owning method used by
        :class:`apeGmsh.opensees.ModelData` to inject the per-element
        ``vecxz`` orientation the viewer's beam join needs
        (``h5_reader.element_local_axes_vecxz``).  Driving the bridge
        Protocol methods (``geomTransf`` / ``element``) from a
        declarative front-door would require pre-setting ``_internal``
        side-channels — exporting bridge-internal coupling into a
        public class.  This method appends a ``_TransformRecord`` and
        per-element ``_ElementRecord`` instances directly, stamping a
        real positive ``fem_eid`` (ADR 0018 INV-6).

        Tag values are file-internal — ``model.h5`` only needs the
        transf record's ``tag`` to match the value stored at the
        registry-defined slot of each element's ``args``.  The reader
        join (h5_reader.py:336-363) never cross-checks against a live
        OpenSees domain.

        Parameters
        ----------
        type_token
            OpenSees element type (``"forceBeamColumn"``,
            ``"elasticBeamColumn"``, ...).  Must yield a transf slot
            via :func:`_transf_arg_tail_index` for ``ndm``; an unknown
            token or one with no transf slot raises (ADR 0018 INV-7).
        vecxz
            Three-tuple ``(vx, vy, vz)``.
        elements
            Iterable of ``(fem_eid, connectivity)`` pairs.  ``fem_eid``
            must be ``> 0`` — sentinel ``-1`` is invalid for a
            declarative inject (ADR 0018 INV-6).
        ndm
            2 or 3.  Selects ``slots_2d`` vs ``slots_3d`` for the
            transf-tag positional slot (ADR 0018 INV-9).

        Raises
        ------
        ValueError
            If ``ndm`` is not in ``{2, 3}``, ``vecxz`` lacks 3
            components, ``type_token`` is unknown / has no transf
            slot, or any ``fem_eid`` is ``<= 0``.
        """
        import math

        from .._element_capabilities import (
            _ELEM_REGISTRY,
            _transf_arg_tail_index,
            known_beam_type_tokens,
        )

        if ndm not in (2, 3):
            raise ValueError(
                f"H5Emitter.add_oriented_elements: ndm must be 2 or 3, "
                f"got {ndm!r}."
            )
        if len(vecxz) != 3:
            raise ValueError(
                f"H5Emitter.add_oriented_elements: vecxz must have 3 "
                f"components, got {len(vecxz)}: {vecxz!r}."
            )
        idx = _transf_arg_tail_index(type_token, ndm, _ELEM_REGISTRY)
        if idx is None:
            valid = ", ".join(known_beam_type_tokens(ndm))
            raise ValueError(
                f"H5Emitter.add_oriented_elements: type_token="
                f"{type_token!r} has no transf slot at ndm={ndm}. "
                f"Valid beam tokens: {valid}."
            )

        items = list(elements)
        for fem_eid, _ in items:
            if int(fem_eid) <= 0:
                raise ValueError(
                    f"H5Emitter.add_oriented_elements: fem_eid must "
                    f"be > 0 (ADR 0018 INV-6 — sentinel -1 is invalid "
                    f"for a declarative inject), got {fem_eid!r}."
                )

        # Allocate file-internal tags.  Counter is monotonic on this
        # emitter; do not mix with bridge-driven ``element()`` calls
        # on the same instance.
        self._orientation_tag_counter += 1
        transf_tag = self._orientation_tag_counter
        self._transforms.append(
            _TransformRecord(
                # geomTransf type-token here is the *class string*
                # (Linear / PDelta / Corotational), independent of the
                # element type.  The reader's orientation join reads
                # only ``vec`` + ``tag`` — geomTransf class does not
                # affect orientation.  Use the schema-stable default.
                type_token="Linear",
                tag=transf_tag,
                vec=(float(vecxz[0]), float(vecxz[1]), float(vecxz[2])),
            )
        )

        # Pad args to (idx + 1) tail length with NaN sentinels — the
        # viewer's orientation join only reads the transf slot; other
        # tail columns are unread for orientation.  See
        # ``_write_element_argstack``: arity = len(connectivity),
        # tail = args[arity:].
        for fem_eid, conn in items:
            conn_t = tuple(int(c) for c in conn)
            self._orientation_tag_counter += 1
            ops_tag = self._orientation_tag_counter
            tail: list[float] = [math.nan] * (idx + 1)
            tail[idx] = float(transf_tag)
            self._elements.append(
                _ElementRecord(
                    type_token=type_token,
                    tag=ops_tag,
                    args=(*conn_t, *tail),
                    connectivity=conn_t,
                    fem_eid=int(fem_eid),
                )
            )
            # Parallel partition_ids row — declarative inject is never
            # rank-scoped (ModelData runs outside the bridge fan-out),
            # so the sentinel -1 always applies.  Keeping the lists
            # parallel is invariant: a future caller mixing this path
            # with bridge-driven element() must not break the
            # element_meta column write.
            self._element_ranks.append(-1)

    # =====================================================================
    # Protocol — Time series
    # =====================================================================

    def timeSeries(
        self, ts_type: str, tag: int, *args: int | float | str,
    ) -> None:
        self._time_series.append(
            _TimeSeriesRecord(
                type_token=ts_type, tag=int(tag), args=tuple(args),
            )
        )

    # =====================================================================
    # Protocol — Patterns
    # =====================================================================

    def pattern_open(
        self, p_type: str, tag: int, *args: int | float | str,
    ) -> None:
        rec = _PatternRecord(
            type_token=p_type, tag=int(tag), args=tuple(args),
        )
        # Stage emission (ADR 0055 Phase 2) — a stage-scoped pattern
        # (7b stage pattern or the ADR 0052 HOLD support pattern)
        # captures into the stage bucket, never the global zone.
        if self._stage_current is not None:
            blk = self._stage_current
            self._flush_stage_open_pattern(blk)
            # P5.1: the partitioned staged pattern fan-out (stage
            # patterns AND the ADR 0052 HOLD pattern) re-opens the
            # same tag once per owning rank — resume the captured
            # record so per-rank line subsets merge, and skip the
            # pattern_seq stamp (the first open stamped it).
            if self._partition_current is not None:
                key = (blk.name, int(tag))
                existing = self._stage_partition_patterns.get(key)
                if existing is not None and existing.type_token == p_type:
                    blk.open_pattern = existing
                    self._stage_open_pattern_resumed = True
                    self._program.repeat(("stage_pattern", *key))
                    return
                self._stage_partition_patterns[key] = rec
                self._program.remember(("stage_pattern", *key))
            blk.pattern_seq.append(blk.next_emit_index())
            blk.open_pattern = rec
            return
        if self._open_pattern is not None:
            # Auto-close a stale open pattern. Defensive; the bridge
            # always pairs open/close, but if a single-line pattern
            # (UniformExcitation) was opened and never explicitly
            # closed before another open, finalize it here.
            self._flush_open_pattern()
        # P5.0a (ADR 0027): the partitioned global pass re-opens the
        # SAME pattern tag once per owning rank with a rank-filtered
        # subset of lines ("shared tag, local copy per rank" —
        # ``_emit_one_pattern_partitioned``).  Under a partition
        # bracket, a re-open of an already-captured tag RESUMES that
        # record so the per-rank line subsets merge into one logical
        # pattern — otherwise two ``_PatternRecord``s with the same tag
        # collide on the ``patterns/<type>_<tag>`` group name at write.
        if self._partition_current is not None:
            existing = self._partition_patterns_by_tag.get(int(tag))
            if existing is not None and existing.type_token == p_type:
                self._open_pattern = existing
                self._open_pattern_resumed = True
                self._program.repeat(("pattern", int(tag)))
                return
            self._partition_patterns_by_tag[int(tag)] = rec
            self._program.remember(("pattern", int(tag)))
        self._open_pattern = rec

    def _flush_open_pattern(self) -> None:
        """Finalize ``_open_pattern`` — append-once aware (P5.0a).

        A RESUMED record (partition-bracketed re-open of an
        already-captured tag) is already in ``_patterns_complete``;
        appending it again would write the pattern group twice.
        """
        if self._open_pattern is None:
            return
        if not self._open_pattern_resumed:
            self._patterns_complete.append(self._open_pattern)
        self._open_pattern = None
        self._open_pattern_resumed = False

    def _flush_stage_open_pattern(self, blk: "_StageEmitBlock") -> None:
        """Finalize a stage bucket's open pattern — append-once aware
        (P5.1): a RESUMED record (partition-bracketed re-open) is
        already in ``patterns_complete``."""
        if blk.open_pattern is None:
            return
        if not self._stage_open_pattern_resumed:
            blk.patterns_complete.append(blk.open_pattern)
        blk.open_pattern = None
        self._stage_open_pattern_resumed = False

    def pattern_close(self) -> None:
        if self._stage_current is not None:
            self._flush_stage_open_pattern(self._stage_current)
            return
        if self._open_pattern is None:
            # Allowed — single-line pattern closes are a no-op in some
            # emitters; mirror that tolerance here.
            return
        self._flush_open_pattern()

    def _active_pattern(self, method: str) -> _PatternRecord:
        """The pattern record body calls append into — stage-aware."""
        rec = (
            self._stage_current.open_pattern
            if self._stage_current is not None
            else self._open_pattern
        )
        if rec is None:
            raise RuntimeError(
                f"H5Emitter.{method} called outside an open pattern."
            )
        return rec

    def load(self, tag: int, *forces: float) -> None:
        rec = _LoadRecord(
            target=int(tag), forces=tuple(float(f) for f in forces),
        )
        pat = self._active_pattern("load")
        # P5.0a/P5.1: a load on a cross-rank SHARED node emits inside
        # every owning rank's copy of the pattern (global AND stage
        # patterns); the merged capture keeps one row.  The scope slot
        # keeps a stage-pattern line distinct from a same-shaped
        # global-pattern line.
        scope = (
            self._stage_current.name
            if self._stage_current is not None else None
        )
        if self._partition_dup(
            ("load", scope, pat.tag, rec.target, rec.forces),
        ):
            return
        pat.loads.append(rec)

    def eleLoad(self, *args: int | float | str) -> None:
        pat = self._active_pattern("eleLoad")
        args_tuple: tuple[int | float | str, ...] = tuple(
            cast("int | float | str", plain_scalar(a)) for a in args
        )
        scope = (
            self._stage_current.name
            if self._stage_current is not None else None
        )
        if self._partition_dup(("eleLoad", scope, pat.tag, args_tuple)):
            return
        pat.ele_loads.append(_EleLoadRecord(args=args_tuple))

    def sp(self, tag: int, dof: int, value: float) -> None:
        rec = _SPRecord(target=int(tag), dof=int(dof), value=float(value))
        pat = self._active_pattern("sp")
        # P5.0a/P5.1: mirror of :meth:`load` — shared-node sp captures once.
        scope = (
            self._stage_current.name
            if self._stage_current is not None else None
        )
        if self._partition_dup(
            ("sp", scope, pat.tag, rec.target, rec.dof, rec.value),
        ):
            return
        pat.sps.append(rec)

    def sp_hold(self, node: int, dof: int) -> None:
        """Capture a stage-bound HOLD line (ADR 0052 / ADR 0055 Phase 2).

        ``sp_hold`` is only ever emitted inside a stage's dedicated
        support pattern; the bucket's open pattern records the
        ``(node, dof)`` pair and the deck emitters re-render the
        ``sp <node> <dof> [nodeDisp …] -const`` form on replay.
        Outside a stage bracket it raises (ADR 0114 D2); the bridge only
        emits it inside a stage's HOLD pattern.
        """
        blk = self._stage_block("sp_hold")
        pat = self._active_pattern("sp_hold")
        # P5.1: a HOLD on a cross-rank shared node emits inside every
        # owning rank's copy of the stage's HOLD pattern — capture once.
        key = ("sp_hold", blk.name, pat.tag, int(node), int(dof))
        if self._partition_dup(key):
            self._decl_claim(key)
            return
        self._decl_note(key, blk.sp_hold_decls.setdefault(pat.tag, []))
        pat.sp_holds.append((int(node), int(dof)))

    # =====================================================================
    # Protocol — Recorders
    # =====================================================================

    def recorder(self, kind: str, *args: int | float | str) -> None:
        blk = self._stage_current
        if blk is not None:
            sink, decls = blk.recorders, blk.recorder_decls
        else:
            sink, decls = self._recorders, self._recorder_decls
        # Recorders are emitted outside any partition bracket, so no
        # replica reaches here: the column takes the open declaration.
        decls.append(self._decl_open)
        self._program.decl = self._decl_open  # A7
        sink.append(_RecorderRecord(
            kind=kind,
            args=tuple(args),
            decl_context=self._decl_context,
        ))

    def region(self, tag: int, *args: int | float | str) -> None:
        if self._stage_current is not None:
            blk = self._stage_current
            # P5.1: the partitioned staged region fan-out
            # (``_emit_stage_regions_partitioned``) emits the SAME tag
            # once per contributing rank with the rank-intersection of
            # members (``-node n1 n2 ...``).  Merge the fragments into
            # the one logical region (member union, first-occurrence
            # order, single emit_index) — the only per-rank stage
            # region producer emits the plain ``-node`` form; anything
            # else (region-scoped rayleigh / damping attach) emits
            # once globally per stage and never re-uses a tag, so a
            # non-mergeable shape falls through to a plain append.
            if self._partition_current is not None:
                key = (blk.name, int(tag))
                idx = self._stage_partition_regions.get(key)
                if idx is not None:
                    merged = _merge_node_region_args(
                        blk.regions[idx].args, tuple(args),
                    )
                    if merged is not None:
                        blk.regions[idx] = _RegionRecord(
                            tag=int(tag), args=merged,
                        )
                        self._program.repeat(("stage_region", *key))
                        return
                else:
                    self._stage_partition_regions[key] = len(blk.regions)
                    self._program.remember(("stage_region", *key))
            blk.region_seq.append(blk.next_emit_index())
            blk.regions.append(_RegionRecord(tag=int(tag), args=tuple(args)))
            return
        self._regions.append(_RegionRecord(tag=int(tag), args=tuple(args)))

    def rayleigh(
        self,
        alpha_m: float,
        beta_k: float,
        beta_k_init: float,
        beta_k_comm: float,
    ) -> None:
        # The GLOBAL form is a domain directive, not a tagged object: it
        # is a ``/opensees/commands`` row (ADR 0114 R3a, schema 2.23.0).
        # Region-scoped Rayleigh (ADR 0053 D2) persists via ``region``,
        # which carries the ``-rayleigh`` tail.
        # ADR 0055 Phase 2: a STAGE-bound global-form rayleigh
        # (``s.damping.rayleigh`` with ``on=()``) is part of the staged
        # program and captures into the stage bucket.
        if self._stage_current is not None:
            blk = self._stage_current
            blk.rayleigh_seq.append(blk.next_emit_index())
            blk.rayleighs.append((
                float(alpha_m), float(beta_k),
                float(beta_k_init), float(beta_k_comm),
            ))
            blk.rayleigh_decls.append(self._decl_open)  # K1-6
            self._program.decl = self._decl_open  # A7
            return
        self._command("rayleigh", (alpha_m, beta_k, beta_k_init, beta_k_comm))

    def damping(
        self, damp_type: str, tag: int, *args: int | float | str,
    ) -> None:
        # ADR 0053 (D3b, schema 2.15.0): persist tagged damping objects
        # under ``/opensees/dampings/`` so they survive a model.h5
        # round-trip and replay through ``_replay_into`` (was a no-op in
        # D3a). Element-flag ``-damp`` attachments ride in the element's
        # own arg tail; region attaches share the regions limitation.
        self._dampings.append(
            _DampingObjectRecord(
                type_token=damp_type, tag=int(tag), args=tuple(args),
            )
        )

    def modal_damping(self, *factors: float) -> None:
        # ADR 0053 (D4): a domain directive, so a ``/opensees/commands``
        # row (ADR 0114 R3a); the ``eigen`` row before it supplies the
        # modes it damps.
        self._command("modal_damping", factors)
