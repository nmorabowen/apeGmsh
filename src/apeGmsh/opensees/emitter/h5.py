"""
H5Emitter — buffers the bridge's emit calls and writes a model-definition
HDF5 archive conforming to ``architecture/h5-schema.md``.

Unlike the stream emitters (Tcl, Py, Live), this is a **structured**
emitter: each Protocol method updates an in-memory buffer; the file is
materialized at the end via :meth:`write`. The structured shape is
required because the schema groups by name
(``/opensees/materials/uniaxial/{name}``,
``/opensees/sections/{name}``, ...), and aggregate sections / patterns
need open / close bracketing while patches, fibers, loads, and sps
land inside them.

Design notes
============

**Names from tags (Option B in ADR 0011's discussion).** The bridge's
typed primitives carry no user-supplied ``name`` field. Every name in
the file is generated from ``<type_token>_<tag>`` (e.g. ``Steel02_1``,
``Fiber_2``, ``forceBeamColumn_3``). Cross-references resolve via the
generated names. No Protocol extension was required.

**Schema layout.** Phase 8.4 partitioned ``model.h5`` into two zones
(see ``architecture/h5-schema.md`` and
https://github.com/nmorabowen/apeGmsh/blob/a9f8b670df700dd5b1a9c40f5c7d1f25dd1d400b/architecture/phase-8-untangle.md §3 (pinned at a9f8b670)):

* ``/meta`` and ``/elements`` stay at the file root.  ``/elements``
  sits in the neutral zone because Phase 8.5 hands the writer from
  the bridge to the broker; for now the bridge still emits it.
* Everything else the bridge writes (materials, sections, transforms,
  beam_integration, time_series, patterns, bcs, recorders, analysis)
  lives under ``/opensees/`` so a future second solver can plug in at
  the same level without colliding.

The reshuffle is a breaking schema change — ``SCHEMA_VERSION`` jumps
``1.1.0 → 2.0.0``.  Phase 7a (ADR 0023) replaced the previous
``EXPECTED_SCHEMA_MAJOR`` constant with per-zone version validation in
:mod:`apeGmsh.opensees._internal.schema_version` (a two-version window
then; a floor per zone since ADR 0113).

**Schema deviation (documented).**  One place where the streaming
Protocol cannot supply the spec-level grouping the schema asks for:

* ``/opensees/transforms/{name}`` — the schema shows one group per
  user-declared transform with a ``per_element_vecxz`` dataset of
  shape ``(n_elements, 3)``.  The orientation-driven fan-out in the bridge's
  build layer emits one ``geomTransf`` line per *distinct* vecxz; the
  H5 emitter sees these as N independent calls and cannot
  reverse-engineer the spec boundary.  We therefore emit one
  ``/opensees/transforms/{type}_{tag}/`` group per ``geomTransf``
  call, each carrying a single-row ``per_element_vecxz`` and
  ``per_element_emitted_tag``.  The viewer's vecxz overlay (which
  only needs ``element_id → vecxz``) still works: it iterates all
  transform groups.

**Elements binned by type token (design choice, not deviation).**  The
schema groups elements by element-type token
(``/elements/forceBeamColumn/``, ``/elements/FourNodeTetrahedron/``,
...).  The streaming Protocol does not surface the PG (``spec.pg`` is
known only inside ``_internal/build/``'s element fan-out), so the
H5 emitter bins by type.  Each group's ``ids`` and ``connectivity``
datasets stack every element of that type across all PGs; the
``args`` / ``args_str`` dataset pair encodes the element's positional
arg list so a vocabulary-aware reader can recover refs.

Both the transform deviation and the elements-by-type choice are
recorded as ``__deviation__`` attrs on the affected groups so a
reader can detect them and degrade gracefully.

**Lazy h5py import.** ``h5py`` is imported only inside :meth:`write` so
constructing an :class:`H5Emitter` (or driving emit) does not pull
the dependency into import time for users who never call ``ops.h5()``.
"""
from __future__ import annotations

import dataclasses
import functools
from collections import Counter
from collections.abc import Callable, Mapping
from types import MappingProxyType
from datetime import datetime, timezone
from typing import Any, NamedTuple, NoReturn, Sequence

from .._target import TargetMode
from .base import DroppedAxisGuard
from .caps import SolveStamp, TargetCaps, value_requires
from .verbs import VERBS, Verb
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
from ._h5_blocks import (
    _PartitionEmitBlock,
    _StageEmitBlock,
)
from ._h5_program import (
    _NO_DECL,
    _NO_ROW,
    _CommandRecord,
    _EmbeddedRebarRecord,
    _ProgramTape,
    _command_kind,
)
from ._h5_write import _set_attr
from .h5_reader import TARGET_MODES, EquationConstraintRecordRO
from ._h5_decls import _H5DeclsMixin, _decl_refs, _json_text
from ._h5_stages import _H5StagesMixin
from ._h5_verbs_model import _H5ModelVerbsMixin
from ._h5_write_model import (  # noqa: F401  (re-exported name helpers)
    _H5WriteModelMixin,
    beam_integration_name,
    element_group_name,
    material_name,
    pattern_name,
    recorder_name,
    section_name_complex,
    section_name_simple,
    time_series_name,
    transform_name,
)

__all__ = [
    "Declaration", "DeclParamsRow", "DeclTagRun", "H5Emitter",
    "SCHEMA_FLOOR", "SCHEMA_VERSION",
    "H5DeclParamsError",
    "H5EquationConstraintDeviationWarning",
    "H5FeatureDeferredWarning",
    "H5LedgerWarning",
    "H5ReinforceDeviationWarning",
    "decl_argv_names", "encode_decl_params",
]


class H5EquationConstraintDeviationWarning(UserWarning):
    """**Dormant — no longer raised** (ADR 0068, Open item 4 RESOLVED).

    Originally signalled that an ``enforce="equation"`` tie (EQ_Constraint)
    was dropped from the OpenSees H5 *deck* zone. That deferral is closed: the
    equation tie is a resolved :class:`InterpolationRecord`, and the neutral
    zone already persists ``enforce`` **and** the projection ``weights``
    (schema 2.14.0). So an equation-tied ``model.h5`` round-trips its interface
    via ``FEMData.from_h5`` → ``apeSees(fem).tcl()/py()/run()`` (the forward
    emit re-runs ``_emit_equation_tie``) — exactly like g.embed / contact /
    reinforce ties, which also no-op silently in the deck zone. The class is
    retained for back-compat (existing imports); new code should not expect it
    to fire."""


class H5LedgerWarning(UserWarning):
    """``apeSees.h5`` wrote an archive that leaves out ``ledger`` calls.

    A ``ledger`` row of ``VERBS`` (ADR 0114, Q3) is a verb ``H5Emitter``
    counts and drops: ``/opensees`` carries no record of the call, so a
    deck replayed from the file omits it (the contact and embed rows; some
    persist in the neutral zone instead). ``equationConstraint`` is not a
    ledger verb: every line is a ``{scope}/constraints/equationConstraint``
    row that replay re-emits (global from opensees 2.35.0, a stage's from
    2.37.0).
    The bridge raises it at most once per ``apeSees`` instance for each
    distinct set of ledgered verbs. It is not an ``H5FeatureDeferredWarning``,
    so the automatic ``model.h5`` write does not silence it."""


class H5FeatureDeferredWarning(UserWarning):
    """A fork-only feature was dropped from the OpenSees H5 deck AND has no
    neutral-zone persistence — native H5 round-trip of the feature is deferred,
    so the feature is lost on round-trip. Emit to Tcl / openseespy (or run live)
    for the complete model.

    ``apeSees.h5`` does not raise it for ``apeSees.equation_constraint``
    rows: each is archived as an ``EquationConstraintRecord`` row of
    ``/opensees/decl_params`` and replayed. The ``model.h5``
    rewrite raises it for program stores it does not write. g.reinforce ties,
    g.constraints.contact,
    g.embed ties, and the ``enforce="equation"`` tie all persist via the
    neutral zone (schema 2.15.0 / 2.21.0 / 2.22.0 / 2.14.0). The class (and its
    back-compat alias) is retained for any future deferred feature. The
    equation-route (EQ_Constraint) deferral is likewise closed — its
    ``H5EquationConstraintDeviationWarning`` is now dormant (ADR 0068, Open
    item 4 resolved)."""


#: Back-compat alias — the warning was originally named for g.reinforce
#: (which no longer warns; its ties persist via the neutral zone). Retained
#: so existing imports keep working; new code should use
#: :class:`H5FeatureDeferredWarning`.
H5ReinforceDeviationWarning = H5FeatureDeferredWarning


class H5RefusedVerb(NotImplementedError):
    """``H5Emitter`` refuses a verb whose ``VERBS`` row is ``refuse`` (ADR 0114 D1).

    Raised by :meth:`H5Emitter._refuse`, the one path every refusal takes,
    so ``apeSees.h5()`` fails loud instead of writing an archive that
    silently lacks the call. ``verb`` is the Protocol verb and ``row`` its
    ``VERBS`` row; ``detail`` says what the call carried and what to use
    instead. A ``NotImplementedError`` subclass: the refusal means "no
    store yet", and the existing ``except NotImplementedError`` sites keep
    catching it.
    """

    def __init__(self, verb: str, row: Verb, detail: str) -> None:
        self.verb = verb
        self.row = row
        super().__init__(
            f"H5Emitter refuses {verb!r} (VERBS row: via={row.via}, "
            f"scope={row.scope}, h5={row.h5}, ADR 0114 D1). {detail}"
        )


#: Schema version string emitted in ``/meta/schema_version``. Bump
#: ``MAJOR`` only on a breaking change; ``MINOR`` for additive groups
#: (such as the ``/beam_integration`` group introduced alongside the
#: Protocol's ``beamIntegration`` method in Phase 4.5).
#:
#: History:
#:   Reading the history: entries that say "per ADR 0023 two-version
#:   reader window, both N-1 and N files are accepted" record the rule
#:   in force when that minor shipped. ADR 0113 retired the window; a
#:   reader now opens every minor from :data:`SCHEMA_FLOOR` (2.12.0) up
#:   to the current one and refuses a newer minor (INV-4).
#:
#:   * 1.0.0 — Phase 6 initial release.
#:   * 1.1.0 — added ``/beam_integration`` group + widened fiber-layer
#:     ``line`` field from float[4] to float[6].
#:   * 2.0.0 — Phase 8.4: bridge-written groups (materials, sections,
#:     transforms, beam_integration, time_series, patterns, bcs,
#:     recorders, analysis) moved under ``/opensees/``.  ``/meta`` and
#:     ``/elements`` stay at root.
#:   * 2.1.0 — Phase 8.5: broker-side neutral zone added (``/nodes``,
#:     ``/elements/{type}``, ``/physical_groups``, ``/labels``,
#:     ``/constraints/{kind}``, ``/loads/{kind}/{pattern}``,
#:     ``/masses``).  ``/elements`` writing handed from bridge to
#:     broker; the bridge no longer emits this group.  Additive — old
#:     v2.0.0 readers still work, they just don't see the new groups.
#:   * 2.2.0 — Phase 8.6: ``fem_eids`` int64 dataset added under each
#:     ``/opensees/element_meta/{type_token}/`` group, parallel to
#:     ``ids``.  Carries the FEM element id each OpenSees tag was
#:     fanned out from (master plan §3 "tag_map").  Sentinel ``-1``
#:     entries mark records emitted outside a bridge fan-out.
#:     Additive — old v2.1.0 readers ignore the new dataset.
#:   * 2.3.0 — Phase 9 commit 6: unified ``/opensees/recorders/``
#:     archive.  Every record group gains a ``kind`` attr —
#:     ``"typed"`` for Node / Element / MPCO primitives (1:1
#:     OpenSees), ``"declared"`` for fan-out calls produced by
#:     ``ops.recorder.declare(...)``. Declared records also carry
#:     the original declaration metadata as attrs:
#:     ``declaration_name``, ``record_name``, ``category``,
#:     ``components``, ``raw``, ``pg``, ``label``, ``selection``,
#:     ``ids``, ``dt``, ``n_steps``, ``file_root``.  Additive — old
#:     v2.2.0 readers see ``kind="declared"`` records as well-formed
#:     recorder groups (they just ignore the extra attrs).
#:   * 2.4.0 — Phase 8.7 commit 2: ``/mesh_selections/`` neutral-zone
#:     group added, mirroring ``/physical_groups`` / ``/labels`` shape.
#:     Carries post-mesh selection sets (``g.mesh_selection`` →
#:     ``fem.mesh_selection``) so the viewer's ``selection=`` selector
#:     round-trips through ``model.h5``.  Additive — old v2.3.0 readers
#:     ignore the new group and lose only the ``selection=`` round-trip
#:     convenience (live mesh_viewer sessions still consult the live
#:     ``fem.mesh_selection`` directly).
#:   * 2.5.0 — apeGmsh.cuts v4: ``/opensees/cuts/`` and
#:     ``/opensees/sweeps/`` groups added for SectionCutDef and
#:     SectionSweepDef persistence (writer lives in
#:     :mod:`apeGmsh.cuts._h5_io`).  Additive — pre-v4 readers ignore
#:     the new groups.
#:   * 2.6.0 — Phase 6 (ADR 0021): the ``/meta/lineage/`` sub-group
#:     is stamped alongside the bridge zone by
#:     :func:`_compose_model_h5`.  Additive — the lineage attrs are
#:     readable by 2.5.x consumers as opaque extra meta children
#:     (the bridge reader ignores them; the read side surfaces a
#:     "lineage absent" warning when the sub-group is missing on a
#:     legacy file).  Per ADR 0023 two-version reader window, both
#:     2.5.x and 2.6.x files are accepted.
#:   * 2.7.0 — Phase 7b (ADR 0022): MP constraint emission fanout.
#:     The H5 emitter gained the ``/opensees/constraints/`` group
#:     with four compound-dtype datasets — ``equalDOF``,
#:     ``rigidLink``, ``rigidDiaphragm``, ``embeddedNode`` — one
#:     row per :meth:`equalDOF` / :meth:`rigidLink` /
#:     :meth:`rigidDiaphragm` / :meth:`embeddedNode` call.  Each
#:     dataset carries the constraint's ``name`` field (declaration
#:     label round-trip, INV-2) and a ``phantom_node_tags`` int64
#:     dataset records phantom-node tags written via
#:     :meth:`node` with ``ndf=6`` so a reader can identify which
#:     nodes were synthesized by the phantom-emit pre-step (INV-3).
#:     Additive — old 2.6.x readers ignore the new group.  Per ADR
#:     0023 two-version reader window, both 2.6.x and 2.7.x files
#:     are accepted.
#:   * 2.8.0 — embeddedNode field rename: the second compound-dtype
#:     column of ``/opensees/constraints/embeddedNode`` was
#:     ``embedding_ele`` in 2.7.0 (a misnomer — the value is the
#:     constrained / slave node id, not an element id) and is
#:     ``cnode`` from 2.8.0 onward (matches the OpenSees ``$Cnode``
#:     vocabulary).  Stored values are unchanged; only the column
#:     name flips.  No in-repo reader consumed the 2.7.0 name, so
#:     no compat shim is required; the two-version reader window
#:     accepts 2.7.x and 2.8.x files but the column name is
#:     version-dependent.
#:   * 2.9.0 — ADR 0024 (Emitter Protocol widening for ``region()``):
#:     the H5 emitter gained the ``/opensees/regions/`` group, with
#:     one ``region_NNN`` sub-group per :meth:`region` call carrying
#:     the integer ``tag`` attribute plus a ``params`` dataset
#:     mirroring the OpenSees flag tail (``-node n1 n2 ...``,
#:     ``-ele e1 e2 ...``, etc.).  Auto-emitted today by the
#:     ``MPCO`` recorder fan-out (``nodes_pg=`` / ``elements_pg=``)
#:     so MPCO output filtered by ``-R $tag`` round-trips through
#:     ``OpenSeesModel.from_h5(...).build("tcl", ...)``.  Additive —
#:     old 2.8.x readers ignore the new group and lose only the
#:     filter round-trip (the MPCO recorder line still carries the
#:     dangling ``-R`` reference; the bridge raises if the
#:     referenced region is missing at re-emit time).  Per ADR 0023
#:     two-version reader window, both 2.8.x and 2.9.x files are
#:     accepted.
#:   * 2.10.0 — ADR 0027 (cross-partition MP-constraint emission policy):
#:     the H5 emitter gained the ``/opensees/partitions/`` group plus
#:     a parallel ``partition_ids`` int64 column on every
#:     ``/opensees/element_meta/{type_token}/`` group.  The new
#:     :meth:`partition_open` / :meth:`partition_close` Protocol
#:     methods bracket per-rank emit blocks; each block populates one
#:     ``partition_NN`` sub-group with ``element_ids`` /
#:     ``node_ids`` / ``boundary_node_ids`` int64 datasets and
#:     ``rank`` / ``n_elements`` / ``n_nodes`` scalar attrs.  The
#:     parent ``/opensees/partitions/`` group carries the
#:     ``n_partitions`` scalar attr.  Boundary nodes are computed on
#:     write as the per-rank set intersected against every other
#:     rank's node set — symmetric across ranks.  Elements emitted
#:     outside a ``partition_open`` / ``partition_close`` bracket get
#:     ``partition_ids`` row value ``-1``.  Additive — old 2.9.x
#:     readers ignore the new group and the new column.  Per ADR 0023
#:     two-version reader window, both 2.9.x and 2.10.x files are
#:     accepted.
#:   * 2.11.0 — bug fix: the bridge now emits **0-based runtime ranks**
#:     (matching ``OpenSeesMP::getPID()``) instead of Gmsh's 1-based
#:     ``PartitionRecord.id``.  Side effect on this zone: per-partition
#:     group ``rank`` attrs and the parallel ``partition_ids`` column
#:     values become 0-based (``0..N-1``) instead of 1-based (``1..N``).
#:     Group naming (``partition_NN`` is just the loop index) is
#:     unchanged.  The broker's Gmsh-side 1-based ``PartitionRecord.id``
#:     is preserved verbatim (only the runtime-rank seam flipped).
#:     Breaking for any reader that mapped ``rank`` attr / partition_ids
#:     values to ``part.id`` directly; per ADR 0023 two-version reader
#:     window, both 2.10.x and 2.11.x files are accepted.
#:   * 2.12.0 — ADR 0035 (ASDEmbeddedNodeElement option exposure): the
#:     ``/opensees/constraints/embeddedNode`` compound dtype gains five
#:     typed columns — ``stiffness`` (float64, ``-K``),
#:     ``stiffness_p`` (float64, ``-KP``) + ``has_stiffness_p`` (uint8
#:     sentinel for ``None``), ``rotational`` (uint8, ``-rot``),
#:     ``pressure`` (uint8, ``-p``).  Defaults written when the user
#:     leaves the kwargs untouched match the C++ parser at
#:     ``ASDEmbeddedNodeElement.cpp:222`` (K = 1e18, ``has_stiffness_p``
#:     = 0, rot = 0, p = 0), so old decks emitted by the legacy path
#:     observe semantically-identical on-deck behaviour.  Additive —
#:     old 2.11.x readers ignore the new columns.  Per ADR 0023 two-
#:     version reader window, both 2.11.x and 2.12.x files are
#:     accepted.
#:   * 2.13.0 — bridge-side name aliases: a ``/opensees/names`` sidecar
#:     group (``name`` / ``kind`` / ``tag`` datasets) persists the
#:     ``ops.<family>.<Type>(..., name=...)`` aliases so the read side
#:     (``OpenSeesModel`` / ``Results`` / viewer) can resolve a name
#:     back to its kind+tag.  Names are labels, not structure: the
#:     group is in ``MODEL_HASH_EXCLUDED_CHILDREN``, so relabelling
#:     never perturbs ``model_hash`` (same carve-out as cuts / sweeps /
#:     regions).  Written only when at least one name is registered, so
#:     name-free files stay byte-identical to 2.12.x.  Additive — old
#:     2.12.x readers ignore the group.  Per ADR 0023 two-version
#:     reader window, both 2.12.x and 2.13.x files are accepted.
#:   * 2.14.0 — ADR 0048/0049 (PR-2): new optional ``/opensees/nodes_ndf``
#:     group (``tags`` int64 + ``ndf`` int8, aligned to the broker node
#:     order) persisting the *effective* per-node ndf the deck emits
#:     (``fem.nodes.ndf_for`` override else the ``ops.model`` envelope).
#:     The one bridge-owned ndf store that the read-side and ADR 0049
#:     ``ops.ndf`` will write/read through; folds into ``model_hash`` (the
#:     opensees zone walks every dataset), never ``fem_hash``.  Written for
#:     broker-backed files only; absent on bridge-only stubs.  Additive —
#:     old 2.13.x readers ignore it.  Per ADR 0023 two-version reader
#:     window, both 2.13.x and 2.14.x files are accepted.
#:   * 2.15.0 — ADR 0053 (D3b): new ``/opensees/dampings/`` group, one
#:     ``damping_NNN`` sub-group per :meth:`damping` call carrying the
#:     ``type`` token (``Uniform`` / ``SecStif`` / ``URD`` / ``URDbeta``)
#:     + integer ``tag`` attr + a ``params`` dataset mirroring the
#:     OpenSees argument tail.  Closes the D3a gap where ``damping``
#:     objects were lost on a ``model.h5`` round-trip (the method was a
#:     no-op).  Replayed by :func:`_replay_into` after ``time_series``
#:     (a ``-factor`` series dependency resolves first) and before
#:     elements (an element's ``-damp $tag`` rides in its own arg tail),
#:     so element-attached damping survives ``from_h5 → build``.  These
#:     are authored model state, so the group **folds into**
#:     ``model_hash`` (unlike the regenerable regions/names carve-outs).
#:     Region-based ``-damp`` / ``-rayleigh`` attachments share the
#:     pre-existing ``/opensees/regions`` limitation (archival, not
#:     re-emitted).  Additive group, but the version bump is a producer
#:     **hard floor** (see 2.16.0 below for the corrected window
#:     semantics): a 2.14.x reader REFUSES a 2.15.x file
#:     (``file.minor > reader.minor`` → ``SchemaVersionError``,
#:     :func:`schema_version.validate_zone_version`).  The window only
#:     lets a *newer* reader open an *older* file, never the reverse —
#:     the earlier "old readers ignore the new group" phrasing in these
#:     bullets is inaccurate and is corrected from 2.16.0 onward.
#:   * 2.16.0 — ADR 0055 Phase 1 (global initial-stress archival): new
#:     ``/opensees/initial_stress`` group, one ``stress_NNN`` sub-group
#:     per global ``ops.initial_stress(...)`` record carrying the
#:     declarative field set (``name`` + ``sigma_xx/yy/zz`` +
#:     ``ramp_steps`` + ``lambda_install`` attrs, and EITHER a ``pg``
#:     attr XOR an ``elements`` int64 dataset).  Pre-resolve / declarative
#:     (no parameter tags, no rendered ramp proc): replay re-runs
#:     :func:`emit_initial_stress_global` /
#:     :func:`emit_initial_stress_addtoparameter` to regenerate the
#:     deck byte-identically.  Closes the global-bucket half of the
#:     ``apeSees.h5`` fail-loud guard (the staged bucket stays loud).
#:     Authored model state → **folds into** ``model_hash``.  Written
#:     only when at least one global initial-stress record exists, so
#:     vanilla files stay byte-identical to 2.15.x.
#:     **Window semantics (hard floor, applies to every bump):** once
#:     the writer stamps 2.16.0, every file it writes — vanilla included
#:     — is 2.16.0, and a still-deployed 2.15.0 reader REFUSES it
#:     (newer-minor branch of
#:     :func:`schema_version.validate_zone_version`).  All readers must
#:     be ≥ 2.16.0 to open any file written after this lands; the window
#:     only buys a 2.16 reader the ability to open 2.15 files, NOT the
#:     reverse.
#:   * 2.17.0 — ADR 0049 (node-pair zeroLength): new optional
#:     ``inline_connectivity`` vlen-int64 dataset under
#:     ``/opensees/element_meta/{type}/`` carrying the endpoint node tags
#:     of node-pair elements (``ops.element.ZeroLength(nodes=...)`` etc.),
#:     which have no backing gmsh cell in the neutral ``/elements`` zone
#:     and so cannot source their connectivity from there on re-emit.
#:     Written only when a type group has ≥1 node-pair (``fem_eid < 0``)
#:     row, so PG-only models are byte-identical and their ``model_hash``
#:     is unchanged.  Folds into ``model_hash`` (connectivity is
#:     model-defining).  Additive group; the hard-floor window semantics
#:     above apply (a 2.16.x reader REFUSES a 2.17.x file; a 2.17 reader
#:     opens 2.16 and 2.17 files).
#:   * 2.18.0 — ADR 0055 Phase 2 (staged-model archival, non-partitioned):
#:     new ``/opensees/stages`` group, one ``stage_NNN`` sub-group per
#:     ``ops.stage(...)`` block in registration order (zero-padded so
#:     name-sorted order == replay order).  Each stage persists the
#:     captured RESOLVED per-stage emit stream (owned node/element tags,
#:     fix/mass, region lines, MP constraints, patterns incl. HOLD
#:     ``sp_holds``, recorders, rayleigh, SSI-2.E mutators, the per-stage
#:     analysis chain + analyze loop) plus the DECLARATIVE complement
#:     (``activated_pgs``, per-stage ``initial_stress`` and
#:     ``activate_absorbing`` sub-tables — same pre-resolve field-set rule
#:     as 2.16.0).  Tri-state mutators are presence-encoded: a never-set
#:     ``set_time`` / ``set_creep_on`` / ``pre_analyze_reset`` simply has
#:     no attribute.  Staged files carry NO global ``/opensees/analysis``
#:     group — each stage's chain is scoped to its own ``stage_NNN``
#:     group.  Closes the staged-bucket half of the ``apeSees.h5``
#:     fail-loud guard for NON-PARTITIONED builds only (partitioned
#:     staged stays loud — ADR 0055 Phase 5).  Authored model state →
#:     **folds into** ``model_hash``.  Written only when at least one
#:     stage exists, so vanilla files stay byte-identical to 2.17.x.
#:     The hard-floor window semantics above apply (a 2.17.x reader
#:     REFUSES a 2.18.x file; a 2.18 reader opens 2.17 and 2.18 files).
#:   * 2.19.0 — ADR 0055 Phase 5 (P5.1, partitioned staged archival):
#:     NO layout change — the bump marks that PARTITIONED staged
#:     archives now exist (the last ``apeSees.h5`` fail-loud guard is
#:     lifted).  The ``/opensees/stages`` zone of a partitioned build
#:     is rank-agnostic by construction: per-rank replicated emission
#:     (fix/mass/MP/remove_sp/HOLD lines, ADR 0027 INV-1/INV-4)
#:     dedupes to one captured record; per-rank pattern and stage-
#:     region fragments merge by tag (member union, first-occurrence
#:     order); foreign ghost-node declarations (INV-2) never enter
#:     ``owned_node_ids``.  Capture order within a stage is RANK-MAJOR
#:     (rank 0's owned subset first) — content-equal, not byte-equal,
#:     to the same model captured unpartitioned.  ``/opensees/
#:     partitions`` carries exactly one ``partition_NN`` group per
#:     rank (stage re-brackets RESUME the rank's accumulator), now
#:     including stage-owned topology.  Authored model state →
#:     **folds into** ``model_hash``.  Vanilla and flat-staged files
#:     stay byte-identical to 2.18.x.  The hard-floor window
#:     semantics above apply (a 2.18.x reader REFUSES a 2.19.x file;
#:     a 2.19 reader opens 2.18 and 2.19 files).
#:   * 2.20.0 — ADR 0078 Amendment A1 (ComputedSection provenance):
#:     additive — new optional ``/opensees/computed_sections`` sidecar
#:     (``tag`` / ``analyzer_name`` / JSON ``payload``), written only
#:     when a ``ComputedSection`` emitted, so every other file stays
#:     byte-identical to 2.19.x.  Provenance metadata, not authored
#:     model state → in ``MODEL_HASH_EXCLUDED_CHILDREN`` (same
#:     carve-out as ``names``).  Standard additive-minor window
#:     semantics: a 2.20 reader opens 2.19 and 2.20 files; a 2.19.x
#:     reader REFUSES a 2.20.x file (INV-4 — no forward tolerance),
#:     and a 2.18.x stamp is now below the hard floor.
#:   * 2.21.0 — Phase SSI-2.E (``s.update_material_stage``): additive
#:     — new optional ``update_material_stage`` dataset under
#:     ``/opensees/stages/stage_NNN/``, written only when a stage
#:     actually flips a SANISAND material, so every other file stays
#:     byte-identical to 2.20.x.  Authored model state, not provenance
#:     → folds into ``model_hash``.  Standard additive-minor window
#:     semantics: a 2.21 reader opens 2.20 and 2.21 files; a 2.20.x
#:     reader REFUSES a 2.21.x file (INV-4 — no forward tolerance).
#:   * 2.22.0 — ADR 0112 amendment 5 (#1304, ``mass_from_model()``
#:     archival): additive — new optional ``/opensees/bcs@mass_from_model``
#:     attr (int8, value 1), written only when the bridge declared
#:     ``mass_from_model()`` and the snapshot carries masses.  The H5
#:     emit then skips the mass stream (the masses already persist in the
#:     neutral zone's ``/masses``), and the marker tells replay
#:     (``OpenSeesModel.build``) to stream ``fem.nodes.masses`` instead
#:     of silently dropping them.  Every other file stays byte-identical
#:     to 2.21.x.  Authored model state → folds into ``model_hash``.
#:     Standard additive-minor window semantics: a 2.22 reader opens
#:     2.21 and 2.22 files; a 2.21.x reader REFUSES a 2.22.x file.
#:   * 2.23.0 — ADR 0114 R2/R3a (K1-4): additive — new ``/opensees/program``
#:     table (the emit order: one run-length row per stretch of calls,
#:     ``first, count, method, store, row, stage, decl``) and an optional
#:     ``/opensees/commands`` store, written only when a call has no typed
#:     store (global ``rayleigh``, ``eigen``, ``modal_damping``, a stage's
#:     ``profiler`` bracket).  Both fold into ``model_hash``, so an
#:     identical model hashes differently once at this minor.  A 2.23
#:     reader opens 2.12 through 2.23; a 2.22.x reader REFUSES a 2.23.x file.
#:   * 2.24.0 — ADR 0114 D6/R4 (K1-5): additive — five optional
#:     attributes on ``/opensees`` itself: ``@will_solve`` (int8 0/1,
#:     ``staged or any(Analysis)`` at emit), ``@solve_mode`` (str, the
#:     archive's own partition mode, ``flat`` or ``partitioned``),
#:     ``@solve_refusals`` (vlen str, the ids of the solve-time gates
#:     that would refuse under that mode), ``@solve_refusals_flat`` (the
#:     same gates' verdict under the flat mode every replay target
#:     emits) and ``@requires`` (vlen str, the sorted union of the
#:     emitted verbs' ``VERBS.requires`` tokens).  Written together,
#:     only once the bridge handed the emitter the stamp
#:     (``set_solve_stamp``); a file without them reads as "no stamp"
#:     (``None``), never as a default.  Attributes of ``/opensees`` fold
#:     into ``model_hash``.
#:     A 2.24 reader opens 2.12 through 2.24; a 2.23.x reader REFUSES a
#:     2.24.x file.
#:   * 2.25.0 — ADR 0114 R5 (K1-6): additive — new optional
#:     ``/opensees/decls`` group, the bridge's declarations: one row per
#:     declaration (``key`` = ``opensees/<family>/<name|#k>``, the
#:     ``/provenance`` path; ``family``; ``name``, ``""`` when unnamed;
#:     ``synth``), the ``tags`` runs (``kind``, ``first``, ``count``,
#:     ``decl``) joining the tagged ones on the tag plan's ``(kind, tag)``
#:     (a fan-out element or transform inherits its spec's row), and
#:     ``rows/<store>`` columns giving the declaration of every row of the
#:     tagless stores (``bcs/fix``, ``bcs/mass``, ``recorders`` and their
#:     ``stages/stage_NNN`` twins). Carries the new ``name=`` of
#:     ``fix`` / ``mass`` (flat and staged) and the recorders' forwarded
#:     one. A label, not structure: ``decls`` is in
#:     ``MODEL_HASH_EXCLUDED_CHILDREN``, so ``model_hash`` is unchanged
#:     and ``/opensees/program``'s hashed ``decl`` column stays ``-1``.
#:     Written only when the bridge drove the emit (``set_declarations``).
#:     A 2.25 reader opens 2.12 through 2.25; a 2.24.x reader REFUSES a
#:     2.25.x file.
#:   * 2.26.0 — ADR 0114 A6/Q4 (K1-7): additive — new optional
#:     ``/opensees/decl_params`` group hanging off ``decls``: every
#:     declaration's parameters **by field name** (one row per ``decls``
#:     row: a primitive, a fix / mass / region / damping / initial-stress
#:     / equation-constraint record, a model-wide declaration), encoded
#:     generically from ``dataclasses.fields(owner)`` (``decl`` = the
#:     ``decls`` row; ``type`` = the owner's class; ``params`` = one
#:     JSON object per row, a reference to a primitive stored as its
#:     declaration key ``{"$decl": key}``; ``params_names`` = the field
#:     name of every argv slot of the declaration's store row when the
#:     argv equals the fields, ``""`` otherwise; ``transf_ref/``,
#:     ``integration_ref/``, ``section_ref/`` = index runs (``first``,
#:     ``count``, ``key``) of every ``GeomTransf`` / ``BeamIntegration``
#:     / ``Section`` key the row references, in field order). A derived view of the hashed
#:     stores: ``decl_params`` is in ``MODEL_HASH_EXCLUDED_CHILDREN``, so
#:     ``model_hash`` is unchanged. Written only when the bridge handed
#:     the primitives in (``set_decl_params``); an unknown field shape
#:     refuses the write. A 2.26 reader opens 2.12 through 2.26; a
#:     2.25.x reader REFUSES a 2.26.x file.
#:   * 2.27.0 — K1-8 (ADR 0114 D4 amended): additive — the
#:     parameter tags the build planned are archived per record, and the
#:     two verbs the writer refused before archive. ``param_tags`` int64
#:     (3,) attr on every ``initial_stress/stress_NNN`` (global and
#:     stage); a ``param_tags`` (k, 2) int64 dataset of ``(rank, pid)``
#:     rows on every ``stages/stage_NNN/activate_absorbing/absorb_NNN``;
#:     new optional ``stages/stage_NNN/update_parameter/update_NNN``
#:     (``s.update_parameter``: ``name``, ``value``, optional ``mat_tag``,
#:     ``pg`` XOR ``elements``, and its ``param_tags`` rows) and
#:     ``stages/stage_NNN/zero_velocities/zero_NNN`` (``s.zero_velocities``:
#:     an optional ``nodes`` dataset, absent for the whole domain). A
#:     replay of a 2.27 archive reads the stored tags and mints none; one
#:     of an older archive keeps re-minting them (``compose.py``, behind
#:     :data:`~apeGmsh.opensees.emitter.h5_reader.PARAM_TAGS_FROM`).
#:     Authored model state: the tags fold into ``model_hash``, so an
#:     initial-stress or absorbing archive hashes differently once at this
#:     minor; a file without either is unchanged. A 2.27 reader opens 2.12
#:     through 2.27; a 2.26.x reader REFUSES a 2.27.x file.
#:   * 2.28.0 — ADR 0115 amendment R2-f: additive — the
#:     ``decl_params`` rows of ``FixRecord`` / ``MassRecord`` and the typed
#:     ``recorder.Node`` / ``recorder.Element`` gain a ``label`` field, the
#:     names a ``label=`` declaration was given (``null`` otherwise).
#:     Provenance only: ids stay authoritative and replay never
#:     re-resolves a label. No path or dtype changes; ``decl_params`` is
#:     hash-excluded. A 2.28 reader opens 2.12 through 2.28; a 2.27.x
#:     reader REFUSES a 2.28.x file.
#:   * 2.29.0 — K2b (ADR 0114 Q3): additive — new optional
#:     ``/opensees/constraints/embeddedRebar`` group, one row per
#:     ``g.reinforce`` ``LadrunoEmbeddedRebar`` tie the build wrote,
#:     formerly a ``ledger`` verb: ``ele_tag`` (int64, the planned tie
#:     tag), ``name`` (the ``mp_constraint_comment`` label, ``""`` when
#:     none) and the ragged positional arguments as in
#:     ``/opensees/commands`` (``arg_offsets``, ``args``, ``args_str``,
#:     ``arg_kinds``). A replay of a 2.29 archive writes the stored rows
#:     and mints no tie tag; one of an older archive keeps re-deriving
#:     the ties from the neutral zone (``compose.py``, behind
#:     :data:`~apeGmsh.opensees.emitter.h5_reader.EMBEDDED_REBAR_FROM`).
#:     Hashed: a reinforced archive hashes differently once at this
#:     minor; a file without a tie is unchanged. A 2.29 reader opens 2.12
#:     through 2.29; a 2.28.x reader REFUSES a 2.29.x file.
#:   * 2.30.0 — R3-S5 (maintainer ruling D7): additive — the
#:     ``decl_params`` rows of the typed ``recorder.Node`` /
#:     ``recorder.Element`` gain a ``precision`` field, the ``-precision N``
#:     significant digits (``null`` when unset, and then the deck carries
#:     no flag). No path or dtype changes; ``decl_params`` is
#:     hash-excluded. A 2.30 reader opens 2.12 through 2.30; a 2.29.x
#:     reader REFUSES a 2.30.x file.
#:   * 2.31.0 — A7: the ``/opensees/program`` ``decl`` column, ``-1`` on
#:     every run below this minor, holds the ``/opensees/decls`` row of the declaration that
#:     made each call when the emit proves it: a tagged call by its
#:     ``(kind, tag)`` run, a tagless call (``fix``, ``mass``,
#:     ``recorder``, the stage removals and flips, ``sp_hold``,
#:     ``rayleigh`` and the ``/opensees/commands`` rows) by the
#:     declaration open on it. Everything else stays an explicit ``-1``
#:     (``h5-schema.md`` lists the cases). No path or dtype changes; runs
#:     now also break where ``decl`` changes. The column is hashed, so
#:     every archive the bridge writes hashes differently once at this
#:     minor; a rewrite echoes its source's column (an older source's
#:     ``-1`` values included), and the hash-excluded
#:     ``/opensees/decls@program_decl`` marker (``"complete"`` from an
#:     emit, ``"absent"`` echoed by a rewrite of an older file) says which
#:     it is. A 2.31 reader opens 2.12 through 2.31 and
#:     refuses a ``decl`` outside ``{-1} ∪ [0, n_decls)``; a 2.30.x reader
#:     REFUSES a 2.31.x file.
#:   * 2.32.0 — K (side-channel decls): the six side-channel verbs
#:     (``step_hook_ramp``, ``addToParameter``, ``flip_element_stage``,
#:     ``update_parameter``, ``set_node_vel``, ``set_node_accel``) carry
#:     the ``/opensees/program`` ``decl`` of the record that drove them,
#:     which the bridge opens around each record's calls; ``-1`` before
#:     this minor. A ``set_node_vel`` / ``set_node_accel`` call on a node
#:     two ``s.zero_velocities`` records name is the first record's (the
#:     deck zeroes it once). No path or dtype changes, and the
#:     ``/opensees/decls@program_decl`` marker keeps its meaning
#:     (``"complete"``: every provable join recorded). The column is
#:     hashed, so an archive with any of these calls (an initial stress,
#:     an absorbing flip, ``s.update_parameter``, ``s.zero_velocities``)
#:     hashes differently once at this minor; any other archive is
#:     unchanged. A 2.32 reader opens 2.12 through 2.32; a 2.31.x reader
#:     REFUSES a 2.32.x file.
#:   * 2.33.0 — value requirement tokens: additive — two
#:     new ``/opensees@requires`` tokens, derived from argument values by
#:     the stamp writer (``H5Emitter.solve_stamp``): ``"mpco"`` (any
#:     ``recorder mpco``) and ``"mpi"`` (a Mumps-family or ``Parallel*``
#:     ``system`` without a runtime fallback, or a partitioned
#:     ``@solve_mode``). No path or dtype changes. Below this
#:     minor ``@requires`` names verb tokens only, and keeps that meaning
#:     on read. ``@requires`` is hashed, so an archive with an MPCO
#:     recorder, a Mumps system or partitions hashes differently once at
#:     this minor; any other archive is unchanged. A 2.33 reader opens
#:     2.12 through 2.33; a 2.32.x reader REFUSES a 2.33.x file.
#:   * 2.34.0 — large arrays as datasets: additive — a ``params`` /
#:     ``params_str`` pair, or a declared recorder's ``ids``, that HDF5
#:     refuses as an attribute for size (an object-header message holds
#:     at most 65,535 bytes) is written as uncompressed datasets of the
#:     same names in its group. The writer tries the attribute first and
#:     falls back only on that refusal. Any other sequence attribute HDF5
#:     refuses raises by name. An array that fits keeps the attribute
#:     form, so every archive a 2.33 writer could write keeps its bytes
#:     and hash. A 2.33
#:     reader would read a dataset-form pair as no arguments, hence the
#:     bump: a 2.34 reader opens 2.12 through 2.34 and refuses a dataset
#:     form below 2.34.0; a 2.33.x reader REFUSES a 2.34.x file.
#:   * 2.35.0 — equationConstraint lines archived: additive — a new
#:     ``/opensees/constraints/equationConstraint`` group holds one row
#:     per global ``equationConstraint`` line, an ``enforce="equation"``
#:     tie's (one per tied DOF) and an ``apeSees.equation_constraint``
#:     row's alike, its ``/opensees/program`` call linking it, so a replay
#:     writes each line at its emit index. Written only when a line
#:     exists, so every other archive keeps its bytes and hash; the group
#:     is hashed, so an archive with one hashes differently once at this
#:     minor. A stage-claimed equation tie's lines stay unarchived (the
#:     replay refuses it by name). A 2.34 reader would replay the model
#:     untied, hence the bump: a 2.35 reader opens 2.12 through 2.35 and
#:     refuses the group below 2.35.0; a 2.34.x reader REFUSES a 2.35.x
#:     file.
#:   * 2.36.0 — the target mode: additive — ``/opensees@target_mode``, a
#:     string ``"fork"`` or ``"stock"``, is the bridge's pinned
#:     ``OpenSeesTarget.mode``. Written only for a
#:     pinned target; an absent attribute reads as ``"auto"``, so every
#:     older archive and every unpinned one keeps its bytes and meaning.
#:     ``binary`` / ``python`` are machine paths and are never archived.
#:     Not hashed (``MODEL_HASH_EXCLUDED_ATTRS``): the same model pinned or
#:     on auto has one ``model_hash``. A replay stamps its deck as the
#:     bridge does and treats a ``"fork"`` archive as needing the fork. A
#:     2.35 reader would replay a fork pin as auto, hence the bump: a 2.36
#:     reader opens 2.12 through 2.36; a 2.35.x reader REFUSES a 2.36.x
#:     file.
#:   * 2.37.0 — stage equationConstraint lines archived: additive — a stage
#:     block's ``constraints`` group gains ``equationConstraint`` (the
#:     global group's columns) and ``equationConstraint_emit_index``, the
#:     lines of every ``enforce="equation"`` tie the stage claims, stamped
#:     from the stage's MP sequence, so a replay writes them in the stage's
#:     own position. ``equationConstraint`` leaves the VERBS ledger (ADR
#:     0114 amendment, #2448). Written only when a stage writes a line, so
#:     every other archive keeps its bytes and hash; the rows are hashed.
#:     A 2.36 reader refuses the group by name, and a replay of a 2.36
#:     archive refuses a stage-claimed equation tie (it stores no line),
#:     hence the bump: a 2.37 reader opens 2.12 through 2.37 and refuses
#:     the stage group below 2.37.0; a 2.36.x reader REFUSES a 2.37.x
#:     file.
SCHEMA_VERSION: str = "2.37.0"

#: Oldest opensees-zone minor the reader opens (ADR 0113 (#1303)). The
#: zone's last non-additive minor is 2.11.0, the 0-based rank flip, but
#: every 2.11-era writer stamped neutral 2.7.0, below the neutral floor
#: 2.10.0, so no 2.11 file opens through today's readers: the schema
#: corpus proved it (#1329) and the maintainer raised the floor to 2.12.0
#: on 2026-10-04 (ADR 0113 D3, the evidence gate; no major bump). From
#: here on the floor moves only with a major bump;
#: :func:`schema_version.reader_floor` reads it from here.
SCHEMA_FLOOR: str = "2.12.0"


# Map known time-series type tokens to "is path-bearing": for a Path
# series the ``args`` carry numeric values; for algorithmic series we
# only record the type + scalar params.
_PATH_SERIES_TOKENS: tuple[str, ...] = ("Path",)


# Pattern type tokens that the bridge opens via ``pattern_open`` and
# closes via ``pattern_close`` with a body of load / sp / eleLoad calls.
_BLOCK_PATTERN_TOKENS: tuple[str, ...] = ("Plain", "MultiSupport")


# ---------------------------------------------------------------------------
# Buffered intermediate representations
# ---------------------------------------------------------------------------
#
# Phase 3 (ADR 0019) lifted the per-record dataclasses out into
# ``apeGmsh.opensees._internal.typed_records`` so both the write path
# (this module) and the read-side broker (``OpenSeesModel``) share one
# definition.  The underscore-prefixed aliases above preserve the
# private import paths that ``model_data.py`` and the test suite
# already depend on.


# ---------------------------------------------------------------------------
# Emit order and the generic store (ADR 0114 R2/R3a, schema 2.23.0)
# ---------------------------------------------------------------------------


class Declaration(NamedTuple):
    """One bridge declaration (ADR 0114 R5, K1-6, schema 2.25.0).

    ``key`` is the declaration path ``opensees/<family>/<name|#k>`` (or a
    synthesised ``<verb>:<owner>[/<role>]`` key), the same path the
    bridge's ``/provenance`` record carries (ADR 0112 D3). ``family`` is
    the bridge allocator kind (``uniaxialMaterial``, ``element``, ...) or
    the tagless verb (``fix``, ``mass``). ``name`` is the user's
    ``name=``, ``""`` when unnamed. ``synth`` marks an object apeGmsh
    synthesised inside a user verb (the HOLD series of ``s.support``).
    """

    key: str
    family: str
    name: str
    synth: bool


#: ``/opensees/decls/tags`` row: ``(kind, first, count, decl)``, the
#: tags ``first .. first + count - 1`` of allocator kind ``kind`` belong
#: to declaration ``decl`` (an index into the declaration rows).
DeclTagRun = tuple[str, int, int, int]


class H5DeclParamsError(TypeError):
    """A primitive field holds a shape ``/opensees/decl_params`` cannot
    store by name (K1-7). Raised at write, never skipped: an archive that
    silently dropped a field would read back as a different declaration.
    """


#: The non-dataclass objects a primitive field may hold, by qualified
#: class name, stored as ``{"$opaque": <class>}``: their lowering lives in
#: the hashed stores (a transform's per-element ``vecxz``, the
#: ``computed_sections`` sidecar), so the name is the whole record. Any
#: other object refuses.
_OPAQUE_FIELD_TYPES: frozenset[str] = frozenset({
    "apeGmsh.opensees._orientation.Cartesian",
    "apeGmsh.opensees._orientation.Cylindrical",
    "apeGmsh.opensees._orientation.Spherical",
    "apeGmsh.opensees._orientation.AlongBeam",
    "apeGmsh.sections._analysis.SectionProperties",
})

#: The ``decls`` families a ``*_ref`` column of ``decl_params`` names:
#: every reference to a primitive of that family among the row's fields,
#: in dataclass field order and depth-first within a field (a ``Hinge*``
#: integration's ``section_i``, ``section_j``, ``section_interior``;
#: the sections of a tuple field in tuple order).
_REF_COLUMNS: tuple[tuple[str, str], ...] = (
    ("transf_ref", "geomTransf"),
    ("integration_ref", "beamIntegration"),
    ("section_ref", "section"),
)

#: The tagged stores whose rows carry one flat argv per tag, the domain
#: of ``params_names`` (ADR 0114 Q4): the archive names a declaration's
#: argv slots only where its store row *is* the fields. An element row
#: carries node tags, a complex section (``Fiber``) spans several calls,
#: a chain component writes no tag: unnamed.
_ARGV_STORE_KINDS: frozenset[str] = frozenset({
    "uniaxialMaterial", "nDMaterial", "section", "beamIntegration",
    "timeSeries", "damping",
})


def _qualname(value: object) -> str:
    t = type(value)
    return f"{t.__module__}.{t.__qualname__}"


def encode_decl_params(
    prim: object, key_of: Callable[[object], str],
) -> dict[str, Any]:
    """Encode ``prim``'s fields by name for ``/opensees/decl_params`` (K1-7).

    ``prim`` is a registered primitive (a dataclass). ``key_of`` maps a
    primitive another field references to its declaration key and raises
    :class:`KeyError` for one the bridge never registered. The result is
    a JSON-ready mapping, field name -> value, in field order:

    * ``None``, ``bool``, ``int``, ``float`` (finite), ``str``: as is;
    * a :class:`Primitive`: ``{"$decl": key}``;
    * a tuple or list: a list, element by element (nested lists kept);
    * a mapping with ``str`` keys: an object, key by key;
    * a non-primitive dataclass (a ``Fiber`` patch, a ``ShellLayer``):
      ``{"$struct": <class>, "fields": {...}}``;
    * an object in :data:`_OPAQUE_FIELD_TYPES`: ``{"$opaque": <class>}``.

    Every other shape (an ``ndarray``, a set, an ``Enum``, a non-finite
    float, a mapping keyed by anything but a plain ``str``, an unlisted
    object) raises :class:`H5DeclParamsError` naming the field.
    """
    import dataclasses

    if not dataclasses.is_dataclass(prim) or isinstance(prim, type):
        raise H5DeclParamsError(
            f"decl_params: {type(prim).__name__} is not a dataclass "
            "primitive; every registered primitive is one (P12).")
    out = {
        f.name: _encode_field(
            getattr(prim, f.name), key_of, f"{type(prim).__name__}.{f.name}")
        for f in dataclasses.fields(prim)
        if not (f.name == "_ele_loads_" and not getattr(prim, "_ele_loads_", ()))
    }
    ele_loads = getattr(prim, "_ele_loads_", ())
    if ele_loads and "_ele_loads_" not in out:
        out["_ele_loads_"] = _encode_field(
            ele_loads, key_of, f"{type(prim).__name__}._ele_loads_")
    return out


def _encode_field(
    value: Any, key_of: Callable[[object], str], where: str,
) -> Any:
    import dataclasses
    import enum
    import fractions
    import math
    import numbers

    from .._internal.types import Primitive

    if isinstance(value, (enum.Enum, fractions.Fraction)):
        # A str- or int-Enum passes the scalar checks below as its base
        # value, a Fraction as a float: both would read back as another
        # type, so neither is stored by name.
        raise H5DeclParamsError(
            f"decl_params: {where} is a {_qualname(value)} ({value!r}); an "
            "Enum member or a Fraction has no exact JSON form, so the field "
            "cannot be stored by name.")
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        v = float(value)
        if not math.isfinite(v):
            raise H5DeclParamsError(
                f"decl_params: {where} is {v!r}; a non-finite value has no "
                "exact JSON form, so the field cannot be stored by name.")
        return v
    if isinstance(value, Primitive):
        try:
            return {"$decl": key_of(value)}
        except KeyError:
            raise H5DeclParamsError(
                f"decl_params: {where} holds a {type(value).__name__} the "
                "bridge never registered, so it has no declaration key; "
                "construct it through the bridge (P11).") from None
    if isinstance(value, (tuple, list)):
        return [_encode_field(v, key_of, f"{where}[{i}]")
                for i, v in enumerate(value)]
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for k, v in value.items():
            if type(k) is not str or k.startswith("$"):
                raise H5DeclParamsError(
                    f"decl_params: {where} is keyed by {k!r}; a mapping "
                    "field is stored as an object, so its keys are plain "
                    "str that do not start with '$'.")
            out[k] = _encode_field(v, key_of, f"{where}[{k!r}]")
        return out
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            "$struct": type(value).__name__,
            "fields": {
                f.name: _encode_field(
                    getattr(value, f.name), key_of, f"{where}.{f.name}")
                for f in dataclasses.fields(value)
            },
        }
    if _qualname(value) in _OPAQUE_FIELD_TYPES:
        return {"$opaque": type(value).__name__}
    raise H5DeclParamsError(
        f"decl_params: {where} holds a {_qualname(value)}, a shape "
        "/opensees/decl_params does not store by name (scalars, "
        "primitives, tuples, str-keyed mappings, value dataclasses and "
        "the listed opaque objects are); the archive refuses rather than "
        "drop the field.")


def _argv_slots(
    name: str, value: Any, tag_of_key: Callable[[str], int | None],
) -> "list[tuple[str, Any]] | None":
    """The ``(slot name, value)`` pairs an encoded field contributes to a
    flat argv, or ``None`` when the field has no slot form: a ``bool``
    (a flag), a mapping, a struct, an opaque object, or a reference whose
    declaration has no single tag. ``None`` contributes nothing (an
    omitted optional)."""
    if value is None:
        return []
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float, str)):
        return [(name, value)]
    if isinstance(value, list):
        out: list[tuple[str, Any]] = []
        for i, v in enumerate(value):
            slots = _argv_slots(f"{name}[{i}]", v, tag_of_key)
            if slots is None:
                return None
            out.extend(slots)
        return out
    if isinstance(value, dict) and set(value) == {"$decl"}:
        tag = tag_of_key(value["$decl"])
        return None if tag is None else [(name, tag)]
    return None


def decl_argv_names(
    params: Mapping[str, Any], argv: "Sequence[Any]",
    tag_of_key: Callable[[str], int | None],
) -> "tuple[str, ...] | None":
    """The field name of every slot of ``argv`` when the argv equals the
    fields (ADR 0114 Q4), else ``None``.

    ``params`` is :func:`encode_decl_params`'s output, ``argv`` the
    declaration's store row after its type token and tag, ``tag_of_key``
    the tag a referenced declaration key resolves to (``None`` when it
    has no single tag). The fields are flattened in order: a scalar is
    one slot named after its field, a sequence one slot per element
    (``points[0]``, nested as ``m_ij[0][1]``), a reference its tag, an
    omitted optional (``None``) no slot. The argv equals the fields iff
    the slot count and every value match. A ``bool`` (a flag), a mapping,
    a struct or an opaque field has no slot form, so its primitive is
    never named: argv flags without a field stay unnamed.
    """
    slots: list[tuple[str, Any]] = []
    for name, value in params.items():
        field_slots = _argv_slots(name, value, tag_of_key)
        if field_slots is None:
            return None
        slots.extend(field_slots)
    if len(slots) != len(argv):
        return None
    for (_name, want), got in zip(slots, argv):
        if isinstance(want, bool) or isinstance(got, bool):
            return None
        if isinstance(want, (int, float)) and isinstance(got, (int, float)):
            if float(want) != float(got):
                return None
        elif want != got:
            return None
    return tuple(name for name, _v in slots)


#: One ``/opensees/decl_params`` row as the writer holds it:
#: ``(decl, type, params, params_names, transf_ref, integration_ref,
#: section_ref)``, ``params`` the JSON text, ``params_names`` the JSON
#: list text or ``""``, each ``*_ref`` the declaration keys of that
#: family in field order (``()`` when the row references none).
DeclParamsRow = tuple[
    int, str, str, str, tuple[str, ...], tuple[str, ...], tuple[str, ...]]


# ---------------------------------------------------------------------------
# H5Emitter
# ---------------------------------------------------------------------------

class H5Emitter(
    _H5ModelVerbsMixin, _H5StagesMixin, _H5WriteModelMixin, _H5DeclsMixin,
):
    """Structured emitter that writes ``model.h5`` per ``h5-schema.md``.

    Construct, drive via :meth:`BuiltModel.emit`, then call
    :meth:`write` to materialize the HDF5 file. The class buffers all
    state in memory; the file is written exactly once.
    """

    #: ADR 0114 D6: the archival target. Solve-time gates do not enforce
    #: here, and partition brackets are stored, not flattened.
    caps: TargetCaps = TargetCaps(
        archival=True,
        supports_partitions=True,
        per_rank_fragments=False,
        suppress_analysis_chain_auto_emit=False,
        model_reissue_purges=False,
        emit_stage_markers=False,
    )

    def __init__(
        self,
        *,
        schema_version: str = SCHEMA_VERSION,
        model_name: str | None = None,
        apegmsh_version: str | None = None,
        snapshot_id: str | None = None,
        target_mode: TargetMode = "auto",
    ) -> None:
        self._schema_version: str = schema_version
        self._model_name: str = model_name or "model"
        self._apegmsh_version: str = apegmsh_version or "unknown"
        self._snapshot_id: str = snapshot_id or ""
        # opensees 2.36.0: the bridge's ``OpenSeesTarget.mode``,
        # written as ``/opensees@target_mode`` when pinned. Only the mode:
        # ``binary`` / ``python`` are machine paths, never archived.
        if target_mode not in TARGET_MODES:
            raise ValueError(
                f"H5Emitter(target_mode={target_mode!r}): the target mode is "
                f"one of {sorted(TARGET_MODES)} (OpenSeesTarget.mode).")
        self._target_mode: TargetMode = target_mode

        self._ndm: int | None = None
        self._ndf: int | None = None
        # The archive keeps every node's xyz, but a deck replayed from it
        # trims to ndm; refuse here what that replay would refuse (#1337).
        self._dropped_axes = DroppedAxisGuard.BEFORE_MODEL

        # Nodes — stored as parallel arrays for compact write.
        self._node_tags: list[int] = []
        self._node_coords: list[tuple[float, float, float]] = []
        # Phantom-node tags written via ``node(tag, *xyz, ndf=6)`` —
        # the per-node DOF override used by the phantom-emit pre-step in
        # ``emit_mp_constraints`` (ADR 0022 INV-3).  Persisted under
        # ``/opensees/constraints/phantom_node_tags`` so a reader can
        # identify which nodes were synthesized.
        self._phantom_node_tags: list[int] = []

        # MP constraints (Phase 7b, ADR 0022).
        self._equal_dofs: list[_EqualDOFRecord] = []
        self._rigid_links: list[_RigidLinkRecord] = []
        self._rigid_diaphragms: list[_RigidDiaphragmRecord] = []
        self._embedded_nodes: list[_EmbeddedNodeRecord] = []
        # Holds the user's declaration label set via
        # ``mp_constraint_comment(name)``; consumed by the next MP
        # constraint emit (INV-2).
        self._pending_mp_name: str = ""

        # g.reinforce (ADR 20 / R2b): the LadrunoEmbeddedRebar ties the
        # build wrote, archived from 2.29.0 (K2b) under
        # ``/opensees/constraints/embeddedRebar``. See ``embedded_rebar``.
        self._embedded_rebars: list[_EmbeddedRebarRecord] = []
        # The global equationConstraint lines (opensees 2.35.0), each a row of
        # ``/opensees/constraints/equationConstraint``. See ``equationConstraint``.
        self._equation_constraints: list[EquationConstraintRecordRO] = []

        # ADR 0114 D1: every drop of a ``ledger`` verb goes through :meth:`_ledger`, which
        # counts it here by verb. The ``ledger`` rows of ``VERBS`` are the
        # only verbs that may reach it (K2's xfail ledger; shrink-only).
        # The equation route (ADR 0068, Open item 4 RESOLVED) is one of
        # them: it round-trips via the neutral InterpolationRecord lane
        # (enforce + weights, schema 2.14.0).
        self._ledger_counts: Counter[str] = Counter()

        # ADR 0114 R2/R3a (schema 2.23.0): the emit order every Protocol
        # call notes (``/opensees/program``), and the generic store for
        # calls with no typed one (``/opensees/commands``). An H5 -> H5
        # rewrite echoes both through :meth:`restore_program`, since a
        # category-major replay cannot regenerate the original order.
        self._program = _ProgramTape()
        self._commands: list[_CommandRecord] = []
        self._command_decls: list[int] = []  # K1-6, parallel to _commands
        self._program_restored: "tuple[tuple[tuple[int, ...], ...], tuple[str, ...], tuple[str, ...], int] | None" = None

        # ADR 0114 D6 (schema 2.24.0): what the archive says about the
        # solve it was emitted for, handed in once through
        # :meth:`set_solve_stamp` (``will_solve``, the partition mode and
        # the refusal ids per mode) and written as ``/opensees@will_solve``,
        # ``@solve_mode``, ``@solve_refusals``, ``@solve_refusals_flat`` and
        # ``@requires``; the last is derived from the program's methods at
        # write time. ``None`` writes nothing.
        self._solve_stamp_input: "tuple[bool, str, tuple[str, ...], tuple[str, ...]] | None" = None

        # ADR 0114 R5 (K1-6, schema 2.25.0): the bridge's declarations,
        # handed in once through :meth:`set_declarations` with the
        # ``(kind, tag)`` runs of the tagged ones, and written as the
        # hash-excluded ``/opensees/decls``. The tagless stores (fix, mass,
        # recorders) note the declaration :meth:`set_declaration` opened
        # in a column parallel to their records; a partition replica that
        # was first captured with none open takes the replica's
        # (``_decl_dup_rows``). ``None``: no bridge declarations, no group.
        self._declarations: "tuple[Declaration, ...] | None" = None
        self._decl_tags: "tuple[DeclTagRun, ...]" = ()
        # A7: ``_decl_tags`` by kind, sorted by ``first``, as parallel
        # ``(firsts, ends, decls)`` lists, so a tagged call finds its
        # declaration by bisection (:meth:`_decl_of_tag`); and the last
        # element run hit, ``(first, end, decl)``, the hot path's cache.
        self._decl_tag_index: "dict[str, tuple[list[int], list[int], list[int]]]" = {}
        self._element_decl_run: "tuple[int, int, int]" = (0, 0, _NO_DECL)
        self._decl_open: int = _NO_DECL
        self._fix_decls: list[int] = []
        self._mass_decls: list[int] = []
        self._recorder_decls: list[int] = []
        self._decl_dup_rows: "dict[tuple[Any, ...], tuple[list[int], int]]" = {}
        # The source's ``rows`` columns, echoed by :meth:`restore_declarations`
        # in place of the live ones (a rewrite opens no declaration).
        self._decl_rows_restored: "dict[str, list[int]] | None" = None
        # A7: the source's ``/opensees/decls@program_decl`` on a rewrite
        # (``restore_declarations``); ``None`` writes ``"complete"``, this
        # emitter's own tape having recorded every call's declaration.
        self._program_decl_restored: "str | None" = None
        # The rows columns of the stores filled through side channels, not
        # Protocol calls (``initial_stress``, ``activate_absorbing``), handed
        # in by :meth:`set_declaration_rows`.
        self._decl_side_rows: "dict[str, list[int]]" = {}
        # K1-7 (schema 2.26.0): the primitives' parameters by field name,
        # ``(decl, type, params)`` per registered primitive, encoded by
        # :meth:`set_decl_params`; the ``params_names`` and ``*_ref``
        # columns are derived at write from the stores. A rewrite echoes
        # the source's rows verbatim (``_decl_params_restored``).
        # ``None``: not handed in, no group.
        self._decl_params: "list[tuple[int, str, dict[str, Any]]] | None" = None
        self._decl_params_restored: "list[DeclParamsRow] | None" = None

        # Constitutive.
        self._uniaxial: list[_MaterialRecord] = []
        self._nd: list[_MaterialRecord] = []

        # Sections.
        self._sections_simple: list[_SectionSimpleRecord] = []
        self._sections_complex: list[_SectionComplexRecord] = []
        self._open_section: _SectionComplexRecord | None = None

        # Transforms (one record per geomTransf call — see module docstring).
        self._transforms: list[_TransformRecord] = []

        # Beam integration rules (Phase 4.5).
        self._beam_integrations: list[_BeamIntegrationRecord] = []

        # Elements.
        self._elements: list[_ElementRecord] = []

        # Time series.
        self._time_series: list[_TimeSeriesRecord] = []

        # Patterns.
        self._patterns_complete: list[_PatternRecord] = []
        self._open_pattern: _PatternRecord | None = None

        # BCs (model-level).
        self._fixes: list[_FixRecord] = []
        self._masses: list[_MassRecord] = []
        # ADR 0112 amendment 5 (#1304): the bridge declared
        # ``mass_from_model()``, so the nodal masses live in the neutral
        # zone and are not streamed here.  Persisted as
        # ``/opensees/bcs@mass_from_model`` so replay re-streams them.
        self._mass_from_model: bool = False

        # Regions (emitted from the recorder fan-out; persisted so MPCO
        # ``-R $tag`` round-trips through ``OpenSeesModel.from_h5``).
        self._regions: list[_RegionRecord] = []

        # Damping objects (ADR 0053 D3b): tagged Uniform / SecStif / URD /
        # URDbeta dissipators, persisted + replayed (was a no-op in D3a).
        self._dampings: list[_DampingObjectRecord] = []

        # Global initial-stress records (ADR 0055 Phase 1).  Handed in via
        # the :meth:`set_initial_stress_records` side-channel (NOT the
        # Protocol stream — the ``addToParameter`` / ``step_hook_ramp``
        # calls the bridge drives carry resolved parameter tags + the
        # rendered ramp proc, whereas archival persists the pre-resolve
        # declarative ``InitialStressRecord`` field set).  Duck-typed: the
        # writer reads ``.name`` / ``.pg`` / ``.elements`` / ``.sigma_*`` /
        # ``.ramp_steps`` / ``.lambda_install`` without importing the
        # record class (avoids a build/ import cycle).
        self._initial_stress_records: list[Any] = []
        # K1-8 (schema 2.27.0): the global ramps' parameter tags as the
        # ``step_hook_ramp`` calls handed them in, by record name, and the
        # stored ``param_tags`` of each record above (one per record).
        self._ramp_tags: dict[str, tuple[int, int, int]] = {}
        self._initial_stress_param_tags: tuple[tuple[int, int, int], ...] = ()

        # Recorders.
        self._recorders: list[_RecorderRecord] = []
        # Active declaration context (set by ``recorder_declaration_begin``,
        # cleared by ``recorder_declaration_end``). While non-None, every
        # ``recorder(...)`` call inherits this context for schema 2.3.0
        # archival.
        self._decl_context: _DeclContext | None = None

        # Analysis chain (collected attrs).
        self._analysis_attrs: dict[str, Any] = {}
        self._analyze_call: tuple[int, float | None] | None = None

        # File-internal tag counter for ``add_oriented_elements`` (ADR
        # 0018 / ModelData declarative front-door).  Must not be mixed
        # with bridge-driven ``element()`` calls on the same instance.
        self._orientation_tag_counter: int = 0

        # Partition emission state (ADR 0027, schema 2.10.0).
        # ``_partition_current`` is the block in flight between
        # ``partition_open(rank)`` and ``partition_close()``; while
        # non-None, every :meth:`node` and :meth:`element` call also
        # appends to it.  ``_partition_blocks`` accumulates closed
        # blocks in emit order (one per rank).  ``_element_ranks`` is
        # a parallel list to ``self._elements`` carrying the rank each
        # element was emitted under (sentinel ``-1`` for outside-bracket
        # emission); it materializes the per-element ``partition_ids``
        # column on ``/opensees/element_meta/{type_token}/``.
        self._partition_current: _PartitionEmitBlock | None = None
        self._partition_blocks: list[_PartitionEmitBlock] = []
        # P5.0a (ADR 0055 Phase 5 groundwork): the global partitioned
        # pass REPLICATES records across owning ranks by design (ADR
        # 0027 INV-1/INV-4 fan-out — a shared-node ``fix`` emits inside
        # every owning rank's bracket; a cross-rank MP constraint
        # replicates byte-identically).  The H5 capture is the flat
        # logical model, so partition-bracketed captures dedupe on full
        # record identity via this seen-set.  Scoped to the GLOBAL
        # sinks only — stage-bucket dedupe/merge is the Phase 5.1
        # surface.  ``_partition_patterns_by_tag`` lets a per-rank
        # ``pattern_open`` of an already-captured tag RESUME that
        # record instead of colliding on the ``patterns/<type>_<tag>``
        # group name at write time; ``_open_pattern_resumed`` keeps the
        # resumed record from being appended to ``_patterns_complete``
        # a second time on close.
        self._partition_seen: set[tuple[Any, ...]] = set()
        self._partition_patterns_by_tag: dict[int, _PatternRecord] = {}
        self._open_pattern_resumed: bool = False
        # P5.1 (ADR 0055 Phase 5): the partitioned STAGED emit re-opens
        # rank brackets once per stage (and per follow-up pass), so a
        # rank accumulates across several ``partition_open`` calls —
        # ``_partition_block_by_rank`` resumes the rank's existing
        # block instead of growing duplicate ``partition_NN`` groups
        # (which would also corrupt the write-time boundary-node
        # intersection: two blocks of the SAME rank would see each
        # other as "another rank").  ``_partition_resumed`` keeps a
        # resumed block from being re-appended on close.  The stage
        # analogs of the P5.0a pattern-resume and region-merge state
        # are keyed by (stage name, tag) since each stage captures
        # into its own bucket.
        self._partition_block_by_rank: dict[int, _PartitionEmitBlock] = {}
        self._partition_resumed: bool = False
        self._stage_partition_patterns: "dict[tuple[str, int], _PatternRecord]" = {}
        self._stage_open_pattern_resumed: bool = False
        self._stage_partition_regions: "dict[tuple[str, int], int]" = {}

        # Stage emission state (ADR 0055 Phase 2, schema 2.18.0).
        # ``_stage_current`` is the capture bucket in flight between
        # ``stage_open(name)`` and ``stage_close()``; while non-None,
        # stage-scoped Protocol calls route into it (see
        # :class:`_StageEmitBlock`).  ``_stage_blocks`` accumulates
        # closed buckets in emit order — one per ``StageRecord``,
        # cross-validated by :meth:`set_stage_records`.
        self._stage_current: _StageEmitBlock | None = None
        self._stage_blocks: list[_StageEmitBlock] = []
        # Set by :meth:`set_stage_records`; ``_write_stages`` refuses
        # to persist captured buckets without it (the direct
        # emit-then-``write()`` path would otherwise archive staged
        # files with the declarative complement silently missing).
        self._stage_records_attached: bool = False
        self._element_ranks: list[int] = []

    # =====================================================================
    # ADR 0114 D1/D2: the one refusal, the one ledger, the one bracket
    # =====================================================================

    def _refuse(self, verb: str, detail: str) -> NoReturn:
        """Raise :class:`H5RefusedVerb` for ``verb``, a ``refuse`` row.

        Every ``refuse`` body ends here, so the lock
        (``test_verbs_lock.py``) can read the refusal off the source and
        the message always names the verb and its ``VERBS`` row.
        """
        row = VERBS[verb]
        if row.h5 != "refuse":
            raise RuntimeError(
                f"H5Emitter._refuse({verb!r}): the VERBS row is "
                f"h5={row.h5!r}, not 'refuse'; fix the row or the body."
            )
        raise H5RefusedVerb(verb, row, detail)

    def _ledger(self, verb: str) -> None:
        """Count a call to ``verb``, a ``ledger`` row, and stay silent.

        The drop is deliberate and observable: ``_ledger_counts[verb]``
        is the number of calls the archive does not carry. Only a
        ``ledger`` row may be counted here; anything else is a body that
        drops what its row says it archives.
        """
        row = VERBS[verb]
        if row.h5 != "ledger":
            raise RuntimeError(
                f"H5Emitter._ledger({verb!r}): the VERBS row is "
                f"h5={row.h5!r}, not 'ledger'; fix the row or the body."
            )
        self._ledger_counts[verb] += 1

    def _stage_block(self, verb: str) -> _StageEmitBlock:
        """Return the open stage bucket a stage-only verb captures into.

        ADR 0114 D2: a ``scope == "stage"`` verb emitted outside a
        ``stage_open`` / ``stage_close`` bracket raises. Before K1-2 the
        bodies returned silently, so a bridge path that emitted a stage
        mutator globally would have archived nothing and said nothing.
        The bracket pair itself and ``flip_element_stage`` (archived by
        the ``set_stage_records`` side channel; the lock pins its body
        trivial) do not use this.
        """
        blk = self._stage_current
        if blk is None:
            raise RuntimeError(
                f"H5Emitter.{verb}: a stage-only verb (VERBS scope "
                "'stage') was emitted outside a stage bracket; it has no "
                "global store, so the archive could not carry it (ADR "
                "0114 D2). Emit it inside stage_open(...) / stage_close()."
            )
        return blk

    def _program_stage(self) -> int:
        """The ``stage`` column of a call now: ``-1`` outside a bracket,
        else the open stage's ``stage_NNN`` ordinal."""
        if self._stage_current is None:
            return _NO_ROW
        return len(self._stage_blocks)

    def _command(
        self,
        method: str,
        args: "tuple[int | float | str, ...]",
        *,
        names: "tuple[str, ...] | None" = None,
        token: str = "",
        stage: int = _NO_ROW,
    ) -> None:
        """Append one ``/opensees/commands`` row (ADR 0114 R3a).

        A global-scope verb inside a stage bracket raises: the archive
        has no stage store for it, so it could not replay there.
        """
        if stage == _NO_ROW and self._stage_current is not None:
            raise RuntimeError(
                f"H5Emitter.{method}: a global-only verb (VERBS scope "
                "'global') was emitted inside stage "
                f"{self._stage_current.name!r}; the archive has no stage "
                "store for it (ADR 0114 D2)."
            )
        for value in args:
            _command_kind(method, value)
        self._commands.append(_CommandRecord(
            method=method, token=token, stage=stage, args=tuple(args),
            names=names if names is not None else ("",) * len(args),
        ))
        # K1-6: the row's declaration (``/opensees/decls/rows/commands``),
        # and the call's (A7).
        self._command_decls.append(self._decl_open)
        self._program.decl = self._decl_open

    @property
    def ledger_counts(self) -> "Mapping[str, int]":
        """``{verb: calls}`` for the ``ledger`` calls the archive dropped.

        ``apeSees.h5`` reads it after the write to warn about what
        ``/opensees`` does not carry (:class:`H5LedgerWarning`).
        """
        return MappingProxyType(dict(self._ledger_counts))

    def solve_stamp(self) -> "SolveStamp | None":
        """The stamp :meth:`write` puts on ``/opensees``: the verbs' tokens
        (:meth:`_H5DeclsMixin.solve_stamp`) joined with the ones the
        stored argument values imply (opensees 2.33.0,
        :func:`~apeGmsh.opensees.emitter.caps.value_requires`): ``"fork"``
        for a fork-only analysis-chain type
        (:data:`~apeGmsh.opensees.emitter.caps.FORK_CHAIN`), ``"mpco"`` for
        a ``recorder mpco`` and ``"mpi"`` for a Mumps-family or
        ``Parallel*`` ``system`` without a runtime fallback or a
        partitioned ``solve_mode``, global and staged stores alike.

        The stores are the source's on an H5 -> H5 rewrite too (the
        recorders and the analysis chain replay into them, the stages are
        restored), so a rewrite of an older archive states what its model
        needs under today's tokens.
        """
        stamp = super().solve_stamp()
        if stamp is None:
            return None
        blocks = self._stage_blocks
        kinds = tuple(r.kind for r in self._recorders) + tuple(
            r.kind for blk in blocks for r in blk.recorders)
        chains = (self._analysis_attrs, *(blk.chain_attrs for blk in blocks))
        extra = value_requires(recorder_kinds=kinds, chains=chains,
                               solve_mode=stamp.solve_mode)
        if extra <= set(stamp.requires):
            return stamp
        return dataclasses.replace(
            stamp, requires=tuple(sorted(set(stamp.requires) | extra)))

    # The two counters tests read, derived from the ledger so the two
    # cannot drift. ``equationConstraint`` left the ledger at opensees
    # 2.37.0, so it has no counter.
    @property
    def _skipped_embed_ties(self) -> int:
        return self._ledger_counts["embedded_node"]

    @property
    def _skipped_contacts(self) -> int:
        return self._ledger_counts["contact_surface"]

    # =====================================================================
    # Output — write the buffered model to disk
    # =====================================================================

    def write(self, path: str) -> None:
        """Materialize the buffered model to an HDF5 file at ``path``.

        Standalone path — no broker integration.  Writes only ``/meta``
        and ``/opensees/...``; the neutral zone (``/nodes``,
        ``/elements/{type}``, …) requires a :class:`FEMData` and is
        emitted by :meth:`apeGmsh.opensees.apeSees.h5` instead.

        h5py is imported here (not at module load) so users who never
        call ``ops.h5()`` do not pay the import cost.
        """
        import h5py  # local import — lazy h5py dep; see module docstring

        with h5py.File(path, "w") as f:
            self._write_meta(f)
            self.write_opensees_into(f)

    def write_opensees_into(self, f: Any) -> None:
        """Write the bridge's ``/opensees/...`` groups into an open file.

        The composition entry point used by
        :meth:`apeGmsh.opensees.apeSees.h5` to layer ``/opensees/``
        content on top of a file the broker has already populated with
        ``/meta`` and the neutral zone.  This method does NOT write
        ``/meta`` and does NOT write ``/elements/{type}`` — both are
        broker-owned post-Phase-8.5.
        """
        self._write_bcs(f)
        self._write_materials(f)
        self._write_sections(f)
        self._write_transforms(f)
        self._write_beam_integration(f)
        self._write_element_meta(f)
        self._write_time_series(f)
        self._write_patterns(f)
        self._write_regions(f)
        self._write_dampings(f)
        self._write_initial_stress(f)
        self._write_recorders(f)
        self._write_constraints(f)
        self._write_partitions(f)
        self._write_analysis(f)
        # ADR 0055 Phase 2 — last, so stages land inside the hashed
        # ``/opensees`` group after every zone they reference
        # (materials / series / dampings / element_meta) is written.
        self._write_stages(f)
        # ADR 0114 R2/R3a (schema 2.23.0): the generic store and the
        # emit order over every store above. Both are hashed.
        self._write_commands(f)
        self._write_program(f)
        # ADR 0114 D6 (schema 2.24.0): the solve stamp, on ``/opensees``
        # itself. Hashed, like every attribute of the group except
        # ``target_mode`` (``MODEL_HASH_EXCLUDED_ATTRS``).
        self._write_solve_stamp(f)
        # opensees 2.36.0: the pinned target mode, beside the
        # stamp; hash-excluded (``MODEL_HASH_EXCLUDED_ATTRS``).
        self._write_target_mode(f)
        # ADR 0114 R5 (K1-6, schema 2.25.0): the declarations. A label,
        # so ``decls`` is in ``MODEL_HASH_EXCLUDED_CHILDREN``.
        self._write_decls(f)
        # K1-7 (schema 2.26.0): the declarations' parameters by field
        # name, a derived view of the stores above; hash-excluded too.
        self._write_decl_params(f)

    def _decl_params_rows(self) -> "list[DeclParamsRow]":
        """The ``/opensees/decl_params`` rows: the handed-in primitives
        with their ``params_names`` and ``*_ref`` columns derived from
        the stores and the declaration joins (K1-7)."""
        assert self._declarations is not None and self._decl_params is not None
        decls = self._declarations
        row_of_key = {d.key: i for i, d in enumerate(decls)}
        runs_of: dict[int, list[DeclTagRun]] = {}
        for run in self._decl_tags:
            runs_of.setdefault(run[3], []).append(run)

        def single_tag(decl: int) -> "tuple[str, int] | None":
            runs = runs_of.get(decl, [])
            if len(runs) == 1 and runs[0][2] == 1:
                return runs[0][0], runs[0][1]
            return None

        def tag_of_key(key: str) -> "int | None":
            row = row_of_key.get(key)
            found = None if row is None else single_tag(row)
            return None if found is None else found[1]

        argv_by_tag = self._argv_by_tag()
        out: list[DeclParamsRow] = []
        for decl, type_name, params in self._decl_params:
            names = ""
            kind_tag = single_tag(decl)
            if kind_tag is not None and kind_tag[0] in _ARGV_STORE_KINDS:
                argv = argv_by_tag[kind_tag[0]].get(kind_tag[1])
                if argv is not None:
                    found = decl_argv_names(params, argv, tag_of_key)
                    if found is not None:
                        names = _json_text(list(found))
            refs: dict[str, list[str]] = {col: [] for col, _f in _REF_COLUMNS}
            for value in params.values():
                for key in _decl_refs(value):
                    family = decls[row_of_key[key]].family
                    for col, want in _REF_COLUMNS:
                        if family == want:
                            refs[col].append(key)
            out.append((decl, type_name, _json_text(params), names,
                        tuple(refs["transf_ref"]),
                        tuple(refs["integration_ref"]),
                        tuple(refs["section_ref"])))
        return out

    def _write_target_mode(self, f: Any) -> None:
        """Persist ``/opensees@target_mode`` for a pinned target.

        ``"auto"`` writes nothing: an absent attribute reads as ``"auto"``,
        so an unpinned model's archive keeps its pre-2.36 bytes.
        """
        if self._target_mode == "auto":
            return
        import h5py
        import numpy as np

        self._ops_group(f).attrs.create(
            "target_mode", np.array(self._target_mode, dtype=object),
            dtype=h5py.string_dtype(encoding="utf-8"))

    def _ops_group(self, f: Any) -> Any:
        """Lazily get or create the ``/opensees/`` namespace group.

        Bridge-written groups (materials, sections, transforms, ...)
        live under ``/opensees/`` since Phase 8.4.  Created on first
        use so the incomplete fixture — no bridge content beyond
        ``/meta`` + ``/elements`` — never carries an empty
        ``/opensees`` group.
        """
        if "opensees" not in f:
            f.create_group("opensees")
        return f["opensees"]

    def _write_meta(self, f: Any) -> None:
        """Create ``/meta`` and populate its attributes."""
        meta = f.create_group("meta")
        for key, value in self._meta_attrs().items():
            _set_attr(meta, key, value)

    def set_declarations(
        self, decls: "Sequence[tuple[str, str, str, bool]]",
        tags: "Sequence[DeclTagRun]",
    ) -> None:
        """Hand in the bridge's declarations (ADR 0114 R5, K1-6, schema 2.25.0).

        Side channel, not a Protocol call (it notes no emit index).
        ``decls`` are the declarations, one per key, as
        :class:`Declaration` fields ``(key, family, name, synth)``; ``tags`` the
        ``(kind, first, count, decl)`` runs joining each tagged
        declaration on the tag plan's ``(kind, tag)``, a fan-out element
        or transform included (it inherits its spec's declaration). Called
        once, before the emit, so :meth:`set_declaration` can open them.
        Refuses a second call, a repeated key, a run whose ``decl`` is out
        of range, and two runs that claim one ``(kind, tag)``.
        """
        if self._declarations is not None:
            raise RuntimeError(
                "H5Emitter.set_declarations: the declarations are already "
                "set; they are handed in once per emit."
            )
        rows = tuple(Declaration(*d) for d in decls)
        seen: set[str] = set()
        for d in rows:
            if not d.key or d.key in seen:
                raise ValueError(
                    f"H5Emitter.set_declarations: declaration key {d.key!r} "
                    "is empty or repeated; each declaration has one key."
                )
            seen.add(d.key)
        runs: list[DeclTagRun] = []
        for kind, first, count, decl in tags:
            if not kind or int(count) < 1 or not 0 <= int(decl) < len(rows):
                raise ValueError(
                    "H5Emitter.set_declarations: tag run "
                    f"{(kind, first, count, decl)!r} needs a kind, a count "
                    f">= 1 and a declaration index below {len(rows)}."
                )
            runs.append((str(kind), int(first), int(count), int(decl)))
        by_kind: dict[str, list[tuple[int, int]]] = {}
        for kind, first, count, _decl in runs:
            by_kind.setdefault(kind, []).append((first, first + count))
        for kind, spans in by_kind.items():
            spans.sort()
            for (_a, end), (start, _b) in zip(spans, spans[1:]):
                if start < end:
                    raise ValueError(
                        "H5Emitter.set_declarations: two declarations claim "
                        f"{kind} tag {start}; a (kind, tag) has one "
                        "declaration."
                    )
        self._declarations = rows
        self._decl_tags = tuple(runs)
        self._element_decl_run = (0, 0, _NO_DECL)
        index: dict[str, tuple[list[int], list[int], list[int]]] = {}
        for kind, first, count, decl in sorted(runs):
            firsts, ends, decls_of = index.setdefault(kind, ([], [], []))
            firsts.append(first)
            ends.append(first + count)
            decls_of.append(decl)
        self._decl_tag_index = index

    def set_decl_params(
        self, items: "Sequence[tuple[int, object]]",
        key_of: Callable[[object], str],
    ) -> None:
        """Hand in the registered primitives for ``/opensees/decl_params``
        (ADR 0114 A6/Q4, K1-7, schema 2.26.0).

        Side channel, not a Protocol call. ``items`` pairs each owner (a
        primitive or a declared record) with its ``decls`` row, in the
        order the rows are written: a key that merged several calls (a
        ``region`` name; the one other name-shared family, ``initial_stress``,
        refuses a repeated name at build on every path) appears once per
        owner, in call order, so no owner is dropped. ``key_of`` maps
        a primitive a field references to its declaration key
        (:class:`KeyError` for an unregistered one). Each owner is
        encoded now by :func:`encode_decl_params`, so an unknown field
        shape refuses here, before the emit, with
        :class:`H5DeclParamsError`. Called once, after
        :meth:`set_declarations`; a second call, a row out of range or a
        row with no owner refuses. The ``params_names`` and ``*_ref``
        columns are derived at :meth:`write` from the stores.
        """
        if self._declarations is None:
            raise RuntimeError(
                "H5Emitter.set_decl_params: call set_declarations first.")
        if self._decl_params is not None or self._decl_params_restored is not None:
            raise RuntimeError(
                "H5Emitter.set_decl_params: the declaration parameters are "
                "already set; they are handed in once per emit.")
        rows: list[tuple[int, str, dict[str, Any]]] = []
        for decl, prim in items:
            decl = int(decl)
            if not 0 <= decl < len(self._declarations):
                raise IndexError(
                    f"H5Emitter.set_decl_params: declaration {decl} is out "
                    f"of range (there are {len(self._declarations)}).")
            rows.append((decl, type(prim).__name__,
                         encode_decl_params(prim, key_of)))
        missing = set(range(len(self._declarations))) - {r[0] for r in rows}
        if missing:
            raise ValueError(
                "H5Emitter.set_decl_params: declaration(s) "
                f"{sorted(self._declarations[i].key for i in missing)} have "
                "no owner; every declaration stores its parameters.")
        self._decl_params = rows

    @staticmethod
    def _drop_absent_stores(
        f: Any,
        rows: "tuple[tuple[int, ...], ...]",
        stores: "tuple[str, ...]",
    ) -> "tuple[tuple[tuple[int, ...], ...], tuple[str, ...]]":
        """Unlink echoed runs whose store this rewrite did not write.

        An echoed program addresses the source's stores; a rewrite that
        drops one (``/opensees/regions`` is not replayed) would leave runs
        naming a path the file lacks. Those runs keep their emit indices
        (the tiling holds) with ``store`` and ``row`` ``-1``, ``@stores``
        loses the name, and the drop is warned, naming the stores.
        """
        import warnings

        absent: set[int] = set()
        for row in rows:
            st, stage = row[3], row[5]
            if st != _NO_ROW and _store_root(stores[st], stage).lstrip("/") not in f:
                absent.add(st)
        if not absent:
            return rows, stores
        keep = [i for i in range(len(stores)) if i not in absent]
        remap = {old: new for new, old in enumerate(keep)}
        out = tuple(
            r if r[3] == _NO_ROW else (
                (r[0], r[1], r[2], _NO_ROW, _NO_ROW, r[5], r[6])
                if r[3] in absent else
                (r[0], r[1], r[2], remap[r[3]], r[4], r[5], r[6]))
            for r in rows
        )
        from .._internal._stacklevel import _stacklevel_outside_package

        names = sorted(stores[i] for i in absent)
        warnings.warn(
            "model.h5 rewrite: the source's /opensees/program names "
            f"store(s) {names} that this rewrite does not write; their runs "
            "keep their emit indices with store -1, so the order survives "
            "but the records do not.",
            H5FeatureDeferredWarning,
            stacklevel=_stacklevel_outside_package(),
        )
        return out, tuple(stores[i] for i in keep)

    # =====================================================================
    # Helpers used by the writer (and by tests inspecting buffer state)
    # =====================================================================

    def _meta_attrs(self) -> dict[str, Any]:
        """Return the ``/meta`` group's attributes as a dict.

        Per ADR 0023 the bridge stamps its own per-zone marker
        ``/meta/opensees_schema_version`` alongside the legacy envelope
        ``schema_version`` key.  Bridge-only standalone files (no broker
        snapshot) carry only the OpenSees per-zone key; ``_compose_model_h5``
        adds the neutral-zone key when a broker FEM was paired in.
        """
        return {
            "schema_version": self._schema_version,
            "opensees_schema_version": self._schema_version,
            "apeGmsh_version": self._apegmsh_version,
            "created_iso": datetime.now(tz=timezone.utc).isoformat(),
            "ndm": int(self._ndm) if self._ndm is not None else 0,
            "ndf": int(self._ndf) if self._ndf is not None else 0,
            "snapshot_id": self._snapshot_id,
            "model_name": self._model_name,
        }


def _store_root(template: str, stage: int) -> str:
    """The HDF5 path a ``VERBS`` store template's records live under.

    ``{scope}`` and the stage prefix resolve by ``stage``; the path is cut
    at the first unresolved field or ``@attribute`` (to the last whole
    segment), so ``{scope}/regions/region_{k:03d}`` is ``/opensees/regions``.
    """
    scope = (
        "/opensees" if stage == _NO_ROW
        else f"/opensees/stages/stage_{stage:03d}")
    path = template.replace("{scope}", scope)
    if stage != _NO_ROW:
        path = path.replace(
            "/opensees/stages/stage_{k:03d}", scope, 1)
    cuts = [i for i in (path.find("{"), path.find("@")) if i >= 0]
    cut = min(cuts) if cuts else len(path)
    literal = path[:cut]
    if cut < len(path) and path[cut] == "{":
        literal = literal.rsplit("/", 1)[0]
    return literal.rstrip("/")


#: The tagged verbs (A7): each call's ``(kind, tag)`` is its record's
#: ``/opensees/decls/tags`` join, ``kind`` the bridge's allocator kind and
#: ``tag`` the positional argument at the index given. ``element`` notes
#: inline (the hot path) with the same join. Every other verb's call is
#: :data:`_NO_DECL` unless its body sets ``tape.decl`` from the
#: declaration its row's ``/opensees/decls/rows`` column took (``fix``,
#: ``mass``, ``recorder``, ``sp_hold``, ``remove_sp``, ``remove_element``,
#: ``update_material_stage``, ``rayleigh`` and the ``/opensees/commands``
#: rows).
_DECL_TAG_ARG: "Mapping[str, tuple[str, int]]" = MappingProxyType({
    "uniaxialMaterial": ("uniaxialMaterial", 1),
    "nDMaterial": ("nDMaterial", 1),
    "section": ("section", 1),
    "section_open": ("section", 1),
    "geomTransf": ("geomTransf", 1),
    "beamIntegration": ("beamIntegration", 1),
    "timeSeries": ("timeSeries", 1),
    "pattern_open": ("pattern", 1),
    "damping": ("damping", 1),
    "region": ("region", 0),
})


#: The side-channel verbs whose call is the declaration open on it
#: (opensees 2.32.0): the bridge opens each record's declaration around
#: the calls that record drives (``stage_emit.DeclCursor``), so the open
#: declaration is proved. Their bodies store nothing per call that a rows
#: column could carry, so :func:`_noted` records it for them.
_SIDE_CHANNEL_DECL_VERBS: frozenset[str] = frozenset({
    "step_hook_ramp", "addToParameter", "flip_element_stage",
    "update_parameter", "set_node_vel", "set_node_accel"})


def _noted(
    name: str, fn: "Callable[..., Any]",
) -> "Callable[..., Any]":
    """Wrap Protocol method ``name`` so each completed call notes one
    ``/opensees/program`` emit index (ADR 0114 R2).

    The store comes from the verb's ``VERBS`` row (the global or the
    stage side of a ``|`` template). A call that raises notes nothing,
    and a Protocol method a body calls internally counts once, as the
    outer call.
    """
    glob, split, stage_side = VERBS[name].store.partition("|")
    # ``"<global>|"``: a stage call writes no row (``equationConstraint``).
    stage_store = stage_side if split else glob
    tagged = _DECL_TAG_ARG.get(name)
    side_channel = name in _SIDE_CHANNEL_DECL_VERBS

    @functools.wraps(fn)
    def noted(self: "H5Emitter", *args: Any, **kwargs: Any) -> Any:
        tape = self._program
        if tape.depth:
            return fn(self, *args, **kwargs)
        blk = self._stage_current
        if blk is None:
            stage, store = _NO_ROW, glob
        else:
            stage, store = len(self._stage_blocks), stage_store
        key = (name, stage)
        tape.pending = tape.ordinal.get(key, 0) if store else _NO_ROW
        tape.override = None
        # A7: no declaration unless the call proves one: a tagged verb by
        # its ``(kind, tag)`` run, a tagless one by setting ``tape.decl``
        # to the declaration its row's column took.
        tape.decl = _NO_DECL
        tape.depth = 1
        try:
            out = fn(self, *args, **kwargs)
        finally:
            tape.depth = 0
        if tagged is not None:
            kind, pos = tagged
            tag = args[pos] if len(args) > pos else kwargs["tag"]
            tape.decl = self._decl_of_tag(kind, int(tag))
        elif side_channel:
            tape.decl = self._decl_open
        if blk is None and self._stage_current is not None:
            # ``stage_open``: the call belongs to the stage it opened.
            stage, store = len(self._stage_blocks), stage_store
            key = (name, stage)
            tape.pending = tape.ordinal.get(key, 0) if store else _NO_ROW
        tape.note(name, store, stage, key)
        return out

    return noted


#: Protocol methods that note inline (the per-node / per-element hot
#: path): each always appends a fresh row and is never replicated, so the
#: generic wrapper's override bookkeeping buys them nothing.
_NOTED_INLINE: frozenset[str] = frozenset({"node", "element"})
for _verb, _row in VERBS.items():
    if _row.via == "protocol" and _verb not in _NOTED_INLINE:
        setattr(H5Emitter, _verb, _noted(_verb, getattr(H5Emitter, _verb)))
del _verb, _row
