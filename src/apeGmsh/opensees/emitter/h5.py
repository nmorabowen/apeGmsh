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
``architecture/phase-8-untangle.md`` §3):

* ``/meta`` and ``/elements`` stay at the file root.  ``/elements``
  sits in the neutral zone because Phase 8.5 hands the writer from
  the bridge to the broker; for now the bridge still emits it.
* Everything else the bridge writes (materials, sections, transforms,
  beam_integration, time_series, patterns, bcs, recorders, analysis)
  lives under ``/opensees/`` so a future second solver can plug in at
  the same level without colliding.

The reshuffle is a breaking schema change — ``SCHEMA_VERSION`` jumps
``1.1.0 → 2.0.0``.  Phase 7a (ADR 0023) replaced the previous
``EXPECTED_SCHEMA_MAJOR`` constant with per-zone two-version-window
validation in :mod:`apeGmsh.opensees._internal.schema_version`.

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
known only inside ``_internal/build.py``'s element fan-out), so the
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

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Iterable, Literal, Sequence

from .._internal.tag_resolution import (
    ATTR_ELEMENT_NODES,
    current_fem_element_id,
)
from .._internal.typed_records import (
    BeamIntegrationRecord as _BeamIntegrationRecord,
    DampingObjectRecord as _DampingObjectRecord,
    DeclContext as _DeclContext,
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

if TYPE_CHECKING:
    from .base import StrategySpec


__all__ = [
    "H5Emitter", "SCHEMA_VERSION",
    "H5EquationConstraintDeviationWarning",
    "H5FeatureDeferredWarning",
    "H5ReinforceDeviationWarning",
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


class H5FeatureDeferredWarning(UserWarning):
    """A fork-only feature was dropped from the OpenSees H5 deck AND has no
    neutral-zone persistence — native H5 round-trip of the feature is deferred,
    so the feature is lost on round-trip. Emit to Tcl / openseespy (or run live)
    for the complete model.

    No emitter currently raises this: g.reinforce ties, g.constraints.contact,
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


#: Schema version string emitted in ``/meta/schema_version``. Bump
#: ``MAJOR`` only on a breaking change; ``MINOR`` for additive groups
#: (such as the ``/beam_integration`` group introduced alongside the
#: Protocol's ``beamIntegration`` method in Phase 4.5).
#:
#: History:
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
SCHEMA_VERSION: str = "2.20.0"


# Map known time-series type tokens to "is path-bearing": for a Path
# series the ``args`` carry numeric values; for algorithmic series we
# only record the type + scalar params.
_PATH_SERIES_TOKENS: tuple[str, ...] = ("Path",)


# Pattern type tokens that the bridge opens via ``pattern_open`` and
# closes via ``pattern_close`` with a body of load / sp / eleLoad calls.
_BLOCK_PATTERN_TOKENS: tuple[str, ...] = ("Plain", "MultiSupport")


# ---------------------------------------------------------------------------
# Low-level write helpers
# ---------------------------------------------------------------------------

def _set_attr(target: Any, key: str, value: Any) -> None:
    """Write ``value`` as an HDF5 attribute on ``target``.

    HDF5 stores attributes as native scalars or arrays — never as JSON
    blobs (per the schema's "structured groups, scalar attrs, array
    datasets" rule). This helper coerces Python values to the closest
    h5py-friendly representation:

    * ``str`` → variable-length UTF-8 string
    * ``bool`` / ``int`` → int64
    * ``float`` → float64
    * ``None`` → empty-string attr (h5py rejects ``None``)
    * tuple/list of numbers → 1-D float64 / int64 array
    * tuple/list of strings → 1-D variable-length string array
    """
    import h5py
    import numpy as np

    if value is None:
        target.attrs[key] = ""
        return
    if isinstance(value, bool):
        target.attrs[key] = np.int64(int(value))
        return
    if isinstance(value, int):
        target.attrs[key] = np.int64(value)
        return
    if isinstance(value, float):
        target.attrs[key] = np.float64(value)
        return
    if isinstance(value, str):
        target.attrs.create(key, value, dtype=h5py.string_dtype(encoding="utf-8"))
        return
    if isinstance(value, (tuple, list)):
        if not value:
            target.attrs.create(
                key, np.array([], dtype=np.float64),
            )
            return
        if all(isinstance(v, str) for v in value):
            target.attrs.create(
                key, list(value),
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
            return
        if all(isinstance(v, bool) or isinstance(v, int) for v in value):
            target.attrs[key] = np.asarray(value, dtype=np.int64)
            return
        # Mixed numeric — coerce to float64.
        target.attrs[key] = np.asarray(
            [float(v) for v in value], dtype=np.float64,
        )
        return
    # Fallback: stringify so we never crash on an unexpected attr type.
    target.attrs.create(
        key, repr(value), dtype=h5py.string_dtype(encoding="utf-8"),
    )


def _scan_flag(
    args: tuple[int | float | str, ...], flag: str,
) -> str | None:
    """Return the argument immediately after ``flag`` (as a string), or
    ``None`` if the flag is not present.

    Used for recorder ``-file`` extraction. Treats only string args as
    flag candidates.
    """
    for i, v in enumerate(args[:-1]):
        if isinstance(v, str) and v == flag:
            nxt = args[i + 1]
            return str(nxt)
    return None


def _write_param_array(
    target: Any, key: str, params: tuple[float | str | int, ...],
) -> None:
    """Write a positional ``*args`` tuple as one or two attributes.

    OpenSees parameter lists are positional: ``Steel02 1 fy E b R0 cR1
    cR2`` is all numeric, but ``forceBeamColumn 1 i j tt it -mass m
    -iter mx tol`` interleaves numerics and flag-string tokens. To
    stay HDF5-native (no JSON blobs), we split into two parallel
    arrays:

    * ``{key}`` — float64 array, ``NaN`` in slots that hold a string.
    * ``{key}_str`` — UTF-8 vlen string array, empty string in slots
      that hold a numeric value. Written ONLY if at least one slot is
      a string (pure-numeric param lists skip ``{key}_str`` entirely).

    Slot ``i`` of the original ``*args`` is reconstructible by reading
    whichever of the two arrays has a non-sentinel value.
    """
    import h5py
    import numpy as np

    if not params:
        target.attrs.create(key, np.array([], dtype=np.float64))
        return
    has_str = any(isinstance(v, str) for v in params)
    nums = np.empty(len(params), dtype=np.float64)
    strs: list[str] = []
    for i, v in enumerate(params):
        if isinstance(v, str):
            nums[i] = float("nan")
            strs.append(v)
        else:
            nums[i] = float(v)
            strs.append("")
    target.attrs.create(key, nums)
    if has_str:
        target.attrs.create(
            f"{key}_str", strs,
            dtype=h5py.string_dtype(encoding="utf-8"),
        )


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
# Partition emission (ADR 0027, schema 2.10.0)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _PartitionEmitBlock:
    """In-memory accumulator for a single ``partition_open(rank)`` block.

    Lives only inside :class:`H5Emitter`; the read-side reuses
    :class:`apeGmsh.opensees.emitter.h5_reader.PartitionEmittedRecord`
    which is the public surface.  The two are intentionally separate —
    the writer carries mutable accumulators (lists; node_set for dedupe),
    the reader carries immutable tuples.
    """

    #: OpenSeesMP rank id this block scopes.
    rank: int
    #: Node tags emitted under this block, in insertion order, with
    #: duplicates dropped (a phantom node declared twice in one rank
    #: block from two MP constraints would still count once).
    node_ids: list[int] = field(default_factory=list)
    #: Element tags emitted under this block, in insertion order.
    element_ids: list[int] = field(default_factory=list)
    #: O(1) dedupe membership test for ``node_ids`` (parallel
    #: shadow — never read directly, only via ``add_node``).
    _node_set: set[int] = field(default_factory=set)

    def add_node(self, tag: int) -> None:
        """Append ``tag`` if not already present (dedupe within block)."""
        if tag not in self._node_set:
            self._node_set.add(tag)
            self.node_ids.append(tag)

    def add_element(self, tag: int) -> None:
        """Append ``tag`` to the block's element list."""
        self.element_ids.append(tag)


# ---------------------------------------------------------------------------
# Stage emission (ADR 0055 Phase 2, schema 2.18.0)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _StageEmitBlock:
    """In-memory capture bucket for one ``stage_open(name)`` block.

    ADR 0055 Phase 2 — while a stage bracket is open, the emitter
    routes in-band Protocol calls here instead of (or, for topology,
    in addition to) the global buffers:

    * **Redirected** (would otherwise pollute the global zones and
      double-apply on replay): ``fix`` / ``mass`` / ``region`` / MP
      constraints / pattern brackets (loads / sps / sp_holds) /
      ``recorder`` / ``rayleigh`` / analysis-chain attrs / ``analyze``
      / the SSI-2.E mutators.  Redirecting the chain calls is what
      prevents the phantom global ``/opensees/analysis`` leak — the
      last stage's chain would otherwise land in ``_analysis_attrs``.
    * **Dual-appended** (global buffers must stay complete — element
      metadata and phantom-node bookkeeping cover ALL elements/nodes):
      ``node`` / ``element``, which additionally record the stage's
      owned tags here, in emit order.

    The declarative complement that never flows through the resolved
    Protocol stream (``activated_pgs``, per-stage initial-stress,
    ``activate_absorbing`` — their emit drives only no-op'd
    ``step_hook_ramp`` / ``addToParameter`` / ``flip_element_stage``
    calls) is attached afterwards by
    :meth:`H5Emitter.set_stage_records` (Phase-1 side-channel pattern).

    Tri-state capture is by *presence*: a mutator that was never
    called leaves its field at the sentinel and the writer omits the
    attribute entirely — never-set is structurally distinct from
    value-0 with no presence companion needed.
    """

    name: str
    # Owned topology (dual-append, emit order — also the replay order).
    owned_node_ids: list[int] = field(default_factory=list)
    owned_element_ids: list[int] = field(default_factory=list)
    # Phantom nodes emitted INSIDE this stage (stage-claimed
    # node_to_surface constraints synthesize them).  Their coordinates
    # land only in the write-only global ``_node_coords`` buffer, so a
    # staged archive listing them in ``owned_node_ids`` would be
    # irreplayable — ``set_stage_records`` fails loud on a non-empty
    # list until a per-stage phantom store exists (gate-2 finding).
    phantom_node_tags: list[int] = field(default_factory=list)
    # Per-stage emit-sequence counter + the slots stamped with it.
    # Regions arrive from four producers (stage regions, region-scoped
    # rayleigh, damping attach, recorder pg-filter) whose interleaving
    # with the global-form ``rayleigh`` call and the stage patterns
    # carries OpenSees overwrite semantics ("region refines global") —
    # the ``emit_index`` stamps preserve the original relative order so
    # replay can reproduce it (gate-2 finding; the lists alone lose it).
    emit_seq: int = 0
    region_seq: list[int] = field(default_factory=list)
    rayleigh_seq: list[int] = field(default_factory=list)
    pattern_seq: list[int] = field(default_factory=list)
    # Redirected stage-scoped pools (same record classes as the
    # global buffers so the per-group writers are reusable verbatim).
    fixes: "list[_FixRecord]" = field(default_factory=list)
    masses: "list[_MassRecord]" = field(default_factory=list)
    regions: "list[_RegionRecord]" = field(default_factory=list)
    equal_dofs: "list[_EqualDOFRecord]" = field(default_factory=list)
    rigid_links: "list[_RigidLinkRecord]" = field(default_factory=list)
    rigid_diaphragms: "list[_RigidDiaphragmRecord]" = field(
        default_factory=list,
    )
    embedded_nodes: "list[_EmbeddedNodeRecord]" = field(
        default_factory=list,
    )
    # ADR 0055 P2.3: per-MP-record emit_index. The bridge interleaves
    # the four MP kinds across one emit pass — crucially a
    # kinematic_coupling emits its equalDOF AFTER rigidDiaphragm
    # (build.py _emit_kinematic_couplings, step 5), so the genuine vs
    # kinematic equalDOFs straddle the rigid_diaphragm emit.  Four
    # separate typed buckets lose that cross-bucket order; the seq
    # stamps recover it (merge-sort on replay, same design as
    # region_seq / rayleigh_seq).
    equal_dof_seq: list[int] = field(default_factory=list)
    rigid_link_seq: list[int] = field(default_factory=list)
    rigid_diaphragm_seq: list[int] = field(default_factory=list)
    embedded_node_seq: list[int] = field(default_factory=list)
    patterns_complete: "list[_PatternRecord]" = field(default_factory=list)
    open_pattern: "_PatternRecord | None" = None
    recorders: "list[_RecorderRecord]" = field(default_factory=list)
    # Global-form stage rayleigh (``on=()``) — four raw coefficients
    # per call.  Region-scoped rayleigh / damping attaches arrive as
    # ``region`` calls with the ``-rayleigh`` / ``-damp`` tail and are
    # captured in ``regions`` above.
    rayleighs: "list[tuple[float, float, float, float]]" = field(
        default_factory=list,
    )
    # SSI-2.E mutators (resolved targets, emit order).
    remove_sps: "list[tuple[int, int]]" = field(default_factory=list)
    remove_elements: "list[int]" = field(default_factory=list)
    # Per-stage analysis chain + analyze loop (the ``_write_analysis``
    # attr shape, scoped to this stage).
    chain_attrs: "dict[str, Any]" = field(default_factory=dict)
    analyze_call: "tuple[int, float | None] | None" = None
    domain_changed: bool = False
    # Presence-captured time-state mutators (None == never called).
    set_time: "float | None" = None
    set_creep_on: "bool | None" = None
    pre_analyze_reset: bool = False
    # Declarative complement — attached by ``set_stage_records``.
    activated_pgs: "tuple[str, ...]" = ()
    initial_stress_records: "tuple[Any, ...]" = ()
    activate_absorbing_records: "tuple[Any, ...]" = ()

    def next_emit_index(self) -> int:
        """Monotone per-stage sequence number (1-based, emit order)."""
        self.emit_seq += 1
        return self.emit_seq


def _merge_node_region_args(
    old: "tuple[int | float | str, ...]",
    new: "tuple[int | float | str, ...]",
) -> "tuple[int | float | str, ...] | None":
    """Union two ``("-node", n1, n2, ...)`` region arg tails (P5.1).

    Returns the merged tail (first-occurrence member order preserved,
    duplicates dropped) or ``None`` when either tail is not the plain
    ``-node`` member-list form — the caller then appends the record
    unmerged rather than corrupting a flag tail it cannot parse.
    """
    def _is_node_form(t: "tuple[int | float | str, ...]") -> bool:
        return (
            len(t) >= 1
            and t[0] == "-node"
            and all(isinstance(v, int) for v in t[1:])
        )

    if not _is_node_form(old) or not _is_node_form(new):
        return None
    seen = set(old[1:])
    merged: "list[int | float | str]" = list(old)
    for v in new[1:]:
        if v not in seen:
            seen.add(v)
            merged.append(v)
    return tuple(merged)


def _partition_blocks_to_ro(
    blocks: "Sequence[_PartitionEmitBlock]",
) -> "list[Any]":
    """Freeze writer partition blocks into read-side
    ``PartitionEmittedRecord`` values (P5.0b).

    Used by ``OpenSeesModel.from_compose_buffers`` so a model
    materialised straight from emitter buffers carries the same
    partition view a write-then-``from_h5`` round trip would produce.
    ``boundary_node_ids`` is computed with the same one-pass symmetric
    intersection :meth:`H5Emitter._write_partitions` uses.
    """
    from .h5_reader import PartitionEmittedRecord

    node_sets = [set(blk.node_ids) for blk in blocks]
    out: "list[Any]" = []
    for idx, blk in enumerate(blocks):
        others: set[int] = set()
        for other_idx, other_set in enumerate(node_sets):
            if other_idx == idx:
                continue
            others.update(other_set)
        out.append(PartitionEmittedRecord(
            rank=int(blk.rank),
            element_ids=tuple(int(e) for e in blk.element_ids),
            node_ids=tuple(int(n) for n in blk.node_ids),
            boundary_node_ids=tuple(
                int(n) for n in blk.node_ids if n in others
            ),
        ))
    return out


def _stage_block_to_ro(blk: "_StageEmitBlock") -> "Any":
    """Freeze a capture bucket into a read-side ``StageRecordRO``.

    Used by ``OpenSeesModel.from_compose_buffers`` so a model
    materialised straight from emitter buffers carries the same
    staged view a write-then-``from_h5`` round trip would produce.
    The bucket's declarative complement must already be attached
    (``set_stage_records`` ran).
    """
    from .._internal.typed_records import StageRecordRO

    if blk.analyze_call is None:
        raise RuntimeError(
            f"stage {blk.name!r}: no analyze call captured — "
            "malformed bracket cannot freeze to a StageRecordRO."
        )
    steps, dt = blk.analyze_call
    return StageRecordRO(
        name=blk.name,
        analyze_steps=int(steps),
        analyze_dt=dt,
        set_time=blk.set_time,
        set_creep_on=blk.set_creep_on,
        pre_analyze_reset=blk.pre_analyze_reset,
        domain_changed=blk.domain_changed,
        activated_pgs=tuple(blk.activated_pgs),
        owned_node_ids=tuple(blk.owned_node_ids),
        owned_element_ids=tuple(blk.owned_element_ids),
        fixes=tuple(blk.fixes),
        masses=tuple(blk.masses),
        regions=tuple(blk.regions),
        region_seq=tuple(blk.region_seq),
        equal_dofs=tuple(blk.equal_dofs),
        rigid_links=tuple(blk.rigid_links),
        rigid_diaphragms=tuple(blk.rigid_diaphragms),
        embedded_nodes=tuple(blk.embedded_nodes),
        equal_dof_seq=tuple(blk.equal_dof_seq),
        rigid_link_seq=tuple(blk.rigid_link_seq),
        rigid_diaphragm_seq=tuple(blk.rigid_diaphragm_seq),
        embedded_node_seq=tuple(blk.embedded_node_seq),
        patterns=tuple(blk.patterns_complete),
        pattern_seq=tuple(blk.pattern_seq),
        recorders=tuple(blk.recorders),
        rayleighs=tuple(blk.rayleighs),
        rayleigh_seq=tuple(blk.rayleigh_seq),
        remove_sps=tuple(blk.remove_sps),
        remove_elements=tuple(blk.remove_elements),
        chain_attrs=dict(blk.chain_attrs),
        initial_stress=tuple(blk.initial_stress_records),
        activate_absorbing=tuple(
            (rec.pg, None if rec.elements is None else tuple(rec.elements))
            for rec in blk.activate_absorbing_records
        ),
    )


def _ro_to_stage_block(ro: "Any") -> "_StageEmitBlock":
    """Thaw a ``StageRecordRO`` back into a capture bucket — the echo
    half of the ``from_h5 → to_h5`` round trip.  Field-for-field
    inverse of :func:`_stage_block_to_ro`; the writer then re-emits
    equivalent stage bytes (hash stability is the acceptance test,
    not raw file bytes — attr creation order may differ)."""
    from .._internal.build import ActivateAbsorbingRecord

    blk = _StageEmitBlock(name=str(ro.name))
    blk.analyze_call = (
        int(ro.analyze_steps),
        None if ro.analyze_dt is None else float(ro.analyze_dt),
    )
    blk.set_time = ro.set_time
    blk.set_creep_on = ro.set_creep_on
    blk.pre_analyze_reset = bool(ro.pre_analyze_reset)
    blk.domain_changed = bool(ro.domain_changed)
    blk.activated_pgs = tuple(ro.activated_pgs)
    blk.owned_node_ids = list(ro.owned_node_ids)
    blk.owned_element_ids = list(ro.owned_element_ids)
    blk.fixes = list(ro.fixes)
    blk.masses = list(ro.masses)
    blk.regions = list(ro.regions)
    blk.region_seq = list(ro.region_seq)
    blk.equal_dofs = list(ro.equal_dofs)
    blk.rigid_links = list(ro.rigid_links)
    blk.rigid_diaphragms = list(ro.rigid_diaphragms)
    blk.embedded_nodes = list(ro.embedded_nodes)
    blk.equal_dof_seq = list(ro.equal_dof_seq)
    blk.rigid_link_seq = list(ro.rigid_link_seq)
    blk.rigid_diaphragm_seq = list(ro.rigid_diaphragm_seq)
    blk.embedded_node_seq = list(ro.embedded_node_seq)
    blk.patterns_complete = list(ro.patterns)
    blk.pattern_seq = list(ro.pattern_seq)
    blk.recorders = list(ro.recorders)
    blk.rayleighs = list(ro.rayleighs)
    blk.rayleigh_seq = list(ro.rayleigh_seq)
    blk.remove_sps = [(int(n), int(d)) for n, d in ro.remove_sps]
    blk.remove_elements = [int(t) for t in ro.remove_elements]
    blk.chain_attrs = dict(ro.chain_attrs)
    blk.initial_stress_records = tuple(ro.initial_stress)
    blk.activate_absorbing_records = tuple(
        ActivateAbsorbingRecord(
            pg=pg, elements=None if elements is None else tuple(elements),
        )
        for pg, elements in ro.activate_absorbing
    )
    return blk


# ---------------------------------------------------------------------------
# H5Emitter
# ---------------------------------------------------------------------------

class H5Emitter:
    """Structured emitter that writes ``model.h5`` per ``h5-schema.md``.

    Construct, drive via :meth:`BuiltModel.emit`, then call
    :meth:`write` to materialize the HDF5 file. The class buffers all
    state in memory; the file is written exactly once.
    """

    def __init__(
        self,
        *,
        schema_version: str = SCHEMA_VERSION,
        model_name: str | None = None,
        apegmsh_version: str | None = None,
        snapshot_id: str | None = None,
    ) -> None:
        self._schema_version: str = schema_version
        self._model_name: str = model_name or "model"
        self._apegmsh_version: str = apegmsh_version or "unknown"
        self._snapshot_id: str = snapshot_id or ""

        self._ndm: int | None = None
        self._ndf: int | None = None

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

        # g.reinforce (ADR 20 / R2b): LadrunoEmbeddedRebar ties are not
        # written into the OpenSees DECK zone (``/opensees/...``) — the
        # dedicated ``reinforceTie`` deck record + deck-replay is a
        # documented follow-on (ADR 0067 P5.1 "A4 full"). The ties are
        # NOT lost: the neutral zone persists them (#706) and travels in
        # the same ``apeSees.h5`` archive. The emitter no-ops each deck
        # tie and counts it (observability only — no warning, since the
        # neutral round-trip is complete). See ``embedded_rebar``.
        self._skipped_reinforce_ties: int = 0
        self._skipped_embed_ties: int = 0
        self._skipped_contacts: int = 0

        # ADR 0068, Open item 4 (RESOLVED): the equation route
        # (EQ_Constraint) round-trips via the neutral InterpolationRecord
        # lane (enforce + weights, schema 2.14.0), so the deck emitter
        # no-ops it silently like the reinforce/embed/contact ties (no
        # warn). Counter retained for symmetry / diagnostics.
        self._skipped_equation_constraints: int = 0

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
        # record class (avoids a build.py import cycle).
        self._initial_stress_records: list[Any] = []

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
    # Protocol — Model
    # =====================================================================

    def model(self, *, ndm: int, ndf: int) -> None:
        self._ndm = ndm
        self._ndf = ndf

    def node(
        self, tag: int, *coords: float, ndf: int | None = None,
    ) -> None:
        # Normalize to 3-tuple by zero-padding 2-D models.
        cs = tuple(float(c) for c in coords)
        if len(cs) == 2:
            x, y = cs
            triple = (x, y, 0.0)
        elif len(cs) == 3:
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
                    ):
                        blk.phantom_node_tags.append(int(tag))
                elif is_stage_owned_node(self, int(tag)):
                    if not self._partition_dup(
                        ("stage_node", blk.name, int(tag)),
                    ):
                        blk.owned_node_ids.append(int(tag))
                # else: foreign decl — partition-block mirror only.

    def _partition_dup(self, key: "tuple[Any, ...]") -> bool:
        """True iff ``key`` was already captured under a partition bracket.

        P5.0a (ADR 0055 Phase 5 groundwork): the partitioned global
        pass replicates fix / mass / MP-constraint / pattern-line
        emission across owning ranks (ADR 0027 INV-1/INV-4); the flat
        H5 capture must record each logical record once.  Outside a
        partition bracket this never reports a duplicate — flat-build
        capture (including genuine user duplicates) is unchanged.
        """
        if self._partition_current is None:
            return False
        if key in self._partition_seen:
            return True
        self._partition_seen.add(key)
        return False

    def fix(self, tag: int, *dofs: int) -> None:
        # Stage emission (ADR 0055 Phase 2) — redirect: a stage-bound
        # fix must NOT land in the global ``/opensees/bcs`` zone (it
        # would double-apply as a t=0 BC on replay).
        rec = _FixRecord(tag=int(tag), dofs=tuple(int(d) for d in dofs))
        if self._stage_current is not None:
            # P5.1: a stage-bound fix on a cross-rank shared node
            # replicates per owning rank (INV-4) — capture once.
            # ``_partition_dup`` is inert outside a partition bracket,
            # so the flat staged path is unchanged.
            if self._partition_dup(
                ("stage_fix", self._stage_current.name, rec.tag, rec.dofs),
            ):
                return
            self._stage_current.fixes.append(rec)
            return
        if self._partition_dup(("fix", rec.tag, rec.dofs)):
            return
        self._fixes.append(rec)

    def mass(self, tag: int, *values: float) -> None:
        rec = _MassRecord(
            tag=int(tag), values=tuple(float(v) for v in values),
        )
        if self._stage_current is not None:
            if self._partition_dup(
                ("stage_mass", self._stage_current.name,
                 rec.tag, rec.values),
            ):
                return
            self._stage_current.masses.append(rec)
            return
        if self._partition_dup(("mass", rec.tag, rec.values)):
            return
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
        raise NotImplementedError(
            "equalDOF_Mixed archival to the OpenSees H5 deck is deferred "
            "(ADR 0069). Emit the model with .tcl() / .py() / live run, or "
            "persist it via the FEMData .h5 snapshot (get_fem_data), which "
            "round-trips equal_dof_mixed records. Deck archival is tracked "
            "as a follow-up."
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
        # ADR 0068 (Open item 4, RESOLVED): the OpenSees *deck* zone
        # (``/opensees/...``) carries no dedicated equationConstraint record
        # (a deck-replay follow-on, shared with reinforce / contact / embed
        # ties). This is NOT data loss: the equation tie is a resolved
        # ``InterpolationRecord``, and the NEUTRAL zone already persists its
        # ``enforce`` route AND the projection ``weights`` (schema 2.14.0).
        # ``apeSees(fem).h5(path)`` writes that neutral zone into the same
        # archive — so an equation-tied model.h5 round-trips its interface via
        # ``FEMData.from_h5`` → ``apeSees(fem).tcl()/py()/run()`` (the forward
        # emit re-runs ``_emit_one_interpolation`` → ``_emit_equation_tie``).
        # Hence no deviation warning here (mirroring embed / contact /
        # reinforce ties); just no-op the deck record and consume any latched
        # mp comment so it can't leak onto the next real MP record.
        del cnode, cdof, ccoef, retained
        self._consume_pending_mp_name()
        self._skipped_equation_constraints += 1

    def embedded_rebar(
        self, ele_tag: int, *args: int | float | str,
    ) -> None:
        # g.reinforce (ADR 20 / R2b): the OpenSees *deck* zone
        # (``/opensees/...``) does not (yet) carry a dedicated
        # ``reinforceTie`` record — that follow-on (ADR 0067 P5.1 "A4
        # full": ``/opensees/constraints/reinforceTie`` + h5_reader
        # reconstruction + ``OpenSeesModel.build`` deck-replay) is a
        # documented open item (see ``internal_docs/plan_rebar_p5.md``).
        #
        # This is NOT a silent data loss: as of ADR 0067 P5.1 (#706) the
        # NEUTRAL zone persists every tie, and ``apeSees(fem).h5(path)``
        # writes that neutral zone into the same archive — so a reinforced
        # model.h5 round-trips its reinforcement via ``FEMData.from_h5`` →
        # ``apeSees(fem).tcl()/py()/run()`` (the forward emit re-runs
        # ``emit_reinforce_ties``). Hence no deviation warning here; just
        # no-op the deck record and consume any latched mp comment so it
        # can't leak onto the next real MP record.
        del ele_tag, args
        self._consume_pending_mp_name()
        self._skipped_reinforce_ties += 1

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
        self._skipped_embed_ties += 1

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
        self._skipped_contacts += 1

    def contact(
        self, tag: int, *args: int | float | str,
    ) -> None:
        # Companion to contact_surface — the warning fires once on the first
        # contactSurface; just consume the call here.
        del tag, args

    def contact_plane(
        self, tag: int, *args: int | float | str,
    ) -> None:
        # g.constraints.contact_plane: deck-zone no-op, recovered via the
        # neutral /contact_planes group (ContactPlaneRecord round-trips through
        # FEMData.to_h5/from_h5 → emit_contact_planes re-emits). The slave-set
        # contactSurface call is consumed by ``contact_surface`` above; this
        # just consumes the contactPlane verb.
        del tag, args

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

    def geomTransf(self, t_type: str, tag: int, *vec: float) -> None:
        self._transforms.append(
            _TransformRecord(
                type_token=t_type, tag=int(tag),
                vec=tuple(float(v) for v in vec),
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
        # (see _internal/build.py emit_element_spec). Reading it here
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
                    return
                self._stage_partition_patterns[key] = rec
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
                return
            self._partition_patterns_by_tag[int(tag)] = rec
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
        self._active_pattern("eleLoad").ele_loads.append(
            _EleLoadRecord(args=tuple(args))
        )

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
        Outside a stage bracket this stays a no-op (unreachable from
        bridge-driven call sites; Protocol-conformance tests drive it
        bare).
        """
        if self._stage_current is None:
            del node, dof
            return
        pat = self._active_pattern("sp_hold")
        # P5.1: a HOLD on a cross-rank shared node emits inside every
        # owning rank's copy of the stage's HOLD pattern — capture once.
        if self._partition_dup(
            ("sp_hold", self._stage_current.name, pat.tag,
             int(node), int(dof)),
        ):
            return
        pat.sp_holds.append((int(node), int(dof)))

    # =====================================================================
    # Protocol — Recorders
    # =====================================================================

    def recorder(self, kind: str, *args: int | float | str) -> None:
        sink = (
            self._stage_current.recorders
            if self._stage_current is not None
            else self._recorders
        )
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
                        return
                else:
                    self._stage_partition_regions[key] = len(blk.regions)
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
        # ADR 0053 (D1): archival of the GLOBAL rayleigh form is deferred —
        # no ``/opensees/`` slot and no schema bump in D1 (same rationale as
        # ``eigen``: this is a domain directive, not a tagged model object).
        # Region-scoped Rayleigh (D2) persists for free via ``region`` since
        # it carries the ``-rayleigh`` tail.
        # ADR 0055 Phase 2: a STAGE-bound global-form rayleigh
        # (``s.damping.rayleigh`` with ``on=()``) is part of the staged
        # program and captures into the stage bucket; the non-staged
        # global form stays unarchived (D1 deferral unchanged).
        if self._stage_current is not None:
            blk = self._stage_current
            blk.rayleigh_seq.append(blk.next_emit_index())
            blk.rayleighs.append((
                float(alpha_m), float(beta_k),
                float(beta_k_init), float(beta_k_comm),
            ))
            return
        del alpha_m, beta_k, beta_k_init, beta_k_comm

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
        # ADR 0053 (D4): modal damping is a domain directive (like
        # ``rayleigh`` / ``eigen``); archival deferred — no-op, no schema bump.
        del factors

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
            in ``_internal/build.py`` calls
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

    # =====================================================================
    # Protocol — Stress control (Phase SSI-1) + Staged analysis (SSI-2)
    # =====================================================================
    #
    # ADR 0055 Phase 2 (schema 2.18.0): the staged-analysis bracket
    # methods CAPTURE into per-stage :class:`_StageEmitBlock` buckets;
    # :meth:`_write_stages` persists them under ``/opensees/stages``.
    # The stress-control trio (``addToParameter`` / ``step_hook_ramp``
    # / ``flip_element_stage``) stays no-op: those calls carry the
    # RESOLVED form (allocated parameter tags, rendered ramp targets),
    # which is non-deterministic across a round-trip.  Their
    # information persists declaratively instead — the global
    # ``set_initial_stress_records`` side-channel (Phase 1) and the
    # per-stage ``initial_stress`` / ``activate_absorbing`` sub-tables
    # attached by :meth:`set_stage_records` (Phase 2).

    def addToParameter(
        self, tag: int, ele_tag: int, response: str,
    ) -> None:
        """No-op — resolved form; persists declaratively (ADR 0055)."""
        del tag, ele_tag, response

    def flip_element_stage(
        self, pid: int, ele_tags: tuple[int, ...],
    ) -> None:
        """No-op — resolved form; persists declaratively via the
        per-stage ``activate_absorbing`` sub-table (ADR 0055 Phase 2)."""
        del pid, ele_tags

    def step_hook_ramp(
        self,
        name: str,
        *,
        targets: tuple[tuple[int, float], ...],
        n_steps_to_full: float,
        phase: Literal["before", "after"] = "before",
    ) -> None:
        """No-op — resolved form; persists declaratively via
        ``set_initial_stress_records`` / the per-stage
        ``initial_stress`` sub-table (ADR 0055)."""
        del name, targets, n_steps_to_full, phase

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
        if self._stage_current is not None:
            self._stage_current.domain_changed = True

    # -- Staged-analysis mutators (Phase SSI-2.E) ---------------------------
    # Capture into the active stage bucket (ADR 0055 Phase 2).  All five
    # are only emitted inside a stage bracket by bridge-driven call
    # sites; bare calls (Protocol-conformance tests) stay no-ops.
    # Presence semantics: a mutator never called leaves its bucket
    # field at the sentinel and the writer omits the attribute —
    # never-set is structurally distinct from value-0.

    def set_time(self, t: float) -> None:
        if self._stage_current is not None:
            self._stage_current.set_time = float(t)

    def set_creep(self, on: bool) -> None:
        if self._stage_current is not None:
            self._stage_current.set_creep_on = bool(on)

    def reset(self) -> None:
        if self._stage_current is not None:
            self._stage_current.pre_analyze_reset = True

    def remove_sp(self, node: int, dof: int) -> None:
        if self._stage_current is not None:
            # P5.1: remove_sp replicates on every rank owning the node
            # (mirrors fix's INV-4 fan-out) — capture once.
            if self._partition_dup(
                ("stage_remove_sp", self._stage_current.name,
                 int(node), int(dof)),
            ):
                return
            self._stage_current.remove_sps.append((int(node), int(dof)))

    def remove_element(self, tag: int) -> None:
        if self._stage_current is not None:
            self._stage_current.remove_elements.append(int(tag))

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
        # The OpenSees *deck* zone (``/opensees/...``) carries no
        # contactSurface/contact records (deck-replay is a follow-on; the
        # contact instead persists in the NEUTRAL zone, schema 2.21.0). The
        # LadrunoContact handler the bridge auto-emits for a contact model
        # would otherwise be recorded against a DECK with no contact data — an
        # inconsistent deck that, on ``OpenSeesModel.from_h5`` replay, emits
        # ``constraints LadrunoContact`` with nothing to handle (and
        # LadrunoContact is Plain-style for MP). Skip it so the replayed deck
        # falls back to the default handler (correct for "deck minus contact").
        # The full model still round-trips via the neutral zone:
        # ``FEMData.from_h5`` → ``apeSees(fem)`` re-runs emit_contacts +
        # re-derives the handler.
        if c_type == "LadrunoContact":
            return
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
        # ``eigen`` is a runtime one-shot retrieval — not part of the
        # model definition the H5 archive captures.  No-op here; the
        # bridge's ``apeSees.eigen(...)`` drives a LiveOpsEmitter
        # directly and never routes through H5.
        del num_modes, solver
        return []

    def modal_properties(
        self, *, unorm: bool = False, out: str | None = None,
    ) -> dict[str, list[float]]:
        # Runtime one-shot retrieval — nothing in the model definition
        # to archive.  No-op, mirroring ``eigen`` above.
        del unorm, out
        return {}

    def eigen_feast(
        self, f_min: float, f_max: float, *, certify: bool = False,
    ) -> list[float]:
        # Runtime one-shot retrieval — no-op, mirroring ``eigen`` above
        # (ADR 0075 INV-2).
        del f_min, f_max, certify
        return []

    def modal_response_history(
        self, *args: int | float | str,
    ) -> None:
        # Runtime analysis command — not model definition.  No-op,
        # mirroring ``eigen`` above (ADR 0075 INV-2).
        del args

    def response_spectrum_analysis(
        self, direction: int, *args: int | float | str,
    ) -> None:
        # Runtime analysis command — not model definition.  No-op,
        # mirroring ``eigen`` above (ADR 0075 INV-2).
        del direction, args

    def profiler(self, *args: int | float | str) -> None:
        # The profiler is runtime telemetry around the analyze loop — there
        # is nothing in the model definition to archive.  No-op, mirroring
        # ``eigen`` above.
        del args

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

    # -- Per-group writers (split out so each step adds one) -------------

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

    def _write_bcs(self, f: Any) -> None:
        """Write ``/opensees/bcs/fix`` and ``/opensees/bcs/mass`` compound datasets.

        Both are emitted only if at least one record exists. The bridge's
        ``fix`` / ``mass`` fan-out has already resolved any ``pg=``
        targets into per-node calls, so every record's ``target_kind``
        is ``"node"`` and ``target`` is the integer tag rendered as a
        string (per the schema's compound-dataset convention).
        """
        if not self._fixes and not self._masses:
            return
        bcs = self._ops_group(f).create_group("bcs")
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
          outside a bridge fan-out (test scenarios).
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
            # outside a bridge fan-out (e.g. standalone test calls).
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

    def set_initial_stress_records(self, records: "Iterable[Any]") -> None:
        """Buffer the bridge's global ``InitialStressRecord``s for archival.

        Side-channel (ADR 0055 Phase 1): :meth:`apeGmsh.opensees.apeSees.h5`
        calls this after ``bm.emit(self)`` and before the compose write, so
        :meth:`_write_initial_stress` can persist the declarative records.
        The Protocol ``addToParameter`` / ``step_hook_ramp`` calls stay
        no-ops — they carry the resolved (parameter-tag-bearing) form, which
        is non-deterministic across a round-trip.
        """
        self._initial_stress_records = list(records)

    def _write_initial_stress(self, f: Any) -> None:
        """Persist ``/opensees/initial_stress/stress_NNN`` groups (ADR 0055).

        One group per global ``ops.initial_stress(...)`` record carrying the
        pre-resolve declarative field set: ``name`` + ``sigma_xx/yy/zz`` +
        ``ramp_steps`` + ``lambda_install`` scalar attrs, and EITHER a ``pg``
        attr (PG-targeted) XOR an ``elements`` int64 dataset (explicit element
        list).  No parameter tags, no rendered ramp proc — replay re-runs
        :func:`emit_initial_stress_global` /
        :func:`emit_initial_stress_addtoparameter` to regenerate the deck.
        Empty when no global initial-stress record exists (vanilla files stay
        byte-identical).  Folds into ``model_hash`` (authored state).
        """
        if not self._initial_stress_records:
            return
        grp = self._ops_group(f).create_group("initial_stress")
        self._write_initial_stress_records_into(
            grp, self._initial_stress_records,
        )

    def _write_initial_stress_records_into(
        self, grp: Any, records: "Iterable[Any]",
    ) -> None:
        """Write ``stress_NNN`` declarative sub-groups under ``grp``.

        Shared by the global ``/opensees/initial_stress`` zone (Phase 1)
        and each stage's ``initial_stress`` sub-group (Phase 2) — same
        pre-resolve field set, same pg-XOR-elements discriminant.
        """
        import numpy as np

        for idx, rec in enumerate(records):
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
        for blk, rec in zip(self._stage_blocks, records):
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
            blk.activated_pgs = tuple(rec.activated_pgs)
            blk.initial_stress_records = tuple(rec.initial_stress_records)
            blk.activate_absorbing_records = tuple(
                rec.activate_absorbing_records
            )
        self._stage_records_attached = True

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
                )
            # Absorbing-boundary stage flip (ADR 0054 AB-3) — declarative
            # pg XOR elements, mirroring the initial-stress discriminant.
            if blk.activate_absorbing_records:
                absorb = g.create_group("activate_absorbing")
                for a_idx, ab_rec in enumerate(
                    blk.activate_absorbing_records
                ):
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
                _set_attr(g, "ids", ctx.ids)
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
            or self._phantom_node_tags
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
