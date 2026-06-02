"""
Typed records — the read-side shape of ``/opensees/...`` zone contents.

These dataclasses are the public, read-side mirror of the bridge's
``H5Emitter`` buffer records.  They serve two roles:

1. **Emitter buffer** — :class:`apeGmsh.opensees.emitter.h5.H5Emitter`
   uses them as its in-memory accumulator while the bridge is driving
   it (mechanical lift from the previous private ``_*Record`` shapes;
   no behavior change).
2. **Read-side broker** — :class:`apeGmsh.opensees.OpenSeesModel`
   (ADR 0019) rehydrates these same shapes from a ``model.h5`` file
   to expose typed accessors (``om.materials()``, ``om.sections()``,
   ...).

History
=======

Phase 3 of the major architectural refactor (ADR 0019) needed typed
record dataclasses for ``OpenSeesModel``'s record-collection
accessors.  The :class:`H5Emitter` already had private
``_MaterialRecord`` / ``_SectionSimpleRecord`` / ``_SectionComplexRecord``
/ ``_TransformRecord`` / ``_BeamIntegrationRecord`` / ``_ElementRecord``
/ ``_TimeSeriesRecord`` / ``_PatternRecord`` / ``_RecorderRecord``
dataclasses that already had the exact shapes the broker needed.
Rather than duplicate, the Phase-3 implementation **lifted** those
private records into this module under public names; the emitter
re-imports the same dataclasses so its on-disk write path is
byte-identical to the previous implementation (verified by the
``test_h5_*`` test suite remaining green).

Design notes
============

**Frozen dataclasses.**  Every record is ``@dataclass(frozen=True,
slots=True)``.  The emitter's previous buffers were mutable (``slots``
only); the lift to ``frozen`` is safe because the emitter constructs
each record once and either appends it to a list or sets it as the
open section / pattern.  The complex (open-then-extend) shapes —
:class:`SectionComplexRecord` and :class:`PatternRecord` — keep
their per-instance sub-lists as ``list[...]`` fields, but those lists
are mutated only between ``section_open`` / ``section_close`` (resp.
``pattern_open`` / ``pattern_close``); after close, the record is
frozen and ``OpenSeesModel`` exposes the sub-lists as tuples through
its accessor (see :meth:`OpenSeesModel.sections`).

**Names match the emitter.**  Each record's field set is identical
to the private predecessor — type tokens, tags, positional arg
tuples, sub-record lists.  This is the only acceptable lift: any
field rename or extension would force a parallel migration in the
write path and risk breaking the existing 2.5.0 schema invariants.

**Cuts / sweeps records carry no lift counterpart.**  The cuts /
sweeps zone was added in schema 2.5.0 with its own dedicated reader
(``apeGmsh.cuts._h5_io.read_cuts_and_sweeps``), which returns the
public ``SectionCutDef`` / ``SectionSweepDef`` types directly.
``OpenSeesModel.cuts()`` / ``.sweeps()`` simply forward those public
types; no new record dataclass is needed.

**MP constraints — Phase 7b shipped (ADR 0022).**  ``/opensees/constraints/``
exists on disk as of schema 2.7.0: four compound-dtype datasets
(``equalDOF`` / ``rigidLink`` / ``rigidDiaphragm`` / ``embeddedNode``)
plus one ``phantom_node_tags`` array.  The matching record types are
:class:`EqualDOFRecord`, :class:`RigidLinkRecord`,
:class:`RigidDiaphragmRecord`, and :class:`EmbeddedNodeRecord` below
— one per ``Emitter`` method.  Each carries the user's declaration
``name`` field for the INV-2 round-trip into emitted Tcl/Py.

See also
========

- :doc:`/architecture/decisions/0019-opensees-model-read-side-broker`
- :doc:`/architecture/decisions/0011-h5-as-fourth-emit-target`
- :doc:`/architecture/decisions/0009-phase-9-unified-recorder-schema`
- :mod:`apeGmsh.opensees.opensees_model` — the consumer of every type
  defined here.
- :mod:`apeGmsh.opensees.emitter.h5` — the producer; re-imports each
  type from this module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .tag_resolution import MISSING_FEM_ELEMENT_ID


__all__ = [
    "MaterialRecord",
    "SectionSimpleRecord",
    "SectionComplexRecord",
    "PatchRecord",
    "FiberRecord",
    "LayerRecord",
    "TransformRecord",
    "BeamIntegrationRecord",
    "ElementRecord",
    "TimeSeriesRecord",
    "LoadRecord",
    "SPRecord",
    "EleLoadRecord",
    "PatternRecord",
    "DeclContext",
    "RecorderRecord",
    "DampingObjectRecord",
    "RegionRecord",
    "FixRecord",
    "MassRecord",
    # MP constraint records (Phase 7b, ADR 0022, schema 2.7.0)
    "EqualDOFRecord",
    "RigidLinkRecord",
    "RigidDiaphragmRecord",
    "EmbeddedNodeRecord",
    # Stress-control records (Phase SSI-1: initial_stress, ramping hooks)
    "ParameterRecord",
    "AddToParameterRecord",
    "StepHookRampRecord",
]


# ---------------------------------------------------------------------------
# Constitutive records
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class MaterialRecord:
    """One uniaxial or nD material declaration.

    Both families share the same shape — only the ``family`` attribute
    on the emitter / reader (``uniaxial`` vs ``nd``) distinguishes
    them at the group-path level.
    """

    type_token: str
    tag: int
    params: tuple[float | str, ...]


@dataclass(frozen=True, slots=True)
class SectionSimpleRecord:
    """A one-liner section declaration (no patch / fiber / layer body).

    Examples: ``Elastic``, ``Aggregator``, ``Uniaxial``.  Complex
    sections (Fiber, FiberThermal, ...) are :class:`SectionComplexRecord`.
    """

    type_token: str
    tag: int
    params: tuple[float | str, ...]


@dataclass(frozen=True, slots=True)
class PatchRecord:
    """One ``patch`` directive inside a complex section.

    Patch kinds: ``"rect"`` (4 coords), ``"quad"`` (8 coords), ``"circ"``
    (6 coords padded to 8 on write).  Schema deviation logged on the
    parent section group's ``__deviation_patches__`` attr if ``kind``
    is unknown.
    """

    kind: str
    args: tuple[int | float, ...]


@dataclass(frozen=True, slots=True)
class FiberRecord:
    """One ``fiber`` directive inside a complex section."""

    y: float
    z: float
    area: float
    mat_tag: int


@dataclass(frozen=True, slots=True)
class LayerRecord:
    """One ``layer`` directive inside a complex section.

    Layer kinds: ``"straight"`` (4 line floats), ``"circ"`` (6 line
    floats).  Schema deviation logged on the parent section group's
    ``__deviation_layers__`` attr if ``kind`` is unknown.
    """

    kind: str
    args: tuple[int | float, ...]


@dataclass(frozen=True, slots=True)
class SectionComplexRecord:
    """A section opened with ``section_open`` and populated with patches
    / fibers / layers before ``section_close``.

    The three sub-record lists are populated in the order the bridge
    emitted them between open and close; their identity within the
    section is positional.  After ``section_close`` the lists are
    treated as immutable by downstream consumers — the
    :class:`OpenSeesModel` accessor (:meth:`OpenSeesModel.sections`)
    re-wraps them as tuples for the read-side surface.
    """

    type_token: str
    tag: int
    params: tuple[float | str, ...]
    patches: list[PatchRecord] = field(default_factory=list)
    fibers: list[FiberRecord] = field(default_factory=list)
    layers: list[LayerRecord] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Transform / beam-integration records
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class TransformRecord:
    """One ``geomTransf`` call.

    **Schema deviation (documented).** The H5 emitter writes one
    ``/opensees/transforms/{type}_{tag}/`` group **per ``geomTransf``
    call**, not per user-declared transform.  Orientation-driven
    transforms fan out across distinct per-element vecxz at build
    time, and each emitted line becomes its own group.  The reader's
    orientation join
    (``h5_reader.element_local_axes_vecxz``) iterates every transform
    group, so the deviation does not affect orientation correctness —
    but consumers walking ``OpenSeesModel.transforms()`` will see one
    record per emitted call, not one per spec.
    """

    type_token: str
    tag: int
    vec: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class BeamIntegrationRecord:
    """One ``beamIntegration`` declaration (Phase 4.5 — Lobatto, ...)."""

    type_token: str
    tag: int
    args: tuple[int | float | str, ...]


# ---------------------------------------------------------------------------
# Element record
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ElementRecord:
    """One ``element`` call.

    ``fem_eid`` is the broker FEM element id the bridge fanned this
    element out from (Phase 8.6 / master-plan §3 ``tag_map``).
    Sentinel ``MISSING_FEM_ELEMENT_ID`` (``-1``) marks records emitted
    outside a bridge fan-out (test stubs, or ``ModelData`` records
    after the v2.1.0 ``from_h5`` rehydrate when the source file
    dropped the connectivity prefix per
    ``H5Emitter._write_element_argstack``).

    ``connectivity`` is the per-element node tag tuple.  For records
    rehydrated by :class:`OpenSeesModel.from_h5` (and the legacy
    :meth:`ModelData.from_h5`), this is empty ``()`` because the
    write path stores only the arg tail (``args[arity:]``) — the
    connectivity prefix is carried by the broker's
    ``/elements/{gmsh_alias}`` zone.
    """

    type_token: str
    tag: int
    args: tuple[int | float | str, ...]
    connectivity: tuple[int, ...]
    fem_eid: int = MISSING_FEM_ELEMENT_ID


# ---------------------------------------------------------------------------
# Time-series / pattern records
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class TimeSeriesRecord:
    """One ``timeSeries`` declaration."""

    type_token: str
    tag: int
    args: tuple[int | float | str, ...]


@dataclass(frozen=True, slots=True)
class LoadRecord:
    """One ``load`` directive inside a :class:`PatternRecord`."""

    target: int
    forces: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class SPRecord:
    """One ``sp`` (single-point constraint) directive inside a pattern.

    Single-point constraints inside a load pattern are distinct from
    boundary fixity (``fix``) — they apply only while the pattern is
    active.  Multi-point constraints (``equalDOF``, ``rigidBeam``,
    ...) live on the broker side (:mod:`apeGmsh.mesh._record_h5`) and
    are out of scope here.
    """

    target: int
    dof: int
    value: float


@dataclass(frozen=True, slots=True)
class EleLoadRecord:
    """One ``eleLoad`` directive inside a :class:`PatternRecord`.

    Element loads carry vocabulary-rich ``*args`` (``-type`` /
    ``-ele`` / ``-eleRange`` flag tokens); the emitter stores them as
    a flat string array per row (see
    ``H5Emitter._write_pattern_ele_loads``).
    """

    args: tuple[int | float | str, ...]


@dataclass(frozen=True, slots=True)
class PatternRecord:
    """A pattern declared via ``pattern_open``.

    ``loads`` / ``sps`` / ``ele_loads`` are populated only if the
    pattern was opened as a block (``Plain`` / ``MultiSupport``).
    Single-line patterns (``UniformExcitation``) carry their body in
    ``args`` and have all three sub-lists empty.
    """

    type_token: str
    tag: int
    args: tuple[int | float | str, ...]
    loads: list[LoadRecord] = field(default_factory=list)
    sps: list[SPRecord] = field(default_factory=list)
    ele_loads: list[EleLoadRecord] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Recorder records (Phase 9 commit 6 — unified schema 2.3.0)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class DeclContext:
    """Declaration metadata captured between
    ``Emitter.recorder_declaration_begin`` and
    ``Emitter.recorder_declaration_end``.

    Attached to each :class:`RecorderRecord` emitted while a context
    is active, so the unified ``/opensees/recorders/`` group (schema
    2.3.0) can persist the original declaration intent alongside the
    fan-out call.
    """

    declaration_name: str
    record_name: str | None
    category: str
    components: tuple[str, ...]
    raw: tuple[str, ...]
    pg: tuple[str, ...]
    label: tuple[str, ...]
    selection: tuple[str, ...]
    ids: tuple[int, ...] | None
    dt: float | None
    n_steps: int | None
    file_root: str


@dataclass(frozen=True, slots=True)
class RecorderRecord:
    """One ``recorder`` call.

    ``kind`` is the OpenSees recorder type (``"Node"``,
    ``"Element"``, ``"MPCO"``, ...).

    ``decl_context`` is non-``None`` for fan-out calls emitted from a
    :class:`apeGmsh.opensees.recorder.RecorderDeclaration` (schema
    2.3.0 ``kind="declared"`` records); ``None`` for direct typed
    primitives (schema 2.3.0 ``kind="typed"`` records).

    .. note::

        The :class:`OpenSeesModel` read-side broker exposes the
        write-time discriminator via the ``decl_context is None``
        check — :attr:`RecorderRecord.kind_label` returns the schema
        2.3.0 ``"typed"`` / ``"declared"`` label without forcing the
        caller to introspect.
    """

    kind: str
    args: tuple[int | float | str, ...]
    decl_context: DeclContext | None = None

    @property
    def kind_label(self) -> Literal["typed", "declared"]:
        """Schema-2.3.0 ``kind`` attr value (``"typed"`` / ``"declared"``).

        Mirrors what the emitter stamps onto the group's ``kind``
        attr; useful for downstream code branching on the
        discriminator without having to check ``decl_context is
        None`` directly.
        """
        return "declared" if self.decl_context is not None else "typed"


# ---------------------------------------------------------------------------
# Region — tagged collection of nodes/elements (recorder filter target)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class RegionRecord:
    """One ``region`` call.

    A region is a tagged OpenSees domain object that bundles nodes
    and/or elements; other commands reference it by tag.  The bridge
    auto-emits regions from the recorder fan-out so :class:`MPCO`
    recorders with ``nodes_pg=`` / ``elements_pg=`` selectors can pass
    ``-R $regTag`` to the MPCO command (per the mpco-recorder skill:
    MPCO records the whole model unless filtered by a region).

    ``args`` carries the raw OpenSees flag sequence following the tag
    (``-node n1 n2 ...``, ``-ele e1 e2 ...``, ``-eleOnly``, ``-nodeOnly``,
    ``-eleRange``, etc.).  Persisted under ``/opensees/regions/`` and
    replayed verbatim by :class:`OpenSeesModel._replay_into`.
    """

    tag: int
    args: tuple[int | float | str, ...]


# ---------------------------------------------------------------------------
# Damping objects — tagged frequency-band dissipators (ADR 0053 D3)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class DampingObjectRecord:
    """One ``damping <Type> $tag ...`` object declaration (ADR 0053 D3).

    The tagged ``Uniform`` / ``SecStif`` / ``URD`` / ``URDbeta`` dissipator,
    inert until attached to elements (via ``region -damp`` or an element's
    ``-damp`` flag).  ``args`` carries the resolved OpenSees argument tail
    after the tag (ζ / freq / β values, ``-activateTime`` / ``-deactivateTime``
    / ``-factor $tsTag``).  Persisted under ``/opensees/dampings/`` and
    replayed verbatim by :class:`OpenSeesModel._replay_into` (after the
    time-series a ``-factor`` may reference, before the elements an
    element-flag ``-damp`` references).
    """

    type_token: str
    tag: int
    args: tuple[int | float | str, ...]


# ---------------------------------------------------------------------------
# Boundary-condition records (model-level fix / mass)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class FixRecord:
    """One ``fix`` directive (model-level boundary fixity).

    Not to be confused with the build-time ``apeGmsh.opensees._internal.build.FixRecord``
    which is the *bridge-side* fix-record before fan-out; this one is
    the *emitter-side* per-node call after fan-out.
    """

    tag: int
    dofs: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class MassRecord:
    """One ``mass`` directive (per-node nodal mass).

    Like :class:`FixRecord`, distinct from the bridge-side
    pre-fan-out :class:`apeGmsh.opensees._internal.build.MassRecord`.
    """

    tag: int
    values: tuple[float, ...]


# ---------------------------------------------------------------------------
# MP constraint records (Phase 7b, ADR 0022, schema 2.7.0)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class EqualDOFRecord:
    """One ``equalDOF`` call.

    Persisted under ``/opensees/constraints/equalDOF`` as a row in a
    compound-dtype dataset (master, slave, dofs[ndf], name).  The
    ``name`` field round-trips the user's declaration label from
    ``g.constraints.equal_dof(name=..., ...)`` into the emitted deck
    (ADR 0022 INV-2).
    """

    master: int
    slave: int
    dofs: tuple[int, ...]
    name: str = ""


@dataclass(frozen=True, slots=True)
class RigidLinkRecord:
    """One ``rigidLink`` call.

    Persisted under ``/opensees/constraints/rigidLink``.  ``kind`` is
    ``"beam"`` (full 6-DOF rigid link) or ``"bar"`` (translation-only
    rigid rod).  The ``name`` field round-trips the user's
    declaration label (ADR 0022 INV-2).
    """

    kind: str
    master: int
    slave: int
    name: str = ""


@dataclass(frozen=True, slots=True)
class RigidDiaphragmRecord:
    """One ``rigidDiaphragm`` call.

    Persisted under ``/opensees/constraints/rigidDiaphragm``.  Also
    used for ``rigid_body`` and ``kinematic_coupling`` records that
    the fan-out pass collapses onto the rigidDiaphragm path; consumers
    can disambiguate via the user's ``name`` field if needed.

    ``perp_dir`` is the global axis perpendicular to the rigid plane
    (1, 2, or 3 — derived from the resolved plane normal at FEM
    snapshot time).
    """

    perp_dir: int
    master: int
    slaves: tuple[int, ...]
    name: str = ""


@dataclass(frozen=True, slots=True)
class EmbeddedNodeRecord:
    """One ``element ASDEmbeddedNodeElement`` call.

    Persisted under ``/opensees/constraints/embeddedNode``.  Covers
    all four surface-coupling kinds (``tie`` / ``distributing`` /
    ``embedded`` / ``tied_contact`` / ``mortar``) that the fan-out
    pass routes through ASDEmbeddedNodeElement.

    ``cnode`` is the constrained (embedded / slave) node — the second
    positional in the OpenSees signature
    ``element ASDEmbeddedNodeElement $tag $Cnode $Rnode1 ...``.
    ``args`` are the host element's node tags ($Rnode1..$RnodeN).

    The four optional flag fields (``stiffness``, ``stiffness_p``,
    ``rotational``, ``pressure``) round-trip the ASDEmbeddedNodeElement
    options ``-K``, ``-KP``, ``-rot``, ``-p`` per ADR 0035.  Defaults
    match the C++ parser at ``ASDEmbeddedNodeElement.cpp:222`` so
    legacy 2.10.x H5 files that lack these columns read back with
    semantically-identical behaviour to the pre-ADR-0035 emit path.
    """

    ele_tag: int
    cnode: int
    args: tuple[int | float, ...]
    stiffness: float = 1.0e18
    stiffness_p: float | None = None
    rotational: bool = False
    pressure: bool = False
    name: str = ""


# ---------------------------------------------------------------------------
# Stress-control records (Phase SSI-1: initial_stress + per-step ramping hooks)
# ---------------------------------------------------------------------------
#
# These are the read-side mirror of the H5 buffer for the parameter /
# addToParameter / step-hook bundle emitted by the InitialStress
# composite primitive.  H5 archival is deferred for Phase SSI-1 — the
# H5 emitter no-ops these methods with a deviation marker.  The
# records exist now so the Emitter Protocol shape is uniform across
# emitters and the future schema bump is mechanical (additive).

@dataclass(frozen=True, slots=True)
class ParameterRecord:
    """One ``parameter $tag`` declaration.

    OpenSees parameters are tag-only at declaration time; subsequent
    :class:`AddToParameterRecord` rows attach element responses to the
    same tag, and :class:`StepHookRampRecord` drives the per-step
    ``updateParameter $tag $delta``.
    """

    tag: int


@dataclass(frozen=True, slots=True)
class AddToParameterRecord:
    """One ``addToParameter $tag element $ele_tag $response`` directive.

    Emitted per-rank inside ``partition_open`` blocks for MP-partitioned
    models — each rank only emits the addToParameter calls for elements
    it owns (see :func:`emit_initial_stress_partitioned` in
    :mod:`apeGmsh.opensees._internal.build`).

    ``response`` is the element-response name OpenSees exposes for the
    parameter (e.g. ``"commitStressIncrementXX"`` on
    ``ASDPlasticMaterial3D``).
    """

    tag: int
    ele_tag: int
    response: str


@dataclass(frozen=True, slots=True)
class StepHookRampRecord:
    """A per-step linear ramping hook bound to one or more parameters.

    Bundles the four things that materialize one
    :class:`apeGmsh.opensees.InitialStress` composite into the deck:

    1. Dispatcher boilerplate (emitted once across the emitter's
       lifetime; subsequent ramps reuse the same dispatcher).
    2. ``parameter $tag`` declaration for each tag in :attr:`targets`.
    3. The per-step procedure body — advances an internal counter,
       computes ``factor = min(count / n_steps_to_full, 1.0)``, then
       for each ``(param_tag, target_value)`` computes the delta
       against the previous step's cumulative value and emits
       ``updateParameter $param_tag $delta``.
    4. Registration with the before / after-step hook dispatcher list
       (``_apesees_before_step_hooks`` / ``_apesees_after_step_hooks``).

    Once any :class:`StepHookRampRecord` has been emitted on an
    emitter, subsequent :meth:`Emitter.analyze` calls MUST wrap the
    analyze loop with hook dispatcher calls.

    ``targets`` is ordered: the first entry of each ramp proc emits
    its updateParameter first.  Order is stable across emitters.
    """

    name: str
    targets: tuple[tuple[int, float], ...]
    n_steps_to_full: float
    phase: Literal["before", "after"]
