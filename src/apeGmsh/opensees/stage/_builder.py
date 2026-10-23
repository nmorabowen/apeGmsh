"""``_StageBuilder`` -- the context manager behind ``ops.stage(name)``.

Moved verbatim out of ``apesees.py`` (program slice S1-a, #1449); no
body changed.  The class is private, so there is no facade in
``apesees.py``: callers import it from here.
"""

from __future__ import annotations

import math
import numbers
import warnings
from typing import TYPE_CHECKING, Any, Callable, Iterable, Sequence

from .._internal.build.errors import BridgeError
from .._internal.build.stage_emit import (
    MATERIAL_STATE_FORK_FLOOR,
    MATERIAL_STATE_WRAPPERS,
    SWEPT_MATERIAL_CLASSES,
)
from .._internal.build.stage_validate import stage_flips
from .._internal.build.records import (
    DampingAttachRecord,
    ActivateAbsorbingRecord,
    ElementRemovalRecord,
    FixRecord,
    InitialStressRecord,
    MassRecord,
    MaterialStageRecord,
    UpdateParameterRecord,
    RayleighRecord,
    RegionAssignmentRecord,
    ProfileRecord,
    SPRemovalRecord,
    StageRecord,
    SupportRecord,
    ZeroVelocityRecord,
)
from .._internal.ns import _StageDampingNS
from .._internal.types import (
    Element,
    NDMaterial,
    Primitive,
    Recorder,
    TimeSeries,
)
from ..analysis.analysis import Static
from ..node import Node, _iter_tags, _select_nodes
from ._chain import (
    CHAIN_SLOTS,
    InertStaticDtWarning,
    StageChain,
    algorithm_needs_test,
)

if TYPE_CHECKING:
    from apeGmsh._kernel.records._constraints import (
        ConstraintRecord,
        InterfaceRecord,
    )
    from ..analysis.strategy import Ladder
    from ..apesees import apeSees
    from ..pattern.pattern import Plain


# nDMaterial classes that honour ``updateMaterialStage`` (their
# ``setParameter`` answers ``updateMaterialStage``).  Matched by
# ``type(handle).__name__`` so the stage verb does not have to import the
# material classes.  Flipping anything else is a silent no-op in
# OpenSees, so ``s.update_material_stage`` refuses it instead.
STAGED_MATERIAL_CLASSES: frozenset[str] = frozenset({
    "ManzariDafalias", "SAniSandMS", "LadrunoSANISAND",
    "PressureIndependMultiYield", "PressureDependMultiYield",
    "PressureDependMultiYield02", "PM4Sand", "PM4Silt",
})

#: The classes that also take stage 2: elastic with pressure-dependent
#: moduli (``loadStage == 2`` in ``PressureDependMultiYield*.cpp``).
STAGE_2_MATERIAL_CLASSES: frozenset[str] = frozenset({
    "PressureDependMultiYield", "PressureDependMultiYield02",
})

#: nDMaterial classes whose ``updateMaterialStage`` flag is a C++
#: ``static``, mapped to that static.  The flag holds ONE value
#: per process: ``updateMaterialStage -material A`` switches every
#: material of the group, and for a flip to 1 only ``A`` runs
#: ``Elastic2Plastic()``, so an unnamed material lands on an answer that
#: matches neither the elastic nor the properly flipped run (measured on
#: stock openseespy 3.8.0 and on the fork alike).  Classes that share one
#: static share one value here, and the emit-time check in
#: ``_internal/build/stage_validate.py`` counts their materials together:
#: ``LadrunoSANISAND`` subclasses ``ManzariDafalias`` and inherits its
#: ``mElastFlag``.  A class whose flag is per tag
#: (the multi-yield soils keep ``loadStagex[matN]``) is not listed.
#: A new class-wide class is one entry ``"<Class>": "<Class>::<static>"``.
#: The static also survives ``ops.wipe()``: an in-process re-run starts at
#: the previous run's value, while these checks assume each model starts
#: at 0.
CLASS_WIDE_STAGE_FLAG: dict[str, str] = {
    "ManzariDafalias": "ManzariDafalias::mElastFlag",
    "LadrunoSANISAND": "ManzariDafalias::mElastFlag",
    "SAniSandMS": "SAniSandMS::mElastFlag",
    "PM4Sand": "PM4Sand::me2p",
    "PM4Silt": "PM4Silt::me2p",
}

#: nDMaterial classes whose ``setParameter`` takes a stage-flag name
#: without checking the material tag that follows it, mapped to the names
#: it takes that way.  A ``materialState <tagX>`` that reaches an element
#: carrying one of them switches its stage, whatever ``tagX`` is: for
#: DruckerPrager ``mElastFlag=1`` is elastic, so a flip of a neighbouring
#: sand to plastic makes it elastic (a tri6n in simple shear: a stress
#: of -18.3 becomes -1115.4 on stock 3.8.0 and the fork,
#: ``tests/opensees/stage/test_tag_blind_stage_parameter.py``).  Read from the
#: ``setParameter`` bodies (``DruckerPrager.cpp``, ``BoundingCamClay.cpp``,
#: ``StressDensityModel.cpp``, ``stressDensity.cpp``); every other
#: ``nD`` class that takes either name compares ``argv[1]`` with its own
#: tag first.  DruckerPrager returns -1 for ``updateMaterialStage``.  The
#: stage close refuses an ``s.update_parameter`` of a
#: :data:`STAGE_FLAG_PARAMETERS` name whose targets carry one of these
#: classes under another tag, and an ``s.update_material_stage`` in a
#: model whose elements carry a class listed here for
#: ``updateMaterialStage`` (``MaterialStageParameter::setDomain`` stops at
#: the first element that accepts it, which could be that one).  An
#: upstream OpenSees defect; a new tag-blind class is one entry here.
TAG_BLIND_STAGE_PARAMETER: dict[str, frozenset[str]] = {
    "DruckerPrager": frozenset({"materialState"}),
    "BoundingCamClay": frozenset({"materialState"}),
    "StressDensityModel": frozenset({"materialState", "updateMaterialStage"}),
    "stressDensity": frozenset({"materialState", "updateMaterialStage"}),
}

#: nDMaterial classes whose ``setParameter`` takes ``materialState``
#: (read from the ``nD`` sources: the only ones with that branch, plus
#: ``LadrunoSANISAND``, which inherits ManzariDafalias's).  On any other
#: class, the multi-yield soils included, ``s.update_parameter(
#: "materialState", v, material=m)`` reaches no object: OpenSees warns
#: and the run keeps its stage, so the call is refused.
MATERIAL_STATE_CLASSES: frozenset[str] = frozenset({
    "ManzariDafalias", "LadrunoSANISAND", "SAniSandMS", "PM4Sand",
    "PM4Silt", "J2CyclicBoundingSurface",
    *(cls for cls, names in TAG_BLIND_STAGE_PARAMETER.items()
      if "materialState" in names),
})

#: The ``setParameter`` names that set a class-wide stage flag.  Every
#: class of :data:`CLASS_WIDE_STAGE_FLAG` answers both (ManzariDafalias,
#: inherited by LadrunoSANISAND, SAniSandMS, PM4Sand and PM4Silt alike),
#: so ``s.update_parameter(<name>, v, material=m)`` flips the group as
#: ``s.update_material_stage`` does and the emit-time check counts it.
#: Only ``materialState`` acts through ``s.update_parameter``: the stage
#: close refuses ``updateMaterialStage`` there, and a
#: ``materialState`` update only on hosts of
#: :data:`MATERIAL_STATE_ELEMENTS`.
STAGE_FLAG_PARAMETERS: frozenset[str] = frozenset(
    {"updateMaterialStage", "materialState"},
)

#: The class-wide classes whose Ladruno-fork constructor resets their
#: static to 0 (elastic).  Every element made with one of them copies the
#: material through that constructor, so a stage that activates such an
#: element after a stage flipped the group to plastic silently puts the
#: whole group back to elastic, on the fork only (stock keeps the flag;
#: SAniSandMS, PM4Sand and PM4Silt keep it on both builds, measured).
#: Fork commit ``43b235bbb`` stops the reset and ``FORK_PIN`` carries it,
#: but every older fork build still resets (79e062367, measured).  The
#: build is known only where the model runs, and a staged model runs only
#: as a deck (``apeSees.analyze`` / ``run`` and ``OpenSeesModel.build(
#: "live")`` refuse stages), so the check is the deck's: a deck that
#: builds an element or material after a flip of the group, and solves
#: before flipping it again, asks the running build for ``ladrunoBuild``
#: and stops unless the sha is one measured to carry the fix
#: (``_internal/build/stage_emit.MD_STATIC_RESET_GUARD``, on every target
#: mode; stock has no ``ladrunoBuild`` and runs).
FORK_RESETS_STAGE_STATIC: frozenset[str] = frozenset(
    {"ManzariDafalias", "LadrunoSANISAND"},
)

#: The classes whose flip to a plastic stage runs a per-instance
#: ``Elastic2Plastic()`` (it re-seats the back-stress on the current
#: stress).  ``updateMaterialStage`` reaches it through
#: ``MaterialStageParameter::setDomain``, which stops at the FIRST element
#: that accepts the parameter, so only that element's integration points
#: are initialised; every other element of the material turns plastic
#: uninitialised (measured on stock 3.8.0 and the fork).  The emit
#: therefore updates each carrying element with ``materialState`` before
#: the ``updateMaterialStage`` line
#: (``_internal/build/stage_emit.MaterialStageCarriers``), and a plastic
#: flip of such a material on more than one element passes only when
#: every element carrying it is of a :data:`MATERIAL_STATE_ELEMENTS`
#: class.  SAniSandMS, PM4Sand and PM4Silt only set their flag.  The emit
#: side's own set, so the two cannot disagree.
FIRST_ELEMENT_ONLY_PLASTIC_INIT: frozenset[str] = SWEPT_MATERIAL_CLASSES

#: The element classes whose ``setParameter`` hands ``materialState
#: <tag>`` to every integration-point material, so the sweep that
#: initialises each element of a plastic flip reaches them.  Measured
#: by the census of ``tests/opensees/integration/
#: test_md_stage_route_elements.py``: two elements of the class on one
#: material, the
#: swept flip against the per-element reference (each element its own
#: material and its own ``updateMaterialStage``), equal bit for bit, with
#: one ``Elastic2Plastic()`` per integration point, on stock openseespy
#: 3.8.0 (``SixNodeTri``, ``FourNodeTetrahedron``, ``SSPquad``,
#: ``SSPbrick``) and on the fork build 79e062367 (the first seven).
#: ``LadrunoQuad`` (std/bbar/ssp), ``LadrunoCST``, ``LadrunoLST``,
#: ``LadrunoBrick`` (std/bbar/uri/ssp/eas, and corot std/bbar) and
#: ``LadrunoBrick20`` (std/uri) are measured on the live-fork lane at
#: ``FORK_PIN`` 100eb9ae (run 39874892907,
#: ``tests/opensees/integration/test_md_stage_ladruno_at_pin.py``); every
#: one dropped the sweep on 79e062367, which predates the fork fix.  They
#: join from the emit side's fork-floor table
#: (``stage_emit.MATERIAL_STATE_FORK_FLOOR``), so one table names them
#: here and in the deck guard that stops a deck by name on a fork build
#: older than the floor.
#: ``SSPquad`` and ``SSPbrick`` have no bridge primitive yet
#: (``element/solid.py`` defers them), so no spec carries those names
#: today; they are listed so the primitive that adds them is not refused.
#: Every other class is refused for a multi-element plastic flip.
#: ``FourNodeQuad``, ``Tri31`` and ``stdBrick`` read any ``material*``
#: name as a per-point address and drop the sweep.
#: ``TenNodeTetrahedron`` forwards on the fork but crashed the stock
#: reference, so it is not proven on every build.  A new class joins only
#: with a census row on each build it runs on, and with its token in
#: ``stage_emit.MATERIAL_STATE_ELEMENT_TOKENS`` (the emit side's carrier
#: table; a class listed here without one would pass the stage close and
#: emit the bare first-element-only flip).
MATERIAL_STATE_ELEMENTS: frozenset[str] = frozenset({
    "SixNodeTri", "FourNodeTetrahedron", "SSPquad", "SSPbrick",
    "BezierTri6", "BezierTet10", "LadrunoUP",
}) | frozenset(MATERIAL_STATE_FORK_FLOOR)


# ---------------------------------------------------------------------------
# Shared validation for initial_stress (used by ops.initial_stress PULL +
# s.initial_stress PUSH).
# ---------------------------------------------------------------------------


def _build_initial_stress_record(
    *,
    source_label: str,
    name: str,
    pg: str | None,
    elements: "Iterable[int] | None",
    sigma_xx: float,
    sigma_yy: float,
    sigma_zz: float,
    ramp_steps: int,
    lambda_install: float,
) -> "InitialStressRecord":
    """Validate inputs and construct an :class:`InitialStressRecord`.

    Shared by :meth:`apeSees.initial_stress` (the bridge-global PULL
    factory) and :meth:`_StageBuilder.initial_stress` (the stage-bound
    PUSH method).  ``source_label`` prefixes every error message so
    users see which API surface they violated.

    Validation rules (identical to the historical inline checks):

    * Exactly one of ``pg=`` / ``elements=``.
    * ``name`` non-empty and a valid Tcl identifier.
    * ``ramp_steps >= 1``.
    * ``lambda_install in (0, 1]``.
    """
    if (pg is None) == (elements is None):
        raise ValueError(
            f"{source_label}: supply exactly one of pg= or "
            f"elements= (got pg={pg!r}, elements={elements!r})."
        )
    if not name:
        raise ValueError(
            f"{source_label}: name= must be non-empty."
        )
    if not name.replace("_", "").isalnum() or name[0].isdigit():
        raise ValueError(
            f"{source_label}: name must be a valid Tcl identifier "
            "(alphanumeric + underscore, not starting with a digit). "
            f"Got name={name!r}."
        )
    if ramp_steps < 1:
        raise ValueError(
            f"{source_label}: ramp_steps must be >= 1, "
            f"got {ramp_steps}."
        )
    if not (0.0 < lambda_install <= 1.0):
        raise ValueError(
            f"{source_label}: lambda_install must be in (0, 1], "
            f"got {lambda_install}."
        )
    elements_tuple = (
        tuple(int(e) for e in elements) if elements is not None else None
    )
    return InitialStressRecord(
        name=str(name),
        pg=pg,
        elements=elements_tuple,
        sigma_xx=float(sigma_xx),
        sigma_yy=float(sigma_yy),
        sigma_zz=float(sigma_zz),
        ramp_steps=int(ramp_steps),
        lambda_install=float(lambda_install),
    )


# ---------------------------------------------------------------------------
# _StageBuilder — context manager backing ops.stage(name) (Phase SSI-2.A)
# ---------------------------------------------------------------------------


class _Omitted:
    """Marks a ``s.analysis`` keyword the caller did not pass."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "<omitted>"


#: Default of the ``s.analysis`` chain keywords: tells "not passed" apart
#: from an explicit ``None`` (a chain plus any explicit slot is refused).
_OMITTED: Any = _Omitted()


class _StageBuilder:
    """Collects per-stage records inside a ``with ops.stage(...) as s:``
    block; emits a :class:`StageRecord` to the bridge on context-exit.

    Lifecycle:

    1. Constructed by :meth:`apeSees.stage` — holds a back-reference
       to the bridge.
    2. Inside the ``with`` block, the user calls ``s.add(record)``,
       ``s.analysis(test=, algorithm=, ...)``, and ``s.run(n=, dt=)``.
    3. On ``__exit__`` (clean exit only), validates that all required
       fields are populated and appends a frozen :class:`StageRecord`
       to ``bridge._stage_records``.  On exception, the stage is
       discarded (caller's exception propagates).

    The builder is NOT a typed primitive — it does not register a tag
    with the bridge.  The records / analysis-chain primitives it
    references ARE registered (independently, via their own
    namespace calls) and therefore appear in
    :attr:`apeSees._primitives` for the topological emit pass.  The
    stage record holds REFERENCES into those primitives, not copies.
    """

    __slots__ = (
        "_bridge", "_name",
        "_initial_stress_records",
        "_activate_absorbing_records",
        "_activated_pgs",
        # Transient → static handover: ``s.zero_velocities`` pool.
        "_zero_velocity_records",
        # Phase SSI-2.D (PR-B + PR-C): stage-bound BC + recorder pools.
        "_fix_records",
        "_mass_records",
        "_region_records",
        "_recorder_specs",
        # ADR 0053 D5: stage-bound damping pools + the ``s.damping`` namespace.
        "_rayleigh_records",
        "_damping_attach_records",
        "damping",
        # ADR 0051 (BL-3): stage-scoped load patterns created via
        # ``s.pattern(series=)``.
        "_pattern_specs",
        # ADR 0052 slice 1: stage-bound HOLD supports (``s.support``) +
        # the lazily-created per-stage ``Plain`` HOLD pattern.
        "_support_records",
        "_support_pattern",
        # Stage-bound constraint pool — populated by s.embedded /
        # s.equal_dof / s.rigid_link / s.tie / s.tied_contact /
        # s.kinematic_coupling / s.node_to_surface.
        "_stage_constraint_records",
        # ADR 0093 S7: stage-bound interface pool — populated by
        # s.interface(name=), kept apart from the MP pool above.
        "_stage_interface_records",
        # Phase SSI-2.E: between-stage Domain mutators.  Removal pools
        # emit BEFORE the stage's new fix / mass / region lines; the
        # three scalar fields emit at well-defined slots (set_time +
        # set_creep right after stage_open; pre_analyze_reset right
        # before analyze).
        "_remove_sp_records",
        "_remove_element_records",
        "_update_material_stage_records",
        # Typed pass-through over parameter / updateParameter.
        "_update_parameter_records",
        "_set_time",
        "_set_creep_on",
        "_pre_analyze_reset",
        "_test", "_algorithm", "_integrator",
        "_constraints", "_numberer", "_system", "_analysis",
        "_n_increments", "_dt",
        # ADR 0057 Phase A: optional solution-strategy ladder for the
        # stage's analyze loop (set via ``s.run(strategy=)``).
        "_strategy",
        "_analysis_set", "_run_set",
        # TIMs A8: optional per-stage profiler bracket (``s.profile``).
        "_profile",
    )

    def __init__(self, bridge: "apeSees", name: str) -> None:
        self._bridge = bridge
        self._name = name
        self._initial_stress_records: list[InitialStressRecord] = []
        self._activate_absorbing_records: list[ActivateAbsorbingRecord] = []
        self._activated_pgs: list[str] = []
        # Transient → static handover: nodal vel / accel zeroing pool.
        self._zero_velocity_records: list[ZeroVelocityRecord] = []
        # Phase SSI-2.D PR-B: stage-bound BC pools (fix + mass).
        self._fix_records: list[FixRecord] = []
        self._mass_records: list[MassRecord] = []
        # Phase SSI-2.D PR-C: stage-bound region + recorder pools.
        self._region_records: list[RegionAssignmentRecord] = []
        self._recorder_specs: list[Recorder] = []
        # ADR 0053 D5: stage-bound damping pools + the ``s.damping``
        # namespace (rayleigh + object forms; modal raises — deferred).
        self._rayleigh_records: list[RayleighRecord] = []
        self._damping_attach_records: list[DampingAttachRecord] = []
        self.damping = _StageDampingNS(bridge, self)
        # ADR 0051 (BL-3): stage-scoped load patterns.
        self._pattern_specs: list["Plain"] = []
        # ADR 0052 slice 1: stage-bound HOLD supports + the dedicated
        # per-stage ``Plain`` HOLD pattern (created lazily on the first
        # ``s.support`` call in this stage; None until then).
        self._support_records: list["SupportRecord"] = []
        self._support_pattern: "Plain | None" = None
        # Stage-bound constraint pool — flat list of resolved
        # ConstraintRecord instances.  Emit-time dispatches by
        # isinstance into the six per-kind emit helpers.
        self._stage_constraint_records: list["ConstraintRecord"] = []
        # ADR 0093 S7: stage-claimed InterfaceRecord rows (see
        # :meth:`interface`).
        self._stage_interface_records: list["InterfaceRecord"] = []
        # Phase SSI-2.E: between-stage Domain mutators.
        self._remove_sp_records: list[SPRemovalRecord] = []
        self._remove_element_records: list[ElementRemovalRecord] = []
        self._update_material_stage_records: list[MaterialStageRecord] = []
        # Typed pass-through over parameter / updateParameter.
        self._update_parameter_records: list[UpdateParameterRecord] = []
        self._set_time: float | None = None
        self._set_creep_on: bool | None = None
        self._pre_analyze_reset: bool = False
        self._test: Primitive | None = None
        self._algorithm: Primitive | None = None
        self._integrator: Primitive | None = None
        self._constraints: Primitive | None = None
        self._numberer: Primitive | None = None
        self._system: Primitive | None = None
        self._analysis: Primitive | None = None
        self._n_increments: int = 0
        self._dt: float | None = None
        self._strategy: "Ladder | None" = None
        self._analysis_set: bool = False
        self._run_set: bool = False
        # TIMs A8: optional per-stage profiler bracket.
        self._profile: "ProfileRecord | None" = None

    def __enter__(self) -> "_StageBuilder":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        # Clear the bridge's open-builder slot regardless of how we
        # exit (exception or clean close) so subsequent
        # ``ops.stage(...)`` calls work.  Set in
        # ``apeSees.stage(name)``.
        self._bridge._open_stage_builder = None
        if exc_type is not None:
            # Don't swallow user's exception; just drop the in-progress
            # stage (no records appended to the bridge).
            return
        # Validate: every stage MUST have a complete analysis chain
        # and a run() call.  Missing-piece errors are caller errors.
        if not self._analysis_set:
            raise ValueError(
                f"Stage {self._name!r}: missing s.analysis(...) — "
                "every stage must declare its analysis chain (test, "
                "algorithm, integrator, constraints, numberer, system, "
                "analysis)."
            )
        if not self._run_set:
            raise ValueError(
                f"Stage {self._name!r}: missing s.run(n_increments=, "
                "dt=) — every stage must declare its analyze loop."
            )
        self._check_chain_against_run()
        self._check_multi_element_plastic_flip()
        self._check_stage_parameter_routes()
        self._check_activation_after_plastic_flip()
        record = StageRecord(
            name=self._name,
            initial_stress_records=tuple(self._initial_stress_records),
            test=self._test,
            algorithm=self._algorithm,
            integrator=self._integrator,
            constraints=self._constraints,
            numberer=self._numberer,
            system=self._system,
            analysis=self._analysis,
            n_increments=int(self._n_increments),
            dt=None if self._dt is None else float(self._dt),
            strategy=self._strategy,
            activated_pgs=tuple(self._activated_pgs),
            zero_velocity_records=tuple(self._zero_velocity_records),
            fix_records=tuple(self._fix_records),
            mass_records=tuple(self._mass_records),
            region_records=tuple(self._region_records),
            recorder_specs=tuple(self._recorder_specs),
            rayleigh_records=tuple(self._rayleigh_records),
            damping_attach_records=tuple(self._damping_attach_records),
            pattern_specs=tuple(self._pattern_specs),
            support_records=tuple(self._support_records),
            support_pattern=self._support_pattern,
            stage_constraint_records=tuple(self._stage_constraint_records),
            stage_interface_records=tuple(self._stage_interface_records),
            remove_sp_records=tuple(self._remove_sp_records),
            remove_element_records=tuple(self._remove_element_records),
            update_material_stage_records=tuple(
                self._update_material_stage_records,
            ),
            update_parameter_records=tuple(self._update_parameter_records),
            set_time=self._set_time,
            set_creep_on=self._set_creep_on,
            pre_analyze_reset=self._pre_analyze_reset,
            activate_absorbing_records=tuple(self._activate_absorbing_records),
            profile=self._profile,
        )
        self._bridge._stage_records.append(record)

    def _class_wide_members(
        self,
    ) -> "tuple[dict[int, Primitive], set[int], dict[int, str]]":
        """The model's :data:`CLASS_WIDE_STAGE_FLAG` materials so far.

        Returns the materials by identity (the ones registered on the
        bridge plus any an element registered so far depends on), the
        identities an element uses, and each tagged material's group.
        """
        members: dict[int, Primitive] = {}
        used: set[int] = set()
        seen: set[int] = set()
        frontier: list[tuple[Primitive, bool]] = [
            (p, isinstance(p, Element)) for p in self._bridge._primitives]
        while frontier:
            prim, under_element = frontier.pop()
            if under_element:
                if id(prim) in used:
                    continue
                used.add(id(prim))
            elif id(prim) in seen:
                continue
            seen.add(id(prim))
            frontier.extend(
                (dep, under_element) for dep in prim.dependencies())
            if type(prim).__name__ in CLASS_WIDE_STAGE_FLAG:
                members[id(prim)] = prim
        group_of_tag: dict[int, str] = {}
        for mat in members.values():
            tag = self._bridge.tag_for(mat)
            if tag is not None:
                group_of_tag[int(tag)] = CLASS_WIDE_STAGE_FLAG[
                    type(mat).__name__]
        return members, used, group_of_tag

    def _first_element_only_carriers(
        self, tags: "set[int]",
    ) -> "tuple[dict[int, list[tuple[Element, Any]]], dict[int, set[str]]]":
        """Each element spec registered so far that carries a tag of ``tags``.

        ``tags`` are nDMaterial tags of
        :data:`FIRST_ELEMENT_ONLY_PLASTIC_INIT` materials.  Returns, per
        tag, ``(spec, fan-out)`` for every element spec whose dependencies
        reach that material, whichever stage activates it.  The fan-out is
        the spec's OpenSees elements: its PG, or one
        ``MISSING_FEM_ELEMENT_ID`` row for a node-pair spec.

        The second dict holds, per tag, the classes of every primitive
        between an element and the material (a wrapper such as
        ``StagedStrain``, or a section), which the plastic-flip check holds
        to :data:`MATERIAL_STATE_WRAPPERS`.
        """
        from .._internal.build.pg_expand import expand_spec_to_elements

        carriers: dict[int, list[tuple[Element, Any]]] = {}
        carrier_via: dict[int, set[str]] = {}
        for spec in self._bridge._primitives:
            if not isinstance(spec, Element):
                continue
            carried: set[int] = set()
            seen: set[int] = set()
            frontier: list[tuple[Primitive, tuple[str, ...]]] = [
                (dep, ()) for dep in spec.dependencies()]
            while frontier:
                dep, via = frontier.pop()
                if id(dep) in seen:
                    continue
                seen.add(id(dep))
                below = (*via, type(dep).__name__)
                frontier.extend((d, below) for d in dep.dependencies())
                if type(dep).__name__ not in FIRST_ELEMENT_ONLY_PLASTIC_INIT:
                    continue
                tag = self._bridge.tag_for(dep)
                if tag is not None and int(tag) in tags:
                    carried.add(int(tag))
                    carrier_via.setdefault(int(tag), set()).update(via)
            if not carried:
                continue
            fanout = expand_spec_to_elements(self._bridge.fem, spec)
            for tag in carried:
                carriers.setdefault(tag, []).append((spec, fanout))
        return carriers, carrier_via

    def _check_multi_element_plastic_flip(self) -> None:
        """Refuse a plastic flip that cannot initialise every element (#2445).

        A :data:`FIRST_ELEMENT_ONLY_PLASTIC_INIT` material flipped to a
        stage ``>= 1``, through either route of ``stage_flips``, on
        more than one OpenSees element needs ``Elastic2Plastic()`` in
        every element.  Elements are counted over every element spec
        registered so far, whichever stage activates them.

        * ``s.update_material_stage`` emits a ``materialState`` update of
          every carrying element of a :data:`MATERIAL_STATE_ELEMENTS`
          class (``stage_emit.MaterialStageCarriers``).  A carrier of any
          other class is refused by name, because the update would skip
          it silently.
        * ``s.update_parameter("materialState", v, material=m)`` reaches
          only the elements it targets.  The same class rule holds, and
          this stage's targets must cover every element carrying ``m``,
          unless this stage also flips ``m`` with
          ``s.update_material_stage``.
        * ``s.update_parameter("updateMaterialStage", ...)`` is refused:
          that parameter id reads ``Information::theInt``, which
          ``updateParameter`` never sets.

        Stage 0 flips and one-element materials pass unchanged.
        """
        from .._internal.build.pg_expand import expand_pg_to_elements
        from .._internal.tag_resolution import MISSING_FEM_ELEMENT_ID

        plastic = {
            tag for tag, value in stage_flips(
                self._update_material_stage_records,
                self._update_parameter_records)
            if value >= 1
        }
        if not plastic:
            return
        carriers, carrier_via = self._first_element_only_carriers(plastic)
        multi = {
            tag: specs for tag, specs in carriers.items()
            if sum(len(fanout) for _spec, fanout in specs) > 1
        }
        if not multi:
            return

        def what(tag: int) -> str:
            n = sum(len(fanout) for _spec, fanout in multi[tag])
            return f"nDMaterial tag {tag} ({n} elements)"

        for tag in sorted(multi):
            refused = sorted({
                type(spec).__name__ for spec, _fanout in multi[tag]
                if type(spec).__name__ not in MATERIAL_STATE_ELEMENTS
            })
            if refused:
                raise BridgeError(
                    f"Stage {self._name!r}: flips {what(tag)} to a plastic "
                    f"stage, but {', '.join(refused)} carries it, and that "
                    "class drops the materialState update that initialises "
                    "each element (issue #2445).  updateMaterialStage alone "
                    "runs Elastic2Plastic() on the first element only, and "
                    "this class reads any 'material...' parameter as a "
                    "per-point address, so every other element would turn "
                    "plastic with its back-stress never re-seated: a wrong "
                    "answer on stock and on the fork alike.  Carry the "
                    "material on one of "
                    f"{', '.join(sorted(MATERIAL_STATE_ELEMENTS))}, give "
                    "each element its own material, or keep it on a single "
                    "element.  A stage 0 (elastic) flip is unaffected."
                )
            between = sorted(
                carrier_via.get(tag, set()) - set(MATERIAL_STATE_WRAPPERS))
            if between:
                raise BridgeError(
                    f"Stage {self._name!r}: flips {what(tag)} to a plastic "
                    f"stage, but its elements reach it through "
                    f"{', '.join(between)}, which is not known to hand the "
                    "materialState update on to the material it wraps "
                    "(issue #2445).  The update that initialises each "
                    "element could stop there silently, leaving every "
                    "element but the first plastic with its back-stress "
                    "never re-seated.  Wrap the material only in "
                    f"{', '.join(sorted(MATERIAL_STATE_WRAPPERS))}, give "
                    "each element its own material, or keep it on a single "
                    "element."
                )
        swept = {
            int(t) for rec in self._update_material_stage_records
            if int(rec.stage) >= 1 for t in rec.mat_tags
        }
        targeted: dict[int, set[int]] = {}
        for prec in self._update_parameter_records:
            if (prec.mat_tag is None or int(prec.mat_tag) not in multi
                    or prec.name not in STAGE_FLAG_PARAMETERS
                    or int(prec.value) < 1):
                continue
            tag = int(prec.mat_tag)
            if prec.name != "materialState":
                raise BridgeError(
                    f"Stage {self._name!r}: s.update_parameter("
                    f"{prec.name!r}, {prec.value}) flips {what(tag)} to a "
                    "plastic stage through the updateMaterialStage "
                    "parameter id, which reads Information::theInt, while "
                    "updateParameter sets only theDouble (issue #2445).  "
                    "Use s.update_material_stage, or s.update_parameter("
                    "'materialState', ...) over every element carrying "
                    "the material."
                )
            if prec.elements is not None:
                eids = {int(e) for e in prec.elements}
            else:
                assert prec.pg is not None  # update_parameter takes one
                eids = {
                    int(e) for e in expand_pg_to_elements(
                        self._bridge.fem, prec.pg).eids}
            targeted.setdefault(tag, set()).update(eids)
        for tag in sorted(targeted):
            if tag in swept:
                continue
            missed = sum(
                1 for _spec, fanout in multi[tag] for eid in fanout.eids
                if int(eid) == MISSING_FEM_ELEMENT_ID
                or int(eid) not in targeted[tag]
            )
            if missed:
                raise BridgeError(
                    f"Stage {self._name!r}: s.update_parameter("
                    f"'materialState', ...) flips {what(tag)} to a plastic "
                    f"stage but leaves {missed} of those elements out of "
                    "its targets (issue #2445).  The flag is class-wide, so "
                    "they turn plastic too, without their own "
                    "Elastic2Plastic(): a wrong answer.  Target every "
                    "element carrying the material (pg= or elements=), or "
                    "use s.update_material_stage, which sweeps them all."
                )

    def _check_stage_parameter_routes(self) -> None:
        """Refuse a stage-flag update that reaches nothing, or the wrong
        material (#2458, #2460).

        For each ``s.update_parameter`` of this stage whose name is in
        :data:`STAGE_FLAG_PARAMETERS`:

        * ``updateMaterialStage`` is refused.  ``updateParameter`` goes
          through ``Parameter::update(double)``, which sets only
          ``theDouble``, and every class that takes the name (each of
          :data:`STAGED_MATERIAL_CLASSES`, PM4Sand, PM4Silt, the
          stress-density models) reads ``Information::theInt`` for it,
          which nothing sets: the stage the run lands on is uninitialised
          memory, while the bridge counted a flip to the value given.
        * ``materialState`` is refused when no targeted element carries the
          named material (the update would reach nothing, yet count as a
          flip).
        * ``materialState`` is refused when a targeted element that carries
          the named material is not of a :data:`MATERIAL_STATE_ELEMENTS`
          class.  ``FourNodeQuad``,
          ``Tri31`` and ``stdBrick`` read any ``material...`` name as their
          own per-point address, so the update never reaches the material.
        * ``materialState`` is refused when a targeted element carries a
          :data:`TAG_BLIND_STAGE_PARAMETER` class under a tag other than
          the ``material=`` it names: that material switches too.

        And an ``s.update_material_stage`` is refused while any element
        carries a class listed there for ``updateMaterialStage``.
        """
        from .._internal.build.pg_expand import (
            expand_pg_to_elements,
            expand_spec_to_elements,
        )

        def call_of(prec: "UpdateParameterRecord") -> str:
            where = (f"pg {prec.pg!r}" if prec.elements is None
                     else f"elements {list(prec.elements)!r}")
            return (f"Stage {self._name!r}: s.update_parameter("
                    f"{prec.name!r}, {prec.value}) on {where}")

        records = [p for p in self._update_parameter_records
                   if p.name in STAGE_FLAG_PARAMETERS]
        for prec in records:
            if prec.name == "updateMaterialStage":
                cls = self._nd_class_of_tag(prec.mat_tag)
                instead = (
                    "s.update_material_stage, or s.update_parameter("
                    "'materialState', ...), which reads theDouble"
                    if cls in MATERIAL_STATE_CLASSES else
                    "s.update_material_stage")
                raise BridgeError(
                    f"{call_of(prec)} does not carry its value: "
                    "updateParameter sets only Parameter::theDouble "
                    "(Parameter::update(double)), while every material that "
                    "takes updateMaterialStage reads Information::theInt for "
                    "it, which is never set.  The stage the run lands on is "
                    "uninitialised memory (measured: the elastic answer in "
                    "one process, a plastic one in another), while the "
                    "bridge counts a flip to the value given (issue #2460).  "
                    f"Use {instead}."
                )
        if not records and not self._update_material_stage_records:
            return
        # Per element spec: the tags of the nDMaterials it reaches, and the
        # (class, tag) of each tag-blind material among them.
        specs: list[tuple[
            Element, set[int], list[tuple[str, int | None]]]] = []
        for spec in self._bridge._primitives:
            if not isinstance(spec, Element):
                continue
            carried: set[int] = set()
            blind: list[tuple[str, int | None]] = []
            seen: set[int] = set()
            frontier: list[Primitive] = list(spec.dependencies())
            while frontier:
                dep = frontier.pop()
                if id(dep) in seen:
                    continue
                seen.add(id(dep))
                frontier.extend(dep.dependencies())
                tag = self._bridge.tag_for(dep)
                if tag is not None and isinstance(dep, NDMaterial):
                    carried.add(int(tag))
                if type(dep).__name__ in TAG_BLIND_STAGE_PARAMETER:
                    blind.append((type(dep).__name__,
                                  None if tag is None else int(tag)))
            specs.append((spec, carried, blind))
        if self._update_material_stage_records:
            swallow = sorted({
                f"{type(spec).__name__} on pg {getattr(spec, 'pg', None)!r} "
                f"({cls})"
                for spec, _carried, blind in specs for cls, _tag in blind
                if "updateMaterialStage" in TAG_BLIND_STAGE_PARAMETER[cls]
            })
            if swallow:
                raise BridgeError(
                    f"Stage {self._name!r}.update_material_stage: "
                    f"{', '.join(swallow)} takes updateMaterialStage "
                    "without checking the material tag, and "
                    "MaterialStageParameter::setDomain stops at the first "
                    "element that takes it: the flip could switch that "
                    "material instead of the one it names (issue #2458).  "
                    "Remove the class from this model, or stage it in a "
                    "separate model."
                )
        fanout: "list[set[int]] | None" = None
        for prec in records:
            call = call_of(prec)
            if prec.elements is not None:
                targets = {int(e) for e in prec.elements}
            else:
                assert prec.pg is not None  # update_parameter takes one
                targets = {int(e) for e in expand_pg_to_elements(
                    self._bridge.fem, prec.pg).eids}
            if fanout is None:
                fanout = [
                    {int(e) for e in expand_spec_to_elements(
                        self._bridge.fem, spec).eids}
                    for spec, _carried, _blind in specs]
            hit = [(spec, carried, blind)
                   for (spec, carried, blind), eids in zip(specs, fanout)
                   if eids & targets]
            # update_parameter refuses a materialState that names no
            # material.  A host is lost only if it carries the named one.
            assert prec.mat_tag is not None
            mat_tag = int(prec.mat_tag)
            if not any(mat_tag in carried for _s, carried, _b in hit):
                cls = self._nd_class_of_tag(mat_tag) or "the material"
                held = sorted({
                    f"{type(spec).__name__} on pg "
                    f"{getattr(spec, 'pg', None)!r}"
                    for spec, _carried, _blind in hit})
                raise BridgeError(
                    f"{call} names {cls} (nDMaterial tag {mat_tag}), but "
                    + (f"its targets ({', '.join(held)}) do not carry it"
                       if held else "it targets no element")
                    + ": the update reaches nothing, while the bridge "
                    "counts a flip of that material (issue #2460).  Target "
                    "the elements that carry it."
                )
            hosts = sorted({
                type(spec).__name__ for spec, carried, _blind in hit
                if type(spec).__name__ not in MATERIAL_STATE_ELEMENTS
                and mat_tag in carried
            })
            if hosts:
                raise BridgeError(
                    f"{call} targets {', '.join(hosts)} elements, which are "
                    "not measured to hand materialState on to their "
                    "materials: FourNodeQuad, Tri31 and stdBrick read any "
                    "'material...' name as their own per-point address, so "
                    "OpenSees warns that no object took the parameter and "
                    "the flip is lost (issue #2460).  Carry the material "
                    "on one of "
                    f"{', '.join(sorted(MATERIAL_STATE_ELEMENTS))}, or use "
                    "s.update_material_stage."
                )
            others = sorted({
                f"{cls} (nDMaterial tag {tag})"
                for _spec, _carried, blind in hit for cls, tag in blind
                if "materialState" in TAG_BLIND_STAGE_PARAMETER[cls]
                and tag != mat_tag
            })
            if others:
                raise BridgeError(
                    f"{call} names nDMaterial tag {mat_tag}, but its targets "
                    "carry "
                    f"{', '.join(others)}, whose setParameter takes "
                    "materialState without checking the tag: the update "
                    "would switch that material's stage too, silently "
                    "(issue #2458; DruckerPrager materialState 1 is "
                    "elastic).  Target only the elements that carry the "
                    "material you flip."
                )

    def _nd_class_of_tag(self, tag: "int | None") -> "str | None":
        """The class name of the nDMaterial with ``tag`` the model holds
        (registered, or a dependency of a registered primitive), else
        ``None``."""
        if tag is None:
            return None
        seen: set[int] = set()
        frontier: list[Primitive] = list(self._bridge._primitives)
        while frontier:
            prim = frontier.pop()
            if id(prim) in seen:
                continue
            seen.add(id(prim))
            frontier.extend(prim.dependencies())
            if (isinstance(prim, NDMaterial)
                    and self._bridge.tag_for(prim) == int(tag)):
                return type(prim).__name__
        return None

    def _check_activation_after_plastic_flip(self) -> None:
        """Refuse activating an element of a plastic first-element-only group.

        An earlier stage may have left a
        :data:`FIRST_ELEMENT_ONLY_PLASTIC_INIT` group plastic.  An element
        this stage activates with such a material then copies it straight
        into the plastic stage, and no flip ever runs its
        ``Elastic2Plastic()`` (#2445).  This is refused on every build,
        unless this stage flips the group again: its flip lines emit after
        the activation and reach the new element.
        """
        if not self._activated_pgs:
            return
        _members, _used, group_of_tag = self._class_wide_members()
        last: dict[str, int] = {}
        for stage in self._bridge._stage_records:
            for tag, value in stage_flips(
                    stage.update_material_stage_records,
                    stage.update_parameter_records):
                group = group_of_tag.get(tag)
                if group is not None:
                    last[group] = value
        plastic_groups = {g for g, v in last.items() if v >= 1}
        if not plastic_groups:
            return
        this_stage = {
            group_of_tag[tag]
            for tag, _value in stage_flips(
                self._update_material_stage_records,
                self._update_parameter_records)
            if tag in group_of_tag
        }
        tags = {
            tag for tag, group in group_of_tag.items()
            if group in plastic_groups and group not in this_stage
        }
        if not tags:
            return
        activated = set(self._activated_pgs)
        for tag, specs in sorted(
                self._first_element_only_carriers(tags)[0].items()):
            new = sorted({
                f"{type(spec).__name__} on pg {getattr(spec, 'pg', None)!r}"
                for spec, _fanout in specs
                if getattr(spec, "pg", None) in activated
            })
            # A material none of whose elements was live before this
            # stage reached nothing at the earlier flip: build-time
            # validator V7 refuses that flip, with its own message.
            earlier_live = any(
                getattr(spec, "pg", None) not in activated
                for spec, _fanout in specs
            )
            if new and earlier_live:
                raise BridgeError(
                    f"Stage {self._name!r}.activate: {', '.join(new)} "
                    f"carries nDMaterial tag {tag}, whose stage flag an "
                    "earlier stage left plastic.  The new element enters "
                    "the plastic stage with no Elastic2Plastic() of its "
                    "own, so its back-stress is never re-seated (issue "
                    "#2445).  Flip the whole group again in this stage "
                    "(the flip runs after the activation and reaches it), "
                    "or activate the element before the flip."
                )

    def _check_chain_against_run(self) -> None:
        """Stage-close checks that need both ``analysis`` and ``run``.

        * A test omitted for ``Linear`` stays required when a strategy
          ladder is attached: the ladder's rungs iterate.
        * A ``Subdivide`` rung needs a ``Transient`` analysis (ADR 0057
          Amendment 1 §3).
        * D1 (R3, #1785; extended R3-S1b, #1822): under ``Static``
          OpenSees ignores ``analyze``'s ``dt``; the integrator sets the
          step.  A given ``dt`` never acts, so it warns
          (:class:`InertStaticDtWarning`), except when it equals the
          integrator's declared step field (``static_step_field``,
          ``"dlam"`` for ``LoadControl`` / ``LadrunoLoadControl``).  An
          integrator without a step field, or without the declaration,
          warns on any ``dt`` (fail loud).  Omitting ``dt`` is silent.
          ``Transient`` keeps ``dt``.
        """
        if self._test is None and self._strategy is not None:
            raise ValueError(
                f"Stage {self._name!r}: a strategy ladder needs a "
                "convergence test, because its rungs iterate; pass "
                "test= to s.analysis (or drop s.run(strategy=))."
            )
        if (self._strategy is not None
                and getattr(self._strategy, "subdivides", False)
                and isinstance(self._analysis, Static)):
            raise ValueError(
                f"Stage {self._name!r}: a Subdivide rung cuts the time "
                "step, and under Static OpenSees ignores analyze's dt (the "
                "integrator sets the step), so each piece would be a full "
                "step. Use it in a Transient stage; a static stage uses "
                "Substep (ADR 0057 Amendment 1)."
            )
        if self._dt is None or not isinstance(self._analysis, Static):
            return
        integ = self._integrator
        kind = type(integ).__name__
        field = getattr(type(integ), "static_step_field", None)
        if field is None:
            warnings.warn(
                f"dt has no effect under Static; {kind} sets the step "
                "and has no dlam for dt to match, so omit dt "
                f"(stage {self._name!r}: dt={self._dt!r}).",
                InertStaticDtWarning,
                stacklevel=3,
            )
            return
        step = float(getattr(integ, field))
        if not math.isclose(self._dt, step, rel_tol=1e-12, abs_tol=0.0):
            warnings.warn(
                f"dt has no effect under Static; {field} sets the step "
                f"(stage {self._name!r}: dt={self._dt!r}, "
                f"{field}={getattr(integ, field)!r}).",
                InertStaticDtWarning,
                stacklevel=3,
            )

    # -- Stage population -------------------------------------------------

    def add(self, record: InitialStressRecord) -> None:
        """Bind a previously-registered record to this stage.

        Currently supports :class:`InitialStressRecord` only.  The
        record is removed from the bridge's global ``_initial_stress_records``
        pool (so it does not also emit in the flat-emit zone) and
        added to this stage's pool.

        Passing a record that's NOT in the bridge's global pool
        raises ``ValueError`` — usually a sign of double-``add``ing
        the same record across stages.
        """
        if isinstance(record, InitialStressRecord):
            try:
                self._bridge._initial_stress_records.remove(record)
            except ValueError as e:
                raise ValueError(
                    f"Stage {self._name!r}.add: InitialStressRecord "
                    f"name={record.name!r} not in the bridge's global "
                    "pool — was it already added to a different stage "
                    "or registered through a different bridge instance?"
                ) from e
            self._initial_stress_records.append(record)
            return
        raise TypeError(
            f"Stage {self._name!r}.add: unsupported record type "
            f"{type(record).__name__!r}.  Phase SSI-2.A supports "
            "InitialStressRecord only; future versions may extend."
        )

    def initial_stress(
        self,
        *,
        name: str,
        pg: str | None = None,
        elements: "Iterable[int] | None" = None,
        sigma_xx: float,
        sigma_yy: float,
        sigma_zz: float,
        ramp_steps: int,
        lambda_install: float = 1.0,
    ) -> "InitialStressRecord":
        """Stage-bound PUSH mirror of :meth:`apeSees.initial_stress`.

        Signature is identical to ``ops.initial_stress(...)``; the
        record is created and appended directly to this stage's pool
        instead of the bridge's global pool — no intermediate
        ``s.add(record)`` step required.

        Equivalent to:

            record = ops.initial_stress(name=..., ...)
            s.add(record)

        but in one call, mirroring the
        ``s.fix`` / ``s.mass`` / ``s.embedded`` PUSH builder methods.
        The existing :meth:`add` PULL path remains supported for
        callers that build records globally and bind them later.

        Per-stage emission ordering is unchanged: this stage's
        initial-stress records emit AFTER the stage's analysis chain
        is established, regardless of which API surface (PUSH vs PULL)
        the record came in through.

        Returns the constructed record so callers can inspect or pass
        it to validators (mirroring ``ops.initial_stress``'s return).
        """
        record = _build_initial_stress_record(
            source_label=f"Stage {self._name!r}.initial_stress",
            name=name, pg=pg, elements=elements,
            sigma_xx=sigma_xx, sigma_yy=sigma_yy, sigma_zz=sigma_zz,
            ramp_steps=ramp_steps, lambda_install=lambda_install,
        )
        # K1-6: one key space with the flat ``ops.initial_stress``.
        self._bridge._declare_by_name(record, "initial_stress", name)
        self._initial_stress_records.append(record)
        return record

    def activate_absorbing(
        self,
        *,
        pg: str | None = None,
        elements: "Iterable[int] | None" = None,
    ) -> "ActivateAbsorbingRecord":
        """Flip this stage's absorbing-boundary elements to absorbing mode.

        Emits the one-way ``ASDAbsorbingBoundary`` stage switch (0→1) — the
        OpenSees ``parameter`` / ``addToParameter ... stage`` /
        ``updateParameter 1`` sequence (ADR 0054 AB-3) — once, after this
        stage's analysis chain is established and before its ``analyze`` loop,
        so the gravity stage has already held the boundary by penalty.  Target
        the elements by ``pg`` (typically
        ``AbsorbingSkinResult.skin_all_pg``) or an explicit ``elements`` list;
        exactly one is required.  Per-partition emission is automatic.

        Usage::

            with ops.stage("dynamic") as s:
                s.activate_absorbing(pg=skin.skin_all_pg)
                s.analysis(...); s.run(...)
        """
        if (pg is None) == (elements is None):
            raise ValueError(
                f"Stage {self._name!r}.activate_absorbing: supply exactly one "
                "of pg= or elements=."
            )
        record = ActivateAbsorbingRecord(
            pg=pg,
            elements=(
                tuple(int(e) for e in elements) if elements is not None else None
            ),
        )
        self._bridge._declare_record(  # K1-6
            record, "activate_absorbing", None)
        self._activate_absorbing_records.append(record)
        return record

    def zero_velocities(
        self, nodes: "Iterable[int | Node] | None" = None,
    ) -> "ZeroVelocityRecord":
        """Zero the nodal velocity AND acceleration state at this stage's
        boundary — the transient → static handover.

        A static stage inherits the previous transient stage's committed
        nodal velocities and accelerations (a static integrator never
        writes either), so ``recorder Node ... -dynamic`` /
        ``reactions -dynamic`` in the static stage keeps reporting the
        previous stage's inertial and damping terms as if they were
        live.  Call this on the static stage to hand it a quiescent
        kinematic state.

        Emits, per targeted node and per DOF of that node's *effective*
        ndf (a u-p node has 4, so it gets DOFs 1..4)::

            setNodeVel   <node> <dof> 0.0 -commit
            setNodeAccel <node> <dof> 0.0 -commit

        ``-commit`` is load-bearing, not decoration.  The stock handler
        (``OpenSeesMiscCommands.cpp`` ``OPS_setNodeVel`` /
        ``OPS_setNodeAccel``) rebuilds the vector from the node's
        COMMITTED state (``Node::getVel`` returns ``commitVel``) and
        writes only the TRIAL vector.  Without committing each call, the
        next DOF's call reads the OLD committed vector back and only the
        last DOF ends up zeroed — and the committed state a static stage
        actually reads is never touched at all.

        Emit slot: LAST in the stage block — after the stage's domain
        mutations, its analysis chain, its patterns and the optional
        ``s.reset()``, immediately before ``analyze``.  ``reset`` reverts
        the Domain to the last ``setTime`` (restoring the velocities),
        so the zeroing has to follow it; and nothing else may run between
        the zeroing and the step that would otherwise read the stale
        state.

        There is no cheaper mechanism.  The Ladruno fork adds
        ``ladrunoSetNodeTrial`` (``OpenSeesMiscCommands.cpp``
        ``OPS_LadrunoSetNodeTrial``), but it writes the TRIAL vectors
        only and never commits, so it cannot express this; neither stock
        nor the fork ships a domain-wide zeroing command.  The deck is
        therefore ``2 x sum(ndf)`` lines — proportional to the model.
        Pass ``nodes=`` to scope it when the whole domain is too many.

        Parameters
        ----------
        nodes
            The nodes to quiet.  ``None`` (default) means the whole
            domain — every node in ``fem.nodes.ids``.  Accepts a mix of
            plain integer tags and :class:`Node` instances, same as
            :meth:`fix`.

        Raises
        ------
        ValueError
            If ``nodes=`` is supplied but empty (an inert directive is
            almost always a mistake; omit the argument for the whole
            domain).
        """
        if nodes is None:
            record = ZeroVelocityRecord(nodes=None)
        else:
            nodes_tuple = _iter_tags(nodes)
            if not nodes_tuple:
                raise ValueError(
                    f"Stage {self._name!r}.zero_velocities: nodes= is "
                    "empty — omit the argument to zero the whole domain."
                )
            record = ZeroVelocityRecord(nodes=nodes_tuple)
        self._bridge._declare_record(  # K1-6
            record, "zero_velocities", None)
        self._zero_velocity_records.append(record)
        return record

    # -- Stage-bound constraints (CLAIM by name) -------------------------

    def embedded(self, *, name: str) -> "tuple[ConstraintRecord, ...]":
        """Claim resolved ``embedded`` constraint records by name for
        this stage.

        Constraint declaration happens at apeGmsh time via
        ``g.constraints.embedded(host_label=..., embedded_label=...,
        name=...)``, which produces resolved ``InterpolationRecord``
        rows on ``fem.elements.constraints``.  This method finds the
        rows matching ``name`` and:

        * appends them to this stage's constraint pool (so they emit
          inside the stage's block, AFTER stage regions and BEFORE the
          stage's ``domain_change``);
        * records their ``id(...)`` in the bridge's
          ``_stage_claimed_constraint_ids`` set so the global MP-
          constraint pass SKIPS them (no double emission).

        The shipped contract is **claim-by-name**, not direct create:
        the kernel resolver runs at apeGmsh / FEMData-build time
        (needs gmsh + parts), so by bridge time the records already
        exist on the FEMData broker.  See ADR 0034 §"Stage-bound
        constraints".

        Parameters
        ----------
        name
            Unique constraint name passed to
            ``g.constraints.embedded(name=...)`` at apeGmsh time.

        Returns
        -------
        tuple[ConstraintRecord, ...]
            The claimed records, in registration order on the broker.

        Raises
        ------
        ValueError
            * No record on ``fem.elements.constraints`` matches
              ``name`` (typo, or missing ``name=`` at declaration).
            * The matched record is already claimed by a different
              stage (double-claim).
        """
        return self._claim_constraints_by_name(
            name=name,
            method_label="s.embedded",
            kind="embedded",
            scope="elements",
        )

    def tie(self, *, name: str) -> "tuple[ConstraintRecord, ...]":
        """Claim resolved ``tie`` constraint records by name (claim-by-
        name; see :meth:`embedded` for the contract)."""
        return self._claim_constraints_by_name(
            name=name,
            method_label="s.tie",
            kind="tie",
            scope="elements",
        )

    def distributing(self, *, name: str) -> "tuple[ConstraintRecord, ...]":
        """Claim resolved ``distributing`` constraint records by name
        (claim-by-name; see :meth:`embedded` for the contract)."""
        return self._claim_constraints_by_name(
            name=name,
            method_label="s.distributing",
            kind="distributing",
            scope="elements",
        )

    def equal_dof(self, *, name: str) -> "tuple[ConstraintRecord, ...]":
        """Claim resolved ``equal_dof`` constraint records by name
        (claim-by-name; see :meth:`embedded` for the contract)."""
        return self._claim_constraints_by_name(
            name=name,
            method_label="s.equal_dof",
            kind="equal_dof",
            scope="nodes",
        )

    def rigid_link(self, *, name: str) -> "tuple[ConstraintRecord, ...]":
        """Claim resolved rigid-link constraint records by name.

        Spans ``rigid_beam``, ``rigid_rod``, and ``rigid_body`` since
        ``g.constraints.rigid_link(...)`` may produce any of the three
        depending on the user's flag (see ConstraintsComposite).
        Claim-by-name semantics; see :meth:`embedded` for the contract.
        """
        return self._claim_constraints_by_name(
            name=name,
            method_label="s.rigid_link",
            kind=frozenset({"rigid_beam", "rigid_rod", "rigid_body"}),
            scope="nodes",
        )

    def rigid_diaphragm(
        self, *, name: str,
    ) -> "tuple[ConstraintRecord, ...]":
        """Claim resolved ``rigid_diaphragm`` constraint records by
        name (claim-by-name; see :meth:`embedded` for the contract)."""
        return self._claim_constraints_by_name(
            name=name,
            method_label="s.rigid_diaphragm",
            kind="rigid_diaphragm",
            scope="nodes",
        )

    def kinematic_coupling(
        self, *, name: str,
    ) -> "tuple[ConstraintRecord, ...]":
        """Claim resolved ``kinematic_coupling`` constraint records by
        name (claim-by-name; see :meth:`embedded` for the contract)."""
        return self._claim_constraints_by_name(
            name=name,
            method_label="s.kinematic_coupling",
            kind="kinematic_coupling",
            scope="nodes",
        )

    def node_to_surface(
        self, *, name: str,
    ) -> "tuple[ConstraintRecord, ...]":
        """Claim resolved ``node_to_surface`` constraint records by
        name (claim-by-name; see :meth:`embedded` for the contract)."""
        return self._claim_constraints_by_name(
            name=name,
            method_label="s.node_to_surface",
            kind="node_to_surface",
            scope="nodes",
        )

    def node_to_surface_spring(
        self, *, name: str,
    ) -> "tuple[ConstraintRecord, ...]":
        """Claim resolved ``node_to_surface_spring`` constraint records
        by name (claim-by-name; see :meth:`embedded` for the contract).
        """
        return self._claim_constraints_by_name(
            name=name,
            method_label="s.node_to_surface_spring",
            kind="node_to_surface_spring",
            scope="nodes",
        )

    def tied_contact(self, *, name: str) -> "tuple[ConstraintRecord, ...]":
        """Claim a resolved ``tied_contact`` surface coupling by name for
        this stage (claim-by-name; see :meth:`embedded` for the contract).

        ``g.constraints.tied_contact(master_label=..., slave_label=...,
        name=...)`` resolves at apeGmsh time to a single
        :class:`SurfaceCouplingRecord` on ``fem.elements.constraints``
        whose ``slave_records`` hold one ``InterpolationRecord`` per slave
        node.  Claiming it routes the whole coupling into this stage's
        block: the stage adapter expands the nested slaves on emit, and
        :meth:`_claimed_constraint_ids` registers those same slave ids so
        the global surface-coupling pass (which sees the expanded slaves,
        not the outer record) correctly skips them — no double emission.

        Both the flat and partitioned stage emit paths handle the
        expansion (``_StageConstraintAdapter.interpolations`` /
        ``_emit_surface_couplings_for_rank``).
        """
        return self._claim_constraints_by_name(
            name=name,
            method_label="s.tied_contact",
            kind="tied_contact",
            scope="elements",
        )

    def interface(self, *, name: str) -> "tuple[InterfaceRecord, ...]":
        """Claim resolved ``g.constraints.interface()`` records by name
        for this stage — the liner-install pattern (ADR 0093 S7 / INV-6).

        ``g.constraints.interface(..., name="RockLiner")`` resolves at
        apeGmsh time into one
        :class:`~apeGmsh._kernel.records._constraints.InterfaceRecord`
        per coincident node pair on ``fem.elements.interfaces``.
        Claiming the name here moves the whole per-pair unit — the
        mixed-ndf phantom, its nested ``equalDOF``, the two
        tributary-scaled uniaxials and the ``zeroLength`` — out of the
        base pass and into this stage's block, emitted after the
        stage's activated topology and before its ``domain_change``.
        The interface is therefore installed on the ground the previous
        stages already equilibrated, and carries only the load
        increments applied from this stage onward.

        Deliberately NOT routed through
        :meth:`_claim_constraints_by_name`: that helper walks
        ``fem.{nodes,elements}.constraints`` and lands its matches in
        the stage's MP pool, whose emit shape is wrong for an
        interface.  This is the FIRST stage-claim path for any
        side-list record and is kept thin on purpose — contacts,
        embeds and reinforce ties still have no stage-claim path
        (ADR 0093 INV-6).

        Parameters
        ----------
        name
            The ``name=`` passed to ``g.constraints.interface(...)`` at
            apeGmsh time.

        Returns
        -------
        tuple[InterfaceRecord, ...]
            The claimed records, in broker registration order.

        Raises
        ------
        ValueError
            Empty ``name=``; no interface record carries that name
            (the message lists the names that exist); or the name is
            already claimed by another stage.
        """
        return self._claim_interfaces_by_name(name=name)

    def _claim_interfaces_by_name(
        self, *, name: str,
    ) -> "tuple[InterfaceRecord, ...]":
        """Walk ``fem.elements.interfaces``, claim matches by name.

        Duck-typed access to the side-list (the neutral-zone writers /
        fem-likes contract): ``getattr(..., "interfaces", None)``, never
        a bare attribute read.
        """
        from apeGmsh._kernel.records._kinds import ConstraintKind

        if not name:
            raise ValueError(
                f"Stage {self._name!r}.s.interface: name= must be "
                "non-empty (claim-by-name requires the user to have "
                "passed a unique name= to g.constraints.interface at "
                "apeGmsh time)."
            )
        fem = self._bridge._fem
        elements = getattr(fem, "elements", None)
        container = (
            getattr(elements, "interfaces", None)
            if elements is not None else None
        )
        if not container:
            raise ValueError(
                f"Stage {self._name!r}.s.interface: this model carries "
                "no g.constraints.interface() records at all "
                "(fem.elements.interfaces is empty) — nothing to claim. "
                f"Declare g.constraints.interface(..., name={name!r}) "
                "at apeGmsh time."
            )
        matched = [
            rec for rec in container
            if getattr(rec, "name", None) == name
            and getattr(rec, "kind", None) == ConstraintKind.INTERFACE
        ]
        if not matched:
            available = sorted({
                str(getattr(rec, "name", None))
                for rec in container
                if getattr(rec, "name", None)
            })
            known = ", ".join(repr(n) for n in available) or "<none named>"
            raise ValueError(
                f"Stage {self._name!r}.s.interface: no resolved "
                f"interface records found with name={name!r} on "
                f"fem.elements.interfaces. Available interface names: "
                f"{known}."
            )
        already = self._bridge._stage_claimed_interface_ids
        for rec in matched:
            if id(rec) in already:
                raise ValueError(
                    f"Stage {self._name!r}.s.interface: interface "
                    f"name={name!r} is already claimed by another "
                    "stage — each named interface may bind to at most "
                    "one stage (ADR 0093 S7)."
                )
        for rec in matched:
            already.add(id(rec))
            self._stage_interface_records.append(rec)
        return tuple(matched)

    # NOTE: s.mortar is intentionally out of scope. As of ADR 0073
    # ``g.constraints.mortar`` delegates to the fork contact-tie and resolves
    # to a ``ContactRecord`` on ``fem.elements.contacts`` (a serial-only
    # subsystem emitted by ``emit_contacts``), NOT a claimable MP
    # ``SurfaceCouplingRecord`` — so there is still no stage-claimable record.

    # -- Internal claim helper -------------------------------------------

    def _claim_constraints_by_name(
        self,
        *,
        name: str,
        method_label: str,
        kind: "str | frozenset[str] | None",
        scope: str,
    ) -> "tuple[ConstraintRecord, ...]":
        """Walk the FEMData constraint broker, claim matches by name.

        Parameters
        ----------
        name
            Constraint name to match (from
            ``g.constraints.<kind>(..., name=name)``).
        method_label
            Display name for error messages (e.g. ``"s.embedded"``).
        kind
            If a ``str``, filter records by ``rec.kind == kind``.
            If a ``frozenset[str]``, filter by membership (used for
            ``s.rigid_link`` which spans rigid_beam / rigid_rod /
            rigid_body).  ``None`` skips the kind check.
        scope
            ``"elements"`` (walk ``fem.elements.constraints``) or
            ``"nodes"`` (walk ``fem.nodes.constraints``).
        """
        if not name:
            raise ValueError(
                f"Stage {self._name!r}.{method_label}: name= must be "
                "non-empty (claim-by-name requires the user to have "
                "passed a unique name= to g.constraints.X at apeGmsh "
                "time)."
            )
        fem = self._bridge._fem
        if scope == "elements":
            container = getattr(
                getattr(fem, "elements", None), "constraints", None,
            )
            scope_attr = "fem.elements.constraints"
        elif scope == "nodes":
            container = getattr(
                getattr(fem, "nodes", None), "constraints", None,
            )
            scope_attr = "fem.nodes.constraints"
        else:
            raise ValueError(
                f"Stage {self._name!r}.{method_label}: invalid scope "
                f"{scope!r} (internal bug)."
            )
        if container is None:
            raise ValueError(
                f"Stage {self._name!r}.{method_label}: {scope_attr} "
                f"is None on this FEMData — no constraint broker to "
                f"claim from."
            )

        kind_check: "Callable[[object], bool] | None"
        kind_label: "str | None"
        if isinstance(kind, str):
            kind_str = kind

            def kind_check(k: object) -> bool:
                return k == kind_str

            kind_label = repr(kind)
        elif kind is not None:
            kind_set = kind

            def kind_check(k: object) -> bool:
                return k in kind_set

            kind_label = repr(sorted(kind))
        else:
            kind_check = None
            kind_label = None
        matched: list["ConstraintRecord"] = []
        for rec in container:
            if getattr(rec, "name", None) != name:
                continue
            if kind_check is not None and not kind_check(
                getattr(rec, "kind", None)
            ):
                continue
            matched.append(rec)
        if not matched:
            raise ValueError(
                f"Stage {self._name!r}.{method_label}: no resolved "
                f"constraint records found with name={name!r}"
                + (f" and kind in {kind_label}" if kind_label else "")
                + f" on {scope_attr}. Did you pass name={name!r} to "
                "the matching g.constraints.X(...) call at apeGmsh "
                "time?"
            )
        already = self._bridge._stage_claimed_constraint_ids
        for rec in matched:
            if id(rec) in already:
                raise ValueError(
                    f"Stage {self._name!r}.{method_label}: constraint "
                    f"name={name!r} is already claimed by another "
                    "stage — each named constraint may bind to at "
                    "most one stage."
                )
        for rec in matched:
            already.add(id(rec))
            self._stage_constraint_records.append(rec)
        return tuple(matched)

    def fix(
        self,
        *,
        pg: str | None = None,
        nodes: "Iterable[int | Node] | None" = None,
        label: str | Sequence[str] | None = None,
        dofs: tuple[int, ...],
        name: str | None = None,
    ) -> None:
        """Apply homogeneous SP constraints (``fix``) bound to this stage.

        Signature mirrors :meth:`apeSees.fix` verbatim: exactly one of
        ``pg`` / ``nodes`` / ``label`` must be supplied; ``nodes``
        accepts a mix of plain integer tags and :class:`Node`
        instances; ``label`` resolves on the snapshot at the call
        (ADR 0115 D4); ``dofs`` is a tuple of 0/1 flags per ndf.  The
        bridge expands ``pg`` to a per-node fan-out at emit time, same
        as the global :meth:`apeSees.fix` path.  Two ``s.fix`` calls of
        one stage on the same ``(node, DOF)`` merge (ADR 0115 D6); an
        overlap with another tier, or with an ``s.support``, raises.

        Stage-bound fix lines emit **inside this stage's block**, after
        the stage's topology (nodes + elements activated via
        :meth:`activate`) and before the stage's initial-stress
        records.  Per-rank fan-out under MP follows the same INV-4
        rules as the global path: each rank only emits ``fix`` for
        nodes it owns.

        Validators V1 / V2 (Phase SSI-2.D PR-A) gate this at build
        time — see :meth:`apeSees._run_staged_bc_validators`.

        Reference frame — absolute (ANCHOR).  ``fix`` is a homogeneous
        single-point constraint at value 0, so adding it mid-stage to a
        node that has already drifted to ``u = d`` drives that DOF back
        toward its ``t = 0`` reference position on the next ``analyze``:
        the node is *moved*, and the attached elements pick up the
        corresponding (physical, not spurious) forces.  Use this when
        you genuinely want the DOF returned to the undeformed position.
        To instead *hold* the node at its current deformed position with
        zero initial force, use :meth:`support` (ADR 0052); ``fix``
        cannot express that, as a homogeneous SP has no value lever.

        Parameters
        ----------
        pg
            Physical group whose nodes receive the fix.  XOR with
            ``nodes`` / ``label``.
        nodes
            Explicit list of node tags (or :class:`Node` instances).
            XOR with ``pg`` / ``label``.
        label
            Geometry label (or list of labels) whose nodes receive the
            fix.  XOR with ``pg`` / ``nodes``; never falls back to a
            physical group.
        dofs
            ``ndf``-length tuple of 0/1 flags — ``1`` means fix that
            DOF, ``0`` leaves it free.
        name
            Labels the declaration (``opensees/fix/<name>`` in
            ``/opensees/decls``), unique among every ``fix``
            declaration, flat or staged; ``#k`` when omitted.  A label
            only: the deck and ``model_hash`` do not change with it.

        Raises
        ------
        ValueError
            If not exactly one of ``pg`` / ``nodes`` / ``label`` is
            supplied, ``label`` is empty or blank, or ``name`` is
            already a ``fix`` declaration's.
        BridgeError
            If ``label`` names no label of the snapshot (the message
            lists the labels that exist) or selects no node.
        """
        pg, nodes_tuple, label_names = _select_nodes(
            self._bridge.fem, pg=pg, nodes=nodes, label=label,
            where=f"Stage {self._name!r}.fix")
        rec = FixRecord(pg=pg, nodes=nodes_tuple, dofs=tuple(dofs),
                        label=label_names)
        # K1-6: one key space with the flat ``ops.fix`` declarations.
        self._bridge._declare_record(rec, "fix", name)
        self._fix_records.append(rec)

    def support(
        self,
        *,
        pg: str | None = None,
        nodes: "Iterable[int | Node] | None" = None,
        dofs: tuple[int, ...],
    ) -> None:
        """Install a stage-bound support that HOLDS the current deformed
        position with zero initial force (ADR 0052).

        The staged-construction counterpart to :meth:`fix`.  Where
        ``fix`` is *absolute* — it drives the DOF back to its ``t = 0``
        reference position — ``support`` *holds* the DOF at wherever it
        has drifted to by the start of this stage.  Each flagged DOF
        emits, inside this stage's dedicated constant pattern::

            sp <node> <dof> [nodeDisp <node> <dof>] -const

        The ``nodeDisp`` value is captured **at runtime** in the emitted
        deck (after the prior stage's ``analyze`` + ``loadConst``), so
        the support is satisfied the instant it is added — zero residual,
        zero jump, zero spurious force.  ``-const`` pins the value so it
        is never scaled by a load factor.

        Signature mirrors :meth:`fix`: exactly one of ``pg`` / ``nodes``;
        ``dofs`` is an ``ndf``-length tuple of 0/1 flags (``1`` = hold
        that DOF).  No value is supplied — it is read from the model at
        runtime.  Emits inside this stage's block (BC region, before
        ``domain_change``) into a per-stage ``Plain`` pattern bound to a
        single shared ``Constant`` series; the pattern is claimed so the
        global / stage-load-pattern passes never double-emit it.

        Reference frame — deformed (HOLD).  Use this for the usual
        staged-construction intent: you add a support to hold what is
        already there.  To instead return a DOF to its ``t = 0`` position
        (a physical restoring force, e.g. releasing then re-anchoring),
        use :meth:`fix`.

        Transient caveat — momentum-kill, not value-jump.  A HOLD support
        introduces no displacement jump (the value equals the current
        position), so it is exactly zero-force in a static stage.  In a
        *transient* stage, however, rigidly pinning a moving DOF cuts its
        velocity in one step — the reaction absorbs the momentum and the
        kinetic energy is removed discontinuously (an impulse).  This is
        **not** fixable by ramping (there is no value trajectory to
        ramp).  If that impulse matters, install the support at a
        quiescent instant, or model the support as a stiff
        spring + dashpot (a ``zeroLength`` element) so the momentum
        bleeds off instead of being cut.

        Validators V1 / V2 gate this at build time: a ``support`` and a
        ``fix`` (or two ``support`` directives) on the same ``(node,
        DOF)`` across tiers is refused — a DOF can carry only one
        single-point constraint.  ``s.remove_sp`` on that target clears
        the registration so a same-stage re-support is allowed.

        Parameters
        ----------
        pg
            Physical group whose nodes are held.  XOR with ``nodes``.
        nodes
            Explicit list of node tags (or :class:`Node` instances).
            XOR with ``pg``.
        dofs
            ``ndf``-length tuple of 0/1 flags — ``1`` means hold that
            DOF at its current displacement, ``0`` leaves it free.

        Raises
        ------
        ValueError
            If both or neither of ``pg`` / ``nodes`` is supplied.
        """
        if (pg is None) == (nodes is None):
            raise ValueError(
                f"Stage {self._name!r}.support: supply exactly one of "
                f"pg= or nodes= (got pg={pg!r}, nodes={nodes!r})."
            )
        nodes_tuple = _iter_tags(nodes) if nodes is not None else None
        # apegmsh-lint: comment-provenance-ok moved verbatim from apesees.py in the S1 pure move
        # Lazily create the shared Constant series (once across all
        # stages) and this stage's dedicated Plain HOLD pattern (once
        # per stage), then claim the pattern so neither the global
        # post-element pattern pass nor the 7b stage-load-pattern pass
        # double-emits it — the dedicated HOLD block drives its emit.
        # Registered directly (not through the namespaces) so each gets
        # its own provenance record, keyed ``support:<owner>[/hold]``,
        # pointing at this ``s.support`` call, ``origin = "synthesised"``
        # (ADR 0112 D3, #1378).  The bridge allows a repeated stage
        # name, so ``<owner>`` is the stage name when its pattern key is
        # free, else ``<stage>@<n>`` with the smallest n >= 2 whose key
        # is free (stages ``s``, ``s``, ``s@2`` give ``s``, ``s@2``,
        # ``s@2@2``).  The shared HOLD series is keyed by the owner of
        # the stage whose first ``support`` creates it; user names share
        # this key space, so a taken HOLD key raises here, before any
        # tag is allocated, and leaves nothing behind.
        if self._support_pattern is None:
            from ..pattern.pattern import Plain as _Plain
            from ..time_series.time_series import Constant as _Constant

            prov = self._bridge._provenance
            owner = self._name
            n = 2
            while prov.has("opensees", "pattern", f"support:{owner}"):
                owner = f"{self._name}@{n}"
                n += 1
            need_hold = self._bridge._hold_series is None
            if need_hold and prov.has(
                    "opensees", "timeSeries", f"support:{owner}/hold"):
                raise ValueError(
                    f"Stage {self._name!r}.support: the provenance key "
                    f"'opensees/timeSeries/support:{owner}/hold' of the "
                    "HOLD series it would create already has a record "
                    "(a declaration was named like it); rename that "
                    "declaration or the stage.  Nothing was registered."
                )
            hold = self._bridge._hold_series
            if hold is None:
                hold = self._bridge._register(
                    _Constant(factor=1.0),
                    synthesised=f"support:{owner}/hold",
                )
                self._bridge._hold_series = hold
            self._support_pattern = self._bridge._register(
                _Plain(series=hold),
                synthesised=f"support:{owner}",
            )
            self._bridge._stage_claimed_pattern_ids.add(
                id(self._support_pattern),
            )
        support = SupportRecord(pg=pg, nodes=nodes_tuple, dofs=tuple(dofs))
        # K1-6: the call's own declaration, beside the synthesised HOLD
        # series and pattern; its ``sp_hold`` rows carry it.
        self._bridge._declare_record(support, "support", None)
        self._support_records.append(support)

    def mass(
        self,
        *,
        pg: str | None = None,
        nodes: "Iterable[int | Node] | None" = None,
        label: str | Sequence[str] | None = None,
        values: tuple[float, ...],
        overwrite: bool = False,
        name: str | None = None,
    ) -> None:
        """Attach lumped nodal mass bound to this stage.

        Signature mirrors :meth:`apeSees.mass` verbatim, plus the
        Phase SSI-2.E ``overwrite=`` flag.  Stage-bound mass lines
        emit alongside stage-bound fix lines (see :meth:`fix` for the
        emit-position rationale).

        OpenSees ``setMass`` silently OVERWRITES a node's mass on
        repeated calls.  Validator V2 (Phase SSI-2.D PR-A) refuses
        the same node receiving mass in more than one tier (global +
        any stage, or stage A + stage B) at build time so the
        physics change is not silent.

        Pass ``overwrite=True`` to opt out of V2 for this record only —
        the user is acknowledging the OpenSees ``setMass`` overwrite is
        intentional (e.g. swapping a temporary construction mass for a
        permanent one between stages).  The emitted ``mass`` line is
        byte-identical with or without the flag; the difference is
        purely a build-time validator-bypass marker.

        Parameters
        ----------
        pg
            Physical group whose nodes receive the mass.  XOR with
            ``nodes`` / ``label``.
        nodes
            Explicit list of node tags (or :class:`Node` instances).
            XOR with ``pg`` / ``label``.
        label
            Geometry label (or list of labels) whose nodes receive the
            mass, resolved at the call (ADR 0115 D4).  XOR with ``pg``
            / ``nodes``.
        values
            ``ndf``-length tuple of mass values per DOF.
        overwrite
            When ``True``, V2 skips the cross-tier duplicate-mass
            check for this record.  Defaults to ``False`` (V2 active).
        name
            Labels the declaration (``opensees/mass/<name>`` in
            ``/opensees/decls``), unique among every ``mass``
            declaration, flat or staged; ``#k`` when omitted.  A label
            only: the deck and ``model_hash`` do not change with it.

        Raises
        ------
        ValueError
            If not exactly one of ``pg`` / ``nodes`` / ``label`` is
            supplied, or ``label`` is empty or blank.
        BridgeError
            If ``label`` names no label of the snapshot (the message
            lists the labels that exist) or selects no node.
        """
        pg, nodes_tuple, label_names = _select_nodes(
            self._bridge.fem, pg=pg, nodes=nodes, label=label,
            where=f"Stage {self._name!r}.mass")
        rec = MassRecord(
            pg=pg, nodes=nodes_tuple, values=tuple(values),
            overwrite=bool(overwrite), label=label_names,
        )
        # K1-6: one key space with the flat ``ops.mass`` declarations.
        self._bridge._declare_record(rec, "mass", name)
        self._mass_records.append(rec)

    # -- Phase SSI-2.E: between-stage Domain mutators --------------------

    def remove_sp(
        self,
        *,
        pg: str | None = None,
        nodes: "Iterable[int | Node] | None" = None,
        dofs: tuple[int, ...],
    ) -> None:
        """Release prior-tier SP constraints on a set of nodes / DOFs
        within this stage (Phase SSI-2.E).

        Stage-bound only.  The emitted ``remove sp $node $dof`` lines
        fire BEFORE the stage's new ``fix`` / ``mass`` / ``region``
        lines, so a stage can release a prior-stage support and then
        re-fix the same DOF with a new value in the same stage block.

        Validator V5 (Phase SSI-2.E) refuses targets whose SP was not
        declared in an earlier scope (global pool OR strictly-earlier
        stage's ``s.fix`` pool), or that was already removed by an
        earlier stage's ``s.remove_sp``.

        Parameters
        ----------
        pg
            Physical group whose nodes have SPs released.  XOR with
            ``nodes``.
        nodes
            Explicit list of node tags (or :class:`Node` instances).
            XOR with ``pg``.
        dofs
            DOF indices to release per node.  Per OpenSees convention,
            DOFs are 1-based — ``(1, 2, 3)`` releases the first three
            DOFs at every resolved node.  Unlike :meth:`fix`, these
            are DOF *indices* (one ``remove sp`` line per index), not
            a fixity flag vector.

        Raises
        ------
        ValueError
            If both or neither of ``pg`` / ``nodes`` is supplied, or
            if ``dofs`` is empty.
        """
        if (pg is None) == (nodes is None):
            raise ValueError(
                f"Stage {self._name!r}.remove_sp: supply exactly one "
                f"of pg= or nodes= (got pg={pg!r}, nodes={nodes!r})."
            )
        dofs_tuple = tuple(int(d) for d in dofs)
        if not dofs_tuple:
            raise ValueError(
                f"Stage {self._name!r}.remove_sp: dofs= must contain "
                "at least one DOF index."
            )
        nodes_tuple = _iter_tags(nodes) if nodes is not None else None
        removal = SPRemovalRecord(pg=pg, nodes=nodes_tuple, dofs=dofs_tuple)
        self._bridge._declare_record(removal, "remove_sp", None)  # K1-6
        self._remove_sp_records.append(removal)

    def remove_bc(
        self,
        *,
        pg: str | None = None,
        nodes: "Iterable[int | Node] | None" = None,
        dofs: tuple[int, ...],
    ) -> None:
        """Release prior-tier boundary conditions on a set of nodes /
        DOFs within this stage — the ``g.constraints.bc``-reading alias
        of :meth:`remove_sp` (ADR 0051 §8).

        Verbatim delegate: ``s.remove_bc(...)`` and ``s.remove_sp(...)``
        produce identical :class:`SPRemovalRecord` rows and identical
        ``remove sp $node $dof`` deck lines.  ``remove_bc`` reads more
        naturally when the released constraint was declared with
        ``g.constraints.bc(...)``; ``remove_sp`` is retained because
        shipped decks and tests reference it.

        DOF convention (unchanged, easy to trip over): ``dofs=`` here are
        **1-based DOF indices** — one ``remove sp`` line per index — NOT
        the 0/1 fixity flag vector that ``ops.fix`` / ``s.fix`` take.
        ``(1, 2, 3)`` releases the first three DOFs at every resolved
        node.

        See :meth:`remove_sp` for the full parameter / validator (V5)
        contract.
        """
        self.remove_sp(pg=pg, nodes=nodes, dofs=dofs)

    def remove_element(
        self,
        *,
        pg: str | None = None,
        elements: "Iterable[int] | None" = None,
    ) -> None:
        """Drop elements from the Domain mid-analysis within this stage
        (Phase SSI-2.E).

        Stage-bound only.  The emitted ``remove element $tag`` lines
        fire BEFORE the stage's new ``fix`` / ``mass`` / ``region`` /
        MP-constraint lines so the same stage can release legacy
        elements and immediately bind new BCs to the survivors.

        Element nodes are NOT removed — they remain in the Domain and
        may continue to carry SP / mass / load declarations from other
        tiers.  Use :meth:`remove_sp` separately if you also want to
        drop SP constraints from those orphaned nodes.

        Validator V6 (Phase SSI-2.E) refuses targets that were not
        previously emitted in an earlier scope (globally emitted OR
        activated by a strictly-earlier stage's ``s.activate(pgs=)``),
        or that were already removed by an earlier stage.

        Parameters
        ----------
        pg
            Physical group whose elements are removed.  XOR with
            ``elements``.
        elements
            Explicit list of FEM element ids (NOT OpenSees ops tags)
            — matches the :class:`recorder.Element` convention.  The
            bridge translates FEM eids to OpenSees ops tags via the
            pre-allocated ``fem_eid_to_ops_tag`` map at emit time, so
            the emitted ``remove element $tag`` line carries the
            OpenSees tag the rest of the deck uses.  XOR with ``pg``.

        Raises
        ------
        ValueError
            If both or neither of ``pg`` / ``elements`` is supplied.
        """
        if (pg is None) == (elements is None):
            raise ValueError(
                f"Stage {self._name!r}.remove_element: supply exactly "
                f"one of pg= or elements= (got pg={pg!r}, "
                f"elements={elements!r})."
            )
        elements_tuple = (
            None if elements is None else tuple(int(e) for e in elements)
        )
        removal = ElementRemovalRecord(pg=pg, elements=elements_tuple)
        self._bridge._declare_record(removal, "remove_element", None)  # K1-6
        self._remove_element_records.append(removal)

    def update_material_stage(
        self,
        *,
        materials: "Iterable[Primitive]",
        stage: int,
    ) -> None:
        """Flip soil materials between the elastic and the elastoplastic
        stage within this stage (Phase SSI-2.E).

        Stage-bound only — there is no top-level
        ``apeSees.update_material_stage``.  Emits one
        ``updateMaterialStage -material $tag -stage $stage`` line per
        material, in the order given.  A flip to ``stage=1`` of a
        :data:`FIRST_ELEMENT_ONLY_PLASTIC_INIT` material on two or more
        elements is preceded by ``setParameter -val 1 -ele <its elements>
        materialState $tag``, which runs each element's own stage update
        (``_internal/build/stage_emit.MaterialStageCarriers``).
        ``stage=0`` is elastic, ``stage=1`` is elastoplastic;
        the OpenSees handler additionally calls ``Elastic2Plastic()``
        when the value is 1.  ``stage=2`` (elastic, pressure-dependent
        moduli) is accepted for ``PressureDependMultiYield`` /
        ``PressureDependMultiYield02`` only.  This is the staged-gravity
        idiom for soil plasticity: build elastic, solve gravity, flip to
        plastic, push.

        The lines emit AFTER this stage's element activation and after
        its ``remove_sp`` / ``remove_element`` block, and BEFORE the
        stage's new ``fix`` / ``mass`` / ``region`` lines — so a stage
        can release, re-fix, and then flip.

        Only materials whose elements are LIVE in the Domain at that
        point are reached: ``MaterialStageParameter::setDomain()`` walks
        the Domain's elements looking for the material tag, so a
        material whose elements are activated by a *later* stage is not
        flipped by an earlier stage's call (OpenSees prints
        ``no effect with material tag N`` and the flip silently does
        nothing).  Validator V7 turns that no-op into a build-time
        error.

        ``materials=`` is a SEQUENCE on purpose.
        ``ManzariDafalias::mElastFlag`` is declared ``static`` — a
        single stage flag shared by every instance in the process, and
        any constructor resets it.  Two SANISAND materials therefore
        cannot sit at different stages, and construction order preserves
        nothing.  The enforced practice is to call this for EVERY
        material tag, EVERY time, immediately before the
        stage-dependent analysis step.  The emit enforces it for the
        :data:`CLASS_WIDE_STAGE_FLAG` groups: a stage that flips some of
        a group's live materials but not all, or flips them to different
        values, raises :class:`BridgeError`.

        Parameters
        ----------
        materials
            The material handles to flip (SANISAND, the multi-yield
            soils, PM4Sand / PM4Silt).  Every handle must
            already be registered on the bridge and must be one of
            :data:`STAGED_MATERIAL_CLASSES`.
        stage
            ``0`` (elastic) or ``1`` (elastoplastic); ``2`` for the
            pressure-dependent multi-yield sands.

        Raises
        ------
        ValueError
            If ``stage`` is not 0 or 1, or if ``materials`` is empty.
        BridgeError
            If a handle was never registered on this bridge, or its
            class does not honour ``updateMaterialStage``.
        """
        if isinstance(stage, bool) or not isinstance(stage, numbers.Integral):
            raise ValueError(
                f"Stage {self._name!r}.update_material_stage: stage= must "
                f"be an int (0, 1 or 2), got {stage!r}."
            )
        stage_i = int(stage)
        if stage_i not in (0, 1, 2):
            raise ValueError(
                f"Stage {self._name!r}.update_material_stage: stage= "
                f"must be 0 (elastic) or 1 (elastoplastic), or 2 for the "
                f"pressure-dependent multi-yield sands, got {stage!r}."
            )
        mats = tuple(materials)
        if not mats:
            raise ValueError(
                f"Stage {self._name!r}.update_material_stage: "
                "materials= must contain at least one material."
            )
        mat_tags: list[int] = []
        for mat in mats:
            tag = self._bridge.tag_for(mat)
            if tag is None:
                raise BridgeError(
                    f"Stage {self._name!r}.update_material_stage: "
                    f"{type(mat).__name__} was never registered on this "
                    "bridge, so it has no nDMaterial tag.  Create it via "
                    "ops.nDMaterial.<Class>(...) (or register it with "
                    "ops.register(mat)) before flipping its stage."
                )
            if type(mat).__name__ not in STAGED_MATERIAL_CLASSES:
                raise BridgeError(
                    f"Stage {self._name!r}.update_material_stage: "
                    f"{type(mat).__name__} does not honour "
                    "updateMaterialStage — the command would be a "
                    "silent no-op.  Supported: "
                    f"{', '.join(sorted(STAGED_MATERIAL_CLASSES))}."
                )
            if stage_i == 2 and type(mat).__name__ not in (
                    STAGE_2_MATERIAL_CLASSES):
                raise ValueError(
                    f"Stage {self._name!r}.update_material_stage: stage=2 "
                    f"(elastic, pressure-dependent moduli) is a "
                    f"{' / '.join(sorted(STAGE_2_MATERIAL_CLASSES))} stage; "
                    f"{type(mat).__name__} takes 0 or 1."
                )
            mat_tags.append(int(tag))
        flip = MaterialStageRecord(mat_tags=tuple(mat_tags), stage=stage_i)
        self._bridge._declare_record(  # K1-6
            flip, "update_material_stage", None)
        self._update_material_stage_records.append(flip)

    def update_parameter(
        self,
        name: str,
        value: float,
        *,
        pg: str | None = None,
        elements: "Iterable[int] | None" = None,
        material: "Primitive | None" = None,
    ) -> "UpdateParameterRecord":
        """Change one element (or element-hosted material) parameter at
        this stage's boundary — a typed pass-through over the same
        ``parameter`` / ``addToParameter`` / ``updateParameter``
        primitive that :meth:`initial_stress` and
        :meth:`activate_absorbing` drive internally.

        Emits, once per record::

            parameter $pid
            addToParameter $pid element $eleTag <name> [<mat_tag>]   # per element
            updateParameter $pid <value>
            remove parameter $pid

        Two target shapes, both addressed through elements:

        * **an element parameter** — ``s.update_parameter("xPerm", 1e-5,
          pg="soil")``.  ``LadrunoUP::setParameter`` matches ``xPerm`` /
          ``yPerm`` / ``zPerm`` directly (fork
          ``LadrunoUP.cpp:1932-1943``; ``zPerm`` is 3-D only).
        * **a material parameter** — ``s.update_parameter("poissonRatio",
          0.35, pg="soil", material=sand)``.  The element forwards the
          unmatched argv to its integration-point materials (the
          catch-all at ``LadrunoUP.cpp:1962-1971``) and the material
          matches on ``argv[0] == name`` **and** ``argv[1] == its own
          tag`` (``ManzariDafalias::setParameter``
          ``ManzariDafalias.cpp:820-857``) — which is why ``material=``
          appends the tag rather than replacing the element target.

        There is deliberately no ``material=``-only form: OpenSees
        ``parameter`` / ``addToParameter`` accept ``node`` / ``element``
        / ``region`` / ``loadPattern`` and nothing else
        (``OpenSeesParameterCommands.cpp`` ``OPS_Parameter`` /
        ``OPS_addToParameter``), so a material is unreachable without an
        element that hosts it.

        No registry of known parameter names — the element's /
        material's own ``setParameter`` is the authority, and a name it
        does not recognise already errors there.

        Parameters
        ----------
        name
            The parameter name the target's ``setParameter`` matches
            (e.g. ``"xPerm"``, ``"poissonRatio"``).
        value
            The value passed to ``updateParameter``.
        pg
            Physical group whose elements carry the parameter.  XOR with
            ``elements``.
        elements
            Explicit list of FEM element ids (NOT OpenSees ops tags —
            same convention as :meth:`remove_element` /
            :meth:`activate_absorbing`).  XOR with ``pg``.
        material
            Optional material handle.  When given, its bridge-allocated
            tag is appended to the ``addToParameter`` argv so the
            element forwards the update to THAT material.  Omit for a
            parameter the element owns itself.

        Raises
        ------
        ValueError
            Empty ``name``; both or neither of ``pg`` / ``elements``.
        BridgeError
            ``material=`` was never registered on this bridge (so it has
            no tag, and the argv would name nothing); or ``name`` is
            ``"materialState"`` with no ``material=``, or with one whose
            class is not in :data:`MATERIAL_STATE_CLASSES` (no object
            would take it).  The stage close refuses the other stage-flag
            routes that reach nothing or the wrong material.
        """
        if not name:
            raise ValueError(
                f"Stage {self._name!r}.update_parameter: name= must be "
                "non-empty."
            )
        if (pg is None) == (elements is None):
            raise ValueError(
                f"Stage {self._name!r}.update_parameter: supply exactly "
                f"one of pg= or elements= (got pg={pg!r}, "
                f"elements={elements!r})."
            )
        mat_tag: int | None = None
        if material is not None:
            tag = self._bridge.tag_for(material)
            if tag is None:
                raise BridgeError(
                    f"Stage {self._name!r}.update_parameter: "
                    f"{type(material).__name__} was never registered on "
                    "this bridge, so it has no tag — the addToParameter "
                    "argv would name nothing and the update would be a "
                    "silent no-op.  Create it via ops.nDMaterial.<Class>"
                    "(...) / ops.uniaxialMaterial.<Class>(...) (or "
                    "register it with ops.register(mat)) first."
                )
            mat_tag = int(tag)
        if name == "materialState":
            if material is None:
                raise BridgeError(
                    f"Stage {self._name!r}.update_parameter('materialState', "
                    f"{value}) names no material.  Every class that answers "
                    "materialState for its own tag needs that tag in the "
                    "argv, so this update would reach only the classes that "
                    "take it for any tag ("
                    f"{', '.join(sorted(TAG_BLIND_STAGE_PARAMETER))}) and "
                    "switch them unasked (issue #2458).  Pass material=."
                )
            cls = type(material).__name__
            if cls not in MATERIAL_STATE_CLASSES:
                instead = (
                    "Flip it with s.update_material_stage."
                    if cls in STAGED_MATERIAL_CLASSES else
                    "It has no stage flag the bridge can set.")
                raise BridgeError(
                    f"Stage {self._name!r}.update_parameter('materialState', "
                    f"{value}, material={cls}): {cls}'s setParameter has no "
                    "materialState, so no object takes the update: OpenSees "
                    "warns and the run keeps its stage (issue #2460).  "
                    + instead
                )
        record = UpdateParameterRecord(
            name=str(name),
            value=float(value),
            pg=pg,
            elements=(
                None if elements is None
                else tuple(int(e) for e in elements)
            ),
            mat_tag=mat_tag,
        )
        self._bridge._declare_record(  # K1-6
            record, "update_parameter", None)
        self._update_parameter_records.append(record)
        return record

    def set_time(self, t: float) -> None:
        """Override the stage's starting pseudo-time (Phase SSI-2.E).

        Emits ``setTime $t`` at the top of the stage block — right
        after ``stage_open``.  Overrides the ``loadConst -time 0.0``
        reset that the previous stage's ``stage_close`` emitted.
        Useful when the dynamic clock of a transient stage should
        begin at a non-zero value (e.g. continuing simulated time
        across multi-record ground motion runs).

        Idempotent at the call-site level: a second ``s.set_time(...)``
        in the same stage overwrites the prior value (only the last
        wins).  Per OpenSees semantics, ``setTime`` does NOT reset
        committed state — node displacements / element forces survive.
        """
        self._set_time = float(t)

    def set_creep(self, on: bool) -> None:
        """Toggle creep for time-dependent concrete materials in this
        stage (Phase SSI-2.E).

        Emits ``setCreep 1`` or ``setCreep 0`` near the top of the
        stage block (after ``set_time``).  Sticky on the OpenSees side
        — apeSees does NOT auto-reset between stages.  Re-assert the
        desired state per stage if you need it scoped.

        A second call in the same stage overwrites the prior value.
        """
        self._set_creep_on = bool(on)

    def reset(self) -> None:
        """Request a ``reset`` command right before this stage's
        ``analyze`` (Phase SSI-2.E).

        Emits the bare OpenSees ``reset`` command, which reverts the
        Domain to its start state (``Domain::revertToStart``): the
        committed and current times both return to 0, so the stage
        clock restarts at 0, not at the last ``setTime``.  Rarely
        needed — kept for parity with the OpenSees surface so unusual
        workflows don't have to drop to raw Tcl.

        Idempotent: multiple ``s.reset()`` calls on the same stage
        produce a single emitted ``reset`` line.
        """
        self._pre_analyze_reset = True

    def region(
        self,
        *,
        name: str,
        pg: str | None = None,
        nodes: "Iterable[int | Node] | None" = None,
    ) -> None:
        """Assign nodes to a named OpenSees Region bound to this stage.

        Signature mirrors :meth:`apeSees.region` verbatim.  Stage-bound
        regions emit **inside this stage's block**, alongside the
        stage's ``fix`` / ``mass`` lines and before the
        ``domain_change`` barrier.

        Under MP the per-stage region tag is allocated once on the
        first rank that contributes members, then re-used across every
        rank that owns members of the same region — same INV-4
        convention the global path uses, but cached per-stage so two
        stages with regions named ``"foo"`` get distinct tags.
        Validator V3 (Phase SSI-2.D PR-A) refuses same-``name``
        regions across scopes, so within any single stage every
        ``name=`` resolves to one unique tag.

        Parameters
        ----------
        name
            Region label.  Must be unique across global + every
            stage's region pool (V3).  Mangle the label to make scope
            explicit when the same conceptual region appears in
            multiple stages (e.g. ``lining_rayleigh_stage2``).
        pg
            Physical group whose nodes join the region.  XOR with
            ``nodes``.
        nodes
            Explicit node list.  XOR with ``pg``.

        Raises
        ------
        ValueError
            If ``name`` is empty or if both / neither of
            ``pg`` / ``nodes`` is supplied.
        """
        if not name:
            raise ValueError(
                f"Stage {self._name!r}.region: name= must be non-empty."
            )
        if (pg is None) == (nodes is None):
            raise ValueError(
                f"Stage {self._name!r}.region: supply exactly one of "
                f"pg= or nodes= (got pg={pg!r}, nodes={nodes!r})."
            )
        nodes_tuple = _iter_tags(nodes) if nodes is not None else None
        region = RegionAssignmentRecord(name=str(name), pg=pg, nodes=nodes_tuple)
        # K1-6: every assignment to one region name is one declaration.
        self._bridge._declare_region(region)
        self._region_records.append(region)

    def recorder(self, spec: Recorder) -> None:
        """Bind a previously-registered recorder to this stage (PULL).

        ``spec`` is a :class:`Recorder` constructed and registered via
        ``ops.recorder.Node(...)`` / ``ops.recorder.Element(...)`` /
        ``ops.recorder.MPCO(...)``.  The recorder keeps its allocated
        tag and stays in the bridge's ``_primitives`` list, but its
        ``id(...)`` lands in
        :attr:`apeSees._stage_claimed_recorder_ids` so the global
        post-element emit loop SKIPS it — the stage's emit pass
        invokes :func:`emit_recorder_spec` inside the stage block
        instead, AFTER the stage's region declarations and analysis
        chain so the recorder sees fully-populated regions when
        OpenSees parses the ``recorder`` line.

        Mirrors the :meth:`add` PULL semantics for
        :class:`InitialStressRecord`; stage-bound recorders ARE
        bridge primitives with registration side effects (tag
        allocation), so PULL is the natural shape.

        Parameters
        ----------
        spec
            A :class:`Recorder` instance already registered with the
            bridge.

        Raises
        ------
        TypeError
            If ``spec`` is not a :class:`Recorder` instance.
        ValueError
            If ``spec`` is not in the bridge's ``_primitives`` (never
            registered through ``ops.recorder.X(...)``) or has
            already been claimed by another stage (double-add).
        """
        if not isinstance(spec, Recorder):
            raise TypeError(
                f"Stage {self._name!r}.recorder: expected a Recorder "
                f"instance (constructed via ops.recorder.Node / "
                f"Element / MPCO); got {type(spec).__name__!r}."
            )
        if id(spec) in self._bridge._stage_claimed_recorder_ids:
            raise ValueError(
                f"Stage {self._name!r}.recorder: recorder spec already "
                "claimed by another stage — each recorder may bind to "
                "at most one stage."
            )
        if spec not in self._bridge._primitives:
            raise ValueError(
                f"Stage {self._name!r}.recorder: recorder spec not in "
                "the bridge's _primitives — was it registered through "
                "this bridge's ``ops.recorder.X(...)`` namespace?"
            )
        self._bridge._stage_claimed_recorder_ids.add(id(spec))
        self._recorder_specs.append(spec)

    def pattern(
        self,
        *,
        series: "TimeSeries | str",
        name: str | None = None,
    ) -> "Plain":
        """Create a stage-scoped ``Plain`` load pattern (ADR 0051 §6).

        Returns a stage-owned :class:`~apeGmsh.opensees.pattern.pattern.Plain`
        that is **both** a typed primitive (registered with the bridge,
        so it gets a tag) **and** a context manager — open it with a
        ``with`` block and call ``p.load(...)`` / ``p.sp(...)`` /
        ``p.from_model(case)`` to populate it, exactly like the global
        ``ops.pattern.Plain(...)``::

            with ops.stage(name="push") as s:
                ts = ops.timeSeries.Linear()
                with s.pattern(series=ts) as p:
                    p.from_model("live")
                    p.load(node=99, forces=(50.0, 0.0, 0.0))
                s.analysis(...)
                s.run(n_increments=10, dt=0.1)

        The pattern emits **inside this stage's block** — after the
        stage's analysis chain and before its ``analyze`` loop — so its
        loads / prescribed displacements drive only this stage and are
        frozen as the permanent baseline by the stage's
        ``stage_close`` ``loadConst``.  It is claimed via
        :attr:`apeSees._stage_claimed_pattern_ids` so the global
        post-element pattern pass SKIPS it (no double emission), exactly
        mirroring how :meth:`recorder` claims recorders.

        The existing **global** ``ops.pattern.Plain(...)`` remains the
        non-staged path; per ADR 0051 §5 a model may not mix a global
        pattern with stages — that no-mixing guard lands in BL-4.

        Parameters
        ----------
        series
            The :class:`~apeGmsh.opensees._internal.types.TimeSeries`
            scaling this pattern's loads — a handle, or the ``name=``
            a series was registered under (dual-mode, same as
            ``ops.pattern.Plain``).
        name
            Optional bridge-side alias for the pattern (see
            ``ops.pattern.Plain``).
        """
        # Delegate construction to the pattern namespace so the series
        # name resolution + registration + tag allocation are identical
        # to the global ``ops.pattern.Plain(...)`` path; then claim it
        # for this stage.
        plain = self._bridge.pattern.Plain(series=series, name=name)
        self._bridge._stage_claimed_pattern_ids.add(id(plain))
        self._pattern_specs.append(plain)
        return plain

    def imposed_path(
        self,
        *,
        node: "int | Node",
        ratios: "Sequence[float]",
        series: "TimeSeries | str",
    ) -> "Plain":
        """Drive ONE node along a prescribed multi-DOF path in this stage.

        The rotational / staged counterpart to
        :meth:`apeSees.imposed_displacement`, which is translations-only
        (``ux`` / ``uy`` / ``uz`` → DOFs 1-3) and global (ADR 0051 §5
        forbids mixing a global pattern with stages).  Here ``ratios``
        is positional over the node's DOFs, so DOFs 4..6 — the rotations
        — are reachable, and the pattern is stage-scoped.

        Creates a stage-scoped ``Plain`` via :meth:`pattern` and records
        one ``sp`` per NON-ZERO ratio::

            sp <node> <i> <ratios[i-1]>

        Zero ratios are skipped, not emitted as ``sp … 0.0``: a
        prescribed zero is a *constraint* (it pins the DOF), not the
        absence of one, so emitting it would silently clamp DOFs the
        caller meant to leave free.  Use ``s.fix`` / ``s.support`` when
        pinning is what you want.

        The ratios are shape only — the magnitude and history come from
        ``series``.  The applied value on DOF ``i`` at time ``t`` is
        ``ratios[i-1] * series(t)``, so a unit-direction vector plus a
        ``Path`` series gives a fault-slip or support-settlement path,
        and the same pattern may carry ordinary ``p.load`` lines on
        other DOFs of the same node (they coexist — a prescribed SP and
        a nodal load are different rows in the same pattern).

        Returns the pattern, so more can be added to it::

            with ops.stage("slip") as s:
                p = s.imposed_path(
                    node=99, ratios=(0.0, 0.0, 0.0, 0.0, 0.0, 0.01),
                    series=ops.timeSeries.Linear(),
                )
                p.load(node=99, forces=(0.0, 0.0, -5e3, 0.0, 0.0, 0.0))

        Parameters
        ----------
        node
            The node to drive — a tag or a :class:`Node`.
        ratios
            Per-DOF ratios, positional from DOF 1.  At most the model's
            ``ndf`` entries; at least one must be non-zero.
        series
            The :class:`~apeGmsh.opensees._internal.types.TimeSeries`
            scaling the path — a handle, or the ``name=`` a series was
            registered under (dual-mode, same as :meth:`pattern`).

        Raises
        ------
        ValueError
            ``ratios`` is empty, is all zeros (an inert directive), or
            is longer than the model's ``ndf``.

        Notes
        -----
        The length check is against the ``ops.model(ndf=)`` **envelope**,
        which is an upper bound on any node's ndf — the per-node
        *effective* ndf map (ADR 0048) is only resolved at build time,
        from every declared element's PG fan-out, so it cannot be probed
        per call without a full mesh walk.  A ratio past the envelope is
        therefore refused here; a ratio past a particular node's lower
        effective ndf reaches OpenSees, which rejects the ``sp`` line.
        """
        ratios_t = tuple(float(r) for r in ratios)
        if not ratios_t:
            raise ValueError(
                f"Stage {self._name!r}.imposed_path: ratios= must "
                "contain at least one entry."
            )
        model_ndf = self._bridge._ndf
        if model_ndf is not None and len(ratios_t) > model_ndf:
            raise ValueError(
                f"Stage {self._name!r}.imposed_path: ratios= has "
                f"{len(ratios_t)} entries but the model's ndf is "
                f"{model_ndf} — DOF {len(ratios_t)} does not exist.  "
                f"Trim ratios= or call ops.model(..., "
                f"ndf={len(ratios_t)}) first."
            )
        if not any(ratios_t):
            raise ValueError(
                f"Stage {self._name!r}.imposed_path: every ratio is "
                "zero, so the pattern would emit no sp lines at all.  "
                "Supply at least one non-zero ratio (a prescribed zero "
                "is a fixity — use s.fix / s.support for that)."
            )
        plain = self.pattern(series=series)
        node_tag = int(_iter_tags([node])[0])
        with plain:
            for dof, ratio in enumerate(ratios_t, start=1):
                if ratio:
                    plain.sp(node=node_tag, dof=dof, value=ratio)
        return plain

    def activate(self, *, pgs: "Iterable[str]") -> None:
        """Mark element PGs as activated by this stage (Phase SSI-2.B).

        Elements whose ``pg=`` matches any activated PG emit their
        ``node`` + ``element`` commands **inside this stage's block**
        (between ``stage_open`` and ``domain_change``), not in the
        global pre-stage emit.  Nodes referenced exclusively by
        stage-activated elements move into the stage's block too;
        nodes shared with global elements stay global.

        May be called multiple times per stage (PGs accumulate as a
        set; duplicates collapse).  Same PG activated in two
        different stages is a build-time error (first-write wins
        is unsafe — the user clearly meant something different).

        Parameters
        ----------
        pgs
            Iterable of element-PG names (e.g. ``["cimbra"]``,
            ``["rock", "lining"]``).  Each must be a non-empty string.
        """
        for pg in pgs:
            if not isinstance(pg, str) or not pg:
                raise ValueError(
                    f"Stage {self._name!r}.activate: pgs= must be an "
                    "iterable of non-empty strings, got "
                    f"{pg!r}."
                )
            if pg not in self._activated_pgs:
                self._activated_pgs.append(pg)

    def analysis(
        self,
        chain: "StageChain | None" = None,
        /,
        *,
        integrator: Primitive,
        test: Primitive | None = _OMITTED,
        algorithm: Primitive | None = _OMITTED,
        constraints: Primitive | None = _OMITTED,
        numberer: Primitive | None = _OMITTED,
        system: Primitive | None = _OMITTED,
        analysis: Primitive | None = _OMITTED,
    ) -> None:
        """Bind the analysis chain for this stage.

        Two forms.  The keyword form names every slot::

            s.analysis(test=..., algorithm=..., integrator=...,
                       constraints=..., numberer=..., system=...,
                       analysis=...)

        The chain form takes a shared :class:`StageChain` built by
        ``ops.analysis.chain(...)`` plus this stage's integrator::

            s.analysis(solver, integrator=ops.integrator.LoadControl(dlam=0.1))

        Both bind the same primitives, so the emitted deck is the same.
        A chain plus an explicit slot it already carries is refused:
        the chain is the one place that slot is stated.

        ``test`` may be omitted only when the algorithm does not iterate
        (``ops.algorithm.Linear``, whose ``needs_test`` is False); the
        stage then emits no ``test`` line.  A strategy ladder
        (``s.run(strategy=)``) still needs one, because its rungs
        iterate.

        Each primitive must already be registered with the bridge
        (e.g. via ``ops.test.NormDispIncr(...)``); the stage holds a
        reference only, not a copy.  Multiple stages may share the
        same primitive instance (e.g. the same ``constraints.Plain()``
        across all stages) — the bridge emits each primitive exactly
        once per stage in which it's referenced, so OpenSees gets a
        fresh ``constraints Plain`` line per stage as required.
        """
        if self._analysis_set:
            raise ValueError(
                f"Stage {self._name!r}.analysis: already called; "
                "stages support one analysis chain each."
            )
        given: dict[str, Primitive | None] = {
            "test": test,
            "algorithm": algorithm,
            "constraints": constraints,
            "numberer": numberer,
            "system": system,
            "analysis": analysis,
        }
        if chain is not None:
            if not isinstance(chain, StageChain):
                raise TypeError(
                    f"Stage {self._name!r}.analysis: the positional "
                    "argument must be a chain from ops.analysis.chain(...), "
                    f"got {type(chain).__name__}."
                )
            doubled = [k for k, v in given.items() if v is not _OMITTED]
            if doubled:
                raise ValueError(
                    f"Stage {self._name!r}.analysis: "
                    f"{', '.join(doubled)} given both by the chain and "
                    "as a keyword; state each slot once (drop the "
                    "keyword, or build a second ops.analysis.chain)."
                )
            slots = {k: getattr(chain, k) for k in CHAIN_SLOTS}
        else:
            # The keyword form binds what it is given, as before; only an
            # omitted slot is refused (the test may be omitted for a
            # non-iterating algorithm, checked below).
            missing = [
                k for k, v in given.items()
                if v is _OMITTED and k != "test"
            ]
            slots = {
                k: (None if v is _OMITTED else v) for k, v in given.items()
            }
            if missing:
                raise TypeError(
                    f"Stage {self._name!r}.analysis: missing "
                    f"{', '.join(missing)}; pass every chain slot as a "
                    "keyword, or a shared ops.analysis.chain(...)."
                )
        if slots["test"] is None and algorithm_needs_test(slots["algorithm"]):
            raise ValueError(
                f"Stage {self._name!r}.analysis: test= is required with "
                f"algorithm {type(slots['algorithm']).__name__}, which "
                "iterates; only a non-iterating algorithm (Linear) may "
                "omit it."
            )
        self._test = slots["test"]
        self._algorithm = slots["algorithm"]
        self._integrator = integrator
        self._constraints = slots["constraints"]
        self._numberer = slots["numberer"]
        self._system = slots["system"]
        self._analysis = slots["analysis"]
        self._analysis_set = True

    def run(
        self, *, n_increments: int, dt: float | None = None,
        strategy: "Ladder | None" = None,
    ) -> None:
        """Set the analyze-loop length + step size for this stage.

        ``strategy`` (ADR 0057 Phase A) attaches a solution-strategy
        ladder to this stage's analyze loop: on a failed increment the
        emitted loop escalates through the ladder's algorithm rungs
        (the chain's own algorithm is rung 0) and restores rung 0
        after a rescue; exhausting the ladder aborts fail-loud.
        Build one via ``ops.strategy.profile("non-smooth")`` or
        ``ops.strategy.Ladder(rungs=[...])``.
        """
        if self._run_set:
            raise ValueError(
                f"Stage {self._name!r}.run: already called; "
                "stages support one analyze loop each."
            )
        if n_increments < 1:
            raise ValueError(
                f"Stage {self._name!r}.run: n_increments must be >= 1, "
                f"got {n_increments}."
            )
        if (strategy is not None and getattr(strategy, "subdivides", False)
                and dt is None):
            raise ValueError(
                f"Stage {self._name!r}.run: a Subdivide rung cuts the time "
                "step, so it needs a transient stage run with dt=; a static "
                "stage uses Substep (ADR 0057 Amendment 1)."
            )
        self._n_increments = int(n_increments)
        self._dt = None if dt is None else float(dt)
        self._strategy = strategy
        self._run_set = True

    def profile(
        self, *,
        deep: bool = False,
        memory: bool = False,
        per_step: bool = False,
    ) -> None:
        """Bracket THIS stage's ``analyze`` loop with the Ladruno
        fork's stack profiler (TIMs A8), reported under this stage's
        own name.

        Reuses the same ``Emitter.profiler(*args)`` machinery the
        bridge-level ``ops.profiler.*`` verbs drive (see
        ``_ProfilerNS.start`` / ``.report``) — NOT a second
        implementation.  Emits ``profiler start [-deep] [-memory]
        [-perStep]`` immediately before this stage's analyze loop and
        ``profiler report <stage name>.h5`` immediately after it
        (before ``stage_close``); the filename is derived from the
        stage's name, so no filename kwarg is needed here.

        ``deep`` / ``memory`` / ``per_step`` mirror
        ``ops.profiler.start``'s three flags exactly. Deck emission
        works on any build; running the deck requires the Ladruno
        fork (stock ``openseespy``/``OpenSees.exe`` rejects the
        ``profiler`` command at run time).
        """
        if self._profile is not None:
            raise ValueError(
                f"Stage {self._name!r}.profile: already called; "
                "stages support one profiler bracket each."
            )
        profile = ProfileRecord(
            deep=bool(deep), memory=bool(memory), per_step=bool(per_step),
        )
        self._bridge._declare_record(profile, "profile", None)  # K1-6
        self._profile = profile
