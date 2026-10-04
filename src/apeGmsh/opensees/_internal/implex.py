"""Build-time half of the IMPL-EX time driver (ADR 0113 slice 1).

The typed declaration and the material predicate live in
:mod:`apeGmsh.opensees.analysis.implex`; this module holds what the
staged emit needs: the refusals (D4, D6), each stage's increment, the
target rows, and the prelude fan-out.  Kept out of ``build.py`` (10k
lines) on purpose; it imports from there, never the reverse.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

from ..analysis.implex import (
    DTIME_PARAMETERS,
    ImplexTime,
    implex_element_specs,
)
from .build import BridgeError, expand_pg_to_elements

if TYPE_CHECKING:
    from ..emitter.base import Emitter
    from .build import ElementPlanRows, StageRecord
    from .tag_allocator import TagAllocator

__all__ = [
    "emit_implex_prelude",
    "implex_target_rows",
    "stage_increment",
    "validate_implex_time",
]

_ADR = "ADR 0113"


def stage_increment(stage: "StageRecord") -> float:
    """The increment every step of ``stage`` takes, or raise.

    ``LoadControl`` (no ``num_iter`` adaptivity): its ``dlam``, which is
    OpenSees' pseudo-time increment ``ops_Dt``.  ``Transient``: the
    stage's fixed ``dt``.  Anything whose increment is not one known
    number (``VariableTransient``, ``DisplacementControl``, arc length,
    an adaptive ``LoadControl``) needs the per-attempt driver of the
    adaptive loop (ADR 0113 D5), which this slice does not emit.
    """
    from ..analysis.analysis import Static, Transient
    from ..analysis.integrator import LoadControl

    analysis, integrator = stage.analysis, stage.integrator
    if isinstance(analysis, Transient):
        if stage.dt is None or not stage.dt > 0.0:
            raise BridgeError(
                f"stage {stage.name!r}: a Transient stage needs run(dt=...) "
                f"> 0 for the IMPL-EX driver to know its increment "
                f"({_ADR} D3); got dt={stage.dt!r}."
            )
        return float(stage.dt)
    if isinstance(analysis, Static) and isinstance(integrator, LoadControl):
        # OpenSees reads numIter only together with min/max
        # (``OPS_LoadControlIntegrator``) and clamps every new increment to
        # [min, max] (``LoadControl::newStep``): min == max == dlam is a
        # constant increment whatever num_iter says.
        lo, hi = integrator.min_lam, integrator.max_lam
        dlam = float(integrator.dlam)
        constant = integrator.num_iter is None or (
            lo is not None and hi is not None
            and float(lo) == dlam == float(hi)
        )
        if not constant:
            raise BridgeError(
                f"stage {stage.name!r}: LoadControl(num_iter=..., "
                f"min_lam={lo!r}, max_lam={hi!r}) adapts its increment step "
                f"by step, so one dTime per stage would be wrong after the "
                f"first change; the per-attempt driver of the adaptive loop "
                f"({_ADR} D5) is not emitted yet.  Drop num_iter / min_lam / "
                f"max_lam, or set min_lam == max_lam == dlam."
            )
        return dlam
    raise BridgeError(
        f"stage {stage.name!r}: the IMPL-EX driver needs a stage whose "
        f"increment is one known number -- Static + LoadControl(dlam) or "
        f"Transient + run(dt=) -- got analysis "
        f"{type(analysis).__name__} with integrator "
        f"{type(integrator).__name__} ({_ADR} D3; variable increments "
        f"wait for the adaptive loop, D5)."
    )


def _dtime_writes(stage: "StageRecord") -> dict[str, float]:
    """``{name: value}`` of the stage's ``s.update_parameter`` records that
    write a ``dTime*`` parameter (last write wins, as in the deck)."""
    return {
        rec.name: float(rec.value)
        for rec in stage.update_parameter_records
        if rec.name in DTIME_PARAMETERS
    }


def _record_eids(fem: Any, rec: Any) -> set[int]:
    """FEM element ids an ``s.update_parameter`` record addresses (the
    ``material=`` form addresses elements too)."""
    if rec.elements is not None:
        return {int(e) for e in rec.elements}
    return {int(e) for e, _conn in expand_pg_to_elements(fem, rec.pg)}


def _target_eids(fem: Any, specs: Iterable[Any]) -> set[int]:
    out: set[int] = set()
    for spec in specs:
        pg = getattr(spec, "pg", None)
        if pg is None:
            continue
        out.update(int(e) for e, _conn in expand_pg_to_elements(fem, pg))
    return out


def validate_implex_time(
    implex: ImplexTime | None,
    stage_records: Sequence["StageRecord"],
    elements: Sequence[Any],
    fem: Any,
) -> None:
    """Every refusal of ADR 0113 slice 1, before anything is emitted.

    With a driver (``mode="stko"``): the model is staged (D3); at least
    one element reaches an IMPL-EX material (D2: a driver with no target
    is a silent no-op); no target is activated by a stage or removed by
    one (D4: persistent parameters hold the material objects collected
    at ``addToParameter`` time); no stage writes ``dTime*`` through
    ``s.update_parameter`` (two owners); every stage's increment is
    known (:func:`stage_increment`).

    With ``mode="off"``: no stage writes ``dTime*``.

    Without a declaration (the dTime trap, D6): an element whose
    materials got any ``dTime*`` write stops following OpenSees'
    increment for good, so the stage of that write and every later one
    must write ``dTime``, equal to its own increment, on every element
    switched so far.
    """
    if implex is not None and implex.mode == "off":
        for st in stage_records:
            w = _dtime_writes(st)
            if w:
                raise BridgeError(
                    f"stage {st.name!r} writes {sorted(w)} through "
                    f"s.update_parameter, but ops.implex_time(mode='off') "
                    f"declares that the materials follow OpenSees' own "
                    f"increment for the whole run ({_ADR} D1).  Drop the "
                    f"writes, or declare mode='stko'."
                )
        return

    if implex is not None and implex.drives:
        if not stage_records:
            raise BridgeError(
                "ops.implex_time(mode='stko') drives staged decks only "
                f"({_ADR} D3): declare the analysis with ops.stage(...).  "
                "A flat run that never writes dTime follows OpenSees' own "
                "increment and needs no driver."
            )
        specs = implex_element_specs(elements)
        if not specs:
            raise BridgeError(
                "ops.implex_time(mode='stko') is declared but no element "
                "reaches an ASDConcrete3D / ASDConcrete1D (implex=True or "
                f"eta > 0) or an ASDSteel1D (implex=True) ({_ADR} D2): the driver would update nothing.  "
                "Drop the declaration or check the materials."
            )
        for spec in specs:
            pg = getattr(spec, "pg", None)
            if pg is not None and not len(expand_pg_to_elements(fem, pg)):
                raise BridgeError(
                    f"IMPL-EX target {type(spec).__name__}(pg={pg!r}) has no "
                    f"elements in the model ({_ADR} D2): the driver would "
                    f"attach nothing to it.  Check the physical group."
                )
        target_pgs: set[str] = {
            str(pg) for pg in (getattr(s, "pg", None) for s in specs)
            if pg is not None
        }
        for st in stage_records:
            hit = target_pgs.intersection(st.activated_pgs)
            if hit:
                raise BridgeError(
                    f"stage {st.name!r} activates IMPL-EX target group(s) "
                    f"{sorted(hit)}: the driver's persistent parameters are "
                    f"built once, before the first stage, from the elements "
                    f"in the domain then ({_ADR} D4).  Stage-activated "
                    f"targets are a follow-up."
                )
        removable = None
        for st in stage_records:
            if not st.remove_element_records:
                continue
            if removable is None:
                removable = _target_eids(fem, specs)
            for rec in st.remove_element_records:
                if rec.pg is not None:
                    gone = {
                        int(e) for e, _c in expand_pg_to_elements(fem, rec.pg)
                    }
                else:
                    gone = {int(e) for e in (rec.elements or ())}
                lost = sorted(gone & removable)
                if lost:
                    raise BridgeError(
                        f"stage {st.name!r} removes IMPL-EX target "
                        f"element(s) {lost[:5]}{'...' if len(lost) > 5 else ''}"
                        f": a persistent parameter keeps pointers to their "
                        f"materials and OpenSees has no way to detach them, "
                        f"so the next driver call would write to freed "
                        f"memory ({_ADR} D4)."
                    )
        for st in stage_records:
            w = _dtime_writes(st)
            if w:
                raise BridgeError(
                    f"stage {st.name!r} writes {sorted(w)} through "
                    f"s.update_parameter while ops.implex_time(mode='stko') "
                    f"drives them ({_ADR} D6: one owner).  Drop the "
                    f"s.update_parameter calls."
                )
            stage_increment(st)
        return

    # No declaration: the trap check, element by element.
    switched: set[int] = set()
    first_writer: str | None = None
    for st in stage_records:
        recs = [
            r for r in st.update_parameter_records
            if r.name in DTIME_PARAMETERS
        ]
        if first_writer is None and not recs:
            continue
        if first_writer is None:
            first_writer = st.name
        dtime_cover: set[int] = set()
        for r in recs:
            eids = _record_eids(fem, r)
            switched |= eids
            if r.name == "dTime":
                dtime_cover |= eids
        stale = sorted(switched - dtime_cover)
        if stale:
            raise BridgeError(
                f"the IMPL-EX dTime trap ({_ADR} D6): from stage "
                f"{first_writer!r} on, s.update_parameter has written dTime* "
                f"on {len(switched)} element(s), which makes their materials "
                f"stop following OpenSees' increment for the rest of the "
                f"run, but stage {st.name!r} does not write dTime on "
                f"{len(stale)} of them (first {stale[:5]}), so they keep a "
                f"stale value.  Declare ops.implex_time(mode='stko') (and "
                f"drop the s.update_parameter dTime* calls), or write dTime "
                f"on every switched element in every stage from "
                f"{first_writer!r} on."
            )
        try:
            inc = stage_increment(st)
        except BridgeError as exc:
            raise BridgeError(
                f"the IMPL-EX dTime trap ({_ADR} D6): stage {st.name!r} "
                f"runs after dTime was written (from stage "
                f"{first_writer!r}) but its increment is not one known "
                f"number, so no single dTime can be right.  {exc}"
            ) from exc
        for r in recs:
            if r.name != "dTime":
                continue
            if not math.isclose(float(r.value), inc, rel_tol=1e-12, abs_tol=0.0):
                raise BridgeError(
                    f"the IMPL-EX dTime trap ({_ADR} D6): stage {st.name!r} "
                    f"writes dTime = {float(r.value)!r} but steps with an "
                    f"increment of {inc!r}; the material would extrapolate "
                    f"and regularize with the wrong time step.  Write "
                    f"dTime = {inc!r}, or declare "
                    f"ops.implex_time(mode='stko')."
                )


def implex_target_rows(
    element_plan: Sequence[tuple[Any, "ElementPlanRows"]],
    elements: Sequence[Any],
) -> list[tuple[int, int]]:
    """``(fem_eid, ops_tag)`` of every IMPL-EX target, in plan order."""
    target_ids = {id(s) for s in implex_element_specs(elements)}
    rows: list[tuple[int, int]] = []
    for spec, sub in element_plan:
        if id(spec) not in target_ids:
            continue
        for eid, _conn, ele_tag in sub:
            rows.append((int(eid), int(ele_tag)))
    return rows


def emit_implex_prelude(
    emitter: "Emitter",
    tags: "TagAllocator",
    rows: Sequence[tuple[int, int]],
    *,
    ranks: Sequence[int] | None = None,
    element_owner: Any = None,
) -> tuple[int, int, int]:
    """Declare the driver and attach its targets (ADR 0113 D3).

    Serial (``ranks is None``): one ``implex_time_targets`` over every
    row.  Partitioned: one per rank inside ``partition_open``, each with
    the rows whose FEM element the rank owns (``element_owner``: FEM eid
    -> runtime rank, the map every other per-rank emit uses).  Returns
    the three bridge-allocated parameter tags.
    """
    ptags = (
        int(tags.allocate("parameter")),
        int(tags.allocate("parameter")),
        int(tags.allocate("parameter")),
    )
    emitter.implex_time_declare(ptags)
    if ranks is None:
        emitter.implex_time_targets(ptags, tuple(t for _e, t in rows))
        return ptags
    for rank in ranks:
        own = tuple(
            t for e, t in rows if element_owner.get(int(e)) == rank
        )
        if not own:
            continue
        emitter.partition_open(rank)
        try:
            emitter.implex_time_targets(ptags, own)
        finally:
            emitter.partition_close()
    return ptags
