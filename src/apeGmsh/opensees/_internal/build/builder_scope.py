"""builder_scope: part of the ``build/`` package (S2-a split of ``_internal/build.py``)."""
from __future__ import annotations
from typing import TYPE_CHECKING

from typing import Any, Iterable, Sequence


from ..types import Element, Primitive
from .errors import BridgeError
from .records import StageRecord
from .ownership import compute_stage_ownership


if TYPE_CHECKING:
    from ...emitter.base import Emitter
    from apeGmsh.mesh.FEMData import FEMData


def needs_builder_ndf_bracket_for_token(
    type_token: str, *, ndm: int, envelope_ndf: int,
) -> bool:
    """True iff ``type_token``'s upstream parser needs a bracket here.

    The token form of :func:`needs_builder_ndf_bracket`, for the replay
    paths that carry deck records (``rec.type_token``) rather than
    ``Element`` specs.  ``element_builder_ndf`` resolves class names AND
    deck tokens (``_CLASS_TOKEN_ALIASES.get(name, name)``), so the two
    forms cannot disagree.

    Note the ``envelope_ndf`` term: a gated class under an envelope that
    ALREADY matches its required builder ndf needs no bracket, and must
    not be hoisted.  Keying on ``_BUILDER_NDF_GATED`` membership alone
    would reorder every ndf=2 quad deck for nothing.
    """
    from ..._element_capabilities import element_builder_ndf

    need = element_builder_ndf(type_token, ndm)
    return need is not None and int(need) != int(envelope_ndf)


def needs_builder_ndf_bracket(
    spec: "Element", *, ndm: int, envelope_ndf: int,
) -> bool:
    """True iff ``spec``'s upstream parser needs a builder-ndf bracket here.

    The single definition of the bracket condition, shared by
    :func:`open_builder_ndf_bracket` (which emits it), the ADR 0099 hoist
    in ``_emit_flat`` (which orders around it), and
    :func:`validate_builder_scope_ordering` (which guards it).
    """
    return needs_builder_ndf_bracket_for_token(
        type(spec).__name__, ndm=ndm, envelope_ndf=envelope_ndf,
    )


#: Emit paths that satisfy ADR 0099 INV-1 by HOISTING their gated element
#: blocks above the builder-scoped declarations, rather than by refusing.
#:
#: ``"flat"``        — S2, ``BuiltModel._emit_flat``.
#: ``"partitioned"`` — S5, ``BuiltModel._emit_partitioned``.  Default
#:   partitioned emit is ONE file with ``if {[getPID] == K}`` brace guards
#:   and the declarations global, outside every guard; a brace is not a
#:   file boundary, so the flat hoist replicates directly as one extra
#:   rank-guard block per rank.
#: ``"split"``       — S6, ``BuiltModel._emit_split``.  File-per-module
#:   (ADR 0043): the hoist moves each gated module's ``source`` line
#:   above the declarations, and the fragment carries its nodes along
#:   unchanged.  A module carrying BOTH a gated element and a
#:   builder-scoped-dependent one emits as two ordered fragments
#:   (``<m>_gated`` / ``<m>_rest``), the shape ADR 0061's per-rank
#:   writer already produces.
#:
#: Still refused: ``"partitioned per_rank"`` (file-per-rank, ADR 0061) —
#: it needs the same source-line move ``split`` got, but applied by the
#: post-emit span writer (``_write_per_rank_tcl``), which slices recorded
#: guard spans out of a finished buffer and cannot reorder them yet
#: (ADR 0099 §"How the two deferred paths should actually be fixed").
_HOISTING_PATHS = frozenset({"flat", "partitioned", "split"})


def validate_builder_scope_ordering(
    elements: "Sequence[Element]",
    primitives: "Iterable[Primitive]",
    fem: "FEMData",
    *,
    ndm: int,
    envelope_ndf: int,
    path: str,
    stage_records: "Sequence[StageRecord]" = (),
) -> None:
    """Fail loud when a builder-ndf bracket would destroy a declaration.

    ADR 0099.  A gated element block is wrapped in a ``model basic -ndf K``
    re-issue + envelope restore (:func:`open_builder_ndf_bracket`), and that
    re-issue DELETES the Tcl model builder — whose destructor purges the
    process-global ``timeSeries`` / ``geomTransf`` / ``beamIntegration`` /
    ``damping`` registries on the way out (fork
    ``SRC/modelbuilder/tcl/TclModelBuilder.cpp:681``; the audit lives in
    :data:`.._element_capabilities._BUILDER_SCOPED_KINDS`).

    The paths in :data:`_HOISTING_PATHS` hoist their gated element blocks
    above those declarations (INV-1).  Everywhere else the deck would die
    late — or, for ``damping``, run to convergence and report an
    **undamped** answer, because ``region -damp`` only warns.  Under
    partitioned emit both outcomes are RANK-LOCAL: a rank owning no gated
    element never executes the bracket, so the failure is
    non-deterministic in ``np``.

    Raises
    ------
    BridgeError
        INV-3 — a gated element directly depends on a builder-scoped
        primitive (``quad(damp=...)``), so no ordering can save it: the
        element's own bracket destroys the declaration it references.
        INV-4 — a gated element brackets on an emit path that cannot
        satisfy INV-1 (``partitioned per_rank`` / a stage-activated
        gated element on the partitioned path — the flat path handles
        that last case with the S7 stage-close replay instead).
    """
    from ..._element_capabilities import (
        builder_scoped_kind,
        element_builder_ndf,
    )

    gated = [
        e for e in elements
        if needs_builder_ndf_bracket(e, ndm=ndm, envelope_ndf=envelope_ndf)
    ]
    if not gated:
        return

    # INV-3 — direct dependencies only: the failure mode is the element
    # line resolving a tag its own bracket just deleted.
    for spec in gated:
        bad = sorted(
            {k for d in spec.dependencies()
             if (k := builder_scoped_kind(d)) is not None}
        )
        if bad:
            raise BridgeError(
                f"{type(spec).__name__} needs a builder-ndf bracket "
                f"(model basic -ndm {ndm} -ndf "
                f"{element_builder_ndf(type(spec).__name__, ndm)}), and that "
                f"bracket destroys the {', '.join(bad)} declaration it "
                f"references — the element would resolve a dead tag and "
                f"OpenSees would only warn. Per ADR 0099 INV-3, drop the "
                f"dependency (attach damping with ops.damping on a region "
                f"instead of the element's damp= argument) or model this "
                f"block with an ungated element."
            )

    scoped = sorted({k for p in primitives
                     if (k := builder_scoped_kind(p)) is not None})
    if not scoped:
        return

    fix = (
        "Per ADR 0099 INV-1 every such declaration must follow the LAST "
        "model line in the deck. The flat, split and default partitioned "
        "emit paths hoist their gated element blocks (for split, the "
        "gated fragments' source lines) above them; per_rank does not "
        "yet."
    )
    if path not in _HOISTING_PATHS:
        raise BridgeError(
            f"emit path {path!r} would declare {', '.join(scoped)} before a "
            f"builder-ndf bracket opened for "
            f"{type(gated[0]).__name__}, and the bracket destroys those "
            f"declarations. {fix} Emit this model on the flat path, or "
            f"remove the gated element / the declaration."
        )

    # ADR 0099 S7: a stage-activated gated element brackets INSIDE the
    # stage block, after the global declarations and (past the first
    # stage) after a completed ``analyze`` — no earlier position exists
    # to hoist to.  The FLAT path fixes it by REPLAYING the purged
    # declarations at bracket close (``replay_builder_scoped_
    # declarations``; Tcl only — the in-process module purges nothing,
    # so a live/py replay would collide on the still-alive tags).  The
    # partitioned path has no replay yet, so it keeps the refusal.
    if stage_records and path != "flat":
        element_owner_stage, _ = compute_stage_ownership(
            tuple(stage_records), list(elements), fem,
        )
        staged = [g for g in gated if id(g) in element_owner_stage]
        if staged:
            raise BridgeError(
                f"{type(staged[0]).__name__} is stage-activated, so its "
                f"builder-ndf bracket emits INSIDE the stage block — after "
                f"the global {', '.join(scoped)} declaration(s), and (for "
                f"any stage past the first) after a pattern and a completed "
                f"analyze. The bracket destroys those declarations, and the "
                f"stage-close replay (ADR 0099 S7) is implemented on the "
                f"flat path only. Emit this model unpartitioned, or "
                f"activate this element's physical group globally instead "
                f"of inside a stage."
            )


def validate_builder_scope_replay(
    elements: "Sequence[Any]",
    *,
    ndm: int,
    envelope_ndf: int,
    already_gated: bool = False,
) -> None:
    """Replay-side sibling of :func:`validate_builder_scope_ordering`.

    ADR 0099 S4b.  The forward validator works over ``Element`` specs and
    ``dependencies()``; replay carries deck RECORDS (a type token, a tag,
    a flat arg tail), so the unfixable-by-ordering case has to be
    recognised differently.  Reachable only from a PRE-S1 archive — the
    bridge refuses to write one today — so this is legacy-data defence,
    not a live path.  (S7 note: this used to carry a second, INV-4 arm
    refusing stage-OWNED gated elements; the stage-close declaration
    replay lifted it, so the arm and its ``stage_owned_tags`` /
    ``scoped_present`` parameters are gone.)

    Raises
    ------
    BridgeError
        INV-3 — a gated element record references a builder-scoped
        declaration through its arg tail (``quad ... -damp N``).  Its own
        bracket destroys that declaration, so no ordering satisfies both:
        hoisted, the element resolves a tag not yet declared; unhoisted,
        the bracket purges it after the fact.  A stage-close replay does
        not save it either — the purge lands at bracket OPEN, before the
        element line parses (measured: aborts at the element line).
    """
    from ..._element_capabilities import (
        builder_scoped_kind_for_arg,
        element_builder_ndf,
    )

    # Gatedness is a property of the TOKEN at a fixed
    # (ndm, envelope_ndf) — resolve it once per distinct token, never
    # per element.  A deck carries a handful of tokens and can carry
    # millions of elements, and this guard runs on every replayed deck.
    # ``already_gated`` lets a caller that has ALREADY partitioned (the
    # ADR 0099 hoist in ``_replay_into``) hand the gated list straight
    # in, so the token pass is not paid twice on the same deck.
    if already_gated:
        gated = list(elements)
    else:
        gated_tokens = {
            tok for tok in {rec.type_token for rec in elements}
            if needs_builder_ndf_bracket_for_token(
                tok, ndm=ndm, envelope_ndf=envelope_ndf,
            )
        }
        if not gated_tokens:
            return
        gated = [rec for rec in elements if rec.type_token in gated_tokens]
    if not gated:
        return

    for rec in gated:
        # ``rec.args`` bare, not ``getattr(rec, "args", ())``: an
        # args-less record is a broken contract, and defaulting to ()
        # would SKIP the INV-3 scan silently — the one outcome this
        # guard exists to prevent.  Matches the bare ``rec.tag`` below.
        bad = sorted({
            k for a in rec.args
            if (k := builder_scoped_kind_for_arg(a)) is not None
        })
        if bad:
            raise BridgeError(
                f"replayed element {rec.type_token!r} (tag "
                f"{int(rec.tag)}) needs a builder-ndf bracket "
                f"(model basic -ndm {ndm} -ndf "
                f"{element_builder_ndf(rec.type_token, ndm)}), and that "
                f"bracket destroys the {', '.join(bad)} declaration its "
                f"own arg tail references. Per ADR 0099 INV-3 no ordering "
                f"of the deck can satisfy both. This archive predates the "
                f"emit-time INV-3 guard; re-author the model attaching "
                f"damping with ops.damping on a region instead of the "
                f"element's damp= argument, or use an ungated element."
            )


def open_builder_ndf_bracket(
    emitter: "Emitter", spec: "Element", *, ndm: int, envelope_ndf: int,
) -> bool:
    """Open a builder-ndf bracket for ``spec`` if its upstream parser needs one.

    ``OPS_FourNodeQuad`` / ``OPS_SixNodeTri`` hard-gate on the BUILDER
    state (``OPS_GetNDF() != 2``), ignoring per-node ndf — under a
    mixed-ndf envelope (ndf=3 because of beams; soil nodes inferred
    ndf=2 per ADR 0048/0049) the deck dies at the first such element.
    Re-issuing ``model basic`` does NOT wipe the domain (the same trick
    STKO decks use for their per-subset ndf switches), so the orchestrators
    bracket each gated element block: ``model basic -ndf 2`` before,
    envelope restore after (:func:`close_builder_ndf_bracket`).

    It DOES, however, delete the Tcl model builder, whose destructor purges
    the process-global ``timeSeries`` / ``geomTransf`` / ``beamIntegration``
    / ``damping`` registries — see ADR 0099 and
    :func:`validate_builder_scope_ordering`.  Callers must emit every such
    declaration AFTER the last bracket line.

    Returns True when a bracket line was emitted (caller must close).
    No-op (False) when the spec's parser carries no gate or the envelope
    already matches.
    """
    from ..._element_capabilities import element_builder_ndf

    if not needs_builder_ndf_bracket(
        spec, ndm=ndm, envelope_ndf=envelope_ndf,
    ):
        return False
    emitter.model(
        ndm=int(ndm),
        ndf=int(element_builder_ndf(type(spec).__name__, ndm)),  # type: ignore[arg-type]
    )
    return True


def close_builder_ndf_bracket(
    emitter: "Emitter", *, ndm: int, envelope_ndf: int,
) -> None:
    """Restore the model envelope after :func:`open_builder_ndf_bracket`."""
    emitter.model(ndm=int(ndm), ndf=int(envelope_ndf))


def replay_builder_scoped_declarations(
    emitter: "Emitter",
    *,
    scoped_primitives: "Sequence[Primitive]",
    tag_for: "dict[int, int]",
    transform_log: "Sequence[tuple[Any, ...]]" = (),
) -> None:
    """Re-declare the builder-scoped declarations a bracket just purged.

    ADR 0099 S7 — the stage-activated gated-element case.  A stage-owned
    gated element brackets INSIDE its stage block, after the global
    ``timeSeries`` / ``geomTransf`` / ``beamIntegration`` / ``damping``
    declarations; the bracket's ``model`` re-issue purges all four
    registries, and there is no earlier position to hoist the element to.
    So the STAGED path re-declares them at bracket close, in the exact
    global emit order (the 5b scoped pass, then the transform fan-out).

    Deliberately NOT a property of the bracket itself (the ADR records
    the measured rejection): re-declaring a tag that was NOT purged
    hard-errors (``MapOfTaggedObjects::addComponent - ... similar tag
    exists``), and the in-process module purges nothing on a ``model``
    re-issue — so a self-healing bracket would collide on ``live`` (and
    on the emitted py deck, which runs under the same module) for
    exactly the M >= 2-bracket models it would exist to fix.  Callers
    invoke this ONLY after a bracket actually fired, and ONLY on an
    emitter whose runtime actually purges (``model_reissue_purges`` —
    the Tcl deck alone today).

    Identity is safe by measurement (probe set recorded in ADR 0099 S7):
    an already-bound ``pattern`` holds a private copy of its series, an
    already-constructed element holds construction-time copies of its
    ``CrdTransf`` / ``BeamIntegration``, and ``region -damp`` copies the
    damping into the elements at the region line — so a re-declaration
    (even a MUTATED one) leaves every already-integrated stage
    bit-identical, and future by-tag references resolve the replayed
    (identical) declarations.

    ``transform_log`` is the capture :func:`emit_transform_specs` filled
    on the first pass — the orientation fan-out ALLOCATES per-vecxz tags,
    so a re-run cannot reproduce it; a log of the emitted lines can.
    Entries are ``("spec", transf, tag)`` (re-run the primitive's own
    deterministic ``_emit``) or ``("line", type_token, tag, vec, jnt)``
    (the bare-2D and fan-out forms, ``jnt`` the ``-jntOffset`` tail or
    ``()``, re-emitted verbatim).
    """
    for p in scoped_primitives:
        p._emit(emitter, tag_for[id(p)])
    for entry in transform_log:
        if entry[0] == "spec":
            _kind, transf, tag = entry
            transf._emit(emitter, tag)
        else:
            _kind, type_token, tag, vec, jnt = entry
            emitter.geomTransf(
                type_token, tag, *vec,
                **({"jnt_offset": jnt} if jnt else {}))
