"""Error types for ADR 0038 `g.compose()` model composition.

These exceptions surface at compose time when the tag-collision
verifier or the namespace/reservation machinery detects a violation
of the merge contract.  They are intentionally narrow — one error
class per failure mode, with a clear `__str__` that names the
offending module label and the colliding identifier.

The verifier (`_tag_collision_verifier.tag_collision_verify`) and
the future Compose facade (`mesh/_compose.py`, ADR 0038 Phase 3)
both raise from this module.  Keeping the exception types in
``core/`` instead of ``mesh/`` lets the verifier — which is a
reusable primitive — be imported by both call sites without
introducing a layering cycle.

ADR 0038 §"Tag-collision verifier" pins the contract.
"""
from __future__ import annotations


class PartTagCollisionError(RuntimeError):
    """Raised when the tag-collision verifier detects a tag-space
    or namespace overlap that would corrupt the host broker.

    Raised by checks 1, 2, and 4 of the verifier:

    - Check 1: an imported tag lands outside the module's reservation
      window (typically into the host's tag range).
    - Check 2: two modules' reservation windows overlap.
    - Check 4: a namespaced physical-group name collides with an
      existing host PG.

    The message names the offending module label and the colliding
    identifier so callers can fix the source H5 or the host author's
    PG-naming choice.
    """


class ComposeCapacityError(RuntimeError):
    """Raised when an explicit ``compose_size_per_module=N`` override
    is smaller than the source's actual tag span.

    Verifier check 5.  Only fires under the explicit override —
    without an override, the auto-sizing scheme computes a window
    that fits the source by construction.

    The message names the source's span and the configured cap so
    callers can either widen the override or remove it to let the
    auto-sizing scheme do its job.
    """


class ComposeInvariantError(RuntimeError):
    """Raised when a constraint reference resolves outside the
    owning module's reservation window.

    Verifier check 3.  This means either the tag-rewrite cover-set
    missed a field (an implementation bug — see ADR 0038
    §"Tag-reference rewrite checklist") or the source H5 carried a
    cross-module reference between sibling composes (forbidden in
    v1 — see ADR 0038 INV-2).

    The message names the offending module label, the constraint
    kind, the field that holds the bad reference, and the
    out-of-range tag value.
    """


class ComposeInterfaceSizeWarning(UserWarning):
    """Advisory warning raised when a compose's interface size sits in
    the regime where downstream cross-rank emit / parse cost may
    become a problem per ADR 0038 §"v1 scope gate".

    Phase 3F.1.  The Phase 1 gate (10k x 4 ranks) passed; the 100k x 8
    cell breached emit/parse/RSS thresholds.  Composing a source with
    more than :data:`apeGmsh.mesh._compose.WARN_INTERFACE_SIZE`
    interface-class constraint records puts the composed assembly
    into the latter regime, so the merge engine emits this warning
    once per compose call.

    Subclass of :class:`UserWarning` so it can be silenced with
    ``warnings.simplefilter("ignore",
    apeGmsh.core._compose_errors.ComposeInterfaceSizeWarning)`` when
    callers accept the cost.  Distinct from
    :class:`~apeGmsh.mesh._compose.ComposeFilterWarning` so users can
    filter the two independently.
    """


class ComposeDepthExceededError(RuntimeError, ValueError):
    """Raised when a nested-compose operation would exceed the
    configured ``max_compose_depth`` per ADR 0038 §"Nested composition".

    Phase 3E.1.  The source's own ``composed_from`` chain has reached
    the configured depth cap and composing it into the host would
    push the new module-entry's depth past the cap.

    The check fires at compose time, before any merge work runs:
    ``source_depth = 1 + max(child.depth)`` over the source's own
    ``composed_from`` records (depth 0 for a never-composed source).
    Composing yields a new entry of depth ``source_depth + 1``; if
    that exceeds ``max_compose_depth`` (default 3), this error is
    raised.

    The default of 3 covers the canonical hierarchy connection →
    frame → building plus one level of headroom.  The cap can be
    lifted per-call via the ``max_compose_depth=N`` kwarg on
    :meth:`~apeGmsh.mesh._compose.Compose.compose`.

    Inherits from both :class:`RuntimeError` (matching the other
    verifier-style errors in this module) AND :class:`ValueError` so
    callers writing ``except ValueError`` continue to catch it,
    consistent with the facade-layer compose errors in
    :mod:`apeGmsh.mesh._compose`.
    """


def chain_phase_guard(session, operation: str) -> None:
    """Tolerant wrapper around ``session._check_chain_phase``.

    Used at every build-phase chokepoint so the call is a no-op for
    stub parents (test fixtures created via ``type("_Stub", (), ...)``
    that don't carry the helper).  Real :class:`_SessionBase`
    subclasses delegate to :meth:`_SessionBase._check_chain_phase`.
    """
    guard = getattr(session, "_check_chain_phase", None)
    if guard is not None:
        guard(operation)


class ChainPhaseError(RuntimeError):
    """Raised when a build-phase API is called on a chain-phase session.

    Phase 3B.2d / ADR 0038.  Once the session has produced its first
    :class:`FEMData` (``apeGmsh._fem is not None``) the session enters
    *chain phase*: the canonical model is the broker snapshot, not the
    live gmsh state.  Geometry / mesh / PG / label / parts / sections
    mutations would diverge the broker from gmsh and silently corrupt
    downstream consumers, so the session refuses them.

    The interface-bridging primitives ``g.constraints.embedded`` /
    ``tied_contact`` / ``equalDOF`` / ``rigid_link`` / ``rigid_diaphragm``
    plus ``g.loads.X`` / ``g.masses.X`` are NOT gated — these are
    documented per ADR 0038 line 45 as the chain-phase composition
    surface and route through ``FEMData.with_*`` transforms.

    Recovery
    --------
    Reload the session from disk (``apeGmsh.from_h5(path)``) or restart
    the build phase before applying the mutation.  Chain-phase
    composition (``g.compose(...)``) is the supported way to extend a
    saved model.
    """


#: Default H5-safe alternatives quoted by :func:`raise_if_no_live_kernel`.
#: Callers whose composite has a more specific counterpart on the broker
#: (e.g. ``g.labels`` -> ``LabelSet``) pass their own ``alternative=``.
_DEFAULT_KERNEL_ALTERNATIVE = (
    "fem.inspect for model summaries and fem.physical (PhysicalGroupSet) "
    "for physical-group queries, where fem = g.mesh.queries.get_fem_data(); "
    "for post-processing use results.inspect on a Results object"
)


def is_kernelless_session(session) -> bool:
    """True when ``session`` was built via :meth:`apeGmsh.from_h5`.

    The predicate behind :func:`raise_if_no_live_kernel`, exposed
    separately for the two ``__repr__`` implementations that must
    *degrade* rather than raise — see that function's note on why
    ``__repr__`` is not a guard site.
    """
    return bool(getattr(session, "_fem_from_h5", False))


def raise_if_no_live_kernel(
    session, verb: str, *, alternative: str | None = None,
) -> None:
    """Guard a kernel-introspection entry point against from_h5 sessions.

    Live-kernel composites (``g.inspect``, the ``g.physical`` and
    ``g.labels`` queries, ``g.view``, ``g.plot``) read the gmsh model
    directly.  A session built via :meth:`apeGmsh.from_h5` has no gmsh
    state of its own, so those reads either die with a raw gmsh error
    ("Gmsh has not been initialized") or — when another session happens
    to hold the kernel — silently answer from an unrelated model.
    Raise :class:`ChainPhaseError` at the composite entry point
    instead, naming the H5-safe alternatives.

    Live sessions (``_fem_from_h5`` absent/False) pass untouched, as do
    stub parents used by low-level test fixtures.

    NOT for ``__repr__``.  Python calls ``__repr__`` from debuggers,
    ``print``, logging, exception formatting and pytest's assertion
    output; a raising ``__repr__`` turns a legible failure into an
    opaque cascading one, and would break the display of any object
    merely *holding* a chain-phase composite.  Those two sites use
    :func:`is_kernelless_session` and return a descriptive string.

    Parameters
    ----------
    session : object
        The owning session (``self._parent`` at every call site).
    verb : str
        Fully-qualified API being called, e.g. ``"g.labels.entities()"``.
        Surfaces first in the message so the offending call is obvious.
    alternative : str or None
        Overrides the quoted H5-safe surfaces for composites with a
        more specific broker counterpart.  ``None`` uses
        :data:`_DEFAULT_KERNEL_ALTERNATIVE`.
    """
    if getattr(session, "_fem_from_h5", False):
        alt = (
            _DEFAULT_KERNEL_ALTERNATIVE if alternative is None
            else alternative
        )
        raise ChainPhaseError(
            f"{verb} requires a live gmsh kernel — this session was "
            f"built via apeGmsh.from_h5 and carries no gmsh state to "
            f"introspect.  Use the H5-safe surfaces instead: {alt}."
        )


def raise_if_from_h5_session(session, verb: str) -> None:
    """Guard a gmsh-resolving verb against from_h5/compose sessions.

    Silent-failures slice 2: ``g.constraints.contact`` /
    ``contact_plane``, ``g.embed``, ``g.reinforce`` and
    ``g.decouple_node`` append defs that resolve from the **live gmsh
    session** at ``get_fem_data()`` extraction.  A ``from_h5`` /
    compose session has no gmsh and never re-extracts, so the def
    would be stored and silently never applied — the model solves
    without the coupling.  Raise :class:`ChainPhaseError` at the verb
    instead.

    Live sessions (gmsh running, ``_fem`` merely a cache) pass — their
    next extraction resolves the def, failing loud there on any bad
    target.
    """
    if getattr(session, "_fem_from_h5", False):
        raise ChainPhaseError(
            f"{verb} requires a live gmsh session — it resolves from "
            f"gmsh geometry at extraction, which never happens in a "
            f"from_h5/compose (chain-phase) session, so the definition "
            f"would be stored but silently never applied.  Declare it "
            f"in the source part session before saving; the resolved "
            f"records round-trip through model.h5 and survive "
            f"g.compose."
        )

