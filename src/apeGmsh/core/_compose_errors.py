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

