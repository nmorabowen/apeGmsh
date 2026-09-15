"""The one classification point for an ``analyze()`` return code.

Every apeGmsh check on an ``analyze`` rc is ``!= 0`` / ``== 0``: while
every non-zero code meant "this increment did not converge", that was
enough, and the cure was always the same — subdivide the step, or
escalate the solution algorithm.

``-4`` is the first rc that does **not** mean that: it says the
**commit was refused**, not that the increment failed to converge.
``StaticAnalysis`` / ``DirectIntegrationAnalysis`` return it for ANY
``Integrator::commit()`` failure, and ``Domain::commit()`` fails for
more than one reason — the commit-time IMPL-EX refusal latch the
Ladruno fork added in PR #838 (a material refusing at
``commitState()``, which ``AnalysisModel::commitDomain()`` turns into
``-2`` and the analysis class into ``-4``), and an ADR-73
``LadrunoPorousOverlay`` fluid solve that fails, among others.  This
module deliberately does **not** name a cause.

What every one of them shares is the reason a retry cannot help: a
failed commit leaves the domain **partially committed** — the nodes
and the sibling integration points have already advanced — so the
model state is inconsistent rather than merely un-advanced.
Subdividing the step or escalating the algorithm cannot rescue it; the
run has to restart from the last good checkpoint.

``-33086`` (``LADRUNO_MATERIAL_REFUSED``) is the **trial**-time
refusal raised under ``-implexControl``.  It is unchanged by #838 and
stays **retryable**: it fails the step it belongs to, before anything
is committed, which is exactly the case a smaller increment rescues.
Only ``-4`` aborts.
"""
from __future__ import annotations

__all__ = [
    "COMMIT_ABORT_MESSAGE",
    "COMMIT_ABORT_RC",
    "AnalysisAbortedError",
    "check_analyze_rc",
]

#: ``analyze()`` rc for a refused commit — NON-retryable.
COMMIT_ABORT_RC: int = -4

#: ``LADRUNO_MATERIAL_REFUSED``, the trial-time refusal — RETRYABLE,
#: and deliberately not classified as an abort here.
MATERIAL_REFUSED_RC: int = -33086

#: The operator-facing explanation of :data:`COMMIT_ABORT_RC`, shared by
#: :func:`check_analyze_rc` and by the ``py`` / ``tcl`` emitters, which
#: interpolate it into the rung ladder they write into a deck.  Kept
#: ASCII-only and free of ``"``, ``$``, ``\``, ``[`` and ``{`` so it can
#: be embedded unescaped in both a Python ``raise`` literal and a Tcl
#: ``error`` literal.
COMMIT_ABORT_MESSAGE: str = (
    f"OpenSees analyze() returned {COMMIT_ABORT_RC}: the COMMIT was "
    "refused. That is the IMPL-EX commit-time refusal latch since Ladruno "
    "fork PR #838, but the same rc is returned for ANY other "
    "Integrator::commit() or Domain::commit() failure (an ADR-73 "
    "LadrunoPorousOverlay fluid solve, for one), so it names the refused "
    "commit and not its cause. A failed commit leaves the domain PARTIALLY "
    "committed -- the nodes and the sibling integration points have already "
    "advanced -- so the model state is INCONSISTENT, not merely "
    "un-advanced: subdividing the step or escalating the algorithm cannot "
    "rescue it. Restart the run from the last good checkpoint. (The "
    f"trial-time refusal {MATERIAL_REFUSED_RC} is the retryable one; this "
    "is not it.)"
)


class AnalysisAbortedError(RuntimeError):
    """An ``analyze()`` rc that no retry can rescue.

    Distinct from ordinary non-convergence: the commit itself was
    refused and the domain is left partially committed, so the retry
    machinery must stop subdividing and stop escalating rungs, and the
    run has to restart from the last good checkpoint.
    """


def check_analyze_rc(rc: int, *, where: str) -> int:
    """Return ``rc``, or raise :class:`AnalysisAbortedError` on an abort.

    Call this **before** subdividing an increment or escalating to the
    next ladder rung — the point of it is to keep a retry from being
    attempted at all.
    """
    rc = int(rc)
    if rc == COMMIT_ABORT_RC:
        raise AnalysisAbortedError(f"{where}: {COMMIT_ABORT_MESSAGE}")
    return rc
