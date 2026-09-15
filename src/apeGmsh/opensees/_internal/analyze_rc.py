"""The one classification point for an ``analyze()`` return code.

Every apeGmsh check on an ``analyze`` rc is ``!= 0`` / ``== 0``: while
every non-zero code meant "this increment did not converge", that was
enough, and the cure was always the same — subdivide the step, or
escalate the solution algorithm.

The Ladruno fork's commit-time IMPL-EX refusal latch (fork PR #838)
introduced the first rc that does **not** mean that.  When a material
refuses at ``commitState()``, ``Domain::commit()`` returns ``-1``
*after* its element loop, ``AnalysisModel::commitDomain()`` turns that
into ``-2`` and every analysis class into ``-4``.  By then the nodes
and the refusing point's sibling Gauss points have **already
committed**, so the model state is inconsistent rather than merely
un-advanced, the material's latch is sticky, and every later step
returns ``-4`` as well.  Retrying it — at any step size, under any
algorithm — cannot succeed.

``-33086`` (``LADRUNO_MATERIAL_REFUSED``) is the **trial**-time
refusal raised under ``-implexControl``.  It is unchanged by #838 and
stays **retryable**: it fails the step it belongs to, before anything
is committed, which is exactly the case a smaller increment rescues.
Only ``-4`` aborts.
"""
from __future__ import annotations

__all__ = [
    "COMMIT_ABORT_RC",
    "MATERIAL_REFUSED_RC",
    "AnalysisAbortedError",
    "check_analyze_rc",
]

#: ``analyze()`` rc for a commit-time material refusal — NON-retryable.
COMMIT_ABORT_RC: int = -4

#: ``LADRUNO_MATERIAL_REFUSED``, the trial-time refusal — RETRYABLE,
#: and deliberately not classified as an abort here.
MATERIAL_REFUSED_RC: int = -33086


class AnalysisAbortedError(RuntimeError):
    """An ``analyze()`` rc that no retry can rescue.

    Distinct from ordinary non-convergence: the retry machinery must
    stop subdividing and stop escalating rungs, and the run has to
    restart from the last good checkpoint.
    """


def check_analyze_rc(rc: int, *, where: str) -> int:
    """Return ``rc``, or raise :class:`AnalysisAbortedError` on an abort.

    Call this **before** subdividing an increment or escalating to the
    next ladder rung — the point of it is to keep a retry from being
    attempted at all.
    """
    rc = int(rc)
    if rc == COMMIT_ABORT_RC:
        raise AnalysisAbortedError(
            f"{where}: OpenSees analyze() returned {COMMIT_ABORT_RC} — a "
            "material refused the COMMIT, so Domain::commit() aborted the "
            "step (Ladruno fork PR #838). The nodes and the refusing "
            "point's sibling Gauss points had already committed, so the "
            "model state is INCONSISTENT, not merely un-advanced: "
            "subdividing the step or escalating the algorithm cannot "
            "rescue it, and every later step will return "
            f"{COMMIT_ABORT_RC} too. Restart the run from the last good "
            "checkpoint. (The trial-time refusal "
            f"{MATERIAL_REFUSED_RC} is the retryable one; this is not it.)"
        )
    return rc
