"""Refuse a ``consistent_tan`` contact that would solve on a symmetric SOE.

``g.constraints.contact(..., consistent_tan=True)`` emits ``-consistanttan``
and ``edge_consistent_tan=True`` emits ``-edgeConsistentTan``: both select
the fork's non-symmetric consistent friction tangent (``ContactDef``'s own
docstring: "REQUIRES a non-symmetric solver"). A half-storage solver reads
only the ``col >= row`` half of each element matrix, drops the off-diagonal
coupling at assembly, and returns a plausible but WRONG solve with rc 0 —
nothing checked this before #1273. Fail-LOUD, like the u-p gate: the answer
is not "less accurate", it is a different system.

Scope mirrors :func:`~apeGmsh.opensees._internal.build.validate_ladruno_up_solver`
exactly: a DECLARED symmetric system is wrong whether or not this emit
solves; the MISSING-system branch (OpenSees' no-``system`` default is
ProfileSPD) is gated on ``enforce`` and skipped for a partitioned deck,
which rides the ADR-0027 INV-5 general auto-emit. Per the #1273
recommendation, a missing ``system`` counts as a violation.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .build import (
    _GENERAL_SOLVER_MSG,
    _UNSYMMETRIC_SAFE_SYSTEMS,
    BridgeError,
    contact_records,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from apeGmsh.mesh.FEMData import FEMData

__all__ = ["validate_consistent_tan_solver"]


def _offenders(fem: "FEMData") -> list[str]:
    """``contact 'name' (flag)`` for every contact asking for a consistent tangent."""
    out: list[str] = []
    for i, rec in enumerate(contact_records(fem, "contact")):
        name = rec.name if getattr(rec, "name", None) else f"#{i}"
        flags = [
            flag for flag in ("consistent_tan", "edge_consistent_tan")
            if getattr(rec, flag, False)
        ]
        if flags:
            out.append(f"contact {name!r} ({', '.join(flags)})")
    return out


def validate_consistent_tan_solver(
    fem: "FEMData",
    *,
    enforce: bool,
    staged: bool,
    partitioned: bool,
    flat_systems: "Sequence[Any]",
    stage_systems: "Sequence[tuple[str, Any | None]]",
) -> None:
    """Raise :class:`BridgeError` when a ``consistent_tan`` /
    ``edge_consistent_tan`` contact would be solved on a symmetric or
    diagonal linear system (#1273).

    Allowed: :data:`_UNSYMMETRIC_SAFE_SYSTEMS`, with ``Pardiso`` / ``Mumps``
    only in their default ``matrix_type="unsymmetric"`` mode. The error names
    the allowed solvers. A FEM with no such contact returns silently.
    """
    offenders = _offenders(fem)
    if not offenders:
        return
    who = ", ".join(offenders)

    def _why(token: str, where: str, detail: str) -> str:
        return (
            f"{who} with system {token} ({where}): {detail} The consistent "
            f"friction tangent (-consistanttan / -edgeConsistentTan) is "
            f"UNSYMMETRIC, so the solve would use only half of it and "
            f"converge to a plausible but WRONG answer with rc 0 (#1273). "
            f"Declare a general solver: {_GENERAL_SOLVER_MSG} — or drop "
            f"consistent_tan (the default symmetric tangent is correct on "
            f"any solver)."
        )

    def _check(system: Any, where: str) -> None:
        token = type(system).__name__
        if token not in _UNSYMMETRIC_SAFE_SYSTEMS:
            raise BridgeError(_why(
                repr(token), where, f"{token} does not store the full matrix.",
            ))
        # Pardiso / Mumps are legal only in their UNSYMMETRIC mode (fork
        # ADR-75 P1d): the half-storage modes are exactly the silent drop
        # this gate exists to stop.
        mtype = getattr(system, "matrix_type", "unsymmetric")
        if mtype != "unsymmetric":
            raise BridgeError(_why(
                f"{token}(matrix_type={mtype!r})", where,
                "this mode stores only the upper triangle.",
            ))

    if staged:
        for name, system in stage_systems:
            if system is not None:
                _check(system, f"stage {name}")
        if enforce:
            for name, system in stage_systems:
                if system is None:
                    raise BridgeError(_why(
                        "'ProfileSPD'", f"stage {name}, undeclared",
                        "the stage analyzes with no linear system, so it "
                        "runs on the OpenSees ProfileSPD default after the "
                        "prior stage's wipeAnalysis.",
                    ))
        return

    # Flat: the last-declared system is the effective one at analyze time.
    if flat_systems:
        _check(flat_systems[-1], "global")
        return
    if not enforce or partitioned:
        # Never solves (archival / eigen-only / model-only), or rides the
        # ADR-0027 INV-5 general auto-emit.
        return
    raise BridgeError(_why(
        "'ProfileSPD'", "undeclared",
        "no linear system is declared, so the deck runs on the OpenSees "
        "no-`system` default, ProfileSPD.",
    ))
