"""``LadrunoRCConcrete`` / ``LadrunoRCFiniteStrain`` C2 flags vs. the bound build.

``-betaC`` and ``-crackedNu`` (``_LadrunoRC.beta_c`` / ``cracked_nu``) were
written against fork PR #873, which was closed UNMERGED on 2026-09-27 and is
being re-landed as fork PR #877. No build of the fork's ``ladruno`` line
carries them yet (verified at ``891978c9e``, 2026-09-28), and that parser's
option loop ends in ``// unknown tokens are ignored (forward-compat)``: a
build without the flags accepts ``-crackedNu 0.2 -betaC 189`` and runs the
elastic Poisson ratio after cracking and ``C = 170``. No error is raised.
The result is a wrong model.

This leaf module (no apeGmsh imports) is shared by the live emitter, which
refuses the flags, and the Tcl / openseespy emitters, which warn.
:mod:`apeGmsh.opensees.material.nd` re-exports the public names beside its
other build floors.
"""
from __future__ import annotations

import warnings

__all__ = [
    "LADRUNO_RC_C2_MIN_BUILD",
    "LadrunoRCBuildWarning",
    "rc_c2_flags",
    "rc_c2_live_refusal",
    "warn_rc_c2_deck",
]

#: Minimum fork build (``ops.ladrunoBuild()``) that parses ``-betaC`` /
#: ``-crackedNu`` on ``LadrunoRCConcrete`` / ``LadrunoRCFiniteStrain``.
#: ``None`` while fork PR #877 is unmerged: no build on the ``ladruno`` line
#: carries them, so the live route refuses them outright. Set it to #877's
#: merge SHA when that PR lands. From then on, the live route refuses only a
#: build with no stamp (stock openseespy, or a fork build that predates
#: ``ladrunoBuild``, fork PR #718). It cannot check that a stamped build
#: descends from the floor, because a bare hash cannot prove ancestry (the
#: ``ASDP_MIN_FORK_BUILD`` convention, ADR 0107 D4).
LADRUNO_RC_C2_MIN_BUILD: str | None = None

_RC_MATERIALS = frozenset({"LadrunoRCConcrete", "LadrunoRCFiniteStrain"})
_C2_FLAGS = ("-betaC", "-crackedNu")


class LadrunoRCBuildWarning(UserWarning):
    """A Tcl / openseespy deck carries ``-betaC`` / ``-crackedNu``.

    Only a fork build that carries fork PR #877 reads them. Every other
    build silently discards them and runs the pre-C2 law.
    """


def rc_c2_flags(
    mat_type: str, params: "tuple[object, ...]",
) -> tuple[str, ...]:
    """The C2 flag tokens among ``params`` of an RC ``nDMaterial`` line."""
    if mat_type not in _RC_MATERIALS:
        return ()
    return tuple(
        f for f in _C2_FLAGS
        if any(isinstance(p, str) and p == f for p in params)
    )


def _floor_text() -> str:
    if LADRUNO_RC_C2_MIN_BUILD is None:
        return (
            "no build of the fork's ladruno branch carries them yet (fork "
            "PR #873 was closed unmerged; they are being re-landed as fork "
            "PR #877)"
        )
    return (
        f"only fork builds at or after {LADRUNO_RC_C2_MIN_BUILD} "
        f"(LADRUNO_RC_C2_MIN_BUILD) carry them"
    )


def warn_rc_c2_deck(mat_type: str, params: "tuple[object, ...]") -> None:
    """Warn when a deck line carries C2 flags (Tcl / openseespy emitters)."""
    flags = rc_c2_flags(mat_type, params)
    if not flags:
        return
    warnings.warn(
        f"nDMaterial {mat_type} emits {' '.join(flags)}: {_floor_text()}. "
        f"A build without them silently discards the flags and runs the "
        f"pre-C2 law (elastic nu after cracking, betaC 170). Run this deck "
        f"only on a build that carries them.",
        LadrunoRCBuildWarning,
        stacklevel=3,
    )


def rc_c2_live_refusal(
    mat_type: str, params: "tuple[object, ...]", build: str | None,
) -> str | None:
    """The live route's refusal message, or ``None`` when the line may run.

    ``build`` is the bound backend's ``ladrunoBuild()`` stamp (``None`` when
    the backend has no such command).
    """
    flags = rc_c2_flags(mat_type, params)
    if not flags:
        return None
    if LADRUNO_RC_C2_MIN_BUILD is not None and build is not None:
        return None
    return (
        f"nDMaterial {mat_type} with {' '.join(flags)} refused on the live "
        f"route: {_floor_text()}, and the bound build "
        f"(ladrunoBuild() = {build!r}) cannot be shown to carry them. A "
        f"build without them does not error: it silently discards the "
        f"flags and runs the pre-C2 law (elastic nu after cracking, betaC "
        f"170). Drop beta_c / cracked_nu, or emit the deck with "
        f"ops.tcl(...) / ops.py(...) and run it on a build that carries "
        f"them."
    )
