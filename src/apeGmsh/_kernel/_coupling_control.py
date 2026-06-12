"""Explicit control knobs for the Ladruno fork coupling elements.

Leaf module (``dataclasses`` only, zero ``apeGmsh.*`` imports) so it can be
imported by **both** :mod:`apeGmsh._kernel.defs.constraints` and
:mod:`apeGmsh._kernel.records._constraints` without re-triggering the
``records`` ↔ ``defs`` package-init cycle (importing a submodule under a
sibling package would run that package's ``__init__``; a top-level
``_kernel`` leaf does not).
"""
from __future__ import annotations

from dataclasses import dataclass

#: Valid ``-enforce`` modes for the fork coupling elements.
_ENFORCE_MODES: tuple[str, ...] = ("penalty", "al")


@dataclass(frozen=True)
class CouplingControl:
    """Explicit penalty / enforcement knobs for the fork coupling elements
    (``LadrunoKinematicCoupling`` / RBE2 and ``LadrunoDistributingCoupling``
    / RBE3).

    Carried on the coupling ``*Def`` and copied onto the resolved record so
    the bridge emits the matching flags. Every field defaults to "unset" ⇒
    the flag is omitted and the fork element's own default applies (``-k``
    ``1e12``, derived ``-kr``, penalty enforcement, no bipenalty, ``g0``
    stress-free birth on). Fields map 1:1 to the fork flags:

    ==================  ==========================  ==========================
    field               flag                        meaning
    ==================  ==========================  ==========================
    ``k``               ``-k $Kt``                  translational penalty (>0)
    ``kr``              ``-kr $Kr``                 rotational penalty (>0);
                                                    else fork-derived ``K_t·ℓ²``
    ``enforce``         ``-enforce {penalty|al}``   ``al`` = augmented
                                                    Lagrangian (implicit only)
    ``bipenalty_dtcr``  ``-bipenalty -dtcr $dt``    explicit critical-step
                                                    target (>0)
    ``absolute``        ``-absolute``               keep the absolute tie
                                                    (skip ``g0`` birth)
    ==================  ==========================  ==========================

    Deferred (host-element auto-scalers — they need a representative host
    element tag the constraint-emit pass can't yet resolve): ``-k auto`` /
    ``-kAlpha`` / ``-host`` / ``-wcap``. Numeric ``k`` + ``bipenalty_dtcr``
    give full manual control meanwhile.
    """
    k: float | None = None
    kr: float | None = None
    enforce: str = "penalty"
    bipenalty_dtcr: float | None = None
    absolute: bool = False

    def __post_init__(self) -> None:
        if self.enforce not in _ENFORCE_MODES:
            raise ValueError(
                f"CouplingControl: enforce must be one of {_ENFORCE_MODES}, "
                f"got {self.enforce!r}."
            )
        for nm, val in (("k", self.k), ("kr", self.kr),
                        ("bipenalty_dtcr", self.bipenalty_dtcr)):
            if val is not None and not (val > 0):
                raise ValueError(
                    f"CouplingControl: {nm} must be > 0 if set, got {val!r}."
                )
        # The fork refuses -enforce al together with -bipenalty: the Uzawa
        # update has no equilibrium iteration to converge against under an
        # explicit integrator (combining them is a parse error there).
        if self.enforce == "al" and self.bipenalty_dtcr is not None:
            raise ValueError(
                "CouplingControl: enforce='al' (augmented Lagrangian, "
                "implicit) cannot be combined with bipenalty_dtcr "
                "(explicit-dynamics control) — the fork refuses this pairing."
            )

    @property
    def is_default(self) -> bool:
        """True when no knob is set — emits no flags, so the resolver can
        store ``None`` on the record instead of a no-op control."""
        return (
            self.k is None and self.kr is None and self.enforce == "penalty"
            and self.bipenalty_dtcr is None and not self.absolute
        )

    def emit_flags(self) -> list[int | float | str]:
        """Order-independent flag tail for the element command (defaults
        elided so the fork's own defaults apply)."""
        out: list[int | float | str] = []
        if self.k is not None:
            out += ["-k", self.k]
        if self.kr is not None:
            out += ["-kr", self.kr]
        if self.enforce != "penalty":
            out += ["-enforce", self.enforce]
        if self.bipenalty_dtcr is not None:
            out += ["-bipenalty", "-dtcr", self.bipenalty_dtcr]
        if self.absolute:
            out += ["-absolute"]
        return out
