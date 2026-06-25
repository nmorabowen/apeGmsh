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
    ``k``               ``-k {Kt|auto}``            translational penalty
                                                    (>0), or ``"auto"`` =
                                                    scale off the ``host``
                                                    element's diagonal
    ``k_alpha``         ``-kAlpha a``               multiplier for
                                                    ``k="auto"`` (fork
                                                    default ``1e3``)
    ``host``            ``-host $eleTag``           representative host
                                                    element — a **FEM
                                                    element id**; the bridge
                                                    translates it to the
                                                    emitted OpenSees tag
    ``kr``              ``-kr $Kr``                 rotational penalty (>0);
                                                    else fork-derived ``K_t·ℓ²``
    ``enforce``         ``-enforce {penalty|al}``   ``al`` = augmented
                                                    Lagrangian (implicit only)
    ``bipenalty_dtcr``  ``-bipenalty -dtcr $dt``    explicit critical-step
                                                    target (>0)
    ``bipenalty_wcap``  ``-bipenalty -wcap $beta``  penalty mass from the
                                                    host frequency
                                                    ``m_p = K_t/(β·ω_host)²``
                                                    (needs ``host``)
    ``absolute``        ``-absolute``               keep the absolute tie
                                                    (skip ``g0`` birth)
    ==================  ==========================  ==========================

    ``host`` is stored as the **FEM element id** (stable across emits); the
    bridge resolves it to the emitted OpenSees element tag at emit time and
    passes it into :meth:`emit_flags` — the control never sees ops tags.
    """
    k: float | str | None = None
    kr: float | None = None
    enforce: str = "penalty"
    bipenalty_dtcr: float | None = None
    absolute: bool = False
    # Host-element auto-scalers (handoff item A).
    k_alpha: float | None = None
    host: int | None = None
    bipenalty_wcap: float | None = None

    def __post_init__(self) -> None:
        if self.enforce not in _ENFORCE_MODES:
            raise ValueError(
                f"CouplingControl: enforce must be one of {_ENFORCE_MODES}, "
                f"got {self.enforce!r}."
            )
        if isinstance(self.k, str):
            if self.k != "auto":
                raise ValueError(
                    f"CouplingControl: k must be a positive number or "
                    f"'auto', got {self.k!r}."
                )
            if self.host is None:
                raise ValueError(
                    "CouplingControl: k='auto' needs a representative host "
                    "element — pass host=<FEM element id>."
                )
        elif self.k is not None and not (self.k > 0):
            raise ValueError(
                f"CouplingControl: k must be > 0 if set, got {self.k!r}."
            )
        for nm, val in (("kr", self.kr),
                        ("bipenalty_dtcr", self.bipenalty_dtcr),
                        ("bipenalty_wcap", self.bipenalty_wcap),
                        ("k_alpha", self.k_alpha)):
            if val is not None and not (val > 0):
                raise ValueError(
                    f"CouplingControl: {nm} must be > 0 if set, got {val!r}."
                )
        if self.k_alpha is not None and self.k != "auto":
            raise ValueError(
                "CouplingControl: k_alpha only scales k='auto' — the fork "
                "ignores -kAlpha next to a numeric -k. Pass k='auto' (with "
                "host=) or drop k_alpha."
            )
        if self.host is not None:
            if not isinstance(self.host, int) or isinstance(self.host, bool) \
                    or self.host <= 0:
                raise ValueError(
                    f"CouplingControl: host must be a positive FEM element "
                    f"id (int), got {self.host!r}."
                )
            if self.k != "auto" and self.bipenalty_wcap is None:
                raise ValueError(
                    "CouplingControl: host has no consumer — the fork only "
                    "reads -host for k='auto' or bipenalty_wcap. Set one of "
                    "those or drop host."
                )
        if self.bipenalty_wcap is not None:
            if self.bipenalty_dtcr is not None:
                raise ValueError(
                    "CouplingControl: bipenalty_dtcr and bipenalty_wcap are "
                    "mutually exclusive — the fork's -bipenalty takes ONE of "
                    "-dtcr <dt> or -wcap <beta>."
                )
            if self.host is None:
                raise ValueError(
                    "CouplingControl: bipenalty_wcap needs the host "
                    "frequency — pass host=<FEM element id>."
                )
        # The fork refuses -enforce al together with -bipenalty: the Uzawa
        # update has no equilibrium iteration to converge against under an
        # explicit integrator (combining them is a parse error there).
        if self.enforce == "al" and (
            self.bipenalty_dtcr is not None or self.bipenalty_wcap is not None
        ):
            raise ValueError(
                "CouplingControl: enforce='al' (augmented Lagrangian, "
                "implicit) cannot be combined with bipenalty_dtcr / "
                "bipenalty_wcap (explicit-dynamics controls) — the fork "
                "refuses this pairing."
            )

    @property
    def is_default(self) -> bool:
        """True when no knob is set — emits no flags, so the resolver can
        store ``None`` on the record instead of a no-op control."""
        return (
            self.k is None and self.kr is None and self.enforce == "penalty"
            and self.bipenalty_dtcr is None and not self.absolute
            and self.k_alpha is None and self.host is None
            and self.bipenalty_wcap is None
        )

    def emit_flags(
        self, *, host_ops_tag: int | None = None,
    ) -> list[int | float | str]:
        """Order-independent flag tail for the element command (defaults
        elided so the fork's own defaults apply).

        ``host_ops_tag`` is the emitted OpenSees tag of the ``host``
        element — the emit site translates the stored FEM eid through the
        bridge's ``fem_eid_to_ops_tag`` map and passes it here. Required
        exactly when ``host`` is set (a control with a host cannot emit
        without the translation; emitting the raw FEM eid would silently
        target the wrong element).
        """
        if self.host is not None and host_ops_tag is None:
            raise ValueError(
                f"CouplingControl: host={self.host} (a FEM element id) "
                "needs the emitted OpenSees tag — the emit site must "
                "translate it via fem_eid_to_ops_tag and pass "
                "host_ops_tag=."
            )
        out: list[int | float | str] = []
        if self.k is not None:
            out += ["-k", self.k]
        if self.k_alpha is not None:
            out += ["-kAlpha", self.k_alpha]
        if self.host is not None:
            out += ["-host", int(host_ops_tag)]  # type: ignore[arg-type]
        if self.kr is not None:
            out += ["-kr", self.kr]
        if self.enforce != "penalty":
            out += ["-enforce", self.enforce]
        if self.bipenalty_dtcr is not None:
            out += ["-bipenalty", "-dtcr", self.bipenalty_dtcr]
        if self.bipenalty_wcap is not None:
            out += ["-bipenalty", "-wcap", self.bipenalty_wcap]
        if self.absolute:
            out += ["-absolute"]
        return out


@dataclass(frozen=True)
class EmbeddedNodeControl(CouplingControl):
    """``CouplingControl`` extended with the ``LadrunoEmbeddedNode``-only
    knobs that the RBE2/RBE3 coupling elements do **not** accept.

    Used as ``control=`` on the ``enforce="penalty_al"`` surface ties
    (``tie`` / ``tied_contact`` / ``embedded``), which emit
    ``LadrunoEmbeddedNode``. A plain :class:`CouplingControl` still works
    there (it is the base); reach for this subclass only when you need the
    extra element features. The base ``-k``/``-kr``/``-enforce``/
    ``-bipenalty``/``-absolute`` knobs behave identically.

    v1 adds the **pressure (u-p) tie**:

    ==========  =====================  ===================================
    field       flag                   meaning
    ==========  =====================  ===================================
    ``pressure``  ``-pressure``        also couple the constrained node's
                                       pressure DOF (index ndm) to the
                                       host's interpolated pressure (needs
                                       all coupled nodes u-p; else U-only)
    ``kp``        ``-kp Kp``           pressure penalty (>0; fork default
                                       ``1e12``). Only with ``pressure``.
    ==========  =====================  ===================================

    The rotation (``-rot``/``-kr``) and material-interface
    (``-matN``/``-matT*``/``-normal``/``-corot``) knobs are deliberately
    NOT here yet — they need host-gradient emission from the node-to-
    surface resolver (the fork parser errors on ``-rot``/``-corot``
    without ``-dNdx``) and uniaxial-material-tag translation, so they land
    together in a follow-up (see ``plan_ladruno_constraints_coverage.md``).
    """
    pressure: bool = False
    kp: float | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.kp is not None and not (self.kp > 0):
            raise ValueError(
                f"EmbeddedNodeControl: kp must be > 0 if set, got {self.kp!r}."
            )
        if self.kp is not None and not self.pressure:
            raise ValueError(
                "EmbeddedNodeControl: kp (-kp) only configures the pressure "
                "tie — pass pressure=True, or drop kp."
            )

    @property
    def is_default(self) -> bool:
        return super().is_default and not self.pressure and self.kp is None

    def emit_flags(
        self, *, host_ops_tag: int | None = None,
    ) -> list[int | float | str]:
        out = super().emit_flags(host_ops_tag=host_ops_tag)
        if self.pressure:
            out += ["-pressure"]
            if self.kp is not None:
                out += ["-kp", self.kp]
        return out
