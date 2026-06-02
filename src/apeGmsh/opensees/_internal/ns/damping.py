"""
``_DampingNS`` — backs ``ops.damping.<verb>(...)`` (ADR 0053).

Damping is a domain-level concern (sibling of ``fix`` / ``mass`` /
``region``), not part of the analysis chain.  Every verb is a declaration
recorded on the bridge and resolved at emit time — there is no ``assign``
step and no user-held object (ADR 0053).

D1 ships ``rayleigh`` (global only).  Region scoping (``on=``), modal damping
(``modal``), and the object-backed forms (``uniform`` / ``sec_stif`` /
``urd`` / ``urd_beta``) land in later slices.
"""
from __future__ import annotations

from collections.abc import Iterable

from collections.abc import Sequence

from ...analysis.rayleigh import Stiffness, rayleigh_from_ratio
from ...damping.damping import SecStif, Uniform
from ..build import (
    DampingAttachRecord,
    ModalDampingRecord,
    RayleighRecord,
)
from ._base import _BridgeNamespace


__all__ = ["_DampingNS"]


class _DampingNS(_BridgeNamespace):
    """``ops.damping.<verb>(...)`` — domain-level damping declarations."""

    def rayleigh(
        self,
        *,
        alpha_m: float | None = None,
        beta_k: float | None = None,
        beta_k_init: float = 0.0,
        beta_k_comm: float = 0.0,
        ratio: float | None = None,
        f_i: float | None = None,
        f_j: float | None = None,
        stiffness: Stiffness = "initial",
        on: str | Iterable[str] | None = None,
    ) -> None:
        """Declare Rayleigh damping → ``rayleigh`` or ``region -rayleigh``.

        Two mutually exclusive coefficient forms:

        * **raw** — supply ``alpha_m`` and/or ``beta_k`` (and optionally
          ``beta_k_init`` / ``beta_k_comm``); the four coefficients pass
          straight through to the OpenSees command.
        * **ratio** — supply ``ratio``, ``f_i``, ``f_j`` (Hz); the
          two-target Rayleigh fit computes ``alpha_m`` and a single ``β``
          placed in the slot named by ``stiffness`` (default ``initial`` =
          ``betaK0``, the nonlinear-safe choice — ADR 0053).  ``stiffness``
          is ignored by the raw form.

        ``on`` is the scope:

        * ``None`` (default) → **global** ``rayleigh αM βK βK0 βKc``.
        * a physical-group name, or a list of them → **region-scoped**: each
          name's elements get one ``region $tag -ele … -rayleigh …`` line
          (``-ele`` membership because βK is stiffness-proportional).

        Because OpenSees overwrites element Rayleigh per element (not
        additive), a global ``rayleigh`` plus a region ``on=`` over the same
        elements means the region value wins — the emit pass warns when both
        coexist (ADR 0053).

        Raises
        ------
        ValueError
            If both coefficient forms (or neither) are supplied, the ratio
            form is missing any of ``ratio`` / ``f_i`` / ``f_j``, or ``on``
            contains a non-string / empty name.
        """
        raw_given = (
            alpha_m is not None
            or beta_k is not None
            or beta_k_init != 0.0
            or beta_k_comm != 0.0
        )
        ratio_given = ratio is not None or f_i is not None or f_j is not None
        if raw_given and ratio_given:
            raise ValueError(
                "ops.damping.rayleigh: supply the raw form "
                "(alpha_m/beta_k/...) OR the ratio form (ratio/f_i/f_j), "
                "not both.",
            )
        if ratio_given:
            if ratio is None or f_i is None or f_j is None:
                raise ValueError(
                    "ops.damping.rayleigh: the ratio form needs all of "
                    f"ratio=, f_i=, f_j= (got ratio={ratio!r}, f_i={f_i!r}, "
                    f"f_j={f_j!r}).",
                )
            coeffs = rayleigh_from_ratio(
                ratio=ratio, f_i=f_i, f_j=f_j, stiffness=stiffness,
            )
        elif raw_given:
            coeffs = (
                float(alpha_m or 0.0),
                float(beta_k or 0.0),
                float(beta_k_init),
                float(beta_k_comm),
            )
        else:
            raise ValueError(
                "ops.damping.rayleigh: supply either the raw form "
                "(alpha_m/beta_k/...) or the ratio form (ratio/f_i/f_j).",
            )
        targets = _normalize_on(on)
        self._bridge._rayleigh_records.append(
            RayleighRecord(*coeffs, on=targets),
        )

    def modal(
        self,
        ratios: float | Sequence[float],
        *,
        modes: int,
        solver: str = "-genBandArpack",
    ) -> None:
        """Declare modal damping → bundled ``eigen`` + ``modalDamping``.

        Computes ``modes`` eigenmodes and assigns damping ratios to them.
        ``ratios`` is either a single float (uniform across **all** modes) or
        a sequence of exactly ``modes`` per-mode ratios.  Domain-wide — modal
        damping has no region scope in OpenSees, so there is no ``on=``.

        The bridge emits ``eigen <solver> <modes>`` immediately followed by
        ``modalDamping <f1> [..]`` driver-post (after the model is built); the
        live emitter runs the eigen solve there, exactly when the factors are
        set.  There is intentionally no ``modal_q`` — OpenSees ``modalDampingQ``
        is a verified upstream anti-damping bug (ADR 0053).

        Raises
        ------
        ValueError
            If ``modes < 1``, or a per-mode ``ratios`` sequence length does
            not equal ``modes``.
        """
        if modes < 1:
            raise ValueError(
                f"ops.damping.modal: modes must be >= 1, got {modes}.",
            )
        if isinstance(ratios, (int, float)):
            factors: tuple[float, ...] = (float(ratios),)
        else:
            factors = tuple(float(r) for r in ratios)
            if len(factors) != modes:
                raise ValueError(
                    "ops.damping.modal: a per-mode ratios sequence must have "
                    f"exactly modes={modes} entries, got {len(factors)}.",
                )
        self._bridge._modal_damping_records.append(
            ModalDampingRecord(factors=factors, modes=modes, solver=solver),
        )

    def uniform(
        self,
        *,
        ratio: float,
        freq_lower: float,
        freq_upper: float,
        on: str | Iterable[str],
        activate_time: float | None = None,
        deactivate_time: float | None = None,
        name: str | None = None,
    ) -> Uniform:
        """Declare a ``damping Uniform`` object and attach it to ``on``.

        Constant damping ratio ``ratio`` (the **physical** ζ — OpenSees
        applies the internal factor of two) across the band
        ``[freq_lower, freq_upper]`` (Hz).  Attaches via
        ``region $tag -ele … -damp $tag`` for each physical group in ``on``
        (required — a damping object with no target is meaningless).

        ``activate_time`` / ``deactivate_time`` window when the object
        dissipates (e.g. off during gravity staging — ADR 0053).  Returns the
        registered :class:`~apeGmsh.opensees.damping.damping.Uniform` handle.
        """
        prim = Uniform(
            zeta=ratio, freq1=freq_lower, freq2=freq_upper,
            activate_time=activate_time, deactivate_time=deactivate_time,
        )
        self._register_damping(prim, on=on, name=name)
        return prim

    def sec_stif(
        self,
        *,
        beta: float,
        on: str | Iterable[str],
        activate_time: float | None = None,
        deactivate_time: float | None = None,
        name: str | None = None,
    ) -> SecStif:
        """Declare a ``damping SecStif`` object and attach it to ``on``.

        Committed (secant) stiffness-proportional damping, coefficient
        ``beta``.  ``on`` (required) and the time-window kwargs behave as in
        :meth:`uniform`.  Returns the registered ``SecStif`` handle.
        """
        prim = SecStif(
            beta=beta,
            activate_time=activate_time, deactivate_time=deactivate_time,
        )
        self._register_damping(prim, on=on, name=name)
        return prim

    def _register_damping(
        self,
        prim: "Uniform | SecStif",
        *,
        on: str | Iterable[str],
        name: str | None,
    ) -> None:
        """Register the object primitive and record its region attachment."""
        targets = _normalize_on(on)
        if not targets:
            raise ValueError(
                f"ops.damping.{type(prim).__name__.lower()}: on= is required "
                "— a damping object with no target attaches to nothing "
                "(there is no global -damp).",
            )
        self._bridge._register(prim, name=name)
        self._bridge._damping_attach_records.append(
            DampingAttachRecord(prim=prim, on=targets),
        )


def _normalize_on(on: "str | Iterable[str] | None") -> tuple[str, ...]:
    """Normalize ``on=`` into a tuple of physical-group names.

    ``None`` → ``()`` (global).  A single name → ``(name,)``.  An iterable
    of names → that tuple.  Every name must be a non-empty string.
    """
    if on is None:
        return ()
    names = (on,) if isinstance(on, str) else tuple(on)
    for name in names:
        if not isinstance(name, str) or not name:
            raise ValueError(
                "ops.damping.rayleigh: on= must be a non-empty physical-group "
                f"name or a list of them (got {on!r}).",
            )
    return names
