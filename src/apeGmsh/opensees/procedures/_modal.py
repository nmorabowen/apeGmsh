"""Modal / eigen / spectrum procedures.

Moved verbatim from ``apesees.apeSees`` (ADR 0114 S1-b); the mixin holds no
state and no ``__init__``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from .._internal.build import (
    BridgeError,
)
from .._internal.types import (
    TimeSeries,
)
from ..node import Node

if TYPE_CHECKING:
    from ..analysis.complex_eigen import ComplexEigenResult
    from ..analysis.eigen import EigenResult
    from ..analysis.footfall_result import FootfallResult
    from ..analysis.modal import (
        ModalHistoryResult,
        ModalPropertiesResult,
        ResponseSpectrumResult,
    )
    from ..pattern.pattern import Plain

from ._host import _ProcedureHost


class _ModalMixin(_ProcedureHost):
    def eigen(
        self,
        num_modes: int,
        *,
        solver: str = "-genBandArpack",
    ) -> "EigenResult":
        """Build + emit + run a one-shot ``eigen`` solve via the live emitter.

        Builds a :class:`BuiltModel`, drives a
        :class:`~apeGmsh.opensees.emitter.live.LiveOpsEmitter` end-to-
        end (model + nodes + elements + bcs + mass), then issues the
        single ``eigen`` call and returns an :class:`EigenResult`
        carrying the eigenvalues plus a back-reference to the live
        emitter for lazy mode-shape access.

        Unlike :meth:`analyze`, ``eigen`` does NOT require an analysis
        chain (constraints / numberer / system / test / algorithm /
        integrator / analysis): it only needs the assembled stiffness
        and mass matrices.

        **Partitioned models — serial-gather stopgap (ADR 0077 Tier 0).**
        On a partition-authored model this runs the eigensolve *serially
        on the full, gathered model* in one process (the live emitter has
        ``supports_partitions = False``): the modes are exact, but the
        whole model is assembled on one rank, so it does **not** scale the
        eigensolve. There is no *distributed* modal path yet — never run a
        bare ``eigen`` under ``OpenSeesMP`` (it solves each rank's LOCAL
        subdomain → wrong modes; ADR 0077 refuted v1). Distributed FEAST
        (ADR 0077 Tier 1) is gated on the classic-Tcl ``-feast`` unlock.

        Parameters
        ----------
        num_modes
            Number of modes to compute. Must be ``>= 1``.
        solver
            OpenSees eigen-solver flag, one of ``-genBandArpack``
            (default), ``-symmBandLapack``, ``-fullGenLapack``,
            ``-frequency``, ``-standard``. Passed through verbatim to
            ``ops.eigen(solver, num_modes)``.

        Returns
        -------
        EigenResult
            Carries ``eigenvalues`` (``λ_i = ω_i²``) plus derived
            ``omega`` / ``freq`` / ``periods`` and a
            :meth:`EigenResult.mode_shape` accessor.

        Raises
        ------
        ValueError
            If ``num_modes < 1``.
        NotImplementedError
            If the model has any registered stages — live execution
            of staged models is unsupported (Phase SSI-2.A).
        """
        if num_modes < 1:
            raise ValueError(
                f"apeSees.eigen: num_modes must be >= 1, got {num_modes}."
            )
        if self._stage_records:
            raise NotImplementedError(
                "apeSees.eigen: live execution does not support staged "
                "models (Phase SSI-2.A) "
                f"(got {len(self._stage_records)} stage(s)).  Eigen "
                "analyses are typically run against an unstaged build; "
                "either drop the stage blocks or emit Tcl/Py and run "
                "the eigen command there."
            )

        # Local imports — keep openseespy + numpy out of bridge import
        # time for Tcl/Py/H5-only users.
        from ..analysis.eigen import EigenResult
        from ..emitter.live import LiveOpsEmitter
        import numpy as np

        bm = self.build()
        self._assert_fork_if_required()
        live_emitter = LiveOpsEmitter(wipe=True)
        bm.emit(live_emitter)
        values = live_emitter.eigen(num_modes, solver=solver)
        return EigenResult(
            eigenvalues=np.asarray(values, dtype=np.float64),
            _live=live_emitter,
        )

    def modal_properties(
        self,
        num_modes: int,
        *,
        solver: str = "-genBandArpack",
        unorm: bool = False,
    ) -> "ModalPropertiesResult":
        """Build + emit + run ``eigen`` + ``modalProperties`` live.

        Like :meth:`eigen`, drives a
        :class:`~apeGmsh.opensees.emitter.live.LiveOpsEmitter` end-to-end
        and needs no analysis chain; after the eigen solve it issues
        ``modalProperties -return`` (upstream ``DomainModalProperties``)
        and wraps the returned dict in a
        :class:`~apeGmsh.opensees.analysis.modal.ModalPropertiesResult`
        carrying participation factors, modal masses, and mass ratios
        per mode and per global component.

        The properties are also stored on the OpenSees Domain, which is
        the prerequisite state for the Ladruno fork's modal-response
        commands (fork ADR 44).

        **Partitioned models — serial-gather stopgap (ADR 0077 Tier 0).**
        Runs serially on the full, gathered model (see :meth:`eigen`), so
        participation factors / effective modal mass are **correct** here.
        This is the *only* correct way to get modal properties on a
        partition-authored model today: the distributed path (ADR 0077
        Tier 1) has no participation surface — upstream
        ``modalProperties`` is MPI-blind — so a distributed run would
        return wrong effective mass. It does not scale the eigensolve
        (whole model on one rank).

        Parameters
        ----------
        num_modes
            Number of modes to compute. Must be ``>= 1``.
        solver
            OpenSees eigen-solver flag, passed through verbatim (see
            :meth:`eigen`). Use ``-fullGenLapack`` on tiny models —
            ARPACK needs ``num_modes < n_dof``.
        unorm
            Request the displacement-normalized eigenvector scaling
            (``modalProperties -unorm``).

        Raises
        ------
        ValueError
            If ``num_modes < 1``.
        NotImplementedError
            If the model has any registered stages — live execution
            of staged models is unsupported (Phase SSI-2.A).
        """
        if num_modes < 1:
            raise ValueError(
                "apeSees.modal_properties: num_modes must be >= 1, "
                f"got {num_modes}."
            )
        if self._stage_records:
            raise NotImplementedError(
                "apeSees.modal_properties: live execution does not "
                "support staged models (Phase SSI-2.A) "
                f"(got {len(self._stage_records)} stage(s)).  Either "
                "drop the stage blocks or emit Tcl/Py and run the "
                "modalProperties command there."
            )

        # Local imports — keep openseespy + numpy out of bridge import
        # time for Tcl/Py/H5-only users.
        from ..analysis.modal import ModalPropertiesResult
        from ..emitter.live import LiveOpsEmitter
        import numpy as np

        bm = self.build()
        self._assert_fork_if_required()
        live_emitter = LiveOpsEmitter(wipe=True)
        bm.emit(live_emitter)
        values = live_emitter.eigen(num_modes, solver=solver)
        properties = live_emitter.modal_properties(unorm=unorm)
        return ModalPropertiesResult(
            eigenvalues=np.asarray(values, dtype=np.float64),
            properties=properties,
            _live=live_emitter,
        )

    def footfall_walking(
        self,
        *,
        num_modes: int,
        body_weight: float,
        g: float,
        response_nodes: "int | Node | Sequence[int | Node]",
        excitation: str = "self",
        excitation_nodes: "int | Node | Sequence[int | Node] | None" = None,
        dof: int = 3,
        occupancy: str = "office",
        limit: str = "curve",
        damp: float | None = None,
        modal_damp: Sequence[float] | None = None,
        rayleigh: tuple[float, float] | None = None,
        f_max: float = 20.0,
        n_extra: int = 30,
        dt: float = 0.005,
        solver: str = "-genBandArpack",
    ) -> "FootfallResult":
        """Evaluate walking footfall vibration on a live modal basis.

        AISC Design Guide 11, 2nd ed., §7.4.1, as ADR 0109 lays it out: one
        ``eigen`` + ``modalProperties`` pair through :meth:`modal_properties`,
        the modal acceleration FRF ``A_ij(Ω) = −Ω² H_ij(Ω)`` summed in numpy
        over the D4 sweep grid, and each response node checked against
        **both** Design Guide branches — Eq 7-1 where the FRF peaks below
        9 Hz (a low-frequency floor, resonant build-up) and Eq 7-4…7-6 where
        it peaks in 9-``f_max`` Hz (a high-frequency floor, footstep impulses
        summed over every mode with ``f_n <= f_max``). The larger governs,
        and the result records which branch it was.

        Runs on **stock openseespy** — ``eigen``, ``modalProperties`` and
        ``nodeEigenvector`` are all upstream; the fork is needed only for the
        band solve the warning below points at.

        **Units are yours.** ``body_weight`` (``Q``, model force units) and
        ``g`` (model acceleration units) are required and have no defaults —
        apeGmsh cannot know whether the model is in N·m or kip·in, and a
        silently wrong ``g`` is a factor of 386 in the answer. ``Q = 168 lb
        ≈ 747 N`` is the Guide's recommended walker weight. Every
        acceleration on the result is a **fraction of g**.

        **Where to put the nodes.** The Guide walks the walker along the
        mid-length of an unobstructed path and seats the occupant at the
        maximum mode-shape ordinate; both at mid-bay is the conservative
        default. ``excitation="self"`` checks the diagonal ``(j, j)`` — the
        walker standing where the occupant sits. ``excitation="full"``
        checks every ``(i, j)`` pair and reports, per response node, the
        excitation node that produced the largest ``a_p``.

        **The eigenvector scale is asserted, not assumed** (ADR 0109 D2): the
        modal sum takes ``m̃_a = 1``, and ``partiMass_C / partiFactor_C²``
        must be 1 for every checkable mode or the evaluation refuses. The
        provenance lands on :attr:`FootfallResult.normalization`.

        Parameters
        ----------
        num_modes
            Modes to extract. The Eq 7-5 sum wants every mode with
            ``f_n <= f_max``; a basis that stops short of ``f_max`` warns.
        body_weight
            Walker weight ``Q`` in model force units. Must be ``> 0``.
        g
            Gravitational acceleration in model units. Must be ``> 0``.
        response_nodes
            Occupant node(s) — a tag, a ``Node``, or a sequence of either.
        excitation
            ``"self"`` (the diagonal only) or ``"full"`` (every pair).
        excitation_nodes
            Walker node(s) for ``excitation="full"``; defaults to
            ``response_nodes``. Refused under ``"self"``, where the
            excitation set IS the response set.
        dof
            1-based DOF the FRF and the mode shapes are read at — the
            vertical translation, 3 on a 3-D floor.
        occupancy
            Tolerance family: ``"office"`` / ``"residence"`` / ``"church"`` /
            ``"school"`` (0.5 %g), ``"shopping"`` / ``"dining"`` /
            ``"indoor_bridge"`` (1.5 %g), ``"outdoor_bridge"`` (5 %g).
        limit
            ``"curve"`` (Fig 2-1 — the flat value scaled by the ISO 2631-2
            base curve) or ``"table"`` (the flat Table 4-1 value).
        damp, modal_damp, rayleigh
            Exactly one ADR 0075 damping channel (ADR 0109 D6): a uniform
            ratio (Table 4-2 is a sum of component ratios), a per-mode list
            in absolute mode order, or ``(a0, a1)`` converted per mode as
            ``ξ_a = a0/(2ω_a) + a1·ω_a/2``.
        f_max
            Upper band edge in Hz — also the Eq 7-5 basis cut. Table 7-1's
            resonant harmonics stop at 20 Hz, so ``0 < f_max <= 20``.
        n_extra
            Linearly spaced sweep points on top of the modal frequencies and
            their ±5 % clusters (ADR 0109 D4).
        dt
            Sampling step of the Eq 7-5 acceleration history, s.
        solver
            Eigen-solver flag, passed through verbatim (see :meth:`eigen`).

        Returns
        -------
        FootfallResult
            One row per response node — see
            :class:`~apeGmsh.opensees.analysis.footfall_result.FootfallResult`.

        Warns
        -----
        UserWarning
            When the highest extracted mode is below ``f_max``: the Eq 7-5
            impulse sum is then missing modes it should carry. Raise
            ``num_modes``, or on the fork use ``eigen_feast`` over the band.

        Raises
        ------
        ValueError
            If ``num_modes < 1``, ``body_weight <= 0``, ``g <= 0``,
            ``dof < 1``, a node set is empty, ``excitation`` is not
            ``"self"`` / ``"full"``, ``limit`` is not ``"curve"`` /
            ``"table"``, ``occupancy`` is unknown, ``f_max`` is outside
            ``(0, 20]``, the damping channel is not exactly one, or the
            eigenvectors are not mass-normalised.
        NotImplementedError
            If the model has any registered stages.
        """
        # Local imports — keep numpy + the kernel out of bridge import
        # time for Tcl/Py/H5-only users.
        import warnings as _warnings

        import numpy as np

        from ..analysis.footfall import (
            dominant_frequency,
            tolerance_limit,
            walking_high_frequency,
            walking_low_frequency,
        )
        from ..analysis.footfall_frf import frf_matrix, grid_for
        from ..analysis.footfall_result import FootfallModes, FootfallResult
        from ..analysis.modal import _damping_channel_args

        context = "apeSees.footfall_walking"
        self._modal_prereqs_and_guards(num_modes, context=context)
        if body_weight <= 0.0:
            raise ValueError(
                f"{context}: body_weight is the walker's weight Q in model "
                f"force units and must be > 0, got {body_weight}."
            )
        if g <= 0.0:
            raise ValueError(
                f"{context}: g is gravitational acceleration in model units "
                f"and must be > 0, got {g} (386.1 in/s^2, 9.81 m/s^2)."
            )
        if dof < 1:
            raise ValueError(f"{context}: dof is 1-based, got {dof}.")
        excitations = ("self", "full")
        if excitation not in excitations:
            raise ValueError(
                f"{context}: excitation must be one of {excitations}, got "
                f"{excitation!r}."
            )
        limits = ("curve", "table")
        if limit not in limits:
            raise ValueError(
                f"{context}: limit must be one of {limits}, got {limit!r}."
            )
        if not (0.0 < f_max <= 20.0):
            raise ValueError(
                f"{context}: f_max must be in (0, 20] Hz — Table 7-1's "
                f"resonant harmonics stop at 20 Hz — got {f_max}."
            )
        # Probe the occupancy/kind pair now rather than after the solve.
        tolerance_limit(occupancy, 8.0, kind=limit)  # type: ignore[arg-type]
        # The same exactly-one-of rule the FRF matrix applies (ADR 0075),
        # run up front so a bad channel does not cost an eigen solve.
        _damping_channel_args(
            damp=damp, rayleigh=rayleigh, modal_damp=modal_damp,
            context=context,
        )

        resp_tags = self._footfall_tags(
            response_nodes, "response_nodes", context,
        )
        if excitation == "self":
            if excitation_nodes is not None:
                raise ValueError(
                    f"{context}: excitation='self' evaluates the diagonal "
                    "(j, j), so the excitation set IS response_nodes — pass "
                    "excitation='full' to drive a different node set."
                )
            exc_tags = resp_tags
        elif excitation_nodes is None:
            exc_tags = resp_tags
        else:
            exc_tags = self._footfall_tags(
                excitation_nodes, "excitation_nodes", context,
            )

        props = self.modal_properties(num_modes, solver=solver)
        modal_f = np.asarray(props.freq, dtype=np.float64)
        if float(np.max(modal_f)) < f_max:
            _warnings.warn(
                f"{context}: the highest extracted mode is "
                f"{float(np.max(modal_f)):.3f} Hz, below f_max={f_max} Hz — "
                "the Eq 7-5 impulse sum is missing modes it should carry. "
                "Raise num_modes, or on the fork use "
                f"eigen_feast(0.0, {f_max + 2.0}) to get the band exactly.",
                UserWarning,
                stacklevel=2,
            )
        f_min = max(float(np.min(modal_f)) - 1.0, 0.1)
        freq = grid_for(props, f_min, f_max, n_extra)
        matrix = frf_matrix(
            props, exc_nodes=exc_tags, resp_nodes=resp_tags, dof=dof,
            freq=freq, damp=damp, modal_damp=modal_damp, rayleigh=rayleigh,
            unorm=False,
        )
        # Read the mode-shape columns while this domain is still the live
        # one — everything after is pure numpy (the ADR 0075 staleness
        # contract: a later eigen/sweep call wipes what ``props`` reads).
        n_modes = int(matrix.modal_freq.size)
        phi = {
            tag: np.asarray(
                [
                    float(props.mode_shape(tag, mode + 1)[dof - 1])
                    for mode in range(n_modes)
                ],
                dtype=np.float64,
            )
            for tag in dict.fromkeys((*exc_tags, *resp_tags))
        }

        modal_damping = matrix.modal_damping
        basis = matrix.modal_freq <= f_max
        hf_band = (freq >= 9.0) & (freq <= f_max)
        n_resp = len(resp_tags)
        nan = float("nan")
        out_f_dom = np.full(n_resp, nan)
        out_f_dom_hf = np.full(n_resp, nan)
        out_frf_max = np.full(n_resp, nan)
        out_a_lf = np.full(n_resp, nan)
        out_a_hf = np.full(n_resp, nan)
        out_a_p = np.full(n_resp, nan)
        out_limit = np.full(n_resp, nan)
        out_ratio = np.full(n_resp, nan)
        out_regime = np.empty(n_resp, dtype=object)
        out_exc = np.zeros(n_resp, dtype=np.int64)

        for row, j_tag in enumerate(resp_tags):
            candidates = (j_tag,) if excitation == "self" else exc_tags
            best: "tuple[float, float, float, float, float, float] | None" = None
            best_exc = j_tag
            best_regime = "none"
            for i_tag in candidates:
                mag = matrix.magnitude(i_tag, j_tag)

                f_lf = dominant_frequency(freq, mag, 9.0)
                a_lf = nan
                if np.isfinite(f_lf):
                    beta_dom = float(modal_damping[
                        int(np.argmin(np.abs(matrix.modal_freq - f_lf)))
                    ])
                    a_lf = walking_low_frequency(
                        float(mag[int(np.argmin(np.abs(freq - f_lf)))]),
                        f_lf, beta_dom, body_weight,
                    ) / g

                f_hf = nan
                a_hf = nan
                if bool(np.any(hf_band)):
                    f_hf = dominant_frequency(
                        freq[hf_band], mag[hf_band], f_max,
                    )
                if np.isfinite(f_hf) and bool(np.any(basis)):
                    a_espa = walking_high_frequency(
                        matrix.modal_freq[basis], phi[i_tag][basis],
                        phi[j_tag][basis], f_hf, modal_damping[basis],
                        body_weight, dt,
                    )[0]
                    a_hf = a_espa / g

                if np.isfinite(a_lf) and np.isfinite(a_hf):
                    regime = "both"
                    a_p = max(a_lf, a_hf)
                    f_gov = f_lf if a_lf >= a_hf else f_hf
                elif np.isfinite(a_lf):
                    regime, a_p, f_gov = "low", a_lf, f_lf
                elif np.isfinite(a_hf):
                    regime, a_p, f_gov = "high", a_hf, f_hf
                else:
                    regime, a_p, f_gov = "none", nan, nan

                # Take the candidate unless the incumbent is finite AND
                # at least as large — a NaN incumbent must never win
                # (R-B finding 5: ``a_p > nan`` is False and would pin
                # the first excitation node forever).
                if best is not None and np.isfinite(best[0]) and not (
                    np.isfinite(a_p) and a_p > best[0]
                ):
                    continue
                frf_at = (
                    float(mag[int(np.argmin(np.abs(freq - f_gov)))])
                    if np.isfinite(f_gov) else nan
                )
                best = (a_p, a_lf, a_hf, f_gov, f_hf, frf_at)
                best_exc = i_tag
                best_regime = regime

            assert best is not None  # every node has >= 1 candidate
            out_a_p[row] = best[0]
            out_a_lf[row] = best[1]
            out_a_hf[row] = best[2]
            out_f_dom[row] = best[3]
            out_f_dom_hf[row] = best[4]
            out_frf_max[row] = best[5]
            out_regime[row] = best_regime
            out_exc[row] = best_exc
            if np.isfinite(best[3]):
                limit_value = tolerance_limit(
                    occupancy, best[3], kind=limit,  # type: ignore[arg-type]
                )
                out_limit[row] = limit_value
                out_ratio[row] = best[0] / limit_value

        return FootfallResult(
            nodes=resp_tags,
            f_dom=out_f_dom,
            frf_max=out_frf_max,
            a_p_lf=out_a_lf,
            a_espa_hf=out_a_hf,
            a_p=out_a_p,
            regime=out_regime,
            exc_node=out_exc,
            limit=out_limit,
            ratio=out_ratio,
            occupancy=occupancy,
            limit_kind=limit,
            g=float(g),
            body_weight=float(body_weight),
            normalization=matrix.normalization,
            freq=freq,
            modes=FootfallModes(f_n=matrix.modal_freq, beta=modal_damping),
            dof=int(dof),
            dt=float(dt),
            f_max=float(f_max),
            _matrix=matrix,
            _phi=phi,
            _f_dom_hf=out_f_dom_hf,
        )

    def _footfall_tags(
        self,
        nodes: "int | Node | Sequence[int | Node]",
        name: str,
        context: str,
    ) -> tuple[int, ...]:
        """Resolve a footfall node set — one tag / ``Node``, or a sequence
        of either — to unique tags in the order given."""
        from ..analysis.modal import _node_tag

        items: "Sequence[int | Node]"
        if isinstance(nodes, (int, Node)):
            items = (nodes,)
        else:
            items = tuple(nodes)
        if not items:
            raise ValueError(
                f"{context}: {name} must carry at least one node."
            )
        seen: dict[int, None] = {}
        for item in items:
            seen.setdefault(_node_tag(item), None)
        return tuple(seen)

    def eigen_feast(
        self,
        f_min: float,
        f_max: float,
        *,
        certify: bool = False,
    ) -> "EigenResult":
        """Band-targeted FEAST eigensolve via the live emitter.

        **Fork-only** (Ladruno ADR-43): ``eigen -feast fmin fmax``
        returns **all** modes whose natural frequency lies in
        ``[f_min, f_max]`` Hz — the mode count is an output
        (``len(result.eigenvalues)``), not an input, which is why this
        is a separate method and not an :meth:`eigen` solver flag.

        ``certify=True`` adds the fork's Sturm/inertia completeness
        certificate: the band content is independently counted via
        LDLᵀ inertia at the two band edges and the solve REFUSES on a
        mismatch with FEAST's count.

        Parameters
        ----------
        f_min, f_max
            Frequency band in Hz; needs ``0 <= f_min < f_max``.
        certify
            Emit ``-certify`` (the completeness certificate).

        Returns
        -------
        EigenResult
            The standard eigen result (possibly zero modes if the band
            is empty) with lazy ``mode_shape`` access.
        """
        if not (0.0 <= f_min < f_max):
            raise ValueError(
                "apeSees.eigen_feast: need 0 <= f_min < f_max, got "
                f"f_min={f_min}, f_max={f_max}."
            )
        if self._stage_records:
            raise NotImplementedError(
                "apeSees.eigen_feast: live execution does not support "
                "staged models (Phase SSI-2.A) "
                f"(got {len(self._stage_records)} stage(s))."
            )
        # The stock ``ops.eigen`` symbol exists on every build, so the
        # missing-attribute gate cannot fire — pre-check the fork probe
        # for the friendly message (a pre-ADR-43 fork build still fails
        # with the OpenSees '-feast' parse error).
        if not self.capabilities().has_fork:
            raise RuntimeError(
                "apeSees.eigen_feast requires the Ladruno fork build of "
                "OpenSees (fork ADR-43 band-targeted FEAST eigensolver); "
                "the in-process openseespy is not the fork."
            )

        from ..analysis.eigen import EigenResult
        from ..emitter.live import LiveOpsEmitter
        import numpy as np

        bm = self.build()
        self._assert_fork_if_required()
        live_emitter = LiveOpsEmitter(wipe=True)
        bm.emit(live_emitter)
        values = live_emitter.eigen_feast(
            float(f_min), float(f_max), certify=certify,
        )
        return EigenResult(
            eigenvalues=np.asarray(values, dtype=np.float64),
            _live=live_emitter,
        )

    def complex_eigen(
        self,
        num_modes: int,
        *,
        solver: str = "-genBandArpack",
        tol: float | None = None,
        closed_form: bool = False,
    ) -> "ComplexEigenResult":
        """Complex / state-space modal analysis via the live emitter.

        **Fork-only** (Ladruno ADR-46, ``complexEigen``): true per-mode
        damping ratios ζ_k, damped frequencies ω_d,k, and phased mode
        shapes for **non-classically damped** models (localized
        dashpots, bearings, radiation damping).  Builds + emits a fresh
        live domain, runs the real ``eigen`` (the projection basis),
        then ``complexEigen`` and parses the flat 7-per-mode return
        into a :class:`ComplexEigenResult`.

        The default route projects the model's **actual** M and C
        (element ``getDamp()``/``getMass()`` + nodal mass/``alphaM``) —
        exactly the C a transient analysis feels.  ``closed_form=True``
        uses the fast global-Rayleigh diagonal closed form instead
        (refuses ``betaKinit``/``betaKcomm``; blind to scoped
        Rayleigh).

        Contract traps (fork guide): damping that does not flow through
        ``getDamp()`` is invisible (``modalDamping``, HHT-α numerical
        damping, elements whose ``-doRayleigh`` defaults OFF — the
        ``Truss``/``zeroLength`` families); the projection spans only
        the retained ``num_modes`` real modes; complex mode shapes are
        recorded via Node-recorder ``raw=("complexEigenRe<k>",)`` /
        ``Im<k>`` tokens, not carried on this result.

        Parameters
        ----------
        num_modes
            Real modes to extract as the projection basis (retain
            enough to cover the band of interest;
            ``-fullGenLapack`` on tiny models).
        tol
            Optional residual tolerance (fork default 1e-8).
        closed_form
            Use the closed-form Rayleigh route (Route A).
        """
        context = "apeSees.complex_eigen"
        self._modal_prereqs_and_guards(num_modes, context=context)
        # Fork guide trap #4: the eigenvector distribution to
        # MP-constrained slave DOFs needs a distributing constraint
        # handler — ``constraints Plain`` + MP constraints yields wrong
        # complex shapes. The bridge-driven eigen path defaults to
        # Transformation when no handler is declared; warn when the
        # user declared Plain.
        from ..analysis.constraint_handler import Plain as _PlainHandler

        if any(isinstance(p, _PlainHandler) for p in self._primitives):
            import warnings

            warnings.warn(
                f"{context}: 'constraints Plain' is declared — on a "
                "model with MP constraints (rigid links, equalDOF, "
                "embedded, ...) complexEigen mode shapes need a "
                "distributing handler (Transformation).",
                UserWarning,
                stacklevel=2,
            )

        from ..analysis.complex_eigen import ComplexEigenResult
        from ..emitter.live import LiveOpsEmitter

        args: list[int | float | str] = []
        if tol is not None:
            if tol <= 0.0:
                raise ValueError(
                    f"{context}: tol must be > 0, got {tol}."
                )
            args.extend(("-tol", float(tol)))
        if closed_form:
            args.append("-closedForm")

        bm = self.build()
        self._assert_fork_if_required()
        live_emitter = LiveOpsEmitter(wipe=True)
        bm.emit(live_emitter)
        live_emitter.eigen(num_modes, solver=solver)
        values = live_emitter.complex_eigen(*args)
        return ComplexEigenResult.from_flat(values)

    def _modal_prereqs_and_guards(
        self, num_modes: int, *, context: str,
    ) -> None:
        """Shared validation for the ADR 0075 modal-response drivers."""
        if num_modes < 1:
            raise ValueError(
                f"{context}: num_modes must be >= 1, got {num_modes}."
            )
        if self._stage_records:
            raise NotImplementedError(
                f"{context}: live execution does not support staged "
                "models (Phase SSI-2.A) "
                f"(got {len(self._stage_records)} stage(s)).  Either "
                "drop the stage blocks or emit Tcl/Py and run the "
                "command there."
            )

    def modal_response_history(
        self,
        *,
        dt: float,
        n_steps: int,
        num_modes: int,
        base_accel: "TimeSeries | str | None" = None,
        direction: int | None = None,
        load: "Plain | str | None" = None,
        series: "TimeSeries | str | None" = None,
        damp: float | None = None,
        rayleigh: tuple[float, float] | None = None,
        modal_damp: Sequence[float] | None = None,
        modes: Sequence[int] | None = None,
        t0: float = 0.0,
        solver: str = "-genBandArpack",
    ) -> "ModalHistoryResult":
        """Run the fork's exact modal-superposition transient live.

        **Fork-only** (Ladruno ADR-44 P1a, ``modalResponseHistory``).
        Builds + emits a fresh live domain, issues ``eigen`` +
        ``modalProperties``, then advances each retained mode by the
        closed-form piecewise-linear recurrence — no iteration, no
        factorization.  One domain step is **committed per station**
        (``t0 … t0 + n_steps·dt``), so every recorder declared on the
        model captures the history exactly as in a direct run.

        Linear models only — superposition is invalid under any
        material or geometric nonlinearity (use ``analyze`` then).

        Parameters
        ----------
        dt, n_steps
            Time step and station count (``n_steps + 1`` commits).
        num_modes
            Modes to extract for the superposition basis (retain
            enough to cover the band of interest).
        base_accel, direction
            Ground-acceleration channel: a registered
            ``ops.timeSeries.*`` handle (or name) sampled at the
            stations, plus the global excitation direction (1-based).
            Response is **relative** to the moving base.  Make the
            record extend at least one sample past ``t0 + n_steps·dt``.
        load, series
            Nodal-force channel ``P(t) = s(t)·P``: an
            ``ops.pattern.Plain`` handle (or name) whose plain nodal
            loads give the reference shape ``P`` (the pattern's own
            timeSeries is IGNORED by the fork), and the scalar
            ``s(t)`` timeSeries.  Response is **absolute**.  Mutually
            exclusive with the base-acceleration channel.
        damp, rayleigh, modal_damp
            Exactly one damping channel (ADR 0075): uniform ratio /
            Rayleigh ``(a0, a1)`` / per-mode ratios.
        modes
            Optional 1-based subset of the extracted modes.
        t0
            Start time (base accel sampled at ``t0 + k·dt``).
        solver
            Eigen-solver flag (``-fullGenLapack`` on tiny models).
        """
        from ..analysis.modal import (
            ModalHistoryResult,
            _damping_channel_args,
        )
        from ..emitter.live import LiveOpsEmitter
        from ..pattern.pattern import Plain as _Plain
        import numpy as np

        context = "apeSees.modal_response_history"
        self._modal_prereqs_and_guards(num_modes, context=context)
        if dt <= 0.0 or n_steps < 1:
            raise ValueError(
                f"{context}: dt must be > 0 and n_steps >= 1, got "
                f"dt={dt}, n_steps={n_steps}."
            )
        damping_args = _damping_channel_args(
            damp=damp, rayleigh=rayleigh, modal_damp=modal_damp,
            context=context,
        )
        excitation = self._resolve_modal_excitation(
            base_accel=base_accel, direction=direction,
            load=load, series=series, context=context,
            plain_cls=_Plain, series_required=True,
        )

        args: list[int | float | str] = [
            "-dt", float(dt), "-nsteps", int(n_steps),
        ]
        if t0 != 0.0:
            args.extend(("-t0", float(t0)))
        args.extend(excitation)
        args.extend(damping_args)
        if modes is not None:
            mode_list = [int(m) for m in modes]
            if not mode_list or any(m < 1 for m in mode_list):
                raise ValueError(
                    f"{context}: modes must be 1-based mode numbers, "
                    f"got {modes!r}."
                )
            args.extend(("-modes", *mode_list))

        bm = self.build()
        self._assert_fork_if_required()
        live_emitter = LiveOpsEmitter(wipe=True)
        bm.emit(live_emitter)
        values = live_emitter.eigen(num_modes, solver=solver)
        live_emitter.modal_properties()
        live_emitter.modal_response_history(*args)
        return ModalHistoryResult(
            eigenvalues=np.asarray(values, dtype=np.float64),
            dt=float(dt),
            n_steps=int(n_steps),
            _live=live_emitter,
        )

    def response_spectrum_analysis(
        self,
        direction: int,
        *,
        periods: Sequence[float],
        accels: Sequence[float],
        combine: str,
        num_modes: int,
        damp: float | None = None,
        modal_damp: Sequence[float] | None = None,
        solver: str = "-genBandArpack",
    ) -> "ResponseSpectrumResult":
        """Run a response-spectrum analysis with native combination.

        **Fork-only** (Ladruno ADR-44 P1b): the ``-combine`` stage on
        ``responseSpectrumAnalysis``.  Builds + emits a fresh live
        domain, issues ``eigen`` + ``modalProperties``, computes the
        per-mode modal displacements against the ``(periods, accels)``
        design spectrum, and commits the **combined** nodal design
        displacement field, read back via
        :meth:`ResponseSpectrumResult.node_disp`.

        Combination is per-quantity and nonlinear — do NOT derive
        combined element forces / drifts from the combined
        displacements (combine those quantities' own per-mode peaks
        instead).

        Parameters
        ----------
        direction
            Global excitation direction (1-based).
        periods, accels
            The design spectrum ``Sa(Tn)`` as parallel lists.
            ``periods`` must be non-negative and strictly increasing;
            a leading ``T = 0`` PGA anchor is legal (the fork clamps
            ``T <= Tn[0]`` to ``Sa[0]``).
        combine
            ``"SRSS"`` | ``"CQC"`` | ``"ABS"`` | ``"TenPercent"``.
            CQC and TenPercent weight closely-spaced modes; CQC
            requires a damping channel.
        num_modes
            Modes to extract; the combination spans all of them
            (``-combine`` and ``-mode`` are mutually exclusive — the
            bridge never emits ``-mode``).
        damp, modal_damp
            Optional damping channel (uniform ratio or per-mode).
            Required for ``CQC``.
        """
        from ..analysis.modal import (
            ResponseSpectrumResult,
            _damping_channel_args,
        )
        from ..emitter.live import LiveOpsEmitter
        import numpy as np

        context = "apeSees.response_spectrum_analysis"
        self._modal_prereqs_and_guards(num_modes, context=context)
        if int(direction) < 1:
            raise ValueError(
                f"{context}: direction is 1-based, got {direction}."
            )
        rules = ("SRSS", "CQC", "ABS", "TenPercent")
        if combine not in rules:
            raise ValueError(
                f"{context}: combine must be one of {rules}, got "
                f"{combine!r}."
            )
        tn = [float(t) for t in periods]
        sa = [float(a) for a in accels]
        if len(tn) != len(sa) or not tn:
            raise ValueError(
                f"{context}: periods and accels must be equal-length "
                f"non-empty lists, got {len(tn)} periods / {len(sa)} "
                "accels."
            )
        if any(t < 0.0 for t in tn) or any(
            b <= a for a, b in zip(tn, tn[1:])
        ):
            raise ValueError(
                f"{context}: periods must be non-negative and strictly "
                "increasing (the fork refuses negative Tn; a leading "
                "T=0 PGA anchor is legal)."
            )
        if combine == "CQC" and damp is None and modal_damp is None:
            raise ValueError(
                f"{context}: CQC needs a damping channel — pass damp= "
                "or modal_damp=."
            )
        damping_args: tuple[float | str, ...] = ()
        if damp is not None or modal_damp is not None:
            damping_args = _damping_channel_args(
                damp=damp, rayleigh=None, modal_damp=modal_damp,
                context=context,
            )

        args: list[int | float | str] = ["-Tn", *tn, "-Sa", *sa]
        args.extend(("-combine", combine))
        args.extend(damping_args)

        bm = self.build()
        self._assert_fork_if_required()
        live_emitter = LiveOpsEmitter(wipe=True)
        bm.emit(live_emitter)
        values = live_emitter.eigen(num_modes, solver=solver)
        live_emitter.modal_properties()
        live_emitter.response_spectrum_analysis(int(direction), *args)
        return ResponseSpectrumResult(
            eigenvalues=np.asarray(values, dtype=np.float64),
            combine=combine,
            _live=live_emitter,
        )

    def _resolve_modal_excitation(
        self,
        *,
        base_accel: "TimeSeries | str | None",
        direction: int | None,
        load: "Plain | str | None",
        series: "TimeSeries | str | None",
        context: str,
        plain_cls: type,
        series_required: bool,
    ) -> tuple[int | float | str, ...]:
        """Resolve one ADR-44 excitation channel to its flag tail.

        Exactly one of the base-acceleration channel
        (``base_accel`` + ``direction`` → ``-baseAccel $ts -dir $d``)
        or the nodal-force channel (``load`` [+ ``series``] →
        ``-load $pat [-series $ts]``) must be given.  Handles resolve
        dual-mode (object or registered name) and must be registered
        on THIS bridge; the fork refuses patterns carrying sp
        constraints or moment tensors, so the bridge pre-checks for
        the friendlier error.
        """
        has_base = base_accel is not None
        has_load = load is not None
        if has_base == has_load:
            raise ValueError(
                f"{context}: supply exactly one excitation channel — "
                "base_accel= (+ direction=) OR load= "
                + ("(+ series=)" if series_required else "")
                + f"; got base_accel={base_accel!r}, load={load!r}."
            )
        if has_base:
            assert base_accel is not None  # has_base == (base_accel is not None)
            if direction is None or int(direction) < 1:
                raise ValueError(
                    f"{context}: the base-acceleration channel needs "
                    f"direction= (1-based), got {direction!r}."
                )
            if series is not None:
                raise ValueError(
                    f"{context}: series= belongs to the load= channel."
                )
            ts = self._resolve(base_accel, base=TimeSeries)
            # _resolve only kind-checks name-string refs; object
            # handles pass through untouched — and per-kind 1-based tag
            # counters mean a wrong-kind handle's tag numerically
            # collides with a real primitive of the expected kind
            # (adversarial-review hardening).
            if not isinstance(ts, TimeSeries):
                raise TypeError(
                    f"{context}: base_accel= needs an ops.timeSeries.* "
                    f"handle (or registered name), got "
                    f"{type(ts).__name__}."
                )
            ts_tag = self.tag_for(ts)
            if ts_tag is None:
                raise BridgeError(
                    f"{context}: the base_accel timeSeries is not "
                    "registered on this bridge — create it via "
                    "ops.timeSeries.<Type>(...)."
                )
            return ("-baseAccel", int(ts_tag), "-dir", int(direction))

        if direction is not None:
            raise ValueError(
                f"{context}: direction= belongs to the base_accel "
                "channel."
            )
        assert load is not None  # XOR check above guarantees it
        pat_tag = self._resolve_load_pattern_tag(
            load, context=context, plain_cls=plain_cls,
        )
        tail: list[int | float | str] = ["-load", int(pat_tag)]
        if series_required:
            if series is None:
                raise ValueError(
                    f"{context}: the load= channel needs series= "
                    "(the scalar s(t) timeSeries; the pattern's own "
                    "timeSeries is ignored by the fork)."
                )
            s = self._resolve(series, base=TimeSeries)
            if not isinstance(s, TimeSeries):
                raise TypeError(
                    f"{context}: series= needs an ops.timeSeries.* "
                    f"handle (or registered name), got "
                    f"{type(s).__name__}."
                )
            s_tag = self.tag_for(s)
            if s_tag is None:
                raise BridgeError(
                    f"{context}: the series timeSeries is not "
                    "registered on this bridge — create it via "
                    "ops.timeSeries.<Type>(...)."
                )
            tail.extend(("-series", int(s_tag)))
        elif series is not None:
            raise ValueError(
                f"{context}: series= is not accepted here (the "
                "harmonic/PSD sweeps carry their own excitation "
                "scale)."
            )
        return tuple(tail)

    def _resolve_load_pattern_tag(
        self,
        load: "Plain | str",
        *,
        context: str,
        plain_cls: type,
    ) -> int:
        """Resolve an ADR-44 ``-load`` pattern handle to its tag.

        The fork refuses ``-load`` patterns carrying anything but plain
        nodal loads; the bridge pre-checks sp constraints and
        moment-tensor sources for the friendlier error.
        """
        pat = self._resolve(load, base=plain_cls)
        # Object handles bypass _resolve's kind check — refuse a
        # wrong-kind handle before its tag collides with a real
        # pattern tag (adversarial-review hardening).
        if not isinstance(pat, plain_cls):
            raise TypeError(
                f"{context}: load= needs an ops.pattern.Plain handle "
                f"(or registered name), got {type(pat).__name__}."
            )
        pat_tag = self.tag_for(pat)
        if pat_tag is None:
            raise BridgeError(
                f"{context}: the load pattern is not registered on "
                "this bridge — create it via ops.pattern.Plain(...)."
            )
        if getattr(pat, "sps", ()):
            raise BridgeError(
                f"{context}: the fork refuses -load patterns carrying "
                "sp constraints — use a pattern with plain nodal "
                "loads only."
            )
        if getattr(pat, "moment_tensors", ()):
            raise BridgeError(
                f"{context}: the fork refuses -load patterns carrying "
                "moment-tensor sources — use a pattern with plain "
                "nodal loads only."
            )
        return int(pat_tag)
