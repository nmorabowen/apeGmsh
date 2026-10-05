"""Frequency-domain (modal FRF) procedures.

Moved verbatim from ``apesees.apeSees`` (ADR 0114 S1-b); the mixin holds no
state and no ``__init__``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

from .._internal.build import (
    BridgeError,
)
from .._internal.types import (
    TimeSeries,
)
from ..node import Node

if TYPE_CHECKING:

    # FEMData is the only mesh symbol the bridge depends on (P3, P9).
    # Imported under TYPE_CHECKING so that constructing apeSees does
    # not transitively import gmsh during static analysis.
    # Use the fully-qualified module path to disambiguate from the
    # similarly-named submodule ``apeGmsh.mesh.FEMData`` under mypy.

    from ..analysis.modal import (
        FrequencyResponseResult,
        RandomResponseResult,
        SteadyStateResult,
    )
    from ..pattern.pattern import Plain

from ._host import _ProcedureHost


class _FrfMixin(_ProcedureHost):
    def frequency_response(
        self,
        *,
        f_min: float,
        f_max: float,
        n_freq: int,
        node: "int | Node",
        dof: int,
        num_modes: int,
        grid: str = "lin",
        base_accel_dir: int | None = None,
        load: "Plain | str | None" = None,
        amp: float = 1.0,
        damp: float | None = None,
        rayleigh: tuple[float, float] | None = None,
        modal_damp: Sequence[float] | None = None,
        resp: str = "disp",
        modes: Sequence[int] | None = None,
        out: str | None = None,
        solver: str = "-genBandArpack",
    ) -> "FrequencyResponseResult":
        """Compute the complex modal FRF of one response DOF, live.

        **Fork-only** (Ladruno ADR-44 P2, ``frequencyResponse``): for a
        harmonic excitation ``amp·e^{iΩt}`` the steady response is a
        dense post-processor on the mode basis — no time stepping.
        Returns a :class:`FrequencyResponseResult` (frequencies in Hz
        + complex FRF).

        Excitation: ``base_accel_dir=`` for uniform harmonic base
        acceleration along a global direction (no timeSeries — the
        sweep is per ``amp``; **relative** response) XOR ``load=`` for
        harmonic nodal forces ``amp·P·e^{iΩt}`` from a plain-nodal-load
        pattern (**absolute** response).

        ``grid``: ``"lin"`` / ``"log"`` / ``"biased"`` — biased adds a
        ±5 % cluster around each in-band modal frequency so sharp
        low-damping peaks are not stepped over.

        ``resp``: ``"disp"`` | ``"vel"`` (``iΩ·û``) | ``"accel"``
        (``−Ω²·û``).  ``out=`` additionally writes the table to an
        ASCII file.
        """
        rows = self._run_modal_sweep(
            command="frequency_response",
            context="apeSees.frequency_response",
            f_min=f_min, f_max=f_max, n_freq=n_freq,
            node=node, dof=dof, num_modes=num_modes, grid=grid,
            base_accel_dir=base_accel_dir, load=load, amp=amp,
            damp=damp, rayleigh=rayleigh, modal_damp=modal_damp,
            resp=resp, modes=modes, out=out, solver=solver,
        )
        import numpy as np

        from ..analysis.modal import FrequencyResponseResult

        table = np.asarray(rows, dtype=np.float64)
        return FrequencyResponseResult(
            freq=table[:, 0],
            response=table[:, 1] + 1j * table[:, 2],
        )

    def steady_state_dynamics(
        self,
        *,
        f_min: float,
        f_max: float,
        n_freq: int,
        node: "int | Node",
        dof: int,
        num_modes: int,
        grid: str = "lin",
        base_accel_dir: int | None = None,
        load: "Plain | str | None" = None,
        amp: float = 1.0,
        damp: float | None = None,
        rayleigh: tuple[float, float] | None = None,
        modal_damp: Sequence[float] | None = None,
        resp: str = "disp",
        modes: Sequence[int] | None = None,
        out: str | None = None,
        solver: str = "-genBandArpack",
    ) -> "SteadyStateResult":
        """Steady-state harmonic response amplitude ``|response|`` per
        sweep frequency — the magnitude companion of
        :meth:`frequency_response` (same flags, fork ADR-44 P2)."""
        rows = self._run_modal_sweep(
            command="steady_state_dynamics",
            context="apeSees.steady_state_dynamics",
            f_min=f_min, f_max=f_max, n_freq=n_freq,
            node=node, dof=dof, num_modes=num_modes, grid=grid,
            base_accel_dir=base_accel_dir, load=load, amp=amp,
            damp=damp, rayleigh=rayleigh, modal_damp=modal_damp,
            resp=resp, modes=modes, out=out, solver=solver,
        )
        import numpy as np

        from ..analysis.modal import SteadyStateResult

        table = np.asarray(rows, dtype=np.float64)
        return SteadyStateResult(
            freq=table[:, 0], magnitude=table[:, 1],
        )

    def random_response(
        self,
        *,
        f_min: float,
        f_max: float,
        n_freq: int,
        node: "int | Node",
        dof: int,
        num_modes: int,
        input_psd: "TimeSeries | str",
        grid: str = "biased",
        base_accel_dir: int | None = None,
        load: "Plain | str | None" = None,
        damp: float | None = None,
        rayleigh: tuple[float, float] | None = None,
        modal_damp: Sequence[float] | None = None,
        resp: str = "disp",
        modes: Sequence[int] | None = None,
        stats: bool = False,
        duration: float | None = None,
        out: str | None = None,
        solver: str = "-genBandArpack",
    ) -> "RandomResponseResult":
        """Stationary random response RMS on the modal FRF, live.

        **Fork-only** (Ladruno ADR-44 P3, ``randomResponse``):
        ``input_psd`` is a **one-sided PSD G(f) in Hz** ((excitation)²/
        Hz), supplied as a registered timeSeries sampled at ``f`` in Hz
        (``Path`` with f→G breakpoints, ``Constant`` for white noise).
        With ``base_accel_dir=`` it is the base-acceleration PSD; with
        ``load=`` the PSD of the scalar multiplying the pattern's
        nodal-load shape (fully correlated).

        ``grid`` defaults to ``"biased"`` — the RMS is a band integral
        and a linear grid mis-integrates sharp resonances (fork guide
        P3).  The band ``[f_min, f_max]`` must cover the input's
        support and every resonance carrying response power; the fork
        refuses zero-damped in-band modes and a rigid-body mode with
        ``f_min = 0``.

        ``stats=`` adds ``ν₀`` (mean zero-upcrossing rate, Hz) and the
        spectral moments ``m0`` / ``m2``; ``duration=`` additionally
        appends the Davenport expected peak over that exposure.
        """
        from ..analysis.modal import RandomResponseResult

        context = "apeSees.random_response"
        ts = self._resolve(input_psd, base=TimeSeries)
        if not isinstance(ts, TimeSeries):
            raise TypeError(
                f"{context}: input_psd= needs an ops.timeSeries.* "
                f"handle (or registered name), got {type(ts).__name__}."
            )
        psd_tag = self.tag_for(ts)
        if psd_tag is None:
            raise BridgeError(
                f"{context}: the input_psd timeSeries is not "
                "registered on this bridge — create it via "
                "ops.timeSeries.<Type>(...)."
            )
        if duration is not None and duration <= 0.0:
            raise ValueError(
                f"{context}: duration must be > 0, got {duration}."
            )
        extra: list[int | float | str] = ["-inputPSD", int(psd_tag)]
        if stats or duration is not None:
            extra.append("-stats")
        if duration is not None:
            extra.extend(("-duration", float(duration)))

        raw = self._run_modal_sweep(
            command="random_response",
            context=context,
            f_min=f_min, f_max=f_max, n_freq=n_freq,
            node=node, dof=dof, num_modes=num_modes, grid=grid,
            base_accel_dir=base_accel_dir, load=load, amp=None,
            damp=damp, rayleigh=rayleigh, modal_damp=modal_damp,
            resp=resp, modes=modes, out=out, solver=solver,
            extra_args=tuple(extra),
        )
        if isinstance(raw, (int, float)):
            return RandomResponseResult(rms=float(raw))
        values = [float(v) for v in raw]
        peak = values[4] if len(values) > 4 else None
        return RandomResponseResult(
            rms=values[0], nu0=values[1], m0=values[2], m2=values[3],
            peak=peak,
        )

    def _run_modal_sweep(
        self,
        *,
        command: str,
        context: str,
        f_min: float,
        f_max: float,
        n_freq: int,
        node: "int | Node",
        dof: int,
        num_modes: int,
        grid: str,
        base_accel_dir: int | None,
        load: "Plain | str | None",
        amp: float | None,
        damp: float | None,
        rayleigh: tuple[float, float] | None,
        modal_damp: Sequence[float] | None,
        resp: str,
        modes: Sequence[int] | None,
        out: str | None,
        solver: str,
        extra_args: tuple[int | float | str, ...] = (),
    ) -> Any:
        """Validate + marshal one ADR-44 frequency-domain sweep and run
        it on a fresh live domain (eigen → modalProperties → command).
        Returns the live emitter's raw return."""
        from ..analysis.modal import _damping_channel_args, _node_tag
        from ..emitter.live import LiveOpsEmitter
        from ..pattern.pattern import Plain as _Plain

        self._modal_prereqs_and_guards(num_modes, context=context)
        if not (0.0 <= f_min < f_max):
            raise ValueError(
                f"{context}: need 0 <= f_min < f_max, got "
                f"f_min={f_min}, f_max={f_max}."
            )
        min_nf = 2 if command == "random_response" else 1
        if n_freq < min_nf:
            raise ValueError(
                f"{context}: n_freq must be >= {min_nf}, got {n_freq}."
            )
        grids = ("lin", "log", "biased")
        if grid not in grids:
            raise ValueError(
                f"{context}: grid must be one of {grids}, got {grid!r}."
            )
        resps = ("disp", "vel", "accel")
        if resp not in resps:
            raise ValueError(
                f"{context}: resp must be one of {resps}, got {resp!r}."
            )
        if dof < 1:
            raise ValueError(
                f"{context}: dof is 1-based, got {dof}."
            )
        damping_args = _damping_channel_args(
            damp=damp, rayleigh=rayleigh, modal_damp=modal_damp,
            context=context,
        )
        has_base = base_accel_dir is not None
        has_load = load is not None
        if has_base == has_load:
            raise ValueError(
                f"{context}: supply exactly one excitation channel — "
                "base_accel_dir= (harmonic base acceleration, no "
                "timeSeries) OR load= (nodal-force pattern); got "
                f"base_accel_dir={base_accel_dir!r}, load={load!r}."
            )
        if base_accel_dir is not None:
            if int(base_accel_dir) < 1:
                raise ValueError(
                    f"{context}: base_accel_dir is 1-based, got "
                    f"{base_accel_dir}."
                )
            excitation: tuple[int | float | str, ...] = (
                "-baseAccel", "-dir", int(base_accel_dir),
            )
        else:
            assert load is not None  # XOR check above guarantees it
            pat_tag = self._resolve_load_pattern_tag(
                load, context=context, plain_cls=_Plain,
            )
            excitation = ("-load", pat_tag)

        args: list[int | float | str] = [
            "-freq", float(f_min), float(f_max), int(n_freq),
            f"-{grid}",
        ]
        args.extend(excitation)
        if amp is not None and amp != 1.0:
            args.extend(("-amp", float(amp)))
        args.extend(damping_args)
        args.extend(("-node", _node_tag(node), "-dof", int(dof)))
        if resp != "disp":
            args.extend(("-resp", resp))
        if modes is not None:
            mode_list = [int(m) for m in modes]
            if not mode_list or any(m < 1 for m in mode_list):
                raise ValueError(
                    f"{context}: modes must be 1-based mode numbers, "
                    f"got {modes!r}."
                )
            args.extend(("-modes", *mode_list))
        args.extend(extra_args)
        if out is not None:
            args.extend(("-out", out))

        bm = self.build()
        self._assert_fork_if_required()
        live_emitter = LiveOpsEmitter(wipe=True)
        bm.emit(live_emitter)
        live_emitter.eigen(num_modes, solver=solver)
        live_emitter.modal_properties()
        runner = getattr(live_emitter, command)
        return runner(*args)
