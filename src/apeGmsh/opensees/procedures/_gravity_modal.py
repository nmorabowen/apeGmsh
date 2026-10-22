"""Gravity-preloaded modal analysis on one live domain (ADR 0123).

``modal_after_gravity`` is the P-Delta route the ETABS "iterative based on
loads" option takes: the gravity preload, ``loadConst -time 0``, then
``eigen`` + ``modalProperties`` on the tangent the gravity state leaves
behind, all on the same domain. The other modal drivers
(:mod:`._modal`) build a fresh, unloaded domain for the eigen solve, so they
see ``K`` only.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .._internal.build.errors import BridgeError

if TYPE_CHECKING:
    from ..analysis.gravity_modal import GravityModalResult
    from .._readback import ReadState

from ._host import _ProcedureHost

#: Gravity is reached at ``steps * dlam == 1`` to this tolerance.
_UNIT_LAMBDA_TOL = 1e-9


class _GravityModalMixin(_ProcedureHost):
    if TYPE_CHECKING:
        _read_state: "ReadState"

    def modal_after_gravity(
        self,
        num_modes: int,
        *,
        p_delta: bool = True,
        solver: str = "-genBandArpack",
    ) -> "GravityModalResult":
        """Gravity preload, ``loadConst``, then ``eigen`` + ``modalProperties``
        on the same live domain.

        The sequence (ADR 0123 D1):

        1. Build and emit the model, as :meth:`analyze` does.
        2. Run the declared analysis chain to ``λ = 1``: every registered
           load pattern is the gravity preload (declare the gravity
           combination, e.g. ``D + SD + 0.25 L``, as the patterns and
           their factors). The chain must be ``Static`` with a
           fixed-increment ``LoadControl(dlam=1/N)``; the driver runs the
           ``N`` steps, and the declared ``test`` and ``algorithm`` set the
           tolerance and the iteration (``Newton`` converges a P-Delta
           model; the step count is the ``dlam``).
        3. ``loadConst -time 0``: the gravity loads stay on, frozen.
        4. ``eigen`` on the current tangent, then ``modalProperties -return``.

        **What ``p_delta`` means.** ``True`` solves step 4 after the preload,
        so the stiffness is ``K - Kg`` with ``Kg`` from the gravity axial
        forces. ``False`` solves it *before* step 2, on the unstressed
        stiffness ``K`` (no initial axial force), and the gravity analysis
        runs afterwards; the result's :attr:`~GravityModalResult.gravity`
        is the same in both. The geometric stiffness comes from the element
        formulation, not from this driver: ``geomTransf PDelta`` (or
        ``Corotational``) frames and geometrically nonlinear shells such as
        :class:`~apeGmsh.opensees.element.shell.ShellNLDKGQ` carry it; a
        ``Linear`` transformation or a linear shell gives the same modes
        either way.

        **Solver.** ``solver`` is passed to ``eigen`` verbatim. The default
        ``-genBandArpack`` is the ARPACK ``ArpackSOE``, which factors through
        the analysis's own linear system, so the declared ``system``
        (``UmfPack``/``BandGen`` on stock, ``Pardiso`` on the fork) is the
        one the eigen solve uses. ARPACK needs ``num_modes`` below the free
        DOF count; use ``-fullGenLapack`` on a tiny model.

        The gravity state is snapshot right after step 3 and re-read after
        step 4; a difference raises (the eigen solve must not move the
        domain). The domain is left at the gravity state, and the bridge's
        read session (``Node.read``) answers there.

        Parameters
        ----------
        num_modes
            Modes to extract, ``>= 1``.
        p_delta
            ``True`` (default): modes on ``K - Kg`` after the preload.
            ``False``: modes on ``K`` before it.
        solver
            The ``eigen`` solver flag.

        Raises
        ------
        ValueError
            ``num_modes < 1``, or the integrator is not a fixed-increment
            ``LoadControl`` reaching ``λ = 1``.
        BridgeError
            The analysis chain is incomplete, or not ``Static``.
        NotImplementedError
            The model has registered stages (live runs flat models only,
            ADR 0121 D2).
        RuntimeError
            The gravity analysis failed to converge, or the eigen solve
            moved the gravity state.
        """
        from ..analysis.gravity_modal import (
            GravityModalResult,
            snapshot_state,
            state_drift,
        )
        from ..analysis.modal import ModalPropertiesResult
        from .._readback import DroppedSession, open_session
        from ..emitter.live import LiveOpsEmitter
        from .._internal.tag_plan import emit_mode
        import numpy as np

        context = "apeSees.modal_after_gravity"
        self._modal_prereqs_and_guards(num_modes, context=context)
        self._check_analysis_chain_for_analyze()
        steps = self._gravity_steps(context)

        bm = self.build()
        self._assert_fork_if_required()
        self._read_state = DroppedSession(
            f"{context}(...) started on this bridge and did not complete"
        )
        live_emitter = LiveOpsEmitter(wipe=True)
        self._live_emitter = live_emitter
        bm.emit(live_emitter)
        self._artifacts.after_emit(self)
        ops = live_emitter.ops

        values: list[float] = []
        properties: dict[str, list[float]] = {}
        if not p_delta:
            values = live_emitter.eigen(num_modes, solver=solver)
            properties = live_emitter.modal_properties()

        rc = int(live_emitter.analyze(steps=steps))
        if rc != 0:
            raise RuntimeError(
                f"{context}: the gravity analysis failed at return code "
                f"{rc} (time {float(ops.getTime()):.6g} of 1.0). Check the "
                "declared test and algorithm (Newton for a P-Delta model) "
                "or take more LoadControl steps."
            )
        reached = float(ops.getTime())
        if abs(reached - 1.0) > _UNIT_LAMBDA_TOL:
            raise RuntimeError(
                f"{context}: the gravity analysis ended at time {reached!r}, "
                "not 1.0."
            )
        ops.loadConst("-time", 0.0)
        gravity = snapshot_state(ops)

        if p_delta:
            values = live_emitter.eigen(num_modes, solver=solver)
            properties = live_emitter.modal_properties()

        disp_drift, reac_drift = state_drift(gravity, snapshot_state(ops))
        if disp_drift != 0.0 or reac_drift != 0.0:
            raise RuntimeError(
                f"{context}: the eigen solve moved the gravity state "
                f"(relative drift: displacement {disp_drift:.3e}, reaction "
                f"{reac_drift:.3e})."
            )

        live_plan = bm._tag_plan(emit_mode(
            bm, split=False,
            supports_partitions=live_emitter.caps.supports_partitions))
        self._read_state = open_session(
            live_emitter, ndm=bm.ndm,
            element_plan=tuple(live_plan.elements.specs),
        )
        return GravityModalResult(
            modal=ModalPropertiesResult(
                eigenvalues=np.asarray(values, dtype=np.float64),
                properties=properties,
                _live=live_emitter,
            ),
            gravity=gravity,
            p_delta=bool(p_delta),
            gravity_steps=steps,
            _live=live_emitter,
        )

    def _gravity_steps(self, context: str) -> int:
        """The step count of the declared gravity ``LoadControl``.

        The last registered integrator and analysis are the ones the
        domain ends up with (emit order). The integrator must be a
        fixed-increment ``LoadControl`` whose ``dlam`` divides 1.
        """
        from .._internal.types import Analysis, Integrator
        from ..analysis.analysis import Static
        from ..analysis.integrator import LoadControl

        integrator = next(
            (p for p in reversed(self._primitives)
             if isinstance(p, Integrator)),
            None,
        )
        analysis = next(
            (p for p in reversed(self._primitives)
             if isinstance(p, Analysis)),
            None,
        )
        if not isinstance(analysis, Static):
            raise BridgeError(
                f"{context}: the gravity preload needs ops.analysis.Static(); "
                f"got {type(analysis).__name__}."
            )
        if not isinstance(integrator, LoadControl):
            raise ValueError(
                f"{context}: the gravity preload needs "
                "ops.integrator.LoadControl(dlam=1/N); got "
                f"{type(integrator).__name__}."
            )
        if integrator.num_iter is not None:
            raise ValueError(
                f"{context}: LoadControl(num_iter=...) adapts the increment, "
                "so the step count that reaches λ = 1 is unknown; declare a "
                "fixed dlam = 1/N."
            )
        dlam = float(integrator.dlam)
        steps = round(1.0 / dlam) if dlam > 0.0 else 0
        if steps < 1 or abs(steps * dlam - 1.0) > _UNIT_LAMBDA_TOL:
            raise ValueError(
                f"{context}: LoadControl(dlam={dlam!r}) does not reach "
                "λ = 1 in a whole number of steps; declare dlam = 1/N."
            )
        return steps


__all__ = ["_GravityModalMixin"]

