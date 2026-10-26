"""Gravity-preloaded modal analysis on one live domain (ADR 0124).

``modal_after_gravity`` is the P-Delta route the ETABS "iterative based on
loads" option takes: the gravity preload, ``loadConst -time 0``, then
``eigen`` + ``modalProperties`` on the tangent the gravity state leaves
behind, all on the same domain. The other modal drivers
(:mod:`._modal`) build a fresh, unloaded domain for the eigen solve, so they
see ``K`` only. ``spectrum_after_gravity`` adds the per-mode response
spectrum about that gravity state (ADR 0124 D4).
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

from .._internal.build.errors import BridgeError

if TYPE_CHECKING:
    import numpy as np

    from ..analysis.gravity_modal import GravityModalResult
    from ..analysis.spectrum import (
        DesignSpectrum,
        ResponseExtractor,
        SpectrumModalResult,
    )
    from ..emitter.live import LiveOpsEmitter
    from .._readback import ReadState

    _Hook = Callable[
        [LiveOpsEmitter, np.ndarray, Mapping[str, Sequence[float]]], None]

from ._host import _ProcedureHost

#: Gravity is reached at ``steps * dlam == 1`` to this tolerance.
_UNIT_LAMBDA_TOL = 1e-9

#: The per-mode solves must leave the domain this close (relative) to the
#: state they started from.
_RESTORE_TOL = 1e-6

#: Each mode's probe moves the largest nodal translation by this fraction of
#: the model's extent, whatever the spectrum's amplitude. Small, because the
#: kink check below separates a first-order effect from a term that grows
#: with the probe; large enough that round-off stays below 1e-10 (on the
#: live test frame: 3.5e-11 here, 2.6e-10 at 1e-9).
_PROBE_REL = 1e-8

#: The two probe legs must meet the same stiffness: the second difference
#: of the nodal resisting forces about the reference, relative to their
#: first difference, stays below this. It grows with the probe for any
#: smooth element (live frames: 2e-8 PDelta, 2e-6 Corotational) and does
#: not for a member whose stiffness differs between the legs (5.5e-2 for
#: Concrete01 braces on their backbone, 0.6 for no-tension braces at zero
#: strain).
_KINK_TOL = 1e-3

#: A mode whose effective mass is below this fraction of the largest, in
#: every requested direction, is not probed.
_NEGLIGIBLE_MASS = 1e-12


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

        The sequence (ADR 0124 D1):

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
        return self._run_gravity_modal(
            num_modes, p_delta=p_delta, solver=solver,
            context="apeSees.modal_after_gravity", hook=None,
        )

    def _run_gravity_modal(
        self,
        num_modes: int,
        *,
        p_delta: bool,
        solver: str,
        context: str,
        hook: "_Hook | None",
    ) -> "GravityModalResult":
        """The D1 sequence; ``hook`` runs once the modes exist, with the
        domain at its reference state (the gravity state, or the unloaded
        one under ``p_delta=False``), and must leave it there."""
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
            if hook is not None:
                hook(live_emitter, np.asarray(values, dtype=np.float64),
                     properties)

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
        if p_delta and hook is not None:
            hook(live_emitter, np.asarray(values, dtype=np.float64),
                 properties)

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

    def spectrum_after_gravity(
        self,
        num_modes: int,
        *,
        spectra: "Sequence[DesignSpectrum]",
        extractors: "Mapping[str, ResponseExtractor]",
        p_delta: bool = True,
        solver: str = "-genBandArpack",
    ) -> "SpectrumModalResult":
        """Per-mode response spectrum about the gravity state (ADR 0124 D4).

        Runs :meth:`modal_after_gravity`'s sequence, then, on the same
        domain and for each mode ``k`` and each sign, applies the probe
        force ``± a_k M φ_k`` as a nodal-load pattern, takes one ``Linear``
        step on the current tangent at ``dλ = 0`` (the tangent the modes
        were solved on, so the domain moves by exactly ``± a_k φ_k / λ_k``),
        calls every extractor there, removes the pattern and solves back to
        the reference state with the declared algorithm. The per-mode value
        under spectrum ``d`` is ``Γ_kd Sa_d(T_k) (r⁺ - r⁻) / (2 a_k)``: the
        state ``responseSpectrumAnalysis -mode k`` imposes, about the
        reference state instead of in place of it. Displacements are the
        modal ones to round-off, PDelta included; element forces and
        reactions are those of the elements at that state. The probe
        amplitude ``a_k`` does not depend on the spectrum: the largest nodal
        translation moves by ``1e-8`` of the model's extent. A mode whose
        effective mass is below ``1e-12`` of the largest in every requested
        direction is skipped (left at zero). The two legs must meet the same
        stiffness: a second difference of the nodal resisting forces above
        ``1e-3`` of the first difference (a no-tension brace or a gap at zero
        strain, a material that loads and unloads on different branches,
        such as ``Concrete01`` on its backbone) raises. Combine the result
        with :meth:`SpectrumModalResult.combine` (CQC / SRSS) and
        :meth:`SpectrumModalResult.directional` (100/30 / SRSS); the gravity
        values are kept apart (:func:`~apeGmsh.opensees.analysis.spectrum.seismic_envelope`).

        The reference state is the gravity state with ``p_delta=True`` and
        the unloaded state with ``p_delta=False`` (modes and per-mode
        responses on ``K``, gravity run afterwards). The domain ends at the
        gravity state either way, checked against the snapshot.

        ``responseSpectrumAnalysis -mode k`` is not used: it replaces the
        committed displacement with the modal one and commits, so the gravity
        state is lost (ADR 0124, Context).

        **Nodal mass only.** The modal force is ``M φ`` built from the nodal
        masses, so a model whose mass sits in its elements (a section or
        material ``rho``, an element ``mass=``) is refused. Lump the mass to
        the nodes (``g.masses`` on the session, or ``ops.mass`` /
        ``mass_from_model``): that is the supported route.

        **No recorders.** Each probe leg and each return solve commits, so a
        declared recorder would write four extra rows per mode that are
        not states of the structure. A model with a declared recorder is
        refused; the gravity state is in
        :attr:`SpectrumModalResult.gravity_values` and in ``Node.read`` on
        the bridge afterwards.

        The driver always adds the extractor ``"base_shear"`` (the total
        support reaction per translational direction), so
        :meth:`SpectrumModalResult.base_shear` gives ``V_t`` per direction.

        Parameters
        ----------
        num_modes
            Modes to extract and combine, ``>= 1``.
        spectra
            One :class:`~apeGmsh.opensees.analysis.spectrum.DesignSpectrum`
            per excitation; each names its translational direction.
        extractors
            Name -> ``extractor(ops) -> 1-D float array`` of fixed length
            (:class:`~apeGmsh.opensees.analysis.spectrum.NodeDisp`,
            :class:`~apeGmsh.opensees.analysis.spectrum.NodeReaction`,
            :class:`~apeGmsh.opensees.analysis.spectrum.ElementResponse`, or
            any callable, such as a story-shear aggregator). The name
            ``"base_shear"`` is reserved.
        p_delta, solver
            As :meth:`modal_after_gravity`.

        Raises
        ------
        BridgeError
            The model carries element mass (refused: the modal force is
            built from nodal masses), declares a recorder, or a spectrum
            names a direction outside ``1..ndm``.
        ValueError
            No spectra or extractors, or an extractor's length changed.
        RuntimeError
            A per-mode solve failed, a mode has a non-positive eigenvalue,
            the two probe legs met different stiffnesses, or the domain did
            not return to the reference state.
        """
        from .._internal.types import Recorder
        from ..analysis.spectrum import SpectrumModalResult

        context = "apeSees.spectrum_after_gravity"
        recorders = sorted({type(p).__name__ for p in self._primitives
                            if isinstance(p, Recorder)})
        if recorders:
            raise BridgeError(
                f"{context}: the model declares recorders ({', '.join(recorders)}). "
                "Every per-mode probe leg and return solve commits, so they "
                "would record four rows per mode that are not states of the "
                "structure. Remove the recorders; the gravity state is in "
                "SpectrumModalResult.gravity_values and Node.read afterwards, "
                "and modal_after_gravity records the gravity run on its own."
            )
        spectra_t = tuple(spectra)
        if not spectra_t:
            raise ValueError(f"{context}: give at least one DesignSpectrum.")
        from ..analysis.spectrum import BASE_SHEAR, _TotalReaction

        named = dict(extractors)
        if not named:
            raise ValueError(f"{context}: give at least one extractor.")
        if BASE_SHEAR in named:
            raise ValueError(
                f"{context}: the extractor name {BASE_SHEAR!r} is reserved "
                "for the driver's own total base reaction."
            )
        ndm = self._ndm
        named[BASE_SHEAR] = _TotalReaction(int(ndm or 0))
        for s in spectra_t:
            if ndm is not None and not 1 <= s.direction <= ndm:
                raise BridgeError(
                    f"{context}: a spectrum acts along direction "
                    f"{s.direction}, outside the translational 1..{ndm}."
                )
        holder: dict[str, object] = {}

        def hook(live_emitter: "LiveOpsEmitter", eigenvalues: "np.ndarray",
                 properties: "Mapping[str, Sequence[float]]") -> None:
            holder["out"] = self._per_mode_spectrum(
                live_emitter, eigenvalues, properties, spectra_t, named,
                context=context,
            )

        modal = self._run_gravity_modal(
            num_modes, p_delta=p_delta, solver=solver, context=context,
            hook=hook,
        )
        gravity_values, per_mode, labels, drift = cast(
            "tuple[dict[str, np.ndarray], dict[str, np.ndarray], "
            "dict[str, tuple[str, ...] | None], tuple[float, float]]",
            holder["out"],
        )
        if not p_delta:
            # The per-mode reference was the unloaded state; the values the
            # gravity combinations need are those of the gravity state.
            ops = modal._live.ops
            gravity_values = {
                name: _evaluate(ex, ops, name, None, context)
                for name, ex in named.items()
            }
        return SpectrumModalResult(
            modal=modal,
            spectra=spectra_t,
            gravity_values=gravity_values,
            per_mode=per_mode,
            labels=labels,
            restore_drift=drift,
        )

    def _per_mode_spectrum(
        self,
        live_emitter: "LiveOpsEmitter",
        eigenvalues: "np.ndarray",
        properties: "Mapping[str, Sequence[float]]",
        spectra: "tuple[DesignSpectrum, ...]",
        extractors: "Mapping[str, ResponseExtractor]",
        *,
        context: str,
    ) -> "tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, tuple[str, ...] | None], tuple[float, float]]":
        """The per-mode loop of ADR 0124 D4, about the domain's current
        committed state; leaves the domain back at it."""
        import numpy as np

        from .._internal.types import SolutionAlgorithm, TimeSeries
        from ..analysis.gravity_modal import snapshot_state, state_drift

        ops = live_emitter.ops
        # The last registered algorithm is the declared one (emit order);
        # the return solves restore it after each Linear leg.
        algorithm = next(
            (p for p in reversed(self._primitives)
             if isinstance(p, SolutionAlgorithm)), None)
        if algorithm is None:
            raise BridgeError(
                f"{context}: the analysis chain declares no algorithm.")
        ndm = int(self._ndm or 0)
        lam = np.asarray(eigenvalues, dtype=np.float64)
        if lam.size == 0 or np.any(lam <= 0.0):
            raise RuntimeError(
                f"{context}: every mode needs a positive eigenvalue to have a "
                f"period; got {lam.tolist()}."
            )
        periods = 2.0 * np.pi / np.sqrt(lam)
        components = ("MX", "MY", "MZ")
        gamma = np.stack([
            np.asarray(properties[f"partiFactor{components[s.direction - 1]}"],
                       dtype=np.float64)
            for s in spectra
        ])
        coef = gamma * np.stack([s.sa(periods) for s in spectra])

        # The modal force is M φ with the nodal masses; element mass (which
        # modalProperties counts) would be missing from it.
        tags = [int(t) for t in ops.getNodeTags()]
        masses = {t: np.asarray(ops.nodeMass(t), dtype=np.float64) for t in tags}
        nodal = np.zeros(ndm)
        for m in masses.values():
            k = min(ndm, m.size)
            nodal[:k] += m[:k]
        total = np.asarray(properties["totalMass"], dtype=np.float64)[:ndm]
        if not np.allclose(nodal, total, rtol=1e-9,
                           atol=1e-12 * max(1.0, float(np.max(np.abs(total))))):
            raise BridgeError(
                f"{context}: modalProperties counts a total mass "
                f"{total.tolist()} but the nodes carry {nodal.tolist()}: the "
                "model has element mass (a section or material rho, an "
                "element mass=), which is refused: the per-mode modal force "
                "is built from nodal masses (ADR 0124 D4). Lump the mass to "
                "the nodes instead (g.masses, or ops.mass / "
                "mass_from_model), the supported route."
            )
        massed = [t for t in tags if np.any(masses[t] != 0.0)]

        reference = snapshot_state(ops)
        ref_values = {
            name: _evaluate(ex, ops, name, None, context)
            for name, ex in extractors.items()
        }
        labels = {name: _labels(ex) for name, ex in extractors.items()}
        n_modes = lam.size
        per_mode = {
            name: np.zeros((len(spectra), n_modes, v.size))
            for name, v in ref_values.items()
        }

        ts_tag = 1 + max(
            (int(tag) for p in self._primitives if isinstance(p, TimeSeries)
             for tag in (self.tag_for(p),) if tag is not None),
            default=0,
        )
        ops.timeSeries("Constant", ts_tag)
        declared = live_emitter._last_integrator
        live_emitter.integrator("LoadControl", 0.0)

        def solve(where: str) -> None:
            rc = int(live_emitter.analyze(steps=1))
            if rc != 0:
                raise RuntimeError(
                    f"{context}: the {where} solve failed (return code {rc}). "
                    "The declared test and algorithm must converge a static "
                    "step about the gravity state."
                )

        # The probe size is a length, independent of the spectrum (ADR 0124
        # D4): the model's largest extent, so a probe is a fixed small
        # fraction of the geometry whatever the units.
        xyz = np.asarray([ops.nodeCoord(t) for t in tags], dtype=np.float64)
        extent = float(np.max(np.ptp(xyz, axis=0))) if xyz.size else 0.0
        probe = _PROBE_REL * (extent if extent > 0.0 else 1.0)

        # A mode whose effective mass is negligible in every requested
        # direction takes no part (a pure-torsion mode about a centred mass
        # node has Γ ~ 1e-12, not 0, and almost no translation there).
        meff = np.stack([
            np.abs(np.asarray(
                properties[f"partiMass{components[s.direction - 1]}"],
                dtype=np.float64))
            for s in spectra
        ])
        active = np.any(
            meff > _NEGLIGIBLE_MASS * np.max(meff, axis=1, keepdims=True),
            axis=0)

        def nodal_forces() -> "dict[int, np.ndarray]":
            # After ``reactions``, every node's reaction is its resisting
            # force minus its applied load (the unbalance at a free node).
            ops.reactions()
            return {t: np.asarray(ops.nodeReaction(t), dtype=np.float64)
                    for t in tags}

        ref_forces = nodal_forces()
        for k in range(n_modes):
            if not active[k] or not np.any(coef[:, k]):
                continue  # the mode takes no part in any spectrum
            phi = {t: np.asarray(ops.nodeEigenvector(t, k + 1),
                                 dtype=np.float64) for t in tags}
            peak = max(float(np.max(np.abs(v[:ndm]), initial=0.0))
                       for v in phi.values())
            if peak == 0.0:
                raise RuntimeError(
                    f"{context}: mode {k + 1} has effective mass but no node "
                    "translates in it."
                )
            amp = probe * float(lam[k]) / peak
            loads = {t: amp * masses[t] * phi[t] for t in massed}
            legs: list[dict[str, np.ndarray]] = []
            forces: list[dict[int, np.ndarray]] = []
            for sign in (1.0, -1.0):
                where = f"mode {k + 1} {'+' if sign > 0 else '-'}"
                pattern = 1 + max(
                    (int(p) for p in ops.getPatterns()), default=0)
                ops.pattern("Plain", pattern, ts_tag, "-fact", sign)
                for t, f in loads.items():
                    ops.load(t, *f.tolist())
                # One step on the current tangent, the one the modes were
                # solved on: the leg moves the domain by exactly ± a φ / λ,
                # whether or not the element's tangent is consistent with
                # its forces (the PDelta transformation's is not).
                live_emitter.algorithm("Linear")
                solve(f"{where} leg")
                legs.append({
                    name: _evaluate(ex, ops, name, ref_values[name].size,
                                    context)
                    for name, ex in extractors.items()
                })
                forces.append(nodal_forces())
                ops.remove("loadPattern", pattern)
                algorithm._emit(live_emitter, self.tag_for(algorithm) or 0)
                solve(f"{where} return")
            # The second difference of the resisting forces about the
            # reference: second order in the probe for any smooth element,
            # first order where a member's stiffness differs between the
            # two legs (a no-tension brace at zero strain, a material that
            # loads and unloads on different branches).
            even = max(float(np.max(np.abs(
                forces[0][t] + forces[1][t] - 2.0 * ref_forces[t]),
                initial=0.0)) for t in tags)
            odd = max(float(np.max(np.abs(
                forces[0][t] - forces[1][t]
                + (2.0 * loads[t] if t in loads else 0.0)), initial=0.0))
                for t in tags)
            if even > _KINK_TOL * odd:
                raise RuntimeError(
                    f"{context}: mode {k + 1}'s two probe legs met different "
                    f"stiffnesses (second difference of the resisting forces "
                    f"{even / odd:.3e} of the first, limit {_KINK_TOL:g}): a "
                    "member changes stiffness at the reference state, such "
                    "as a no-tension brace or a gap at zero strain, or a "
                    "material that loads and unloads on different branches "
                    "(Concrete01, Steel02 on their backbone). The response "
                    "spectrum is linear about that state and is refused."
                )
            for name in extractors:
                unit = (legs[0][name] - legs[1][name]) / (2.0 * amp)
                per_mode[name][:, k, :] = coef[:, k, None] * unit[None, :]

        if declared is not None:
            live_emitter.integrator(str(declared[0]), *declared[1:])
        drift = state_drift(reference, snapshot_state(ops))
        if max(drift) > _RESTORE_TOL:
            raise RuntimeError(
                f"{context}: the domain did not return to its reference state "
                f"after the per-mode solves (relative drift: displacement "
                f"{drift[0]:.3e}, reaction {drift[1]:.3e}; limit "
                f"{_RESTORE_TOL:g}). Tighten the declared convergence test."
            )
        return ref_values, per_mode, labels, drift


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


def _evaluate(
    extractor: "ResponseExtractor",
    ops: Any,
    name: str,
    size: int | None,
    context: str,
) -> "np.ndarray":
    """Call one extractor and hold it to its contract: a 1-D float array
    of the length it had at the reference state."""
    import numpy as np

    values = np.asarray(extractor(ops), dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(
            f"{context}: extractor {name!r} returned an array of shape "
            f"{values.shape}; it must return a 1-D array."
        )
    if size is not None and values.size != size:
        raise ValueError(
            f"{context}: extractor {name!r} changed length from {size} to "
            f"{values.size} between calls."
        )
    return values


def _labels(extractor: "ResponseExtractor") -> "tuple[str, ...] | None":
    labels = getattr(extractor, "labels", None)
    return None if labels is None else tuple(str(x) for x in labels)


__all__ = ["_GravityModalMixin"]

