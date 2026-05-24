"""
LiveOpsEmitter — drives openseespy in-process for live execution.

The only emitter that imports :mod:`openseespy.opensees`. Each
Protocol method is a thin wrapper around the matching ``ops.X(...)``
call.

The Tcl-block ``section_open`` / ``section_close`` and ``pattern_open``
/ ``pattern_close`` pairs map to ``ops.section(...)`` / ``ops.pattern(...)``
because openseespy itself maintains the implicit current-section /
current-pattern state — ``section_close`` / ``pattern_close`` are
no-ops here (the next non-fiber/patch/layer or non-load/sp/eleLoad
call ends the implicit scope).

This is the only place ``import openseespy.opensees`` may appear in
``apeGmsh.opensees`` (per Phase-4 architecture rules).
"""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Callable, Literal

if TYPE_CHECKING:
    from types import ModuleType


__all__ = ["LiveOpsEmitter"]


class _NoOpOps:
    """A no-op stand-in for the openseespy module.

    Used by :class:`LiveOpsEmitter` while a non-zero partition block
    is open. Every attribute access returns a callable that swallows
    its arguments and returns ``0`` (the convention :meth:`analyze`
    uses for "no failure"). This lets every emit-style method on
    :class:`LiveOpsEmitter` continue to call ``self._ops.X(...)``
    unchanged while suppression is active.
    """

    __slots__ = ()

    def __getattr__(self, _name: str) -> Any:
        def _noop(*_args: Any, **_kwargs: Any) -> int:
            return 0
        return _noop


def _get_ops() -> "ModuleType":
    """Lazy-import openseespy. Raises a clear error if not installed."""
    try:
        import openseespy.opensees as ops
    except ImportError as e:
        raise ImportError(
            "LiveOpsEmitter requires openseespy. Install it via the "
            "opensees venv (see memory: opensees venv) or pip-install "
            "openseespy in the active environment."
        ) from e
    # mypy stubs for openseespy don't ship a ModuleType return; wrap
    # to silence the no-any-return warning.
    from types import ModuleType
    assert isinstance(ops, ModuleType)
    return ops


class LiveOpsEmitter:
    """Drives openseespy in-process for live execution.

    Construct with ``wipe=True`` (default) to call ``ops.wipe()`` at
    construction so prior session state is cleared.

    Each Protocol method calls the matching ``ops.X(...)``. Return
    values are forwarded for :meth:`analyze` (the only method whose
    return value the bridge cares about).
    """

    def __init__(self, *, wipe: bool = True) -> None:
        self._ops = _get_ops()
        if wipe:
            self._ops.wipe()
        # Partition-emission state (ADR 0027 / P4). LiveOps is
        # single-process; it cannot drive OpenSeesMP. ``partition_open(0)``
        # passes through; ``partition_open(K!=0)`` swaps ``self._ops``
        # for a :class:`_NoOpOps` so every subsequent emit method
        # silently no-ops until the matching ``partition_close``. The
        # real openseespy module is stashed in ``_real_ops`` and
        # restored on ``partition_close``. A one-shot ``UserWarning``
        # fires on the first non-zero ``partition_open`` to surface
        # the contract mismatch (live cannot run partitioned models).
        self._real_ops: "ModuleType" = self._ops
        self._partition_warned: bool = False
        self._in_partition: bool = False
        # Step-hook state (Phase SSI-1).  ``_before_step_hooks`` /
        # ``_after_step_hooks`` are lists of zero-arg callables fired
        # before / after each ``analyze 1`` call when the hook-wrapped
        # path is active (i.e. ``_step_hooks_registered is True``).
        # Each closure captures its own state (counter + per-parameter
        # cumulative value) so multiple ramps coexist cleanly.
        self._before_step_hooks: list[Callable[[], None]] = []
        self._after_step_hooks: list[Callable[[], None]] = []
        self._step_hooks_registered: bool = False

    # -- Model ---------------------------------------------------------------

    def model(self, *, ndm: int, ndf: int) -> None:
        self._ops.model("basic", "-ndm", ndm, "-ndf", ndf)

    def node(
        self, tag: int, *coords: float, ndf: int | None = None,
    ) -> None:
        if ndf is None:
            self._ops.node(tag, *coords)
        else:
            # ``ops.node(tag, *xyz, '-ndf', n)`` is the openseespy idiom
            # for the per-node DOF override (ADR 0022 INV-3).
            self._ops.node(tag, *coords, "-ndf", ndf)

    def fix(self, tag: int, *dofs: int) -> None:
        self._ops.fix(tag, *dofs)

    def mass(self, tag: int, *values: float) -> None:
        self._ops.mass(tag, *values)

    # -- MP constraints (ADR 0022, Phase 7b) -----------------------------

    def equalDOF(self, master: int, slave: int, *dofs: int) -> None:
        self._ops.equalDOF(master, slave, *dofs)

    def rigidLink(self, kind: str, master: int, slave: int) -> None:
        self._ops.rigidLink(kind, master, slave)

    def rigidDiaphragm(
        self, perp_dir: int, master: int, *slaves: int,
    ) -> None:
        self._ops.rigidDiaphragm(perp_dir, master, *slaves)

    def embeddedNode(
        self, ele_tag: int, cnode: int,
        *args: int | float,
    ) -> None:
        self._ops.element(
            "ASDEmbeddedNodeElement", ele_tag, cnode, *args,
        )

    def mp_constraint_comment(self, name: str) -> None:
        # No-op — live execution can't carry comments. Argument exists
        # so the Protocol shape is uniform across emitters (INV-4).
        del name

    # -- Regions -------------------------------------------------------------

    def region(self, tag: int, *args: int | float | str) -> None:
        self._ops.region(tag, *args)

    # -- Constitutive --------------------------------------------------------

    def uniaxialMaterial(
        self, mat_type: str, tag: int, *params: float | str,
    ) -> None:
        self._ops.uniaxialMaterial(mat_type, tag, *params)

    def nDMaterial(
        self, mat_type: str, tag: int, *params: float | str,
    ) -> None:
        self._ops.nDMaterial(mat_type, tag, *params)

    def section(
        self, sec_type: str, tag: int, *params: float | str,
    ) -> None:
        self._ops.section(sec_type, tag, *params)

    def geomTransf(self, t_type: str, tag: int, *vec: float) -> None:
        self._ops.geomTransf(t_type, tag, *vec)

    # -- Sections that take blocks (Fiber) ----------------------------------

    def section_open(
        self, sec_type: str, tag: int, *params: float | str,
    ) -> None:
        # openseespy's ops.section(...) sets the current-section state.
        # Subsequent ops.patch / ops.layer / ops.fiber attach to this
        # section by openseespy convention.
        self._ops.section(sec_type, tag, *params)

    def section_close(self) -> None:
        # No-op — openseespy has no explicit close. The current-section
        # context ends implicitly when the next non-fiber/patch/layer
        # command runs.
        pass

    def patch(self, kind: str, *args: int | float) -> None:
        self._ops.patch(kind, *args)

    def fiber(
        self, y: float, z: float, area: float, mat_tag: int,
    ) -> None:
        self._ops.fiber(y, z, area, mat_tag)

    def layer(self, kind: str, *args: int | float) -> None:
        self._ops.layer(kind, *args)

    # -- Beam integration rules ---------------------------------------------

    def beamIntegration(
        self, rule_type: str, tag: int, *args: int | float | str,
    ) -> None:
        self._ops.beamIntegration(rule_type, tag, *args)

    # -- Topology ------------------------------------------------------------

    def element(
        self, ele_type: str, tag: int, *args: int | float | str,
    ) -> None:
        self._ops.element(ele_type, tag, *args)

    # -- Time series --------------------------------------------------------

    def timeSeries(
        self, ts_type: str, tag: int, *args: int | float | str,
    ) -> None:
        self._ops.timeSeries(ts_type, tag, *args)

    # -- Patterns -----------------------------------------------------------

    def pattern_open(
        self, p_type: str, tag: int, *args: int | float | str,
    ) -> None:
        self._ops.pattern(p_type, tag, *args)

    def pattern_close(self) -> None:
        # No-op — see section_close.
        pass

    def load(self, tag: int, *forces: float) -> None:
        self._ops.load(tag, *forces)

    def eleLoad(self, *args: int | float | str) -> None:
        self._ops.eleLoad(*args)

    def sp(self, tag: int, dof: int, value: float) -> None:
        self._ops.sp(tag, dof, value)

    # -- Recorders ----------------------------------------------------------

    def recorder(self, kind: str, *args: int | float | str) -> None:
        self._ops.recorder(kind, *args)

    def recorder_declaration_begin(
        self,
        *,
        declaration_name: str,
        record_name: str | None,
        category: str,
        components: tuple[str, ...],
        raw: tuple[str, ...] = (),
        pg: tuple[str, ...] = (),
        label: tuple[str, ...] = (),
        selection: tuple[str, ...] = (),
        ids: tuple[int, ...] | None = None,
        dt: float | None = None,
        n_steps: int | None = None,
        file_root: str = ".",
    ) -> None:
        """Phase 9 schema 2.3.0 declaration metadata — no-op for live ops."""
        del (
            declaration_name, record_name, category, components, raw,
            pg, label, selection, ids, dt, n_steps, file_root,
        )

    def recorder_declaration_end(self) -> None:
        """Phase 9 schema 2.3.0 declaration metadata — no-op for live ops."""

    # -- Analysis chain -----------------------------------------------------

    def constraints(self, c_type: str, *args: float) -> None:
        self._ops.constraints(c_type, *args)

    def numberer(self, n_type: str) -> None:
        self._ops.numberer(n_type)

    def system(self, s_type: str, *args: int | float | str) -> None:
        self._ops.system(s_type, *args)

    def test(self, t_type: str, *args: int | float | str) -> None:
        self._ops.test(t_type, *args)

    def algorithm(self, a_type: str, *args: int | float | str) -> None:
        self._ops.algorithm(a_type, *args)

    def integrator(self, i_type: str, *args: int | float | str) -> None:
        self._ops.integrator(i_type, *args)

    def analysis(self, a_type: str) -> None:
        self._ops.analysis(a_type)

    def analyze(self, *, steps: int, dt: float | None = None) -> int:
        if not self._step_hooks_registered:
            if dt is None:
                ret: Any = self._ops.analyze(steps)
            else:
                ret = self._ops.analyze(steps, dt)
            return int(ret)
        # Hook-wrapped form: drive the loop in-process, firing
        # registered closures between each ``ops.analyze(1)`` call.
        # On the first non-zero return from openseespy we break with
        # that exit code so the caller can react to non-convergence —
        # matching the contract of unhook-wrapped ``ops.analyze(N)``
        # (which short-circuits internally on failure).
        last_ret: int = 0
        for _ in range(int(steps)):
            for fn in self._before_step_hooks:
                fn()
            if dt is None:
                r: Any = self._ops.analyze(1)
            else:
                r = self._ops.analyze(1, dt)
            ri = int(r)
            if ri != 0:
                last_ret = ri
                break
            for fn in self._after_step_hooks:
                fn()
        return last_ret

    def eigen(
        self, num_modes: int, *, solver: str = "-genBandArpack",
    ) -> list[float]:
        # openseespy: ``ops.eigen(solver, num_modes)`` returns a list of
        # eigenvalues ``λ_i = ω_i²``. Modal shapes are queried separately
        # via ``ops.nodeEigenvector(node_tag, mode_idx)`` — see
        # :class:`apeGmsh.opensees.analysis.eigen.EigenResult`.
        values: Any = self._ops.eigen(solver, num_modes)
        return [float(v) for v in values]

    # -- Partition emission scoping (ADR 0027, P4) --------------------------

    def partition_open(self, rank: int) -> None:
        """Begin a per-rank emission block.

        LiveOps runs in a single Python process; it cannot drive
        OpenSeesMP. The implementation treats itself as rank 0:

        * ``rank == 0``: pass through — subsequent emits go to the
          real openseespy module.
        * ``rank != 0``: suppress every subsequent emit until the
          matching ``partition_close`` by swapping ``self._ops`` for a
          :class:`_NoOpOps`. A one-shot ``UserWarning`` fires on the
          first non-zero call to surface the contract mismatch.
        """
        self._in_partition = True
        if rank != 0:
            if not self._partition_warned:
                warnings.warn(
                    "LiveOpsEmitter is single-process; "
                    f"partition_open(rank={rank}) with rank!=0 will "
                    "suppress emission. Use ops.tcl(...) / ops.py(...) "
                    "+ `mpirun -np N OpenSeesMP` for true parallel runs.",
                    UserWarning,
                    stacklevel=2,
                )
                self._partition_warned = True
            # Swap to a no-op stand-in. The real ops module is
            # restored in ``partition_close``.
            self._ops = _NoOpOps()  # type: ignore[assignment]
        # rank == 0: pass through; ``self._ops`` already the real module.

    def partition_close(self) -> None:
        """End the current per-rank block; restore real openseespy."""
        # Restore unconditionally; cheap and idempotent.
        self._ops = self._real_ops
        self._in_partition = False

    # -- Partition runtime-conditional fallback (ADR 0027 INV-5) ------------

    def parallel_runtime_fallback_numberer(
        self, primary: str, fallback: str,
    ) -> None:
        """Try the primary numberer in-process; fall back on any
        ``Exception`` with a single ``UserWarning``.

        Mirrors the runtime conditional emitted by Tcl/Py emitters but
        actually executes the choice against openseespy at emit time
        (live mode runs in-process).  ``ParallelPlain`` is typically
        not available in a single-process openseespy build, so this
        path almost always lands in the fallback branch — but the
        warning surfaces the contract mismatch loudly.
        """
        try:
            self._ops.numberer(primary)
        except Exception:
            warnings.warn(
                f"{primary!r} numberer not available in this "
                "openseespy build; falling back to "
                f"{fallback!r}. Use OpenSeesMP for true parallel runs.",
                UserWarning,
                stacklevel=2,
            )
            self._ops.numberer(fallback)

    def parallel_runtime_fallback_system(
        self, primary: str, fallback: str,
    ) -> None:
        """Try the primary system in-process; fall back on any
        ``Exception`` with a single ``UserWarning``.

        Mirror of :meth:`parallel_runtime_fallback_numberer`.
        """
        try:
            self._ops.system(primary)
        except Exception:
            warnings.warn(
                f"{primary!r} system not available in this "
                "openseespy build; falling back to "
                f"{fallback!r}. Use OpenSeesMP for true parallel runs.",
                UserWarning,
                stacklevel=2,
            )
            self._ops.system(fallback)

    # -- Staged analysis (Phase SSI-2.A) ------------------------------------
    #
    # Live execution does not currently support staged builds.  The
    # bridge's :meth:`apeSees.analyze` builds a deck, drives it
    # through this emitter, and calls :meth:`analyze` once — a model
    # shape with multiple stages would need separate analysis-chain
    # re-binding, per-stage analyze loops, ``loadConst`` /
    # ``wipeAnalysis`` interleaved, and hook-list clearing in
    # between.  Lifting this is feasible but deferred: Tcl + Py text
    # emit are the supported execution paths for staged decks in
    # Phase SSI-2.A.

    def stage_open(self, name: str) -> None:
        raise NotImplementedError(
            "LiveOpsEmitter does not support staged models in Phase "
            f"SSI-2.A (got stage_open(name={name!r})).  Use "
            "``ops.tcl(...)`` / ``ops.py(...)`` to emit a staged deck "
            "and run it via OpenSees.exe / openseespy subprocess "
            "instead."
        )

    def stage_close(self) -> None:
        raise NotImplementedError(
            "LiveOpsEmitter does not support staged models in Phase "
            "SSI-2.A.  Use ``ops.tcl(...)`` / ``ops.py(...)`` to emit "
            "a staged deck and run it via OpenSees.exe / openseespy "
            "subprocess instead."
        )

    # -- Stress control (Phase SSI-1: initial_stress + ramping hooks) -------

    def addToParameter(
        self, tag: int, ele_tag: int, response: str,
    ) -> None:
        self._ops.addToParameter(tag, "element", ele_tag, response)

    def step_hook_ramp(
        self,
        name: str,
        *,
        targets: tuple[tuple[int, float], ...],
        n_steps_to_full: float,
        phase: Literal["before", "after"] = "before",
    ) -> None:
        """Build a Python closure that ramps the parameters per step
        and register it with the in-process hook dispatcher.

        Side effects (matching the Tcl / Py emitters):

        1. Declare each parameter in the live openseespy domain.
        2. Capture per-hook state (step counter + cumulative value per
           parameter) in a dict the closure mutates.
        3. Append the closure to the before- or after-step hook list.
        4. Flip ``_step_hooks_registered`` so :meth:`analyze` wraps
           the loop with hook calls.

        The ``name`` argument is recorded for diagnostics but is not
        otherwise observable from the closure — the live dispatch does
        not need a global name lookup.
        """
        del name  # diagnostics-only; live dispatch is by reference.

        # 1. Declare the parameters in the live domain.
        for tag, _target in targets:
            self._ops.parameter(int(tag))

        # 2. Per-hook state — closed over by the hook function below.
        captured_targets = tuple((int(t), float(v)) for t, v in targets)
        captured_divisor = float(n_steps_to_full)
        state: dict[str, Any] = {"count": 0}
        for tag, _ in captured_targets:
            state[f"cum_{tag}"] = 0.0

        # 3. The ramp closure — same algorithm as the emitted Tcl proc:
        # advance counter, compute capped factor, emit one
        # ``updateParameter $tag $delta`` per target.
        def _ramp() -> None:
            state["count"] = state["count"] + 1
            factor = min(state["count"] / captured_divisor, 1.0)
            for tag, target in captured_targets:
                current = target * factor
                delta = current - state[f"cum_{tag}"]
                self._ops.updateParameter(tag, delta)
                state[f"cum_{tag}"] = current

        # 4. Register + flip the flag.
        if phase == "before":
            self._before_step_hooks.append(_ramp)
        else:
            self._after_step_hooks.append(_ramp)
        self._step_hooks_registered = True

    # -- Direct accessor for tests / diagnostics ----------------------------

    @property
    def ops(self) -> "ModuleType":
        """Return the openseespy module — lets live-mode users query state.

        Useful for ``ret = ops_emitter.ops.nodeDisp(2, 1)`` after an
        analysis runs.
        """
        return self._ops
