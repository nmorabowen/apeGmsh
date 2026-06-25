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

import os
import warnings
from typing import TYPE_CHECKING, Any, Callable, Literal, Sequence

from .base import StrategySpec

if TYPE_CHECKING:
    from types import ModuleType


__all__ = ["LiveOpsEmitter", "get_backend_name"]


#: Raised by :meth:`LiveOpsEmitter.profiler` when the live openseespy build
#: lacks the fork-only ``profiler`` command (i.e. stock openseespy). Deck
#: emission still works on any build — only running the profiled deck needs
#: the Ladruno fork.
_PROFILER_FORK_REQUIRED = (
    "ops.profiler / analyze(profile=...) requires the Ladruno fork build of "
    "OpenSees (the 'profiler' command is fork-only). Deck emission via "
    "ops.tcl(...) / ops.py(...) works on any build; only running the profiled "
    "deck needs the fork. The Tcl deck (ops.tcl(run=True)) is the recommended "
    "profiled path."
)

#: Raised by :meth:`LiveOpsEmitter.ladruno_projection_tie_force` when the live
#: build lacks the fork-only ``ladrunoProjectionTieForce`` query (i.e. stock
#: openseespy). The query also requires the active constraint handler to be
#: ``LadrunoProjection`` — the fork command itself fail-louds in that case.
_TIE_FORCE_FORK_REQUIRED = (
    "ops.ladrunoProjectionTieForce requires the Ladruno fork build of OpenSees "
    "(fork-only, ADR-30 P3). The active constraint handler must also be "
    "LadrunoProjection (the equation-tie route, enforce='equation'). Deck "
    "emission via ops.tcl(...) / ops.py(...) works on any build; only the live "
    "query needs the fork."
)

#: Element types that exist only in the Ladruno fork build (private ≥33000
#: class-tag band). Emitting their ``element …`` line works on any build;
#: the fork is required only to *run* the deck in-process — gated in
#: :meth:`LiveOpsEmitter.element` below.
_FORK_ONLY_ELEMENTS = frozenset(
    {"BezierTri6", "BezierTet10", "LadrunoEmbeddedRebar",
     "LadrunoQuad", "LadrunoCST",
     "LadrunoKinematicCoupling", "LadrunoDistributingCoupling",
     "LadrunoEmbeddedNode", "LadrunoRigidBody",
     "LadrunoDispBeamColumn", "LadrunoIMKBeam"})


def _fork_element_required(ele_type: str) -> str:
    return (
        f"element {ele_type} requires the Ladruno fork build of OpenSees "
        f"(fork-only, private ≥33000 class-tag band) — the live build "
        f"does not know this element. Meshing and deck emission "
        f"(ops.tcl(...) / ops.py(...)) work on any build; only running the "
        f"deck in-process needs the fork. To run on a stock build, use the "
        f"direct-drive fallback (see bezier_apegmsh_integration.md)."
    )


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


#: Module-level cache for the resolved OpenSees backend. The backend is
#: resolved once on first use and reused for the life of the process, so
#: ``APEGMSH_OPENSEES_BIN`` must be set *before* the first emit / first
#: :func:`get_backend_name` call to take effect.
_OPS_CACHE: "ModuleType | None" = None
_BACKEND_NAME: str = "unresolved"


def _looks_like_opensees(mod: "ModuleType") -> bool:
    """True if ``mod`` exposes the core OpenSees command API.

    Guards against an importable-but-unrelated top-level ``opensees`` (e.g.
    apeGmsh's own ``tests/opensees`` package shadowing the name under
    pytest's importlib mode) being mistaken for an OpenSees backend.
    """
    return all(hasattr(mod, a) for a in ("wipe", "model", "element"))


def _resolve_ops() -> "tuple[ModuleType, str]":
    """Resolve the OpenSees backend module (auto-detect, env-overridable).

    Resolution order:

    1. ``APEGMSH_OPENSEES_BIN`` set → add it as a DLL directory (Windows;
       the Ladruno fork co-locates its MKL DLLs next to ``opensees.pyd``),
       put it on ``sys.path``, and ``import opensees`` (the fork's module
       name) → ``"ladruno-fork"``.
    2. else bare ``import opensees`` (fork already importable / on
       ``PYTHONPATH``) → ``"ladruno-fork"``.
    3. else ``import openseespy.opensees`` (stock PyPI build) →
       ``"stock-openseespy"``.

    The backend is tagged ``"ladruno-fork"`` only when the resolved module
    exposes a fork-only symbol (``criticalTimeStep``); this keeps detection
    honest even if a bare ``opensees`` module ever ships from elsewhere.
    """
    from types import ModuleType

    # Suppress the fork's splash banner in library / test output.
    os.environ.setdefault("LADRUNO_OPENSEES_QUIET", "1")

    bin_dir = os.environ.get("APEGMSH_OPENSEES_BIN")
    loaders: "list[Callable[[], ModuleType]]" = []

    if bin_dir:
        def _from_bin() -> ModuleType:
            import sys
            import importlib
            import importlib.util
            if hasattr(os, "add_dll_directory") and os.path.isdir(bin_dir):
                os.add_dll_directory(bin_dir)  # MKL DLLs (Windows)
            # If the fork is already loaded under ``opensees``, reuse it.
            existing = sys.modules.get("opensees")
            if existing is not None and hasattr(existing, "criticalTimeStep"):
                return existing
            # Load the extension the user pointed at *by explicit path*, so a
            # shadowing top-level ``opensees`` on sys.path (e.g. a local
            # package, or apeGmsh's own ``tests/opensees`` package registered
            # by pytest's importlib mode) cannot mask the fork build.
            pyd = next(
                (os.path.join(bin_dir, n) for n in
                 ("opensees.pyd", "opensees.so")
                 if os.path.isfile(os.path.join(bin_dir, n))),
                None,
            )
            if pyd is None:  # fall back to path-based import
                if bin_dir not in sys.path:
                    sys.path.insert(0, bin_dir)
                saved = sys.modules.pop("opensees", None)
                try:
                    importlib.invalidate_caches()
                    mod = importlib.import_module("opensees")
                except BaseException:
                    if saved is not None:
                        sys.modules["opensees"] = saved
                    raise
                if saved is not None and not hasattr(saved, "criticalTimeStep"):
                    sys.modules["opensees"] = saved
                return mod
            spec = importlib.util.spec_from_file_location("opensees", pyd)
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot load fork extension at {pyd}")
            mod = importlib.util.module_from_spec(spec)
            saved = sys.modules.get("opensees")
            sys.modules["opensees"] = mod  # exec_module expects registration
            try:
                spec.loader.exec_module(mod)
            except BaseException:
                if saved is not None:
                    sys.modules["opensees"] = saved
                else:
                    sys.modules.pop("opensees", None)
                raise
            # Restore a non-fork shadower so we don't disturb its owner (the
            # fork stays in our own cache regardless).
            if saved is not None and not hasattr(saved, "criticalTimeStep"):
                sys.modules["opensees"] = saved
            return mod
        loaders.append(_from_bin)

    def _bare_fork() -> ModuleType:
        import opensees as _ops
        return _ops
    loaders.append(_bare_fork)

    def _stock() -> ModuleType:
        import openseespy.opensees as _ops
        return _ops
    loaders.append(_stock)

    last_err: "ImportError | None" = None
    for load in loaders:
        try:
            ops = load()
        except ImportError as e:
            last_err = e
            continue
        assert isinstance(ops, ModuleType)
        if not _looks_like_opensees(ops):
            # An importable but non-OpenSees module shadowed the name (e.g.
            # apeGmsh's own ``tests/opensees`` package registered as a
            # top-level ``opensees`` by pytest's importlib mode). Skip it and
            # try the next loader rather than returning a backend with no
            # ``wipe`` / ``model`` / ``element``.
            continue
        name = (
            "ladruno-fork" if hasattr(ops, "criticalTimeStep")
            else "stock-openseespy"
        )
        return ops, name

    raise ImportError(
        "LiveOpsEmitter could not import an OpenSees backend. Point "
        "APEGMSH_OPENSEES_BIN at the Ladruno fork's dist\\bin (the folder "
        "holding opensees.pyd), put the fork on PYTHONPATH, or pip-install "
        "stock openseespy in the active environment."
    ) from last_err


def _get_ops() -> "ModuleType":
    """Lazy-import + cache the OpenSees backend module."""
    global _OPS_CACHE, _BACKEND_NAME
    if _OPS_CACHE is None:
        _OPS_CACHE, _BACKEND_NAME = _resolve_ops()
    return _OPS_CACHE


def get_backend_name() -> str:
    """Return the resolved backend: ``'ladruno-fork'`` or ``'stock-openseespy'``.

    Resolves the backend on first call (same path / cache as
    :func:`_get_ops`). Tests use this to skip fork-only cases cleanly on a
    stock build.
    """
    _get_ops()
    return _BACKEND_NAME


class LiveOpsEmitter:
    """Drives openseespy in-process for live execution.

    Construct with ``wipe=True`` (default) to call ``ops.wipe()`` at
    construction so prior session state is cleared.

    Each Protocol method calls the matching ``ops.X(...)``. Return
    values are forwarded for :meth:`analyze` (the only method whose
    return value the bridge cares about).
    """

    #: LiveOps runs in a single process and cannot drive OpenSeesMP, so
    #: it cannot consume the per-rank ``partition_open`` / ``partition_close``
    #: brackets the partitioned emit path produces.  The bridge reads this
    #: flag (default ``True`` for partition-capable emitters such as the
    #: Tcl/Py/MPI writers) and flattens a partitioned model into one domain
    #: via ``_emit_flat`` when emitting here — so a *composed* model
    #: (auto-partitioned one-rank-per-module by ADR 0038) still emits every
    #: module's nodes/elements into the single live domain and analyzes.
    supports_partitions: bool = False

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
        # Fork-only element gate (B3). Once a fork-only element (BezierTri6
        # / BezierTet10) is confirmed to actually build on the live ops, the
        # check is skipped for the rest of the session (O(1) overhead).
        self._fork_element_verified: bool = False
        # ADR 0057 Phase A: live harvest of strategy-ladder escalations —
        # one ``(label, increment, rung_index, rung_args)`` per escalation
        # (rung-0 attempts are not logged; an empty list after a laddered
        # run means the base algorithm carried every increment).
        self.strategy_events: list[
            tuple[str, int, int, tuple[int | float | str, ...]]
        ] = []

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

    def equalDOF_mixed(
        self, master: int, slave: int,
        dof_pairs: "Sequence[tuple[int, int]]",
    ) -> None:
        flat = [int(d) for pair in dof_pairs for d in pair]
        self._ops.equalDOF_Mixed(master, slave, len(dof_pairs), *flat)

    def rigidLink(self, kind: str, master: int, slave: int) -> None:
        self._ops.rigidLink(kind, master, slave)

    def rigidDiaphragm(
        self, perp_dir: int, master: int, *slaves: int,
    ) -> None:
        self._ops.rigidDiaphragm(perp_dir, master, *slaves)

    def embeddedNode(
        self, ele_tag: int, cnode: int, *master_nodes: int,
        stiffness: float = 1.0e18,
        stiffness_p: float | None = None,
        rotational: bool = False,
        pressure: bool = False,
    ) -> None:
        from .base import _build_embedded_flag_args
        flag_args = _build_embedded_flag_args(
            stiffness, stiffness_p, rotational, pressure,
        )
        self._ops.element(
            "ASDEmbeddedNodeElement", ele_tag, cnode,
            *master_nodes, *flag_args,
        )

    def embedded_rebar(
        self, ele_tag: int, *args: int | float | str,
    ) -> None:
        # LadrunoEmbeddedRebar coupling (g.reinforce, ADR 20). Routed
        # through self.element so the fork-only gate (it is in
        # _FORK_ONLY_ELEMENTS) raises a clear "requires the Ladruno fork
        # build" error on a stock OpenSees instead of a cryptic parser
        # failure.
        self.element("LadrunoEmbeddedRebar", ele_tag, *args)

    def equationConstraint(
        self, cnode: int, cdof: int, ccoef: float,
        retained: "Sequence[tuple[int, int, float]]",
    ) -> None:
        # EQ_Constraint via live openseespy (ADR 0068): one call per tied
        # DOF, flat varargs.
        flat: list[int | float] = []
        for rn, rd, rc in retained:
            flat += [int(rn), int(rd), float(rc)]
        self._ops.equationConstraint(
            int(cnode), int(cdof), float(ccoef), *flat,
        )

    def mp_constraint_comment(self, name: str) -> None:
        # No-op — live execution can't carry comments. Argument exists
        # so the Protocol shape is uniform across emitters (INV-4).
        del name

    # -- Regions -------------------------------------------------------------

    def region(self, tag: int, *args: int | float | str) -> None:
        self._ops.region(tag, *args)

    # -- Damping (ADR 0053) --------------------------------------------------

    def rayleigh(
        self,
        alpha_m: float,
        beta_k: float,
        beta_k_init: float,
        beta_k_comm: float,
    ) -> None:
        self._ops.rayleigh(alpha_m, beta_k, beta_k_init, beta_k_comm)

    def damping(
        self, damp_type: str, tag: int, *args: int | float | str,
    ) -> None:
        self._ops.damping(damp_type, tag, *args)

    def modal_damping(self, *factors: float) -> None:
        self._ops.modalDamping(*factors)

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
        # Fork-only elements (B3): verify the live build actually knows the
        # element instead of letting it fail later with a cryptic openseespy
        # error. Skipped while a non-zero partition block is open (the
        # ``_NoOpOps`` stand-in has no real domain to probe) and after the
        # first successful build.
        if (
            ele_type in _FORK_ONLY_ELEMENTS
            and not self._in_partition
            and not self._fork_element_verified
        ):
            self._element_fork_gated(ele_type, tag, args)
            return
        self._ops.element(ele_type, tag, *args)

    def _element_fork_gated(
        self, ele_type: str, tag: int, args: "tuple[int | float | str, ...]",
    ) -> None:
        """Create a fork-only element, raising a clear fork-build error if
        the live build rejects it (by raising) or silently drops it (no
        element created — stock openseespy warns and returns)."""
        try:
            self._ops.element(ele_type, tag, *args)
        except Exception as e:
            raise RuntimeError(_fork_element_required(ele_type)) from e
        try:
            tags = self._ops.getEleTags()
        except Exception:
            # Build lacks getEleTags — don't false-positive; the element
            # call did not raise, so trust it.
            self._fork_element_verified = True
            return
        if isinstance(tags, int):  # some builds return a bare int for 1 elem
            tags = [tags]
        if tag not in (tags or []):
            raise RuntimeError(_fork_element_required(ele_type))
        self._fork_element_verified = True

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

    def sp_hold(self, node: int, dof: int) -> None:
        # HOLD support (ADR 0052): in-process, the runtime displacement
        # is available now via ``nodeDisp``, so capture it directly.
        self._ops.sp(node, dof, self._ops.nodeDisp(node, dof), "-const")

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

    def constraints(self, c_type: str, *args: int | float | str) -> None:
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

    def analyze(
        self, *, steps: int, dt: float | None = None,
        label: str | None = None,
        strategy: StrategySpec | None = None,
    ) -> int:
        # ``label`` names the loop in the DECK emitters' fail-loud
        # banners; live runs in-process and reports failure through the
        # returned rc instead (the staged orchestrator raises on it).
        if strategy is not None:
            return self._analyze_ladder(
                steps=int(steps), dt=dt, label=label, strategy=strategy,
            )
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

    def _analyze_ladder(
        self, *, steps: int, dt: float | None,
        label: str | None, strategy: StrategySpec,
    ) -> int:
        # ADR 0057 Phase A: in-process rung walk, mirroring the deck
        # emitters' loop — rung 0 first every increment, escalate on a
        # failed ``analyze(1)``, restore rung 0 after a rescue.  Every
        # escalation prints a loud provenance line and is appended to
        # ``self.strategy_events`` (the live harvest); exhaustion
        # returns the failing rc (the orchestrator raises on it).
        where = f" of stage '{label}'" if label else ""
        rungs = strategy.rungs
        for i in range(steps):
            for fn in self._before_step_hooks:
                fn()
            carried = -1
            last_rc = 0
            for r, rung in enumerate(rungs):
                if r:
                    print(
                        f"apeGmsh strategy '{strategy.name}': increment "
                        f"{i + 1}/{steps}{where} -> rung {r} {rung}"
                    )
                    self._ops.algorithm(*rung)
                    self.strategy_events.append(
                        (label or "", i + 1, r, rung)
                    )
                rc: Any = (
                    self._ops.analyze(1) if dt is None
                    else self._ops.analyze(1, dt)
                )
                last_rc = int(rc)
                if last_rc == 0:
                    carried = r
                    break
            if carried < 0:
                return last_rc
            if carried > 0:
                self._ops.algorithm(*rungs[0])
            for fn in self._after_step_hooks:
                fn()
        return 0

    def eigen(
        self, num_modes: int, *, solver: str = "-genBandArpack",
    ) -> list[float]:
        # openseespy: ``ops.eigen(solver, num_modes)`` returns a list of
        # eigenvalues ``λ_i = ω_i²``. Modal shapes are queried separately
        # via ``ops.nodeEigenvector(node_tag, mode_idx)`` — see
        # :class:`apeGmsh.opensees.analysis.eigen.EigenResult`.
        values: Any = self._ops.eigen(solver, num_modes)
        return [float(v) for v in values]

    def profiler(self, *args: int | float | str) -> None:
        # The fork's ``profiler`` command exists only in the Ladruno build.
        # Stock openseespy has no ``ops.profiler`` attribute — gate on that
        # and re-raise a friendly error instead of a bare AttributeError.
        profiler_fn = getattr(self._ops, "profiler", None)
        if profiler_fn is None:
            raise RuntimeError(_PROFILER_FORK_REQUIRED)
        profiler_fn(*args)

    def ladruno_projection_tie_force(self, node: int, dof: int) -> float:
        # openseespy (Ladruno fork, ADR-30 P3): ``ops.ladrunoProjectionTieForce``
        # returns the projection constraint tie force f = M(a_raw - a_proj) at
        # (node, dof) from the last projection step — the analogue of LS-DYNA
        # ``*DATABASE_NCFORC``. ``dof`` is 1-based (the fork command converts
        # to a 0-based local index internally). Stock openseespy has no such
        # attribute — gate on it and raise a friendly error instead of a bare
        # AttributeError.
        fn = getattr(self._ops, "ladrunoProjectionTieForce", None)
        if fn is None:
            raise RuntimeError(_TIE_FORCE_FORK_REQUIRED)
        return float(fn(int(node), int(dof)))

    def critical_time_step(self) -> float:
        # openseespy (Ladruno fork): ``ops.criticalTimeStep()`` returns the
        # active explicit integrator's critical time step (dt_cr), or a
        # sentinel: ``0.0`` not-computed, ``-1.0`` not-applicable/disabled.
        # A valid value needs an explicit integrator with ``-cfl`` (or
        # ``-tangent``/``-recompute``) AND at least one prior ``analyze`` /
        # ``domainChanged`` step.
        return float(self._ops.criticalTimeStep())

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

    def domain_change(self) -> None:
        """``ops.domainChange()`` — rebuild the renumbered DOF map.

        Live emit is currently NOT staged (``stage_open`` /
        ``stage_close`` raise), so this is reachable only from a
        non-staged user-driven call (e.g. someone driving live with
        a custom topology-rebuild workflow).  Forward straight to
        openseespy.
        """
        self._ops.domainChange()

    # -- Staged-analysis mutators (Phase SSI-2.E) ---------------------------
    # Same rationale as :meth:`domain_change`: reachable only outside the
    # staged emit path (which raises at ``stage_open``).  Forwards to
    # openseespy so live callers driving a hand-rolled multi-step workflow
    # have these primitives available.

    def set_time(self, t: float) -> None:
        self._ops.setTime(float(t))

    def set_creep(self, on: bool) -> None:
        self._ops.setCreep(1 if on else 0)

    def reset(self) -> None:
        self._ops.reset()

    def remove_sp(self, node: int, dof: int) -> None:
        self._ops.remove("sp", int(node), int(dof))

    def remove_element(self, tag: int) -> None:
        self._ops.remove("element", int(tag))

    # -- Stress control (Phase SSI-1: initial_stress + ramping hooks) -------

    def addToParameter(
        self, tag: int, ele_tag: int, response: str,
    ) -> None:
        self._ops.addToParameter(tag, "element", ele_tag, response)

    def flip_element_stage(
        self, pid: int, ele_tags: tuple[int, ...],
    ) -> None:
        self._ops.parameter(int(pid))
        for et in ele_tags:
            self._ops.addToParameter(int(pid), "element", int(et), "stage")
        self._ops.updateParameter(int(pid), 1)
        self._ops.remove("parameter", int(pid))

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
