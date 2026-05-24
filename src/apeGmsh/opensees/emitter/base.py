"""
Emitter Protocol — the frozen interface every emit target satisfies.

The Protocol is **load-bearing for every primitive's _emit method**.
After Phase 0 it is read-only: adding or removing a method is an
architecture event that requires coordination across all primitives
and all concrete emitters (TclEmitter, PyEmitter, LiveOpsEmitter,
H5Emitter, RecordingEmitter).

Per P12, user-facing signatures forbid ``**kwargs`` and positional
``*args``. The Protocol is **internal** — it sits between primitives
and the OpenSees vocabulary, which inherently takes variadic tail
parameters. The carve-out is documented in ADR 0008.

Note on runtime checking: the Protocol is **not** marked
``@runtime_checkable``. ``isinstance()`` checks against Protocols
that contain ``*args`` / ``**kwargs`` are unreliable; type-only use
is sufficient (mypy / pyright check static conformance).

The deliberate ``*_open`` / ``*_close`` pairs (``section_open`` /
``section_close``, ``pattern_open`` / ``pattern_close``) bridge the
Tcl curly-brace block dialect against openseespy's stateful
"current section / pattern" dialect. Each concrete emitter handles
its own dialect.

**Architecture event — ADR 0022 (Phase 7b, May 2026).** The Protocol
was widened with five MP-constraint methods (``equalDOF``,
``rigidLink``, ``rigidDiaphragm``, ``embeddedNode``,
``mp_constraint_comment``) closing the §3.3 deferral so
``apeSees(fem).tcl(p)`` finally produces a runnable deck for models
declaring ``g.constraints.rigid_diaphragm(...)`` /
``g.constraints.tied_contact(...)`` etc. The H5 emitter additionally
gained an ``ndf=`` kwarg on :meth:`node` (additive, default ``None``)
to express the per-node DOF override used for the 6-DOF phantom nodes
in mixed-ndf models.

**Architecture event — ADR 0024 (late-May 2026).** The Protocol was
widened with one new method (:meth:`region`) so the MPCO recorder can
filter its output via OpenSees ``region $tag -node ... -ele ...`` +
``-R $tag``.  Auto-emitted by the build pipeline when
``ops.recorder.MPCO(nodes_pg=..., elements_pg=...)`` is declared.
Schema bumped 2.7.0 → 2.8.0 for the new ``/opensees/regions/`` zone.

**Architecture event — ADR 0025 (late-May 2026).** The Protocol was
widened with one new method (:meth:`eigen`) so the bridge can drive
one-shot modal extractions via OpenSees ``eigen [solver] $numModes``.
Unlike the stepped ``analyze`` driver, ``eigen`` requires no
preceding ``analysis <Type>`` chain and returns eigenvalues directly
to the caller — the live emitter returns ``list[float]`` while Tcl /
py emit the line and return an empty list; H5 / recording are no-op
/ record.  Driven by :meth:`apeGmsh.opensees.apeSees.eigen`, a bridge
driver method parallel to ``analyze``; wraps the eigenvalues in an
:class:`apeGmsh.opensees.analysis.eigen.EigenResult` carrying derived
``omega`` / ``freq`` / ``periods`` and a lazy ``mode_shape(node,
mode)`` accessor over ``ops.nodeEigenvector``.  No schema bump — the
H5 emitter no-ops on ``eigen`` because the call is a runtime
retrieval, not a model-definition declaration.

**Architecture event — ADR 0027 INV-5 amendment (May 2026).** The
Protocol was widened with two runtime-conditional fallback methods
(:meth:`parallel_runtime_fallback_numberer`,
:meth:`parallel_runtime_fallback_system`) that emit a
primary / fallback pair so the same partitioned deck runs under both
OpenSeesMP (``ParallelPlain`` + ``Mumps``) and single-process
OpenSees (``RCM`` + ``UmfPack``).  Tcl wraps the primary in
``catch``; Py wraps it in ``try / except``; LiveOps invokes the
primary and falls back on exception with a ``UserWarning``; H5
records the primary as canonical and stores a ``runtime_fallback``
attribute; Recording captures the pair.  Restores end-to-end
shim-consistency with the ``proc getPID`` partition-open shim.

**Architecture event — ADR 0027 (P4, May 2026).** The Protocol was
widened with two emission-scoping methods (:meth:`partition_open`,
:meth:`partition_close`) that bracket a per-rank emission block.
Tcl wraps the block in ``if {[getPID] == K} { ... }``; Py wraps it
in ``if getPID() == K: ...``; LiveOps treats itself as rank 0 and
suppresses emission on non-zero rank blocks; H5 collects per-rank
content into a per-partition sub-group under ``/opensees/partitions/``;
Recording captures both ``partition_open(rank)`` and
``partition_close()`` events for tests. The Tcl / Py emitters also
inject a one-shot runtime shim on the first ``partition_open`` call
so single-process OpenSees still runs the deck: Tcl declares
``proc getPID {} { return 0 }`` if not already present; Py wraps
``from openseespy.opensees import getPID`` in a ``try / except
ImportError`` with a fallback ``def getPID(): return 0``.  Schema
bump opensees ``2.9.0 → 2.10.0`` per ADR 0023 — additive minor.

**Architecture event — Phase SSI-1 (initial_stress, May 2026).** The
Protocol was widened with two stress-control methods,
:meth:`addToParameter` and :meth:`step_hook_ramp`, that materialize
the STKO-style initial-stress-injection-via-committed-stress-increment
pattern.  ``addToParameter`` is a single-line per-rank-scoped
directive; ``step_hook_ramp`` is a multi-line bundle (dispatcher
init + parameter declarations + per-step proc + lappend
registration) emitted globally.  Once any ``step_hook_ramp`` has
been emitted, the emitter's :meth:`analyze` MUST wrap the analyze
loop with hook-dispatcher calls so the ramp actually advances
between steps.  H5 archival is deferred (no schema bump in Phase
SSI-1); H5 / Recording / Live emitters implement the methods as
no-ops / capture-only / in-process closures respectively.

**Architecture event — Phase SSI-2.A (staged analysis, May 2026).**
The Protocol gained two stage-bracketing methods,
:meth:`stage_open` and :meth:`stage_close`, that frame a per-stage
analysis block in multi-stage decks.  ``stage_open`` emits a
human-readable comment delimiter (``# === Stage: <name> ===``);
``stage_close`` emits the canonical between-stages cleanup:
``loadConst -time 0.0`` + ``wipeAnalysis`` + (if step hooks are
registered) clears the ``_apesees_before_step_hooks`` /
``_apesees_after_step_hooks`` dispatcher lists so the next stage's
hooks fire on the right step counters.  The emitted ``proc``
definitions persist across ``stage_close`` — only the lappend
lists are reset, not the proc bodies themselves.  H5 / Recording
emitters no-op / capture for tests; Live raises
``NotImplementedError`` (staged live execution deferred — the
bridge's :meth:`apeSees.analyze` only supports non-staged models).

**Architecture event — Phase SSI-2.B (per-stage topology activation,
May 2026).** The Protocol gained :meth:`domain_change`, which emits
the OpenSees ``domainChange`` command.  Staged decks need this
after activating new elements / nodes mid-analysis so OpenSees
rebuilds its renumbered DOF map.  Tcl emits ``domainChange``;
Py emits ``ops.domainChange()``; Live calls ``ops.domainChange()``
in-process; H5 no-ops (no model-state change to archive);
Recording captures for tests.
"""
from __future__ import annotations

from typing import Literal, Protocol


class Emitter(Protocol):
    """Frozen Protocol covering every OpenSees command the bridge emits.

    See ``architecture/emitter.md`` for the full rationale and the
    matrix of how each method maps to Tcl, openseespy, and live calls.
    """

    # -- Model -----------------------------------------------------------
    def model(self, *, ndm: int, ndf: int) -> None: ...
    def node(
        self, tag: int, *coords: float, ndf: int | None = None,
    ) -> None: ...
    def fix(self, tag: int, *dofs: int) -> None: ...
    def mass(self, tag: int, *values: float) -> None: ...

    # -- MP constraints (ADR 0022, Phase 7b) -----------------------------
    # Five methods closing the §3.3 deferral. Build-time fan-out in
    # ``opensees._internal.build.emit_mp_constraints`` calls these after
    # element emission and before pattern emission (INV-5). Phantom
    # nodes from ``NodeToSurfaceRecord`` are emitted via ``node(...,
    # ndf=6)`` before any constraint references them (INV-3).
    def equalDOF(self, master: int, slave: int, *dofs: int) -> None: ...
    def rigidLink(
        self, kind: Literal["beam", "bar"], master: int, slave: int,
    ) -> None: ...
    def rigidDiaphragm(
        self, perp_dir: int, master: int, *slaves: int,
    ) -> None: ...
    def embeddedNode(
        self, ele_tag: int, cnode: int,
        *args: int | float,
    ) -> None: ...
    def mp_constraint_comment(self, name: str) -> None: ...

    # -- Constitutive ----------------------------------------------------
    def uniaxialMaterial(
        self, mat_type: str, tag: int, *params: float | str
    ) -> None: ...
    def nDMaterial(
        self, mat_type: str, tag: int, *params: float | str
    ) -> None: ...
    def section(
        self, sec_type: str, tag: int, *params: float | str
    ) -> None: ...
    def geomTransf(
        self, t_type: str, tag: int, *vec: float
    ) -> None: ...

    # -- Sections that take blocks (Fiber) -------------------------------
    def section_open(
        self, sec_type: str, tag: int, *params: float | str
    ) -> None: ...
    def section_close(self) -> None: ...
    def patch(self, kind: str, *args: int | float) -> None: ...
    def fiber(
        self, y: float, z: float, area: float, mat_tag: int
    ) -> None: ...
    def layer(self, kind: str, *args: int | float) -> None: ...

    # -- Beam integration rules ------------------------------------------
    # Single-line; no block. References its constituent sections by tag,
    # not by composition. Beam-column elements then reference this
    # integration rule's tag rather than carrying ``section`` + ``n_ip``
    # directly — this mirrors modern OpenSees and is what openseespy
    # requires for ``forceBeamColumn`` / ``dispBeamColumn`` to parse.
    def beamIntegration(
        self, rule_type: str, tag: int, *args: int | float | str
    ) -> None: ...

    # -- Topology --------------------------------------------------------
    def element(
        self, ele_type: str, tag: int, *args: int | float | str
    ) -> None: ...

    # -- Time series -----------------------------------------------------
    def timeSeries(
        self, ts_type: str, tag: int, *args: int | float | str
    ) -> None: ...

    # -- Patterns (Tcl wants a block; py wants a stateful current) ------
    def pattern_open(
        self, p_type: str, tag: int, *args: int | float | str
    ) -> None: ...
    def pattern_close(self) -> None: ...
    def load(self, tag: int, *forces: float) -> None: ...
    def eleLoad(self, *args: int | float | str) -> None: ...
    def sp(self, tag: int, dof: int, value: float) -> None: ...

    # -- Regions ---------------------------------------------------------
    # ``region`` declares a named OpenSees region (a tagged collection of
    # nodes and/or elements) that other commands can reference. Today
    # the bridge emits it from the recorder fan-out to filter MPCO
    # output via ``-R $regTag`` (per the mpco-recorder skill: MPCO
    # records the whole model unless an explicit region filter is
    # supplied). The ``args`` tail carries the raw OpenSees flag
    # sequence (``-node n1 n2 ...``, ``-ele e1 e2 ...``, ``-eleOnly``,
    # ``-nodeOnly``, ``-eleRange``, etc.) — see the OpenSees manual.
    def region(self, tag: int, *args: int | float | str) -> None: ...

    # -- Recorders -------------------------------------------------------
    def recorder(self, kind: str, *args: int | float | str) -> None: ...

    # -- Recorder declaration archival (Phase 9 schema 2.3.0) ------------
    # These two methods bracket the file-emit fan-out of a single
    # :class:`apeGmsh.opensees.recorder.RecorderRecord`. Every
    # :meth:`recorder` call issued between ``recorder_declaration_begin``
    # and ``recorder_declaration_end`` is associated with the same
    # declaration metadata; emitters that archive model state (the H5
    # emitter, schema 2.3.0+) persist that metadata alongside each
    # fan-out call. Tcl / py / live / recording emitters implement
    # both methods as no-ops — they don't archive declaration intent,
    # only the OpenSees commands themselves.
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
    ) -> None: ...

    def recorder_declaration_end(self) -> None: ...

    # -- Analysis chain --------------------------------------------------
    def constraints(self, c_type: str, *args: int | float | str) -> None: ...
    def numberer(self, n_type: str) -> None: ...
    def system(self, s_type: str, *args: int | float | str) -> None: ...
    def test(self, t_type: str, *args: int | float | str) -> None: ...
    def algorithm(self, a_type: str, *args: int | float | str) -> None: ...
    def integrator(self, i_type: str, *args: int | float | str) -> None: ...
    def analysis(self, a_type: str) -> None: ...

    # Behavior change (Phase SSI-1): once :meth:`step_hook_ramp` has
    # been called on this emitter, ``analyze`` MUST wrap the analyze
    # loop with per-step hook-dispatcher calls so registered ramps
    # actually advance between steps.  Tcl emits ``for`` + dispatcher
    # invocations; Py emits ``for`` + dispatcher invocations; LiveOps
    # runs the loop in-process and invokes the captured closures.
    def analyze(self, *, steps: int, dt: float | None = None) -> int: ...

    # -- Stress control (Phase SSI-1: initial_stress + ramping hooks) ----
    # ``addToParameter`` attaches one element's response to a previously
    # declared parameter.  Emitted per-rank inside ``partition_open``
    # blocks for MP-partitioned models — each rank emits only the
    # addToParameter calls for elements it owns.  The parameter
    # declarations themselves come from :meth:`step_hook_ramp` (global).
    def addToParameter(
        self, tag: int, ele_tag: int, response: str,
    ) -> None: ...

    # ``step_hook_ramp`` emits the multi-line bundle that materializes
    # one ``InitialStress`` composite into the deck:
    #
    # 1. Dispatcher boilerplate (idempotent — emitted once across the
    #    emitter's lifetime).  Includes the
    #    ``_apesees_before_step_hooks`` / ``_apesees_after_step_hooks``
    #    list and the ``_apesees_call_before_step`` /
    #    ``_apesees_call_after_step`` dispatcher procs.
    # 2. ``parameter $tag`` declaration for each tag in ``targets``.
    # 3. The per-step procedure body.  Computes
    #    ``factor = min(count / n_steps_to_full, 1.0)`` then emits
    #    one ``updateParameter $param_tag $delta`` per target, where
    #    ``$delta = target_value * factor - previous_cumulative``.
    # 4. Registration via ``lappend`` to the dispatcher list selected
    #    by ``phase``.
    #
    # The ``lambda_install`` parameter of the user-facing
    # :func:`apeSees.initial_stress` is baked into ``target_value``
    # (target_value = sigma * lambda_install), so the proc body
    # always ramps factor 0 → 1.0.
    def step_hook_ramp(
        self,
        name: str,
        *,
        targets: tuple[tuple[int, float], ...],
        n_steps_to_full: float,
        phase: Literal["before", "after"] = "before",
    ) -> None: ...

    # -- Staged analysis (Phase SSI-2.A) -----------------------------------
    # ``stage_open(name)`` emits a human-readable comment delimiter
    # so the multi-stage deck stays grep-friendly.  ``stage_close()``
    # emits the canonical between-stages cleanup: ``loadConst -time
    # 0.0`` + ``wipeAnalysis`` + (if any step hook was registered)
    # clears the dispatcher lists so the next stage's hooks fire on
    # the right step counters.  Proc definitions persist across
    # ``stage_close`` — only the ``lappend`` lists are reset.
    def stage_open(self, name: str) -> None: ...

    def stage_close(self) -> None: ...

    # -- Topology activation (Phase SSI-2.B) -------------------------------
    # ``domain_change()`` emits the OpenSees ``domainChange`` command,
    # which tells OpenSees to rebuild its internal DOF map after the
    # set of active nodes / elements has changed.  Staged decks call
    # this after a stage's element activation block, before its
    # analysis chain emits.
    def domain_change(self) -> None: ...

    # -- Eigen (one-shot, returns values from live emitter) ---------------
    # Issues ``eigen [solver] $numModes`` — does not require an
    # ``analysis <Type>`` chain.  The live emitter returns the list of
    # eigenvalues ``λ_i = ω_i²`` from openseespy; Tcl / py emit the line
    # and return an empty list; h5 / recording archive / record the call.
    def eigen(
        self, num_modes: int, *, solver: str = "-genBandArpack",
    ) -> list[float]: ...

    # -- Partition emission scoping (ADR 0027, P4) -----------------------
    # Bracket a per-rank emission block. Every emit call between
    # ``partition_open(K)`` and the matching ``partition_close()`` is
    # scoped to rank ``K``. Tcl wraps the block in
    # ``if {[getPID] == K} { ... }``; Py wraps it in
    # ``if getPID() == K: ...``; LiveOps treats itself as rank 0 and
    # suppresses emission on non-zero rank blocks; H5 collects per-rank
    # content into a per-partition sub-group; Recording captures both
    # events. The Tcl / Py emitters inject a one-shot runtime shim on
    # the first ``partition_open`` so single-process OpenSees still runs
    # the deck — Tcl defines ``proc getPID {} { return 0 }`` and Py
    # defines a ``getPID()`` fallback wrapped in a ``try / except
    # ImportError`` from ``openseespy.opensees``.
    def partition_open(self, rank: int) -> None:
        """Open a per-rank emission block.

        All subsequent emit calls until the matching
        :meth:`partition_close` belong to the specified ``rank``.
        Implementations MUST be idempotent w.r.t. preamble emission:
        the partition-runtime shim (defining ``getPID`` fallback) is
        emitted on the first ``partition_open`` call only.
        """
        ...

    def partition_close(self) -> None:
        """Close the current per-rank emission block."""
        ...

    # -- Partition runtime-conditional fallback (ADR 0027 INV-5, amended) --
    # Emit a runtime conditional so the same deck runs under both
    # OpenSeesMP (uses ``primary``, e.g. ``ParallelPlain`` / ``Mumps``)
    # and single-process OpenSees (falls back to ``fallback``, e.g.
    # ``RCM`` / ``UmfPack``).  Tcl wraps the primary call in a Tcl
    # ``catch`` block; Py wraps it in a Python ``try / except``;
    # LiveOps actually invokes openseespy and catches the exception in-
    # process (firing a ``UserWarning`` on fallback); H5 records the
    # primary choice as the canonical numberer / system and stores a
    # ``runtime_fallback`` attribute documenting the fallback;
    # Recording captures the pair as ``(primary, fallback)``.  This
    # restores end-to-end shim-consistency with the ``proc getPID``
    # single-process fallback in :meth:`partition_open` (ADR 0027 INV-5
    # amendment 2026-05-23).
    def parallel_runtime_fallback_numberer(
        self, primary: str, fallback: str,
    ) -> None:
        """Emit a runtime-conditional numberer (primary / fallback).

        The primary numberer is tried at runtime; on failure (e.g.
        ``ParallelPlain`` not registered under single-process OpenSees),
        the fallback numberer is used instead.
        """
        ...

    def parallel_runtime_fallback_system(
        self, primary: str, fallback: str,
    ) -> None:
        """Emit a runtime-conditional system of equations (primary / fallback).

        The primary system is tried at runtime; on failure (e.g.
        ``Mumps`` not compiled into a single-process OpenSees build),
        the fallback system is used instead.
        """
        ...
