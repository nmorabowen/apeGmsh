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
        self, ele_tag: int, embedding_ele: int,
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
    def constraints(self, c_type: str, *args: float) -> None: ...
    def numberer(self, n_type: str) -> None: ...
    def system(self, s_type: str, *args: int | float | str) -> None: ...
    def test(self, t_type: str, *args: int | float | str) -> None: ...
    def algorithm(self, a_type: str, *args: int | float | str) -> None: ...
    def integrator(self, i_type: str, *args: int | float | str) -> None: ...
    def analysis(self, a_type: str) -> None: ...
    def analyze(self, *, steps: int, dt: float | None = None) -> int: ...
