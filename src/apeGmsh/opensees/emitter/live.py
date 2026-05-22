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

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from types import ModuleType


__all__ = ["LiveOpsEmitter"]


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
        if dt is None:
            ret: Any = self._ops.analyze(steps)
        else:
            ret = self._ops.analyze(steps, dt)
        return int(ret)

    def eigen(
        self, num_modes: int, *, solver: str = "-genBandArpack",
    ) -> list[float]:
        # openseespy: ``ops.eigen(solver, num_modes)`` returns a list of
        # eigenvalues ``λ_i = ω_i²``. Modal shapes are queried separately
        # via ``ops.nodeEigenvector(node_tag, mode_idx)`` — see
        # :class:`apeGmsh.opensees.analysis.eigen.EigenResult`.
        values: Any = self._ops.eigen(solver, num_modes)
        return [float(v) for v in values]

    # -- Direct accessor for tests / diagnostics ----------------------------

    @property
    def ops(self) -> "ModuleType":
        """Return the openseespy module — lets live-mode users query state.

        Useful for ``ret = ops_emitter.ops.nodeDisp(2, 1)`` after an
        analysis runs.
        """
        return self._ops
