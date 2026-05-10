"""
RecordingEmitter — captures every Emitter call as a structured record.

Used exclusively by tests. Each Protocol method appends a tuple of
``(method_name, positional_args, keyword_args)`` to ``self.calls``.

This is the only emitter that does NOT serialize anywhere — it lives
in memory, is constructible with no arguments, and never imports
``openseespy``.

The boilerplate is written out by hand rather than synthesized via a
metaclass. Per the project's "no clever code" stance, explicit beats
implicit here: anyone debugging a primitive's ``_emit`` can read this
file and see exactly what gets recorded.
"""
from __future__ import annotations

from typing import Any


class RecordingEmitter:
    """Records every emitter call as ``(name, args, kwargs)``."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    # -- Model -----------------------------------------------------------
    def model(self, *, ndm: int, ndf: int) -> None:
        self.calls.append(("model", (), {"ndm": ndm, "ndf": ndf}))

    def node(self, tag: int, *coords: float) -> None:
        self.calls.append(("node", (tag, *coords), {}))

    def fix(self, tag: int, *dofs: int) -> None:
        self.calls.append(("fix", (tag, *dofs), {}))

    def mass(self, tag: int, *values: float) -> None:
        self.calls.append(("mass", (tag, *values), {}))

    # -- Constitutive ----------------------------------------------------
    def uniaxialMaterial(
        self, mat_type: str, tag: int, *params: float | str
    ) -> None:
        self.calls.append(
            ("uniaxialMaterial", (mat_type, tag, *params), {})
        )

    def nDMaterial(
        self, mat_type: str, tag: int, *params: float | str
    ) -> None:
        self.calls.append(("nDMaterial", (mat_type, tag, *params), {}))

    def section(
        self, sec_type: str, tag: int, *params: float | str
    ) -> None:
        self.calls.append(("section", (sec_type, tag, *params), {}))

    def geomTransf(self, t_type: str, tag: int, *vec: float) -> None:
        self.calls.append(("geomTransf", (t_type, tag, *vec), {}))

    # -- Sections that take blocks ---------------------------------------
    def section_open(
        self, sec_type: str, tag: int, *params: float | str
    ) -> None:
        self.calls.append(("section_open", (sec_type, tag, *params), {}))

    def section_close(self) -> None:
        self.calls.append(("section_close", (), {}))

    def patch(self, kind: str, *args: int | float) -> None:
        self.calls.append(("patch", (kind, *args), {}))

    def fiber(
        self, y: float, z: float, area: float, mat_tag: int
    ) -> None:
        self.calls.append(("fiber", (y, z, area, mat_tag), {}))

    def layer(self, kind: str, *args: int | float) -> None:
        self.calls.append(("layer", (kind, *args), {}))

    # -- Beam integration rules ------------------------------------------
    def beamIntegration(
        self, rule_type: str, tag: int, *args: int | float | str
    ) -> None:
        self.calls.append(("beamIntegration", (rule_type, tag, *args), {}))

    # -- Topology --------------------------------------------------------
    def element(
        self, ele_type: str, tag: int, *args: int | float | str
    ) -> None:
        self.calls.append(("element", (ele_type, tag, *args), {}))

    # -- Time series -----------------------------------------------------
    def timeSeries(
        self, ts_type: str, tag: int, *args: int | float | str
    ) -> None:
        self.calls.append(("timeSeries", (ts_type, tag, *args), {}))

    # -- Patterns --------------------------------------------------------
    def pattern_open(
        self, p_type: str, tag: int, *args: int | float | str
    ) -> None:
        self.calls.append(("pattern_open", (p_type, tag, *args), {}))

    def pattern_close(self) -> None:
        self.calls.append(("pattern_close", (), {}))

    def load(self, tag: int, *forces: float) -> None:
        self.calls.append(("load", (tag, *forces), {}))

    def eleLoad(self, *args: int | float | str) -> None:
        self.calls.append(("eleLoad", tuple(args), {}))

    def sp(self, tag: int, dof: int, value: float) -> None:
        self.calls.append(("sp", (tag, dof, value), {}))

    # -- Recorders -------------------------------------------------------
    def recorder(self, kind: str, *args: int | float | str) -> None:
        self.calls.append(("recorder", (kind, *args), {}))

    # -- Analysis chain --------------------------------------------------
    def constraints(self, c_type: str, *args: float) -> None:
        self.calls.append(("constraints", (c_type, *args), {}))

    def numberer(self, n_type: str) -> None:
        self.calls.append(("numberer", (n_type,), {}))

    def system(self, s_type: str, *args: int | float | str) -> None:
        self.calls.append(("system", (s_type, *args), {}))

    def test(self, t_type: str, *args: int | float | str) -> None:
        self.calls.append(("test", (t_type, *args), {}))

    def algorithm(self, a_type: str, *args: int | float | str) -> None:
        self.calls.append(("algorithm", (a_type, *args), {}))

    def integrator(self, i_type: str, *args: int | float | str) -> None:
        self.calls.append(("integrator", (i_type, *args), {}))

    def analysis(self, a_type: str) -> None:
        self.calls.append(("analysis", (a_type,), {}))

    def analyze(self, *, steps: int, dt: float | None = None) -> int:
        self.calls.append(("analyze", (), {"steps": steps, "dt": dt}))
        return 0
