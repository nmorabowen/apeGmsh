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

from typing import TYPE_CHECKING, Any, Literal, Sequence

if TYPE_CHECKING:
    from .base import StrategySpec


class RecordingEmitter:
    """Records every emitter call as ``(name, args, kwargs)``."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    # -- Model -----------------------------------------------------------
    def model(self, *, ndm: int, ndf: int) -> None:
        self.calls.append(("model", (), {"ndm": ndm, "ndf": ndf}))

    def node(
        self, tag: int, *coords: float, ndf: int | None = None,
    ) -> None:
        kwargs: dict[str, Any] = {}
        if ndf is not None:
            kwargs["ndf"] = ndf
        self.calls.append(("node", (tag, *coords), kwargs))

    def fix(self, tag: int, *dofs: int) -> None:
        self.calls.append(("fix", (tag, *dofs), {}))

    def mass(self, tag: int, *values: float) -> None:
        self.calls.append(("mass", (tag, *values), {}))

    # -- MP constraints (ADR 0022, Phase 7b) -----------------------------

    def equalDOF(self, master: int, slave: int, *dofs: int) -> None:
        self.calls.append(("equalDOF", (master, slave, *dofs), {}))

    def equalDOF_mixed(
        self, master: int, slave: int,
        dof_pairs: "Sequence[tuple[int, int]]",
    ) -> None:
        flat = [int(d) for pair in dof_pairs for d in pair]
        self.calls.append(
            ("equalDOF_Mixed", (master, slave, len(dof_pairs), *flat), {})
        )

    def rigidLink(self, kind: str, master: int, slave: int) -> None:
        self.calls.append(("rigidLink", (kind, master, slave), {}))

    def rigidDiaphragm(
        self, perp_dir: int, master: int, *slaves: int,
    ) -> None:
        self.calls.append(
            ("rigidDiaphragm", (perp_dir, master, *slaves), {})
        )

    def embeddedNode(
        self, ele_tag: int, cnode: int, *master_nodes: int,
        stiffness: float = 1.0e18,
        stiffness_p: float | None = None,
        rotational: bool = False,
        pressure: bool = False,
    ) -> None:
        self.calls.append((
            "embeddedNode",
            (ele_tag, cnode, *master_nodes),
            {
                "stiffness": stiffness,
                "stiffness_p": stiffness_p,
                "rotational": rotational,
                "pressure": pressure,
            },
        ))

    def embedded_rebar(
        self, ele_tag: int, *args: int | float | str,
    ) -> None:
        self.calls.append(("embedded_rebar", (ele_tag, *args), {}))

    def equationConstraint(
        self, cnode: int, cdof: int, ccoef: float,
        retained: "Sequence[tuple[int, int, float]]",
    ) -> None:
        self.calls.append((
            "equationConstraint",
            (int(cnode), int(cdof), float(ccoef),
             tuple((int(rn), int(rd), float(rc)) for rn, rd, rc in retained)),
            {},
        ))

    def embedded_node(
        self, ele_tag: int, *args: int | float | str,
    ) -> None:
        self.calls.append(("embedded_node", (ele_tag, *args), {}))

    def contact_surface(
        self, tag: int, *args: int | float | str,
    ) -> None:
        self.calls.append(("contact_surface", (tag, *args), {}))

    def contact(
        self, tag: int, *args: int | float | str,
    ) -> None:
        self.calls.append(("contact", (tag, *args), {}))

    def contact_plane(
        self, tag: int, *args: int | float | str,
    ) -> None:
        self.calls.append(("contact_plane", (tag, *args), {}))

    def mp_constraint_comment(self, name: str) -> None:
        self.calls.append(("mp_constraint_comment", (name,), {}))

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

    def sp_hold(self, node: int, dof: int) -> None:
        self.calls.append(("sp_hold", (node, dof), {}))

    # -- Regions ---------------------------------------------------------
    def region(self, tag: int, *args: int | float | str) -> None:
        self.calls.append(("region", (tag, *args), {}))

    # -- Damping (ADR 0053) ----------------------------------------------
    def rayleigh(
        self,
        alpha_m: float,
        beta_k: float,
        beta_k_init: float,
        beta_k_comm: float,
    ) -> None:
        self.calls.append(
            ("rayleigh", (alpha_m, beta_k, beta_k_init, beta_k_comm), {}),
        )

    def damping(
        self, damp_type: str, tag: int, *args: int | float | str,
    ) -> None:
        self.calls.append(("damping", (damp_type, tag, *args), {}))

    def modal_damping(self, *factors: float) -> None:
        self.calls.append(("modal_damping", factors, {}))

    # -- Recorders -------------------------------------------------------
    def recorder(self, kind: str, *args: int | float | str) -> None:
        self.calls.append(("recorder", (kind, *args), {}))

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
        """Capture Phase 9 declaration-begin events for tests."""
        self.calls.append((
            "recorder_declaration_begin", (), {
                "declaration_name": declaration_name,
                "record_name": record_name,
                "category": category,
                "components": components,
                "raw": raw,
                "pg": pg,
                "label": label,
                "selection": selection,
                "ids": ids,
                "dt": dt,
                "n_steps": n_steps,
                "file_root": file_root,
            },
        ))

    def recorder_declaration_end(self) -> None:
        """Capture Phase 9 declaration-end events for tests."""
        self.calls.append(("recorder_declaration_end", (), {}))

    # -- Analysis chain --------------------------------------------------
    def constraints(self, c_type: str, *args: int | float | str) -> None:
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

    def analyze(
        self, *, steps: int, dt: float | None = None,
        label: str | None = None,
        strategy: "StrategySpec | None" = None,
    ) -> int:
        kwargs: dict[str, Any] = {"steps": steps, "dt": dt}
        if label is not None:
            kwargs["label"] = label
        if strategy is not None:
            kwargs["strategy"] = strategy
        self.calls.append(("analyze", (), kwargs))
        return 0

    def eigen(
        self, num_modes: int, *, solver: str = "-genBandArpack",
    ) -> list[float]:
        self.calls.append(
            ("eigen", (), {"num_modes": num_modes, "solver": solver}),
        )
        return []

    def profiler(self, *args: int | float | str) -> None:
        self.calls.append(("profiler", tuple(args), {}))

    # -- Partition emission scoping (ADR 0027, P4) -----------------------

    def partition_open(self, rank: int) -> None:
        self.calls.append(("partition_open", (rank,), {}))

    def partition_close(self) -> None:
        self.calls.append(("partition_close", (), {}))

    # -- Partition runtime-conditional fallback (ADR 0027 INV-5) ---------

    def parallel_runtime_fallback_numberer(
        self, primary: str, fallback: str,
    ) -> None:
        self.calls.append(
            ("parallel_runtime_fallback_numberer", (primary, fallback), {}),
        )

    def parallel_runtime_fallback_system(
        self, primary: str, fallback: str,
    ) -> None:
        self.calls.append(
            ("parallel_runtime_fallback_system", (primary, fallback), {}),
        )

    # -- Stress control (Phase SSI-1: initial_stress + ramping hooks) ----

    def addToParameter(
        self, tag: int, ele_tag: int, response: str,
    ) -> None:
        self.calls.append(
            ("addToParameter", (tag, ele_tag, response), {})
        )

    def flip_element_stage(
        self, pid: int, ele_tags: tuple[int, ...],
    ) -> None:
        self.calls.append(
            ("flip_element_stage", (pid, tuple(ele_tags)), {})
        )

    def step_hook_ramp(
        self,
        name: str,
        *,
        targets: tuple[tuple[int, float], ...],
        n_steps_to_full: float,
        phase: Literal["before", "after"] = "before",
    ) -> None:
        self.calls.append((
            "step_hook_ramp",
            (name,),
            {
                "targets": tuple(targets),
                "n_steps_to_full": n_steps_to_full,
                "phase": phase,
            },
        ))

    # -- Staged analysis (Phase SSI-2.A) ------------------------------------

    def stage_open(self, name: str) -> None:
        self.calls.append(("stage_open", (name,), {}))

    def stage_close(self) -> None:
        self.calls.append(("stage_close", (), {}))

    def domain_change(self) -> None:
        self.calls.append(("domain_change", (), {}))

    # -- Staged-analysis mutators (Phase SSI-2.E) ---------------------------

    def set_time(self, t: float) -> None:
        self.calls.append(("set_time", (float(t),), {}))

    def set_creep(self, on: bool) -> None:
        self.calls.append(("set_creep", (bool(on),), {}))

    def reset(self) -> None:
        self.calls.append(("reset", (), {}))

    def remove_sp(self, node: int, dof: int) -> None:
        self.calls.append(("remove_sp", (int(node), int(dof)), {}))

    def remove_element(self, tag: int) -> None:
        self.calls.append(("remove_element", (int(tag),), {}))
