"""The ``VERBS`` table: one row per Emitter verb (ADR 0114).

Data only. This module imports nothing from apeGmsh, so the AST lock in
``tests/opensees/contract/test_verbs_lock.py`` and any Python-free
reader can trust it without importing the bridge.

Each row records what a verb does **today**; later K1 PRs move rows,
and the lock pins the moves:

``via``
    ``"protocol"``: a method on ``base.py::Emitter``. ``"command"``: a
    token routed through ``Emitter.command()``, which is itself the
    last Protocol method (K1-2). The ``command`` rows are the channel's
    allow-list (``base.command_row``): a token without one raises on
    every emitter. None ships yet; K4 moves typed fork verbs onto it.
``family``
    The kind of model fact the verb states; one of :data:`FAMILIES`.
``scope``
    Where the archive places the record today: ``"global"`` (under
    ``/opensees``), ``"stage"`` (under ``/opensees/stages/stage_{k}``),
    or ``"both"``. For a ``refuse`` or ``ledger`` row, which writes
    nothing, it is where the bridge emits the verb.
``h5``
    What ``H5Emitter`` does with the call today. ``"archive"``: the
    call's information reaches ``model.h5``. ``"refuse"``: the call
    raises, so ``apeSees.h5()`` fails loud. ``"ledger"``: the call is
    dropped silently (in at least one scope). The ``ledger`` rows are
    K2's xfail ledger; their count may only shrink
    (``tests/opensees/contract/verbs_ledger.txt``).
``store``
    The archive path template, ``""`` when nothing is written. ``{scope}``
    expands to ``/opensees`` or ``/opensees/stages/stage_{k:03d}``;
    ``@name`` names an attribute; ``|`` separates the global and the
    stage store where their shapes differ.
``requires``
    Capability tokens a solving backend must have (``"fork"``: the
    Ladruno build). Value-dependent requirements, such as the
    ``LadrunoContact`` handler token of ``constraints``, are not here.
``returns``
    The return annotation on ``base.py::Emitter``, verbatim.
``seq_op`` / ``seq_what``
    The ``/sequence.changes`` row the verb yields (ADR 0112, #1283
    decision 9), ``""`` when it yields none. Verbs that fill the
    ``/sequence.stages`` columns (the chain, ``analyze``, ``set_time``,
    ``domain_change``) carry ``""``. V3b locks this mapping.
``decl``
    The call opens a record that K1-6 keys by declaration
    (``<zone>/<family>/<name|#k>``).
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal

Via = Literal["protocol", "command"]
Scope = Literal["global", "stage", "both"]
H5 = Literal["archive", "refuse", "ledger"]

#: The families a row may name.
FAMILIES: Final[frozenset[str]] = frozenset({
    "model", "node", "bc", "constraint", "embed", "contact",
    "material", "section", "transform", "integration", "element",
    "time_series", "pattern", "load", "region", "damping", "recorder",
    "analysis", "parameter", "stage", "domain", "modal", "profiler",
    "partition",
    # The command channel itself; the fact a token states is its own row's.
    "channel",
})

#: The ``op`` and ``what`` vocabularies of ``/sequence.changes``.
SEQ_OPS: Final[frozenset[str]] = frozenset(
    {"", "add", "remove", "activate", "set"})
SEQ_WHATS: Final[frozenset[str]] = frozenset({
    "", "pattern", "recorder", "fix", "mass", "element", "group",
    "region", "rayleigh", "initial_stress", "absorbing",
    "material_stage", "constraint",
})


@dataclass(frozen=True, slots=True)
class Verb:
    """One row of :data:`VERBS`."""

    verb: str
    via: Via
    family: str
    scope: Scope
    h5: H5
    store: str
    requires: frozenset[str]
    returns: str
    seq_what: str
    seq_op: str
    decl: bool


_NONE: Final[frozenset[str]] = frozenset()
_FORK: Final[frozenset[str]] = frozenset({"fork"})
_S: Final[str] = "/opensees/stages/stage_{k:03d}"
#: The generic store of K1-4 (opensees 2.23.0): one row per call, with a
#: ``stage`` column (``-1`` global).
_CMD: Final[str] = "/opensees/commands"


def _p(
    verb: str, family: str, scope: Scope, h5: H5, store: str, *,
    requires: frozenset[str] = _NONE, returns: str = "None",
    seq: tuple[str, str] = ("", ""), decl: bool = False,
) -> Verb:
    return Verb(
        verb=verb, via="protocol", family=family, scope=scope, h5=h5,
        store=store, requires=requires, returns=returns,
        seq_op=seq[0], seq_what=seq[1], decl=decl,
    )


_ROWS: Final[tuple[Verb, ...]] = (
    # -- Model ------------------------------------------------------------
    _p("model", "model", "global", "archive", "/meta@ndm"),
    # Real nodes live in the neutral zone; /opensees keeps only the
    # phantom, partition and stage-owned tag lists.
    _p("node", "node", "both", "archive", "/nodes"),
    _p("fix", "bc", "both", "archive", "{scope}/bcs/fix",
       seq=("add", "fix"), decl=True),
    _p("mass", "bc", "both", "archive", "{scope}/bcs/mass",
       seq=("add", "mass"), decl=True),
    # -- MP constraints ---------------------------------------------------
    _p("equalDOF", "constraint", "both", "archive",
       "{scope}/constraints/equalDOF", seq=("add", "constraint"), decl=True),
    _p("equalDOF_mixed", "constraint", "both", "refuse", "",
       seq=("add", "constraint"), decl=True),
    _p("rigidLink", "constraint", "both", "archive",
       "{scope}/constraints/rigidLink", seq=("add", "constraint"), decl=True),
    _p("rigidDiaphragm", "constraint", "both", "archive",
       "{scope}/constraints/rigidDiaphragm", seq=("add", "constraint"),
       decl=True),
    _p("embeddedNode", "constraint", "both", "archive",
       "{scope}/constraints/embeddedNode", seq=("add", "constraint"),
       decl=True),
    _p("embedded_rebar", "embed", "global", "ledger", "",
       requires=_FORK, seq=("add", "constraint"), decl=True),
    _p("equationConstraint", "constraint", "both", "ledger", "",
       seq=("add", "constraint"), decl=True),
    _p("embedded_node", "embed", "global", "ledger", "",
       requires=_FORK, seq=("add", "constraint"), decl=True),
    _p("contact_surface", "contact", "global", "ledger", "",
       requires=_FORK, decl=True),
    _p("contact", "contact", "global", "ledger", "",
       requires=_FORK, seq=("add", "constraint"), decl=True),
    _p("contact_plane", "contact", "global", "ledger", "",
       requires=_FORK, seq=("add", "constraint"), decl=True),
    # Names the next constraint row (its ``name`` field).
    _p("mp_constraint_comment", "constraint", "both", "archive",
       "{scope}/constraints/{kind}"),
    # -- Constitutive -----------------------------------------------------
    _p("uniaxialMaterial", "material", "global", "archive",
       "/opensees/materials/uniaxial/{type}_{tag}", decl=True),
    _p("nDMaterial", "material", "global", "archive",
       "/opensees/materials/nd/{type}_{tag}", decl=True),
    _p("section", "section", "global", "archive",
       "/opensees/sections/{type}_{tag}", decl=True),
    _p("geomTransf", "transform", "global", "archive",
       "/opensees/transforms/{type}_{tag}", decl=True),
    # -- Fiber sections ---------------------------------------------------
    _p("section_open", "section", "global", "archive",
       "/opensees/sections/{type}_{tag}", decl=True),
    _p("section_close", "section", "global", "archive",
       "/opensees/sections/{type}_{tag}"),
    _p("patch", "section", "global", "archive",
       "/opensees/sections/{type}_{tag}/patches"),
    _p("fiber", "section", "global", "archive",
       "/opensees/sections/{type}_{tag}/fibers"),
    _p("layer", "section", "global", "archive",
       "/opensees/sections/{type}_{tag}/layers"),
    _p("beamIntegration", "integration", "global", "archive",
       "/opensees/beam_integration/{type}_{tag}", decl=True),
    # -- Topology ---------------------------------------------------------
    _p("element", "element", "both", "archive",
       "/opensees/element_meta/{type}|" + _S + "/owned_element_ids",
       seq=("activate", "element"), decl=True),
    # -- Time series and patterns -----------------------------------------
    _p("timeSeries", "time_series", "global", "archive",
       "/opensees/time_series/{type}_{tag}", decl=True),
    _p("pattern_open", "pattern", "both", "archive",
       "{scope}/patterns/{type}_{tag}", seq=("add", "pattern"), decl=True),
    _p("pattern_close", "pattern", "both", "archive",
       "{scope}/patterns/{type}_{tag}"),
    _p("load", "load", "both", "archive",
       "{scope}/patterns/{type}_{tag}/loads"),
    _p("eleLoad", "load", "both", "archive",
       "{scope}/patterns/{type}_{tag}/element_loads"),
    _p("sp", "load", "both", "archive",
       "{scope}/patterns/{type}_{tag}/sps"),
    _p("sp_hold", "load", "stage", "archive",
       _S + "/patterns/{type}_{tag}/sp_holds"),
    # -- Regions and damping ----------------------------------------------
    _p("region", "region", "both", "archive",
       "{scope}/regions/region_{k:03d}", seq=("add", "region"), decl=True),
    # The global form is a ``/opensees/commands`` row (K1-4); the stage
    # form keeps its stage sub-table (ADR 0055 Phase 2).
    _p("rayleigh", "damping", "both", "archive",
       _CMD + "|" + _S + "/rayleigh", seq=("set", "rayleigh")),
    _p("damping", "damping", "global", "archive",
       "/opensees/dampings/damping_{k:03d}", decl=True),
    _p("modal_damping", "damping", "global", "archive", _CMD),
    # -- Recorders --------------------------------------------------------
    _p("recorder", "recorder", "both", "archive",
       "{scope}/recorders/{kind}_{k}", seq=("add", "recorder"), decl=True),
    _p("recorder_declaration_begin", "recorder", "both", "archive",
       "{scope}/recorders/{kind}_{k}@declaration_name"),
    _p("recorder_declaration_end", "recorder", "both", "archive",
       "{scope}/recorders/{kind}_{k}@declaration_name"),
    # -- Analysis chain ---------------------------------------------------
    # ``constraints("LadrunoContact")`` is skipped silently; every other
    # handler archives.
    _p("constraints", "analysis", "both", "archive",
       "{scope}/analysis@handler"),
    _p("numberer", "analysis", "both", "archive",
       "{scope}/analysis@numberer"),
    _p("system", "analysis", "both", "archive", "{scope}/analysis@system"),
    _p("test", "analysis", "both", "archive", "{scope}/analysis@test"),
    _p("algorithm", "analysis", "both", "archive",
       "{scope}/analysis@algorithm"),
    _p("integrator", "analysis", "both", "archive",
       "{scope}/analysis@integrator"),
    _p("analysis", "analysis", "both", "archive",
       "{scope}/analysis@analysis"),
    _p("analyze", "analysis", "both", "archive",
       "/opensees/analysis@analyze_steps|" + _S + "@analyze_steps",
       returns="int"),
    # -- Stress control ---------------------------------------------------
    # The trio's calls carry resolved tags and write nothing; their
    # information is archived declaratively through the
    # ``set_initial_stress_records`` / ``set_stage_records`` side
    # channels (ADR 0055).
    _p("addToParameter", "parameter", "both", "archive",
       "{scope}/initial_stress/stress_{k:03d}"),
    _p("flip_element_stage", "parameter", "stage", "archive",
       _S + "/activate_absorbing/absorb_{k:03d}",
       seq=("activate", "absorbing")),
    _p("update_parameter", "parameter", "stage", "refuse", ""),
    _p("step_hook_ramp", "parameter", "both", "archive",
       "{scope}/initial_stress/stress_{k:03d}",
       seq=("add", "initial_stress")),
    # -- Staged analysis --------------------------------------------------
    _p("stage_open", "stage", "stage", "archive", _S + "@name"),
    _p("stage_close", "stage", "stage", "archive", _S),
    _p("domain_change", "domain", "stage", "archive", _S + "@domain_change"),
    _p("set_time", "domain", "stage", "archive", _S + "@set_time"),
    _p("set_creep", "domain", "stage", "archive", _S + "@set_creep_on"),
    _p("reset", "domain", "stage", "archive", _S + "@pre_analyze_reset"),
    _p("set_node_vel", "domain", "stage", "refuse", ""),
    _p("set_node_accel", "domain", "stage", "refuse", ""),
    _p("remove_sp", "bc", "stage", "archive", _S + "/remove_sp",
       seq=("remove", "fix")),
    _p("remove_element", "element", "stage", "archive",
       _S + "/remove_element", seq=("remove", "element")),
    _p("update_material_stage", "material", "stage", "archive",
       _S + "/update_material_stage", seq=("set", "material_stage")),
    # -- Modal family (runtime retrieval and fork analyses) ---------------
    _p("eigen", "modal", "global", "archive", _CMD, returns="list[float]"),
    _p("modal_properties", "modal", "global", "ledger", "",
       returns="dict[str, list[float]]"),
    _p("modal_response_history", "modal", "global", "ledger", "",
       requires=_FORK),
    _p("response_spectrum_analysis", "modal", "global", "ledger", "",
       requires=_FORK),
    _p("eigen_feast", "modal", "global", "ledger", "",
       requires=_FORK, returns="list[float]"),
    # Archived as a stage's ``/opensees/commands`` row (``s.profile``);
    # the bridge's own ``ops.profiler`` brackets reach decks only.
    _p("profiler", "profiler", "stage", "archive", _CMD, requires=_FORK),
    # -- Partitions -------------------------------------------------------
    _p("partition_open", "partition", "both", "archive",
       "/opensees/partitions/partition_{rank:02d}"),
    _p("partition_close", "partition", "both", "archive",
       "/opensees/partitions/partition_{rank:02d}"),
    _p("parallel_runtime_fallback_numberer", "analysis", "both", "archive",
       "{scope}/analysis@numberer_runtime_fallback"),
    _p("parallel_runtime_fallback_system", "analysis", "both", "archive",
       "{scope}/analysis@system_runtime_fallback"),
    # -- Command channel (ADR 0114 D2/D3) ---------------------------------
    # Method 75, the last. H5 writes an allow-listed token as a
    # ``/opensees/commands`` row; an unknown token raises on every emitter.
    _p("command", "channel", "both", "archive", _CMD),
)

#: One row per verb, keyed by verb name.
VERBS: Final[Mapping[str, Verb]] = MappingProxyType(
    {row.verb: row for row in _ROWS})

#: The number of methods on ``base.py::Emitter``. ADR 0114 D2 freezes it
#: at 75: ``command`` (K1-2) is the last method, and a new verb is a
#: ``command()`` token with a row here.
EMITTER_METHOD_COUNT: Final[int] = 75

#: Public names an emitter may define beyond the Protocol, per emitter
#: module: emitter-specific side channels, not verbs. The class-level
#: capability flags (``model_reissue_purges``, ``supports_partitions``)
#: are listed too until the bridge reads ``caps`` (E2 ``TargetCaps``,
#: ADR 0114 D6) instead of them. ``caps`` itself is the Protocol's one
#: attribute, declared on every emitter, so it is not a side channel.
SIDE_CHANNELS: Final[Mapping[str, frozenset[str]]] = MappingProxyType({
    "tcl": frozenset({
        "lines", "line_count", "line_buffer", "write_to", "preamble",
        "stream_to", "stream_fragment_count", "stream_finish",
        "stream_abort", "eigen_feast_parallel", "eigen_parallel",
        "partition_spans", "model_reissue_purges",
    }),
    "py": frozenset({"lines", "line_count", "line_buffer", "write_to"}),
    "live": frozenset({
        "frequency_response", "steady_state_dynamics", "complex_eigen",
        "random_response", "ladruno_projection_tie_force",
        "ladruno_contact_force", "ladruno_contact_info",
        "ladruno_mortar_penetration", "ladruno_mortar_tie_residual",
        "critical_time_step", "augment", "ops", "supports_partitions",
    }),
    "h5": frozenset({
        "mark_mass_from_model", "add_oriented_elements", "write",
        "write_opensees_into", "set_initial_stress_records",
        "set_stage_records", "restore_partition_blocks",
        "restore_stage_blocks", "restore_program", "ledger_counts",
        "set_solve_stamp",
    }),
    "recording": frozenset(),
})
