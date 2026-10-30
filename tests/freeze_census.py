"""The freeze census (ADR 0121 D4; invariants 6 and 7).

Census for ``tests/test_freeze_lock.py``. ADR 0121 D4 replaces ADR
0102 D2's prose freeze with a lock on four registries (the MCP row is
two sets here, the tools and the habitat verbs):

* ``primitive families``: the ``name`` of each ``tests/families.py::FAMILIES``
  entry;
* ``constraint kinds``: the string constants of
  ``apeGmsh._kernel.records._kinds.ConstraintKind``;
* ``studio mcp tools``: the ``@mcp.tool`` functions in
  ``src/apeGmsh/studio/mcp.py``;
* ``habitat tools``: ``apeGmsh.studio._profile.HABITAT_TOOLS``;
* ``emitter verb families``: ``verbs.FAMILIES`` and every row ``family``
  in ``apeGmsh.opensees.emitter.verbs.VERBS``.

``FROZEN`` is the registries as they stood at the freeze point, the
merge commit of the PR that added this file. Those entries are
grandfathered. The lock test pins them by digest, so nothing joins
``FROZEN`` after the freeze point.

``ADMITTED`` holds every entry added after the freeze point, keyed
``entry -> ADR number`` (four digits, e.g. ``"0121"``). The cited ADR's
Status, as ``scripts/adr_index.py::parse_adr`` reads it, must be
Accepted. Growing a registry is: write and accept the ADR, add the
member, add its ``ADMITTED`` line. Removing a member removes its line
(from ``ADMITTED``, or from ``FROZEN`` together with the digest).
"""
from __future__ import annotations

FROZEN: dict[str, frozenset[str]] = {
    "primitive families": frozenset({
        "uniaxial material", "nd material", "section", "geomTransf",
        "time series", "pattern", "recorder", "beam-column element",
        "truss element", "zero-length element", "shell element",
        "solid element", "constraint handler", "numberer", "system",
        "convergence test", "algorithm", "integrator", "analysis",
    }),
    "constraint kinds": frozenset({
        "equal_dof", "equal_dof_mixed", "rigid_beam", "rigid_beam_stiff",
        "rigid_rod", "rigid_diaphragm", "rigid_body", "kinematic_coupling",
        "penalty", "node_to_surface", "node_to_surface_spring", "tie",
        "distributing", "embedded", "tied_contact", "mortar", "interface",
    }),
    "studio mcp tools": frozenset({
        "status", "get_selection", "lookup", "run_until", "assess",
        "render", "animate", "results_pin", "emit_report", "highlight",
        "promote_selection",
    }),
    "habitat tools": frozenset({
        "status", "get_selection", "run_until", "assess", "render",
        "animate", "results_pin", "emit_report", "highlight",
        "promote_selection",
    }),
    "emitter verb families": frozenset({
        "model", "node", "bc", "constraint", "embed", "contact",
        "material", "section", "transform", "integration", "element",
        "time_series", "pattern", "load", "region", "damping", "recorder",
        "analysis", "parameter", "stage", "domain", "modal", "profiler",
        "partition", "channel",
    }),
}

ADMITTED: dict[str, dict[str, str]] = {
    "primitive families": {"modal basis": "0127"},
    "constraint kinds": {},
    "studio mcp tools": {},
    "habitat tools": {},
    "emitter verb families": {},
}
