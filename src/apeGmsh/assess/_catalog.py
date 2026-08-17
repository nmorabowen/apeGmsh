"""Closed v1 finding catalog (ADR 0094). New codes are an ADR amendment."""
from __future__ import annotations

from typing import Literal

Severity = Literal["error", "warning", "info"]

EVIDENCE_CAP = 8

# ADR 0094 Amendment 4. Last-step |ERR| (percent) above which
# RES.ENERGY_ERR promotes from info to warning. Basis: the one
# healthy dogfood implicit transient closes at 0.141% (a 35x margin
# below this line); ~5% is the conventional energy-balance alarm in
# explicit-dynamics practice. Single-datapoint basis — revisit with
# explicit-run data by amendment.
ENERGY_ERR_WARN_PCT = 5.0

# FAIL is reserved for the error codes. RES.U_VS_DIAG / RES.ENERGY_ERR
# are never error (ADR 0094 Amendment 1).
CATALOG_SEVERITY: dict[str, Severity] = {
    "MODEL.EMPTY": "error",
    "MESH.INVERTED": "error",
    "MESH.COINCIDENT_UNBRIDGED": "warning",
    "MESH.EMPTY_PG": "warning",
    "RES.UNBOUND_FEM": "error",
    "RES.NO_STAGE": "error",
    "RES.NAN": "error",
    "RES.INF": "error",
    "RES.LINEAGE": "warning",
    "RES.U_VS_DIAG": "info",
    # Base severity — see CONDITIONAL_SEVERITY below (ADR 0094
    # Amendment 4: promotes to "warning" when last-step |ERR| >
    # ENERGY_ERR_WARN_PCT).
    "RES.ENERGY_ERR": "info",
    "RES.ENERGY_NONFINITE": "warning",
    # ADR 0094 Amendment 2 — OpenSeesModel solver-zone catalog.
    # Every trigger has a legal reading (free-free eigen, free
    # vibration, multi-deck workflows) — none is error in v1.
    "OSM.NO_ANALYZE": "info",
    "OSM.NO_SUPPORT": "warning",
    "OSM.NO_PATTERN": "warning",
    "OSM.LOADS_UNIMPORTED": "warning",
    # ADR 0094 Amendment 4 — promoted info -> warning: across six
    # dogfood-assessed runs, one true positive and zero false
    # positives.
    "RES.ZERO_U": "warning",
}

# ADR 0094 Amendment 4. Codes whose emitted severity is conditional on
# runtime data rather than fixed by CATALOG_SEVERITY (which holds each
# such code's BASE/lowest severity). Maps code -> the set of severities
# it may legally emit. Tests consult this to allow either value instead
# of weakening the single-valued-severity assertion for every code.
CONDITIONAL_SEVERITY: dict[str, frozenset[Severity]] = {
    "RES.ENERGY_ERR": frozenset({"info", "warning"}),
}

FAIL_RESERVED: frozenset[str] = frozenset({
    "MODEL.EMPTY",
    "MESH.INVERTED",
    "RES.UNBOUND_FEM",
    "RES.NO_STAGE",
    "RES.NAN",
    "RES.INF",
})
