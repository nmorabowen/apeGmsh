"""Closed v1 finding catalog (ADR 0094). New codes are an ADR amendment."""
from __future__ import annotations

from typing import Literal

Severity = Literal["error", "warning", "info"]

EVIDENCE_CAP = 8

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
    "RES.ENERGY_ERR": "info",
    "RES.ENERGY_NONFINITE": "warning",
    # ADR 0094 Amendment 2 — OpenSeesModel solver-zone catalog.
    # Every trigger has a legal reading (free-free eigen, free
    # vibration, multi-deck workflows) — none is error in v1.
    "OSM.NO_ANALYZE": "info",
    "OSM.NO_SUPPORT": "warning",
    "OSM.NO_PATTERN": "warning",
    "OSM.LOADS_UNIMPORTED": "warning",
    "RES.ZERO_U": "info",
}

FAIL_RESERVED: frozenset[str] = frozenset({
    "MODEL.EMPTY",
    "MESH.INVERTED",
    "RES.UNBOUND_FEM",
    "RES.NO_STAGE",
    "RES.NAN",
    "RES.INF",
})
