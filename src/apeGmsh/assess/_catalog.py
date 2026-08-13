"""Closed v1 finding catalog (ADR 0094). New codes are an ADR amendment."""
from __future__ import annotations

from typing import Literal

Severity = Literal["error", "warning", "info"]

EVIDENCE_CAP = 8

# FAIL is reserved for the error codes. RES.U_VS_DIAG is never error.
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
    "RES.ENERGY_ERR": "warning",
}
