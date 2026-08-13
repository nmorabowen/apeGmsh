"""Finding compiler for agents (ADR 0094 S2).

Standalone sidecar — types + pure runners — the same shape as
:mod:`apeGmsh.hpc`. Not a session composite and not re-exported from
``apeGmsh.__init__`` (the root import chain eagerly pulls ``gmsh``).

Public doors are the thin methods on the brokers an agent already
holds::

    report = fem.assess()
    report = results.assess()

``figures=True`` ships in S3; this slice implements findings only.
"""

from ._catalog import CATALOG_SEVERITY, EVIDENCE_CAP
from ._fem import assess_fem
from ._results import assess_results
from ._types import AssessmentReport, Finding

__all__ = [
    "AssessmentReport",
    "CATALOG_SEVERITY",
    "EVIDENCE_CAP",
    "Finding",
    "assess_fem",
    "assess_results",
]
