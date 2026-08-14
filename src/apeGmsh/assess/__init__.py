"""Finding compiler for agents (ADR 0094 S2).

Standalone sidecar — types + pure runners — the same shape as
:mod:`apeGmsh.hpc`. Not a session composite and not re-exported from
``apeGmsh.__init__`` (the root import chain eagerly pulls ``gmsh``).

Public doors are the thin methods on the brokers an agent already
holds::

    report = fem.assess()
    report = results.assess()
    report = results.assess(figures=True)  # + render_pack

``figures=True`` (default ``False``) calls ``results.render_pack`` /
``fem.render`` on the already-imported broker. This package does not
import ``apeGmsh.viewers`` or ``gmsh``.
"""

from ._catalog import CATALOG_SEVERITY, EVIDENCE_CAP, FAIL_RESERVED
from ._fem import assess_fem
from ._results import assess_results
from ._types import AssessmentReport, Finding

__all__ = [
    "AssessmentReport",
    "CATALOG_SEVERITY",
    "EVIDENCE_CAP",
    "FAIL_RESERVED",
    "Finding",
    "assess_fem",
    "assess_results",
]
