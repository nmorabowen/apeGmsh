"""Agent + script + viewer habitat (ADR 0095 S1–S2).

Standalone sidecar — same administrative shape as :mod:`apeGmsh.hpc`
and :mod:`apeGmsh.assess`. Not a session composite and not re-exported
from ``apeGmsh.__init__``.

Public door::

    python -m apeGmsh.studio script.py
    python -m apeGmsh.studio script.py --phase mesh
    python -m apeGmsh.studio --status

``run_until(phase=)`` is a real gate: ``model`` stops before
``generate()``, ``mesh`` before ``apeSees`` / ``Results``. A successful
stop writes ``.apegmsh/names.json`` and appends ``.apegmsh/runs.jsonl``.
Picks write a names-first :class:`SelectionEnvelope` to
``.apegmsh/selection.json`` (cwd). Types live here so a test or skill
can project / read the envelope without opening Qt.
"""

from ._envelope import (
    ENVELOPE_SCHEMA,
    NameRecord,
    PickEvidence,
    SelectionEnvelope,
    dumps,
    loads,
    project,
    project_state,
    read_envelope,
    write_envelope,
)
from ._ledger import last_run, read_runs
from ._paths import (
    DEFAULT_ENVELOPE_REL,
    DEFAULT_LEDGER_REL,
    DEFAULT_NAMES_REL,
    envelope_path,
    ledger_path,
    names_path,
)
from ._replay import PHASES, ReplayResult, ReplayRunner, StopAtPhase
from ._status import collect_status, format_status

__all__ = [
    "DEFAULT_ENVELOPE_REL",
    "DEFAULT_LEDGER_REL",
    "DEFAULT_NAMES_REL",
    "ENVELOPE_SCHEMA",
    "PHASES",
    "NameRecord",
    "PickEvidence",
    "ReplayResult",
    "ReplayRunner",
    "SelectionEnvelope",
    "StopAtPhase",
    "collect_status",
    "dumps",
    "envelope_path",
    "format_status",
    "last_run",
    "ledger_path",
    "loads",
    "names_path",
    "project",
    "project_state",
    "read_envelope",
    "read_runs",
    "write_envelope",
]
