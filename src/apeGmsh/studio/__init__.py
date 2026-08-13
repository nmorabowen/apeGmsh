"""Agent + script + viewer habitat (ADR 0095 S1–S2).

Standalone sidecar — same administrative shape as :mod:`apeGmsh.hpc`
and :mod:`apeGmsh.assess`. Not a session composite and not re-exported
from ``apeGmsh.__init__``.

Public door::

    python -m apeGmsh.studio script.py

Opens MeshViewer if the script generated a mesh, otherwise
ModelViewer. Picks in the Qt host write a names-first
:class:`SelectionEnvelope`
to ``.apegmsh/selection.json`` (cwd). Types live here so a test or
skill can project / read the envelope without opening Qt.
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
from ._paths import DEFAULT_ENVELOPE_REL, envelope_path
from ._replay import ReplayResult, ReplayRunner

__all__ = [
    "DEFAULT_ENVELOPE_REL",
    "ENVELOPE_SCHEMA",
    "NameRecord",
    "PickEvidence",
    "ReplayResult",
    "ReplayRunner",
    "SelectionEnvelope",
    "dumps",
    "envelope_path",
    "loads",
    "project",
    "project_state",
    "read_envelope",
    "write_envelope",
]
