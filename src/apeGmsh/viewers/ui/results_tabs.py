"""Results-viewer tab assembly.

Constructs the right-side tab widgets the ``ResultsViewer`` shows in
its ``ResultsWindow`` dock:

* **Settings** — per-diagram styling controls
* **Inspector** — picked-entity details + time-history launcher
* **Probes** — point / line / plane probes (appended after the probe
  overlay is constructed)

Stages and Diagrams used to live as tabs here; they migrated into the
left-rail outline tree (B1). Settings / Inspector / Probes follow into
the right-rail details panel (B2+).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ._diagram_settings_tab import DiagramSettingsTab
from ._inspector_tab import InspectorTab
from ._probes_tab import ProbesTab

if TYPE_CHECKING:
    from ..diagrams._director import ResultsDirector


@dataclass
class ResultsTabs:
    """Container holding constructed tab widgets for the results window."""
    settings: DiagramSettingsTab
    inspector: InspectorTab
    probes: ProbesTab | None = None

    def to_pairs(self) -> list[tuple[str, object]]:
        """Return the list of ``(name, widget)`` pairs for the tab dock."""
        pairs: list[tuple[str, object]] = [
            ("Settings", self.settings.widget),
            ("Inspector", self.inspector.widget),
        ]
        if self.probes is not None:
            pairs.append(("Probes", self.probes.widget))
        return pairs


def build_results_tabs(
    director: "ResultsDirector",
    on_open_history=None,
    probe_overlay=None,
) -> ResultsTabs:
    """Construct the residual right-dock tab set.

    ``on_open_history(node_id, component)`` is invoked when the user
    clicks the Inspector's "Open time history…" button; the viewer
    shell uses it to dock a ``TimeHistoryPanel``.
    """
    settings = DiagramSettingsTab(director)
    inspector = InspectorTab(director, on_open_history=on_open_history)
    probes = ProbesTab(probe_overlay) if probe_overlay is not None else None

    return ResultsTabs(
        settings=settings,
        inspector=inspector,
        probes=probes,
    )
