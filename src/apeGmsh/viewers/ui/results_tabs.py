"""Results-viewer tab assembly.

Constructs the right-side tab widgets the ``ResultsViewer`` shows in
its ``ViewerWindow`` dock:

* **Stages** — analysis stage list with active-stage selection
* **Diagrams** — diagram list + add / remove / reorder
* **Settings** — per-diagram styling controls (Phase 1+)

Future tabs (Inspector, Probes, Visibility, Session) attach in their
own phases.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ._stages_tab import StagesTab
from ._diagrams_tab import DiagramsTab
from ._diagram_settings_tab import DiagramSettingsTab
from ._inspector_tab import InspectorTab
from ._probes_tab import ProbesTab

if TYPE_CHECKING:
    from ..diagrams._director import ResultsDirector


@dataclass
class ResultsTabs:
    """Container holding constructed tab widgets for the results window."""
    stages: StagesTab
    diagrams: DiagramsTab
    settings: DiagramSettingsTab
    inspector: InspectorTab
    probes: ProbesTab | None = None

    def to_pairs(self) -> list[tuple[str, object]]:
        """Return the list of ``(name, widget)`` pairs for ``ViewerWindow``."""
        pairs: list[tuple[str, object]] = [
            ("Stages", self.stages.widget),
            ("Diagrams", self.diagrams.widget),
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
    """Construct the tab set and wire diagrams<->settings selection.

    ``on_open_history(node_id, component)`` is invoked when the user
    clicks the Inspector's "Open time history…" button; the viewer
    shell uses it to dock a ``TimeHistoryPanel``.
    """
    stages = StagesTab(director)
    diagrams = DiagramsTab(director)
    settings = DiagramSettingsTab(director)
    inspector = InspectorTab(director, on_open_history=on_open_history)
    probes = ProbesTab(probe_overlay) if probe_overlay is not None else None

    # When the user selects a diagram in the Diagrams tab, route it
    # to the Settings panel so it can render the right controls.
    diagrams.on_diagram_selected(settings.set_selected)

    return ResultsTabs(
        stages=stages,
        diagrams=diagrams,
        settings=settings,
        inspector=inspector,
        probes=probes,
    )
