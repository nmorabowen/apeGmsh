"""Results-viewer tab assembly.

Constructs the residual right-side tab widgets the ``ResultsViewer``
shows in its ``ResultsWindow`` dock:

* **Inspector** — picked-entity details + time-history launcher.

Stages / Diagrams migrated to the left-rail outline tree (B1).
The Settings tab content moved into the right-rail DetailsPanel (B2),
but the :class:`DiagramSettingsTab` instance is still constructed
here — DetailsPanel re-hosts its widget.
Probes moved to the viewport HUD palette (B3); the right dock now
hosts only the Inspector. The Inspector itself migrates into the
details panel + a node-pick HUD readout in B5.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ._diagram_settings_tab import DiagramSettingsTab
from ._inspector_tab import InspectorTab

if TYPE_CHECKING:
    from ..diagrams._director import ResultsDirector


@dataclass
class ResultsTabs:
    """Container holding constructed tab widgets for the results window."""
    settings: DiagramSettingsTab
    inspector: InspectorTab

    def to_pairs(self) -> list[tuple[str, object]]:
        """Return the list of ``(name, widget)`` pairs for the tab dock.

        The Settings tab is intentionally absent — its content is
        re-hosted by the right-rail :class:`DetailsPanel` (B2).
        """
        return [
            ("Inspector", self.inspector.widget),
        ]


def build_results_tabs(
    director: "ResultsDirector",
    on_open_history=None,
) -> ResultsTabs:
    """Construct the residual right-dock tab set.

    ``on_open_history(node_id, component)`` is invoked when the user
    clicks the Inspector's "Open time history…" button; the viewer
    shell uses it to add a tab in the plot pane.
    """
    settings = DiagramSettingsTab(director)
    inspector = InspectorTab(director, on_open_history=on_open_history)
    return ResultsTabs(settings=settings, inspector=inspector)
