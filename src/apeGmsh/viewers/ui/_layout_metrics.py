"""Layout metrics — single source of truth for ResultsViewer dock geometry.

Centralises the dimensions previously scattered as inline literals across
``_results_window.py``, ``_plot_pane.py``, ``_details_panel.py``,
``_outline_tree.py``, ``_time_scrubber.py``, and
``_diagram_settings_tab.py``.

Tiered structure so future PREFERENCES promotion is straightforward:

- **Tier 1** — Dock dimensions. Most likely to become user-tunable
  (e.g. compact / comfortable density, multi-monitor users wanting
  a thinner outline column).
- **Tier 3** — Panel internals (header bar heights, label minimums,
  PlotPane row size, etc.).
- **Tier 4** — Visual aesthetics (corners, separators).

(Tier 2 originally held the in-window title-bar metrics; the title
bar was removed in v3.8 — those fields are gone.)

Migration path to PREFERENCES, when a field becomes user-tunable:

1. Add a matching field to ``Preferences`` in ``preferences_manager.py``.
2. Switch the consumer from ``LAYOUT.foo`` to a small accessor that
   prefers the user override and falls back to ``LAYOUT.foo``.
3. Optional: remove the field from ``LayoutMetrics`` if it's now
   fully owned by Preferences.

Defaults reflect the post-audit values (Tier 1 mins relaxed,
``plot_tab_list_max_rows`` raised from 6, ``diagram_btn_max_width``
from 60). Behaviour change vs. pre-centralisation is intentional and
limited to those three relaxations.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LayoutMetrics:
    """Frozen container of layout dimensions used by the ResultsViewer."""

    # ── Tier 1: Dock dimensions ──────────────────────────────────────
    # Outline (left) dock — separate min and initial so users can shrink.
    outline_min_width: int = 180
    outline_initial_width: int = 260

    # Right-side docks (Plots, Details, Session).
    right_min_width: int = 240
    right_initial_width: int = 380

    # Time-scrubber (bottom) dock.
    scrubber_min_height: int = 60
    scrubber_initial_height: int = 84

    # ── Tier 3: Panel internals ──────────────────────────────────────
    # Header strip inside Outline / PlotPane / etc. — visual chrome.
    panel_header_height: int = 28
    # DetailsPanel header is intentionally smaller than panel_header_height.
    details_header_height: int = 24

    # PlotPane tab-row sizing and tab-list cap (in rows).
    plot_row_height: int = 24
    plot_tab_list_max_rows: int = 12

    # Time-scrubber readout-label minimum widths (prevent collapse).
    scrubber_step_label_min_width: int = 80
    scrubber_time_label_min_width: int = 140

    # ── Tier 4: Visual aesthetics (corners + separators) ─────────────
    # Corner radius scale — most widgets use the default; small for tight
    # circles (×, dots), large for containers/dialogs/panels.
    corner_radius: int = 4
    corner_radius_small: int = 2
    corner_radius_large: int = 6

    # Gap between adjacent docks (drawn as the QMainWindow::separator).
    # Bigger = more obvious visual breathing room between docks; smaller =
    # docks sit flush. Also the click target for resizing.
    dock_separator_width: int = 4

    # Gap between QSplitter panes (mostly irrelevant in dock layout but
    # used in dialog splitters elsewhere — kept for consistency).
    splitter_handle_width: int = 4


# Default singleton consumed across the viewer. Treat as immutable; mutate
# by constructing a new ``LayoutMetrics`` instance and rebinding ``LAYOUT``
# (or, preferably, by promoting the field to ``Preferences`` — see module
# docstring).
LAYOUT = LayoutMetrics()
