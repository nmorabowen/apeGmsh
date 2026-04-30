"""Manual smoke-check script — opens the results viewer with a fixed
fixture and prints a checklist of things to verify by hand.

This is **not a pytest module**. It is a runnable script that opens a
real Qt window. Run it after non-trivial viewer changes, before
shipping a PR, or whenever the automated suite passes but you want
human eyes on the rendering.

Usage::

    # From the project root:
    python tests/viewers/manual_check.py

    # Or run a specific fixture:
    python tests/viewers/manual_check.py springs
    python tests/viewers/manual_check.py frame

The script:

1. Prints the checklist (read it BEFORE opening the window so you know
   what you are looking for).
2. Opens the viewer with the chosen fixture.
3. After you close the window, prompts you to record any failures.

Scope — what manual verification covers that the automated suite does
not:

* Visual correctness (colormap orientation, glyph rendering,
  scrubber moves smoothly).
* Real-mouse picking interactions (the automated suite mocks the
  callbacks; this exercises ``enable_point_picking`` end-to-end).
* Qt event-loop integration (the suite uses ``off_screen=True``).
* Side-panel docking (matplotlib in Qt for fiber / layer / time-
  history panels).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional


# --------------------------------------------------------------------- #
# Fixture catalog
# --------------------------------------------------------------------- #

_REPO_ROOT = Path(__file__).resolve().parents[2]

_FIXTURES = {
    "frame": _REPO_ROOT / "tests" / "fixtures" / "results" / "elasticFrame.mpco",
    "springs": _REPO_ROOT / "tests" / "fixtures" / "results" / "zl_springs.mpco",
}


# --------------------------------------------------------------------- #
# Checklist
# --------------------------------------------------------------------- #

CHECKLIST = """\
══════════════════════════════════════════════════════════════════════
  apeGmsh ResultsViewer — manual smoke checklist
══════════════════════════════════════════════════════════════════════

WINDOW + SCRUBBER
  [ ] Window opens; the mesh is visible (gray substrate)
  [ ] Status bar shows "Mesh: N nodes, M cells | Stages: K"
  [ ] Time scrubber at the bottom shows current step + t = …
  [ ] Dragging the scrubber moves time smoothly (no jank)
  [ ] ◀ / ▶ buttons step backward / forward
  [ ] ⏪ / ⏩ jump to first / last
  [ ] Camera orbit (Shift+Scroll), pan (MMB), zoom (wheel) all work

STAGES TAB
  [ ] Stage list shows the file's stages with name / kind / steps
  [ ] Active stage is highlighted in bold
  [ ] Clicking another stage switches the active one (mesh stays;
      scrubber resets to step 0)

DIAGRAMS TAB — Add… opens the dialog
  [ ] Add → "Contour" → component "displacement_z" (or what's available)
       → contour appears, scalar bar visible
  [ ] Drag scrubber → contour values update; no flicker; no actor
       re-creation (geometry is stable)
  [ ] Add → "Deformed shape" → mesh warps; undeformed reference
       shown in light gray
  [ ] Add → "Line force diagram" → hatched fill perpendicular to
       each beam (frame fixture only)
  [ ] Row selection in Diagrams highlights, Settings tab populates
  [ ] Up / Down reorders; Remove drops the diagram

SETTINGS TAB (per-diagram controls)
  [ ] Contour: cmap combo, clim min/max, "Auto-fit at current step",
       opacity slider all live-update
  [ ] Deformed: scale spinner + ×1/×10/×100/×1000 presets work
  [ ] Line force: scale, axis combo (Auto / y / z), Flip sign

INSPECTOR TAB
  [ ] Node ID input → Lookup → coords + values appear
  [ ] Values column lists each active diagram's component
  [ ] Step changes update the values (without re-typing the ID)
  [ ] Component textbox + "Open time history…" docks a chart
  [ ] Time-history chart shows a vertical red line at current step;
       dragging the scrubber moves the line without re-fetching data

PROBES TAB
  [ ] Point Probe → click on mesh → marker + values text below
  [ ] Line Probe → click A → click B → matplotlib chart pops up
  [ ] Plane Probe (X / Y / Z) → slice surface appears; result text
       lists slice point/cell counts
  [ ] Stop cancels an in-progress interactive probe
  [ ] Clear removes all markers and resets history

LIFECYCLE
  [ ] Closing the window exits cleanly (no Python tracebacks)
  [ ] Re-running this script opens a fresh window (no leftover state)

══════════════════════════════════════════════════════════════════════
"""


# --------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------- #

def _resolve_fixture(name: str) -> Path:
    if name not in _FIXTURES:
        keys = ", ".join(sorted(_FIXTURES))
        raise SystemExit(
            f"Unknown fixture {name!r}. Available: {keys}"
        )
    path = _FIXTURES[name]
    if not path.exists():
        raise SystemExit(
            f"Fixture file missing: {path}\n"
            f"Run from the project root, not the worktree's parent."
        )
    return path


def _open_viewer(fixture_path: Path) -> None:
    """Open the results viewer on the chosen fixture (blocks until close)."""
    # Imports kept inside the function so the script's --help / banner
    # path works without a Qt environment.
    from apeGmsh.results import Results

    print(f"Opening: {fixture_path}\n")
    if fixture_path.suffix.lower() == ".mpco":
        results = Results.from_mpco(fixture_path)
    else:
        results = Results.from_native(fixture_path)

    print(
        f"  stages : {[s.name for s in results.stages]}\n"
        f"  ndm    : (from FEM)\n"
        f"  fem    : {'bound' if results.fem else 'NOT bound'}\n"
    )

    print("Launching viewer (close the window to exit)…\n")
    results.viewer()


def _print_post_checklist() -> None:
    print(
        "\n──────────────────────────────────────────────────────────"
        "────────────\n"
        "  Window closed.\n\n"
        "  If anything in the checklist failed, jot it down — bugs the\n"
        "  automated suite can't catch land here. Open issues with:\n"
        "    * which fixture you ran ('frame' or 'springs')\n"
        "    * which checklist line failed\n"
        "    * the actual vs. expected behaviour\n"
        "──────────────────────────────────────────────────────────"
        "────────────"
    )


def main(argv: Optional[list[str]] = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])

    if args and args[0] in {"-h", "--help"}:
        print(__doc__)
        return 0

    fixture_name = args[0] if args else "frame"
    fixture_path = _resolve_fixture(fixture_name)

    print(CHECKLIST)
    try:
        _open_viewer(fixture_path)
    except KeyboardInterrupt:
        print("\nInterrupted before window close.")
        return 130

    _print_post_checklist()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
