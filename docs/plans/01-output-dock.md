# 01 — Output / Log Dock

**Status:** ✅ **Landed** for `results.viewer` via [PR #168](https://github.com/nmorabowen/apeGmsh/pull/168) (2026-05-14).
Mesh/model viewer rollout + VTK `vtkOutputWindow` capture deferred — see *Deferred from this implementation* below.
**Cost:** ~2 days  ·  **Depends on:** none

## Goal

Add a docked log panel to all three viewers that captures Python tracebacks, VTK
warnings/errors, and user-facing info messages, color-coded by severity, with a
status-bar badge that increments on new warnings/errors.

## Why

Today: a `Diagram.update_to_step` exception during scrubbing flickers the UI and the
traceback lands in the launching REPL. If the viewer was launched with
`results.viewer(blocking=False)` (Phase 6+ subprocess mode), the traceback is lost
entirely. VTK warnings (e.g. "no scalar array named X") are silent.

ParaView pattern: one Qt message handler funnels `qDebug`/`qWarning`/`qCritical` and
VTK's `vtkOutputWindow` into a single dock, thread-safely. Users have one place to look.

## ParaView reference

- Workflow: [`paraview-flows/index.html` → "VTK warning fires → routed to Output dock"](../paraview-flows/index.html)
- Files:
  - `Qt/Core/pqOutputWidget.h` / `.cxx` — the dock + Qt message handler installation.
  - `Remoting/Core/vtkPVOutputWindow.h` — VTK→Qt bridge.
  - `Qt/ApplicationComponents/pqStatusBar.h` — badge counter.

## Files to add / modify

**New:**
- `src/apeGmsh/viewers/ui/_output_dock.py` — the `OutputDock` widget (QDockWidget + QTextEdit).
- `src/apeGmsh/viewers/ui/_log_router.py` — installs Python `logging.Handler`, `sys.excepthook`, and a VTK `vtkOutputWindow` subclass; emits a Qt signal for the dock to consume.

**Modify:**
- `src/apeGmsh/viewers/ui/viewer_window.py` — instantiate `OutputDock`, dock to bottom by default, add to View menu as a toggle. Wire `LogRouter` into the window's lifecycle (install on show, restore on close).
- `src/apeGmsh/viewers/model_viewer.py`, `mesh_viewer.py`, `results_viewer.py` — no per-viewer code; pick up the dock from `ViewerWindow`.

## API sketch

```python
# _log_router.py
class LogRouter(QObject):
    message = Signal(str, str)   # (text, severity in {"info","warning","error"})

    def install(self) -> None:
        """Hook Python logging, sys.excepthook, and vtkOutputWindow."""
    def uninstall(self) -> None:
        """Restore originals. Must be idempotent."""

# _output_dock.py
class OutputDock(QDockWidget):
    def __init__(self, router: LogRouter, parent: QWidget): ...
    def append(self, text: str, severity: str) -> None: ...
    def clear(self) -> None: ...
    @property
    def counts(self) -> dict[str, int]: ...   # for status-bar badge
```

Severity → color mapping is hardcoded (red/yellow/gray). No user setting — keep it simple.

## Risks

- **Thread safety.** VTK warnings may fire from background threads (rare in our case but
  possible). `LogRouter.message` is a Qt signal — emission is thread-safe by Qt's design;
  receiver runs in the UI thread. Verify with a synthetic test that emits from a
  `QThreadPool` task.
- **Recursive logging.** If `OutputDock.append` itself logs, infinite loop. Mitigation:
  `append` does not use the `logging` module; uses direct `QTextEdit.append`.
- **Existing `print()` calls.** We don't intercept stdout. Accept this — `print()` should
  be migrated to `logging.info` over time, but not in this plan.
- **Subprocess viewer mode** (Phase 6+ deferred): in subprocess mode, the parent process
  loses access to child logs. Out of scope — this dock fixes the in-process case, which
  is the common one.

## Done criteria

- [ ] Opening any of the three viewers shows an Output dock (collapsed by default at the bottom).
- [ ] `logging.warning("test")` from anywhere in apeGmsh (including a Diagram) appears in the dock.
- [ ] An unhandled Python exception raised during a diagram update (e.g. a bad selector
      that raises `KeyError`) appears in the dock as an error with full traceback;
      the viewer does NOT crash.
- [ ] A VTK warning (e.g. coloring by a non-existent array) appears in the dock as a warning.
- [ ] Status bar shows a badge "⚠ 3" after three warnings; clicking the badge raises the dock.
- [ ] Closing the viewer restores the original Python `excepthook` and `vtkOutputWindow`
      (verifiable by a unit test that opens + closes a viewer and checks state).
- [ ] No regression in existing viewer tests.

## Out of scope

- Per-message filtering UI (severity filter, search) — defer to a follow-up if useful.
- Persisting log history across viewer launches — logs are session-scoped.
- Routing logs from subprocess viewers back to the parent — separate problem.
- Migrating existing `print()` calls in apeGmsh to `logging`.

## Deferred from this implementation (PR #168)

- **VTK `vtkOutputWindow` capture.** Python subclassing of VTK's wrapper is
  version-fragile and the failure modes are silent. The three Python channels
  (logging + excepthook + unraisablehook) cover the daily-pain case; re-evaluate
  if VTK warnings become a real problem.
- **Mesh / model viewer rollout.** Output dock + LogRouter exist for
  `results.viewer` only. `ViewerWindow` doesn't yet have the extension-dock API
  that `ResultsWindow` got in PR #168 — generalizing it is the prerequisite for
  this rollout.
- **Status-bar badge for mesh / model.** Same prereq.
