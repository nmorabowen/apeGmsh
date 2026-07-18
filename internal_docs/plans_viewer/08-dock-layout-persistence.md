# 08 — Dock Registry + Layout State Persistence

**Status:** ✅ **Landed** for `results.viewer` via [PR #168](https://github.com/nmorabowen/apeGmsh/pull/168) (2026-05-14).
Step 1: `DockSpec`, `DockRegistry`, `LayoutPersistence`, plus module-level helpers `mount_dock_spec`,
`build_view_menu`, `add_view_menu_toggle` (the testability extraction). Step 2: `ResultsWindow.extension_docks`
+ `add_extension_dock(spec)` + `DOCK_*` constants + auto-View-menu. Mesh/model viewer migration to the
registry is a separate follow-up — see *Deferred* below.
**Cost:** ~3–4 days  ·  **Depends on:** none

## Goal

Replace the current ad-hoc dock construction in each viewer (`viewer_window.py` plus
per-viewer tab assembly) with a registry-driven dock system that persists window
geometry, dock positions, sizes, visibility, floating state, and tabification across
viewer launches. The View menu auto-populates with one checkable toggle per registered
dock.

## Why

Today:

- Each viewer (`model_viewer.py`, `mesh_viewer.py`, `results_viewer.py`) constructs its
  tabs inline in the constructor. Adding a new dock means editing the main viewer class.
- `viewer_window.py` doesn't save state on close — every launch starts with the
  default layout. Users re-tile every session.
- No tabification of related docks. ResultsViewer ends up with 5–8 tabs on the right and
  there's no way to consolidate or split them.
- The View menu doesn't auto-list docks. Toggling visibility is inconsistent across viewers.
- Closing a floated dock loses its position permanently.

ParaView pattern: `pqSettings.saveState(QMainWindow&)` + `restoreState(QMainWindow&)`
inside a behavior (`PersistentMainWindowStateBehavior`) handles persistence. Docks are
registered via interfaces (`pqDockWindowInterface`), and the main window populates the
View menu from the registry.

## ParaView reference

- Files:
  - `Qt/Core/pqSettings.h:23-78` — `saveState` / `restoreState` wrapper.
  - `Qt/ApplicationComponents/pqParaViewBehaviors.h:81` — `PersistentMainWindowStateBehavior`.
  - `Qt/Components/pqDockWindowInterface.h:17-34` — plugin dock registration.
  - `Clients/ParaView/ParaViewMainWindow.cxx:141-149` — tabifyDockWidget grouping.
  - Workflow: this isn't surfaced as a `flows.json` entry yet — consider adding one
    ("Close viewer → restore on next launch") in the next pass.

## Files to add / modify

**New:**
- `src/apeGmsh/viewers/ui/_dock_registry.py` — `DockRegistry` that holds dock specs and
  mounts them onto a `QMainWindow`.
- `src/apeGmsh/viewers/ui/_layout_persistence.py` — `LayoutPersistence` wrapping
  `QSettings` save/restore, with versioning.
- `src/apeGmsh/viewers/ui/_view_menu.py` — auto-builds the View menu from the registry
  (one checkable action per dock, plus "Reset Layout" and "Save Layout As…").

**Modify:**
- `src/apeGmsh/viewers/ui/viewer_window.py` — instead of constructing tabs/docks inline,
  walk a `DockRegistry` instance passed in by the viewer. Wire `LayoutPersistence` to
  `showEvent` (restore) and `closeEvent` (save).
- `src/apeGmsh/viewers/model_viewer.py`, `mesh_viewer.py`, `results_viewer.py` —
  migrate per-viewer dock construction into `register_docks(registry)` calls. Each tab
  becomes a separate registered dock with a default area, default visibility, and an
  optional `tabify_with=` pointer to group related ones.

## API sketch

```python
# _dock_registry.py
@dataclass
class DockSpec:
    dock_id: str                       # stable id, used as QSettings key
    title: str                         # menu/tab label
    factory: Callable[[QWidget], QWidget]   # builds the dock's content widget
    default_area: Qt.DockWidgetArea = Qt.RightDockWidgetArea
    default_visible: bool = True
    default_floating: bool = False
    tabify_with: str | None = None     # dock_id of an existing dock to tabify with
    allowed_areas: Qt.DockWidgetAreas = (
        Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea
        | Qt.TopDockWidgetArea | Qt.BottomDockWidgetArea
    )

class DockRegistry:
    """Per-viewer registry. Built by the viewer class before window mount."""

    def register(self, spec: DockSpec) -> None: ...
    def mount(self, window: QMainWindow) -> dict[str, QDockWidget]:
        """Instantiate all registered docks; honor tabify groups in declaration order."""

# _layout_persistence.py
class LayoutPersistence:
    """Saves/restores window geometry + dockwidget state via QSettings."""

    SCHEMA_VERSION = 1   # bump if dock id space changes incompatibly

    def __init__(self, window_key: str): ...
    def save(self, window: QMainWindow) -> None: ...
    def restore(self, window: QMainWindow) -> bool:
        """Returns True if a saved state was found and applied. False = use defaults."""
    def reset(self) -> None: ...     # delete saved state for this window_key
```

Per-viewer usage:

```python
# results_viewer.py (illustrative)
def _build_docks(self) -> DockRegistry:
    reg = DockRegistry()
    reg.register(DockSpec("stages",   "Stages",     self._stages_factory))
    reg.register(DockSpec("time",     "Time",       self._time_factory,
                          tabify_with="stages"))
    reg.register(DockSpec("outline",  "Outline",    self._outline_factory,
                          default_area=Qt.LeftDockWidgetArea))
    reg.register(DockSpec("diagrams", "Diagrams",   self._diagrams_factory,
                          tabify_with="outline"))
    reg.register(DockSpec("session",  "Settings",   self._session_factory,
                          default_area=Qt.RightDockWidgetArea, default_visible=False))
    reg.register(DockSpec("output",   "Output",     self._output_factory,   # from plan 01
                          default_area=Qt.BottomDockWidgetArea,
                          default_visible=False))
    return reg
```

## Risks

- **Migration touches every dock site.** Three viewers, each with 5–8 tabs. Mitigation:
  migrate one viewer at a time. Keep the old inline-construction path coexisting until
  all three are migrated. Don't open one giant PR.
- **Stable `dock_id` discipline.** If a dock's id changes, restored state silently
  becomes orphaned and the dock falls back to defaults. Mitigation: ids are documented
  constants per viewer, and the `LayoutPersistence` SCHEMA_VERSION bumps if we ever
  rename one.
- **PyVista's QtInteractor is not a dock — it's the central widget.** That stays
  central. Only the side/bottom panels go through the registry.
- **Some tabs today are mutually exclusive** (e.g. Mesh viewer's Loads tab only exists
  if a `LoadsComposite` is set). Mitigation: registry supports conditional registration
  — viewer calls `reg.register(...)` only if the precondition holds. State for
  unregistered ids stays in QSettings but is ignored.
- **Cross-platform QSettings paths.** Windows registry vs `.config` on Linux vs
  `Preferences/` on macOS. Qt handles this; we just need a stable application + window
  name. Use `"apeGmsh"` as org, `"viewers.{model|mesh|results}"` as window_key.
- **Reset Layout** must work even when QSettings state is corrupted (older schema,
  partial write). `restore()` returns False on failure; window falls back to defaults.

## Done criteria

- [ ] `DockRegistry` and `LayoutPersistence` implemented with unit tests
      (serialize → deserialize roundtrip; reset clears state).
- [ ] All three viewers migrated; no inline `addDockWidget` calls remain in
      `viewer_window.py` or per-viewer code outside the registry.
- [ ] Closing a viewer saves: dock positions, sizes, floating state, visibility,
      tabification, window geometry, splitter sizes.
- [ ] Reopening the same viewer restores all of the above perfectly.
- [ ] View menu auto-populates with one checkable `QAction` per registered dock,
      sorted by registration order. Action title matches dock title.
- [ ] "Reset Layout" action in View menu clears saved state and reapplies defaults
      without closing the window.
- [ ] Tabifying multiple docks via `tabify_with=` produces a tabbed dock group, with
      tab order matching registration order.
- [ ] Adding a new dock = one `reg.register(DockSpec(...))` call. No edits to
      `viewer_window.py` or `_view_menu.py`.
- [ ] Existing viewer tests still pass.

## Out of scope

- Saving multiple named layouts (e.g. "Editing layout" vs "Presentation layout").
  ParaView supports this; we don't need it now.
- Cross-viewer layout sharing.
- Drag-and-drop reordering of tab order at runtime. Tab order is whatever Qt picks from
  the tabify group; users can rearrange via Qt's built-in drag, but we don't persist a
  custom order separately.
- Layout templates shipped with the app. Defaults come from `DockSpec.default_*`.

## Deferred from this implementation (PR #168)

- **`ViewerWindow` dock-registry path.** `ResultsWindow` got the extension-dock API
  + auto-View-menu, but `ViewerWindow` (used directly by mesh / model viewers) still
  uses its legacy `tabs=` + `extra_docks=` parameters. Generalizing `ViewerWindow` so
  mesh / model viewers can also register docks via `DockSpec` is the prerequisite for
  plan 01's rollout to those viewers (and for plan 04 step 3 / 4 panel additions).
- **`ResultsWindow` internals onto `DockRegistry`.** Its seven built-in docks are
  hardcoded; only extension docks go through the registry. Full unification is
  optional cleanup, not on the critical path.
