# 07 — Property Descriptors + Unified Settings Dialog

**Status:** pending  ·  **Cost:** ~1 week  ·  **Depends on:** 04 (active-objects), 06 (color-map editor — for one concrete migration case)

## Goal

Introduce `Property[T]` typed descriptors with domain metadata, then build a single
Settings dialog that auto-generates a tab per "settings owner" and a widget per property.
Replace the three per-viewer preference dialogs with this one dialog.

## Why

Two pains, one solution.

**Pain 1 — diagram kinds are hardcoded.** Adding a new diagram type means editing the
Add-Diagram dialog, a style class, sometimes the registry. The Style classes
(`ContourStyle`, `DeformedShapeStyle`, …) are dataclasses with no metadata — the UI has
to know how to render each one.

**Pain 2 — three preference panels.** `model.viewer`, `mesh.viewer`, `results.viewer`
each have their own preferences dialog (`preferences_dialog.py`). Same options
(background color, grid color, default colormap) duplicated three times with subtle
divergences. No search; users hunt for "the dark mode toggle."

ParaView solves both with the same mechanism: properties carry type + domain metadata;
UI is generated from that metadata. New filter? Declare a proxy with properties. New
setting? Register a "settings" proxy. UI auto-rebuilds.

## ParaView reference

- Workflow: [`paraview-flows/index.html` → "Open Settings → tabs auto-generated from proxies"](../paraview-flows/index.html)
- Files:
  - `Remoting/ServerManager/vtkSMProperty.h:5-100` — typed property + domain.
  - `Qt/Components/pqProxyWidget.h:28` — auto-generates widgets from a proxy.
  - `Qt/Components/pqSettingsDialog.h` — dialog that enumerates settings proxies.
  - `Qt/Components/pqSearchBox.h` — cross-tab search.

## Files to add / modify

**New:**
- `src/apeGmsh/viewers/core/_property.py` — the `Property[T]` descriptor and domain types.
- `src/apeGmsh/viewers/ui/_property_widget.py` — domain → Qt widget factory (range →
  slider, enum → combo, bool → checkbox, color → color picker, etc.).
- `src/apeGmsh/viewers/ui/_owner_panel.py` — given an object whose class declares
  `Property[T]` fields, build a `QFormLayout` of widgets.
- `src/apeGmsh/viewers/ui/_settings_dialog.py` — the unified dialog. Enumerates settings
  owners, tabs them, adds global search.
- `src/apeGmsh/viewers/core/_settings_registry.py` — register settings owners (one per
  concern: Theme, Camera defaults, Grid, Picking, Diagrams). Tabs are derived from this
  list.

**Modify:**
- `src/apeGmsh/viewers/diagrams/_styles.py` — migrate `ContourStyle` (only — proof of
  concept) to use `Property[T]` descriptors. Other styles stay as-is in this plan.
- `src/apeGmsh/viewers/ui/_diagram_settings_tab.py` — use `OwnerPanel` to render the
  active diagram's properties.
- `src/apeGmsh/viewers/ui/preferences_dialog.py`, `preferences.py` (and the three
  per-viewer variants) — delete; replaced by `_settings_dialog.py`.

## API sketch

```python
# _property.py
T = TypeVar("T")

class Domain(Generic[T]):
    """Constrains valid values. Drives the UI widget choice."""

class Range(Domain[float]):
    def __init__(self, lo: float, hi: float, step: float = 0.01): ...

class Enum(Domain[str]):
    def __init__(self, choices: list[str]): ...

class Color(Domain[tuple[int, int, int]]): ...

class Bool(Domain[bool]): ...

class Property(Generic[T]):
    """A typed, validated, observable property."""

    def __init__(self, default: T, domain: Domain[T], *, label: str = "",
                 help: str = ""): ...

    # Descriptor protocol
    def __get__(self, obj, objtype=None) -> T: ...
    def __set__(self, obj, value: T) -> None:
        """Validates against domain, fires `obj.property_changed(name, value)`."""

# Usage
class ContourStyle:
    array_name = Property[str]("", Enum([]), label="Color by")
    opacity    = Property[float](1.0, Range(0.0, 1.0), label="Opacity")
    log_scale  = Property[bool](False, Bool(), label="Log scale")
    line_width = Property[float](1.0, Range(0.5, 10.0), label="Line width")
```

```python
# _settings_dialog.py
class SettingsDialog(QDialog):
    """
    Tabs: one per owner. Each tab is an OwnerPanel of the owner's Property fields.
    Top: search box that filters property labels/help across all tabs.
    Buttons: OK (apply), Cancel (revert), Defaults (reset all).
    """
```

## Migration strategy (the hard part)

Don't migrate everything. Land in three steps:

1. **Descriptor + widget factory + one settings owner** (`Theme`). Settings dialog
   exists, has one tab. Proves the pattern. ~2 days.
2. **Migrate `ContourStyle`** to `Property[T]`. Diagram settings panel renders it via
   `OwnerPanel`. Other styles unchanged — old code path stays for them. ~2 days.
3. **Migrate the remaining settings owners** (Camera, Grid, Picking). Delete the old
   `preferences_dialog.py`. ~3 days.

Other style classes (DeformedShapeStyle, FiberSectionStyle, …) stay in the old dataclass
form until they need updates. The new pattern coexists with the old.

## Risks

- **Big API surface.** `Property`, `Domain`, `Range`, `Enum`, `Color`, `Bool` is a lot.
  Mitigation: ship the descriptor and only three domains in v1 (Range, Enum, Bool).
  Add Color and others lazily as concrete needs arise.
- **`@property` collisions.** Class-level descriptors interact with Python's
  built-in `@property`. If a style class already has `@property` methods, the migration
  needs care. Mitigation: descriptors only on dataclass-shaped style classes; mixed
  classes stay unchanged.
- **Serialization.** Today's `DiagramSpec` JSON serialization (`_session.py` schema v4)
  walks dataclass fields. Property-descriptor classes need a parallel serializer.
  Mitigation: descriptor exposes `to_dict()` / `from_dict()` on the owner class;
  schema bumps to v5; load path handles both.
- **Performance.** Descriptor `__set__` running validation + signal emission on every
  property write is fine for UI use, possibly excessive in tight loops. Mitigation:
  no hot path uses these descriptors — they're for user-facing settings only.
- **Discoverability of search.** Users may not notice the search box. Mitigation:
  placeholder text "Search 28 settings…" with the count.

## Done criteria

- [ ] `Property[T]` descriptor implemented; 3 domains shipped (Range, Enum, Bool); tests.
- [ ] Settings dialog exists, has search box, has tabs.
- [ ] At minimum 4 settings owners populated: Theme, Camera, Grid, Picking.
- [ ] Search filters property labels and help text across all tabs in real time.
- [ ] `ContourStyle` migrated to `Property[T]`. ContourStyle's settings panel renders
      via `OwnerPanel`, not a hand-built widget.
- [ ] Old per-viewer preferences dialogs deleted; all three viewers now route Edit →
      Settings to the unified dialog.
- [ ] Session save/load (`_session.py`) round-trips a Contour diagram with property
      changes; schema v5 documented.
- [ ] No regression in existing viewer tests.

## Out of scope

- Migrating every diagram style class. Just ContourStyle.
- Per-property undo. Out of scope for both this plan and 05.
- User-defined custom settings (a "User Tab"). Could be a follow-up.
- Plugin-defined settings owners. Implies a plugin system, which is in `future/`.
- Cross-viewer settings sync. Each viewer has its own registry; sharing is a follow-up.
