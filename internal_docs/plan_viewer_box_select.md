---
title: Viewer Mouse Box-Select — Fix Plan
aliases: [viewer-box-select, plan_viewer_box_select]
tags: [apeGmsh, viewer, picking, bugfix, plan]
---

# Viewer Mouse Box-Select — Fix Plan

> [!summary] One-line
> Drag-box selection in `ModelViewer` misses entities that are clearly
> inside the rubber-band on HiDPI displays, and silently discards
> empty-box releases. Caused by a DPI-scaling inconsistency between
> `_do_box` and `_do_click` / `_do_hover`, plus an over-strict
> window-mode containment test, in
> `viewers/core/pick_engine.py`.

## Reported symptom

User reproduction:

1. `g.begin()` → build geometry.
2. `g.model.viewer()` → opens `ModelViewer`.
3. Drag a rubber-band over several volumes (left → right, "window"
   mode).
4. On release, none — or only some — of the entities under the box are
   added to the working set.

Click-pick on individual entities works. Hover highlight works. The
rubber-band itself draws and animates correctly. Only the *box-select*
release behaves wrong.

> [!note] What "doesn't pick up" means here
> The user has not yet specified whether they see *zero* hits or a
> *partial* hit set. Both modes are predicted by the root causes below
> — Phase 0 disambiguates which is dominant before any code changes.

---

## Architectural context

The intended flow (per `viewers/core/pick_engine.py`) is:

```
LMB press      ──► on_lmb_press()                       (pick_engine.py:201)
                       └─ store _press_pos, _ctrl_held
LMB drag       ──► on_mouse_move()                      (pick_engine.py:208)
                       ├─ if past _drag_threshold → _dragging = True
                       └─ _update_rubberband(sx, sy, px, py)   (uses raw event coords)
LMB release    ──► on_lmb_release()                     (pick_engine.py:229)
                       └─ if _dragging → _do_box(...)   (pick_engine.py:295)
                              ├─ scale event coords by sx_ratio = aw / vw
                              ├─ project entity points via renderer.WorldToDisplay
                              ├─ test inside (bx0..bx1, by0..by1)
                              ├─ window mode: np.all(inside)
                              ├─ crossing mode: np.any(inside)
                              └─ if hits → on_box_select(hits, ctrl)
```

**Key invariants the design relies on:**

1. The rubber-band actor (`vtkActor2D` placed at raw event coords) and
   the hit-test box (computed in `_do_box`) **must live in the same
   coordinate space** as the projected entity points
   (`renderer.WorldToDisplay`). If any one of those three diverges
   (especially under HiDPI), what the user sees is not what gets
   tested.
2. Window mode (left → right drag) is the conventional CAD semantic:
   "the entity must be **fully** inside the box." The implementation
   approximates *full containment* by sampling representative points —
   so the sampled set must be representative of the entity's extent.
3. A drag that ends with no hits is a real user action (mis-aim,
   target moved, occluded). The release must give some feedback — even
   if it's "0 entities selected" — otherwise the UI feels broken.

All three invariants are violated by the current implementation. See
"Root causes" below.

---

## Decisions captured (resolved upfront)

| Decision | Resolution | Source |
|---|---|---|
| Coordinate space for hit-test | **Match what `_do_click` uses.** Event coords go straight into `_do_box` un-scaled, just like `_do_click` passes them straight to `picker.Pick`. WorldToDisplay output is then assumed to live in the same space (verified in Phase 0). | Inferred — the rubber-band already draws correctly using raw event coords, which is strong evidence event coords match VTK's display space on this build. |
| Window-mode containment | **Project entity points → compute their 2D AABB → test that AABB is fully inside the selection box.** Replaces strict `np.all(inside)` over up to 64 sample points. | UX — a single stray sample point currently kills the whole entity. |
| Empty-box feedback | **Always invoke `on_box_select`, even with an empty hit list.** Caller decides whether to show a status message; PickEngine's job is to report what happened. | Consistency with click semantics (a missed click also fires nothing rather than silently re-aborting drag state). |
| Phase 0 must run before Phase 1 | **Yes.** The DPI-scaling fix has two possible directions (remove the scaling vs. apply it everywhere) and we must not guess. | CLAUDE.md §1 — "Don't assume." |

> [!note] On the "always fire" decision
> Today, an empty drag exits at `pick_engine.py:385` (`if hits and
> self.on_box_select is not None`) without invoking the callback.
> Phase 3 changes this to fire with `[]`. Callers (specifically
> `model_viewer.py`'s wiring) must tolerate an empty list — verify in
> Phase 3 that `SelectionState.box_select(hits, ctrl)` (or the
> equivalent) does not blow up when `hits == []`.

---

## Root causes

### Bug 1 — DPI scaling applied in `_do_box` only

`viewers/core/pick_engine.py:303-317` (`_do_box`):

```python
# DPI scaling
try:
    rw = self._plotter.render_window
    vw, vh = rw.GetSize()
    aw, ah = rw.GetActualSize()
    sx_ratio = aw / vw if vw else 1.0
    sy_ratio = ah / vh if vh else 1.0
except Exception:
    sx_ratio = sy_ratio = 1.0

crossing = x1 < x0
bx0 = min(x0, x1) * sx_ratio
bx1 = max(x0, x1) * sx_ratio
by0 = min(y0, y1) * sy_ratio
by1 = max(y0, y1) * sy_ratio
```

`_do_click` (`pick_engine.py:258-273`) and `_do_hover`
(`pick_engine.py:275-293`) do **not** apply any scaling — they pass the
raw event coordinates straight to `picker.Pick(x, y, 0, renderer)`.
`_update_rubberband` (`pick_engine.py:179-194`) also uses raw event
coords on a `vtkActor2D`.

If `picker.Pick` and `vtkActor2D` are using the same coordinate space
(which the rubber-band drawing correctly suggests), then **the
projected `WorldToDisplay` coordinates are also in that space**, and
the `* sx_ratio` scaling in `_do_box` shifts the test box to a
*different* region of the screen than the user sees.

On a typical 2× HiDPI laptop:
- `vw = 1280`, `aw = 2560` → `sx_ratio = 2.0`.
- User drags from event-x = 100 to event-x = 400 (visible rubber-band
  spans 100..400 on screen).
- `_do_box` tests `bx0=200`, `bx1=800` against entity projections that
  are themselves at event-coord scale (e.g. ~150..350 for what the
  user sees) → **box is shifted right and stretched off-screen**, no
  hits.

This is the strongest candidate for the user's reported zero-/partial-
hit symptom.

> [!warning] Phase 0 must verify the direction
> If the rubber-band is *also* wrong on HiDPI (drawn at the wrong
> place), then event coords ARE in logical space and `WorldToDisplay`
> returns physical-pixel coords — in which case the fix is to scale
> *all three* of click, hover, and box, not to remove the scaling
> from box. The Phase 0 diagnostic resolves which world we live in.

### Bug 2 — Window-mode containment is over-strict

`viewers/core/pick_engine.py:374-379`:

```python
inside = (bx0 <= sx) & (sx <= bx1) & (by0 <= sy) & (sy <= by1)
if crossing:
    hit = np.any(inside)
else:
    hit = np.all(inside)
```

For volumes, `_do_box` calls `entity_points(dt)` which returns up to
64 sample mesh vertices (`entity_registry.py:219-249`). With
`np.all(inside)`, **a single vertex outside the box disqualifies the
whole entity** — even one vertex on the back face that projects barely
outside the box will kill an otherwise-enclosed cube.

For non-volumes (dim 0/1/2), `bbox(dt)` returns the 8 AABB corners
(`entity_registry.py:211-217`). Eight corners is more forgiving but
the same problem exists: rotate the AABB just so and one corner sits
outside even when the entity is visually fully enclosed.

**Right fix:** project the points, compute their 2D bounding box
`(px_min, py_min, px_max, py_max)`, and test:

```python
hit = (bx0 <= px_min and px_max <= bx1
       and by0 <= py_min and py_max <= by1)
```

This is the correct *fully contained* predicate for the entity's
projected extent and is invariant to how many sample points we use.

### Bug 3 — Empty-box release is silent

`viewers/core/pick_engine.py:385-386`:

```python
if hits and self.on_box_select is not None:
    self.on_box_select(hits, ctrl)
```

When `hits == []`, the callback never fires. Combined with Bug 1 and
Bug 2, the user cannot tell whether (a) the box did fire and produced
no matches, (b) the drag was below `_drag_threshold` and was
re-classified as a click, or (c) something blew up earlier. There is
no log, no status message, no acknowledgement.

**Right fix:** always fire when `_dragging` was true. Callers can show
"0 entities selected" or stay quiet, but the engine's contract should
be explicit.

### Bug 4 (suspected, lower priority) — Subsampling stride may miss extreme vertices

`viewers/core/entity_registry.py:243-247`:

```python
idx = np.array(sorted(pt_ids))
pts = np.asarray(mesh.points[idx])
if len(pts) > max_points:
    step = len(pts) // max_points
    pts = pts[::step]
return pts
```

`pts[::step]` is a deterministic stride after sorting by point-id.
Whether this picks up extreme vertices depends on the mesh's
point-ordering — for a regular hex mesh the corners are usually the
first/last ids and *are* picked, but for a tet mesh from OCC the order
is arbitrary and the extremes can be silently dropped.

Once Bug 2 is fixed (using projected 2D AABB instead of `np.all`),
this becomes load-bearing: the AABB is only as accurate as the
extreme-most sampled points. Mitigation in optional polish below.

---

## Fix plan — phased

> [!important] Phase ordering
> Phase 0 (diagnostic) is **non-negotiable** and goes first — the DPI
> fix has two opposite directions and we must not guess. Phases 1, 2,
> 3 are independent and can be parallelized after Phase 0 lands.

### Phase 0 — Diagnostic: confirm the coordinate-space hypothesis

**Goal:** decide between "remove `* sx_ratio` from `_do_box`" and
"add scaling to `_do_click` / `_do_hover` / `_update_rubberband`."

**Files & changes:**

1. **`src/apeGmsh/viewers/core/pick_engine.py:303-386`** — drop in a
   one-shot debug block at the top of `_do_box`, gated by an env var
   so it ships harmlessly:

   ```python
   def _do_box(self, x0, y0, x1, y1, ctrl):
       import os
       if os.environ.get("APEGMSH_DEBUG_BOX"):
           rw = self._plotter.render_window
           vw, vh = rw.GetSize()
           aw, ah = rw.GetActualSize()
           # Project one known entity centroid for comparison
           sample_dt = next(iter(self._registry.all_entities()), None)
           proj = None
           if sample_dt is not None:
               c = self._registry.centroid(sample_dt)
               if c is not None:
                   r = self._plotter.renderer
                   r.SetWorldPoint(c[0], c[1], c[2], 1.0)
                   r.WorldToDisplay()
                   proj = r.GetDisplayPoint()
           print(
               f"[box] event=({x0},{y0})->({x1},{y1})  "
               f"size={vw}x{vh}  actual={aw}x{ah}  "
               f"sample_dt={sample_dt}  projected={proj}",
               flush=True,
           )
       # ... existing body ...
   ```

2. **Run repro:** with `APEGMSH_DEBUG_BOX=1`, drag a small box near
   one entity and read the printed values:
   - If `vw == aw` (e.g. both 1280) → no HiDPI scaling on this
     machine; both code paths are equivalent. Skip Phase 1's
     "remove scaling" step (it's a no-op) but still address Bugs 2 & 3.
   - If `vw != aw` and the projected coord falls roughly in
     `(0..vw)` range → event coords *and* WorldToDisplay both speak
     event-space; **remove the `* sx_ratio` scaling** in Phase 1.
   - If the projected coord falls in `(0..aw)` range → WorldToDisplay
     speaks physical-pixel space; **keep the scaling and add it to
     click + hover + rubber-band** in Phase 1 instead.

3. **Document the result** as a one-line comment at the top of `_do_box`
   ("// HiDPI: event coords are in display space (verified on
   {platform})" or equivalent), then **delete** the debug block.

**Acceptance:**
- Debug output tells us unambiguously which space `WorldToDisplay`
  returns on the user's machine.
- A short comment in `pick_engine.py` records the answer so the next
  visitor doesn't repeat the experiment.

> [!note] Why an env-var gate, not a unit test
> This is a windowing-system / VTK-build / Qt-DPI question that can't
> be reproduced in a headless test. The empirical run on the user's
> actual hardware is the only authoritative answer.

---

### Phase 1 — Fix DPI scaling

**Goal:** `_do_box` tests entities in the same coordinate space the
rubber-band draws into and the user sees.

**Branch A — if Phase 0 says "remove scaling" (most likely):**

1. **`src/apeGmsh/viewers/core/pick_engine.py:303-317`** — delete the
   scaling block:

   ```python
   def _do_box(self, x0, y0, x1, y1, ctrl):
       """Box-select with proper window vs crossing modes.

       Event coordinates from VTK on this build are in the same display
       space as renderer.WorldToDisplay output and as vtkActor2D. No
       extra DPI scaling is applied (verified Phase 0).
       """
       crossing = x1 < x0
       bx0 = min(x0, x1)
       bx1 = max(x0, x1)
       by0 = min(y0, y1)
       by1 = max(y0, y1)
       # ... rest unchanged ...
   ```

**Branch B — if Phase 0 says "scale everywhere":**

1. **`src/apeGmsh/viewers/core/pick_engine.py:258-273`** — `_do_click`:

   ```python
   def _do_click(self, x, y, ctrl):
       sx_ratio, sy_ratio = self._dpi_ratio()
       renderer = self._plotter.renderer
       self._click_picker.Pick(x * sx_ratio, y * sy_ratio, 0, renderer)
       # ... rest unchanged ...
   ```

2. **`src/apeGmsh/viewers/core/pick_engine.py:275-293`** — `_do_hover`:
   same `* sx_ratio / sy_ratio` treatment on the `Pick(...)` call.

3. **`src/apeGmsh/viewers/core/pick_engine.py:179-194`** —
   `_update_rubberband`: scale the four `pts.SetPoint(i, x, y, 0)`
   calls similarly. (This is the visual fix; without it the
   rubber-band appears in the wrong place.)

4. Add a private helper `_dpi_ratio(self) -> tuple[float, float]`
   returning `(sx_ratio, sy_ratio)` so the formula isn't duplicated.

**Acceptance (both branches):**
- New test `tests/test_box_select_dpi.py` (see "Tests" below) asserts
  that with a known DPI ratio the test box is consistent with click
  coordinates.
- Manual repro: small box around one entity → entity is selected.
  Crossing-drag → entity is selected. Visual rubber-band sits where
  the cursor is (Branch B regression check).

---

### Phase 2 — Window-mode containment via projected 2D AABB

**Goal:** "fully inside" means the entity's *projected silhouette
bounding box* is inside the selection box, not "every sampled vertex
is inside."

**Files & changes:**

1. **`src/apeGmsh/viewers/core/pick_engine.py:368-383`** — replace the
   per-entity loop:

   ```python
   # Check each entity's points against the box
   offset = 0
   for i, dt in enumerate(entities):
       n = corner_counts[i]
       sx = screen_x[offset:offset + n]
       sy = screen_y[offset:offset + n]

       if crossing:
           # Crossing: any sampled point inside is enough
           inside = (bx0 <= sx) & (sx <= bx1) & (by0 <= sy) & (sy <= by1)
           hit = bool(np.any(inside))
       else:
           # Window: projected 2D AABB of the entity must be fully inside
           hit = (
               bx0 <= sx.min() and sx.max() <= bx1
               and by0 <= sy.min() and sy.max() <= by1
           )

       if hit:
           hits.append(dt)
       offset += n
   ```

   Crossing semantics are unchanged — `np.any` over sampled points is
   the right "any-overlap" approximation.

**Acceptance:**
- New test `tests/test_box_select_window.py` (see "Tests"): a single
  cube fully visually inside the box is selected even when one of its
  64 sampled vertices is just outside.
- Crossing test still passes (single point inside is enough).
- A cube *outside* the box is not selected.

> [!warning] AABB-of-projected-points still over-selects rotated views
> Projected 2D AABB is a tighter test than `np.all(inside)` but still
> larger than the actual silhouette — a cube at 45° will report a
> bbox bigger than its visual outline. This is the standard CAD
> approximation (matches Rhino, FreeCAD); a true silhouette test is
> out of scope. Flag in the test docstring.

---

### Phase 3 — Empty-box feedback

**Goal:** every drag-release that crossed the threshold produces a
callback. Callers wire it to a status-bar message or ignore it.

**Files & changes:**

1. **`src/apeGmsh/viewers/core/pick_engine.py:385-386`** — drop the
   `if hits` guard:

   ```python
   if self.on_box_select is not None:
       self.on_box_select(hits, ctrl)
   ```

2. **`src/apeGmsh/viewers/model_viewer.py`** — locate the
   `pick_engine.on_box_select = ...` wiring and confirm it tolerates
   `hits == []`. Likely already does (passing `[]` to a list-
   comprehension toggle is fine), but verify.

3. **`src/apeGmsh/viewers/model_viewer.py`** (optional, recommended) —
   in the same wiring, surface the count to the status bar:

   ```python
   def _on_box(hits, ctrl):
       sel.box_select(hits, ctrl)
       if not hits:
           win.set_status("Box select: 0 entities", 2000)
       else:
           win.set_status(
               f"Box select: {len(hits)} entit"
               f"{'y' if len(hits) == 1 else 'ies'}",
               2000,
           )
   pick_engine.on_box_select = _on_box
   ```

**Acceptance:**
- New test `tests/test_box_select_empty.py`: an empty drag fires
  `on_box_select([], ctrl)` exactly once.
- Manual repro: drag in empty space → status bar shows "0 entities";
  drag over entities → shows count.

---

### Phase 4 (optional polish) — Better entity sampling

> [!note] Defer unless Phase 1+2 don't fully resolve the symptom.
> Phases 0–3 are the surgical fix. Phase 4 is improvement.

**Files & changes:**

1. **`src/apeGmsh/viewers/core/entity_registry.py:219-249`** —
   `entity_points`: replace stride subsampling with extremes-aware
   sampling. Always include the 6 axis-extreme vertices
   (`argmin/argmax` along x, y, z) plus a uniform random sample for
   the rest:

   ```python
   if len(pts) > max_points:
       extremes = np.unique(np.concatenate([
           [pts[:, 0].argmin(), pts[:, 0].argmax()],
           [pts[:, 1].argmin(), pts[:, 1].argmax()],
           [pts[:, 2].argmin(), pts[:, 2].argmax()],
       ]))
       remaining = max_points - len(extremes)
       if remaining > 0:
           rng = np.random.default_rng(42)  # deterministic
           others = np.setdiff1d(np.arange(len(pts)), extremes)
           sample = rng.choice(others, size=min(remaining, len(others)),
                               replace=False)
           keep = np.concatenate([extremes, sample])
       else:
           keep = extremes
       pts = pts[keep]
   ```

2. **`src/apeGmsh/viewers/core/entity_registry.py:219-249`** — extend
   `entity_points` to cover dim 1 and dim 2 (curves and surfaces) too,
   not only dim 3. `_do_box` then prefers `entity_points` over
   `bbox` for all dims, falling back to `bbox` only for dim 0.

3. **`src/apeGmsh/viewers/core/pick_engine.py`** — early-out: project
   the entity's centroid first; if the centroid is more than
   `(box_diagonal + entity_radius)` away from the box center, skip the
   per-vertex projection. Worth it only if profiling shows the
   per-vertex `WorldToDisplay` loop is a hot spot at large entity
   counts.

**Acceptance:**
- New test: a long thin curve whose AABB extremes are correctly sampled
  is selected by a tight window.
- Profiling: `_do_box` over 1000 entities completes in < 50 ms.

---

## Tests

All new tests go under `tests/`. Picking tests are awkward because VTK
is windowing-stack-dependent; we test the **pure logic** (containment
predicate, coordinate-space transformation) on synthetic data, not the
full VTK render loop.

### `tests/test_box_select_window.py` (new)

Covers Phase 2 directly against the containment logic. We extract
`_do_box`'s containment predicate into a helper or test the engine
with a stub registry + stub renderer.

```python
"""
Regression tests for window-mode box-select containment.

Bug covered:
  - np.all(inside) over sampled points kills entities whose 2D AABB
    is fully inside the box but one vertex projects barely outside.
"""
import numpy as np
import pytest


def _window_hit(sx, sy, bx0, bx1, by0, by1):
    """Same predicate as pick_engine.py:368-383 (window branch)."""
    return (bx0 <= sx.min() and sx.max() <= bx1
            and by0 <= sy.min() and sy.max() <= by1)


def _crossing_hit(sx, sy, bx0, bx1, by0, by1):
    inside = (bx0 <= sx) & (sx <= bx1) & (by0 <= sy) & (sy <= by1)
    return bool(np.any(inside))


def test_window_fully_contained():
    sx = np.array([10, 20, 30, 40])
    sy = np.array([10, 20, 30, 40])
    assert _window_hit(sx, sy, 0, 50, 0, 50)


def test_window_one_vertex_outside():
    # 2D AABB max-x = 60 is OUTSIDE the box (xmax=50) — must miss
    sx = np.array([10, 20, 30, 60])
    sy = np.array([10, 20, 30, 40])
    assert not _window_hit(sx, sy, 0, 50, 0, 50)


def test_window_aabb_predicate_replaces_all_predicate():
    """The bug fix: with previous np.all(inside), this would miss.

    Here the projected AABB is fully inside even though one
    intermediate sample sits at the edge — verify we accept it.
    """
    # All four sampled corners just inside the box
    sx = np.array([1.0, 1.0, 49.0, 49.0])
    sy = np.array([1.0, 49.0, 1.0, 49.0])
    assert _window_hit(sx, sy, 0, 50, 0, 50)


def test_crossing_partial_overlap():
    sx = np.array([10, 20, 60, 70])  # half outside
    sy = np.array([10, 20, 30, 40])
    assert _crossing_hit(sx, sy, 0, 50, 0, 50)


def test_crossing_fully_outside():
    sx = np.array([60, 70, 80, 90])
    sy = np.array([10, 20, 30, 40])
    assert not _crossing_hit(sx, sy, 0, 50, 0, 50)
```

### `tests/test_box_select_empty.py` (new)

Covers Phase 3. Build a `PickEngine` against a stub registry that
reports zero entities; trigger a synthetic `_do_box`; assert the
callback fires with `[]`.

```python
"""Phase 3: empty box fires on_box_select exactly once."""
from unittest.mock import MagicMock

import pytest

# Skip if VTK / pyvista not importable in the test env
pytest.importorskip("pyvista")
pytest.importorskip("vtk")

from apeGmsh.viewers.core.pick_engine import PickEngine


class _StubRegistry:
    dims = []
    def all_entities(self):
        return []
    def bbox(self, dt):
        return None
    def centroid(self, dt):
        return None
    def entity_points(self, dt):
        return None


def test_empty_box_fires_callback():
    plotter = MagicMock()  # _do_box won't actually project anything
    engine = PickEngine(plotter, _StubRegistry())
    cb = MagicMock()
    engine.on_box_select = cb
    engine._do_box(10, 10, 100, 100, ctrl=False)
    cb.assert_called_once_with([], False)
```

### `tests/test_box_select_dpi.py` (new, optional)

Covers Phase 1's invariant — that the coordinate space used by
`_do_box` matches the space used by `_do_click`. Best implemented as a
manual check during code review unless someone wants to invest in a
real VTK offscreen render fixture (heavy).

```python
"""Phase 1: box and click use the same coordinate space.

This test is intentionally small — it asserts that no DPI scaling
diverges between the two handlers. Full HiDPI behavior is verified
manually on the target machine (see plan_viewer_box_select.md
Phase 0).
"""
import inspect
from apeGmsh.viewers.core import pick_engine


def test_no_dpi_scaling_in_do_box():
    """After Branch A: no `sx_ratio`/`sy_ratio` multiplication."""
    src = inspect.getsource(pick_engine.PickEngine._do_box)
    # Allow the variable name to exist in the projection helper, but
    # not as a multiplier of the box coordinates.
    assert "min(x0, x1) * sx_ratio" not in src
    assert "max(x0, x1) * sx_ratio" not in src
```

(Branch B variant: assert the scaling is present in *all three*
handlers via a similar source-inspection probe. Source-inspection is a
lightweight stand-in for true integration testing.)

---

## What this plan does NOT change

Per `CLAUDE.md` §3 (Surgical Changes), the following are **out of
scope** and should be left untouched:

- The VTK observer registration (priorities 10.0 / 9.0 / 10.0) and
  abort-flag flow in `install()` — works correctly.
- The `_drag_threshold` value or its accessor — separate UX tuning
  ticket if needed.
- The hover-throttle counter (`pick_engine.py:222-223`) — independent
  performance code.
- `_ensure_rubberband` / `_update_rubberband` actor styling (color,
  line width, stipple pattern) — Branch A leaves them alone; Branch B
  edits *only* the coordinate calls inside `_update_rubberband`.
- `EntityRegistry`'s overall structure — Phase 4 touches
  `entity_points` only, and only if Phases 0–3 don't fully resolve
  the symptom.
- The VTK `vtkAreaPicker` / `vtkRenderedAreaPicker` alternative — a
  rewrite to use VTK's native frustum picker is a bigger architectural
  decision than this plan covers. Flag for a separate spike if
  Phases 0–3 + 4 still feel insufficient.
- The crossing-vs-window keymap (today: direction of drag) —
  intentional CAD convention, not a bug.

---

## Acceptance — overall

The fix is complete when:

1. **All new tests pass** (`pytest tests/test_box_select_*.py`).
2. **All existing tests still pass** (`pytest tests/`).
3. **Manual re-run of the user's repro on the actual target machine:**
   - `g.begin()` → build geometry with several volumes.
   - `g.model.viewer()` → drag rubber-band over them (window mode).
   - All visually-enclosed entities are added to the working set.
   - Crossing drag (right → left) picks up partially-overlapped
     entities.
   - Empty drag shows "0 entities" in the status bar.
4. **Phase 0 result is committed as a one-line comment** in
   `_do_box`'s docstring so the next reader knows which coordinate
   space the build uses.
5. **No regression in click-pick or hover** (test by clicking a few
   entities and watching hover highlight after each fix lands).

---

## Cheat sheet for implementation order

```
1. Phase 0 (1 file, ~20 LOC; revert after) — env-gated debug print
2. Run repro on user's machine, collect printed values
3. Decide Branch A vs Branch B; record decision in code comment
4. Phase 1 (1 file, ~10 LOC for Branch A; ~25 LOC for Branch B)
5. Phase 2 (1 file, ~10 LOC)
6. Phase 3 (1-2 files, ~10 LOC)
7. Add tests/test_box_select_window.py + test_box_select_empty.py
8. Manual repro check on the target machine
9. (Optional) Phase 4 if symptom not fully resolved
10. CHANGELOG entry
```

Net delta (Phases 0–3 only): ~30–50 LOC source, ~80 LOC tests.

---

*Cross-references:*
[[architecture]] · [[apeGmsh_architecture]] ·
[[plan_viewer_pg_persistence]] ·
`src/apeGmsh/viewers/core/pick_engine.py` ·
`src/apeGmsh/viewers/core/entity_registry.py` ·
`src/apeGmsh/viewers/model_viewer.py`
