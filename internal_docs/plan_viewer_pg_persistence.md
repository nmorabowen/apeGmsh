---
title: Viewer Physical Group Persistence — Fix Plan
aliases: [viewer-pg-persistence, plan_viewer_pg_persistence]
tags: [apeGmsh, viewer, physical-groups, bugfix, plan]
---

# Viewer Physical Group Persistence — Fix Plan

> **Status: delivered** — PG persistence shipped per recent viewer commits.

> [!summary] One-line
> Physical groups created in `ModelViewer` do not always end up in
> `gmsh` after the viewer closes. Multiple bugs in
> `viewers/core/selection.py` and `viewers/ui/viewer_window.py`
> conspire to make the failure invisible.

## Reported symptom

User reproduction:

1. `g.begin()` → build geometry.
2. `g.model.viewer()` → opens `ModelViewer`.
3. In the viewer, click **New Group**, name it `Foo`, pick some entities.
4. Close the viewer.
5. `g.physical.get_all()` (or `g.physical.entities("Foo")`) → group is missing.

The user sees **no error message** — the viewer simply closes and the
group is gone.

---

## Architectural context

The intended flow (per `docs/architecture.md` §2 and the viewer code in
`src/apeGmsh/viewers/`) is:

```
User clicks New Group ──► _on_new_group()                       (model_viewer.py:141)
                              │
                              ├─ sel._staged_groups[name] = picks
                              └─ sel.set_active_group(name)     (selection.py:216)
                                       │
                                       └─ fire on_changed
                                              │
                                              └─ commit_active_group() (selection.py:252)
                                                     │
                                                     └─ _write_group(name, picks)  → gmsh

User picks/unpicks ───► sel.toggle(dt)                          (selection.py:115)
                              └─ fire on_changed → commit_active_group() → _write_group()

User closes window ──► closeEvent()                              (viewer_window.py:101)
                              └─ _on_close()                     (model_viewer.py:125)
                                     └─ sel.flush_to_gmsh()      (selection.py:289)
```

**Key invariants the design relies on:**

1. `commit_active_group()` runs on **every** pick change — the active
   group is always synced live.
2. `flush_to_gmsh()` on close is a safety net for any staged group
   that was edited but never had `commit_active_group` fire.
3. Exceptions are NOT silently swallowed — the user must see errors.

Three of these are violated. See "Root causes" below.

---

## Decisions captured (resolved upfront)

| Decision | Resolution | Source |
|---|---|---|
| Empty active group on close → ? | **Delete from `gmsh`.** "What the user sees in the viewer = what `gmsh` has." | Inferred from user's "select then apply" mental model + safest UX |
| New Group with current selection → ? | **Inherit current picks.** Existing behavior is intentional. | User confirmed |
| Surface errors how? | **Log at error level + show on status bar in red.** Never silently swallow. | User confirmed |

> [!note] On the "empty group" decision
> The user's stated mental model is *select → apply*. The current
> implementation is *select-and-continuously-sync*. There is **no**
> Apply button in the viewer today. This plan keeps the live-sync
> behavior but adds a deletion path so an emptied active group is
> properly removed from `gmsh` (matching the user's intuitive mental
> model of WYSIWYG). If the user later wants an explicit Apply
> workflow, that is a separate UX change tracked outside this plan.

---

## Root causes

### Bug 1 — Empty active groups are never propagated

`viewers/core/selection.py:252-256` (`commit_active_group`):

```python
def commit_active_group(self) -> None:
    if self._active_group is not None and self._picks:
        self._staged_groups[self._active_group] = list(self._picks)
        _write_group(self._active_group, self._picks)
```

`viewers/core/selection.py:289-299` (`flush_to_gmsh`):

```python
for name, members in self._staged_groups.items():
    if members:
        _write_group(name, members)
        n += 1
```

`viewers/core/selection.py:233-238` (`set_active_group`, outgoing
write):

```python
if self._active_group is not None:
    members = list(self._picks)
    self._staged_groups[self._active_group] = members
    if members:
        _write_group(self._active_group, members)
```

All three guard with `if members:` (or `and self._picks`). Consequence:
once an active group has been emptied (Escape, unpick all entities, or
created empty and never populated), gmsh keeps the **previous
non-empty state** indefinitely. There is no path to delete a group's
contents from the viewer except an explicit `delete_group`.

**This is the most likely explanation for the user's reported repro.**
A common path: user creates an empty group first (`New Group` with no
prior selection), names it `Foo`, then picks entities. The first pick
fires `commit_active_group` and writes — so this path *should* work.
But if `commit_active_group` raises silently (see Bug 3), the write is
lost and the user has no signal.

### Bug 2 — Rename leaves a stale PG in gmsh

`viewers/core/selection.py:264-268`:

```python
def rename_group(self, old: str, new: str) -> None:
    if old in self._staged_groups:
        self._staged_groups[new] = self._staged_groups.pop(old)
    if self._active_group == old:
        self._active_group = new
```

This mutates the in-viewer dict only. It never calls
`_delete_group_by_name(old)`. The new name is written by the next
`commit_active_group` or by `flush_to_gmsh`, but **the old name is
never removed from `gmsh`**. After rename + close, `gmsh` contains
both `old` (stale) and `new`.

`_group_order` is also not updated — `old` stays in the order list,
`new` is missing.

### Bug 3 — Silent exception swallowing

`viewers/ui/viewer_window.py:101-106`:

```python
def closeEvent(self, event):
    if ui_self._on_close is not None:
        try:
            ui_self._on_close()
        except Exception:
            pass
```

`viewers/core/selection.py:323-328` (`_fire`):

```python
def _fire(self) -> None:
    for cb in self.on_changed:
        try:
            cb()
        except Exception:
            pass
```

Both bare `except Exception: pass`. Every callback failure —
including `commit_active_group` and `flush_to_gmsh` — vanishes
without trace. This is the reason the user has no diagnostic when
groups don't arrive.

### Bug 4 — `apply_group` does not register in `_group_order`

`viewers/core/selection.py:258-262`:

```python
def apply_group(self, name: str) -> None:
    self._staged_groups[name] = list(self._picks)
    if self._picks:
        _write_group(name, self._picks)
```

Used by `ModelViewer.to_physical()` (the public, headless API). The
group is written to gmsh but the in-viewer browser tree's order list
is not updated. Lower priority — only affects programmatic users who
combine `to_physical()` with a subsequent viewer reopen.

---

## Fix plan — phased

> [!important] Phase ordering
> Phase 0 (diagnostic) is **non-negotiable** and goes first — without
> it we cannot confirm whether subsequent fixes resolve the user's
> repro. Phases 1 and 2 are independent and can be parallelized.

### Phase 0 — Stop swallowing exceptions

**Goal:** every PG write/flush failure becomes visible.

**Files & changes:**

1. **`src/apeGmsh/viewers/ui/viewer_window.py:101-106`** — replace bare
   `except Exception: pass` in `closeEvent`:

   ```python
   def closeEvent(self, event):
       if ui_self._on_close is not None:
           try:
               ui_self._on_close()
           except Exception as exc:
               # Log to stderr — Qt status bar may already be torn down
               import traceback, sys
               print(
                   f"[viewer] on_close failed: {exc}",
                   file=sys.stderr,
               )
               traceback.print_exc(file=sys.stderr)
           # Always tear down the interactor
           try:
               ui_self._qt_interactor.close()
           except Exception:
               pass
           super().closeEvent(event)
   ```

2. **`src/apeGmsh/viewers/core/selection.py:323-328`** — `_fire`:
   replace silent swallow with logging. Keep the loop running (we
   don't want one bad callback to break the rest), but emit warnings:

   ```python
   def _fire(self) -> None:
       import logging
       log = logging.getLogger("apeGmsh.viewer.selection")
       for cb in self.on_changed:
           try:
               cb()
           except Exception:
               log.exception("on_changed callback failed: %r", cb)
   ```

3. **`src/apeGmsh/viewers/core/selection.py`** (`_write_group`,
   `_delete_group_by_name`, `_load_group_members`) — replace bare
   `except Exception: pass` with the same logging pattern. Specifically:
   - `_load_group_members`: lines 36-38, 41 — log instead of swallow.
   - `_delete_group_by_name`: lines 49-53 — log.
   - `_write_group`: no current swallow but wrap `addPhysicalGroup` and
     `setPhysicalName` in a try/log so a failure on dim N doesn't lose
     dims < N silently.

4. **`src/apeGmsh/viewers/model_viewer.py:125-129`** — make `_on_close`
   surface failures via the status bar in addition to logging:

   ```python
   def _on_close():
       try:
           n = sel.flush_to_gmsh()
       except Exception as exc:
           win.set_status(
               f"Failed to write physical groups: {exc}", 8000,
           )
           raise  # re-raise so closeEvent's logger picks it up
       if self._parent._verbose:
           print(
               f"[viewer] closed — {n} physical group(s) written, "
               f"{len(sel.picks)} picks in working set"
           )
   ```

   Note: `win.set_status` may not render if the window is already
   tearing down. The `print`/log path in step 1 is the reliable
   surface.

**Acceptance:**
- Manually inject a raise in `_write_group` (e.g. `raise
  RuntimeError("test")`); close the viewer; confirm the error appears
  in stderr.
- Existing tests pass unchanged.
- New test `tests/test_viewer_pg_logging.py` (see "Tests" below)
  asserts that an exception in a fake `_write_group` is logged, not
  swallowed.

---

### Phase 1 — Empty-active-group propagation

**Goal:** when the user empties an active group's picks, the group is
**deleted** from `gmsh` on the next commit/flush. Matches the
"viewer = ground truth" mental model.

**Files & changes:**

1. **`src/apeGmsh/viewers/core/selection.py:252-256`** —
   `commit_active_group`: drop the `and self._picks` guard. Empty picks
   means "delete":

   ```python
   def commit_active_group(self) -> None:
       """Write the current active group to gmsh, or delete it if empty."""
       if self._active_group is None:
           return
       self._staged_groups[self._active_group] = list(self._picks)
       if self._picks:
           _write_group(self._active_group, self._picks)
       else:
           _delete_group_by_name(self._active_group)
   ```

2. **`src/apeGmsh/viewers/core/selection.py:289-299`** —
   `flush_to_gmsh`: same treatment. Track which staged groups were
   emptied and delete them:

   ```python
   def flush_to_gmsh(self) -> int:
       """Write all staged groups to gmsh. Empty groups are deleted.

       Returns the count of groups *written* (deletions not counted).
       """
       if self._active_group is not None:
           self._staged_groups[self._active_group] = list(self._picks)
       n = 0
       for name, members in self._staged_groups.items():
           if members:
               _write_group(name, members)
               n += 1
           else:
               _delete_group_by_name(name)
       return n
   ```

3. **`src/apeGmsh/viewers/core/selection.py:233-238`** —
   `set_active_group`, outgoing-write branch: same treatment. If
   switching away from a group whose picks are empty, delete it:

   ```python
   if self._active_group is not None:
       members = list(self._picks)
       self._staged_groups[self._active_group] = members
       if members:
           _write_group(self._active_group, members)
       else:
           _delete_group_by_name(self._active_group)
   ```

> [!warning] Subtle interaction with Escape
> `clear()` (selection.py:121-128) sets `_active_group = None` AND
> clears picks. After `clear()`, `commit_active_group` is a no-op
> (active is None) — so Escape no longer triggers a deletion. This is
> deliberate: Escape means "clear the working set, don't touch the
> group." Confirm this matches the intended UX before merging.
>
> If the user wants Escape to also empty (and therefore delete) the
> active group, that's a one-line change in `clear()` — but flag it
> for review rather than assuming.

**Acceptance:**
- New test: create group `Foo` with picks; unpick all; `flush_to_gmsh`;
  assert `gmsh` has no PG named `Foo`.
- New test: create group `Foo` with picks; `set_active_group("Bar")`
  (switching away while empty); assert `Foo` is gone from gmsh.
- Existing test for "non-empty group survives close" still passes.

---

### Phase 2 — Rename writes immediately + cleans up old name

**Goal:** `rename_group(old, new)` removes `old` from gmsh and creates
`new` in the same call, with `_group_order` kept consistent.

**Files & changes:**

1. **`src/apeGmsh/viewers/core/selection.py:264-268`** — rewrite:

   ```python
   def rename_group(self, old: str, new: str) -> None:
       """Rename a staged group and propagate to gmsh atomically."""
       if old == new:
           return

       # Move staged entry
       members: list["DimTag"] = []
       if old in self._staged_groups:
           members = self._staged_groups.pop(old)
           self._staged_groups[new] = members

       # Update active pointer
       if self._active_group == old:
           self._active_group = new

       # Update creation-order list in place
       try:
           idx = self._group_order.index(old)
           self._group_order[idx] = new
       except ValueError:
           if new not in self._group_order:
               self._group_order.append(new)

       # Atomic gmsh rewrite: delete old, write new (if non-empty)
       _delete_group_by_name(old)
       if members:
           _write_group(new, members)
       # else: the group is empty and we already deleted old; nothing
       # to write.
   ```

2. **`src/apeGmsh/viewers/core/selection.py:258-262`** — `apply_group`:
   add the missing `_group_order` registration (Bug 4):

   ```python
   def apply_group(self, name: str) -> None:
       """Stage current picks as group *name* and write to gmsh."""
       self._staged_groups[name] = list(self._picks)
       if name not in self._group_order:
           self._group_order.append(name)
       if self._picks:
           _write_group(name, self._picks)
       else:
           _delete_group_by_name(name)  # consistency with Phase 1
   ```

**Acceptance:**
- New test: create `Foo` with picks; `rename_group("Foo", "Bar")`;
  assert `gmsh` has `Bar` and does **not** have `Foo`.
- New test: rename → switch active → flush; final state is `Bar`
  alone.
- New test: `apply_group("X")` then `apply_group("Y")` — `_group_order`
  contains `["X", "Y"]` in order.

---

## Tests

All new tests go under `tests/`. Use the existing pattern (likely
`pytest` with the `gmsh` fixture from `conftest.py`). **Do not** mock
`gmsh` — the architecture audit (`(v)` and `(xii)`) is explicit that
viewer/PG code touches the real gmsh. Use a tiny in-memory model.

### `tests/test_viewer_pg_persistence.py` (new)

Cover Phase 1 and Phase 2 directly against `SelectionState` and the
gmsh-IO helpers. No Qt — these are pure-state tests.

```python
"""
Regression tests for ModelViewer physical-group persistence.

Bugs covered:
  - empty active group not propagated (commit + flush)
  - rename leaves stale PG in gmsh
  - apply_group missing from _group_order
"""
import gmsh
import pytest

from apeGmsh.viewers.core.selection import (
    SelectionState, _load_group_members, _write_group,
)


@pytest.fixture
def gmsh_session():
    gmsh.initialize()
    gmsh.model.add("test_pg_persistence")
    # Create a few entities to use as PG members
    gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)   # vol 1
    gmsh.model.occ.addBox(2, 0, 0, 1, 1, 1)   # vol 2
    gmsh.model.occ.synchronize()
    yield
    gmsh.finalize()


def _pg_names() -> set[str]:
    return {
        gmsh.model.getPhysicalName(d, t)
        for d, t in gmsh.model.getPhysicalGroups()
    }


# ── Phase 1 ─────────────────────────────────────────────────────────

def test_empty_active_group_deleted_on_commit(gmsh_session):
    sel = SelectionState()
    sel.set_active_group("Foo")
    sel.pick((3, 1))
    sel.commit_active_group()
    assert "Foo" in _pg_names()

    sel.unpick((3, 1))
    sel.commit_active_group()
    assert "Foo" not in _pg_names()


def test_empty_group_deleted_on_flush(gmsh_session):
    sel = SelectionState()
    sel.set_active_group("Foo")
    sel.pick((3, 1))
    sel.commit_active_group()
    sel.unpick((3, 1))
    sel.flush_to_gmsh()
    assert "Foo" not in _pg_names()


def test_switching_from_empty_group_deletes_it(gmsh_session):
    sel = SelectionState()
    sel.set_active_group("Foo")
    sel.pick((3, 1))
    sel.commit_active_group()
    sel.unpick((3, 1))
    sel.set_active_group("Bar")  # switching away while Foo is empty
    assert "Foo" not in _pg_names()


# ── Phase 2 ─────────────────────────────────────────────────────────

def test_rename_removes_old_pg(gmsh_session):
    sel = SelectionState()
    sel.set_active_group("Foo")
    sel.pick((3, 1))
    sel.commit_active_group()
    assert "Foo" in _pg_names()

    sel.rename_group("Foo", "Bar")
    names = _pg_names()
    assert "Bar" in names
    assert "Foo" not in names


def test_rename_updates_group_order(gmsh_session):
    sel = SelectionState()
    sel.set_active_group("Foo")
    sel.pick((3, 1))
    sel.commit_active_group()
    sel.rename_group("Foo", "Bar")
    assert "Foo" not in sel.group_order
    assert "Bar" in sel.group_order


def test_apply_group_registers_order(gmsh_session):
    sel = SelectionState()
    sel.pick((3, 1))
    sel.apply_group("X")
    sel.unpick((3, 1))
    sel.pick((3, 2))
    sel.apply_group("Y")
    assert sel.group_order == ["X", "Y"]


# ── Reported repro ──────────────────────────────────────────────────

def test_reported_repro_create_then_query(gmsh_session):
    """Reproduction: create a group, query it after — must be there."""
    sel = SelectionState()
    sel.pick((3, 1))
    sel.pick((3, 2))
    # Simulates _on_new_group in model_viewer.py
    sel._staged_groups["Foo"] = list(sel._picks)
    sel.set_active_group("Foo")
    # Simulates close
    sel.flush_to_gmsh()

    assert "Foo" in _pg_names()
    members = _load_group_members("Foo")
    assert set(members) == {(3, 1), (3, 2)}
```

### `tests/test_viewer_pg_logging.py` (new)

Verify Phase 0: failures are logged, not swallowed.

```python
import logging
import pytest

from apeGmsh.viewers.core.selection import SelectionState


def test_on_changed_failure_is_logged(caplog):
    sel = SelectionState()
    def bad_cb():
        raise RuntimeError("boom")
    sel.on_changed.append(bad_cb)
    with caplog.at_level(logging.ERROR, logger="apeGmsh.viewer.selection"):
        sel.pick((3, 1))
    assert any("boom" in r.message or "boom" in str(r.exc_info)
               for r in caplog.records)
```

(A Qt-level test for `closeEvent` is intentionally skipped — Qt
testing is heavy and the stderr `print` is sufficient verification.)

---

## What this plan does NOT change

Per `CLAUDE.md` §3 (Surgical Changes), the following are **out of
scope** and should be left untouched:

- The on_changed callback chain wiring in `model_viewer.py` (works
  correctly, just needs better error surfacing).
- `_load_group_members` cross-dim semantics (matches by name across
  dims — separate concern, not the reported bug).
- `_on_new_group` inheriting current picks (confirmed intentional by
  user).
- The Qt event loop / `win.exec()` semantics in `viewer_window.py`
  (works in standalone scripts; Jupyter is a separate ticket if it
  surfaces).
- `g.physical` / `PhysicalGroups` — read side is correct, no cache.
- The HTML/trame backend commitment (separate roadmap item per
  architecture §11.2).

---

## Acceptance — overall

The fix is complete when:

1. **All four new tests pass.**
2. **All existing tests still pass** (`pytest tests/`).
3. **Manual re-run of the user's repro:**
   - `g.begin()` → build geometry
   - `g.model.viewer()` → create group `Foo` with picks → close
   - `g.physical.entities("Foo")` returns the picked entities.
4. **Manual error-path check:** monkeypatch `_write_group` to raise;
   close the viewer; confirm error appears in stderr (not silently
   swallowed).
5. **`docs/architecture.md` §11 (consistency audit) is updated** with a
   one-line note acknowledging that empty-group deletion is now part
   of the viewer contract.

---

## Cheat sheet for implementation order

```
1. Phase 0 changes (5 files, ~40 LOC)
2. Run existing tests — confirm green
3. Add tests/test_viewer_pg_logging.py — confirm green
4. Phase 1 changes (1 file, ~15 LOC across 3 methods)
5. Phase 2 changes (1 file, ~25 LOC across 2 methods)
6. Add tests/test_viewer_pg_persistence.py — confirm green
7. Manual repro check
8. Update docs/architecture.md §11 (one line)
9. CHANGELOG entry
```

Net delta: ~80 LOC source, ~150 LOC tests.

---

*Cross-references:*
[[architecture]] · [[apeGmsh_architecture]] ·
`src/apeGmsh/viewers/model_viewer.py` ·
`src/apeGmsh/viewers/core/selection.py` ·
`src/apeGmsh/viewers/ui/viewer_window.py`
