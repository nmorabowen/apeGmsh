# Analysis profiling — FEM runtime metrics (Ladruno / apeGmsh)

Companion to agentic postmortems: when a model (or stage) is **worth
measuring**, capture OpenSees **stack profiler** output so we can improve
mesh, solvers, contact, and skills (`opensees-performance`,
`opensees-expert`) with evidence — not folklore.

This is **analysis-time** profiling (`profile.h5`), distinct from the
Studio **token** sidecar (`python -m apeGmsh.studio.profile`).

---

## When to profile

| Do profile | Skip (for now) |
|------------|----------------|
| First acceptable stage of a progressive ladder | Tiny throwaway syntax checks |
| Before/after a solver, mesh, or contact change | Every single debug iteration |
| Cases that will become golden oracles | Runs that already fail setup |
| Comparing tet vs hex, kn, threads, algorithms | |

Discernment: same as progressive modeling — measure when the metric will
change a decision or feed a skill/backlog row.

---

## How (apeGmsh bridge)

**Deck path (recommended for Tcl batch):**

```python
ops.profiler.start(deep=False, memory=False, per_step=False)
ops.profiler.report(str(layout.root / "results" / "profile.h5"), run=case_id)
ops.tcl(str(layout.tcl), analyze_steps=n, run=True, bin=..., log=...)
```

**Live path:**

```python
ops.analyze(
    steps=n,
    profile=str(path_to_profile_h5),
    profile_run=case_id,
    profile_deep=False,
    profile_memory=False,
    profile_per_step=False,
)
```

## Windows / Tcl note

Pass **POSIX paths** to `ops.profiler.report(...)` (use `Path.as_posix()`).
Backslash paths are eaten by Tcl escapes and the report file never appears.
Call `ops.profiler.stop()` before `report` when the fork warns about rollup
while still enabled.

**Read:**

```powershell
$env:LADRUNO_PROFILER_VIEWER = "<github>/OpenSees/Ladruno_tools/profiler_viewer"
```

```python
import apeGmsh.profiler as profiler  # bridge to Ladruno profiler_viewer
with profiler.open("…/profile.h5") as pr:
    ...  # manifest / rollup / series / diff
# profiler.show_web("…/profile.h5")  # optional UI
```

Pattern (worked example: steel-connection) — the elastic driver reads a
profile env var (there `STEEL_PROFILE=1`, or a path) alongside its run flag
(there `STEEL_RUN=1`); see
`models/shear_connection/src/steel_connection_elastic.py` in that repo.
Give your own `models/<model_id>/src/` driver the same pattern.

---

## Where artifacts live

| Artifact | Location |
|----------|----------|
| `profile.h5` | `models/<id>/cases/<case>/results/profile.h5` (preferred) |
| Short rollup notes | case `run.json` extras or `logs/profile_notes.md` |
| Session-level takeaways | `postmortem/sessions/…/01_metrics.md` § analysis profiling |
| Backlog / skill feed | `postmortem/backlog/open.md` tag `fem-profile` → promote to skills |

---

## Two profilers (do not confuse)

| Profiler | Command / API | Answers |
|----------|---------------|---------|
| **FEM stack** | `ops.profiler.*` / `analyze(profile=…)` → `profile.h5` | Where analyze time goes (elements, solver, contact, …) |
| **Agent token** | `python -m apeGmsh.studio.profile` (sidecar; **not** an MCP tool) | Where agent tokens go (`src_search_rate`, habitat MCP, …) |

Habitat MCP (`status`, `run_until`, `lookup`, …) helps operate Studio; it
does **not** replace either profiler (ADR 0096: no profiler MCP tool).

---

## Feeding skills

After a useful profile:

1. Note hotspots in the case / postmortem (one paragraph + numbers).
2. If the pattern is general, open a `skills` or `opensees` backlog row
   aimed at **opensees-performance** / **opensees-expert** (examples,
   anti-patterns, solver ladders).
3. Prefer one concrete recommendation (“UmfPack→Pardiso on this DOF
   count”, “contact onset dominates after step k”) over dumping raw HDF5
   into chat.
