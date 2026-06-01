# OpenSees fork (Ladruno) integration

apeGmsh can target the **Ladruno fork** of OpenSees (`nmorabowen/OpenSees`,
branch `ladruno`) in addition to stock `openseespy`. The fork adds features
apeGmsh emits/reads that stock OpenSees does **not** have. Stock `openseespy`
stays first-class — the fork is **opt-in**; gate fork-only features at the
point of use, never force the fork.

## Fork-only features apeGmsh touches

| Feature | Kind | Notes |
|---|---|---|
| **BezierTri6** | element | fork-only element |
| **ExplicitBathe / ExplicitBatheLNVD / CentralDifferenceLadruno** | explicit integrators | not in stock OpenSees |
| **EnergyBalance** | recorder | fork-only |
| **`.ladruno` recorder** | recorder | `recorder ladruno` — note `.ladruno`, a sibling of the vanilla `.mpco` |

The `.ladruno` recorder **does** write `MODEL/LOCAL_AXES` (per-class quaternion
`FRAME`) for beams — unlike vanilla `.mpco`, which omits beam local axes. Don't
carry the stale "MPCO carries no beam LOCAL_AXES" assumption into `.ladruno`
readers.

`Results.from_ladruno(...)` (model_h5 **optional** — a `.ladruno` is self-sufficient)
surfaces this as **`results.elements.local_axes(...)`** → a `LocalAxes` with per-element
scalar-first quaternions plus `.matrices` / `.x_axis` / `.y_axis` / `.z_axis`. The local
axes are the **rows** of each matrix (OpenSees `quatFromMat` stores the transpose), in
global coords — verified: a beam's `.x_axis` points along node1→node2. So beam
orientation for line/section-force diagrams comes straight from `.ladruno` (wired
classes; ElasticBeam3d today), **not** the native `vecxz` path. Energy lands via
**`results.energy(region=)`** → a DataFrame `KE/IE/DW/ULW/RES/ERR` (recorder `-G energy`).

## Contract lives in the fork repo

The exact emit/read contracts — command grammar, apeGmsh touch-points
(`_ELEM_REGISTRY` / `_response_catalog` / `Results.from_ladruno`), the
class-tag band, and the `.ladruno` schema notes — live in the fork's own
reference doc:

> `Ladruno_implementation/ladruno_apegmsh_contract.md` in
> `nmorabowen/OpenSees@ladruno`
> raw: `raw.githubusercontent.com/nmorabowen/OpenSees/ladruno/Ladruno_implementation/ladruno_apegmsh_contract.md`

**Read it before wiring any fork-only emitter or reader.**

## Class-tag band

Fork-only class tags live in the **private `≥33000` band**. Don't hardcode
the dead sub-300 values — read them live from the fork's `classTags.h` /
ledger. (See also `~/.claude/CLAUDE.md`: the OpenSees C++ source is at
`C:\Users\nmora\Github\OpenSees_Compile\OpenSees`.)
