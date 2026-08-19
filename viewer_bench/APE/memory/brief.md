# Project brief — viewer_bench

**Habitat:** The canonical fixture for exercising apeGmsh's Results
viewers — a soil-structure model built to feed every part of the ADR 0098
`ResultsSession` surface, not to answer an engineering question.
**Domain note:** Instructions under `APE/instructions/` are generic FEM
mechanics; *this file* is only this habitat's memory.
**Units:** kN, m, s, t — so E is kPa, density t/m³, mass tonnes. (Note
this overrides the template default of N, mm.)
**Status:** `ssi_frame_wall` is built, verified and recorded as a case.
Four findings so far, one fixed upstream and three open.

## Intent

The viewer is the product under test, so the model's job is coverage, not
insight: three element families in one scene, mixed 3/6-DOF nodes, nine
physical groups, three stages (static, six modes, transient), and real
data behind all seven §4 result slots. "Done" is `check_slots.py` green —
every slot filled, every pose rendered, the dynamic stage measurably
moving — because a slot with nothing behind it makes the viewer look
broken when it is not, and a frozen transient makes it look fine when it
is not.

Everything about *why the model is shaped the way it is* lives in the
habitat [`README.md`](../../README.md). This file does not restate it;
one source of truth, or the two drift.

## Members / layout

| Role | Section | Length | Notes |
|------|---------|--------|--------|
| soil block | 30 × 30 × 15 m, tet4 | — | E 320 MPa, ν 0.35, ρ 1.9; no self-weight, mass only |
| raft | 14 × 14 × 0.8 m, tet4 | — | conforming with the soil (fragmented, not tied) |
| columns | 0.50 × 0.50 | 3.2 m | `forceBeamColumn`, elastic section |
| beams | 0.30 × 0.50 | 6.0 m | 3 levels, 2 × 2 bays |
| grade beams | 0.40 × 0.60 | 6.0 m | z = 0 grillage; the tie's slave set |
| shear wall | t = 0.25 m | 6 × 9.6 m | `ShellMITC4`, one edge bay |
| slabs | t = 0.15 m | 12 × 12 m | `ShellMITC4`, 3 levels |

Gravity acts −z, applied as consistent nodal loads on the `dead` case
(raft volume, slab/wall traction, frame line load). The structure is
tied into the raft at z = 0 by `g.constraints.embedded(rotational=True)`;
the soil box is fixed at its base and rollered on its sides.

## Done when

1. `check_slots.py` reports PASS for all seven slots and all five poses.
2. The dynamic stage's peak-to-peak roof sway exceeds the gravity
   deflection by more than 10× (the frozen-transient guard).
3. Cases recorded under `models/ssi_frame_wall/cases/` with `run.json`.
4. Narrative + figures under `reports/` (not only `.apegmsh/visors/`).

## Open / next (engineering)

- Two viewer gaps are open, both "the data exists, the slot refuses":
  `contour`/`vector` cannot see derived scalars (`von_mises_stress`), and
  `sand` reads nodal components only. `check_slots.py` probes both every
  run and says `NOW SUPPORTED` the day either is fixed.
- The Tcl emitter writes numpy reprs into a `Path` time series.
- The `materials` scope axis still refuses (no element→material index).

## Agent intake (copy into new chats)

kN·m·s·t, fully elastic. A 30 × 30 × 15 m soil block with a 14 × 14 × 0.8 m
raft fragmented into its top surface (conforming, shared nodes), carrying a
3-storey 2 × 2-bay frame at 6 m bays and 3.2 m storeys: `forceBeamColumn`
columns/beams/grade-beams, `ShellMITC4` wall and slabs. Solid nodes are
ndf 3 and frame/shell nodes ndf 6; they meet only through
`ASDEmbeddedNodeElement` ties on the z = 0 grade-beam grillage. Soil base
fixed, sides rollered, soil self-weight deliberately off. Gravity ramps
over 10 static steps, then six modes, then 600 Newmark steps at dt = 0.01
under a generated 0.30 g pulse created *after* `loadConst` (it freezes
every pattern otherwise). Scripts in `models/ssi_frame_wall/src/`:
`build.py` is the driver, `check_slots.py` the verify. Both read and write
the current directory. Habitat root = this folder.

**APE door:** start at `APE/README.md`, then `APE/instructions/how-we-work.md`,
then this brief.

**Skills:** `APE/skills/catalog.md` + `catalog.json` (GitHub-canonical).
Refresh: `python APE/skills/harvest.py`.

**Libraries:** `APE/libraries/catalog.md` — bridges, apeGmsh, design,
access points. Refresh: `python APE/libraries/harvest.py`.
