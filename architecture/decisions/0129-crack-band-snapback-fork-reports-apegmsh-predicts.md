# ADR 0129 — Crack-band snap-back: the fork reports it, apeGmsh predicts it, a live test reconciles them

**Status:** Proposed (2026-11-01)

**Owner:** nmora

**Relation to earlier ADRs:**

- [ADR 0044](0044-asdconcrete-regularization-contract.md) is **extended**.
  Its Decision 5 rule still holds: the emit-time size check warns and never
  blocks emit. This ADR widens it from ASDConcrete on solids to every
  crack-band concrete and host, and states which side is the authority.
- The fork-side counter in D1 needs its own fork ADR in
  `Ladruno_implementation/`. This ADR states the contract apeGmsh relies on;
  the fork ADR owns the C++ design.

## Context

A crack-band concrete spreads its softening over one element length `lch`,
rescaling the fracture energy so the answer does not depend on the mesh.
Past a ceiling `l_max` the rescaled energy would be below the elastic
energy at peak (snap-back). The binary then clamps it to a floor, and the
element responds more brittle than the material and depends on the mesh.
The model is not malformed, so nothing fails and nothing in the results
shows it (ADR 0044, "The two facts").

Each side holds a different part of the picture. At FORK_PIN 23d937bc4:

| | The fork (run time) | apeGmsh (emit time) |
|---|---|---|
| The exact `lch` | yes. The element hands it to the material, per integration point, in the reference configuration, curved geometry included | mirrored. It re-implements each element's `getCharacteristicLength` from the corner nodes |
| Whether the clamp actually fires | yes, only where the material softens | no. It can only say the clamp *would* fire if that point softened |
| When the user hears about it | after the run has started, possibly hours in | before the run |
| How the report reads | per material tag and element tag, if at all | per physical group, one aggregated line |

What the fork reports today is uneven:

- `LadrunoConcrete3D` warns on the compressive limit:
  `"the element is too large for this Gc (compressive snap-back limit)"`
  (`LadrunoConcrete3D.cpp:712-715`).
- No tension-side warning was found.
- `ASDConcrete3D` clamps silently (`ASDConcrete3DMaterial.cpp:961-965`, per ADR 0044).
- The `_LadrunoRC` table floor (`regularize()`) clamps silently, per the #2648 review.

apeGmsh's emit-time sweep (`sweep_asdconcrete_element_size`) covers the
materials and hosts below. Concrete nested inside fiber, layered-shell and
wall sections is being added in a separate session.

| Material | Hosts |
|---|---|
| ASDConcrete | solids (ADR 0044) |
| `LadrunoConcrete3D`, `_LadrunoRC` | solids and plane elements (#2659) |

The fork's rule differs per element:

| Element family | `lch` at the pin |
|---|---|
| `LadrunoBrick`, `LadrunoBrick20` | `cbrt(V)` |
| `BezierTet10` | `cbrt(6V)` |
| `LadrunoQuad` | `sqrt(A)` |
| triangles | `sqrt(2A)` |
| `ASDShellQ4` | min node distance, halved with EAS |
| `MVLEM`, `MVLEM_3D` | the wall height `h` |
| force-based and Ladruno beams | `wt_i · L` per integration point, unless `-lch` or `-totalElementLch` overrides it |
| everything else | min node distance |

Mirroring those rules buys the early warning, but each one can drift
silently when the fork changes. That is the cost this ADR sets out to bound.

## Decision

**D1. The fork is the authority on whether snap-back happened.**

- Every crack-band material counts its own clamps. That covers
  ASDConcrete1D/3D, `LadrunoConcrete3D` and the `_LadrunoRC` family, on
  both the tension and the compression side.
- It keeps one count per material tag and side: how many integration
  points were clamped, and the worst `lch / l_max` seen. It warns once per
  material tag, never per step or per point.
- The counts can be read after `analyze`. The fork ADR picks the channel;
  the existing refusal counters are the model.
- apeGmsh treats a nonzero count as the fact. Any prediction of its own is
  advisory.

**D2. apeGmsh predicts it before the run, as a warning only.**

- The emit-time sweep stays read-only and never blocks emit (ADR 0044 D5).
- It emits one `ASDRegularizationWarning` per physical group and names the
  lever: refine the mesh or raise `Gf` for solids and shells; the
  integration rule or `-lch` / `-totalElementLch` for beams.
- Every mirrored `lch` rule cites its fork file and line at the pin it was
  read from.
- A host whose rule is not mirrored uses the base rule. It is listed in the
  sweep's docstring, so it is never a silent guess.

**D3. A live test reconciles the two (the drift guard).**

- The live-fork lane runs one deliberately oversized model per mirrored
  family: brick, tet, quad, triangle, shell, wall and beam.
- For each family, the elements apeGmsh flags must be exactly the elements
  the fork counts as clamped.
- A mismatch fails the lane. A FORK_PIN bump that changes an `lch` rule
  therefore fails before it ships, and does not leave a stale mirror.

**D4. Beams: which integration points the prediction checks. The owner decides.**

- In a force-based beam each integration point regularizes over its own
  tributary length `wt_i · L` (`ForceBeamColumn3d.cpp:1183-1188`).
- With Gauss-Lobatto the end weights are the smallest. For 5 points they
  are 0.05, 0.272, 0.356, 0.272 and 0.05, so the largest length is the
  interior `0.356 L`: 1067 mm on a 3 m member.
- An ASDConcrete1D at `E=30000`, `Gf=0.1`, `ft=3` has `l_max ≈ 667 mm`.
  Checking every point would therefore warn on most ordinary fiber beams.

| Option | Checks | Pro | Con |
|---|---|---|---|
| **(a) every point** | `max wt_i · L` | the honest contract: any point that softens can clamp | warns on most fiber beams, so the warning gets ignored |
| **(b) end points** | `wt_0 · L`, `wt_n · L` | where flexural hinges localize under lateral load | misses midspan localization under gravity |
| **(c) both, split** | the end points warn; the interior points are counted in the same message, labelled advisory | keeps the signal and the honesty | a longer message |

The proposal is **(c)**. D1 settles what actually clamped at run time, so
the prediction can be graded, not exhaustive. The owner confirms or picks
(a) or (b) before the beam slice lands.

## Consequences

- A user hears about a coarse mesh before the run, and gets an exact count
  after it.
- The mirror can still be wrong between pin bumps, but D3 catches it at the
  next bump.
- The fork gains counters on materials that clamp silently today. That is
  fork work, scoped by its own ADR.
- Until D1 ships, the D3 test cannot run, so the mirror is checked only by
  hand-oracle unit tests and source citations.

## Slices

1. **Done.** #2659: Ladruno concretes on solid and plane hosts.
2. **In progress.** Shells and walls, in the section-nested sweep session.
3. Beams, after the owner rules on D4.
4. A fork ADR and its implementation for D1, in the fork repository.
5. The D3 live reconciliation test, after slice 4 lands and FORK_PIN moves
   past it.

## Open questions

- D4: options (a), (b) or (c).
- D1: which channel carries the counts out of the fork (a recorder, a
  `getParameter` read, or the existing refusal-counter path). The fork
  ADR decides.
