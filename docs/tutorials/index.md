# Tutorials

**Learning-oriented. One path, zero forks, guaranteed success.**

A tutorial is a lesson you can't fail. You follow it top to bottom, you
type what it says, and at the end something works — a real model that
runs OpenSees and prints a number you can check by hand. No decisions to
agonize over, no "it depends," no detours into theory. That comes later.

If you already know FEM and OpenSees but have never touched apeGmsh,
**these pages are how you build trust in the library.** Each one ends with
a closed-form answer (tip deflection, midspan moment, axial stress) so you
*know* the machinery did the right thing — not just that it ran without an
exception.

Once you've earned that trust, the [How-to recipes](../how-to/index.md)
answer specific "how do I…?" questions, the [Concepts](../concepts/index.md)
pages explain *why* the library is shaped the way it is, and the
[Examples](../examples/index.md) ladder works recognizable structural
problems end to end.

## Start here

→ **[T1 · Your first model in 10 minutes](first-model.md)**

Build an end-loaded cantilever start to finish in under 40 lines: open a
session, draw geometry, tag a physical group, mesh, snapshot the
`FEMData`, drive OpenSees through the typed `apeSees(fem)` bridge, read
results with `Results.from_native`, and view them in the browser with
`show_web`. You'll verify the tip deflection against **δ = PL³/3EI** and
see the deformed shape render inline. Zero Parts, zero naming theory, zero
strategy forks — just the spine, working.

## The track

| Tutorial | What you'll learn | You check |
|---|---|---|
| **[T1 · Your first model in 10 minutes](first-model.md)** | The whole apeGmsh → OpenSees spine on a line-element cantilever, in one unbroken path. | δ = PL³/3EI |
| **T2 · A plate in tension** *(coming)* | The same typed bridge on a 2D **solid**: `nDMaterial`, a pressure-loaded element set, reading a field back by physical group, and contouring it. | u = σL/E |
| **T3 · A simply-supported beam, the apeGmsh way** *(coming)* | The **composites** that make apeGmsh declarative: `g.loads.line` and load patterns, `g.masses`, `ops.section` — declare-then-resolve, no hand-written tributary loops. | δ = 5wL⁴/384EI, M = wL²/8 |
| **T4 · Save, reload, view** *(coming)* | Native persistence (`save_to` / `from_h5`) and the notebook-safe results loop — why `Results` needs `model=`, and why `show_web` is the viewer that doesn't crash your kernel. | Reloaded model reproduces δ |

Work them in order. T1 stands alone; T2–T4 each build on the muscle memory
of the one before. When you've finished T4 you'll have the full
build → solve → persist → view loop in your hands, and the
[Examples](../examples/index.md) ladder (portal frames, modal analysis,
fiber sections, pushover, staged SSI) is open to you.
