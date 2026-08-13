# Handoff — ADR 0094 review findings (fix round)

> **Audience:** the implementing agent (Grok 4.6 experiment), responding
> to the maintainer review of PRs #930 / #938 / #932.
> **Scope:** two work items — **(A)** fix PR #932 on its own branch
> (`grok/adr94-s2-assess`, still open) before merge; **(B)** one new PR
> `grok/adr94-s1-fixes` on `main` for the follow-ups to merged PR #938.
> **Out of scope:** S3–S5, the deferred list at the bottom, anything not
> named here. Same rules as before: every changed line traces to a
> finding; ambiguities you resolve go in the PR description.
>
> Overall review verdict, for context: S0 clean; S1 faithful with
> follow-ups; S2 well-built with one **merge-blocking correctness bug**.
> PR-body judgment-call documentation was exemplary — keep doing that.

---

## A — PR #932 (S2 assess): fix in place, then it can merge

### A1. BLOCKER — hex8 signed volume is wrong (false `MESH.INVERTED` errors)

`src/apeGmsh/assess/_inverted.py` reuses the kernel's `_HEX8_TETS`
decomposition **signed**. In the kernel that list is consumed with
`abs()` per tet, which hides that the tet `(0, 7, 2, 6)` is listed with
**negative orientation**. Consequences (all verified numerically):

- On a perfect unit cube the signed sum is **2/3, not 1** (five tets
  +1/6, one −1/6).
- In a 200k random-hex sweep, ~0.2% of clearly **valid** hexes get a
  signed sum ≤ 0 → false `MESH.INVERTED`, which is FAIL-reserved
  severity (ADR 0094 INV-4). Zero false negatives were observed, but
  the false FAILs alone block merge.

**Fix:** make all six tets positively oriented. Flipping the offending
tet to `(0, 2, 7, 6)` is verified: all six volumes = +1/6 on the unit
cube, sum exactly 1.0. Leave the kernel's `_HEX8_TETS` untouched — its
`abs()` use is correct for volume; only the assess copy is wrong.

**Required regression test** (this exact hex defeats the current code:
every corner tet of the canonical decomposition has volume ≥ 0.149,
true volume ≈ 1.50, yet the current signed sum is −0.0033):

```python
_VALID_SKEWED_HEX8 = [
    ( 0.258, -0.357, -0.363),
    ( 1.087, -0.350, -0.320),
    ( 0.588,  1.381, -0.319),
    ( 0.260,  1.254,  0.224),
    (-0.369,  0.155,  0.888),
    ( 0.855, -0.028,  0.712),
    ( 1.416,  1.030,  0.805),
    (-0.392,  0.747,  0.764),
]
# assert: this hex8 does NOT produce MESH.INVERTED
# and: the same coords with two corners swapped (e.g. nodes 6/7 in one
# face, producing a genuinely negative canonical volume) DOES.
```

Also add a plain unit-cube hex8 case asserting no finding (the current
code passes it only because 2/3 > 0 — after the fix it should pass
because the sum is exactly the volume).

### A2. 2D cells in 3D space — orientation is a convention, not inversion

`_check_inverted` judges every `tri3` / `quad4` by projected-normal
sign (`src/apeGmsh/assess/_fem.py` ~87, `_inverted.py` `_tri_signed_area`).
For a **3D shell surface** (curved, or just not axis-aligned-planar) the
sign depends on viewing convention — a legally oriented shell facing
"down" is not an inverted element. This produces false FAILs on shell
models.

**Fix:** judge 2D cell types only when the mesh is planar — simplest
honest rule: all node coords coplanar to a small tolerance relative to
the model diagonal (a 2D model in any plane qualifies; a curved shell
does not). Otherwise add the 2D types to the skipped list with reason
`"2D cells in non-planar 3D space — orientation is not inversion"`
(INV-3: skip ≠ pass, listed). Test: a tri3 on an inclined plane in a
planar mesh → judged; the same tri3 in a mesh with out-of-plane nodes
→ skip-listed, no finding.

### A3. INV-3 gaps — checks that cannot run must be skip-listed

`src/apeGmsh/assess/_results.py`:

- `RES.UNBOUND_FEM` path (~line 40): the four mesh checks silently
  vanish. Add `("MESH.*", "no bound FEMData")` (one line covering the
  family is fine) to `skipped`.
- `RES.NO_STAGE` path (~line 66): `RES.U_VS_DIAG` and `RES.ENERGY_ERR`
  are skip-listed but the NaN/Inf check is not. Add
  `("RES.NAN", "no stages")`.

Test: assert both reasons appear under "Skipped checks" in
`report.text`.

### A4. Energy check — non-finite ERR is silently ignored

`_check_energy` (`_results.py` ~298): `if not np.isfinite(err) …:
continue` drops a NaN/Inf `ERR` on the floor. A non-finite energy
balance is at least as alarming as a large one.

**Fix:** a non-finite last-step `ERR` emits `RES.ENERGY_ERR` (message
says non-finite), it does not `continue`. While there, document in the
PR body (judgment call) that `np.isclose(err, 0.0)` uses an absolute
atol and is therefore unit-scale-sensitive in the office N·mm·tonne
convention — either keep it with that note, or compare |ERR| against a
relative reference if one is cheaply available from the energy table.
Test: patch an energy frame whose last `ERR` is NaN → finding present.

### A5. Minor — energy probe uses `stages[0]` only

The capability probe (`_results.py` ~270) calls
`results.energy(stage=stages[0].id)`; if energy happens to be absent
on stage 0 but present on a later stage, the whole check is skipped.
Cheap fix: treat the probe's `TypeError` as reader-wide skip (correct),
but on `ValueError` continue to the per-stage loop instead of
returning. Document either way as a judgment call.

---

## B — new PR `grok/adr94-s1-fixes` (follow-ups to merged #938)

All in `src/apeGmsh/viewers/render.py` unless noted. These land
**before S3 starts** — `render_pack` will sit directly on these paths.

### B1. Extract a shared deform-field reader (retire the third copy)

`_read_deform_field` (~385–417) re-implements the per-node dict-lookup
loop from `scripts/render_showcase/stills.py` (~95–111), unvectorized,
while the viewer has a vectorized reader closure-trapped in
`results_viewer.py` (~1683). **Fix:** extract one shared, vectorized
reader — natural home is next to `_compose_substrate_points` in the
pump-set module — and call it from `render.py`. Do **not** touch
`stills.py` (open question 3) and do **not** refactor
`results_viewer.py`'s internal use in this PR unless it is a pure
call-site swap; if it is bigger than that, leave the viewer side alone
and say so in the PR body. Test: deformed render output unchanged
(same PNG produced / same composed points) on the existing fixture.

### B2. Stop classifying every exception as "no GL"

`_open_plotter` / `_screenshot` (~214–219, ~269–276) catch broad
exceptions and report `[skip viewer] no GL context`, so a real bug
(bad argument, VTK API change) masquerades as an environment skip —
and S3's "no stills; numbers only" report line would silently mask
defects. **Fix:** only treat GL-context/initialization failures as the
skip path (match on the exception types VTK/pyvista actually raise at
plotter creation); everything else propagates. Test: monkeypatch the
screenshot to raise `TypeError` → the call raises, not skip.

### B3. Never delete a file this call did not create

`_screenshot` (~271–276) unlinks `out` on failure even when the file
pre-dated the call — a failed re-render destroys the previous good
still. **Fix:** only unlink when the file did not exist before the
attempt (or write to a temp name and rename on success). Test: seed a
file at the path, force a failure, assert it survives.

### B4. Out-of-range `step=` raises

`_resolve_step` (~329–335) clamps `step=99` → last. The PR body
claimed Python-style indexing; Python raises on out-of-range. A still
labeled with a step that was never rendered misleads an agent.
**Fix:** raise `ValueError` naming the valid range; keep genuine
negative indexing (`-1`, `-2`, …). Test both directions.

### B5. `view="deformed"` with no displacement raises

`_apply_deform` (~351–353) silently returns undeformed when no
`displacement_*` is recorded, so the agent reads a "deformed" PNG that
is not. Missing contour components already raise `ValueError`
(~131–136) — match that behavior. Test: results without displacement +
`view="deformed"` → `ValueError`.

### B6. CI must notice if real rendering silently dies

`test_render_without_skip_is_path_or_none` accepts `None`, so a VTK
offscreen regression on CI degrades the live-render test to a no-op
with no signal. **Fix:** env-gated hard assertion — when
`APEGMSH_EXPECT_GL=1` the test requires a `Path`; set that variable in
the CI `suite` lane (the runners have Mesa and already run exact-pixel
tests — see `tests/viewers/conftest.py`). Locally without the variable
the tolerant behavior stays.

### Optional riders (fold in if trivial, skip if not)

- CHANGELOG: add the missing blank line before `### REMOVED` from the
  #930 entry.
- `render.py` skip-env check runs before input validation (~82–90), so
  a bad call behaves differently headless vs live — reorder validation
  first if it stays a small diff.

---

## Process (same contract as before)

- One PR per work item: A on the existing `grok/adr94-s2-assess`
  branch; B as `grok/adr94-s1-fixes` from `main`.
- Tests land in the same PR as the fix they lock. Run
  `C:\Users\nmora\venv\opensees_venv\Scripts\python.exe -m pytest -q
  tests/assess tests/viewers/test_render.py` before pushing.
- PR bodies: numbered judgment calls, as you did — that part of the
  last round was exactly right.
- CHANGELOG per `internal_docs/changelog_workflow.md` (B only; A can
  amend the existing #932 entry if wording changes).

## Explicitly deferred (do not touch)

Substrate/theme pixel parity (rides with open question 3),
`registry.add` composition-ordering nit, regenerating
`docs/api-flows/atlas.html` / `flows.json`, `MESH.INVERTED` support
for `prism6` / `pyramid5`, any S3–S5 work.
