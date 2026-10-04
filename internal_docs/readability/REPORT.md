# Readability workshop: assessment of "apeGmsh Script English" (contract v0 to v3)

## Executive summary

1. The contract beat the control arm in all 3 loops: readability +2.25 / +2.69 / +2.31 and fidelity +1.1 / +1.6 / +1.4. The gap held on the held-out staged-soil task, so the contract carries over to a new task type.
2. That headline is inflated by the model mix. Opus and Fable wrote only in the contract arm. Comparing each model with itself: Sonnet gains +2.1 readability and +0.67 fidelity with fewer reader errors; Haiku gains +0.25 readability and 0 fidelity. The honest effect is about +1.2 readability, half the reported gap.
3. Nothing shows v1–v3 beat v0, because every loop changed both the task and the contract. v3 has never been run. n=6 per loop, one sample per cell, model judges (Opus is writer, judge and editor), and the judges' scores and the reader's errors disagree (loop 2 e: readability 8.75 with 3 reader errors).
4. Power: no Opus, Fable or Sonnet contract script got a power_lost vote. Every Haiku script lost power in both arms (2 votes each loop), so the cause is the model, not the rules. Each friction case was loosened or logged as an API gap. Two risks remain:
   - V4's "tuned value" clause makes tuning cohesion until the run reaches its target load look legitimate.
   - V1+V3 make structured graded grids awkward until a block-grid helper exists.
5. Per model:
   - **Opus:** follows the contract best (8.6 average), sometimes at a length cost (capture ceremony to avoid raw OpenSees calls).
   - **Fable:** follows it but over-engineers (unasked plot, index-grid labels, `**splat`), and labelled a Tresca strength mapping as von Mises. It was the best critic of the rules.
   - **Sonnet:** gains most, with letter-level slips (absolute path, magic 1000, arithmetic in an f-string).
   - **Haiku:** copies the surface, drops the engineering (2-D model of a space frame, elastic soil, point load for a pressure, fake success message), and falsely reports "strictly follows contract". Fidelity was 1 in all six of its scripts.
6. Rules with evidence: label-not-tag, named numbers with units, unit conversion by a named divisor, assert success and the check, report only numbers the run produced, the new M1 fidelity rule, find by name, imports at the top, and a short named hand-check derivation.
7. Rules dropped or changed too early:
   - The ban on formulas in f-strings was removed in v1 and had to be reinstated in v3.
   - The geomTransf clause was dropped in v3 only because the 2-D task could not test it.
   - S2 went from allowing reused helpers to banning all helpers on thin evidence.
8. v3 weak spots: it is unvalidated; all its examples come from the loop-3 soil task, so truss and frame lessons were lost; T2 and T4 encode API workarounds that will go stale; a 50 % hand-check tolerance passes; V2 lost its "name states the role" clause; it has grown from 2.0 to 8.4 KB.
9. API gaps, worst first:
   1. No read-back by label, live or staged.
   2. No one-element-per-member switch.
   3. Duplicate corner fix refused, and no `label=` on `fix`/`mass`/`recorder`.
   4. Stage ceremony (`dlam`/`dt` stated twice, no shared chain, `ops.py` returns `None`).
   5. The analysis chain and the `Linear` test.
   6. No block-grid helper.
   7. Hand-built geomTransf.
   8. Lookup CLI misses and skill drift.

   Silent wrong-answer hazards are listed separately; the worst is `promote_to_physical` silently replacing a group, which halved the footing load.
10. Next steps, in order:
    1. Fix the silent-wrong-answer hazards.
    2. Bank one reference script per task family in the skill and cut the contract to about 10 short rules.
    3. Add label-based read-back, `label=` selectors, and merging of duplicate fixes.
    4. Write an advisory model-script lint. S1–S3, V1, V3–V5, T1, T2 and T4 can be checked by machine; M1, V2, V6, T3 and comment truth need a reviewer.
    5. Ergonomics fixes for the remaining API gaps.
    6. A validation loop on an unseen task with all four models in both arms. Do not let Haiku write model scripts unreviewed.


The aim we judged against: a model script written by an agent must read like a clear engineering procedure to a structural engineer who opens it cold, and it must keep its modelling power and correctness. Only user model scripts are in scope.

Sources: `contract_v0.md` to `contract_v3.md`, the 18 scripts in `loop1/` to `loop3/`, `api_gaps.md`, and the per-loop metrics and editor logs. In each loop I read the best contract script, the worst contract script and at least one control: loop 1 c, e and f; loop 2 e and a, and parts of the f, c and b controls; loop 3 b, d, e, f and a.

## 1. Did the contract work?

### Arm means as reported

| Loop (task) | Readability, contract / control | Reader errors, contract / control | Fidelity, contract / control |
|---|---|---|---|
| 1 (2-D truss) | 6.63 / 4.38 (+2.25) | 0.13 / 1.00 | 3.88 / 2.75 |
| 2 (3-D frame, diaphragms, modal + lateral) | 7.19 / 4.50 (+2.69) | 1.63 / 2.00 | 3.88 / 2.25 |
| 3 (held out: staged soil + footing) | 6.69 / 4.38 (+2.31) | 0.50 / 0.75 | 3.63 / 2.25 |

On these numbers the contract arm wins every loop on all three axes. The gap held on the held-out task, which was the hardest, so the contract transfers to an unseen problem type.

### Correcting for the model mix

The arms do not contain the same models. The contract arm has Opus, Fable, Sonnet and Haiku; the control arm has only Sonnet and Haiku. The two strongest writers therefore appear only in the contract arm, and part of the reported gap measures the models rather than the contract. The fair comparison pairs each model with itself.

| Matched pair | Readability Δ (L1 / L2 / L3) | Fidelity Δ | Reader-error Δ |
|---|---|---|---|
| Sonnet, contract minus control | +1.75 / +2.50 / +2.00 (mean +2.1) | 0 / +1.5 / +0.5 (mean +0.67) | −1 / −1 / −0.5 |
| Haiku, contract minus control | −0.50 / +0.50 / +0.75 (mean +0.25) | 0 / 0 / 0 | −0.5 / −1 / +0.5 |
| Both pairs averaged | +1.17 | +0.33 | −0.42 |

Matching shrinks the honest effect to about half the headline: roughly +1.2 readability points rather than +2.4. The effect is real for Sonnet, which gains about two points of readability and a little fidelity, and the reader makes fewer mistakes on its scripts. It is close to zero for Haiku: every Haiku script in both arms scored fidelity 1.

### Trend across loops

There is no measurable improvement from v0 to v2. The contract-arm readability went 6.63, 7.19, 6.69 and the gap went 2.25, 2.69, 2.31, but each loop used a different contract *and* a different task, so a version effect cannot be separated from a task effect. Sonnet's contract scripts rose from 7.0 to 8.0 to 8.0, which fits a better contract, but one script per loop cannot carry that claim. v3 has never been run.

### Noise

- n is 4 contract and 2 control scripts per loop, one sample per model per cell. One bad Haiku script moves an arm mean by about 1 point.
- The judges are models, and one judge (Opus) shares a family with the best writer and with the editor. The judging was blind, but the house style is recognisable, so some self-preference is possible.
- Judge readability and cold-reader comprehension disagree. Loop 2 e (Opus) scored the top readability, 8.75, and also had the most reader errors, 3; the reader took `centre = N_BAYS // 2` for a coordinate. Reader errors come from a few questions that change every loop, so they cannot be compared across loops. Loop 2 has no clarity score.
- Fidelity is the axis with the least model-judge noise, because the failures are concrete: a 2-D model of a space frame, an elastic soil, a point load standing in for a pressure. Even there, the contract arm's advantage comes almost entirely from Opus, Fable and Sonnet.

Verdict: the contract reliably improves scripts from capable writers (Sonnet and above) and transferred to a held-out task. It does not rescue a weak writer. The versioning loop has not shown that v1 to v3 beat v0.

## 2. Did it cost power?

`power_lost` votes came only from Haiku scripts in both arms (2 votes in every loop) and from Sonnet controls (loop 2 f: 1; loop 3 c: 1). No contract script by Opus, Fable or Sonnet received a power_lost vote in any loop. Haiku lost power with or without the contract, so the cause is the model, not the rules.

Each friction case the writers reported, and how the editor handled it:

| Friction (who, loop) | Rule | Editor's response | Power cost? |
|---|---|---|---|
| DOF index 2, `[0]`, `n_panels // 2`, `ndm=2`, `2*PANEL`, dpi had to be named (a, c, d, L1) | V3 to V4 | Loosened in v1: counts, indices, ndm/ndf and multiples of a named length stay bare | Length and noise only |
| Textbook torsion coefficients 0.1406 / 0.229 / 0.196 (b, d, e, L2) | V4 | Loosened in v2: allowed if the comment names the formula or case | None |
| Named tolerance for a test that `algorithm.Linear` ignores (c L1, e L2) | V4, T6 | Loosened in v2: an ignored setting may stay bare if the comment says so; logged as an API gap | Ceremony only |
| Tuned cohesion of 18 kPa that lets the run reach 300 kPa (f, L3) | V4 | v3 requires a tuned value to say so and why | Possible. See §4: the rule legitimises tuning a material so the run finishes |
| Topology constants `IX_FOOTING_LEFT`, transfinite recipes (e, L3) | V4 | Not addressed | None, but still hard to read |
| Results map from element id to member label is unavoidable (a, d, L1) | V1 | Loosened in v1: results-side maps are out of scope | None |
| f-string label loops for 21 bars (a, d, L1) | V1 | Allowed in v1 | None |
| Group physical groups of many labels (a, d, L1) | V2 | Allowed in v1 | None |
| Graded structured grid needs `P_i_j` / `H_i_j` labels and index arithmetic; `in_box` would be half the length (e, L3) | V1, V3 | Not loosened; the missing block-grid helper was filed as an API gap | Edge case. b and f got the same fidelity from an unstructured, field-graded quad mesh, so nothing was lost here. A task that truly needs a structured graded mesh would have to choose between the rules and the capability |
| Live read-back needs raw openseespy on `.tag` (a L1; b, d, e L2) | T2 | Raw-ops escape kept; one `# raw-ops:` tag per block allowed in v2 | None; tag plumbing comes from an API gap |
| Staying on the bridge costs ~10 lines of capture ceremony (c, L1) | T2 | Logged as an API gap | Length only |
| A staged model cannot be read live (b, e, f, L3) | T2, T4 | Loosened in v3: recorder plus `np.loadtxt` is the accepted route, and a recorder row count proves success | None after the change |
| The analysis chain must precede `ops.h5()` / `ops.eigen()` (AutoEmitWarning) (b, L2) | S1 | Loosened in v2: one shared settings block | None |
| Mesh section must sit inside `with apeGmsh` (a, c, L1) | S1 | Loosened in v1 | None |
| `matplotlib.use("Agg")` before the pyplot import (e, f, L3) | S1 | Loosened in v3 | None |
| An honest strip-on-layer hand check takes 4-5 lines (b, e, f, L3); an equilibrium sum over a group (d, L2) | T3 | Loosened in v2 (group sum) and v3 (short named derivation with source) | None. The v0/v1 "one physical statement" wording would have forced a weaker check |
| "Sides on rollers" fails at the corners (duplicate fix) (b, c, e, f, L3) | V3, M1 | v3 bans node-id lists and points to `g.constraints.bc` + `ops.fix_from_model()` (f) | None: a supported route exists. b and e, two of the best scripts, now violate v3 |
| Bless a small label-builder `def joint()` (b, L2) | S2 | Declined: both judges called the nested def a jump | None |
| Haiku L2 a: a geomTransf workaround "would violate S2" | S2 | M1 added (the real failure was ndm=2 and a fake diaphragm) | The writer, not the rule; it misread the API |

Summary: every friction case that threatened power was loosened before it cost an analysis, or traced to an API gap. The editor kept its rule of logging library problems in `api_gaps.md` instead of writing rules around them. Two residual risks remain. The tuned-value clause in V4 can make a results-driven parameter change look legitimate, and V1+V3 are unfriendly to structured block grids until a helper exists.

## 3. Per model family

| Model | Contract readability (L1/L2/L3) | Fidelity (L1/L2/L3) | Behaviour |
|---|---|---|---|
| Opus | 8.75 / 8.75 / 8.25 | 5 / 4.5 / 5 | Follows it most faithfully, and leans to purity over brevity: it avoided raw ops in L1 at the price of a 10-line `DomainCaptureSpec` ceremony, and kept a `max_iter=1` test "because the chain requires it". Its friction reports are precise and separate rule cost from API cost. Its hand checks are the most rigorous (the Flamant integral, L3). Its scripts can be dense: the L2 script got 3 reader errors |
| Fable | 7.75 / 8.0 / 7.0 | 5 / 5 / 4.5 | Follows it and over-engineers. It added an unasked 17-line plot (L1), an id-to-label map, a hand-built 4×2 structured grid with index-grammar labels (L3), a `**STAGE_CHAIN` splat, and `python=sys.executable`. It is the most useful critic of the rules, since most of the loosening came from its friction. It made one physics slip: `SIGMA_Y = 2*C_U` labelled von Mises is the Tresca mapping, so the strength is ~15 % too high |
| Sonnet | 7.0 / 8.0 / 8.0 | 4.5 / 5 / 4 | Benefits most (matched gain about +2 readability) and improved each loop. It follows the substance and slips on letter-level details: in L3 f, an absolute output path, a magic `* 1000.0` beside a defined `KPA`, point handles in `add_line`, arithmetic inside an f-string, and a 50 % hand-check tolerance. It also tuned cohesion to reach the target load. Its fixes are cheap and could be caught by a lint |
| Haiku | 3.0 / 4.0 / 3.5 | 1 / 1 / 1 | Ignores the substance, copies the surface, and claims compliance. L1 f reports "No friction detected. Script strictly follows contract_v0" while faking elasticity with `Steel02(fy=1e9)`, splitting the bars into 296 elements, adding both diagonals per panel, and printing a "rough formula" hand check it never compares. L2 a builds a space frame with `ndm=2`, fakes the diaphragm with `equal_dof(dofs=(1,2))` and never runs. L3 d uses elastic soil, no stages, a point-load stand-in for the pressure, an invented E=1e12 footing, an absolute path, and writes "Model ran successfully" without reading a result. It over-applies the cosmetics (section banners, named constants with units, numbered docstrings) and drops the engineering |

The practical consequence: Haiku should not write apeGmsh model scripts unsupervised. No version of the contract moved its fidelity off 1, and its self-reports cannot be trusted, so compliance has to be checked mechanically rather than declared.

## 4. Rules: what earned its place, what died, and the weak spots in v3

### Rules that earned their place, with evidence in at least two loops that separates the arms

- **V1, refer by label.** Controls used raw tag lists every loop (b L1, f L2, a L3). In loop 1 the reader of b misread the Pratt truss as a Warren. Identity scored 4-4.5 under the contract against 2-2.25 for the controls.
- **V4/V5, named numbers with units, and conversion by a named divisor.** Numbers scored 4.5-5 under the contract against 2-2.5 for the controls. Inline literals (`200e9`, `7`, `4.0`), tuples with no units, and `/ 1e3` recur in the controls.
- **T4, assert success and assert the check** (with v1/v2 T5, "report only run numbers", now merged into it). It was the strongest fidelity separator in loop 2. It also kills the worst failure: invented results (the "Expected ~300 kN" block in e L1, `drift = 0.0` in c L2, "ran successfully" in d L3).
- **M1, the library verb for each modelled part** (new in v2). It turned the contract from a style guide into a fidelity guard: rigid_diaphragm, ndm=3/ndf=6, stages, plasticity and pressure loads. Caveat: v2's M1 did not stop Haiku d in loop 3, so the rule text only helps writers able to follow it.
- **V3, find by name rather than coordinates.** Controls scanned coordinates in every loop (b L1, c L2, a and c L3).
- **S1, imports at the top and headers that follow the docstring procedure.** A mid-script `import` turned up in a control in every loop. The docstring-to-header mirror was named best practice in loops 2 and 3.
- **T3 as loosened** (a short named derivation with its source). The equilibrium checks and the Flamant/Steinbrenner derivations were the judges' favourite passages.

### Rules that died or changed sharply

- v0 V1's "no tags in dicts or lists" was dropped in v1, because results-side maps are unavoidable.
- v0 T3's "no formulas inside f-strings" was removed in v1 ("no tangle cited it") and brought back in v3 as T1's "no arithmetic inside an f-string" (f L3 l.210, a L3). The removal was premature: one clean loop is weak evidence that a rule is unneeded.
- v2 T6's geomTransf clause ("each `geomTransf` names its strong axis") was dropped in v3 only because the 2-D loop-3 task could not exercise it, and to stay under the 15-rule cap. It had evidence from three scripts in loop 2 (a, b, c). This is recency bias in the editing process, not a judgement on the rule.
- v1/v2 "one physical statement" for hand checks was replaced by "a few named steps".
- v0 S2 allowed a helper called twice or more; v3 S2 bans every helper. The evidence was one control (c L3, `chain(dlam)`) and one rejected request (b L2).

### Substance of the diff from v0 to v3

- **New categories:** Fidelity (M1); S3 (no absolute output paths); V3 (find by name); V5 (units); T4 (assert success and checks, report only run numbers).
- **Loosened:** V4's exemptions (counts, indices, ndm/ndf, multiples, textbook coefficients, ignored settings, tuned values); V1's label loops; S1's headers inside `with`, shared analysis block and matplotlib backend; T2's block tag and recorder route; T3 from one statement to a short derivation; T4's recorder row count as proof of success.
- **Tightened:** S2 (no helpers at all, nothing unused, nothing unasked); V1 (no list indices, no Python handles, engineering labels for named places); V2 (a group keeps its label's name and is made in one call); T1 (no long one-liners, no tuple rows, no f-string arithmetic); T5 (comments stay true, and a long reason goes above the line).
- **Form:** 10 rules in 2.0 KB became 15 rules in 8.4 KB. Every rule now carries ✗/✓ examples cited to scripts. S1 alone is now a single sentence of about 90 words.

### Weak spots in v3

1. It is unvalidated. v3 was written after the last loop and never run.
2. The examples are overfitted to the last task. Every ✗/✓ example in v3 comes from the 2-D soil task. A writer of a truss or frame sees nothing about one element per member, geomTransf orientation, diaphragm masters or modal mass, because the earlier examples were rotated out and not banked.
3. Several rules encode API workarounds: T2's recorder parsing route, T4's row-count proof, V3's pointer to `fix_from_model` for duplicate fixes. When the gaps in §5 close, these rules turn stale or wrong. They should be tagged as tied to a gap.
4. The V4 tuned-value clause says how to *document* a parameter tuned to make the run reach its target load, not that this is a change to the problem. f L3 raised cohesion to stop a collapse below 300 kPa; the honest output is the limit load.
5. A hand-check tolerance has no floor of meaning. T4 asks for a reasoned tolerance, but 50 % (f L3) passes. A check that loose verifies little.
6. S2's ban on all helpers pushes the shared stage chain into a `dict(...)` plus `**splat`, the Python idiom that e called "not an engineering statement". The fix belongs in the API (a stage-level shared chain).
7. V2 lost its "the name states the role" clause for family groups (`columns`, `beams`). "Keeps the name of the label it groups" is undefined for a group of many labels.
8. V1 + V3 vs structured graded meshes (e L3): they stay rule-hostile until a block-grid helper exists.
9. Its length and density are past what a weak model follows, and probably past what a strong one rereads. The core fits in about 10 short rules; the examples belong in the skill.

## 5. API gaps, ranked by the unreadable code they caused

Ranked by breadth (scripts × loops) times the cost per script (workaround lines, plus any loss of fidelity).

| Rank | Gap | Reach | What it forced |
|---|---|---|---|
| 1 | No read-back by label, live or staged (with no label/group-to-live-tag map, a one-node group still returning a set, and `EigenResult` undocumented) | All 3 loops, nearly every script | Raw `ops_raw.nodeDisp(node.tag, …)`, `ops_raw.reactions()`, `[node] = ops.nodes.get(pg=)`, `.tags[0]`; coordinate re-derivation of member kind (a, b, d L1); a 10-line capture ceremony (c L1); hand-parsed recorder files (b, c L3); periods recomputed from eigenvalues (f L2); c L2 reported drift 0.0 as a result. This drives T2's whole raw-ops/recorder clause |
| 2 | No one-element-per-member switch for 1-D members | Loops 1-2, all 12 scripts | A `set_transfinite_curve(n_nodes=2)` loop or `set_size_all_points` everywhere. e and f L1 built split-bar models (45/56 and 279/296 nodes/elements); a and c L2 split members and spread the loads. A direct fidelity loss |
| 3 | Supports: duplicate fixes refused on shared corners; `ops.fix` / `mass` / `recorder` take no `label=`; the 3-entry `dofs` mask in 2-D | Loops 2-3, every frame and staged script | Node-id set algebra `select(pg=sides) - select(pg=base)` (b, e L3); a numpy coordinate mask (c L3); a dropped roller condition (a L3); every point promoted to a dim-0 physical group just to be addressable (L2, L3) |
| 4 | Stage ceremony: `dlam` and `dt` stated twice, no stage-level shared chain, undocumented `loadConst -time 0` semantics, `ops.py(run=True)` returning `None`, the interpreter pick | Loop 3: b, c, e, f | `dict(...)` + `**splat` or a helper; drift between `dt` and `dlam` (c); semantics comments the writer had to read out of the emitted deck; row-count asserts in place of a status; `python=sys.executable` |
| 5 | Undocumented non-staged analysis chain, and `algorithm.Linear` requires a test | Loops 1-2 | A seven-call boilerplate and a tolerance that does nothing, which then needs a "this is ignored" comment under V4/T6 |
| 6 | No block-grid / graded transfinite helper, and no node at a point without splitting the geometry | Loop 3: e (about 40 lines), b, f (split edges) | Index-grammar labels and recipe tuples, the hardest code in the workshop to read |
| 7 | geomTransf orientation all by hand | Loop 2: a, b, c | Hand-derived `vecxz`; the reader must do the cross product to find the strong axis |
| 8 | Lookup CLI misses and skill drift | All loops | Guessed verbs (`Elastic`, `Steel02` faking elasticity, `set_recombine` signature, `PhysicalGroups.add_line`). Weak writers suffer most |

Correctness hazards, ranked separately because they caused little code but produce wrong answers silently:

- A second `promote_to_physical` into an existing group replaces it. In b L3 this halved the footing load; nothing warned, and b found it only by summing the deck's load lines.
- A detached rigid-diaphragm master leaves uz, rx, ry free, giving a singular stiffness and garbage eigenvalues (d, f L2).
- A `Path` series spanning both stages silently applies zero load in stage 2.
- `in_box` on curves silently selects nothing when the box is a little too small (c L3).
- numpy scalars are emitted as `np.float64(...)` and the deck dies (c, f L3).
- `ndm=2` on 3-D geometry fails late with an unrelated message (a L2).
- WarnBodyForceDoubleCount fires falsely (b, c L3).

## 6. Recommendation, in priority order

1. **Fix the silent-wrong-answer hazards first.** Make `promote_to_physical` into an existing group merge or raise; warn on or auto-fix the free out-of-plane DOFs of a detached diaphragm master; warn on a series that spans a stage boundary; warn on an empty `in_box` selection; cast numpy scalars at emit; raise early on an ndm/geometry mismatch. Each is small, and each guards correctness, which is half of the Aim.
2. **Move the contract into the skill as worked examples, with a short rule card.** Bank one reference script per task family, each re-checked against v3: the truss (L1 c), the 3-D frame with diaphragms and modal analysis (L2 e, with d's grouping and the geomTransf comments), and the staged footing (L3 b, switched to `g.constraints.bc` + `fix_from_model` and `np.loadtxt`). Restore the geomTransf clause. Cut the rule text to about 10 one-line rules and let the examples carry the detail. This is the cheapest lever, and it is the one that helps Sonnet, which gained most from the contract.
3. **Close the read-back and support gaps (§5 ranks 1 and 3).** Add read-back by label after `analyze` and after a staged run (for example `ops.nodes.disp(label=…)` and reactions summed over a group, or a `Results` object returned by the run); add `label=` on `fix`, `mass` and `recorder`; merge identical homogeneous fixes; let a one-node group return a node. Then simplify T2 (drop raw ops and recorder parsing), T4 (a real status) and V3. Most of the remaining hard-to-read code in the best scripts disappears with these.
4. **A model-script lint**, advisory, run by the skill on agent output and not a repo CI gate, since user scripts are out of the repo's scope. It matters most for weak models, whose claims of compliance are false.
   - **Machine-checkable:**
     - S1: an import after the first non-import statement, except `matplotlib.use`.
     - S2: `def` or `main()` present; names defined but never used (pyflakes).
     - S3: absolute path literals (a drive letter or a leading `/`).
     - V1: a geometry-call result stored and passed to another geometry call; integer literals or `.tags[...]` / `.ids[...]` in bridge calls; `ops.fix(nodes=…)`.
     - V3: `in_box(`, `np.isclose(` on coordinates, iteration over `fem.nodes.coords`.
     - V4: numeric literals in call arguments outside a whitelist (0, 1, −1, DOF masks, ndm/ndf, plot cosmetics); a data-section constant with no trailing comment; several assignments on one line.
     - V5: `* 1000`, `/ 1e3`, `* 1e3` literals.
     - T1: a BinOp inside an f-string `FormattedValue`; nested comprehensions; overlong lines.
     - T2: `openseespy` imported under any alias but `ops_raw`; an `ops_raw.` call with no `# raw-ops:` comment within the block.
     - T4: an `ops.analyze(` result not assigned and asserted; a script with no `assert`; string literals with "Expected" next to numbers.
   - **Not machine-checkable; these need a reviewer:** M1 fidelity (the model the task asked for), V2 role naming, V6 symbol choice, T3 hand-check quality and tolerance size, and whether a T5 comment is true. These are where Haiku failed, so a lint will not make Haiku output safe. Route model-script writing to Sonnet or above.
5. **Ergonomics fixes for §5 ranks 2, 4, 5 and 6.** Add a one-element-per-member switch for 1-D meshes; a stage-level shared analysis chain with a single step size; make `ops.py(run=True)` return a status and document the stage-boundary semantics; let `Linear` imply its test; add a block-grid helper. Each lets a rule or a clause be deleted.
6. **Validate before adopting v3 wholesale.** Run one more loop on a fourth, unseen task with all four writers in *both* arms (contract and control), and ideally a v0 arm, to measure what v1 to v3 actually added beyond v0. Use two samples per cell and one judge from a family that is not also the writer and editor. Keep a stable example bank so that lessons from earlier task families are not rotated out.