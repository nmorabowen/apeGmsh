# apeGmsh Script English — contract v1

## Aim (fixed; never edit this section)

A model script written by an agent must read like a clear engineering procedure
to a structural engineer who opens it cold: top to bottom, one step at a time,
every object with one name, every number explained. The contract must NOT cost
modelling power or correctness: if a rule forces a weaker analysis, a hack, or a
dropped capability, the rule is wrong. Scope is user model scripts only, never
library code.

## Shape

S1. Order the script in sections, each opened by a one-line header: docstring (what, units) -> imports -> data -> geometry -> groups & loads -> mesh -> OpenSees model -> analysis -> checks & report; headers may sit inside the `with apeGmsh(...)` block, and every import goes at the top.
    ✗ `import openseespy.opensees as osp` at line 86, in the results block (b)   ✓ all imports before `# --- Data` (c)
S2. Write a flat procedure: no `main()`, no function called only once, and no step the task did not ask for.
    ✗ masses "for completeness" in a static run; an unasked 17-line plot (f, d)   ✓ the steps the task names, in order (c)

## Vocabulary

V1. Refer to model objects by label, never by tag; building labels in a loop over a named count is fine.
    ✗ `top[i]` / `bot[i+1]` as raw gmsh tags (b)   ✓ `add_point(i * PANEL, H, 0, label=f"T{i}")` (c)
V2. One object, one name: a group may collect many labels, but its name states the members' role.
    ✗ end posts filed in `"diagonals"`; `fix(pg="B0")  # pin` (d, c)   ✓ `name="pin_support"` (a)
V3. Find a joint or member by its name, never by its coordinates, and key results by that name too.
    ✗ `in_box((SPAN - eps, ...))`; `force_at[((xi, yi), (xj, yj))]` (b, a)   ✓ `gauss.get(label="B2-B3", ...)` (c)
V4. Every engineering number (dimension, material, load, mesh size, tolerance) is named in the data section with its unit on the same line; counts, DOF and array indices, `ndm`/`ndf` and multiples of a named length may stay bare.
    ✗ `7`, `6`, `4.0`, `200e9` inline (e)   ✓ `PANEL = 4.0  # m`, then `x_T2 = 2 * PANEL` (c)
V5. State the unit system once in the docstring, and convert for printing by dividing by a named unit.
    ✗ `kN = 1e-3; force * kN`, `* 1e3` for mm (d)   ✓ `KN = 1e3  # N per kN`, then `N / KN` (c)
V6. Use a textbook symbol only where the textbook uses it (E, A, H, P, L); otherwise spell the name out.
    ✗ `a` for the panel length (d)   ✓ `PANEL = 4.0  # m, panel length` (c)

## Statements

T1. One modelling action per statement: no nested comprehensions, no parenthesised chains, no positional tuple rows.
    ✗ `(g.model.select(...).in_box(...).to_physical("Pin"))`; `r[4]` (b)   ✓ one `fix(...)` per support (c)
T2. Use the apeSees bridge as the one OpenSees handle; where it cannot express a step, raw openseespy is allowed with `# raw-ops: <why>` on the line.
    ✗ an untagged `osp.eleResponse(...)` (b)   ✓ `ops_raw.eleResponse(ele, "axialForce")  # raw-ops: element force read-back` (a)
T3. Write a hand check as the physical statement (named cut, named moment point) in named variables, not as a general loop.
    ✗ `for k in range(1, n_loads_left + 1): lever = ...` (a)   ✓ `M_T2 = R_support * x_T2 - P * (x_T2 - PANEL)` (c)
T4. Assert that the analysis succeeded and that the hand check agrees within a named tolerance.
    ✗ `ops.analyze(steps=1)` with no status, check only printed (a, b)   ✓ `assert status == 0`; `assert abs(err) < TOL_CHECK` (c, d)
T5. Every reported number comes from the run; never type expected results by hand.
    ✗ a text block "Expected: ~300 kN chord, 10-15 mm" (e, f)   ✓ `N_checked = results.elements.gauss.get(...)` (c)
T6. Comments explain why, and every non-default setting carries its reason.
    ✗ `renumber(dim=1, base=1)` and `tol=1e-3, UmfPack` with no reason (d, f)   ✓ `# roller: free to slide, so no spurious chord force` (c)

## Changelog

v1 (loop 1). Contract scripts beat control on every axis (readability 6.6 vs 4.4, reader errors 0.13 vs 1), so the v0 core stays.
Loosened: V4 (old V3) now exempts counts, indices, `ndm`/`ndf` and multiples of named lengths (friction a, c, d); V1 allows label loops and no longer bans results-side maps (d); S1 allows headers inside the `with` block (a, c).
Clarified: V2 allows family groups of many labels, and asks that the name state the role (a, c, d friction; c/d tangles).
Added from tangles in two or more scripts: V3 by name not coordinates (a, b), V5 units (b, d, e), T4 asserts (a, b), T5 no hand-typed results (e, f), S2 no unasked steps (d, f), imports at top in S1 (b, f), T6 reasons for settings (d, f).
API gaps (raw read-back, tag mapping, capture ceremony, one element per truss member) go to api_gaps.md, not into rules.
