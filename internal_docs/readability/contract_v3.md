# apeGmsh Script English — contract v3

## Aim (fixed; never edit this section)

A model script written by an agent must read like a clear engineering procedure
to a structural engineer who opens it cold: top to bottom, one step at a time,
every object with one name, every number explained. The contract must NOT cost
modelling power or correctness: if a rule forces a weaker analysis, a hack, or a
dropped capability, the rule is wrong. Scope is user model scripts only, never
library code.

## Shape

S1. Order the script in sections that follow the docstring's numbered procedure, each opened by a one-line header (docstring with what, units and procedure -> imports -> data -> geometry -> groups, supports & loads -> mesh -> OpenSees model -> analysis settings -> analyses -> checks & report); imports go at the top, where a backend switch such as `matplotlib.use("Agg")` may sit between them, and the analysis chain may sit in one shared block ahead of the first analysis.
    ✗ `import opensees` and `import csv` at line 88, after the mesh (a)   ✓ docstring "5. Stage 1, geostatic ... 6. Stage 2, footing", then `# --- Stage 1: geostatic self-weight` (b)
S2. Write a flat procedure: no `main()`, no helper function, no step or part the task did not ask for, and nothing defined that the script never uses.
    ✗ `def chain(dlam): return dict(...)` called once per stage (c); an E = 1e12 footing block and an unused `import openseespy.opensees as ops_raw` (d)   ✓ two `with ops.stage(...)` blocks read straight down (b, f)
S3. Write outputs to a folder beside the script, never to a hard-coded absolute path.
    ✗ `Path(r"C:\Users\nmora\...\loop3\out_f")`; `out_dir + "\\summary.txt"` (f, d)   ✓ `OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_b")` (b)

## Fidelity

M1. Build the model the task describes with the library verb that names each part (material law, stages, supports, loads as given), never an elastic, unstaged, point-load, partial or hand-rebuilt stand-in.
    ✗ `ElasticIsotropic` soil with gravity as an element body force and `p.load(pg="Footing_top", forces=(0, -footing_force))` for a 300 kPa pressure (d); 600 kN on one node (a)   ✓ `g.loads.line(half, magnitude=FOOTING_PRESSURE * THICKNESS, ...)` in case "footing", then `p.from_model("footing")` in stage 2 (b)

## Vocabulary

V1. Refer to model objects by label, never by tag, list index or Python handle; every place the procedure names gets an engineering label, and loop-built labels are only for repeated members.
    ✗ `add_line(corner_bl, corner_br, ...)` on handles (f); `footing_lines = [f"H_{IX_FOOTING_LEFT}_{IY_SURFACE}", ...]` (e)   ✓ `geo.add_line("footing_centre", "footing_left_edge", label="footing_left_half")` (b)
V2. One object, one name, made once: a physical group keeps the name of the label it groups, and each group is made in one call from a list.
    ✗ `promote_to_physical("soil", pg_name="Soil")` (b; also c, d); `add_curve("side_right", name="sides")` then `add_curve("side_left", name="sides")` (f)   ✓ `g.physical.add_surface("soil", name="soil")` (f); `add_curve(side_lines, name="sides")` (e)
V3. Find an edge, joint or support by the label it got at creation, never by coordinates, a tolerance box or a node-id list, and key results by that name too.
    ✗ `np.isclose(abs(x), W/2) & (y > -D + 1e-6)` for the rollers; four `in_box(..., tol)` boxes to name unlabelled rectangle edges (c, d)   ✓ `g.constraints.bc("sides", dofs=[1, 0, 0])`, then `ops.fix_from_model()` (f)
V4. Every engineering number (dimension, material, load, mesh size, tolerance) is named in the data section, one per line, with its unit; counts, indices, `ndm`/`ndf`, multiples of a named length, textbook coefficients whose comment names their formula, and a setting whose comment says the analysis ignores it may stay bare, and a tuned value says it was tuned and why.
    ✗ `W, D, B = 20.0, 10.0, 2.0`; `H_FINE, H_COARSE = 0.25, 1.5` with no unit (c)   ✓ `SU = 75.0  # kPa, undrained shear strength (firm clay)` (b)
V5. State the unit system once in the docstring, and convert for printing by dividing by a named unit.
    ✗ `E, nu = 30e6, 0.3` with no unit system anywhere (a); `settlement * 1000.0` beside a defined `KPA` (f)   ✓ `MM = 1.0e-3  # m per mm`, then `settlement_final / MM` (b)
V6. Use a short symbol or formula form only as the textbook writes it; spell everything else out, and say in the name whether it is an index or a length.
    ✗ `K`, `G`, `M` as top-level names; `body_force=(0.0, -9.81 * GAMMA_SOIL / 9.81)` (c, d)   ✓ `b = HALF_FOOTING` inside the Flamant derivation, which writes b (b)

## Statements

T1. One modelling action per statement, short enough to read at a glance: no nested comprehensions, no long one-line formulas, no positional tuple rows, and no arithmetic inside an f-string.
    ✗ `print(f"final-slope / initial-slope : {(settlement[-1] - settlement[-2]) / (...) / slope:.2f}")` (f)   ✓ `err_slope = (slope_fe - slope_hand) / slope_hand`, then `print(f"FE / hand - 1 = {err_slope:+.1%}")` (b)
T2. Use the apeSees bridge as the one OpenSees handle: read results from its recorders with `np.loadtxt` under a comment saying what one row holds (the only route for a staged model), or, for live read-back, raw openseespy imported at the top as `ops_raw` with one `# raw-ops: <why>`.
    ✗ `for row in f: time, uy = row.split()` (b); `uy.reshape(len(uy), -1)[:, -1]` (c); `opensees.nodeDisp` imported mid-script (a)   ✓ `settlement_history = np.loadtxt(file_settlement)  # m, vertical displacement of the centre` (f)
T3. Write the hand check, and any constant derived from a textbook mapping, as a few named steps whose comment gives the formula or source and the convention used; prefer an equilibrium check of a named cut, and never use an unexplained factor.
    ✗ `(300.0 * 1e3 * footing_width / E) * 0.5` (a); `SIGMA_Y = 2.0 * C_U  # von Mises` (e, a Tresca mapping)   ✓ `SIGMA_Y = math.sqrt(3) * SU  # kPa, von Mises uniaxial yield for strength SU in pure shear` (b)
T4. Assert that the run succeeded (its status, or for a deck run one recorder row per increment) and that each check agrees within a named tolerance whose comment says why it is that size; report only numbers the run produced.
    ✗ ratio printed, never asserted (c); `summary.txt` says "Model ran successfully" with no result read (d)   ✓ `assert len(history) == N_STEPS_GRAVITY + N_STEPS_FOOTING, "a load step failed to converge"`; `TOL_SLOPE = 0.15  # ... (hand ignores the rigid base)` (e, b)
T5. Comments explain why and stay true to the code beside them: every non-default setting and any step a reader would not expect carries its reason, and a reason longer than the line goes above it.
    ✗ `# Emit and run via Tcl to avoid live analysis complexity` over a deck with no analysis (d); a 130-character trailing comment on `COHESION` (f)   ✓ `sig0=SIGMA_Y, sigInf=SIGMA_Y,   # perfectly plastic: no hardening` (b)

## Changelog

v3 (loop 3, strip footing). Contract beat control (readability 6.69 vs 4.38, fidelity 3.63 vs 2.25), but d shows the contract alone does not stop a weak writer from dropping the plasticity, the stages and the analysis, so M1 now names those parts.
Loosened: S1 lets a matplotlib backend switch sit among the imports (e, f friction); T2 makes recorder files the accepted staged route (b, e, f friction); T3 allows a short derivation instead of one statement (b, e, f friction); T4 accepts a recorder row count as proof of success (b, e friction). V3 stays strict although e found the grid route long: b and f name every edge at creation without a hack, and c, d's tolerance boxes were judged tangles; the missing block-grid helper is an API gap.
Added: S3 no absolute output paths (d, f tangles); V2 keeps one name per group (b, c, d) and one call per group (f); T1 bans arithmetic in f-strings (a, f); T2 asks for `np.loadtxt` over hand parsing (b, c); T3 covers derived material constants (e, f).
Merged: v2's T4 and T5 into T4 (both about not claiming unchecked results), to stay at 15 rules; T5 (was T6) drops its frame-only geomTransf clause, which this 2-D task could not test.
API gaps (double fix on shared corners, dt repeated with dlam, stage semantics, ops.py status, silent PG replacement, block grids, and more) went to api_gaps.md, not into rules.
