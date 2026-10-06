# apeGmsh Script English — contract v2

## Aim (fixed; never edit this section)

A model script written by an agent must read like a clear engineering procedure
to a structural engineer who opens it cold: top to bottom, one step at a time,
every object with one name, every number explained. The contract must NOT cost
modelling power or correctness: if a rule forces a weaker analysis, a hack, or a
dropped capability, the rule is wrong. Scope is user model scripts only, never
library code.

## Shape

S1. Order the script in sections, each opened by a one-line header: docstring (what, units, numbered procedure) -> imports -> data -> geometry -> groups & constraints -> mesh -> OpenSees model -> analysis settings -> analyses -> checks & report; the headers follow the docstring's procedure, may sit inside the `with apeGmsh(...)` block, every import goes at the top, and the analysis chain may sit in one shared block ahead of the first analysis.
    ✗ `import openseespy.opensees as op` at line 115; a load pattern placed before the modal step the procedure lists first (f, d)   ✓ "1. modal ... 2. lateral load" in the docstring, then `# --- Analysis settings (shared by both runs)` before `ops.eigen` (e, b)
S2. Write a flat procedure: no `main()`, no function called only once, no step the task did not ask for, and nothing defined that the script never uses.
    ✗ `nDMaterial.ElasticIsotropic(...)` and `section.Elastic(...)` built and never used (a, c)   ✓ `elasticBeamColumn(pg="columns", A=A_COL, E=E, G=G, J=J_COL, ...)` with nothing else built (b)

## Fidelity

M1. Build the model the task describes with the library verb that names each part, never a 2-D, partial or hand-rebuilt stand-in.
    ✗ `ops.model(ndm=2, ndf=3)` for a space frame; `equal_dof(..., dofs=(1, 2))` called a rigid diaphragm; periods as `2 * math.pi / np.sqrt(eigenvalues)` (a, f)   ✓ `ops.model(ndm=3, ndf=6)`, `g.constraints.rigid_diaphragm(...)`, `modal.periods` (b)

## Vocabulary

V1. Refer to model objects by label, never by tag; building labels in a loop over a named count is fine.
    ✗ `add_line(pts[k - 1, i, j], pts[k, i, j])` on raw tags; `col_0`, `col_1` from a counter (f, c)   ✓ `add_point(i * BAY, j * BAY, k * STOREY, label=f"J{i}{j}_L{k}")` (e)
V2. One object, one name, made once: collect each label into its group list in the loop that creates it, and name the group for its members' role.
    ✗ column and beam lists rebuilt by repeating the creation loops; `nodes_3d` tags beside `node_labels_by_level` for the same joints (a, d, c)   ✓ `columns.append(label)` right after `add_line(..., label=label)`, then `add_curve(columns, name="columns")` (b)
V3. Find a joint or member by its name, never by its coordinates, and key results by that name too.
    ✗ floor nodes found by scanning `fem.nodes.coords` for z within 0.01 (c)   ✓ `p.load(pg=f"floor_{k}_centre", ...)` (e)
V4. Every engineering number (dimension, material, load, mesh size, tolerance) is named in the data section with its unit on the same line; counts, indices, `ndm`/`ndf`, multiples of a named length, textbook coefficients whose comment names the case they belong to, and a setting whose comment says the analysis ignores it may stay bare.
    ✗ `NBAY, BAY, NSTOREY, H = 2, 6.0, 2, 3.2` with no units; `tol=1e-10` beside a named `TOL_EQUILIBRIUM` (f, d)   ✓ `J_BEAM = 0.229 * BEAM_DEPTH * BEAM_WIDTH**3  # m^4, torsion constant for depth/width = 2` (d)
V5. State the unit system once in the docstring, and convert for printing by dividing by a named unit.
    ✗ `SEISMIC_FORCE_TOTAL / 1e3` (a)   ✓ `KN = 1e3  # N per kN`, then `V_BASE / KN` (b)
V6. Use a short symbol or formula form only as the textbook writes it; spell everything else out, and say in the name whether it is an index or a length.
    ✗ `bc`, `hb`; `FLOOR_MASS * 2 * L**2 / 12`; `centre = N_BAYS // 2` read as a coordinate (f, d, e)   ✓ `M_FLOOR * (LX**2 + LY**2) / 12`; `ix_centre` (b)

## Statements

T1. One modelling action per statement, short enough to read at a glance: no nested comprehensions, no long one-line generators or formulas, no positional tuple rows.
    ✗ `R_base_x = sum(ops_raw.nodeReaction(node, 1) for node in fem.nodes.select(...).ids)`; a 130-character `J_BEAM` (b)   ✓ `for node in base_nodes: reaction_x += ops_raw.nodeReaction(node.tag, 1)` (d)
T2. Use the apeSees bridge as the one OpenSees handle; where it cannot express a step, raw openseespy is allowed, imported at the top as `ops_raw`, with one `# raw-ops: <why>` on the line or over the block.
    ✗ an untagged `op.nodeDisp(t, 1)` with `op` next to `ops` (f)   ✓ `ux_1 = ops_raw.nodeDisp(floor_1_centre.tag, 1)  # raw-ops: live displacement read-back` (e)
T3. Write the hand check as one physical statement (equilibrium of a named cut, moment about a named point) in named variables; summing reactions over a named group is fine.
    ✗ `for k in range(1, n_loads_left + 1): lever = ...` (loop 1, a)   ✓ `err_base_shear = (R_base_x + V_BASE) / V_BASE` (b)
T4. Assert that the analysis succeeded and that the hand check agrees within a named tolerance.
    ✗ `ops.analyze(steps=1)` with the status dropped and no check; only "periods positive" tested (f, a)   ✓ `assert status == 0`; `assert abs(err_base_shear) < TOL_CHECK` (b, d, e)
T5. Every reported number comes from the run; never type results by hand.
    ✗ `story1_drift = 0.0` written to the CSV as a result (c)   ✓ `u_L1 = ops_raw.nodeDisp(node_L1, 1)`, then `drift_L1 = u_L1 / STOREY` (b)
T6. Comments explain why and stay true to the line beside them; every non-default setting, each `geomTransf` and any step a reader would not expect carries its reason.
    ✗ `set_global_size(1.0)  # one element per member`; `floor_mass_total / n_storeys`; a bare `ops.run(wipe=True)` (a, c, d)   ✓ `# Beams: local z along global Z, so local y is horizontal and Iy is the strong axis.` (e)

## Changelog

v2 (loop 2). Contract beat control again (readability 7.19 vs 4.5, fidelity 3.88 vs 2.25), so the v1 core stays; a's failure (2-D model, fake diaphragm) shows the contract must guard fidelity, not only style.
Added: M1 library verb for each modelled part (a, c, f tangles); V2 now asks for labels collected where created (a, d tangles; c parallel identities); S2 bans unused objects (a, c).
Loosened: S1 allows a shared analysis-settings block before the first analysis (b friction, AutoEmitWarning); T2 allows one raw-ops tag per block (b, e friction); T3 allows a sum over a named group (d friction); V4 lets textbook coefficients and ignored settings stay bare (b, d, e friction).
Sharpened: T1 bans long one-liners (b), V6 covers formula forms and index-vs-length names (d, e, f), T6 asks comments to stay true and each geomTransf to name its strong axis (a, b, c, d, e).
API gaps from loop 2 (live read-back, NodeSet singletons, detached diaphragm masters, one element per member, AutoEmitWarning order, lookup misses) went to api_gaps.md, not into rules.
