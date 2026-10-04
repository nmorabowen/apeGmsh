# apeGmsh Script English — contract v0

## Aim (fixed; never edit this section)

A model script written by an agent must read like a clear engineering procedure
to a structural engineer who opens it cold: top to bottom, one step at a time,
every object with one name, every number explained. The contract must NOT cost
modelling power or correctness: if a rule forces a weaker analysis, a hack, or a
dropped capability, the rule is wrong. Scope is user model scripts only, never
library code.

## Shape

S1. Order the script in sections, each opened by a one-line comment header:
    docstring (what, units) -> data -> geometry -> groups & loads -> mesh ->
    OpenSees model -> analysis -> checks & report.
S2. Write it as a flat procedure. No `main()`; no function called only once.
    A helper called two or more times is fine, and should be defined just
    before its first use.

## Vocabulary

V1. Refer to model objects by label, never by tag. Pass `label=` to every
    geometry call whose result you need later; do not keep tags in dicts or lists.
V2. One object, one name. Do not create a physical group whose name differs from
    an existing label only in case or plural (`plate` / `Plate`). Reuse the name.
V3. Name every number. All dimensions, material values, loads and mesh sizes live
    in the data section, with units in a trailing comment. In calls, only 0, 1,
    -1 and DOF masks may appear bare.
V4. Textbook symbols (L, E, I, fy) are welcome as names, with units stated once.

## Statements

T1. One modelling action per statement. No nested comprehensions, no unpacking
    from generators, no clever one-liners.
T2. Use one OpenSees handle: the apeSees bridge. If the bridge cannot express a
    step, raw openseespy is allowed, but the line carries a comment
    `# raw-ops: <why the bridge can't>`.
T3. Hand checks go in named variables (`delta_eb = P * L**3 / (3 * E * I)`),
    then get printed. No formulas inside f-strings.
T4. Comments explain why, not what.
