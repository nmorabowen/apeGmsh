### Skill: model-scripts re-cut onto the R3 API (R5-a, #1867)

The three reference scripts now use `one_element_per_curve`, `ops.analysis.chain`, the D3
`rigid_diaphragm` default and a dt-free `Static` run, so no `API gap` marker remains. Rule 8
teaches `ops_raw` and `# raw-ops: <why>` again. The bridge, cheatsheet and results references
cover `RunReport`, `ops.wipe()`, recorder `precision=`, `InertStaticDtWarning` and
`CurvesSkippedWarning`.
