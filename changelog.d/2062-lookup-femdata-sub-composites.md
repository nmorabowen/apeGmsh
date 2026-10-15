### ADDED — `lookup` resolves `FEMData.nodes`, `.elements` and `.inspect` (SK2-S4b, #2062)

`python -m apeGmsh.studio.lookup` now indexes the FEMData sub-composites: `FEMData.nodes.*`
(with `.constraints.*`, `.loads.*`, `.masses.*` and `.sp.*`), `FEMData.elements.*` and
`FEMData.inspect.*`, 51 new entries, each routed to `references/fem-broker.md`. Before,
`lookup FEMData.nodes.select` missed. Every existing entry is unchanged. Three
`_SKILL_BY_PREFIX` rows whose value equalled the `skill_for` default (`g.model.select`,
`Cluster`, `Job`) are removed, so the studio surface shrinks by one line.
