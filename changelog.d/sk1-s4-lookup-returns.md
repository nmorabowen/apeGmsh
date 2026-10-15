### ADDED — `lookup` resolves methods of returned objects (SK1-S4, #2008)

`python -m apeGmsh.studio.lookup` now indexes the methods of the objects
that indexed calls return, under the bare class name: `EigenResult`
(`ops.eigen`), `ComplexEigenResult` (`ops.complex_eigen`),
`ParallelModalResult` and `FEMData` (`get_fem_data`). So
`lookup EigenResult.to_native` and `lookup EigenResult.results` resolve,
and a bare `lookup to_native` lists `EigenResult.to_native` among its
matches. The classes come from an explicit allow-list in the index
builder, which raises if a listed class stops importing or has no skill
pointer of its own. The eigen family points at
`references/opensees-bridge.md` and `FEMData` at
`references/fem-broker.md`, not at the cheatsheet.
