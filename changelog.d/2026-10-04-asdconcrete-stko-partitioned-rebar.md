### ADDED — ASDConcrete3D.from_stko, ASDConcrete3D implex_alpha, and partitioned emit of embedded rebar

`ASDConcrete3D.from_stko(...)` builds the material from the STKO preset inputs
(`ft`, `fc0`, `fcp`, `fcr`, `ecp`, `Gt`, `Gc`, `pscale_t`, `pscale_c`); omitted
values fall back to the 1P preset, and `from_fc` is unchanged. It is exposed on
the bridge as `ops.nDMaterial.ASDConcrete3D_stko`. `ASDConcrete3D` also gains
`implex_alpha` (`-implexAlpha`, emitted only when different from 1).

`g.rebar.resolve` now reads bar cells from the partition entities of each
curve, so a partitioned mesh keeps its bars. The partitioned emit routes each
`LadrunoEmbeddedRebar` tie to the rank that owns its host nodes and each bar
`CorotTruss` cell to the rank that owns both of its nodes; a rebar node on
another rank is ghost-declared (the shared-node mechanism already used for
`ASDEmbeddedNodeElement`). The single-process warning of `g.rebar.place` with
embedded coupling is removed.
