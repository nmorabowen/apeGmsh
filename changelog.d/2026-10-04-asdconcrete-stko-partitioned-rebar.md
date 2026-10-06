### ADDED — ASDConcrete3D.from_stko, ASDConcrete3D implex_alpha, and partitioned emit of embedded rebar

`ASDConcrete3D.from_stko(...)` builds the material from the STKO preset inputs
(`ft`, `fc0`, `fcp`, `fcr`, `ecp`, `Gt`, `Gc`, `pscale_t`, `pscale_c`); omitted
values fall back to the 1P preset (the default `Gt` assumes N and mm), and
`from_fc` is unchanged. It is exposed on the bridge as
`ops.nDMaterial.ASDConcrete3DSTKO`. `ASDConcrete3D` also gains `implex_alpha`
(`-implexAlpha`, `>= 0` with `0` turning the extrapolation off, emitted only
when different from 1; set without `implex` it warns). The STKO translator now
carries `implexAlpha` through to `ASDConcrete3D`, `-crackPlanes` included.

`g.rebar.resolve` now reads bar cells from the partition entities of each
curve, so a partitioned mesh keeps its bars. The partitioned emit routes each
`LadrunoEmbeddedRebar` tie to the rank that owns its host nodes and each bar
`CorotTruss` cell to the rank that owns both of its nodes; a rebar node on
another rank is ghost-declared (the shared-node mechanism already used for
`ASDEmbeddedNodeElement`). The single-process warning of `g.rebar.place` with
embedded coupling is removed.
