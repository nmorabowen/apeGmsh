### CHANGED — `Results.demo()` writes a `.ladruno` (#2778, ADR 0131 D10 stage 0)

`make_demo_results()` (and `Results.demo()`) used to write a composed native
`demo_results.h5` and open it with `Results.from_native`. It now writes
`demo_results.ladruno`, stamped against the emitted `demo_model.h5` (ADR 0122
D3), and opens it with `Results.from_ladruno(path, model_h5=demo_model.h5)`.
The displacements are the same. The stage is now named `MODEL_STAGE[1]`
rather than `pushover`, because the demo model declares no `ops.stage`.

The file holds nodal displacements only. It writes no element bucket and says
so in `PROVENANCE@unsupported` (`["elements"]`); the elements bind from
`demo_model.h5`. No engine solved the values, so `opensees_backend` is
`"unknown"`. The file passes the fork's own `ladruno_format.validate()` at
`FORK_PIN`. Code that opened `demo_results.h5` as a composed native file must
read `demo_model.h5` for the model instead.
