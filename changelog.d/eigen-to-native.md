### ADDED — `EigenResult.to_native()` and `EigenResult.results()`: modes on pip openseespy (#1972)

`ops.eigen(n).to_native(path=None)` writes the modes as `kind="mode"`
stages of a native results file and returns its path (a fresh temp
directory when `path` is `None`). The layout is the one
`DomainCapture.capture_modes` writes: one stage per eigenvalue with
`eigenvalue`, `frequency_hz`, `period_s` and `mode_index`, and
`displacement_*` (plus `rotation_*` when the envelope `ndf` is at least 6)
read once per node from `nodeEigenvector`, zero-filled for a node with a
shorter vector. The file embeds `/model` and `/opensees`, so
`Results.from_native(p, model=OpenSeesModel.from_h5(p))` opens it alone;
`EigenResult.results(model_h5=None)` does that for you. `source_type` is
`"eigen"`, and the root attrs name the module that solved.

It raises `RuntimeError` when the live domain was rebuilt after the solve
(any later `LiveOpsEmitter`, from `modal_properties`, another `eigen` or a
second bridge, moves `domain_epoch()`), and `ValueError` without writing
anything for an empty `eigen_feast` band. `ParallelModalResult.to_native`
now shares its stage loop (`analysis/modal.py::_write_mode_stages`); its
output is unchanged.
