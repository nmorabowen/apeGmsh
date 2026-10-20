### FIXED — `Results.from_ladruno(model_h5=...)` binds the model's physical groups and labels, so the fork's default route reads by `pg=` and `label=` (#2329)

With `model_h5=`, `from_ladruno` bound the FEMData built from the file's own
`MODEL` group, which has no physical groups or labels. On the fork,
`emit_default` opens its `.ladruno` this way, so
`results.nodes.get(pg="Tip")` raised `KeyError: No group named 'Tip'`.

`from_ladruno` now picks the FEMData the way `from_mpco` does. `fem=` still
wins. Otherwise it binds the FEMData archived in `model_h5`, with its
physical groups and labels. A `model_h5` from another model or another mesh
warns `ModelFemMismatchWarning`, names the archive, and binds the file's
`MODEL` group instead. Without `model_h5=` nothing changes.

On a fork run of the two-node truss, the `pg=` and `label=` reads equal the
same reads from the stock route's `.mpco` at `rtol=1e-12`.
