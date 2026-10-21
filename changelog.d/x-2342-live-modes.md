### ADDED — `capture_live_modes`: the modes of a plain openseespy deck, on any build (program slice #2342, gap a)

`apeGmsh.results.live.capture_live_modes(model_h5_or_fem, n_modes, path, *, ops=None)` solves
`ops.eigen(n_modes)` on the live domain of a deck the bridge did not build and writes one native
`mode_<k>` stage per mode, in the layout `EigenResult.to_native` writes. It works on pip openseespy,
where `emit_default` has no modal route and MPCO is absent. Each fem node is paired with the live node of
its tag by fact (the live domain holds it, at the FEM coordinates), its ndf is the live `getNDF`, and its
rotation slots follow the live elements' DOF roles. A rotation is written only where a beam, shell or
zero-length element touches the node. The vectors are the raw `nodeEigenvector` values. A node that is off
its FEM coordinates or missing from the deck, or a non-positive eigenvalue, refuses by name, and nothing is
written. The `emit_default` refusal for modal records now points at it.
