### ADDED — live read-back of fibres and layered shells off DomainCapture (#1951, #1767 G4)

Fibres and layered shells now have value tests on the recorder routes, for a
model the bridge builds. The fork lane runs
`tests/opensees/integration_ladruno/test_fiber_layer_readback_live.py` on
`.ladruno` and MPCO, and checks at `rtol=1e-10` against the live domain:

- A `forceBeamColumn` fibre section: fibre stress and strain against
  `eleResponse(e, "section", ip, "fiber", y, z, "stressStrain")`, and each
  fibre's `y`, `z`, area, material and Lobatto station against the section
  the bridge defined.
- An `ASDShellQ4` with a `LayeredShell`: the five components of layer stress
  and strain against `eleResponse(e, "material", gp, "fiber", k, ...)`, layer
  thickness against the `ShellLayer` definitions, and the local-axes
  quaternion against the frame `results/_shell_geometry.py` computes. That
  frame is itself checked against a closed form and against the quaternion
  the MPCO writer stores for a beam with the same axes.

Two routes return a silent wrong answer. They are strict xfails citing
#1995 (the `.ladruno` reader returns an empty slab for a layered shell) and
#1996 (the MPCO reader fills an identity quaternion for a shell). Pip
`openseespy` has no MPCO or `.ladruno` recorder, and `.out` carries neither
kind, so on a pip install fibres and layers have no route.
