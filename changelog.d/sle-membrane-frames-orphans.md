### FIXED — a storey of `LadrunoMembrane(pg=...)` panels carries its load: per-panel frames, no orphan interior nodes

A floor modelled as several `LadrunoMembrane(pg=...)` panels with
`frames=("Beams",)` solved with `sum Rz = 0` and every eigenvalue
undetermined, and the fork printed `Singular matrix in
LadrunoMembrane::solve_linear_system`. Two bridge bugs caused it.

- `frames=` passed every element of the group to every panel, including the
  beams of the other panels and the out-of-plane columns. The kernel goes
  singular on a frame outside its panel and then transfers no load. The
  bridge now keeps, per panel, the frames whose nodes and chord midpoint lie
  in the panel's plane, on its outer boundary or inside it, and not inside a
  hole, within `1e-6` of the panel's bounding-box diagonal. A group (a PG or
  label name, or a `pg=` Element spec) is filtered. An explicitly named frame
  (an int tag, or a node-pair Element spec) that falls outside the panel is
  refused. When filtering leaves a panel with no frame and no
  `support_nodes=`, the build refuses and names the panel.
- The panel's interior mesh nodes, which only the panel's gmsh cells touch,
  were emitted as `node` lines with no element, so K was singular. They are
  no longer emitted on the flat, staged and partitioned paths, nor by an
  archive replay (`OpenSeesModel.build`, which reads the membranes' `pg` from
  `/opensees/decl_params`). FEMData keeps them. A `fix`, `mass`, nodal load,
  `sp`, constraint, `from_model` load, `mass_from_model()` mass or node
  recorder that targets one is refused, and the message says what to use
  instead (the membrane's `rho=`, `p.surface_load(ele=...)`, boundary
  nodes). `EigenResult.to_native` / `to_ladruno` and `capture_live_modes`
  write the deck's nodes.

The fork condenses a membrane's in-plane stiffness and mass onto its outer
loop only, so a hole node that no other element touches has no stiffness.
The build now warns `MembraneHoleNodeFreeWarning`; frame the opening.

The fork lane solves a three-panel storey (an interior beam, a framed hole,
columns and an `ASDShellT3` wall): `sum Rz = q * sum(A_i)` at rtol 1e-9, with
finite positive periods.
