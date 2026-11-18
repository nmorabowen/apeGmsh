### ADDED — `ops.element.SSPquadUP`: the stock OpenSees u-p quad as a typed primitive

`ops.element.SSPquadUP(pg=, material=, thickness=, fluid_bulk=,
fluid_density=, perm=(k1, k2), void_ratio=, alpha=, body=(b1, b2))` emits the
UWelements `element SSPquadUP` (McGann & Arduino) to Tcl, openseespy, the live
route and model.h5, in `OPS_SSPquadUP`'s positional order. It runs on stock
openseespy. Every node is inferred `ndf 3` (ux, uy, p). The element joins the
capability registry, the u-p rotation-role families (DOF 3 is a pore
pressure, never `rotation_z`) and the response catalog (`stress` / `strain`,
one Gauss point, class tag 120). `s.update_parameter("hPerm" | "vPerm", k,
pg=...)` changes its permeability between stages.

Its pore pressure is the **velocity** of DOF 3 (`nodeVel(n, 3)`), and its
coupling term sits in the damping matrix, so reactions need `reactions
-dynamic`. The docstring and the skill say so. The bridge's typed results
routes do not yet expose that p as `pore_pressure`, and the live
`.read("reaction_force_*")` still calls a plain `reactions`.
