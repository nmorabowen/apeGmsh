### ADDED — Assembly v2 parity: anchor=, tie knobs, equal_dof_mixed, rigid_body, RBE knobs (AS5-b′, #1588)

`Assembly` v2 gains the v1 features the maintainer ruled to build (#1585):

- `instance(..., anchor="{instance}.{pg|label}")` places an instance at the
  centroid of a port of an instance declared before it (a physical group
  first, then a label), as v1's `compose(anchor=)` did. It refuses a nonzero
  `translate=`. The archive stores the resolved translate.
- `tie(..., stiffness=, stiffness_p=, rotational=, pressure=, control=,
  outward=)` passes the `g.constraints.tie` penalty knobs to the same
  `TieDef`.
- New verbs `equal_dof_mixed(master, slave, dof_pairs=)` and
  `rigid_body(master, slave, master_point=, as_element=, mass=, omega=)`.
- `couple(kind="kinematic"|"distributing", k=, kr=, enforce=)`, plus
  `al_update=` on RBE2. `k="auto"` and `k_alpha` need a host element, which
  the assembly form does not take, so they are refused.

Every new option is validated before it is recorded and persists in the
`/assembly/ties` params. A knob is written only when set, so existing
archives and their rows are unchanged; the zone stays at 1.0.0. `Assembly.h5`
cannot archive an `equal_dof_mixed` bridge yet: `apeSees.h5` refuses
`equalDOF_Mixed` (ADR 0069), before writing the file. ADR 0117 records that
cross-instance `penalty` and `tied_contact` are deferred.
