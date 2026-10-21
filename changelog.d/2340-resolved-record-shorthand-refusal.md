### FIXED — a resolved recorder record refuses a shorthand that records nothing, by name (#2340)

`ResolvedRecorderRecord(category="nodes", components=("displacement",))`
wrote no recorder on any route: no `.out` file, no MPCO `-N` token and
nothing in the fork's `.ladruno` file. The run then read back empty, and
nothing warned. A resolved record carries canonical names, and no route
expands a shorthand there. The record now refuses such a component at
construction, by name. The message gives the component's
`expand_shorthand(...)` expansion and points to `ops.recorder.declare(...)`,
which expands shorthands.

The refusal covers:

- the twelve nodal family shorthands (`displacement`, `velocity`,
  `acceleration`, `displacement_increment`, `rotation`, `angular_velocity`,
  `angular_acceleration`, `force`, `moment`, `reaction`, `reaction_force`,
  `reaction_moment`);
- `section_force`, `section_deformation`, `shell_resultant` and
  `shell_deformation` on a gauss record, and every other shorthand outside
  the category it records in;
- any name outside the canonical vocabulary, such as a typo.

Four shorthands still record as written: `stress`, `strain` and
`plastic_strain` on a gauss record, and `section_force` on a line-stations
record. Each one emits the family's whole response. The three gauss
shorthands read back the same values as their explicit components on stock
pip and on the fork. `section_force` does the same on the `.out` route; the
MPCO and fork routes already refused it by name.
`ops.recorder.declare(...)` and `DomainCaptureSpec` already expanded
shorthands, so they do not change.
