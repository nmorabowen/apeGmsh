### FIXED — `.ladruno` and MPCO reactions read `REACTION_FORCE` by precedence, never the Rayleigh damping force (#2122)

`REACTION_FORCE`, `REACTION_FORCE_INCLUDING_INERTIA` and `RAYLEIGH_FORCE`
all translated to `reaction_force_*`, and both readers took the first
`ON_NODES` group in HDF5 iteration order. A file written without
`track_order` iterates by name, so `RAYLEIGH_FORCE` came first and a
reactions read returned the Rayleigh damping force with no warning. The
fork's `.ladruno` recorder keeps the `-N` token order, so recording
`rayleighForce` before `reactionForce` did the same.

The group now comes from an explicit precedence: the static flavor, then
the inertia-including one, for `reaction_force`, `reaction_moment`, `force`
and `moment` alike. A Rayleigh group (`RAYLEIGH_FORCE`, `RAYLEIGH_MOMENT`)
holds the damping force alone, not a reaction, so it no longer translates
to any canonical name. A `reaction_*` read from a stage that records only
Rayleigh groups raises `NodeComponentAmbiguous` and names the group, and so
does a read whose leading group lacks the requested column while a lower
group holds it. Partitioned `.ladruno` and MPCO reads follow the same rule.
