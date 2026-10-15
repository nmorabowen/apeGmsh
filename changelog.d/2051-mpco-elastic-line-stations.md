### FIXED — an MPCO line-stations record reads elastic beams, or warns that it cannot (#2051)

A line-stations record on `spec.emit_mpco`, `to_mpco_tcl_command` or
`to_mpco_python_command` emitted `-E section.force` only. Elastic beams
(`ElasticBeam2d/3d`, `ElasticTimoshenkoBeam2d/3d`, `ModElasticBeam2d`) do not
answer `section.force`, so `Results.from_mpco(...)` read their stations empty,
with no warning. A record whose `element_class_name` is one of these classes
now also records `localForce`. The MPCO reader builds each elastic beam's two
end stations from it, and keeps `section.force` for the elements that answer
it, so one record over elastic and force beams reads both. A record that names
no class, or a class the response catalog does not list as a section-force
beam, still records `section.force` only, and now raises a `UserWarning`
naming the record and the elastic classes that would read empty.

On the fork lane each elastic class, and a mixed `ElasticBeam3d` +
`forceBeamColumn` record, read back their stations at `rtol=1e-10` against the
cantilever closed form.
