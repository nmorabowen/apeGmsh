### FIXED — MPCO line stations warn when one beam's components read different station sets, and accept the element command as the class (#2070)

A `forceBeamColumn` whose sections answer no shear (an `Elastic` section
without `-alpha`, say) writes a `section.force` bucket of `P, Mz, My, T`
only. Its axial force, torsion and moments read at every section, but its
`shear_y` and `shear_z` read at the two `localForce` end stations. Each
value is right, and nothing said that the station sets differed. The MPCO
reader now warns `MPCOLineStationSourcesMixed` on such a read, once per
read and by element class, naming the component that changed source, its
station count, and the components the `section.force` bucket records with
theirs. The read still answers.

An MPCO line-stations record whose `element_class_name` is an OpenSees
command (`elasticBeamColumn`, `ElasticTimoshenkoBeam`, `forceBeamColumn`,
`dispBeamColumn`, `LadrunoForceBeamColumn`, `LadrunoDispBeamColumn`) now
stands for the classes that command builds. Before, each warned that the
response catalog did not list it, and `elasticBeamColumn` recorded
`section.force` alone, so its beams read back empty. `MixedBeamColumn3d`
and `TimoshenkoBeamColumn3d`, which the catalog does not list, keep that
warning. The emit's token-routing tests now name a class, so they no
longer raise 19 "names no element class" warnings; the warning stays for
records that name none.
