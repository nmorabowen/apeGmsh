### FIXED — MPCO reads buckets with a header index above 0 (#2033, #2096, #2093)

MPCO opens a new response header (`<tag>-<Class>[<rule>:<cust>:1]`, `:2`, ...)
when one element class's elements write different META: ASDConcrete3D
(`d+,d-`) and LadrunoConcrete3D (`C1,C2`) on one LadrunoBrick class, a
plane-stress and a plane-strain material on one quad class, fiber or layered
sections of different sizes on one beam or shell class. The material-state,
Gauss, fiber and layer readers kept header 0 only, so
`Results.from_mpco(...).elements.gauss.get(component="damage_tension")`
returned the header-0 elements alone and the rest vanished without a warning.
Every header is now read, each resolved from its own META, `NUM_COLUMNS` and
section assignments. A header-1 layer bucket, which warned
`LayerBucketUnreadWarning` since #2091, is now read. A header-1 nodal-force
bucket is still named in `NodalForceBucketSkippedWarning`.
Gauss columns are now named by each bucket's own META labels (`sigma33` is
`stress_zz` wherever it sits); MPCO's unnamed `C1..Cn` keep the catalog
order, and a label that names none of the catalog's components refuses the
read with a `ValueError` naming the bucket, where it used to be read under
the catalog's names.
