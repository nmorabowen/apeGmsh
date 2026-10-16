### FIXED — line stations take one ranked source per element, and `globalForce` is never one, on `.ladruno` and MPCO (#2109)

A `.ladruno` line-station read fed every element-level token into the
stations. A file that recorded only `globalForce` read the global end
moments as `bending_moment_*`: a 3-D cantilever root read `-8600` where the
section moment is `+8600`. A beam that recorded `basicForce` and
`globalForce` got four stations, two of them with the wrong sign.

Each element now reads one source, the first it records in this order:
`section.force`, then `localForce`, then the `basicForce` end moments (a
truss's `axialForce` ranks with `basicForce`). `globalForce` and every other
element-level token are never a source. When such a token is the only one
that names a requested element's component, the read returns no stations
for it and warns, naming the token, the bucket and the elements
(`LineStationBucketSkippedWarning` on `.ladruno`, `MPCOBucketSkippedWarning`
on MPCO).

MPCO follows the same rule. It now reads `basicForce` end moments
(`Mz_1,Mz_2`, `My_1,My_2`) as stations under the `localForce` signs, as
`.ladruno` does. Other two-station `basicForce` components read nothing and
warn. MPCO's refusal of a malformed `localForce` bucket no longer blocks
elements that have `section.force` stations, so a read with no element
filter works again.

A `.ladruno` read also names a source bucket whose `COLUMN_MAP` does not
cover its data, rather than dropping it silently.
