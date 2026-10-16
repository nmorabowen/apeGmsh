### FIXED — a `basicForce`-only `.ladruno` file reads 3-D end moments as stations in the section convention (#2011)

A 3-D beam's `basicForce` bucket is `N,Mz_1,Mz_2,My_1,My_2,T`. With no
`section.force` or `localForce` recorded, `Results.from_ladruno(...)` read
`Mz_1,Mz_2` and `My_1,My_2` as `bending_moment_z` / `bending_moment_y` line
stations with the basic-system signs. Station 1 then had the wrong sign
against the `localForce` reading of the same element and against statics: a
cantilever root read `-8600` where the section moment is `+8600`.

Every fork class that names these columns reports its `localForce` end
moments as the same values, member loads included. The pair now takes the
`localForce` station signs, the one table every route uses since #2009, so
the stations read `[-M_1, +M_2]`. They equal the `localForce` stations of the
same element. `axial_force` and `torsion` remain single stations equal to the
live `basicForce` values. Any other two-station `basicForce` component lists
and reads no stations, and both calls warn with the bucket and the component.

The fork lane checks an `elasticBeamColumn` and a `forceBeamColumn` 3-D
cantilever against `eleResponse` and statics at rtol 1e-10.
