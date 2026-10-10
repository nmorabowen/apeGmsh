### ADDED — `g.parts.add_station_soil_box`: a DRM layer on the `.h5drm` stations (ADR 0118)

`SoilLattice.from_h5drm(path, crd_scale=, transform=, x0=)` reads a two-shell
ShakerMaker station grid through the frame the fork's `pattern H5DRM` applies
(any units, z up or down, signed-permutation `T`, offset `x0`; non-uniform
spacing per axis accepted; complete grids, missing stations, wrong `internal`
flags and rotations refused with counts). `g.parts.add_station_soil_box`
builds from it either a DRM box (interior + a DRM layer whose nodes are exactly
the stations + `Exterior` + fixed or absorbing boundary) or the same interior
wrapped by an ASD absorbing skin, with an optional `NearField` block tied by
embedded nodes and a `Pit` void. The result carries the PGs, the
`ops.pattern.H5DRM` keyword arguments, expected counts and
`station_check(fem)`, which checks the meshed DRM layer against the stations.
`boundary="absorbing"` needs a z-up lattice.
