# ADR 0118 — Station-aligned DRM soil box: one builder for the DRM box and the absorbing box

**Status:** Proposed (2026-10-10)

**Owner:** nmora

**Builds on** [ADR 0066](0066-h5drm-drm-authoring.md) (H5DRM pattern and the
complete-grid box) and [ADR 0054](0054-asd-absorbing-boundary.md) (the ASD skin).
**Consumer:** the San Ramón Tier 3/4 port (`Epistemic Uncertanty`,
`models/apegmsh/recipes/tiers234.md` §3.3-3.4, §4 gaps G3 + G4, author
decisions A3-A7 of 2026-10-10).

## Context

`add_DRM_box_from_h5drm` (ADR 0066) accepts only a complete, isotropic grid in
the identity frame (metres, z down). A ShakerMaker DRMBox file is not that: it
holds two station shells (the San Ramón file: 8 178 stations of a 45 × 45 × 14
lattice at 2.5 m, 4 313 on the outer shell with `internal = 0`, 3 865 on the
inner shell with `internal = 1`), in km, with the deck's `pattern H5DRM` moving
it to millimetres, swapping x/y and flipping z (`T = ((0,1,0),(1,0,0),(0,0,-1))`,
`x0 = (22000, 15500, 0)`, `crd_scale = 1e6`).

The fork (`H5DRMLoadPattern.cpp`, `node_matching_BruteForce`) visits every domain
node with ndf ≤ 6, takes its nearest station, and matches it when
`d < distance_tolerance`. An element is a DRM element when all its nodes are
matched. The as-run Tier-4 decks sit on a 5 m grid and match with a 2 500 mm
tolerance: 4 126 nodes of 8 178 stations, 2 789 of them off the stations by up
to 2.4 m. The author decided (A5) that the port's DRM layer lies exactly on the
station grid with a ~1 mm tolerance, that one soil mesh serves all Cases of a
Tier (A4), that the near-field block stays at least one cell inside the DRM
layer (A6), that it is coupled to the rest of the soil by
`ASDEmbeddedNodeElement` as in the decks (A7), and that the same builder makes
the realigned Tier-3 box (A3): the Tier-4 interior wrapped by an absorbing skin.

## Decision

1. **`SoilLattice`** is the interior mesh rule both configurations share:
   ascending x / y / z lines in the model frame and the free-surface end of z.
   `SoilLattice.from_h5drm(path, crd_scale=, transform=, x0=,
   distance_tolerance=)` applies the fork's transform
   `T · ((xyz − drmbox_x0) · crd_scale) + x0` to the stations, reads the lattice
   off the transformed coordinates, and keeps the inner shell and everything
   inside it as the interior and the outer shell as the DRM layer's outer face.
   `SoilLattice.regular(lo=, hi=, spacing=)` builds one without a file.
2. **Accepted grids:** the two-shell layout of any lattice, with any spacing per
   axis, uniform or not (a non-uniform axis becomes uniform runs). **Refused,
   with counts:** a complete grid (pointed to `add_DRM_box_from_h5drm`), missing,
   extra or duplicated stations, `internal` flags that contradict their shell,
   a `T` that is not a signed permutation or moves the depth axis off model z,
   a tolerance outside (0, half the smallest gap). Default tolerance: 4e-4 of the
   smallest station gap (1 mm on 2.5 m in mm; 1e-3 in m).
3. **`g.parts.add_station_soil_box(lattice, *, drm, exterior, boundary,
   skin_thickness, nearfield, pit, name)`** builds one sliced box (structured,
   transfinite per sub-volume) whose axes are the interior runs, then, outward,
   the one-cell DRM layer (`drm=True`), the `Exterior(thickness, size)` zone, and
   for `boundary="absorbing"` a one-cell ASD skin; `boundary="fixed"` returns a
   dim-2 PG for `ops.fix`. `drm=True` needs an exterior. The tangential spacing
   of the exterior follows the lattice (conforming, no transition elements).
4. **The near-field block** (`NearField(lo, hi, size, lines, couple, stiffness)`)
   fills a hole of the lattice whose faces are lattice lines, reaches the free
   surface, and stays at least one lattice cell inside the interior. It has its
   own spacing (2.5 m × 2.5 m × 1 m for San Ramón) and lines through the pit
   planes. Its interface nodes are embedded in the lattice hole faces
   (`g.constraints.embedded`, one `ASDEmbeddedNodeElement` each). The finer
   depth spacing lives only there: lattice nodes on the inner shell must be
   stations, so the lattice cannot be refined in depth.
5. **The pit** (`Pit(lo, hi)`) is a void reaching the free surface, cut from the
   near-field block, or from the lattice when its planes are lattice lines.
6. **Physical groups by name** (prefix `name`): volumes `domain`, `interior`
   (lattice + near field, the same in both configurations), `lattice`,
   `nearfield`, `drm`, `exterior`, the skin PGs of ADR 0054; surfaces
   `free_surface`, `boundary`, `pit_bottom`, `pit_walls`, `pit`,
   `lattice_interface`, `nearfield_interface`.
7. **The result carries the gates:** `frame.pattern_kwargs()` for
   `ops.pattern.H5DRM`, hex counts per PG and the node count computed from the
   axes, and a pre-mesh `station_check` (every DRM-layer node is a station and
   every station a DRM-layer node, maximum distance below the tolerance).
   `h5drm_matches` reproduces the fork's matching rule on a mesh.

## Consequences

- On the San Ramón file the DRM box matches 8 178 of 8 178 stations at a maximum
  distance of 8.7e-7 mm with a 1 mm tolerance; the nearest unmatched node is
  2 500 mm away. The Tier-3 box built from the same lattice has the same
  interior node for node.
- The box is larger than the as-run 4D (72 723 nodes against 55 980): the
  exterior keeps the 2.5 m tangential spacing (27 320 exterior hexes against
  10 422).
- Deferred: a coarser exterior in the tangential direction (needs a
  non-conforming exterior tie or transition elements), stratified soil layers,
  rotated frames, complete grids in this builder, and conformal building–soil
  coupling (A7 study option).
