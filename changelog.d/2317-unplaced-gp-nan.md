### FIXED — an unplaced Gauss point is NaN for every cause, never the origin (#2317)

`GaussSlab.global_coords`, `GaussCensus.global_coords` and
`compute_global_coords_from_arrays` put a Gauss point they could not place at
the origin `(0, 0, 0)` when its element was missing from the FEMData, its
connectivity named an unknown node, or the FEMData had no nodes. The origin is
a plausible coordinate. A FEMData with no nodes returned it with no warning.
Every unplaced row is now NaN in all three columns, as a row without natural
coordinates already was (#2315), and `WarnGaussCoordsApproximate` names the
element ids. A negative node id in the connectivity no longer indexes the
lookup from its end and borrows another node's coordinates: that element is
unplaced and named.

Not fixed here: the Qt viewer's click pick (`_nearest`) does not mask
non-finite rows. When any Gauss target is NaN, a click selects the first
unplaced point instead of the nearest placed one. The viewer is in sunset and
takes crash fixes only. Box-select already excludes NaN rows, and the marker
layers render without error.
