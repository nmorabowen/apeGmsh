### FIXED — `.ladruno` has one spatial-dimension rule, and it refuses an invalid `SPATIAL_DIM` (program slice B, #2098)

`LadrunoReader` and the `localForce` 2-D gate each had their own copy of the rule that reads a `.ladruno` file's
spatial dimension: `INFO/SPATIAL_DIM`, else the one coordinate width of 1, 2 or 3 that every stage agrees on, else
unknown. The reader now calls the gate's helper, so the two cannot drift apart. An `INFO/SPATIAL_DIM` that is not an
integral 1, 2 or 3 now raises `SpatialDimInvalid`, a `SpatialDimUnknown` that names the file and the value. Before
this change, 7, 0 and -1 passed through unchanged, so `opensees_model()` built `ndm=ndf=7` without a warning, and 2.9
was truncated to 2. `2.0` still reads as 2, and inference when the attribute is absent is unchanged. A table test in
`tests/results/readers/test_ladruno_reader.py` pins the answer for every input class.
