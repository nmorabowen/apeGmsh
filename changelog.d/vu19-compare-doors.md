### apeGmshViewer: the compare set's model and `.user.h5` through its own doors (VU19)

A variant opened as the compare set now has doors of its own for its model and its
`.user.h5` (`compare:model`, `compare:userRead`, `compare:userReadResult`; ADR 0132 D9.5).
They read only the set main holds in its compare slot, take no path and no row, and refuse
once the slot is closed, switched onto or dropped by a page load. The Variants panel lists
the compare variant's own user results, each judged by A6 against that variant's own model
and results, and draws a state-free nodal one on the window's mesh by node id when the two
models are one mesh (the same `fem_hash`); two meshes that do not pair are refused with the
reason, never interpolated. The window set's user doors are unchanged and still refuse the
compare set's file, and no compare door writes anything.
