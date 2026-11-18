### FIXED — apeGmshViewer's capture and measure harness reports what failed (#2669)

A cut-face still names the condition that failed. `measure` accepts a solid on
which some section planes cut nothing, and a history scan that finished is no
longer reported as unfinished. A model that cannot give a row (one-step or
eigen-only results, a refused contour, no plane cutting a cell) is skipped with
the reason printed.
