### CHANGED — apeGmshViewer: load arrows thin out evenly along curves

Where load arrows crowd on screen, the viewer used to keep one arrow per grid cell, which left runs and holes where
a curved member crossed the cell corners. It now keeps an arrow only when no larger load is closer than the
thinning distance, so a curve thins evenly: the arrows kept are at least that far apart, and every arrow left out has
a larger one within it. On a 100 000-node load case the arrows take about 0.4 s to build instead of 0.2 s; on
models of a few thousand loads the difference is a few milliseconds, and zooming still costs nothing.
