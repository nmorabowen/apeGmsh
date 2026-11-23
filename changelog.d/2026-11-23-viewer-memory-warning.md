### ADDED — apeGmshViewer warns when memory is short for a model load (ADR 0137 D4 stage 1, #3057)

Before opening a model, and every second while it loads, the viewer reads the machine's memory and
the model's cell count. When available RAM is under 2 GB, or (on Windows) the memory that can still
be committed is under twice the load's estimated peak, the banner says so, with the memory available
and the estimate: "Memory is short: 1.5 GB of RAM is available, under 2.0 GB, for a load estimated to
peak at about 2.1 GB (128,923 cells). The app may be slow or stop responding; closing other programs
frees memory." The estimate, 900 MB plus 10 KB per cell for the whole app, comes from the #3056
measurements. It only warns: no open is refused. Render, capture and describe runs never show it.
