### CHANGED — apeGmshViewer load memory re-measured after the load fixes

`apeGmshViewer/MEASUREMENTS.md` has a new section, "Load memory after the load
fixes". With #3047, #3048 and #3049 on `main`, five models (24k to 301k cells)
loaded with the page at 1.3 to 10 KB per cell. ADR 0137 took 42 KB per cell
from `yoo_building_SLE_T2`, measured before those fixes. The section proposes
a warning estimate (page about 250 MB + 10 KB per cell) and records that on
Windows, Electron's `getSystemMemoryInfo().swapFree` is the commit headroom.
The building itself could not be re-measured on the shared machine.
