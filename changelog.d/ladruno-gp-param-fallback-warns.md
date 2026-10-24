### FIXED - `.ladruno` line stations name the evenly spaced GP_PARAM fallback (#2474)

A section-level line station (`section.force` / `section.deformation`) takes its natural
coordinate from the class's `QUADRATURE/GP_PARAM` row keyed by its `GAUSS_ID`. When the file
carried no `GP_PARAM` for the class, or a `GAUSS_ID` fell outside it, the `.ladruno` reader
silently placed those stations evenly over `[-1, +1]`, which is not where a force beam integrates.
It now warns `LineStationGpParamFallbackWarning`, naming the component, the bucket, its element
class, its element count, and the unmapped `GAUSS_ID`s. The values and the evenly spaced
placement are unchanged; a file whose stations all map stays silent.
