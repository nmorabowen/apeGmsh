### CHANGED — apeGmshViewer: a partitioned `.ladruno` set reads each part once for all components (#2297)

`LadrunoPartSet.readNodalRows`, the read behind a part set's steps and its
contour history scan, called each part once per component. N components of
one `DATA` on P parts were therefore N x P slices of it. It now reads each
part once for every asked component (one hyperslab per `DATA`, #2270) and
stitches the parts' columns in memory, so the count is P. The values are
unchanged, bit for bit, against the per-component stitch for every component
over every range of the committed part set and of 2- and 3-part scratch sets
whose interface copies differ. NONE still keeps the first part's copy, and
SUM still adds the copies in part order.

One refusal moves earlier. Every asked component is now checked before any
part is read, so an unknown component is refused before any read. Before, an
earlier component's part read ran first, and a refusal from that read could
surface instead. The range check still comes first. No timed row exists,
because there is no partitioned measure set (`apeGmshViewer/MEASUREMENTS.md`).
