### FIXED — apeGmshViewer hangs on a model with many inline zeroLength springs (#3130)

A model of about 20 000 inline `zeroLength` springs (a pile-supported mat on
soil springs) now opens and draws; before, the window never showed the mesh.
The frame extrusion found each OpenSees-only line's declaration by scanning every
declaration, so the load was quadratic in the number of springs; it now looks the
declaration up by its key, which also joins the line to its own element type's
row rather than the first type with the same row number. The model's answer also
reaches the page as one structured clone instead of `contextBridge`'s
object-by-object copy, which did not finish for this model's 170 MB of
declarations.
