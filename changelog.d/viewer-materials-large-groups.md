### FIXED — apeGmshViewer's Materials window freezes on a model with ~20k spring materials (#3133)

The Materials & sections window now opens at once on a model with one material per
soil spring (19 512 on the YOO Guayaquil mat). A group of more than 100 rows shows a
summary by type (count and the range of each numeric parameter), a filter by name, HDF5 name,
tag, declaration key or type, and its rows 100 at a time ("show 100 more"). The
lookups of `/opensees/decls/tags` are indexed once per file, which also speeds the
reader's tag check and the declaration keys of every such model; the list is kept
per read, so a selection no longer rebuilds it.
