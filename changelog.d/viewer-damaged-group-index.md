### FIXED — apeGmshViewer refuses a damaged HDF5 group index instead of listing it short

h5wasm ignores the status of its group listing, so a group whose index was
damaged came back with only some of its children and no error. A stage could
open without its nodal results or with fewer components. Every file the viewer
opens now has its listings checked: a listing HDF5 could not finish stops with
an error naming the group, and a `.ladruno` stage that cannot be listed follows
the reader's existing rules for a stage that does not read.
