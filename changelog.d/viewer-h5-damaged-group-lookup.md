### FIXED — apeGmshViewer refuses a lookup through a damaged HDF5 group index

h5wasm ignores the status of its lookup by name, so with a damaged group index
an existing dataset, and a name that does not exist, both came back as groups
with no error. A lookup that cannot read the index of a group on its path now
stops with an error naming the path. Lookups that fail normally in a sound file
(below a dataset, through a dangling soft link or an external link) answer as
before, and so does an object whose own header does not read.
