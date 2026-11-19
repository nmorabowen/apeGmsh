### FIXED — apeGmshViewer refuses a dataset HDF5 cannot read instead of showing garbage (#3010)

h5wasm ignores the status of its dataset reads, so a damaged chunk index could
reach the screen as ordinary numbers. Every file the viewer opens now has its
reads checked, and a refused read stops with an error naming the dataset.
