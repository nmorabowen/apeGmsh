### FIXED — apeGmshViewer no longer reads one HDF5 file as another (lesson #2260, slice #2261)

h5wasm's Node build gave HDF5 each file's id rounded to a JavaScript double. On NTFS, two
different files can round to the same id once the file id passes 2^53, and HDF5 then hands back
the file that is already open: with a results file held open, opening a model could read the
results file in its place, with no error. Every viewer open now goes through
`apeGmshViewer/src/reader/h5open.ts` (`openH5`), which makes HDF5 compare exact 64-bit file ids and
refuses to open anything when it cannot. The app-wall lint's new rule W5 keeps every h5wasm open in
`src/` and `scripts/` inside that file. Python (h5py) was never affected.
