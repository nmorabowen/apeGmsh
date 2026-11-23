### CHANGED — apeGmshViewer opens models in a loader process (ADR 0137 D2, #3058)

Opening a model, its zones (`/provenance`, `/meta/lineage`, the `/sequence` element tags), a
`/geometry` sibling and a compare set's model now runs in a separate Electron utility process
instead of the main process, so the windows keep responding while a large file is read. The
process starts with the window and comes back on the next open if it stops; an open it had not
finished is refused with why. On `yoo_slab_N56_75` the longest stall of the main process during a
load fell from about 800 ms to about 100 ms, with the load time unchanged. Opening results files,
history scans and step reads still run in the main process; they move in the next slice.
