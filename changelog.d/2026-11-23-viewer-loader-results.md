### CHANGED — apeGmshViewer reads results in the loader process (ADR 0137 D2/D3, #3058)

Opening a results file, its history scans, step reads, hinge histories, the particles and a compare
set's results now run in the loader process that already opens models, so a long scan or a large
index no longer holds up the windows. Both results hosts, the window set's and the compare set's,
live there with their open file, index and running scans. A step read costs about 3 ms more on a
1.7 MB step (`tims_c3b`), which ADR 0137 D3 asked to measure before moving step reads. If the loader
stops, the reads in flight are refused with why, and the files it had open are opened again before
the next read.
