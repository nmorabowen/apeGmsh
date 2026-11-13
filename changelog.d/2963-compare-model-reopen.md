### CHANGED — apeGmshViewer: a re-open of the same compare set answers `compare:model` without reading its model again (VF3-e, #2963)

Measured on a 1,000 x 1,000 FourNodeQuad compare set (1,002,001 nodes, 1,004,004 elements, a
382 MB model file), one `compare:model` read cost 1.5-1.8 s and about 620 MB of peak RSS, and the page
asks for it on every compare open. Main now holds the last answer's node and element counts, never the
model, keyed by the file's path, size and mtime and by the `fem_hash` and `model_hash` read from it on
each call. Those hashes are checked against the pairing before the held answer is used. A re-open of
the same set (`compare:close`, then `variants:compare` on the same row) now takes about 1 ms. A file
touched since the read is read again. A file rewritten during the read is refused. Pairing another row
drops the held answer.
