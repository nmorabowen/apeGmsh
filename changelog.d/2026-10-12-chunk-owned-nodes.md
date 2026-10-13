### CHANGED — partitioned emit walks each rank's owned nodes in slices (#1758)

The owned-node pass of partitioned emit (#1670) resolved and converted a
whole rank's nodes at once, which held about 240 bytes per node while it
ran. It now walks the sorted ids in slices of 16,384, so the pass holds
the sorted ids plus one slice. On a 1M-node box over 4 ranks the pass
peaks at 8.1 MiB per rank, against 58.8 MiB after #1670 and 9.8 MiB
before it. Emitted decks are byte-identical.
