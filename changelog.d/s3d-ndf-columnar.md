### CHANGED — per-node ndf inference reads the columnar fan-out (program slice S3-d, #1922)

`infer_node_ndf` now takes node ids straight from each element spec's `PGElementFanout` connectivity
arrays, with one `np.unique` per block. It no longer builds one Python tuple per element. Those
tuples, and the flatten over them, set the process emit peak at about 580 B per global node (#1920).
A class whose fan-out mixes node counts is split by width, and a strict per-slot class (ADR 0074)
maps its slot floors column by column. The inferred map, every refusal message and the emitted
decks are unchanged.
