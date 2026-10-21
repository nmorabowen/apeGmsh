### FIXED — Gauss reads that mix 2-D and 3-D classes refuse by name in every reader; an element read twice refuses (program slice B, #2221)

A Gauss read joining a 2-D-parent class (a quad or shell) and a 3-D-parent class (a tet or brick) now
refuses in all four readers, naming one source of each width and the file. A read narrowed to one class
(`element_ids`, or a physical group holding one class) answers as before. The MPCO single-file read already
raised `MPCOGaussDimensionMixed`. The partitioned MPCO read now raises it too, where it used to pad the quad
rows with NaN. The `.ladruno` read used to return the quad rows entirely NaN, and the native read failed
with a bare numpy `ValueError`; both now raise `GaussDimensionMixed`. The partitioned MPCO and `.ladruno`
stitches also refuse an element id that comes back from two part files, and the `.ladruno` element read
refuses an id in two buckets of one token, with `MPCOElementInTwoBuckets`. Before, that element was read
twice. A boundary node shared by two parts still reads once.
