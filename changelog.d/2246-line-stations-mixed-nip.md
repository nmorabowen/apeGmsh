### FIXED — `.out` line stations refuse a record whose elements differ in integration-point count (#2246)

`Results.from_recorders` sliced a line-stations record into equal per-element
blocks, because the `.out` files carry no per-element count. A record over a
Lobatto 3 and a Lobatto 5 force beam gave 4 columns each and read without
error: one element took the next one's first station. The transcoder now
reads each element's count from the model's `beamIntegration` and refuses a
record whose elements differ, naming them and their counts; split such a
record into one per count. Each element's stations must also increase strictly
within [-1, 1], the only check a transcode without a model can make. A record
whose elements share one count reads as before.
