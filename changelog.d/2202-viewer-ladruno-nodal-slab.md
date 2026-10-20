### CHANGED — apeGmshViewer: one `.ladruno` nodal step is one hyperslab per `DATA` (#2202)

`LadrunoFile.readNodalStep` sliced `DATA[step, :, col]` once per component,
so one step of three displacement components inflated the same chunk three
times. It now reads the step's columns of each `DATA` in one slice and splits
them in memory (`reader/ladrunoNodal.ts`). The values are unchanged, bit for
bit, for every component at every step of every committed `.ladruno`. On a
4 096-node, 500-step fork run (chunks [2, 4096, 3], shuffle, deflate), one
warm step drops from about 1.7 ms to 0.63 ms, and costs 1.5 to 1.8 times the
native `results.h5` read instead of 4.8 to 5.4 times (`MEASUREMENTS.md`).
