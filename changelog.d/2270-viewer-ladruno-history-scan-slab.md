### CHANGED — apeGmshViewer: the `.ladruno` contour history scan reads one hyperslab per `DATA` (#2270)

`LadrunoFile.readNodalRows`, the read behind the contour's whole-history
range, sliced `DATA[k0:k1, :, col]` once per component, so a magnitude scan
of three components inflated every chunk of the history three times. It now
reads the range's columns of each `DATA` in one slice and splits them in
memory, through the same helper as one step (#2202). The values are
unchanged, bit for bit, for every component over every range of every
committed `.ladruno`. On a 4 096-node, 500-step fork run (chunks
[2, 4096, 3], shuffle, deflate), the history scan of the displacement
magnitude drops from about 1.0 s to 0.43 s in the app, and its reads from
about 430 ms to 157 ms (`MEASUREMENTS.md`).
