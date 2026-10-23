### FIXED — apeGmshViewer reads `.ladruno` line stations past 2^24 elements (#2348)

The viewer's `.ladruno` line-station read (`readLineStations`, and the step
line diagram that uses it) kept three `Set`s sized by the element count: the
elements asked for, the elements a section bucket serves, and the elements an
earlier bucket had already given. A `Set` holds at most 2^24 entries in V8, so
a file with more beam elements failed with an unnamed `RangeError`. The read
now uses sorted typed arrays for these lookups, as the frame lookup does since
#2336. Its output is unchanged, bit for bit, on every committed `.ladruno`.

A `MODEL/LOCAL_AXES` group that the reader does not read is reported once,
among the warnings when the file opens (#2336). The app's step reads do not
repeat the warning at each step. A test now checks this through the app's own
open answer.
