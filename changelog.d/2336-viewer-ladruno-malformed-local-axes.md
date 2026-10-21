### FIXED — apeGmshViewer says when a `.ladruno` `MODEL/LOCAL_AXES` group is not read, and reads past 2^24 framed elements (#2336)

The viewer's `.ladruno` reader skipped a `MODEL/LOCAL_AXES/<class>` group whose
`FRAME` could not be paired with its `ID`, with no warning. Its elements kept
no recorded frame, so a beam diagram computed theirs from the model (the
`geomTransf` vecxz, or the default), and nothing said the file's frames had
been set aside.

The reader still falls back the same way, as the Python reader and MPCO do
(#2090), but it now says so. A `FRAME` with more or fewer rows than its `ID`,
or with rows that are not 3, 4 or 9 wide, warns when the file opens, among the
open warnings of a stage that has line stations. The check reads only the
datasets' shapes. The same warning is also given to the `warningsOut` of
`readLocalAxes` and the line-station reads. It names the file, the dataset
and both counts. An empty `ID` beside `FRAME` rows is refused by name when
the frames are read, as a `LadrunoError` of kind `layout`, because those
frames belong to no element.

`readLocalAxes` and the line-station reads now resolve the frames with sorted
typed arrays instead of a `Map` per element. A `Map` or `Set` holds at most
2^24 entries in V8, so a file with more framed elements failed with an
unnamed `RangeError`. The frames these reads return are unchanged, bit for
bit, on every committed `.ladruno`.
