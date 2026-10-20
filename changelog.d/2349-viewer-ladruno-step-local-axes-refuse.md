### FIXED — apeGmshViewer refuses a malformed per-step `.ladruno` `LOCAL_AXES` bucket by name (#2349)

This hardens the reader for the per-step `RESULTS/ON_ELEMENTS/LOCAL_AXES` stream,
which schema §5 specifies but the fork at `FORK_PIN` does not write yet. The
viewer's `.ladruno` reader used to skip a per-step bucket it could not pair
with its ids or with the stage's steps, and said nothing. A step read
(`readLocalAxes(stage, {step})`, and so `readLineStationStep`) then returned
the `MODEL/LOCAL_AXES` reference frame in place of the stored step frame. For
a co-rotational element that is the wrong frame.

Such a bucket is now refused by name as a `LadrunoError` of kind `layout`. The
message names the file, the bucket and the step. The refusal covers:

- an `ID` or a `DATA` that is missing or is not a dataset;
- a `DATA` that is not shaped `(steps, elements, components)`;
- a step count that is not the stage's;
- a row count that is not the `ID`'s;
- rows that are not 3, 4 or 9 wide;
- an empty `ID` beside rows;
- a row that is not a frame, such as a value that is not finite or an all-zero
  quaternion (HDF5's fill value for an unwritten chunk). This message also
  names the element and its row.

A `step` that is not an integer inside the stage is now refused by
`readLocalAxes` as a `LadrunoError` of kind `argument`. Before, a step out of
range quietly read the reference frames, and a fractional step threw a raw
`TypeError`.

Unlike `MODEL/LOCAL_AXES` (#2336), nothing falls back with a warning, because
at a step the stored frame is the answer. A step-less read does not touch
these buckets. A well-formed bucket reads exactly as before, and no committed
`.ladruno` holds one.
