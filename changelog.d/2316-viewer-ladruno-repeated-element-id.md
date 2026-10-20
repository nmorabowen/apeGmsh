### FIXED — apeGmshViewer refuses a `.ladruno` that lists one element twice in a result, or a part set that holds one element in two parts, by name (#2316)

The viewer's `.ladruno` reader used to take an element bucket's `ID` as
written. An element id listed twice in one bucket, or in two buckets of one
result, read as two rows of one element, and the element whose id was
overwritten disappeared, with no warning.

Opening a `.ladruno` now raises a `LadrunoError` of kind `layout` when a
result under a stage's `RESULTS/ON_ELEMENTS` lists an element id twice, in
one bucket or across its buckets. Every bucket is checked, Gauss or not, so
line-station, fiber and `LOCAL_AXES` buckets are covered too. The error
names the file, the result, and each repeated id with the `ID` dataset and
row it sits at. Two results that hold one element, such as `stress` and
`strain`, are the norm and still open.

A part set is refused at open when the Gauss buckets of one result hold an
element in two part files, naming both parts: each element lives on one
rank. A node held by two parts is still legal. The Python reader refuses
the same repeats (#2310).

The fork is not known to write a repeated element id; this guards against a
malformed or merged file. Every committed `.ladruno` fixture still opens.
