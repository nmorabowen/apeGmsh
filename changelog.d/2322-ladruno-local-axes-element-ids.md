### FIXED — a `.ladruno` whose `MODEL/LOCAL_AXES` gives an element two frames, or holds frames under an empty `ID`, is refused when it opens, by name (#2322)

`MODEL/LOCAL_AXES` keeps one `<classTag>-<Class>/{ID, FRAME}` group per
element class. A repeated element id, within one class's `ID` or across two
classes, was resolved by the last write: the element took another row's
frame and the repeated row's element lost its own. A group whose `ID` was
empty while its `FRAME` held frames was skipped without a word.

Opening such a file now raises `LadrunoRepeatedElementId`, naming the file,
the `ID` dataset or datasets and each repeated id with its places, or
`LadrunoEmptyLocalAxesId`, naming the file, the class group and the `FRAME`
shape. `Results.from_ladruno` refuses a single file and a part set alike. An
empty group (no id, no frame) still opens, and any other `FRAME`/`ID` count
mismatch keeps its layered-shell warning (#2090). The viewer's twins are
#2321 and #2336.
