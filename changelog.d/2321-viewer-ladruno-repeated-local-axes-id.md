### FIXED — apeGmshViewer refuses a `.ladruno` whose `MODEL/LOCAL_AXES` lists one element twice, by name (#2321)

The viewer's `.ladruno` reader took the reference local frames under a
stage's `MODEL/LOCAL_AXES` as written. An element id listed twice in one
class's `ID`, or in two classes' groups, kept the last row read: the element
took another row's frame, and the element whose row it was had no frame,
with no warning.

Reading the local axes of such a stage (`readLocalAxes`, and every
line-station read, which attaches them) now raises a `LadrunoError` of kind
`layout`. The error names the file, `MODEL/LOCAL_AXES`, and each repeated id
with the `ID` dataset and row it sits at. The file still opens, and its other
stages read: open does not read `MODEL/LOCAL_AXES`. The fork writes each
element once, so only a malformed file is refused.
