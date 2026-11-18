### CHANGED — apeGmshViewer names a compare user result by variant and entry; the palette and the legend offer one range rule (#2956)

A compare variant's user result was named in a manifest only by the display
string the legend shows, `<variant>: <entry>`, which can equal a window
`.user.h5` entry of that literal name. Its manifest contour now also states
`user.compare: { variant, entry }`, and a window user result states no
`compare`, so a reader tells the two apart. A `--render` spec can now draw a
compare user result: a shot with `contour.source: "user"` and `compare:
<variant>` names that variant's own entry (`contour.field`) and column
(`contour.component`). The shot opens the compare set, waits for its
`.user.h5`, and draws the column. An unknown variant is refused with the
variants the manifest has, and an unknown entry or component with what the
compare file offers. Every shot now starts with no compare user result drawn.

The legend's scale and the command palette now read one colour-range rule
(`rangeOptionsOf`, through `state/selectors.ts`), so the legend no longer
keeps its own copy. The palette now offers "fixed" with no stored range as
well. Picking it starts from the range shown, as the legend's button does.
