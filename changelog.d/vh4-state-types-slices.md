### CHANGED — apeGmshViewer: `state/types.ts` is per-slice files behind the same import path (VH4 #2867)

`apeGmshViewer/src/state/types.ts` (1,832 lines) is now a barrel of re-exports.
Each slice's types live in `state/types/<slice>.ts` (`model`, `results`,
`user`, `variants`, `contour`, `deform`, `clip`, `lineDiagram`, `overlays`,
`windows`, `offers`, `camera`), `State` in `types/state.ts`, and the `Event`
union in `types/event.ts`, composed from one union per slice in
`types/events/<slice>.ts` (`ClipEvent`, `ContourEvent`, ...). Importers do not
change: the barrel exports the same 143 names, each with the same type or
value kind, and every one is type-identical to before. A change to one slice
now edits that slice's file, and a lock can name it.
