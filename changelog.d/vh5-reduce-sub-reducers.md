### CHANGED — apeGmshViewer: `state/reduce.ts` delegates to one sub-reducer per slice (VH5 #2869)

`apeGmshViewer/src/state/reduce.ts` (1,786 lines) is now 862. The overlay,
line-diagram, windows (open windows, spotlight and search), offers, camera,
user-file and variants events each go to a pure sub-reducer under
`state/reduce/<slice>.ts`, the way the clip events already went to
`reduceClip`: it takes its own part of the state, and what it reads of the
rest as explicit inputs, returns the same object when nothing changed, and
ends in its own `never` over its slice's event union. The cases that read or
write several slices (the files, the selection, the phase, the results, the
contour and the deformed shape) stay in `reduce.ts`, as do the follow-ups a
moved case needs from another slice (the phase after a new user file, the
line-diagram chains after a new scale). `reduce.ts` exports the same 24
names, each with the same type, and every event gives the same next state:
a replay of the test suite's 3,392 recorded transitions, all 109 event tags,
through the old and the new reducer agrees step by step.
