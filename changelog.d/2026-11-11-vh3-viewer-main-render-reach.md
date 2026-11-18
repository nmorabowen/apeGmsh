### CHANGED — apeGmshViewer: the agent render and reach modes move out of `main.ts` (VH3, #2868)

The ADR 0128 render path (`--render`, `--describe` and the capture stills: the
`render:*` and `capture:*` handlers) moves to `main/modes/render.ts`, and
`--present` / `--snapshot` with the offer queue move to `main/modes/reach.ts`.
`main.ts` keeps the window, the session, the file doors and the
`second-instance` handler, which hands an agent's call to the reach module
before it can open, restore or focus anything (INV-5). The modules take what
they need of `main.ts` through one registration call each, and the preload
script's path now comes from `main.ts` to `main/modes/measure.ts` the same way.
Nothing changes for a user: the IPC channels, the exit codes, the stills and
the manifests are the same.
