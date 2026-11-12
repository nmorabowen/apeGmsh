### CHANGED — apeGmshViewer: `main.ts` reads its command line from `main/args.ts`, and the measurement harness moves to `main/modes/measure.ts` (VH2, #2866)

The main process parses its command line once, in `main/args.ts` (`mode`,
`file`, `out` and the rest), so the later splits of `main.ts` read the mode
without importing it. The read-timing and results-measure code (the `measure:*`
handlers, the `.ladruno` read row and `--mode=measure-results`) moves to
`main/modes/measure.ts`, which takes what it needs of `main.ts` through one
`registerMeasure` call. Nothing changes for a user: the IPC channels, the exit
codes and the printed rows are the same.
