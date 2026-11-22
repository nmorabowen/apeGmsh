### FIXED — apeGmshViewer's still capture waits for a long history scan while it advances

`npm run capture` stopped on the real Case 1D building (13,459 nodes, 101 recorded steps in 8 parts) with "the history scan did not finish": the contour stills waited a fixed 30 s for the scan of the whole history, and the scan was still advancing (16 of 101 steps). The harness now waits while the scan advances (`untilSteady` in renderer/harness/context.ts). It fails if the scan stalls for 30 s, and stops at 10 minutes as a runaway guard. A fixture's capture is unchanged.
