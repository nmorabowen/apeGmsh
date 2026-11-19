### FIXED — `npm run profile` reads frame times from when the frame callback ran

The profiler in `apeGmshViewer/` took frame intervals from the
`requestAnimationFrame` timestamp. That timestamp stays on the display's
refresh grid while the page thread is late, so the reported fps and p95 frame
times could read better than the truth. The probe now uses the time the
callback actually ran. Scenario times, steps drawn, flicker counts and CPU
profiles never used the timestamp and are unchanged. `apeGmshViewer/MEASUREMENTS.md`
notes which earlier numbers this affects.
