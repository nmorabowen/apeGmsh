### ADDED — a probe workflow asks whether apeGmshViewer's headless capture writes a PNG on Linux (#2649, VA0p)

`.github/workflows/viewer-capture-probe.yml` runs `npm run capture` under
`xvfb-run` on ubuntu three ways: plain, with `ELECTRON_DISABLE_SANDBOX=1`, and
with the setuid fix on `chrome-sandbox`. It records the exit code, the PNG size
and whether the PNG is non-blank in the step summary. It runs on a manual
dispatch and on a pull request that edits the file, always ends green, and is
not a required check.

Result: no PNG on ubuntu under xvfb. WebGL2 is blocklisted under software
rendering, and the switches that might unblock it (`--use-angle=swiftshader`,
`--enable-unsafe-swiftshader`, `--ignore-gpu-blocklist`) can only be set from
app code, which is VA2's job.
