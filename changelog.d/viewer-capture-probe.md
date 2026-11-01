### ADDED — a probe workflow asks whether apeGmshViewer's headless capture writes a PNG on Linux (#2649, VA0p)

`.github/workflows/viewer-capture-probe.yml` runs `npm run capture` under
`xvfb-run` on ubuntu, plain and with `--no-sandbox` and software GL, and
records the exit code, the PNG size and whether the PNG is non-blank in the
step summary. It runs on a manual dispatch and on a pull request that edits
the file, always ends green, and is not a required check.
