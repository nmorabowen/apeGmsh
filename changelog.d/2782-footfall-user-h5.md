### CHANGED — footfall writes `<stem>.user.h5`, not a native results file (#2782, ADR 0131 D6)

`FootfallResult.to_results(fem, path, *, model_h5)` now writes the
evaluation into the model's user-results file as one state-free user result
named `"footfall"` on the nodes (ADR 0130 D7). It has three components:
`footfall_ap` (fraction of g), `footfall_ratio` and `footfall_fdom` (Hz),
over every node in `fem`, `NaN` off the response nodes. Read it back with
`UserFile(path).read_result("footfall")` from `apeGmsh.results.user`.

- **Breaking:** `model_h5=` is required. It is the model archive of the same
  `fem` (`ops.h5(...)` or `fem.to_h5(...)`); an archive of another fem is
  refused. `path` must end in `.user.h5`. No native `results.h5` is written,
  so `Results.from_fem(fem, path, kind="native")` no longer reads a footfall
  map.
- A `fem` with no nodes, or a response node outside `fem`, raises
  `UserWriteRefused` by name, and nothing is written.
- An existing `.user.h5` keeps its other entries; the `"footfall"` entry is
  added or replaced.
- `/meta@writer` records the driver and the OpenSees backend the modal basis
  ran on (the fork build's sha, or `stock`), in place of the native file's
  `opensees_backend` and `opensees_build` attributes.
