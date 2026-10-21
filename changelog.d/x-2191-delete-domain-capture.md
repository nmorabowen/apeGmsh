### REMOVED — the in-process `DomainCapture` route (program slice X 7, #2191)

`apeGmsh.results.capture` (`DomainCapture`, `DomainCaptureSpec` and the resolved
records) and `apeSees.domain_capture` are gone: `from apeGmsh.results.capture
import DomainCapture` raises `ImportError`. Every job it did has a route
(parity table on #2191):

- record an in-process run: `spec.emit_default(path, model_h5=...)`, read with
  `run.results()`; several stages in one notebook run: `spec.emit_recorders`;
- modes: `ops.eigen(n)` (`EigenResult.to_native` / `.results()`), and
  `apeGmsh.results.live.capture_live_modes` for a plain openseespy deck;
- fibres, layers and per-material Gauss points: MPCO, the fork's `.ladruno`,
  and pip `.out` fibres.

Files it wrote stay readable with `Results.from_native`: the
`source_type="domain_capture"` value is kept until the native format retires.
The two catalog helpers only it called, `has_dead_gauss_probe` and
`needs_per_material_query`, are removed with it; the bridge emit still warns on
a stale upstream-shell probe (`StaleShellProbeWarning`).

The `ssi_frame_wall` viewer bench now has two result sources (maintainer
ruling (A), 2026-10-21): the recorder set (`gravity` + `dynamic`, through
`emit_recorders`, transcoded on open) and `modes.h5` (`ops.eigen(6).to_native`),
both opened by `build.open_results`, which `check_slots.py` and
`open_viewer.py` use.
