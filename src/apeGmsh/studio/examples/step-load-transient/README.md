# step-load-transient

A small generic solid cantilever (density on the material, a Constant-
series step load, Newmark transient integration) plus its static twin
built off the same mesh — enough to compute a dynamic amplification
factor and check the Ladruno recorder's energy-balance channel.

Teaches:

- Density lives on the material (`ops.nDMaterial.ElasticIsotropic(rho=...)`),
  not the element.
- A step load (`ops.timeSeries.Constant()` applied at `t=0`) excites a
  real transient response, unlike a ramped `Linear` series.
- `ops.recorder.Ladruno(..., energy=True)` — the `-G energy` channel,
  read back via `Results.from_ladruno(...).energy()` — the headline
  solution-quality diagnostic (`KE` / `IE` / `DW` / `ULW` / `RES` /
  `ERR`) for a transient run.

The geometry here is a small generic box (not the specimen this
pattern was ported from — see `manifest.json`'s `provenance`).

Requires: `ladruno` (the classic OpenSees exe; both decks are emitted
Tcl and run as subprocesses — no in-process openseespy).

Run:

```
python step_load_transient.py
python verify.py
```

Writes the static + transient Tcl decks, their Node-recorder text
outputs, and the transient's `.ladruno` energy file to the current
directory. See `manifest.json` for the oracle metrics `verify.py`
checks (DAF, last-step energy error, KE+IE vs ULW closure).
