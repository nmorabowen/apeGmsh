### ADDED — `recorder.Ladruno(precision="f64" | "f32")` (program slice X, #2613)

`ops.recorder.Ladruno(...)` and the `recorder.Ladruno` primitive take a
`precision` field. The default `"f64"` changes nothing: no flag is emitted
and a default deck is byte-identical to one written before the field
existed. `precision="f32"` emits `-precision f32` right after the file name
in the Tcl and the Python deck, and the fork then stores the per-step result
data as float32, about half the bytes at ~7 significant digits (the file
stamps the choice in `INFO/STORED_PRECISION`). Any other value raises
`ValueError` at construction; the fork would otherwise warn and silently
keep f64. The value is archived in `ops.h5` with the recorder's other fields
(the `/opensees/decl_params` row, and the recorder's `params` argv), so a
reader knows the recorded precision without opening the `.ladruno`.
