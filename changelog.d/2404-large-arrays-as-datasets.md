### FIXED — a long `Path` time series archives: arrays past the HDF5 attribute limit are datasets (opensees 2.34.0, #2404)

`ops.timeSeries.Path(values=<16,384 floats>)` ran its deck, but the
automatic `model.h5` write failed with `OSError: object header message is
too large`, and the archive was lost behind one `BridgeArtifactWarning`.
Every call's positional arguments were stored as one HDF5 attribute, and
an attribute holds at most 65,535 bytes. That is about 8,180 numbers, or
4,090 slots once any argument is a string, so a `Path` series failed from
about 4,090 values.

- A `params` / `params_str` pair whose payload is past 64,000 bytes is now
  written as two uncompressed datasets of the same names in its group, as
  an element's `args` / `args_str` are. That covers every typed store:
  materials, sections, beam integrations, time series, patterns,
  dampings, regions and recorders, global and staged. A declared
  recorder's `ids` follows the same rule.
- Any other sequence attribute past the limit refuses at write with a
  `ValueError` naming the key and the group, instead of the bare HDF5
  error.
- Every array that fitted keeps the attribute form, so every archive a
  2.33 writer could write keeps its bytes and its `model_hash`. The golden
  corpus changes only its version stamps.
- The reader, the assembly region-damping check and the viewer read
  either form. A key in both forms, or a dataset form in a file stamped
  below 2.34.0, is refused as malformed.
- Opensees schema 2.34.0 (additive): a 2.33 reader would have read a
  dataset-form pair as no arguments, so it now refuses a 2.34 file.
