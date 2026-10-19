### CHANGED — the `.out` fibre read checks its stations by the line-station reader's rule (#2256)

The `.out` fibre read now uses the station checks the line-station read uses
(#2246), so both refuse in the same way:

- A station may sit up to 1e-4 outside the natural range [-1, 1]. A file
  written at OpenSees's default `-precision 6` puts an end station near
  `1 + 3e-6`; the fibre read used to refuse it, and now reads it.
- Elements that share one `beamIntegration` rule whose point count is not
  known are refused by name. They used to be read unchecked. Every rule the
  bridge emits has a known count, so a bridge model is not affected.
- A record whose elements have different point counts, or a file whose count
  is not the model's, gets the line-station read's message.

The fibre read still needs the model, and still refuses repeated or
out-of-order stations with #2250's message.
