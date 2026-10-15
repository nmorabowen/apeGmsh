### CHANGED — spec-emitted `.out` recorders write full precision (program slice X, #1974)

Recorders emitted from a `ResolvedRecorderSpec` (Tcl, Python and live
`emit_recorders`) now carry `-precision 17`. OpenSees writes 6 significant
digits by default, which held every `.out` value oracle near rtol 1e-5;
17 digits round-trip an IEEE double, so a `.out` file now matches
`ops.nodeDisp` / `ops.eleResponse` to rtol 1e-12, like `.ladruno` and MPCO.
`LogicalRecorder` gains a `precision` field (default 17; a value that is not
a positive int raises), and the three formatters emit it in lockstep, just
before the response tokens. Output files are larger, about 2.5 times the
characters per value. The `.out` parsers read the longer numbers unchanged.
