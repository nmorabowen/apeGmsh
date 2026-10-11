### ADDED — apeGmshViewer reads the model-definition data and opens five model-definition windows (V3c0, #1641)

The app's reader now loads `/opensees/decls`, `/opensees/decl_params`
(parameters by field name, `{"$decl": key}` references), the analysis chain,
time series, patterns, recorders, fix records and
`/opensees/stages/stage_NNN` into the one store, and the root `/sequence`
zone when a file carries one (absent is "not in this file", never an error).
A malformed `decl_params` row refuses the file with its path, as Python's
`H5Model.declarations()` does. The app reads OpenSees schema 2.26, so a
2.26.0 `model.h5` opens with no "newer than this app" banner. A **Windows**
menu opens Stages, Analysis, Patterns & series, Recorders, and Materials &
sections, each a placeholder listing the rows its V3 slice will show. The
viewer README now says how to install the app so "Open with" lists it.
