### apeGmshViewer: the phase bar lists state families, the contour picker lists User Results (ADR 0130 slice 3b)

The phase bar now shows the results by family. Stages and Modes are the results
file's stages by kind, and a click opens the first one; the stage picker then lists
that family only. Spectra and User Combinations come from the set's `.user.h5`.
They are listed and not drawn yet: their buttons are disabled, and the tooltip
names each member. A family says how many of its members are stale, and how many
rows the reader refused. The contour field picker groups its fields as Results and
User Results, and the legend lists each user result with its target, whether it is
per state, and whether it is stale.

The window reads the `.user.h5` after the model's first read has landed, so the
first frame never waits on it. It reads it again when the user file, the model or
the results change, because staleness depends on all three. A file the app cannot
take is shown as refused with the reason, and a zone it does not know is named in a
warning. A results stage that records no step is listed and never opened.
