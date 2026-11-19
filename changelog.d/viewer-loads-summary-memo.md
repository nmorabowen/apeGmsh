### FIXED — apeGmshViewer: a step change no longer re-parses every load pattern (#2836)

The loads panel asked for the loads summary on each store event, and the summary
parsed every load table of the model each time: 71% of the renderer's busy time
when stepping through `tims_c3b`, and 77% during playback. The parsed patterns are
now kept per model and stage, and the summary is recomputed only when a series
factor, the units or the load settings change.
