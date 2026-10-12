### CHANGED — apeGmshViewer: stage selection, selected-row highlight, LadrunoQuad, opensees 2.27 (V4f, #1708)

A click on a stage card in the **Stages** window now selects that stage, and
the selected card is highlighted; a second click clears it. The selection
lives in the store, so it survives a file change and a re-read while the same
stage is still at that position, and is cleared otherwise. Selecting a stage
does not change the viewport.

The model-definition windows now redraw when the selection changes or a
go-to-source answer arrives, so the row of the selected declaration is
highlighted in the **Materials & sections** and **Recorders** windows, and each window keeps its
scroll position.

`LadrunoQuad` elements now resolve to their material, so the materials window
no longer reports that the definition chain and `decl_params` disagree on a
LadrunoQuad model. Files at opensees schema 2.27.0 open with no
"newer than this app" banner: the app lists the two new stage records
(`update_parameter`, `zero_velocities`) and does not need the archived
parameter tags. The `/sequence` floor now sits in the app's one floor table,
which the drift test compares with Python's.
