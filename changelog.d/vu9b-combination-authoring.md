### apeGmshViewer: write a User Combination in the app, saved into `<stem>.user.h5` (ADR 0130 A4, VU9b)

- A `+` beside the phase bar's families opens the **New combination**
  panel: a name, a rule (with its envelope, fraction or damping) and terms,
  each a source address and a factor. The draft is checked as it is typed,
  first by the readers' rules beside the file's rows, then by the app's
  evaluator for one nodal component of the `/spectra` cases it reaches. A
  refused draft shows its reason and cannot be saved. A draft of stage or
  mode terms stands but is not evaluated in the app.
- **Save** adds a generation `g<N+1>` to `/combinations` of the open set's
  `.user.h5` through main's `combinations:save` door, which never takes a
  path: the file is copied to the writer's own partial, the copy is
  appended to, read back, and replaces the original only if `/meta@revision`
  has not moved. The live file is never opened for writing; a killed save
  leaves it byte-identical, and the next open sweeps the dead partial. No
  other zone changes. **Revert last save** undoes the save made last in the
  window, once, while the file is as that save left it.
- A file another program holds is a refusal naming it, never a retry. A
  disk without a second file's worth of free space refuses before the copy.
  The app never creates a `.user.h5`, and only the human's window saves.
- Python's `UserFile` reads the generations the app writes, and evaluates
  them to the app's values.
