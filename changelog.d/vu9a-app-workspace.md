### apeGmshViewer: named sets saved to `<stem>.workspace.h5` (ADR 0132, VU9a)

- The **Sets** panel names a selection as a node or element set, renames
  and deletes sets, and shows each set's verdict against the model on
  screen (current, moved, partial "k of n" with Accept, lost) with its
  reason, as Python's `WorkspaceFile.verdicts()` gives it. Stored plots and
  views are listed by name.
- `.workspace.h5` pairs with its set as a suffix of its own, and the open
  set gains a `workspace` slot; its rewrite re-reads the panel, never the set.
- It is the app's one file write, from the human's window only, through
  main's `workspace:save` door, which never takes a path. A save writes a
  partial, verifies it and replaces the file only if no one else changed it;
  a killed save leaves the old file whole. A file another program holds
  is a refusal naming it ("close it there and save again"), never a retry;
  an unreadable file is moved aside and named in the banner; a newer file
  is read-only.
