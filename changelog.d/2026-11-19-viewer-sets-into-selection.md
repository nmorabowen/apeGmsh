### ADDED — apeGmshViewer: saved sets into the selection (ADR 0138 SEL-2)

- The selection bar has a Sets list: the workspace's saved sets of the active kind. Each can be selected (replacing the
  selection), added to it or subtracted from it, and an element set can be hidden or isolated.
- An element set selects the elements carrying its OpenSees tags; a node set selects its nodes and makes Nodes active.
  A member the set's verdict says no longer matches the model is left out, and the list says how many of how many come in.
- Selecting a set is one step of the selection's undo. Hiding or isolating a node set (the elements its nodes belong to)
  is not there yet.
