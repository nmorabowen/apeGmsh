### ADDED — ADR 0137 (Proposed): apeGmshViewer large models and tool hosting. No code changes.

- Design only: the stated model support, loading in a utility process, memory warnings before any refusal, one shared step
  read, and no prefetch or step worker yet.
- Tools get one contract and several placements: separate-renderer tiled windows, a dock and anchored cards for light tools.
  The host owns scheduling, canvases and the drawing budget, because a separate renderer isolates CPU work and not GPU work.
- Adds a deny-by-default window and navigation policy, and the implementation order.
