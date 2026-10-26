### ADDED — `apeGmshViewer` 2-D multi-beam chain plot and interactive 3-D/2-D hover readout in `#line-diagram` (Step 4, #2500)

- **State & pure 2-D SVG chain plot model (`src/state/types.ts`, `src/state/lineDiagram.ts`, `src/state/lineDiagramChainSvg.ts`, `src/state/reduce.ts`, `src/effects.ts`)**:
  - `LineDiagramState` tracks plain-data `drawnBeams: readonly LineDiagramBeamSummary[]`, `chains: readonly BeamChain[]` (built from selected 1-D beam elements via `buildSelectedBeamChainsFromState`), and `hover: LineDiagramHoverReadout | null` (projected via `projectHoverNeedle`).
  - `BeamChainSpan` records `xiMin`, `xiMax`, `startEndKnown`, and `endEndKnown` so Gauss-Legendre interior stations (`xi in (-1, +1)`) are plotted only over `[xiMin, xiMax]` with explicit dashed unknown-end gap markers rather than clamping or extrapolating to the joints.
- **2-D multi-beam chain SVG & hover readout in `#line-diagram` (`src/panels/lineDiagram.ts`)**:
  - Renders `.line-diagram-hover-readout` (showing element ID, natural coordinate `xi`, chain arc-length `s`, interpolated value, and `(outside station span)` when hovering in an unrecorded Gauss end gap) and `.line-diagram-chains` (`.line-diagram-chain-empty` when no beam is selected, or stacked `.line-diagram-chain-card` SVG plots with joint markers, span labels, fill polygons, min/max extrema markers, unknown-end dashed markers/badge, and `.line-diagram-chain-hover-needle`).
  - Pointer movement on `.line-diagram-chain-svg` dispatches `{ type: "hover", at }` and synchronizes both the 2-D SVG hover needle and the 3-D viewport hover needle.
- **3-D viewport hover needle (`src/renderer/lineDiagram.ts`, `src/renderer/viewport.ts`)**:
  - `LineDiagramLayer` renders `hoverNeedle: THREE.LineSegments` from `chordPoint` to `ribbonPoint` whenever `s.lineDiagram.on && s.lineDiagram.hover !== null`, and `Viewport` dispatches `{ type: "hover", at: this.pick(x, y) }` on `pointermove` / `pointerleave`.
