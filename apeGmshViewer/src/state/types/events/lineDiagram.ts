// The lineDiagram slice's events: its part of the store's Event union (types/event.ts).
// Not re-exported by src/state/types.ts, which exports only the whole union.

import type { ContourRange, ContourScan } from "../contour.ts";
import type { LineDiagramBeamSummary, LineDiagramCmap, LineDiagramPlane, LineStationComponent } from "../lineDiagram.ts";

export type LineDiagramEvent =
  | { type: "setLineDiagram"; on: boolean }
  | { type: "setLineDiagramComponent"; component: LineStationComponent }
  | { type: "setLineDiagramScale"; scale: number }
  | { type: "setLineDiagramPlane"; plane: LineDiagramPlane }
  | { type: "setLineDiagramCmap"; cmap: LineDiagramCmap }
  | { type: "setLineDiagramFill"; fill: boolean }
  | { type: "setLineDiagramFlipSign"; flipSign: boolean | null }
  | { type: "setLineDiagramRange"; range: ContourRange; fixed?: { min: number; max: number } }
  | {
      type: "lineDiagramStepRange";
      key: string;
      min: number | null;
      max: number | null;
      notice?: string | null;
      drawnBeams?: readonly LineDiagramBeamSummary[];
    }
  | { type: "lineDiagramScan"; scan: ContourScan };
