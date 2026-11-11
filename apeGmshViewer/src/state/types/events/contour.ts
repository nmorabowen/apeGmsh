// The contour slice's events: its part of the store's Event union (types/event.ts).
// Not re-exported by src/state/types.ts, which exports only the whole union.

import type { ContourCmap, ContourRange, ContourScan, ContourSource, ElementJoinVerdict, GaussMode, UnitDisplay } from "../contour.ts";
import type { UnitSystemId } from "../model.ts";

export type ContourEvent =
  /**
   * V4c1: pick the contour's field and component (a component of state/contour.ts
   * `nodalFields` of the stage shown); V4c2a: `source: "element"` picks an
   * element field (`elementFields`). Absent `source` is nodal.
   */
  | { type: "setContourField"; field: string; component: string; source?: ContourSource }
  /** V4c2a: how an element field is drawn */
  | { type: "setContourGauss"; mode: GaussMode }
  /** V4c2a (effects): the join of the picked element field's rows to the mesh */
  | { type: "contourElementJoin"; verdict: ElementJoinVerdict }
  /** V4c1: the colour range mode; `fixed` takes the user's min and max */
  | { type: "setContourRange"; range: ContourRange; fixed?: { min: number; max: number } }
  | { type: "setContourCmap"; cmap: ContourCmap }
  /** baseUnits: set the file's assumed native unit system and/or the display unit system */
  | {
      type: "setContourUnits";
      source?: UnitSystemId | null;
      target?: UnitDisplay;
      sourceUnits?: UnitSystemId | null;
      targetUnits?: UnitDisplay;
    }
  /** V4c1 (effects): the finite range of the step shown, for the key it was measured on */
  | { type: "contourStepRange"; key: string; min: number | null; max: number | null }
  /** V4c1 (effects): the history scan started, progressed, finished or failed */
  | { type: "contourScan"; scan: ContourScan };
