// The viewport's side of the nodal contour (V4c1, #1756): the vertex
// colours of the drawn primitives for the step on screen. The values and the
// colours come from state/contour.ts (pure, tested in Node); this module
// picks the step data the effects fetched, checks it is the step the state
// shows, and converts the sRGB ramp to the linear colours three.js expects
// for a vertex colour attribute (it writes sRGB on output). The GPU then
// interpolates the colours inside each primitive.

import type { BlobStore } from "../state/blobs.ts";
import { columnOf, NO_DATA_RGB, primitiveColours, vertexValues } from "../state/contour.ts";
import type { ContourView } from "../state/selectors.ts";
import type { MeshInfo, State } from "../state/types.ts";

/** One step's nodal values, as the effects hold them (effects.ts `StepData`). */
export interface StepValues {
  stage: number;
  step: number;
  nodal: Record<string, Float64Array>;
}

/** sRGB (0..1) to linear, per channel (the sRGB transfer function). */
export const toLinear = (c: number): number => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);

function linearise(rgb: Float32Array): Float32Array {
  for (let i = 0; i < rgb.length; i++) rgb[i] = toLinear(rgb[i]!);
  return rgb;
}

/** Every drawn vertex in the no-data colour (nothing can be coloured: the legend says why). */
export function noDataColours(map: Int32Array, perPrim: number): Float32Array {
  const out = new Float32Array(map.length * perPrim * 3);
  const c = NO_DATA_RGB.map(toLinear);
  for (let i = 0; i < out.length; i += 3) out.set(c, i);
  return out;
}

let columnMemo: { ids: Float64Array; column: Map<number, number> } | null = null;

/** What a set of contour colours shows: the step and component, the range and the colormap (the viewport skips a recolour with the same key). */
export const contourKey = (view: Extract<ContourView, { state: "ready" }>): string => `${view.keys.step}|${view.clim.min}|${view.clim.max}|${view.cmap}`;

/**
 * The linear vertex colours of the drawn lines and triangles for the step
 * on screen, and the key of that step; null when the step's data or its
 * measured range is not here yet (the viewport keeps what it shows).
 */
export function contourColours(
  s: State,
  view: Extract<ContourView, { state: "ready" }>,
  mesh: MeshInfo,
  blobs: BlobStore,
  step: StepValues | null,
  lineMap: Int32Array,
  triMap: Int32Array,
): { line: Float32Array; tri: Float32Array; key: string } | null {
  const at = s.phase.at;
  if (at?.kind !== "results" || !step || step.stage !== at.stage || step.step !== at.step || !view.stepMeasured) return null;
  const rows = view.component.datasets.map((d) => step.nodal[d]);
  if (!rows.every((r): r is Float64Array => r instanceof Float64Array)) {
    throw new Error(`contour: the step data has no ${view.component.datasets.filter((d) => !(step.nodal[d] instanceof Float64Array)).join(", ")}`);
  }
  const ids = blobs.get(view.nodeIds);
  if (!(ids instanceof Float64Array)) throw new TypeError(`contour: ${view.nodeIds.id} is not f64`);
  if (!columnMemo || columnMemo.ids !== ids) columnMemo = { ids, column: columnOf(ids) };
  const vn = mesh.vertexNodes;
  const lineNodes = blobs.get(vn.line) as Float64Array;
  const triNodes = blobs.get(vn.tri) as Float64Array;
  const clim = { min: view.clim.min, max: view.clim.max };
  const line = primitiveColours(vertexValues(lineNodes, columnMemo.column, rows, view.component.magnitude), lineMap, 2, clim, view.cmap).rgb;
  const tri = primitiveColours(vertexValues(triNodes, columnMemo.column, rows, view.component.magnitude), triMap, 3, clim, view.cmap).rgb;
  return { line: linearise(line), tri: linearise(tri), key: contourKey(view) };
}
