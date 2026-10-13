// The viewport's side of the nodal contour (V4c1, #1756): the vertex
// colours of the drawn primitives for the step on screen. The values and the
// colours come from state/contour.ts (pure, tested in Node); this module
// picks the step data the effects fetched, checks it is the step the state
// shows, and converts the sRGB ramp to the linear colours three.js expects
// for a vertex colour attribute (it writes sRGB on output). The GPU then
// interpolates the colours inside each primitive.

import type { BlobStore } from "../state/blobs.ts";
import { cellVertexValues, columnOf, elementCellValues, gaussCellValues, isNodeTopology, joinElements, meshFemIds, NO_DATA_RGB, primitiveColours, vertexValues } from "../state/contour.ts";
import { applyPlan, cornerVertexValues, planFor, vertexCorners, type NodeTopology } from "../state/extrapolation.ts";
import type { ContourView } from "../state/selectors.ts";
import type { MeshInfo, State } from "../state/types.ts";

/** One step's nodal values, as the effects hold them (effects.ts `StepData`). */
export interface StepValues {
  stage: number;
  step: number;
  nodal: Record<string, Float64Array>;
  /** V4c2a: the element component fetched with the step (null: none), and its values by gauss group, (E * G,) */
  element?: string | null;
  gauss?: Record<string, Float64Array>;
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

/**
 * What a set of contour colours shows: the results read (a re-read at the
 * same path can hold other values), the step and component, the range and
 * the colormap. The viewport skips a recolour with the same key.
 */
export const contourKey = (view: Extract<ContourView, { state: "ready" }>): string => `read ${view.reads}|${view.keys.step}|${view.clim.min}|${view.clim.max}|${view.cmap}`;

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

// -- V4c2a: element (Gauss) contours, flat per cell -------------------------------

/** What a set of element colours shows: the nodal key's parts, and the join the rows went through (it names the read, the mesh and the rows). */
export const elementContourKey = (view: Extract<ContourView, { state: "element" }>): string =>
  `read ${view.reads}|${view.keys.step}|${view.clim.min}|${view.clim.max}|${view.cmap}|${view.topology}|join ${view.joinKey}`;

/** The key of the colours a contour view shows once they are applied (nodal, element, or no data with its reason); null when off. */
export function contourShownKey(v: ContourView): string | null {
  switch (v.state) {
    case "ready":
      return contourKey(v);
    case "element":
      return elementContourKey(v);
    case "unavailable":
    case "refused":
    case "not-yet":
      return `no-data:${v.reason}`;
    case "off":
      return null;
    default: {
      const unknown: never = v;
      throw new Error(`unknown contour view ${JSON.stringify(unknown)}`);
    }
  }
}

let joinMemo:{ key: string; at: Map<number, { group: number; row: number }> } | null = null;
let femMemo: { elements: readonly string[]; decls: unknown; ids: Float64Array } | null = null;

/**
 * The linear vertex colours of the drawn lines and triangles for an element
 * field at the step on screen: each primitive in its element's value (the
 * cell's one Gauss value, or the mean of its Gauss values), joined to the
 * element by FEM id only. A drawn element no group has is no data. Null while
 * the step's data or its measured range is not here.
 */
export function elementColours(
  s: State,
  view: Extract<ContourView, { state: "element" }>,
  mesh: MeshInfo,
  blobs: BlobStore,
  step: StepValues | null,
  lineMap: Int32Array,
  triMap: Int32Array,
): { line: Float32Array; tri: Float32Array; key: string } | null {
  const at = s.phase.at;
  if (at?.kind !== "results" || !step || step.stage !== at.stage || step.step !== at.step || !view.stepMeasured) return null;
  if (step.element !== view.component.key || !step.gauss) return null;
  if (!joinMemo || joinMemo.key !== view.joinKey) {
    const j = joinElements(
      s.decls,
      view.groups.map((g) => ({ id: g.id, eids: blobs.get(g.eids) as Float64Array })),
    );
    // The state says this join is ok (the effects checked it for this key); anything else is a bug, never a guessed join.
    if (!j.ok) throw new Error(`element contour: the join ${view.joinKey} is refused (${j.reason}) while the state shows it`);
    joinMemo = { key: view.joinKey, at: j.at };
  }
  if (!femMemo || femMemo.elements !== mesh.elements || femMemo.decls !== s.decls) femMemo = { elements: mesh.elements, decls: s.decls, ids: meshFemIds(mesh.elements, s.decls) };
  if (isNodeTopology(view.topology)) return nodeColours(s, { ...view, topology: view.topology }, mesh, blobs, step.gauss, lineMap, triMap);
  const perGroup = view.groups.map((g) => {
    const v = step.gauss![g.id];
    if (!(v instanceof Float64Array)) throw new Error(`element contour: the step data has no ${view.component.key} for group ${g.id}`);
    return gaussCellValues(v, g.nElements, g.nGauss);
  });
  const values = elementCellValues(femMemo.ids, joinMemo.at, perGroup);
  const clim = { min: view.clim.min, max: view.clim.max };
  const line = primitiveColours(cellVertexValues(values, blobs.i32(mesh.lineElement), 2), lineMap, 2, clim, view.cmap).rgb;
  const tri = primitiveColours(cellVertexValues(values, blobs.i32(mesh.triElement), 3), triMap, 3, clim, view.cmap).rgb;
  return { line: linearise(line), tri: linearise(tri), key: elementContourKey(view) };
}

// -- V4c2b: element fields carried to the nodes -------------------------------------

let cornerMemo: { key: string; femIds: Float64Array; line: Int32Array; tri: Int32Array } | null = null;

/**
 * The linear vertex colours for a node-located topology (V4c2b): the step's
 * Gauss values carried through the field's plan (state/extrapolation.ts),
 * each vertex coloured by its own element's corner at its node (discrete) or
 * by that node's average (averaged); a vertex of an element with no data is
 * no data, whatever its neighbours hold.
 */
function nodeColours(
  s: State,
  view: Extract<ContourView, { state: "element" }> & { topology: NodeTopology },
  mesh: MeshInfo,
  blobs: BlobStore,
  gauss: Record<string, Float64Array>,
  lineMap: Int32Array,
  triMap: Int32Array,
): { line: Float32Array; tri: Float32Array; key: string } {
  const plan = planFor(view, s, blobs);
  const key = `${view.joinKey}|${view.topology}|${mesh.triElement.id}`;
  if (!cornerMemo || cornerMemo.key !== key || cornerMemo.femIds !== femMemo!.ids) {
    const fem = femMemo!.ids;
    const vn = mesh.vertexNodes;
    cornerMemo = {
      key,
      femIds: fem,
      line: vertexCorners(plan, joinMemo!.at, fem, blobs.i32(mesh.lineElement), blobs.get(vn.line) as Float64Array, 2),
      tri: vertexCorners(plan, joinMemo!.at, fem, blobs.i32(mesh.triElement), blobs.get(vn.tri) as Float64Array, 3),
    };
  }
  const r = applyPlan(
    plan,
    view.groups.map((g) => {
      const v = gauss[g.id];
      if (!(v instanceof Float64Array)) throw new Error(`element contour: the step data has no ${view.component.key} for group ${g.id}`);
      return v;
    }),
  );
  const clim = { min: view.clim.min, max: view.clim.max };
  const line = primitiveColours(cornerVertexValues(plan, r, cornerMemo.line), lineMap, 2, clim, view.cmap).rgb;
  const tri = primitiveColours(cornerVertexValues(plan, r, cornerMemo.tri), triMap, 3, clim, view.cmap).rgb;
  return { line: linearise(line), tri: linearise(tri), key: elementContourKey(view) };
}
