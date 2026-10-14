// The cut face of a solid (V4e3, #1904; ADR 0083 S3): a cap on each section
// plane that cuts a volumetric mesh. Clipping discards fragments, so a plane
// through a solid opens onto an empty box; the cap fills it with the slice
// of the drawn solid cells (state/slice.ts), deformed as drawn and carrying
// the contour on screen.
//
// - Source: the volumetric cells (tet, hex, wedge, pyramid) of the model's
//   neutral blocks, minus the cells the legend hides or the stage has not
//   activated, over the full node table (`MeshInfo.nodes`: the cut meets
//   interior nodes no drawn face has). A shell, beam or 2-D mesh has none:
//   it gets no cap and the panel says so, never a fake fill.
// - Values: the contour's own, at each cell corner: a nodal field's node
//   value, an element field's cell value (`gauss_cell`, `element_mean`: the
//   cap is coloured per cell) or its carried corner value (the node-located
//   modes), through the same join and plan as the surfaces. The cap's clim,
//   colormap and no-data colour are the contour's. No data stays no data
//   (state/slice.ts), never 0. With no contour the cap is one flat fill.
// - Trim and exemption: each cap is cut by the other active planes
//   geometrically (`sliceSolids`' trims) and is clip-exempt (the stamping in
//   renderer/clip.ts skips its subtree): it lies on its own plane, where the
//   GPU's keep-or-discard is a coin toss. It steps CAP_STEP of its own extent
//   into the discarded half, so it never speckles against a coplanar face or
//   the V4e2 gizmo quad (the Qt S3 measurement).
// - When: on drag-release and on a step change, never per drag tick; mid-drag
//   the caps are hidden. A change of clim or colormap alone recolours the
//   caps without slicing again.

import * as THREE from "three";
import type { BlobStore } from "../state/blobs.ts";
import { cutsOf, type Cut } from "../state/clip.ts";
import { columnOf, elementCellValues, gaussCellValues, isNodeTopology, NO_DATA_RGB, primitiveColours, vertexValues } from "../state/contour.ts";
import { vertexDisplacements } from "../state/deform.ts";
import { applyPlan, cornerVertexValues, planFor, vertexCorners } from "../state/extrapolation.ts";
import { activationNow, contourView, deformView, NO_SOLID_CELLS, type ContourView } from "../state/selectors.ts";
import { SOLID_CORNERS, SOLID_SHAPES, sliceSolids, sliceExtent, solidShapeOf, type Slice, type Solids } from "../state/slice.ts";
import type { Decl, DeclPath, State } from "../state/types.ts";
import { DARK_ROLE } from "../theme/tokens.ts";
import { elementJoinAt, toLinear, type StepValues } from "./contour.ts";

/** How far a cap steps into the discarded half: a fraction of its own extent (the Qt S3 measurement: less speckles against the gizmo quad). */
export const CAP_STEP = 1e-3;

/** The flat fill of a cap with no contour (sRGB 0..1): the concrete tone, never the no-data grey. */
export const CAP_FILL: readonly [number, number, number] = (() => {
  const n = parseInt(DARK_ROLE.concrete.slice(1), 16);
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
})();

/** Why a model gets no cap (the panel shows it: selectors `cutFaceNote`). */
export const NO_NODE_TABLE = "the model was read without its node table, so the cut face cannot be built";

/** The volumetric cells of one mesh read, over the full node table. */
export interface SolidTable {
  solids: Solids;
  /** per cell, its FEM id and its declaration */
  fem: Float64Array;
  decl: DeclPath[];
  /** per corner, its node tag */
  cornerNodes: Float64Array;
  /** the node table: ids and reference coordinates (3 per node) */
  ids: Float64Array;
  xyz: Float64Array;
}

let tableMemo: { key: readonly unknown[]; value: SolidTable | { none: string } } | null = null;

/** The volumetric cells of the state's mesh (once per read), or why there are none. */
export function solidTable(s: Pick<State, "decls" | "blocks" | "mesh">, blobs: BlobStore): SolidTable | { none: string } {
  const mesh = s.mesh;
  if (!mesh) return { none: "no mesh is loaded" };
  const key = [s.decls, s.blocks, mesh];
  if (tableMemo && tableMemo.key.every((k, i) => k === key[i])) return tableMemo.value;
  const value = buildTable(s, blobs);
  tableMemo = { key, value };
  return value;
}

function buildTable(s: Pick<State, "decls" | "blocks" | "mesh">, blobs: BlobStore): SolidTable | { none: string } {
  const mesh = s.mesh!;
  if (mesh.counts.solidCells === 0) return { none: NO_SOLID_CELLS };
  if (!mesh.nodes) return { none: NO_NODE_TABLE };
  const ids = blobs.get(mesh.nodes.ids), xyz = blobs.get(mesh.nodes.xyz);
  if (!(ids instanceof Float64Array) || !(xyz instanceof Float64Array)) throw new TypeError("caps: the node table is not f64");
  const row = new Map<number, number>();
  ids.forEach((id, i) => row.set(id, i));
  const conn = new Map<number, Float64Array>();
  const shape: number[] = [], start: number[] = [0], slot: number[] = [], fem: number[] = [], decl: DeclPath[] = [], corner: number[] = [];
  for (const d of Object.values(s.decls)) {
    const cell = d.element?.cell;
    if (!cell) continue;
    const b = s.blocks[cell.block];
    if (!b) throw new Error(`caps: ${d.path} names block ${cell.block}, which the state does not have`);
    const sh = solidShapeOf(b.code);
    if (sh === null) continue;
    let c = conn.get(cell.block);
    if (!c) {
      const x = blobs.get(b.connectivity);
      if (!(x instanceof Float64Array)) throw new TypeError(`blob ${b.connectivity.id} is not f64`);
      conn.set(cell.block, (c = x));
    }
    const nc = SOLID_CORNERS[sh];
    const rows: number[] = [];
    for (let k = 0; k < nc; k++) {
      const r = row.get(c[cell.row * b.npe + k]!);
      if (r === undefined) break; // a tag not in /nodes: mesh/build.ts warns and draws no face of it either
      rows.push(r);
    }
    if (rows.length !== nc) continue;
    shape.push(SOLID_SHAPES.indexOf(sh));
    for (const r of rows) {
      slot.push(r);
      corner.push(ids[r]!);
    }
    start.push(slot.length);
    fem.push(d.element!.femId ?? NaN);
    decl.push(d.path);
  }
  if (!shape.length) return { none: NO_SOLID_CELLS };
  return {
    solids: { shape: Uint8Array.from(shape), start: Int32Array.from(start), slot: Int32Array.from(slot) },
    fem: Float64Array.from(fem),
    decl,
    cornerNodes: Float64Array.from(corner),
    ids,
    xyz,
  };
}

/** The colour group of a cell: the smallest element-side physical group holding it, ties by name (mesh/build.ts `colourGroups`). */
function colourGroupOf(d: Decl, decls: Readonly<Record<DeclPath, Decl>>): DeclPath | null {
  let best: Decl | null = null;
  for (const p of d.element?.groups ?? []) {
    const g = decls[p];
    if (!g?.group) continue;
    if (!best || g.group.count < best.group!.count || (g.group.count === best.group!.count && g.name.localeCompare(best.name) < 0)) best = g;
  }
  return best?.path ?? null;
}

/** Which cells are drawn (1) given the hidden legend rows and the stage's activation; null when every cell is. */
export function shownCells(s: State, t: SolidTable): Uint8Array | null {
  const hidden = new Set(s.visibility.hidden);
  const a = activationNow(s)?.activation;
  const inactive = a?.state === "filtered" ? a.inactiveTags : null;
  if (!hidden.size && !inactive?.size) return null;
  const out = new Uint8Array(t.decl.length);
  t.decl.forEach((p, c) => {
    const d = s.decls[p]!;
    const g = hidden.size ? colourGroupOf(d, s.decls) : null;
    const off = (g !== null && hidden.has(g)) || (inactive !== null && (d.element?.metas ?? []).some((m) => inactive.has(m.tag)));
    out[c] = off ? 0 : 1;
  });
  return out;
}

/** The cells of `t` that `mask` keeps (all when null), with their FEM ids and corner node tags. */
export function subsetOf(t: SolidTable, mask: Uint8Array | null): { solids: Solids; fem: Float64Array; cornerNodes: Float64Array; cornerCell: Int32Array } {
  const n = t.solids.shape.length;
  const cornerCellOf = (solids: Solids) => {
    const out = new Int32Array(solids.slot.length);
    for (let c = 0; c + 1 < solids.start.length; c++) out.fill(c, solids.start[c]!, solids.start[c + 1]!);
    return out;
  };
  if (!mask) return { solids: t.solids, fem: t.fem, cornerNodes: t.cornerNodes, cornerCell: cornerCellOf(t.solids) };
  const shape: number[] = [], start: number[] = [0], slot: number[] = [], fem: number[] = [], corner: number[] = [];
  for (let c = 0; c < n; c++) {
    if (!mask[c]) continue;
    shape.push(t.solids.shape[c]!);
    for (let k = t.solids.start[c]!; k < t.solids.start[c + 1]!; k++) {
      slot.push(t.solids.slot[k]!);
      corner.push(t.cornerNodes[k]!);
    }
    start.push(slot.length);
    fem.push(t.fem[c]!);
  }
  const solids = { shape: Uint8Array.from(shape), start: Int32Array.from(start), slot: Int32Array.from(slot) };
  return { solids, fem: Float64Array.from(fem), cornerNodes: Float64Array.from(corner), cornerCell: cornerCellOf(solids) };
}

/** What the caps show: a slice key (what is cut, where, and which values) and a colour key (how values become colours); null when no cap is wanted. */
export function capKeys(s: State): { cuts: string; slice: string; colour: string } | null {
  const cuts = cutsOf(s.clip);
  if (!s.mesh || !s.clip.cutFace || !cuts.length || s.phase.at?.kind === "geometry") return null;
  const cutKey = JSON.stringify(cuts);
  const a = activationNow(s)?.activation;
  const shown = `hidden ${s.visibility.hidden.join(",")}|act ${a?.state === "filtered" ? `${a.row}:${a.name}` : "all"}`;
  const d = deformView(s);
  const deform = d.state === "loading" || d.state === "ready" || d.state === "partial" ? `${d.keys.disp}|scale ${d.scale}` : "base";
  const v = contourView(s);
  let values: string, colour: string;
  switch (v.state) {
    case "off":
      values = "fill";
      colour = "fill";
      break;
    case "ready":
      values = `read ${v.reads}|${v.keys.step}`;
      colour = `${v.clim.min}|${v.clim.max}|${v.cmap}`;
      break;
    case "element":
      values = `read ${v.reads}|${v.keys.step}|${v.topology}|join ${v.joinKey}`;
      colour = `${v.clim.min}|${v.clim.max}|${v.cmap}`;
      break;
    default:
      values = `no-data:${v.reason}`;
      colour = "no-data";
  }
  return { cuts: cutKey, slice: `${cutKey}|mesh ${s.mesh.triPositions.id}|${shown}|${deform}|${values}`, colour };
}

/** The positions, values and cuts of one cap pass, or `wait` while the step shown is not all here, or why there is no cap. */
export type CapInputs =
  | { solids: Solids; fem: Float64Array; xyz: Float64Array; values: Float64Array | null; cuts: Cut[] }
  | { wait: true }
  | { none: string };

const stepIsShown = (s: State, step: StepValues | null): step is StepValues => {
  const at = s.phase.at;
  return at?.kind === "results" && step !== null && step.stage === at.stage && step.step === at.step;
};

function row(step: StepValues, name: string, what: string): Float64Array {
  const r = step.nodal[name];
  if (!(r instanceof Float64Array)) throw new Error(`${what}: the step data has no ${name}`);
  return r;
}

/** The cap pass's inputs from the state, the blobs and the effects' step data. */
export function capInputs(s: State, blobs: BlobStore, step: StepValues | null): CapInputs {
  const t = solidTable(s, blobs);
  if ("none" in t) return t;
  const sub = subsetOf(t, shownCells(s, t));
  if (!sub.solids.shape.length) return { none: "every volumetric cell is hidden" };
  // Positions: the node table, deformed as the surfaces are (a node with no displacement stays put, as it is drawn).
  let xyz = t.xyz;
  const d = deformView(s);
  if (d.state === "loading" || d.state === "ready" || d.state === "partial") {
    if (!stepIsShown(s, step)) return { wait: true };
    const [dx, dy, dz] = d.datasets;
    const rows = [row(step, dx, "deformed shape"), row(step, dy, "deformed shape"), dz === null ? null : row(step, dz, "deformed shape")] as const;
    const ids = blobs.get(d.nodeIds);
    if (!(ids instanceof Float64Array)) throw new TypeError(`deformed shape: ${d.nodeIds.id} is not f64`);
    const u = vertexDisplacements(t.ids, columnOf(ids), rows).d;
    xyz = new Float64Array(t.xyz.length);
    for (let i = 0; i < xyz.length; i++) xyz[i] = t.xyz[i]! + d.scale * u[i]!;
  }
  const v = contourView(s);
  let values: Float64Array | null;
  switch (v.state) {
    case "off":
      values = null;
      break;
    case "ready": {
      if (!stepIsShown(s, step) || !v.stepMeasured) return { wait: true };
      const ids = blobs.get(v.nodeIds);
      if (!(ids instanceof Float64Array)) throw new TypeError(`contour: ${v.nodeIds.id} is not f64`);
      values = vertexValues(sub.cornerNodes, columnOf(ids), v.component.datasets.map((n) => row(step, n, "contour")), v.component.magnitude);
      break;
    }
    case "element": {
      if (!stepIsShown(s, step) || !v.stepMeasured || step.element !== v.component.key || !step.gauss) return { wait: true };
      const gauss = v.groups.map((g) => {
        const x = step.gauss![g.id];
        if (!(x instanceof Float64Array)) throw new Error(`element contour: the step data has no ${v.component.key} for group ${g.id}`);
        return x;
      });
      const at = elementJoinAt(s, v, blobs);
      if (isNodeTopology(v.topology)) {
        const plan = planFor({ ...v, topology: v.topology }, s, blobs);
        // Each corner a one-vertex primitive of its own cell: the cell's corner at that node (discrete) or the node's average.
        const corners = vertexCorners(plan, at, sub.fem, sub.cornerCell, sub.cornerNodes, 1);
        values = cornerVertexValues(plan, applyPlan(plan, gauss), corners);
      } else {
        const cell = elementCellValues(sub.fem, at, v.groups.map((g, i) => gaussCellValues(gauss[i]!, g.nElements, g.nGauss)));
        values = Float64Array.from(sub.cornerCell, (c) => cell[c]!);
      }
      break;
    }
    default:
      values = new Float64Array(sub.solids.slot.length).fill(NaN);
  }
  return { solids: sub.solids, fem: sub.fem, xyz, values, cuts: cutsOf(s.clip) };
}

/** The linear vertex colours of a cap: the contour's ramp, clim and no-data rule, or the flat fill. */
export function capColours(slice: Slice, view: ContourView): Float32Array {
  const n = slice.cell.length;
  let rgb: Float32Array;
  if (view.state === "off") {
    rgb = new Float32Array(9 * n);
    for (let i = 0; i < rgb.length; i += 3) rgb.set(CAP_FILL, i);
  } else if (view.state === "ready" || view.state === "element") {
    rgb = primitiveColours(slice.values!, Int32Array.from({ length: n }, (_, i) => i), 3, view.clim, view.cmap).rgb;
  } else {
    rgb = new Float32Array(9 * n);
    for (let i = 0; i < rgb.length; i += 3) rgb.set(NO_DATA_RGB, i);
  }
  for (let i = 0; i < rgb.length; i++) rgb[i] = toLinear(rgb[i]!);
  return rgb;
}

/** One cap drawn: its plane, its slice and its mesh. */
interface Cap {
  cut: Cut;
  slice: Slice;
  mesh: THREE.Mesh;
}

/** The caps of the viewport: one mesh per cutting plane through a solid, under one clip-exempt group. */
export class CapLayer {
  readonly root = new THREE.Group();
  /** `sliceSolids` calls so far (F-SETTLED counts them; the capture reports them) */
  slices = 0;
  /** why there is no cap although one is wanted (a shell model, every cell hidden); null otherwise */
  note: string | null = null;
  private caps: Cap[] = [];
  private keys: { cuts: string; slice: string; colour: string } | null = null;
  private dim = 1;

  constructor() {
    this.root.name = "section caps";
    // renderer/clip.ts skips this subtree: no plane cuts a cap (it is trimmed by the others itself).
    this.root.userData["clipExempt"] = true;
    this.root.visible = false;
  }

  /** How many caps are drawn now. */
  get count(): number {
    return this.root.visible ? this.caps.length : 0;
  }

  /** The caps as drawn (tests, capture). */
  get drawn(): readonly { cut: Cut; slice: Slice; mesh: THREE.Mesh }[] {
    return this.caps;
  }

  /**
   * Follow the state: hidden while a plane is dragged, sliced again when what
   * is cut changed (planes, cells, positions, values), recoloured when only
   * the colour key did, kept while the step shown is still on its way. True
   * when the drawing changed.
   */
  sync(s: State, blobs: BlobStore, step: StepValues | null, dragging: boolean): boolean {
    const keys = capKeys(s);
    if (!keys) {
      this.note = null;
      const had = this.caps.length > 0 || this.root.visible;
      this.clear();
      this.keys = null;
      return had;
    }
    if (dragging) {
      if (!this.root.visible) return false;
      this.root.visible = false;
      return true;
    }
    if (this.keys?.slice === keys.slice) {
      let changed = !this.root.visible && this.caps.length > 0;
      this.root.visible = this.caps.length > 0;
      if (this.keys.colour !== keys.colour) {
        const view = contourView(s);
        for (const c of this.caps) this.paint(c, view);
        this.keys = keys;
        changed = true;
      }
      return changed;
    }
    const inputs = capInputs(s, blobs, step);
    if ("wait" in inputs) {
      // The step is on its way: keep the caps while they are still on the same planes, else hide them.
      const show = this.keys?.cuts === keys.cuts && this.caps.length > 0;
      const changed = show !== this.root.visible;
      this.root.visible = show;
      return changed;
    }
    this.clear();
    this.keys = keys;
    if ("none" in inputs) {
      this.note = inputs.none;
      return true;
    }
    this.note = null;
    const view = contourView(s);
    inputs.cuts.forEach((cut, i) => {
      const trims = inputs.cuts.filter((_, j) => j !== i);
      this.slices++;
      const slice = sliceSolids(inputs.solids, inputs.xyz, inputs.values, cut, trims);
      if (!slice) return; // the plane cuts no drawn cell: no cap, never an empty one
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.BufferAttribute(Float32Array.from(slice.positions), 3));
      g.setAttribute("color", new THREE.BufferAttribute(new Float32Array(slice.positions.length), 3));
      g.computeVertexNormals();
      const m = new THREE.MeshStandardMaterial({ vertexColors: true, side: THREE.DoubleSide, roughness: 0.75, metalness: 0 });
      m.color.setScalar(this.dim);
      const mesh = new THREE.Mesh(g, m);
      mesh.name = "section cap";
      // CAP_STEP of its own extent into the discarded half: never coplanar with its plane's gizmo quad or a face on it.
      const step = CAP_STEP * sliceExtent(slice);
      mesh.position.set(-cut.normal[0] * step, -cut.normal[1] * step, -cut.normal[2] * step);
      mesh.raycast = () => {}; // the pick sees the model, not its caps (ADR 0083 open question 3)
      const cap = { cut, slice, mesh };
      this.paint(cap, view);
      this.caps.push(cap);
      this.root.add(mesh);
    });
    this.root.visible = this.caps.length > 0;
    return true;
  }

  /** Dim the caps with the unselected model (viewport `applyDim`). */
  setDim(k: number): void {
    this.dim = k;
    for (const c of this.caps) (c.mesh.material as THREE.MeshStandardMaterial).color.setScalar(k);
  }

  private paint(c: Cap, view: ContourView): void {
    const a = c.mesh.geometry.getAttribute("color") as THREE.BufferAttribute;
    (a.array as Float32Array).set(capColours(c.slice, view));
    a.needsUpdate = true;
  }

  private clear(): void {
    for (const c of this.caps) {
      c.mesh.geometry.dispose();
      (c.mesh.material as THREE.Material).dispose();
      this.root.remove(c.mesh);
    }
    this.caps = [];
    this.root.visible = false;
  }

  dispose(): void {
    this.clear();
  }
}

/**
 * A listen target that runs `after` once the listeners it was given have
 * handled a pointer release (`pointerup`, `pointercancel`, `pointerleave`).
 * The gizmo interactor (renderer/gizmo.ts) ends its drag in those handlers
 * and dispatches nothing then (a release moves no plane), so the viewport
 * hands it its canvas through this wrapper to learn the drag is over: the
 * moment the caps are sliced again.
 */
export function releaseAware<T extends { addEventListener(type: string, l: (e: Event) => void, o?: AddEventListenerOptions | boolean): void; removeEventListener(type: string, l: (e: Event) => void, o?: EventListenerOptions | boolean): void }>(
  target: T,
  after: () => void,
): T {
  const RELEASE = new Set(["pointerup", "pointercancel", "pointerleave"]);
  const wrapped = new Map<string, Map<(e: Event) => void, (e: Event) => void>>();
  return new Proxy(target, {
    get(t, prop) {
      if (prop === "addEventListener") {
        return (type: string, l: (e: Event) => void, o?: AddEventListenerOptions | boolean) => {
          if (!RELEASE.has(type)) return t.addEventListener(type, l, o);
          const w = (e: Event) => {
            try {
              l(e);
            } finally {
              after();
            }
          };
          let m = wrapped.get(type);
          if (!m) wrapped.set(type, (m = new Map()));
          m.set(l, w);
          t.addEventListener(type, w, o);
        };
      }
      if (prop === "removeEventListener") {
        return (type: string, l: (e: Event) => void, o?: EventListenerOptions | boolean) => t.removeEventListener(type, wrapped.get(type)?.get(l) ?? l, o);
      }
      const v = Reflect.get(t, prop, t) as unknown;
      return typeof v === "function" ? (v as (...a: unknown[]) => unknown).bind(t) : v;
    },
  });
}
