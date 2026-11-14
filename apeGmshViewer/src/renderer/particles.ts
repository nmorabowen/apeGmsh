// The MPM particles' points layer (ADR 0134 D5; chain M, M2-S4 #2983).
//
// One `THREE.Points` per particle stage, made by renderer/nodes.ts `makeNodePoints`
// (the node markers' round, rimmed point, one vertex each) with a colour per vertex.
// The effects read one output at a time (main's `particles:readOutput`, a hyperslab of
// `OFFSETS[k]..OFFSETS[k+1]`) and hand it here as a `ParticleFrame`; this layer copies
// it into the stage's buffers, which grow when an output holds more particles than any
// before it and are otherwise reused, so a scrub allocates nothing and the history is
// never in memory.
//
// - Colour: one component of one field through the contour's ramp (state/contour.ts
//   `sampleCmap`, `rampT`), linearised as the contour texture is; a NaN (a `STATE_*`
//   field on a particle whose material does not answer) is the contour's no-value
//   colour (`NO_DATA_RGB`). With no field the points take one plain colour.
// - The grid (`MODEL/PARTICLES/GRID`) is an optional wire box: its outline, at z = 0 in 2-D.
// - Seams: a new stage key (another stage, file, or a re-read of it) disposes the old
//   stage's geometry and material before the new one is made; `show(null)` disposes
//   both, and an output with no particle draws none (draw range 0).

import * as THREE from "three";
import { NO_DATA_RGB, rampT, sampleCmap } from "../state/contour.ts";
import type { ContourCmap } from "../state/types/contour.ts";
import { toLinear } from "./colormapTexture.ts";
import { makeNodePoints, setNodeSize } from "./nodes.ts";

/** One output of one particle stage, as the effects hand it to the viewport. */
export interface ParticleFrame {
  /** the stage's identity: a new key disposes the old stage's Points and makes new ones */
  stageKey: string;
  ndm: 2 | 3;
  /** (n * ndm,) row-major */
  positions: Float64Array;
  /** (n,) the colour component, or null for one plain colour */
  scalar: Float64Array | null;
  /** the colour range (finite ends), or null: every point is no value when `scalar` is set */
  range: { min: number; max: number } | null;
  cmap: ContourCmap;
  /** the grid's wire box, or null when it is not drawn */
  grid: { origin: number[]; cellSize: number[]; numCells: number[] } | null;
  /** fit the camera to these particles (the first frame of a set that has no mesh) */
  fit: boolean;
}

/** How the layer looks, from the theme: the plain colour, the rim, the grid line colour. */
export interface ParticleLook {
  plain: number;
  rim: number;
  grid: number;
}

/** Point size in pixels by count: big enough to read a few points, small enough not to merge a million. */
export function particleSize(n: number): number {
  return n <= 100 ? 16 : n <= 1e3 ? 10 : n <= 1e4 ? 6 : n <= 1e5 ? 4 : n <= 5e5 ? 3 : 2;
}

/** `positions` (n * ndm) into xyz (n * 3) at `out`, z = 0 in 2-D. */
export function fillPositions(positions: Float64Array, ndm: 2 | 3, out: Float32Array): number {
  const n = positions.length / ndm;
  if (!Number.isInteger(n)) throw new RangeError(`particle positions: ${positions.length} values are not whole ${ndm}-D rows`);
  if (out.length < n * 3) throw new RangeError(`particle positions: ${n} particles do not fit a buffer of ${out.length / 3}`);
  for (let i = 0; i < n; i++) {
    out[3 * i] = positions[ndm * i]!;
    out[3 * i + 1] = positions[ndm * i + 1]!;
    out[3 * i + 2] = ndm === 3 ? positions[3 * i + 2]! : 0;
  }
  return n;
}

/**
 * Linear RGB per particle into `out` (n * 3): the ramp colour of `scalar[i]` over
 * `range`, the no-value colour for a NaN (or every value, with no finite range), or
 * `plain` (linear RGB) for every particle when `scalar` is null. Returns the number of
 * particles with no value.
 */
export function fillColours(n: number, scalar: Float64Array | null, range: { min: number; max: number } | null, cmap: ContourCmap, plain: readonly [number, number, number], out: Float32Array): number {
  const none = [toLinear(NO_DATA_RGB[0]), toLinear(NO_DATA_RGB[1]), toLinear(NO_DATA_RGB[2])];
  if (scalar !== null && scalar.length !== n) throw new RangeError(`particle colours: ${scalar.length} values for ${n} particles`);
  let noValue = 0;
  for (let i = 0; i < n; i++) {
    let c: readonly number[];
    if (scalar === null) c = plain;
    else {
      const v = scalar[i]!;
      if (!Number.isFinite(v) || range === null) {
        c = none;
        noValue++;
      } else {
        const s = sampleCmap(cmap, rampT(v, range.min, range.max));
        c = [toLinear(s[0]), toLinear(s[1]), toLinear(s[2])];
      }
    }
    out[3 * i] = c[0]!;
    out[3 * i + 1] = c[1]!;
    out[3 * i + 2] = c[2]!;
  }
  return noValue;
}

/** The finite min and max of `values`, or null when none is finite; and how many are not finite. */
export function particleRange(values: Float64Array): { range: { min: number; max: number } | null; noValue: number } {
  let min = Infinity;
  let max = -Infinity;
  let noValue = 0;
  for (let i = 0; i < values.length; i++) {
    const v = values[i]!;
    if (!Number.isFinite(v)) {
      noValue++;
      continue;
    }
    if (v < min) min = v;
    if (v > max) max = v;
  }
  return { range: min <= max ? { min, max } : null, noValue };
}

/** Component `c` of row-major (n * nComp) values. */
export function componentOf(values: Float64Array, nComp: number, c: number): Float64Array {
  if (!Number.isInteger(c) || c < 0 || c >= nComp) throw new RangeError(`component ${c} of ${nComp}`);
  const n = values.length / nComp;
  if (!Number.isInteger(n)) throw new RangeError(`${values.length} values are not whole rows of ${nComp}`);
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) out[i] = values[i * nComp + c]!;
  return out;
}

/** The grid's outline as line segments (xyz pairs): 4 edges of a rectangle at z = 0 in 2-D, 12 of a box in 3-D. */
export function gridBoxSegments(grid: { origin: number[]; cellSize: number[]; numCells: number[] }, ndm: 2 | 3): Float32Array {
  const lo = [0, 1, 2].map((a) => (a < ndm ? grid.origin[a]! : 0));
  const hi = [0, 1, 2].map((a) => (a < ndm ? grid.origin[a]! + grid.cellSize[a]! * grid.numCells[a]! : 0));
  const corner = (bits: number): number[] => [0, 1, 2].map((a) => ((bits >> a) & 1 ? hi[a]! : lo[a]!));
  const edges: number[] = [];
  const axes = ndm === 3 ? [0, 1, 2] : [0, 1];
  const span = ndm === 3 ? 8 : 4;
  for (let b = 0; b < span; b++) for (const a of axes) if (!((b >> a) & 1)) edges.push(...corner(b), ...corner(b | (1 << a)));
  return Float32Array.from(edges);
}

/** The bounding sphere of the particles (xyz, z = 0 in 2-D), or null with none. */
export function particleBounds(positions: Float64Array, ndm: 2 | 3): { center: [number, number, number]; radius: number } | null {
  const n = positions.length / ndm;
  if (n === 0) return null;
  const lo = [Infinity, Infinity, ndm === 3 ? Infinity : 0];
  const hi = [-Infinity, -Infinity, ndm === 3 ? -Infinity : 0];
  for (let i = 0; i < n; i++) {
    for (let a = 0; a < ndm; a++) {
      const v = positions[ndm * i + a]!;
      if (v < lo[a]!) lo[a] = v;
      if (v > hi[a]!) hi[a] = v;
    }
  }
  const center: [number, number, number] = [(lo[0]! + hi[0]!) / 2, (lo[1]! + hi[1]!) / 2, (lo[2]! + hi[2]!) / 2];
  const radius = Math.max(1e-9, Math.hypot(hi[0]! - lo[0]!, hi[1]! - lo[1]!, hi[2]! - lo[2]!) / 2);
  return { center, radius };
}

/** The particles of one stage, as one `THREE.Points` whose buffers are refilled per output. */
export class ParticleLayer {
  readonly root = new THREE.Group();
  private points: THREE.Points | null = null;
  private stageKey: string | null = null;
  private capacity = 0;
  private box: THREE.LineSegments | null = null;
  private boxKey: string | null = null;

  constructor() {
    this.root.name = "particles";
  }

  /** The Points drawn now (tests, measure). */
  get drawn(): THREE.Points | null {
    return this.points;
  }

  /** How many particles are drawn now. */
  get count(): number {
    return this.points ? this.points.geometry.drawRange.count : 0;
  }

  /** Draw `frame`, or nothing (null disposes the layer's geometry and material). */
  show(frame: ParticleFrame | null, look: ParticleLook): void {
    if (frame === null) {
      this.disposePoints();
      this.disposeBox();
      return;
    }
    const n = frame.positions.length / frame.ndm;
    if (frame.stageKey !== this.stageKey || n > this.capacity) {
      // Another stage (or a re-read): its buffers go before the new ones are made; a larger output grows them.
      this.disposePoints();
      const cap = Math.max(n, 1);
      const pts = makeNodePoints(new Float32Array(cap * 3), particleSize(n), 0xffffff, look.rim);
      pts.geometry.setAttribute("color", new THREE.BufferAttribute(new Float32Array(cap * 3), 3));
      const m = pts.material as THREE.PointsMaterial;
      m.vertexColors = true;
      m.needsUpdate = true;
      pts.name = "particles";
      // The points span what the file says; a scrub moves them, so the bounds are not cached for culling.
      pts.frustumCulled = false;
      this.points = pts;
      this.capacity = cap;
      this.stageKey = frame.stageKey;
      this.root.add(pts);
    }
    const g = this.points!.geometry;
    const pos = g.getAttribute("position") as THREE.BufferAttribute;
    const col = g.getAttribute("color") as THREE.BufferAttribute;
    fillPositions(frame.positions, frame.ndm, pos.array as Float32Array);
    const plain = new THREE.Color(look.plain);
    fillColours(n, frame.scalar, frame.range, frame.cmap, [plain.r, plain.g, plain.b], col.array as Float32Array);
    pos.needsUpdate = true;
    col.needsUpdate = true;
    g.setDrawRange(0, n);
    setNodeSize(this.points!, particleSize(n));
    this.showBox(frame, look);
  }

  private showBox(frame: ParticleFrame, look: ParticleLook): void {
    if (frame.grid === null) {
      this.disposeBox();
      return;
    }
    const key = `${frame.stageKey}|${look.grid}`;
    if (this.box && this.boxKey === key) return;
    this.disposeBox();
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(gridBoxSegments(frame.grid, frame.ndm), 3));
    this.box = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: look.grid }));
    this.box.name = "particle grid";
    this.box.raycast = () => {};
    this.boxKey = key;
    this.root.add(this.box);
  }

  private disposePoints(): void {
    if (!this.points) return;
    this.root.remove(this.points);
    this.points.geometry.dispose();
    (this.points.material as THREE.Material).dispose();
    this.points = null;
    this.stageKey = null;
    this.capacity = 0;
  }

  private disposeBox(): void {
    if (!this.box) return;
    this.root.remove(this.box);
    this.box.geometry.dispose();
    (this.box.material as THREE.Material).dispose();
    this.box = null;
    this.boxKey = null;
  }

  dispose(): void {
    this.disposePoints();
    this.disposeBox();
  }
}
