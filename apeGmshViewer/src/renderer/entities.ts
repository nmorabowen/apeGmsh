// ADR 0130 D8 (VU8): the `/entities` of the set's `.user.h5`, drawn where they
// are: a section cut as its plane, a storey as its outline at its level, a
// point set as its points. The geometry is `entityDrawing`, a pure function of
// the zone and the model's nodes (tested in Node); `EntitiesLayer` turns it
// into three.js objects.
//
// - A cut with a bounding polygon draws that polygon; one without draws its
//   plane where it crosses the model's box. A short tick from the plane
//   points to the kept side (`positive` along the normal), and a marker sits
//   at the reference its moments are about.
// - A storey draws the outline of its nodes in plan (their convex hull) at
//   its level, faintly filled so it reads apart from the floor's beams, with
//   a marker at its reference point (the centre of mass).
//   A storey naming a node the model does not hold is not drawn, and says why.
// - A point set draws a marker at each point.
//
// Drawn only while the zone reads and is current: A6 draws stale content only
// on an explicit request, and no request exists yet. Refused rows are never
// drawn (the reducer keeps only standing rows).

import * as THREE from "three";
import type { BlobStore } from "../state/blobs.ts";
import type { State, UserCut, UserInfo, UserPointSet, UserStorey, Vec3 } from "../state/types.ts";
import { themeTokens, toHexNumber, type ThemeMode } from "../theme/tokens.ts";
import { makeNodePoints, setNodeSize } from "./nodes.ts";

/** The zone as the reducer keeps it, when it reads. */
export type EntitiesIn = Extract<NonNullable<UserInfo["entities"]>, { ok: true }>;

/** An axis-aligned box: `[min, max]`. */
export type Box = readonly [Vec3, Vec3];

/** What the layer draws, as flat `xyz` arrays. */
export interface EntityDrawing {
  /** cut outlines and side ticks, 6 floats per segment */
  cutLines: number[];
  /** cut faces, 9 floats per triangle */
  cutTris: number[];
  /** storey outlines, 6 floats per segment */
  storeyLines: number[];
  /** storey floors (the outline filled), 9 floats per triangle */
  storeyTris: number[];
  /** the point each cut's moments are about, 3 floats each */
  cutReferences: number[];
  /** each storey's reference point, 3 floats each */
  storeyReferences: number[];
  /** point-set points, 3 floats each */
  points: number[];
  /** a row that could not be drawn, and why */
  notes: string[];
}

const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a: Vec3, b: Vec3): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const mul = (a: Vec3, k: number): Vec3 => [a[0] * k, a[1] * k, a[2] * k];
const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: Vec3, b: Vec3): Vec3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const len = (a: Vec3) => Math.sqrt(dot(a, a));
const unit = (a: Vec3): Vec3 => mul(a, 1 / len(a));

/** The box of the model's nodes (`xyz`, 3 per node), grown by `pad` of its diagonal on every side. */
export function boxOf(xyz: ArrayLike<number>, pad = 0.05): Box | null {
  if (xyz.length < 3) return null;
  const lo: [number, number, number] = [Infinity, Infinity, Infinity];
  const hi: [number, number, number] = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i + 2 < xyz.length; i += 3) {
    for (let k = 0; k < 3; k++) {
      lo[k] = Math.min(lo[k]!, xyz[i + k]!);
      hi[k] = Math.max(hi[k]!, xyz[i + k]!);
    }
  }
  const d = Math.max(len(sub(hi, lo)), 1e-9) * pad;
  return [sub(lo, [d, d, d]), add(hi, [d, d, d])];
}

/** Where the plane `(point, normal)` crosses `box`: a convex polygon in order, or [] when it misses. */
export function planeInBox(point: Vec3, normal: Vec3, box: Box): Vec3[] {
  const [lo, hi] = box;
  const corner = (i: number): Vec3 => [i & 1 ? hi[0] : lo[0], i & 2 ? hi[1] : lo[1], i & 4 ? hi[2] : lo[2]];
  const edges: [number, number][] = [];
  for (let i = 0; i < 8; i++) for (const bit of [1, 2, 4]) if (!(i & bit)) edges.push([i, i | bit]);
  const tol = 1e-12 * Math.max(len(sub(hi, lo)), 1);
  const out: Vec3[] = [];
  for (const [i, j] of edges) {
    const a = corner(i);
    const b = corner(j);
    const da = dot(sub(a, point), normal);
    const db = dot(sub(b, point), normal);
    if ((da > tol && db > tol) || (da < -tol && db < -tol) || da === db) continue;
    const p = add(a, mul(sub(b, a), da / (da - db)));
    if (!out.some((q) => len(sub(p, q)) <= tol)) out.push(p);
  }
  if (out.length < 3) return [];
  return ordered(out, normal);
}

/** `pts`, which lie in one plane, ordered around their centroid by angle about `normal`. */
function ordered(pts: Vec3[], normal: Vec3): Vec3[] {
  const c = mul(pts.reduce(add, [0, 0, 0]), 1 / pts.length);
  const u = unit(sub(pts[0]!, c));
  const v = cross(unit(normal), u);
  return [...pts].sort((p, q) => Math.atan2(dot(sub(p, c), v), dot(sub(p, c), u)) - Math.atan2(dot(sub(q, c), v), dot(sub(q, c), u)));
}

/** The convex hull of `pts` in plan (x, y), counter-clockwise (Andrew's monotone chain); collinear points dropped. */
export function planHull(pts: readonly (readonly [number, number])[]): [number, number][] {
  const p = [...new Map(pts.map((q) => [`${q[0]},${q[1]}`, [q[0], q[1]] as [number, number]])).values()].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  if (p.length < 3) return p;
  const turn = (o: [number, number], a: [number, number], b: [number, number]) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower: [number, number][] = [];
  for (const q of p) {
    while (lower.length >= 2 && turn(lower.at(-2)!, lower.at(-1)!, q) <= 0) lower.pop();
    lower.push(q);
  }
  const upper: [number, number][] = [];
  for (const q of [...p].reverse()) {
    while (upper.length >= 2 && turn(upper.at(-2)!, upper.at(-1)!, q) <= 0) upper.pop();
    upper.push(q);
  }
  return [...lower.slice(0, -1), ...upper.slice(0, -1)];
}

/** A closed loop as segments. */
function loop(out: number[], ring: readonly Vec3[]): void {
  ring.forEach((a, i) => out.push(...a, ...ring[(i + 1) % ring.length]!));
}

function drawCut(d: EntityDrawing, c: UserCut, box: Box): void {
  const ring = c.polygon.length >= 3 ? [...c.polygon] : planeInBox(c.point, c.normal, box);
  if (ring.length < 3) {
    d.notes.push(`cut ${JSON.stringify(c.name)}: its plane misses the model, so it is not drawn`);
    return;
  }
  loop(d.cutLines, ring);
  for (let i = 1; i + 1 < ring.length; i++) d.cutTris.push(...ring[0]!, ...ring[i]!, ...ring[i + 1]!);
  // The kept side: a tick from the polygon's centroid, a tenth of the box diagonal long.
  const at = mul(ring.reduce(add, [0, 0, 0]), 1 / ring.length);
  const tick = 0.1 * len(sub(box[1], box[0]));
  d.cutLines.push(...at, ...add(at, mul(c.normal, c.side === "positive" ? tick : -tick)));
  d.cutReferences.push(...c.reference);
}

function drawStorey(d: EntityDrawing, s: UserStorey, nodes: ReadonlyMap<number, Vec3> | null): void {
  if (nodes === null) {
    d.notes.push(`storey ${JSON.stringify(s.name)}: the model's nodes are not read, so it is not drawn`);
    return;
  }
  const missing = s.nodeIds.filter((n) => !nodes.has(n));
  if (missing.length) {
    d.notes.push(`storey ${JSON.stringify(s.name)} names node${missing.length > 1 ? "s" : ""} ${missing.slice(0, 5).join(", ")}${missing.length > 5 ? ", ..." : ""}, which the model does not hold, so it is not drawn`);
    return;
  }
  const hull = planHull(s.nodeIds.map((n) => [nodes.get(n)![0], nodes.get(n)![1]] as const));
  const ring: Vec3[] = hull.map(([x, y]) => [x, y, s.level]);
  if (ring.length >= 3) {
    loop(d.storeyLines, ring);
    // The hull is convex, so a fan fills it.
    for (let i = 1; i + 1 < ring.length; i++) d.storeyTris.push(...ring[0]!, ...ring[i]!, ...ring[i + 1]!);
  } else if (ring.length === 2) d.storeyLines.push(...ring[0]!, ...ring[1]!);
  d.storeyReferences.push(...s.reference);
}

function drawPointSet(d: EntityDrawing, p: UserPointSet): void {
  for (const x of p.xyz) d.points.push(...x);
}

/**
 * Everything the layer draws for `e`: `nodes` maps a node tag to its
 * coordinates (null when the model's nodes are not read), and `box` is where a
 * cut with no polygon is drawn.
 */
export function entityDrawing(e: EntitiesIn, nodes: ReadonlyMap<number, Vec3> | null, box: Box): EntityDrawing {
  const d: EntityDrawing = { cutLines: [], cutTris: [], storeyLines: [], storeyTris: [], cutReferences: [], storeyReferences: [], points: [], notes: [] };
  for (const c of e.cuts) drawCut(d, c, box);
  for (const s of e.storeys) drawStorey(d, s, nodes);
  for (const p of e.pointSets) drawPointSet(d, p);
  return d;
}

/** The model's nodes by tag, from the mesh's `nodes` blobs, or null when the mesh was built without them. */
export function nodeMap(s: State, blobs: BlobStore): Map<number, Vec3> | null {
  const n = s.mesh?.nodes;
  if (!n) return null;
  const ids = blobs.get(n.ids) as Float64Array;
  const xyz = blobs.get(n.xyz) as Float64Array;
  const out = new Map<number, Vec3>();
  for (let i = 0; i < ids.length; i++) out.set(ids[i]!, [xyz[3 * i]!, xyz[3 * i + 1]!, xyz[3 * i + 2]!]);
  return out;
}

/** The zone to draw, or null: it must read, be current, and the model must be drawn. */
export function drawnEntities(s: State): EntitiesIn | null {
  if (!s.mesh || s.phase.at?.kind === "geometry" || s.user?.status !== "ready") return null;
  const e = s.user.info.entities;
  return e && e.ok && !e.stale.stale ? e : null;
}

/** Larger than a node marker (5 px), so a point set reads apart from the nodes. */
const POINT_SIZE = 12;
const REFERENCE_SIZE = 8;

export class EntitiesLayer {
  readonly root = new THREE.Group();
  private theme: ThemeMode = "dark";
  private appliedKey: unknown[] | null = null;
  /** the rows the last drawing could not draw, and why */
  notes: string[] = [];

  constructor() {
    this.root.visible = false;
  }

  setTheme(mode: ThemeMode): void {
    this.theme = mode;
  }

  /** Draw `s`'s entities; true when what is drawn changed. */
  sync(s: State, blobs: BlobStore): boolean {
    const e = drawnEntities(s);
    if (e === null) {
      const was = this.root.visible;
      this.root.visible = false;
      this.appliedKey = null;
      return was;
    }
    const key = [e, s.mesh, this.theme];
    if (this.root.visible && this.appliedKey?.every((k, i) => k === key[i])) return false;
    const nodes = nodeMap(s, blobs);
    const box = (nodes && boxOf([...nodes.values()].flat())) ?? sphereBox(s.mesh!.center, s.mesh!.radius);
    const d = entityDrawing(e, nodes, box);
    for (const n of d.notes) if (!this.notes.includes(n)) console.warn(`apeGmshViewer /entities: ${n}`);
    this.notes = d.notes;
    this.build(d);
    this.root.visible = true;
    this.appliedKey = key;
    return true;
  }

  private build(d: EntityDrawing): void {
    this.clear();
    const t = themeTokens(this.theme);
    const cut = toHexNumber(t.ok);
    const storey = toHexNumber(t.warn);
    const lines = (xyz: number[], color: number) => {
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.Float32BufferAttribute(xyz, 3));
      return new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color, depthTest: false, transparent: true }));
    };
    const faces = (xyz: number[], color: number, opacity: number) => {
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.Float32BufferAttribute(xyz, 3));
      return new THREE.Mesh(g, new THREE.MeshBasicMaterial({ color, transparent: true, opacity, side: THREE.DoubleSide, depthWrite: false }));
    };
    if (d.cutTris.length) this.root.add(faces(d.cutTris, cut, 0.18));
    if (d.storeyTris.length) this.root.add(faces(d.storeyTris, storey, 0.14));
    if (d.cutLines.length) this.root.add(lines(d.cutLines, cut));
    if (d.storeyLines.length) this.root.add(lines(d.storeyLines, storey));
    if (d.points.length) this.root.add(this.markers(d.points, toHexNumber(t.text), POINT_SIZE));
    if (d.cutReferences.length) this.root.add(this.markers(d.cutReferences, cut, REFERENCE_SIZE));
    if (d.storeyReferences.length) this.root.add(this.markers(d.storeyReferences, storey, REFERENCE_SIZE));
    for (const o of this.root.children) o.renderOrder = 10;
  }

  private markers(xyz: number[], fill: number, size: number): THREE.Points {
    const p = makeNodePoints(Float32Array.from(xyz), size, fill, toHexNumber(themeTokens(this.theme).bg0));
    setNodeSize(p, size);
    return p;
  }

  private clear(): void {
    for (const o of [...this.root.children]) {
      this.root.remove(o);
      const obj = o as THREE.Mesh;
      obj.geometry?.dispose();
      (obj.material as THREE.Material | undefined)?.dispose();
    }
  }

  dispose(): void {
    this.clear();
  }
}

function sphereBox(c: readonly [number, number, number], r: number): Box {
  return [
    [c[0] - r, c[1] - r, c[2] - r],
    [c[0] + r, c[1] + r, c[2] + r],
  ];
}
