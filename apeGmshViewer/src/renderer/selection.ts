// Selection emphasis (R1 follow-up on #1308): the highlight objects of a
// selection, a screen-space halo of fixed pixel width drawn on top of
// everything, a short pulse when the selection changes, and the bounds `F`
// frames.
//
// The halo is a post pass: the highlight objects (they sit on HALO_LAYER as
// well as the default layer) are rendered alone into a mask target, then a
// full-screen quad paints every pixel that is outside the mask but within
// HALO_PX CSS pixels of it (HALO_PX times the device pixel ratio in device
// pixels), so it reads the same for a beam, a shell or a solid at any zoom
// and on any display. The mask exists only while something is selected. The
// pure parts (the selected primitives, highlight objects, pulse envelope,
// selection bounds, fit distance) are tested in test/selection.test.ts.

import * as THREE from "three";
import type { BlobStore } from "../state/blobs.ts";
import { selectedElementsOf } from "../state/selectors.ts";
import type { State } from "../state/types.ts";
import { DARK, themeTokens, toHexNumber, type ThemeMode } from "../theme/tokens.ts";

/** The layer the highlight objects also live on; the mask pass renders it alone. */
export const HALO_LAYER = 1;
/** Halo width in CSS pixels (times the device pixel ratio on screen). */
export const HALO_PX = 3;
/** The halo is white; the core (outline and fill) is the office accent (#1308 round 3, item 3). */
export const HALO_COLOUR = toHexNumber(DARK.halo);
export const SELECTION_COLOUR = toHexNumber(DARK.selection);
/** The pulse: HALO_PX swells to PULSE_SCALE x HALO_PX and back over PULSE_MS. */
export const PULSE_MS = 650;
export const PULSE_SCALE = 3;

/** Halo width multiplier `t` ms after a select: 1 at rest, PULSE_SCALE at the middle of the pulse. */
export function pulseAt(t: number, duration = PULSE_MS, scale = PULSE_SCALE): number {
  if (!(t >= 0) || t >= duration) return 1;
  return 1 + (scale - 1) * Math.sin((Math.PI * t) / duration);
}

/** The drawn primitives of one element: segment and triangle indices. */
export interface Prims {
  lines: readonly number[];
  tris: readonly number[];
}

/**
 * ADR 0138 D8: what of a selection is drawn, as indices over the drawn
 * geometry. `selected[e]` is 1 for a selected element (`e` indexes
 * `MeshInfo.elements`). `triIndex` holds 3 vertex indices per selected drawn
 * triangle, into the faces' own position buffer (drawn order), so a highlight
 * over that buffer moves with the deformed shape. `segments` are the original
 * indices of the selected drawn segments. A hidden element is not drawn, so it
 * is not highlighted (ADR 0138 D4).
 */
export function selectedPrims(
  drawn: { lineMap: Int32Array; triMap: Int32Array },
  lineElement: Int32Array,
  triElement: Int32Array,
  selected: Uint8Array,
): { triIndex: Uint32Array; segments: Int32Array } {
  let nTri = 0;
  for (let d = 0; d < drawn.triMap.length; d++) if (selected[triElement[drawn.triMap[d]!]!]) nTri++;
  const triIndex = new Uint32Array(3 * nTri);
  for (let d = 0, k = 0; d < drawn.triMap.length; d++) {
    if (!selected[triElement[drawn.triMap[d]!]!]) continue;
    triIndex[k++] = 3 * d;
    triIndex[k++] = 3 * d + 1;
    triIndex[k++] = 3 * d + 2;
  }
  const segments: number[] = [];
  for (const seg of drawn.lineMap) if (selected[lineElement[seg]!]) segments.push(seg);
  return { triIndex, segments: Int32Array.from(segments) };
}

/**
 * Each selected element's outline (the look of #1308): the edges of its drawn
 * triangles that only one of them has, 6 floats per edge, from the drawn
 * positions `triPos` (9 floats per drawn triangle, drawn order). Edges are
 * keyed by their node-tag pair within one element at a time, so the cost is
 * one small map per element, not one map of coordinate strings for the whole
 * selection (ADR 0138 D8).
 */
export function elementOutlines(
  triMap: Int32Array,
  triElement: Int32Array,
  selected: Uint8Array,
  triNodes: Float64Array,
  triPos: Float32Array,
): Float32Array {
  const out: number[] = [];
  const edges = new Map<number | string, { n: number; d: number; k: number }>();
  const flush = () => {
    for (const e of edges.values()) {
      if (e.n !== 1) continue;
      const a = 9 * e.d + 3 * e.k, b = 9 * e.d + 3 * ((e.k + 1) % 3);
      out.push(triPos[a]!, triPos[a + 1]!, triPos[a + 2]!, triPos[b]!, triPos[b + 1]!, triPos[b + 2]!);
    }
    edges.clear();
  };
  let current = -1;
  for (let d = 0; d < triMap.length; d++) {
    const t = triMap[d]!, el = triElement[t]!;
    if (!selected[el]) continue;
    if (el !== current) {
      flush();
      current = el;
    }
    for (let k = 0; k < 3; k++) {
      const p = triNodes[3 * t + k]!, q = triNodes[3 * t + ((k + 1) % 3)]!;
      const lo = Math.min(p, q), hi = Math.max(p, q);
      const key = Number.isInteger(lo) && lo >= 0 && hi < 2 ** 26 ? lo * 2 ** 26 + hi : `${lo}|${hi}`;
      const e = edges.get(key);
      if (e) e.n++;
      else edges.set(key, { n: 1, d, k });
    }
  }
  flush();
  return new Float32Array(out);
}

/**
 * The drawn position of each selected node: the first drawn vertex carrying
 * its tag (2 tags per segment and 3 per triangle in `vertexNodes`), 3 floats
 * per node found, in the order found. A node that no drawn primitive carries
 * (hidden, or interior) is not drawn.
 */
export function nodeVertexPositions(
  tags: readonly number[],
  vertexNodes: { line: Float64Array; tri: Float64Array },
  drawn: { lineMap: Int32Array; triMap: Int32Array },
  pos: { line: Float32Array; tri: Float32Array },
): Float32Array {
  const want = new Set(tags);
  const out: number[] = [];
  const take = (id: number, src: Float32Array, o: number) => {
    if (!want.delete(id)) return;
    out.push(src[o]!, src[o + 1]!, src[o + 2]!);
  };
  for (const seg of drawn.lineMap) for (let k = 0; k < 2 && want.size; k++) take(vertexNodes.line[2 * seg + k]!, pos.line, 6 * seg + 3 * k);
  for (const t of drawn.triMap) for (let k = 0; k < 3 && want.size; k++) take(vertexNodes.tri[3 * t + k]!, pos.tri, 9 * t + 3 * k);
  return new Float32Array(out);
}

/**
 * The node a click picks (ADR 0138 D2): of the hit primitive's vertices, the
 * one nearest the hit point, where it is drawn (so the deformed shape counts).
 * `prim` is the hit's original segment or triangle; a frame face's hit is its
 * segment. Null when the vertices carry no node tag.
 */
export function nearestNodeOf(
  prim: { kind: "line" | "tri"; index: number },
  point: readonly [number, number, number],
  vertexNodes: { line: Float64Array; tri: Float64Array },
  pos: { line: Float32Array; tri: Float32Array },
): number | null {
  const n = prim.kind === "line" ? 2 : 3, tags = prim.kind === "line" ? vertexNodes.line : vertexNodes.tri, src = prim.kind === "line" ? pos.line : pos.tri;
  let best: number | null = null, bestD = Infinity;
  for (let k = 0; k < n; k++) {
    const id = tags[n * prim.index + k];
    if (id === undefined || !Number.isFinite(id)) continue;
    const o = 3 * n * prim.index + 3 * k;
    const d = Math.hypot(src[o]! - point[0], src[o + 1]! - point[1], src[o + 2]! - point[2]);
    if (d < bestD) [best, bestD] = [id, d];
  }
  return best;
}

/**
 * The objects that draw a selection over the model (ADR 0138 D8): the selected
 * triangles filled, as an index over the faces' own position attribute; the
 * selected segments as thick lines and each selected element's outline as
 * thinner ones; the selected nodes as points. With `quiet` (another kind is
 * active, ADR 0138 D2) the element kind is drawn faint and left off the halo. All without
 * depth test, and every one on HALO_LAYER as well as the default layer, so the
 * halo pass rings the whole selection.
 */
export function highlightObjects(
  facePositions: THREE.BufferAttribute | null,
  triIndex: Uint32Array,
  segmentPositions: Float32Array,
  outlinePositions: Float32Array,
  nodePositions: Float32Array,
  makeLines: (positions: Float32Array, widthPx: number) => THREE.Object3D,
  makePoints: (positions: Float32Array) => THREE.Object3D,
  quiet = false,
): THREE.Object3D[] {
  const out: THREE.Object3D[] = [];
  /** objects of the element kind; with `quiet` they stay faint and off the halo, so another kind's markers read over them */
  const elementParts: THREE.Object3D[] = [];
  const noDepth = (o: THREE.Object3D, order: number) => {
    o.renderOrder = order;
    const m = (o as THREE.Mesh).material as THREE.Material | undefined;
    if (m) m.depthTest = false;
    out.push(o);
    return o;
  };
  if (segmentPositions.length) elementParts.push(noDepth(makeLines(segmentPositions, quiet ? 4 : 9), 10));
  if (outlinePositions.length && !quiet) elementParts.push(noDepth(makeLines(outlinePositions, 6), 10));
  if (facePositions && triIndex.length) {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", facePositions);
    g.setIndex(new THREE.BufferAttribute(triIndex, 1));
    const fill = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ color: SELECTION_COLOUR, side: THREE.DoubleSide, transparent: true, opacity: quiet ? 0.2 : 0.6 }));
    fill.frustumCulled = false;
    elementParts.push(noDepth(fill, 9));
  }
  if (nodePositions.length) noDepth(makePoints(nodePositions), 11);
  for (const o of out) if (!(quiet && elementParts.includes(o))) o.layers.enable(HALO_LAYER);
  return out;
}

/** What a hit names: the element, the point, and the primitive (original segment, triangle, or frame-face triangle index). */
export interface Hit {
  element: number;
  point: THREE.Vector3;
  prim: { kind: "line" | "tri" | "frame"; index: number };
}

/**
 * The element under the raycaster: the drawn lines first (lines win over
 * faces), then the faces; the hit is wherever those objects are drawn now,
 * so on a deformed mesh it is the deformed position (V4d; renderer/deform.ts
 * writes the positions and the bounds the raycaster tests). Among coincident
 * line hits an OpenSees-only element (no cell) wins, since it would be
 * unreachable behind a cell drawn on the same nodes. Null on a miss.
 *
 * V4e1 (ADR 0083 C-PICK): `clipped` says whether a world point is cut away
 * by a section plane. Each object's nearest hit counts only where the cut
 * kept it: a cut-away line hit leaves the faces to answer, and a cut-away
 * face hit is a miss. Hits behind a cut-away one are never walked to (no
 * pick-through the hole, ADR 0083 open question 3).
 *
 * ADR 0138 D5: with `only`, the element is already chosen (by the click's ID
 * image, renderer/picking.ts `ClickPicker`), and this returns where the ray
 * meets that element: its nearest kept hit, on lines, faces or frame faces.
 * The ray cast alone is not used to choose: it tries lines before faces along
 * the whole ray and takes the line hit nearest the eye within its pixel
 * tolerance, so it picked beams hidden behind a slab and the neighbour of a
 * short segment (internal_docs/spike_viewer_selection.md, "Clicks").
 */
export function hitElement(
  raycaster: THREE.Raycaster,
  lines: THREE.Object3D | null,
  faces: THREE.Object3D | null,
  drawn: { lineMap: Int32Array; triMap: Int32Array },
  elements: readonly string[],
  lineElement: Int32Array,
  triElement: Int32Array,
  radius: number,
  clipped: (p: THREE.Vector3) => boolean = () => false,
  frameFaces: THREE.Object3D | null = null,
  frameTriMap: Int32Array | null = null,
  frameTriElement: Int32Array | null = null,
  only: number | null = null,
): { element: number; point: THREE.Vector3; prim: Hit["prim"] } | null {
  if (only !== null) return hitOn(only, raycaster, lines, faces, drawn, lineElement, triElement, clipped, frameFaces, frameTriMap, frameTriElement);
  if (lines && lines.visible) {
    const hits = raycaster.intersectObject(lines, false).filter((h) => h.faceIndex !== undefined && h.faceIndex !== null);
    if (hits.length && !clipped(hits[0]!.point)) {
      const near = hits[0]!.distance + 1e-6 * radius;
      const coincident = hits.filter((h) => h.distance <= near);
      const ops = coincident.find((h) => elements[lineElement[drawn.lineMap[h.faceIndex!]!]!]!.startsWith("opensees/element/"));
      const h = ops ?? coincident[0]!;
      const seg = drawn.lineMap[h.faceIndex!]!;
      return { element: lineElement[seg]!, point: h.point, prim: { kind: "line", index: seg } };
    }
  }
  let bestFaceHit: { element: number; point: THREE.Vector3; distance: number; prim: Hit["prim"] } | null = null;
  if (faces && faces.visible) {
    const hit = raycaster.intersectObject(faces, false)[0];
    if (hit && hit.faceIndex !== undefined && hit.faceIndex !== null && !clipped(hit.point)) {
      const t = drawn.triMap[hit.faceIndex]!;
      bestFaceHit = { element: triElement[t]!, point: hit.point, distance: hit.distance, prim: { kind: "tri", index: t } };
    }
  }
  if (frameFaces && frameFaces.visible && frameTriMap && frameTriElement) {
    const hit = raycaster.intersectObject(frameFaces, false)[0];
    if (hit && hit.faceIndex !== undefined && hit.faceIndex !== null && !clipped(hit.point)) {
      if (!bestFaceHit || hit.distance < bestFaceHit.distance) {
        const f = frameTriMap[hit.faceIndex]!;
        bestFaceHit = { element: frameTriElement[f]!, point: hit.point, distance: hit.distance, prim: { kind: "frame", index: f } };
      }
    }
  }
  if (bestFaceHit) return { element: bestFaceHit.element, point: bestFaceHit.point, prim: bestFaceHit.prim };
  return null;
}

/** `hitElement` with the element chosen: the nearest kept hit on one of its lines, else on one of its faces or frame faces. */
function hitOn(
  only: number,
  raycaster: THREE.Raycaster,
  lines: THREE.Object3D | null,
  faces: THREE.Object3D | null,
  drawn: { lineMap: Int32Array; triMap: Int32Array },
  lineElement: Int32Array,
  triElement: Int32Array,
  clipped: (p: THREE.Vector3) => boolean,
  frameFaces: THREE.Object3D | null,
  frameTriMap: Int32Array | null,
  frameTriElement: Int32Array | null,
): Hit | null {
  const first = (o: THREE.Object3D | null, map: Int32Array | null, element: Int32Array | null) =>
    o && o.visible && map && element
      ? raycaster.intersectObject(o, false).find((h) => h.faceIndex !== undefined && h.faceIndex !== null && element[map[h.faceIndex]!] === only && !clipped(h.point))
      : undefined;
  const l = first(lines, drawn.lineMap, lineElement);
  if (l) return { element: only, point: l.point, prim: { kind: "line", index: drawn.lineMap[l.faceIndex!]! } };
  const t = first(faces, drawn.triMap, triElement), f = first(frameFaces, frameTriMap, frameTriElement);
  if (t && (!f || t.distance <= f.distance)) return { element: only, point: t.point, prim: { kind: "tri", index: drawn.triMap[t.faceIndex!]! } };
  if (f) return { element: only, point: f.point, prim: { kind: "frame", index: frameTriMap![f.faceIndex!]! } };
  return null;
}

export interface Focus {
  center: readonly [number, number, number];
  radius: number;
}

/**
 * The bounding sphere of the selected primitives (6 floats per segment, 9
 * per triangle): the box centre and half its diagonal, never smaller than
 * `minRadius`. Null when nothing is drawn for the selection.
 */
export function selectionBounds(
  linePositions: Float32Array,
  triPositions: Float32Array,
  lines: readonly number[],
  tris: readonly number[],
  minRadius: number,
): Focus | null {
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  let any = false;
  const take = (a: Float32Array, o: number) => {
    any = true;
    for (let k = 0; k < 3; k++) {
      lo[k] = Math.min(lo[k]!, a[o + k]!);
      hi[k] = Math.max(hi[k]!, a[o + k]!);
    }
  };
  for (const i of lines) for (let v = 0; v < 2; v++) take(linePositions, 6 * i + 3 * v);
  for (const i of tris) for (let v = 0; v < 3; v++) take(triPositions, 9 * i + 3 * v);
  if (!any) return null;
  const center: [number, number, number] = [0, 1, 2].map((k) => (lo[k]! + hi[k]!) / 2) as [number, number, number];
  const radius = Math.max(minRadius, Math.hypot(hi[0]! - lo[0]!, hi[1]! - lo[1]!, hi[2]! - lo[2]!) / 2);
  return { center, radius };
}

/**
 * What `frameSelection` frames: the bounds of the selected elements' drawn
 * primitives, read from the state's mesh blobs; null when nothing is selected
 * (or nothing of it is drawn), which means the whole model.
 */
export function focusOf(s: State, blobs: BlobStore): Focus | null {
  const info = s.mesh;
  const sel = selectedElementsOf(s);
  if (!info || sel.length === 0) return null;
  const selSet = new Set<string>(sel);
  const selected = new Set<number>();
  info.elements.forEach((p, i) => {
    if (selSet.has(p)) selected.add(i);
  });
  if (selected.size === 0) return null;
  const lines: number[] = [], tris: number[] = [];
  blobs.i32(info.lineElement).forEach((e, i) => {
    if (selected.has(e)) lines.push(i);
  });
  blobs.i32(info.triElement).forEach((e, i) => {
    if (selected.has(e)) tris.push(i);
  });
  return selectionBounds(blobs.f32(info.linePositions), blobs.f32(info.triPositions), lines, tris, 1e-3 * info.radius);
}

/**
 * The camera distance that fits a sphere of `radius` in a view of vertical
 * field `fovDeg` whose free area is `freePx` wide and `heightPx` tall: the
 * sphere touches the narrower half-angle, with 8 % of margin.
 */
export function fitDistance(radius: number, fovDeg: number, freePx: number, heightPx: number): number {
  const half = THREE.MathUtils.degToRad(fovDeg / 2);
  const halfH = Math.atan(Math.tan(half) * (Math.max(1, freePx) / Math.max(1, heightPx)));
  return (radius / Math.sin(Math.min(half, halfH))) * 1.08;
}

/** Bytes a mask target of a canvas takes (RGBA8, one texel per device pixel). */
export function maskBytes(widthPx: number, heightPx: number, pixelRatio: number): number {
  return Math.round(widthPx * pixelRatio) * Math.round(heightPx * pixelRatio) * 4;
}

const VERTEX = /* glsl */ `
  varying vec2 vUv;
  void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }
`;

const FRAGMENT = /* glsl */ `
  uniform sampler2D tMask;
  uniform vec2 texel;
  uniform float radius;
  uniform vec3 colour;
  varying vec2 vUv;
  void main() {
    if (texture2D(tMask, vUv).a > 0.5) discard;
    float hit = 0.0;
    for (int i = 0; i < 16; i++) {
      float a = float(i) * 0.39269908;
      vec2 d = vec2(cos(a), sin(a)) * texel;
      hit = max(hit, texture2D(tMask, vUv + d * radius).a);
      hit = max(hit, texture2D(tMask, vUv + d * radius * 0.5).a);
    }
    if (hit < 0.5) discard;
    gl_FragColor = vec4(colour, 1.0);
  }
`;

export class SelectionHalo {
  private readonly renderer: THREE.WebGLRenderer;
  /** allocated on the first render with a selection, released on `release()` */
  private mask: THREE.WebGLRenderTarget | null = null;
  private readonly quad: THREE.Mesh<THREE.PlaneGeometry, THREE.ShaderMaterial>;
  private readonly pass = new THREE.Scene();
  private readonly ortho = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  private pulseStart: number | null = null;
  private width = 1;
  private height = 1;
  private pixelRatio = 1;

  constructor(renderer: THREE.WebGLRenderer) {
    this.renderer = renderer;
    this.quad = new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.ShaderMaterial({
        uniforms: {
          tMask: { value: null },
          texel: { value: new THREE.Vector2(1, 1) },
          radius: { value: HALO_PX },
          colour: { value: new THREE.Color(HALO_COLOUR) },
        },
        vertexShader: VERTEX,
        fragmentShader: FRAGMENT,
        transparent: true,
        depthTest: false,
        depthWrite: false,
      }),
    );
    this.pass.add(this.quad);
  }

  /** Update the halo ring colour for the active theme. */
  setTheme(mode: ThemeMode): void {
    (this.quad.material.uniforms["colour"]!.value as THREE.Color).setHex(toHexNumber(themeTokens(mode).halo));
  }

  /** The canvas size in CSS pixels and the device pixel ratio; the mask follows on its next use. */
  resize(width: number, height: number, pixelRatio: number): void {
    this.width = width;
    this.height = height;
    this.pixelRatio = pixelRatio;
    if (this.mask) this.fitMask();
  }

  /** The halo radius in device pixels at rest: HALO_PX CSS pixels. */
  get radiusPx(): number {
    return HALO_PX * this.pixelRatio;
  }

  /** Whether the mask target is allocated (for tests and the memory note). */
  get allocated(): boolean {
    return this.mask !== null;
  }

  private fitMask(): void {
    const w = Math.max(1, Math.round(this.width * this.pixelRatio)), h = Math.max(1, Math.round(this.height * this.pixelRatio));
    if (!this.mask) {
      this.mask = new THREE.WebGLRenderTarget(w, h, { depthBuffer: false, stencilBuffer: false });
      this.quad.material.uniforms["tMask"]!.value = this.mask.texture;
    } else if (this.mask.width !== w || this.mask.height !== h) this.mask.setSize(w, h);
    (this.quad.material.uniforms["texel"]!.value as THREE.Vector2).set(1 / w, 1 / h);
  }

  /** Free the mask target (nothing selected). */
  release(): void {
    if (!this.mask) return;
    this.mask.dispose();
    this.mask = null;
    this.quad.material.uniforms["tMask"]!.value = null;
    this.pulseStart = null;
  }

  /** Start the pulse (on a selection change). */
  pulse(): void {
    this.pulseStart = performance.now();
  }

  /** Whether the pulse is still running, so the viewport keeps rendering. */
  get pulsing(): boolean {
    return this.pulseStart !== null && performance.now() - this.pulseStart < PULSE_MS;
  }

  /**
   * Draw the halo over the frame already rendered: the HALO_LAYER objects of
   * `scene` into the mask, then the ring pass onto the screen.
   */
  render(scene: THREE.Scene, camera: THREE.Camera): void {
    this.fitMask();
    const t = this.pulseStart === null ? Infinity : performance.now() - this.pulseStart;
    this.quad.material.uniforms["radius"]!.value = this.radiusPx * pulseAt(t);
    if (t >= PULSE_MS) this.pulseStart = null;
    const r = this.renderer;
    const autoClear = r.autoClear, mask = camera.layers.mask;
    camera.layers.set(HALO_LAYER);
    r.setRenderTarget(this.mask);
    // The mask is cleared to alpha 0; its colour is never read.
    r.setClearColor(0, 0);
    r.clear();
    r.render(scene, camera);
    r.setRenderTarget(null);
    camera.layers.mask = mask;
    r.autoClear = false;
    r.render(this.pass, this.ortho);
    r.autoClear = autoClear;
  }

  dispose(): void {
    this.release();
    this.quad.geometry.dispose();
    this.quad.material.dispose();
  }
}
