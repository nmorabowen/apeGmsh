// Box selection from an ID image (ADR 0138 D5, D7).
//
// At a press, the viewport draws what the active kind can select into an
// off-screen target, one 24-bit ID per pixel (0 is nothing), and reads it back
// once. The image uses the drawn positions, section planes and hidden set, so
// a box takes what the user sees. Every update of the drag then reads that
// copy in memory; nothing is drawn per frame. Elements are the faces, frame
// faces and lines; nodes are discs over the faces, which are drawn as depth
// only so a node behind a face is not visible.
//
// For elements, a box drawn left to right takes the elements fully inside it:
// those it shows that have no drawn vertex outside it (the vertices are
// projected once, at the press). Right to left takes every element it shows.
//
// A click (and the hover mark, which names what a click would take) reads the
// same kind of image (`ClickPicker`), drawn again only when the view or what
// is drawn changes. It takes the element drawn at the pixel, but a line drawn
// within CLICK_LINE_PX wins: the image draws lines 1 px wide where the screen
// draws them 3 px, and a beam must stay as easy to hit as the ray cast made
// it. Only what is drawn is in the image, so a hidden beam never wins, and the
// nearest pixel wins, so a short segment is never beaten by its neighbour.
// The pure parts (`boxOf`, the region scan, `clickId`) are tested in
// test/picking.test.ts; the GPU part is proven by the capture stills.

import * as THREE from "three";

export type PickKind = "element" | "node";

/** An ID is the low 24 bits of a pixel (RGB); the alpha byte is not part of it. 2^24 is ADR 0137 D1's limit. */
const ID_MASK = 2 ** 24 - 1;

/** The rectangle a drag covers, in canvas pixels from the top left, and whether it was drawn right to left (crossing). */
export interface Box {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  crossing: boolean;
}

/** The box between a press and the pointer now, in canvas pixels; a drag toward the left is a crossing box. */
export function boxOf(ax: number, ay: number, bx: number, by: number): Box {
  return { x0: Math.min(ax, bx), y0: Math.min(ay, by), x1: Math.max(ax, bx), y1: Math.max(ay, by), crossing: bx < ax };
}

/**
 * The IDs inside a box of an ID image (`ids` row 0 at the bottom, as WebGL
 * reads it; `w` x `h`), each once, by a stamp array that clears in O(1).
 */
export function idsInBox(ids: Uint32Array, w: number, h: number, box: Box, dpr: number, seen: Uint32Array, stamp: number, into: number[]): void {
  const x0 = Math.max(0, Math.floor(box.x0 * dpr)), x1 = Math.min(w - 1, Math.ceil(box.x1 * dpr));
  const top = Math.max(0, Math.floor(box.y0 * dpr)), bottom = Math.min(h - 1, Math.ceil(box.y1 * dpr));
  for (let row = h - 1 - bottom; row <= h - 1 - top; row++) {
    for (let i = row * w + x0, end = row * w + x1; i <= end; i++) {
      const id = ids[i]! & ID_MASK;
      if (id === 0 || seen[id] === stamp) continue;
      seen[id] = stamp;
      into.push(id - 1);
    }
  }
}

const CLIP_V = "#include <clipping_planes_pars_vertex>\n";
const CLIP_F = "#include <clipping_planes_pars_fragment>\n";

/** A material that writes each primitive's ID (texture texel `gl_VertexID / perPrim`). */
function primMaterial(ids: THREE.DataTexture, perPrim: number, behind: boolean, planes: THREE.Plane[]): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    uniforms: { ids: { value: ids }, perPrim: { value: perPrim } },
    vertexShader: `${CLIP_V}uniform highp sampler2D ids; uniform int perPrim; flat varying vec4 vId;
      void main() {
        int d = gl_VertexID / perPrim;
        vId = texelFetch(ids, ivec2(d % 4096, d / 4096), 0);
        vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
        gl_Position = projectionMatrix * mvPosition;
        #include <clipping_planes_vertex>
      }`,
    fragmentShader: `${CLIP_F}flat varying vec4 vId;
      void main() {
        #include <clipping_planes_fragment>
        gl_FragColor = vId;
      }`,
    side: THREE.DoubleSide,
    blending: THREE.NoBlending,
    clipping: true,
    clippingPlanes: planes,
    // Faces sit a little behind, so a line drawn on a face wins there, as a click does.
    polygonOffset: behind,
    polygonOffsetFactor: 1,
    polygonOffsetUnits: 1,
  });
}

/** Discs of `sizePx` whose ID is their vertex index + 1, pulled a little toward the eye so a node on a face wins. */
function pointMaterial(sizePx: number, planes: THREE.Plane[]): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    uniforms: { size: { value: sizePx } },
    vertexShader: `${CLIP_V}uniform float size; flat varying vec4 vId;
      void main() {
        int i = gl_VertexID + 1;
        vId = vec4(float(i & 255), float((i >> 8) & 255), float((i >> 16) & 255), 255.0) / 255.0;
        vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
        gl_Position = projectionMatrix * mvPosition;
        gl_Position.z -= 2e-4 * gl_Position.w;
        gl_PointSize = size;
        #include <clipping_planes_vertex>
      }`,
    fragmentShader: `${CLIP_F}flat varying vec4 vId;
      void main() {
        #include <clipping_planes_fragment>
        if (length(gl_PointCoord - 0.5) > 0.5) discard;
        gl_FragColor = vId;
      }`,
    blending: THREE.NoBlending,
    clipping: true,
    clippingPlanes: planes,
  });
}

/** IDs as an RGBA8 texture 4096 wide, one texel per primitive. */
function idTexture(ids: ArrayLike<number>): THREE.DataTexture {
  const h = Math.max(1, Math.ceil(ids.length / 4096));
  const px = new Uint8Array(4096 * h * 4);
  for (let i = 0; i < ids.length; i++) {
    const v = ids[i]!, o = 4 * i;
    px[o] = v & 255;
    px[o + 1] = (v >> 8) & 255;
    px[o + 2] = (v >> 16) & 255;
    px[o + 3] = 255;
  }
  const t = new THREE.DataTexture(px, 4096, h, THREE.RGBAFormat, THREE.UnsignedByteType);
  t.minFilter = t.magFilter = THREE.NearestFilter;
  t.generateMipmaps = false;
  t.needsUpdate = true;
  return t;
}

/** What the viewport draws now, as the picker needs it. Positions are drawn order (deformed while the deformed shape shows). */
export interface PickSource {
  /** the faces' own position buffer and the element index of each drawn triangle */
  faces: { positions: THREE.BufferAttribute; element: Int32Array } | null;
  /** extruded frame faces, as `faces` */
  frameFaces: { positions: THREE.BufferAttribute; element: Int32Array } | null;
  /** the drawn segments: 6 floats each, and their element index */
  lines: { positions: Float32Array; element: Int32Array } | null;
  /** the drawn nodes: 3 floats each, and their tags */
  nodes: { positions: Float32Array; tags: Float64Array } | null;
  /** the number of elements (`MeshInfo.elements.length`) */
  elements: number;
  planes: THREE.Plane[];
}

/** What `kind` can select, drawn as IDs: faces and frame faces (depth only for nodes), then lines or node discs. */
function pickScene(kind: PickKind, src: PickSource, pixelRatio: number, owned: { dispose(): void }[]): THREE.Scene {
  const scene = new THREE.Scene();
  const add = (o: THREE.Object3D, ...own: { dispose(): void }[]) => {
    o.frustumCulled = false;
    scene.add(o);
    owned.push(...own);
  };
  const { faces, frameFaces, lines, nodes, planes } = src;
  const facesOf = (f: NonNullable<PickSource["faces"]>) => {
    const tex = idTexture(Int32Array.from(f.element, (e) => e + 1));
    const m = primMaterial(tex, 3, true, planes);
    m.colorWrite = kind !== "node";
    const g = new THREE.BufferGeometry().setAttribute("position", f.positions);
    add(new THREE.Mesh(g, m), tex, m, g);
  };
  if (faces) facesOf(faces);
  if (frameFaces) facesOf(frameFaces);
  if (kind === "element" && lines && lines.element.length) {
    const tex = idTexture(Int32Array.from(lines.element, (e) => e + 1));
    const m = primMaterial(tex, 2, false, planes);
    const g = new THREE.BufferGeometry().setAttribute("position", new THREE.BufferAttribute(lines.positions, 3));
    add(new THREE.LineSegments(g, m), tex, m, g);
  }
  if (kind === "node" && nodes && nodes.tags.length) {
    const m = pointMaterial(11 * pixelRatio, planes);
    const g = new THREE.BufferGeometry().setAttribute("position", new THREE.BufferAttribute(nodes.positions, 3));
    add(new THREE.Points(g, m), m, g);
  }
  return scene;
}

/** Draw `scene` as IDs into `target` (cleared to 0, nothing); the renderer's target and clear colour are left as they were. */
function drawIds(renderer: THREE.WebGLRenderer, target: THREE.WebGLRenderTarget, scene: THREE.Scene, camera: THREE.Camera): void {
  camera.updateMatrixWorld();
  const colour = renderer.getClearColor(new THREE.Color()), alpha = renderer.getClearAlpha(), was = renderer.getRenderTarget();
  renderer.setRenderTarget(target);
  renderer.setClearColor(0, 0);
  renderer.clear();
  renderer.render(scene, camera);
  renderer.setRenderTarget(was);
  renderer.setClearColor(colour, alpha);
}

/** How far from the cursor, in CSS pixels, a drawn line still takes a click from the face under it (ADR 0138 D5). */
export const CLICK_LINE_PX = 4;

/**
 * The element a click takes from a window of an ID image (`ids` row 0 at the
 * bottom, `w` x `h`), at window pixel (`cx`, `cy`): the line nearest the
 * centre within `r` pixels, else the element at the centre. An ID is the
 * element index + 1. Returns the element index, or -1 for nothing.
 */
export function clickId(ids: Uint32Array, w: number, h: number, cx: number, cy: number, r: number, isLine: (element: number) => boolean): number {
  let best = -1, bestD = r * r + 1;
  for (let y = Math.max(0, cy - r); y <= Math.min(h - 1, cy + r); y++) {
    for (let x = Math.max(0, cx - r); x <= Math.min(w - 1, cx + r); x++) {
      const id = ids[y * w + x]! & ID_MASK, d = (x - cx) ** 2 + (y - cy) ** 2;
      if (id !== 0 && d < bestD && isLine(id - 1)) [best, bestD] = [id - 1, d];
    }
  }
  if (best >= 0) return best;
  return cx >= 0 && cx < w && cy >= 0 && cy < h ? (ids[cy * w + cx]! & ID_MASK) - 1 : -1;
}

/**
 * The element ID image a click and the hover mark read (ADR 0138 D5). It is
 * drawn again only when what is drawn (`sceneKey`, compared item by item) or
 * the view (camera, drawing size, section planes) changes; each pick reads a
 * window of (2 r + 1)^2 pixels around the pointer.
 */
export class ClickPicker {
  private readonly renderer: THREE.WebGLRenderer;
  private target: THREE.WebGLRenderTarget | null = null;
  private scene: THREE.Scene | null = null;
  private planes: THREE.Plane[] = [];
  private readonly owned: { dispose(): void }[] = [];
  private isLine = new Uint8Array(0);
  private sceneKey: readonly unknown[] = [];
  private viewKey = "";
  private window = new Uint8Array(0);

  constructor(renderer: THREE.WebGLRenderer) {
    this.renderer = renderer;
  }

  /**
   * The element a click at a canvas point (CSS pixels from the top left)
   * takes, with that point moved to the centre of its device pixel (where
   * the image was sampled); null on nothing. `source` is called only when
   * `sceneKey` changed.
   */
  elementAt(
    camera: THREE.Camera,
    sceneKey: readonly unknown[],
    source: () => PickSource | null,
    x: number,
    y: number,
    cssWidth: number,
  ): { element: number; x: number; y: number } | null {
    if (sceneKey.length !== this.sceneKey.length || sceneKey.some((k, i) => k !== this.sceneKey[i])) {
      this.dropScene();
      const src = source();
      if (!src) return null;
      this.scene = pickScene("element", src, this.renderer.getPixelRatio(), this.owned);
      this.planes = src.planes;
      this.isLine = new Uint8Array(src.elements);
      if (src.lines) for (const e of src.lines.element) this.isLine[e] = 1;
      this.sceneKey = sceneKey;
    }
    const size = this.renderer.getDrawingBufferSize(new THREE.Vector2());
    if (!this.target || this.target.width !== size.x || this.target.height !== size.y) {
      this.target?.dispose();
      this.target = new THREE.WebGLRenderTarget(size.x, size.y, { minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter, depthBuffer: true, stencilBuffer: false });
      this.viewKey = "";
    }
    camera.updateMatrixWorld();
    const view = [
      size.x,
      size.y,
      ...camera.matrixWorld.elements,
      ...(camera as THREE.PerspectiveCamera).projectionMatrix.elements,
      ...this.planes.flatMap((p) => [p.normal.x, p.normal.y, p.normal.z, p.constant]),
    ].join(",");
    if (view !== this.viewKey) {
      drawIds(this.renderer, this.target, this.scene!, camera);
      this.viewKey = view;
    }
    const dpr = size.x / cssWidth, px = Math.floor(x * dpr), py = size.y - 1 - Math.floor(y * dpr);
    if (px < 0 || py < 0 || px >= size.x || py >= size.y) return null;
    const r = Math.round(CLICK_LINE_PX * dpr), x0 = Math.max(0, px - r), y0 = Math.max(0, py - r);
    const w = Math.min(size.x, px + r + 1) - x0, h = Math.min(size.y, py + r + 1) - y0;
    if (this.window.length < 4 * w * h) this.window = new Uint8Array(4 * (2 * r + 1) ** 2);
    const bytes = this.window.subarray(0, 4 * w * h);
    this.renderer.readRenderTargetPixels(this.target, x0, y0, w, h, bytes);
    const element = clickId(new Uint32Array(bytes.buffer, bytes.byteOffset, w * h), w, h, px - x0, py - y0, r, (e) => this.isLine[e] === 1);
    return element < 0 ? null : { element, x: (px + 0.5) / dpr, y: (size.y - 1 - py + 0.5) / dpr };
  }

  private dropScene(): void {
    for (const d of this.owned.splice(0)) d.dispose();
    this.scene = null;
    this.sceneKey = [];
    this.viewKey = "";
  }

  dispose(): void {
    this.dropScene();
    this.target?.dispose();
    this.target = null;
  }
}

/** Which sides of a box a canvas point is beyond: 1 left, 2 right, 4 above, 8 below; 0 inside. */
function outcode(x: number, y: number, box: Box): number {
  return (x < box.x0 ? 1 : 0) | (x > box.x1 ? 2 : 0) | (y < box.y0 ? 4 : 0) | (y > box.y1 ? 8 : 0);
}

/**
 * ADR 0138 D7, through the model: the elements a box takes from their
 * projected vertices (`xy`, canvas pixels; NaN for one a section plane cuts
 * away), each vertex with its element and its primitive (a primitive's
 * vertices are consecutive). Left to right (`crossing` false): the elements
 * with every kept vertex inside. Right to left: the elements with a primitive
 * not wholly beyond one side of the box (a bounding test, ADR 0138 "Not
 * established"). Element indices, ascending.
 */
export function elementsThrough(xy: Float32Array, element: Int32Array, prim: Int32Array, box: Box, elements: number): number[] {
  const taken = new Uint8Array(elements), out = new Uint8Array(elements);
  for (let v = 0; v < element.length; ) {
    const p = prim[v]!, e = element[v]!;
    let and = 15, kept = 0;
    for (; v < element.length && prim[v] === p; v++) {
      const x = xy[2 * v]!;
      if (Number.isNaN(x)) continue;
      const c = outcode(x, xy[2 * v + 1]!, box);
      kept++;
      and &= c;
      if (c !== 0) out[e] = 1;
    }
    if (kept > 0 && (box.crossing ? and === 0 : true)) taken[e] = 1;
  }
  const found: number[] = [];
  for (let e = 0; e < elements; e++) if (taken[e] && (box.crossing || !out[e])) found.push(e);
  return found;
}

/** ADR 0138 D7, through the model: the indices of the points (`xy`, NaN when cut away) inside a box. */
export function pointsThrough(xy: Float32Array, box: Box): number[] {
  const found: number[] = [];
  for (let i = 0; i < xy.length / 2; i++) {
    const x = xy[2 * i]!;
    if (!Number.isNaN(x) && outcode(x, xy[2 * i + 1]!, box) === 0) found.push(i);
  }
  return found;
}

/**
 * ADR 0138 D7: a lasso's closed path (canvas CSS pixels, x y pairs) filled
 * into a mask of device pixels over its bounding box, by the even-odd rule
 * (pixel centres; row 0 at the top).
 */
export interface LassoMask {
  x0: number;
  y0: number;
  w: number;
  h: number;
  dpr: number;
  bits: Uint8Array;
}

export function lassoMask(path: ArrayLike<number>, dpr: number): LassoMask {
  const n = path.length / 2;
  let lx = Infinity, ly = Infinity, hx = -Infinity, hy = -Infinity;
  for (let i = 0; i < n; i++) {
    lx = Math.min(lx, path[2 * i]!);
    hx = Math.max(hx, path[2 * i]!);
    ly = Math.min(ly, path[2 * i + 1]!);
    hy = Math.max(hy, path[2 * i + 1]!);
  }
  const x0 = Math.max(0, Math.floor(lx * dpr)), y0 = Math.max(0, Math.floor(ly * dpr));
  const w = n < 3 ? 0 : Math.max(0, Math.ceil(hx * dpr) - x0 + 1), h = n < 3 ? 0 : Math.max(0, Math.ceil(hy * dpr) - y0 + 1);
  const bits = new Uint8Array(w * h), xs: number[] = [];
  for (let j = 0; j < h; j++) {
    const yc = (y0 + j + 0.5) / dpr;
    xs.length = 0;
    for (let i = 0; i < n; i++) {
      const ax = path[2 * i]!, ay = path[2 * i + 1]!, k = (i + 1) % n, bx = path[2 * k]!, by = path[2 * k + 1]!;
      if ((ay <= yc && yc < by) || (by <= yc && yc < ay)) xs.push(ax + ((yc - ay) * (bx - ax)) / (by - ay));
    }
    xs.sort((a, b) => a - b);
    for (let p = 0; p + 1 < xs.length; p += 2) {
      const from = Math.max(0, Math.ceil(xs[p]! * dpr - 0.5) - x0), to = Math.min(w - 1, Math.floor(xs[p + 1]! * dpr - 0.5) - x0);
      for (let x = from; x <= to; x++) bits[j * w + x] = 1;
    }
  }
  return { x0, y0, w, h, dpr, bits };
}

/** Whether a canvas point (CSS pixels) is inside a lasso's mask. */
export function inLasso(m: LassoMask, x: number, y: number): boolean {
  const px = Math.floor(x * m.dpr) - m.x0, py = Math.floor(y * m.dpr) - m.y0;
  return px >= 0 && py >= 0 && px < m.w && py < m.h && m.bits[py * m.w + px] === 1;
}

/**
 * Whether a closed path is drawn clockwise on screen (y down): a window lasso,
 * which takes what is fully inside; counter-clockwise takes what it touches,
 * as a box drawn right to left does.
 */
export function lassoClockwise(path: ArrayLike<number>): boolean {
  let sum = 0;
  const n = path.length / 2;
  for (let i = 0; i < n; i++) {
    const k = (i + 1) % n;
    sum += path[2 * i]! * path[2 * k + 1]! - path[2 * k]! * path[2 * i + 1]!;
  }
  return sum > 0;
}

/** The IDs under a lasso's mask in an ID image (`ids` row 0 at the bottom, `w` x `h`), each once. */
export function idsInLasso(ids: Uint32Array, w: number, h: number, m: LassoMask, seen: Uint32Array, stamp: number, into: number[]): void {
  for (let j = 0; j < m.h; j++) {
    const row = h - 1 - (m.y0 + j);
    if (row < 0 || row >= h) continue;
    for (let i = 0; i < m.w; i++) {
      if (!m.bits[j * m.w + i]) continue;
      const x = m.x0 + i;
      if (x >= w) break;
      const id = ids[row * w + x]! & ID_MASK;
      if (id === 0 || seen[id] === stamp) continue;
      seen[id] = stamp;
      into.push(id - 1);
    }
  }
}

/**
 * Through the model, a lasso (ADR 0138 D7): the elements with every kept
 * vertex inside (`crossing` false), or with any kept vertex inside. Element
 * indices, ascending.
 */
export function elementsInLasso(xy: Float32Array, element: Int32Array, m: LassoMask, elements: number, crossing: boolean): number[] {
  const any = new Uint8Array(elements), out = new Uint8Array(elements);
  for (let v = 0; v < element.length; v++) {
    const x = xy[2 * v]!;
    if (Number.isNaN(x)) continue;
    if (inLasso(m, x, xy[2 * v + 1]!)) any[element[v]!] = 1;
    else out[element[v]!] = 1;
  }
  const found: number[] = [];
  for (let e = 0; e < elements; e++) if (any[e] && (crossing || !out[e])) found.push(e);
  return found;
}

/** One press's ID image and projected vertices; dropped at the release. */
export class BoxPicker {
  /** the press's ID image; null through the model (D7), which reads projected positions only */
  private readonly target: THREE.WebGLRenderTarget | null;
  private readonly ids: Uint32Array;
  private readonly seen: Uint32Array;
  private stamp = 0;
  private readonly disposables: { dispose(): void }[] = [];
  /** per element: the stamp of the last box that found one of its drawn vertices outside */
  private readonly outside: Uint32Array | null;
  /** canvas-pixel position of every drawn vertex (x, y), its element and its primitive, for "fully inside" and through */
  private readonly verts: { xy: Float32Array; element: Int32Array; prim: Int32Array } | null;
  /** through the model, nodes: every drawn node in canvas pixels */
  private readonly points: Float32Array | null;
  private readonly through: boolean;
  /** device pixels per CSS pixel of the canvas */
  private readonly dpr: number;

  private readonly kind: PickKind;
  private readonly src: PickSource;
  private readonly cssWidth: number;
  private readonly cssHeight: number;

  constructor(renderer: THREE.WebGLRenderer, camera: THREE.Camera, kind: PickKind, src: PickSource, cssWidth: number, cssHeight: number, through = false) {
    this.kind = kind;
    this.src = src;
    this.cssWidth = cssWidth;
    this.cssHeight = cssHeight;
    this.through = through;
    this.dpr = renderer.getDrawingBufferSize(new THREE.Vector2()).x / cssWidth;
    const count = kind === "element" ? src.elements : (src.nodes?.tags.length ?? 0);
    this.seen = new Uint32Array(count + 1);
    camera.updateMatrixWorld();
    if (through) {
      this.target = null;
      this.ids = new Uint32Array(0);
    } else {
      const size = renderer.getDrawingBufferSize(new THREE.Vector2());
      this.target = new THREE.WebGLRenderTarget(size.x, size.y, { minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter, depthBuffer: true, stencilBuffer: false });
      const bytes = new Uint8Array(size.x * size.y * 4);
      this.ids = new Uint32Array(bytes.buffer);
      drawIds(renderer, this.target, pickScene(kind, src, renderer.getPixelRatio(), this.disposables), camera);
      renderer.readRenderTargetPixels(this.target, 0, 0, size.x, size.y, bytes);
    }
    this.outside = kind === "element" ? new Uint32Array(src.elements) : null;
    this.verts = kind === "element" ? this.project(camera) : null;
    this.points = kind === "node" && through && src.nodes ? this.projectPoints(camera, src.nodes.positions) : null;
  }

  /** Points in canvas pixels; NaN where a section plane cuts one away, far outside when behind the eye. */
  private projectPoints(camera: THREE.Camera, pos: ArrayLike<number>): Float32Array {
    const m = new THREE.Matrix4().multiplyMatrices((camera as THREE.PerspectiveCamera).projectionMatrix, camera.matrixWorldInverse).elements;
    const xy = new Float32Array((2 * pos.length) / 3);
    for (let i = 0; i < pos.length / 3; i++) this.place(m, pos[3 * i]!, pos[3 * i + 1]!, pos[3 * i + 2]!, xy, i);
    return xy;
  }

  private place(m: number[], x: number, y: number, z: number, xy: Float32Array, o: number): void {
    // A vertex a section plane cuts away is not drawn, so it never counts as outside.
    if (!this.src.planes.every((p) => p.normal.x * x + p.normal.y * y + p.normal.z * z + p.constant >= 0)) {
      xy[2 * o] = NaN;
      return;
    }
    const w = m[3]! * x + m[7]! * y + m[11]! * z + m[15]!;
    if (w <= 0) {
      xy[2 * o] = -1e9;
      xy[2 * o + 1] = -1e9;
      return;
    }
    xy[2 * o] = (((m[0]! * x + m[4]! * y + m[8]! * z + m[12]!) / w) * 0.5 + 0.5) * this.cssWidth;
    xy[2 * o + 1] = (0.5 - ((m[1]! * x + m[5]! * y + m[9]! * z + m[13]!) / w) * 0.5) * this.cssHeight;
  }

  /** Every drawn vertex in canvas pixels, with its element and primitive; a vertex behind the eye is placed far outside. */
  private project(camera: THREE.Camera): { xy: Float32Array; element: Int32Array; prim: Int32Array } {
    const parts: { pos: ArrayLike<number>; element: Int32Array; per: number }[] = [];
    if (this.src.faces) parts.push({ pos: this.src.faces.positions.array, element: this.src.faces.element, per: 3 });
    if (this.src.frameFaces) parts.push({ pos: this.src.frameFaces.positions.array, element: this.src.frameFaces.element, per: 3 });
    if (this.src.lines) parts.push({ pos: this.src.lines.positions, element: this.src.lines.element, per: 2 });
    const n = parts.reduce((a, p) => a + p.pos.length / 3, 0);
    const xy = new Float32Array(2 * n), element = new Int32Array(n), prim = new Int32Array(n);
    const m = new THREE.Matrix4().multiplyMatrices((camera as THREE.PerspectiveCamera).projectionMatrix, camera.matrixWorldInverse).elements;
    let o = 0, primBase = 0;
    for (const p of parts) {
      for (let i = 0, nv = p.pos.length / 3; i < nv; i++, o++) {
        element[o] = p.element[Math.floor(i / p.per)]!;
        prim[o] = primBase + Math.floor(i / p.per);
        this.place(m, p.pos[3 * i]!, p.pos[3 * i + 1]!, p.pos[3 * i + 2]!, xy, o);
      }
      primBase += p.pos.length / 3 / p.per;
    }
    return { xy, element, prim };
  }

  /**
   * What a box takes: element indices, or node tags. Elements fully inside
   * for a box drawn left to right, every one it shows for a crossing box.
   */
  query(box: Box): number[] {
    this.stamp++;
    if (this.through) {
      if (this.kind === "node") return this.points ? pointsThrough(this.points, box).map((i) => this.src.nodes!.tags[i]!) : [];
      return this.verts ? elementsThrough(this.verts.xy, this.verts.element, this.verts.prim, box, this.src.elements) : [];
    }
    const found: number[] = [];
    const size = this.target!;
    idsInBox(this.ids, size.width, size.height, box, size.width / this.cssWidth, this.seen, this.stamp, found);
    if (this.kind === "node") {
      const tags = this.src.nodes!.tags;
      return found.map((i) => tags[i]!);
    }
    if (box.crossing || !this.verts || !this.outside) return found;
    const { xy, element } = this.verts, out = this.outside, stamp = this.stamp;
    for (let v = 0; v < element.length; v++) {
      const x = xy[2 * v]!;
      if (Number.isNaN(x)) continue;
      const y = xy[2 * v + 1]!;
      if (x < box.x0 || x > box.x1 || y < box.y0 || y > box.y1) out[element[v]!] = stamp;
    }
    return found.filter((e) => out[e] !== stamp);
  }

  /**
   * What a lasso takes (ADR 0138 D7): element indices, or node tags. Drawn
   * clockwise, the elements fully inside; counter-clockwise (`crossing`),
   * every one it reaches. Visible only, or through the model.
   */
  queryLasso(path: Float32Array): { found: number[]; crossing: boolean } {
    this.stamp++;
    const crossing = !lassoClockwise(path), m = lassoMask(path, this.dpr);
    if (m.w === 0) return { found: [], crossing };
    if (this.through) {
      if (this.kind === "node") {
        if (!this.points) return { found: [], crossing };
        const tags = this.src.nodes!.tags, found: number[] = [];
        for (let i = 0; i < this.points.length / 2; i++) {
          const x = this.points[2 * i]!;
          if (!Number.isNaN(x) && inLasso(m, x, this.points[2 * i + 1]!)) found.push(tags[i]!);
        }
        return { found, crossing };
      }
      return { found: this.verts ? elementsInLasso(this.verts.xy, this.verts.element, m, this.src.elements, crossing) : [], crossing };
    }
    const found: number[] = [];
    const size = this.target!;
    idsInLasso(this.ids, size.width, size.height, m, this.seen, this.stamp, found);
    if (this.kind === "node") return { found: found.map((i) => this.src.nodes!.tags[i]!), crossing };
    if (crossing || !this.verts || !this.outside) return { found, crossing };
    const { xy, element } = this.verts, out = this.outside, stamp = this.stamp;
    for (let v = 0; v < element.length; v++) {
      const x = xy[2 * v]!;
      if (!Number.isNaN(x) && !inLasso(m, x, xy[2 * v + 1]!)) out[element[v]!] = stamp;
    }
    return { found: found.filter((e) => out[e] !== stamp), crossing };
  }

  dispose(): void {
    this.target?.dispose();
    for (const d of this.disposables.splice(0)) d.dispose();
  }
}
