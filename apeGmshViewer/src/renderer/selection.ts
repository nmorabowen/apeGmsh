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
// pure parts (highlight objects, pulse envelope, selection bounds, fit
// distance) are tested in test/selection.test.ts.

import * as THREE from "three";
import type { BlobStore } from "../state/blobs.ts";
import type { State } from "../state/types.ts";
import { DARK, toHexNumber } from "../theme/tokens.ts";

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

/** The outline of an element's triangles: the edges that belong to one of them only (6 floats per edge). */
export function outlineOf(tp: Float32Array, tris: readonly number[]): Float32Array {
  const coords = new Map<string, [number, number, number, number, number, number]>();
  const seen = new Map<string, number>();
  for (const t of tris) {
    for (let k = 0; k < 3; k++) {
      const a = 9 * t + 3 * k, b = 9 * t + 3 * ((k + 1) % 3);
      const pa = [tp[a]!, tp[a + 1]!, tp[a + 2]!], pb = [tp[b]!, tp[b + 1]!, tp[b + 2]!];
      const ka = pa.join(","), kb = pb.join(",");
      const key = ka < kb ? `${ka}|${kb}` : `${kb}|${ka}`;
      seen.set(key, (seen.get(key) ?? 0) + 1);
      coords.set(key, [...pa, ...pb] as [number, number, number, number, number, number]);
    }
  }
  const out: number[] = [];
  for (const [key, n] of seen) if (n === 1) out.push(...coords.get(key)!);
  return new Float32Array(out);
}

/**
 * The objects that draw a selection over the model: its segments as thick
 * lines, its triangles' outline as thick lines, and its triangles filled, all
 * without depth test, and every one on HALO_LAYER as well as the default
 * layer (the halo pass renders that layer alone). `makeLines` builds a thick
 * line object for the viewport's resolution.
 */
export function highlightObjects(
  lp: Float32Array,
  tp: Float32Array,
  prims: readonly Prims[],
  makeLines: (positions: Float32Array, widthPx: number) => THREE.Object3D,
): THREE.Object3D[] {
  const seg: number[] = [], tri: number[] = [], outline: number[] = [];
  for (const p of prims) {
    for (const i of p.lines) for (let k = 0; k < 6; k++) seg.push(lp[6 * i + k]!);
    for (const i of p.tris) for (let k = 0; k < 9; k++) tri.push(tp[9 * i + k]!);
    outline.push(...outlineOf(tp, p.tris));
  }
  const out: THREE.Object3D[] = [];
  const thick = (pos: Float32Array, width: number) => {
    const o = makeLines(pos, width);
    o.renderOrder = 10;
    const m = (o as THREE.Mesh).material as THREE.Material | undefined;
    if (m) m.depthTest = false;
    return o;
  };
  if (seg.length) out.push(thick(new Float32Array(seg), 9));
  if (outline.length) out.push(thick(new Float32Array(outline), 6));
  if (tri.length) {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(tri), 3));
    const m = new THREE.MeshBasicMaterial({
      color: SELECTION_COLOUR, side: THREE.DoubleSide, transparent: true, opacity: 0.6, depthTest: false,
    });
    const fill = new THREE.Mesh(g, m);
    fill.renderOrder = 9;
    out.push(fill);
  }
  for (const o of out) o.layers.enable(HALO_LAYER);
  return out;
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
  if (!info || s.selection.decls.length === 0) return null;
  const selected = new Set<number>();
  info.elements.forEach((p, i) => {
    if (s.selection.decls.includes(p)) selected.add(i);
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
