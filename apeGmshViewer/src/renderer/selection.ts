// Selection emphasis (R1 follow-up on #1308): a screen-space halo of fixed
// pixel width drawn on top of everything, a short pulse when the selection
// changes, and the bounds `F` frames.
//
// The halo is a post pass: the highlight objects (they sit on HALO_LAYER as
// well as the default layer) are rendered alone into a mask target, then a
// full-screen quad paints every pixel that is outside the mask but within
// HALO_PX of it. Its width is in device pixels, so it reads the same for a
// beam, a shell or a solid at any zoom. The pure parts (pulse envelope,
// selection bounds, fit distance) are tested in test/selection.test.ts.

import * as THREE from "three";

/** The layer the highlight objects also live on; the mask pass renders it alone. */
export const HALO_LAYER = 1;
/** Halo width in device pixels. */
export const HALO_PX = 3;
export const HALO_COLOUR = 0xffffff;
/** The pulse: HALO_PX swells to PULSE_SCALE x HALO_PX and back over PULSE_MS. */
export const PULSE_MS = 650;
export const PULSE_SCALE = 3;

/** Halo width multiplier `t` ms after a select: 1 at rest, PULSE_SCALE at the middle of the pulse. */
export function pulseAt(t: number, duration = PULSE_MS, scale = PULSE_SCALE): number {
  if (!(t >= 0) || t >= duration) return 1;
  return 1 + (scale - 1) * Math.sin((Math.PI * t) / duration);
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
): { center: [number, number, number]; radius: number } | null {
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
 * The camera distance that fits a sphere of `radius` in a view of vertical
 * field `fovDeg` whose free area is `freePx` wide and `heightPx` tall: the
 * sphere touches the narrower half-angle, with 8 % of margin.
 */
export function fitDistance(radius: number, fovDeg: number, freePx: number, heightPx: number): number {
  const half = THREE.MathUtils.degToRad(fovDeg / 2);
  const halfH = Math.atan(Math.tan(half) * (Math.max(1, freePx) / Math.max(1, heightPx)));
  return (radius / Math.sin(Math.min(half, halfH))) * 1.08;
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
  private readonly mask: THREE.WebGLRenderTarget;
  private readonly quad: THREE.Mesh<THREE.PlaneGeometry, THREE.ShaderMaterial>;
  private readonly pass = new THREE.Scene();
  private readonly ortho = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  private pulseStart: number | null = null;

  constructor(renderer: THREE.WebGLRenderer) {
    this.renderer = renderer;
    this.mask = new THREE.WebGLRenderTarget(1, 1, { depthBuffer: false, stencilBuffer: false });
    this.quad = new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.ShaderMaterial({
        uniforms: {
          tMask: { value: this.mask.texture },
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

  /** Size the mask to the canvas (CSS pixels times the device pixel ratio). */
  resize(width: number, height: number, pixelRatio: number): void {
    const w = Math.max(1, Math.round(width * pixelRatio)), h = Math.max(1, Math.round(height * pixelRatio));
    this.mask.setSize(w, h);
    (this.quad.material.uniforms["texel"]!.value as THREE.Vector2).set(1 / w, 1 / h);
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
    const t = this.pulseStart === null ? Infinity : performance.now() - this.pulseStart;
    this.quad.material.uniforms["radius"]!.value = HALO_PX * pulseAt(t);
    if (t >= PULSE_MS) this.pulseStart = null;
    const r = this.renderer;
    const autoClear = r.autoClear, mask = camera.layers.mask;
    camera.layers.set(HALO_LAYER);
    r.setRenderTarget(this.mask);
    r.setClearColor(0x000000, 0);
    r.clear();
    r.render(scene, camera);
    r.setRenderTarget(null);
    camera.layers.mask = mask;
    r.autoClear = false;
    r.render(this.pass, this.ortho);
    r.autoClear = autoClear;
  }

  dispose(): void {
    this.mask.dispose();
    this.quad.geometry.dispose();
    this.quad.material.dispose();
  }
}
