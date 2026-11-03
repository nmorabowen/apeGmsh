// 3-D structural loading and prescribed-SP layer: renders `buildLoads3D`
// (`state/loads.ts`) as instances, one per arrow record: `THREE.LineSegments`
// for the shafts and SP bars, `THREE.Mesh` for the heads, and a second
// `THREE.LineSegments` for the beam comb bars. The vertex shader (`ARROW_VERTEX`)
// sizes each glyph on screen: a flat 2:1 head `headSize` pixels long, turned
// toward the camera; a shaft whose nominal length is capped to a pixel band;
// and the arrows below the thinning level the zoom needs hidden. Zooming and
// orbiting therefore rebuild nothing; only a change of loads, step, colour or
// scale rebuilds the records. Section-clip planes, theme and the deformed mesh
// are forwarded as before.

import * as THREE from "three";
import type { BlobStore } from "../state/blobs.ts";
import { ARROW, ARROW_STRIDE, buildLoads3D, buildReactions3D, COMB_STRIDE, THIN_LEVELS } from "../state/loads.ts";
import type { State } from "../state/types.ts";
import type { ThemeMode } from "../theme/tokens.ts";
import { toLinear, type StepValues } from "./contour.ts";
import type { FullPositions } from "./deform.ts";

/** A shaft is never drawn shorter or longer than this on screen (CSS px, times the user scale). */
export const SHAFT_MIN_PX = 15;
export const SHAFT_MAX_PX = 80;
/**
 * Thinning shows the level whose distance (state/loads.ts `thinningLevels`) is
 * the largest power-of-two step no longer on screen than this multiple of the
 * head length, so arrows 3/4 of a head length apart never merge and arrows
 * closer than about 3/8 of one always do.
 */
export const SPACING_PER_HEAD = 0.75;

/**
 * The shader's sizing, written once: each is a GLSL expression that is also
 * JavaScript once `clamp`, `min`, `max`, `floor` and `log2` are defined, so the
 * shader embeds these strings and `screenArrow` evaluates the same strings.
 * In world units: the drawn shaft `L` of an arrow `len` long at `wpp` world
 * units per pixel, its head `H`, and the thinning level the zoom needs.
 */
export const SIZING_EXPR = {
  shaft: "len <= 0.0 ? 0.0 : clamp( len / wpp, uMinPx, uMaxPx ) * wpp",
  head: "min( uHeadPx * wpp, L )",
  need: `min( ${THIN_LEVELS.toFixed(1)}, max( 0.0, ${THIN_LEVELS.toFixed(1)} + floor( log2( uSpacingPx * wpp / uRadius ) ) ) )`,
} as const;

const SIZING_NAMES = ["len", "wpp", "L", "uHeadPx", "uMinPx", "uMaxPx", "uSpacingPx", "uRadius"] as const;
let twin: Record<keyof typeof SIZING_EXPR, (...v: number[]) => number> | null = null;

/**
 * `SIZING_EXPR` evaluated in JavaScript, for tests: the drawn shaft length and
 * head length (world units) of an arrow `len` long at `wpp` world units per
 * pixel, and whether its thinning `level` is shown. The expressions compile on
 * the first call only, so the app (which never calls this) needs no `eval`.
 */
export function screenArrow(len: number, level: number, wpp: number, u: { headPx: number; minPx: number; maxPx: number; radius: number; thin: boolean }): { L: number; H: number; shown: boolean } {
  twin ??= Object.fromEntries(
    Object.entries(SIZING_EXPR).map(([k, expr]) => [
      k,
      new Function(...SIZING_NAMES, `const clamp = (v, a, b) => Math.min(Math.max(v, a), b), min = Math.min, max = Math.max, floor = Math.floor, log2 = Math.log2; return (${expr});`) as (...v: number[]) => number,
    ]),
  ) as NonNullable<typeof twin>;
  const vars = (L: number) => [len, wpp, L, u.headPx, u.minPx, u.maxPx, u.headPx * SPACING_PER_HEAD, u.radius];
  const L = twin.shaft(...vars(0));
  const H = twin.head(...vars(L));
  return { L, H, shown: !u.thin || level >= twin.need(...vars(L)) };
}

const SIZING = `
uniform float uHeadPx;
uniform float uMinPx;
uniform float uMaxPx;
uniform float uSpacingPx;
uniform float uRadius;
uniform float uViewportH;
uniform float uThin;
float worldPerPixel(vec4 mv) {
  float depth = projectionMatrix[3][3] == 1.0 ? 1.0 : -mv.z;
  return 2.0 * depth / (projectionMatrix[1][1] * uViewportH);
}
float shaftLength(float len, float wpp) {
  return ${SIZING_EXPR.shaft};
}
`;

/**
 * Arrow instances: `aShape` = (×shaft length, ×head length, ×head length across), `aPart` 0 shaft or
 * first head, 1 SP bar, 2 second head, 3 reaction tick (2/3 of a head long, at 45° to the shaft's
 * perpendicular, a fifth of the shaft in from its tail).
 */
const ARROW_VERTEX = `
vec4 mvTip = modelViewMatrix * vec4( aTip, 1.0 );
float wpp = worldPerPixel( mvTip );
float L = shaftLength( aLen, wpp );
float H = ${SIZING_EXPR.head};
vec3 dV = normalize( mat3( modelViewMatrix ) * aDir );
vec3 toEye = projectionMatrix[ 3 ][ 3 ] == 1.0 ? vec3( 0.0, 0.0, 1.0 ) : normalize( -mvTip.xyz );
vec3 across = cross( dV, toEye );
if ( length( across ) < 1e-3 ) across = cross( dV, vec3( 0.0, 1.0, 0.0 ) );
if ( length( across ) < 1e-3 ) across = vec3( 1.0, 0.0, 0.0 );
across = normalize( across );
vec4 mvPosition = vec4( mvTip.xyz - dV * ( aShape.x * L + aShape.y * H ) + across * ( aShape.z * H ), 1.0 );
float need = ${SIZING_EXPR.need};
bool hidden = ( aPart == 1.0 && aBar != 1.0 ) || ( aPart == 3.0 && aBar != 2.0 ) || ( aPart == 2.0 && aHeads < 1.5 ) || ( uThin > 0.5 && aLevel < need );
gl_Position = hidden ? vec4( 2.0, 2.0, 2.0, 1.0 ) : projectionMatrix * mvPosition;
`;

/** The arrow instances' vertex source as compiled (sizing functions and body), for the test that it embeds `SIZING_EXPR`. */
export const arrowShaderSource = (): string => SIZING + ARROW_VERTEX;

/** Comb bars: each end at its arrow's drawn tail, `aEnd` 0 or 1. */
const COMB_VERTEX = `
vec3 tip = aEnd < 0.5 ? aTipA : aTipB;
vec3 dir = aEnd < 0.5 ? aDirA : aDirB;
float len = aEnd < 0.5 ? aLenA : aLenB;
vec4 mvTip = modelViewMatrix * vec4( tip, 1.0 );
float L = shaftLength( len, worldPerPixel( mvTip ) );
vec3 dV = length( dir ) > 0.5 ? normalize( mat3( modelViewMatrix ) * dir ) : vec3( 0.0 );
vec4 mvPosition = vec4( mvTip.xyz - dV * L, 1.0 );
gl_Position = projectionMatrix * mvPosition;
`;

type Uniforms = Record<"uHeadPx" | "uMinPx" | "uMaxPx" | "uSpacingPx" | "uRadius" | "uViewportH" | "uThin", { value: number }>;

/** Replace the material's projection with `body` (which sets `mvPosition` for the clip planes, and `gl_Position`). */
function instanced(material: THREE.Material, attributes: string, body: string, uniforms: Uniforms): void {
  material.onBeforeCompile = (shader) => {
    Object.assign(shader.uniforms, uniforms);
    shader.vertexShader = shader.vertexShader
      .replace("#include <common>", `#include <common>\n${attributes}\n${SIZING}`)
      .replace("#include <project_vertex>", body);
  };
}

function template(shape: readonly number[][], extra: Record<string, readonly number[]>): THREE.InstancedBufferGeometry {
  const g = new THREE.InstancedBufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(new Float32Array(shape.length * 3), 3));
  for (const [name, values] of Object.entries(extra)) g.setAttribute(name, new THREE.Float32BufferAttribute(values as number[], values.length / shape.length));
  g.instanceCount = 0;
  return g;
}

const TICK = Math.SQRT1_2 / 3;
const SHAFT_SHAPE = [[1, 0, 0], [0, 1, 0], [0, 0, -0.5], [0, 0, 0.5], [0.8, -TICK, -TICK], [0.8, TICK, TICK]];
const HEAD_SHAPE = [[0, 0, 0], [0, 1, -0.25], [0, 1, 0.25], [0, 0.72, 0], [0, 1.72, -0.25], [0, 1.72, 0.25]];
const shaftTemplate = () => template(SHAFT_SHAPE, { aShape: SHAFT_SHAPE.flat(), aPart: [0, 0, 1, 1, 3, 3] });
const headTemplate = () => template(HEAD_SHAPE, { aShape: HEAD_SHAPE.flat(), aPart: [0, 0, 0, 2, 2, 2] });
const combTemplate = () => template([[0], [0]], { aEnd: [0, 1] });

/** How a card sizes its arrows: the head in pixels, the shaft band's multiplier, and thinning. */
export interface ArrowSizing {
  readonly headSize: number;
  readonly scale: number;
  readonly thin: boolean;
}

/**
 * The instanced arrows of one card: shafts and marks, heads, and comb bars,
 * drawn from `ARROW_STRIDE` records with their own pixel sizing. The loads and
 * the reactions each own one, so each card sizes, colours and thins its own.
 */
export class ArrowLayer {
  readonly root = new THREE.Group();
  /** The shafts, SP bars and reaction ticks. */
  readonly lines: THREE.LineSegments;
  /** The flat heads. */
  readonly cones: THREE.Mesh;
  /** The beam comb bars. */
  readonly combs: THREE.LineSegments;
  readonly uniforms: Uniforms = {
    uHeadPx: { value: 12 },
    uMinPx: { value: SHAFT_MIN_PX },
    uMaxPx: { value: SHAFT_MAX_PX },
    uSpacingPx: { value: 12 * SPACING_PER_HEAD },
    uRadius: { value: 1 },
    uViewportH: { value: 800 },
    uThin: { value: 1 },
  };

  private readonly lineMaterial = new THREE.LineBasicMaterial({ vertexColors: true });
  private readonly coneMaterial = new THREE.MeshBasicMaterial({
    vertexColors: true,
    side: THREE.DoubleSide,
    polygonOffset: true,
    polygonOffsetFactor: -1,
    polygonOffsetUnits: -1,
  });
  private readonly combMaterial = new THREE.LineBasicMaterial({ vertexColors: true });
  protected theme: ThemeMode = "dark";

  constructor() {
    const arrowAttrs = "attribute vec3 aTip;\nattribute vec3 aDir;\nattribute float aLen;\nattribute float aHeads;\nattribute float aBar;\nattribute float aLevel;\nattribute vec3 aShape;\nattribute float aPart;";
    instanced(this.lineMaterial, arrowAttrs, ARROW_VERTEX, this.uniforms);
    instanced(this.coneMaterial, arrowAttrs, ARROW_VERTEX, this.uniforms);
    instanced(
      this.combMaterial,
      "attribute vec3 aTipA;\nattribute vec3 aDirA;\nattribute float aLenA;\nattribute vec3 aTipB;\nattribute vec3 aDirB;\nattribute float aLenB;\nattribute float aEnd;",
      COMB_VERTEX,
      this.uniforms,
    );
    this.lines = new THREE.LineSegments(shaftTemplate(), this.lineMaterial);
    this.cones = new THREE.Mesh(headTemplate(), this.coneMaterial);
    this.combs = new THREE.LineSegments(combTemplate(), this.combMaterial);
    const size = new THREE.Vector2();
    for (const o of [this.lines, this.cones, this.combs]) {
      // The template has no extent of its own: every vertex is placed by the shader.
      o.frustumCulled = false;
      o.onBeforeRender = (renderer) => {
        this.uniforms.uViewportH.value = Math.max(1, renderer.getSize(size).y);
      };
    }
    this.root.add(this.lines, this.cones, this.combs);
    this.root.visible = false;
  }

  setTheme(mode: ThemeMode): void {
    this.theme = mode;
  }

  /** Set the pixel sizing (uniforms: no rebuild); true when it changed. */
  protected size(z: ArrowSizing): boolean {
    const changed = this.uniforms.uHeadPx.value !== z.headSize || this.uniforms.uThin.value !== (z.thin ? 1 : 0) || this.uniforms.uMinPx.value !== SHAFT_MIN_PX * z.scale;
    this.uniforms.uHeadPx.value = z.headSize;
    this.uniforms.uSpacingPx.value = z.headSize * SPACING_PER_HEAD;
    this.uniforms.uMinPx.value = SHAFT_MIN_PX * z.scale;
    this.uniforms.uMaxPx.value = SHAFT_MAX_PX * z.scale;
    this.uniforms.uThin.value = z.thin ? 1 : 0;
    return changed;
  }

  /** Hide the layer; true when it was showing. */
  protected hide(): boolean {
    const was = this.root.visible;
    this.root.visible = false;
    return was;
  }

  /** Draw these records (hidden when there are none); true when anything was or is now drawn. */
  protected upload(arrows: Float32Array, combs: Float32Array, radius: number): boolean {
    const nArrows = arrows.length / ARROW_STRIDE;
    const nCombs = combs.length / COMB_STRIDE;
    if (nArrows === 0 && nCombs === 0) return this.hide();
    this.uniforms.uRadius.value = radius > 0 ? radius : 1;
    // The records are read in place: one interleaved buffer, one attribute view per field.
    const arrowBuf = new THREE.InstancedInterleavedBuffer(arrows, ARROW_STRIDE, 1);
    const arrowColours = new Float32Array(nArrows * 3);
    for (let i = 0; i < nArrows; i++) for (let c = 0; c < 3; c++) arrowColours[i * 3 + c] = toLinear(arrows[i * ARROW_STRIDE + ARROW.rgb + c]!);
    // Fresh templates per rebuild: disposing the old geometry frees its GPU buffers, which replacing attributes would not.
    this.lines.geometry.dispose();
    this.cones.geometry.dispose();
    this.combs.geometry.dispose();
    this.lines.geometry = shaftTemplate();
    this.cones.geometry = headTemplate();
    this.combs.geometry = combTemplate();
    for (const o of [this.lines, this.cones]) {
      const g = o.geometry as THREE.InstancedBufferGeometry;
      g.setAttribute("aTip", new THREE.InterleavedBufferAttribute(arrowBuf, 3, ARROW.tip));
      g.setAttribute("aDir", new THREE.InterleavedBufferAttribute(arrowBuf, 3, ARROW.dir));
      g.setAttribute("aLen", new THREE.InterleavedBufferAttribute(arrowBuf, 1, ARROW.len));
      g.setAttribute("aHeads", new THREE.InterleavedBufferAttribute(arrowBuf, 1, ARROW.heads));
      g.setAttribute("aBar", new THREE.InterleavedBufferAttribute(arrowBuf, 1, ARROW.bar));
      g.setAttribute("aLevel", new THREE.InterleavedBufferAttribute(arrowBuf, 1, ARROW.level));
      g.setAttribute("color", new THREE.InstancedBufferAttribute(arrowColours, 3));
      g.instanceCount = nArrows;
    }
    const combBuf = new THREE.InstancedInterleavedBuffer(combs, COMB_STRIDE, 1);
    const combColours = new Float32Array(nCombs * 3);
    for (let i = 0; i < nCombs; i++) for (let c = 0; c < 3; c++) combColours[i * 3 + c] = toLinear(combs[i * COMB_STRIDE + 14 + c]!);
    const cg = this.combs.geometry as THREE.InstancedBufferGeometry;
    cg.setAttribute("aTipA", new THREE.InterleavedBufferAttribute(combBuf, 3, 0));
    cg.setAttribute("aDirA", new THREE.InterleavedBufferAttribute(combBuf, 3, 3));
    cg.setAttribute("aLenA", new THREE.InterleavedBufferAttribute(combBuf, 1, 6));
    cg.setAttribute("aTipB", new THREE.InterleavedBufferAttribute(combBuf, 3, 7));
    cg.setAttribute("aDirB", new THREE.InterleavedBufferAttribute(combBuf, 3, 10));
    cg.setAttribute("aLenB", new THREE.InterleavedBufferAttribute(combBuf, 1, 13));
    cg.setAttribute("color", new THREE.InstancedBufferAttribute(combColours, 3));
    cg.instanceCount = nCombs;
    this.lines.visible = nArrows > 0;
    this.cones.visible = nArrows > 0;
    this.combs.visible = nCombs > 0;
    this.root.visible = true;
    return true;
  }

  setClippingPlanes(planes: THREE.Plane[]): void {
    for (const mat of [this.lineMaterial, this.coneMaterial, this.combMaterial]) {
      mat.clippingPlanes = planes;
      mat.needsUpdate = true;
    }
  }

  dispose(): void {
    this.lines.geometry.dispose();
    this.cones.geometry.dispose();
    this.combs.geometry.dispose();
    this.lineMaterial.dispose();
    this.coneMaterial.dispose();
    this.combMaterial.dispose();
  }
}

const NO_RECORDS = new Float32Array(0);

const drawnOpts = (theme: ThemeMode, full: FullPositions | null) => ({ theme, ...(full ? { drawnLinePositions: full.line, drawnTriPositions: full.tri } : {}) });

/** The Loads & SPs card's arrows (`s.loads`). */
export class LoadsLayer extends ArrowLayer {
  private appliedKey: string | null = null;

  sync(s: State, blobs: BlobStore, full: FullPositions | null = null): boolean {
    if (!s.loads.on || !s.mesh || s.phase.at?.kind === "geometry") {
      this.appliedKey = null;
      return this.hide();
    }
    const sized = this.size(s.loads);
    const phaseKey = s.phase.at
      ? s.phase.at.kind === "results"
        ? `results:${s.phase.at.stage}:${s.phase.at.step}`
        : s.phase.at.kind === "stage"
          ? `stage:${s.phase.at.index}`
          : s.phase.at.kind
      : "none";
    const nextKey = [
      s.mesh.linePositions.id,
      s.mesh.triPositions.id,
      phaseKey,
      s.stage ?? "",
      full ? full.key : "base",
      s.visibility.hidden.join(","),
      s.loads.scale,
      s.loads.showNodalForces ? "F" : "",
      s.loads.showNodalMoments ? "M" : "",
      s.loads.showElementLoads ? "E" : "",
      s.loads.showSp ? "SP" : "",
      s.loads.includeSupportReactions ? "SR" : "",
      s.loads.scaleBySeries ? "TS" : "",
      s.loads.colorBy,
      s.loads.selectedPatterns === null ? "*" : s.loads.selectedPatterns.join(","),
      s.contour.sourceUnits ?? "",
      s.contour.targetUnits ?? "",
      this.theme,
    ].join("|");
    if (this.root.visible && this.appliedKey === nextKey) return sized;
    const geom = buildLoads3D(s, blobs, drawnOpts(this.theme, full));
    this.appliedKey = geom.arrows.length || geom.combs.length ? nextKey : null;
    return this.upload(geom.arrows, geom.combs, geom.radius);
  }
}

/** The Reactions card's arrows (`s.reactions`): the reaction forces of the results step on screen. */
export class ReactionsLayer extends ArrowLayer {
  private appliedKey: string | null = null;
  /** The step data drawn: a re-read of the same step rebuilds the arrows. */
  private drawnStep: StepValues | null = null;
  /** The results stage the arrows on screen were built for. */
  private drawnStage: number | null = null;

  sync(s: State, blobs: BlobStore, full: FullPositions | null = null, step: StepValues | null = null): boolean {
    const at = s.phase.at;
    if (!s.reactions.on || !s.mesh || at?.kind !== "results") {
      this.appliedKey = null;
      this.drawnStep = null;
      this.drawnStage = null;
      return this.hide();
    }
    const sized = this.size(s.reactions);
    const ids = s.results?.stages[at.stage]?.nodeIds;
    // Until the step's data is here (a scrub), the last step's arrows stay.
    const here = step && step.stage === at.stage && step.step === at.step && ids && blobs.has(ids) ? step : null;
    const nextKey = [s.mesh.linePositions.id, full ? full.key : "base", s.reactions.scale, s.reactions.colorBy, s.reactions.component, s.reactions.allNodes, s.contour.sourceUnits ?? "", s.contour.targetUnits ?? "", this.theme].join("|");
    if (!here) {
      // A scrub within the stage keeps the last arrows until the step arrives; another stage's, or a
      // step whose read failed, are not this step's reactions.
      const failed = s.resultStep?.stage === at.stage && s.resultStep.step === at.step && s.resultStep.status === "failed";
      if (this.drawnStage !== at.stage || failed) {
        this.appliedKey = null;
        this.drawnStep = null;
        this.drawnStage = null;
        return this.hide() || sized;
      }
      return sized;
    }
    if (this.appliedKey === nextKey && here === this.drawnStep) return sized;
    const arrows = buildReactions3D(s, blobs, here.nodal, blobs.get(ids!) as Float64Array, drawnOpts(this.theme, full)).arrows;
    this.drawnStep = here;
    this.drawnStage = at.stage;
    this.appliedKey = nextKey;
    return this.upload(arrows, NO_RECORDS, s.mesh.radius);
  }
}
