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
 * Thinning merges arrows that share a cell no wider on screen than this
 * multiple of the head length (the widest power-of-two cell that fits), so
 * arrows a head length apart never merge and arrows closer than about half of
 * one always do.
 */
export const SPACING_PER_HEAD = 0.75;

/**
 * The CPU twin of the shader's sizing, for tests: the drawn shaft length and
 * head length (world units) of an arrow `len` long at `wpp` world units per
 * pixel, and whether its thinning `level` is shown.
 */
export function screenArrow(len: number, level: number, wpp: number, u: { headPx: number; minPx: number; maxPx: number; radius: number; thin: boolean }): { L: number; H: number; shown: boolean } {
  const L = Math.min(Math.max(len / wpp, u.minPx), u.maxPx) * wpp;
  const H = Math.min(u.headPx * wpp, L);
  const need = Math.min(THIN_LEVELS, Math.max(0, THIN_LEVELS + Math.floor(Math.log2((SPACING_PER_HEAD * u.headPx * wpp) / u.radius))));
  return { L, H, shown: !u.thin || level >= need };
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
  return len <= 0.0 ? 0.0 : clamp(len / wpp, uMinPx, uMaxPx) * wpp;
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
float H = min( uHeadPx * wpp, L );
vec3 dV = normalize( mat3( modelViewMatrix ) * aDir );
vec3 toEye = projectionMatrix[ 3 ][ 3 ] == 1.0 ? vec3( 0.0, 0.0, 1.0 ) : normalize( -mvTip.xyz );
vec3 across = cross( dV, toEye );
if ( length( across ) < 1e-3 ) across = cross( dV, vec3( 0.0, 1.0, 0.0 ) );
if ( length( across ) < 1e-3 ) across = vec3( 1.0, 0.0, 0.0 );
across = normalize( across );
vec4 mvPosition = vec4( mvTip.xyz - dV * ( aShape.x * L + aShape.y * H ) + across * ( aShape.z * H ), 1.0 );
float need = min( ${THIN_LEVELS.toFixed(1)}, max( 0.0, ${THIN_LEVELS.toFixed(1)} + floor( log2( uSpacingPx * wpp / uRadius ) ) ) );
bool hidden = ( aPart == 1.0 && aBar != 1.0 ) || ( aPart == 3.0 && aBar != 2.0 ) || ( aPart == 2.0 && aHeads < 1.5 ) || ( uThin > 0.5 && aLevel < need );
gl_Position = hidden ? vec4( 2.0, 2.0, 2.0, 1.0 ) : projectionMatrix * mvPosition;
`;

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

export class LoadsLayer {
  readonly root = new THREE.Group();
  /** The shafts and SP bars. */
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
  private theme: ThemeMode = "dark";
  private appliedKey: string | null = null;
  /** The step data the reactions were built from: a re-read of the same step rebuilds them. */
  private reactionStep: StepValues | null = null;
  private reactionArrows: Float32Array = new Float32Array(0);

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

  sync(s: State, blobs: BlobStore, full: FullPositions | null = null, step: StepValues | null = null): boolean {
    if (!(s.loads.on || s.loads.showReactions) || !s.mesh || s.phase.at?.kind === "geometry") {
      const wasVisible = this.root.visible;
      this.root.visible = false;
      this.appliedKey = null;
      return wasVisible;
    }

    // The pixel sizes are uniforms: a new head size or thinning switch redraws without a rebuild.
    const headPx = s.loads.headSize;
    const uniformsChanged = this.uniforms.uHeadPx.value !== headPx || this.uniforms.uThin.value !== (s.loads.thin ? 1 : 0) || this.uniforms.uMinPx.value !== SHAFT_MIN_PX * s.loads.scale;
    this.uniforms.uHeadPx.value = headPx;
    this.uniforms.uSpacingPx.value = headPx * SPACING_PER_HEAD;
    this.uniforms.uMinPx.value = SHAFT_MIN_PX * s.loads.scale;
    this.uniforms.uMaxPx.value = SHAFT_MAX_PX * s.loads.scale;
    this.uniforms.uThin.value = s.loads.thin ? 1 : 0;

    const phaseKey = s.phase.at
      ? s.phase.at.kind === "results"
        ? `results:${s.phase.at.stage}:${s.phase.at.step}`
        : s.phase.at.kind === "stage"
          ? `stage:${s.phase.at.index}`
          : s.phase.at.kind
      : "none";
    const deformKey = full ? full.key : "base";
    const nextKey = [
      s.mesh.linePositions.id,
      s.mesh.triPositions.id,
      phaseKey,
      s.stage ?? "",
      deformKey,
      s.visibility.hidden.join(","),
      s.loads.scale,
      s.loads.showNodalForces ? "F" : "",
      s.loads.showNodalMoments ? "M" : "",
      s.loads.showElementLoads ? "E" : "",
      s.loads.showSp ? "SP" : "",
      s.loads.includeSupportReactions ? "SR" : "",
      s.loads.scaleBySeries ? "TS" : "",
      s.loads.colorBy,
      s.loads.on ? "L" : "",
      s.loads.showReactions ? "R" : "",
      s.loads.selectedPatterns === null ? "*" : s.loads.selectedPatterns.join(","),
      s.contour.sourceUnits ?? "",
      s.contour.targetUnits ?? "",
      this.theme,
    ].join("|");

    // The reactions of the step on screen, once its data is here; until then (a scrub) the last ones stay.
    const at = s.phase.at;
    const ids = at?.kind === "results" ? s.results?.stages[at.stage]?.nodeIds : undefined;
    const stepHere = s.loads.showReactions && at?.kind === "results" && step && step.stage === at.stage && step.step === at.step && ids && blobs.has(ids) ? step : null;
    if (this.root.visible && this.appliedKey === nextKey && (!stepHere || stepHere === this.reactionStep)) {
      return uniformsChanged;
    }

    const opts = {
      theme: this.theme,
      ...(full
        ? {
            drawnLinePositions: full.line,
            drawnTriPositions: full.tri,
          }
        : {}),
    };
    const loads = s.loads.on ? buildLoads3D(s, blobs, opts) : { arrows: new Float32Array(0), combs: new Float32Array(0), radius: s.mesh.radius };
    if (stepHere && ids) this.reactionArrows = buildReactions3D(s, blobs, stepHere.nodal, blobs.get(ids) as Float64Array, opts).arrows;
    else if (!s.loads.showReactions || at?.kind !== "results") this.reactionArrows = new Float32Array(0);
    if (stepHere) this.reactionStep = stepHere;
    const reactions = this.reactionArrows;
    const geom = { arrows: new Float32Array(loads.arrows.length + reactions.length), combs: loads.combs, radius: loads.radius };
    geom.arrows.set(loads.arrows);
    geom.arrows.set(reactions, loads.arrows.length);

    const nArrows = geom.arrows.length / ARROW_STRIDE;
    const nCombs = geom.combs.length / COMB_STRIDE;
    if (nArrows === 0 && nCombs === 0) {
      const wasVisible = this.root.visible;
      this.root.visible = false;
      this.appliedKey = null;
      return wasVisible;
    }

    this.uniforms.uRadius.value = geom.radius > 0 ? geom.radius : 1;
    // The records are read in place: one interleaved buffer, one attribute view per field.
    const arrowBuf = new THREE.InstancedInterleavedBuffer(geom.arrows, ARROW_STRIDE, 1);
    const arrowColours = new Float32Array(nArrows * 3);
    for (let i = 0; i < nArrows; i++) for (let c = 0; c < 3; c++) arrowColours[i * 3 + c] = toLinear(geom.arrows[i * ARROW_STRIDE + ARROW.rgb + c]!);
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
    const combBuf = new THREE.InstancedInterleavedBuffer(geom.combs, COMB_STRIDE, 1);
    const combColours = new Float32Array(nCombs * 3);
    for (let i = 0; i < nCombs; i++) for (let c = 0; c < 3; c++) combColours[i * 3 + c] = toLinear(geom.combs[i * COMB_STRIDE + 14 + c]!);
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
    this.appliedKey = nextKey;
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
