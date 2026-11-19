// The Hinges card's markers (`s.hinges`, state/hinges.ts `buildHinges3D`): one
// round point per moment hinge, on the element's chord, following the deformed
// shape. Sized in screen pixels like the node markers, with the card's size or
// the one chosen for the model (state/hinges.ts `smartHingeSize`). Drawn over the
// model, since an embedded hinge sits inside its frame.

import * as THREE from "three";
import type { BlobStore } from "../state/blobs.ts";
import { buildHinges3D, type HingeStepData } from "../state/hinges.ts";
import { hingeJoinOf } from "../state/selectors.ts";
import type { State } from "../state/types.ts";
import { themeTokens, toHexNumber, type ThemeMode } from "../theme/tokens.ts";
import { toLinear } from "./colormapTexture.ts";
import type { StepValues } from "./contour.ts";
import type { FullPositions } from "./deform.ts";
import { makeNodePoints, setNodeSize } from "./nodes.ts";

/** The size until the effects have chosen one for the model. */
const HINGE_SIZE_FALLBACK = 6;
/** The rim is the theme's text colour: it separates a marker from the background and from its neighbours, whatever its fill. */
const rimOf = (theme: ThemeMode): number => toHexNumber(themeTokens(theme).text);
/** Below this size a rim would leave no room for the colour, so the marker is all fill. */
const RIM_FROM_PX = 8;
const RIM_PX = 1;

export class HingesLayer {
  readonly root = new THREE.Group();
  private points: THREE.Points | null = null;
  private capacity = 0;
  private planes: THREE.Plane[] = [];
  private theme: ThemeMode = "dark";
  private appliedKey: string | null = null;
  private drawnStep: HingeStepData | null = null;
  private drawnStage: number | null = null;
  private drawnSize: number | null = null;
  /** what the last build drew */
  drawn = { count: 0, unplaced: 0, noValue: 0, top: null as number | null, size: 0 };

  constructor() {
    this.root.visible = false;
    this.root.renderOrder = 20;
  }

  setClippingPlanes(planes: THREE.Plane[]): void {
    this.planes = planes;
    if (this.points) {
      const m = this.points.material as THREE.PointsMaterial;
      m.clippingPlanes = planes;
      m.needsUpdate = true;
    }
  }

  setTheme(mode: ThemeMode): void {
    this.theme = mode;
    if (this.points) this.rim(this.points);
  }

  private rim(points: THREE.Points): void {
    const uniforms = (points.material as THREE.PointsMaterial).userData["nodeUniforms"] as { nodeRim: { value: THREE.Color } };
    uniforms.nodeRim.value.set(rimOf(this.theme));
  }

  /** The marker size, with this layer's own rim: thinner than a node's, and none on a small marker. */
  private setSize(points: THREE.Points, size: number): void {
    setNodeSize(points, size);
    const uniforms = (points.material as THREE.PointsMaterial).userData["nodeUniforms"] as { nodeRimAt: { value: number } };
    uniforms.nodeRimAt.value = size >= RIM_FROM_PX ? 1 - (2 * RIM_PX) / size : 2;
  }

  private hide(): boolean {
    const was = this.root.visible;
    this.root.visible = false;
    this.appliedKey = null;
    this.drawnStep = null;
    this.drawnStage = null;
    this.drawn = { count: 0, unplaced: 0, noValue: 0, top: null, size: 0 };
    return was;
  }

  /** A size change touches the material only: a dragged slider never rebuilds the markers. */
  private applySize(size: number): boolean {
    this.drawn.size = size;
    if (!this.points || size === this.drawnSize) return false;
    this.setSize(this.points, size);
    this.drawnSize = size;
    return true;
  }

  /** True when the scene changed. */
  sync(s: State, blobs: BlobStore, full: FullPositions | null = null, step: StepValues | null = null): boolean {
    if (!s.hinges.on || !s.mesh) return this.hide();
    const size = s.hinges.size ?? HINGE_SIZE_FALLBACK;
    const at = s.phase.at;
    const stage = at?.kind === "results" ? at.stage : null;
    const here = at?.kind === "results" && step && step.stage === at.stage && step.step === at.step && step.hinges ? step : null;
    if (at?.kind === "results" && !here && (s.results?.stages[at.stage]?.hinges ?? 0) > 0) {
      // Until the step arrives the last markers stay; another stage's, or a failed read, are redrawn without values.
      const failed = (s.resultStep?.stage === at.stage && s.resultStep.step === at.step && s.resultStep.status === "failed") || (step?.stage === at.stage && step.step === at.step && step.hinges === null);
      if (this.drawnStage === at.stage && this.drawnStep && !failed) return this.applySize(size);
    }
    const data = here?.hinges ?? null;
    const key = [s.mesh.linePositions.id, full ? full.key : "base", s.hinges.colorBy, s.visibility.hidden.length].join("|");
    if (key === this.appliedKey && data === this.drawnStep && stage === this.drawnStage) return this.applySize(size);
    const join = data ? hingeJoinOf(s) : null;
    const idOf = join?.ok ? (d: string) => join.byDeclPath.get(d) : null;
    const g = buildHinges3D(s, blobs, data, idOf, full ? { drawnLinePositions: full.line, drawnTriPositions: full.tri } : undefined);
    this.appliedKey = key;
    this.drawnStep = data;
    this.drawnStage = stage;
    this.drawn = { count: g.count, unplaced: g.unplaced, noValue: g.noValue, top: g.top, size };
    if (!this.points || this.capacity < g.count) {
      this.disposePoints();
      this.capacity = Math.max(g.count, 1);
      const points = makeNodePoints(new Float32Array(this.capacity * 3), size, toHexNumber(themeTokens(this.theme).text), rimOf(this.theme));
      (points.material as THREE.PointsMaterial).color.setRGB(1, 1, 1);
      points.geometry.setAttribute("color", new THREE.BufferAttribute(new Float32Array(this.capacity * 3), 3));
      const m = points.material as THREE.PointsMaterial;
      m.vertexColors = true;
      m.depthTest = false;
      m.transparent = true;
      m.clippingPlanes = this.planes;
      points.frustumCulled = false;
      points.renderOrder = 20;
      this.setSize(points, size);
      this.points = points;
      this.drawnSize = size;
      this.root.add(points);
    }
    const position = this.points.geometry.getAttribute("position") as THREE.BufferAttribute;
    const color = this.points.geometry.getAttribute("color") as THREE.BufferAttribute;
    (position.array as Float32Array).set(g.positions);
    const rgb = color.array as Float32Array;
    for (let i = 0; i < g.colors.length; i++) rgb[i] = toLinear(g.colors[i]!);
    position.needsUpdate = true;
    color.needsUpdate = true;
    this.points.geometry.setDrawRange(0, g.count);
    this.applySize(size);
    this.root.visible = g.count > 0;
    return true;
  }

  private disposePoints(): void {
    if (!this.points) return;
    this.root.remove(this.points);
    this.points.geometry.dispose();
    (this.points.material as THREE.Material).dispose();
    this.points = null;
    this.drawnSize = null;
  }

  dispose(): void {
    this.disposePoints();
  }
}
