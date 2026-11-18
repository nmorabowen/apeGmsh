// The Hinges card's markers (`s.hinges`, state/hinges.ts `buildHinges3D`): one
// small sphere per moment hinge, on the element's chord, following the deformed
// shape. Drawn over the model, since an embedded hinge sits inside its frame.

import * as THREE from "three";
import type { BlobStore } from "../state/blobs.ts";
import { buildHinges3D, type HingeStepData } from "../state/hinges.ts";
import { hingeJoinOf } from "../state/selectors.ts";
import type { State } from "../state/types.ts";
import { toLinear } from "./colormapTexture.ts";
import type { StepValues } from "./contour.ts";
import type { FullPositions } from "./deform.ts";

/** A marker's radius as a fraction of the model's. */
export const HINGE_RADIUS = 0.018;

export class HingesLayer {
  readonly root = new THREE.Group();
  private readonly sphere = new THREE.SphereGeometry(1, 16, 12);
  private readonly material = new THREE.MeshBasicMaterial({ depthTest: false, transparent: true });
  private mesh: THREE.InstancedMesh | null = null;
  private capacity = 0;
  private appliedKey: string | null = null;
  private drawnStep: HingeStepData | null = null;
  private drawnStage: number | null = null;
  /** what the last build drew */
  drawn = { count: 0, unplaced: 0, noValue: 0, top: null as number | null };

  constructor() {
    this.root.visible = false;
    this.root.renderOrder = 20;
  }

  setClippingPlanes(planes: THREE.Plane[]): void {
    this.material.clippingPlanes = planes;
    this.material.needsUpdate = true;
  }

  private hide(): boolean {
    const was = this.root.visible;
    this.root.visible = false;
    this.appliedKey = null;
    this.drawnStep = null;
    this.drawnStage = null;
    this.drawn = { count: 0, unplaced: 0, noValue: 0, top: null };
    return was;
  }

  /** True when the scene changed. */
  sync(s: State, blobs: BlobStore, full: FullPositions | null = null, step: StepValues | null = null): boolean {
    if (!s.hinges.on || !s.mesh) return this.hide();
    const at = s.phase.at;
    const stage = at?.kind === "results" ? at.stage : null;
    const here = at?.kind === "results" && step && step.stage === at.stage && step.step === at.step && step.hinges ? step : null;
    if (at?.kind === "results" && !here && (s.results?.stages[at.stage]?.hinges ?? 0) > 0) {
      // Until the step arrives the last markers stay; another stage's, or a failed read, are redrawn without values.
      const failed = (s.resultStep?.stage === at.stage && s.resultStep.step === at.step && s.resultStep.status === "failed") || (step?.stage === at.stage && step.step === at.step && step.hinges === null);
      if (this.drawnStage === at.stage && this.drawnStep && !failed) return false;
    }
    const data = here?.hinges ?? null;
    const key = [s.mesh.linePositions.id, full ? full.key : "base", s.hinges.size, s.hinges.colorBy, s.visibility.hidden.length].join("|");
    if (key === this.appliedKey && data === this.drawnStep && stage === this.drawnStage) return false;
    const join = data ? hingeJoinOf(s) : null;
    const idOf = join?.ok ? (d: string) => join.byDeclPath.get(d) : null;
    const g = buildHinges3D(s, blobs, data, idOf, full ? { drawnLinePositions: full.line, drawnTriPositions: full.tri } : undefined);
    this.appliedKey = key;
    this.drawnStep = data;
    this.drawnStage = stage;
    this.drawn = { count: g.count, unplaced: g.unplaced, noValue: g.noValue, top: g.top };
    if (!this.mesh || this.capacity < g.count) {
      if (this.mesh) {
        this.root.remove(this.mesh);
        this.mesh.dispose();
      }
      this.capacity = Math.max(g.count, 1);
      this.mesh = new THREE.InstancedMesh(this.sphere, this.material, this.capacity);
      this.mesh.frustumCulled = false;
      this.mesh.renderOrder = 20;
      this.root.add(this.mesh);
    }
    const r = (s.mesh.radius > 0 ? s.mesh.radius : 1) * HINGE_RADIUS * s.hinges.size;
    const m = new THREE.Matrix4(), c = new THREE.Color();
    for (let i = 0; i < g.count; i++) {
      m.makeScale(r, r, r).setPosition(g.positions[i * 3]!, g.positions[i * 3 + 1]!, g.positions[i * 3 + 2]!);
      this.mesh.setMatrixAt(i, m);
      this.mesh.setColorAt(i, c.setRGB(toLinear(g.colors[i * 3]!), toLinear(g.colors[i * 3 + 1]!), toLinear(g.colors[i * 3 + 2]!)));
    }
    this.mesh.count = g.count;
    this.mesh.instanceMatrix.needsUpdate = true;
    if (this.mesh.instanceColor) this.mesh.instanceColor.needsUpdate = true;
    this.root.visible = g.count > 0;
    return true;
  }

  dispose(): void {
    this.mesh?.dispose();
    this.sphere.dispose();
    this.material.dispose();
  }
}
