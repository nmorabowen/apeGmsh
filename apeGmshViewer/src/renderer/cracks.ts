// The Cracks card's discs (`s.cracks`, state/cracks.ts): at each Gauss point of
// the stage's crack pattern, a disc in each crack plane, coloured by the opening,
// following the deformed shape as the Gauss markers and the principal arrows do.
// One unlit, double-sided mesh of non-indexed triangles with vertex colours: a
// disc has no front, and its colour is the opening, never a shade.

import * as THREE from "three";
import type { BlobStore } from "../state/blobs.ts";
import { buildCracks3D, crackHalfSize, crackVectors } from "../state/cracks.ts";
import { deformedPositions } from "../state/deform.ts";
import { gaussBasePositions, gaussDisplacements } from "../state/gaussPoints.ts";
import { principalPointsOf, type PrincipalPoints } from "../state/principal.ts";
import { crackFieldOf } from "../state/selectors.ts";
import type { State } from "../state/types.ts";
import { toLinear } from "./colormapTexture.ts";
import type { StepValues } from "./contour.ts";
import type { FullPositions } from "./deform.ts";

export class CracksLayer {
  readonly root = new THREE.Group();
  private readonly geometry = new THREE.BufferGeometry();
  private readonly material = new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide });
  private readonly mesh = new THREE.Mesh(this.geometry, this.material);
  private appliedKey: string | null = null;
  private drawnStep: StepValues | null = null;
  private drawnStage: number | null = null;
  /** The field's points, placed once per stage, field and visibility, and half a disc's width from their spacing. */
  private points: { deps: unknown[]; gen: number; points: PrincipalPoints | null; base: Float32Array | null; half: number } | null = null;
  /** what the last build drew: discs, the points with one, and the largest opening (the colour range's top) */
  drawn = { discs: 0, points: 0, maxOpening: 0 };

  constructor() {
    this.mesh.frustumCulled = false;
    this.root.add(this.mesh);
    this.root.visible = false;
  }

  private hide(): boolean {
    const was = this.root.visible;
    this.root.visible = false;
    this.drawn = { discs: 0, points: 0, maxOpening: 0 };
    return was;
  }

  /** Rebuild when the state, the deformed shape or the step's crack vectors change; true when the scene changed. */
  sync(s: State, blobs: BlobStore, full: FullPositions | null = null, step: StepValues | null = null): boolean {
    const at = s.phase.at;
    const field = s.cracks.on ? crackFieldOf(s) : null;
    if (!field || field.groups.length === 0 || !s.mesh?.nodes || at?.kind !== "results") {
      this.appliedKey = null;
      this.drawnStep = null;
      this.drawnStage = null;
      return this.hide();
    }
    const tensor = step?.tensor;
    const here = step && step.stage === at.stage && step.step === at.step && tensor && field.components.every((c) => tensor[c]) ? step : null;
    if (!here) {
      // A scrub within the stage keeps the last discs until the step arrives; another stage's, or a failed read, do not.
      const failed = s.resultStep?.stage === at.stage && s.resultStep.step === at.step && s.resultStep.status === "failed";
      if (this.drawnStage !== at.stage || failed) {
        this.appliedKey = null;
        this.drawnStep = null;
        this.drawnStage = null;
        return this.hide();
      }
      return false;
    }
    const deps = [s.results, at.stage, s.visibility.hidden, s.mesh, s.decls, s.stageElements, s.sequence, field.components.join(",")];
    if (!this.points || this.points.deps.some((d, i) => d !== deps[i])) {
      const points = principalPointsOf(s, blobs, field);
      const nodes = s.mesh.nodes;
      const base = points ? gaussBasePositions(points.layout, blobs.get(nodes.ids) as Float64Array, blobs.get(nodes.xyz) as Float64Array) : null;
      this.points = { deps, gen: (this.points?.gen ?? 0) + 1, points, base, half: base ? crackHalfSize(base) : 1 };
    }
    const { points, base, half } = this.points;
    const disp = full && full.key !== "base" && full.disp?.nodal ? full.disp : null;
    const c = s.cracks;
    const nextKey = [this.points.gen, disp ? full!.key : "base", c.minOpening, c.cmap, c.scale].join("|");
    if (this.appliedKey === nextKey && here === this.drawnStep) return false;
    this.drawnStep = here;
    this.drawnStage = at.stage;
    this.appliedKey = nextKey;
    if (!points || !base) return this.hide();
    const at3 = disp ? deformedPositions(base, gaussDisplacements(points.layout, disp.nodal!.column, disp.nodal!.rows), s.deform.scale) : base;
    const r = buildCracks3D(at3, crackVectors(points, field, here.tensor!), { minOpening: c.minOpening, cmap: c.cmap, halfSize: half * c.scale });
    this.drawn = { discs: r.discs, points: r.points, maxOpening: r.maxOpening };
    for (let i = 0; i < r.colours.length; i++) r.colours[i] = toLinear(r.colours[i]!);
    this.geometry.setAttribute("position", new THREE.BufferAttribute(r.positions, 3));
    this.geometry.setAttribute("color", new THREE.BufferAttribute(r.colours, 3));
    this.geometry.computeBoundingSphere();
    this.root.visible = r.discs > 0;
    return true;
  }

  setClippingPlanes(planes: THREE.Plane[]): void {
    this.material.clippingPlanes = planes;
    this.material.needsUpdate = true;
  }

  dispose(): void {
    this.geometry.dispose();
    this.material.dispose();
  }
}
