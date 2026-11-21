// The Cracks card's discs (`s.cracks`, state/cracks.ts): at each Gauss point of
// the stage's crack pattern, a disc in each crack plane, coloured by the opening,
// following the deformed shape as the Gauss markers and the principal arrows do.
// One unlit, double-sided mesh of non-indexed triangles with vertex colours: a
// disc has no front, and its colour is the opening, never a shade.
//
// Layered shells (state/shellCracks.ts): a tick per crack on each face shown,
// red toward the positive global axis, blue the other, one line-segment batch
// with vertex colours. The ticks sit at their layer's face, +-t/2 along e3: on the
// slab's faces when the shells are drawn extruded (`visibility.extrudeFrames`,
// renderer/shellSlab.ts), and off the drawn mid-surface when they are not.

import * as THREE from "three";
import type { BlobStore } from "../state/blobs.ts";
import { buildCracks3D, crackHalfSize, crackVectors } from "../state/cracks.ts";
import { deformedPositions } from "../state/deform.ts";
import { gaussBasePositions, gaussDisplacements } from "../state/gaussPoints.ts";
import { principalPointsOf, type PrincipalPoints } from "../state/principal.ts";
import { crackFieldOf, shellCrackFieldOf } from "../state/selectors.ts";
import { buildCrackLines, buildRibbons, type CrackLines } from "../state/crackLines.ts";
import { elementCornersOf, lineVertexLayoutOf, shellCrackPointsOf, type ShellCrackPoints } from "../state/shellCrackPoints.ts";
import { buildShellTicks, type ShellTickGroup } from "../state/shellCracks.ts";
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
  private readonly tickGeometry = new THREE.BufferGeometry();
  private readonly tickMaterial = new THREE.LineBasicMaterial({ vertexColors: true });
  private readonly ticks = new THREE.LineSegments(this.tickGeometry, this.tickMaterial);
  private tickKey: string | null = null;
  private tickStep: StepValues | null = null;
  private tickStage: number | null = null;
  private tickPoints: { deps: unknown[]; gen: number; points: ShellCrackPoints | null; base: Float32Array | null; half: number } | null = null;
  /** what the last shell build drew: ticks, the points with one, the points never drawn for want of a frame, and the largest opening */
  drawnShell = { ticks: 0, points: 0, unframed: 0, maxOpening: 0 };
  private readonly ribbonGeometry = new THREE.BufferGeometry();
  private readonly ribbonMaterial = new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide });
  private readonly ribbons = new THREE.Mesh(this.ribbonGeometry, this.ribbonMaterial);
  /** The traced lines of the step on screen, kept while only the deformation changes (no re-trace for a scale). */
  private traced: { key: unknown[]; lines: CrackLines; layout: ReturnType<typeof lineVertexLayoutOf>; base: Float32Array } | null = null;
  /** what the last line build drew: lines, ribbon segments, panels, and the largest opening on a line */
  drawnLines = { lines: 0, segments: 0, panels: 0, maxOpening: 0 };

  constructor() {
    this.mesh.frustumCulled = false;
    this.ticks.frustumCulled = false;
    this.ribbons.frustumCulled = false;
    this.mesh.visible = false;
    this.ticks.visible = false;
    this.ribbons.visible = false;
    this.root.add(this.mesh, this.ticks, this.ribbons);
    this.root.visible = false;
  }

  private hide(): boolean {
    const was = this.mesh.visible;
    this.mesh.visible = false;
    this.drawn = { discs: 0, points: 0, maxOpening: 0 };
    return was;
  }

  private hideTicks(): boolean {
    const was = this.ticks.visible || this.ribbons.visible;
    this.ticks.visible = false;
    this.ribbons.visible = false;
    this.traced = null;
    this.drawnLines = { lines: 0, segments: 0, panels: 0, maxOpening: 0 };
    this.tickKey = null;
    this.tickStep = null;
    this.tickStage = null;
    this.drawnShell = { ticks: 0, points: 0, unframed: 0, maxOpening: 0 };
    return was;
  }

  /** Rebuild when the state, the deformed shape or the step's crack vectors change; true when the scene changed. */
  sync(s: State, blobs: BlobStore, full: FullPositions | null = null, step: StepValues | null = null): boolean {
    const discs = this.syncDiscs(s, blobs, full, step);
    const ticks = this.syncTicks(s, blobs, full, step);
    this.root.visible = this.mesh.visible || this.ticks.visible || this.ribbons.visible;
    return discs || ticks;
  }

  /** The layered shells' ticks (state/shellCracks.ts); true when they changed. */
  private syncTicks(s: State, blobs: BlobStore, full: FullPositions | null, step: StepValues | null): boolean {
    const at = s.phase.at;
    const field = s.cracks.on ? shellCrackFieldOf(s) : null;
    if (!field || field.groups.length === 0 || !s.mesh?.nodes || at?.kind !== "results") {
      this.tickPoints = null;
      return this.hideTicks();
    }
    const layers = step?.layers;
    const here = step && step.stage === at.stage && step.step === at.step && layers && field.groups.every((g) => layers[g.id]) ? step : null;
    if (!here) {
      // As the discs: a scrub within the stage keeps the last ticks until the step arrives.
      const failed = s.resultStep?.stage === at.stage && s.resultStep.step === at.step && s.resultStep.status === "failed";
      if (this.tickStage !== at.stage || failed) return this.hideTicks();
      return false;
    }
    const st = s.results!.stages[at.stage]!;
    const deps = [s.results, at.stage, s.visibility.hidden, s.mesh, s.decls, s.stageElements, s.sequence, field.groups.map((g) => g.id).join(",")];
    if (!this.tickPoints || this.tickPoints.deps.some((d, i) => d !== deps[i])) {
      const points = shellCrackPointsOf(s, blobs, field);
      const nodes = s.mesh.nodes;
      const base = points ? gaussBasePositions(points.layout, blobs.get(nodes.ids) as Float64Array, blobs.get(nodes.xyz) as Float64Array) : null;
      // A tick is 0.35 x the element across (CRACK_PLOTS_SPEC.md §3.2): 0.7 x the Gauss pitch, half of it each way.
      this.tickPoints = { deps, gen: (this.tickPoints?.gen ?? 0) + 1, points, base, half: base ? (0.35 * crackHalfSize(base)) / 0.4 : 1 };
    }
    const { points, base, half } = this.tickPoints;
    const disp = full && full.key !== "base" && full.disp?.nodal ? full.disp : null;
    const c = s.cracks;
    const nextKey = [this.tickPoints.gen, disp ? full!.key : "base", s.deform.scale, c.minOpening, c.scale, c.face, c.style].join("|");
    if (this.tickKey === nextKey && here === this.tickStep) return false;
    this.tickKey = nextKey;
    this.tickStep = here;
    this.tickStage = at.stage;
    if (!points || !base) {
      this.ticks.visible = false;
      this.ribbons.visible = false;
      return true;
    }
    const groups: ShellTickGroup[] = field.groups.map((fg) => {
      const g = st.layers!.find((x) => x.id === fg.id)!;
      const v = layers![fg.id]!;
      return {
        values: v.values,
        layers: v.layers,
        nLayers: g.nLayers,
        nGauss: g.nGauss,
        quaternions: blobs.get(g.quaternions!) as Float64Array,
        recorded: blobs.get(g.recorded!) as Uint8Array,
        thickness: g.thickness ? (blobs.get(g.thickness) as Float64Array) : null,
      };
    });
    const pitch = half / 0.35;
    if (c.style === "lines" || c.style === "continuous") {
      this.ticks.visible = false;
      this.drawnShell = { ticks: 0, points: 0, unframed: 0, maxOpening: 0 };
      // Trace once per step, face and threshold, in the reference geometry; a deformation only moves the vertices.
      const traceKey = [this.tickPoints.gen, here, c.minOpening, c.face, c.style];
      if (!this.traced || this.traced.key.some((k, i) => k !== traceKey[i])) {
        const elementIds = field.groups.map((fg) => blobs.get(st.layers!.find((x) => x.id === fg.id)!.elementIds!) as Float64Array);
        const lines = buildCrackLines(
          { base, points, groups, elementIds, corners: elementCornersOf(s, blobs) },
          { face: c.face, minOpening: c.minOpening, pitch, eps: 0.02 * half, continuous: c.style === "continuous" },
        );
        const layout = lineVertexLayoutOf(s, blobs, lines.elementId, lines.xi);
        const nodes = s.mesh.nodes;
        const vbase = gaussBasePositions(layout.layout, blobs.get(nodes.ids) as Float64Array, blobs.get(nodes.xyz) as Float64Array);
        this.traced = { key: traceKey, lines, layout, base: vbase };
      }
      const { lines, layout, base: vbase } = this.traced;
      const placedAt = disp ? deformedPositions(vbase, gaussDisplacements(layout.layout, disp.nodal!.column, disp.nodal!.rows), s.deform.scale) : vbase;
      const atV = new Float32Array(3 * lines.elementId.length).fill(NaN);
      layout.placed.forEach((k, v) => {
        if (k >= 0) atV.set(placedAt.subarray(3 * k, 3 * k + 3), 3 * v);
      });
      const rb = buildRibbons(lines, atV, 0.008 * pitch * c.scale, 0.05 * pitch * c.scale, c.minOpening);
      this.drawnLines = { lines: lines.face.length, segments: rb.segments, panels: lines.panels, maxOpening: lines.maxOpening };
      for (let i = 0; i < rb.colours.length; i++) rb.colours[i] = toLinear(rb.colours[i]!);
      this.ribbonGeometry.setAttribute("position", new THREE.BufferAttribute(rb.positions, 3));
      this.ribbonGeometry.setAttribute("color", new THREE.BufferAttribute(rb.colours, 3));
      this.ribbonGeometry.computeBoundingSphere();
      this.ribbons.visible = rb.segments > 0;
      return true;
    }
    this.ribbons.visible = false;
    this.traced = null;
    this.drawnLines = { lines: 0, segments: 0, panels: 0, maxOpening: 0 };
    const at3 = disp ? deformedPositions(base, gaussDisplacements(points.layout, disp.nodal!.column, disp.nodal!.rows), s.deform.scale) : base;
    const r = buildShellTicks(at3, points, groups, { face: c.face, minOpening: c.minOpening, halfLength: half * c.scale, eps: 0.02 * half });
    this.drawnShell = { ticks: r.ticks, points: r.points, unframed: r.unframed, maxOpening: r.maxOpening };
    for (let i = 0; i < r.colours.length; i++) r.colours[i] = toLinear(r.colours[i]!);
    this.tickGeometry.setAttribute("position", new THREE.BufferAttribute(r.positions, 3));
    this.tickGeometry.setAttribute("color", new THREE.BufferAttribute(r.colours, 3));
    this.tickGeometry.computeBoundingSphere();
    this.ticks.visible = r.ticks > 0;
    return true;
  }

  /** The solids' discs (state/cracks.ts); true when they changed. */
  private syncDiscs(s: State, blobs: BlobStore, full: FullPositions | null, step: StepValues | null): boolean {
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
    this.mesh.visible = r.discs > 0;
    return true;
  }

  setClippingPlanes(planes: THREE.Plane[]): void {
    this.material.clippingPlanes = planes;
    this.material.needsUpdate = true;
    this.tickMaterial.clippingPlanes = planes;
    this.tickMaterial.needsUpdate = true;
    this.ribbonMaterial.clippingPlanes = planes;
    this.ribbonMaterial.needsUpdate = true;
  }

  dispose(): void {
    this.geometry.dispose();
    this.material.dispose();
    this.tickGeometry.dispose();
    this.tickMaterial.dispose();
    this.ribbonGeometry.dispose();
    this.ribbonMaterial.dispose();
  }
}
