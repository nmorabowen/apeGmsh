// 3-D structural loading and prescribed-SP layer: renders `buildLoads3D`
// (`state/loads.ts`) as `THREE.LineSegments` (shafts, moment double-head chevrons,
// beam load combs, SP ticks) and `THREE.Mesh` (arrowheads) with vertex colours,
// section-clip plane forwarding, theme awareness, and deformed-mesh support.

import * as THREE from "three";
import type { BlobStore } from "../state/blobs.ts";
import { buildLoads3D } from "../state/loads.ts";
import type { State } from "../state/types.ts";
import type { ThemeMode } from "../theme/tokens.ts";
import { toLinear } from "./contour.ts";
import type { FullPositions } from "./deform.ts";

function srgbToLinearArray(src: Float32Array): Float32Array {
  const out = new Float32Array(src.length);
  for (let i = 0; i < src.length; i++) {
    out[i] = toLinear(src[i]!);
  }
  return out;
}

export class LoadsLayer {
  readonly root = new THREE.Group();
  readonly lines: THREE.LineSegments;
  readonly cones: THREE.Mesh;

  private readonly lineMaterial: THREE.LineBasicMaterial;
  private readonly coneMaterial: THREE.MeshBasicMaterial;
  private theme: ThemeMode = "dark";
  private appliedKey: string | null = null;

  constructor() {
    this.lineMaterial = new THREE.LineBasicMaterial({
      vertexColors: true,
    });
    this.coneMaterial = new THREE.MeshBasicMaterial({
      vertexColors: true,
      side: THREE.DoubleSide,
      polygonOffset: true,
      polygonOffsetFactor: -1,
      polygonOffsetUnits: -1,
    });

    this.lines = new THREE.LineSegments(new THREE.BufferGeometry(), this.lineMaterial);
    this.cones = new THREE.Mesh(new THREE.BufferGeometry(), this.coneMaterial);

    this.root.add(this.lines, this.cones);
    this.root.visible = false;
  }

  setTheme(mode: ThemeMode): void {
    this.theme = mode;
  }

  sync(s: State, blobs: BlobStore, full: FullPositions | null = null): boolean {
    if (!s.loads.on || !s.mesh || s.phase.at?.kind === "geometry") {
      const wasVisible = this.root.visible;
      this.root.visible = false;
      this.appliedKey = null;
      return wasVisible;
    }

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
      s.loads.selectedPatterns === null ? "*" : s.loads.selectedPatterns.join(","),
      s.contour.sourceUnits ?? "",
      s.contour.targetUnits ?? "",
      this.theme,
    ].join("|");

    if (this.root.visible && this.appliedKey === nextKey) {
      return false;
    }

    const geom = buildLoads3D(s, blobs, {
      theme: this.theme,
      ...(full
        ? {
            drawnLinePositions: full.line,
            drawnTriPositions: full.tri,
          }
        : {}),
    });

    if (geom.linePositions.length === 0 && geom.triPositions.length === 0) {
      const wasVisible = this.root.visible;
      this.root.visible = false;
      this.appliedKey = null;
      return wasVisible;
    }

    const lineGeom = new THREE.BufferGeometry();
    lineGeom.setAttribute("position", new THREE.BufferAttribute(geom.linePositions, 3));
    lineGeom.setAttribute("color", new THREE.BufferAttribute(srgbToLinearArray(geom.lineColors), 3));
    if (geom.linePositions.length > 0) lineGeom.computeBoundingSphere();
    this.lines.geometry.dispose();
    this.lines.geometry = lineGeom;
    this.lines.visible = geom.linePositions.length > 0;

    const coneGeom = new THREE.BufferGeometry();
    coneGeom.setAttribute("position", new THREE.BufferAttribute(geom.triPositions, 3));
    coneGeom.setAttribute("color", new THREE.BufferAttribute(srgbToLinearArray(geom.triColors), 3));
    if (geom.triPositions.length > 0) coneGeom.computeBoundingSphere();
    this.cones.geometry.dispose();
    this.cones.geometry = coneGeom;
    this.cones.visible = geom.triPositions.length > 0;

    this.root.visible = true;
    this.appliedKey = nextKey;
    return true;
  }

  setClippingPlanes(planes: THREE.Plane[]): void {
    for (const mat of [this.lineMaterial, this.coneMaterial]) {
      mat.clippingPlanes = planes;
      mat.needsUpdate = true;
    }
  }

  dispose(): void {
    this.lines.geometry.dispose();
    this.cones.geometry.dispose();
    this.lineMaterial.dispose();
    this.coneMaterial.dispose();
  }
}
