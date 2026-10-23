// 3-D line diagram layer (axial N, shear Vy/Vz, torsion T, moment My/Mz):
// builds ribbon, profile curve, and riser segments from `buildLineDiagram3D`
// (`state/lineDiagram.ts`) and renders them in a single `THREE.Group` with
// sign-based or continuous colormaps, selection dimming, deformed baseline
// support, and section-clip plane forwarding.

import * as THREE from "three";
import type { DisplacementData } from "../effects.ts";
import type { LadrunoLineStationStep } from "../reader/ladruno.ts";
import type { BlobStore } from "../state/blobs.ts";
import {
  buildLineDiagram3D,
  colorForValue,
  effectiveFlipSign,
  joinLadrunoElements,
  lineDiagramClimOf,
  lineDiagramKeysOf,
  projectHoverNeedle,
  type BeamDiagramSpan3D,
} from "../state/lineDiagram.ts";
import { deformView, resultsPairing } from "../state/selectors.ts";
import type { LineDiagramCmap, State } from "../state/types.ts";
import type { ThemeMode } from "../theme/tokens.ts";
import { toLinear } from "./contour.ts";
import { SELECTION_COLOUR } from "./selection.ts";
import { elementMaskOf, visibleMaps } from "./visible.ts";

/** Unselected beam dimming multiplier in linear space (matches viewport.ts). */
const DIM = 0.07;

export type LineDiagramDeformInput =
  | { dispKey: string; line: Float64Array | Float32Array }
  | { key: string; line: Float32Array }
  | DisplacementData
  | null
  | undefined;

function buildTriLinearColors(
  srgbTriColors: Float32Array,
  triBeamIndex: Int32Array,
  beams: readonly BeamDiagramSpan3D[],
  selectedDecls: readonly string[],
): Float32Array {
  const out = new Float32Array(srgbTriColors.length);
  const selectedSet = selectedDecls.length > 0 ? new Set(selectedDecls) : null;
  const nTri = triBeamIndex.length;
  for (let t = 0; t < nTri; t++) {
    const bIdx = triBeamIndex[t]!;
    const decl = beams[bIdx]?.decl;
    const dim = selectedSet !== null && (decl === undefined || !selectedSet.has(decl)) ? DIM : 1;
    const base = 9 * t;
    for (let j = 0; j < 9; j++) {
      out[base + j] = toLinear(srgbTriColors[base + j]!) * dim;
    }
  }
  return out;
}

function buildSegmentLinearColors(
  segValues: Float32Array,
  segBeamIndex: Int32Array,
  beams: readonly BeamDiagramSpan3D[],
  selectedDecls: readonly string[],
  cmap: LineDiagramCmap,
  clim: readonly [number, number],
  theme: ThemeMode = "dark",
): Float32Array {
  const nSeg = segBeamIndex.length;
  const out = new Float32Array(nSeg * 6);
  const selectedSet = selectedDecls.length > 0 ? new Set(selectedDecls) : null;
  for (let s = 0; s < nSeg; s++) {
    const bIdx = segBeamIndex[s]!;
    const decl = beams[bIdx]?.decl;
    const dim = selectedSet !== null && (decl === undefined || !selectedSet.has(decl)) ? DIM : 1;
    const v0 = segValues[2 * s]!;
    const v1 = segValues[2 * s + 1]!;
    const c0 = colorForValue(v0, cmap, clim[0], clim[1], 0, theme);
    const c1 = colorForValue(v1, cmap, clim[0], clim[1], 0, theme);
    const base = 6 * s;
    out[base] = toLinear(c0[0]) * dim;
    out[base + 1] = toLinear(c0[1]) * dim;
    out[base + 2] = toLinear(c0[2]) * dim;
    out[base + 3] = toLinear(c1[0]) * dim;
    out[base + 4] = toLinear(c1[1]) * dim;
    out[base + 5] = toLinear(c1[2]) * dim;
  }
  return out;
}

export class LineDiagramLayer {
  readonly root = new THREE.Group();
  get group(): THREE.Group {
    return this.root;
  }
  readonly mesh: THREE.Mesh;
  readonly profileLines: THREE.LineSegments;
  readonly riserLines: THREE.LineSegments;
  readonly hoverNeedle: THREE.LineSegments;

  private readonly meshMaterial: THREE.MeshBasicMaterial;
  private readonly profileMaterial: THREE.LineBasicMaterial;
  private readonly riserMaterial: THREE.LineBasicMaterial;
  private readonly hoverMaterial: THREE.LineBasicMaterial;
  private camera: THREE.Camera | null;
  private theme: ThemeMode = "dark";

  private lastState: State | null = null;
  private lastBlobs: BlobStore | null = null;
  private lastStep: { key: string; step: LadrunoLineStationStep } | null = null;
  private lastDeform: LineDiagramDeformInput = null;
  private lastVisibleLineMap: Int32Array | undefined = undefined;
  private lastBeams: readonly BeamDiagramSpan3D[] = [];
  private lastEffectiveScale = 0;
  private appliedKey: string | null = null;
  private appliedHoverKey: string | null = null;

  constructor(camera: THREE.Camera | null = null) {
    this.camera = camera;
    this.meshMaterial = new THREE.MeshBasicMaterial({
      vertexColors: true,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.55,
      depthWrite: false,
      polygonOffset: true,
      polygonOffsetFactor: -1,
      polygonOffsetUnits: -1,
    });
    this.profileMaterial = new THREE.LineBasicMaterial({
      vertexColors: true,
    });
    this.riserMaterial = new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.65,
    });
    this.hoverMaterial = new THREE.LineBasicMaterial({
      color: SELECTION_COLOUR,
      depthTest: false,
    });

    this.mesh = new THREE.Mesh(new THREE.BufferGeometry(), this.meshMaterial);
    this.profileLines = new THREE.LineSegments(new THREE.BufferGeometry(), this.profileMaterial);
    this.riserLines = new THREE.LineSegments(new THREE.BufferGeometry(), this.riserMaterial);
    this.hoverNeedle = new THREE.LineSegments(new THREE.BufferGeometry(), this.hoverMaterial);
    this.hoverNeedle.renderOrder = 2;
    this.hoverNeedle.visible = false;

    this.root.add(this.mesh, this.profileLines, this.riserLines, this.hoverNeedle);
    this.root.visible = false;
  }

  setCamera(camera: THREE.Camera | null): void {
    this.camera = camera;
  }

  setTheme(mode: ThemeMode): void {
    this.theme = mode;
  }

  private getViewDir(): [number, number, number] {
    if (!this.camera) return [0, 0, -1];
    const d = this.camera.getWorldDirection(new THREE.Vector3());
    return [d.x, d.y, d.z];
  }

  private hoverKeyOf(s: State): string {
    if (!s.hover || typeof s.hover === "string" || !s.hover.at) {
      if (s.lineDiagram.hover) {
        const h = s.lineDiagram.hover;
        return `ro:${h.decl}|${h.pBase.join(",")}|${h.pTop.join(",")}`;
      }
      return "none";
    }
    return `pick:${s.hover.decl}|${s.hover.at[0]},${s.hover.at[1]},${s.hover.at[2]}`;
  }

  private syncHoverNeedle(s: State, beams: readonly BeamDiagramSpan3D[], effectiveScale: number): void {
    let pBase: readonly [number, number, number] | null = null;
    let pTop: readonly [number, number, number] | null = null;

    if (s.hover && typeof s.hover !== "string" && s.hover.at) {
      const hoverDecl = s.hover.decl;
      const hitPoint = s.hover.at;
      const candidates = beams.filter((b) => b.decl === hoverDecl);
      if (candidates.length > 0) {
        let beam = candidates[0]!;
        if (candidates.length > 1) {
          let bestDist = Infinity;
          for (const c of candidates) {
            const dx = c.pEnd[0] - c.pStart[0];
            const dy = c.pEnd[1] - c.pStart[1];
            const dz = c.pEnd[2] - c.pStart[2];
            const lenSq = dx * dx + dy * dy + dz * dz;
            const t =
              lenSq > 1e-24
                ? Math.max(0, Math.min(1, ((hitPoint[0] - c.pStart[0]) * dx + (hitPoint[1] - c.pStart[1]) * dy + (hitPoint[2] - c.pStart[2]) * dz) / lenSq))
                : 0;
            const qx = c.pStart[0] + t * dx - hitPoint[0];
            const qy = c.pStart[1] + t * dy - hitPoint[1];
            const qz = c.pStart[2] + t * dz - hitPoint[2];
            const d = qx * qx + qy * qy + qz * qz;
            if (d < bestDist) {
              bestDist = d;
              beam = c;
            }
          }
        }
        const needle = projectHoverNeedle(beam, hitPoint, effectiveScale, effectiveFlipSign(s.lineDiagram));
        pBase = needle.pBase;
        pTop = needle.pTop;
      }
    } else if (s.lineDiagram.hover) {
      pBase = s.lineDiagram.hover.pBase;
      pTop = s.lineDiagram.hover.pTop;
    }

    const needleGeom = new THREE.BufferGeometry();
    if (pBase && pTop) {
      const pos = new Float32Array([
        pBase[0],
        pBase[1],
        pBase[2],
        pTop[0],
        pTop[1],
        pTop[2],
      ]);
      needleGeom.setAttribute("position", new THREE.BufferAttribute(pos, 3));
      needleGeom.computeBoundingSphere();
      this.hoverNeedle.geometry.dispose();
      this.hoverNeedle.geometry = needleGeom;
      this.hoverNeedle.visible = true;
    } else {
      needleGeom.setAttribute("position", new THREE.BufferAttribute(new Float32Array(0), 3));
      this.hoverNeedle.geometry.dispose();
      this.hoverNeedle.geometry = needleGeom;
      this.hoverNeedle.visible = false;
    }
  }

  sync(
    s: State,
    blobs: BlobStore,
    step: { key: string; step: LadrunoLineStationStep } | null,
    deform?: LineDiagramDeformInput,
    forceCamera = false,
    visibleLineMap?: Int32Array,
  ): boolean {
    this.lastState = s;
    this.lastBlobs = blobs;
    this.lastStep = step;
    this.lastDeform = deform;
    this.lastVisibleLineMap = visibleLineMap;

    const ldKeys = lineDiagramKeysOf(s);
    if (
      !s.lineDiagram.on ||
      !s.mesh ||
      s.phase.at?.kind !== "results" ||
      !s.results ||
      resultsPairing(s).state !== "paired" ||
      !step ||
      !ldKeys ||
      step.key !== ldKeys.step
    ) {
      const wasVisible = this.root.visible;
      this.root.visible = false;
      this.hoverNeedle.visible = false;
      this.lastBeams = [];
      this.appliedKey = null;
      this.appliedHoverKey = null;
      return wasVisible;
    }

    const join = joinLadrunoElements(s.decls, resultsPairing(s), s.results.elementIndexSpace);
    if (!join.ok) {
      const wasVisible = this.root.visible;
      this.root.visible = false;
      this.hoverNeedle.visible = false;
      this.lastBeams = [];
      this.appliedKey = null;
      this.appliedHoverKey = null;
      return wasVisible;
    }

    const basePos = blobs.f32(s.mesh.linePositions);
    let deformKey = "undeformed";
    let drawnLinePositions = basePos;
    let deformApplied = false;
    if (s.deform.on) {
      const dv = deformView(s);
      if (deform) {
        if ("dispKey" in deform) {
          const rawKey = deform.dispKey;
          const rawLine = deform.line;
          const validDeform =
            dv.state === "ready" || dv.state === "partial" || dv.state === "loading"
              ? rawKey === dv.keys.disp
              : false;
          if (validDeform && rawLine.length === basePos.length) {
            deformKey = `${rawKey}|${s.deform.scale}`;
            drawnLinePositions = new Float32Array(basePos.length);
            const scale = s.deform.scale;
            for (let i = 0; i < basePos.length; i++) {
              drawnLinePositions[i] = basePos[i]! + scale * rawLine[i]!;
            }
            deformApplied = true;
          }
        } else if (deform.line instanceof Float32Array) {
          // Pre-deformed full-mesh positions from Viewport (`this.full`)
          const validDeform =
            deform.key !== "base" &&
            (dv.state === "ready" || dv.state === "partial" || dv.state === "loading"
              ? deform.key === dv.keys.disp || deform.key === `${dv.keys.disp}|scale ${s.deform.scale}`
              : false);
          if (validDeform && deform.line.length === basePos.length) {
            deformKey = deform.key;
            drawnLinePositions = deform.line;
            deformApplied = true;
          }
        } else {
          const rawKey = deform.key;
          const rawLine = deform.line.d;
          const validDeform =
            dv.state === "ready" || dv.state === "partial" || dv.state === "loading"
              ? rawKey === dv.keys.disp
              : false;
          if (validDeform && rawLine.length === basePos.length) {
            deformKey = `${rawKey}|${s.deform.scale}`;
            drawnLinePositions = new Float32Array(basePos.length);
            const scale = s.deform.scale;
            for (let i = 0; i < basePos.length; i++) {
              drawnLinePositions[i] = basePos[i]! + scale * rawLine[i]!;
            }
            deformApplied = true;
          }
        }
      }
      if (!deformApplied) {
        const wasVisible = this.root.visible;
        this.root.visible = false;
        this.appliedKey = null;
        return wasVisible;
      }
    }

    const viewDir = this.getViewDir();
    const clim = lineDiagramClimOf(s);
    const viewKey =
      s.lineDiagram.plane === "screen"
        ? `${viewDir[0].toFixed(5)},${viewDir[1].toFixed(5)},${viewDir[2].toFixed(5)}`
        : s.lineDiagram.plane;
    const nextKey = [
      step.key,
      s.mesh.linePositions.id,
      s.visibility.hidden.join(","),
      s.selection.decls.join(","),
      viewKey,
      s.lineDiagram.scale,
      s.lineDiagram.cmap,
      s.lineDiagram.fill ? "fill" : "nofill",
      String(s.lineDiagram.flipSign),
      String(clim.min),
      String(clim.max),
      deformKey,
      this.theme,
    ].join("|");
    const hoverKey = this.hoverKeyOf(s);

    if (!forceCamera && this.root.visible && this.appliedKey === nextKey) {
      if (this.appliedHoverKey === hoverKey) {
        return false;
      }
      this.syncHoverNeedle(s, this.lastBeams, this.lastEffectiveScale);
      this.appliedHoverKey = hoverKey;
      return true;
    }

    const visibleIndices =
      visibleLineMap ??
      visibleMaps(
        s.mesh.legend,
        s.visibility.hidden,
        blobs.i32(s.mesh.lineGroup),
        blobs.i32(s.mesh.triGroup),
        elementMaskOf(s, s.mesh, blobs),
      ).lineMap;

    const climTuple: readonly [number, number] | null =
      clim.min !== null && clim.max !== null ? [clim.min, clim.max] : null;

    const geom = buildLineDiagram3D({
      decls: s.decls,
      meshElements: s.mesh.elements,
      lineElement: blobs.i32(s.mesh.lineElement),
      drawnLinePositions,
      lineNodes: blobs.get(s.mesh.vertexNodes.line) as Float64Array,
      visibleLineMap: visibleIndices,
      stepData: step.step,
      join,
      ndm: s.ndm.ndm,
      charLength: 0.8 * s.mesh.radius,
      deformed: deformApplied,
      scale: s.lineDiagram.scale,
      flipSign: s.lineDiagram.flipSign,
      cmap: s.lineDiagram.cmap,
      plane: s.lineDiagram.plane,
      cameraViewDir: viewDir,
      clim: climTuple,
      theme: this.theme,
    });

    if (geom.beams.length === 0) {
      const wasVisible = this.root.visible;
      this.root.visible = false;
      this.hoverNeedle.visible = false;
      this.lastBeams = [];
      this.lastEffectiveScale = 0;
      this.appliedKey = null;
      this.appliedHoverKey = null;
      return wasVisible;
    }
    this.lastBeams = geom.beams;
    this.lastEffectiveScale = geom.effectiveScale;

    const ribbonColors = buildTriLinearColors(
      geom.triColors,
      geom.triBeamIndex,
      geom.beams,
      s.selection.decls,
    );
    const profileColors = buildSegmentLinearColors(
      geom.profileValues,
      geom.profileBeamIndex,
      geom.beams,
      s.selection.decls,
      s.lineDiagram.cmap,
      geom.clim,
      this.theme,
    );
    const riserColors = buildSegmentLinearColors(
      geom.riserValues,
      geom.riserBeamIndex,
      geom.beams,
      s.selection.decls,
      s.lineDiagram.cmap,
      geom.clim,
      this.theme,
    );

    const meshGeom = new THREE.BufferGeometry();
    meshGeom.setAttribute("position", new THREE.BufferAttribute(geom.triPositions, 3));
    meshGeom.setAttribute("color", new THREE.BufferAttribute(ribbonColors, 3));
    if (geom.triPositions.length > 0) meshGeom.computeBoundingSphere();
    this.mesh.geometry.dispose();
    this.mesh.geometry = meshGeom;
    this.mesh.visible = s.lineDiagram.fill;

    const profGeom = new THREE.BufferGeometry();
    profGeom.setAttribute("position", new THREE.BufferAttribute(geom.profilePositions, 3));
    profGeom.setAttribute("color", new THREE.BufferAttribute(profileColors, 3));
    if (geom.profilePositions.length > 0) profGeom.computeBoundingSphere();
    this.profileLines.geometry.dispose();
    this.profileLines.geometry = profGeom;

    const riserGeom = new THREE.BufferGeometry();
    riserGeom.setAttribute("position", new THREE.BufferAttribute(geom.riserPositions, 3));
    riserGeom.setAttribute("color", new THREE.BufferAttribute(riserColors, 3));
    if (geom.riserPositions.length > 0) riserGeom.computeBoundingSphere();
    this.riserLines.geometry.dispose();
    this.riserLines.geometry = riserGeom;

    this.syncHoverNeedle(s, geom.beams, geom.effectiveScale);
    this.root.visible = true;
    this.appliedKey = nextKey;
    this.appliedHoverKey = hoverKey;
    return true;
  }

  setClippingPlanes(planes: THREE.Plane[]): void {
    for (const mat of [this.meshMaterial, this.profileMaterial, this.riserMaterial, this.hoverMaterial]) {
      mat.clippingPlanes = planes;
      mat.needsUpdate = true;
    }
  }

  onCameraChange(): boolean {
    if (!this.root.visible || !this.lastState || !this.lastBlobs || !this.lastStep) return false;
    if (this.lastState.lineDiagram.plane !== "screen") return false;
    return this.sync(this.lastState, this.lastBlobs, this.lastStep, this.lastDeform, true, this.lastVisibleLineMap);
  }

  dispose(): void {
    this.mesh.geometry.dispose();
    this.profileLines.geometry.dispose();
    this.riserLines.geometry.dispose();
    this.hoverNeedle.geometry.dispose();
    this.meshMaterial.dispose();
    this.profileMaterial.dispose();
    this.riserMaterial.dispose();
    this.hoverMaterial.dispose();
  }
}
