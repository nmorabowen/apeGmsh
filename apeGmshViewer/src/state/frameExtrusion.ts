// Pure 3-D cross-section prism extrusion and rigid joint-offset geometry
// builder for 1-D OpenSees frame elements.
//
// Operates on plain declarations (`Decl`) and neutral-zone 1-D line segments.
// Every output buffer is a TypedArray destined for `BlobStore` (never stored
// directly in `State`), and every vertex records its owning node ID (`nodeI`
// or `nodeJ`) so `redeform` translates both the rigid offset link and the
// extruded prism end-cap by the exact same nodal displacement vector.

import type { ElementRef } from "../chain/resolve.ts";
import { ELEMENT_SYNTAX, INTEGRATION_SYNTAX } from "../chain/signatures.ts";
import type { ElementMeta, Param, SectionProfile } from "../model/types.ts";
import { triangulateSectionProfile } from "../mesh/polygon.ts";
import { equivalentRectProfile } from "../reader/equivalentSection.ts";
import { cellPath, opsRowPath } from "./decls.ts";
import {
  beamLocalAxes,
  collectTransformJntOffsets,
  resolveElementJointOffset,
  type ElementJointOffset,
} from "./lineDiagram.ts";
import type { Decl, DeclPath, Definitions } from "./types.ts";

export { collectTransformJntOffsets, resolveElementJointOffset, type ElementJointOffset };

export interface FrameExtrusionInput {
  readonly linePositions: Float32Array;
  readonly lineNodes: Float64Array;
  readonly lineElementIds: Int32Array;
  readonly lineRefs: readonly ElementRef[];
  readonly decls: Readonly<Record<DeclPath, Decl>>;
  readonly definitions?: Definitions | null;
  readonly elementMeta?: readonly ElementMeta[] | null;
  readonly ndm?: 2 | 3 | null;
  readonly stepLocalAxesZRec?: ReadonlyMap<number, readonly [number, number, number]> | null;
  readonly modelLocalAxesZRec?: ReadonlyMap<number, readonly [number, number, number]> | null;
}

export interface FrameExtrusionData {
  /** (nTri * 3 * 3) float32 xyz positions of extruded prism triangles */
  readonly frameTriPositions: Float32Array;
  /** (nTri * 3) float64 node IDs for deformed-shape translation */
  readonly frameTriNodes: Float64Array;
  /** (nTri) int32 index into `lineRefs` / `lineElementIds` (0..nSeg-1) */
  readonly frameTriSegments: Int32Array;
  /** (nTri) int32 FEM element ID per triangle (-1 for inline ops elements) */
  readonly frameTriElementIds: Int32Array;
  /** ElementRef per extruded triangle for raycast picking in `selection.ts` */
  readonly frameTriRefs: ElementRef[];
  /** (nEdge * 2 * 3) float32 xyz positions of prism feature & cap edges */
  readonly frameEdgePositions: Float32Array;
  /** (nEdge * 2) float64 node IDs for deformed-shape translation */
  readonly frameEdgeNodes: Float64Array;
  /** (nEdge) int32 index into `lineRefs` (0..nSeg-1) */
  readonly frameEdgeSegments: Int32Array;
  /** (nSeg) uint8 mask: 1 if segment `s` has extruded 3-D prism geometry */
  readonly extrudedSegmentMask: Uint8Array;
  /** (nSeg * 2 * 3) float32 xyz positions of 1-D segments shifted to flexible clear span [pI', pJ'] when offsets exist */
  readonly clearLinePositions: Float32Array;
  /** (nOffsetSeg * 2 * 3) float32 xyz positions of rigid joint-offset links */
  readonly offsetLinePositions: Float32Array;
  /** (nOffsetSeg * 2) float64 node IDs for deformed-shape translation */
  readonly offsetLineNodes: Float64Array;
  /** (nOffsetSeg) int32 index into `lineRefs` (0..nSeg-1) */
  readonly offsetLineSegments: Int32Array;
  /** (nOffsetPt * 3) float32 xyz positions of joint-offset end markers */
  readonly offsetJointPositions: Float32Array;
  /** (nOffsetPt) float64 node IDs for deformed-shape translation */
  readonly offsetJointNodes: Float64Array;
  /** (nOffsetPt) int32 index into `lineRefs` (0..nSeg-1) */
  readonly offsetJointSegments: Int32Array;
  readonly hasExtrudableFrames: boolean;
  readonly hasJointOffsets: boolean;
  /** Count of 1-D segments with a non-axisymmetric section skipped because no 3-D orientation (vecxz / localAxesZRec) was available */
  readonly skippedUnorientedCount: number;
}

/** Collect emitted `transfTag -> vecxz` across all 3-D `geomTransf` declarations, plus the set of 2-D transform tags. */
function collectTransformAxesInfo(decls: Readonly<Record<DeclPath, Decl>>): {
  vecxzByTag: Map<number, [number, number, number]>;
  transf2DTags: Set<number>;
  transfDeclByTag: Map<number, Decl>;
} {
  const vecxzByTag = new Map<number, [number, number, number]>();
  const transf2DTags = new Set<number>();
  const transfDeclByTag = new Map<number, Decl>();
  for (const d of Object.values(decls)) {
    if (d.kind !== "geomTransf") continue;
    if (d.tag !== null) transfDeclByTag.set(d.tag, d);
    if (d.jntOffset) {
      for (const row of d.jntOffset) transfDeclByTag.set(row.tag, d);
    }
    if (d.vecxz && d.vecxz.length > 0) {
      for (const row of d.vecxz) {
        vecxzByTag.set(row.tag, [row.vecxz[0], row.vecxz[1], row.vecxz[2]]);
        transfDeclByTag.set(row.tag, d);
      }
      continue;
    }
    if (
      d.tag !== null &&
      (d.params.length === 3 ||
        d.params.length === 9 ||
        (d.params.length === 10 && d.params[3] === "-jntOffset")) &&
      typeof d.params[0] === "number" &&
      typeof d.params[1] === "number" &&
      typeof d.params[2] === "number" &&
      Number.isFinite(d.params[0]) &&
      Number.isFinite(d.params[1]) &&
      Number.isFinite(d.params[2])
    ) {
      vecxzByTag.set(d.tag, [d.params[0], d.params[1], d.params[2]]);
      continue;
    }
    // A geomTransf with no 3-D vecxz is a 2-D transform
    if (d.tag !== null) transf2DTags.add(d.tag);
    if (d.jntOffset) {
      for (const row of d.jntOffset) transf2DTags.add(row.tag);
    }
  }
  return { vecxzByTag, transf2DTags, transfDeclByTag };
}

function leadingNumCount(args: readonly Param[]): number {
  let n = 0;
  while (n < args.length && typeof args[n] === "number" && !Number.isNaN(args[n])) n++;
  return n;
}

function rectLoop(yMin: number, yMax: number, zMin: number, zMax: number): [number, number][] {
  return [
    [yMin, zMin],
    [yMax, zMin],
    [yMax, zMax],
    [yMin, zMax],
  ];
}

function circleLoop(r: number, nSeg = 24): [number, number][] {
  const pts: [number, number][] = [];
  for (let i = 0; i < nSeg; i++) {
    const theta = (2 * Math.PI * i) / nSeg;
    pts.push([r * Math.cos(theta), r * Math.sin(theta)]);
  }
  return pts;
}

function inlineElementProfile(metaType: string, args: readonly Param[]): SectionProfile | null {
  const n = leadingNumCount(args);
  if (metaType === "elasticBeamColumn" || metaType === "LadrunoIMKBeam") {
    const is2D = n === 4 || (metaType === "LadrunoIMKBeam" && n === 6);
    const is3D = n === 7 || (metaType === "LadrunoIMKBeam" && n === 9);
    if (is2D) {
      const A = args[0] as number;
      const Iz = args[2] as number;
      return equivalentRectProfile(A, Iz);
    } else if (is3D) {
      const A = args[0] as number;
      const Iy = args[4] as number;
      const Iz = args[5] as number;
      return equivalentRectProfile(A, Iz, Iy);
    }
  } else if (metaType === "ElasticTimoshenkoBeam" || metaType === "elasticTimoshenkoBeam") {
    if (n === 6) {
      const A = args[2] as number;
      const Iz = args[3] as number;
      return equivalentRectProfile(A, Iz);
    } else if (n === 9) {
      const A = args[2] as number;
      const Iy = args[4] as number;
      const Iz = args[5] as number;
      return equivalentRectProfile(A, Iz, Iy);
    }
  } else if (metaType === "ModElasticBeam2d" && n === 7) {
    const A = args[0] as number;
    const Iz = args[2] as number;
    return equivalentRectProfile(A, Iz);
  } else if ((metaType === "Truss" || metaType === "CorotTruss") && n >= 2) {
    const A = args[0] as number;
    if (typeof A === "number" && Number.isFinite(A) && A > 0) {
      return { loops: [circleLoop(Math.sqrt(A / Math.PI))], axisymmetric: true, source: "elastic" };
    }
  }
  return null;
}

/**
 * The ETABS (2, 3) extent of an `InsertionPoint` as a local (y, z) rectangle,
 * following `joint_offset.insertion_point_offsets`: `mirror_2` flips the 3
 * extent, `mirror_3` flips the 2 extent; `axis_2="y"` maps 2→y, 3→z, while
 * `axis_2="z"` maps 2→z and 3→e1×ez = −y.
 */
function insertionPointRect(
  min2: number,
  max2: number,
  min3: number,
  max3: number,
  axis2: "y" | "z",
  mirror2: boolean,
  mirror3: boolean,
): [number, number][] {
  if (mirror2) [min3, max3] = [-max3, -min3];
  if (mirror3) [min2, max2] = [-max2, -min2];
  return axis2 === "y" ? rectLoop(min2, max2, min3, max3) : rectLoop(-max3, -min3, min2, max2);
}

/**
 * Tier 2: Build a bounding-rectangle `SectionProfile` from an `InsertionPoint`
 * struct in `/opensees/decl_params` (`depth`, `width`, `bbox`, `axis_2`,
 * `mirror_2`, `mirror_3`, or a referenced `section` `$decl`).
 */
export function insertionPointProfileFromFields(
  fields: Readonly<Record<string, unknown>>,
  secByDeclKey: ReadonlyMap<string, Decl>,
): SectionProfile | null {
  const secRef = fields["section"];
  if (secRef && typeof secRef === "object" && !Array.isArray(secRef) && typeof (secRef as Record<string, unknown>)["$decl"] === "string") {
    const secDecl = secByDeclKey.get((secRef as Record<string, unknown>)["$decl"] as string);
    if (secDecl?.sectionProfile) return secDecl.sectionProfile;
  }

  const axis2 = fields["axis_2"] === "z" ? "z" : "y";
  const mirror2 = fields["mirror_2"] === true;
  const mirror3 = fields["mirror_3"] === true;

  const rawBbox = fields["bbox"];
  if (
    Array.isArray(rawBbox) &&
    rawBbox.length === 4 &&
    rawBbox.every((v) => typeof v === "number" && Number.isFinite(v))
  ) {
    const [min2, max2, min3, max3] = rawBbox as [number, number, number, number];
    if (max2 > min2 && max3 > min3) {
      return { loops: [insertionPointRect(min2, max2, min3, max3, axis2, mirror2, mirror3)], source: "parametric" };
    }
  }

  // Python reads a missing width as 0 (`joint_offset._section_bbox`): a
  // zero-thickness section has no prism to draw, so refuse rather than guess.
  const depth = typeof fields["depth"] === "number" && Number.isFinite(fields["depth"]) && fields["depth"] > 0 ? fields["depth"] : null;
  const width = typeof fields["width"] === "number" && Number.isFinite(fields["width"]) && fields["width"] > 0 ? fields["width"] : null;
  if (depth !== null && width !== null) {
    return {
      loops: [insertionPointRect(-depth / 2, depth / 2, -width / 2, width / 2, axis2, mirror2, mirror3)],
      source: "parametric",
    };
  }
  return null;
}

/** Unwrap `Aggregator` (`-section secTag`) or `InsertionPoint` (`secTag -offset ...`) to find a `SectionProfile`. */
function resolveSectionDeclProfile(
  secTag: number,
  secByTag: ReadonlyMap<number, Decl>,
): SectionProfile | null {
  let currTag: number | null = secTag;
  for (let depth = 0; depth < 5 && currTag !== null; depth++) {
    const sec = secByTag.get(currTag);
    if (!sec) return null;
    if (sec.sectionProfile) return sec.sectionProfile;
    if (sec.type === "Aggregator") {
      const idx = sec.params.indexOf("-section");
      const next = idx >= 0 ? sec.params[idx + 1] : undefined;
      currTag = typeof next === "number" && Number.isInteger(next) ? next : null;
    } else if (sec.type === "InsertionPoint") {
      const next = sec.params[0];
      currTag = typeof next === "number" && Number.isInteger(next) ? next : null;
    } else {
      return null;
    }
  }
  return null;
}

export { triangulateSectionProfile };

function arbitraryPerpendicularTriad(
  dx: number,
  dy: number,
  dz: number,
): { xLocal: [number, number, number]; yLocal: [number, number, number]; zLocal: [number, number, number] } | null {
  const L = Math.hypot(dx, dy, dz);
  if (L <= 1e-12) return null;
  const xLocal: [number, number, number] = [dx / L, dy / L, dz / L];
  const ref: [number, number, number] = Math.abs(xLocal[2]) < 0.9 ? [0, 0, 1] : [0, 1, 0];
  return beamLocalAxes(xLocal, ref);
}

/**
 * Build 3-D cross-section prism triangles/edges and rigid joint-offset
 * link/marker buffers across all 1-D OpenSees elements in the model.
 */
export function buildFrameExtrusion(input: FrameExtrusionInput): FrameExtrusionData {
  const {
    linePositions,
    lineNodes,
    lineElementIds,
    lineRefs,
    decls,
    definitions,
    elementMeta,
    ndm,
    stepLocalAxesZRec,
    modelLocalAxesZRec,
  } = input;
  const nSeg = lineRefs.length;
  const metaGroupByPath = new Map<string, ElementMeta>();
  if (elementMeta) {
    for (const m of elementMeta) metaGroupByPath.set(m.path, m);
  }

  const { vecxzByTag, transfDeclByTag } = collectTransformAxesInfo(decls);
  const jntByTransfTag = collectTransformJntOffsets(decls);

  const secByTag = new Map<number, Decl>();
  const integByTag = new Map<number, Decl>();
  const secByDeclKey = new Map<string, Decl>();
  for (const d of Object.values(decls)) {
    if (d.kind === "section" && d.tag !== null) {
      secByTag.set(d.tag, d);
      secByDeclKey.set(d.path, d);
    } else if (d.kind === "beamIntegration" && d.tag !== null) {
      integByTag.set(d.tag, d);
    }
  }

  // Tier 2: Map geomTransf DeclPath -> InsertionPoint SectionProfile from /opensees/decl_params
  const insertionProfileByTransfPath = new Map<DeclPath, SectionProfile>();
  if (definitions?.decls) {
    for (const rec of definitions.decls.records) {
      if (rec.family !== "geomTransf") continue;
      for (const pRow of rec.params) {
        const jntVal = pRow.params["jnt_offset"];
        if (jntVal && typeof jntVal === "object" && !Array.isArray(jntVal) && jntVal["$struct"] === "InsertionPoint") {
          const fields = jntVal["fields"];
          if (fields && typeof fields === "object" && !Array.isArray(fields)) {
            const prof = insertionPointProfileFromFields(fields as Record<string, unknown>, secByDeclKey);
            if (prof) insertionProfileByTransfPath.set(rec.key, prof);
          }
        }
      }
    }
  }

  const triPos: number[] = [];
  const triNodes: number[] = [];
  const triSegs: number[] = [];
  const triEids: number[] = [];
  const triRefs: ElementRef[] = [];

  const edgePos: number[] = [];
  const edgeNodes: number[] = [];
  const edgeSegs: number[] = [];

  const offsetLinePos: number[] = [];
  const offsetLineNodes: number[] = [];
  const offsetLineSegs: number[] = [];

  const offsetJointPos: number[] = [];
  const offsetJointNodes: number[] = [];
  const offsetJointSegs: number[] = [];

  const extrudedSegmentMask = new Uint8Array(nSeg);
  const clearLinePositions = new Float32Array(linePositions);
  const meshCache = new Map<SectionProfile, ReturnType<typeof triangulateSectionProfile>>();
  const seenOffsets = new Set<string>();
  let skippedUnorientedCount = 0;

  interface ExtrudedSegRecord {
    s: number;
    ref: ElementRef;
    eid: number;
    nodeI: number;
    nodeJ: number;
    pIPrime: [number, number, number];
    pJPrime: [number, number, number];
    profile: SectionProfile;
    secMesh: ReturnType<typeof triangulateSectionProfile>;
    xLocal: [number, number, number];
    yLocal: [number, number, number];
    zLocal: [number, number, number];
  }
  const extrudedRecords: ExtrudedSegRecord[] = [];

  for (let s = 0; s < nSeg; s++) {
    const ref = lineRefs[s]!;
    const eid = lineElementIds[s]!;
    const declPath = ref.kind === "fem" ? cellPath(eid) : undefined;
    let eleDecl: Decl | undefined = declPath ? decls[declPath] : undefined;
    if (!eleDecl && ref.kind === "ops") {
      // The ops row's declaration, by its key (load.ts `pathOfRef`). A scan of every decl here
      // was quadratic: 19 476 inline zeroLength springs never finished loading (#3130).
      const m = elementMeta?.[ref.metaIndex];
      const d = m ? decls[opsRowPath(m.type, ref.row)] : undefined;
      if (d?.kind === "element" && d.element?.cell === null) eleDecl = d;
    }
    const meta = eleDecl?.element?.metas[0];
    if (!meta) continue;

    // P0-2: Ensure (nodeI, pI) and (nodeJ, pJ) match OpenSees (iNode, jNode) order
    let nodeI = lineNodes[2 * s]!;
    let nodeJ = lineNodes[2 * s + 1]!;
    let pI: [number, number, number] = [
      linePositions[6 * s]!,
      linePositions[6 * s + 1]!,
      linePositions[6 * s + 2]!,
    ];
    let pJ: [number, number, number] = [
      linePositions[6 * s + 3]!,
      linePositions[6 * s + 4]!,
      linePositions[6 * s + 5]!,
    ];
    const declNodes = eleDecl?.element?.inlineNodes;
    const inlineNodes =
      declNodes && declNodes.length >= 2
        ? declNodes
        : metaGroupByPath.get(meta.h5)?.inlineConnectivity?.[meta.row];
    const iNodeArg = inlineNodes && inlineNodes.length >= 2 ? inlineNodes[0] : undefined;
    const jNodeArg = inlineNodes && inlineNodes.length >= 2 ? inlineNodes[inlineNodes.length - 1] : undefined;
    const reversed =
      typeof iNodeArg === "number" &&
      typeof jNodeArg === "number" &&
      nodeI === jNodeArg &&
      nodeJ === iNodeArg &&
      nodeI !== nodeJ;
    if (reversed) {
      const tmpN = nodeI;
      nodeI = nodeJ;
      nodeJ = tmpN;
      const tmpP = pI;
      pI = pJ;
      pJ = tmpP;
    }

    // 1. Resolve joint offsets (dI, dJ)
    const jnt = resolveElementJointOffset(meta.type, meta.args, pI, pJ, jntByTransfTag);
    const pIPrime: [number, number, number] = jnt
      ? [pI[0] + jnt.dI[0], pI[1] + jnt.dI[1], pI[2] + jnt.dI[2]]
      : pI;
    const pJPrime: [number, number, number] = jnt
      ? [pJ[0] + jnt.dJ[0], pJ[1] + jnt.dJ[1], pJ[2] + jnt.dJ[2]]
      : pJ;

    if (jnt) {
      const pStartPrime = reversed ? pJPrime : pIPrime;
      const pEndPrime = reversed ? pIPrime : pJPrime;
      clearLinePositions[6 * s] = pStartPrime[0];
      clearLinePositions[6 * s + 1] = pStartPrime[1];
      clearLinePositions[6 * s + 2] = pStartPrime[2];
      clearLinePositions[6 * s + 3] = pEndPrime[0];
      clearLinePositions[6 * s + 4] = pEndPrime[1];
      clearLinePositions[6 * s + 5] = pEndPrime[2];

      if (Math.hypot(jnt.dI[0], jnt.dI[1], jnt.dI[2]) > 1e-12) {
        const keyI = `${nodeI}:${pIPrime[0].toFixed(6)},${pIPrime[1].toFixed(6)},${pIPrime[2].toFixed(6)}`;
        if (!seenOffsets.has(keyI)) {
          seenOffsets.add(keyI);
          offsetLinePos.push(pI[0], pI[1], pI[2], pIPrime[0], pIPrime[1], pIPrime[2]);
          offsetLineNodes.push(nodeI, nodeI);
          offsetLineSegs.push(s);
          offsetJointPos.push(pI[0], pI[1], pI[2], pIPrime[0], pIPrime[1], pIPrime[2]);
          offsetJointNodes.push(nodeI, nodeI);
          offsetJointSegs.push(s, s);
        }
      }
      if (Math.hypot(jnt.dJ[0], jnt.dJ[1], jnt.dJ[2]) > 1e-12) {
        const keyJ = `${nodeJ}:${pJPrime[0].toFixed(6)},${pJPrime[1].toFixed(6)},${pJPrime[2].toFixed(6)}`;
        if (!seenOffsets.has(keyJ)) {
          seenOffsets.add(keyJ);
          offsetLinePos.push(pJ[0], pJ[1], pJ[2], pJPrime[0], pJPrime[1], pJPrime[2]);
          offsetLineNodes.push(nodeJ, nodeJ);
          offsetLineSegs.push(s);
          offsetJointPos.push(pJ[0], pJ[1], pJ[2], pJPrime[0], pJPrime[1], pJPrime[2]);
          offsetJointNodes.push(nodeJ, nodeJ);
          offsetJointSegs.push(s, s);
        }
      }
    }

    // 2. Resolve SectionProfile (Tier 1 -> Tier 2 -> Tier 3)
    let profile: SectionProfile | null = null;
    let transfTag: number | null = null;
    const dec = ELEMENT_SYNTAX[meta.type]?.(meta.args as Param[]);
    if (dec && "slots" in dec) {
      for (const slot of dec.slots) {
        const tagVal = meta.args[slot.slot];
        if (typeof tagVal !== "number" || !Number.isInteger(tagVal)) continue;
        if (slot.family === "geomTransf") {
          transfTag = tagVal;
        } else if (slot.family === "section") {
          profile = resolveSectionDeclProfile(tagVal, secByTag);
        } else if (slot.family === "beamIntegration") {
          const integ = integByTag.get(tagVal);
          if (integ) {
            const iDec = INTEGRATION_SYNTAX[integ.type]?.(integ.params as Param[]);
            if (iDec && "slots" in iDec) {
              for (const iSlot of iDec.slots) {
                if (iSlot.family !== "section") continue;
                const sTag = integ.params[iSlot.slot];
                if (typeof sTag === "number" && Number.isInteger(sTag)) {
                  const cand = resolveSectionDeclProfile(sTag, secByTag);
                  if (cand && (!profile || (profile.source === "elastic" && cand.source !== "elastic"))) {
                    profile = cand;
                  }
                }
              }
            }
          }
        }
      }
    }

    // Tier 2: If element has no Tier 1 section (or only a Tier 3 elastic section) and its
    // geomTransf carries an InsertionPoint profile in /opensees/decl_params, use Tier 2!
    if ((!profile || profile.source === "elastic") && transfTag !== null) {
      const tDecl = transfDeclByTag.get(transfTag);
      if (tDecl) {
        const basePath = tDecl.path.split("@")[0]!;
        const insProf = insertionProfileByTransfPath.get(basePath);
        if (insProf) profile = insProf;
      }
    }

    // Tier 3: Inline element cross-section properties (elasticBeamColumn, LadrunoIMKBeam, Truss, etc.)
    if (!profile) {
      profile = inlineElementProfile(meta.type, meta.args);
    }
    if (!profile || profile.loops.length === 0) continue;

    // 3. Resolve 3-D local triad (xLocal, yLocal, zLocal) along clear span [pIPrime, pJPrime]
    const dx = pJPrime[0] - pIPrime[0];
    const dy = pJPrime[1] - pIPrime[1];
    const dz = pJPrime[2] - pIPrime[2];
    const L = Math.hypot(dx, dy, dz);
    if (L <= 1e-12) continue;

    const xUnit: [number, number, number] = [dx / L, dy / L, dz / L];
    const zRec = stepLocalAxesZRec?.get(eid) ?? modelLocalAxesZRec?.get(eid) ?? null;

    let triad: { xLocal: [number, number, number]; yLocal: [number, number, number]; zLocal: [number, number, number] } | null = null;
    if (zRec) {
      triad = beamLocalAxes(xUnit, zRec);
    }
    if (!triad && transfTag !== null && vecxzByTag.has(transfTag)) {
      triad = beamLocalAxes(xUnit, vecxzByTag.get(transfTag)!);
    }
    if (!triad && ndm === 2 && Math.abs(dz) <= 1e-7 * L) {
      const Lxy = Math.hypot(dx, dy);
      if (Lxy > 1e-12) {
        const cx = dx / Lxy;
        const cy = dy / Lxy;
        triad = {
          xLocal: [cx, cy, 0],
          yLocal: [-cy, cx, 0],
          zLocal: [0, 0, 1],
        };
      }
    }
    if (!triad && profile.axisymmetric) {
      triad = arbitraryPerpendicularTriad(dx, dy, dz);
    }
    // Fail-closed: never guess strong/weak axis orientation for a non-axisymmetric section
    if (!triad) {
      skippedUnorientedCount++;
      continue;
    }

    let secMesh = meshCache.get(profile);
    if (!secMesh) {
      secMesh = triangulateSectionProfile(profile);
      meshCache.set(profile, secMesh);
    }
    if (secMesh.vertices.length < 3 || secMesh.triangles.length === 0) continue;

    extrudedSegmentMask[s] = 1;
    extrudedRecords.push({
      s,
      ref,
      eid,
      nodeI,
      nodeJ,
      pIPrime,
      pJPrime,
      profile,
      secMesh,
      xLocal: triad.xLocal,
      yLocal: triad.yLocal,
      zLocal: triad.zLocal,
    });
  }

  // Index extruded segments by node so interior mesh nodes shared between two
  // collinear sub-elements of the same section omit redundant internal partition caps.
  const recordsByNode = new Map<number, ExtrudedSegRecord[]>();
  for (const rec of extrudedRecords) {
    let listI = recordsByNode.get(rec.nodeI);
    if (!listI) {
      listI = [];
      recordsByNode.set(rec.nodeI, listI);
    }
    listI.push(rec);
    let listJ = recordsByNode.get(rec.nodeJ);
    if (!listJ) {
      listJ = [];
      recordsByNode.set(rec.nodeJ, listJ);
    }
    listJ.push(rec);
  }

  const isCollinearContinuation = (
    rec: ExtrudedSegRecord,
    end: "I" | "J",
  ): boolean => {
    const node = end === "I" ? rec.nodeI : rec.nodeJ;
    const pt = end === "I" ? rec.pIPrime : rec.pJPrime;
    const peers = recordsByNode.get(node);
    if (!peers || peers.length < 2) return false;
    for (const other of peers) {
      if (other.s === rec.s || other.profile !== rec.profile) continue;
      // Match when `other` meets `node` with opposite outward normal along the continuous beam
      const otherAtJ = other.nodeJ === node;
      const otherAtI = other.nodeI === node;
      if (!otherAtJ && !otherAtI) continue;
      const otherPt = otherAtJ ? other.pJPrime : other.pIPrime;
      if (Math.hypot(otherPt[0] - pt[0], otherPt[1] - pt[1], otherPt[2] - pt[2]) > 1e-6) continue;
      const dirSign = (end === "I" && otherAtJ) || (end === "J" && otherAtI) ? 1 : -1;
      const dotX =
        (rec.xLocal[0] * other.xLocal[0] +
          rec.xLocal[1] * other.xLocal[1] +
          rec.xLocal[2] * other.xLocal[2]) *
        dirSign;
      const dotY =
        rec.yLocal[0] * other.yLocal[0] +
        rec.yLocal[1] * other.yLocal[1] +
        rec.yLocal[2] * other.yLocal[2];
      const dotZ =
        rec.zLocal[0] * other.zLocal[0] +
        rec.zLocal[1] * other.zLocal[1] +
        rec.zLocal[2] * other.zLocal[2];
      if (dotX > 0.999 && dotY > 0.999 && dotZ > 0.999) return true;
    }
    return false;
  };

  for (const rec of extrudedRecords) {
    const { s, ref, eid, nodeI, nodeJ, pIPrime, pJPrime, secMesh, yLocal, zLocal } = rec;
    const capI = !isCollinearContinuation(rec, "I");
    const capJ = !isCollinearContinuation(rec, "J");

    // Precompute 3-D vertices at I-end and J-end
    const nV = secMesh.vertices.length;
    const vI = new Float64Array(nV * 3);
    const vJ = new Float64Array(nV * 3);
    for (let k = 0; k < nV; k++) {
      const [yl, zl] = secMesh.vertices[k]!;
      const ox = yl * yLocal[0] + zl * zLocal[0];
      const oy = yl * yLocal[1] + zl * zLocal[1];
      const oz = yl * yLocal[2] + zl * zLocal[2];
      vI[3 * k] = pIPrime[0] + ox;
      vI[3 * k + 1] = pIPrime[1] + oy;
      vI[3 * k + 2] = pIPrime[2] + oz;
      vJ[3 * k] = pJPrime[0] + ox;
      vJ[3 * k + 1] = pJPrime[1] + oy;
      vJ[3 * k + 2] = pJPrime[2] + oz;
    }

    const pushTri = (
      ax: number, ay: number, az: number, nA: number,
      bx: number, by: number, bz: number, nB: number,
      cx: number, cy: number, cz: number, nC: number,
    ) => {
      triPos.push(ax, ay, az, bx, by, bz, cx, cy, cz);
      triNodes.push(nA, nB, nC);
      triSegs.push(s);
      triEids.push(eid);
      triRefs.push(ref);
    };

    const pushEdge = (
      ax: number, ay: number, az: number, nA: number,
      bx: number, by: number, bz: number, nB: number,
    ) => {
      edgePos.push(ax, ay, az, bx, by, bz);
      edgeNodes.push(nA, nB);
      edgeSegs.push(s);
    };

    // Side quads + perimeter cap edges + longitudinal feature edges per loop
    for (const { start, count } of secMesh.loopRanges) {
      for (let i = 0; i < count; i++) {
        const k0 = start + i;
        const k1 = start + ((i + 1) % count);
        const kPrev = start + ((i - 1 + count) % count);

        const i0x = vI[3 * k0]!, i0y = vI[3 * k0 + 1]!, i0z = vI[3 * k0 + 2]!;
        const i1x = vI[3 * k1]!, i1y = vI[3 * k1 + 1]!, i1z = vI[3 * k1 + 2]!;
        const j0x = vJ[3 * k0]!, j0y = vJ[3 * k0 + 1]!, j0z = vJ[3 * k0 + 2]!;
        const j1x = vJ[3 * k1]!, j1y = vJ[3 * k1 + 1]!, j1z = vJ[3 * k1 + 2]!;

        pushTri(i0x, i0y, i0z, nodeI, j0x, j0y, j0z, nodeJ, j1x, j1y, j1z, nodeJ);
        pushTri(i0x, i0y, i0z, nodeI, j1x, j1y, j1z, nodeJ, i1x, i1y, i1z, nodeI);

        // End-cap outline edges at I and J
        if (capI) pushEdge(i0x, i0y, i0z, nodeI, i1x, i1y, i1z, nodeI);
        pushEdge(j0x, j0y, j0z, nodeJ, j1x, j1y, j1z, nodeJ);

        // Longitudinal feature edge at sharp corners (or 4 silhouette lines on smooth circles)
        const [py, pz] = secMesh.vertices[kPrev]!;
        const [cy, cz] = secMesh.vertices[k0]!;
        const [ny, nz] = secMesh.vertices[k1]!;
        const uy = cy - py, uz = cz - pz;
        const vy = ny - cy, vz = nz - cz;
        const nu = Math.hypot(uy, uz), nv = Math.hypot(vy, vz);
        const cosAngle = nu > 0 && nv > 0 ? (uy * vy + uz * vz) / (nu * nv) : 1;
        if (cosAngle < 0.9397 || (count >= 16 && i % Math.max(1, Math.floor(count / 4)) === 0)) {
          pushEdge(i0x, i0y, i0z, nodeI, j0x, j0y, j0z, nodeJ);
        }
      }
    }

    // End-cap triangles at I (-xLocal) and J (+xLocal) on true member ends
    if (capI || capJ) {
      for (const [a, b, c] of secMesh.triangles) {
        if (capI) {
          pushTri(
            vI[3 * a]!, vI[3 * a + 1]!, vI[3 * a + 2]!, nodeI,
            vI[3 * c]!, vI[3 * c + 1]!, vI[3 * c + 2]!, nodeI,
            vI[3 * b]!, vI[3 * b + 1]!, vI[3 * b + 2]!, nodeI,
          );
        }
        if (capJ) {
          pushTri(
            vJ[3 * a]!, vJ[3 * a + 1]!, vJ[3 * a + 2]!, nodeJ,
            vJ[3 * b]!, vJ[3 * b + 1]!, vJ[3 * b + 2]!, nodeJ,
            vJ[3 * c]!, vJ[3 * c + 1]!, vJ[3 * c + 2]!, nodeJ,
          );
        }
      }
    }
  }

  return {
    frameTriPositions: new Float32Array(triPos),
    frameTriNodes: new Float64Array(triNodes),
    frameTriSegments: new Int32Array(triSegs),
    frameTriElementIds: new Int32Array(triEids),
    frameTriRefs: triRefs,
    frameEdgePositions: new Float32Array(edgePos),
    frameEdgeNodes: new Float64Array(edgeNodes),
    frameEdgeSegments: new Int32Array(edgeSegs),
    extrudedSegmentMask,
    clearLinePositions,
    offsetLinePositions: new Float32Array(offsetLinePos),
    offsetLineNodes: new Float64Array(offsetLineNodes),
    offsetLineSegments: new Int32Array(offsetLineSegs),
    offsetJointPositions: new Float32Array(offsetJointPos),
    offsetJointNodes: new Float64Array(offsetJointNodes),
    offsetJointSegments: new Int32Array(offsetJointSegs),
    hasExtrudableFrames: triRefs.length > 0,
    hasJointOffsets: offsetLineSegs.length > 0,
    skippedUnorientedCount,
  };
}
