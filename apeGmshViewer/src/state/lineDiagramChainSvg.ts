// Pure 2-D multi-beam chain plot model, drawn-beam summary, and hover readout
// helpers for 1-D beam line diagrams.

import { fmtValue } from "./contour.ts";
import {
  buildSelectedBeamChainsFromState,
  effectiveFlipSign,
  LINE_DIAGRAM_COMPONENT_META,
  projectHoverNeedle,
  type LineDiagram3DGeometry,
} from "./lineDiagram.ts";
import type {
  BeamChain,
  DeclPath,
  LineDiagramBeamSummary,
  LineDiagramHoverReadout,
  LineStationComponent,
  Pick as ModelPick,
  State,
} from "./types.ts";

export interface BeamChainJointPlot {
  s: number;
  x: number;
  nodeId: number;
  label: string;
  isInternal: boolean;
}

export interface BeamChainExtremaPlot {
  s: number;
  x: number;
  y: number;
  value: number;
  signedValue: number;
  label: string;
}

export interface BeamChainStationDotPlot {
  s: number;
  u: number;
  xi: number;
  x: number;
  y: number;
  value: number;
  signedValue: number;
  isUnknownBoundary: boolean;
}

export interface BeamChainUnknownGapPlot {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  stationX: number;
  stationY: number;
}

export interface BeamChainSpanPlotModel {
  decl: DeclPath;
  elementId: number;
  nodeStart: number;
  nodeEnd: number;
  reversed: boolean;
  xStart: number;
  xEnd: number;
  length: number;
  x0: number;
  x1: number;
  midX: number;
  measuredX0: number;
  measuredX1: number;
  xiMin: number;
  xiMax: number;
  startEndKnown: boolean;
  endEndKnown: boolean;
  label: string;
  polylinePoints: string;
  polygonPoints: string | null;
  unknownStartGap: BeamChainUnknownGapPlot | null;
  unknownEndGap: BeamChainUnknownGapPlot | null;
  stationDots: BeamChainStationDotPlot[];
  minMarker: BeamChainExtremaPlot | null;
  maxMarker: BeamChainExtremaPlot | null;
}

export interface BeamChainHoverPlot {
  elementId: number;
  xi: number;
  s: number;
  x: number;
  y: number;
  zeroY: number;
  value: number;
  signedValue: number;
  withinStationSpan: boolean;
  label: string;
}

export interface BeamChainPlotModel {
  chainIndex: number;
  totalChains: number;
  headerLabel: string;
  width: number;
  height: number;
  padLeft: number;
  padRight: number;
  padTop: number;
  padBottom: number;
  zeroY: number;
  totalLength: number;
  maxAbs: number;
  flipSign: boolean;
  fill: boolean;
  hasUnknownEnds: boolean;
  unknownEndBadge: string | null;
  joints: BeamChainJointPlot[];
  spans: BeamChainSpanPlotModel[];
  hoverMarker: BeamChainHoverPlot | null;
}

export interface BuildBeamChainPlotOptions {
  flipSign: boolean;
  fill: boolean;
  component: LineStationComponent | null;
  hover?: LineDiagramHoverReadout | null;
  width?: number;
  height?: number;
}

/**
 * Convert 3-D beam spans from `buildLineDiagram3D` into plain-data summaries
 * suitable for storing in `State.lineDiagram.drawnBeams`.
 */
export function summarizeDrawnBeams(
  geom: LineDiagram3DGeometry,
  lineNodes?: ArrayLike<number>,
): LineDiagramBeamSummary[] {
  const out: LineDiagramBeamSummary[] = [];
  for (const b of geom.beams) {
    const segIdx = b.segmentIndex;
    const endSegIdx = b.endSegmentIndex ?? segIdx;
    const nodeStart =
      b.nodeStart !== undefined
        ? b.nodeStart
        : lineNodes && 2 * segIdx < lineNodes.length
          ? Math.round(Number(lineNodes[2 * segIdx]))
          : 0;
    const nodeEnd =
      b.nodeEnd !== undefined
        ? b.nodeEnd
        : lineNodes && 2 * endSegIdx + 1 < lineNodes.length
          ? Math.round(Number(lineNodes[2 * endSegIdx + 1]))
          : 0;
    out.push({
      decl: b.decl,
      elementId: b.elementId,
      nodeStart,
      nodeEnd,
      pStart: [b.pStart[0], b.pStart[1], b.pStart[2]],
      pEnd: [b.pEnd[0], b.pEnd[1], b.pEnd[2]],
      length: b.length,
      xLocal: [b.xLocal[0], b.xLocal[1], b.xLocal[2]],
      extrudeDir: [b.extrudeDir[0], b.extrudeDir[1], b.extrudeDir[2]],
      xi: Array.from(b.xi, (v) => Number(v)),
      values: Array.from(b.values, (v) => Number(v)),
      autoScale: geom.autoScale,
      ...(b.spanMode !== undefined ? { spanMode: b.spanMode } : {}),
      ...(b.spanReason !== undefined ? { spanReason: b.spanReason } : {}),
    });
  }
  return out;
}

function distSqToSegment(
  pt: readonly [number, number, number],
  p0: readonly [number, number, number],
  p1: readonly [number, number, number],
): number {
  const dx = p1[0] - p0[0];
  const dy = p1[1] - p0[1];
  const dz = p1[2] - p0[2];
  const lenSq = dx * dx + dy * dy + dz * dz;
  const t =
    lenSq > 1e-24
      ? Math.max(0, Math.min(1, ((pt[0] - p0[0]) * dx + (pt[1] - p0[1]) * dy + (pt[2] - p0[2]) * dz) / lenSq))
      : 0;
  const qx = p0[0] + t * dx - pt[0];
  const qy = p0[1] + t * dy - pt[1];
  const qz = p0[2] + t * dz - pt[2];
  return qx * qx + qy * qy + qz * qz;
}

/**
 * Compute the active 1-D beam hover readout from `s.hover` and `drawnBeams`.
 * Returns `null` when `lineDiagram.on` is false, `s.hover` is null or a bare
 * `DeclPath`, `s.hover.at` is null, or `s.hover.decl` is not a drawn 1-D beam.
 */
export function computeLineDiagramHover(
  s: Pick<State, "hover" | "lineDiagram">,
  drawnBeams?: readonly LineDiagramBeamSummary[],
): LineDiagramHoverReadout | null {
  if (!s.lineDiagram.on || !s.hover || typeof s.hover === "string" || !s.hover.at) {
    return null;
  }
  const beams = drawnBeams ?? s.lineDiagram.drawnBeams;
  if (beams.length === 0) return null;

  const hoverPick = s.hover;
  const hitPoint = hoverPick.at;
  if (!hitPoint) return null;
  const candidates = beams.filter((b) => b.decl === hoverPick.decl);
  if (candidates.length === 0) return null;

  let beam = candidates[0]!;
  if (candidates.length > 1) {
    let bestDist = Infinity;
    for (const c of candidates) {
      const d = distSqToSegment(hitPoint, c.pStart, c.pEnd);
      if (d < bestDist) {
        bestDist = d;
        beam = c;
      }
    }
  }

  const flipSign = effectiveFlipSign(s.lineDiagram);
  const effectiveScale = beam.autoScale * s.lineDiagram.scale;
  const needle = projectHoverNeedle(beam, hitPoint, effectiveScale, flipSign);

  let chainIndex: number | null = null;
  let sCoord = needle.x;
  const chains = s.lineDiagram.chains;
  for (let cIdx = 0; cIdx < chains.length; cIdx++) {
    const chain = chains[cIdx]!;
    const span = chain.spans.find((sp) => sp.elementId === beam.elementId && sp.decl === beam.decl);
    if (span) {
      chainIndex = cIdx;
      const chainU = span.reversed ? 1 - needle.xRel : needle.xRel;
      sCoord = span.xStart + chainU * span.length;
      break;
    }
  }

  return {
    decl: beam.decl,
    elementId: beam.elementId,
    xRel: needle.xRel,
    x: needle.x,
    s: sCoord,
    chainIndex,
    length: needle.length,
    xi: needle.xi,
    value: needle.value,
    withinStationSpan: needle.withinStationSpan,
    pBase: [needle.pBase[0], needle.pBase[1], needle.pBase[2]],
    pTop: [needle.pTop[0], needle.pTop[1], needle.pTop[2]],
    needlePositions: [
      needle.pBase[0],
      needle.pBase[1],
      needle.pBase[2],
      needle.pTop[0],
      needle.pTop[1],
      needle.pTop[2],
    ],
  };
}

/**
 * Recompute `s.lineDiagram.chains` and `s.lineDiagram.hover` from current
 * selection, hover, and drawn beams.
 */
export function syncLineDiagramDerived(s: State): State {
  if (!s.lineDiagram.on || s.lineDiagram.drawnBeams.length === 0) {
    if (s.lineDiagram.chains.length === 0 && s.lineDiagram.hover === null) return s;
    return {
      ...s,
      lineDiagram: {
        ...s.lineDiagram,
        chains: [],
        hover: null,
      },
    };
  }
  const chains = buildSelectedBeamChainsFromState(s, { beams: s.lineDiagram.drawnBeams });
  const nextLineDiagram = {
    ...s.lineDiagram,
    chains,
  };
  const hover = computeLineDiagramHover({ hover: s.hover, lineDiagram: nextLineDiagram }, s.lineDiagram.drawnBeams);
  return {
    ...s,
    lineDiagram: {
      ...nextLineDiagram,
      hover,
    },
  };
}

/**
 * Map a horizontal arc-length coordinate `sCoord` in `[0, chain.totalLength]`
 * on a 2-D chain plot back to a 3-D `Pick` on the underlying drawn beam so
 * hovering the 2-D SVG plot drives the same `hover` store event as 3-D picking.
 */
export function pickChainPlotAtS(
  chain: BeamChain,
  sCoord: number,
  drawnBeams: readonly LineDiagramBeamSummary[],
): ModelPick | null {
  if (chain.spans.length === 0 || chain.totalLength <= 0) return null;
  const clampedS = Math.max(0, Math.min(chain.totalLength, sCoord));
  let span = chain.spans[chain.spans.length - 1]!;
  for (const sp of chain.spans) {
    if (clampedS >= sp.xStart && clampedS <= sp.xEnd + 1e-9) {
      span = sp;
      break;
    }
  }
  const beam = drawnBeams.find((b) => b.elementId === span.elementId && b.decl === span.decl);
  if (!beam) return null;
  const chainU = span.length > 1e-12 ? Math.max(0, Math.min(1, (clampedS - span.xStart) / span.length)) : 0;
  const t = span.reversed ? 1 - chainU : chainU;
  const at: [number, number, number] = [
    beam.pStart[0] + t * (beam.pEnd[0] - beam.pStart[0]),
    beam.pStart[1] + t * (beam.pEnd[1] - beam.pStart[1]),
    beam.pStart[2] + t * (beam.pEnd[2] - beam.pStart[2]),
  ];
  return {
    decl: span.decl,
    at,
    nodes: [span.nodeStart, span.nodeEnd],
  };
}

/**
 * Format the hover readout bar text `(E<id>, xi, s, value)` for the panel.
 */
export function formatHoverReadout(
  hover: LineDiagramHoverReadout,
  component: LineStationComponent | null,
): string {
  const compLabel = component ?? "val";
  const meta = component ? LINE_DIAGRAM_COMPONENT_META[component] : undefined;
  const canonical = meta ? ` (${meta.canonical})` : "";
  const extrapNote = hover.withinStationSpan ? "" : " [clamped to station end]";
  return `E${hover.elementId} · ξ = ${hover.xi.toFixed(3)} · s = ${fmtValue(hover.s)} · ${compLabel}${canonical} = ${fmtValue(hover.value)}${extrapNote}`;
}

const fmtCoord = (n: number): string => Number(n.toFixed(2)).toString();

/**
 * Build the pure 2-D SVG plot model for one `BeamChain`.
 *
 * - Plots `s in [0, chain.totalLength]` horizontally.
 * - Marks joints `N<nodeId>` and span labels `E<elementId>`.
 * - Draws each span's polyline and optional fill polygon ONLY over its
 *   measured station range `[s_k(u_min), s_k(u_max)]` — never extrapolating
 *   interior Gauss stations to the element ends `xi = +-1`.
 * - Emits explicit dashed/unfilled end gap descriptors (`unknownStartGap`,
 *   `unknownEndGap`) and `unknownEndBadge` whenever `!startEndKnown || !endEndKnown`.
 */
export function buildBeamChainPlotModel(
  chain: BeamChain,
  chainIndex: number,
  totalChains: number,
  opts: BuildBeamChainPlotOptions,
): BeamChainPlotModel {
  const width = opts.width ?? 280;
  const height = opts.height ?? 116;
  const padLeft = 24;
  const padRight = 14;
  const padTop = 18;
  const padBottom = 22;
  const plotW = Math.max(1, width - padLeft - padRight);
  const plotH = Math.max(1, height - padTop - padBottom);
  const zeroY = Number((padTop + plotH / 2).toFixed(2));
  const halfH = (plotH / 2) * 0.82;
  const totalLength = Math.max(chain.totalLength, 1e-12);
  const sign = opts.flipSign ? -1 : 1;

  let maxAbs = 0;
  for (const sp of chain.spans) {
    for (const st of sp.stations) {
      if (Number.isFinite(st.value)) {
        const a = Math.abs(st.value);
        if (a > maxAbs) maxAbs = a;
      }
    }
  }
  const scaleMax = maxAbs > 1e-12 ? maxAbs : 1;

  const sToX = (s: number): number => Number((padLeft + (Math.max(0, Math.min(totalLength, s)) / totalLength) * plotW).toFixed(2));
  const vToY = (v: number): number => {
    if (!Number.isFinite(v)) return zeroY;
    const sv = sign * v;
    return Number((zeroY - (sv / scaleMax) * halfH).toFixed(2));
  };

  const joints: BeamChainJointPlot[] = chain.joints.map((j, idx) => ({
    s: j.x,
    x: sToX(j.x),
    nodeId: j.nodeId,
    label: `N${j.nodeId}`,
    isInternal: idx > 0 && idx < chain.joints.length - 1,
  }));

  let hasUnknownEnds = false;
  const spans: BeamChainSpanPlotModel[] = chain.spans.map((sp) => {
    if (!sp.startEndKnown || !sp.endEndKnown) hasUnknownEnds = true;
    const x0 = sToX(sp.xStart);
    const x1 = sToX(sp.xEnd);
    const midX = Number(((x0 + x1) / 2).toFixed(2));

    const stationDots: BeamChainStationDotPlot[] = sp.stations.map((st, idx) => ({
      s: st.x,
      u: sp.length > 1e-12 ? (st.x - sp.xStart) / sp.length : 0,
      xi: st.xi,
      x: sToX(st.x),
      y: vToY(st.value),
      value: st.value,
      signedValue: sign * st.value,
      isUnknownBoundary:
        (idx === 0 && !sp.startEndKnown) || (idx === sp.stations.length - 1 && !sp.endEndKnown),
    }));

    const firstDot = stationDots[0];
    const lastDot = stationDots[stationDots.length - 1];
    const measuredX0 = firstDot ? firstDot.x : x0;
    const measuredX1 = lastDot ? lastDot.x : x1;

    const polylinePoints = stationDots.map((d) => `${fmtCoord(d.x)},${fmtCoord(d.y)}`).join(" ");
    const polygonPoints =
      opts.fill && stationDots.length >= 2
        ? [
            `${fmtCoord(measuredX0)},${fmtCoord(zeroY)}`,
            ...stationDots.map((d) => `${fmtCoord(d.x)},${fmtCoord(d.y)}`),
            `${fmtCoord(measuredX1)},${fmtCoord(zeroY)}`,
          ].join(" ")
        : null;

    const unknownStartGap: BeamChainUnknownGapPlot | null =
      !sp.startEndKnown && firstDot
        ? {
            x0,
            y0: zeroY,
            x1: firstDot.x,
            y1: firstDot.y,
            stationX: firstDot.x,
            stationY: firstDot.y,
          }
        : null;

    const unknownEndGap: BeamChainUnknownGapPlot | null =
      !sp.endEndKnown && lastDot
        ? {
            x0: lastDot.x,
            y0: lastDot.y,
            x1,
            y1: zeroY,
            stationX: lastDot.x,
            stationY: lastDot.y,
          }
        : null;

    let minSt = sp.stations[0];
    let maxSt = sp.stations[0];
    for (const st of sp.stations) {
      if (minSt && st.value < minSt.value) minSt = st;
      if (maxSt && st.value > maxSt.value) maxSt = st;
    }

    const minMarker: BeamChainExtremaPlot | null =
      minSt && Number.isFinite(minSt.value)
        ? {
            s: minSt.x,
            x: sToX(minSt.x),
            y: vToY(minSt.value),
            value: minSt.value,
            signedValue: sign * minSt.value,
            label: fmtValue(minSt.value),
          }
        : null;

    const maxMarker: BeamChainExtremaPlot | null =
      maxSt && Number.isFinite(maxSt.value)
        ? {
            s: maxSt.x,
            x: sToX(maxSt.x),
            y: vToY(maxSt.value),
            value: maxSt.value,
            signedValue: sign * maxSt.value,
            label: fmtValue(maxSt.value),
          }
        : null;

    return {
      decl: sp.decl,
      elementId: sp.elementId,
      nodeStart: sp.nodeStart,
      nodeEnd: sp.nodeEnd,
      reversed: sp.reversed,
      xStart: sp.xStart,
      xEnd: sp.xEnd,
      length: sp.length,
      x0,
      x1,
      midX,
      measuredX0,
      measuredX1,
      xiMin: sp.xiMin,
      xiMax: sp.xiMax,
      startEndKnown: sp.startEndKnown,
      endEndKnown: sp.endEndKnown,
      label: `E${sp.elementId}`,
      polylinePoints,
      polygonPoints,
      unknownStartGap,
      unknownEndGap,
      stationDots,
      minMarker,
      maxMarker,
    };
  });

  const firstSpan = chain.spans[0];
  const headerLabel =
    chain.spans.length === 1 && firstSpan
      ? `Chain ${chainIndex + 1}: E${firstSpan.elementId} (N${firstSpan.nodeStart} → N${firstSpan.nodeEnd}, L = ${fmtValue(chain.totalLength)})`
      : `Chain ${chainIndex + 1} (${chain.spans.length} spans, L = ${fmtValue(chain.totalLength)}): ${chain.spans.map((sp) => `E${sp.elementId}`).join(" → ")}`;

  let hoverMarker: BeamChainHoverPlot | null = null;
  if (opts.hover && opts.hover.chainIndex === chainIndex) {
    const hx = sToX(opts.hover.s);
    const hy = vToY(opts.hover.value);
    hoverMarker = {
      elementId: opts.hover.elementId,
      xi: opts.hover.xi,
      s: opts.hover.s,
      x: hx,
      y: hy,
      zeroY,
      value: opts.hover.value,
      signedValue: sign * opts.hover.value,
      withinStationSpan: opts.hover.withinStationSpan,
      label: `E${opts.hover.elementId}: ${fmtValue(opts.hover.value)}`,
    };
  }

  return {
    chainIndex,
    totalChains,
    headerLabel,
    width,
    height,
    padLeft,
    padRight,
    padTop,
    padBottom,
    zeroY,
    totalLength: chain.totalLength,
    maxAbs,
    flipSign: opts.flipSign,
    fill: opts.fill,
    hasUnknownEnds,
    unknownEndBadge: hasUnknownEnds ? "ends unrecorded (Gauss stations)" : null,
    joints,
    spans,
    hoverMarker,
  };
}
