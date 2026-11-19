// The line diagram slice: 1-D beam station diagrams and their chains.
// One slice of the store's types; src/state/types.ts re-exports it, and the
// panels import it from there. A slice file imports sibling files, never the barrel.

import type { ContourCmap, ContourRange, ContourScan } from "./contour.ts";
import type { DeclPath } from "./model.ts";

export const LINE_STATION_COMPONENTS = [
  "N", "Vy", "Vz", "T", "My", "Mz",
  "eps", "gammaY", "gammaZ", "theta", "kappaY", "kappaZ",
] as const;
export type LineStationComponent = (typeof LINE_STATION_COMPONENTS)[number];

export type LineDiagramCmap = "sign" | ContourCmap;
export type LineDiagramPlane = "local" | "screen";

export interface BeamChainStationPoint {
  /** Natural coordinate in the span's original orientation (`[-1, +1]`). */
  xi: number;
  /** Cumulative distance `x` along the chain `[0, totalLength]`. */
  x: number;
  /** Raw physical value at this station. */
  value: number;
}

export interface BeamChainSpan {
  decl: DeclPath;
  elementId: number;
  /** Entry node ID in chain walk order. */
  nodeStart: number;
  /** Exit node ID in chain walk order. */
  nodeEnd: number;
  /** True when the beam's element orientation (`nodeEnd -> nodeStart`) was reversed to keep the chain contiguous. */
  reversed: boolean;
  /** Start cumulative distance along the chain. */
  xStart: number;
  /** End cumulative distance along the chain (`xStart + length`). */
  xEnd: number;
  length: number;
  /** Minimum natural coordinate recorded on this span (`-1` when single-station). */
  xiMin: number;
  /** Maximum natural coordinate recorded on this span (`+1` when single-station). */
  xiMax: number;
  /** True when the value at `xStart` of the chain span is measured (station at endpoint or single-station constant). */
  startEndKnown: boolean;
  /** True when the value at `xEnd` of the chain span is measured (station at endpoint or single-station constant). */
  endEndKnown: boolean;
  /** Stations ordered by increasing cumulative `x` along the chain. */
  stations: BeamChainStationPoint[];
}

export interface BeamChainJoint {
  nodeId: number;
  x: number;
}

export interface BeamChain {
  spans: BeamChainSpan[];
  /** Joint markers at span boundaries (`[0, L_1, L_1 + L_2, ..., totalLength]`). */
  joints: BeamChainJoint[];
  totalLength: number;
  /** True when all input beams formed a single contiguous node-to-node chain. */
  contiguous: boolean;
}

export interface LineDiagramBeamSummary {
  decl: DeclPath;
  elementId: number;
  nodeStart: number;
  nodeEnd: number;
  pStart: readonly [number, number, number];
  pEnd: readonly [number, number, number];
  length: number;
  xLocal: readonly [number, number, number];
  extrudeDir: readonly [number, number, number];
  xi: readonly number[];
  values: readonly number[];
  autoScale: number;
  /** Whether the span curve was reconstructed analytically from model loads (`"exact"`) or station-interpolated (`"interpolated"`). */
  spanMode?: "exact" | "interpolated";
  /** Fallback reason when `spanMode === "interpolated"` (`null` when `"exact"`). */
  spanReason?: string | null;
}

export interface LineDiagramHoverReadout {
  decl: DeclPath;
  elementId: number;
  /** Relative distance along beam chord x/L in [0, 1]. */
  xRel: number;
  /** Distance along beam chord x in [0, L]. */
  x: number;
  /** Cumulative distance s along the selected chain [0, totalLength], or x when not in a selected chain. */
  s: number;
  /** Index of the selected chain containing this beam, or null when not in a selected chain. */
  chainIndex: number | null;
  /** Total chord length L. */
  length: number;
  /** Natural coordinate xi = 2*(x/L) - 1 in [-1, +1]. */
  xi: number;
  /** Linearly interpolated station value at xi. */
  value: number;
  /** True when xi lies within the recorded station range [xiMin, xiMax] (or single-station constant span); false in unmeasured Gauss end zones. */
  withinStationSpan: boolean;
  /** Projected point on the beam chord. */
  pBase: readonly [number, number, number];
  /** Corresponding point on the diagram profile. */
  pTop: readonly [number, number, number];
  /** 6 floats as a plain tuple: `[pBase[0], pBase[1], pBase[2], pTop[0], pTop[1], pTop[2]]`. */
  needlePositions: readonly [number, number, number, number, number, number];
}

export interface LineDiagramState {
  on: boolean;
  component: LineStationComponent | null;
  /** User scale multiplier (default 1, > 0). */
  scale: number;
  plane: LineDiagramPlane;
  cmap: LineDiagramCmap;
  fill: boolean;
  /** null means use component default (`true` for My/Mz, `false` otherwise); boolean overrides. */
  flipSign: boolean | null;
  range: ContourRange;
  fixed: { min: number; max: number } | null;
  stepRange: { key: string; min: number | null; max: number | null } | null;
  scan: ContourScan | null;
  notice: string | null;
  /** Drawn 1-D beam spans for the current step (plain data). */
  drawnBeams: readonly LineDiagramBeamSummary[];
  /** Selected 1-D beam chains for the 2-D SVG span plot. */
  chains: readonly BeamChain[];
  /** Active 3-D and 2-D hover readout on a drawn 1-D beam, or null. */
  hover: LineDiagramHoverReadout | null;
}
