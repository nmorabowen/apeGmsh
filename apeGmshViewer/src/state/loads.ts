// Pure 3-D structural loading and prescribed-SP selector & geometry builder.
//
// Pure: no three.js and no DOM, so every function runs in Node tests.
// Reads both `/opensees/patterns` (global and staged) and neutral
// `/loads/{nodal,element,sp}/{case}` tables, applies stage activation,
// `remove_sp`, `role="support_reaction"` filtering, optional `timeSeries`
// scaling λ(t), fail-closed 3-D beam `vecxz` checks, and `baseUnits`
// conversion (`resolveFieldUnits`).

import { DARK_MAIN, hexToRgb, LIGHT_MAIN, themeTokens, type ThemeMode } from "../theme/tokens.ts";
import { activationOf } from "./activation.ts";
import type { BlobStore } from "./blobs.ts";
import { beamLocalAxes, elementTransformVecxzFromState } from "./lineDiagram.ts";
import type {
  Decl,
  DeclPath,
  DeclRecord,
  DeclValue,
  Definitions,
  DefObject,
  State,
  Table,
} from "./types.ts";
import { resolveFieldUnits } from "./units.ts";

export interface ResolvedUnits {
  readonly factor: number;
  readonly unit: string | null;
  readonly note: string;
}

function declOfTag(defs: Definitions, kind: string, tag: number | null): DeclRecord | null {
  if (!defs.decls || tag === null) return null;
  const run = defs.decls.tags.find((t) => t.kind === kind && tag >= t.first && tag < t.first + t.count);
  return run ? (defs.decls.records[run.decl] ?? null) : null;
}

function inactiveElements(s: State): { mask: Uint8Array } | null {
  if (s.sequence.status !== "ready" || !s.mesh) return null;
  const at = s.phase.at;
  let stageName: string | null = null;
  if (at?.kind === "stage") {
    const nameCol = s.sequence.stages.columns.indexOf("name");
    const row = s.sequence.stages.rows[at.index];
    stageName = row && nameCol >= 0 ? String(row[nameCol] ?? "") : null;
  } else if (at?.kind === "results" && s.results) {
    stageName = s.results.stages[at.stage]?.name ?? null;
  } else if (s.stage !== null) {
    const nameCol = s.sequence.stages.columns.indexOf("name");
    const row = s.sequence.stages.rows[s.stage];
    stageName = row && nameCol >= 0 ? String(row[nameCol] ?? "") : null;
  }
  if (!stageName) return null;
  const act = activationOf(s.sequence, s.stageElements, s.definitions, stageName);
  if (act.state !== "filtered") return null;
  const tags = act.inactiveTags;
  const mask = new Uint8Array(s.mesh.elements.length);
  s.mesh.elements.forEach((p, i) => {
    for (const m of s.decls[p]?.element?.metas ?? []) if (tags.has(m.tag)) mask[i] = 1;
  });
  return { mask };
}

export interface LoadsUnitLabels {
  readonly force: ResolvedUnits;
  readonly moment: ResolvedUnits;
  readonly beamLoad: ResolvedUnits;
  readonly surfacePressure: ResolvedUnits;
  readonly displacement: ResolvedUnits;
  readonly rotation: ResolvedUnits;
  readonly acceleration: ResolvedUnits;
}

export interface LoadPatternCounts {
  readonly nodalForces: number;
  readonly nodalMoments: number;
  readonly elementLoads: number;
  readonly sps: number;
  readonly spHolds: number;
  readonly total: number;
}

export interface LoadPatternEntry {
  /** Unique pattern key: HDF5 path (`/opensees/patterns/Plain_1`) or `neutral:<case>` */
  readonly id: string;
  readonly zone: "opensees" | "neutral";
  readonly title: string;
  readonly stage: string;
  /** -1 for global/neutral, or 0-based stage index */
  readonly stageIndex: number;
  readonly type: string;
  readonly tag: number | null;
  readonly role: string | null;
  readonly synth: boolean;
  /** Evaluated timeSeries factor λ(t) at current step time (1 when nominal or off results). */
  readonly seriesFactor: number;
  readonly seriesLabel: string | null;
  readonly active: boolean;
  readonly counts: LoadPatternCounts;
}

export interface LoadsPeaks {
  readonly nodalForce: number;
  readonly nodalMoment: number;
  readonly beamUniform: number;
  readonly beamPoint: number;
  readonly surfacePressure: number;
  readonly spTranslation: number;
  readonly spRotation: number;
  readonly baseExcitation: number;
}

export interface LoadsSummary {
  readonly hasLoads: boolean;
  readonly patterns: readonly LoadPatternEntry[];
  readonly activePatterns: readonly LoadPatternEntry[];
  readonly activeStageIndex: number | null;
  readonly activeStageLabel: string;
  readonly time: number | null;
  readonly units: LoadsUnitLabels;
  readonly counts: {
    readonly nodalForces: number;
    readonly nodalMoments: number;
    readonly elementLoads: number;
    readonly beamUniform: number;
    readonly beamPoint: number;
    readonly surfacePressure: number;
    readonly sps: number;
    readonly spTranslations: number;
    readonly spRotations: number;
    readonly baseExcitations: number;
    readonly total: number;
  };
  readonly peaks: LoadsPeaks;
  readonly skippedUnorientedBeams: number;
  readonly notice: string | null;
}

export interface Loads3DGeometry {
  /** 6 floats per line segment (arrow shafts, moment double-heads, distributed load comb bars, SP cross-bars). */
  readonly linePositions: Float32Array;
  /** 6 floats per line segment (sRGB vertex colours). */
  readonly lineColors: Float32Array;
  /** 9 floats per arrowhead triangle. */
  readonly conePositions: Float32Array;
  /** 9 floats per arrowhead triangle (sRGB vertex colours). */
  readonly coneColors: Float32Array;
  /** Alias for `conePositions` (9 floats per arrowhead triangle). */
  readonly triPositions: Float32Array;
  /** Alias for `coneColors` (9 floats per arrowhead triangle). */
  readonly triColors: Float32Array;
  readonly nodalForcesCount: number;
  readonly nodalMomentsCount: number;
  readonly beamUniformCount: number;
  readonly beamPointCount: number;
  readonly surfacePressureCount: number;
  readonly spCount: number;
  readonly nodalForceArrows: number;
  readonly nodalMomentArrows: number;
  readonly elementLoadArrows: number;
  readonly spArrows: number;
  readonly skippedUnorientedBeams: number;
  readonly skippedBeamDecls: readonly DeclPath[];
  readonly notice: string | null;
  readonly maxForceMag: number;
  readonly maxMomentMag: number;
  readonly maxElementLoadMag: number;
  readonly maxBeamUniformMag: number;
  readonly maxBeamPointMag: number;
  readonly maxSurfacePressureMag: number;
  readonly maxSpMag: number;
  readonly maxSpTranslationMag: number;
  readonly maxSpRotationMag: number;
  readonly maxBaseExcitationMag: number;
  readonly units: LoadsUnitLabels;
}

export interface BuildLoads3DOptions {
  readonly theme?: ThemeMode;
  /** Optional deformed node coordinates (`nodeId -> [x, y, z]`) when `s.deform.on` is active. */
  readonly deformedNodeCoords?: ReadonlyMap<number, readonly [number, number, number]>;
  /** Optional drawn line segment positions (6 floats per segment, deformed when `s.deform.on` is active). */
  readonly drawnLinePositions?: Float32Array;
  /** Optional drawn triangle positions (9 floats per triangle, deformed when `s.deform.on` is active). */
  readonly drawnTriPositions?: Float32Array;
}

export function loadsColourPalette(mode: ThemeMode = "dark"): {
  readonly force: [number, number, number];
  readonly moment: [number, number, number];
  readonly nodalForce: [number, number, number];
  readonly nodalMoment: [number, number, number];
  readonly elementLoad: [number, number, number];
  readonly sp: [number, number, number];
} {
  const t = themeTokens(mode);
  const ring = mode === "light" ? LIGHT_MAIN : DARK_MAIN;
  const force = hexToRgb(t.warn);
  const moment = hexToRgb(ring[5]!);
  return {
    force,
    moment,
    nodalForce: force,
    nodalMoment: moment,
    elementLoad: hexToRgb(t.ok),
    sp: hexToRgb(ring[4]!),
  };
}

/**
 * Resolve unit conversion factors and display labels for all structural load
 * and prescribed-SP categories using `s.contour.{declaredUnits, sourceUnits, targetUnits}`.
 */
export function loadsUnitsOf(s: State): LoadsUnitLabels {
  const c = s.contour;
  const native = c.declaredUnits ?? c.sourceUnits;
  const effectiveTarget = c.targetUnits === "as_stored" && native !== null ? native : c.targetUnits;
  const res = (field: string, source: "nodal" | "element"): ResolvedUnits =>
    resolveFieldUnits(field, source, c.sourceUnits, effectiveTarget, undefined, c.declaredUnits);
  return {
    force: res("reaction_force", "nodal"),
    moment: res("reaction_moment", "nodal"),
    beamLoad: res("membrane_force", "element"),
    surfacePressure: res("pressure", "element"),
    displacement: res("displacement", "nodal"),
    rotation: res("rotation", "nodal"),
    acceleration: res("acceleration", "nodal"),
  };
}

// ---- Time Series Evaluation λ(t) --------------------------------------------

function interpolateLinear(ts: readonly number[], vs: readonly number[], t: number): number {
  const n = ts.length;
  if (n === 0) return 0;
  if (t < ts[0]! - 1e-12 || t > ts[n - 1]! + 1e-12) return 0;
  if (n === 1 || t <= ts[0]!) return vs[0]!;
  if (t >= ts[n - 1]!) return vs[n - 1]!;
  for (let i = 0; i + 1 < n; i++) {
    const t0 = ts[i]!;
    const t1 = ts[i + 1]!;
    if (t >= t0 && t <= t1) {
      const dt = t1 - t0;
      const alpha = dt > 1e-15 ? (t - t0) / dt : 0;
      return vs[i]! + alpha * (vs[i + 1]! - vs[i]!);
    }
  }
  return vs[n - 1]!;
}

function evalSeriesParams(type: string, p: Record<string, DeclValue>, t: number): number | null {
  const getNum = (k: string, fb?: number): number | null => {
    const v = p[k];
    if (v === undefined || v === null) return fb ?? null;
    return typeof v === "number" && Number.isFinite(v) ? v : null;
  };
  switch (type) {
    case "Constant": {
      const f = getNum("factor", 1);
      return f ?? 1;
    }
    case "Linear": {
      const f = getNum("factor", 1);
      return (f ?? 1) * t;
    }
    case "Trig": {
      const t0 = getNum("t_start");
      const t1 = getNum("t_end");
      const T = getNum("period");
      const f = getNum("factor", 1);
      const shift = getNum("shift", 0) ?? 0;
      const zs = getNum("zero_shift", 0) ?? 0;
      if (t0 === null || t1 === null || T === null || f === null || !(t1 > t0) || !(T > 0)) return null;
      if (t < t0 || t > t1) return 0;
      const ratio = zs === 0 ? 0 : zs / f;
      if (Math.abs(ratio) > 1) return null;
      const phi = shift - (T / (2 * Math.PI)) * Math.asin(ratio);
      return f * Math.sin((2 * Math.PI * (t - t0)) / T + phi) + zs;
    }
    case "Path": {
      const valsRaw = p["values"];
      if (!Array.isArray(valsRaw) || !valsRaw.every((x) => typeof x === "number" && Number.isFinite(x))) return null;
      const values = valsRaw as number[];
      if (values.length === 0) return null;
      const f = getNum("factor", 1) ?? 1;
      const prepend = p["prepend_zero"] === true;
      const timeRaw = p["time"];
      let tsArr: number[];
      let vsArr = values.map((x) => f * x);
      if (Array.isArray(timeRaw) && timeRaw.every((x) => typeof x === "number" && Number.isFinite(x))) {
        if (prepend || timeRaw.length !== values.length) return null;
        tsArr = timeRaw as number[];
      } else {
        const dt = getNum("dt");
        if (dt === null || !(dt > 0)) return null;
        const t0 = getNum("start_time", 0) ?? 0;
        if (prepend) vsArr = [0, ...vsArr];
        tsArr = vsArr.map((_, i) => t0 + i * dt);
      }
      return interpolateLinear(tsArr, vsArr, t);
    }
    default:
      return null;
  }
}

function findPatternSeries(defs: Definitions, pat: DefObject): { ts: DefObject | null; label: string | null } {
  const patDecl = declOfTag(defs, "pattern", pat.tag);
  const rec = patDecl?.params[0] ?? null;
  const seriesV = rec?.params["series"];
  if (typeof seriesV === "object" && seriesV !== null && !Array.isArray(seriesV) && typeof seriesV["$decl"] === "string") {
    const key = seriesV["$decl"];
    for (const ts of defs.timeSeries) {
      const d = declOfTag(defs, "timeSeries", ts.tag);
      if (d?.key === key) return { ts, label: d.name || ts.groupName };
    }
  }
  const seriesRef = pat.attrs["series_ref"];
  if (typeof seriesRef === "string") {
    const ts = defs.timeSeries.find((x) => x.path === seriesRef) ?? null;
    if (ts) {
      const d = declOfTag(defs, "timeSeries", ts.tag);
      return { ts, label: d?.name || ts.groupName };
    }
    return { ts: null, label: seriesRef.slice(seriesRef.lastIndexOf("/") + 1) };
  }
  if (pat.type === "Plain" && typeof pat.params[0] === "number") {
    const tsTag = pat.params[0];
    const ts = defs.timeSeries.find((x) => x.tag === tsTag) ?? null;
    if (ts) {
      const d = declOfTag(defs, "timeSeries", ts.tag);
      return { ts, label: d?.name || ts.groupName };
    }
  }
  return { ts: null, label: null };
}

/**
 * Evaluate the `timeSeries` multiplier λ(t) for `pat` at time `time`.
 * Falls back to `1` when `time === null` or when the series cannot be sampled.
 */
export function evaluatePatternSeriesFactor(defs: Definitions | null, pat: DefObject, time: number | null): number {
  if (!defs || time === null || !Number.isFinite(time)) return 1;
  const { ts } = findPatternSeries(defs, pat);
  if (!ts) return 1;
  const col = (x: Table | undefined): number[] | null =>
    x && x.columns.length === 1 && x.rows.every((r) => typeof r[0] === "number" && Number.isFinite(r[0]))
      ? x.rows.map((r) => r[0] as number)
      : null;
  const archT = col(ts.tables["time"]);
  const archV = col(ts.tables["values"]);
  if (archT && archV && archT.length === archV.length && archT.length > 0) {
    return interpolateLinear(archT, archV, time);
  }
  const d = declOfTag(defs, "timeSeries", ts.tag);
  const rec = d?.params[0] ?? null;
  if (rec) {
    const val = evalSeriesParams(rec.type, rec.params, time);
    if (val !== null && Number.isFinite(val)) return val;
  }
  // Fallback to positional params on `ts` (`Constant` / `Linear`)
  let factor = 1;
  const fIdx = ts.params.indexOf("-factor");
  if (fIdx >= 0 && typeof ts.params[fIdx + 1] === "number" && Number.isFinite(ts.params[fIdx + 1] as number)) {
    factor = ts.params[fIdx + 1] as number;
  } else if (typeof ts.params[0] === "number" && Number.isFinite(ts.params[0])) {
    factor = ts.params[0];
  }
  if (ts.type === "Constant") return factor;
  if (ts.type === "Linear") return factor * time;
  return 1;
}

// ---- Stage & Pattern Resolution ---------------------------------------------

function resolveActiveStage(s: State): { index: number | null; label: string; time: number | null } {
  const at = s.phase.at;
  if (at?.kind === "results" && s.results) {
    const resSt = s.results.stages[at.stage];
    const time = resSt && resSt.kind !== "mode" && resSt.time[at.step] !== undefined ? resSt.time[at.step]! : null;
    if (resSt && resSt.name) {
      const defMatch = s.definitions?.stages.find((st) => st.name === resSt.name);
      if (defMatch) {
        return { index: defMatch.index, label: `stage ${defMatch.index} · ${defMatch.name}`, time };
      }
      if (s.sequence.status === "ready") {
        const nameCol = s.sequence.stages.columns.indexOf("name");
        const orderCol = s.sequence.stages.columns.indexOf("order");
        if (nameCol >= 0) {
          const rowIdx = s.sequence.stages.rows.findIndex((r) => String(r[nameCol] ?? "") === resSt.name);
          if (rowIdx >= 0) {
            const ord = orderCol >= 0 && typeof s.sequence.stages.rows[rowIdx]![orderCol] === "number"
              ? (s.sequence.stages.rows[rowIdx]![orderCol] as number)
              : rowIdx;
            return { index: ord, label: `stage ${ord} · ${resSt.name}`, time };
          }
        }
      }
    }
    if (s.stage !== null) {
      return { index: s.stage, label: `stage ${s.stage}`, time };
    }
    return { index: null, label: resSt ? `results · ${resSt.name || resSt.id}` : "all stages", time };
  }
  if (s.stage !== null) {
    const defSt = s.definitions?.stages.find((st) => st.index === s.stage);
    const label = defSt ? `stage ${defSt.index} · ${defSt.name}` : `stage ${s.stage}`;
    return { index: s.stage, label, time: null };
  }
  return { index: null, label: "all stages", time: null };
}

function removedPatternTagsUpTo(s: State, maxStage: number | null): Set<number> {
  const out = new Set<number>();
  if (s.sequence.status !== "ready") return out;
  const cols = s.sequence.changes.columns;
  const stIdx = cols.indexOf("stage");
  const opIdx = cols.indexOf("op");
  const whatIdx = cols.indexOf("what");
  const idIdx = cols.indexOf("id");
  const tagIdx = cols.indexOf("tag");
  if (opIdx < 0 || whatIdx < 0) return out;
  for (const row of s.sequence.changes.rows) {
    const st = stIdx >= 0 && typeof row[stIdx] === "number" ? (row[stIdx] as number) : 0;
    if (maxStage !== null && st > maxStage) continue;
    const op = String(row[opIdx] ?? "");
    const what = String(row[whatIdx] ?? "");
    if (op === "remove" && what === "pattern") {
      const rawTag = tagIdx >= 0 ? row[tagIdx] : idIdx >= 0 ? row[idIdx] : undefined;
      const t = Number(rawTag);
      if (Number.isInteger(t)) out.add(t);
    }
  }
  return out;
}

function removedSpStageMapUpTo(s: State, maxStage: number | null): Map<string, number> {
  const out = new Map<string, number>();
  for (const st of s.definitions?.stages ?? []) {
    if (maxStage !== null && st.index > maxStage) continue;
    const tbl = st.removeSp;
    if (!tbl) continue;
    const nCol = tbl.columns.indexOf("node_tag") >= 0 ? tbl.columns.indexOf("node_tag") : tbl.columns.indexOf("target") >= 0 ? tbl.columns.indexOf("target") : 0;
    const dCol = tbl.columns.indexOf("dof") >= 0 ? tbl.columns.indexOf("dof") : 1;
    for (const row of tbl.rows) {
      const node = Number(row[nCol]);
      const dof = Number(row[dCol]);
      if (Number.isInteger(node) && Number.isInteger(dof)) {
        const key = `${node}:${dof}`;
        const prev = out.get(key);
        if (prev === undefined || st.index > prev) {
          out.set(key, st.index);
        }
      }
    }
  }
  return out;
}

// ---- Table Row Parsers ------------------------------------------------------

interface ParsedNodalLoad {
  readonly nodeId: number;
  readonly force: [number, number, number];
  readonly moment: [number, number, number];
}

interface ParsedSp {
  readonly nodeId: number;
  readonly dof: number;
  readonly value: number;
  readonly isHold: boolean;
}

interface ParsedElementLoad {
  readonly elementIds: readonly number[];
  readonly idSpace: "ops_tag" | "fem";
  readonly pgName: string | null;
  readonly kind: "beamUniform" | "beamPoint" | "surfacePressure" | "bodyForce";
  /** For `beamUniform`: `[wy, wz, wx]` in local frame unless `globalDirection` is set. */
  readonly wy: number;
  readonly wz: number;
  readonly wx: number;
  /** Relative position `x/L` in `[0, 1]` for `beamPoint`. */
  readonly xL: number;
  /** Surface pressure scalar (positive along face normal). */
  readonly pressure: number;
  /** Optional global unit/scaled direction vector `[dx, dy, dz]` (from neutral `line_uniform`). */
  readonly globalVector: readonly [number, number, number] | null;
}

function finiteOrZero(v: unknown): number {
  const n = typeof v === "number" ? v : Number(v);
  return Number.isFinite(n) ? n : 0;
}

function resolveTargetNodes(
  s: State,
  targetKind: string,
  targetRaw: unknown,
): number[] {
  if (targetKind === "node" || targetKind === "") {
    const n = Number(targetRaw);
    return Number.isInteger(n) ? [n] : [];
  }
  const name = String(targetRaw ?? "");
  const fromNeutral = s.neutralLoads?.nodeGroups?.[name];
  if (fromNeutral && fromNeutral.length > 0) return [...fromNeutral];
  return [];
}

function parseOpsPatternNodalLoads(s: State, table: Table | undefined, is2D: boolean): ParsedNodalLoad[] {
  if (!table || table.rows.length === 0) return [];
  const out: ParsedNodalLoad[] = [];
  const cols = table.columns;
  const nodeTagIdx = cols.indexOf("node_tag") >= 0 ? cols.indexOf("node_tag") : cols.indexOf("node_id");
  const targetKindIdx = cols.indexOf("target_kind");
  const targetIdx = cols.indexOf("target");
  const forcesIdx = cols.indexOf("forces") >= 0 ? cols.indexOf("forces") : cols.length - 1;

  for (const row of table.rows) {
    const rawForces = row[forcesIdx];
    const fArr = Array.isArray(rawForces) ? rawForces.map(finiteOrZero) : [];
    if (fArr.length === 0) continue;

    let force: [number, number, number];
    let moment: [number, number, number];
    if (is2D && fArr.length <= 3) {
      force = [fArr[0] ?? 0, fArr[1] ?? 0, 0];
      moment = [0, 0, fArr[2] ?? 0];
    } else {
      force = [fArr[0] ?? 0, fArr[1] ?? 0, fArr[2] ?? 0];
      moment = [fArr[3] ?? 0, fArr[4] ?? 0, fArr[5] ?? 0];
    }
    const fMag = Math.hypot(force[0], force[1], force[2]);
    const mMag = Math.hypot(moment[0], moment[1], moment[2]);
    if (fMag === 0 && mMag === 0) continue;

    const nodes =
      nodeTagIdx >= 0
        ? [Number(row[nodeTagIdx])].filter(Number.isInteger)
        : resolveTargetNodes(
            s,
            targetKindIdx >= 0 ? String(row[targetKindIdx] ?? "node") : "node",
            targetIdx >= 0 ? row[targetIdx] : row[0],
          );
    for (const nodeId of nodes) {
      out.push({ nodeId, force, moment });
    }
  }
  return out;
}

function parseNeutralNodalLoads(s: State, table: Table | undefined): ParsedNodalLoad[] {
  if (!table || table.rows.length === 0) return [];
  const out: ParsedNodalLoad[] = [];
  const cols = table.columns;
  const nodeIdIdx = cols.indexOf("node_id");
  const targetKindIdx = cols.indexOf("target_kind");
  const targetIdx = cols.indexOf("target");
  const forceIdx = cols.indexOf("force_xyz");
  const momentIdx = cols.indexOf("moment_xyz");
  const fxIdx = cols.indexOf("fx");
  const fyIdx = cols.indexOf("fy");
  const fzIdx = cols.indexOf("fz");
  const mxIdx = cols.indexOf("mx");
  const myIdx = cols.indexOf("my");
  const mzIdx = cols.indexOf("mz");
  const payloadIdx = cols.indexOf("payload");

  for (const row of table.rows) {
    let nodeId = nodeIdIdx >= 0 ? Number(row[nodeIdIdx]) : Number.NaN;
    let fRaw: unknown = forceIdx >= 0 ? row[forceIdx] : null;
    let mRaw: unknown = momentIdx >= 0 ? row[momentIdx] : null;

    if (payloadIdx >= 0 && row[payloadIdx] !== undefined) {
      const p = typeof row[payloadIdx] === "string" ? JSON.parse(row[payloadIdx] as string) : row[payloadIdx];
      if (p && typeof p === "object" && !Array.isArray(p)) {
        const rec = p as Record<string, unknown>;
        if (!Number.isInteger(nodeId) && rec["node_id"] !== undefined) nodeId = Number(rec["node_id"]);
        if (!fRaw) fRaw = rec["force_xyz"] ?? rec["force"] ?? rec["values"];
        if (!mRaw) mRaw = rec["moment_xyz"] ?? rec["moment"];
      }
    }

    if (!fRaw && (fxIdx >= 0 || fyIdx >= 0 || fzIdx >= 0)) {
      fRaw = [
        fxIdx >= 0 ? row[fxIdx] : 0,
        fyIdx >= 0 ? row[fyIdx] : 0,
        fzIdx >= 0 ? row[fzIdx] : 0,
      ];
    }
    if (!mRaw && (mxIdx >= 0 || myIdx >= 0 || mzIdx >= 0)) {
      mRaw = [
        mxIdx >= 0 ? row[mxIdx] : 0,
        myIdx >= 0 ? row[myIdx] : 0,
        mzIdx >= 0 ? row[mzIdx] : 0,
      ];
    }

    const fArr = Array.isArray(fRaw) ? fRaw.map(finiteOrZero) : [0, 0, 0];
    const mArr = Array.isArray(mRaw) ? mRaw.map(finiteOrZero) : [0, 0, 0];
    const force: [number, number, number] = [fArr[0] ?? 0, fArr[1] ?? 0, fArr[2] ?? 0];
    const moment: [number, number, number] = [mArr[0] ?? 0, mArr[1] ?? 0, mArr[2] ?? 0];
    if (Math.hypot(force[0], force[1], force[2]) === 0 && Math.hypot(moment[0], moment[1], moment[2]) === 0) {
      continue;
    }

    const nodes = Number.isInteger(nodeId)
      ? [nodeId]
      : resolveTargetNodes(
          s,
          targetKindIdx >= 0 ? String(row[targetKindIdx] ?? "node") : "node",
          targetIdx >= 0 ? row[targetIdx] : row[0],
        );
    for (const nid of nodes) {
      out.push({ nodeId: nid, force, moment });
    }
  }
  return out;
}

function parsePatternSps(
  s: State,
  table: Table | undefined,
  removedSpStage: ReadonlyMap<string, number>,
  patternStageIndex: number,
  isHold: boolean,
): ParsedSp[] {
  if (!table || table.rows.length === 0) return [];
  const out: ParsedSp[] = [];
  const cols = table.columns;
  const nodeIdIdx = cols.indexOf("node_id") >= 0 ? cols.indexOf("node_id") : cols.indexOf("node_tag");
  const targetKindIdx = cols.indexOf("target_kind");
  const targetIdx = cols.indexOf("target");
  const dofIdx = cols.indexOf("dof") >= 0 ? cols.indexOf("dof") : 1;
  const valIdx = cols.indexOf("value");
  const homoIdx = cols.indexOf("is_homogeneous");
  const payloadIdx = cols.indexOf("payload");

  for (const row of table.rows) {
    let nodeId = nodeIdIdx >= 0 ? Number(row[nodeIdIdx]) : Number.NaN;
    let dof = dofIdx >= 0 && dofIdx < row.length ? Number(row[dofIdx]) : Number.NaN;
    let value = valIdx >= 0 ? Number(row[valIdx]) : isHold ? 1 : Number.NaN;
    let isHomo = homoIdx >= 0 ? Boolean(Number(row[homoIdx])) : false;

    if (payloadIdx >= 0 && row[payloadIdx] !== undefined) {
      const p = typeof row[payloadIdx] === "string" ? JSON.parse(row[payloadIdx] as string) : row[payloadIdx];
      if (p && typeof p === "object" && !Array.isArray(p)) {
        const rec = p as Record<string, unknown>;
        if (!Number.isInteger(nodeId) && rec["node_id"] !== undefined) nodeId = Number(rec["node_id"]);
        if (!Number.isInteger(dof) && rec["dof"] !== undefined) dof = Number(rec["dof"]);
        if (!Number.isFinite(value) && rec["value"] !== undefined) value = Number(rec["value"]);
        if (rec["is_homogeneous"] !== undefined) isHomo = Boolean(Number(rec["is_homogeneous"]));
      }
    }

    if (isHold && !Number.isFinite(value)) value = 1;
    if (isHomo || !Number.isInteger(dof) || dof < 1) continue;
    if (!isHold && (!Number.isFinite(value) || Math.abs(value) === 0)) continue;

    const nodes = Number.isInteger(nodeId)
      ? [nodeId]
      : resolveTargetNodes(
          s,
          targetKindIdx >= 0 ? String(row[targetKindIdx] ?? "node") : "node",
          targetIdx >= 0 ? row[targetIdx] : row[0],
        );
    for (const nid of nodes) {
      const removedAt = removedSpStage.get(`${nid}:${dof}`);
      // Per compose.py stage slot ordering, stage k's remove_sp runs at slot 5
      // BEFORE stage k's hold_patterns (slot 6) and load_patterns (slot 11).
      if (removedAt !== undefined && patternStageIndex < removedAt) continue;
      out.push({ nodeId: nid, dof, value, isHold });
    }
  }
  return out;
}

function decodeEleLoadParams(
  rawKind: string,
  params: readonly number[],
  elementIds: readonly number[],
  idSpace: "ops_tag" | "fem",
  pgName: string | null,
  is2D: boolean,
): ParsedElementLoad | null {
  const cleanKind = rawKind.replace(/^-/, "");
  if (cleanKind === "beamUniform" || cleanKind === "line_uniform" || cleanKind === "lineUniform") {
    let wy = 0, wz = 0, wx = 0;
    if (params.length === 1) {
      wy = finiteOrZero(params[0]);
    } else if (params.length === 2) {
      if (is2D) {
        wy = finiteOrZero(params[0]);
        wx = finiteOrZero(params[1]);
      } else {
        wy = finiteOrZero(params[0]);
        wz = finiteOrZero(params[1]);
      }
    } else if (params.length >= 3) {
      wy = finiteOrZero(params[0]);
      wz = finiteOrZero(params[1]);
      wx = finiteOrZero(params[2]);
    }
    if (wy === 0 && wz === 0 && wx === 0) return null;
    return {
      elementIds,
      idSpace,
      pgName,
      kind: "beamUniform",
      wy,
      wz,
      wx,
      xL: 0.5,
      pressure: 0,
      globalVector: null,
    };
  }
  if (cleanKind === "beamPoint") {
    let Py = 0, Pz = 0, xL = 0.5, Px = 0;
    if (is2D) {
      Py = finiteOrZero(params[0]);
      xL = params[1] !== undefined && Number.isFinite(params[1]) ? params[1]! : 0.5;
      Px = finiteOrZero(params[2]);
    } else if (params.length === 2) {
      Py = finiteOrZero(params[0]);
      xL = params[1] !== undefined && Number.isFinite(params[1]) ? params[1]! : 0.5;
    } else if (params.length >= 3) {
      Py = finiteOrZero(params[0]);
      Pz = finiteOrZero(params[1]);
      xL = params[2] !== undefined && Number.isFinite(params[2]) ? params[2]! : 0.5;
      Px = finiteOrZero(params[3]);
    }
    if (Py === 0 && Pz === 0 && Px === 0) return null;
    return {
      elementIds,
      idSpace,
      pgName,
      kind: "beamPoint",
      wy: Py,
      wz: Pz,
      wx: Px,
      xL: Math.min(1, Math.max(0, xL)),
      pressure: 0,
      globalVector: null,
    };
  }
  if (cleanKind === "surfaceLoad" || cleanKind === "surfacePressure" || cleanKind === "surface_pressure") {
    const p = finiteOrZero(params[0]);
    if (p === 0) return null;
    return {
      elementIds,
      idSpace,
      pgName,
      kind: "surfacePressure",
      wy: 0,
      wz: 0,
      wx: 0,
      xL: 0.5,
      pressure: p,
      globalVector: null,
    };
  }
  if (cleanKind === "brickW" || cleanKind === "selfWeight") {
    const bx = finiteOrZero(params[0]);
    const by = finiteOrZero(params[1]);
    const bz = finiteOrZero(params[2]);
    if (bx === 0 && by === 0 && bz === 0) return null;
    return {
      elementIds,
      idSpace,
      pgName,
      kind: "bodyForce",
      wy: 0,
      wz: 0,
      wx: 0,
      xL: 0.5,
      pressure: 0,
      globalVector: [bx, by, bz],
    };
  }
  return null;
}

function parseOpsPatternElementLoads(table: Table | undefined, is2D: boolean): ParsedElementLoad[] {
  if (!table || table.rows.length === 0) return [];
  const out: ParsedElementLoad[] = [];
  const cols = table.columns;
  const kindIdx = cols.indexOf("kind");

  if (kindIdx >= 0) {
    // Structured table: `["element_tags", "kind", "params", "target_kind", "target"]`
    const tagsIdx = cols.indexOf("element_tags");
    const paramsIdx = cols.indexOf("params");
    const tkIdx = cols.indexOf("target_kind");
    const tIdx = cols.indexOf("target");
    for (const row of table.rows) {
      const kind = String(row[kindIdx] ?? "");
      const rawTags = tagsIdx >= 0 ? row[tagsIdx] : undefined;
      const eids: number[] = Array.isArray(rawTags)
        ? rawTags.map(Number).filter(Number.isInteger)
        : typeof rawTags === "number" && Number.isInteger(rawTags)
          ? [rawTags]
          : [];
      const tk = tkIdx >= 0 ? String(row[tkIdx] ?? "") : "";
      const targetStr = tIdx >= 0 ? String(row[tIdx] ?? "") : "";
      if (eids.length === 0 && (tk === "element" || tk === "")) {
        const n = Number(targetStr);
        if (Number.isInteger(n)) eids.push(n);
      }
      const pgName = tk === "pg" || tk === "group" ? targetStr : null;
      const rawParams = paramsIdx >= 0 ? row[paramsIdx] : [];
      const pArr = Array.isArray(rawParams) ? rawParams.map(Number) : [Number(rawParams)];
      const parsed = decodeEleLoadParams(kind, pArr, eids, "ops_tag", pgName, is2D);
      if (parsed) out.push(parsed);
    }
    return out;
  }

  // Raw OpenSees token rows (`["-ele", "1", "2", "-type", "-beamUniform", "-10.0"]`)
  // or positional rows (`[eleTag, "-beamUniform", -10, 0, NaN, NaN]`)
  for (const row of table.rows) {
    if (
      row.length >= 3 &&
      (typeof row[0] === "number" || Array.isArray(row[0])) &&
      typeof row[1] === "string" &&
      row[1] !== "-ele" &&
      row[1] !== "-range" &&
      row[1] !== "-eleRange"
    ) {
      const eids = Array.isArray(row[0])
        ? row[0].map(Number).filter(Number.isInteger)
        : Number.isInteger(row[0])
          ? [row[0] as number]
          : [];
      const kind = row[1];
      const params: number[] = [];
      for (let j = 2; j < row.length; j++) {
        const v = Number(row[j]);
        if (Number.isFinite(v)) params.push(v);
      }
      if (kind && eids.length > 0) {
        const parsed = decodeEleLoadParams(kind, params, eids, "ops_tag", null, is2D);
        if (parsed) out.push(parsed);
      }
      continue;
    }

    const tokens = row.map((x) => String(x ?? "").trim()).filter((x) => x.length > 0);
    if (tokens.length === 0) continue;
    const eids: number[] = [];
    let kind = "";
    const params: number[] = [];
    let i = 0;
    while (i < tokens.length) {
      const tok = tokens[i]!;
      if (tok === "-ele") {
        i++;
        while (i < tokens.length && !tokens[i]!.startsWith("-")) {
          const id = Number(tokens[i]);
          if (Number.isInteger(id)) eids.push(id);
          i++;
        }
      } else if (tok === "-range" || tok === "-eleRange") {
        const start = Number(tokens[i + 1]);
        const end = Number(tokens[i + 2]);
        if (Number.isInteger(start) && Number.isInteger(end) && end >= start && end - start <= 100000) {
          for (let e = start; e <= end; e++) eids.push(e);
        }
        i += 3;
      } else if (tok === "-type" && i + 1 < tokens.length) {
        kind = tokens[i + 1]!;
        i += 2;
        while (i < tokens.length) {
          const val = Number(tokens[i]);
          if (Number.isFinite(val)) params.push(val);
          i++;
        }
      } else {
        i++;
      }
    }
    if (kind && eids.length > 0) {
      const parsed = decodeEleLoadParams(kind, params, eids, "ops_tag", null, is2D);
      if (parsed) out.push(parsed);
    }
  }
  return out;
}

function parseNeutralElementLoads(table: Table | undefined, is2D: boolean): ParsedElementLoad[] {
  if (!table || table.rows.length === 0) return [];
  const out: ParsedElementLoad[] = [];
  const cols = table.columns;
  const eidIdx = cols.indexOf("element_id");
  const tkIdx = cols.indexOf("target_kind");
  const tIdx = cols.indexOf("target");
  const pkIdx = cols.indexOf("payload_kind");
  const ltIdx = cols.indexOf("load_type");
  const pjIdx = cols.indexOf("params_json");
  const payloadIdx = cols.indexOf("payload");

  for (const row of table.rows) {
    let eid = eidIdx >= 0 ? Number(row[eidIdx]) : Number.NaN;
    const tk = tkIdx >= 0 ? String(row[tkIdx] ?? "") : "";
    const targetStr = tIdx >= 0 ? String(row[tIdx] ?? "") : "";
    let loadType = ltIdx >= 0 ? String(row[ltIdx] ?? "") : pkIdx >= 0 ? String(row[pkIdx] ?? "") : "";
    let paramsObj: Record<string, unknown> = {};

    const rawJson = pjIdx >= 0 ? row[pjIdx] : payloadIdx >= 0 ? row[payloadIdx] : undefined;
    if (typeof rawJson === "string" && rawJson.trim().startsWith("{")) {
      try {
        paramsObj = JSON.parse(rawJson) as Record<string, unknown>;
      } catch {
        paramsObj = {};
      }
    } else if (rawJson && typeof rawJson === "object" && !Array.isArray(rawJson)) {
      paramsObj = { ...(rawJson as Record<string, unknown>) };
    }
    for (let c = 0; c < cols.length; c++) {
      const col = cols[c]!;
      if (paramsObj[col] === undefined && row[c] !== undefined) {
        paramsObj[col] = row[c];
      }
    }

    if (!Number.isInteger(eid) && paramsObj["element_id"] !== undefined) {
      eid = Number(paramsObj["element_id"]);
    }
    if (!Number.isInteger(eid) && (tk === "element" || tk === "")) {
      const n = Number(targetStr);
      if (Number.isInteger(n)) eid = n;
    }
    if (!loadType && typeof paramsObj["load_type"] === "string") {
      loadType = paramsObj["load_type"] as string;
    }

    const eids = Number.isInteger(eid) ? [eid] : [];
    const pgName = tk === "pg" || tk === "group" || tk === "physical_group" ? targetStr : null;
    if (eids.length === 0 && !pgName) continue;

    const cleanKind = loadType.replace(/^-/, "");
    // Check if neutral record provides an explicit global direction + magnitude (`line_uniform` or `direction`)
    if (Array.isArray(paramsObj["direction"])) {
      const dir = (paramsObj["direction"] as unknown[]).map(finiteOrZero);
      const mag = paramsObj["magnitude"] !== undefined ? finiteOrZero(paramsObj["magnitude"]) : 1;
      const gx = (dir[0] ?? 0) * mag;
      const gy = (dir[1] ?? 0) * mag;
      const gz = (dir[2] ?? 0) * mag;
      if (Math.hypot(gx, gy, gz) > 0) {
        out.push({
          elementIds: eids,
          idSpace: "fem",
          pgName,
          kind: "beamUniform",
          wy: 0,
          wz: 0,
          wx: 0,
          xL: 0.5,
          pressure: 0,
          globalVector: [gx, gy, gz],
        });
        continue;
      }
    }

    if (cleanKind === "beamUniform" || cleanKind === "line_uniform" || cleanKind === "lineUniform") {
      const wy = finiteOrZero(paramsObj["wy"] ?? (Array.isArray(paramsObj["values"]) ? paramsObj["values"][0] : 0));
      const wz = finiteOrZero(paramsObj["wz"] ?? (Array.isArray(paramsObj["values"]) ? paramsObj["values"][1] : 0));
      const wx = finiteOrZero(paramsObj["wx"] ?? (Array.isArray(paramsObj["values"]) ? paramsObj["values"][2] : 0));
      const parsed = decodeEleLoadParams("beamUniform", [wy, wz, wx], eids, "fem", pgName, is2D);
      if (parsed) out.push(parsed);
    } else if (cleanKind === "beamPoint") {
      const Py = finiteOrZero(paramsObj["Py"] ?? paramsObj["py"] ?? 0);
      const Pz = finiteOrZero(paramsObj["Pz"] ?? paramsObj["pz"] ?? 0);
      const Px = finiteOrZero(paramsObj["Px"] ?? paramsObj["px"] ?? 0);
      const xL = paramsObj["xL"] !== undefined ? finiteOrZero(paramsObj["xL"]) : paramsObj["x"] !== undefined ? finiteOrZero(paramsObj["x"]) : 0.5;
      const parsed = decodeEleLoadParams("beamPoint", [Py, Pz, xL, Px], eids, "fem", pgName, is2D);
      if (parsed) out.push(parsed);
    } else if (
      cleanKind === "surfacePressure" ||
      cleanKind === "surface_pressure" ||
      cleanKind === "surfaceLoad"
    ) {
      const p = finiteOrZero(paramsObj["pressure"] ?? paramsObj["p"] ?? paramsObj["value"] ?? paramsObj["magnitude"] ?? 0);
      const parsed = decodeEleLoadParams("surfacePressure", [p], eids, "fem", pgName, is2D);
      if (parsed) out.push(parsed);
    }
  }
  return out;
}

// ---- Pattern / Case Summary Selector (`loadsSummaryOf`) ---------------------

interface ResolvedPatternData {
  readonly entry: LoadPatternEntry;
  readonly nodalLoads: readonly ParsedNodalLoad[];
  readonly elementLoads: readonly ParsedElementLoad[];
  readonly sps: readonly ParsedSp[];
  /** For UniformExcitation: 1-based global direction (1=X, 2=Y, 3=Z) */
  readonly excitationDirection: number | null;
}

function collectResolvedPatterns(s: State): {
  readonly patterns: readonly ResolvedPatternData[];
  readonly activeStageIndex: number | null;
  readonly activeStageLabel: string;
  readonly time: number | null;
  readonly units: LoadsUnitLabels;
} {
  const units = loadsUnitsOf(s);
  const { index: activeStageIndex, label: activeStageLabel, time } = resolveActiveStage(s);
  const removedPatterns = removedPatternTagsUpTo(s, activeStageIndex);
  const removedSpStage = removedSpStageMapUpTo(s, activeStageIndex);
  const is2D = s.ndm.ndm === 2;
  const defs = s.definitions;
  const selected = s.loads.selectedPatterns ? new Set(s.loads.selectedPatterns) : null;

  const out: ResolvedPatternData[] = [];
  const opsNames = new Set<string>();

  const addOpsPattern = (pat: DefObject, stageLabel: string, stageIndex: number) => {
    const d = defs ? declOfTag(defs, "pattern", pat.tag) : null;
    const title = d ? d.name || d.key.split("/").slice(2).join("/") || d.key : pat.groupName;
    if (d?.name) opsNames.add(d.name);
    opsNames.add(pat.groupName);

    const roleAttr = pat.attrs["role"];
    const role = typeof roleAttr === "string" ? roleAttr : null;
    const synth = d?.synth ?? role === "support_reaction";

    const nodalLoads = parseOpsPatternNodalLoads(s, pat.tables["loads"], is2D);
    const elementLoads = parseOpsPatternElementLoads(pat.tables["element_loads"], is2D);
    const spsRegular = parsePatternSps(s, pat.tables["sps"], removedSpStage, stageIndex, false);
    const spsHolds = parsePatternSps(s, pat.tables["sp_holds"], removedSpStage, stageIndex, true);
    const sps = s.loads.includeSupportReactions ? [...spsRegular, ...spsHolds] : spsRegular;

    let excitationDirection: number | null = null;
    if (pat.type === "UniformExcitation") {
      const dirAttr = pat.attrs["direction"] ?? pat.params[0];
      const dirNum = Number(dirAttr);
      if (Number.isInteger(dirNum) && dirNum >= 1 && dirNum <= 6) {
        excitationDirection = dirNum;
      }
    }

    let nodalForcesCount = 0;
    let nodalMomentsCount = 0;
    for (const nl of nodalLoads) {
      if (Math.hypot(nl.force[0], nl.force[1], nl.force[2]) > 0) nodalForcesCount++;
      if (Math.hypot(nl.moment[0], nl.moment[1], nl.moment[2]) > 0) nodalMomentsCount++;
    }
    const spCount = spsRegular.length + (excitationDirection !== null ? 1 : 0);
    const spHoldsCount = spsHolds.length;
    const elemCount = elementLoads.length;
    const total = nodalForcesCount + nodalMomentsCount + elemCount + spCount + (s.loads.includeSupportReactions ? spHoldsCount : 0);

    const isSupportReaction = role === "support_reaction" || (nodalForcesCount === 0 && nodalMomentsCount === 0 && elemCount === 0 && spCount === 0 && spHoldsCount > 0);
    const seriesInfo = defs ? findPatternSeries(defs, pat) : { ts: null, label: null };
    const seriesFactor = s.loads.scaleBySeries ? evaluatePatternSeriesFactor(defs, pat, time) : 1;

    let active: boolean;
    if (selected !== null) {
      active = selected.has(pat.path);
    } else {
      const inStage = activeStageIndex === null || stageIndex <= activeStageIndex;
      const isRemoved = pat.tag !== null && removedPatterns.has(pat.tag);
      const allowedRole = !isSupportReaction || s.loads.includeSupportReactions;
      active = inStage && !isRemoved && allowedRole && total > 0;
    }

    out.push({
      entry: {
        id: pat.path,
        zone: "opensees",
        title,
        stage: stageLabel,
        stageIndex,
        type: pat.type,
        tag: pat.tag,
        role,
        synth,
        seriesFactor,
        seriesLabel: seriesInfo.label,
        active,
        counts: {
          nodalForces: nodalForcesCount,
          nodalMoments: nodalMomentsCount,
          elementLoads: elemCount,
          sps: spCount,
          spHolds: spHoldsCount,
          total,
        },
      },
      nodalLoads,
      elementLoads,
      sps,
      excitationDirection,
    });
  };

  if (defs) {
    for (const p of defs.patterns) addOpsPattern(p, "global", -1);
    for (const st of defs.stages) {
      const stLabel = `stage ${st.index} · ${st.name}`;
      for (const p of st.patterns) addOpsPattern(p, stLabel, st.index);
    }
  }

  // Neutral `/loads/{nodal,element,sp}/{case}`
  const nl = s.neutralLoads;
  if (nl) {
    const cases = new Set<string>([
      ...Object.keys(nl.nodal),
      ...Object.keys(nl.element),
      ...Object.keys(nl.sp),
    ]);
    for (const caseName of [...cases].sort()) {
      const id = `neutral:${caseName}`;
      const nodalLoads = parseNeutralNodalLoads(s, nl.nodal[caseName]);
      const elementLoads = parseNeutralElementLoads(nl.element[caseName], is2D);
      const sps = parsePatternSps(s, nl.sp[caseName], removedSpStage, -1, false);
      let nodalForcesCount = 0;
      let nodalMomentsCount = 0;
      for (const n of nodalLoads) {
        if (Math.hypot(n.force[0], n.force[1], n.force[2]) > 0) nodalForcesCount++;
        if (Math.hypot(n.moment[0], n.moment[1], n.moment[2]) > 0) nodalMomentsCount++;
      }
      const total = nodalForcesCount + nodalMomentsCount + elementLoads.length + sps.length;
      if (total === 0) continue;

      const duplicatedInOps = opsNames.has(caseName);
      const active = selected !== null ? selected.has(id) : !duplicatedInOps;
      out.push({
        entry: {
          id,
          zone: "neutral",
          title: caseName,
          stage: "neutral",
          stageIndex: -1,
          type: "Neutral",
          tag: null,
          role: null,
          synth: false,
          seriesFactor: 1,
          seriesLabel: null,
          active,
          counts: {
            nodalForces: nodalForcesCount,
            nodalMoments: nodalMomentsCount,
            elementLoads: elementLoads.length,
            sps: sps.length,
            spHolds: 0,
            total,
          },
        },
        nodalLoads,
        elementLoads,
        sps,
        excitationDirection: null,
      });
    }
  }

  return { patterns: out, activeStageIndex, activeStageLabel, time, units };
}

/**
 * Pure summary of all load patterns and neutral load cases in `s`, their active
 * status for the current phase/stage, converted peak magnitudes per load kind,
 * and unit labels.
 */
export function loadsSummaryOf(s: State): LoadsSummary {
  const { patterns, activeStageIndex, activeStageLabel, time, units } = collectResolvedPatterns(s);
  const entries = patterns.map((p) => p.entry);
  const activeResolved = patterns.filter((p) => p.entry.active);
  const activePatterns = activeResolved.map((p) => p.entry);
  const is2D = s.ndm.ndm === 2;

  let nodalForces = 0;
  let nodalMoments = 0;
  let elementLoads = 0;
  let beamUniform = 0;
  let beamPoint = 0;
  let surfacePressure = 0;
  let sps = 0;
  let spTranslations = 0;
  let spRotations = 0;
  let baseExcitations = 0;

  const nodalForcesByNode = new Map<number, [number, number, number]>();
  const nodalMomentsByNode = new Map<number, [number, number, number]>();
  const spVecByNode = new Map<number, { trans: [number, number, number]; rot: [number, number, number] }>();

  let peakBeamUniform = 0;
  let peakBeamPoint = 0;
  let peakSurfacePressure = 0;
  let peakBaseExcitation = 0;
  const skippedBeamSet = new Set<DeclPath>();

  // Element lookup maps for 3-D beam vecxz check in summary
  const byOpsTag = new Map<number, DeclPath>();
  const byFemId = new Map<number, DeclPath>();
  const byPgName = new Map<string, DeclPath[]>();
  if (!is2D && s.loads.showElementLoads) {
    for (const [path, d] of Object.entries(s.decls)) {
      if (d.kind !== "element" || !d.element) continue;
      if (d.element.femId !== null) byFemId.set(d.element.femId, path);
      for (const m of d.element.metas) byOpsTag.set(m.tag, path);
      for (const gPath of d.element.groups) {
        const gName = s.decls[gPath]?.name ?? gPath.split("/").pop() ?? "";
        if (gName) {
          let list = byPgName.get(gName);
          if (!list) {
            list = [];
            byPgName.set(gName, list);
          }
          list.push(path);
        }
      }
    }
  }

  for (const pat of activeResolved) {
    const p = pat.entry;
    const factor = p.seriesFactor;
    if (s.loads.showNodalForces) {
      nodalForces += p.counts.nodalForces;
      for (const nl of pat.nodalLoads) {
        const fx = nl.force[0] * factor * units.force.factor;
        const fy = nl.force[1] * factor * units.force.factor;
        const fz = nl.force[2] * factor * units.force.factor;
        if (Math.hypot(fx, fy, fz) > 0) {
          const cur = nodalForcesByNode.get(nl.nodeId) ?? [0, 0, 0];
          nodalForcesByNode.set(nl.nodeId, [cur[0] + fx, cur[1] + fy, cur[2] + fz]);
        }
      }
    }
    if (s.loads.showNodalMoments) {
      nodalMoments += p.counts.nodalMoments;
      for (const nl of pat.nodalLoads) {
        const mx = nl.moment[0] * factor * units.moment.factor;
        const my = nl.moment[1] * factor * units.moment.factor;
        const mz = nl.moment[2] * factor * units.moment.factor;
        if (Math.hypot(mx, my, mz) > 0) {
          const cur = nodalMomentsByNode.get(nl.nodeId) ?? [0, 0, 0];
          nodalMomentsByNode.set(nl.nodeId, [cur[0] + mx, cur[1] + my, cur[2] + mz]);
        }
      }
    }
    if (s.loads.showElementLoads) {
      elementLoads += p.counts.elementLoads;
      for (const el of pat.elementLoads) {
        if (el.kind === "beamUniform") {
          beamUniform++;
          const rawMag = el.globalVector
            ? Math.hypot(el.globalVector[0], el.globalVector[1], el.globalVector[2])
            : Math.hypot(el.wy, el.wz, el.wx);
          const conv = rawMag * Math.abs(factor) * units.beamLoad.factor;
          if (conv > peakBeamUniform) peakBeamUniform = conv;
        } else if (el.kind === "beamPoint") {
          beamPoint++;
          const rawMag = Math.hypot(el.wy, el.wz, el.wx);
          const conv = rawMag * Math.abs(factor) * units.force.factor;
          if (conv > peakBeamPoint) peakBeamPoint = conv;
        } else {
          surfacePressure++;
          const rawMag = el.globalVector
            ? Math.hypot(el.globalVector[0], el.globalVector[1], el.globalVector[2])
            : Math.abs(el.pressure);
          const conv = rawMag * Math.abs(factor) * units.surfacePressure.factor;
          if (conv > peakSurfacePressure) peakSurfacePressure = conv;
        }
        if (
          !is2D &&
          (el.kind === "beamUniform" || el.kind === "beamPoint") &&
          !el.globalVector &&
          (Math.abs(el.wy) > 0 || Math.abs(el.wz) > 0)
        ) {
          const targetDecls: DeclPath[] = [];
          for (const eid of el.elementIds) {
            const decl =
              el.idSpace === "ops_tag"
                ? (byOpsTag.get(eid) ?? byFemId.get(eid))
                : (byFemId.get(eid) ?? byOpsTag.get(eid));
            if (decl) targetDecls.push(decl);
          }
          if (targetDecls.length === 0 && el.pgName) {
            targetDecls.push(...(byPgName.get(el.pgName) ?? []));
          }
          for (const decl of targetDecls) {
            const d = s.decls[decl];
            if (!d) continue;
            const vecxz = elementTransformVecxzFromState(s.decls, d);
            if (!vecxz || Math.hypot(vecxz[0], vecxz[1], vecxz[2]) < 1e-6) {
              skippedBeamSet.add(decl);
            }
          }
        }
      }
    }
    if (s.loads.showSp) {
      sps += p.counts.sps + (s.loads.includeSupportReactions ? p.counts.spHolds : 0);
      for (const sp of pat.sps) {
        const entry = spVecByNode.get(sp.nodeId) ?? { trans: [0, 0, 0], rot: [0, 0, 0] };
        const val = sp.value * factor;
        const isRot = is2D ? sp.dof === 3 : sp.dof >= 4 && sp.dof <= 6;
        if (isRot) {
          spRotations++;
          if (is2D) entry.rot[2] += val * units.rotation.factor;
          else entry.rot[sp.dof - 4]! += val * units.rotation.factor;
        } else if (sp.dof >= 1 && sp.dof <= (is2D ? 2 : 3)) {
          spTranslations++;
          entry.trans[sp.dof - 1]! += val * units.displacement.factor;
        }
        spVecByNode.set(sp.nodeId, entry);
      }
      if (pat.excitationDirection !== null) {
        baseExcitations++;
        const excMag = Math.abs(factor) * units.acceleration.factor;
        if (excMag > peakBaseExcitation) peakBaseExcitation = excMag;
      }
    }
  }

  let peakNodalForce = 0;
  for (const f of nodalForcesByNode.values()) {
    const m = Math.hypot(f[0], f[1], f[2]);
    if (m > peakNodalForce) peakNodalForce = m;
  }
  let peakNodalMoment = 0;
  for (const mVec of nodalMomentsByNode.values()) {
    const m = Math.hypot(mVec[0], mVec[1], mVec[2]);
    if (m > peakNodalMoment) peakNodalMoment = m;
  }
  let peakSpTranslation = 0;
  let peakSpRotation = 0;
  for (const sp of spVecByNode.values()) {
    const mt = Math.hypot(sp.trans[0], sp.trans[1], sp.trans[2]);
    const mr = Math.hypot(sp.rot[0], sp.rot[1], sp.rot[2]);
    if (mt > peakSpTranslation) peakSpTranslation = mt;
    if (mr > peakSpRotation) peakSpRotation = mr;
  }

  const skippedCount = skippedBeamSet.size;
  const notice =
    skippedCount > 0
      ? `${skippedCount} 3-D beam${skippedCount === 1 ? "" : "s"} skipped in 3-D loads: missing geomTransf vecxz (transverse beam loads require a valid non-parallel vecxz; never guessed in 3-D).`
      : null;

  const hasLoads = entries.some((e) => e.counts.total > 0);
  return {
    hasLoads,
    patterns: entries,
    activePatterns,
    activeStageIndex,
    activeStageLabel,
    time,
    units,
    counts: {
      nodalForces,
      nodalMoments,
      elementLoads,
      beamUniform,
      beamPoint,
      surfacePressure,
      sps,
      spTranslations,
      spRotations,
      baseExcitations,
      total: nodalForces + nodalMoments + elementLoads + sps,
    },
    peaks: {
      nodalForce: peakNodalForce,
      nodalMoment: peakNodalMoment,
      beamUniform: peakBeamUniform,
      beamPoint: peakBeamPoint,
      surfacePressure: peakSurfacePressure,
      spTranslation: peakSpTranslation,
      spRotation: peakSpRotation,
      baseExcitation: peakBaseExcitation,
    },
    skippedUnorientedBeams: skippedCount,
    notice,
  };
}

// ---- 3-D Arrow & Comb Primitive Geometry Builder ----------------------------

function orthoBasis(dir: readonly [number, number, number]): {
  u: [number, number, number];
  v: [number, number, number];
} {
  const [dx, dy, dz] = dir;
  const ref: [number, number, number] = Math.abs(dz) < 0.85 ? [0, 0, 1] : [1, 0, 0];
  let ux = dy * ref[2] - dz * ref[1];
  let uy = dz * ref[0] - dx * ref[2];
  let uz = dx * ref[1] - dy * ref[0];
  const uNorm = Math.hypot(ux, uy, uz) || 1;
  ux /= uNorm;
  uy /= uNorm;
  uz /= uNorm;
  const vx = dy * uz - dz * uy;
  const vy = dz * ux - dx * uz;
  const vz = dx * uy - dy * ux;
  return { u: [ux, uy, uz], v: [vx, vy, vz] };
}

function pushSegment(
  linePos: number[],
  lineCol: number[],
  p0: readonly [number, number, number],
  p1: readonly [number, number, number],
  rgb: readonly [number, number, number],
): void {
  linePos.push(p0[0], p0[1], p0[2], p1[0], p1[1], p1[2]);
  lineCol.push(rgb[0], rgb[1], rgb[2], rgb[0], rgb[1], rgb[2]);
}

function pushConeHead(
  conePos: number[],
  coneCol: number[],
  tip: readonly [number, number, number],
  dir: readonly [number, number, number],
  headLen: number,
  headRadius: number,
  rgb: readonly [number, number, number],
): void {
  const { u, v } = orthoBasis(dir);
  const bx = tip[0] - dir[0] * headLen;
  const by = tip[1] - dir[1] * headLen;
  const bz = tip[2] - dir[2] * headLen;
  const nSides = 6;
  const ring: [number, number, number][] = [];
  for (let i = 0; i < nSides; i++) {
    const ang = (2 * Math.PI * i) / nSides;
    const c = Math.cos(ang) * headRadius;
    const s = Math.sin(ang) * headRadius;
    ring.push([
      bx + c * u[0] + s * v[0],
      by + c * u[1] + s * v[1],
      bz + c * u[2] + s * v[2],
    ]);
  }
  for (let i = 0; i < nSides; i++) {
    const a = ring[i]!;
    const b = ring[(i + 1) % nSides]!;
    conePos.push(tip[0], tip[1], tip[2], a[0], a[1], a[2], b[0], b[1], b[2]);
    for (let k = 0; k < 3; k++) coneCol.push(rgb[0], rgb[1], rgb[2]);
  }
}

/**
 * Push a 3-D arrow whose tip touches `targetPoint` (pointing along `dir`) and
 * whose shaft has length `arrowLen`.
 */
function pushArrowToPoint(
  linePos: number[],
  lineCol: number[],
  conePos: number[],
  coneCol: number[],
  targetPoint: readonly [number, number, number],
  dir: readonly [number, number, number],
  arrowLen: number,
  rgb: readonly [number, number, number],
  doubleHead = false,
  spCrossbar = false,
): void {
  const tail: [number, number, number] = [
    targetPoint[0] - dir[0] * arrowLen,
    targetPoint[1] - dir[1] * arrowLen,
    targetPoint[2] - dir[2] * arrowLen,
  ];
  pushSegment(linePos, lineCol, tail, targetPoint, rgb);
  const headLen = Math.min(arrowLen * 0.34, arrowLen * 0.48);
  const headRad = headLen * 0.36;
  pushConeHead(conePos, coneCol, targetPoint, dir, headLen, headRad, rgb);
  if (doubleHead) {
    const secondTip: [number, number, number] = [
      targetPoint[0] - dir[0] * (headLen * 0.72),
      targetPoint[1] - dir[1] * (headLen * 0.72),
      targetPoint[2] - dir[2] * (headLen * 0.72),
    ];
    pushConeHead(conePos, coneCol, secondTip, dir, headLen, headRad, rgb);
  }
  if (spCrossbar) {
    const { u, v } = orthoBasis(dir);
    const barHalf = headRad * 1.35;
    pushSegment(
      linePos,
      lineCol,
      [targetPoint[0] - u[0] * barHalf, targetPoint[1] - u[1] * barHalf, targetPoint[2] - u[2] * barHalf],
      [targetPoint[0] + u[0] * barHalf, targetPoint[1] + u[1] * barHalf, targetPoint[2] + u[2] * barHalf],
      rgb,
    );
    pushSegment(
      linePos,
      lineCol,
      [targetPoint[0] - v[0] * barHalf, targetPoint[1] - v[1] * barHalf, targetPoint[2] - v[2] * barHalf],
      [targetPoint[0] + v[0] * barHalf, targetPoint[1] + v[1] * barHalf, targetPoint[2] + v[2] * barHalf],
      rgb,
    );
  }
}

function buildNodeCoordsMap(
  s: State,
  blobs: BlobStore,
  opts?: BuildLoads3DOptions,
): Map<number, [number, number, number]> {
  const out = new Map<number, [number, number, number]>();
  const mesh = s.mesh;
  if (!mesh) return out;

  if (mesh.nodes && blobs.has(mesh.nodes.ids) && blobs.has(mesh.nodes.xyz)) {
    const ids = blobs.get(mesh.nodes.ids) as Float64Array;
    const xyz = blobs.get(mesh.nodes.xyz) as Float64Array;
    for (let i = 0; i < ids.length; i++) {
      const id = ids[i]!;
      out.set(id, [xyz[i * 3]!, xyz[i * 3 + 1]!, xyz[i * 3 + 2]!]);
    }
  }

  // Also index drawn vertex nodes (covers synthetic test meshes without `mesh.nodes` and deformed vertex positions)
  const linePos = opts?.drawnLinePositions ?? (blobs.has(mesh.linePositions) ? blobs.f32(mesh.linePositions) : null);
  if (linePos && blobs.has(mesh.vertexNodes.line)) {
    const vNodes = blobs.get(mesh.vertexNodes.line) as Float64Array;
    const nVerts = Math.min(vNodes.length, Math.floor(linePos.length / 3));
    for (let i = 0; i < nVerts; i++) {
      const nid = vNodes[i]!;
      if (opts?.drawnLinePositions || !out.has(nid)) {
        out.set(nid, [linePos[i * 3]!, linePos[i * 3 + 1]!, linePos[i * 3 + 2]!]);
      }
    }
  }
  const triPos = opts?.drawnTriPositions ?? (blobs.has(mesh.triPositions) ? blobs.f32(mesh.triPositions) : null);
  if (triPos && blobs.has(mesh.vertexNodes.tri)) {
    const vNodes = blobs.get(mesh.vertexNodes.tri) as Float64Array;
    const nVerts = Math.min(vNodes.length, Math.floor(triPos.length / 3));
    for (let i = 0; i < nVerts; i++) {
      const nid = vNodes[i]!;
      if (opts?.drawnTriPositions || !out.has(nid)) {
        out.set(nid, [triPos[i * 3]!, triPos[i * 3 + 1]!, triPos[i * 3 + 2]!]);
      }
    }
  }

  if (opts?.deformedNodeCoords) {
    for (const [nid, xyz] of opts.deformedNodeCoords) {
      out.set(nid, [xyz[0], xyz[1], xyz[2]]);
    }
  }
  return out;
}

function isElementVisible(s: State, declPath: DeclPath, hiddenSet: ReadonlySet<DeclPath>, inactiveMask: Uint8Array | null, elemIndexMap: ReadonlyMap<DeclPath, number>): boolean {
  if (hiddenSet.has(declPath)) return false;
  const d = s.decls[declPath];
  if (!d) return false;
  if (d.element && d.element.groups.length > 0 && d.element.groups.every((g) => hiddenSet.has(g))) {
    return false;
  }
  if (inactiveMask) {
    const idx = elemIndexMap.get(declPath);
    if (idx !== undefined && inactiveMask[idx] === 1) return false;
  }
  return true;
}

/**
 * Build the 3-D arrow, moment, comb, surface-pressure, and prescribed-SP
 * geometry for the currently active patterns/cases in `s`.
 */
export function buildLoads3D(
  s: State,
  blobs: BlobStore,
  opts?: BuildLoads3DOptions,
): Loads3DGeometry {
  const units = loadsUnitsOf(s);
  const empty = (): Loads3DGeometry => {
    const z = new Float32Array(0);
    return {
      linePositions: z,
      lineColors: z,
      conePositions: z,
      coneColors: z,
      triPositions: z,
      triColors: z,
      nodalForcesCount: 0,
      nodalMomentsCount: 0,
      beamUniformCount: 0,
      beamPointCount: 0,
      surfacePressureCount: 0,
      spCount: 0,
      nodalForceArrows: 0,
      nodalMomentArrows: 0,
      elementLoadArrows: 0,
      spArrows: 0,
      skippedUnorientedBeams: 0,
      skippedBeamDecls: [],
      notice: null,
      maxForceMag: 0,
      maxMomentMag: 0,
      maxElementLoadMag: 0,
      maxBeamUniformMag: 0,
      maxBeamPointMag: 0,
      maxSurfacePressureMag: 0,
      maxSpMag: 0,
      maxSpTranslationMag: 0,
      maxSpRotationMag: 0,
      maxBaseExcitationMag: 0,
      units,
    };
  };

  if (!s.mesh) return empty();

  const { patterns } = collectResolvedPatterns(s);
  const activePatterns = patterns.filter((p) => p.entry.active);
  if (activePatterns.length === 0) return empty();

  const palette = loadsColourPalette(opts?.theme ?? "dark");
  const nodeCoords = buildNodeCoordsMap(s, blobs, opts);
  const hiddenSet = new Set(s.visibility.hidden);
  const inact = inactiveElements(s);
  const elemIndexMap = new Map<DeclPath, number>();
  s.mesh.elements.forEach((p, idx) => elemIndexMap.set(p, idx));

  // Build element lookup maps (by ops_tag, by femId, and by physical_group name)
  const byOpsTag = new Map<number, DeclPath>();
  const byFemId = new Map<number, DeclPath>();
  const byPgName = new Map<string, DeclPath[]>();
  for (const [path, d] of Object.entries(s.decls)) {
    if (d.kind !== "element" || !d.element) continue;
    if (d.element.femId !== null) byFemId.set(d.element.femId, path);
    for (const m of d.element.metas) byOpsTag.set(m.tag, path);
    for (const gPath of d.element.groups) {
      const gName = s.decls[gPath]?.name ?? gPath.split("/").pop() ?? "";
      if (gName) {
        let list = byPgName.get(gName);
        if (!list) {
          list = [];
          byPgName.set(gName, list);
        }
        list.push(path);
      }
    }
  }

  // Index drawn line segments and triangles by DeclPath
  const linePosBlob = opts?.drawnLinePositions ?? (blobs.has(s.mesh.linePositions) ? blobs.f32(s.mesh.linePositions) : new Float32Array(0));
  const lineElemBlob = blobs.has(s.mesh.lineElement) ? blobs.i32(s.mesh.lineElement) : new Int32Array(0);
  const lineSegsByDecl = new Map<DeclPath, { startSeg: number; endSeg: number }>();
  for (let seg = 0; seg < lineElemBlob.length; seg++) {
    const eIdx = lineElemBlob[seg]!;
    const decl = s.mesh.elements[eIdx];
    if (!decl) continue;
    const cur = lineSegsByDecl.get(decl);
    if (cur) cur.endSeg = seg;
    else lineSegsByDecl.set(decl, { startSeg: seg, endSeg: seg });
  }

  const triPosBlob = opts?.drawnTriPositions ?? (blobs.has(s.mesh.triPositions) ? blobs.f32(s.mesh.triPositions) : new Float32Array(0));
  const triElemBlob = blobs.has(s.mesh.triElement) ? blobs.i32(s.mesh.triElement) : new Int32Array(0);
  const trisByDecl = new Map<DeclPath, number[]>();
  for (let tri = 0; tri < triElemBlob.length; tri++) {
    const eIdx = triElemBlob[tri]!;
    const decl = s.mesh.elements[eIdx];
    if (!decl) continue;
    let list = trisByDecl.get(decl);
    if (!list) {
      list = [];
      trisByDecl.set(decl, list);
    }
    list.push(tri);
  }

  // Helper to get element node IDs from connectivity block or inlineNodes
  const elementNodesOf = (d: Decl): number[] => {
    if (!d.element) return [];
    if (d.element.inlineNodes.length > 0) return [...d.element.inlineNodes];
    if (d.element.cell) {
      const blk = s.blocks[d.element.cell.block];
      if (blk && blobs.has(blk.connectivity)) {
        const conn = blobs.get(blk.connectivity) as Float64Array;
        const start = d.element.cell.row * blk.npe;
        const nodes: number[] = [];
        for (let k = 0; k < blk.npe; k++) nodes.push(conn[start + k]!);
        return nodes;
      }
    }
    return [];
  };

  // Characteristic glyph length from model bounding sphere
  const radius = s.mesh.radius > 0 && Number.isFinite(s.mesh.radius) ? s.mesh.radius : 1;
  const baseLen = Math.max(radius * 0.22, 1e-4) * s.loads.scale;
  const is2D = s.ndm.ndm === 2;

  // 1. Aggregate nodal forces and moments by nodeId across active patterns
  const nodalForcesByNode = new Map<number, [number, number, number]>();
  const nodalMomentsByNode = new Map<number, [number, number, number]>();
  const spVecByNode = new Map<number, { trans: [number, number, number]; rot: [number, number, number] }>();
  let maxBaseExcitationMag = 0;

  for (const pat of activePatterns) {
    const factor = pat.entry.seriesFactor;
    if (s.loads.showNodalForces || s.loads.showNodalMoments) {
      for (const nl of pat.nodalLoads) {
        if (!nodeCoords.has(nl.nodeId)) continue;
        if (s.loads.showNodalForces) {
          const fx = nl.force[0] * factor * units.force.factor;
          const fy = nl.force[1] * factor * units.force.factor;
          const fz = nl.force[2] * factor * units.force.factor;
          if (Math.hypot(fx, fy, fz) > 0) {
            const cur = nodalForcesByNode.get(nl.nodeId) ?? [0, 0, 0];
            nodalForcesByNode.set(nl.nodeId, [cur[0] + fx, cur[1] + fy, cur[2] + fz]);
          }
        }
        if (s.loads.showNodalMoments) {
          const mx = nl.moment[0] * factor * units.moment.factor;
          const my = nl.moment[1] * factor * units.moment.factor;
          const mz = nl.moment[2] * factor * units.moment.factor;
          if (Math.hypot(mx, my, mz) > 0) {
            const cur = nodalMomentsByNode.get(nl.nodeId) ?? [0, 0, 0];
            nodalMomentsByNode.set(nl.nodeId, [cur[0] + mx, cur[1] + my, cur[2] + mz]);
          }
        }
      }
    }

    if (s.loads.showSp) {
      for (const sp of pat.sps) {
        if (!nodeCoords.has(sp.nodeId)) continue;
        const entry = spVecByNode.get(sp.nodeId) ?? { trans: [0, 0, 0], rot: [0, 0, 0] };
        const val = sp.value * factor;
        if (is2D) {
          if (sp.dof === 1) entry.trans[0] += val * units.displacement.factor;
          else if (sp.dof === 2) entry.trans[1] += val * units.displacement.factor;
          else if (sp.dof === 3) entry.rot[2] += val * units.rotation.factor;
        } else {
          if (sp.dof >= 1 && sp.dof <= 3) entry.trans[sp.dof - 1]! += val * units.displacement.factor;
          else if (sp.dof >= 4 && sp.dof <= 6) entry.rot[sp.dof - 4]! += val * units.rotation.factor;
        }
        spVecByNode.set(sp.nodeId, entry);
      }
      if (pat.excitationDirection !== null) {
        const excMag = Math.abs(factor) * units.acceleration.factor;
        if (excMag > maxBaseExcitationMag) maxBaseExcitationMag = excMag;
      }
    }
  }

  let maxForceMag = 0;
  for (const f of nodalForcesByNode.values()) {
    const m = Math.hypot(f[0], f[1], f[2]);
    if (m > maxForceMag) maxForceMag = m;
  }
  let maxMomentMag = 0;
  for (const mVec of nodalMomentsByNode.values()) {
    const m = Math.hypot(mVec[0], mVec[1], mVec[2]);
    if (m > maxMomentMag) maxMomentMag = m;
  }
  let maxSpTranslationMag = 0;
  let maxSpRotationMag = 0;
  for (const sp of spVecByNode.values()) {
    const mt = Math.hypot(sp.trans[0], sp.trans[1], sp.trans[2]);
    const mr = Math.hypot(sp.rot[0], sp.rot[1], sp.rot[2]);
    if (mt > maxSpTranslationMag) maxSpTranslationMag = mt;
    if (mr > maxSpRotationMag) maxSpRotationMag = mr;
  }
  const maxSpMag = Math.max(maxSpTranslationMag, maxSpRotationMag, maxBaseExcitationMag);

  const linePos: number[] = [];
  const lineCol: number[] = [];
  const conePos: number[] = [];
  const coneCol: number[] = [];

  let nodalForcesCount = 0;
  let nodalMomentsCount = 0;
  let beamUniformCount = 0;
  let beamPointCount = 0;
  let surfacePressureCount = 0;
  let spCount = 0;
  const skippedBeamSet = new Set<DeclPath>();

  // Render Nodal Forces
  if (s.loads.showNodalForces && maxForceMag > 0) {
    for (const [nodeId, f] of nodalForcesByNode) {
      const pt = nodeCoords.get(nodeId);
      if (!pt) continue;
      const mag = Math.hypot(f[0], f[1], f[2]);
      if (mag <= 1e-30) continue;
      const dir: [number, number, number] = [f[0] / mag, f[1] / mag, f[2] / mag];
      const len = baseLen * (0.35 + 0.65 * (mag / maxForceMag));
      pushArrowToPoint(linePos, lineCol, conePos, coneCol, pt, dir, len, palette.force, false, false);
      nodalForcesCount++;
    }
  }

  // Render Nodal Moments (double-headed arrows along moment vector)
  if (s.loads.showNodalMoments && maxMomentMag > 0) {
    for (const [nodeId, mVec] of nodalMomentsByNode) {
      const pt = nodeCoords.get(nodeId);
      if (!pt) continue;
      const mag = Math.hypot(mVec[0], mVec[1], mVec[2]);
      if (mag <= 1e-30) continue;
      const dir: [number, number, number] = [mVec[0] / mag, mVec[1] / mag, mVec[2] / mag];
      const len = baseLen * (0.35 + 0.65 * (mag / maxMomentMag));
      pushArrowToPoint(linePos, lineCol, conePos, coneCol, pt, dir, len, palette.moment, true, false);
      nodalMomentsCount++;
    }
  }

  // Render Prescribed Non-Homogeneous SPs & Base Excitation
  if (s.loads.showSp) {
    for (const [nodeId, sp] of spVecByNode) {
      const pt = nodeCoords.get(nodeId);
      if (!pt) continue;
      const tMag = Math.hypot(sp.trans[0], sp.trans[1], sp.trans[2]);
      if (tMag > 1e-30 && maxSpTranslationMag > 0) {
        const dir: [number, number, number] = [sp.trans[0] / tMag, sp.trans[1] / tMag, sp.trans[2] / tMag];
        const len = baseLen * 0.85 * (0.35 + 0.65 * (tMag / maxSpTranslationMag));
        pushArrowToPoint(linePos, lineCol, conePos, coneCol, pt, dir, len, palette.sp, false, true);
        spCount++;
      }
      const rMag = Math.hypot(sp.rot[0], sp.rot[1], sp.rot[2]);
      if (rMag > 1e-30 && maxSpRotationMag > 0) {
        const dir: [number, number, number] = [sp.rot[0] / rMag, sp.rot[1] / rMag, sp.rot[2] / rMag];
        const len = baseLen * 0.85 * (0.35 + 0.65 * (rMag / maxSpRotationMag));
        pushArrowToPoint(linePos, lineCol, conePos, coneCol, pt, dir, len, palette.sp, true, true);
        spCount++;
      }
    }
    // UniformExcitation base motion indicator at model center
    for (const pat of activePatterns) {
      if (pat.excitationDirection !== null) {
        const dIdx = pat.excitationDirection;
        const dir: [number, number, number] =
          dIdx === 1 ? [1, 0, 0] : dIdx === 2 ? [0, 1, 0] : [0, 0, 1];
        pushArrowToPoint(
          linePos,
          lineCol,
          conePos,
          coneCol,
          s.mesh.center,
          dir,
          baseLen * 1.1,
          palette.sp,
          false,
          true,
        );
        spCount++;
      }
    }
  }

  // Resolve and Render Distributed Element Loads (1D beamUniform/beamPoint & 2D/3D surfacePressure)
  let maxBeamUniformMag = 0;
  let maxBeamPointMag = 0;
  let maxSurfacePressureMag = 0;
  interface ResolvedBeamLoadItem {
    readonly decl: DeclPath;
    readonly kind: "beamUniform" | "beamPoint";
    readonly pStart: [number, number, number];
    readonly pEnd: [number, number, number];
    readonly worldVec: [number, number, number];
    readonly mag: number;
    readonly xL: number;
  }
  interface ResolvedSurfaceLoadItem {
    readonly decl: DeclPath;
    readonly points: readonly [number, number, number][];
    readonly normal: [number, number, number];
    readonly mag: number;
  }

  const beamItems: ResolvedBeamLoadItem[] = [];
  const surfaceItems: ResolvedSurfaceLoadItem[] = [];

  if (s.loads.showElementLoads) {
    for (const pat of activePatterns) {
      const factor = pat.entry.seriesFactor;
      for (const elLoad of pat.elementLoads) {
        const targetDecls: DeclPath[] = [];
        for (const eid of elLoad.elementIds) {
          const decl =
            elLoad.idSpace === "ops_tag"
              ? (byOpsTag.get(eid) ?? byFemId.get(eid))
              : (byFemId.get(eid) ?? byOpsTag.get(eid));
          if (decl) targetDecls.push(decl);
        }
        if (targetDecls.length === 0 && elLoad.pgName) {
          const pgDecls = byPgName.get(elLoad.pgName) ?? [];
          targetDecls.push(...pgDecls);
        }

        for (const decl of targetDecls) {
          if (!isElementVisible(s, decl, hiddenSet, inact?.mask ?? null, elemIndexMap)) continue;
          const d = s.decls[decl];
          if (!d) continue;

          if (elLoad.kind === "beamUniform" || elLoad.kind === "beamPoint") {
            // Find beam endpoints pStart and pEnd
            let pStart: [number, number, number] | null = null;
            let pEnd: [number, number, number] | null = null;
            const segRange = lineSegsByDecl.get(decl);
            if (segRange && linePosBlob.length >= (segRange.endSeg + 1) * 6) {
              const o0 = segRange.startSeg * 6;
              const o1 = segRange.endSeg * 6;
              pStart = [linePosBlob[o0]!, linePosBlob[o0 + 1]!, linePosBlob[o0 + 2]!];
              pEnd = [linePosBlob[o1 + 3]!, linePosBlob[o1 + 4]!, linePosBlob[o1 + 5]!];
            } else {
              const eNodes = elementNodesOf(d);
              if (eNodes.length >= 2) {
                const c0 = nodeCoords.get(eNodes[0]!);
                const c1 = nodeCoords.get(eNodes[eNodes.length - 1]!);
                if (c0 && c1) {
                  pStart = [c0[0], c0[1], c0[2]];
                  pEnd = [c1[0], c1[1], c1[2]];
                }
              }
            }
            if (!pStart || !pEnd) continue;

            const dx = pEnd[0] - pStart[0];
            const dy = pEnd[1] - pStart[1];
            const dz = pEnd[2] - pStart[2];
            const L = Math.hypot(dx, dy, dz);
            if (!Number.isFinite(L) || L < 1e-12) continue;
            const xLocal: [number, number, number] = [dx / L, dy / L, dz / L];

            const unitConv = elLoad.kind === "beamPoint" ? units.force.factor : units.beamLoad.factor;
            let worldVec: [number, number, number];

            if (elLoad.globalVector) {
              worldVec = [
                elLoad.globalVector[0] * factor * unitConv,
                elLoad.globalVector[1] * factor * unitConv,
                elLoad.globalVector[2] * factor * unitConv,
              ];
            } else {
              const hasTransverse = Math.abs(elLoad.wy) > 0 || Math.abs(elLoad.wz) > 0;
              let yLocal: [number, number, number] = [0, 1, 0];
              let zLocal: [number, number, number] = [0, 0, 1];
              if (hasTransverse) {
                if (is2D) {
                  const axes = beamLocalAxes(xLocal, [0, 0, 1]);
                  yLocal = axes.yLocal;
                  zLocal = axes.zLocal;
                } else {
                  // Strict fail-closed 3-D beam local frame rule: require explicit non-parallel vecxz
                  const vecxz = elementTransformVecxzFromState(s.decls, d);
                  if (!vecxz) {
                    skippedBeamSet.add(decl);
                    continue;
                  }
                  const vNorm = Math.hypot(vecxz[0], vecxz[1], vecxz[2]);
                  const dot = vecxz[0] * xLocal[0] + vecxz[1] * xLocal[1] + vecxz[2] * xLocal[2];
                  const zx = vecxz[0] - dot * xLocal[0];
                  const zy = vecxz[1] - dot * xLocal[1];
                  const zz = vecxz[2] - dot * xLocal[2];
                  const zNorm = Math.hypot(zx, zy, zz);
                  if (
                    !Number.isFinite(vNorm) ||
                    vNorm < 1e-12 ||
                    !Number.isFinite(zNorm) ||
                    zNorm < 1e-6 ||
                    zNorm < 1e-6 * vNorm
                  ) {
                    skippedBeamSet.add(decl);
                    continue;
                  }
                  const axes = beamLocalAxes(xLocal, vecxz);
                  yLocal = axes.yLocal;
                  zLocal = axes.zLocal;
                }
              }
              const wx = elLoad.wx * factor * unitConv;
              const wy = elLoad.wy * factor * unitConv;
              const wz = elLoad.wz * factor * unitConv;
              worldVec = [
                wx * xLocal[0] + wy * yLocal[0] + wz * zLocal[0],
                wx * xLocal[1] + wy * yLocal[1] + wz * zLocal[1],
                wx * xLocal[2] + wy * yLocal[2] + wz * zLocal[2],
              ];
            }

            const mag = Math.hypot(worldVec[0], worldVec[1], worldVec[2]);
            if (mag <= 1e-30) continue;
            if (elLoad.kind === "beamPoint") {
              if (mag > maxBeamPointMag) maxBeamPointMag = mag;
            } else {
              if (mag > maxBeamUniformMag) maxBeamUniformMag = mag;
            }
            beamItems.push({
              decl,
              kind: elLoad.kind,
              pStart,
              pEnd,
              worldVec,
              mag,
              xL: elLoad.xL,
            });
          } else if (elLoad.kind === "surfacePressure" || elLoad.kind === "bodyForce") {
            const tris = trisByDecl.get(decl);
            const pts: [number, number, number][] = [];
            let nx = 0, ny = 0, nz = 0;

            if (tris && tris.length > 0 && triPosBlob.length > 0) {
              let cx = 0, cy = 0, cz = 0, nVerts = 0;
              for (const tIdx of tris) {
                const o = tIdx * 9;
                if (o + 8 >= triPosBlob.length) continue;
                const ax = triPosBlob[o]!, ay = triPosBlob[o + 1]!, az = triPosBlob[o + 2]!;
                const bx = triPosBlob[o + 3]!, by = triPosBlob[o + 4]!, bz = triPosBlob[o + 5]!;
                const dx = triPosBlob[o + 6]!, dy = triPosBlob[o + 7]!, dz = triPosBlob[o + 8]!;
                const ux = bx - ax, uy = by - ay, uz = bz - az;
                const vx = dx - ax, vy = dy - ay, vz = dz - az;
                nx += uy * vz - uz * vy;
                ny += uz * vx - ux * vz;
                nz += ux * vy - uy * vx;
                cx += ax + bx + dx;
                cy += ay + by + dy;
                cz += az + bz + dz;
                nVerts += 3;
                pts.push([(ax + bx + dx) / 3, (ay + by + dy) / 3, (az + bz + dz) / 3]);
              }
              if (nVerts > 0) {
                pts.unshift([cx / nVerts, cy / nVerts, cz / nVerts]);
              }
            } else {
              const eNodes = elementNodesOf(d);
              const coords = eNodes.map((nid) => nodeCoords.get(nid)).filter((c): c is [number, number, number] => c !== undefined);
              if (coords.length >= 3) {
                const [a, b, c] = coords as [[number, number, number], [number, number, number], [number, number, number]];
                const ux = b[0] - a[0], uy = b[1] - a[1], uz = b[2] - a[2];
                const vx = c[0] - a[0], vy = c[1] - a[1], vz = c[2] - a[2];
                nx = uy * vz - uz * vy;
                ny = uz * vx - ux * vz;
                nz = ux * vy - uy * vx;
                const cx = coords.reduce((acc, p) => acc + p[0], 0) / coords.length;
                const cy = coords.reduce((acc, p) => acc + p[1], 0) / coords.length;
                const cz = coords.reduce((acc, p) => acc + p[2], 0) / coords.length;
                pts.push([cx, cy, cz]);
              }
            }

            if (pts.length === 0) continue;
            let normal: [number, number, number];
            let mag: number;
            if (elLoad.globalVector) {
              const gx = elLoad.globalVector[0] * factor * units.surfacePressure.factor;
              const gy = elLoad.globalVector[1] * factor * units.surfacePressure.factor;
              const gz = elLoad.globalVector[2] * factor * units.surfacePressure.factor;
              mag = Math.hypot(gx, gy, gz);
              if (mag <= 1e-30) continue;
              normal = [gx / mag, gy / mag, gz / mag];
            } else {
              const nNorm = Math.hypot(nx, ny, nz);
              if (!Number.isFinite(nNorm) || nNorm < 1e-12) continue;
              const pVal = elLoad.pressure * factor * units.surfacePressure.factor;
              mag = Math.abs(pVal);
              if (mag <= 1e-30) continue;
              const sign = Math.sign(pVal);
              normal = [(sign * nx) / nNorm, (sign * ny) / nNorm, (sign * nz) / nNorm];
            }
            if (mag > maxSurfacePressureMag) maxSurfacePressureMag = mag;
            surfaceItems.push({ decl, points: pts.slice(0, 5), normal, mag });
          }
        }
      }
    }

    // Emit 3-D primitives for beam loads
    for (const item of beamItems) {
      const dir: [number, number, number] = [
        item.worldVec[0] / item.mag,
        item.worldVec[1] / item.mag,
        item.worldVec[2] / item.mag,
      ];
      const kindMax = item.kind === "beamPoint" ? maxBeamPointMag : maxBeamUniformMag;
      const len = baseLen * 0.75 * (0.35 + 0.65 * (item.mag / (kindMax || 1)));
      if (item.kind === "beamPoint") {
        const pt: [number, number, number] = [
          item.pStart[0] + item.xL * (item.pEnd[0] - item.pStart[0]),
          item.pStart[1] + item.xL * (item.pEnd[1] - item.pStart[1]),
          item.pStart[2] + item.xL * (item.pEnd[2] - item.pStart[2]),
        ];
        pushArrowToPoint(linePos, lineCol, conePos, coneCol, pt, dir, len, palette.elementLoad, false, false);
        beamPointCount++;
      } else {
        const nTicks = 5;
        const tails: [number, number, number][] = [];
        for (let i = 0; i < nTicks; i++) {
          const u = i / (nTicks - 1);
          const pt: [number, number, number] = [
            item.pStart[0] + u * (item.pEnd[0] - item.pStart[0]),
            item.pStart[1] + u * (item.pEnd[1] - item.pStart[1]),
            item.pStart[2] + u * (item.pEnd[2] - item.pStart[2]),
          ];
          const tail: [number, number, number] = [
            pt[0] - dir[0] * len,
            pt[1] - dir[1] * len,
            pt[2] - dir[2] * len,
          ];
          tails.push(tail);
          pushArrowToPoint(linePos, lineCol, conePos, coneCol, pt, dir, len, palette.elementLoad, false, false);
        }
        // Top profile bar connecting the comb arrow tails
        for (let i = 0; i + 1 < tails.length; i++) {
          pushSegment(linePos, lineCol, tails[i]!, tails[i + 1]!, palette.elementLoad);
        }
        beamUniformCount++;
      }
    }

    // Emit 3-D primitives for surface pressure loads
    for (const item of surfaceItems) {
      const len = baseLen * 0.65 * (0.35 + 0.65 * (item.mag / (maxSurfacePressureMag || 1)));
      for (const pt of item.points) {
        pushArrowToPoint(linePos, lineCol, conePos, coneCol, pt, item.normal, len, palette.elementLoad, false, false);
      }
      surfacePressureCount++;
    }
  }

  const maxElementLoadMag = Math.max(maxBeamUniformMag, maxBeamPointMag, maxSurfacePressureMag);
  const skippedBeamDecls = [...skippedBeamSet];
  const skippedCount = skippedBeamDecls.length;
  const notice =
    skippedCount > 0
      ? `${skippedCount} 3-D beam${skippedCount === 1 ? "" : "s"} skipped in 3-D loads: missing geomTransf vecxz (transverse beam loads require a valid non-parallel vecxz; never guessed in 3-D).`
      : null;

  const linePositions = new Float32Array(linePos);
  const lineColors = new Float32Array(lineCol);
  const conePositions = new Float32Array(conePos);
  const coneColors = new Float32Array(coneCol);
  const elementLoadArrows = beamUniformCount * 5 + beamPointCount + surfacePressureCount;

  return {
    linePositions,
    lineColors,
    conePositions,
    coneColors,
    triPositions: conePositions,
    triColors: coneColors,
    nodalForcesCount,
    nodalMomentsCount,
    beamUniformCount,
    beamPointCount,
    surfacePressureCount,
    spCount,
    nodalForceArrows: nodalForcesCount,
    nodalMomentArrows: nodalMomentsCount,
    elementLoadArrows,
    spArrows: spCount,
    skippedUnorientedBeams: skippedCount,
    skippedBeamDecls,
    notice,
    maxForceMag,
    maxMomentMag,
    maxElementLoadMag,
    maxBeamUniformMag,
    maxBeamPointMag,
    maxSurfacePressureMag,
    maxSpMag,
    maxSpTranslationMag,
    maxSpRotationMag,
    maxBaseExcitationMag,
    units,
  };
}
