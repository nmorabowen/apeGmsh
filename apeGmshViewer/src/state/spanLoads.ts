// Exact analytical beam-span internal-force reconstruction (Slice E2).
//
// Pure state/math module: no three.js, no DOM, and no panel imports, so every
// function runs in Node tests and obeys the state-layer dependency wall.
//
// Parity targets:
// - `src/apeGmsh/results/_beam_diagrams.py` (`BeamSpanSegmentLoad`,
//   `BeamSpanDiagram`, `parse_ele_load_beam_uniform`,
//   `reconstruct_beam_element_diagram`),
// - `src/apeGmsh/opensees/_response_catalog.py` (`local_force_station_signs`),
// - `architecture/h5-schema.md` (`/opensees/patterns/<P>/element_loads` and
//   exact span-load reconstruction contract, Slice E1).

import { ELEMENT_SYNTAX } from "../chain/signatures.ts";
import type { Param } from "../model/types.ts";
import type { LadrunoLineStationStep, LadrunoProvenance, LineStationComponent } from "../reader/ladruno.ts";
import type { ResultsPairing } from "./results.ts";
import type {
  AnalysisDef,
  Decl,
  DeclParams,
  DeclPath,
  DeclRecord,
  DeclValue,
  Definitions,
  DefObject,
  ResultsStage,
  SequenceZone,
  SpatialNdm,
  State,
  Table,
} from "./types.ts";

export type LadrunoElementJoin =
  | {
      ok: true;
      space: "ops_tag" | "fem";
      /** Map from `.ladruno` element ID to `DeclPath` in `s.decls`. */
      byLadrunoId: Map<number, DeclPath>;
      /** Inverse map from `DeclPath` to `.ladruno` element ID. */
      byDeclPath: Map<DeclPath, number>;
    }
  | {
      ok: false;
      reason: string;
    };

export function joinLadrunoElements(
  decls: Readonly<Record<DeclPath, Decl>>,
  pairing: ResultsPairing,
  provenance?: LadrunoProvenance | string | null,
): LadrunoElementJoin {
  if (pairing.state !== "paired") {
    return { ok: false, reason: `results are not paired with the model (${pairing.state})` };
  }
  const isFemSpace =
    typeof provenance === "string"
      ? provenance === "fem"
      : provenance?.state === "present" &&
        provenance.attrs.element_index_space?.state === "present" &&
        provenance.attrs.element_index_space.value === "fem";

  if (pairing.by === "fem_hash") {
    if (!isFemSpace) {
      return {
        ok: false,
        reason:
          'paired by fem_hash only and INFO/PROVENANCE@element_index_space is not "fem": .ladruno element IDs are OpenSees tags and cannot be joined without /opensees/element_meta',
      };
    }
    const byLadrunoId = new Map<number, DeclPath>();
    const byDeclPath = new Map<DeclPath, number>();
    for (const [path, d] of Object.entries(decls)) {
      if (d.kind === "element" && d.element?.femId !== null && d.element?.femId !== undefined) {
        byLadrunoId.set(d.element.femId, path);
        byDeclPath.set(path, d.element.femId);
      }
    }
    return { ok: true, space: "fem", byLadrunoId, byDeclPath };
  }

  if (isFemSpace) {
    const byLadrunoId = new Map<number, DeclPath>();
    const byDeclPath = new Map<DeclPath, number>();
    for (const [path, d] of Object.entries(decls)) {
      if (d.kind === "element" && d.element?.femId !== null && d.element?.femId !== undefined) {
        byLadrunoId.set(d.element.femId, path);
        byDeclPath.set(path, d.element.femId);
      }
    }
    return { ok: true, space: "fem", byLadrunoId, byDeclPath };
  }

  const byLadrunoId = new Map<number, DeclPath>();
  const byDeclPath = new Map<DeclPath, number>();
  for (const [path, d] of Object.entries(decls)) {
    if (d.kind !== "element" || !d.element) continue;
    for (const m of d.element.metas) {
      byLadrunoId.set(m.tag, path);
      if (!byDeclPath.has(path)) byDeclPath.set(path, m.tag);
    }
  }
  return { ok: true, space: "ops_tag", byLadrunoId, byDeclPath };
}

/** The 6 internal-force components supported by exact span equilibrium reconstruction. */
export type BeamForceComponent = "N" | "Vy" | "Vz" | "T" | "My" | "Mz";

export const BEAM_FORCE_COMPONENTS: readonly BeamForceComponent[] = ["N", "Vy", "Vz", "T", "My", "Mz"];

export function isBeamForceComponent(comp: LineStationComponent | string): comp is BeamForceComponent {
  return (
    comp === "N" ||
    comp === "Vy" ||
    comp === "Vz" ||
    comp === "T" ||
    comp === "My" ||
    comp === "Mz"
  );
}

/** Normalize `2 | 3 | null | undefined | SpatialNdm` into `2 | 3 | null`. */
export function unwrapSpatialNdm(ndm: 2 | 3 | null | undefined | SpatialNdm): 2 | 3 | null {
  if (ndm === 2 || ndm === 3) return ndm;
  if (ndm !== null && ndm !== undefined && typeof ndm === "object" && (ndm.ndm === 2 || ndm.ndm === 3)) {
    return ndm.ndm;
  }
  return null;
}

/**
 * Partial or full-span trapezoidal distributed load on a 1-D beam element in
 * local element coordinates (`0 <= aL < bL <= 1`).
 *
 * Components `(wy, wz, wx)` are expressed in the element local coordinate
 * system `(x_local, y_local, z_local)` and already include the active pattern
 * load factor `\lambda(t)` when scaled for a specific analysis step.
 */
export interface BeamSpanSegmentLoad {
  readonly aL: number;
  readonly bL: number;
  readonly wya: number;
  readonly wyb: number;
  readonly wza: number;
  readonly wzb: number;
  readonly wxa: number;
  readonly wxb: number;
}

/**
 * Exact internal-force diagram reconstructed along a single 1-D beam element.
 *
 * Sign conventions match `.ladruno` `section_force` (`LOCAL_FORCE_STATION_SIGNS`
 * in `src/reader/ladruno.ts` and `src/apeGmsh/results/_beam_diagrams.py`):
 * - `N(0)  = -N_I`,  `N(x)  = N(0)  - \int_0^x w_x(s) ds`
 * - `Vy(0) = +Vy_I`, `Vy(x) = Vy(0) + \int_0^x w_y(s) ds`
 * - `Vz(0) = -Vz_I`, `Vz(x) = Vz(0) - \int_0^x w_z(s) ds`
 * - `T(0)  = -T_I`,  `T(x)  = T(0)`
 * - `Mz(0) = -Mz_I`, `Mz(x) = Mz(0) + \int_0^x Vy(s) ds`
 * - `My(0) = -My_I`, `My(x) = My(0) + \int_0^x Vz(s) ds`
 */
export interface BeamSpanDiagram {
  readonly elementId: number;
  readonly length: number;
  /** Natural coordinates in `[-1, +1]`, monotonically increasing from `-1` to `+1`. */
  readonly xi: Float64Array;
  /** Local coordinates in `[0, L]`, `x = 0.5 * (xi + 1) * L`. */
  readonly x: Float64Array;
  readonly N: Float64Array;
  readonly Vy: Float64Array;
  readonly Vz: Float64Array;
  readonly T: Float64Array;
  readonly My: Float64Array;
  readonly Mz: Float64Array;
}

/** Section forces at the element start (`xi = -1`, `x = 0`), after `LOCAL_FORCE_STATION_SIGNS`. */
export interface BeamSectionEndForcesI {
  readonly N0?: number;
  readonly Vy0?: number;
  readonly Vz0?: number;
  readonly T0?: number;
  readonly My0?: number;
  readonly Mz0?: number;
}

/** Section forces at the element end (`xi = +1`, `x = L`), after `LOCAL_FORCE_STATION_SIGNS`. */
export interface BeamSectionEndForcesJ {
  readonly NJ?: number;
  readonly VyJ?: number;
  readonly VzJ?: number;
  readonly TJ?: number;
  readonly MyJ?: number;
  readonly MzJ?: number;
}

/** Raw OpenSees nodal end reactions `[N_I, Vy_I, Vz_I, T_I, My_I, Mz_I]` at node `i` (before `LOCAL_FORCE_STATION_SIGNS`). */
export interface BeamNodalReactionsI {
  readonly NI?: number;
  readonly VyI?: number;
  readonly VzI?: number;
  readonly TI?: number;
  readonly MyI?: number;
  readonly MzI?: number;
}

/** Convert raw OpenSees nodal end forces at node `i` into section forces at `x = 0` (`xi = -1`). */
export function nodalReactionsIToSectionForces(r: BeamNodalReactionsI): Required<BeamSectionEndForcesI> {
  return {
    N0: -(r.NI ?? 0),
    Vy0: +(r.VyI ?? 0),
    Vz0: -(r.VzI ?? 0),
    T0: -(r.TI ?? 0),
    My0: -(r.MyI ?? 0),
    Mz0: -(r.MzI ?? 0),
  };
}

/** Scale a `BeamSpanSegmentLoad` by a time-series / pattern factor `\lambda(t)`. */
export function scaleBeamSpanLoad(load: BeamSpanSegmentLoad, factor: number): BeamSpanSegmentLoad {
  return {
    aL: load.aL,
    bL: load.bL,
    wya: load.wya * factor,
    wyb: load.wyb * factor,
    wza: load.wza * factor,
    wzb: load.wzb * factor,
    wxa: load.wxa * factor,
    wxb: load.wxb * factor,
  };
}

// -- 1. Parsing `eleLoad` rows (`-beamUniform` and unsupported load types) -----

export type ParsedEleLoad =
  | {
      readonly kind: "beamUniform";
      readonly eleTags: number[];
      readonly load: BeamSpanSegmentLoad;
      /**
       * `"canonical"` when parsed from the `h5-schema.md` / `pattern.py` /
       * `_beam_diagrams.py` token order; `"tcl_cpp_3d8"` when 3-D 8-arg tokens
       * had `aL, bL` at indices `[2, 3]` (`Wya Wzb aL bL Wyb Wza Wxa Wxb`).
       */
      readonly tokenOrder: "canonical" | "tcl_cpp_3d8";
    }
  | {
      readonly kind: "unsupported_load";
      readonly loadType: string;
      readonly eleTags: number[];
      readonly reason: string;
    };

function normalizeEleLoadTokens(tokens: readonly unknown[]): string[] {
  const cleaned: string[] = [];
  for (const t of tokens) {
    if (t === null || t === undefined) continue;
    const s = String(t).trim();
    if (s !== "") cleaned.push(s);
  }
  return cleaned;
}

/**
 * Unpack an `element_loads` `Table` from `DefObject.tables["element_loads"]`
 * into individual token rows. Handles both flat row arrays (`string[][]`) and
 * `h5wasm`'s nested 2-D string dataset representation (`string[][][]`).
 */
export function extractElementLoadRows(table: Table | undefined | null): string[][] {
  if (!table || table.rows.length === 0) return [];
  const out: string[][] = [];
  for (const row of table.rows) {
    if (!Array.isArray(row) || row.length === 0) continue;
    if (row.every((cell) => Array.isArray(cell))) {
      for (const sub of row) {
        const toks = normalizeEleLoadTokens(sub as readonly unknown[]);
        if (toks.length > 0) out.push(toks);
      }
    } else {
      const toks = normalizeEleLoadTokens(row);
      if (toks.length > 0) out.push(toks);
    }
  }
  return out;
}

/**
 * Parse a single `eleLoad` argv row into either a `beamUniform` segment load
 * or an `unsupported_load` record (for `-beamPoint`, `-beamThermal`,
 * `-selfWeight`, malformed rows, etc.) without throwing.
 */
export function parseEleLoadRow(tokens: readonly (string | number)[], ndm: 2 | 3): ParsedEleLoad {
  const cleaned = normalizeEleLoadTokens(tokens);
  if (cleaned.length === 0) {
    return { kind: "unsupported_load", loadType: "empty", eleTags: [], reason: "empty eleLoad token sequence" };
  }

  const eleTags: number[] = [];
  let i = 0;
  const n = cleaned.length;
  let loadType: string | null = null;

  while (i < n) {
    const tok = cleaned[i]!;
    if (tok === "-ele") {
      i += 1;
      while (i < n && !cleaned[i]!.startsWith("-")) {
        const tag = Number(cleaned[i]!);
        if (!Number.isInteger(tag)) {
          return {
            kind: "unsupported_load",
            loadType: "malformed",
            eleTags,
            reason: `invalid integer element tag '${cleaned[i]!}' in eleLoad -ele`,
          };
        }
        eleTags.push(tag);
        i += 1;
      }
    } else if (tok === "-range" || tok === "-eleRange") {
      if (i + 2 >= n) {
        return {
          kind: "unsupported_load",
          loadType: "malformed",
          eleTags,
          reason: `eleLoad '${tok}' requires start and end element tags`,
        };
      }
      const start = Number(cleaned[i + 1]!);
      const end = Number(cleaned[i + 2]!);
      if (!Number.isInteger(start) || !Number.isInteger(end) || end < start) {
        return {
          kind: "unsupported_load",
          loadType: "malformed",
          eleTags,
          reason: `invalid eleLoad ${tok} [${cleaned[i + 1]!}, ${cleaned[i + 2]!}]`,
        };
      }
      for (let tag = start; tag <= end; tag++) eleTags.push(tag);
      i += 3;
    } else if (tok === "-type") {
      if (i + 1 >= n) {
        return {
          kind: "unsupported_load",
          loadType: "missing",
          eleTags,
          reason: "eleLoad '-type' missing load type token",
        };
      }
      loadType = cleaned[i + 1]!;
      i += 2;
      break;
    } else {
      return {
        kind: "unsupported_load",
        loadType: "unknown",
        eleTags,
        reason: `unexpected token '${tok}' in eleLoad before '-type'`,
      };
    }
  }

  if (loadType === null) {
    return {
      kind: "unsupported_load",
      loadType: "missing",
      eleTags,
      reason: "eleLoad missing '-type' specification",
    };
  }

  if (loadType !== "-beamUniform") {
    return {
      kind: "unsupported_load",
      loadType,
      eleTags,
      reason: `element has '${loadType}' load (exact span reconstruction requires '-beamUniform' only)`,
    };
  }

  const vals: number[] = [];
  for (let k = i; k < n; k++) {
    const v = Number(cleaned[k]!);
    if (!Number.isFinite(v)) {
      return {
        kind: "unsupported_load",
        loadType,
        eleTags,
        reason: `non-numeric -beamUniform parameter '${cleaned[k]!}'`,
      };
    }
    vals.push(v);
  }

  let aL = 0.0;
  let bL = 1.0;
  let wya = 0.0;
  let wyb = 0.0;
  let wza = 0.0;
  let wzb = 0.0;
  let wxa = 0.0;
  let wxb = 0.0;
  let tokenOrder: "canonical" | "tcl_cpp_3d8" = "canonical";

  if (ndm === 2) {
    if (vals.length === 1) {
      wya = wyb = vals[0]!;
    } else if (vals.length === 2) {
      wya = wyb = vals[0]!;
      wxa = wxb = vals[1]!;
    } else if (vals.length === 4) {
      wya = wyb = vals[0]!;
      wxa = wxb = vals[1]!;
      aL = vals[2]!;
      bL = vals[3]!;
    } else if (vals.length === 6) {
      wya = vals[0]!;
      wxa = vals[1]!;
      aL = vals[2]!;
      bL = vals[3]!;
      wyb = vals[4]!;
      wxb = vals[5]!;
    } else {
      return {
        kind: "unsupported_load",
        loadType,
        eleTags,
        reason: `unsupported 2D -beamUniform value count ${vals.length} (expected 1, 2, 4, or 6)`,
      };
    }
  } else if (ndm === 3) {
    if (vals.length === 2) {
      wya = wyb = vals[0]!;
      wza = wzb = vals[1]!;
    } else if (vals.length === 3) {
      wya = wyb = vals[0]!;
      wza = wzb = vals[1]!;
      wxa = wxb = vals[2]!;
    } else if (vals.length === 5) {
      wya = wyb = vals[0]!;
      wza = wzb = vals[1]!;
      wxa = wxb = vals[2]!;
      aL = vals[3]!;
      bL = vals[4]!;
    } else if (vals.length === 8) {
      const canonicalValid = 0.0 <= vals[3]! && vals[3]! < vals[4]! && vals[4]! <= 1.0;
      const tclCppValid = 0.0 <= vals[2]! && vals[2]! < vals[3]! && vals[3]! <= 1.0;
      if (!canonicalValid && tclCppValid) {
        wya = vals[0]!;
        wzb = vals[1]!;
        aL = vals[2]!;
        bL = vals[3]!;
        wyb = vals[4]!;
        wza = vals[5]!;
        wxa = vals[6]!;
        wxb = vals[7]!;
        tokenOrder = "tcl_cpp_3d8";
      } else {
        wya = vals[0]!;
        wza = vals[1]!;
        wxa = vals[2]!;
        aL = vals[3]!;
        bL = vals[4]!;
        wyb = vals[5]!;
        wzb = vals[6]!;
        wxb = vals[7]!;
      }
    } else {
      return {
        kind: "unsupported_load",
        loadType,
        eleTags,
        reason: `unsupported 3D -beamUniform value count ${vals.length} (expected 2, 3, 5, or 8)`,
      };
    }
  } else {
    return { kind: "unsupported_load", loadType, eleTags, reason: `ndm must be 2 or 3, got ${String(ndm)}` };
  }

  if (!(0.0 <= aL && aL < bL && bL <= 1.0)) {
    return {
      kind: "unsupported_load",
      loadType,
      eleTags,
      reason: `invalid -beamUniform segment bounds aL=${aL}, bL=${bL} (must satisfy 0 <= aL < bL <= 1)`,
    };
  }

  return { kind: "beamUniform", eleTags, load: { aL, bL, wya, wyb, wza, wzb, wxa, wxb }, tokenOrder };
}

/**
 * Direct parity wrapper for Python's `parse_ele_load_beam_uniform(tokens, ndm)`:
 * returns `{ eleTags, load }` for `-beamUniform` rows and throws `RangeError`
 * if the row is malformed or names a non-`-beamUniform` load type.
 */
export function parseEleLoadBeamUniform(
  tokens: readonly (string | number)[],
  ndm: 2 | 3,
): { eleTags: number[]; load: BeamSpanSegmentLoad; tokenOrder: "canonical" | "tcl_cpp_3d8" } {
  const res = parseEleLoadRow(tokens, ndm);
  if (res.kind !== "beamUniform") throw new RangeError(res.reason);
  return { eleTags: res.eleTags, load: res.load, tokenOrder: res.tokenOrder };
}

// -- 2. Time-series factor `\lambda(t)` evaluation ----------------------------

export type TimeSeriesEvalResult =
  | { readonly ok: true; readonly factor: number }
  | { readonly ok: false; readonly reason: string };

function asFiniteNumber(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

function asFiniteNumberArray(v: unknown): number[] | null {
  if (!Array.isArray(v)) return null;
  const out: number[] = [];
  for (const item of v) {
    if (typeof item !== "number" || !Number.isFinite(item)) return null;
    out.push(item);
  }
  return out;
}

function flattenTableNumbers(t: Table | undefined): number[] | null {
  if (!t || t.rows.length === 0) return null;
  const out: number[] = [];
  const visit = (x: unknown): boolean => {
    if (Array.isArray(x)) {
      for (const item of x) {
        if (!visit(item)) return false;
      }
      return true;
    }
    if (typeof x === "number" && Number.isFinite(x)) {
      out.push(x);
      return true;
    }
    return false;
  };
  for (const r of t.rows) {
    if (!visit(r)) return null;
  }
  return out.length > 0 ? out : null;
}

function interpolatePathSeries(
  time: number,
  knotsT: readonly number[],
  knotsV: readonly number[],
  factor: number,
  useLast: boolean,
): TimeSeriesEvalResult {
  const n = knotsT.length;
  if (n === 0 || knotsV.length !== n) {
    return { ok: false, reason: "Path timeSeries has empty or mismatched time/values arrays" };
  }
  for (let i = 1; i < n; i++) {
    if (!(knotsT[i]! > knotsT[i - 1]!)) {
      return { ok: false, reason: "Path timeSeries time vector is not strictly increasing" };
    }
  }
  const tFirst = knotsT[0]!;
  const tLast = knotsT[n - 1]!;
  const eps = 1e-12 * Math.max(1.0, Math.abs(tLast - tFirst));

  if (time < tFirst - eps) return { ok: true, factor: 0.0 };
  if (time <= tFirst + eps) return { ok: true, factor: factor * knotsV[0]! };
  if (time > tLast + eps) return { ok: true, factor: useLast ? factor * knotsV[n - 1]! : 0.0 };
  if (time >= tLast - eps) return { ok: true, factor: factor * knotsV[n - 1]! };

  let lo = 0;
  let hi = n - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (knotsT[mid]! <= time) lo = mid;
    else hi = mid;
  }
  const t0 = knotsT[lo]!;
  const t1 = knotsT[hi]!;
  const v0 = knotsV[lo]!;
  const v1 = knotsV[hi]!;
  return { ok: true, factor: factor * (v0 + ((time - t0) / (t1 - t0)) * (v1 - v0)) };
}

function evalTrigSeries(
  time: number,
  tStart: number | null,
  tFinish: number | null,
  period: number | null,
  factor: number,
  shift: number,
  zeroShift: number,
): TimeSeriesEvalResult {
  if (tStart === null || tFinish === null || period === null || !(period > 0) || !(tFinish > tStart)) {
    return { ok: false, reason: "Trig timeSeries has invalid or missing t_start, t_end/t_finish, or period" };
  }
  if (!Number.isFinite(factor) || !Number.isFinite(shift) || !Number.isFinite(zeroShift)) {
    return { ok: false, reason: "Trig timeSeries has non-finite factor, shift, or zero_shift" };
  }
  const ratio = zeroShift === 0.0 ? 0.0 : zeroShift / factor;
  if (!Number.isFinite(ratio) || !(Math.abs(ratio) <= 1.0)) {
    return { ok: false, reason: `Trig timeSeries has invalid zero_shift/factor ratio (${String(zeroShift)} / ${String(factor)})` };
  }
  const eps = 1e-12 * Math.max(1.0, Math.abs(tFinish - tStart));
  if (time < tStart - eps || time > tFinish + eps) return { ok: true, factor: 0.0 };
  const clampedT = Math.min(Math.max(time, tStart), tFinish);
  const phi = shift - (period / (2.0 * Math.PI)) * Math.asin(ratio);
  return { ok: true, factor: factor * Math.sin(((2.0 * Math.PI * (clampedT - tStart)) / period) + phi) + zeroShift };
}

function evalTimeSeriesFromDeclParams(rec: DeclParams, tables: Readonly<Record<string, Table>> | undefined, time: number): TimeSeriesEvalResult {
  const type = rec.type;
  const p = rec.params;
  if (type === "Constant") return { ok: true, factor: asFiniteNumber(p["factor"]) ?? 1.0 };
  if (type === "Linear") {
    const factor = asFiniteNumber(p["factor"]) ?? 1.0;
    const tStart = asFiniteNumber(p["t_start"] ?? p["tStart"]) ?? 0.0;
    return { ok: true, factor: time < tStart - 1e-12 ? 0.0 : factor * time };
  }
  if (type === "Trig" || type === "Sine") {
    return evalTrigSeries(
      time,
      asFiniteNumber(p["t_start"] ?? p["tStart"]),
      asFiniteNumber(p["t_end"] ?? p["tEnd"] ?? p["t_finish"] ?? p["tFinish"]),
      asFiniteNumber(p["period"]),
      asFiniteNumber(p["factor"]) ?? 1.0,
      asFiniteNumber(p["shift"]) ?? 0.0,
      asFiniteNumber(p["zero_shift"] ?? p["zeroShift"]) ?? 0.0,
    );
  }
  if (type === "Path") {
    const filePath = p["file"] ?? p["file_path"] ?? p["filePath"];
    const fileTime = p["file_time"] ?? p["fileTime"];
    const values = asFiniteNumberArray(p["values"]) ?? flattenTableNumbers(tables?.["values"]);
    if (!values || values.length === 0) {
      if (typeof filePath === "string" && filePath !== "") {
        return { ok: false, reason: `Path timeSeries references external file '${filePath}' without inline values` };
      }
      if (typeof fileTime === "string" && fileTime !== "") {
        return { ok: false, reason: `Path timeSeries references external time file '${fileTime}' without inline values` };
      }
      return { ok: false, reason: "Path timeSeries has no inline values" };
    }
    const factor = asFiniteNumber(p["factor"]) ?? 1.0;
    const useLast = p["use_last"] === true || p["useLast"] === true;
    const prependZero = p["prepend_zero"] === true || p["prependZero"] === true;
    const startTime = asFiniteNumber(p["start_time"] ?? p["startTime"]) ?? 0.0;
    const explicitTime = asFiniteNumberArray(p["time"]) ?? flattenTableNumbers(tables?.["time"]);
    const dt = asFiniteNumber(p["dt"]);

    if (explicitTime !== null) {
      if (prependZero) return { ok: false, reason: "Path timeSeries cannot combine prepend_zero with an explicit time vector" };
      if (explicitTime.length !== values.length) {
        return { ok: false, reason: `Path timeSeries time (${explicitTime.length}) and values (${values.length}) lengths differ` };
      }
      return interpolatePathSeries(time, explicitTime, values, factor, useLast);
    }
    if (dt === null || !(dt > 0)) {
      return { ok: false, reason: `Path timeSeries requires a positive dt or explicit time vector, got dt=${String(dt)}` };
    }
    const knotsV = prependZero ? [0.0, ...values] : [...values];
    const knotsT = knotsV.map((_, i) => startTime + i * dt);
    return interpolatePathSeries(time, knotsT, knotsV, factor, useLast);
  }
  return { ok: false, reason: `unsupported timeSeries type '${type}'` };
}

function evalTimeSeriesFromDefObject(ts: DefObject, time: number): TimeSeriesEvalResult {
  const type = ts.type;
  const params = ts.params;
  const findFlagNumber = (flag: string): number | null => {
    const idx = params.indexOf(flag);
    if (idx >= 0 && idx + 1 < params.length) {
      const n = Number(params[idx + 1]);
      return Number.isFinite(n) ? n : null;
    }
    return null;
  };

  if (type === "Constant") {
    const byFlag = findFlagNumber("-factor");
    if (byFlag !== null) return { ok: true, factor: byFlag };
    if (params.length > 0 && typeof params[0] === "number" && Number.isFinite(params[0])) return { ok: true, factor: params[0] };
    return { ok: true, factor: 1.0 };
  }
  if (type === "Linear") {
    const byFlag = findFlagNumber("-factor");
    const tStart = findFlagNumber("-tStart") ?? 0.0;
    const factor = byFlag ?? (params.length > 0 && typeof params[0] === "number" && Number.isFinite(params[0]) ? params[0] : 1.0);
    return { ok: true, factor: time < tStart - 1e-12 ? 0.0 : factor * time };
  }
  if (type === "Trig" || type === "Sine") {
    if (params.length < 3) {
      return { ok: false, reason: "Trig timeSeries has fewer than 3 positional parameters (tStart, tFinish, period)" };
    }
    return evalTrigSeries(
      time,
      asFiniteNumber(params[0]),
      asFiniteNumber(params[1]),
      asFiniteNumber(params[2]),
      findFlagNumber("-factor") ?? asFiniteNumber(ts.attrs["factor"]) ?? 1.0,
      findFlagNumber("-shift") ?? asFiniteNumber(ts.attrs["shift"]) ?? 0.0,
      findFlagNumber("-zeroShift") ?? asFiniteNumber(ts.attrs["zero_shift"] ?? ts.attrs["zeroShift"]) ?? 0.0,
    );
  }
  if (type === "Path") {
    if ((params.includes("-filePath") || params.includes("-fileTime")) && !flattenTableNumbers(ts.tables["values"])) {
      return { ok: false, reason: "Path timeSeries uses external -filePath/-fileTime without archived table values" };
    }
    const readNumberRunAfterFlag = (flag: string): number[] | null => {
      const idx = params.indexOf(flag);
      if (idx < 0) return null;
      const out: number[] = [];
      for (let i = idx + 1; i < params.length; i++) {
        const tok = params[i]!;
        if (typeof tok === "string" && tok.startsWith("-") && Number.isNaN(Number(tok))) break;
        const v = Number(tok);
        if (!Number.isFinite(v)) break;
        out.push(v);
      }
      return out.length > 0 ? out : null;
    };
    const values = flattenTableNumbers(ts.tables["values"]) ?? asFiniteNumberArray(ts.attrs["values"]) ?? readNumberRunAfterFlag("-values");
    if (!values || values.length === 0) return { ok: false, reason: "Path timeSeries has no inline or archived values" };
    const explicitTime = flattenTableNumbers(ts.tables["time"]) ?? asFiniteNumberArray(ts.attrs["time"]) ?? readNumberRunAfterFlag("-time");
    const dt = findFlagNumber("-dt") ?? asFiniteNumber(ts.attrs["dt"]);
    const factor = findFlagNumber("-factor") ?? asFiniteNumber(ts.attrs["factor"]) ?? 1.0;
    const startTime = findFlagNumber("-startTime") ?? asFiniteNumber(ts.attrs["start_time"] ?? ts.attrs["startTime"]) ?? 0.0;
    const useLast = params.includes("-useLast") || ts.attrs["use_last"] === 1 || ts.attrs["use_last"] === "true";
    const prependZero = params.includes("-prependZero") || ts.attrs["prepend_zero"] === 1 || ts.attrs["prepend_zero"] === "true";

    if (explicitTime !== null) {
      if (prependZero) return { ok: false, reason: "Path timeSeries cannot combine -prependZero with an explicit -time vector" };
      if (explicitTime.length !== values.length) {
        return { ok: false, reason: `Path timeSeries time (${explicitTime.length}) and values (${values.length}) lengths differ` };
      }
      return interpolatePathSeries(time, explicitTime, values, factor, useLast);
    }
    if (dt === null || !(dt > 0)) {
      return { ok: false, reason: `Path timeSeries requires a positive -dt or explicit -time vector, got dt=${String(dt)}` };
    }
    const knotsV = prependZero ? [0.0, ...values] : [...values];
    const knotsT = knotsV.map((_, i) => startTime + i * dt);
    return interpolatePathSeries(time, knotsT, knotsV, factor, useLast);
  }
  return { ok: false, reason: `unsupported timeSeries type '${type}'` };
}

function findDeclRecordByTag(defs: Definitions | null | undefined, kind: string, tag: number | null): DeclRecord | null {
  if (!defs?.decls || tag === null) return null;
  const run = defs.decls.tags.find((t) => t.kind === kind && tag >= t.first && tag < t.first + t.count);
  return run ? (defs.decls.records[run.decl] ?? null) : null;
}

const PRIMITIVE_TIME_SERIES_TYPES: ReadonlySet<string> = new Set(["Constant", "Linear", "Trig", "Sine", "Path"]);

/** Evaluate an OpenSees `timeSeries` at pseudo-time / step time `time`. */
export function evaluateTimeSeriesFactor(
  ts: DefObject | DeclParams,
  time: number,
  defs?: Definitions | null,
): TimeSeriesEvalResult {
  if (!Number.isFinite(time)) return { ok: false, reason: `step time is non-finite (${String(time)})` };
  if ("groupName" in ts) {
    const declRec = findDeclRecordByTag(defs, "timeSeries", ts.tag);
    const declParams = declRec?.params[0];
    if (declParams) {
      const fromDecl = evalTimeSeriesFromDeclParams(declParams, ts.tables, time);
      if (fromDecl.ok || PRIMITIVE_TIME_SERIES_TYPES.has(declParams.type)) return fromDecl;
    }
    return evalTimeSeriesFromDefObject(ts, time);
  }
  return evalTimeSeriesFromDeclParams(ts, undefined, time);
}

// -- 3. Exact analytical integration & zero-shear root finding ----------------

/** Analytical single integral `I_1(x) = \int_0^x w(s) ds` for a trapezoidal segment load on `[aL * L, bL * L]`. */
export function integrateSegmentLoadSingle(
  x: number,
  length: number,
  aL: number,
  bL: number,
  wa: number,
  wb: number,
): number {
  const xa = aL * length;
  const xb = bL * length;
  const d = xb - xa;
  if (d <= 0.0 || x <= xa) return 0.0;
  const slope = (wb - wa) / d;
  const xiEnd = Math.min(x, xb) - xa;
  return wa * xiEnd + 0.5 * slope * xiEnd * xiEnd;
}

/** Analytical double integral `I_2(x) = \int_0^x \int_0^s w(u) du ds` for a trapezoidal segment load on `[aL * L, bL * L]`. */
export function integrateSegmentLoadDouble(
  x: number,
  length: number,
  aL: number,
  bL: number,
  wa: number,
  wb: number,
): number {
  const xa = aL * length;
  const xb = bL * length;
  const d = xb - xa;
  if (d <= 0.0 || x <= xa) return 0.0;
  const slope = (wb - wa) / d;
  if (x <= xb) {
    const dx = x - xa;
    return 0.5 * wa * dx * dx + (slope * dx * dx * dx) / 6.0;
  }
  const totalQ = wa * d + 0.5 * slope * d * d;
  const momentAboutXb = 0.5 * wa * d * d + (slope * d * d * d) / 6.0;
  return momentAboutXb + totalQ * (x - xb);
}

function evalShearScalar(
  x: number,
  V0: number,
  length: number,
  loads: readonly BeamSpanSegmentLoad[],
  component: "y" | "z",
  sign: 1 | -1,
): number {
  let acc = 0.0;
  for (const ld of loads) {
    const wa = component === "y" ? ld.wya : ld.wza;
    const wb = component === "y" ? ld.wyb : ld.wzb;
    acc += integrateSegmentLoadSingle(x, length, ld.aL, ld.bL, wa, wb);
  }
  return V0 + sign * acc;
}

function dedupeSortedCoords(coords: readonly number[], length: number, tolScale = 1e-12): number[] {
  if (coords.length === 0) return [];
  const sorted = [...coords].sort((a, b) => a - b);
  const tol = tolScale * Math.max(1.0, length);
  const out: number[] = [sorted[0]!];
  for (let i = 1; i < sorted.length; i++) {
    const v = sorted[i]!;
    if (v - out[out.length - 1]! > tol) out.push(v);
  }
  return out;
}

/** Find interior points `x \in (0, L)` where `V(x) = V0 + sign * \sum \int_0^x w(s) ds = 0`. */
export function findZeroShearRoots(
  V0: number,
  length: number,
  loads: readonly BeamSpanSegmentLoad[],
  component: "y" | "z",
  sign: 1 | -1,
): number[] {
  const rawBp: number[] = [0.0, length];
  for (const ld of loads) rawBp.push(ld.aL * length, ld.bL * length);
  const bp = dedupeSortedCoords(rawBp, length);
  const roots: number[] = [];
  const epsX = 1e-11 * Math.max(1.0, length);

  for (let k = 0; k + 1 < bp.length; k++) {
    const x0 = bp[k]!;
    const x1 = bp[k + 1]!;
    const h = x1 - x0;
    if (h <= epsX) continue;
    const xm = 0.5 * (x0 + x1);
    const v0 = evalShearScalar(x0, V0, length, loads, component, sign);
    const vm = evalShearScalar(xm, V0, length, loads, component, sign);
    const v1 = evalShearScalar(x1, V0, length, loads, component, sign);

    const c = v0;
    const a = (2.0 * (v0 - 2.0 * vm + v1)) / (h * h);
    const b = (v1 - v0 - a * h * h) / h;
    const scaleV = Math.max(1.0, Math.abs(v0), Math.abs(vm), Math.abs(v1));
    const candidates: number[] = [];

    if (Math.abs(a) * h * h <= 1e-12 * scaleV) {
      if (Math.abs(b) * h > 1e-14 * scaleV) candidates.push(-c / b);
    } else {
      const disc = b * b - 4.0 * a * c;
      if (disc >= 0.0) {
        const sqrtD = Math.sqrt(disc);
        const q = -0.5 * (b >= 0.0 ? b + sqrtD : b - sqrtD);
        if (Math.abs(a) > 0.0) candidates.push(q / a);
        if (Math.abs(q) > 0.0) candidates.push(c / q);
      }
    }

    for (const sRaw of candidates) {
      if (sRaw > -epsX && sRaw < h + epsX) {
        let xr = Math.min(Math.max(x0 + sRaw, x0), x1);
        for (let iter = 0; iter < 2; iter++) {
          const fv = evalShearScalar(xr, V0, length, loads, component, sign);
          const dv = b + 2.0 * a * (xr - x0);
          if (Math.abs(dv) * h > 1e-14 * scaleV) {
            const xNew = xr - fv / dv;
            if (xNew >= x0 && xNew <= x1) xr = xNew;
          }
        }
        if (xr > epsX && xr < length - epsX) roots.push(xr);
      }
    }
  }
  return roots;
}

export interface ReconstructBeamElementDiagramOptions {
  readonly elementId?: number;
  readonly length: number;
  readonly endForcesI?: BeamSectionEndForcesI;
  readonly nodalReactionsI?: BeamNodalReactionsI;
  readonly loads?: readonly BeamSpanSegmentLoad[];
  readonly nSub?: number;
  readonly extraXi?: ArrayLike<number>;
}

/**
 * Reconstruct exact internal force diagrams `(N, Vy, Vz, T, My, Mz)` along a
 * 1-D beam element from node-`i` section forces and scaled `-beamUniform` loads.
 */
export function reconstructBeamElementDiagram(opts: ReconstructBeamElementDiagramOptions): BeamSpanDiagram {
  const L = opts.length;
  if (!(L > 0.0) || !Number.isFinite(L)) {
    throw new RangeError(`element length must be positive and finite, got ${String(L)}`);
  }
  const nSub = opts.nSub ?? 16;
  if (!Number.isInteger(nSub) || nSub < 1) {
    throw new RangeError(`nSub must be >= 1, got ${String(nSub)}`);
  }

  const secI: Required<BeamSectionEndForcesI> = opts.endForcesI
    ? {
        N0: opts.endForcesI.N0 ?? 0.0,
        Vy0: opts.endForcesI.Vy0 ?? 0.0,
        Vz0: opts.endForcesI.Vz0 ?? 0.0,
        T0: opts.endForcesI.T0 ?? 0.0,
        My0: opts.endForcesI.My0 ?? 0.0,
        Mz0: opts.endForcesI.Mz0 ?? 0.0,
      }
    : opts.nodalReactionsI
      ? nodalReactionsIToSectionForces(opts.nodalReactionsI)
      : { N0: 0, Vy0: 0, Vz0: 0, T0: 0, My0: 0, Mz0: 0 };

  if (
    !Number.isFinite(secI.N0) ||
    !Number.isFinite(secI.Vy0) ||
    !Number.isFinite(secI.Vz0) ||
    !Number.isFinite(secI.T0) ||
    !Number.isFinite(secI.My0) ||
    !Number.isFinite(secI.Mz0)
  ) {
    throw new RangeError("endForcesI / nodalReactionsI contains non-finite force value");
  }

  const loads = opts.loads ?? [];
  for (const ld of loads) {
    if (!Number.isFinite(ld.aL) || !Number.isFinite(ld.bL) || !(0.0 <= ld.aL && ld.aL < ld.bL && ld.bL <= 1.0)) {
      throw new RangeError(`invalid segment load bounds aL=${ld.aL}, bL=${ld.bL}`);
    }
    if (
      !Number.isFinite(ld.wxa) ||
      !Number.isFinite(ld.wxb) ||
      !Number.isFinite(ld.wya) ||
      !Number.isFinite(ld.wyb) ||
      !Number.isFinite(ld.wza) ||
      !Number.isFinite(ld.wzb)
    ) {
      throw new RangeError("segment load contains non-finite intensity value");
    }
  }

  const rawCrit: number[] = [0.0, L];
  for (const ld of loads) rawCrit.push(ld.aL * L, ld.bL * L);
  for (const r of findZeroShearRoots(secI.Vy0, L, loads, "y", +1)) rawCrit.push(r);
  for (const r of findZeroShearRoots(secI.Vz0, L, loads, "z", -1)) rawCrit.push(r);

  if (opts.extraXi) {
    for (let i = 0; i < opts.extraXi.length; i++) {
      const xiVal = opts.extraXi[i]!;
      if (Number.isFinite(xiVal) && xiVal >= -1.0 && xiVal <= 1.0) {
        rawCrit.push(0.5 * (xiVal + 1.0) * L);
      }
    }
  }

  const crit = dedupeSortedCoords(rawCrit, L, 1e-11);
  const denseRaw: number[] = [];
  for (let k = 0; k + 1 < crit.length; k++) {
    const c0 = crit[k]!;
    const c1 = crit[k + 1]!;
    const limit = k + 2 < crit.length ? nSub : nSub + 1;
    for (let j = 0; j < limit; j++) denseRaw.push(c0 + ((c1 - c0) * j) / nSub);
  }
  const xCoords = dedupeSortedCoords(denseRaw, L, 1e-12);
  if (xCoords.length === 0) {
    xCoords.push(0.0, L);
  } else {
    xCoords[0] = 0.0;
    xCoords[xCoords.length - 1] = L;
  }

  const m = xCoords.length;
  const xi = new Float64Array(m);
  const x = new Float64Array(m);
  const N = new Float64Array(m);
  const Vy = new Float64Array(m);
  const Vz = new Float64Array(m);
  const T = new Float64Array(m);
  const My = new Float64Array(m);
  const Mz = new Float64Array(m);

  for (let i = 0; i < m; i++) {
    const xv = xCoords[i]!;
    x[i] = xv;
    xi[i] = i === 0 ? -1.0 : i === m - 1 ? 1.0 : (2.0 * xv) / L - 1.0;

    let intWx = 0.0;
    let intWy = 0.0;
    let intWz = 0.0;
    let iintWy = 0.0;
    let iintWz = 0.0;

    for (const ld of loads) {
      intWx += integrateSegmentLoadSingle(xv, L, ld.aL, ld.bL, ld.wxa, ld.wxb);
      intWy += integrateSegmentLoadSingle(xv, L, ld.aL, ld.bL, ld.wya, ld.wyb);
      intWz += integrateSegmentLoadSingle(xv, L, ld.aL, ld.bL, ld.wza, ld.wzb);
      iintWy += integrateSegmentLoadDouble(xv, L, ld.aL, ld.bL, ld.wya, ld.wyb);
      iintWz += integrateSegmentLoadDouble(xv, L, ld.aL, ld.bL, ld.wza, ld.wzb);
    }

    N[i] = secI.N0 - intWx;
    Vy[i] = secI.Vy0 + intWy;
    Vz[i] = secI.Vz0 - intWz;
    T[i] = secI.T0;
    Mz[i] = secI.Mz0 + secI.Vy0 * xv + iintWy;
    My[i] = secI.My0 + secI.Vz0 * xv - iintWz;
  }

  return { elementId: opts.elementId ?? 0, length: L, xi, x, N, Vy, Vz, T, My, Mz };
}

/** Alias for `reconstructBeamElementDiagram` used by `buildLineDiagram3D` / E2b callers. */
export const reconstructElementSpanDiagram = reconstructBeamElementDiagram;

// -- 4. Node-`j` equilibrium closure check ------------------------------------

export interface BeamClosureCheckOptions {
  readonly rtol?: number;
  readonly atol?: number;
}

export type BeamClosureCheckResult =
  | { readonly ok: true }
  | {
      readonly ok: false;
      readonly component: BeamForceComponent;
      readonly reconstructed: number;
      readonly recorded: number;
      readonly diff: number;
      readonly tol: number;
      readonly reason: string;
    };

/** Verify that reconstructed right-end (`x = L`, `xi = +1`) section forces match recorded `.ladruno` station forces. */
export function checkBeamSpanClosure(
  diagram: BeamSpanDiagram,
  endForcesJ: BeamSectionEndForcesJ,
  opts?: BeamClosureCheckOptions,
): BeamClosureCheckResult {
  const rtol = opts?.rtol ?? 1e-4;
  const atol = opts?.atol ?? 1e-5;
  const last = diagram.x.length - 1;
  const L = diagram.length;

  const checkOne = (
    comp: BeamForceComponent,
    reconstructed: number,
    recorded: number | undefined,
    extraScale: number,
  ): BeamClosureCheckResult => {
    if (recorded === undefined || !Number.isFinite(recorded)) return { ok: true };
    const diff = Math.abs(reconstructed - recorded);
    const scale = Math.max(Math.abs(reconstructed), Math.abs(recorded), extraScale, 1.0);
    const tol = atol + rtol * scale;
    if (diff > tol) {
      return {
        ok: false,
        component: comp,
        reconstructed,
        recorded,
        diff,
        tol,
        reason: `end-j equilibrium closure failed on ${comp} (reconstructed ${reconstructed.toPrecision(6)} vs recorded ${recorded.toPrecision(6)}, |diff|=${diff.toExponential(3)} > tol=${tol.toExponential(3)})`,
      };
    }
    return { ok: true };
  };

  const forceScaleY = Math.max(Math.abs(diagram.Vy[0]!), Math.abs(diagram.Vy[last]!));
  const forceScaleZ = Math.max(Math.abs(diagram.Vz[0]!), Math.abs(diagram.Vz[last]!));

  const checks: BeamClosureCheckResult[] = [
    checkOne("N", diagram.N[last]!, endForcesJ.NJ, Math.abs(diagram.N[0]!)),
    checkOne("Vy", diagram.Vy[last]!, endForcesJ.VyJ, forceScaleY),
    checkOne("Vz", diagram.Vz[last]!, endForcesJ.VzJ, forceScaleZ),
    checkOne("T", diagram.T[last]!, endForcesJ.TJ, Math.abs(diagram.T[0]!)),
    checkOne("My", diagram.My[last]!, endForcesJ.MyJ, Math.max(Math.abs(diagram.My[0]!), forceScaleZ * L)),
    checkOne("Mz", diagram.Mz[last]!, endForcesJ.MzJ, Math.max(Math.abs(diagram.Mz[0]!), forceScaleY * L)),
  ];

  for (const c of checks) {
    if (!c.ok) return c;
  }
  return { ok: true };
}

// -- 5. Resolving transform, active patterns, and span loads from `State` -----

function candidateElementFamilyTags(elementDecl: Decl, family: string): number[] {
  const out: number[] = [];
  for (const m of elementDecl.element?.metas ?? []) {
    const decoder = ELEMENT_SYNTAX[m.type];
    if (!decoder) continue;
    const dec = decoder(m.args as Param[]);
    if ("reason" in dec) continue;
    for (const s of dec.slots) {
      if (s.family === family) {
        const tag = m.args[s.slot];
        if (typeof tag === "number" && Number.isInteger(tag)) out.push(tag);
      }
    }
  }
  return out;
}

/** Resolve the OpenSees `geomTransf` type (`"Linear" | "PDelta" | "Corotational"`) for a 1-D element `Decl`. */
export function resolveElementGeomTransfType(
  decls: Readonly<Record<DeclPath, Decl>>,
  elementDecl: Decl,
  defs?: Definitions | null,
): string | null {
  const transfPath = elementDecl.refs["transfTag"];
  if (transfPath && decls[transfPath]?.kind === "geomTransf") return decls[transfPath]!.type;
  const tags = candidateElementFamilyTags(elementDecl, "geomTransf");
  if (tags.length === 0) return null;

  const matchedTypes = new Set<string>();
  for (const d of Object.values(decls)) {
    if (d.kind !== "geomTransf") continue;
    if ((d.tag !== null && tags.includes(d.tag)) || d.vecxz?.some((p) => tags.includes(p.tag))) {
      matchedTypes.add(d.type);
    }
  }
  if (matchedTypes.size === 1) return [...matchedTypes][0]!;
  if (defs?.decls) {
    for (const tag of tags) {
      const t = findDeclRecordByTag(defs, "geomTransf", tag)?.params[0]?.type;
      if (t) matchedTypes.add(t);
    }
    if (matchedTypes.size === 1) return [...matchedTypes][0]!;
  }
  return null;
}

const INTERIOR_ONLY_BEAM_INTEGRATIONS: ReadonlySet<string> = new Set([
  "Legendre",
  "Radau",
  "MidDistance",
  "Chebyshev",
]);

/** Resolve the OpenSees `beamIntegration` type (`"Lobatto" | "Legendre" | ...`) for a 1-D beam-column `Decl`. */
export function resolveElementBeamIntegrationType(
  decls: Readonly<Record<DeclPath, Decl>>,
  elementDecl: Decl,
  defs?: Definitions | null,
): string | null {
  const intPath = elementDecl.refs["integrationTag"];
  if (intPath && decls[intPath]?.kind === "beamIntegration") return decls[intPath]!.type;
  const tags = candidateElementFamilyTags(elementDecl, "beamIntegration");
  if (tags.length === 0) return null;

  const matchedTypes = new Set<string>();
  for (const d of Object.values(decls)) {
    if (d.kind === "beamIntegration" && d.tag !== null && tags.includes(d.tag)) matchedTypes.add(d.type);
  }
  if (matchedTypes.size === 1) return [...matchedTypes][0]!;
  if (defs?.decls) {
    for (const tag of tags) {
      const t = findDeclRecordByTag(defs, "beamIntegration", tag)?.params[0]?.type;
      if (t) matchedTypes.add(t);
    }
    if (matchedTypes.size === 1) return [...matchedTypes][0]!;
  }
  return null;
}

function resolvePatternFactor(pattern: DefObject, defs: Definitions | null | undefined): number {
  const declRec = findDeclRecordByTag(defs, "pattern", pattern.tag);
  const p = declRec?.params[0]?.params;
  if (p && typeof p["factor"] === "number" && Number.isFinite(p["factor"])) return p["factor"];
  const idx = pattern.params.indexOf("-factor");
  if (idx >= 0 && idx + 1 < pattern.params.length) {
    const v = Number(pattern.params[idx + 1]);
    if (Number.isFinite(v)) return v;
  }
  if (typeof pattern.attrs["factor"] === "number" && Number.isFinite(pattern.attrs["factor"])) {
    return pattern.attrs["factor"];
  }
  return 1.0;
}

function resolvePatternTimeSeries(
  pattern: DefObject,
  defs: Definitions,
): { ok: true; series: DefObject | DeclParams } | { ok: false; reason: string } {
  const seriesRef = pattern.attrs["series_ref"];
  if (typeof seriesRef === "string" && seriesRef !== "") {
    const byPath = defs.timeSeries.find((ts) => ts.path === seriesRef);
    if (byPath) return { ok: true, series: byPath };
  }
  const declRec = findDeclRecordByTag(defs, "pattern", pattern.tag);
  const seriesVal: DeclValue | undefined =
    declRec?.params[0]?.params["series"] ?? declRec?.params[0]?.params["time_series"];
  if (typeof seriesVal === "object" && seriesVal !== null && !Array.isArray(seriesVal) && typeof seriesVal["$decl"] === "string") {
    const key = seriesVal["$decl"];
    const tsDecl = defs.decls?.records.find((r) => r.key === key);
    if (tsDecl) {
      const tagRun = defs.decls?.tags.find((t) => t.kind === "timeSeries" && t.decl === tsDecl.row);
      if (tagRun) {
        const byTag = defs.timeSeries.find((ts) => ts.tag !== null && ts.tag >= tagRun.first && ts.tag < tagRun.first + tagRun.count);
        if (byTag) return { ok: true, series: byTag };
      }
      if (tsDecl.params[0]) return { ok: true, series: tsDecl.params[0] };
    }
  }
  if (pattern.params.length > 0 && typeof pattern.params[0] === "number" && Number.isInteger(pattern.params[0])) {
    const tsTag = pattern.params[0];
    const byTag = defs.timeSeries.find((ts) => ts.tag === tsTag);
    if (byTag) return { ok: true, series: byTag };
    const tsDecl = findDeclRecordByTag(defs, "timeSeries", tsTag);
    if (tsDecl?.params[0]) return { ok: true, series: tsDecl.params[0] };
  }
  return { ok: false, reason: `pattern '${pattern.groupName}' (tag ${String(pattern.tag)}) has no resolvable timeSeries` };
}

export type ResolvedStageSpanLoads =
  | {
      readonly ok: true;
      readonly loadsByElementTag: Map<number, BeamSpanSegmentLoad[]>;
      readonly unsupportedByElementTag: Map<number, string>;
    }
  | {
      readonly ok: false;
      readonly reason: string;
    };

export interface ResolveStageSpanLoadsOptions {
  readonly definitions: Definitions | null | undefined;
  readonly sequence?: SequenceZone | null;
  readonly ndm: 2 | 3 | null | undefined | SpatialNdm;
  readonly resultsStage?: Pick<ResultsStage, "id" | "name" | "kind"> | null;
  readonly stageIndex?: number;
  readonly time: number;
}

/** Resolve all active patterns and their scaled `-beamUniform` element loads at step `time` for a results stage. */
export function resolveStageSpanLoads(opts: ResolveStageSpanLoadsOptions): ResolvedStageSpanLoads {
  const defs = opts.definitions;
  if (!defs) return { ok: false, reason: "model has no /opensees definitions" };
  const ndm = unwrapSpatialNdm(opts.ndm);
  if (ndm !== 2 && ndm !== 3) {
    return { ok: false, reason: `model spatial dimension ndm must be 2 or 3, got ${String(ndm)}` };
  }
  if (!Number.isFinite(opts.time)) return { ok: false, reason: `step time is non-finite (${String(opts.time)})` };
  if (opts.resultsStage && opts.resultsStage.kind !== "static") {
    return {
      ok: false,
      reason: `results stage '${opts.resultsStage.name || opts.resultsStage.id}' is '${opts.resultsStage.kind}' (exact span reconstruction requires 'static')`,
    };
  }
  if (defs.program?.["loadConst"] !== undefined) {
    return { ok: false, reason: "model program invokes 'loadConst' with unarchived freeze time" };
  }

  let activeAnalysis: AnalysisDef | null = defs.analysis;
  const activePatterns: DefObject[] = [...defs.patterns];

  const isConstantPatternTimeSeries = (pat: DefObject): boolean => {
    const tsRes = resolvePatternTimeSeries(pat, defs);
    if (!tsRes.ok) return false;
    if ("groupName" in tsRes.series) {
      const declRec = findDeclRecordByTag(defs, "timeSeries", tsRes.series.tag);
      const declType = declRec?.params[0]?.type;
      if (declType) return declType === "Constant";
      return tsRes.series.type === "Constant";
    }
    return tsRes.series.type === "Constant";
  };

  if (defs.stages.length > 0) {
    const stageIdx =
      opts.resultsStage?.name
        ? defs.stages.findIndex((s) => s.name === opts.resultsStage!.name)
        : (opts.stageIndex ?? 0);
    const resolvedIdx = stageIdx >= 0 ? stageIdx : (opts.stageIndex ?? -1);
    const stage = defs.stages[resolvedIdx];
    if (!stage) {
      return {
        ok: false,
        reason: `results stage '${opts.resultsStage?.name ?? String(opts.stageIndex ?? 0)}' could not be matched to /opensees/stages`,
      };
    }

    if (resolvedIdx > 0) {
      for (const rootPat of defs.patterns) {
        const rootRows = extractElementLoadRows(rootPat.tables["element_loads"]);
        if (rootRows.length > 0 && !isConstantPatternTimeSeries(rootPat)) {
          return {
            ok: false,
            reason: `root pattern '${rootPat.groupName}' with non-Constant timeSeries has element_loads across stage boundary; multi-stage loadConst state is not archived`,
          };
        }
      }
    }

    for (let sIdx = 0; sIdx <= resolvedIdx; sIdx++) {
      const st = defs.stages[sIdx]!;
      if (
        st.attrs["load_const"] !== undefined ||
        st.attrs["loadConst"] !== undefined ||
        st.attrs["pre_analyze_reset"] !== undefined ||
        st.other.includes("load_const") ||
        st.other.includes("loadConst")
      ) {
        return { ok: false, reason: `stage '${st.name}' invokes loadConst without archived freeze time` };
      }
      if (st.attrs["set_time"] !== undefined) {
        return {
          ok: false,
          reason: `stage '${st.name}' resets domain time via set_time (${String(st.attrs["set_time"])}), making pattern load factors ambiguous`,
        };
      }
      if (sIdx < resolvedIdx && st.patterns.length > 0) {
        for (const priorPat of st.patterns) {
          const priorRows = extractElementLoadRows(priorPat.tables["element_loads"]);
          if (priorRows.length > 0 && !isConstantPatternTimeSeries(priorPat)) {
            return {
              ok: false,
              reason: `prior stage '${st.name}' added element_loads pattern '${priorPat.groupName}' with non-Constant timeSeries; multi-stage loadConst state is not archived`,
            };
          }
        }
      }
      activePatterns.push(...st.patterns);
    }
    activeAnalysis = stage.analysis ?? defs.analysis;
  }

  if (opts.sequence?.status === "ready") {
    const changes = opts.sequence.changes;
    const opIdx = changes.columns.indexOf("op");
    const whatIdx = changes.columns.indexOf("what");
    if (opIdx >= 0 && whatIdx >= 0) {
      for (const row of changes.rows) {
        if (row[opIdx] === "remove" && row[whatIdx] === "pattern") {
          return { ok: false, reason: "/sequence records a 'remove pattern' operation; active pattern history is ambiguous" };
        }
      }
    }
  }

  if (activeAnalysis) {
    const analysisType = activeAnalysis.attrs["analysis"];
    if (typeof analysisType === "string" && analysisType !== "" && analysisType !== "Static") {
      return { ok: false, reason: `analysis type is '${analysisType}' (exact span reconstruction requires 'Static')` };
    }
    const integrator = activeAnalysis.attrs["integrator"];
    if (typeof integrator === "string" && integrator !== "" && integrator !== "LoadControl") {
      return { ok: false, reason: `analysis integrator is '${integrator}' (exact span reconstruction requires 'LoadControl')` };
    }
  }

  const loadsByElementTag = new Map<number, BeamSpanSegmentLoad[]>();
  const unsupportedByElementTag = new Map<number, string>();

  for (const pat of activePatterns) {
    const rows = extractElementLoadRows(pat.tables["element_loads"]);
    if (rows.length === 0) {
      if (pat.type !== "Plain") {
        return { ok: false, reason: `active pattern '${pat.groupName}' has type '${pat.type}' (requires 'Plain')` };
      }
      continue;
    }
    if (pat.type !== "Plain") {
      return { ok: false, reason: `pattern '${pat.groupName}' with element_loads has type '${pat.type}' (requires 'Plain')` };
    }
    const tsRes = resolvePatternTimeSeries(pat, defs);
    if (!tsRes.ok) return tsRes;
    const factorRes = evaluateTimeSeriesFactor(tsRes.series, opts.time, defs);
    if (!factorRes.ok) {
      return {
        ok: false,
        reason: `pattern '${pat.groupName}' timeSeries cannot be evaluated at t=${opts.time}: ${factorRes.reason}`,
      };
    }
    const totalFactor = factorRes.factor * resolvePatternFactor(pat, defs);

    for (const row of rows) {
      const parsed = parseEleLoadRow(row, ndm);
      if (parsed.kind === "unsupported_load") {
        if (parsed.eleTags.length === 0) return { ok: false, reason: parsed.reason };
        for (const tag of parsed.eleTags) {
          if (!unsupportedByElementTag.has(tag)) unsupportedByElementTag.set(tag, parsed.reason);
        }
        continue;
      }
      const scaled = scaleBeamSpanLoad(parsed.load, totalFactor);
      for (const tag of parsed.eleTags) {
        const list = loadsByElementTag.get(tag) ?? [];
        list.push(scaled);
        loadsByElementTag.set(tag, list);
      }
    }
  }

  return { ok: true, loadsByElementTag, unsupportedByElementTag };
}

// -- 6. High-level per-element & per-step reconstruction + fallback -----------

export type ElementSpanDiagramReconstruction =
  | {
      readonly mode: "exact";
      readonly elementId: number;
      readonly decl: DeclPath | null;
      readonly diagram: BeamSpanDiagram;
      readonly xi: Float64Array;
      readonly values: Float64Array;
    }
  | {
      readonly mode: "interpolated";
      readonly elementId: number;
      readonly decl: DeclPath | null;
      readonly reason: string;
      readonly xi: Float64Array;
      readonly values: Float64Array;
    };

export interface StepSectionForceMap {
  readonly N?: LadrunoLineStationStep;
  readonly Vy?: LadrunoLineStationStep;
  readonly Vz?: LadrunoLineStationStep;
  readonly T?: LadrunoLineStationStep;
  readonly My?: LadrunoLineStationStep;
  readonly Mz?: LadrunoLineStationStep;
}

function extractElementStations(
  step: LadrunoLineStationStep | undefined,
  elementId: number,
): { xi: Float64Array; values: Float64Array } {
  if (!step) return { xi: new Float64Array(0), values: new Float64Array(0) };
  const pairs: { xi: number; val: number }[] = [];
  let needsSort = false;
  let allFiniteXi = true;
  for (let i = 0; i < step.elementIndex.length; i++) {
    if (step.elementIndex[i] === elementId) {
      const xVal = step.stationNaturalCoord[i]!;
      const vVal = step.values[i]!;
      if (!Number.isFinite(xVal)) allFiniteXi = false;
      if (pairs.length > 0 && xVal < pairs[pairs.length - 1]!.xi) needsSort = true;
      pairs.push({ xi: xVal, val: vVal });
    }
  }
  if (needsSort && allFiniteXi) pairs.sort((a, b) => a.xi - b.xi);
  const xi = new Float64Array(pairs.length);
  const values = new Float64Array(pairs.length);
  for (let i = 0; i < pairs.length; i++) {
    xi[i] = pairs[i]!.xi;
    values[i] = pairs[i]!.val;
  }
  return { xi, values };
}

function findElementEndStations(
  step: LadrunoLineStationStep | undefined,
  elementId: number,
  endTol = 1e-3,
): { ok: true; vI: number; vJ: number; xi: Float64Array; values: Float64Array } | { ok: false; reason: string; xi: Float64Array; values: Float64Array } {
  if (!step) {
    return { ok: false, reason: "component step data is not available", xi: new Float64Array(0), values: new Float64Array(0) };
  }
  const { xi, values } = extractElementStations(step, elementId);
  const count = xi.length;
  if (count === 0) {
    return { ok: false, reason: `element ${elementId} is not present in .ladruno step data`, xi, values };
  }
  if (count < 2) {
    return {
      ok: false,
      reason: `element ${elementId} has ${count} recorded station(s); exact span reconstruction requires end stations at xi = -1 and xi = +1`,
      xi,
      values,
    };
  }
  for (let k = 0; k < count; k++) {
    if (!Number.isFinite(xi[k]!) || !Number.isFinite(values[k]!)) {
      return { ok: false, reason: `element ${elementId} station coordinates or values are non-finite`, xi, values };
    }
  }
  const xiFirst = xi[0]!;
  const xiLast = xi[count - 1]!;
  if (Math.abs(xiFirst - -1.0) > endTol || Math.abs(xiLast - 1.0) > endTol) {
    return {
      ok: false,
      reason: `element ${elementId} stations [${xiFirst.toFixed(4)}, ..., ${xiLast.toFixed(4)}] do not include element ends xi = -1 and xi = +1`,
      xi,
      values,
    };
  }
  return { ok: true, vI: values[0]!, vJ: values[count - 1]!, xi, values };
}

export interface ReconstructStepSpanDiagramsOptions {
  readonly component: LineStationComponent;
  readonly stepByComponent: Readonly<Partial<Record<LineStationComponent, LadrunoLineStationStep>>>;
  readonly elementLengths: ReadonlyMap<number, number>;
  readonly join?: LadrunoElementJoin | null;
  readonly decls?: Readonly<Record<DeclPath, Decl>>;
  readonly definitions?: Definitions | null;
  readonly sequence?: SequenceZone | null;
  readonly ndm: 2 | 3 | null | undefined | SpatialNdm;
  readonly resultsStage?: Pick<ResultsStage, "id" | "name" | "kind"> | null;
  readonly stageIndex?: number;
  readonly time: number;
  readonly nSub?: number;
  readonly closureTolerance?: BeamClosureCheckOptions;
}

/**
 * Reconstruct exact or fallback-interpolated span diagrams for every element in
 * `stepByComponent[component]` at analysis `time`.
 */
export function reconstructSpanDiagramsForStep(
  opts: ReconstructStepSpanDiagramsOptions,
): Map<number, ElementSpanDiagramReconstruction> {
  const out = new Map<number, ElementSpanDiagramReconstruction>();
  const primaryStep = opts.stepByComponent[opts.component];
  if (!primaryStep) return out;

  const isForce = isBeamForceComponent(opts.component);
  const stageLoadsRes = isForce
    ? resolveStageSpanLoads({
        definitions: opts.definitions,
        sequence: opts.sequence ?? null,
        ndm: opts.ndm,
        resultsStage: opts.resultsStage ?? null,
        ...(opts.stageIndex !== undefined ? { stageIndex: opts.stageIndex } : {}),
        time: opts.time,
      })
    : {
        ok: false as const,
        reason: `component '${opts.component}' is a section_deformation quantity (exact equilibrium integration applies to section_force only)`,
      };

  const companionStep = primaryStep.companionStep ?? undefined;
  const stepVy = opts.stepByComponent.Vy ?? (companionStep?.component === "Vy" ? companionStep : undefined);
  const stepVz = opts.stepByComponent.Vz ?? (companionStep?.component === "Vz" ? companionStep : undefined);
  const stepMy = opts.stepByComponent.My ?? (companionStep?.component === "My" ? companionStep : undefined);
  const stepMz = opts.stepByComponent.Mz ?? (companionStep?.component === "Mz" ? companionStep : undefined);

  const seenElementIds = new Set<number>();
  const elementIds: number[] = [];
  for (let i = 0; i < primaryStep.elementIndex.length; i++) {
    const eid = primaryStep.elementIndex[i]!;
    if (!seenElementIds.has(eid)) {
      seenElementIds.add(eid);
      elementIds.push(eid);
    }
  }

  for (const elementId of elementIds) {
    const { xi: rawXi, values: rawValues } = extractElementStations(primaryStep, elementId);
    const declPath = opts.join && opts.join.ok ? (opts.join.byLadrunoId.get(elementId) ?? null) : null;
    const setInterpolated = (decl: DeclPath | null, reason: string): void => {
      out.set(elementId, { mode: "interpolated", elementId, decl, reason, xi: rawXi, values: rawValues });
    };

    if (!isForce) {
      setInterpolated(declPath, stageLoadsRes.ok ? "" : stageLoadsRes.reason);
      continue;
    }
    if (!opts.join || !opts.join.ok) {
      setInterpolated(null, opts.join ? opts.join.reason : "results are not paired with model declarations");
      continue;
    }
    if (!stageLoadsRes.ok) {
      setInterpolated(declPath, stageLoadsRes.reason);
      continue;
    }

    const elemDecl = declPath && opts.decls ? opts.decls[declPath] : undefined;
    if (!elemDecl) {
      setInterpolated(declPath, `element ${elementId} has no matching declaration in model.h5`);
      continue;
    }

    const opsTags = elemDecl.element?.metas.map((m) => m.tag) ?? (opts.join.space === "ops_tag" ? [elementId] : []);
    if (opsTags.length === 0) {
      setInterpolated(declPath, `element ${elementId} (${declPath}) has no OpenSees element_meta tag`);
      continue;
    }

    let unsupportedReason: string | null = null;
    for (const t of opsTags) {
      const r = stageLoadsRes.unsupportedByElementTag.get(t);
      if (r) {
        unsupportedReason = r;
        break;
      }
    }
    if (unsupportedReason !== null) {
      setInterpolated(declPath, unsupportedReason);
      continue;
    }

    const transfType = resolveElementGeomTransfType(opts.decls ?? {}, elemDecl, opts.definitions);
    if (!transfType) {
      setInterpolated(declPath, `element ${elementId} (${declPath}) has no resolved geomTransf`);
      continue;
    }
    if (/^corot/i.test(transfType)) {
      setInterpolated(
        declPath,
        `element ${elementId} uses '${transfType}' geomTransf (large-displacement); exact span reconstruction requires 'Linear' or 'PDelta'`,
      );
      continue;
    }
    if (transfType !== "Linear" && transfType !== "PDelta") {
      setInterpolated(declPath, `element ${elementId} uses unsupported geomTransf '${transfType}' (requires 'Linear' or 'PDelta')`);
      continue;
    }

    const integrationType = resolveElementBeamIntegrationType(opts.decls ?? {}, elemDecl, opts.definitions);
    if (integrationType !== null && INTERIOR_ONLY_BEAM_INTEGRATIONS.has(integrationType)) {
      setInterpolated(
        declPath,
        `element ${elementId} uses '${integrationType}' beamIntegration whose section stations do not include element ends xi = -1 and xi = +1`,
      );
      continue;
    }

    const L = opts.elementLengths.get(elementId);
    if (L === undefined || !(L > 0) || !Number.isFinite(L)) {
      setInterpolated(declPath, `element ${elementId} has invalid or missing length (${String(L)})`);
      continue;
    }

    const primaryEnds = findElementEndStations(primaryStep, elementId);
    if (!primaryEnds.ok) {
      setInterpolated(declPath, primaryEnds.reason);
      continue;
    }

    const elemTag = opsTags[0]!;
    const elemLoads = stageLoadsRes.loadsByElementTag.get(elemTag) ?? [];

    const nEnds = findElementEndStations(opts.stepByComponent.N, elementId);
    const vyEnds = findElementEndStations(stepVy, elementId);
    const vzEnds = findElementEndStations(stepVz, elementId);
    const tEnds = findElementEndStations(opts.stepByComponent.T, elementId);
    const myEnds = findElementEndStations(stepMy, elementId);
    const mzEnds = findElementEndStations(stepMz, elementId);

    if (opts.component === "Mz" && !vyEnds.ok) {
      setInterpolated(declPath, `exact Mz reconstruction on element ${elementId} requires companion shear Vy end stations (${vyEnds.reason})`);
      continue;
    }
    if (opts.component === "My" && !vzEnds.ok) {
      setInterpolated(declPath, `exact My reconstruction on element ${elementId} requires companion shear Vz end stations (${vzEnds.reason})`);
      continue;
    }
    if (opts.component === "Vy" && elemLoads.length > 0 && !mzEnds.ok) {
      setInterpolated(declPath, `exact Vy reconstruction on loaded element ${elementId} requires companion moment Mz end stations (${mzEnds.reason})`);
      continue;
    }
    if (opts.component === "Vz" && elemLoads.length > 0 && !myEnds.ok) {
      setInterpolated(declPath, `exact Vz reconstruction on loaded element ${elementId} requires companion moment My end stations (${myEnds.reason})`);
      continue;
    }

    const endForcesI: BeamSectionEndForcesI = {
      N0: nEnds.ok ? nEnds.vI : 0.0,
      Vy0: vyEnds.ok ? vyEnds.vI : 0.0,
      Vz0: vzEnds.ok ? vzEnds.vI : 0.0,
      T0: tEnds.ok ? tEnds.vI : 0.0,
      My0: myEnds.ok ? myEnds.vI : 0.0,
      Mz0: mzEnds.ok ? mzEnds.vI : 0.0,
    };

    const endForcesJ: BeamSectionEndForcesJ = {
      ...(nEnds.ok ? { NJ: nEnds.vJ } : {}),
      ...(vyEnds.ok ? { VyJ: vyEnds.vJ } : {}),
      ...(vzEnds.ok ? { VzJ: vzEnds.vJ } : {}),
      ...(tEnds.ok ? { TJ: tEnds.vJ } : {}),
      ...(myEnds.ok && vzEnds.ok ? { MyJ: myEnds.vJ } : {}),
      ...(mzEnds.ok && vyEnds.ok ? { MzJ: mzEnds.vJ } : {}),
    };

    const diagram = reconstructBeamElementDiagram({
      elementId,
      length: L,
      endForcesI,
      loads: elemLoads,
      ...(opts.nSub !== undefined ? { nSub: opts.nSub } : {}),
      extraXi: rawXi,
    });

    const closure = checkBeamSpanClosure(diagram, endForcesJ, opts.closureTolerance);
    if (!closure.ok) {
      setInterpolated(declPath, closure.reason);
      continue;
    }

    const compValues = diagram[opts.component];
    out.set(elementId, {
      mode: "exact",
      elementId,
      decl: declPath,
      diagram,
      xi: diagram.xi,
      values: compValues,
    });
  }

  return out;
}

export interface StateForSpanReconstruction {
  readonly decls: Readonly<Record<DeclPath, Decl>>;
  readonly definitions: Definitions | null;
  readonly sequence: SequenceZone;
  readonly ndm?: SpatialNdm | 2 | 3 | null;
  readonly results?: {
    readonly path: string;
    readonly elementIndexSpace?: string | null;
    readonly stages: readonly (Pick<ResultsStage, "id" | "name" | "kind"> & {
      readonly time: ArrayLike<number>;
      readonly elementIndexSpace?: string | null;
    })[];
  } | null;
  readonly phase: State["phase"];
}

/** Convenience helper to reconstruct span diagrams directly from a `State` snapshot and a map of `.ladruno` step data. */
export function reconstructSpanDiagramsFromState(
  state: StateForSpanReconstruction,
  stepByComponent: Readonly<Partial<Record<LineStationComponent, LadrunoLineStationStep>>>,
  elementLengths: ReadonlyMap<number, number>,
  pairing: Parameters<typeof joinLadrunoElements>[1],
  component: LineStationComponent,
  opts?: {
    nSub?: number;
    closureTolerance?: BeamClosureCheckOptions;
    stageIndex?: number;
    time?: number;
    provenance?: LadrunoProvenance | string | null;
  },
): Map<number, ElementSpanDiagramReconstruction> {
  const at = state.phase.at;
  const stageIdx = opts?.stageIndex ?? (at?.kind === "results" ? at.stage : 0);
  const resultsStage = state.results?.stages[stageIdx] ?? null;
  const stepIdx = at?.kind === "results" ? at.step : 0;
  const time = opts?.time ?? resultsStage?.time[stepIdx] ?? Number.NaN;
  const prov = opts?.provenance ?? resultsStage?.elementIndexSpace ?? state.results?.elementIndexSpace ?? null;
  const join = joinLadrunoElements(state.decls, pairing, prov);

  return reconstructSpanDiagramsForStep({
    component,
    stepByComponent,
    elementLengths,
    join,
    decls: state.decls,
    definitions: state.definitions,
    sequence: state.sequence,
    ndm: state.ndm,
    resultsStage,
    stageIndex: stageIdx,
    time,
    ...(opts?.nSub !== undefined ? { nSub: opts.nSub } : {}),
    ...(opts?.closureTolerance !== undefined ? { closureTolerance: opts.closureTolerance } : {}),
  });
}
