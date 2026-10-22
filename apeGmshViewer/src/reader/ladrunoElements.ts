// Gauss-point results of a `.ladruno`'s element buckets (ADR 0122 slice 8a, #2187).
//
// reader/ladruno.ts indexes a stage; this module turns each structured
// `RESULTS/ON_ELEMENTS/<result>/<classTag>-<ClassName>[...]` bucket that holds one
// value per Gauss point into the `GaussGroupInfo` reader/results.ts serves, and
// reads it back as `(E * G,)` element-major, as the native reader does. The layout
// is the fork's: `Ladruno_implementation/ladruno_schema_v1.md` §3 (MODEL/ELEMENTS,
// QUADRATURE/GP_PARAM) and §7.2 (COLUMN_MAP).
//
// Rules this file keeps:
// - a bucket is a Gauss bucket when its COLUMN_MAP has a GAUSS_ID row per block,
//   every one >= 0, no block repeated (MULTIPLICITY 1: a repeated block is a fiber
//   or layer set) and no FIBER_ID, and its element group's parametric space is not
//   1-D (a 1-D one is a beam's line stations, read by reader/ladruno.ts). Any
//   other bucket is not read here: it stays in `unread`, by name;
// - components are the file's own: `<result>/<label>`, the label from COMP_NAMES
//   (ADR 0122 D7: no token table), so a Gauss component never shares a name with a
//   nodal `<RESULT>/<label>` by accident of spelling;
// - positions are the file's: natural coordinates are QUADRATURE/GP_PARAM of the
//   bucket's MODEL/ELEMENTS group, row g for GAUSS_ID g, never a TS table of rules.
//   They are offered (for the Gauss-to-node extrapolation) only on PARAM_DOMAIN
//   "[-1,1]", the domain state/extrapolation.ts's shape functions read; another
//   domain ("[0,1]", "bary") warns and offers none, never a re-mapped guess;
// - a Gauss bucket that disagrees with itself is refused by name, with the reason,
//   in `GaussGroupInfo.refused`, and never read: a GAUSS_ID count that is not
//   GP_PARAM's (or a GLOBAL_GP_COORDS width that is no multiple of it), NUM_COLUMNS
//   that is not the sum of MULTIPLICITY * NUM_COMP or DATA's width, a DATA step axis
//   that disagrees with TIME and STEP (a partial write is never read short), a
//   TIME that is not the stage's, and a bucket with no Gauss point (G == 0, as
//   V4c2a refuses natively);
// - element rows are the bucket's ID: FEM element ids only when
//   INFO/PROVENANCE@element_index_space is the well-formed "fem" (#2079 Q6);
// - an element id is listed once per token (#2316): every bucket of every
//   ON_ELEMENTS token is checked at open, Gauss or not (line-station, fiber and
//   LOCAL_AXES buckets too), and an id twice in one bucket, or in two buckets of one
//   token, refuses the file by name (reader/ladrunoChecks.ts). A bucket whose ID is
//   absent or not an (n, 1) integer dataset holds no element row any read takes, so it
//   lists none. Two tokens holding one element is the norm (stress and strain).
//
// Browser-safe: no Node import, and no import of reader/ladruno.ts (it imports this).

import { uniqueElementIds, type IdList } from "./ladrunoChecks.ts";
import type { ElementIds, GaussGroupInfo } from "./results.ts";

const H5T_INTEGER = 0;
const H5T_FLOAT = 1;
/** The PARAM_DOMAIN whose GP_PARAM rows are the natural coordinates state/extrapolation.ts reads. */
const EXTRAPOLATION_DOMAIN = "[-1,1]";

interface ENode {
  path: string;
  attrs: Record<string, { value: unknown }>;
}
export interface EGroup extends ENode {
  keys(): string[];
  get(name: string): unknown;
}
/** The open file: its root group, and the name a refusal gives it. */
export interface EFile extends EGroup {
  filename: string;
}
interface EDataset extends ENode {
  shape: number[] | null;
  metadata: { type: number };
  value: unknown;
  slice(ranges: ([] | [number | null] | [number | null, number | null])[]): unknown;
}
export interface ECtors {
  Group: abstract new (...args: never[]) => unknown;
  Dataset: abstract new (...args: never[]) => unknown;
}
export type ElementsFail = (kind: "layout" | "dtype" | "argument", message: string) => Error;

/** One INFO/PROVENANCE attribute as reader/ladruno.ts reports it (`ProvenanceFact`). */
export type SpaceFact = { state: "present"; value: string } | { state: "absent" } | { state: "malformed"; reason: string };
export type ElementIndexVerdict = { verified: true } | { verified: false; reason: string };

/** One Gauss bucket as indexed: its info, where its values are, and each component's DATA column per Gauss point. */
interface GaussBucket {
  info: GaussGroupInfo;
  dataPath: string;
  /** the bucket's ID rows (E) */
  ids: Float64Array;
  /** component -> DATA column of Gauss point g (g = the GP_PARAM row, 0..G-1) */
  columns: Map<string, Int32Array>;
}

const not = (reason: string) => ({ reason });

/**
 * Whether a `.ladruno`'s element rows are FEM element ids: only a well-formed
 * `element_index_space = "fem"` says so (#2079 Q6). The one verdict: main's meta
 * and every element read take it from here.
 */
export function elementIndexVerdictOf(space: SpaceFact | null): ElementIndexVerdict {
  const tail = "its element rows are not known to be FEM element ids";
  if (space === null) return { verified: false, reason: `the file has no INFO/PROVENANCE group, so ${tail}` };
  if (space.state === "malformed") return { verified: false, reason: `${space.reason}; ${tail}` };
  if (space.state === "absent" || space.value !== "fem") return { verified: false, reason: `INFO/PROVENANCE has no @element_index_space = "fem", so ${tail}` };
  return { verified: true };
}

/**
 * Each bucket's DATA columns `[lo, hi)` that hold `cols`, and the gather of one
 * read of them into `(nSteps * E * G,)`, step-major, element-major, then Gauss
 * point (reader/results.ts's `readGaussRows` order).
 */
export function gatherGauss(raw: Float64Array, nSteps: number, E: number, lo: number, hi: number, cols: Int32Array): Float64Array {
  const w = hi - lo;
  const G = cols.length;
  if (raw.length !== nSteps * E * w) throw new RangeError(`gatherGauss: ${raw.length} values for ${nSteps} steps x ${E} elements x ${w} columns`);
  const out = new Float64Array(nSteps * E * G);
  for (let r = 0; r < nSteps * E; r++) for (let g = 0; g < G; g++) out[r * G + g] = raw[r * w + cols[g]! - lo]!;
  return out;
}

export class LadrunoGauss {
  private readonly h5: ECtors;
  private readonly file: EFile;
  private readonly fail: ElementsFail;
  private readonly buckets = new Map<string, Map<string, GaussBucket>>();

  constructor(h5: ECtors, file: EFile, fail: ElementsFail) {
    this.h5 = h5;
    this.file = file;
    this.fail = fail;
  }

  /**
   * Index one stage's ON_ELEMENTS: its Gauss groups (read or refused), and the
   * results this module does not read, as `ON_ELEMENTS/<result>` (none of its
   * buckets read) or `ON_ELEMENTS/<result>/<bucket>` (some of them read).
   */
  indexStage(stageId: string, onElements: EGroup, modelElements: EGroup | null, time: Float64Array, warnings: string[]): { gauss: GaussGroupInfo[]; unread: string[] } {
    const byId = new Map<string, GaussBucket>();
    const unread: string[] = [];
    for (const result of onElements.keys()) {
      const rg = this.groupOf(onElements, result);
      if (!rg) {
        unread.push(`ON_ELEMENTS/${result} (not a group)`);
        continue;
      }
      const left: string[] = [];
      let read = 0;
      // The token's element ids: its buckets', and an ID on the token itself (a flat LOCAL_AXES, which a step read refuses, #2451).
      const lists: IdList[] = [];
      const own = this.elementIdsIn(rg);
      if (own) lists.push({ at: `${rg.path}/ID`, ids: own });
      for (const key of rg.keys()) {
        const bg = this.groupOf(rg, key);
        const ids = bg ? this.elementIdsIn(bg) : null;
        if (ids) lists.push({ at: `${bg!.path}/ID`, ids });
        const b = bg ? this.bucket(result, key, bg, ids, modelElements, time, warnings) : null;
        if (b) {
          byId.set(b.info.id, b);
          read++;
        } else left.push(key);
      }
      uniqueElementIds(
        lists,
        `${this.file.filename}: ${rg.path}`,
        "the recorder writes each element once per token, so a repeat would read as two rows of one element; the file is not read",
        (m) => this.fail("layout", m),
      );
      if (read === 0) unread.push(`ON_ELEMENTS/${result}`);
      else for (const key of left) unread.push(`ON_ELEMENTS/${result}/${key}`);
    }
    this.buckets.set(stageId, byId);
    return { gauss: [...byId.values()].map((b) => b.info), unread };
  }

  /** Steps `k0` to `k1` (exclusive) of one component of one Gauss group of `stage`: ((k1 - k0) * E * G,). */
  read(stage: { id: string; nSteps: number }, groupId: string, component: string, k0: number, k1: number): Float64Array {
    const stageId = stage.id;
    if (!Number.isInteger(k0) || !Number.isInteger(k1) || k0 < 0 || k1 > stage.nSteps || k0 >= k1) {
      throw this.fail("argument", `steps ${k0}..${k1} are not a range inside stage ${stageId} (0..${stage.nSteps})`);
    }
    const b = this.bucketOf(stageId, groupId);
    const cols = b.columns.get(component);
    if (!cols) {
      throw this.fail("argument", `gauss group ${groupId} of stage ${stageId} has no component ${JSON.stringify(component)}; it has ${b.info.components.join(", ") || "none"}`);
    }
    let lo = Infinity;
    let hi = -Infinity;
    for (const c of cols) {
      lo = Math.min(lo, c);
      hi = Math.max(hi, c + 1);
    }
    const ds = this.file.get(b.dataPath);
    if (!(ds instanceof this.h5.Dataset)) throw this.fail("layout", `${b.dataPath} is not a dataset`);
    const v = (ds as unknown as EDataset).slice([[k0, k1], [], [lo, hi]]);
    const raw = v instanceof Float64Array ? v : v instanceof Float32Array ? Float64Array.from(v) : null;
    if (!raw) throw this.fail("dtype", `${b.dataPath}: read returned ${Object.prototype.toString.call(v)}, not float32 or float64`);
    return gatherGauss(raw, k1 - k0, b.ids.length, lo, hi, cols);
  }

  /** A Gauss group's element rows under `verdict`. */
  elementIds(stageId: string, groupId: string, verdict: ElementIndexVerdict): ElementIds {
    const ids = this.bucketOf(stageId, groupId).ids;
    return verdict.verified ? { mapping: "fem_eid", sourceType: "ladruno", ids } : { mapping: "unverified", sourceType: "ladruno", reason: verdict.reason, rawIndex: ids };
  }

  private bucketOf(stageId: string, groupId: string): GaussBucket {
    const b = this.buckets.get(stageId)?.get(groupId);
    if (!b) throw this.fail("argument", `stage ${stageId} has no gauss group ${JSON.stringify(groupId)}`);
    return b;
  }

  /** A Gauss bucket, read or refused; null when the bucket is not one (it stays unread). `idRows` is `elementIdsIn(bg)`. */
  private bucket(result: string, key: string, bg: EGroup, idRows: Float64Array | null, modelElements: EGroup | null, time: Float64Array, warnings: string[]): GaussBucket | null {
    const cm = this.groupOf(bg, "COLUMN_MAP");
    const gidDs = cm ? this.datasetOf(cm, "GAUSS_ID") : null;
    if (!cm || !gidDs || gidDs.metadata.type !== H5T_INTEGER) return null;
    const gids = ints(gidDs);
    const mult = this.intsOf(cm, "MULTIPLICITY");
    const fiber = this.intsOf(cm, "FIBER_ID");
    if (gids.some((x) => x < 0) || mult?.some((m) => m !== 1) || fiber?.some((x) => x >= 0)) return null;
    const model = modelGroupFor(modelElements, key, (n) => this.groupOf(modelElements!, n));
    const quad = model ? this.groupOf(model, "QUADRATURE") : null;
    const gp = quad ? this.datasetOf(quad, "GP_PARAM") : null;
    const ndir = gp?.shape?.length === 2 ? gp.shape[1]! : gp?.shape?.length === 1 ? 1 : null;
    if (oneValue(model?.attrs["TOPOLOGY"]?.value) === "line" || ndir === 1) return null;

    const id = `${result}/${key}`;
    const classTag = Number(/^(\d+)-/.exec(key)?.[1] ?? -1);
    const rule = intOf(oneValue(model?.attrs["INTEGRATION_RULE"]?.value));
    const blocks = parseCompNames(cm.path, cm.attrs["COMP_NAMES"]?.value, gids.length, "GAUSS_ID");
    const labels = typeof blocks === "string" ? null : blocks[0] ?? [];
    const refuse = (reason: string, ids: Float64Array = new Float64Array(0)): GaussBucket => ({
      info: {
        id,
        path: bg.path,
        classTag,
        intRule: rule ?? -1,
        nElements: ids.length,
        nGauss: gids.length,
        components: [],
        refused: labels && labels.length ? labels.map((l) => ({ component: `${result}/${l}`, reason })) : [{ component: result, reason }],
        localAxes: false,
        naturalCoords: null,
      },
      dataPath: `${bg.path}/DATA`,
      ids,
      columns: new Map(),
    });

    // The datasets, and the step and element axes (seam "partial writes").
    const ds = ["ID", "DATA", "TIME", "STEP"].map((k) => this.datasetOf(bg, k));
    const missing = ["ID", "DATA", "TIME", "STEP"].filter((_, i) => !ds[i]);
    if (missing.length) return refuse(`${bg.path} has no dataset ${missing.join(", ")}`);
    const [idDs, dataDs, timeDs, stepDs] = ds as EDataset[];
    if (!idRows) return refuse(`${idDs!.path} is not an (n, 1) integer dataset`);
    const ids = idRows;
    const shape = dataDs!.shape ?? [];
    if (shape.length !== 3 || dataDs!.metadata.type !== H5T_FLOAT) return refuse(`${dataDs!.path} is not a 3-D floating-point dataset (T, elements, columns)`, ids);
    const [T, nE, width] = shape as [number, number, number];
    const nTime = timeDs!.shape?.length === 1 ? timeDs!.shape[0]! : -1;
    const nStep = stepDs!.shape?.length === 1 ? stepDs!.shape[0]! : -1;
    if (T !== nTime || T !== nStep) return refuse(`${dataDs!.path} holds ${T} steps, TIME ${nTime} and STEP ${nStep}: the step axes disagree (a recorder stopped mid-step)`, ids);
    if (nE !== ids.length) return refuse(`${dataDs!.path} has ${nE} element rows and ${idDs!.path} ${ids.length} ids (a partial write)`, ids);
    if (timeDs!.metadata.type !== H5T_FLOAT || !sameTime(timeDs!.value, time)) return refuse(`${timeDs!.path} is not the stage's time (${time.length} steps)`, ids);

    // The column map against itself and DATA (schema §7.2's invariant).
    if (gids.length === 0) return refuse(`${cm.path} lists no Gauss point (G == 0): the bucket has no value to show`, ids);
    if (typeof blocks === "string") return refuse(blocks, ids);
    const numComp = this.intsOf(cm, "NUM_COMP");
    if (!numComp || numComp.length !== gids.length) return refuse(`${cm.path}/NUM_COMP is not an integer dataset with one row per GAUSS_ID row (${gids.length})`, ids);
    const bad = numComp.findIndex((n, i) => n !== blocks[i]!.length);
    if (bad >= 0) return refuse(`${cm.path}: NUM_COMP[${bad}] = ${numComp[bad]} but COMP_NAMES line ${bad} names ${blocks[bad]!.length} components`, ids);
    const sum = numComp.reduce((a, n, i) => a + (mult?.[i] ?? 1) * n, 0);
    const numColumns = intOf(oneValue(bg.attrs["NUM_COLUMNS"]?.value));
    if (numColumns === null) return refuse(`${bg.path} has no integer NUM_COLUMNS attribute`, ids);
    if (numColumns !== sum || numColumns !== width) {
      return refuse(`${bg.path}@NUM_COLUMNS = ${numColumns}, the column map sums MULTIPLICITY * NUM_COMP to ${sum}, and DATA has ${width} columns: they must agree`, ids);
    }
    if (blocks.some((b) => b.join(",") !== labels!.join(","))) return refuse(`${cm.path}@COMP_NAMES: the Gauss points name different components (${blocks.map((b) => b.join(",")).join(" | ")})`, ids);

    // The Gauss points against the file's quadrature (seam "positions").
    if (!model) return refuse(`no MODEL/ELEMENTS group matches ${key}, so its Gauss points cannot be placed`, ids);
    if (!gp || gp.metadata.type !== H5T_FLOAT || ndir === null) return refuse(`${model.path} has no floating-point QUADRATURE/GP_PARAM, so its Gauss points cannot be placed`, ids);
    const G = gp.shape![0]!;
    if (gids.length !== G) return refuse(`${cm.path}/GAUSS_ID lists ${gids.length} Gauss points and ${gp.path} ${G}: the counts disagree`, ids);
    if (new Set(gids).size !== G || gids.some((g) => g >= G)) return refuse(`${cm.path}/GAUSS_ID = [${gids.join(", ")}] is not each GP_PARAM row 0..${G - 1} once`, ids);
    const glob = this.datasetOf(model, "GLOBAL_GP_COORDS");
    if (glob && (glob.shape?.length !== 2 || glob.shape[1]! % G !== 0)) return refuse(`${glob.path} has shape (${(glob.shape ?? []).join(", ")}), not (elements, ${G} x dimension)`, ids);

    const columns = new Map<string, Int32Array>();
    labels!.forEach((label, c) => {
      const cols = new Int32Array(G);
      gids.forEach((g, i) => (cols[g] = i * labels!.length + c));
      columns.set(`${result}/${label}`, cols);
    });
    const domain = oneValue(model.attrs["PARAM_DOMAIN"]?.value);
    let naturalCoords: number[][] | null = null;
    if (domain === EXTRAPOLATION_DOMAIN) {
      const flat = gp.value instanceof Float64Array ? gp.value : Float64Array.from(gp.value as ArrayLike<number>);
      naturalCoords = Array.from({ length: G }, (_, r) => Array.from(flat.subarray(r * ndir, (r + 1) * ndir)));
    } else {
      warnings.push(`${bg.path}: ${model.path}@PARAM_DOMAIN is ${JSON.stringify(domain ?? null)}, not "${EXTRAPOLATION_DOMAIN}"; its Gauss points are read, but not extrapolated to nodes`);
    }
    return {
      info: { id, path: bg.path, classTag, intRule: rule ?? -1, nElements: ids.length, nGauss: G, components: [...columns.keys()], refused: [], localAxes: false, naturalCoords },
      dataPath: dataDs!.path,
      ids,
      columns,
    };
  }

  /** The element ids of a bucket's ID, when it is an (n, 1) or (n,) integer dataset; else null (no read takes its rows). */
  private elementIdsIn(bg: EGroup): Float64Array | null {
    const ds = this.datasetOf(bg, "ID");
    const shape = ds?.shape ?? [];
    if (!ds || ds.metadata.type !== H5T_INTEGER || !(shape.length === 1 || (shape.length === 2 && shape[1] === 1))) return null;
    const v = ds.value;
    if (v instanceof BigInt64Array || v instanceof BigUint64Array) return Float64Array.from(v, Number);
    if (ArrayBuffer.isView(v) && !(v instanceof DataView)) return Float64Array.from(v as unknown as ArrayLike<number>);
    throw this.fail("dtype", `${ds.path}: read returned ${Object.prototype.toString.call(v)}, not an integer array`);
  }

  private groupOf(parent: EGroup, name: string): EGroup | null {
    if (!parent.keys().includes(name)) return null;
    const g = parent.get(name);
    return g instanceof this.h5.Group ? (g as unknown as EGroup) : null;
  }

  private datasetOf(parent: EGroup, name: string): EDataset | null {
    if (!parent.keys().includes(name)) return null;
    const d = parent.get(name);
    return d instanceof this.h5.Dataset ? (d as unknown as EDataset) : null;
  }

  /** An optional integer dataset, flattened; null when absent or not integer. */
  private intsOf(parent: EGroup, name: string): number[] | null {
    const d = this.datasetOf(parent, name);
    return d && d.metadata.type === H5T_INTEGER ? ints(d) : null;
  }
}

/**
 * The MODEL/ELEMENTS group of an ON_ELEMENTS bucket: the same
 * `<classTag>-<ClassName>`, whose bracket fields are the first fields of the
 * bucket's (`31-FourNodeQuad[201:0]` for `31-FourNodeQuad[201:0:0]`). Null when none
 * matches. The one match: the Gauss and the line-station reads both use it.
 */
export function modelGroupFor<G>(modelElements: { keys(): string[] } | null, key: string, groupOf: (name: string) => G | null): G | null {
  if (!modelElements) return null;
  const parse = (k: string) => /^(?<cls>[^[]+)(?:\[(?<f>[^\]]*)\])?$/.exec(k)?.groups;
  const b = parse(key);
  if (!b) return null;
  const bf = (b.f ?? "").split(":");
  for (const name of modelElements.keys()) {
    const m = parse(name);
    if (!m || m.cls !== b.cls) continue;
    const mf = (m.f ?? "").split(":");
    if (mf.length <= bf.length && mf.every((x, i) => x === bf[i])) return groupOf(name);
  }
  return null;
}

/**
 * A COLUMN_MAP's COMP_NAMES (schema §7.2: one CSV line per block) as one label
 * list per block, or why it cannot be read: not a string, a line count that is
 * not the `nBlocks` rows of `rowsOf`, an empty label or a repeated one. The one
 * parser: the Gauss and the line-station reads both use it (fail closed: an empty
 * label is refused, never dropped).
 */
export function parseCompNames(cmPath: string, attr: unknown, nBlocks: number, rowsOf: "GAUSS_ID" | "LEVELS"): string[][] | string {
  const raw = oneValue(attr);
  if (typeof raw !== "string") return `${cmPath} has no string COMP_NAMES attribute`;
  const lines = nBlocks === 0 && raw === "" ? [] : raw.split("\n");
  if (lines.length !== nBlocks) return `${cmPath}@COMP_NAMES has ${lines.length} lines and ${rowsOf} ${nBlocks} rows`;
  const out: string[][] = [];
  for (const line of lines) {
    const names = line.split(",").map((t) => t.trim());
    if (names.some((t) => t === "")) return `${cmPath}@COMP_NAMES line ${JSON.stringify(line)} has an empty label`;
    if (new Set(names).size !== names.length) return `${cmPath}@COMP_NAMES line ${JSON.stringify(line)} repeats a label`;
    out.push(names);
  }
  return out;
}

function sameTime(v: unknown, time: Float64Array): boolean {
  const t = v instanceof Float64Array ? v : v instanceof Float32Array ? Float64Array.from(v) : null;
  return t !== null && t.length === time.length && t.every((x, i) => x === time[i]);
}

/** A scalar attribute, or the one element of a one-element array (the fork writes shape (1,)). Shared with reader/ladruno.ts. */
export function oneValue(v: unknown): unknown {
  if (Array.isArray(v)) return v.length === 1 ? v[0] : v;
  if (ArrayBuffer.isView(v) && !(v instanceof DataView)) {
    const a = v as unknown as ArrayLike<unknown>;
    return a.length === 1 ? a[0] : v;
  }
  return v;
}

/** An integer attribute value (number or bigint), else null. Shared with reader/ladruno.ts. */
export function intOf(v: unknown): number | null {
  if (typeof v === "bigint") return Number(v);
  if (typeof v === "number" && Number.isInteger(v)) return v;
  return null;
}

function ints(ds: EDataset): number[] {
  const v = ds.value;
  if (v instanceof BigInt64Array) return Array.from(v, (x) => Number(x));
  if (ArrayBuffer.isView(v) && !(v instanceof DataView)) return Array.from(v as unknown as ArrayLike<number>, Number);
  return [];
}
