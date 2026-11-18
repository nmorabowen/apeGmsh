// Read and write a `<stem>.workspace.h5` (ADR 0132 D1-D7; h5-schema.md
// "`<stem>.workspace.h5`: what the user makes").
//
// The app's twin of src/apeGmsh/results/user/_workspace.py. Three levels of
// refusal, as Python's: a file that is not HDF5, has no version key, is outside
// the version floor or lacks a table refuses the file (`WorkspaceRefusal`);
// malformed columns refuse that table (`broken`); a bad row refuses alone, with
// a reason (`refused`). One difference, by ADR 0113 D7: a same-major newer
// file opens here read-only (`readOnly`), where Python refuses it; no writer
// writes over it (D2).
//
// A set's verdict is judged against the paired `<stem>.h5` (`verdictsOf`):
// `current` when every identity hash is equal, otherwise each member is looked
// for at its stored coordinates, and a member matches only when its own tag
// sits within `1e-9` of the model's bounding-box diagonal **and no other** item
// of the set's kind does. The search is Python's, step for step (the same
// projection direction, slack and distance formula), so the two agree on every
// member.
//
// The write (`buildWrite`, `writePartial`, `verifyPartial`) builds the whole
// new file in a partial and reads it back; main/workspace.ts owns the disk
// steps around it (the revision check, the rename aside, the replace). Every
// file this module opens it reads whole and closes before returning: it never
// keeps a handle (HG9 rule 1).
//
// No Electron import: the tests run in Node.

import { createH5, openH5, type H5FS } from "./h5open.ts";
import { addressProblem, nameProblem } from "./userRules.ts";
import { validateSpec } from "../spec/spec.ts";
import type { DefRowIn, RowRefusalIn, SetKind, SetVerdictIn, SetVerdictName, WorkspaceSetIn } from "../state/types/workspace.ts";

/** The writer's version of the file, and the oldest one this reader opens (ADR 0113). */
export const WORKSPACE_SCHEMA_VERSION = "1.0.0";
export const WORKSPACE_SCHEMA_FLOOR = "1.0.0";
export const VERSION_KEY = "workspace_schema_version";
export const SUFFIX = ".workspace.h5";
export const PLOT_GRAMMAR = "apegmsh-plot/1";
export const VIEW_GRAMMAR = "apegmsh-view/1";
export const ID_SPACE = "opensees";
export const KINDS: readonly SetKind[] = ["nodes", "elements"];
export const TABLES = ["sets", "plots", "views"] as const;
export type Table = (typeof TABLES)[number];
/** A member matches within this fraction of the model's bounding-box diagonal. */
export const TOLERANCE = 1e-9;
/** The refusals that make an old file *unreadable*: renamed aside, never overwritten (D6). */
export const UNREADABLE: ReadonlySet<string> = new Set(["not-hdf5", "no-version", "bad-version", "bad-meta"]);

const VERSION = /^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$/;
const SET_TEXT = ["name", "kind", "id_space", "fem_hash", "model_hash"] as const;
const ROW_TEXT = ["name", "grammar", "body"] as const;
const I4_MIN = -(2 ** 31);
const I4_MAX = 2 ** 31 - 1;
const H5T_INTEGER = 0;
const H5T_FLOAT = 1;
const H5T_STRING = 3;
/** The direction the coordinate search sorts along (Python's `_DIRECTION`); irrational, so mesh planes rarely collapse onto one projected value. */
const DIRECTION = (() => {
  const d = [1.0, 0.6180339887498949, 0.41421356237309515];
  const n = Math.sqrt(d[0]! * d[0]! + d[1]! * d[1]! + d[2]! * d[2]!);
  return [d[0]! / n, d[1]! / n, d[2]! / n] as const;
})();
const EPS = 2 ** -52;
const show = (x: unknown): string => {
  try {
    return JSON.stringify(x) ?? String(x);
  } catch {
    return String(x);
  }
};

// ---------------------------------------------------------------------------
// What this module needs of h5wasm (`h5wasm/node`, after `await h5wasm.ready`)
// ---------------------------------------------------------------------------

export interface WsMeta {
  type: number;
  size: number;
  signed?: boolean;
  vlen?: boolean;
}
export interface WsNode {
  path: string;
  attrs: Record<string, { value: unknown; metadata: WsMeta }>;
}
export interface WsGroup extends WsNode {
  keys(): string[];
  get(name: string): unknown;
}
export interface WsDataset extends WsNode {
  shape: number[] | null;
  metadata: WsMeta;
  value: unknown;
}
export interface WsWriteGroup {
  create_group(name: string): WsWriteGroup;
  create_dataset(args: { name: string; data: unknown; shape: number[]; dtype: string }): unknown;
  create_attribute(name: string, data: unknown, shape: number[] | null, dtype: string): unknown;
}
export interface WsFile extends WsGroup, WsWriteGroup {
  file_id: bigint | number;
  close(): unknown;
}
export interface WorkspaceH5 {
  File: new (filename: string, mode: "r" | "x") => WsFile;
  readonly FS: H5FS | null;
  Group: abstract new (...args: never[]) => WsGroup;
  Dataset: abstract new (...args: never[]) => WsDataset;
}

/** A file-level refusal, with the schema's word (`not-hdf5`, `no-version`, `newer`, `missing-table`, `unpaired`, ...). */
export class WorkspaceRefusal extends Error {
  readonly reason: string;
  readonly detail: string;
  constructor(reason: string, detail: string) {
    super(`${reason}: ${detail}`);
    this.reason = reason;
    this.detail = detail;
  }
}

const isGroup = (h5: WorkspaceH5, x: unknown): x is WsGroup => x instanceof h5.Group;
const isDataset = (h5: WorkspaceH5, x: unknown): x is WsDataset => x instanceof h5.Dataset;
const has = (g: WsGroup, name: string) => g.keys().includes(name);

/** Open `path` read-only, run `read`, and close it whatever happens: no handle outlives the call. */
function withFile<T>(h5: WorkspaceH5, path: string, read: (f: WsFile) => T): T {
  const f = openH5(h5 as unknown as Parameters<typeof openH5<WsFile>>[0], path);
  try {
    if (Number(f.file_id) < 0) throw new Error(`HDF5 could not open ${path}`);
    return read(f);
  } finally {
    if (Number(f.file_id) >= 0) f.close();
  }
}

// ---------------------------------------------------------------------------
// Strict JSON (Python's `strict_json`)
// ---------------------------------------------------------------------------

/** A body as read: its value, and whether a number was written as a float token (`1.0`, `1e2`) rather than an integer. */
export interface JsonRead {
  value: unknown;
  floatAt(container: object, key: string | number): boolean;
}

const JSON_WS = new Set([" ", "\t", "\n", "\r"]);
const JSON_NUMBER = /-?(?:0|[1-9]\d*)(\.\d+)?([eE][+-]?\d+)?/y;
const JSON_ESCAPES: Record<string, string> = { '"': '"', "\\": "\\", "/": "/", b: "\b", f: "\f", n: "\n", r: "\r", t: "\t" };

/**
 * RFC 8259 as Python's `strict_json` reads it: no `NaN` or `Infinity` token,
 * no key twice, no float token that overflows to infinity. It keeps what
 * `JSON.parse` loses and the plot grammar needs: whether a number was written
 * as an integer, because Python's `int` and `float` differ there and the
 * grammar's integer keys (`steps`, `step`, `mode`) refuse a float.
 */
export function strictJson(text: string): JsonRead {
  const floats = new WeakMap<object, Set<string | number>>();
  let i = 0;
  const fail = (why: string): never => {
    throw new Error(`not JSON: ${why} at character ${i}`);
  };
  const skip = () => {
    while (i < text.length && JSON_WS.has(text[i]!)) i++;
  };
  const markFloat = (container: object, key: string | number) => {
    let s = floats.get(container);
    if (!s) floats.set(container, (s = new Set()));
    s.add(key);
  };
  const str = (): string => {
    i++;
    let out = "";
    for (;;) {
      if (i >= text.length) fail("an unterminated string");
      const c = text[i]!;
      if (c === '"') {
        i++;
        return out;
      }
      if (c < " ") fail("a control character in a string");
      if (c !== "\\") {
        out += c;
        i++;
        continue;
      }
      const e = text[i + 1];
      if (e === "u") {
        const hex = text.slice(i + 2, i + 6);
        if (!/^[0-9a-fA-F]{4}$/.test(hex)) fail("a bad \\u escape");
        out += String.fromCharCode(parseInt(hex, 16));
        i += 6;
        continue;
      }
      const v = e === undefined ? undefined : JSON_ESCAPES[e];
      if (v === undefined) fail(`a bad escape \\${e ?? ""}`);
      out += v;
      i += 2;
    }
  };
  const value = (isFloat: () => void): unknown => {
    skip();
    const c = text[i];
    if (c === undefined) fail("the text ended where a value was expected");
    if (c === "{") {
      i++;
      const obj: Record<string, unknown> = {};
      const seen = new Set<string>();
      skip();
      if (text[i] === "}") {
        i++;
        return obj;
      }
      for (;;) {
        skip();
        if (text[i] !== '"') fail("an object key that is not a string");
        const key = str();
        if (seen.has(key)) throw new Error(`the key ${show(key)} appears twice`);
        seen.add(key);
        skip();
        if (text[i] !== ":") fail("a missing ':'");
        i++;
        // `__proto__` is an own key, never the prototype.
        Object.defineProperty(obj, key, { value: value(() => markFloat(obj, key)), enumerable: true, writable: true, configurable: true });
        skip();
        if (text[i] === ",") {
          i++;
          continue;
        }
        if (text[i] === "}") {
          i++;
          return obj;
        }
        fail("a missing ',' or '}'");
      }
    }
    if (c === "[") {
      i++;
      const arr: unknown[] = [];
      skip();
      if (text[i] === "]") {
        i++;
        return arr;
      }
      for (;;) {
        const at = arr.length;
        arr.push(value(() => markFloat(arr, at)));
        skip();
        if (text[i] === ",") {
          i++;
          continue;
        }
        if (text[i] === "]") {
          i++;
          return arr;
        }
        fail("a missing ',' or ']'");
      }
    }
    if (c === '"') return str();
    for (const [word, v] of [["true", true], ["false", false], ["null", null]] as const) {
      if (text.startsWith(word, i)) {
        i += word.length;
        return v;
      }
    }
    for (const t of ["NaN", "Infinity", "-Infinity"]) if (text.startsWith(t, i)) throw new Error(`${t} is not a JSON number`);
    JSON_NUMBER.lastIndex = i;
    const m = JSON_NUMBER.exec(text);
    if (!m) return fail("an unexpected character");
    i += m[0].length;
    const n = Number(m[0]);
    if (m[1] !== undefined || m[2] !== undefined) {
      if (!Number.isFinite(n)) throw new Error(`${m[0]} overflows to infinity`);
      isFloat();
    }
    return n;
  };
  const root = value(() => undefined);
  skip();
  if (i !== text.length) fail("extra data after the value");
  return { value: root, floatAt: (container, key) => floats.get(container)?.has(key) ?? false };
}

// ---------------------------------------------------------------------------
// The plot grammar, apegmsh-plot/1 (closed keys; Python's `plot_problem`)
// ---------------------------------------------------------------------------

const PLOT_KEYS: Record<string, readonly [readonly string[], readonly string[]]> = {
  history: [["kind", "over", "series"], ["kind", "over", "series"]],
  xy: [["kind", "over", "x", "y"], ["kind", "over", "x", "y"]],
  profile: [["kind", "state", "axis", "series"], ["kind", "state", "axis", "series"]],
};
const SERIES_KEYS = [["label", "quantity", "target", "reduce", "minus", "scale"], ["quantity", "target", "reduce"]] as const;
const REDUCE = ["each", "sum", "mean", "min", "max", "absmax"];
const GROUPS = ["results", "user_results"];
const TARGET_KINDS = ["set", "physical", "label", "mesh_selection"];
const SWEEP_AXES = ["time", "step"];
const PROFILE_AXES = ["member", "x", "y", "z", "path"];
const pyTuple = (xs: readonly string[]) => `(${xs.map((x) => `'${x}'`).join(", ")})`;
const isObj = (x: unknown): x is Record<string, unknown> => typeof x === "object" && x !== null && !Array.isArray(x);
const sorted = (xs: Iterable<string>) => [...xs].sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));

function closed(obj: unknown, allowed: readonly string[], required: readonly string[], where: string): string | null {
  if (!isObj(obj)) return `${where} is not an object`;
  const extra = sorted(Object.keys(obj).filter((k) => !allowed.includes(k)));
  if (extra.length) return `${where}: the key ${show(extra[0])} is not allowed`;
  const missing = sorted(required.filter((k) => !Object.hasOwn(obj, k)));
  if (missing.length) return `${where}: the key ${show(missing[0])} is required`;
  return null;
}

/** A JSON number that is no boolean and is finite (Python's `finite_number`). */
function finiteNumber(v: unknown): boolean {
  return typeof v === "number" && Number.isFinite(v);
}

function targetProblem(target: unknown, where: string): string | null {
  const p = closed(target, ["kind", "name"], ["kind", "name"], where);
  if (p !== null) return p;
  const t = target as Record<string, unknown>;
  if (!TARGET_KINDS.includes(t["kind"] as string)) return `${where}.kind ${show(t["kind"])} is not one of ${pyTuple(TARGET_KINDS)}`;
  const n = nameProblem(t["name"]);
  return n !== null ? `${where}.name: ${n}` : null;
}

function seriesProblem(series: unknown, where: string, rule: "each" | "one" | "any"): string | null {
  const p = closed(series, SERIES_KEYS[0], SERIES_KEYS[1], where);
  if (p !== null) return p;
  const s = series as Record<string, unknown>;
  if ("label" in s && nameProblem(s["label"]) !== null) return `${where}.label: ${nameProblem(s["label"])}`;
  const q = s["quantity"];
  const qp = closed(q, ["group", "field", "component"], ["group", "field", "component"], `${where}.quantity`);
  if (qp !== null) return qp;
  const qo = q as Record<string, unknown>;
  if (!GROUPS.includes(qo["group"] as string)) return `${where}.quantity.group ${show(qo["group"])} is not one of ${pyTuple(GROUPS)}`;
  for (const key of ["field", "component"]) if (nameProblem(qo[key]) !== null) return `${where}.quantity.${key}: ${nameProblem(qo[key])}`;
  const tp = targetProblem(s["target"], `${where}.target`);
  if (tp !== null) return tp;
  const reduce = s["reduce"];
  if (!REDUCE.includes(reduce as string)) return `${where}.reduce ${show(reduce)} is not one of ${pyTuple(REDUCE)}`;
  if (rule === "each" && reduce !== "each") return `${where}.reduce is ${show(reduce)}; every profile series reduces 'each'`;
  if (rule === "one" && reduce === "each") return `${where}.reduce is 'each'; an xy series reduces to one value`;
  if ("minus" in s) {
    const m = s["minus"];
    const mp = closed(m, ["target", "reduce"], ["target", "reduce"], `${where}.minus`);
    if (mp !== null) return mp;
    const mo = m as Record<string, unknown>;
    const mtp = targetProblem(mo["target"], `${where}.minus.target`);
    if (mtp !== null) return mtp;
    if (!REDUCE.slice(1).includes(mo["reduce"] as string)) return `${where}.minus.reduce ${show(mo["reduce"])} must reduce to one value`;
  }
  if ("scale" in s && !finiteNumber(s["scale"])) return `${where}.scale ${show(s["scale"])} is not a finite number`;
  return null;
}

function overProblem(over: unknown, json: JsonRead): string | null {
  const p = closed(over, ["kind", "source", "steps", "axis"], ["kind", "source", "axis"], "over");
  if (p !== null) return p;
  const o = over as Record<string, unknown>;
  if (o["kind"] !== "stage") return `over.kind ${show(o["kind"])}: 1.0.0 sweeps a stage only`;
  if (nameProblem(o["source"]) !== null) return `over.source: ${nameProblem(o["source"])}`;
  if (!SWEEP_AXES.includes(o["axis"] as string)) return `over.axis ${show(o["axis"])} is not one of ${pyTuple(SWEEP_AXES)}`;
  if ("steps" in o) {
    const steps = o["steps"];
    const isInt = (arr: unknown[], k: number) => Number.isInteger(arr[k]) && !json.floatAt(arr, k);
    if (!Array.isArray(steps) || steps.length !== 2 || !steps.every((v, k) => isInt(steps, k) && (v as number) >= -1)) {
      return `over.steps ${show(steps)} is not [from, to] of integers >= -1`;
    }
  }
  return null;
}

function seriesListProblem(value: unknown, where: string, rule: "each" | "one" | "any"): string | null {
  if (!Array.isArray(value) || value.length === 0) return `${where} is not a non-empty list`;
  for (const [i, s] of value.entries()) {
    const p = seriesProblem(s, `${where}[${i}]`, rule);
    if (p !== null) return p;
  }
  return null;
}

/** Why `plot` breaks `apegmsh-plot/1` (naming the key), or null. */
export function plotProblem(plot: Record<string, unknown>, json: JsonRead): string | null {
  const kind = plot["kind"];
  if (typeof kind !== "string" || !Object.hasOwn(PLOT_KEYS, kind)) return `kind ${show(kind)} is not one of ${pyTuple(Object.keys(PLOT_KEYS))}`;
  const p = closed(plot, PLOT_KEYS[kind]![0], PLOT_KEYS[kind]![1], `a ${kind} plot`);
  if (p !== null) return p;
  if (kind === "profile") {
    const state = plot["state"];
    const sp = closed(state, ["kind", "source", "step", "mode"], ["kind", "source", "step", "mode"], "state");
    if (sp !== null) return sp;
    const st = state as Record<string, unknown>;
    // Python's `_is_i4`: an integer token, never a float token or a boolean.
    const intOr = (k: "step" | "mode") => (typeof st[k] === "number" && !json.floatAt(st, k) ? (st[k] as number) : Number.NaN);
    const ap = addressProblem({ kind: st["kind"] as never, source: st["source"] as string, step: intOr("step"), mode: intOr("mode") });
    // The integer check names the values as written, not the NaN that stood in for a float token.
    if (ap !== null) return `state: ${/^step .* must be 32-bit integers$/.test(ap) ? `step ${show(st["step"])} and mode ${show(st["mode"])} must be 32-bit integers` : ap}`;
    if (!PROFILE_AXES.includes(plot["axis"] as string)) return `axis ${show(plot["axis"])} is not one of ${pyTuple(PROFILE_AXES)}`;
    return seriesListProblem(plot["series"], "series", "each");
  }
  const op = overProblem(plot["over"], json);
  if (op !== null) return op;
  if (kind === "history") return seriesListProblem(plot["series"], "series", "any");
  const xp = seriesProblem(plot["x"], "x", "one");
  if (xp !== null) return xp;
  return seriesListProblem(plot["y"], "y", "one");
}

/** `[data, "", ""]` for a body that stands, else `[null, reason, detail]` (Python's `_body`, plus the view check D5 gives the app). */
export function bodyProblem(table: "plots" | "views", body: string): [Record<string, unknown> | null, string, string] {
  let json: JsonRead;
  try {
    json = strictJson(body);
  } catch (err) {
    return [null, "bad-json", err instanceof Error ? err.message : String(err)];
  }
  const data = json.value;
  if (!isObj(data)) return [null, "bad-json", `the body is a ${data === null ? "null" : Array.isArray(data) ? "list" : typeof data}, not one JSON object`];
  if (table === "plots") {
    const p = plotProblem(data, json);
    if (p !== null) return [null, "bad-plot", p];
  } else {
    // D5: the app validates a view as a shot of `apegmsh-view/1` (Python checks only that it is an object).
    const v = validateSpec({ spec: VIEW_GRAMMAR, files: { model: "workspace.h5" }, shots: [data] });
    if (!v.ok) return [null, "bad-shot", `${v.key.replace(/^shots\[0\]\.?/, "") || "shot"}: ${v.reason}`];
  }
  return [data, "", ""];
}

// ---------------------------------------------------------------------------
// Reading the file
// ---------------------------------------------------------------------------

/** One `/sets` row as stored: text cells as h5wasm decoded them, members as copies. */
export interface RawSetRow {
  name: string;
  kind: string;
  id_space: string;
  fem_hash: string;
  model_hash: string;
  ids: Int32Array;
  /** row-major (m, 3) */
  xyz: Float64Array;
}
export interface RawDefRow {
  name: string;
  grammar: string;
  body: string;
}

/** A set that stands, with its row in `/sets`. */
export interface StandingSet {
  row: number;
  name: string;
  kind: SetKind;
  fem_hash: string;
  model_hash: string;
  ids: Int32Array;
  xyz: Float64Array;
}

/** An opened workspace: its `/meta`, every table as stored (verbatim, for a write), and what stands. */
export interface WorkspaceData {
  version: string;
  revision: number;
  writer: string | null;
  saveId: string | null;
  /** the banner of a same-major newer file (ADR 0113 D7), or null; such a file is never written over */
  newer: string | null;
  raw: { sets: RawSetRow[] | null; plots: RawDefRow[] | null; views: RawDefRow[] | null };
  /** tables whose columns are malformed, by name, with the reason */
  broken: Partial<Record<Table, string>>;
  sets: StandingSet[];
  plots: DefRowIn[];
  views: DefRowIn[];
  refused: RowRefusalIn[];
}

/** A stored string as text, or "" when h5wasm could not decode it as UTF-8 (Python's `text_or_empty`). */
export function textOrEmpty(raw: unknown): string {
  return typeof raw === "string" && !raw.includes("�") ? raw : "";
}

function scalarText(n: WsNode, key: string): string | null {
  const a = n.attrs[key];
  if (a === undefined) return null;
  let v = a.value;
  if (Array.isArray(v) || ArrayBuffer.isView(v)) {
    const arr = v as ArrayLike<unknown>;
    if (arr.length !== 1) return "";
    v = arr[0];
  }
  return a.metadata.type === H5T_STRING ? textOrEmpty(v) : "";
}

/** The file-level refusal for `version` under ADR 0113's floor rule, or null (Python's `version_problem`). */
export function versionProblem(version: string): string | null {
  const m = VERSION.exec(version);
  if (m === null) return "bad-version";
  const [major, minor] = [Number(m[1]), Number(m[2])];
  const current = WORKSPACE_SCHEMA_VERSION.split(".").map(Number);
  const floor = WORKSPACE_SCHEMA_FLOOR.split(".").map(Number);
  if (major !== current[0]) return "major";
  if (minor > current[1]!) return "newer";
  if (minor < floor[1]!) return "too-old";
  return null;
}

const shapeOf = (ds: WsDataset) => ds.shape ?? [];

function stringCells(h5: WorkspaceH5, g: WsGroup, name: string, where: string): string[] {
  const ds = has(g, name) ? g.get(name) : null;
  if (!isDataset(h5, ds) || ds.metadata.type !== H5T_STRING) throw new Error(`${where} is not a string column`);
  if (shapeOf(ds).length !== 1) throw new Error(`${where} has shape (${shapeOf(ds).join(", ")}), not (K,)`);
  const v = ds.value;
  const arr = Array.isArray(v) ? v : [v];
  return arr.map((x) => (typeof x === "string" ? x : ""));
}

function numbers(ds: WsDataset): number[] {
  const v = ds.value;
  if (v instanceof BigInt64Array || v instanceof BigUint64Array) return Array.from(v, Number);
  if (ArrayBuffer.isView(v)) return Array.from(v as unknown as ArrayLike<number>);
  if (Array.isArray(v)) return v.map(Number);
  return [Number(v)];
}

function readSets(h5: WorkspaceH5, g: WsGroup): RawSetRow[] {
  const cols = Object.fromEntries(SET_TEXT.map((c) => [c, stringCells(h5, g, c, `/sets/${c}`)])) as Record<(typeof SET_TEXT)[number], string[]>;
  const k = cols.name.length;
  if (SET_TEXT.some((c) => cols[c].length !== k)) throw new Error("the string columns do not share one length");
  const ds = (c: string): WsDataset => {
    const d = has(g, c) ? g.get(c) : null;
    if (!isDataset(h5, d)) throw new Error(`/sets/${c} is missing`);
    return d;
  };
  const offsets = ds("member_offsets");
  const ids = ds("ids");
  const xyz = ds("xyz");
  const dt = (d: WsDataset) => `${d.metadata.type === H5T_FLOAT ? "f" : d.metadata.type === H5T_INTEGER ? (d.metadata.signed ? "i" : "u") : "?"}${d.metadata.size}`;
  const shp = (d: WsDataset) => `(${shapeOf(d).join(", ")}${shapeOf(d).length === 1 ? "," : ""})`;
  const signedInt = (d: WsDataset) => d.metadata.type === H5T_INTEGER && d.metadata.signed === true;
  if (!signedInt(offsets) || shapeOf(offsets).length !== 1 || shapeOf(offsets)[0] !== k + 1) throw new Error(`member_offsets is ${dt(offsets)} ${shp(offsets)}, not i4 (${k + 1},)`);
  if (!signedInt(ids) || shapeOf(ids).length !== 1) throw new Error(`ids is ${dt(ids)} ${shp(ids)}, not i4 (M,)`);
  const m = shapeOf(ids)[0]!;
  if (xyz.metadata.type !== H5T_FLOAT || shapeOf(xyz).length !== 2 || shapeOf(xyz)[0] !== m || shapeOf(xyz)[1] !== 3) throw new Error(`xyz is ${dt(xyz)} ${shp(xyz)}, not f8 (${m}, 3)`);
  const off = numbers(offsets);
  const idv = numbers(ids);
  const xv = numbers(xyz);
  if (off[0] !== 0 || off[k] !== m || off.some((o, i) => i > 0 && o < off[i - 1]!)) throw new Error(`member_offsets ${show(off)} do not run from 0 up to ${m}`);
  if (m && idv.some((t) => t < I4_MIN || t > I4_MAX)) throw new Error("ids do not fit i4");
  const rows: RawSetRow[] = [];
  for (let r = 0; r < k; r++) {
    const a = off[r]!, b = off[r + 1]!;
    rows.push({
      name: cols.name[r]!,
      kind: cols.kind[r]!,
      id_space: cols.id_space[r]!,
      fem_hash: cols.fem_hash[r]!,
      model_hash: cols.model_hash[r]!,
      ids: Int32Array.from(idv.slice(a, b)),
      xyz: Float64Array.from(xv.slice(3 * a, 3 * b)),
    });
  }
  return rows;
}

function readRows(h5: WorkspaceH5, g: WsGroup, table: Table): RawDefRow[] {
  const cols = Object.fromEntries(ROW_TEXT.map((c) => [c, stringCells(h5, g, c, `/${table}/${c}`)])) as Record<(typeof ROW_TEXT)[number], string[]>;
  const k = cols.name.length;
  if (ROW_TEXT.some((c) => cols[c].length !== k)) throw new Error("the columns do not share one length");
  return Array.from({ length: k }, (_, r) => ({ name: cols.name[r]!, grammar: cols.grammar[r]!, body: cols.body[r]! }));
}

function nameCounts(names: readonly string[]): Map<string, number> {
  const counts = new Map<string, number>();
  for (const n of names) if (nameProblem(n) === null) counts.set(n, (counts.get(n) ?? 0) + 1);
  return counts;
}

/** The sets that stand, and each set row that refuses alone (Python's `_judge_sets`). */
export function judgeSets(rows: readonly RawSetRow[], refused: RowRefusalIn[]): StandingSet[] {
  const names = rows.map((r) => textOrEmpty(r.name));
  const counts = nameCounts(names);
  const out: StandingSet[] = [];
  rows.forEach((raw, row) => {
    const name = names[row]!;
    const kind = textOrEmpty(raw.kind);
    const idSpace = textOrEmpty(raw.id_space);
    const problem = nameProblem(name);
    let reason = "", detail = "";
    if (problem !== null) [reason, detail] = ["empty-name", `name ${show(name)}: ${problem}`];
    else if ((counts.get(name) ?? 0) > 1) [reason, detail] = ["duplicate", `${counts.get(name)} sets are named ${show(name)}`];
    else if (!(KINDS as readonly string[]).includes(kind)) [reason, detail] = ["bad-param", `kind ${show(kind)} is not nodes or elements`];
    else if (idSpace !== ID_SPACE) [reason, detail] = ["bad-param", `id_space ${show(idSpace)} is not ${show(ID_SPACE)}`];
    else if (raw.ids.length === 0) [reason, detail] = ["bad-members", "it has no member"];
    else if (new Set(raw.ids).size !== raw.ids.length) [reason, detail] = ["bad-members", "a tag appears twice"];
    else if (kind === "elements" && !textOrEmpty(raw.model_hash)) [reason, detail] = ["bad-param", "an element set needs a model_hash"];
    else {
      out.push({ row, name, kind: kind as SetKind, fem_hash: textOrEmpty(raw.fem_hash), model_hash: textOrEmpty(raw.model_hash), ids: raw.ids, xyz: raw.xyz });
      return;
    }
    refused.push({ table: "sets", row, name, reason, detail });
  });
  return out;
}

/** The plot or view rows that stand, and each that refuses alone (Python's `_judge_rows`). */
export function judgeRows(table: "plots" | "views", rows: readonly RawDefRow[], refused: RowRefusalIn[]): DefRowIn[] {
  const wanted = table === "plots" ? PLOT_GRAMMAR : VIEW_GRAMMAR;
  const names = rows.map((r) => textOrEmpty(r.name));
  const counts = nameCounts(names);
  const out: DefRowIn[] = [];
  rows.forEach((raw, row) => {
    const name = names[row]!;
    const grammar = textOrEmpty(raw.grammar);
    const body = textOrEmpty(raw.body);
    const problem = nameProblem(name);
    let reason = "", detail = "";
    if (problem !== null) [reason, detail] = ["empty-name", `name ${show(name)}: ${problem}`];
    else if ((counts.get(name) ?? 0) > 1) [reason, detail] = ["duplicate", `${counts.get(name)} ${table} are named ${show(name)}`];
    else if (grammar !== wanted) [reason, detail] = ["unknown-grammar", `grammar ${show(grammar)} is not ${show(wanted)}`];
    else {
      const [data, r, d] = bodyProblem(table, body);
      if (data !== null) {
        out.push({ table, row, name, grammar, body });
        return;
      }
      [reason, detail] = [r, d];
    }
    refused.push({ table, row, name, reason, detail });
  });
  return out;
}

/**
 * Read `path` whole and close it. Throws `WorkspaceRefusal` for a file-level
 * refusal; a same-major newer file reads, with `newer` set. `label` names the
 * file in messages (the target's path when `path` is a partial).
 */
export function readWorkspaceFile(h5: WorkspaceH5, path: string, label: string = path): WorkspaceData {
  let f: WsFile;
  try {
    f = openH5(h5 as unknown as Parameters<typeof openH5<WsFile>>[0], path);
  } catch (err) {
    throw new WorkspaceRefusal("not-hdf5", `${label} is not HDF5: ${err instanceof Error ? err.message : String(err)}`);
  }
  if (Number(f.file_id) < 0) throw new WorkspaceRefusal("not-hdf5", `${label} is not HDF5 (HDF5 could not open it)`);
  try {
    const meta = has(f, "meta") ? f.get("meta") : null;
    if (!isGroup(h5, meta)) throw new WorkspaceRefusal("no-version", `${label} has no /meta`);
    const version = scalarText(meta, VERSION_KEY);
    if (version === null) throw new WorkspaceRefusal("no-version", `${label} has no /meta@${VERSION_KEY}`);
    const vp = versionProblem(version);
    if (vp !== null && vp !== "newer") {
      throw new WorkspaceRefusal(vp, `${label}: ${VERSION_KEY} is ${show(version)}; this reader opens ${WORKSPACE_SCHEMA_FLOOR} to ${WORKSPACE_SCHEMA_VERSION}`);
    }
    const rev = meta.attrs["revision"];
    const rv = rev === undefined ? [] : Array.isArray(rev.value) || ArrayBuffer.isView(rev.value) ? Array.from(rev.value as ArrayLike<number | bigint>, Number) : [Number(rev.value)];
    if (rev === undefined || rev.metadata.type !== H5T_INTEGER || rev.metadata.signed !== true || rv.length !== 1 || !(rv[0]! >= 1)) {
      throw new WorkspaceRefusal("bad-meta", `${label}: /meta@revision is not an integer >= 1`);
    }
    const raw: WorkspaceData["raw"] = { sets: null, plots: null, views: null };
    const broken: WorkspaceData["broken"] = {};
    for (const table of TABLES) {
      const g = has(f, table) ? f.get(table) : null;
      if (!isGroup(h5, g)) throw new WorkspaceRefusal("missing-table", `${label} has no /${table}`);
      try {
        if (table === "sets") raw.sets = readSets(h5, g);
        else raw[table] = readRows(h5, g, table);
      } catch (err) {
        if (err instanceof WorkspaceRefusal) throw err;
        broken[table] = err instanceof Error ? err.message : String(err);
      }
    }
    const refused: RowRefusalIn[] = [];
    const sets = raw.sets ? judgeSets(raw.sets, refused) : [];
    const plots = raw.plots ? judgeRows("plots", raw.plots, refused) : [];
    const views = raw.views ? judgeRows("views", raw.views, refused) : [];
    return {
      version,
      revision: rv[0]!,
      writer: scalarText(meta, "writer"),
      saveId: scalarText(meta, "save_id"),
      newer:
        vp === "newer"
          ? `${label} is workspace ${version}, newer than this app's ${WORKSPACE_SCHEMA_VERSION}: it is shown read-only, and nothing writes over it (tables and keys this app does not know are not shown)`
          : null,
      raw,
      broken,
      sets,
      plots,
      views,
      refused,
    };
  } finally {
    f.close();
  }
}

// ---------------------------------------------------------------------------
// The model side: tags, positions, and the per-member check
// ---------------------------------------------------------------------------

/** One kind of model item: tags with positions (NaN rows have none), and the search Python's `_Items` runs. */
export class Items {
  readonly tol: number;
  private readonly sortedTags: Float64Array;
  private readonly sortedXyz: Float64Array;
  private readonly repeated: Set<number>;
  private readonly proj: Float64Array;
  private readonly pts: Float64Array;
  private readonly reach: number;

  constructor(tags: ArrayLike<number>, xyz: ArrayLike<number>, tol: number) {
    this.tol = tol;
    const n = tags.length;
    const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => tags[a]! - tags[b]! || a - b);
    this.sortedTags = Float64Array.from(order, (i) => tags[i]!);
    this.sortedXyz = new Float64Array(3 * n);
    order.forEach((i, k) => this.sortedXyz.set([xyz[3 * i]!, xyz[3 * i + 1]!, xyz[3 * i + 2]!], 3 * k));
    this.repeated = new Set();
    for (let k = 1; k < n; k++) if (this.sortedTags[k] === this.sortedTags[k - 1]) this.repeated.add(this.sortedTags[k]!);
    const finite: number[] = [];
    for (let i = 0; i < n; i++) if (Number.isFinite(xyz[3 * i]!) && Number.isFinite(xyz[3 * i + 1]!) && Number.isFinite(xyz[3 * i + 2]!)) finite.push(i);
    const p = finite.map((i) => projected(xyz[3 * i]!, xyz[3 * i + 1]!, xyz[3 * i + 2]!));
    const byProj = Array.from({ length: finite.length }, (_, k) => k).sort((a, b) => p[a]! - p[b]! || a - b);
    this.proj = Float64Array.from(byProj, (k) => p[k]!);
    this.pts = new Float64Array(3 * finite.length);
    byProj.forEach((k, j) => {
      const i = finite[k]!;
      this.pts.set([xyz[3 * i]!, xyz[3 * i + 1]!, xyz[3 * i + 2]!], 3 * j);
    });
    let reach = 0;
    for (const v of this.proj) reach = Math.max(reach, Math.abs(v));
    this.reach = reach;
  }

  /** Each tag's position (3 per tag); NaN for a tag the model lacks, repeats or cannot place. */
  positions(ids: ArrayLike<number>): Float64Array {
    const out = new Float64Array(3 * ids.length).fill(Number.NaN);
    for (let j = 0; j < ids.length; j++) {
      const t = ids[j]!;
      if (this.repeated.has(t)) continue;
      const k = lowerBound(this.sortedTags, t);
      if (k < this.sortedTags.length && this.sortedTags[k] === t) out.set(this.sortedXyz.subarray(3 * k, 3 * k + 3), 3 * j);
    }
    return out;
  }

  /** How many placed items lie within `tol` of each stored position. */
  near(xyz: ArrayLike<number>): number[] {
    const m = xyz.length / 3;
    const counts = new Array<number>(m).fill(0);
    if (!this.proj.length) return counts;
    for (let i = 0; i < m; i++) {
      const [x, y, z] = [xyz[3 * i]!, xyz[3 * i + 1]!, xyz[3 * i + 2]!];
      if (!(Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z))) continue;
      const s = projected(x, y, z);
      const slack = this.tol * (1.0 + 1e-9) + 16.0 * EPS * (Math.abs(s) + this.reach);
      const lo = lowerBound(this.proj, s - slack);
      const hi = upperBound(this.proj, s + slack);
      for (let k = lo; k < hi; k++) if (distance(this.pts, k, x, y, z) <= this.tol) counts[i]!++;
    }
    return counts;
  }
}

/** `p @ _DIRECTION`, as numpy's matmul sums it for one row of three. */
function projected(x: number, y: number, z: number): number {
  return x * DIRECTION[0] + y * DIRECTION[1] + z * DIRECTION[2];
}

/** `np.linalg.norm(a - b)` of one row of three: the square root of the summed squares, in order. */
function norm3(dx: number, dy: number, dz: number): number {
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

function distance(pts: Float64Array, k: number, x: number, y: number, z: number): number {
  return norm3(pts[3 * k]! - x, pts[3 * k + 1]! - y, pts[3 * k + 2]! - z);
}

function lowerBound(a: Float64Array, v: number): number {
  let lo = 0, hi = a.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (a[mid]! < v) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

function upperBound(a: Float64Array, v: number): number {
  let lo = 0, hi = a.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (a[mid]! <= v) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

/** An element set on a model with no `/opensees` tag map, or one whose map cannot be read. */
export class NoTagMap extends Error {}

/** The paired `<stem>.h5`: its identity hashes now, and its items when a check needs them (read whole, then closed). */
export class ModelSide {
  readonly path: string;
  readonly femHash: string | null;
  readonly modelHash: string | null;
  readonly hasTagMap: boolean;
  private readonly h5: WorkspaceH5;
  private nodes: { ids: Float64Array; xyz: Float64Array } | null = null;
  private readonly items = new Map<SetKind, Items>();

  constructor(h5: WorkspaceH5, path: string) {
    this.h5 = h5;
    this.path = path;
    let lineage: { fem: string | null; model: string | null; tagMap: boolean };
    try {
      lineage = withFile(h5, path, (f) => {
        const meta = has(f, "meta") ? f.get("meta") : null;
        const lin = isGroup(h5, meta) && has(meta, "lineage") ? meta.get("lineage") : null;
        const text = (k: string) => (isGroup(h5, lin) ? scalarText(lin, k) || null : null);
        const os = has(f, "opensees") ? f.get("opensees") : null;
        const tagMap = isGroup(h5, os) && has(os, "element_meta") && isGroup(h5, os.get("element_meta"));
        return { fem: text("fem_hash"), model: text("model_hash"), tagMap };
      });
    } catch (err) {
      throw new WorkspaceRefusal("unpaired", `${path} is not a readable model archive: ${err instanceof Error ? err.message : String(err)}`);
    }
    this.femHash = lineage.fem;
    this.modelHash = lineage.model;
    this.hasTagMap = lineage.tagMap;
  }

  private nodeArrays(f: WsFile): { ids: Float64Array; xyz: Float64Array } {
    if (this.nodes === null) {
      const g = has(f, "nodes") ? f.get("nodes") : null;
      const ids = isGroup(this.h5, g) && has(g, "ids") ? g.get("ids") : null;
      const coords = isGroup(this.h5, g) && has(g, "coords") ? g.get("coords") : null;
      this.nodes = isDataset(this.h5, ids) && isDataset(this.h5, coords) ? { ids: Float64Array.from(numbers(ids)), xyz: Float64Array.from(numbers(coords)) } : { ids: new Float64Array(0), xyz: new Float64Array(0) };
    }
    return this.nodes;
  }

  /** `TOLERANCE` times the diagonal of the box of every finite node (0 with none). */
  static tolerance(xyz: Float64Array): number {
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    let any = false;
    for (let i = 0; i < xyz.length / 3; i++) {
      const p = [xyz[3 * i]!, xyz[3 * i + 1]!, xyz[3 * i + 2]!];
      if (!p.every(Number.isFinite)) continue;
      any = true;
      for (let k = 0; k < 3; k++) {
        lo[k] = Math.min(lo[k]!, p[k]!);
        hi[k] = Math.max(hi[k]!, p[k]!);
      }
    }
    return any ? TOLERANCE * norm3(hi[0]! - lo[0]!, hi[1]! - lo[1]!, hi[2]! - lo[2]!) : 0;
  }

  itemsOf(kind: SetKind): Items {
    const cached = this.items.get(kind);
    if (cached) return cached;
    const items = withFile(this.h5, this.path, (f) => {
      const { ids, xyz } = this.nodeArrays(f);
      const tol = ModelSide.tolerance(xyz);
      if (kind === "nodes") return new Items(ids, xyz, tol);
      if (!this.hasTagMap) throw new NoTagMap(`${this.path} has no /opensees/element_meta`);
      const { tags, centroids } = elementCentroids(this.h5, f, ids, xyz);
      return new Items(tags, centroids, tol);
    });
    this.items.set(kind, items);
    return items;
  }
}

/** Each wanted node's coordinates (3 per node), NaN where the model lacks it (Python's `_node_lookup`). */
function nodeLookup(xyz: Float64Array, index: Map<number, number>, wanted: ArrayLike<number>): Float64Array {
  const out = new Float64Array(3 * wanted.length).fill(Number.NaN);
  for (let j = 0; j < wanted.length; j++) {
    const i = index.get(wanted[j]!);
    if (i !== undefined) out.set(xyz.subarray(3 * i, 3 * i + 3), 3 * j);
  }
  return out;
}

/** The mean of `n` points (3 per point), as numpy's `mean(axis=0)` sums them: in order, then divided. */
function meanOf(p: Float64Array, n: number): [number, number, number] {
  const s = [0, 0, 0];
  for (let k = 0; k < n; k++) for (let c = 0; c < 3; c++) s[c]! += p[3 * k + c]!;
  return [s[0]! / n, s[1]! / n, s[2]! / n];
}

/** Every `/opensees/element_meta` tag and its centroid (NaN when it has no nodes; Python's `_element_centroids`). */
function elementCentroids(h5: WorkspaceH5, f: WsFile, nodeIds: Float64Array, xyz: Float64Array): { tags: Float64Array; centroids: Float64Array } {
  // First occurrence wins, as numpy's searchsorted over a stable sort of the ids.
  const index = new Map<number, number>();
  nodeIds.forEach((id, i) => {
    if (!index.has(id)) index.set(id, i);
  });
  const femCentroid = new Map<number, [number, number, number]>();
  const elements = has(f, "elements") ? f.get("elements") : null;
  if (isGroup(h5, elements)) {
    for (const alias of elements.keys()) {
      const g = elements.get(alias);
      if (!isGroup(h5, g) || !has(g, "ids") || !has(g, "connectivity")) continue;
      const ids = numbers(g.get("ids") as WsDataset);
      const conn = numbers(g.get("connectivity") as WsDataset);
      const npe = ids.length ? conn.length / ids.length : 0;
      ids.forEach((id, r) => {
        if (femCentroid.has(id)) return;
        femCentroid.set(id, meanOf(nodeLookup(xyz, index, conn.slice(r * npe, (r + 1) * npe)), npe));
      });
    }
  }
  const tags: number[] = [];
  const cents: number[] = [];
  const meta = (f.get("opensees") as WsGroup).get("element_meta") as WsGroup;
  for (const token of meta.keys()) {
    const g = meta.get(token);
    if (!isGroup(h5, g) || !has(g, "ids")) throw new NoTagMap(`/opensees/element_meta/${token} has no ids`);
    const ids = numbers(g.get("ids") as WsDataset);
    const fem = has(g, "fem_eids") ? numbers(g.get("fem_eids") as WsDataset) : ids.map(() => -1);
    if (fem.length !== ids.length) throw new NoTagMap(`/opensees/element_meta/${token}: fem_eids and ids differ in length`);
    const inline = has(g, "inline_connectivity") ? (g.get("inline_connectivity") as WsDataset).value : null;
    ids.forEach((tag, r) => {
      let c: [number, number, number] = [Number.NaN, Number.NaN, Number.NaN];
      if (fem[r]! >= 0) c = femCentroid.get(fem[r]!) ?? c;
      else if (Array.isArray(inline)) {
        const nodes = Array.from(inline[r] as ArrayLike<number | bigint>, Number);
        if (nodes.length) c = meanOf(nodeLookup(xyz, index, nodes), nodes.length);
      }
      tags.push(tag);
      cents.push(...c);
    });
  }
  return { tags: Float64Array.from(tags), centroids: Float64Array.from(cents) };
}

/** Every identity hash of the row equals the model's; a missing model hash differs (Python's `_hashes_equal`). */
function hashesEqual(s: StandingSet, model: ModelSide): boolean {
  if (model.femHash === null || s.fem_hash !== model.femHash) return false;
  if (s.kind === "elements") return model.modelHash !== null && s.model_hash === model.modelHash;
  return true;
}

/** `(matched, distance)` per member: its own tag within `tol`, and no other item (Python's `match_members`). */
export function matchMembers(items: Items, ids: ArrayLike<number>, xyz: ArrayLike<number>): { matched: boolean[]; distance: number[] } {
  const own = items.positions(ids);
  const near = items.near(xyz);
  const matched: boolean[] = [];
  const dist: number[] = [];
  for (let i = 0; i < ids.length; i++) {
    const d = norm3(xyz[3 * i]! - own[3 * i]!, xyz[3 * i + 1]! - own[3 * i + 1]!, xyz[3 * i + 2]! - own[3 * i + 2]!);
    dist.push(d);
    matched.push(d <= items.tol && near[i] === 1);
  }
  return { matched, distance: dist };
}

/** Every standing set's verdict against `model`, and the sets the model refuses (`no-tag-map`; Python's `_verdicts`). */
export function verdictsOf(sets: readonly StandingSet[], model: ModelSide): { verdicts: Map<string, SetVerdictIn>; refused: RowRefusalIn[] } {
  const verdicts = new Map<string, SetVerdictIn>();
  const refused: RowRefusalIn[] = [];
  for (const s of sets) {
    if (s.kind === "elements" && !model.hasTagMap) {
      refused.push({ table: "sets", row: s.row, name: s.name, reason: "no-tag-map", detail: `${model.path} has no /opensees zone to name element tags` });
      continue;
    }
    const n = s.ids.length;
    if (hashesEqual(s, model)) {
      verdicts.set(s.name, { verdict: "current", matched: n, total: n, members: new Array<boolean>(n).fill(true), checked: false, maxDistance: null });
      continue;
    }
    let items: Items;
    try {
      items = model.itemsOf(s.kind);
    } catch (err) {
      if (!(err instanceof NoTagMap)) throw err;
      refused.push({ table: "sets", row: s.row, name: s.name, reason: "no-tag-map", detail: err.message });
      continue;
    }
    const { matched, distance } = matchMembers(items, s.ids, s.xyz);
    const k = matched.filter(Boolean).length;
    const verdict: SetVerdictName = k === n ? "moved" : k === 0 ? "lost" : "partial";
    const worst = k ? Math.max(...distance.filter((_, i) => matched[i])) : null;
    verdicts.set(s.name, { verdict, matched: k, total: n, members: matched, checked: true, maxDistance: worst });
  }
  return { verdicts, refused };
}

/** The sets as the page shows them: each standing row with its verdict (null when the verdict refused it). */
export function setsIn(sets: readonly StandingSet[], verdicts: Map<string, SetVerdictIn>): WorkspaceSetIn[] {
  return sets.map((s) => ({ row: s.row, name: s.name, kind: s.kind, ids: Array.from(s.ids), verdict: verdicts.get(s.name) ?? null }));
}

// ---------------------------------------------------------------------------
// Writing: the new content (Python's `write_workspace`, before the disk steps)
// ---------------------------------------------------------------------------

/** A write refused before anything was replaced; the old file is whole. */
export class WriteRefusal extends Error {
  readonly reason: string;
  readonly detail: string;
  constructor(reason: string, detail: string) {
    super(`${reason}: ${detail}`);
    this.reason = reason;
    this.detail = detail;
  }
}

export interface SetDef {
  name: string;
  kind: string;
  ids: readonly unknown[];
}
export interface BodyDef {
  name: string;
  body: string;
}

/** What one write changes. `rename` keeps a row verbatim under another name (the app's; Python has no rename). */
export interface Change {
  sets?: readonly SetDef[];
  plots?: readonly BodyDef[];
  views?: readonly BodyDef[];
  remove?: Partial<Record<Table, readonly string[]>>;
  rename?: readonly { table: Table; from: string; to: string }[];
}

export interface Tables {
  sets: RawSetRow[];
  plots: RawDefRow[];
  views: RawDefRow[];
}

export interface Planned {
  revision: number;
  tables: Tables;
}

function setIds(d: SetDef): Int32Array {
  const raw = [...d.ids];
  if (raw.length === 0) throw new WriteRefusal("bad-members", `set ${show(d.name)} has no member`);
  for (const tag of raw) {
    if (typeof tag !== "number" || !Number.isInteger(tag)) throw new WriteRefusal("bad-members", `set ${show(d.name)}: ${show(tag)} is not a tag`);
    if (tag < I4_MIN || tag > I4_MAX) throw new WriteRefusal("bad-members", `set ${show(d.name)}: tag ${tag} does not fit i4`);
  }
  const ids = Int32Array.from(raw as number[]);
  if (new Set(ids).size !== ids.length) throw new WriteRefusal("bad-members", `set ${show(d.name)}: a tag appears twice`);
  return ids;
}

/** One new `/sets` row: positions and hashes from the model, refusing a tag it lacks (Python's `_set_row`). */
export function setRow(d: SetDef, model: ModelSide): RawSetRow {
  if (!(KINDS as readonly string[]).includes(d.kind)) throw new WriteRefusal("bad-param", `set ${show(d.name)}: kind ${show(d.kind)}`);
  const kind = d.kind as SetKind;
  const ids = setIds(d);
  if (model.femHash === null) throw new WriteRefusal("bad-row", `${model.path} has no fem_hash to stamp`);
  if (kind === "elements") {
    if (!model.hasTagMap) throw new WriteRefusal("no-tag-map", `${model.path} has no /opensees zone`);
    if (model.modelHash === null) throw new WriteRefusal("bad-param", `${model.path} has no model_hash to stamp`);
  }
  let items: Items;
  try {
    items = model.itemsOf(kind);
  } catch (err) {
    if (err instanceof NoTagMap) throw new WriteRefusal("no-tag-map", err.message);
    throw err;
  }
  const xyz = items.positions(ids);
  const lacking = Array.from(ids).filter((_, i) => ![xyz[3 * i]!, xyz[3 * i + 1]!, xyz[3 * i + 2]!].every(Number.isFinite));
  if (lacking.length) throw new WriteRefusal("bad-row", `set ${show(d.name)}: the model has no single placed ${kind.slice(0, -1)} ${show(lacking.slice(0, 10))}`);
  return { name: d.name, kind, id_space: ID_SPACE, fem_hash: model.femHash, model_hash: kind === "elements" ? (model.modelHash ?? "") : "", ids, xyz };
}

function bodyRow(table: "plots" | "views", d: BodyDef): RawDefRow {
  const [data, reason, detail] = bodyProblem(table, d.body);
  if (data === null) throw new WriteRefusal(reason, `${table} ${show(d.name)}: ${detail}`);
  return { name: d.name, grammar: table === "plots" ? PLOT_GRAMMAR : VIEW_GRAMMAR, body: d.body };
}

function given(table: Table, rows: readonly { name: string }[]): string[] {
  const names: string[] = [];
  for (const r of rows) {
    const p = nameProblem(r.name);
    if (p !== null) throw new WriteRefusal("empty-name", `${table} name ${show(r.name)}: ${p}`);
    if (names.includes(r.name)) throw new WriteRefusal("duplicate", `${table} ${show(r.name)} is given twice`);
    names.push(r.name);
  }
  return names;
}

/** Old rows in place, verbatim, unless replaced or removed; new names after them (Python's `_merge`). */
function merge<R extends { name: string }>(old: readonly R[], fresh: Map<string, R>, removing: ReadonlySet<string>): R[] {
  const out: R[] = [];
  const placed = new Set<string>();
  for (const raw of old) {
    const name = textOrEmpty(raw.name);
    if (removing.has(name)) continue;
    if (fresh.has(name)) {
      if (!placed.has(name)) {
        out.push(fresh.get(name)!);
        placed.add(name);
      }
      continue;
    }
    out.push(raw);
  }
  for (const [name, row] of fresh) if (!placed.has(name)) out.push(row);
  return out;
}

/** The old `/sets` rows, verbatim, but a `moved` set takes the model's hashes (Python's `_restamp_moved`). */
function restampMoved(previous: WorkspaceData, model: ModelSide): RawSetRow[] {
  const rows = previous.raw.sets!.map((r) => ({ ...r }));
  const { verdicts } = verdictsOf(previous.sets, model);
  for (const s of previous.sets) {
    if (verdicts.get(s.name)?.verdict !== "moved") continue;
    rows[s.row]!.fem_hash = model.femHash ?? "";
    if (s.kind === "elements") rows[s.row]!.model_hash = model.modelHash ?? "";
  }
  return rows;
}

/**
 * The content of the next file: the old rows verbatim (refused rows included),
 * the given rows in place of the rows of the same name or after them, the
 * removals, the renames, and `moved` sets restamped (Python's `write_workspace`
 * up to the partial). `previous` is null for no file or an unreadable one.
 */
export function buildWrite(previous: WorkspaceData | null, model: ModelSide, change: Change, target: string): Planned {
  const removing: Record<Table, Set<string>> = { sets: new Set(), plots: new Set(), views: new Set() };
  for (const [table, names] of Object.entries(change.remove ?? {})) {
    if (!(TABLES as readonly string[]).includes(table)) throw new WriteRefusal("bad-param", `remove names an unknown table ${show(table)}`);
    for (const n of names ?? []) removing[table as Table].add(n);
  }
  const giv: Record<Table, string[]> = { sets: given("sets", change.sets ?? []), plots: given("plots", change.plots ?? []), views: given("views", change.views ?? []) };
  for (const table of TABLES) {
    const clash = [...removing[table]].filter((n) => giv[table].includes(n));
    if (clash.length) throw new WriteRefusal("bad-param", `${table} ${show(sorted(clash))} written and removed`);
  }
  if (previous !== null) {
    if (previous.newer !== null) throw new WriteRefusal("newer", `${target} is workspace ${previous.version}, newer than this app's ${WORKSPACE_SCHEMA_VERSION}: writing would drop what this app cannot read; nothing was replaced`);
    const [table, why] = Object.entries(previous.broken)[0] ?? [];
    if (table !== undefined) throw new WriteRefusal("bad-table", `${target}: /${table} cannot be copied verbatim (${why}); nothing was replaced`);
    for (const t of TABLES) {
      const rows = previous.raw[t]!;
      for (const r of rows) {
        for (const v of Object.values(r)) {
          if (typeof v === "string" && v.includes("�")) throw new WriteRefusal("bad-table", `${target}: /${t} holds text this app cannot copy verbatim (it is not UTF-8); nothing was replaced`);
        }
      }
    }
  }
  const fresh = {
    sets: new Map((change.sets ?? []).map((d) => [d.name, setRow(d, model)] as const)),
    plots: new Map((change.plots ?? []).map((d) => [d.name, bodyRow("plots", d)] as const)),
    views: new Map((change.views ?? []).map((d) => [d.name, bodyRow("views", d)] as const)),
  };
  const old: Tables = previous !== null ? { sets: restampMoved(previous, model), plots: [...previous.raw.plots!], views: [...previous.raw.views!] } : { sets: [], plots: [], views: [] };
  for (const r of change.rename ?? []) {
    if (!(TABLES as readonly string[]).includes(r.table)) throw new WriteRefusal("bad-param", `rename names an unknown table ${show(r.table)}`);
    const p = nameProblem(r.to);
    if (p !== null) throw new WriteRefusal("empty-name", `${r.table} name ${show(r.to)}: ${p}`);
    const rows = old[r.table] as { name: string }[];
    const at = rows.flatMap((x, i) => (textOrEmpty(x.name) === r.from ? [i] : []));
    if (at.length === 0) throw new WriteRefusal("bad-param", `no ${r.table} named ${show(r.from)} to rename`);
    if (at.length > 1) throw new WriteRefusal("duplicate", `${r.table} ${show(r.from)} names more than one row of ${target}; nothing was replaced`);
    if (rows.some((x) => textOrEmpty(x.name) === r.to) || giv[r.table].includes(r.to)) throw new WriteRefusal("duplicate", `${r.table} ${show(r.to)} is already a name in ${target}`);
    rows[at[0]!] = { ...rows[at[0]!]!, name: r.to };
  }
  for (const table of TABLES) {
    const held = new Map<string, number>();
    for (const r of old[table]) held.set(textOrEmpty(r.name), (held.get(textOrEmpty(r.name)) ?? 0) + 1);
    const absent = [...removing[table]].filter((n) => !held.has(n));
    if (absent.length) throw new WriteRefusal("bad-param", `no ${table} named ${show(sorted(absent))} to remove`);
    // A name two old rows share names neither: replacing or removing one would drop the other (INV-2).
    const twice = sorted(new Set([...giv[table], ...removing[table]].filter((n) => (held.get(n) ?? 0) > 1)));
    if (twice.length) throw new WriteRefusal("duplicate", `${table} ${show(twice)} name more than one row of ${target}; nothing was replaced`);
  }
  return {
    revision: previous === null ? 1 : previous.revision + 1,
    tables: {
      sets: merge(old.sets, fresh.sets, removing.sets),
      plots: merge(old.plots, fresh.plots, removing.plots),
      views: merge(old.views, fresh.views, removing.views),
    },
  };
}

/** `/meta` of a write. */
export interface WriteMeta {
  revision: number;
  saveId: string;
  writer: string;
}

/** Write the whole file into `partial`, created exclusively (`"x"`: another writer's file is never touched), and close it. */
export function writePartial(h5: WorkspaceH5, partial: string, meta: WriteMeta, tables: Tables): void {
  let f: WsFile;
  try {
    f = createH5(h5 as unknown as Parameters<typeof createH5<WsFile>>[0], partial);
  } catch (err) {
    throw new WriteRefusal("cannot-create", `${partial} cannot be created; nothing was replaced: ${err instanceof Error ? err.message : String(err)}`);
  }
  try {
    const m = f.create_group("meta");
    m.create_attribute(VERSION_KEY, WORKSPACE_SCHEMA_VERSION, null, "S");
    m.create_attribute("revision", meta.revision, null, "<i4");
    m.create_attribute("writer", meta.writer, null, "S");
    m.create_attribute("save_id", meta.saveId, null, "S");
    const sets = f.create_group("sets");
    const rows = tables.sets;
    for (const c of SET_TEXT) sets.create_dataset({ name: c, data: rows.map((r) => r[c]), shape: [rows.length], dtype: "S" });
    const offsets = [0];
    for (const r of rows) offsets.push(offsets[offsets.length - 1]! + r.ids.length);
    const m_ = offsets[offsets.length - 1]!;
    const ids = new Int32Array(m_);
    const xyz = new Float64Array(3 * m_);
    rows.forEach((r, k) => {
      ids.set(r.ids, offsets[k]!);
      xyz.set(r.xyz, 3 * offsets[k]!);
    });
    sets.create_dataset({ name: "member_offsets", data: Int32Array.from(offsets), shape: [offsets.length], dtype: "<i4" });
    sets.create_dataset({ name: "ids", data: ids, shape: [m_], dtype: "<i4" });
    // "<d", not "<f8": h5wasm 0.10.3 reads the width of a float dtype from its letter, and "<f8" writes f4.
    sets.create_dataset({ name: "xyz", data: xyz, shape: [m_, 3], dtype: "<d" });
    for (const table of ["plots", "views"] as const) {
      const g = f.create_group(table);
      for (const c of ROW_TEXT) g.create_dataset({ name: c, data: tables[table].map((r) => r[c]), shape: [tables[table].length], dtype: "S" });
    }
  } finally {
    f.close();
  }
}

/** Bit for bit, so a NaN or a signed zero compares as stored. */
function sameBits(a: Float64Array, b: Float64Array): boolean {
  if (a.length !== b.length) return false;
  const x = new Uint8Array(a.buffer, a.byteOffset, a.byteLength), y = new Uint8Array(b.buffer, b.byteOffset, b.byteLength);
  return x.every((v, i) => v === y[i]);
}

/** Read the partial back: every table, and `ids` and `xyz` bit for bit (Python's `_verify`). */
export function verifyPartial(h5: WorkspaceH5, partial: string, meta: WriteMeta, tables: Tables): void {
  let back: WorkspaceData;
  try {
    back = readWorkspaceFile(h5, partial);
  } catch (err) {
    throw new WriteRefusal("partial-mismatch", `${partial} does not read: ${err instanceof Error ? err.message : String(err)}`);
  }
  if (Object.keys(back.broken).length || back.revision !== meta.revision || back.saveId !== meta.saveId || back.version !== WORKSPACE_SCHEMA_VERSION) {
    throw new WriteRefusal("partial-mismatch", `${partial}: /meta did not read back`);
  }
  for (const table of TABLES) {
    const wrote = tables[table] as unknown as Record<string, unknown>[];
    const read = back.raw[table] as unknown as Record<string, unknown>[];
    if (wrote.length !== read.length) throw new WriteRefusal("partial-mismatch", `${partial}: /${table} row count`);
    wrote.forEach((a, i) => {
      for (const [key, v] of Object.entries(a)) {
        const b = read[i]![key];
        const same = v instanceof Int32Array ? b instanceof Int32Array && v.length === b.length && v.every((x, j) => x === b[j]) : v instanceof Float64Array ? b instanceof Float64Array && sameBits(v, b) : v === b;
        if (!same) throw new WriteRefusal("partial-mismatch", `${partial}: /${table}/${key} did not read back`);
      }
    });
  }
}
