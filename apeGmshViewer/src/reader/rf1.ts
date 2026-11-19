// `rf1`: the fingerprint of a `.ladruno` (or its part set) that a `<stem>.user.h5`
// stores as `/meta@results_hash` (ADR 0130 A6; h5-schema.md "Staleness").
//
// This is the app's twin of src/apeGmsh/results/user/_fingerprint.py and must equal
// it byte for byte: that module's docstring is the encoding, and its tests hold the
// goldens this file's tests repeat (tests/results/user/test_user_fingerprint.py).
// `line(s)` hashes `s + "\n"` as UTF-8; arrays hash as their little-endian bytes.
//
//   1. line("rf1"), once.
//   2. Each part, in partition order: line("<K>=<v>") for GENERATOR, FORMAT_VERSION,
//      RUN_ID of INFO's attributes, "-" for one that is absent.
//   3. Each MODEL_STAGE[i] by i: line("MODEL_STAGE[i]|<KIND>"); then ON_NODES results
//      by name, ON_ELEMENTS results and their buckets by name; then every other child
//      of RESULTS by name (descending into groups that hold no DATA).
//   4. The stage's sample, once, after its results.
//
// A native `results.h5` (no INFO group) gives `native:<meta/lineage@results_hash>`.
//
// What Python's `str()` would print for an attribute and this file cannot print the
// same way (a float32, a compound) refuses, so the two never disagree silently: a
// refusal reads as "cannot be fingerprinted", which the staleness table shows stale.
// No Node import at the top: the renderer bundle may import types from here.

import { openH5, type H5FS } from "./h5open.ts";
import { partOf, partSiblings } from "./ladrunoParts.ts";

/** HDF5 type classes, as h5wasm reports them in `metadata.type`. */
const H5T_INTEGER = 0;
const H5T_FLOAT = 1;
const H5T_STRING = 3;

/** The h5wasm metadata of a dataset or an attribute that this file reads. */
export interface H5Meta {
  type: number;
  size: number;
  signed?: boolean;
  littleEndian?: boolean;
  vlen?: boolean;
  shape?: number[] | null;
}
export interface FpAttr {
  value: unknown;
  metadata: H5Meta;
}
export interface FpNode {
  path: string;
  attrs: Record<string, FpAttr>;
}
export interface FpGroup extends FpNode {
  keys(): string[];
  get(name: string): unknown;
}
export interface FpDataset extends FpNode {
  shape: number[] | null;
  metadata: H5Meta;
  value: unknown;
  slice(ranges: [number, number][]): unknown;
}
export interface FpFile extends FpGroup {
  close(): unknown;
}
/** What the fingerprint needs of h5wasm (`h5wasm/node`, after `await h5wasm.ready`). */
export interface FpH5 {
  File: new (filename: string, mode?: "r") => FpFile;
  readonly FS: H5FS | null;
  Group: abstract new (...args: never[]) => FpGroup;
  Dataset: abstract new (...args: never[]) => FpDataset;
}

/** No fingerprint can be computed; the message says why. Never a guessed hash. */
export class FingerprintRefused extends Error {}

const STAGE = "MODEL_STAGE[";
const SAMPLED = ["ON_NODES", "ON_ELEMENTS"] as const;

/** Python's `sorted()` order on `str`: by code point (JS `<` compares UTF-16 units). */
export function byCodePoint(a: string, b: string): number {
  const x = Array.from(a);
  const y = Array.from(b);
  for (let i = 0; i < Math.min(x.length, y.length); i++) {
    const d = x[i]!.codePointAt(0)! - y[i]!.codePointAt(0)!;
    if (d !== 0) return d;
  }
  return x.length - y.length;
}
const sortedKeys = (g: FpGroup): string[] => [...g.keys()].sort(byCodePoint);

/** A SHA-256 that takes text lines and raw bytes. */
interface Hasher {
  update(data: string | Uint8Array): unknown;
  digest(enc: "hex"): string;
}

function sha256(): Hasher {
  const crypto = process.getBuiltinModule("node:crypto");
  return crypto.createHash("sha256");
}

const UTF8 = new TextEncoder();
const line = (h: Hasher, text: string) => h.update(UTF8.encode(`${text}\n`));

/**
 * `rf1:<hex>` of a `.ladruno` or its part set, or `native:<hash>` of a native
 * `results.h5`. `path` may be any part of a set: every part is hashed, in
 * partition order, and a set whose indices are not 0..n-1 refuses. `list(dir)`
 * names a directory's files (the part discovery).
 */
export function resultsFingerprint(h5: FpH5, path: string, list: (dir: string) => readonly string[]): string {
  const parts = partFiles(path, list);
  const h = sha256();
  line(h, "rf1");
  const fs = process.getBuiltinModule("node:fs");
  for (const part of parts) {
    if (!fs.statSync(part, { throwIfNoEntry: false })?.isFile()) throw new FingerprintRefused(`no such results file: ${part}`);
    let f: FpFile;
    try {
      f = openH5(h5, part);
    } catch (err) {
      throw new FingerprintRefused(`${part} cannot be opened: ${message(err)}`);
    }
    try {
      if (!f.keys().includes("INFO")) {
        if (parts.length !== 1) throw new FingerprintRefused(`${part} has no INFO group`);
        return nativeHash(h5, f, part);
      }
      hashPart(h5, h, f, part);
    } catch (err) {
      if (err instanceof FingerprintRefused) throw err;
      throw new FingerprintRefused(`${part} cannot be fingerprinted: ${message(err)}`);
    } finally {
      f.close();
    }
  }
  return `rf1:${h.digest("hex")}`;
}

const message = (err: unknown) => (err instanceof Error ? err.message : String(err));

/** The files to hash: the path alone, or every part of its set (Python's `discover_partition_files`). */
function partFiles(path: string, list: (dir: string) => readonly string[]): string[] {
  if (partOf(path) === null) return [path];
  let sibs: { index: number; path: string }[];
  try {
    sibs = partSiblings(path, list);
  } catch (err) {
    throw new FingerprintRefused(message(err));
  }
  if (sibs.length === 0) return [path];
  const indices = sibs.map((s) => s.index);
  if (indices.some((i, k) => i !== k)) {
    throw new FingerprintRefused(`the part files of ${path} are not contiguous from 0: found indices ${indices.join(", ")}`);
  }
  return sibs.map((s) => s.path);
}

function isGroup(h5: FpH5, x: unknown): x is FpGroup {
  return x instanceof (h5.Group as unknown as abstract new () => unknown);
}
function isDataset(h5: FpH5, x: unknown): x is FpDataset {
  return x instanceof (h5.Dataset as unknown as abstract new () => unknown);
}
/** h5py's `name in group`: a link that resolves to an object (a dangling soft link does not). */
function has(h5: FpH5, g: FpGroup, name: string): boolean {
  if (!g.keys().includes(name)) return false;
  const x = g.get(name);
  return isGroup(h5, x) || isDataset(h5, x);
}
/** h5py's `group[name]`: the object, or a refusal for a link that does not resolve. */
function child(h5: FpH5, g: FpGroup, name: string): FpGroup | FpDataset {
  const x = g.get(name);
  if (isGroup(h5, x) || isDataset(h5, x)) return x;
  throw new FingerprintRefused(`${g.path}/${name} is not a group or a dataset`);
}
function group(h5: FpH5, g: FpGroup, name: string, where: string): FpGroup {
  const x = child(h5, g, name);
  if (!isGroup(h5, x)) throw new FingerprintRefused(`${where} is not a result group`);
  return x;
}

/** Python's `repr` of a float64 (what `str(np.float64(x))` prints). */
export function pyFloat(x: number): string {
  if (Number.isNaN(x)) return "nan";
  if (!Number.isFinite(x)) return x > 0 ? "inf" : "-inf";
  if (x === 0) return Object.is(x, -0) ? "-0.0" : "0.0";
  // Shortest round-trip digits, as both languages choose them.
  const [mant, expText] = x.toExponential().split("e") as [string, string];
  const exp = Number(expText);
  const neg = mant.startsWith("-");
  const digits = mant.replace("-", "").replace(".", "");
  const sign = neg ? "-" : "";
  if (exp < -4 || exp >= 16) {
    const m = digits.length > 1 ? `${digits[0]}.${digits.slice(1)}` : digits;
    const e = Math.abs(exp).toString().padStart(2, "0");
    return `${sign}${m}e${exp < 0 ? "-" : "+"}${e}`;
  }
  if (exp < 0) return `${sign}0.${"0".repeat(-exp - 1)}${digits}`;
  const int = digits.slice(0, exp + 1).padEnd(exp + 1, "0");
  const frac = digits.slice(exp + 1);
  return `${sign}${int}.${frac === "" ? "0" : frac}`;
}

/** Python's `_one`: an attribute's first element as text, trailing NULs removed. */
function one(attr: FpAttr, where: string): string {
  const { metadata } = attr;
  let v = attr.value;
  if (ArrayBuffer.isView(v) || Array.isArray(v)) {
    const a = v as ArrayLike<unknown>;
    if (a.length === 0) return "";
    v = a[0];
  }
  if (metadata.type === H5T_STRING) {
    if (typeof v !== "string") throw new FingerprintRefused(`${where} is a string attribute h5wasm did not read as text`);
    return v.replace(/\0+$/, "");
  }
  if (metadata.type === H5T_INTEGER && (typeof v === "number" || typeof v === "bigint")) return String(v);
  if (metadata.type === H5T_FLOAT && metadata.size === 8 && typeof v === "number") return pyFloat(v);
  throw new FingerprintRefused(`${where} is an attribute of a type rf1 cannot print as Python does (class ${metadata.type}, size ${metadata.size})`);
}

/** The little-endian bytes of a numeric value h5wasm read (a typed array, or a scalar). */
function bytesOf(meta: H5Meta, value: unknown, where: string): Uint8Array {
  if (meta.type !== H5T_INTEGER && meta.type !== H5T_FLOAT) {
    throw new FingerprintRefused(`${where} is not numeric; rf1 hashes only numbers`);
  }
  if (meta.littleEndian === false) throw new FingerprintRefused(`${where} is big-endian; rf1 reads little-endian data only`);
  if (ArrayBuffer.isView(value)) return new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
  // A scalar dataset: h5wasm hands back a number or a bigint.
  const out = new ArrayBuffer(meta.size);
  const view = new DataView(out);
  if (meta.type === H5T_FLOAT && meta.size === 8 && typeof value === "number") view.setFloat64(0, value, true);
  else if (meta.type === H5T_FLOAT && meta.size === 4 && typeof value === "number") view.setFloat32(0, value, true);
  else if (meta.type === H5T_INTEGER && typeof value === "number" && meta.size <= 4) {
    if (meta.size === 1) view.setInt8(0, value);
    else if (meta.size === 2) view.setInt16(0, value, true);
    else view.setInt32(0, value, true);
  } else if (meta.type === H5T_INTEGER && meta.size === 8 && (typeof value === "bigint" || typeof value === "number")) {
    view.setBigInt64(0, BigInt(value), true);
  } else throw new FingerprintRefused(`${where} holds a scalar rf1 cannot encode`);
  return new Uint8Array(out);
}

const shapeOf = (ds: FpDataset): number[] => ds.shape ?? [];
const shapeText = (ds: FpDataset) => shapeOf(ds).join(",");

/** `f<size>` or `i<size>`; anything else refuses (Python's `_kind`). */
function kindOf(ds: FpDataset, where: string): string {
  const m = ds.metadata;
  if (m.type === H5T_FLOAT) return `f${m.size}`;
  if (m.type === H5T_INTEGER && m.signed) return `i${m.size}`;
  throw new FingerprintRefused(`${where} has a dtype rf1 does not know (class ${m.type}, size ${m.size}, signed ${String(m.signed)}); rf1 knows f and i`);
}

/** Python's `_index`: the integer after `prefix`, a trailing "]" removed. */
function indexOf(name: string, prefixLen: number, where: string): number {
  const text = name.slice(prefixLen).replace(/\]+$/, "");
  if (!/^[+-]?\d+$/.test(text)) throw new FingerprintRefused(`${where}: ${JSON.stringify(name)} has no integer index`);
  return Number(text);
}
const byIndex = (names: string[], prefixLen: number, where: string): string[] => {
  const keyed = names.sort(byCodePoint).map((n) => [indexOf(n, prefixLen, where), n] as const);
  return keyed.sort((a, b) => a[0] - b[0]).map(([, n]) => n);
};

function lastSlab(ds: FpDataset, where: string): Uint8Array | null {
  const shape = shapeOf(ds);
  if (shape.length === 0) throw new FingerprintRefused(`${where} is a scalar: it has no time axis`);
  const t = shape[0]!;
  if (t === 0) return null;
  return bytesOf(ds.metadata, ds.slice([[t - 1, t]]), where);
}

function nativeHash(h5: FpH5, f: FpFile, path: string): string {
  if (has(h5, f, "meta")) {
    const meta = child(h5, f, "meta");
    if (isGroup(h5, meta) && has(h5, meta, "lineage")) {
      const lineage = child(h5, meta, "lineage");
      const attr = lineage.attrs["results_hash"];
      if (attr !== undefined) {
        const text = one(attr, `${path}:/meta/lineage@results_hash`);
        if (text !== "") return `native:${text}`;
      }
    }
  }
  throw new FingerprintRefused(`${path} is neither a recorder results file (no INFO group) nor a native results file with /meta/lineage@results_hash`);
}

function hashPart(h5: FpH5, h: Hasher, f: FpFile, path: string): void {
  const info = child(h5, f, "INFO");
  for (const key of ["GENERATOR", "FORMAT_VERSION", "RUN_ID"]) {
    const attr = info.attrs[key];
    line(h, `${key}=${attr === undefined ? "-" : one(attr, `${path}:/INFO@${key}`)}`);
  }
  const stages = f.keys().filter((k) => k.startsWith(STAGE));
  for (const stage of byIndex(stages, STAGE.length, path)) {
    const g = child(h5, f, stage);
    const where = `${path}:${stage}`;
    const kind = g.attrs["KIND"];
    if (kind === undefined) throw new FingerprintRefused(`${where} has no KIND`);
    line(h, `${stage}|${one(kind, `${where}@KIND`)}`);
    if (isGroup(h5, g) && has(h5, g, "RESULTS")) {
      const results = child(h5, g, "RESULTS");
      if (!isGroup(h5, results)) throw new FingerprintRefused(`${where}/RESULTS is not a group`);
      hashStageResults(h5, h, results, where);
    }
  }
}

function hashStageResults(h5: FpH5, h: Hasher, results: FpGroup, where: string): void {
  let sample: Uint8Array | null = null;
  for (const family of SAMPLED) {
    if (!has(h5, results, family)) continue;
    const fam = group(h5, results, family, `${where}/${family}`);
    for (const name of sortedKeys(fam)) {
      const result = child(h5, fam, name);
      let holders: [string, FpGroup | FpDataset][];
      if (family === "ON_NODES") holders = [[`${family}/${name}`, result]];
      else {
        if (!isGroup(h5, result)) throw new FingerprintRefused(`${where}/${family}/${name} is not a result group`);
        holders = sortedKeys(result).map((b) => [`${family}/${name}/${b}`, child(h5, result, b)]);
      }
      for (const [path, holder] of holders) {
        const here = `${where}/${path}`;
        if (!isGroup(h5, holder)) throw new FingerprintRefused(`${here} is not a result group`);
        if (!has(h5, holder, "DATA")) continue;
        const data = child(h5, holder, "DATA");
        // Only the sample's slab is read (Python's comment: reading every result's
        // last slab would cost what the fingerprint exists to avoid).
        const want = sample === null;
        let found: Uint8Array | null;
        if (isDataset(h5, data)) found = hashDataset(h5, h, holder, data, path, here, want);
        else {
          if (family !== "ON_NODES") throw new FingerprintRefused(`${here}/DATA is a group on an element bucket`);
          found = hashModes(h5, h, data, path, here, want);
        }
        if (want) sample = found;
      }
    }
  }
  hashIndex(h5, h, results, "", where, SAMPLED);
  if (sample !== null) {
    h.update(sample);
    return;
  }
  const other = otherSample(h5, results, SAMPLED);
  if (other !== null) h.update(other);
}

/** The sample of a stage with no nodal or element slab: the first result by name that has values. */
function otherSample(h5: FpH5, g: FpGroup, skip: readonly string[] = []): Uint8Array | null {
  if (has(h5, g, "DATA")) {
    const data = child(h5, g, "DATA");
    if (isDataset(h5, data)) return lastSlab(data, `${g.path}/DATA`);
    let last: FpDataset | null = null;
    for (const step of byIndex(data.keys(), "STEP_".length, g.path)) {
      const s = group(h5, data, step, `${data.path}/${step}`);
      for (const mode of byIndex(s.keys(), "MODE_".length, g.path)) last = modeDataset(h5, s, mode);
    }
    return last === null ? null : bytesOf(last.metadata, last.value, last.path);
  }
  const names = sortedKeys(g).filter((n) => !skip.includes(n));
  const held = names.map((n) => child(h5, g, n)).filter((x): x is FpDataset => isDataset(h5, x));
  if (held.length > 0) {
    // An envelope: no time axis, so every statistic it holds.
    const parts = held.map((ds) => bytesOf(ds.metadata, ds.value, ds.path));
    const total = parts.reduce((s, p) => s + p.length, 0);
    if (total === 0) return null;
    const out = new Uint8Array(total);
    let at = 0;
    for (const p of parts) {
      out.set(p, at);
      at += p.length;
    }
    return out;
  }
  for (const name of names) {
    const x = child(h5, g, name);
    if (!isGroup(h5, x)) continue;
    const found = otherSample(h5, x);
    if (found !== null) return found;
  }
  return null;
}

/** The index of everything the sample does not come from, with no values. */
function hashIndex(h5: FpH5, h: Hasher, g: FpGroup, prefix: string, where: string, skip: readonly string[] = []): void {
  for (const name of sortedKeys(g)) {
    if (skip.includes(name)) continue;
    const c = child(h5, g, name);
    const path = `${prefix}${name}`;
    const here = `${where}/${path}`;
    if (isDataset(h5, c)) line(h, `${path}|${shapeText(c)}|${kindOf(c, here)}`);
    else if (!has(h5, c, "DATA")) hashIndex(h5, h, c, `${path}/`, where);
    else {
      const data = child(h5, c, "DATA");
      if (isDataset(h5, data)) hashDataset(h5, h, c, data, path, here, false);
      else hashModes(h5, h, data, path, here, false);
    }
  }
}

function hashDataset(h5: FpH5, h: Hasher, holder: FpGroup, data: FpDataset, path: string, where: string, want: boolean): Uint8Array | null {
  if (shapeOf(data).length === 0) throw new FingerprintRefused(`${where}/DATA is a scalar: it has no time axis`);
  for (const name of ["TIME", "STEP"]) {
    if (!has(h5, holder, name)) throw new FingerprintRefused(`${where} has no ${name}`);
  }
  line(h, `${path}|${shapeText(data)}|${kindOf(data, `${where}/DATA`)}`);
  for (const name of ["TIME", "STEP"]) {
    const ds = child(h5, holder, name);
    if (!isDataset(h5, ds)) throw new FingerprintRefused(`${where}/${name} is not a dataset`);
    h.update(bytesOf(ds.metadata, ds.value, `${where}/${name}`));
  }
  return want ? lastSlab(data, `${where}/DATA`) : null;
}

function modeDataset(h5: FpH5, step: FpGroup, mode: string): FpDataset {
  const ds = child(h5, step, mode);
  if (!isDataset(h5, ds)) throw new FingerprintRefused(`${step.path}/${mode} is not a dataset`);
  return ds;
}

function hashModes(h5: FpH5, h: Hasher, data: FpGroup, path: string, where: string, want: boolean): Uint8Array | null {
  let last: FpDataset | null = null;
  for (const step of byIndex(data.keys(), "STEP_".length, where)) {
    const s = group(h5, data, step, `${where}/${step}`);
    for (const mode of byIndex(s.keys(), "MODE_".length, where)) {
      const ds = modeDataset(h5, s, mode);
      const lambda = ds.attrs["LAMBDA"];
      if (lambda === undefined) throw new FingerprintRefused(`${where}/${step}/${mode} has no LAMBDA`);
      const values = lambdaValues(lambda, `${where}/${step}/${mode}@LAMBDA`);
      if (values.length !== 1) throw new FingerprintRefused(`${where}/${step}/${mode}@LAMBDA holds ${values.length} values`);
      line(h, `${path}/${step}/${mode}|${shapeText(ds)}`);
      h.update(new Uint8Array(Float64Array.of(values[0]!).buffer));
      last = ds;
    }
  }
  return want && last !== null ? bytesOf(last.metadata, last.value, last.path) : null;
}

/** `np.asarray(attr, dtype="<f8").ravel()` of a numeric attribute. */
function lambdaValues(attr: FpAttr, where: string): number[] {
  const { metadata, value } = attr;
  if (metadata.type !== H5T_FLOAT && metadata.type !== H5T_INTEGER) throw new FingerprintRefused(`${where} is not numeric`);
  if (ArrayBuffer.isView(value)) return Array.from(value as unknown as ArrayLike<number | bigint>, Number);
  if (Array.isArray(value)) return value.map(Number);
  if (typeof value === "number" || typeof value === "bigint") return [Number(value)];
  throw new FingerprintRefused(`${where} cannot be read as numbers`);
}
