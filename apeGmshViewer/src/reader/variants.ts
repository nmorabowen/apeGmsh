// `<study>.variants.json`, the variants manifest (ADR 0132 D8, D9; the layout
// is architecture/h5-schema.md, section "`<study>.variants.json`", and its
// JSON Schema architecture/schemas/variants.schema.json). The app reads it and
// never writes it.
//
// Main reads a manifest only from a path the user gave (D9.1) and keeps the
// rows that pass, each with its resolved path; the page sees a row's index,
// label and params, and names a variant by index and label, never by path
// (D9.2). This module is that read: pure apart from the file-system functions
// a caller passes in, so the rules are tested in Node on a fake disk.
//
// - The envelope: `format` "apegmsh-variants", `version` a MAJOR.MINOR.PATCH
//   string (ADR 0113 floor rule, current and floor 1.0.0) and `variants`, a
//   non-empty list. Anything else refuses the whole file.
// - A same-major newer minor opens with one banner: every key the 1.0 schema
//   does not name is dropped, at any depth, and the rest is read as 1.0. At
//   1.0, an unknown key refuses the file at the top and the row in a row.
// - A bad row refuses alone, with its code (`RowCode`), and the file opens.

import { dirname, isAbsolute, join, relative } from "node:path";

export const VARIANTS_FORMAT = "apegmsh-variants";
/** The app's version of the manifest, and its floor (ADR 0132 D2, ADR 0113). */
export const VARIANTS_TARGET = "1.0.0";
export const VARIANTS_FLOOR = "1.0.0";

/** A manifest's file name: `<study>.variants.json`, matched case-insensitively. */
export const MANIFEST_SUFFIX = /\.variants\.json$/i;

export function isManifestPath(path: string): boolean {
  return MANIFEST_SUFFIX.test(path) && path.length > ".variants.json".length && !/[\\/]\.variants\.json$/i.test(path);
}

export type VariantParam = string | number | boolean;

/** A row that passed D8, with the path main opens (`resolved`). The page never receives `resolved`. */
export interface VariantRow {
  /** the row's position in `variants`, 0-based: the page names a row by it and its label */
  index: number;
  label: string;
  /** as the manifest writes it, relative to the manifest's folder */
  path: string;
  /** the model file, joined to the manifest's folder and confined to it (after links resolve) */
  resolved: string;
  params: Record<string, VariantParam>;
}

/**
 * The row refusals of h5-schema.md, plus `bad-row` for a row that is not an
 * object of `{label, path, params}` (a missing key, or at 1.0 a key the schema
 * does not name): the section lists no code for that shape, and the schema
 * refuses such a row.
 */
export type RowCode = "empty-name" | "duplicate" | "bad-path" | "escapes" | "missing" | "duplicate-path" | "bad-param" | "bad-row";

export interface RowRefusal {
  index: number;
  /** the row's label when it has a usable one */
  label: string | null;
  code: RowCode;
  reason: string;
}

export type FileCode = "unreadable" | "not-json" | "not-a-manifest" | "bad-version" | "below-floor" | "other-major" | "malformed";

export type VariantsRead =
  | {
      ok: true;
      manifest: string;
      version: string;
      /** the banner of a same-major newer file (read-only, unknown keys dropped), or null */
      newer: string | null;
      rows: VariantRow[];
      refused: RowRefusal[];
    }
  | { ok: false; manifest: string; code: FileCode; reason: string };

export interface VariantsFs {
  readFile(path: string): string;
  /** whether an existing path is a regular file; false when it does not exist */
  isFile(path: string): boolean;
  /** the path with every link and junction resolved (fs.realpathSync.native) */
  realpath(path: string): string;
  /** whether paths compare case-folded (Windows, macOS) */
  fold: boolean;
}

const VERSION = /^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$/;
const TOP_KEYS = ["format", "version", "variants"] as const;
const ROW_KEYS = ["label", "path", "params"] as const;
/** The largest finite double: a number past it overflowed to infinity (`1e400`). */
const MAX = 1.7976931348623157e308;

const isObject = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null && !Array.isArray(v);
const show = (v: unknown): string => (v === undefined ? "nothing" : JSON.stringify(v) ?? String(v));

/** ADR 0130 A2's name rule: non-empty, no control character, no leading or trailing whitespace, NFC. Null when it passes. */
export function nameProblem(v: unknown): string | null {
  if (typeof v !== "string") return `must be a string; got ${show(v)}`;
  if (v.length === 0) return "is empty";
  if (/[\u0000-\u001f\u007f-\u009f]/.test(v)) return `holds a control character: ${show(v)}`;
  if (/^\s|\s$/.test(v)) return `starts or ends with whitespace: ${show(v)}`;
  if (v.normalize("NFC") !== v) return `is not in Unicode NFC: ${show(v)}`;
  return null;
}

/** The reserved sibling suffixes (D8): a variant's path names a model file, never one of these. */
const RESERVED = /\.(geometry|results|user|workspace)\.h5$/i;

/** Why `p` is not a D8 path form (the schema's checks), or null. */
export function pathProblem(p: unknown): string | null {
  if (typeof p !== "string") return `must be a string; got ${show(p)}`;
  if (p.length === 0) return "is empty";
  if (/[\\:\u0000-\u001f\u007f]/.test(p)) return `holds a backslash, a colon or a control character, so it could name a drive or an absolute path: ${show(p)}`;
  if (p.startsWith("/")) return `is absolute: ${show(p)}`;
  const segs = p.split("/");
  for (const [i, s] of segs.entries()) {
    if (s === "" || s === "." || s === "..") return `has an empty, "." or ".." segment, so it could leave the manifest's folder: ${show(p)}`;
    if (i < segs.length - 1 && /[. ]$/.test(s)) return `has a folder segment ending in a dot or a space (${show(s)}), which Windows strips: ${show(p)}`;
  }
  const last = segs[segs.length - 1]!;
  if (!/\.h5$/i.test(last) || last.length <= 3) return `must name a model file <stem>.h5 with a non-empty stem: ${show(p)}`;
  if (/\.ladruno$/i.test(last) || RESERVED.test(last)) return `names a sibling of a set (.geometry.h5, .results.h5, .user.h5, .workspace.h5 or .ladruno), not a model file: ${show(p)}`;
  return null;
}

function paramProblem(v: unknown): string | null {
  if (typeof v === "string" || typeof v === "boolean") return null;
  if (typeof v === "number") return Number.isFinite(v) && Math.abs(v) <= MAX ? null : `is not a finite number (it overflows to infinity): ${String(v)}`;
  return `must be a string, a finite number or a boolean; got ${show(v)}`;
}

/** Whether `inner` lies inside `outer` (both resolved): no `..` step and not absolute. */
function inside(outer: string, inner: string): boolean {
  const r = relative(outer, inner);
  return r !== "" && !r.startsWith("..") && !isAbsolute(r);
}

/** Read and check the manifest at `manifest` (an absolute path the user gave). Never throws. */
export function readVariants(manifest: string, fs: VariantsFs): VariantsRead {
  const refuse = (code: FileCode, reason: string): VariantsRead => ({ ok: false, manifest, code, reason: `${manifest}: ${reason}` });
  if (!isAbsolute(manifest)) return refuse("unreadable", "a manifest is read from an absolute path");
  let text: string;
  try {
    text = fs.readFile(manifest);
  } catch (err) {
    return refuse("unreadable", `cannot be read (${err instanceof Error ? err.message : String(err)})`);
  }
  // UTF-8; a reader strips a byte-order mark.
  if (text.charCodeAt(0) === 0xfeff) text = text.slice(1);
  let raw: unknown;
  try {
    // JSON.parse refuses NaN and Infinity tokens; an overflowing number is caught per param.
    raw = JSON.parse(text);
  } catch (err) {
    return refuse("not-json", `is not a JSON document (${err instanceof Error ? err.message : String(err)})`);
  }
  if (!isObject(raw)) return refuse("not-a-manifest", `must be a JSON object; got ${show(raw)}`);
  if (raw["format"] !== VARIANTS_FORMAT) return refuse("not-a-manifest", `format must be "${VARIANTS_FORMAT}"; got ${show(raw["format"])}`);
  const v = raw["version"];
  if (typeof v !== "string" || !VERSION.test(v)) return refuse("bad-version", `version must be a MAJOR.MINOR.PATCH string; got ${show(v)}`);
  const [major, minor] = v.split(".").map(Number) as [number, number, number];
  const [tMajor, tMinor] = VARIANTS_TARGET.split(".").map(Number) as [number, number, number];
  const [fMajor, fMinor] = VARIANTS_FLOOR.split(".").map(Number) as [number, number, number];
  if (major !== tMajor) return refuse("other-major", `version ${v} is of another major than the app's ${VARIANTS_TARGET}; it is not read`);
  if (major === fMajor && minor < fMinor) return refuse("below-floor", `version ${v} is below the floor ${VARIANTS_FLOOR}`);
  const newer = minor > tMinor;
  if (!newer) {
    const extra = Object.keys(raw).filter((k) => !(TOP_KEYS as readonly string[]).includes(k));
    if (extra.length) return refuse("malformed", `${extra.map((k) => JSON.stringify(k)).join(", ")} is not a key of a ${VARIANTS_TARGET} manifest; its keys are ${TOP_KEYS.join(", ")}`);
  }
  const list = raw["variants"];
  if (!Array.isArray(list) || list.length === 0) return refuse("malformed", `variants must be a non-empty list of rows; got ${show(list)}`);
  const folder = dirname(manifest);
  let realFolder: string;
  try {
    realFolder = fs.realpath(folder);
  } catch (err) {
    return refuse("unreadable", `its folder cannot be resolved (${err instanceof Error ? err.message : String(err)})`);
  }

  const refused: RowRefusal[] = [];
  type Pending = Omit<VariantRow, "resolved"> & { resolved: string; key: string };
  const pending: Pending[] = [];
  for (const [index, row] of list.entries()) {
    const no = (code: RowCode, reason: string, label: string | null = null) => refused.push({ index, label, code, reason: `variants[${index}]: ${reason}` });
    if (!isObject(row)) {
      no("bad-row", `must be an object {label, path, params}; got ${show(row)}`);
      continue;
    }
    const label = nameProblem(row["label"]) === null ? (row["label"] as string) : null;
    if (!newer) {
      const extra = Object.keys(row).filter((k) => !(ROW_KEYS as readonly string[]).includes(k));
      if (extra.length) {
        no("bad-row", `${extra.map((k) => JSON.stringify(k)).join(", ")} is not a key of a ${VARIANTS_TARGET} row; its keys are ${ROW_KEYS.join(", ")}`, label);
        continue;
      }
    }
    const lp = nameProblem(row["label"]);
    if (lp !== null) {
      no("empty-name", `label ${lp}`);
      continue;
    }
    const pp = pathProblem(row["path"]);
    if (pp !== null) {
      no("bad-path", `path ${pp}`, label);
      continue;
    }
    const params = row["params"];
    if (!isObject(params)) {
      no("bad-param", `params must be an object of names to values; got ${show(params)}`, label);
      continue;
    }
    let bad: { code: RowCode; reason: string } | null = null;
    for (const [k, x] of Object.entries(params)) {
      const kp = nameProblem(k);
      if (kp !== null) bad ??= { code: "empty-name", reason: `a params key ${kp}` };
      const xp = paramProblem(x);
      if (xp !== null) bad ??= { code: "bad-param", reason: `params ${JSON.stringify(k)} ${xp}` };
    }
    if (bad) {
      no(bad.code, bad.reason, label);
      continue;
    }
    const rel = row["path"] as string;
    const joined = join(folder, ...rel.split("/"));
    if (!fs.isFile(joined)) {
      no("missing", `path ${JSON.stringify(rel)} names no file (${joined})`, label);
      continue;
    }
    let real: string;
    try {
      real = fs.realpath(joined);
    } catch (err) {
      no("missing", `path ${JSON.stringify(rel)} cannot be resolved (${err instanceof Error ? err.message : String(err)})`, label);
      continue;
    }
    if (!inside(realFolder, real)) {
      no("escapes", `path ${JSON.stringify(rel)} resolves to ${real}, outside the manifest's folder ${realFolder}`, label);
      continue;
    }
    pending.push({ index, label: label!, path: rel, resolved: joined, params: { ...(params as Record<string, VariantParam>) }, key: fs.fold ? real.toLowerCase() : real });
  }
  // A label or a path two rows share refuses both rows.
  const twice = (key: (p: Pending) => string) => {
    const n = new Map<string, number>();
    for (const p of pending) n.set(key(p), (n.get(key(p)) ?? 0) + 1);
    return (p: Pending) => (n.get(key(p)) ?? 0) > 1;
  };
  const dupLabel = twice((p) => p.label);
  const dupPath = twice((p) => p.key);
  const rows: VariantRow[] = [];
  for (const p of pending) {
    if (dupLabel(p)) refused.push({ index: p.index, label: p.label, code: "duplicate", reason: `variants[${p.index}]: label ${JSON.stringify(p.label)} names another row too` });
    else if (dupPath(p)) refused.push({ index: p.index, label: p.label, code: "duplicate-path", reason: `variants[${p.index}]: path ${JSON.stringify(p.path)} names the same file as another row` });
    else rows.push({ index: p.index, label: p.label, path: p.path, resolved: p.resolved, params: p.params });
  }
  refused.sort((a, b) => a.index - b.index);
  return {
    ok: true,
    manifest,
    version: v,
    newer: newer ? `${manifest} is version ${v}, newer than this app's ${VARIANTS_TARGET}: it is read as ${VARIANTS_TARGET}, and the keys ${VARIANTS_TARGET} does not name are not shown` : null,
    rows,
    refused,
  };
}

/** The row `index` when its label is `label`, else why not (D9.2: main refuses a request whose label does not match its row). */
export function rowOf(read: VariantsRead | null, index: unknown, label: unknown): { ok: true; row: VariantRow } | { ok: false; reason: string } {
  if (read === null || !read.ok) return { ok: false, reason: "no variants manifest is open" };
  if (typeof index !== "number" || !Number.isInteger(index) || typeof label !== "string") return { ok: false, reason: `a variant is named by its row index and label; got ${show(index)}, ${show(label)}` };
  const row = read.rows.find((r) => r.index === index);
  if (!row) {
    const r = read.refused.find((x) => x.index === index);
    return { ok: false, reason: r ? `variant row ${index} is refused (${r.code}: ${r.reason})` : `the manifest has no variant row ${index}` };
  }
  if (row.label !== label) return { ok: false, reason: `variant row ${index} is ${JSON.stringify(row.label)}, not ${JSON.stringify(label)}` };
  return { ok: true, row };
}
