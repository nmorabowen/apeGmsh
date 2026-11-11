// Read a `<stem>.user.h5` (ADR 0130 Amendment 1; h5-schema.md "`<stem>.user.h5`").
//
// The app's twin of src/apeGmsh/results/user/_reader.py and _staleness.py, for
// the zones this slice reads: `/meta`, `/combinations` and the index of
// `/results`. Three levels of refusal, as Python's: a bad `/meta` refuses the
// file (`UserFileRefused`); a malformed zone, or one outside its version range,
// refuses that zone and the others still read (`ZoneRead.ok === false`); a bad
// row refuses alone, with a reason (`refused`).
//
// What the index never does: read the `DATA` of a `/results` entry or of a
// `/spectra` case (only its shape). `readEntityValues` reads one `ON_ENTITIES`
// entry's `DATA` on request, bound to its rows. `/modes` is read whole
// (reader/userModes.ts judges it), and each row gets A6's lambda verdict
// against the paired `.ladruno` (`Staleness.modes`). `/spectra` is read as an
// index of its cases (reader/userSpectra.ts judges each): a bad case refuses
// alone, as in Python's `read_spectra`. `/entities` is read whole
// (reader/userEntities.ts judges its rows): a bad row refuses alone and keeps
// its position, and an `ON_ENTITIES` user result naming a row the zone does not
// hold or refuses is refused alone, `unresolved` (Python's `_bound_entries`).
// An element entry's `GP_PARAM` frame is not checked against its topology
// (Python's `_check_frame`); its placement attributes and shapes are.
//
// The staleness table is A6's, as `staleness()` in _staleness.py: whatever cannot
// be compared is stale, so nothing is drawn as current on no proof.

import { openH5 } from "./h5open.ts";
import { checkZoneVersion, COMBINATIONS_TARGET, ENTITIES_TARGET, MODES_TARGET, OPENSEES_TARGET, SchemaError, SPECTRA_TARGET, USER_META_TARGET, USER_RESULTS_TARGET, ZONE_FLOOR } from "./read.ts";
import { byCodePoint, FingerprintRefused, resultsFingerprint, type FpDataset, type FpFile, type FpGroup, type FpH5, type H5Meta } from "./rf1.ts";
import { LambdasRefused, modeTableProblem, recordedLambdas, rowVerdicts, type ModeRow, type ModeTable, type RowVerdict } from "./userModes.ts";
import { caseOf, caseProblem, npDtype, pyList, pyShape, pyStr, SPECTRA_FAMILIES, type CaseDraft, type SpectrumCase, type SpectrumQuantity, type SpectrumTarget } from "./userSpectra.ts";
import { addressProblem, nameProblem, resolve, type Combination, type Refusal, type StateAddress, type Term } from "./userRules.ts";
import { bindingProblem, ELEMENT_INDEX_SPACE, entityRow, isEntityKind, noEntities, rowRefusals, type CutEntity, type Entities, type EntityKind, type PointSet, type Storey, type Vec3 } from "./userEntities.ts";

export type { Combination, Refusal, StateAddress, Term } from "./userRules.ts";
export type { CutEntity, Entities, EntityKind, EntityRefusal, PointSet, Storey } from "./userEntities.ts";
export type { ModeRow, ModeTable, RowVerdict } from "./userModes.ts";
export type { SpectrumCase, SpectrumMode, SpectrumQuantity } from "./userSpectra.ts";
export type UserH5 = FpH5;

/** The file cannot be opened at all: its `/meta` is missing or malformed. */
export class UserFileRefused extends Error {}
/** One zone cannot be read; the file still opens. The message starts with the zone. */
class ZoneRefused extends Error {
  constructor(zone: string, detail: string) {
    super(`${zone}: ${detail}`);
  }
}
/** One `/results` entry is refused, with its reason. */
class EntryRefused extends Error {
  readonly reason: Refusal["reason"];
  readonly detail: string;
  constructor(reason: Refusal["reason"], detail: string) {
    super(`${reason}: ${detail}`);
    this.reason = reason;
    this.detail = detail;
  }
}

/** A zone that read, or why it did not. */
export type ZoneRead<T> = { ok: true; value: T } | { ok: false; reason: string };

/** The combinations in force across both files, and each refused row (A3). */
export interface CombinationTable {
  /** the `/combinations` generation read (`@current`), null when the file holds no zone */
  generation: number | null;
  /** VU9b: every `g<N>` the zone holds, ascending ([] with no zone): the authoring panel's revert reads it */
  generations: number[];
  /** the `.user.h5` rows that stand */
  rows: Combination[];
  /** the model.h5 `/opensees/combinations` rows that stand */
  modelRows: Combination[];
  refused: Refusal[];
}

/** A `/results` entry, judged without reading its `DATA`. */
export interface ResultEntry {
  name: string;
  target: "nodes" | "elements" | "entities" | "domain";
  basis: "results" | "model";
  perState: boolean;
  linear: boolean;
  /** the `r<k>` link under `/results` */
  link: string;
  /** one per row of `DATA` axis 0; empty for a state-free entry */
  states: StateAddress[];
  /**
   * VU16: the `@COMPONENTS` of a nodes, entities or domain entry, in `DATA`
   * column order; [] for an elements entry, whose columns its buckets'
   * `COLUMN_MAP` names.
   */
  components: string[];
}
export interface ResultIndex {
  entries: ResultEntry[];
  refused: Refusal[];
}

/** One `.user.h5`, as the app reads it on request (never on the first frame). */
export interface UserIndex {
  path: string;
  femHash: string;
  modelHash: string | null;
  resultsHash: string | null;
  writer: string | null;
  units: string | null;
  revision: number;
  /** every root member other than `/meta`, by name: `combinations`, `results`, `modes`, ... */
  zones: string[];
  /** a zone newer than this app: it opens, and the banner says so */
  warnings: string[];
  combinations: ZoneRead<CombinationTable>;
  results: ZoneRead<ResultIndex>;
  /** `/modes`, the modal table, or null when the file holds none */
  modes: ZoneRead<ModeTable> | null;
  /** `/spectra`, its cases (every value but DATA), or null when the file holds none */
  spectra: ZoneRead<SpectraIndex> | null;
  /** `/entities`, its cuts, storeys and point sets, or null when the file holds none */
  entities: ZoneRead<Entities> | null;
}

/** The `/spectra` cases that stand, and each refused one (Python's `SpectraIndex`). */
export interface SpectraIndex {
  cases: SpectrumCase[];
  refused: Refusal[];
}

/** Whether one zone or entry is stale, and why ("" when current). */
export interface ZoneState {
  stale: boolean;
  reason: string;
}
/** A6 over the file and its set (h5-schema.md "Staleness"). */
export interface Staleness {
  /** by root name: `meta`, and each present zone of `combinations`, `results`, `modes`, `spectra`, `entities` */
  zones: Record<string, ZoneState>;
  /** by user-result name */
  results: Record<string, ZoneState>;
  /** the standing combinations with a stale source, directly or through another combination */
  combinations: string[];
  /** A6's verdict on each `/modes` row, in table order; [] when the table does not read */
  modes: RowVerdict[];
}

/** What `user:read` answers: the file and its staleness against the set, or why it cannot be read. */
export type UserRead = { ok: true; user: UserIndex; staleness: Staleness } | { ok: false; error: string };

const H5T_INTEGER = 0;
const H5T_FLOAT = 1;
const H5T_STRING = 3;
const TARGETS = ["nodes", "elements", "entities", "domain"] as const;
const BASES = ["results", "model"] as const;
const ON: Record<ResultEntry["target"], string> = { nodes: "ON_NODES", elements: "ON_ELEMENTS", entities: "ON_ENTITIES", domain: "ON_DOMAIN" };
const ENTITY_KINDS = ["cut", "storey", "point_set"];
const MODEL_ZONES = ["modes", "spectra", "entities"] as const;
/** A term's data comes from these zones; a `stage` or `mode` term reads the `.ladruno`, current by definition. */
const TERM_ZONE: Record<string, string> = { spectrum: "spectra", reference: "spectra" };

const message = (err: unknown) => (err instanceof Error ? err.message : String(err));

// ---------------------------------------------------------------------------
// h5wasm access, with h5py's meanings of `in` and `[]`
// ---------------------------------------------------------------------------

class Tree {
  private readonly h5: UserH5;
  constructor(h5: UserH5) {
    this.h5 = h5;
  }
  isGroup(x: unknown): x is FpGroup {
    return x instanceof (this.h5.Group as unknown as abstract new () => unknown);
  }
  isDataset(x: unknown): x is FpDataset {
    return x instanceof (this.h5.Dataset as unknown as abstract new () => unknown);
  }
  /** h5py's `name in group`: a link that resolves (a path may have several parts). */
  has(g: FpGroup, path: string): boolean {
    let at: FpGroup = g;
    const parts = path.split("/");
    for (let i = 0; i < parts.length; i++) {
      if (!at.keys().includes(parts[i]!)) return false;
      const x = at.get(parts[i]!);
      if (i === parts.length - 1) return this.isGroup(x) || this.isDataset(x);
      if (!this.isGroup(x)) return false;
      at = x;
    }
    return false;
  }
  group(g: FpGroup, path: string): FpGroup | null {
    if (!this.has(g, path)) return null;
    const x = g.get(path);
    return this.isGroup(x) ? x : null;
  }
  dataset(g: FpGroup, path: string): FpDataset | null {
    if (!this.has(g, path)) return null;
    const x = g.get(path);
    return this.isDataset(x) ? x : null;
  }
}

/** A stored text attribute (Python's `_text_attr`): null when absent, "" when it is not one string. */
function textAttr(n: { attrs: Record<string, { value: unknown; metadata: H5Meta }> }, key: string): string | null {
  const a = n.attrs[key];
  if (a === undefined) return null;
  let v = a.value;
  if (Array.isArray(v) || ArrayBuffer.isView(v)) {
    const arr = v as ArrayLike<unknown>;
    if (arr.length !== 1) return "";
    v = arr[0];
  }
  return a.metadata.type === H5T_STRING && typeof v === "string" ? v : "";
}

/** One integer attribute (signed, or `iu` when `unsigned`), or null when it is not one. */
function intAttr(n: { attrs: Record<string, { value: unknown; metadata: H5Meta }> }, key: string, unsigned: boolean): number | null {
  const a = n.attrs[key];
  if (a === undefined || a.metadata.type !== H5T_INTEGER || (!a.metadata.signed && !unsigned)) return null;
  let v = a.value;
  if (Array.isArray(v) || ArrayBuffer.isView(v)) {
    const arr = v as ArrayLike<unknown>;
    if (arr.length !== 1) return null;
    v = arr[0];
  }
  return typeof v === "number" || typeof v === "bigint" ? Number(v) : null;
}

const shape = (ds: FpDataset): number[] => ds.shape ?? [];
const sameShape = (a: readonly number[], b: readonly number[]) => a.length === b.length && a.every((x, i) => x === b[i]);

/** Whether a dataset is `code` ("str", "f8", "i4"); Python compares `(kind, itemsize)`. */
function isCode(ds: FpDataset, code: string): boolean {
  const m = ds.metadata;
  if (code === "str") return m.type === H5T_STRING;
  const size = Number(code.slice(1));
  if (code[0] === "f") return m.type === H5T_FLOAT && m.size === size;
  return m.type === H5T_INTEGER && m.signed === true && m.size === size;
}
const dtypeText = (ds: FpDataset) => {
  const m = ds.metadata;
  return m.type === H5T_STRING ? "str" : `${m.type === H5T_FLOAT ? "f" : m.signed ? "i" : "u"}${m.size}`;
};

/** The values of a small column as an array (strings, or numbers). */
function values(ds: FpDataset): unknown[] {
  const v = ds.value;
  if (Array.isArray(v)) return v;
  if (ArrayBuffer.isView(v)) return Array.from(v as unknown as ArrayLike<number | bigint>, (x) => (typeof x === "bigint" ? Number(x) : x));
  return [v];
}

/** A table cell as text; a cell that is not text refuses the zone (Python's `decode_cell`). */
function cell(raw: unknown, zone: string, where: string): string {
  if (typeof raw !== "string") throw new ZoneRefused(zone, `${where} holds a cell that is not a string: ${JSON.stringify(raw)}`);
  return raw;
}

// ---------------------------------------------------------------------------
// The file
// ---------------------------------------------------------------------------

/**
 * Read `path` and judge it against the set (`model`, `results`: the set's
 * model.h5 and results file, or null). Never raises: a file that cannot be
 * read answers `ok: false` with the reason.
 */
export function readUserSet(h5: UserH5, path: string, model: string | null, results: string | null, list: (dir: string) => readonly string[]): UserRead {
  try {
    const user = readUserFile(h5, path, model);
    return { ok: true, user, staleness: staleness(h5, user, model, results, list) };
  } catch (err) {
    return { ok: false, error: message(err) };
  }
}

/** The file's `/meta`, its combinations judged with `model`'s, and its `/results` index. */
export function readUserFile(h5: UserH5, path: string, model: string | null): UserIndex {
  const fs = process.getBuiltinModule("node:fs");
  if (!fs.statSync(path, { throwIfNoEntry: false })?.isFile()) throw new UserFileRefused(`no such file: ${path}`);
  let f: FpFile;
  try {
    f = openH5(h5, path);
  } catch (err) {
    throw new UserFileRefused(`${path} is not a readable HDF5 file: ${message(err)}`);
  }
  const t = new Tree(h5);
  try {
    const meta = readMeta(t, f, path);
    // Python's staleness passes the model only when its hashes read (`_readable`).
    let modelRows: ZoneRead<Combination[]> = { ok: true, value: [] };
    if (model !== null && modelHashes(h5, model).ok) modelRows = readModelCombinations(h5, model, meta.warnings);
    // Read once: the index lists it, and the /results binding checks its rows against it.
    // A stamped key with no group refuses here, as Python's `_held_entities`, and lists nothing.
    const held = zoneRead(() => readEntities(t, f, meta.warnings));
    return {
      ...meta,
      combinations: zoneRead(() => readCombinations(t, f, meta.warnings, modelRows)),
      results: zoneRead(() => readResults(t, f, meta.warnings, held)),
      modes: f.keys().includes("modes") ? zoneRead(() => readModes(t, f, meta.warnings)) : null,
      spectra: f.keys().includes("spectra") ? zoneRead(() => readSpectra(t, f, meta.warnings)) : null,
      entities: f.keys().includes("entities") ? held : null,
    };
  } finally {
    f.close();
  }
}

function zoneRead<T>(read: () => T): ZoneRead<T> {
  try {
    return { ok: true, value: read() };
  } catch (err) {
    if (err instanceof UserFileRefused) throw err;
    return { ok: false, reason: `refused: ${message(err)}` };
  }
}

function readMeta(t: Tree, f: FpFile, path: string): Omit<UserIndex, "combinations" | "results" | "modes" | "spectra" | "entities"> {
  const m = t.group(f, "meta");
  if (m === null) throw new UserFileRefused(`${path} has no /meta group`);
  const key = "user_meta_schema_version";
  const raw = m.attrs[key];
  if (raw === undefined) throw new UserFileRefused(`${path} has no /meta@${key}: it is not a .user.h5`);
  const warnings: string[] = [];
  try {
    checkZoneVersion(textAttr(m, key) as string, key, USER_META_TARGET, ZONE_FLOOR.user_meta, warnings);
  } catch (err) {
    throw new UserFileRefused(`${path}: ${message(err)}`);
  }
  const femHash = textAttr(m, "fem_hash");
  if (!femHash) throw new UserFileRefused(`${path}: /meta@fem_hash is required`);
  const revision = intAttr(m, "revision", false);
  if (revision === null) throw new UserFileRefused(`${path}: /meta@revision is not an integer`);
  return {
    path,
    femHash,
    modelHash: textAttr(m, "model_hash") || null,
    resultsHash: textAttr(m, "results_hash") || null,
    writer: textAttr(m, "writer"),
    units: textAttr(m, "units"),
    revision,
    zones: f.keys().filter((k) => k !== "meta").sort(byCodePoint),
    warnings,
  };
}

/**
 * Whether the zone is present, refusing a key and a group that disagree, or a
 * version outside the app's range (Python's `_zone_version_checked`).
 */
function zoneVersionChecked(t: Tree, f: FpFile, zone: string, root: string, target: { major: number; minor: number }, floor: number, warnings: string[]): boolean {
  const meta = t.group(f, "meta")!;
  const key = `${zone}_schema_version`;
  const present = f.keys().includes(root);
  const raw = meta.attrs[key];
  if (raw === undefined && present) throw new ZoneRefused(`/${root}`, `/meta has no version key for zone '${zone}'`);
  if (raw !== undefined && !present) throw new ZoneRefused(`/${root}`, `/meta stamps zone '${zone}' but the file has no /${root}`);
  if (raw === undefined) return false;
  try {
    checkZoneVersion(textAttr(meta, key) as string, key, target, floor, warnings);
  } catch (err) {
    if (err instanceof SchemaError) throw new ZoneRefused(`/${root}`, err.message);
    throw err;
  }
  return true;
}

// ---------------------------------------------------------------------------
// /combinations (and model.h5's /opensees/combinations)
// ---------------------------------------------------------------------------

function readCombinations(t: Tree, f: FpFile, warnings: string[], modelRows: ZoneRead<Combination[]>): CombinationTable {
  if (!modelRows.ok) throw new ZoneRefused("/opensees/combinations", modelRows.reason);
  if (!zoneVersionChecked(t, f, "combinations", "combinations", COMBINATIONS_TARGET, ZONE_FLOOR.combinations, warnings)) {
    return { generation: null, generations: [], ...resolve([], modelRows.value) };
  }
  const zone = t.group(f, "combinations");
  const current = zone === null ? null : intAttr(zone, "current", false);
  if (zone === null || zone.attrs["current"] === undefined) throw new ZoneRefused("/combinations", "it has no @current");
  if (current === null) throw new ZoneRefused("/combinations", "@current is not an integer");
  const g = t.group(zone, `g${current}`);
  if (g === null) throw new ZoneRefused("/combinations", `@current names g${current}, which it does not hold`);
  return { generation: current, generations: generationsOf(t, zone), ...resolve(readGeneration(t, g, "/combinations"), modelRows.value) };
}

const GENERATION = /^g(\d+)$/;
/** Every `g<N>` group of a `/combinations` zone, ascending (Python's `_GENERATION`). */
function generationsOf(t: Tree, zone: FpGroup): number[] {
  return zone
    .keys()
    .flatMap((k) => {
      const m = GENERATION.exec(k);
      return m && t.group(zone, k) !== null ? [Number(m[1])] : [];
    })
    .sort((a, b) => a - b);
}

// ---------------------------------------------------------------------------
// VU9b (ADR 0130 A4): what a combination save reads of the file before it writes
// ---------------------------------------------------------------------------

/** A file a combination save may not write, with the reason (`no-version`, `newer`, `refused`, ...). */
export class UserWriteRefused extends Error {
  readonly reason: string;
  readonly detail: string;
  constructor(reason: string, detail: string) {
    super(`${reason}: ${detail}`);
    this.reason = reason;
    this.detail = detail;
  }
}

/** What a save needs of the file it copies: `/meta@revision`, the root zones, and the generations with the rows of `@current` (and of the one a revert asks for). */
export interface UserWriteBase {
  revision: number;
  zones: string[];
  /** `@current`, or null when the file holds no `/combinations` (a first save creates it) */
  current: number | null;
  /** every `g<N>`, ascending */
  held: number[];
  /** the rows of `@current` as stored, refused ones included */
  rows: Combination[];
  /** generation `also` and its rows, when the caller asked for one and the zone holds it; else null */
  also: { generation: number; rows: Combination[] } | null;
}

/** The root zones a `.user.h5` writer must understand, and their `/meta` keys (Python's `_check_known_versions`). */
const WRITE_ZONES: readonly [string, string][] = [
  ["combinations", "combinations_schema_version"],
  ["results", "user_results_schema_version"],
  ["modes", "modes_schema_version"],
  ["spectra", "spectra_schema_version"],
  ["entities", "entities_schema_version"],
];

/**
 * Read `path` (read-only, closed before this returns) for a combination save.
 * Refuses a file whose `/meta` or `/combinations` is newer than this app
 * writes (A4: no writer touches a zone newer than it knows; `/meta@revision`
 * is written too), a zone group with no version key, and a `/combinations`
 * this reader refuses: the save would otherwise carry it forward unread.
 */
export function readForWrite(h5: UserH5, path: string, also: number | null = null): UserWriteBase {
  let f: FpFile;
  try {
    f = openH5(h5, path);
  } catch (err) {
    throw new UserWriteRefused("cannot-read", `${path} is not a readable HDF5 file: ${message(err)}`);
  }
  const t = new Tree(h5);
  try {
    let meta: ReturnType<typeof readMeta>;
    try {
      meta = readMeta(t, f, path);
    } catch (err) {
      throw new UserWriteRefused("refused", message(err));
    }
    if (meta.warnings.length) throw new UserWriteRefused("newer", `${path}: ${meta.warnings.join("; ")}; this app does not write over a newer file`);
    const m = t.group(f, "meta")!;
    for (const [root, key] of WRITE_ZONES) {
      if (f.keys().includes(root) && m.attrs[key] === undefined) throw new UserWriteRefused("no-version", `${path}: /${root} has no ${key} in /meta; nothing was replaced`);
    }
    const warnings: string[] = [];
    let present: boolean;
    try {
      present = zoneVersionChecked(t, f, "combinations", "combinations", COMBINATIONS_TARGET, ZONE_FLOOR.combinations, warnings);
    } catch (err) {
      throw new UserWriteRefused("refused", `${path}: ${message(err)}`);
    }
    if (warnings.length) throw new UserWriteRefused("newer", `${path}: ${warnings.join("; ")}; this app does not write a zone newer than it knows (ADR 0130 A4)`);
    const base = { revision: meta.revision, zones: meta.zones };
    if (!present) return { ...base, current: null, held: [], rows: [], also: null };
    try {
      const zone = t.group(f, "combinations");
      const current = zone === null ? null : intAttr(zone, "current", false);
      if (zone === null || current === null) throw new ZoneRefused("/combinations", "it has no integer @current");
      const g = t.group(zone, `g${current}`);
      if (g === null) throw new ZoneRefused("/combinations", `@current names g${current}, which it does not hold`);
      const held = generationsOf(t, zone);
      const other = also === null ? null : t.group(zone, `g${also}`);
      return {
        ...base,
        current,
        held,
        rows: readGeneration(t, g, "/combinations"),
        also: also === null || other === null ? null : { generation: also, rows: readGeneration(t, other, "/combinations") },
      };
    } catch (err) {
      throw new UserWriteRefused("refused", `${path}: /combinations cannot be read, so it is not written over: ${message(err)}`);
    }
  } finally {
    f.close();
  }
}

/** `/meta@revision` of `path` now (read-only, closed at once): a save compares it immediately before the replace. */
export function readRevision(h5: UserH5, path: string): number {
  const f = openH5(h5, path);
  try {
    return readMeta(new Tree(h5), f, path).revision;
  } finally {
    f.close();
  }
}

/** The model archive's `/opensees/combinations` rows as stored, refused ones included (Python's `read_model_combinations`); throws when it cannot be read. */
export function modelCombinationRows(h5: UserH5, model: string): Combination[] {
  const r = readModelCombinations(h5, model, []);
  if (!r.ok) throw new UserWriteRefused("refused", `${model}: ${r.reason}`);
  return r.value;
}

function column(t: Tree, g: FpGroup, name: string, code: string, zone: string, length: number | null): unknown[] {
  if (!g.keys().includes(name)) throw new ZoneRefused(zone, `${g.path} has no ${name} column`);
  const ds = t.dataset(g, name);
  if (ds === null || shape(ds).length !== 1) throw new ZoneRefused(zone, `${g.path}/${name} is not a one-dimensional dataset`);
  if (!isCode(ds, code)) {
    throw new ZoneRefused(zone, code === "str" ? `${ds.path} is not a string column (${dtypeText(ds)})` : `${ds.path} is ${dtypeText(ds)}, expected ${code}`);
  }
  if (length !== null && shape(ds)[0] !== length) throw new ZoneRefused(zone, `${ds.path} has ${shape(ds)[0]} rows, expected ${length}`);
  return values(ds);
}

/** The rows of one `g<N>`; malformed columns refuse the zone (Python's `read_generation`). */
function readGeneration(t: Tree, g: FpGroup, zone: string): Combination[] {
  const names = column(t, g, "name", "str", zone, null);
  const k = names.length;
  const rules = column(t, g, "rule", "str", zone, k);
  const envelopes = column(t, g, "envelope", "str", zone, k);
  const fractions = column(t, g, "fraction", "f8", zone, k) as number[];
  const dampings = column(t, g, "damping", "f8", zone, k) as number[];
  const offsets = column(t, g, "term_offsets", "i4", zone, k + 1) as number[];
  const terms = t.group(g, "terms");
  if (terms === null) throw new ZoneRefused(zone, `${g.path} has no terms group`);
  const kinds = column(t, terms, "kind", "str", zone, null);
  const n = kinds.length;
  const sources = column(t, terms, "source", "str", zone, n);
  const steps = column(t, terms, "step", "i4", zone, n) as number[];
  const modes = column(t, terms, "mode", "i4", zone, n) as number[];
  const factors = column(t, terms, "factor", "f8", zone, n) as number[];
  if (offsets[0] !== 0 || offsets[k] !== n || offsets.some((o, i) => i > 0 && o < offsets[i - 1]!)) {
    throw new ZoneRefused(zone, `${g.path}/term_offsets does not partition ${n} terms`);
  }
  const c = (raw: unknown) => cell(raw, zone, g.path);
  const out: Combination[] = [];
  for (let i = 0; i < k; i++) {
    const rowTerms: Term[] = [];
    for (let j = offsets[i]!; j < offsets[i + 1]!; j++) {
      rowTerms.push({ kind: c(kinds[j]), source: c(sources[j]), step: steps[j]!, mode: modes[j]!, factor: factors[j]! });
    }
    out.push({
      name: c(names[i]),
      rule: c(rules[i]),
      terms: rowTerms,
      envelope: c(envelopes[i]),
      fraction: Number.isNaN(fractions[i]) ? null : fractions[i]!,
      damping: Number.isNaN(dampings[i]) ? null : dampings[i]!,
    });
  }
  return out;
}

/** The rows of `/opensees/combinations/g0` in a model archive, or `[]` (Python's `read_model_combinations`). */
function readModelCombinations(h5: UserH5, model: string, warnings: string[]): ZoneRead<Combination[]> {
  const where = "/opensees/combinations";
  const t = new Tree(h5);
  let f: FpFile;
  try {
    f = openH5(h5, model);
  } catch (err) {
    return { ok: false, reason: `${where}: it cannot be read: ${message(err)}` };
  }
  try {
    if (!t.has(f, "opensees/combinations")) return { ok: true, value: [] };
    const meta = t.group(f, "meta");
    const key = "opensees_schema_version";
    if (meta === null || meta.attrs[key] === undefined) throw new ZoneRefused(where, "the archive has no opensees_schema_version");
    // ADR 0113 D7: a newer minor opens with one banner, here the model's, named as such.
    const newer: string[] = [];
    checkZoneVersion(textAttr(meta, key) as string, key, OPENSEES_TARGET, ZONE_FLOOR.opensees, newer);
    for (const w of newer) warnings.push(`${model} (read for /opensees/combinations): ${w}`);
    const zone = t.group(f, "opensees/combinations");
    const g0 = zone === null ? null : t.group(zone, "g0");
    if (g0 === null) throw new ZoneRefused(where, "it has no g0");
    return { ok: true, value: readGeneration(t, g0, where) };
  } catch (err) {
    return { ok: false, reason: err instanceof ZoneRefused ? err.message : `${where}: ${message(err)}` };
  } finally {
    f.close();
  }
}

// ---------------------------------------------------------------------------
// /results: the index, never DATA
// ---------------------------------------------------------------------------

function readResults(t: Tree, f: FpFile, warnings: string[], held: ZoneRead<Entities>): ResultIndex {
  if (!zoneVersionChecked(t, f, "user_results", "results", USER_RESULTS_TARGET, ZONE_FLOOR.user_results, warnings)) {
    return { entries: [], refused: [] };
  }
  const zone = t.group(f, "results");
  if (zone === null) throw new ZoneRefused("/results", "it is not a group");
  const links = zone
    .keys()
    .filter((k) => /^r\d+$/.test(k))
    .sort((a, b) => Number(a.slice(1)) - Number(b.slice(1)));
  let entries: ResultEntry[] = [];
  const refused: Refusal[] = [];
  links.forEach((link, at) => {
    try {
      const g = t.group(zone, link);
      if (g === null) throw new EntryRefused("bad-param", `${zone.path}/${link} is not a group`);
      entries.push(readEntryMeta(t, link, g));
    } catch (err) {
      if (err instanceof ZoneRefused) throw err;
      const name = entryName(t, zone, link);
      if (err instanceof EntryRefused) refused.push({ name, reason: err.reason, detail: err.detail, origin: "user", row: at });
      else refused.push({ name, reason: "bad-param", detail: message(err), origin: "user", row: at });
    }
  });
  // Python's `_bound_entries`: an ON_ENTITIES entry naming a row /entities does
  // not hold, or refuses, is refused alone; so is every one when the zone does not read.
  entries = entries.filter((e) => {
    if (e.target !== "entities") return true;
    let problem: string | null;
    if (!held.ok) {
      problem = `user result ${pyStr(e.name)}: /entities cannot be read: ${held.reason.replace(/^refused: /, "")}`;
    } else {
      const { kinds, rows } = entityColumns(t, zone, e);
      problem = bindingProblem(held.value, e.name, kinds, rows);
    }
    if (problem === null) return true;
    refused.push({ name: e.name, reason: "unresolved", detail: problem, origin: "user", row: links.indexOf(e.link) });
    return false;
  });
  const counts = new Map<string, number>();
  for (const e of entries) counts.set(e.name, (counts.get(e.name) ?? 0) + 1);
  for (const e of entries) {
    const n = counts.get(e.name)!;
    if (n > 1) refused.push({ name: e.name, reason: "duplicate", detail: `${n} user results are named ${JSON.stringify(e.name)}`, origin: "user", row: links.indexOf(e.link) });
  }
  return { entries: entries.filter((e) => counts.get(e.name) === 1), refused };
}

function entryName(t: Tree, zone: FpGroup, link: string): string {
  try {
    const g = t.group(zone, link);
    return g === null ? "" : (textAttr(g, "name") ?? "");
  } catch {
    return "";
  }
}

/** A flag attribute: null when absent; 0 or 1, else refused (Python's `_attr_flag`). */
function flagAttr(g: FpGroup, key: string): 0 | 1 | null {
  if (g.attrs[key] === undefined) return null;
  const v = intAttr(g, key, true);
  if (v !== 0 && v !== 1) throw new EntryRefused("bad-param", `@${key} is not 0 or 1`);
  return v;
}

/** A dataset of `kind` and rank, or the entry refuses (Python's `_dataset`). */
function entryDataset(t: Tree, g: FpGroup, name: string, code: string, ndim: number): FpDataset {
  const ds = t.dataset(g, name);
  if (ds === null) throw new EntryRefused("bad-param", `${g.path} has no ${name} dataset`);
  if (!isCode(ds, code) || shape(ds).length !== ndim) {
    throw new EntryRefused("bad-param", `${ds.path} is ${dtypeText(ds)} (${shape(ds).join(", ")}), expected ${code}, rank ${ndim}`);
  }
  return ds;
}

function readStates(t: Tree, g: FpGroup): StateAddress[] {
  const kinds = values(entryDataset(t, g, "kind", "str", 1));
  const sources = values(entryDataset(t, g, "source", "str", 1));
  const steps = values(entryDataset(t, g, "step", "i4", 1)) as number[];
  const modes = values(entryDataset(t, g, "mode", "i4", 1)) as number[];
  const n = kinds.length;
  if (sources.length !== n || steps.length !== n || modes.length !== n || n === 0) {
    throw new EntryRefused("bad-param", `${g.path} columns disagree or are empty`);
  }
  const out = kinds.map((k, i) => ({ kind: cell(k, "/results", g.path), source: cell(sources[i], "/results", g.path), step: steps[i]!, mode: modes[i]! }));
  for (const s of out) {
    const problem = addressProblem(s);
    if (problem !== null) throw new EntryRefused("bad-term", problem);
  }
  return out;
}

/** Validate an entry without reading `DATA` (Python's `read_entry_meta`). */
function readEntryMeta(t: Tree, link: string, g: FpGroup): ResultEntry {
  const name = textAttr(g, "name");
  const problem = name === null ? "absent" : nameProblem(name);
  if (problem !== null) throw new EntryRefused("empty-name", `${g.path}@name: ${problem}`);
  const target = textAttr(g, "target");
  if (!(TARGETS as readonly (string | null)[]).includes(target)) throw new EntryRefused("bad-param", `@target ${JSON.stringify(target)} is not one of ${TARGETS.join(", ")}`);
  const basis = textAttr(g, "basis");
  if (!(BASES as readonly (string | null)[]).includes(basis)) throw new EntryRefused("bad-param", `@basis ${JSON.stringify(basis)} is not one of ${BASES.join(", ")}`);
  const perState = flagAttr(g, "per_state");
  if (perState === null) throw new EntryRefused("bad-param", `${g.path} has no @per_state`);
  const linear = flagAttr(g, "linear") ?? 0;
  let states: StateAddress[] = [];
  if (perState) {
    const sg = t.group(g, "STATES");
    if (sg === null) throw new EntryRefused("bad-param", `${g.path} is per state and has no STATES`);
    states = readStates(t, sg);
  }
  const s = perState ? states.length : 1;
  const tg = target as ResultEntry["target"];
  const on = ON[tg];
  if (!t.has(g, `RESULTS/${on}/values`)) throw new EntryRefused("bad-param", `${g.path} has no RESULTS/${on}/values`);
  const vals = t.group(g, `RESULTS/${on}/values`);
  if (vals === null) throw new EntryRefused("bad-param", `${g.path}/RESULTS/${on}/values is not a group`);
  const holders = tg === "elements" ? vals.keys().map((b) => [b, t.group(vals, b)] as const) : ([["", vals]] as const);
  if (holders.length === 0) throw new EntryRefused("bad-param", `${vals.path} has no bucket`);
  for (const [bucket, holder] of holders) {
    if (holder === null) throw new EntryRefused("bad-param", `${vals.path}/${bucket} is not a bucket group`);
    checkHolder(t, holder, tg, s);
    if (tg === "elements") checkPlacement(t, g, bucket);
  }
  // checkHolder held @COMPONENTS to one name per DATA column.
  const components = tg === "elements" ? [] : textAttr(vals, "COMPONENTS")!.split(",");
  return { name: name!, target: tg, basis: basis as ResultEntry["basis"], perState: perState === 1, linear: linear === 1, link, states, components };
}

/** The shapes of one holder; `DATA` is never read, only its shape (Python's `_check_holder`). */
function checkHolder(t: Tree, holder: FpGroup, target: ResultEntry["target"], s: number): void {
  const data = entryDataset(t, holder, "DATA", "f8", 3);
  const [ds0, n, c] = shape(data) as [number, number, number];
  if (ds0 !== s) throw new EntryRefused("bad-param", `${data.path} has ${ds0} rows on axis 0, expected ${s}`);
  if (target === "nodes" || target === "elements") {
    const ids = entryDataset(t, holder, "ID", "i4", 2);
    if (!sameShape(shape(ids), [n, 1])) throw new EntryRefused("bad-param", `${ids.path} has shape (${shape(ids).join(", ")}), expected (${n}, 1)`);
  } else if (target === "entities") {
    const kinds = entryDataset(t, holder, "ENTITY_KIND", "str", 1);
    if (!sameShape(shape(kinds), [n]) || !sameShape(shape(entryDataset(t, holder, "ENTITY_ROW", "i4", 1)), [n])) {
      throw new EntryRefused("bad-param", `${holder.path}: entity columns must have ${n} rows`);
    }
    for (const k of values(kinds)) {
      const kind = cell(k, "/results", kinds.path);
      if (!ENTITY_KINDS.includes(kind)) throw new EntryRefused("bad-param", `${kinds.path}: entity kind ${JSON.stringify(kind)} is not one of ${ENTITY_KINDS.join(", ")}`);
    }
  } else if (n !== 1) {
    throw new EntryRefused("bad-param", `${data.path}: a domain value has n = 1, got ${n}`);
  }
  if (!sameShape(shape(entryDataset(t, holder, "STEP", "i4", 1)), [s]) || !sameShape(shape(entryDataset(t, holder, "TIME", "f8", 1)), [s])) {
    throw new EntryRefused("bad-param", `${holder.path}: STEP and TIME must have ${s} rows`);
  }
  if (target === "elements") {
    checkColumnMap(t, holder, c);
  } else {
    const text = textAttr(holder, "COMPONENTS");
    if (!text || text.split(",").length !== c) throw new EntryRefused("bad-param", `${holder.path}@COMPONENTS ${JSON.stringify(text)} does not name ${c} columns`);
  }
}

function checkColumnMap(t: Tree, holder: FpGroup, width: number): void {
  const cmap = t.group(holder, "COLUMN_MAP");
  if (cmap === null) throw new EntryRefused("bad-param", `${holder.path} has no COLUMN_MAP`);
  const text = textAttr(cmap, "COMP_NAMES");
  if (!text) throw new EntryRefused("bad-param", `${cmap.path} has no @COMP_NAMES`);
  const lines = text.split("\n");
  const mult = values(entryDataset(t, cmap, "MULTIPLICITY", "i4", 2)) as number[];
  const gauss = values(entryDataset(t, cmap, "GAUSS_ID", "i4", 2));
  if (mult.length !== lines.length || gauss.length !== lines.length) throw new EntryRefused("bad-param", `${cmap.path} rows disagree with its @COMP_NAMES lines`);
  const total = lines.reduce((sum, line, i) => sum + line.split(",").length * mult[i]!, 0);
  if (total !== width) throw new EntryRefused("bad-param", `${cmap.path} covers ${total} columns, DATA has ${width}`);
}

/** A stage carries its own `GP_PARAM` (Python's `_check_placement`, less `_check_frame`: see the header). */
function checkPlacement(t: Tree, entry: FpGroup, bucket: string): void {
  const where = `MODEL/ELEMENTS/${bucket}`;
  const placed = t.group(entry, where);
  if (placed === null) throw new EntryRefused("bad-param", `${entry.path} has no ${where}`);
  for (const key of ["TOPOLOGY", "PARAM_DOMAIN", "FAMILY"]) {
    if (!textAttr(placed, key)) throw new EntryRefused("bad-param", `${placed.path} has no @${key}`);
  }
  const quad = t.group(placed, "QUADRATURE");
  if (quad === null) throw new EntryRefused("bad-param", `${placed.path} has no QUADRATURE`);
  const gp = entryDataset(t, quad, "GP_PARAM", "f8", 2);
  const [rows, ndir] = shape(gp) as [number, number];
  if (intAttr(placed, "NUM_GP", true) !== rows) throw new EntryRefused("bad-param", `${placed.path}@NUM_GP does not match GP_PARAM (${rows}, ${ndir})`);
  const order = placed.attrs["ORDER"];
  if (order === undefined || order.metadata.type !== H5T_INTEGER) throw new EntryRefused("bad-param", `${placed.path}@ORDER is absent or not integers`);
  const ov = order.value;
  const list = ArrayBuffer.isView(ov) || Array.isArray(ov) ? Array.from(ov as ArrayLike<number | bigint>, Number) : [Number(ov)];
  const rank = order.metadata.shape?.length ?? 0;
  if (rank !== 1 || list.length !== ndir || list.some((x) => x < 1)) {
    throw new EntryRefused("bad-param", `${placed.path}: ORDER [${list.join(", ")}] needs one entry >= 1 per GP_PARAM column, and GP_PARAM has shape (${rows}, ${ndir})`);
  }
}

// ---------------------------------------------------------------------------
// /modes (Python's `read_modes`; reader/userModes.ts holds `table_problem`)
// ---------------------------------------------------------------------------

function modeColumn(t: Tree, zone: FpGroup, name: string, code: string, ndim: number): { values: unknown[]; shape: number[] } {
  const ds = t.dataset(zone, name);
  if (ds === null) throw new ZoneRefused("/modes", `it has no ${name} column`);
  if (!isCode(ds, code) || shape(ds).length !== ndim) {
    throw new ZoneRefused("/modes", `${name} is ${npDtype(ds.metadata)} ${pyShape(shape(ds))}, expected ${code}, rank ${ndim}`);
  }
  return { values: values(ds), shape: shape(ds) };
}

function readModes(t: Tree, f: FpFile, warnings: string[]): ModeTable {
  if (!zoneVersionChecked(t, f, "modes", "modes", MODES_TARGET, ZONE_FLOOR.modes, warnings)) {
    throw new ZoneRefused("/modes", "this file holds no modal table");
  }
  const zone = t.group(f, "modes");
  if (zone === null) throw new ZoneRefused("/modes", "it is not a group");
  if (zone.attrs["components"] === undefined) throw new ZoneRefused("/modes", "it has no @components");
  const components = (textAttr(zone, "components") ?? "").split(",");
  const host = modeColumn(t, zone, "host", "str", 1).values.map((x) => cell(x, "/modes", "host"));
  const kind = modeColumn(t, zone, "kind", "str", 1).values.map((x) => cell(x, "/modes", "kind"));
  const step = modeColumn(t, zone, "step", "i4", 1).values as number[];
  const mode = modeColumn(t, zone, "mode", "i4", 1).values as number[];
  const lam = modeColumn(t, zone, "lambda", "f8", 1).values as number[];
  const period = modeColumn(t, zone, "period", "f8", 1).values as number[];
  const mass = modeColumn(t, zone, "generalized_mass", "f8", 1).values as number[];
  const tableNames = ["participation", "mass_ratio", "cumulative_ratio"] as const;
  const tables = tableNames.map((n) => modeColumn(t, zone, n, "f8", 2));
  const m = host.length;
  const c = components.length;
  // The lengths are checked here, in Python's `table_problem` words; the rules on the values are `modeTableProblem`'s.
  if (kind.length !== m) throw new ZoneRefused("/modes", `${kind.length} kinds for ${m} rows`);
  for (const [what, col] of [["step", step], ["mode", mode]] as const) {
    if (col.length !== m) throw new ZoneRefused("/modes", `${what} must be ${m} integers, got int32 ${pyShape([col.length])}`);
  }
  for (const [what, col] of [["lambda", lam], ["period", period], ["generalized_mass", mass]] as const) {
    if (col.length !== m) throw new ZoneRefused("/modes", `${what} has shape ${pyShape([col.length])}, expected (${m},)`);
  }
  tableNames.forEach((what, k) => {
    if (!sameShape(tables[k]!.shape, [m, c])) throw new ZoneRefused("/modes", `${what} has shape (${tables[k]!.shape.join(", ")}), expected (${m}, ${c})`);
  });
  const slab = (k: number, i: number) => (tables[k]!.values as number[]).slice(i * c, (i + 1) * c);
  const rows: ModeRow[] = host.map((h, i) => ({
    host: h,
    kind: kind[i]!,
    step: step[i]!,
    mode: mode[i]!,
    lambda: lam[i]!,
    period: period[i]!,
    generalizedMass: mass[i]!,
    participation: slab(0, i),
    massRatio: slab(1, i),
    cumulativeRatio: slab(2, i),
  }));
  const table: ModeTable = { components, rows };
  const problem = modeTableProblem(table);
  if (problem !== null) throw new ZoneRefused("/modes", problem);
  return table;
}

// ---------------------------------------------------------------------------
// /spectra (Python's `read_spectra`, `read_case`; reader/userSpectra.ts holds `case_problem`)
// ---------------------------------------------------------------------------

function readSpectra(t: Tree, f: FpFile, warnings: string[]): SpectraIndex {
  if (!zoneVersionChecked(t, f, "spectra", "spectra", SPECTRA_TARGET, ZONE_FLOOR.spectra, warnings)) return { cases: [], refused: [] };
  const zone = t.group(f, "spectra");
  if (zone === null) throw new ZoneRefused("/spectra", "it is not a group");
  const links = zone
    .keys()
    .filter((k) => /^s\d+$/.test(k))
    .sort((a, b) => Number(a.slice(1)) - Number(b.slice(1)));
  const read: [number, SpectrumCase][] = [];
  const refused: Refusal[] = [];
  links.forEach((link, at) => {
    try {
      const g = t.group(zone, link);
      if (g === null) throw new EntryRefused("bad-param", `${zone.path}/${link} is not a group`);
      read.push([at, readCase(t, link, g)]);
    } catch (err) {
      // A non-UTF-8 table cell refuses the zone.
      if (err instanceof ZoneRefused) throw err;
      const name = entryName(t, zone, link);
      if (err instanceof EntryRefused) refused.push({ name, reason: err.reason, detail: err.detail, origin: "user", row: at });
      else refused.push({ name, reason: "bad-param", detail: message(err), origin: "user", row: at });
    }
  });
  const counts = new Map<string, number>();
  for (const [, c] of read) counts.set(c.name, (counts.get(c.name) ?? 0) + 1);
  for (const [at, c] of read) {
    const n = counts.get(c.name)!;
    if (n > 1) refused.push({ name: c.name, reason: "duplicate", detail: `${n} cases are named ${pyStr(c.name)}`, origin: "user", row: at });
  }
  return { cases: read.filter(([, c]) => counts.get(c.name) === 1).map(([, c]) => c), refused };
}

/** One numeric attribute of a numpy kind in `kinds` ("iu" or "f"), or the case refuses (Python's `_attr` and `_number`). */
function numberAttr(g: FpGroup, key: string, kinds: "iu" | "f"): number {
  const a = g.attrs[key];
  if (a === undefined) throw new EntryRefused("bad-param", `${g.path} has no @${key}`);
  let v = a.value;
  if (Array.isArray(v) || ArrayBuffer.isView(v)) {
    const arr = v as ArrayLike<unknown>;
    if (arr.length !== 1) throw new EntryRefused("bad-param", `${g.path}@${key} holds ${arr.length} values`);
    v = arr[0];
  }
  const ok = kinds === "f" ? a.metadata.type === H5T_FLOAT : a.metadata.type === H5T_INTEGER;
  if (!ok || (typeof v !== "number" && typeof v !== "bigint")) throw new EntryRefused("bad-param", `${g.path}@${key} is ${npDtype(a.metadata)}, expected ${kinds}`);
  return Number(v);
}

/** One `modes/` or `spectrum/` column: one-dimensional and of its dtype, or the case refuses (Python's `_column`). */
function caseColumn(t: Tree, g: FpGroup, name: string, code: string): unknown[] {
  const ds = t.dataset(g, name);
  if (ds === null) throw new EntryRefused("bad-param", `${g.path} has no ${name} column`);
  if (!isCode(ds, code) || shape(ds).length !== 1) throw new EntryRefused("bad-param", `${ds.path} is ${npDtype(ds.metadata)} ${pyShape(shape(ds))}, expected ${code}`);
  return values(ds);
}

/** One case, judged by `caseProblem`, or `EntryRefused` (Python's `read_case`, in its order). Its DATA is not read. */
function readCase(t: Tree, link: string, g: FpGroup): SpectrumCase {
  if (g.attrs["name"] === undefined) throw new EntryRefused("bad-param", `${g.path} has no @name`);
  const name = textAttr(g, "name") ?? "";
  const problem = nameProblem(name);
  if (problem !== null) throw new EntryRefused("empty-name", `${g.path}@name: ${problem}`);
  if (g.attrs["KIND"] === undefined) throw new EntryRefused("bad-param", `${g.path} has no @KIND`);
  const kindAttr = textAttr(g, "KIND") ?? "";
  if (kindAttr !== "static") throw new EntryRefused("bad-param", `${g.path}@KIND is ${pyStr(kindAttr)}, not 'static'`);
  const spectrum = t.group(g, "spectrum");
  const modes = t.group(g, "modes");
  const reference = t.group(g, "reference");
  if (spectrum === null || modes === null || reference === null) throw new EntryRefused("bad-param", `${g.path} needs spectrum, modes and reference`);
  // Python builds the case in this order: the first fault it meets is the one it names.
  const direction = numberAttr(g, "direction", "iu");
  const scale = numberAttr(g, "scale", "f");
  const periods = caseColumn(t, spectrum, "T", "f8") as number[];
  const sa = caseColumn(t, spectrum, "Sa", "f8") as number[];
  const draft: CaseDraft = {
    name,
    link,
    direction,
    scale,
    periods,
    sa,
    columns: {
      mode: caseColumn(t, modes, "mode", "i4") as number[],
      kind: caseColumn(t, modes, "kind", "str").map((x) => cell(x, "/spectra", `${modes.path}/kind`)),
      lambda: caseColumn(t, modes, "lambda", "f8") as number[],
      saPeriod: caseColumn(t, modes, "sa_period", "f8") as number[],
      gamma: caseColumn(t, modes, "gamma", "f8") as number[],
      sa: caseColumn(t, modes, "sa", "f8") as number[],
      damping: caseColumn(t, modes, "damping", "f8") as number[],
    },
    results: readQuantities(t, g, "RESULTS"),
    reference: readQuantities(t, reference, "reference/RESULTS"),
  };
  const bad = caseProblem(draft);
  if (bad !== null) throw new EntryRefused("bad-param", `${g.path}: ${bad}`);
  return caseOf(draft);
}

/** `ON_DOMAIN` and `ON_NODES` to their targets (Python's `_TARGET`). */
const SPECTRUM_TARGET_OF: ReadonlyMap<string, SpectrumTarget> = new Map(Object.entries(SPECTRA_FAMILIES).map(([k, v]) => [v, k as SpectrumTarget]));

/** A case's results, by shape; a family outside 1.0.0 refuses the case (Python's `_read_quantities`). */
function readQuantities(t: Tree, root: FpGroup, where: string): SpectrumQuantity[] {
  const results = t.group(root, "RESULTS");
  if (results === null) throw new EntryRefused("bad-param", `${root.path} has no ${where}`);
  const out: SpectrumQuantity[] = [];
  for (const on of results.keys()) {
    const target = SPECTRUM_TARGET_OF.get(on);
    if (target === undefined) {
      throw new EntryRefused("bad-param", `${results.path}/${on}: the family ${on} is not in spectra 1.0.0, whose cases hold ${pyList([...SPECTRUM_TARGET_OF.keys()].sort())} only`);
    }
    const family = t.group(results, on);
    if (family === null) throw new EntryRefused("bad-param", `${results.path}/${on} is not a group`);
    for (const label of family.keys()) {
      const holder = t.group(family, label);
      if (holder === null || !t.has(holder, "DATA")) throw new EntryRefused("bad-param", `${family.path}/${label} is not a result group`);
      const data = t.dataset(holder, "DATA");
      if (data === null || !isCode(data, "f8") || shape(data).length !== 3) throw new EntryRefused("bad-param", `${holder.path}/DATA is not f8 of rank 3`);
      const [s, n, c] = shape(data) as [number, number, number];
      for (const axis of ["STEP", "TIME"]) {
        const ds = t.dataset(holder, axis);
        if (ds === null || !sameShape(shape(ds), [s])) throw new EntryRefused("bad-param", `${holder.path}: ${axis} must have ${s} rows`);
      }
      let ids: number[] | null = null;
      if (target === "nodes") {
        const id = t.dataset(holder, "ID");
        if (id === null || !isCode(id, "i4") || !sameShape(shape(id), [n, 1])) throw new EntryRefused("bad-param", `${holder.path}/ID must be i4 (${n}, 1)`);
        ids = values(id) as number[];
      }
      if (holder.attrs["COMPONENTS"] === undefined) throw new EntryRefused("bad-param", `${holder.path} has no @COMPONENTS`);
      const components = (textAttr(holder, "COMPONENTS") ?? "").split(",");
      out.push({ label, target, components, shape: [s, n, c], ids });
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// /entities (Python's `read_entities`; reader/userEntities.ts holds `row_refusals`)
// ---------------------------------------------------------------------------

const ENTITIES = "/entities";

/** A column's values; [] for a dataset with an empty axis, whose h5wasm `value` is not an array. */
function columnValues(ds: FpDataset): unknown[] {
  return shape(ds).some((n) => n === 0) ? [] : values(ds);
}

/** `(None, 3)` for a wanted shape, as Python prints it. */
const wantedShape = (want: readonly (number | null)[]) => (want.length === 1 ? `(${want[0] ?? "None"},)` : `(${want.map((w) => w ?? "None").join(", ")})`);

/** One column of `code` and shape `want` (null: any length), or the zone refuses (Python's `_col`). */
function entityColumn(t: Tree, g: FpGroup, name: string, code: string, want: readonly (number | null)[]): unknown[] {
  const ds = t.dataset(g, name);
  if (ds === null) throw new ZoneRefused(ENTITIES, `${g.path} has no ${name} column`);
  const got = shape(ds);
  const fits = got.length === want.length && want.every((w, i) => w === null || got[i] === w);
  if (!isCode(ds, code) || !fits) throw new ZoneRefused(ENTITIES, `${ds.path} is ${npDtype(ds.metadata)} ${pyShape(got)}, expected ${code} ${wantedShape(want)}`);
  const v = columnValues(ds);
  return code === "str" ? v.map((x) => cell(x, ENTITIES, ds.path)) : v;
}

/** The rows' `[start, end)` from `<kind>_offsets`, which runs from 0 to the column's length (Python's `_spans`). */
function spans(t: Tree, g: FpGroup, offsets: string, k: number, total: number): [number, number][] {
  const off = entityColumn(t, g, offsets, "i4", [k + 1]) as number[];
  if (off[0] !== 0 || off[k] !== total || off.some((o, i) => i > 0 && o < off[i - 1]!)) {
    throw new ZoneRefused(ENTITIES, `${g.path}/${offsets} [${off.join(", ")}] does not run from 0 to ${total}`);
  }
  return off.slice(0, -1).map((a, i) => [a, off[i + 1]!]);
}

function entityTable(t: Tree, zone: FpGroup, name: string): FpGroup {
  const g = t.group(zone, name);
  if (g === null) throw new ZoneRefused(ENTITIES, `it has no ${name} table`);
  return g;
}

const vec3 = (flat: readonly number[], i: number): Vec3 => [flat[3 * i]!, flat[3 * i + 1]!, flat[3 * i + 2]!];
const points = (flat: readonly number[], [a, b]: [number, number]): Vec3[] => Array.from({ length: b - a }, (_, j) => vec3(flat, a + j));

/** `/entities`: a missing table, a malformed column or bad offsets refuse the zone; a bad row refuses alone. */
function readEntities(t: Tree, f: FpFile, warnings: string[]): Entities {
  if (!zoneVersionChecked(t, f, "entities", "entities", ENTITIES_TARGET, ZONE_FLOOR.entities, warnings)) return noEntities();
  const zone = t.group(f, "entities");
  if (zone === null) throw new ZoneRefused(ENTITIES, "it is not a group");
  const space = zone.attrs["element_index_space"] === undefined ? null : textAttr(zone, "element_index_space");
  if (space !== ELEMENT_INDEX_SPACE) {
    throw new ZoneRefused(ENTITIES, `@element_index_space is ${space === null ? "None" : pyStr(space)}; this reader knows ${pyStr(ELEMENT_INDEX_SPACE)} only`);
  }

  let g = entityTable(t, zone, "cuts");
  let names = entityColumn(t, g, "name", "str", [null]) as string[];
  let k = names.length;
  const side = entityColumn(t, g, "side", "str", [k]) as string[];
  const [point, normal, reference] = ["point", "normal", "reference"].map((c) => entityColumn(t, g, c, "f8", [k, 3]) as number[]) as [number[], number[], number[]];
  const eids = entityColumn(t, g, "element_ids", "i4", [null]) as number[];
  const poly = entityColumn(t, g, "polygon", "f8", [null, 3]) as number[];
  const eSpan = spans(t, g, "element_offsets", k, eids.length);
  const pSpan = spans(t, g, "polygon_offsets", k, poly.length / 3);
  const cuts: CutEntity[] = names.map((name, i) => ({
    name,
    side: side[i]!,
    point: vec3(point, i),
    normal: vec3(normal, i),
    reference: vec3(reference, i),
    elementIds: eids.slice(...eSpan[i]!),
    polygon: points(poly, pSpan[i]!),
  }));

  g = entityTable(t, zone, "storeys");
  names = entityColumn(t, g, "name", "str", [null]) as string[];
  k = names.length;
  const level = entityColumn(t, g, "level", "f8", [k]) as number[];
  const ref = entityColumn(t, g, "reference", "f8", [k, 3]) as number[];
  const nids = entityColumn(t, g, "node_ids", "i4", [null]) as number[];
  const nSpan = spans(t, g, "node_offsets", k, nids.length);
  const storeys: Storey[] = names.map((name, i) => ({ name, level: level[i]!, reference: vec3(ref, i), nodeIds: nids.slice(...nSpan[i]!) }));

  g = entityTable(t, zone, "point_sets");
  names = entityColumn(t, g, "name", "str", [null]) as string[];
  k = names.length;
  const xyz = entityColumn(t, g, "xyz", "f8", [null, 3]) as number[];
  const host = entityColumn(t, g, "host_element", "i4", [xyz.length / 3]) as number[];
  const xSpan = spans(t, g, "point_offsets", k, xyz.length / 3);
  const pointSets: PointSet[] = names.map((name, i) => ({ name, xyz: points(xyz, xSpan[i]!), hostElement: host.slice(...xSpan[i]!) }));

  return { cuts, storeys, pointSets, refused: rowRefusals({ cuts, storeys, pointSets }) };
}

/** An `ON_ENTITIES` entry's `ENTITY_KIND` and `ENTITY_ROW`; `checkHolder` has checked their shapes and kinds. */
function entityColumns(t: Tree, zone: FpGroup, e: ResultEntry): { kinds: string[]; rows: number[] } {
  const v = t.group(zone, `${e.link}/RESULTS/ON_ENTITIES/values`)!;
  const kinds = t.dataset(v, "ENTITY_KIND")!;
  return {
    kinds: columnValues(kinds).map((x) => cell(x, "/results", kinds.path)),
    rows: columnValues(t.dataset(v, "ENTITY_ROW")!) as number[],
  };
}

/** One `ON_ENTITIES` user result with its values, each `DATA` row bound to the entity it names. */
export interface EntityValues {
  name: string;
  components: string[];
  /** one per `DATA` axis-0 row; null for a state-free result */
  states: StateAddress[] | null;
  /** one per `DATA` axis-1 row: the `(ENTITY_KIND, ENTITY_ROW)` it names, and that row */
  rows: { kind: EntityKind; row: number; entity: CutEntity | Storey | PointSet }[];
  /** `DATA` `(S, n, c)`, row-major */
  shape: [number, number, number];
  data: number[];
}

/** A user result that cannot be given: absent, refused, or not on entities. */
export class EntityValuesRefused extends Error {}

/**
 * The values of the `ON_ENTITIES` user result `name` in the `.user.h5` at
 * `path`, read on request (never by the index). It must stand in the index,
 * which binds every row it names to `/entities`; otherwise it refuses with why.
 */
export function readEntityValues(h5: UserH5, path: string, name: string): EntityValues {
  const user = readUserFile(h5, path, null);
  if (!user.results.ok) throw new EntityValuesRefused(`${path}: /results ${user.results.reason}`);
  const entry = user.results.value.entries.find((e) => e.name === name);
  if (entry === undefined) {
    const why = user.results.value.refused.find((r) => r.name === name);
    throw new EntityValuesRefused(why ? `user result ${pyStr(name)} is refused: ${why.reason}: ${why.detail}` : `${path} holds no user result ${pyStr(name)}`);
  }
  if (entry.target !== "entities") throw new EntityValuesRefused(`user result ${pyStr(name)} is on ${entry.target}, not on entities`);
  // The entry stands, so it bound its rows to a zone that read.
  const held = user.entities;
  if (held === null || !held.ok) throw new EntityValuesRefused(`user result ${pyStr(name)} stands with no /entities that reads`);
  const t = new Tree(h5);
  const f = openH5(h5, path);
  try {
    const zone = t.group(f, "results")!;
    const { kinds, rows } = entityColumns(t, zone, entry);
    const v = t.group(zone, `${entry.link}/RESULTS/ON_ENTITIES/values`)!;
    const data = t.dataset(v, "DATA")!;
    return {
      name,
      components: (textAttr(v, "COMPONENTS") ?? "").split(","),
      states: entry.perState ? entry.states : null,
      rows: kinds.map((kind, i) => {
        if (!isEntityKind(kind)) throw new EntityValuesRefused(`user result ${pyStr(name)}: entity kind ${pyStr(kind)}`);
        return { kind, row: rows[i]!, entity: entityRow(held.value, kind, rows[i]!) };
      }),
      shape: shape(data) as [number, number, number],
      data: columnValues(data) as number[],
    };
  } finally {
    f.close();
  }
}

// ---------------------------------------------------------------------------
// VU16: one component of a state-free ON_NODES user result, as a contour column
// ---------------------------------------------------------------------------

/** One component of a state-free `ON_NODES` user result: the node ids and their values, in `DATA` row order. */
export interface UserNodalColumn {
  name: string;
  component: string;
  /** every component of the entry, in `DATA` column order */
  components: string[];
  /** the entry's `ID` column (FEM node ids) */
  ids: Float64Array;
  /** `DATA[0, :, column]`; NaN is no value at that node (drawn as no data) */
  values: Float64Array;
}

/** What `user:readResult` answers: the column, or why it is not given (`stale`: A6; `refused`: anything else). */
export type UserColumnRead = { ok: true; value: UserNodalColumn } | { ok: false; reason: "refused" | "stale"; detail: string };

/** A user result whose column cannot be given: absent, refused, on another target, or per state. */
export class UserColumnRefused extends Error {}

/**
 * The column of `component` of the state-free `ON_NODES` user result `name`,
 * read on request (never by the index). It must stand in `results` (the index
 * `readUserFile` read from the same file), so its `DATA` and `ID` shapes are
 * already checked; an id listed twice refuses, because a node takes one value.
 */
export function readUserNodalColumn(h5: UserH5, path: string, results: ZoneRead<ResultIndex>, name: string, component: string): UserNodalColumn {
  if (!results.ok) throw new UserColumnRefused(`${path}: /results ${results.reason}`);
  const entry = results.value.entries.find((e) => e.name === name);
  if (entry === undefined) {
    const why = results.value.refused.find((r) => r.name === name);
    throw new UserColumnRefused(why ? `user result ${pyStr(name)} is refused: ${why.reason}: ${why.detail}` : `${path} holds no user result ${pyStr(name)}`);
  }
  if (entry.target !== "nodes") throw new UserColumnRefused(`user result ${pyStr(name)} is on ${entry.target}; this app draws a user result on nodes only`);
  if (entry.perState) throw new UserColumnRefused(`user result ${pyStr(name)} is per state (${entry.states.length} states); this app draws a state-free one only`);
  const column = entry.components.indexOf(component);
  if (column < 0) throw new UserColumnRefused(`user result ${pyStr(name)} has no component ${pyStr(component)}; its components are ${entry.components.join(", ")}`);
  const t = new Tree(h5);
  const f = openH5(h5, path);
  try {
    const v = t.group(f, `results/${entry.link}/RESULTS/ON_NODES/values`)!;
    const data = t.dataset(v, "DATA")!;
    const [, n, c] = shape(data) as [number, number, number];
    const raw = data.value as Float64Array;
    const ids = Float64Array.from(columnValues(t.dataset(v, "ID")!) as number[]);
    const values = new Float64Array(n);
    for (let k = 0; k < n; k++) values[k] = raw[k * c + column]!;
    const seen = new Set<number>();
    for (const id of ids) {
      if (seen.has(id)) throw new UserColumnRefused(`user result ${pyStr(name)} lists node ${id} twice in its ID column`);
      seen.add(id);
    }
    return { name, component, components: [...entry.components], ids, values };
  } finally {
    f.close();
  }
}

/**
 * `user:readResult`'s answer for the `.user.h5` at `path` judged against the
 * set's `model` and `results` (A6): a stale entry is refused with its reason
 * before its `DATA` is read; then `readUserNodalColumn`. Never raises.
 */
export function readUserResultColumn(
  h5: UserH5,
  path: string,
  model: string | null,
  results: string | null,
  list: (dir: string) => readonly string[],
  name: string,
  component: string,
): UserColumnRead {
  const set = readUserSet(h5, path, model, results, list);
  if (!set.ok) return { ok: false, reason: "refused", detail: set.error };
  // A standing entry with no own verdict is stale (A6: nothing is read as current on no proof), as reduce.ts
  // `NO_VERDICT` holds; a name `__proto__` sets the record's prototype, never an own key. An entry that does not
  // stand is refused by `readUserNodalColumn`, with why.
  const standing = set.user.results.ok && set.user.results.value.entries.some((e) => e.name === name);
  const mark = Object.hasOwn(set.staleness.results, name) ? set.staleness.results[name]! : null;
  if (standing && mark === null) return { ok: false, reason: "stale", detail: `user result ${pyStr(name)} is stale (A6): the reader gave no staleness verdict for it` };
  if (mark?.stale) return { ok: false, reason: "stale", detail: `user result ${pyStr(name)} is stale (A6): ${mark.reason}` };
  try {
    return { ok: true, value: readUserNodalColumn(h5, path, set.user.results, name, component) };
  } catch (err) {
    return { ok: false, reason: "refused", detail: message(err) };
  }
}

// ---------------------------------------------------------------------------
// Staleness (A6)
// ---------------------------------------------------------------------------

/** `(fem_hash, model_hash)` from a model's `/meta/lineage`, exactly as stored (Python's `model_hashes`). */
function modelHashes(h5: UserH5, model: string): { ok: true; femHash: string; modelHash: string | null } | { ok: false; reason: string } {
  const fs = process.getBuiltinModule("node:fs");
  if (!fs.statSync(model, { throwIfNoEntry: false })?.isFile()) return { ok: false, reason: `no such model archive: ${model}` };
  const t = new Tree(h5);
  let f: FpFile;
  try {
    f = openH5(h5, model);
  } catch (err) {
    return { ok: false, reason: `${model} is not a readable model archive: ${message(err)}` };
  }
  try {
    if (t.group(f, "meta") === null) return { ok: false, reason: `${model} has no /meta` };
    const lineage = t.group(f, "meta/lineage");
    const fem = lineage === null ? null : textAttr(lineage, "fem_hash");
    if (!fem) return { ok: false, reason: `${model}: /meta/lineage has no fem_hash to pair with` };
    return { ok: true, femHash: fem, modelHash: (lineage && textAttr(lineage, "model_hash")) || null };
  } finally {
    f.close();
  }
}

function modelState(h5: UserH5, user: UserIndex, model: string | null): ZoneState {
  if (model === null) return { stale: true, reason: "no model file to compare with" };
  const m = modelHashes(h5, model);
  if (!m.ok) return { stale: true, reason: `the model hashes cannot be read: ${m.reason}` };
  if (m.femHash !== user.femHash) return { stale: true, reason: `fem_hash differs: ${user.femHash} here, ${m.femHash} in the model` };
  if (m.modelHash !== null && user.modelHash !== null && m.modelHash !== user.modelHash) {
    return { stale: true, reason: `model_hash differs: ${user.modelHash} here, ${m.modelHash} in the model` };
  }
  return { stale: false, reason: "" };
}

function resultsState(h5: UserH5, user: UserIndex, results: string | null, list: (dir: string) => readonly string[]): ZoneState {
  if (user.resultsHash === null) return { stale: true, reason: "this file has no results_hash" };
  if (results === null) return { stale: true, reason: "no results file to compare with" };
  let current: string;
  try {
    current = resultsFingerprint(h5, results, list);
  } catch (err) {
    if (!(err instanceof FingerprintRefused)) throw err;
    return { stale: true, reason: `the results file cannot be fingerprinted: ${err.message}` };
  }
  if (current !== user.resultsHash) return { stale: true, reason: `results_hash differs: ${user.resultsHash} here, ${current} on disk` };
  return { stale: false, reason: "" };
}

/** A6 applied to `user` against the set's model and results files. */
export function staleness(h5: UserH5, user: UserIndex, model: string | null, results: string | null, list: (dir: string) => readonly string[]): Staleness {
  const byModel = modelState(h5, user, model);
  const byResults = resultsState(h5, user, results, list);
  const zones: Record<string, ZoneState> = { meta: { stale: false, reason: "" } };
  for (const root of MODEL_ZONES) if (user.zones.includes(root)) zones[root] = byModel;
  // Python's `_mode_rows` reads the table: a zone that refuses is stale with the refusal.
  if (user.modes !== null && !user.modes.ok) zones["modes"] = { stale: true, reason: user.modes.reason };

  const entries: Record<string, ZoneState> = {};
  if (user.results.ok) {
    for (const e of user.results.value.entries) entries[e.name] = e.basis === "results" ? byResults : byModel;
    if (user.zones.includes("results")) zones["results"] = { stale: Object.values(entries).some((s) => s.stale), reason: "by @basis, per entry" };
  } else {
    zones["results"] = { stale: true, reason: user.results.reason };
  }

  let stale: string[] = [];
  if (user.combinations.ok) {
    if (user.zones.includes("combinations")) zones["combinations"] = { stale: false, reason: "" };
    stale = staleCombinations(user.combinations.value.rows, zones);
  } else {
    zones["combinations"] = { stale: true, reason: user.combinations.reason };
  }
  const modes = user.modes !== null && user.modes.ok ? modeRows(h5, user.modes.value, zones["modes"]!, results, list) : [];
  return { zones, results: entries, combinations: stale, modes };
}

/** A6's row rule over `/modes`; a row of a stale zone is stale (Python's `_mode_rows`). */
function modeRows(h5: UserH5, table: ModeTable, zone: ZoneState, results: string | null, list: (dir: string) => readonly string[]): RowVerdict[] {
  const all = (v: RowVerdict) => table.rows.map(() => v);
  if (zone.stale) return all(zone);
  if (results === null) return all({ stale: true, reason: "no results file to compare lambda with" });
  let lambdas: ReturnType<typeof recordedLambdas>;
  try {
    lambdas = recordedLambdas(h5, results, list);
  } catch (err) {
    // Python catches OSError, KeyError and ValueError: the file, or its modal record, cannot be read.
    if (!(err instanceof LambdasRefused)) throw err;
    return all({ stale: true, reason: `the results file's modes cannot be read: ${message(err)}` });
  }
  return rowVerdicts(table, lambdas);
}

/** Rows with a stale source, directly or through another combination. */
function staleCombinations(rows: readonly Combination[], zones: Record<string, ZoneState>): string[] {
  const stale = new Set<string>();
  for (const row of rows) {
    for (const term of row.terms) {
      const zone = TERM_ZONE[term.kind];
      if (zone !== undefined && zones[zone]?.stale) stale.add(row.name);
    }
  }
  let changed = true;
  while (changed) {
    changed = false;
    for (const row of rows) {
      if (!stale.has(row.name) && row.terms.some((t) => t.kind === "combination" && stale.has(t.source))) {
        stale.add(row.name);
        changed = true;
      }
    }
  }
  return rows.filter((r) => stale.has(r.name)).map((r) => r.name);
}
