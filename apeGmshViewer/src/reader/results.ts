// Read a native results.h5, lazily, with h5wasm and nothing else (V4a, #1707).
//
// The layout is architecture/h5-schema.md, "`results.h5`: the native results
// file" (V4s, #1706 / PR #1709), results zone 1.1.0, floor 1.0.0. This is the
// smallest read path that can be timed for the ADR 0112 D5 decision: it lists
// stages, steps and time, reads one component at one step by hyperslab, one
// Gauss-point component at one step, and one node's whole history. It builds
// no store state and draws nothing.
//
// Rules this file keeps (the same as read.ts):
// - the version gate runs before any read; a file outside the window raises;
// - a required child that is missing, a shape that disagrees with the time
//   vector, a child this reader does not know, or a non-float component
//   raises a ResultsError, never a guessed default;
// - element categories the spike does not read (line_stations, nodal_forces,
//   fibers, layers) are listed in `unread`, not skipped silently;
// - `_element_index` is a FEM element id only for a `domain_capture` file
//   (an allow-list, #1711): any other or missing `source_type` gives the
//   typed "unverified" mapping, whose rows carry another field name so a
//   caller cannot join them to /model/elements by accident.
//
// One partition only: every writer in src/ writes `partition_0`; a file with
// several is refused ("unsupported"), not merged (the doc's union rule is a
// follow-up for the real reader).

import { NEUTRAL_TARGET, OPENSEES_TARGET, ZONE_FLOOR } from "./read.ts";

/** The results zone this reader was written against, and its floor (ADR 0113). */
export const RESULTS_TARGET = { major: 1, minor: 1 } as const;
export const RESULTS_FLOOR = { major: 1, minor: 0 } as const;

/** The one `source_type` whose `_element_index` is verified to hold FEM ids (#1711). */
const VERIFIED_ELEMENT_IDS = new Set(["domain_capture"]);

const STAGE_KINDS = new Set(["static", "transient", "mode"]);
const ELEMENT_CATEGORIES = new Set(["gauss_points", "line_stations", "nodal_forces", "fibers", "layers"]);
const GAUSS_INDEX = new Set(["_element_index", "_natural_coords", "_local_axes_quaternion"]);
const H5T_FLOAT = 1;

export type ResultsErrorKind = "open" | "version" | "layout" | "dtype" | "unsupported" | "argument";

export class ResultsError extends Error {
  readonly kind: ResultsErrorKind;
  constructor(kind: ResultsErrorKind, message: string) {
    super(message);
    this.kind = kind;
  }
}

/** The parts of h5wasm this reader uses (h5wasm/node, opened by path). */
export interface ResultsH5 {
  File: new (filename: string, mode?: "r") => RFile;
  Group: abstract new (...args: never[]) => RGroup;
  Dataset: abstract new (...args: never[]) => RDataset;
}
interface RNode {
  path: string;
  attrs: Record<string, { value: unknown }>;
}
interface RGroup extends RNode {
  keys(): string[];
  get(name: string): unknown;
}
interface RDataset extends RNode {
  shape: number[] | null;
  metadata: { type: number; size: number };
  value: unknown;
  slice(ranges: ([] | [number | null] | [number | null, number | null])[]): unknown;
}
interface RFile extends RGroup {
  file_id: bigint | number;
  close(): unknown;
}

export interface NodalInfo {
  path: string;
  /** Node ids, column order of every component (`_ids`, int64 in the file). */
  ids: Float64Array;
  components: string[];
}

export interface GaussGroupInfo {
  id: string;
  path: string;
  classTag: number;
  intRule: number;
  nElements: number;
  nGauss: number;
  components: string[];
}

export interface StageInfo {
  id: string;
  name: string;
  kind: "static" | "transient" | "mode";
  time: Float64Array;
  nSteps: number;
  partition: string;
  nodes: NodalInfo | null;
  gauss: GaussGroupInfo[];
  /** Element categories present in the file that this spike does not read. */
  unread: string[];
}

export type ElementIds =
  | { mapping: "fem_eid"; sourceType: string; ids: Float64Array }
  /** Rows of `_element_index` whose meaning is not verified: never a FEM id. */
  | { mapping: "unverified"; sourceType: string | null; reason: string; rawIndex: Float64Array };

export interface ResultsIndex {
  path: string;
  version: string;
  sourceType: string | null;
  /** Embedded zones below their floor (ADR 0113 D9): `/stages` reads, these do not. */
  unavailableZones: string[];
  stages: StageInfo[];
  warnings: string[];
  /**
   * `/meta/lineage` (results 1.1.0, h5-schema.md "how to pair a file with its
   * `model.h5`"): the hashes the pairing rule compares. Null when the file has
   * no `/meta/lineage` (results 1.0.0, or a 1.1.0 file whose writer has not
   * stamped it): such a file cannot be paired by content (V4b, #1728).
   */
  lineage: ResultsLineage | null;
  /**
   * Why the file looks unfinished, or null: a 1.1.0 file whose lineage has no
   * `results_hash`, which the writer stamps at close. It opens read-only; the
   * steps it lists are what was written so far (V4b seam "partial writes").
   */
  partial: string | null;
}

/** The `/meta/lineage` hashes, each a 32-character hex string or null when the writer did not stamp it. */
export interface ResultsLineage {
  femHash: string | null;
  modelHash: string | null;
  resultsHash: string | null;
}

/** Open a results.h5 and index it; reads go through the returned object. */
export function openResults(h5: ResultsH5, path: string): ResultsFile {
  const f = new h5.File(path, "r");
  // h5wasm does not throw when HDF5 cannot open a file (a truncated file, a
  // file that is not HDF5): it returns a File whose id is negative.
  if (Number(f.file_id) < 0) throw new ResultsError("open", `HDF5 could not open ${path} (truncated, or not an HDF5 file)`);
  try {
    return new ResultsFile(h5, f, path);
  } catch (err) {
    f.close();
    throw err;
  }
}

export class ResultsFile {
  readonly index: ResultsIndex;
  private readonly h5: ResultsH5;
  private readonly f: RFile;
  private readonly nodeColumn = new Map<string, Map<number, number>>();

  constructor(h5: ResultsH5, f: RFile, path: string) {
    this.h5 = h5;
    this.f = f;
    const warnings: string[] = [];
    const version = checkResultsVersion(f.attrs, warnings);
    const modelMeta = f.keys().includes("model") && this.group(f, "model").keys().includes("meta") ? this.group(this.group(f, "model"), "meta") : null;
    const unavailableZones = checkEmbeddedZones(f, modelMeta, warnings);
    const st = f.attrs["source_type"]?.value;
    const sourceType = typeof st === "string" ? st : null;
    const stagesGroup = this.group(f, "stages");
    const ids = sortStageIds(stagesGroup.keys());
    const lineage = this.readLineage(f);
    this.index = {
      path,
      version,
      sourceType,
      unavailableZones,
      stages: ids.map((id) => this.readStage(this.group(stagesGroup, id), id)),
      warnings,
      lineage,
      partial: partialOf(version, lineage),
    };
  }

  /** `/meta/lineage@{fem,model,results}_hash`; a stamp that is not a string is refused, never read as absent. */
  private readLineage(f: RFile): ResultsLineage | null {
    if (!f.keys().includes("meta")) return null;
    const meta = this.group(f, "meta");
    if (!meta.keys().includes("lineage")) return null;
    const g = this.group(meta, "lineage");
    const hash = (key: string): string | null => {
      const v = g.attrs[key]?.value;
      if (v === undefined) return null;
      if (typeof v !== "string" || v === "") throw new ResultsError("layout", `${g.path}@${key} is not a non-empty string`);
      return v;
    };
    return { femHash: hash("fem_hash"), modelHash: hash("model_hash"), resultsHash: hash("results_hash") };
  }

  close(): void {
    this.f.close();
  }

  stage(id: string): StageInfo {
    const s = this.index.stages.find((x) => x.id === id);
    if (!s) throw new ResultsError("argument", `no stage ${JSON.stringify(id)}; stages: ${this.index.stages.map((x) => x.id).join(", ")}`);
    return s;
  }

  /** One component per entry of `components`, each (N,) at `step`, by hyperslab. */
  readNodalStep(stageId: string, components: readonly string[], step: number): Float64Array[] {
    const s = this.stage(stageId);
    const nodes = this.requireNodes(s);
    checkStep(s, step);
    return components.map((c) => {
      const ds = this.component(nodes.path, nodes.components, c);
      return toFloat64(ds.slice([[step, step + 1], []]), ds.path);
    });
  }

  /** One Gauss-group component at `step`: (E * G,), element-major. */
  readGaussStep(stageId: string, groupId: string, component: string, step: number): Float64Array {
    const s = this.stage(stageId);
    const g = s.gauss.find((x) => x.id === groupId);
    if (!g) throw new ResultsError("argument", `stage ${stageId} has no gauss group ${JSON.stringify(groupId)}`);
    checkStep(s, step);
    const ds = this.component(g.path, g.components, component);
    return toFloat64(ds.slice([[step, step + 1], [], []]), ds.path);
  }

  /** Node `nodeId`'s whole history of one component: (T,). */
  readNodeHistory(stageId: string, component: string, nodeId: number): Float64Array {
    const s = this.stage(stageId);
    const nodes = this.requireNodes(s);
    let col = this.nodeColumn.get(nodes.path);
    if (!col) {
      col = new Map();
      nodes.ids.forEach((id, j) => col!.set(id, j));
      this.nodeColumn.set(nodes.path, col);
    }
    const j = col.get(nodeId);
    if (j === undefined) throw new ResultsError("argument", `stage ${stageId} records no node ${nodeId}`);
    const ds = this.component(nodes.path, nodes.components, component);
    return toFloat64(ds.slice([[], [j, j + 1]]), ds.path);
  }

  /** A gauss group's element rows, with their verified-or-not mapping (#1711). */
  elementIds(stageId: string, groupId: string): ElementIds {
    const s = this.stage(stageId);
    const g = s.gauss.find((x) => x.id === groupId);
    if (!g) throw new ResultsError("argument", `stage ${stageId} has no gauss group ${JSON.stringify(groupId)}`);
    const rows = intsAsNumbers(this.dataset(this.group(this.f, g.path), "_element_index"));
    const st = this.index.sourceType;
    if (st !== null && VERIFIED_ELEMENT_IDS.has(st)) return { mapping: "fem_eid", sourceType: st, ids: rows };
    return {
      mapping: "unverified",
      sourceType: st,
      reason:
        st === null
          ? "the file has no source_type: its _element_index is not verified to hold FEM element ids"
          : `source_type ${JSON.stringify(st)}: _element_index is verified to hold FEM element ids only for domain_capture files (#1711)`,
      rawIndex: rows,
    };
  }

  // -- index ---------------------------------------------------------------

  private readStage(g: RGroup, id: string): StageInfo {
    const name = g.attrs["name"]?.value;
    const kind = g.attrs["kind"]?.value;
    if (typeof name !== "string") throw new ResultsError("layout", `${g.path} has no string @name`);
    if (typeof kind !== "string" || !STAGE_KINDS.has(kind)) {
      throw new ResultsError("layout", `${g.path}@kind = ${JSON.stringify(kind)}; expected static, transient or mode`);
    }
    const timeDs = this.dataset(g, "time");
    if (timeDs.shape?.length !== 1) throw new ResultsError("layout", `${timeDs.path} is not 1-D`);
    const time = toFloat64(timeDs.value, timeDs.path);
    const T = time.length;
    for (const k of g.keys()) {
      if (k !== "time" && k !== "partitions") throw new ResultsError("layout", `${g.path}/${k}: not a stage child this reader knows`);
    }
    const parts = this.group(g, "partitions").keys();
    if (parts.length !== 1) {
      throw new ResultsError("unsupported", `${g.path}/partitions has ${parts.length} partitions (${parts.join(", ")}); this reader reads exactly one`);
    }
    const partition = parts[0]!;
    const pg = this.group(this.group(g, "partitions"), partition);
    let nodes: NodalInfo | null = null;
    const gauss: GaussGroupInfo[] = [];
    const unread: string[] = [];
    for (const child of pg.keys()) {
      if (child === "nodes") nodes = this.readNodes(this.group(pg, "nodes"), T);
      else if (child === "elements") {
        const eg = this.group(pg, "elements");
        for (const cat of eg.keys()) {
          if (cat === "_ids") continue; // reserved, written by no writer (h5-schema.md "What is absent")
          if (!ELEMENT_CATEGORIES.has(cat)) throw new ResultsError("layout", `${eg.path}/${cat}: not an element category this reader knows`);
          const cg = this.group(eg, cat);
          if (cat === "gauss_points") for (const gid of cg.keys()) gauss.push(this.readGauss(this.group(cg, gid), gid, T));
          else if (cg.keys().length > 0) unread.push(`${cat} (${cg.keys().length} groups)`);
        }
      } else throw new ResultsError("layout", `${pg.path}/${child}: not a partition child this reader knows`);
    }
    return { id, name, kind: kind as StageInfo["kind"], time, nSteps: T, partition, nodes, gauss, unread };
  }

  private readNodes(g: RGroup, T: number): NodalInfo {
    const ids = intsAsNumbers(this.dataset(g, "_ids"));
    const components: string[] = [];
    for (const k of g.keys()) {
      if (k === "_ids") continue;
      if (k.startsWith("_")) throw new ResultsError("layout", `${g.path}/${k}: not a nodal index dataset this reader knows`);
      const ds = this.dataset(g, k);
      expectShape(ds, [T, ids.length]);
      expectFloat(ds);
      components.push(k);
    }
    return { path: g.path, ids, components };
  }

  private readGauss(g: RGroup, id: string, T: number): GaussGroupInfo {
    const eidx = this.dataset(g, "_element_index");
    const nc = this.dataset(g, "_natural_coords");
    const E = eidx.shape?.length === 1 ? eidx.shape[0]! : -1;
    const G = nc.shape?.length === 2 ? nc.shape[0]! : -1;
    if (E < 0) throw new ResultsError("layout", `${eidx.path} is not 1-D`);
    if (G < 0) throw new ResultsError("layout", `${nc.path} is not 2-D`);
    const components: string[] = [];
    for (const k of g.keys()) {
      if (k.startsWith("_")) {
        if (!GAUSS_INDEX.has(k)) throw new ResultsError("layout", `${g.path}/${k}: not a gauss index dataset this reader knows`);
        continue;
      }
      const ds = this.dataset(g, k);
      expectShape(ds, [T, E, G]);
      expectFloat(ds);
      components.push(k);
    }
    return { id, path: g.path, classTag: intAttr(g, "class_tag"), intRule: intAttr(g, "int_rule"), nElements: E, nGauss: G, components };
  }

  // -- helpers ---------------------------------------------------------------

  private requireNodes(s: StageInfo): NodalInfo {
    if (!s.nodes) throw new ResultsError("argument", `stage ${s.id} has no nodal results`);
    return s.nodes;
  }

  private component(groupPath: string, known: string[], name: string): RDataset {
    if (!known.includes(name)) throw new ResultsError("argument", `${groupPath} has no component ${JSON.stringify(name)}; it has ${known.join(", ")}`);
    return this.dataset(this.group(this.f, groupPath), name);
  }

  private group(parent: RGroup, name: string): RGroup {
    const path = name.startsWith("/") ? name : parent.path === "/" ? `/${name}` : `${parent.path}/${name}`;
    if (!name.startsWith("/") && !parent.keys().includes(name)) throw new ResultsError("layout", `missing group ${path}`);
    const g = name.startsWith("/") ? this.f.get(name) : parent.get(name);
    if (!(g instanceof this.h5.Group)) throw new ResultsError("layout", `${path} is not a group`);
    return g as unknown as RGroup;
  }

  private dataset(parent: RGroup, name: string): RDataset {
    const path = parent.path === "/" ? `/${name}` : `${parent.path}/${name}`;
    if (!parent.keys().includes(name)) throw new ResultsError("layout", `missing dataset ${path}`);
    const d = parent.get(name);
    if (!(d instanceof this.h5.Dataset)) throw new ResultsError("layout", `${path} is not a dataset`);
    return d as unknown as RDataset;
  }
}

/**
 * The results-zone gate (ADR 0113, h5-schema.md "Root attributes"): the root
 * `results_schema_version`, else the envelope `schema_version`; neither is
 * "not a native results.h5". Another major is refused (below the floor or
 * newer); a newer minor opens with a warning, as the app's other zones do.
 */
export function checkResultsVersion(attrs: Record<string, { value: unknown }>, warnings: string[]): string {
  let raw = attrs["results_schema_version"]?.value;
  let key = "results_schema_version";
  if (raw === undefined) {
    raw = attrs["schema_version"]?.value;
    key = "schema_version";
  }
  if (raw === undefined) throw new ResultsError("version", "no results_schema_version or schema_version root attribute: not a native apeGmsh results.h5");
  const { major, minor } = parseSchemaVersion(raw, key);
  const stamp = String(raw);
  const floor = `${RESULTS_FLOOR.major}.${RESULTS_FLOOR.minor}.0`;
  if (major < RESULTS_FLOOR.major || (major === RESULTS_FLOOR.major && minor < RESULTS_FLOOR.minor)) {
    throw new ResultsError("version", `${key} ${stamp}: below the results floor ${floor}`);
  }
  if (major > RESULTS_TARGET.major) {
    throw new ResultsError("version", `${key} ${stamp}: a newer major than this app reads (${RESULTS_TARGET.major}.x)`);
  }
  if (minor > RESULTS_TARGET.minor) {
    warnings.push(`${key} ${stamp} is newer than this app (${RESULTS_TARGET.major}.${RESULTS_TARGET.minor}.x): the file opens, and what that apeGmsh added is not shown`);
  }
  return stamp;
}

/** One Python `int()` part: optional whitespace and sign, digits with single `_` separators. */
const PY_INT = /^\s*[+-]?\d+(?:_\d+)*\s*$/;

/**
 * A version stamp parsed as Python's `SchemaVersion.parse` parses it
 * (`opensees/_internal/schema_version.py`): `"X.Y.Z"` or `"X.Y"` (patch 0;
 * pre-key envelope stamps such as `"1.0"` are two-part), each part an integer
 * as `int()` reads it. Anything else is refused.
 */
export function parseSchemaVersion(raw: unknown, key: string): { major: number; minor: number; patch: number } {
  if (typeof raw !== "string") throw new ResultsError("version", `${key} is not a string`);
  if (raw === "") throw new ResultsError("version", `${key} is empty`);
  const parts = raw.split(".");
  if (parts.length < 2 || parts.length > 3) throw new ResultsError("version", `${key} ${JSON.stringify(raw)} is not X.Y or X.Y.Z`);
  if (!parts.every((p) => PY_INT.test(p))) throw new ResultsError("version", `${key} ${JSON.stringify(raw)} has non-integer parts`);
  const n = (p: string) => Number(p.replace(/_/g, "").trim());
  return { major: n(parts[0]!), minor: n(parts[1]!), patch: parts.length === 3 ? n(parts[2]!) : 0 };
}

/**
 * The embedded model zones (ADR 0113 D9, h5-schema.md "The embedded model"):
 * the neutral zone when `/model/meta` is present, the opensees zone when
 * `/opensees` is. A newer major than this app reads refuses the whole file; a
 * zone below its floor leaves `/stages` readable and is reported in
 * `unavailable` with a warning; a newer minor opens with a warning (the app's
 * rule for its other zones, read.ts `checkZoneVersion`).
 */
export function checkEmbeddedZones(
  root: { attrs: Record<string, { value: unknown }>; keys(): string[] },
  modelMeta: { attrs: Record<string, { value: unknown }> } | null,
  warnings: string[],
): string[] {
  const unavailable: string[] = [];
  const zones: { zone: "neutral" | "opensees"; raw: unknown; key: string; target: { major: number; minor: number }; floor: number }[] = [];
  if (modelMeta) {
    const own = modelMeta.attrs["neutral_schema_version"]?.value;
    zones.push({
      zone: "neutral",
      raw: own ?? modelMeta.attrs["schema_version"]?.value,
      key: own !== undefined ? "/model/meta@neutral_schema_version" : "/model/meta@schema_version",
      target: NEUTRAL_TARGET,
      floor: ZONE_FLOOR.neutral,
    });
  }
  if (root.keys().includes("opensees")) {
    const sources: [string, unknown][] = [
      ["@opensees_schema_version", root.attrs["opensees_schema_version"]?.value],
      ["/model/meta@opensees_schema_version", modelMeta?.attrs["opensees_schema_version"]?.value],
      ["/model/meta@schema_version", modelMeta?.attrs["schema_version"]?.value],
    ];
    const hit = sources.find(([, v]) => v !== undefined);
    if (!hit) throw new ResultsError("version", "/opensees is embedded but no opensees_schema_version stamp was found");
    zones.push({ zone: "opensees", raw: hit[1], key: hit[0], target: OPENSEES_TARGET, floor: ZONE_FLOOR.opensees });
  }
  for (const z of zones) {
    if (z.raw === undefined) throw new ResultsError("version", `the embedded ${z.zone} zone has no version stamp`);
    const v = parseSchemaVersion(z.raw, z.key);
    const raw = String(z.raw);
    if (v.major > z.target.major) {
      throw new ResultsError("version", `${z.key} ${raw}: the embedded ${z.zone} zone is a newer major than this app reads (${z.target.major}.x)`);
    }
    if (v.major < z.target.major || v.minor < z.floor) {
      unavailable.push(z.zone);
      warnings.push(`${z.key} ${raw}: the embedded ${z.zone} zone is below its floor ${z.target.major}.${z.floor}.0; /stages still reads, the ${z.zone} zone is unavailable`);
    } else if (v.minor > z.target.minor) {
      warnings.push(`${z.key} ${raw} is newer than this app (${z.target.major}.${z.target.minor}.x): the file opens, and what that apeGmsh added is not shown`);
    }
  }
  return unavailable;
}

/**
 * Whether a file looks unfinished (V4b seam "partial writes"): from results
 * 1.1.0 on, `/meta/lineage` is always written and `results_hash` is stamped
 * when the writer closes the file, so a 1.1.0 file without either is still
 * being written (or its writer died). A 1.0.0 file predates the lineage.
 */
export function partialOf(version: string, lineage: ResultsLineage | null): string | null {
  const { major, minor } = parseSchemaVersion(version, "results version");
  if (major === 1 && minor < 1) return null;
  if (lineage === null) return `results ${version} has no /meta/lineage, which its writer always stamps: the file may still be being written`;
  if (lineage.resultsHash === null) return "/meta/lineage has no results_hash, which the writer stamps when it closes the file: it may still be being written";
  return null;
}

/** `stage_<int>` by the integer, then the other ids lexically (h5-schema.md). */
export function sortStageIds(ids: string[]): string[] {
  const n = (s: string) => (/^stage_(\d+)$/.exec(s) ? Number(/^stage_(\d+)$/.exec(s)![1]) : null);
  return [...ids].sort((a, b) => {
    const x = n(a);
    const y = n(b);
    if (x !== null && y !== null) return x - y;
    if (x !== null) return -1;
    if (y !== null) return 1;
    return a < b ? -1 : a > b ? 1 : 0;
  });
}

function checkStep(s: StageInfo, step: number): void {
  if (!Number.isInteger(step) || step < 0 || step >= s.nSteps) {
    throw new ResultsError("argument", `step ${step} is outside stage ${s.id} (0..${s.nSteps - 1})`);
  }
}

function expectShape(ds: RDataset, want: number[]): void {
  const got = ds.shape ?? [];
  if (got.length !== want.length || got.some((v, i) => v !== want[i])) {
    throw new ResultsError("layout", `${ds.path} has shape (${got.join(", ")}); expected (${want.join(", ")})`);
  }
}

/** A component must be floating point (the writer does not coerce: float32 and float64 both occur). */
function expectFloat(ds: RDataset): void {
  if (ds.metadata.type !== H5T_FLOAT) {
    throw new ResultsError("dtype", `${ds.path} is not a floating-point dataset (HDF5 class ${ds.metadata.type}, ${ds.metadata.size} bytes)`);
  }
}

/** float64 passes through; float32 is cast; anything else is refused. */
function toFloat64(v: unknown, where: string): Float64Array {
  if (v instanceof Float64Array) return v;
  if (v instanceof Float32Array) return Float64Array.from(v);
  throw new ResultsError("dtype", `${where}: read returned ${Object.prototype.toString.call(v)}, not float32 or float64`);
}

function intsAsNumbers(ds: RDataset): Float64Array {
  const v = ds.value;
  if (v instanceof BigInt64Array) {
    const out = new Float64Array(v.length);
    for (let i = 0; i < v.length; i++) {
      const x = v[i]!;
      if (x > BigInt(Number.MAX_SAFE_INTEGER) || x < BigInt(Number.MIN_SAFE_INTEGER)) throw new ResultsError("dtype", `${ds.path}[${i}] = ${x} is not a safe integer`);
      out[i] = Number(x);
    }
    return out;
  }
  if (v instanceof Int32Array || v instanceof Uint32Array || v instanceof Int16Array || v instanceof Int8Array) return Float64Array.from(v);
  throw new ResultsError("dtype", `${ds.path} is not an integer dataset`);
}

function intAttr(g: RGroup, key: string): number {
  const v = g.attrs[key]?.value;
  if (typeof v === "bigint") return Number(v);
  if (typeof v === "number" && Number.isInteger(v)) return v;
  throw new ResultsError("layout", `${g.path} has no integer @${key}`);
}
