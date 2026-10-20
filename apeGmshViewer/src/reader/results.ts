// Read a native results.h5, lazily, with h5wasm and nothing else (V4a, #1707).
//
// The layout is architecture/h5-schema.md, "`results.h5`: the native results
// file" (V4s, #1706 / PR #1709), results zone 1.2.0, floor 1.0.0. This is the
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
// - V4c2a: a gauss component whose element or Gauss axis disagrees with the
//   group's index datasets (a partial write), and every component of a group
//   with no Gauss point, is refused alone (`GaussGroupInfo.refused`);
// - `_element_index` is a FEM element id if and only if the root attribute
//   `element_index_space` is the scalar variable-length UTF-8 string "fem", the
//   encoding NativeWriter.stamp_element_index_space writes and the one
//   Python's NativeReader.element_index_verified accepts (h5-schema.md
//   §`element_index_space`, results 1.2.0; stamp-only, #1768, #1811). `source_type`
//   and the version never verify a file: an absent stamp, a non-string one,
//   or any other value gives the typed "unverified" mapping, whose rows carry
//   another field name so a caller cannot join them to /model/elements by
//   accident.
//
// A node's history column is found by binary search over `_ids`
// (ladrunoNodal.ts::nodeColumnLookup, #2358), not a Map from id to column: V8
// caps a Map at 2^24 entries, so a stage of more nodes threw an unnamed
// RangeError on its first node-history read. No Map or Set here is sized by the
// node or element count; the ones left hold fixed vocabularies, stage paths or
// class tags.
//
// One partition only: every writer in src/ writes `partition_0`; a file with
// several is refused ("unsupported"), not merged (the doc's union rule is a
// follow-up for the real reader).

import { LINE_STATION_COMPONENTS, type LineStationComponent, type UnitSystemId } from "../state/types.ts";
import { parseDeclaredUnits } from "../state/units.ts";
import { NEUTRAL_TARGET, OPENSEES_TARGET, programDecls, ZONE_FLOOR } from "./read.ts";
import type { ProgramDecls } from "../model/types.ts";
import { SIDE_CHANNEL_DECL_VERBS } from "../state/results.ts";
import { openH5, type H5FS } from "./h5open.ts";
import { gmshSimplexRows } from "./legacySimplex.ts";
import { uniqueNodeIds } from "./ladrunoChecks.ts";
import { nodeColumnLookup } from "./ladrunoNodal.ts";

/** The results zone this reader was written against, and its floor (ADR 0113). */
export const RESULTS_TARGET = { major: 1, minor: 2 } as const;
export const RESULTS_FLOOR = { major: 1, minor: 0 } as const;

/**
 * The root attribute that verifies element row ids, and its one verifying value
 * (h5-schema.md §`element_index_space`; Python `ELEMENT_INDEX_SPACES`). Any other
 * value, a non-string, or no attribute at all reads as unverified (#1768).
 * The stamp must also be a scalar, variable-length, UTF-8 string (#1811): h5wasm
 * strips a fixed-length string's padding and decodes an ASCII one, so S3, NUL-padded
 * S8 or variable-length ASCII "fem" all read as "fem" here, while Python's
 * `NativeReader.element_index_verified` rejects every one of them (#1810 review).
 */
const ELEMENT_INDEX_SPACE = "element_index_space";
const ELEMENT_INDEX_SPACE_FEM = "fem";

const STAGE_KINDS = new Set(["static", "transient", "mode"]);
const ELEMENT_CATEGORIES = new Set(["gauss_points", "line_stations", "nodal_forces", "fibers", "layers"]);
const GAUSS_INDEX = new Set(["_element_index", "_natural_coords", "_local_axes_quaternion"]);
const LINE_STATION_INDEX = new Set(["_element_index", "_station_natural_coord", "_local_axes_quaternion"]);
const H5T_FLOAT = 1;
const H5T_STRING = 3;
/** Why a native stage lists a node once: its one partition holds each node's column once (#2358). */
const NATIVE_NODE_ONCE = "a node is listed once per partition, and this reader reads one partition per stage";

/** HDF5 `H5T_CSET_UTF8`; h5wasm reports it as `metadata.cset` (`H5T_CSET_ASCII` is 0). */
const H5T_CSET_UTF8 = 1;

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
  readonly FS: H5FS | null;
  Group: abstract new (...args: never[]) => RGroup;
  Dataset: abstract new (...args: never[]) => RDataset;
}
interface RNode {
  path: string;
  attrs: Record<string, RAttr>;
}
interface RAttr {
  value: unknown;
  metadata: { type: number; vlen: boolean; cset?: number; shape: number[] | null };
}
interface RGroup extends RNode {
  keys(): string[];
  get(name: string): unknown;
}
interface RDataset extends RNode {
  shape: number[] | null;
  metadata: { type: number; size: number };
  /** a compound dataset's `[name, type]` members (read by `readProgram` only) */
  dtype: unknown;
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
  /**
   * Components whose value columns disagree with `_ids` (a partial write, V4c1
   * #1756): refused one by one, with the reason, and never listed in
   * `components`. A step axis that disagrees with `time` still refuses the file.
   */
  refused: { component: string; reason: string }[];
}

export interface GaussGroupInfo {
  id: string;
  path: string;
  classTag: number;
  intRule: number;
  nElements: number;
  nGauss: number;
  components: string[];
  /**
   * V4c2a: components refused one by one (a partial write: value shape that
   * disagrees with `_element_index` or `_natural_coords`), and every component
   * of a group with no Gauss point (`G == 0`). Never listed in `components`.
   */
  refused: { component: string; reason: string }[];
  /** the group carries `_local_axes_quaternion` (shells) */
  localAxes: boolean;
  /**
   * V4c2b: `_natural_coords`, one row (its parent-space coordinates) per
   * Gauss point; null when the dataset is not floating point (no
   * extrapolation is then offered: it reads "unsupported", never a guess).
   */
  naturalCoords: number[][] | null;
}

export interface LineStationGroupInfo {
  id: string;
  path: string;
  classTag: number;
  intRule: number;
  nElements: number;
  nStations: number;
  components: string[];
  refused: { component: string; reason: string }[];
  localAxes: boolean;
  /** `_station_natural_coord` (S,) in [-1, +1]. */
  stationNaturalCoord: Float64Array;
}

export type LocalAxesSource = "step_local_axes" | "model_local_axes";

/** One stage's multi-step (or single-step) line-station slab. */
export interface LadrunoLineStationSlab {
  component: LineStationComponent;
  /** Row-major (nSteps, nStationsTotal) station values. */
  values: Float64Array;
  nSteps: number;
  nStationsTotal: number;
  /** (nStationsTotal,) element id per station column. */
  elementIndex: Float64Array;
  /** (nStationsTotal,) natural coordinate xi in [-1, +1] per station column. */
  stationNaturalCoord: Float64Array;
  /** (nSteps,) time values. */
  time: Float64Array;
  /** (nStationsTotal * 4,) scalar-first quaternion (w, x, y, z) per station, or null when no frames recorded; NaN rows for unrecorded elements. */
  localAxesQuaternion: Float64Array | null;
  /** (nStationsTotal * 3,) reference z_rec vector per station from step/model local axes, or null when no frames recorded; NaN rows for unrecorded elements. */
  localAxesZRec: Float64Array | null;
  /** (nStationsTotal,) provenance of each station's local axes (`"step_local_axes"` or `"model_local_axes"`), or null when no frames recorded. */
  localAxesSources?: readonly (LocalAxesSource | null)[] | null;
}

/** One step's line-station data. */
export interface LadrunoLineStationStep {
  component: LineStationComponent;
  /** (nStationsTotal,) station values at `step`. */
  values: Float64Array;
  /** (nStationsTotal,) element id per station. */
  elementIndex: Float64Array;
  /** (nStationsTotal,) natural coordinate xi in [-1, +1] per station. */
  stationNaturalCoord: Float64Array;
  time: number;
  /** (nStationsTotal * 4,) scalar-first quaternion (w, x, y, z) per station, or null when no frames recorded. */
  localAxesQuaternion: Float64Array | null;
  /** (nStationsTotal * 3,) reference z_rec vector per station, or null when no frames recorded. */
  localAxesZRec: Float64Array | null;
  /** (nStationsTotal,) provenance of each station's local axes (`"step_local_axes"` or `"model_local_axes"`), or null when no frames recorded. */
  localAxesSources?: readonly (LocalAxesSource | null)[] | null;
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
  lineStations?: LineStationGroupInfo[];
  /** Element categories present in the file that this spike does not read. */
  unread: string[];
}

export type ElementIds =
  | { mapping: "fem_eid"; sourceType: string | null; ids: Float64Array }
  /** Rows of `_element_index` whose meaning is not verified: never a FEM id. */
  | { mapping: "unverified"; sourceType: string | null; reason: string; rawIndex: Float64Array };

export interface ResultsIndex {
  path: string;
  version: string;
  sourceType: string | null;
  /**
   * Root `@element_index_space` when it is a string, else null (absent or not a
   * string). Only `"fem"` verifies element row ids (h5-schema.md §`element_index_space`).
   */
  elementIndexSpace: string | null;
  /** Embedded zones below their floor (ADR 0113 D9): `/stages` reads, these do not. */
  unavailableZones: string[];
  /**
   * The embedded opensees zone's version stamp as `checkEmbeddedZones` reads
   * it (unhashed), or null when the file embeds no `/opensees`. The pairing
   * rule reads it to name the 2.31.0 `model_hash` change (V4h, #1856).
   */
  openseesVersion: string | null;
  /**
   * The embedded `/opensees/decls@program_decl` marker (V4h): "absent" when
   * `/opensees` has no `decls` group or no marker (h5-schema.md: a file below
   * 2.31.0 reads as "absent"); a non-string marker as its JSON; null when the
   * file embeds no `/opensees`.
   */
  programDecl: string | null;
  /**
   * The embedded `/opensees/program`'s side-channel calls and their
   * declaration proof (read.ts `programDecls`, #2189): the pairing names the
   * opensees 2.32.0 `model_hash` change from it. null when the file embeds no
   * `/opensees`, its decls marker is not "complete", it has no program, or a
   * warning says the table could not be read.
   */
  program: ProgramDecls | null;
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
  /** Consistent unit system declared in `/model/meta/@units` or `@units`, or null when undeclared. */
  units: UnitSystemId | null;
}

/** The `/meta/lineage` hashes, each a 32-character hex string or null when the writer did not stamp it. */
export interface ResultsLineage {
  femHash: string | null;
  modelHash: string | null;
  resultsHash: string | null;
}

/** Open a results.h5 and index it; reads go through the returned object. */
export function openResults(h5: ResultsH5, path: string): ResultsFile {
  const f = openH5(h5, path);
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
  private readonly nodeColumn = new Map<string, (nodeId: number) => number>(); // per nodal block: sized by the stage count, the lookup by typed arrays (#2358)
  /** Whether `_element_index` holds FEM ids, from the root stamp alone (#1768). */
  private readonly elementSpace: { verified: true } | { verified: false; reason: string };
  /** The path the file opened from, named in the index's warnings; and the class tags already warned about (#2274). */
  private readonly path: string;
  private readonly simplexWarned = new Set<number>();

  constructor(h5: ResultsH5, f: RFile, path: string) {
    this.h5 = h5;
    this.f = f;
    this.path = path;
    const warnings: string[] = [];
    const version = checkResultsVersion(f.attrs, warnings);
    const modelMeta = f.keys().includes("model") && this.group(f, "model").keys().includes("meta") ? this.group(this.group(f, "model"), "meta") : null;
    const embedded = checkEmbeddedZones(f, modelMeta, warnings);
    const st = f.attrs["source_type"]?.value;
    const sourceType = typeof st === "string" ? st : null;
    const eisAttr = f.attrs[ELEMENT_INDEX_SPACE];
    const eis = eisAttr?.value;
    this.elementSpace = elementSpaceVerdict(eisAttr);
    const stagesGroup = this.group(f, "stages");
    const ids = sortStageIds(stagesGroup.keys());
    const lineage = this.readLineage(f);
    const units = parseDeclaredUnits(modelMeta?.attrs["units"]?.value ?? f.attrs["units"]?.value);
    this.index = {
      path,
      version,
      sourceType,
      elementIndexSpace: typeof eis === "string" ? eis : null,
      unavailableZones: embedded.unavailable,
      openseesVersion: embedded.openseesVersion,
      programDecl: this.readProgramDecl(f),
      program: this.readProgram(f, warnings),
      stages: ids.map((id) => this.readStage(this.group(stagesGroup, id), id, warnings)),
      warnings,
      lineage,
      partial: partialOf(version, lineage),
      units,
    };
  }

  /** The embedded `/opensees/decls@program_decl` (see `ResultsIndex.programDecl`). */
  private readProgramDecl(f: RFile): string | null {
    if (!f.keys().includes("opensees")) return null;
    const ops = this.group(f, "opensees");
    if (!ops.keys().includes("decls")) return "absent";
    const marker = this.group(ops, "decls").attrs["program_decl"]?.value;
    return marker === undefined ? "absent" : typeof marker === "string" ? marker : JSON.stringify(marker);
  }

  /**
   * The embedded `/opensees/program` (see `ResultsIndex.program`), read only
   * when the embedded decls marker reads "complete" (the 2.32.0 rule needs
   * it); an unreadable table warns and reads as null.
   */
  private readProgram(f: RFile, warnings: string[]): ProgramDecls | null {
    if (this.readProgramDecl(f) !== "complete") return null;
    const ops = this.group(f, "opensees");
    if (!ops.keys().includes("program")) return null;
    const node = ops.get("program");
    if (!(node instanceof this.h5.Dataset)) {
      warnings.push(`${ops.path}/program is not a dataset; the results pairing does not read it`);
      return null;
    }
    const ds = node as RDataset;
    const columns = Array.isArray(ds.dtype) ? (ds.dtype as unknown[]).map((m) => String((m as unknown[])[0])) : [];
    const out = programDecls(ds.path, ds.attrs["methods"]?.value, columns, () => ds.value, SIDE_CHANNEL_DECL_VERBS);
    if (typeof out === "string") {
      warnings.push(`${out}; the results pairing does not read it`);
      return null;
    }
    return out;
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

  /**
   * Steps `k0` to `k1` (exclusive) of each entry of `components`, by one
   * hyperslab each: ((k1 - k0) * N,), step-major (V4c1: the history-range scan
   * reads a stage in chunks of steps, never the whole (T, N) at once).
   */
  readNodalRows(stageId: string, components: readonly string[], k0: number, k1: number): Float64Array[] {
    const s = this.stage(stageId);
    const nodes = this.requireNodes(s);
    if (!Number.isInteger(k0) || !Number.isInteger(k1) || k0 < 0 || k1 > s.nSteps || k0 >= k1) {
      throw new ResultsError("argument", `steps ${k0}..${k1} are not a range inside stage ${s.id} (0..${s.nSteps})`);
    }
    return components.map((c) => {
      const ds = this.component(nodes.path, nodes.components, c);
      return toFloat64(ds.slice([[k0, k1], []]), ds.path);
    });
  }

  /** Whether `_element_index` holds FEM element ids (the root stamp alone, #1768), and why not. */
  elementIndexVerdict(): { verified: true } | { verified: false; reason: string } {
    return this.elementSpace;
  }

  /**
   * Steps `k0` to `k1` (exclusive) of one Gauss-group component, by one
   * hyperslab: ((k1 - k0) * E * G,), step-major then element-major (V4c2a:
   * the element history scan reads a stage in chunks of steps).
   */
  readGaussRows(stageId: string, groupId: string, component: string, k0: number, k1: number): Float64Array {
    const s = this.stage(stageId);
    const g = s.gauss.find((x) => x.id === groupId);
    if (!g) throw new ResultsError("argument", `stage ${stageId} has no gauss group ${JSON.stringify(groupId)}`);
    if (!Number.isInteger(k0) || !Number.isInteger(k1) || k0 < 0 || k1 > s.nSteps || k0 >= k1) {
      throw new ResultsError("argument", `steps ${k0}..${k1} are not a range inside stage ${s.id} (0..${s.nSteps})`);
    }
    const ds = this.component(g.path, g.components, component);
    return toFloat64(ds.slice([[k0, k1], [], []]), ds.path);
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
    let columnOf = this.nodeColumn.get(nodes.path);
    if (!columnOf) {
      columnOf = nodeColumnLookup(nodes.ids, (m) => new ResultsError("layout", `${nodes.path}/_ids: ${m}`));
      this.nodeColumn.set(nodes.path, columnOf);
    }
    const j = columnOf(nodeId);
    if (j < 0) throw new ResultsError("argument", `stage ${stageId} records no node ${nodeId}`);
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
    const v = this.elementSpace;
    if (v.verified) return { mapping: "fem_eid", sourceType: st, ids: rows };
    return { mapping: "unverified", sourceType: st, reason: v.reason, rawIndex: rows };
  }

  /**
   * Canonical 1-D beam line-station components (`N, Vy, Vz, T, My, Mz, ...`)
   * recorded in `stageId` under `elements/line_stations/<group>` (verified files only).
   */
  availableLineStations(stageId: string): LineStationComponent[] {
    if (!this.elementSpace.verified) return [];
    const s = this.stage(stageId);
    const found = new Set<LineStationComponent>();
    for (const g of s.lineStations ?? []) {
      for (const comp of g.components) {
        const canon = normalizeLineStationComponent(comp);
        if (canon) found.add(canon);
      }
    }
    return LINE_STATION_COMPONENTS.filter((c) => found.has(c));
  }

  /**
   * Read one canonical (or raw) line-station component across all `line_stations`
   * groups in `stageId`, either for all steps or for a single `options.step`.
   */
  readLineStations(
    stageId: string,
    component: string,
    options?: { step?: number },
  ): LadrunoLineStationSlab {
    const canon = normalizeLineStationComponent(component);
    if (!canon) {
      throw new ResultsError("argument", `unknown line-station component ${JSON.stringify(component)}`);
    }
    if (!this.elementSpace.verified) {
      throw new ResultsError("unsupported", `line-station results of ${this.index.path} are refused: ${this.elementSpace.reason}`);
    }
    const s = this.stage(stageId);
    const step = options?.step;
    if (step !== undefined) checkStep(s, step);
    const nStepsRead = step !== undefined ? 1 : s.nSteps;
    const timeOut = step !== undefined ? Float64Array.of(s.time[step]!) : Float64Array.from(s.time);

    const groupChunks: {
      g: LineStationGroupInfo;
      eids: Float64Array;
      raw: Float64Array;
      quats: Float64Array | null;
    }[] = [];
    let nStationsTotal = 0;
    let anyLocalAxes = false;

    for (const g of s.lineStations ?? []) {
      const dsName = g.components.find((c) => normalizeLineStationComponent(c) === canon);
      if (!dsName) continue;
      const grp = this.group(this.f, g.path);
      const eids = intsAsNumbers(this.dataset(grp, "_element_index"));
      const ds = this.component(g.path, g.components, dsName);
      const raw =
        step !== undefined
          ? toFloat64(ds.slice([[step, step + 1], [], []]), ds.path)
          : toFloat64(ds.value, ds.path);
      let quats: Float64Array | null = null;
      if (g.localAxes && grp.keys().includes("_local_axes_quaternion")) {
        const qDs = this.dataset(grp, "_local_axes_quaternion");
        if (qDs.metadata.type === H5T_FLOAT && qDs.shape?.length === 2 && qDs.shape[0] === g.nElements && qDs.shape[1] === 4) {
          quats = toFloat64(qDs.value, qDs.path);
          anyLocalAxes = true;
        }
      }
      groupChunks.push({ g, eids, raw, quats });
      nStationsTotal += g.nElements * g.nStations;
    }

    if (nStationsTotal === 0) {
      return {
        component: canon,
        values: new Float64Array(0),
        nSteps: nStepsRead,
        nStationsTotal: 0,
        elementIndex: new Float64Array(0),
        stationNaturalCoord: new Float64Array(0),
        time: timeOut,
        localAxesQuaternion: null,
        localAxesZRec: null,
        localAxesSources: null,
      };
    }

    const elementIndex = new Float64Array(nStationsTotal);
    const stationNaturalCoord = new Float64Array(nStationsTotal);
    const values = new Float64Array(nStepsRead * nStationsTotal);
    const localAxesQuaternion = anyLocalAxes ? new Float64Array(nStationsTotal * 4) : null;
    const localAxesZRec = anyLocalAxes ? new Float64Array(nStationsTotal * 3) : null;
    const localAxesSources: (LocalAxesSource | null)[] | null = anyLocalAxes ? new Array(nStationsTotal).fill(null) : null;

    let colOffset = 0;
    for (const { g, eids, raw, quats } of groupChunks) {
      const E = g.nElements;
      const S = g.nStations;
      const groupCols = E * S;
      for (let e = 0; e < E; e++) {
        const eid = eids[e]!;
        let qw = Number.NaN, qx = Number.NaN, qy = Number.NaN, qz = Number.NaN;
        let zx = Number.NaN, zy = Number.NaN, zz = Number.NaN;
        let hasQuat = false;
        if (quats) {
          qw = quats[e * 4]!;
          qx = quats[e * 4 + 1]!;
          qy = quats[e * 4 + 2]!;
          qz = quats[e * 4 + 3]!;
          const zRec = quaternionToZRec(qw, qx, qy, qz);
          if (Number.isFinite(zRec[0])) {
            [zx, zy, zz] = zRec;
            hasQuat = true;
          }
        }
        for (let st = 0; st < S; st++) {
          const col = colOffset + e * S + st;
          elementIndex[col] = eid;
          stationNaturalCoord[col] = g.stationNaturalCoord[st]!;
          if (localAxesQuaternion && localAxesZRec && localAxesSources) {
            localAxesQuaternion[col * 4] = hasQuat ? qw : Number.NaN;
            localAxesQuaternion[col * 4 + 1] = hasQuat ? qx : Number.NaN;
            localAxesQuaternion[col * 4 + 2] = hasQuat ? qy : Number.NaN;
            localAxesQuaternion[col * 4 + 3] = hasQuat ? qz : Number.NaN;
            localAxesZRec[col * 3] = zx;
            localAxesZRec[col * 3 + 1] = zy;
            localAxesZRec[col * 3 + 2] = zz;
            localAxesSources[col] = hasQuat ? "model_local_axes" : null;
          }
        }
      }
      for (let t = 0; t < nStepsRead; t++) {
        const srcStart = t * groupCols;
        const dstStart = t * nStationsTotal + colOffset;
        values.set(raw.subarray(srcStart, srcStart + groupCols), dstStart);
      }
      colOffset += groupCols;
    }

    return {
      component: canon,
      values,
      nSteps: nStepsRead,
      nStationsTotal,
      elementIndex,
      stationNaturalCoord,
      time: timeOut,
      localAxesQuaternion,
      localAxesZRec,
      localAxesSources,
    };
  }

  /** Read one step of a canonical (or raw) line-station component in `stageId`. */
  readLineStationStep(stageId: string, component: string, step: number): LadrunoLineStationStep {
    const slab = this.readLineStations(stageId, component, { step });
    return {
      component: slab.component,
      values: slab.values,
      elementIndex: slab.elementIndex,
      stationNaturalCoord: slab.stationNaturalCoord,
      time: slab.time[0] ?? 0,
      localAxesQuaternion: slab.localAxesQuaternion,
      localAxesZRec: slab.localAxesZRec,
      localAxesSources: slab.localAxesSources ?? null,
    };
  }

  // -- index ---------------------------------------------------------------

  private readStage(g: RGroup, id: string, warnings: string[]): StageInfo {
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
    const lineStations: LineStationGroupInfo[] = [];
    const unread: string[] = [];
    for (const child of pg.keys()) {
      if (child === "nodes") nodes = this.readNodes(this.group(pg, "nodes"), T);
      else if (child === "elements") {
        const eg = this.group(pg, "elements");
        for (const cat of eg.keys()) {
          if (cat === "_ids") continue; // reserved, written by no writer (h5-schema.md "What is absent")
          if (!ELEMENT_CATEGORIES.has(cat)) throw new ResultsError("layout", `${eg.path}/${cat}: not an element category this reader knows`);
          const cg = this.group(eg, cat);
          if (cat === "gauss_points") for (const gid of cg.keys()) gauss.push(this.readGauss(this.group(cg, gid), gid, T, warnings));
          else if (cat === "line_stations" && this.elementSpace.verified) {
            for (const gid of cg.keys()) lineStations.push(this.readLineStationGroup(this.group(cg, gid), gid, T));
          } else if (cg.keys().length > 0) unread.push(`${cat} (${cg.keys().length} groups)`);
        }
      } else throw new ResultsError("layout", `${pg.path}/${child}: not a partition child this reader knows`);
    }
    return {
      id,
      name,
      kind: kind as StageInfo["kind"],
      time,
      nSteps: T,
      partition,
      nodes,
      gauss,
      ...(lineStations.length > 0 ? { lineStations } : {}),
      unread,
    };
  }

  private readNodes(g: RGroup, T: number): NodalInfo {
    // A node listed twice is refused at open, by name (#2358): readNodalStep would
    // return both columns, a contour keep the first and a history the other. This
    // reader reads exactly one partition (readStage refuses a stage with several), so
    // `_ids` is one chunk and a repeat is a repeat within one chunk, as #2360 refuses
    // in Python. Boundary nodes copied across partitions, which Python's native
    // reader dedups first-wins, never reach this reader.
    const idDs = this.dataset(g, "_ids");
    const ids = uniqueNodeIds(intsAsNumbers(idDs), this.path, idDs.path, (m) => new ResultsError("layout", m), NATIVE_NODE_ONCE);
    const components: string[] = [];
    const refused: { component: string; reason: string }[] = [];
    for (const k of g.keys()) {
      if (k === "_ids") continue;
      if (k.startsWith("_")) throw new ResultsError("layout", `${g.path}/${k}: not a nodal index dataset this reader knows`);
      const ds = this.dataset(g, k);
      // V4c1 seam "partial writes": a step axis that matches `time` with value
      // columns that do not match `_ids` refuses this component, not the file.
      const shape = ds.shape ?? [];
      if (shape.length === 2 && shape[0] === T && shape[1] !== ids.length) {
        refused.push({ component: k, reason: `${ds.path} has ${shape[1]} value columns and _ids has ${ids.length} node ids (a partial write)` });
        continue;
      }
      expectShape(ds, [T, ids.length]);
      expectFloat(ds);
      components.push(k);
    }
    return { path: g.path, ids, components, refused };
  }

  private readGauss(g: RGroup, id: string, T: number, warnings: string[]): GaussGroupInfo {
    const eidx = this.dataset(g, "_element_index");
    const nc = this.dataset(g, "_natural_coords");
    const E = eidx.shape?.length === 1 ? eidx.shape[0]! : -1;
    const G = nc.shape?.length === 2 ? nc.shape[0]! : -1;
    if (E < 0) throw new ResultsError("layout", `${eidx.path} is not 1-D`);
    if (G < 0) throw new ResultsError("layout", `${nc.path} is not 2-D`);
    const components: string[] = [];
    const refused: { component: string; reason: string }[] = [];
    let localAxes = false;
    for (const k of g.keys()) {
      if (k.startsWith("_")) {
        if (!GAUSS_INDEX.has(k)) throw new ResultsError("layout", `${g.path}/${k}: not a gauss index dataset this reader knows`);
        if (k === "_local_axes_quaternion") localAxes = true;
        continue;
      }
      const ds = this.dataset(g, k);
      // V4c2a seam "empty": a group with no Gauss point has no value to show.
      if (G === 0) {
        refused.push({ component: k, reason: `${nc.path} has 0 rows: the group ${id} has no Gauss point` });
        continue;
      }
      // V4c2a seam "partial writes" (as V4c1 does for nodal data): a step axis
      // that matches `time` with element or Gauss axes that disagree with the
      // index datasets refuses this component, not the file.
      const shape = ds.shape ?? [];
      if (shape.length === 3 && shape[0] === T && (shape[1] !== E || shape[2] !== G)) {
        refused.push({
          component: k,
          reason: `${ds.path} has shape (${shape.join(", ")}) and the group indexes ${E} elements x ${G} Gauss points (a partial write)`,
        });
        continue;
      }
      expectShape(ds, [T, E, G]);
      expectFloat(ds);
      components.push(k);
    }
    // V4c2b: the natural coordinates themselves (G x D, small), for the Gauss-to-node extrapolation.
    let naturalCoords: number[][] | null = null;
    if (nc.metadata.type === H5T_FLOAT) {
      const D = nc.shape![1]!;
      const flat = toFloat64(nc.value, nc.path);
      naturalCoords = Array.from({ length: G }, (_, r) => Array.from(flat.subarray(r * D, (r + 1) * D)));
    }
    // #2274: a triangle or tet group written before #2238 holds node-0-first rows; they read in gmsh
    // coordinates (legacySimplex.ts), or, matching neither table, offer none and warn once per class.
    const classTag = intAttr(g, "class_tag");
    if (naturalCoords) {
      const m = gmshSimplexRows(classTag, naturalCoords, `${this.path}: Gauss group ${g.path}`);
      naturalCoords = m.rows;
      if (m.refused !== null && !this.simplexWarned.has(classTag)) {
        this.simplexWarned.add(classTag);
        warnings.push(m.refused);
      }
    }
    return { id, path: g.path, classTag, intRule: intAttr(g, "int_rule"), nElements: E, nGauss: G, components, refused, localAxes, naturalCoords };
  }

  private readLineStationGroup(g: RGroup, id: string, T: number): LineStationGroupInfo {
    const eidx = this.dataset(g, "_element_index");
    const snc = this.dataset(g, "_station_natural_coord");
    const E = eidx.shape?.length === 1 ? eidx.shape[0]! : -1;
    const S = snc.shape?.length === 1 ? snc.shape[0]! : -1;
    if (E < 0) throw new ResultsError("layout", `${eidx.path} is not 1-D`);
    if (S < 0) throw new ResultsError("layout", `${snc.path} is not 1-D`);
    expectFloat(snc);
    const stationNaturalCoord = toFloat64(snc.value, snc.path);
    const components: string[] = [];
    const refused: { component: string; reason: string }[] = [];
    let localAxes = false;
    for (const k of g.keys()) {
      if (k.startsWith("_")) {
        if (!LINE_STATION_INDEX.has(k)) throw new ResultsError("layout", `${g.path}/${k}: not a line_stations index dataset this reader knows`);
        if (k === "_local_axes_quaternion") localAxes = true;
        continue;
      }
      const ds = this.dataset(g, k);
      if (S === 0 || E === 0) {
        refused.push({ component: k, reason: `${g.path} indexes ${E} elements x ${S} stations: nothing to read` });
        continue;
      }
      const shape = ds.shape ?? [];
      if (shape.length === 3 && shape[0] === T && (shape[1] !== E || shape[2] !== S)) {
        refused.push({
          component: k,
          reason: `${ds.path} has shape (${shape.join(", ")}) and the group indexes ${E} elements x ${S} stations (a partial write)`,
        });
        continue;
      }
      expectShape(ds, [T, E, S]);
      expectFloat(ds);
      components.push(k);
    }
    return {
      id,
      path: g.path,
      classTag: intAttr(g, "class_tag"),
      intRule: intAttr(g, "int_rule"),
      nElements: E,
      nStations: S,
      components,
      refused,
      localAxes,
      stationNaturalCoord,
    };
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
 * The element-id verdict from the root `@element_index_space` alone (#1768):
 * verified iff it is a scalar, variable-length, UTF-8 string equal to "fem".
 * `source_type` and the version play no part.
 */
function elementSpaceVerdict(attr: RAttr | undefined): { verified: true } | { verified: false; reason: string } {
  const value = attr?.value;
  const m = attr?.metadata;
  const scalarString = m !== undefined && m.type === H5T_STRING && Array.isArray(m.shape) && m.shape.length === 0;
  const stampForm = scalarString && m.vlen === true && m.cset === H5T_CSET_UTF8;
  if (stampForm && value === ELEMENT_INDEX_SPACE_FEM) return { verified: true };
  const why =
    attr === undefined
      ? `the file has no @${ELEMENT_INDEX_SPACE}`
      : typeof value !== "string" || !scalarString
        ? `@${ELEMENT_INDEX_SPACE} is not a string scalar`
        : m.vlen !== true
          ? `@${ELEMENT_INDEX_SPACE} is a fixed-length string, not the variable-length UTF-8 string the writer stamps`
          : m.cset !== H5T_CSET_UTF8
            ? `@${ELEMENT_INDEX_SPACE} is an ASCII-charset string, not the variable-length UTF-8 string the writer stamps`
            : `@${ELEMENT_INDEX_SPACE} = ${JSON.stringify(value)} is not "${ELEMENT_INDEX_SPACE_FEM}"`;
  return { verified: false, reason: `${why}: _element_index is not verified to hold FEM element ids (only @${ELEMENT_INDEX_SPACE} = "${ELEMENT_INDEX_SPACE_FEM}" verifies it)` };
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
 * rule for its other zones, read.ts `checkZoneVersion`). Returns the
 * unavailable zones and the opensees zone's stamp (null with no `/opensees`).
 */
export function checkEmbeddedZones(
  root: { attrs: Record<string, { value: unknown }>; keys(): string[] },
  modelMeta: { attrs: Record<string, { value: unknown }> } | null,
  warnings: string[],
): { unavailable: string[]; openseesVersion: string | null } {
  const unavailable: string[] = [];
  let openseesVersion: string | null = null;
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
    if (z.zone === "opensees") openseesVersion = raw;
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
  return { unavailable, openseesVersion };
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

const CANONICAL_COMPONENT_ALIASES: Readonly<Record<string, LineStationComponent>> = {
  N: "N",
  axial_force: "N",
  Vy: "Vy",
  shear_y: "Vy",
  Vz: "Vz",
  shear_z: "Vz",
  T: "T",
  torsion: "T",
  My: "My",
  bending_moment_y: "My",
  Mz: "Mz",
  bending_moment_z: "Mz",
  eps: "eps",
  axial_strain: "eps",
  gammaY: "gammaY",
  shear_strain_y: "gammaY",
  gammaZ: "gammaZ",
  shear_strain_z: "gammaZ",
  theta: "theta",
  torsional_strain: "theta",
  kappaY: "kappaY",
  curvature_y: "kappaY",
  kappaZ: "kappaZ",
  curvature_z: "kappaZ",
};

/** Normalize a short (`"Mz"`) or canonical (`"bending_moment_z"`) line-station component name. */
export function normalizeLineStationComponent(raw: string): LineStationComponent | null {
  return CANONICAL_COMPONENT_ALIASES[raw.trim()] ?? null;
}

/**
 * Convert a scalar-first unit quaternion `(w, x, y, z)` (global -> local,
 * schema §5, where rows `(0, 1, 2)` of the 3x3 matrix are `(x_local, y_local, z_local)`)
 * into row 2 (`z_local` in global coords), matching Python `LocalAxes.matrices`
 * (`src/apeGmsh/results/_slabs.py`) and `axes_from_quaternion` (`_beam_geometry.py`).
 */
export function quaternionToZRec(w: number, x: number, y: number, z: number): [number, number, number] {
  const norm = Math.hypot(w, x, y, z);
  if (!Number.isFinite(norm) || norm < 1e-15) return [Number.NaN, Number.NaN, Number.NaN];
  const wn = w / norm;
  const xn = x / norm;
  const yn = y / norm;
  const zn = z / norm;
  return [
    2 * (xn * zn - yn * wn),
    2 * (yn * zn + xn * wn),
    1 - 2 * (xn * xn + yn * yn),
  ];
}
