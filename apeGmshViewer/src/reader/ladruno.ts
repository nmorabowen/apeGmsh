// Read one fork-written `.ladruno`, lazily, with h5wasm (ADR 0122 D1, slice 4a, #2126).
//
// The layout is the fork's: `Ladruno_implementation/ladruno_schema_v1.md` on the
// fork's `ladruno` branch (§1 tree, §2 INFO, §2a MODEL_STAGE and KIND, §7.0 and
// §7.1 ON_NODES, §7b.1 INFO/PROVENANCE). ADR 0122 D4: the fork owns the format,
// apeGmsh owns only the lowercase PROVENANCE attributes. The index has the shape
// reader/results.ts gives main (the same `StageInfo` and `NodalInfo`), so the next
// slice can dispatch on the file kind. This slice wires nothing into main.
//
// Rules this file keeps:
// - the gate runs before any other read: `INFO@GENERATOR` is "Ladruno" and
//   `INFO@FORMAT_VERSION` is in LADRUNO_SUPPORTED_FORMAT_VERSIONS (Python's set
//   of the same name, src/apeGmsh/results/schema/_versions.py, pinned by
//   tests/test_app_ladruno_gate_drift.py); outside it a LadrunoError, never a guess;
// - one file only: a `<stem>.part-<N>.ladruno` is refused "unsupported" by its
//   name, and a file whose INFO says it is one part of several is refused too;
//   a part set is read by reader/ladrunoParts.ts (#2188), which opens each part `part`;
// - components are spelled as the file spells them, `<RESULT>/<label>` with the
//   label from the result's COMPONENTS attribute. ADR 0122 D7: no token table,
//   so nothing here maps `DISPLACEMENT/Ux` to an apeGmsh name;
// - a nodal result whose datasets disagree (DATA's step axis against TIME and
//   STEP, its node axis against ID, its component axis against COMPONENTS) is
//   refused by name in `NodalInfo.refused`, with the reason, never read;
// - Gauss-point element buckets are read by reader/ladrunoElements.ts (#2187); other
//   element results, envelopes, the domain and region buckets are listed in `unread`;
// - `KIND=eigen` is not a mode (Python `_ladruno.py::_KIND_MAP`): the fork labels
//   every stage of a recorder that asks for modesOfVibration `eigen`, so the stage
//   keeps its steps as a static stage, and each mode under
//   `ON_NODES/MODES_OF_VIBRATION(U|R)/DATA/STEP_<s>/MODE_<k>` becomes its own
//   stage of kind "mode", listed after its host, as Python's
//   `_mpco_nodal_io::discover_modes` and `mode_stage_layout` find them;
// - INFO/PROVENANCE is reported attribute by attribute: present (a scalar,
//   variable-length UTF-8 string, the §7b.1 encoding), absent, or malformed (any
//   other encoding, or a JSON value of the wrong shape), never coerced. Lineage
//   exists only when both hashes are present; pairing itself is the next slice.
//
// Browser-safe: no Node import.

import type { UnitSystemId } from "../state/types.ts";
import { parseDeclaredUnits, UNIT_SYSTEM_IDS } from "../state/units.ts";
import { elementIndexVerdictOf, intOf, LadrunoGauss, modelGroupFor, oneValue, parseCompNames, type ElementIndexVerdict } from "./ladrunoElements.ts";
import { spatialDimOf } from "./ladrunoInfo.ts";
import { readColumnsBySlab } from "./ladrunoNodal.ts";
import type { ElementIds, NodalInfo, ResultsH5, StageInfo } from "./results.ts";
import { openH5 } from "./h5open.ts";

/** `INFO@GENERATOR` of every file this reader opens (schema §2, "reader branch key"). */
export const LADRUNO_GENERATOR = "Ladruno";
/** The `INFO@FORMAT_VERSION` values this reader opens; equal to Python's `LADRUNO_SUPPORTED_FORMAT_VERSIONS`. */
export const LADRUNO_SUPPORTED_FORMAT_VERSIONS: readonly number[] = [1] as const;

/** A partitioned run's part file (schema "File identity", D5). */
const PART_FILE = /^(?<stem>.+?)\.part-(?<idx>\d+)\.ladruno$/;
const STAGE_PREFIX = "MODEL_STAGE[";
const STAGE_NAME = /^MODEL_STAGE\[(\d+)\]$/;
/** Ladruno `KIND` → stage kind; `eigen` is static (see the header). */
const KIND_MAP: Record<string, "static" | "transient"> = { static: "static", transient: "transient", eigen: "static" };
const MODAL_GROUPS = ["MODES_OF_VIBRATION(U)", "MODES_OF_VIBRATION(R)"];
/** RESULTS buckets this reader lists as unread (it reads ON_NODES). */
const UNREAD_BUCKETS = new Set(["ON_ELEMENTS", "ON_DOMAIN", "ON_REGIONS", "ENVELOPES"]);
const STAGE_CHILDREN = new Set(["MODEL", "RESULTS"]);

/** The consumer-owned attributes of schema §7b.1, in its order. */
export const PROVENANCE_ATTRS = ["model_hash", "fem_hash", "stage_names", "element_index_space", "unsupported", "writer", "units"] as const;
export type ProvenanceAttr = (typeof PROVENANCE_ATTRS)[number];
const PROVENANCE_JSON_LISTS = new Set<ProvenanceAttr>(["stage_names", "unsupported"]);

const H5T_INTEGER = 0;
const H5T_FLOAT = 1;
const H5T_STRING = 3;
/** HDF5 `H5T_CSET_UTF8`; h5wasm reports it as `metadata.cset` (`H5T_CSET_ASCII` is 0). */
const H5T_CSET_UTF8 = 1;

export type LadrunoErrorKind = "open" | "generator" | "format_version" | "layout" | "dtype" | "unsupported" | "argument";

export class LadrunoError extends Error {
  readonly kind: LadrunoErrorKind;
  constructor(kind: LadrunoErrorKind, message: string) {
    super(message);
    this.kind = kind;
  }
}

interface LNode {
  path: string;
  attrs: Record<string, LAttr>;
}
interface LAttr {
  value: unknown;
  metadata: { type: number; vlen: boolean; cset?: number; shape: number[] | null };
}
interface LGroup extends LNode {
  keys(): string[];
  get(name: string): unknown;
}
interface LDataset extends LNode {
  shape: number[] | null;
  metadata: { type: number; size: number };
  value: unknown;
  slice(ranges: ([] | [number | null] | [number | null, number | null])[]): unknown;
}
interface LFile extends LGroup {
  file_id: bigint | number;
  close(): unknown;
}

/** One attribute of INFO/PROVENANCE as the file holds it. */
export type ProvenanceFact =
  | { state: "present"; value: string }
  | { state: "absent" }
  /** not a scalar variable-length UTF-8 string, or (stage_names, unsupported) not a JSON array of strings */
  | { state: "malformed"; reason: string };

export type LadrunoProvenance =
  /** the file has no INFO/PROVENANCE group (the fork creates it empty at open) */
  | { state: "absent"; reason: string }
  | {
      state: "present";
      attrs: Record<ProvenanceAttr, ProvenanceFact>;
      /** `stage_names` decoded, or null when it is not present and well formed */
      stageNames: string[] | null;
      /** `unsupported` decoded, or null when it is not present and well formed */
      unsupported: string[] | null;
      /** `units` decoded when it is a valid `UnitSystemId`, else null */
      units: UnitSystemId | null;
      /** other attribute names in the group: UPPERCASE ones are the fork's (§7b.1 rule 4), lowercase ones unknown */
      other: string[];
    };

/** The pairing hashes, copied byte-equal from a model.h5's /meta/lineage by the stamp. */
export interface LadrunoLineage {
  modelHash: string;
  femHash: string;
}

/** One recorded mode: the stage that holds it and its eigen data (Python `ModeEntry`). */
export interface LadrunoMode {
  stageId: string;
  hostStageId: string;
  stepKey: string;
  modeKey: string;
  /** 1-based, as in a native results file */
  modeIndex: number;
  /** LAMBDA = omega^2 */
  eigenvalue: number;
  /** from LAMBDA, as Python's `eigen_scalars`; 0 for a non-positive eigenvalue (which warns) */
  frequencyHz: number;
  periodS: number;
}

export interface LadrunoIndex {
  path: string;
  generator: typeof LADRUNO_GENERATOR;
  formatVersion: number;
  /** `INFO@FORMAT_MINOR_VERSION`; absent means 0 (schema §2) */
  formatMinorVersion: number;
  /** `INFO@SPATIAL_DIM` (1, 2 or 3; any other value is refused, reader/ladrunoInfo.ts), null when absent */
  spatialDim: number | null;
  /** host stages (`stage_<i>`, MODEL_STAGE order) and, after each, its mode stages */
  stages: StageInfo[];
  /** the KIND each host stage carries in the file (`eigen` reads as static), by stage id */
  fileKinds: Record<string, string>;
  modes: LadrunoMode[];
  provenance: LadrunoProvenance;
  /** both hashes present and well formed, else null with `lineageReason` */
  lineage: LadrunoLineage | null;
  lineageReason: string | null;
  /** Consistent unit system stamped on `INFO/PROVENANCE@units`, or null when absent/malformed */
  units: UnitSystemId | null;
  /** INFO@PARTITION_ID, NUM_PARTITIONS, RUN_ID and RUN_ID_SCOPE (null when absent) of a file opened as one part, else null */
  partition: { id: number; count: number; runId: string | null; runIdScope: string | null } | null;
  warnings: string[];
}

/** Canonical line-station components (force resultants and generalized section deformations). */
export const LINE_STATION_COMPONENTS = [
  "N",
  "Vy",
  "Vz",
  "T",
  "My",
  "Mz",
  "eps",
  "gammaY",
  "gammaZ",
  "theta",
  "kappaY",
  "kappaZ",
] as const;

export type LineStationComponent = (typeof LINE_STATION_COMPONENTS)[number];

export type LocalAxesSource = "step_local_axes" | "model_local_axes";

/** One stage's multi-step (or single-step) line-station slab from `.ladruno`. */
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

/** One step's line-station data from `.ladruno`. */
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

/** Per-element local frames from `.ladruno` (`RESULTS/ON_ELEMENTS/LOCAL_AXES` and `MODEL/LOCAL_AXES`). */
export interface LadrunoLocalAxes {
  /** (nElements,) element IDs. */
  elementIds: Float64Array;
  /** (nElements * 4,) scalar-first unit quaternion (w, x, y, z) per element (identity [1,0,0,0] fallback when unrecorded). */
  quaternions: Float64Array;
  /** (nElements * 3,) local z_rec reference vector in global coords per element (NaN when unrecorded). */
  zRec: Float64Array;
  /** (nElements,) 1 when the file recorded a frame for this element, 0 when fallback. */
  hasRecorded: Uint8Array;
  /** (nElements,) provenance of each element's recorded frame (`"step_local_axes"`, `"model_local_axes"`, or `null` when unrecorded). */
  sources: (LocalAxesSource | null)[];
}
/** Where one component of one stage lives: a 3-D DATA[T, n, c] or a 2-D MODE_<k>[n, c]. */
interface Column {
  path: string;
  col: number;
  modal: boolean;
}

/**
 * Open a `.ladruno` and index it; reads go through the returned object. `part`
 * opens one file of a partitioned set, for reader/ladrunoParts.ts only: the set
 * is what the app reads, never one part alone (#2188).
 */
export function openLadruno(h5: ResultsH5, path: string, opts: { part?: boolean } = {}): LadrunoFile {
  const base = path.split(/[\\/]/).pop() ?? path;
  const part = PART_FILE.exec(base);
  if (part && !opts.part) {
    throw new LadrunoError(
      "unsupported",
      `${path} is part ${part.groups!.idx} of the partitioned set ${part.groups!.stem}.part-<N>.ladruno; this reader reads one file, not a part set`,
    );
  }
  const f = openH5(h5, path) as unknown as LFile;
  // h5wasm does not throw when HDF5 cannot open a file: it returns a File whose id is negative.
  if (Number(f.file_id) < 0) throw new LadrunoError("open", `HDF5 could not open ${path} (truncated, or not an HDF5 file)`);
  try {
    return new LadrunoFile(h5, f, path, opts.part === true);
  } catch (err) {
    f.close();
    throw err;
  }
}

export class LadrunoFile {
  readonly index: LadrunoIndex;
  private readonly h5: ResultsH5;
  private readonly f: LFile;
  private readonly columns = new Map<string, Map<string, Column>>();
  private readonly nodeColumn = new Map<string, Map<number, number>>();
  private readonly gauss: LadrunoGauss;

  constructor(h5: ResultsH5, f: LFile, path: string, part = false) {
    this.h5 = h5;
    this.f = f;
    this.gauss = new LadrunoGauss(h5, f, (kind, message) => new LadrunoError(kind, message));
    const warnings: string[] = [];
    const info = this.gate(path);
    const formatMinorVersion = optionalInt(info, "FORMAT_MINOR_VERSION") ?? 0;
    const formatVersion = optionalInt(info, "FORMAT_VERSION")!;
    const partitions = optionalInt(info, "NUM_PARTITIONS");
    const partitionId = optionalInt(info, "PARTITION_ID");
    const partitioned = optionalInt(info, "PARTITIONED") === 1 || (partitions !== null && partitions > 1);
    if (partitioned && !part) {
      throw new LadrunoError("unsupported", `${path}: INFO says this file is one part of a partitioned run (PARTITIONED, NUM_PARTITIONS = ${partitions}); this reader reads one file`);
    }
    if (part && (!partitioned || partitions === null || partitionId === null)) {
      throw new LadrunoError("layout", `${path}: named as a part file, but INFO@PARTITIONED, NUM_PARTITIONS and PARTITION_ID do not say it is one part of a partitioned run (schema §2)`);
    }
    const dim = spatialDimOf(info.attrs.SPATIAL_DIM, path, (m) => new LadrunoError("layout", m));
    for (const k of f.keys()) {
      if (k !== "INFO" && !k.startsWith(STAGE_PREFIX)) warnings.push(`/${k}: not a root group of the ladruno schema; not read`);
    }
    const stageNames = f.keys().filter((k) => k.startsWith(STAGE_PREFIX));
    for (const n of stageNames) {
      if (!STAGE_NAME.test(n)) throw new LadrunoError("layout", `/${n}: a MODEL_STAGE whose id is not an integer`);
    }
    stageNames.sort((a, b) => Number(STAGE_NAME.exec(a)![1]) - Number(STAGE_NAME.exec(b)![1]));
    const hosts = stageNames.map((name, i) => this.readHost(this.group(f, name), `stage_${i}`, warnings));
    const modes = this.layModes(hosts, warnings);
    const stages: StageInfo[] = [];
    for (const h of hosts) {
      stages.push(h.stage);
      for (const m of modes.stages) if (m.host === h.stage.id) stages.push(m.stage);
    }
    const provenance = this.readProvenance(info);
    const { lineage, reason } = lineageOf(provenance);
    this.index = {
      path,
      generator: LADRUNO_GENERATOR,
      formatVersion,
      formatMinorVersion,
      spatialDim: dim,
      stages,
      fileKinds: Object.fromEntries(hosts.map((h) => [h.stage.id, h.fileKind])),
      modes: modes.facts,
      provenance,
      lineage,
      lineageReason: reason,
      units: provenance.state === "present" ? provenance.units : null,
      partition: part ? { id: partitionId!, count: partitions!, runId: optionalText(info, "RUN_ID"), runIdScope: optionalText(info, "RUN_ID_SCOPE") } : null,
      warnings,
    };
  }

  /**
   * The `PARTITION_REDUCTION` text of the ON_NODES group that holds `component`
   * (schema §7.1), trimmed, or null when the group has none. Not checked here:
   * reader/ladrunoParts.ts refuses an absent or unknown value by name.
   */
  nodalReduction(stageId: string, component: string): string | null {
    const col = this.column(this.stage(stageId), component);
    const at = col.path.indexOf("/ON_NODES/");
    const group = this.f.get(col.path.slice(0, col.path.indexOf("/", at + "/ON_NODES/".length)));
    const attr = (group as unknown as LGroup | null)?.attrs["PARTITION_REDUCTION"];
    return attr === undefined ? null : String(oneValue(attr.value)).replace(/[\0\s]+$/, "");
  }

  close(): void {
    this.f.close();
  }

  stage(id: string): StageInfo {
    const s = this.index.stages.find((x) => x.id === id);
    if (!s) throw new LadrunoError("argument", `no stage ${JSON.stringify(id)}; stages: ${this.index.stages.map((x) => x.id).join(", ")}`);
    return s;
  }

  /** The mode a mode stage holds, or null for a host stage. */
  mode(stageId: string): LadrunoMode | null {
    this.stage(stageId);
    return this.index.modes.find((m) => m.stageId === stageId) ?? null;
  }

  /** One array per entry of `components` (`<RESULT>/<label>`), each (N,) at `step`: one hyperslab per DATA (reader/ladrunoNodal.ts). */
  readNodalStep(stageId: string, components: readonly string[], step: number): Float64Array[] {
    const s = this.stage(stageId);
    checkStep(s, step);
    return readColumnsBySlab(components.map((c) => this.column(s, c)), (col, c0, c1) => {
      const ds = this.datasetAt(col.path);
      return toFloat64(col.modal ? ds.slice([[], [c0, c1]]) : ds.slice([[step, step + 1], [], [c0, c1]]), ds.path);
    }, (m) => new LadrunoError("layout", m));
  }

  /** Steps `k0` to `k1` (exclusive) of each entry of `components`: ((k1 - k0) * N,), step-major. */
  readNodalRows(stageId: string, components: readonly string[], k0: number, k1: number): Float64Array[] {
    const s = this.stage(stageId);
    if (!Number.isInteger(k0) || !Number.isInteger(k1) || k0 < 0 || k1 > s.nSteps || k0 >= k1) {
      throw new LadrunoError("argument", `steps ${k0}..${k1} are not a range inside stage ${s.id} (0..${s.nSteps})`);
    }
    return components.map((c) => {
      const col = this.column(s, c);
      const ds = this.datasetAt(col.path);
      return toFloat64(col.modal ? ds.slice([[], [col.col, col.col + 1]]) : ds.slice([[k0, k1], [], [col.col, col.col + 1]]), ds.path);
    });
  }

  /** Node `nodeId`'s whole history of one component: (T,); a mode stage's is its one value. */
  readNodeHistory(stageId: string, component: string, nodeId: number): Float64Array {
    const s = this.stage(stageId);
    const nodes = s.nodes;
    if (!nodes) throw new LadrunoError("argument", `stage ${stageId} has no nodal results`);
    let map = this.nodeColumn.get(s.id);
    if (!map) {
      map = new Map();
      nodes.ids.forEach((id, j) => map!.set(id, j));
      this.nodeColumn.set(s.id, map);
    }
    const j = map.get(nodeId);
    if (j === undefined) throw new LadrunoError("argument", `stage ${stageId} records no node ${nodeId}`);
    const col = this.column(s, component);
    const ds = this.datasetAt(col.path);
    return toFloat64(col.modal ? ds.slice([[j, j + 1], [col.col, col.col + 1]]) : ds.slice([[], [j, j + 1], [col.col, col.col + 1]]), ds.path);
  }

  /** One Gauss-group component at `step`: (E * G,), element-major (reader/results.ts `readGaussStep`). */
  readGaussStep(stageId: string, groupId: string, component: string, step: number): Float64Array {
    checkStep(this.stage(stageId), step);
    return this.gauss.read(this.stage(stageId), groupId, component, step, step + 1);
  }

  /** Steps `k0` to `k1` (exclusive) of one Gauss-group component: ((k1 - k0) * E * G,), step-major then element-major. */
  readGaussRows(stageId: string, groupId: string, component: string, k0: number, k1: number): Float64Array {
    return this.gauss.read(this.stage(stageId), groupId, component, k0, k1);
  }

  /** A Gauss group's element rows: FEM ids only under a well-formed element_index_space = "fem" (#2079 Q6). */
  elementIds(stageId: string, groupId: string): ElementIds {
    return this.gauss.elementIds(this.stage(stageId).id, groupId, this.elementIndexVerdict());
  }

  elementIndexVerdict(): ElementIndexVerdict {
    const p = this.index.provenance;
    return elementIndexVerdictOf(p.state === "present" ? p.attrs.element_index_space : null);
  }

  // -- line stations & local axes (Escalón 2) ---------------------------------

  /**
   * Infer the file's spatial dimension: `INFO@SPATIAL_DIM` when present, else
   * from `MODEL/NODES/COORDINATES` column width (`shape[1]` in `{1, 2, 3}`),
   * matching Python `_file_spatial_dim` (#2081).
   */
  inferSpatialDim(): number | null {
    if (this.index.spatialDim !== null) return this.index.spatialDim;
    const dims = new Set<number>();
    for (const k of this.f.keys()) {
      if (!k.startsWith(STAGE_PREFIX)) continue;
      const st = this.group(this.f, k);
      if (!st.keys().includes("MODEL")) continue;
      const model = this.group(st, "MODEL");
      if (!model.keys().includes("NODES")) continue;
      const nodes = this.group(model, "NODES");
      if (!nodes.keys().includes("COORDINATES")) continue;
      const ds = this.dataset(nodes, "COORDINATES");
      const shape = ds.shape ?? [];
      if (shape.length === 2 && shape[0]! > 0) {
        const w = shape[1]!;
        if (w === 1 || w === 2 || w === 3) dims.add(w);
      }
    }
    if (dims.size === 1) return [...dims][0]!;
    return null;
  }

  /**
   * Canonical line-station components (`N`, `Vy`, `Vz`, `T`, `My`, `Mz`, `eps`,
   * `gammaY`, `gammaZ`, `theta`, `kappaY`, `kappaZ`) available in `stageId`.
   */
  availableLineStations(stageId: string, warningsOut?: string[]): LineStationComponent[] {
    const s = this.stage(stageId);
    if (s.kind === "mode") return [];
    const hostGrp = this.group(this.f, s.name);
    const onElements = this.optionalSubgroup(hostGrp, ["RESULTS", "ON_ELEMENTS"]);
    if (!onElements) return [];
    const ndm = this.inferSpatialDim();
    const found = new Set<LineStationComponent>();
    for (const token of onElements.keys()) {
      const rank = tokenRank(token);
      if (rank === null) continue;
      const tokenGrp = this.optionalGroup(onElements, token);
      if (!tokenGrp) continue;
      for (const key of tokenGrp.keys()) {
        const bucket = this.optionalGroup(tokenGrp, key);
        if (!bucket) continue;
        const parsed = this.parseBucketBlocks(bucket);
        if ("reason" in parsed) {
          if (token === "localForce" || token === "localForces") {
            warningsOut?.push(
              `localForce bucket ${JSON.stringify(key)} (element class ${JSON.stringify(classNameOfBucket(key) || "<unparsed>")}) is unreadable: ${parsed.reason} Its line stations are left out.`,
            );
          }
          continue;
        }
        const { blocks, width } = parsed;
        const gate = localForce2dGate(key, blocks, width, token, ndm);
        if (!gate.ok) {
          warningsOut?.push(`${gate.reason} Its line stations are left out.`);
          continue;
        }
        const elementStations = new Map<LineStationComponent, number[]>();
        for (const b of blocks) {
          for (const name of b.compNames) {
            if (b.level === 2 && rank === 0) {
              const c = sectionComponentOf(name, token, ndm);
              if (c !== null) found.add(c);
            } else if (b.level === 0 && rank >= 1) {
              const bc = elementStationComponentOf(name, token, gate.alias2d);
              if (bc !== null) {
                const list = elementStations.get(bc.component);
                const st = bc.station ?? 1;
                if (list) list.push(st);
                else elementStations.set(bc.component, [st]);
              }
            }
          }
        }
        for (const [comp, stations] of elementStations) {
          stations.sort((a, b) => a - b);
          if (token === "basicForce" || token === "basicForces") {
            const refusal = basicForceRefusal(key, comp, stations);
            if (refusal !== null) {
              warningsOut?.push(refusal);
              continue;
            }
          }
          found.add(comp);
        }
      }
    }
    return LINE_STATION_COMPONENTS.filter((c) => found.has(c));
  }

  /** Alias for `availableLineStations(stageId, warningsOut)`. */
  lineStationComponents(stageId: string, warningsOut?: string[]): LineStationComponent[] {
    return this.availableLineStations(stageId, warningsOut);
  }

  /**
   * Read a line-station slab from `stageId` for `component` (either all steps or a
   * single `step`), applying the rank 0 > rank 1 > rank 2 > rank 3 per-element
   * source precedence and `local_force_station_signs`.
   */
  readLineStations(
    stageId: string,
    component: string,
    options?: { elementIds?: ArrayLike<number>; step?: number; warningsOut?: string[] },
  ): LadrunoLineStationSlab {
    const s = this.stage(stageId);
    if (s.kind === "mode") {
      throw new LadrunoError("argument", `${this.index.path}: stage ${stageId} is a mode stage; it holds mode shapes at nodes only, so line-station component ${JSON.stringify(component)} cannot be read from it`);
    }
    const canon = normalizeLineStationComponent(component);
    if (!canon) {
      throw new LadrunoError("argument", `unknown line-station component ${JSON.stringify(component)}; expected one of ${LINE_STATION_COMPONENTS.join(", ")}`);
    }
    const step = options?.step;
    if (step !== undefined) checkStep(s, step);
    const hostGrp = this.group(this.f, s.name);
    const onElements = this.optionalSubgroup(hostGrp, ["RESULTS", "ON_ELEMENTS"]);
    if (!onElements || onElements.keys().length === 0) {
      throw new LadrunoError(
        "layout",
        `No element results in ${this.index.path} (stage ${stageId}): RESULTS/ON_ELEMENTS is ${onElements ? "empty" : "absent"}, so line-station component ${JSON.stringify(component)} cannot be read`,
      );
    }
    const modelElements = this.optionalSubgroup(hostGrp, ["MODEL", "ELEMENTS"]);
    const ndm = this.inferSpatialDim();
    const filterSet = options?.elementIds ? new Set<number>(Array.from(options.elementIds, Number)) : null;

    interface BucketPart {
      rank: 0 | 1 | 2 | 3;
      /** values per element: each entry is (nStepsRead * nSt) for one element */
      perElement: { eid: number; nSt: number; xi: Float64Array; values: Float64Array }[];
      nStepsRead: number;
    }
    const found: BucketPart[] = [];
    const nStepsTotal = s.nSteps;
    const nStepsRead = step !== undefined ? 1 : nStepsTotal;

    // Pass 1: Rank 0 (section.*, LEVELS == 2) buckets that actually resolve `canon`.
    const sectionServed = new Set<number>();
    for (const token of onElements.keys()) {
      const baseRank = tokenRank(token);
      if (baseRank !== 0) continue;
      const tokenGrp = this.optionalGroup(onElements, token);
      if (!tokenGrp) continue;
      for (const key of tokenGrp.keys()) {
        const bucket = this.optionalGroup(tokenGrp, key);
        if (!bucket) continue;
        const parsed = this.parseBucketBlocks(bucket);
        if ("reason" in parsed) continue;
        if (!parsed.blocks.some((b) => b.level === 2 && b.multiplicity > 0 && b.compNames.some((n) => sectionComponentOf(n, token, ndm) === canon))) {
          continue;
        }
        for (const eid of parsed.ids) sectionServed.add(eid);
        const part = this.readSectionBucketStations(parsed, canon, key, token, ndm, modelElements, filterSet, step, nStepsTotal);
        if (part) found.push({ rank: 0, perElement: part, nStepsRead });
      }
    }

    // Pass 2: Rank 1..3 (localForce, basicForce, axialForce) buckets.
    for (const token of onElements.keys()) {
      const baseRank = tokenRank(token);
      if (baseRank === null || baseRank === 0) continue;
      const tokenGrp = this.optionalGroup(onElements, token);
      if (!tokenGrp) continue;
      for (const key of tokenGrp.keys()) {
        const bucket = this.optionalGroup(tokenGrp, key);
        if (!bucket) continue;
        const parsed = this.parseBucketBlocks(bucket);
        if ("reason" in parsed) continue;
        const hasLevel2 = parsed.blocks.some((b) => b.level === 2);
        if (!hasLevel2) {
          const part = this.readElementBucketStations(
            parsed,
            canon,
            key,
            token,
            baseRank as 1 | 2 | 3,
            ndm,
            filterSet,
            sectionServed,
            step,
            nStepsTotal,
            options?.warningsOut,
          );
          if (part) found.push({ rank: baseRank as 1 | 2 | 3, perElement: part, nStepsRead });
        }
      }
    }

    // Per-element rank precedence: rank 0 > rank 1 > rank 2 > rank 3.
    const covered = new Set<number>();
    const chosen: { eid: number; nSt: number; xi: Float64Array; values: Float64Array }[] = [];
    for (const rank of [0, 1, 2, 3] as const) {
      const servedInRank = new Set<number>();
      for (const part of found) {
        if (part.rank !== rank) continue;
        for (const elem of part.perElement) {
          if (covered.has(elem.eid) || servedInRank.has(elem.eid)) continue;
          chosen.push(elem);
          servedInRank.add(elem.eid);
        }
      }
      for (const eid of servedInRank) covered.add(eid);
    }

    const timeSlice = step !== undefined ? new Float64Array([s.time[step] ?? 0]) : s.time;
    if (chosen.length === 0) {
      return {
        component: canon,
        values: new Float64Array(0),
        nSteps: nStepsRead,
        nStationsTotal: 0,
        elementIndex: new Float64Array(0),
        stationNaturalCoord: new Float64Array(0),
        time: timeSlice,
        localAxesQuaternion: null,
        localAxesZRec: null,
        localAxesSources: null,
      };
    }

    const nStationsTotal = chosen.reduce((acc, e) => acc + e.nSt, 0);
    const values = new Float64Array(nStepsRead * nStationsTotal);
    const elementIndex = new Float64Array(nStationsTotal);
    const stationNaturalCoord = new Float64Array(nStationsTotal);
    let colOffset = 0;
    for (const elem of chosen) {
      for (let st = 0; st < elem.nSt; st++) {
        elementIndex[colOffset + st] = elem.eid;
        stationNaturalCoord[colOffset + st] = elem.xi[st]!;
      }
      for (let t = 0; t < nStepsRead; t++) {
        const srcBase = t * elem.nSt;
        const dstBase = t * nStationsTotal + colOffset;
        for (let st = 0; st < elem.nSt; st++) {
          values[dstBase + st] = elem.values[srcBase + st]!;
        }
      }
      colOffset += elem.nSt;
    }

    const { localAxesQuaternion, localAxesZRec, localAxesSources } = this.stationLocalAxes(stageId, elementIndex, step);
    return {
      component: canon,
      values,
      nSteps: nStepsRead,
      nStationsTotal,
      elementIndex,
      stationNaturalCoord,
      time: timeSlice,
      localAxesQuaternion,
      localAxesZRec,
      localAxesSources,
    };
  }

  /** Convenience single-step line-station read at `step`. */
  readLineStationStep(
    stageId: string,
    component: string,
    step: number,
    options?: { elementIds?: ArrayLike<number>; warningsOut?: string[] },
  ): LadrunoLineStationStep {
    const slab = this.readLineStations(stageId, component, { ...options, step });
    return {
      component: slab.component,
      values: slab.values,
      elementIndex: slab.elementIndex,
      stationNaturalCoord: slab.stationNaturalCoord,
      time: slab.time[0] ?? 0,
      localAxesQuaternion: slab.localAxesQuaternion,
      localAxesZRec: slab.localAxesZRec,
      localAxesSources: slab.localAxesSources,
    };
  }

  /**
   * Per-element local frames from `RESULTS/ON_ELEMENTS/LOCAL_AXES` (at `step`, when
   * `options?.step !== undefined`) and `MODEL/LOCAL_AXES/<class>/{ID, FRAME}` (reference).
   */
  readLocalAxes(stageId: string, options?: { elementIds?: ArrayLike<number>; step?: number }): LadrunoLocalAxes {
    const s = this.stage(stageId);
    const hostId = this.mode(stageId)?.hostStageId ?? s.id;
    const hostStage = this.stage(hostId);
    const hostGrp = this.group(this.f, hostStage.name);

    const idToQuat = new Map<number, [number, number, number, number]>();
    const idToZRec = new Map<number, [number, number, number]>();
    const idToSource = new Map<number, LocalAxesSource>();

    // 1. Reference MODEL/LOCAL_AXES
    const modelLa = this.optionalSubgroup(hostGrp, ["MODEL", "LOCAL_AXES"]);
    if (modelLa) {
      this.collectModelLocalAxes(modelLa, idToQuat, idToZRec, idToSource);
    }

    // 2. Per-step RESULTS/ON_ELEMENTS/LOCAL_AXES (only when `options?.step !== undefined`; wins over MODEL/LOCAL_AXES)
    const step = options?.step;
    if (step !== undefined) {
      const stepLa = this.optionalSubgroup(hostGrp, ["RESULTS", "ON_ELEMENTS", "LOCAL_AXES"]);
      if (stepLa && step >= 0 && step < hostStage.nSteps) {
        this.collectStepLocalAxes(stepLa, step, idToQuat, idToZRec, idToSource);
      }
    }

    const outIds = options?.elementIds
      ? Float64Array.from(options.elementIds, Number)
      : Float64Array.from([...new Set([...idToQuat.keys(), ...idToZRec.keys()])].sort((a, b) => a - b));
    const n = outIds.length;
    const quaternions = new Float64Array(n * 4);
    const zRec = new Float64Array(n * 3);
    const hasRecorded = new Uint8Array(n);
    const sources: (LocalAxesSource | null)[] = new Array(n).fill(null);
    for (let k = 0; k < n; k++) {
      const eid = outIds[k]!;
      const q = idToQuat.get(eid);
      const z = idToZRec.get(eid);
      if (q) {
        quaternions[k * 4] = q[0];
        quaternions[k * 4 + 1] = q[1];
        quaternions[k * 4 + 2] = q[2];
        quaternions[k * 4 + 3] = q[3];
      } else {
        quaternions[k * 4] = 1;
        quaternions[k * 4 + 1] = 0;
        quaternions[k * 4 + 2] = 0;
        quaternions[k * 4 + 3] = 0;
      }
      if (z) {
        zRec[k * 3] = z[0];
        zRec[k * 3 + 1] = z[1];
        zRec[k * 3 + 2] = z[2];
        hasRecorded[k] = 1;
        sources[k] = idToSource.get(eid) ?? "model_local_axes";
      } else {
        zRec[k * 3] = Number.NaN;
        zRec[k * 3 + 1] = Number.NaN;
        zRec[k * 3 + 2] = Number.NaN;
        hasRecorded[k] = 0;
        sources[k] = null;
      }
    }
    return { elementIds: outIds, quaternions, zRec, hasRecorded, sources };
  }

  // -- the gate ---------------------------------------------------------------

  /** `INFO@GENERATOR` and `INFO@FORMAT_VERSION`, before any other read (schema §2). */
  private gate(path: string): LGroup {
    if (!this.f.keys().includes("INFO")) throw new LadrunoError("generator", `${path} has no INFO group: not a .ladruno written by the Ladruno recorder`);
    const info = this.group(this.f, "INFO");
    const gen = info.attrs["GENERATOR"];
    if (gen === undefined) throw new LadrunoError("generator", `${path} has no INFO@GENERATOR: not a .ladruno written by the Ladruno recorder`);
    const g = oneValue(gen.value);
    if (g !== LADRUNO_GENERATOR) {
      throw new LadrunoError("generator", `${path}: INFO@GENERATOR is ${JSON.stringify(g)}, not ${JSON.stringify(LADRUNO_GENERATOR)} (an .mpco or a foreign HDF5 file)`);
    }
    const fv = info.attrs["FORMAT_VERSION"];
    const v = fv === undefined ? undefined : intOf(oneValue(fv.value));
    if (v === undefined || v === null || !LADRUNO_SUPPORTED_FORMAT_VERSIONS.includes(v)) {
      const shown = fv === undefined ? "absent" : JSON.stringify(String(oneValue(fv.value)));
      throw new LadrunoError(
        "format_version",
        `${path}: INFO@FORMAT_VERSION is ${shown}; this app reads ${LADRUNO_SUPPORTED_FORMAT_VERSIONS.join(", ")} (a newer or older Ladruno format)`,
      );
    }
    return info;
  }

  // -- stages -----------------------------------------------------------------

  private readHost(g: LGroup, id: string, warnings: string[]): { stage: StageInfo; fileKind: string; onNodes: LGroup | null } {
    const kindAttr = g.attrs["KIND"];
    const kind = kindAttr === undefined ? undefined : oneValue(kindAttr.value);
    if (typeof kind !== "string" || !(kind in KIND_MAP)) {
      throw new LadrunoError("layout", `${g.path}@KIND = ${JSON.stringify(kind ?? null)}; expected static, transient or eigen (schema §2a)`);
    }
    for (const k of g.keys()) if (!STAGE_CHILDREN.has(k)) warnings.push(`${g.path}/${k}: not a stage child of the ladruno schema; not read`);
    const results = g.keys().includes("RESULTS") ? this.group(g, "RESULTS") : null;
    const unread: string[] = [];
    let onNodes: LGroup | null = null;
    let [onElements, elementsAt]: [LGroup | null, number] = [null, 0];
    if (results) {
      for (const b of results.keys()) {
        if (b === "ON_NODES") onNodes = this.group(results, b);
        else if (b === "ON_ELEMENTS") [onElements, elementsAt] = [this.group(results, b), unread.length];
        else if (UNREAD_BUCKETS.has(b)) {
          const bg = this.group(results, b);
          for (const r of bg.keys()) unread.push(`${b}/${r}`);
        } else unread.push(`${b} (not a RESULTS bucket this reader knows)`);
      }
    }
    const refused: NodalInfo["refused"] = [];
    const accepted: { name: string; group: LGroup; labels: string[]; ids: Float64Array; time: Float64Array }[] = [];
    if (onNodes) {
      for (const name of onNodes.keys()) {
        if (MODAL_GROUPS.includes(name)) continue;
        const r = this.checkResult(onNodes, name);
        if ("reason" in r) refused.push({ component: name, reason: r.reason });
        else accepted.push({ name, ...r });
      }
    }
    // The stage's time is its first readable nodal result's; one that disagrees is refused.
    const first = accepted[0];
    const time = first ? first.time : this.otherTime(results);
    const ids = first ? first.ids : new Float64Array(0);
    const cols = new Map<string, Column>();
    const components: string[] = [];
    for (const a of accepted) {
      if (!sameValues(a.ids, ids)) {
        refused.push({ component: a.name, reason: `${a.group.path}/ID is not the node set of ${first!.group.path}/ID (${a.ids.length} and ${ids.length} nodes)` });
        continue;
      }
      if (!sameValues(a.time, time)) {
        refused.push({ component: a.name, reason: `${a.group.path}/TIME is not the stage's time (${first!.group.path}/TIME)` });
        continue;
      }
      a.labels.forEach((label, c) => {
        const comp = `${a.name}/${label}`;
        components.push(comp);
        cols.set(comp, { path: `${a.group.path}/DATA`, col: c, modal: false });
      });
    }
    const elements = onElements ? this.gauss.indexStage(id, onElements, this.optionalSubgroup(g, ["MODEL", "ELEMENTS"]), time, warnings) : null;
    unread.splice(elementsAt, 0, ...(elements?.unread ?? []));
    const hasNodal = onNodes !== null && onNodes.keys().some((k) => !MODAL_GROUPS.includes(k));
    const nodes: NodalInfo | null = hasNodal ? { path: onNodes!.path, ids, components, refused } : null;
    this.columns.set(id, cols);
    const stage: StageInfo = {
      id,
      name: g.path.slice(1),
      kind: KIND_MAP[kind]!,
      time,
      nSteps: time.length,
      partition: "partition_0",
      nodes,
      gauss: elements?.gauss ?? [],
      unread,
    };
    return { stage, fileKind: kind, onNodes };
  }

  /** One ON_NODES result, or why it is refused (the seams "degenerate names" and "partial writes"). */
  private checkResult(onNodes: LGroup, name: string): { group: LGroup; labels: string[]; ids: Float64Array; time: Float64Array } | { reason: string } {
    const g = onNodes.get(name);
    const path = `${onNodes.path}/${name}`;
    if (!(g instanceof this.h5.Group)) return { reason: `${path} is not a group` };
    const grp = g as unknown as LGroup;
    const labels = componentLabels(grp);
    if (typeof labels === "string") return { reason: labels };
    for (const k of ["ID", "DATA", "TIME", "STEP"]) if (!grp.keys().includes(k)) return { reason: `${path} has no ${k}` };
    const data = grp.get("DATA");
    if (data instanceof this.h5.Group) return { reason: `${path}/DATA is a group of per-step datasets (the superseded layout of schema §7.0), which this reader does not read` };
    const ds = [grp.get("ID"), data, grp.get("TIME"), grp.get("STEP")];
    if (!ds.every((d) => d instanceof this.h5.Dataset)) return { reason: `${path}: ID, DATA, TIME and STEP must all be datasets` };
    const [idDs, dataDs, timeDs, stepDs] = ds as unknown as LDataset[];
    const idShape = idDs!.shape ?? [];
    if (!(idShape.length === 1 || (idShape.length === 2 && idShape[1] === 1))) return { reason: `${idDs!.path} has shape (${idShape.join(", ")}); expected (n, 1)` };
    if (idDs!.metadata.type !== H5T_INTEGER) return { reason: `${idDs!.path} is not an integer dataset` };
    const shape = dataDs!.shape ?? [];
    if (shape.length !== 3) return { reason: `${dataDs!.path} has shape (${shape.join(", ")}); expected (T, n, components)` };
    if (dataDs!.metadata.type !== H5T_FLOAT) return { reason: `${dataDs!.path} is not a floating-point dataset (HDF5 class ${dataDs!.metadata.type})` };
    if ((timeDs!.shape ?? []).length !== 1 || timeDs!.metadata.type !== H5T_FLOAT) return { reason: `${timeDs!.path} is not a 1-D floating-point dataset` };
    if ((stepDs!.shape ?? []).length !== 1 || stepDs!.metadata.type !== H5T_INTEGER) return { reason: `${stepDs!.path} is not a 1-D integer dataset` };
    const [T, n, c] = shape as [number, number, number];
    const nTime = timeDs!.shape![0]!;
    const nStep = stepDs!.shape![0]!;
    if (T !== nTime || T !== nStep) {
      return { reason: `${dataDs!.path} holds ${T} steps, TIME ${nTime} and STEP ${nStep}: the step axes disagree (a recorder stopped mid-step)` };
    }
    if (n !== idShape[0]) return { reason: `${dataDs!.path} has ${n} node rows and ${idDs!.path} ${idShape[0]} node ids (a partial write)` };
    if (c !== labels.length) return { reason: `${dataDs!.path} has ${c} component columns and COMPONENTS names ${labels.length} (${labels.join(",")})` };
    return { group: grp, labels, ids: intsAsNumbers(idDs!), time: toFloat64(timeDs!.value, timeDs!.path) };
  }

  /** The time axis of a stage with no readable nodal result: the first TIME under another bucket (Python `_time_for_group`). */
  private otherTime(results: LGroup | null): Float64Array {
    if (!results) return new Float64Array(0);
    for (const b of ["ON_ELEMENTS", "ON_DOMAIN", "ON_REGIONS"]) {
      if (!results.keys().includes(b)) continue;
      const t = this.findTime(this.group(results, b));
      if (t) return toFloat64(t.value, t.path);
    }
    return new Float64Array(0);
  }

  private findTime(g: LGroup): LDataset | null {
    for (const k of g.keys()) {
      const item = g.get(k);
      if (k === "TIME" && item instanceof this.h5.Dataset) return item as unknown as LDataset;
      if (item instanceof this.h5.Group) {
        const t = this.findTime(item as unknown as LGroup);
        if (t) return t;
      }
    }
    return null;
  }

  // -- modes ------------------------------------------------------------------

  /** Python `discover_modes` per host, then `mode_stage_layout` over the file. */
  private layModes(hosts: { stage: StageInfo; onNodes: LGroup | null }[], warnings: string[]): { stages: { host: string; stage: StageInfo }[]; facts: LadrunoMode[] } {
    const perHost = hosts.map((h) => (h.onNodes ? this.discoverModes(h.onNodes, warnings) : []));
    const nSets = perHost.reduce((n, sets) => n + sets.length, 0);
    const stages: { host: string; stage: StageInfo }[] = [];
    const facts: LadrunoMode[] = [];
    hosts.forEach((h, hi) => {
      const sets = perHost[hi]!;
      for (const set of sets) {
        for (const m of set.modes) {
          const id = sets.length === 1 ? `${h.stage.id}_mode_${m.index}` : `${h.stage.id}_${set.stepKey.toLowerCase()}_mode_${m.index}`;
          const name = nSets === 1 ? `mode_${m.index}` : `${h.stage.name}/${set.stepKey}/mode_${m.index}`;
          const [frequencyHz, periodS] = eigenScalars(m.lambda, m.index, `${h.stage.name}`, warnings);
          facts.push({ stageId: id, hostStageId: h.stage.id, stepKey: set.stepKey, modeKey: m.key, modeIndex: m.index, eigenvalue: m.lambda, frequencyHz, periodS });
          stages.push({ host: h.stage.id, stage: this.modeStage(h.onNodes!, id, name, set.stepKey, m.key) });
        }
      }
    });
    return { stages, facts };
  }

  /**
   * Every modal set under ON_NODES, ordered by step, modes by index. The
   * translational group is the reference; the rotational one, when both
   * exist, lists the same steps, modes and eigenvalues, or the file is
   * refused. A group without DATA, or a MODE_<k> without LAMBDA, is refused.
   */
  private discoverModes(onNodes: LGroup, warnings: string[]): { stepKey: string; modes: { key: string; index: number; lambda: number }[] }[] {
    const names = MODAL_GROUPS.filter((n) => onNodes.keys().includes(n));
    if (names.length === 0) return [];
    const perGroup = names.map((name) => {
      const g = this.group(onNodes, name);
      if (!g.keys().includes("DATA")) throw new LadrunoError("layout", `${g.path} has no DATA group; the modal record is incomplete`);
      const data = this.group(g, "DATA");
      const sets: { stepKey: string; modes: { key: string; index: number; lambda: number }[] }[] = [];
      for (const stepKey of keyed(data, "STEP_", warnings)) {
        const sg = this.group(data, stepKey);
        const modes = keyed(sg, "MODE_", warnings).map((key) => {
          const ds = this.dataset(sg, key);
          const lam = ds.attrs["LAMBDA"];
          const v = lam === undefined ? undefined : oneValue(lam.value);
          if (typeof v !== "number") throw new LadrunoError("layout", `${ds.path} has no numeric LAMBDA attribute; the eigenvalue of this mode is not recorded`);
          return { key, index: Number(key.slice("MODE_".length)) + 1, lambda: v };
        });
        if (modes.length > 0) sets.push({ stepKey, modes });
      }
      return sets;
    });
    const ref = JSON.stringify(perGroup[0]);
    names.slice(1).forEach((name, i) => {
      if (JSON.stringify(perGroup[i + 1]) !== ref) {
        throw new LadrunoError("layout", `${onNodes.path}/${name} lists different modes or eigenvalues than ${onNodes.path}/${names[0]}; the modal record is inconsistent`);
      }
    });
    return perGroup[0]!;
  }

  /** A mode stage: one step at time 0, the modal groups' columns as its components. */
  private modeStage(onNodes: LGroup, id: string, name: string, stepKey: string, modeKey: string): StageInfo {
    const refused: NodalInfo["refused"] = [];
    const cols = new Map<string, Column>();
    const components: string[] = [];
    let ids: Float64Array | null = null;
    let firstPath = "";
    for (const group of MODAL_GROUPS) {
      if (!onNodes.keys().includes(group)) continue;
      const g = this.group(onNodes, group);
      const labels = componentLabels(g);
      if (typeof labels === "string") {
        refused.push({ component: group, reason: labels });
        continue;
      }
      if (!g.keys().includes("ID")) {
        refused.push({ component: group, reason: `${g.path} has no ID` });
        continue;
      }
      const idDs = this.dataset(g, "ID");
      const idShape = idDs.shape ?? [];
      if (!(idShape.length === 1 || (idShape.length === 2 && idShape[1] === 1)) || idDs.metadata.type !== H5T_INTEGER) {
        refused.push({ component: group, reason: `${idDs.path} is not an (n, 1) integer dataset` });
        continue;
      }
      const ds = this.dataset(this.group(this.group(g, "DATA"), stepKey), modeKey);
      const shape = ds.shape ?? [];
      if (ds.metadata.type !== H5T_FLOAT) {
        refused.push({ component: group, reason: `${ds.path} is not a floating-point dataset` });
        continue;
      }
      if (shape.length !== 2 || shape[0] !== idShape[0] || shape[1] !== labels.length) {
        refused.push({ component: group, reason: `${ds.path} has shape (${shape.join(", ")}); ID has ${idShape[0]} nodes and COMPONENTS names ${labels.length} (a partial write)` });
        continue;
      }
      const these = intsAsNumbers(idDs);
      if (ids === null) {
        ids = these;
        firstPath = idDs.path;
      } else if (!sameValues(ids, these)) {
        refused.push({ component: group, reason: `${idDs.path} is not the node set of ${firstPath}` });
        continue;
      }
      labels.forEach((label, c) => {
        const comp = `${group}/${label}`;
        components.push(comp);
        cols.set(comp, { path: ds.path, col: c, modal: true });
      });
    }
    this.columns.set(id, cols);
    return {
      id,
      name,
      kind: "mode",
      time: new Float64Array([0]),
      nSteps: 1,
      partition: "partition_0",
      nodes: { path: onNodes.path, ids: ids ?? new Float64Array(0), components, refused },
      gauss: [],
      unread: [],
    };
  }

  // -- provenance -------------------------------------------------------------

  /** INFO/PROVENANCE, attribute by attribute (schema §7b.1). */
  private readProvenance(info: LGroup): LadrunoProvenance {
    if (!info.keys().includes("PROVENANCE")) {
      return { state: "absent", reason: "the file has no INFO/PROVENANCE group (the recorder creates it empty; this file was not written by a recorder that does)" };
    }
    const g = this.group(info, "PROVENANCE");
    const attrs = {} as Record<ProvenanceAttr, ProvenanceFact>;
    let stageNames: string[] | null = null;
    let unsupported: string[] | null = null;
    let units: UnitSystemId | null = null;
    for (const name of PROVENANCE_ATTRS) {
      const fact = provenanceFact(g.attrs[name], name);
      if (fact.state === "present" && PROVENANCE_JSON_LISTS.has(name)) {
        const list = jsonStringList(fact.value);
        if (list === null) {
          attrs[name] = { state: "malformed", reason: `@${name} = ${JSON.stringify(fact.value)} is not a JSON array of strings` };
          continue;
        }
        if (name === "stage_names") stageNames = list;
        else unsupported = list;
      }
      if (fact.state === "present" && name === "element_index_space" && fact.value !== "fem") {
        attrs[name] = { state: "malformed", reason: `@element_index_space = ${JSON.stringify(fact.value)}; §7b.1 allows "fem" or no attribute` };
        continue;
      }
      if (fact.state === "present" && name === "units") {
        const parsed = parseDeclaredUnits(fact.value);
        if (parsed === null) {
          attrs[name] = { state: "malformed", reason: `@units = ${JSON.stringify(fact.value)}; expected one of ${UNIT_SYSTEM_IDS.join(", ")}` };
          continue;
        }
        units = parsed;
      }
      if (fact.state === "present" && (name === "model_hash" || name === "fem_hash") && fact.value === "") {
        attrs[name] = { state: "malformed", reason: `@${name} is an empty string` };
        continue;
      }
      attrs[name] = fact;
    }
    const known = new Set<string>(PROVENANCE_ATTRS);
    const other = Object.keys(g.attrs).filter((k) => !known.has(k)).sort();
    return { state: "present", attrs, stageNames, unsupported, units, other };
  }

  // -- helpers ----------------------------------------------------------------

  private column(s: StageInfo, component: string): Column {
    const col = this.columns.get(s.id)?.get(component);
    if (!col) {
      const known = s.nodes?.components ?? [];
      throw new LadrunoError("argument", `stage ${s.id} has no component ${JSON.stringify(component)}; it has ${known.length > 0 ? known.join(", ") : "none"}`);
    }
    return col;
  }

  private datasetAt(path: string): LDataset {
    const d = this.f.get(path);
    if (!(d instanceof this.h5.Dataset)) throw new LadrunoError("layout", `${path} is not a dataset`);
    return d as unknown as LDataset;
  }

  private group(parent: LGroup, name: string): LGroup {
    const path = parent.path === "/" ? `/${name}` : `${parent.path}/${name}`;
    if (!parent.keys().includes(name)) throw new LadrunoError("layout", `missing group ${path}`);
    const g = parent.get(name);
    if (!(g instanceof this.h5.Group)) throw new LadrunoError("layout", `${path} is not a group`);
    return g as unknown as LGroup;
  }

  private dataset(parent: LGroup, name: string): LDataset {
    const path = parent.path === "/" ? `/${name}` : `${parent.path}/${name}`;
    if (!parent.keys().includes(name)) throw new LadrunoError("layout", `missing dataset ${path}`);
    const d = parent.get(name);
    if (!(d instanceof this.h5.Dataset)) throw new LadrunoError("layout", `${path} is not a dataset`);
    return d as unknown as LDataset;
  }

  private optionalGroup(parent: LGroup, name: string): LGroup | null {
    if (!parent.keys().includes(name)) return null;
    const g = parent.get(name);
    return g instanceof this.h5.Group ? (g as unknown as LGroup) : null;
  }

  private optionalSubgroup(parent: LGroup, path: readonly string[]): LGroup | null {
    let cur: LGroup | null = parent;
    for (const seg of path) {
      if (!cur) return null;
      cur = this.optionalGroup(cur, seg);
    }
    return cur;
  }

  /**
   * Parse a `.ladruno` element bucket (`COLUMN_MAP` + `COMP_NAMES`, or direct
   * `@COMPONENTS` + `@LEVELS`), plus its `ID` and `DATA` datasets.
   */
  private parseBucketBlocks(
    bucket: LGroup,
  ): { ids: Float64Array; dataDs: LDataset; width: number; blocks: BucketBlock[] } | { reason: string } {
    for (const k of ["ID", "DATA", "TIME", "STEP"]) {
      if (!bucket.keys().includes(k)) return { reason: `${bucket.path} has no ${k}` };
    }
    const idNode = bucket.get("ID");
    const dataNode = bucket.get("DATA");
    const timeNode = bucket.get("TIME");
    const stepNode = bucket.get("STEP");
    if (
      !(idNode instanceof this.h5.Dataset) ||
      !(dataNode instanceof this.h5.Dataset) ||
      !(timeNode instanceof this.h5.Dataset) ||
      !(stepNode instanceof this.h5.Dataset)
    ) {
      return { reason: `${bucket.path}: ID, DATA, TIME and STEP must all be datasets` };
    }
    const idDs = idNode as unknown as LDataset;
    const dataDs = dataNode as unknown as LDataset;
    const timeDs = timeNode as unknown as LDataset;
    const stepDs = stepNode as unknown as LDataset;
    const idShape = idDs.shape ?? [];
    if (!(idShape.length === 1 || (idShape.length === 2 && idShape[1] === 1)) || idDs.metadata.type !== H5T_INTEGER) {
      return { reason: `${idDs.path} is not an (n, 1) integer dataset` };
    }
    const shape = dataDs.shape ?? [];
    if (shape.length !== 3 || dataDs.metadata.type !== H5T_FLOAT) {
      return { reason: `${dataDs.path} is not a 3-D floating-point dataset` };
    }
    const [T, nE, width] = shape as [number, number, number];
    if (T !== (timeDs.shape?.[0] ?? -1) || T !== (stepDs.shape?.[0] ?? -1)) {
      return { reason: `${dataDs.path} step axis (${T}) disagrees with TIME/STEP` };
    }
    if (nE !== idShape[0]) {
      return { reason: `${dataDs.path} element axis (${nE}) disagrees with ID (${idShape[0]})` };
    }
    const ids = intsAsNumbers(idDs);

    if (bucket.keys().includes("COLUMN_MAP")) {
      const cm = this.optionalGroup(bucket, "COLUMN_MAP");
      if (!cm) return { reason: `${bucket.path}/COLUMN_MAP is not a group` };
      if (!cm.keys().includes("LEVELS") || !cm.keys().includes("GAUSS_ID")) {
        return { reason: `${cm.path} is missing LEVELS or GAUSS_ID` };
      }
      const levelsDs = this.dataset(cm, "LEVELS");
      const gaussDs = this.dataset(cm, "GAUSS_ID");
      const levels = intsAsNumbers(levelsDs);
      const gauss = intsAsNumbers(gaussDs);
      const nBlocks = levels.length;
      const lines = parseCompNames(cm.path, cm.attrs["COMP_NAMES"]?.value, nBlocks, "LEVELS");
      if (typeof lines === "string") return { reason: lines };
      const mult = cm.keys().includes("MULTIPLICITY") ? intsAsNumbers(this.dataset(cm, "MULTIPLICITY")) : null;
      const blocks: BucketBlock[] = [];
      let col = 0;
      for (let i = 0; i < nBlocks; i++) {
        const names = lines[i]!;
        const m = mult && i < mult.length ? Math.trunc(mult[i]!) : 1;
        blocks.push({
          level: Math.trunc(levels[i]!),
          gaussId: Math.trunc(gauss[i] ?? -1),
          compNames: names,
          colStart: col,
          multiplicity: m,
        });
        col += names.length * m;
      }
      if (col !== width) {
        return { reason: `COLUMN_MAP block widths sum to ${col} but DATA has ${width} columns — malformed .ladruno bucket.` };
      }
      return { ids, dataDs, width, blocks };
    }

    // Fallback for simplified buckets carrying `@COMPONENTS` and `@LEVELS` directly on `bucket`.
    const labels = componentLabels(bucket);
    if (typeof labels === "string") return { reason: labels };
    if (width % labels.length !== 0) {
      return { reason: `${bucket.path}/DATA width ${width} is not a multiple of COMPONENTS length ${labels.length}` };
    }
    const level = optionalInt(bucket, "LEVELS") ?? 0;
    const mult = width / labels.length;
    if (level === 2) {
      const gaussIds = bucket.keys().includes("GAUSS_ID") ? intsAsNumbers(this.dataset(bucket, "GAUSS_ID")) : null;
      const blocks: BucketBlock[] = [];
      for (let s = 0; s < mult; s++) {
        blocks.push({
          level: 2,
          gaussId: gaussIds && s < gaussIds.length ? Math.trunc(gaussIds[s]!) : s,
          compNames: labels,
          colStart: s * labels.length,
          multiplicity: 1,
        });
      }
      return { ids, dataDs, width, blocks };
    }
    return {
      ids,
      dataDs,
      width,
      blocks: [{ level, gaussId: -1, compNames: labels, colStart: 0, multiplicity: mult }],
    };
  }

  /** The first natural coordinate of each GP_PARAM row of the bucket's element group (`modelGroupFor`), or null. */
  private gpParamFor(modelElements: LGroup | null, bucketKey: string): Float64Array | null {
    const grp = modelGroupFor(modelElements, bucketKey, (n) => this.optionalGroup(modelElements!, n));
    const quad = grp ? this.optionalGroup(grp, "QUADRATURE") : null;
    if (!quad || !quad.keys().includes("GP_PARAM")) return null;
    const ds = this.dataset(quad, "GP_PARAM");
    const shape = ds.shape ?? [];
    const cols = shape.length === 2 ? shape[1]! : 1;
    const nRows = shape[0] ?? 0;
    if (nRows === 0 || cols === 0) return null;
    const raw = toFloat64(ds.value, ds.path);
    if (cols === 1) return raw;
    const firstCol = new Float64Array(nRows);
    for (let i = 0; i < nRows; i++) firstCol[i] = raw[i * cols]!;
    return firstCol;
  }

  private readSectionBucketStations(
    parsed: { ids: Float64Array; dataDs: LDataset; width: number; blocks: BucketBlock[] },
    component: LineStationComponent,
    bucketKey: string,
    token: string,
    ndm: number | null,
    modelElements: LGroup | null,
    filterSet: ReadonlySet<number> | null,
    step: number | undefined,
    nStepsTotal: number,
  ): { eid: number; nSt: number; xi: Float64Array; values: Float64Array }[] | null {
    const stationEntries: { gid: number; col: number; seq: number }[] = [];
    let seqIdx = 0;
    for (const b of parsed.blocks) {
      if (b.level !== 2 || b.multiplicity <= 0) continue;
      for (let off = 0; off < b.compNames.length; off++) {
        if (sectionComponentOf(b.compNames[off]!, token, ndm) === component) {
          if (b.multiplicity > 1) {
            for (let gp = 0; gp < b.multiplicity; gp++) {
              stationEntries.push({
                gid: seqIdx + gp,
                col: b.colStart + gp * b.compNames.length + off,
                seq: seqIdx + gp,
              });
            }
            seqIdx += b.multiplicity;
          } else {
            stationEntries.push({
              gid: b.gaussId,
              col: b.colStart + off,
              seq: seqIdx,
            });
            seqIdx++;
          }
          break;
        }
      }
    }
    if (stationEntries.length === 0) return null;

    const selectedRows: { row: number; eid: number }[] = [];
    for (let row = 0; row < parsed.ids.length; row++) {
      const eid = parsed.ids[row]!;
      if (!filterSet || filterSet.has(eid)) selectedRows.push({ row, eid });
    }
    if (selectedRows.length === 0) return null;

    const gpParam = this.gpParamFor(modelElements, bucketKey);
    const nSt = stationEntries.length;
    const distinctNonNegGids =
      stationEntries.every((s) => s.gid >= 0) &&
      new Set(stationEntries.map((s) => s.gid)).size === nSt;
    // Order stations by ascending GAUSS_ID when all GAUSS_IDs are non-negative and distinct; otherwise keep block order.
    const orderedEntries = [...stationEntries].sort((a, b) =>
      distinctNonNegGids ? a.gid - b.gid : a.seq - b.seq,
    );
    const gaussIds = orderedEntries.map((s) => s.gid);
    const fallback = stationXi(nSt);
    const allZeroBased =
      distinctNonNegGids &&
      gpParam !== null &&
      gaussIds.every((gid) => gid >= 0 && gid < gpParam.length);
    const allOneBased =
      distinctNonNegGids &&
      !allZeroBased &&
      gpParam !== null &&
      gaussIds.every((gid) => gid >= 1 && gid <= gpParam.length);
    const rawStations = orderedEntries.map((entry, sIdx) => ({
      gid: entry.gid,
      col: entry.col,
      xi: allZeroBased
        ? gpParam![entry.gid]!
        : allOneBased
          ? gpParam![entry.gid - 1]!
          : fallback[sIdx]!,
    }));
    // Sort stations by ascending xi so GP_PARAM order is always respected left-to-right.
    rawStations.sort((a, b) => a.xi - b.xi);
    const xi = Float64Array.from(rawStations.map((s) => s.xi));
    const cols = rawStations.map((s) => s.col);

    return this.sliceElementStationColumns(parsed.dataDs, parsed.ids.length, parsed.width, selectedRows, cols, xi, null, step, nStepsTotal);
  }

  private readElementBucketStations(
    parsed: { ids: Float64Array; dataDs: LDataset; width: number; blocks: BucketBlock[] },
    component: LineStationComponent,
    bucketKey: string,
    token: string,
    rank: 1 | 2 | 3,
    ndm: number | null,
    filterSet: ReadonlySet<number> | null,
    sectionServed: ReadonlySet<number>,
    step: number | undefined,
    nStepsTotal: number,
    warningsOut?: string[],
  ): { eid: number; nSt: number; xi: Float64Array; values: Float64Array }[] | null {
    const selectedRows: { row: number; eid: number }[] = [];
    for (let row = 0; row < parsed.ids.length; row++) {
      const eid = parsed.ids[row]!;
      if (!filterSet || filterSet.has(eid)) selectedRows.push({ row, eid });
    }
    const gate = localForce2dGate(bucketKey, parsed.blocks, parsed.width, token, ndm);
    if (!gate.ok) {
      if (selectedRows.length > 0) {
        const bare = selectedRows.map((r) => r.eid).filter((eid) => !sectionServed.has(eid));
        if (bare.length > 0) {
          throw new LadrunoError(
            "layout",
            `${gate.reason} (requested element(s) [${bare.slice(0, 10).join(", ")}] are in this bucket, with no section stations for ${JSON.stringify(component)})`,
          );
        }
      }
      return null;
    }

    const stationToCol = new Map<number, number>();
    for (const b of parsed.blocks) {
      if (b.level !== 0) continue;
      for (let off = 0; off < b.compNames.length; off++) {
        const bc = elementStationComponentOf(b.compNames[off]!, token, gate.alias2d);
        if (bc === null || bc.component !== component) continue;
        const st = bc.station ?? 1;
        stationToCol.set(st, b.colStart + off);
      }
    }
    if (stationToCol.size === 0 || selectedRows.length === 0) return null;

    const stations = [...stationToCol.keys()].sort((a, b) => a - b);
    if (rank === 2) {
      const refusal = basicForceRefusal(bucketKey, component, stations);
      if (refusal !== null) {
        warningsOut?.push(refusal);
        return null;
      }
    }
    const nSt = stations.length;
    const cols = stations.map((st) => stationToCol.get(st)!);
    const xi = stationXi(nSt);
    const signs = (rank === 1 || rank === 2) && nSt === 2 ? (LOCAL_FORCE_STATION_SIGNS[component] ?? null) : null;
    return this.sliceElementStationColumns(parsed.dataDs, parsed.ids.length, parsed.width, selectedRows, cols, xi, signs, step, nStepsTotal);
  }

  private sliceElementStationColumns(
    dataDs: LDataset,
    nElementsInBucket: number,
    width: number,
    selectedRows: readonly { row: number; eid: number }[],
    cols: readonly number[],
    xi: Float64Array,
    signs: readonly [number, number] | null,
    step: number | undefined,
    nStepsTotal: number,
  ): { eid: number; nSt: number; xi: Float64Array; values: Float64Array }[] {
    const nSt = cols.length;
    const nStepsRead = step !== undefined ? 1 : nStepsTotal;
    const raw =
      step !== undefined
        ? toFloat64(dataDs.slice([[step, step + 1], [], []]), dataDs.path)
        : toFloat64(dataDs.value, dataDs.path);
    const elemStride = width;
    const stepStride = nElementsInBucket * width;

    return selectedRows.map(({ row, eid }) => {
      const values = new Float64Array(nStepsRead * nSt);
      for (let t = 0; t < nStepsRead; t++) {
        const base = t * stepStride + row * elemStride;
        for (let s = 0; s < nSt; s++) {
          let v = raw[base + cols[s]!]!;
          if (signs) v *= signs[s]!;
          values[t * nSt + s] = v;
        }
      }
      return { eid, nSt, xi, values };
    });
  }

  private stationLocalAxes(
    stageId: string,
    elementIndex: Float64Array,
    step: number | undefined,
  ): {
    localAxesQuaternion: Float64Array | null;
    localAxesZRec: Float64Array | null;
    localAxesSources: (LocalAxesSource | null)[] | null;
  } {
    const recorded = this.readLocalAxes(stageId, { step });
    if (recorded.elementIds.length === 0) {
      return { localAxesQuaternion: null, localAxesZRec: null, localAxesSources: null };
    }
    const idToRow = new Map<number, number>();
    recorded.elementIds.forEach((eid, idx) => idToRow.set(eid, idx));
    const n = elementIndex.length;
    const quats = new Float64Array(n * 4);
    const zRec = new Float64Array(n * 3);
    const sources: (LocalAxesSource | null)[] = new Array(n).fill(null);
    for (let i = 0; i < n; i++) {
      const idx = idToRow.get(elementIndex[i]!);
      if (idx !== undefined && recorded.hasRecorded[idx]) {
        quats[i * 4] = recorded.quaternions[idx * 4]!;
        quats[i * 4 + 1] = recorded.quaternions[idx * 4 + 1]!;
        quats[i * 4 + 2] = recorded.quaternions[idx * 4 + 2]!;
        quats[i * 4 + 3] = recorded.quaternions[idx * 4 + 3]!;
        zRec[i * 3] = recorded.zRec[idx * 3]!;
        zRec[i * 3 + 1] = recorded.zRec[idx * 3 + 1]!;
        zRec[i * 3 + 2] = recorded.zRec[idx * 3 + 2]!;
        sources[i] = recorded.sources[idx] ?? "model_local_axes";
      } else {
        quats[i * 4] = Number.NaN;
        quats[i * 4 + 1] = Number.NaN;
        quats[i * 4 + 2] = Number.NaN;
        quats[i * 4 + 3] = Number.NaN;
        zRec[i * 3] = Number.NaN;
        zRec[i * 3 + 1] = Number.NaN;
        zRec[i * 3 + 2] = Number.NaN;
        sources[i] = null;
      }
    }
    return { localAxesQuaternion: quats, localAxesZRec: zRec, localAxesSources: sources };
  }

  private collectModelLocalAxes(
    modelLa: LGroup,
    idToQuat: Map<number, [number, number, number, number]>,
    idToZRec: Map<number, [number, number, number]>,
    idToSource: Map<number, LocalAxesSource>,
  ): void {
    const readSub = (sub: LGroup) => {
      const frameName = sub.keys().includes("FRAME") ? "FRAME" : sub.keys().includes("LOCAL_AXES") ? "LOCAL_AXES" : null;
      if (!sub.keys().includes("ID") || !frameName) return;
      const idDs = this.dataset(sub, "ID");
      const frameDs = this.dataset(sub, frameName);
      const ids = intsAsNumbers(idDs);
      const raw = toFloat64(frameDs.value, frameDs.path);
      const nE = ids.length;
      const frameShape = frameDs.shape ?? [];
      if (frameShape.length >= 2 && frameShape[0] !== nE) return;
      if (nE <= 0 || raw.length === 0 || raw.length % nE !== 0) return;
      const cols = raw.length / nE;
      for (let i = 0; i < nE; i++) {
        const eid = ids[i]!;
        const decoded = decodeLocalAxesRow(raw, i * cols, cols);
        if (decoded) {
          if (decoded.quat) idToQuat.set(eid, decoded.quat);
          else idToQuat.delete(eid);
          idToZRec.set(eid, decoded.zRec);
          idToSource.set(eid, "model_local_axes");
        }
      }
    };
    if (modelLa.keys().includes("ID")) {
      readSub(modelLa);
    }
    for (const k of modelLa.keys()) {
      const sub = this.optionalGroup(modelLa, k);
      if (sub) readSub(sub);
    }
  }

  private collectStepLocalAxes(
    stepLa: LGroup,
    step: number,
    idToQuat: Map<number, [number, number, number, number]>,
    idToZRec: Map<number, [number, number, number]>,
    idToSource: Map<number, LocalAxesSource>,
  ): void {
    const readBucket = (b: LGroup) => {
      if (!b.keys().includes("ID") || !b.keys().includes("DATA")) return;
      const idNode = b.get("ID");
      const dataNode = b.get("DATA");
      if (!(idNode instanceof this.h5.Dataset) || !(dataNode instanceof this.h5.Dataset)) return;
      const idDs = idNode as unknown as LDataset;
      const dataDs = dataNode as unknown as LDataset;
      const shape = dataDs.shape ?? [];
      if (shape.length !== 3 || step >= shape[0]!) return;
      const ids = intsAsNumbers(idDs);
      const nE = ids.length;
      if (nE <= 0 || shape[1] !== nE) return;
      const slice = toFloat64(dataDs.slice([[step, step + 1], [], []]), dataDs.path);
      if (slice.length === 0 || slice.length % nE !== 0) return;
      const cols = slice.length / nE;
      if (cols !== shape[2]) return;
      for (let i = 0; i < nE; i++) {
        const eid = ids[i]!;
        const decoded = decodeLocalAxesRow(slice, i * cols, cols);
        if (decoded) {
          if (decoded.quat) idToQuat.set(eid, decoded.quat);
          else idToQuat.delete(eid);
          idToZRec.set(eid, decoded.zRec);
          idToSource.set(eid, "step_local_axes");
        }
      }
    };
    if (stepLa.keys().includes("ID") && stepLa.keys().includes("DATA")) {
      readBucket(stepLa);
    }
    for (const k of stepLa.keys()) {
      const sub = this.optionalGroup(stepLa, k);
      if (sub) readBucket(sub);
    }
  }
}

/** One §7b.1 attribute: present only as a scalar, variable-length UTF-8 string. */
function provenanceFact(attr: LAttr | undefined, name: string): ProvenanceFact {
  if (attr === undefined) return { state: "absent" };
  const m = attr.metadata;
  const scalar = Array.isArray(m.shape) && m.shape.length === 0;
  // The HDF5 class first, then the shape: h5wasm returns a shape-(1,) string as an array, so the
  // value's JS type is checked last, once a scalar string is all it can be.
  if (m.type !== H5T_STRING) return { state: "malformed", reason: `@${name} is not a string (HDF5 class ${m.type})` };
  if (!scalar) return { state: "malformed", reason: `@${name} is a string array of shape (${(m.shape ?? []).join(", ")}), not a scalar` };
  if (m.vlen !== true) return { state: "malformed", reason: `@${name} is a fixed-length string, not the variable-length UTF-8 string §7b.1 specifies` };
  if (m.cset !== H5T_CSET_UTF8) return { state: "malformed", reason: `@${name} is an ASCII-charset string, not the variable-length UTF-8 string §7b.1 specifies` };
  if (typeof attr.value !== "string") return { state: "malformed", reason: `@${name} read as ${Object.prototype.toString.call(attr.value)}, not a string` };
  return { state: "present", value: attr.value };
}

/** Lineage only when both hashes are present and well formed; otherwise why not. */
function lineageOf(p: LadrunoProvenance): { lineage: LadrunoLineage | null; reason: string | null } {
  if (p.state === "absent") return { lineage: null, reason: p.reason };
  const m = p.attrs.model_hash;
  const f = p.attrs.fem_hash;
  if (m.state === "present" && f.state === "present") return { lineage: { modelHash: m.value, femHash: f.value }, reason: null };
  const why = (key: string, x: ProvenanceFact) => (x.state === "absent" ? `INFO/PROVENANCE has no @${key}` : x.state === "malformed" ? x.reason : null);
  const reasons = [why("model_hash", m), why("fem_hash", f)].filter((r): r is string => r !== null);
  return { lineage: null, reason: `${reasons.join("; ")}: the file is not stamped for pairing (schema §7b.1)` };
}

function jsonStringList(text: string): string[] | null {
  let v: unknown;
  try {
    v = JSON.parse(text);
  } catch {
    return null;
  }
  return Array.isArray(v) && v.every((x) => typeof x === "string") ? (v as string[]) : null;
}

/** A result's COMPONENTS labels, or why there are none (the seam "empty names"). */
function componentLabels(g: LGroup): string[] | string {
  const a = g.attrs["COMPONENTS"];
  if (a === undefined) return `${g.path} has no COMPONENTS attribute`;
  const v = oneValue(a.value);
  if (typeof v !== "string") return `${g.path}@COMPONENTS is not a string`;
  const labels = v.split(",").map((t) => t.trim());
  if (labels.length === 0 || labels.some((t) => t === "")) return `${g.path}@COMPONENTS = ${JSON.stringify(v)} has an empty label`;
  if (new Set(labels).size !== labels.length) return `${g.path}@COMPONENTS = ${JSON.stringify(v)} repeats a label`;
  return labels;
}

/** `<prefix><int>` children, sorted by the integer; any other child warns (Python `_keyed`). */
function keyed(g: LGroup, prefix: string, warnings: string[]): string[] {
  const out: string[] = [];
  for (const k of g.keys()) {
    if (!k.startsWith(prefix)) {
      warnings.push(`${g.path}/${k}: not a ${prefix}<k> child; not read`);
      continue;
    }
    if (!/^\d+$/.test(k.slice(prefix.length))) throw new LadrunoError("layout", `${g.path}/${k}: ${prefix} is not followed by an integer`);
    out.push(k);
  }
  return out.sort((a, b) => Number(a.slice(prefix.length)) - Number(b.slice(prefix.length)));
}

/** Python `eigen_scalars`: (frequency, period) from LAMBDA; a non-positive one warns and reads 0, 0. */
function eigenScalars(lambda: number, modeIndex: number, source: string, warnings: string[]): [number, number] {
  if (lambda > 0) {
    const omega = Math.sqrt(lambda);
    return [omega / (2 * Math.PI), (2 * Math.PI) / omega];
  }
  warnings.push(`${source}: mode ${modeIndex} has a non-positive eigenvalue (${lambda}); a spurious or unstable mode. Its frequency and period read 0`);
  return [0, 0];
}

function optionalInt(g: LGroup, key: string): number | null {
  const a = g.attrs[key];
  if (a === undefined) return null;
  const v = intOf(oneValue(a.value));
  if (v === null) throw new LadrunoError("layout", `${g.path}@${key} is not an integer`);
  return v;
}

/** A text attribute, trimmed of NULs and spaces as Python's `partition_manifest` does, or null when absent. */
const optionalText = (g: LGroup, key: string): string | null => (g.attrs[key] === undefined ? null : String(oneValue(g.attrs[key]!.value)).replace(/^[\0 ]+|[\0 ]+$/g, ""));

function checkStep(s: StageInfo, step: number): void {
  if (!Number.isInteger(step) || step < 0 || step >= s.nSteps) {
    throw new LadrunoError("argument", `step ${step} is outside stage ${s.id} (0..${s.nSteps - 1})`);
  }
}

function sameValues(a: Float64Array, b: Float64Array): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

/** float64 passes through; float32 is cast; anything else is refused. */
function toFloat64(v: unknown, where: string): Float64Array {
  if (v instanceof Float64Array) return v;
  if (v instanceof Float32Array) return Float64Array.from(v);
  throw new LadrunoError("dtype", `${where}: read returned ${Object.prototype.toString.call(v)}, not float32 or float64`);
}

function intsAsNumbers(ds: LDataset): Float64Array {
  const v = ds.value;
  if (v instanceof BigInt64Array) {
    const out = new Float64Array(v.length);
    for (let i = 0; i < v.length; i++) {
      const x = v[i]!;
      if (x > BigInt(Number.MAX_SAFE_INTEGER) || x < BigInt(Number.MIN_SAFE_INTEGER)) throw new LadrunoError("dtype", `${ds.path}[${i}] = ${x} is not a safe integer`);
      out[i] = Number(x);
    }
    return out;
  }
  if (v instanceof Int32Array || v instanceof Uint32Array || v instanceof Int16Array || v instanceof Int8Array) return Float64Array.from(v);
  throw new LadrunoError("dtype", `${ds.path} is not an integer dataset`);
}

// -- line-station & local-axes helpers ----------------------------------------

interface BucketBlock {
  level: number;
  gaussId: number;
  compNames: string[];
  colStart: number;
  multiplicity: number;
}

/**
 * Per-station sign multipliers mapping OpenSees 2-node beam `localForce` end
 * reactions `[end_1, end_2]` into internal-section forces at `[xi=-1, xi=+1]`
 * (`src/apeGmsh/opensees/_response_catalog.py::local_force_station_signs`).
 *
 * `end_1` (station 1): `{N: -1, Vy: +1, Vz: -1, T: -1, My: -1, Mz: -1}`
 * `end_2` (station 2): `{N: +1, Vy: -1, Vz: +1, T: +1, My: +1, Mz: +1}`
 */
export const LOCAL_FORCE_STATION_SIGNS: Readonly<Record<string, readonly [number, number]>> = {
  N: [-1, 1],
  Vy: [1, -1],
  Vz: [-1, 1],
  T: [-1, 1],
  My: [-1, 1],
  Mz: [-1, 1],
};

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

function decodeLocalAxesRow(
  raw: Float64Array,
  offset: number,
  cols: number,
): { quat: [number, number, number, number] | null; zRec: [number, number, number] } | null {
  if (cols === 4) {
    const w = raw[offset]!;
    const x = raw[offset + 1]!;
    const y = raw[offset + 2]!;
    const z = raw[offset + 3]!;
    if (!Number.isFinite(w) || !Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) return null;
    const zRec = quaternionToZRec(w, x, y, z);
    if (!Number.isFinite(zRec[0])) return null;
    return { quat: [w, x, y, z], zRec };
  }
  if (cols === 9) {
    // 3x3 row-major rotation matrix [x_local(3), y_local(3), z_local(3)]: row 2 is cols 6..8.
    const zx = raw[offset + 6]!;
    const zy = raw[offset + 7]!;
    const zz = raw[offset + 8]!;
    if (!Number.isFinite(zx) || !Number.isFinite(zy) || !Number.isFinite(zz)) return null;
    return { quat: null, zRec: [zx, zy, zz] };
  }
  if (cols === 3) {
    const zx = raw[offset]!;
    const zy = raw[offset + 1]!;
    const zz = raw[offset + 2]!;
    if (!Number.isFinite(zx) || !Number.isFinite(zy) || !Number.isFinite(zz)) return null;
    return { quat: null, zRec: [zx, zy, zz] };
  }
  return null;
}

/**
 * Source rank of an `ON_ELEMENTS` token:
 * - 0: `section.force`, `section.deformation`, or any `section.*` (non-fiber) bucket
 * - 1: `localForce` / `localForces`
 * - 2: `basicForce` / `basicForces`
 * - 3: `axialForce`
 * - null: not a line-station token (e.g. `globalForce`, `stress`, `LOCAL_AXES`)
 */
function tokenRank(token: string): 0 | 1 | 2 | 3 | null {
  if (token.includes(".fiber.")) return null;
  if (token === "section.force" || token === "section.deformation" || token.startsWith("section.")) return 0;
  if (token === "localForce" || token === "localForces") return 1;
  if (token === "basicForce" || token === "basicForces") return 2;
  if (token === "axialForce") return 3;
  return null;
}

const SECTION_FORCE_MAP: Readonly<Record<string, LineStationComponent>> = {
  P: "N",
  N: "N",
  Vy: "Vy",
  Vz: "Vz",
  T: "T",
  My: "My",
  Mz: "Mz",
};

const SECTION_FORCE_2D_MAP: Readonly<Record<string, LineStationComponent>> = {
  V: "Vy",
  M: "Mz",
};

const SECTION_DEFORM_MAP: Readonly<Record<string, LineStationComponent>> = {
  eps: "eps",
  epsilon: "eps",
  gammaY: "gammaY",
  gammaZ: "gammaZ",
  theta: "theta",
  kappaY: "kappaY",
  kappaZ: "kappaZ",
};

const SECTION_DEFORM_2D_MAP: Readonly<Record<string, LineStationComponent>> = {
  gamma: "gammaY",
  alpha: "theta",
  kappa: "kappaZ",
};

function sectionComponentOf(label: string, token: string, ndm: number | null): LineStationComponent | null {
  const t = label.trim();
  const allow2d = ndm === 2;
  if (token === "section.deformation") {
    return SECTION_DEFORM_MAP[t] ?? (allow2d ? SECTION_DEFORM_2D_MAP[t] : undefined) ?? null;
  }
  if (token === "section.force") {
    return SECTION_FORCE_MAP[t] ?? (allow2d ? SECTION_FORCE_2D_MAP[t] : undefined) ?? null;
  }
  return (
    SECTION_FORCE_MAP[t] ??
    SECTION_DEFORM_MAP[t] ??
    (allow2d ? (SECTION_FORCE_2D_MAP[t] ?? SECTION_DEFORM_2D_MAP[t]) : undefined) ??
    null
  );
}

const BEAM_BASE_MAP: Readonly<Record<string, LineStationComponent>> = {
  N: "N",
  Vy: "Vy",
  Vz: "Vz",
  T: "T",
  My: "My",
  Mz: "Mz",
};

const LOCAL_FORCE_2D_ALIAS: Readonly<Record<string, string>> = {
  V: "Vy",
  M: "Mz",
};

const LOCAL_FORCE_TOKEN_RE = /^(?<base>[A-Za-z]+)_?(?<station>\d+)$/;
const BEAM_TOKEN_RE = /^(?<base>[A-Za-z]+?)(?:_(?<station>\d+))?$/;
const LOCAL_FORCE_3D_BASES = new Set(["Vx", "Vy", "Vz", "Mx", "My", "Mz", "T"]);

function elementStationComponentOf(
  label: string,
  token: string,
  alias2d: boolean,
): { component: LineStationComponent; station: number | null } | null {
  let name = label.trim();
  if (token === "axialForce") {
    if (name === "N" || name === "P") return { component: "N", station: null };
    return null;
  }
  if (token === "localForce" || token === "localForces") {
    const m = LOCAL_FORCE_TOKEN_RE.exec(name);
    if (m?.groups) {
      const rawBase = m.groups.base!;
      const base = alias2d ? (LOCAL_FORCE_2D_ALIAS[rawBase] ?? rawBase) : rawBase;
      const comp = BEAM_BASE_MAP[base];
      if (!comp) return null;
      return { component: comp, station: Number(m.groups.station!) };
    }
  }
  const m = BEAM_TOKEN_RE.exec(name);
  if (!m?.groups) return null;
  const comp = BEAM_BASE_MAP[m.groups.base!];
  if (!comp) return null;
  const st = m.groups.station !== undefined ? Number(m.groups.station) : null;
  return { component: comp, station: st };
}

function classPrefixOfBucket(bucketKey: string): string {
  const idx = bucketKey.indexOf("[");
  return idx >= 0 ? bucketKey.slice(0, idx) : bucketKey;
}

function classNameOfBucket(bucketKey: string): string {
  const prefix = classPrefixOfBucket(bucketKey);
  const dash = prefix.indexOf("-");
  return dash >= 0 ? prefix.slice(dash + 1) : prefix;
}

function localForce2dGate(
  bucketKey: string,
  blocks: readonly BucketBlock[],
  width: number,
  token: string,
  ndm: number | null,
): { ok: true; alias2d: boolean } | { ok: false; reason: string } {
  if (token !== "localForce" && token !== "localForces") return { ok: true, alias2d: false };
  const names = blocks.flatMap((b) => b.compNames.map((n) => n.trim()));
  const bases = new Set<string>();
  for (const n of names) {
    const m = LOCAL_FORCE_TOKEN_RE.exec(n);
    if (m?.groups?.base) bases.add(m.groups.base);
  }
  const spelled2d = [...bases].filter((b) => b in LOCAL_FORCE_2D_ALIAS).sort();
  if (spelled2d.length === 0) return { ok: true, alias2d: false };

  const cls = classNameOfBucket(bucketKey) || "<unparsed>";
  const where = `localForce bucket ${JSON.stringify(bucketKey)} (element class ${JSON.stringify(cls)})`;
  if (blocks.some((b) => b.multiplicity !== 1) || names.length !== width) {
    return {
      ok: false,
      reason: `${where}: its DATA is ${width} columns wide per element but it tags ${names.length} components [${names.join(", ")}]; the 2-D names cannot be matched to the columns.`,
    };
  }
  const spelled3d = [...bases].filter((b) => LOCAL_FORCE_3D_BASES.has(b)).sort();
  if (spelled3d.length > 0) {
    return {
      ok: false,
      reason: `${where}: mixes the 2-D names [${spelled2d.join(", ")}] with the 3-D [${spelled3d.join(", ")}]; the shear and moment axes are ambiguous.`,
    };
  }
  if (ndm !== 2) {
    return {
      ok: false,
      reason: `${where}: tags its columns with the 2-D names [${names.join(", ")}], but the file's spatial dimension is ${JSON.stringify(ndm)}, not 2; the shear and moment axes cannot be read from them.`,
    };
  }
  if (!classNameOfBucket(bucketKey) || cls.toLowerCase().endsWith("3d")) {
    return {
      ok: false,
      reason: `${where}: the class records its localForce under the 2-D names [${names.join(", ")}]; a 3-D class (or an unparsed key) cannot be read as a 2-D beam.`,
    };
  }
  return { ok: true, alias2d: true };
}

function basicForceRefusal(bucketKey: string, component: LineStationComponent, stations: readonly number[]): string | null {
  if (stations.length === 1) return null;
  if ((component === "My" || component === "Mz") && stations.length === 2 && stations[0] === 1 && stations[1] === 2) {
    return null;
  }
  const cls = classNameOfBucket(bucketKey) || "<unparsed>";
  return (
    `basicForce bucket ${JSON.stringify(bucketKey)} (element class ${JSON.stringify(cls)}) names ${JSON.stringify(component)} at stations [${stations.join(", ")}]; ` +
    `only the end moments ["My", "Mz"] at stations [1, 2] map to the section convention. Its ${JSON.stringify(component)} line stations are left out.`
  );
}

function stationXi(nStations: number): Float64Array {
  if (nStations <= 0) return new Float64Array(0);
  if (nStations === 1) return new Float64Array([0]);
  const out = new Float64Array(nStations);
  for (let i = 0; i < nStations; i++) {
    out[i] = -1 + (2 * i) / (nStations - 1);
  }
  return out;
}
