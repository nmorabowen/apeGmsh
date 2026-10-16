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
//   name, and a file whose INFO says it is one part of several is refused too
//   (part-N sets are slice 8);
// - components are spelled as the file spells them, `<RESULT>/<label>` with the
//   label from the result's COMPONENTS attribute. ADR 0122 D7: no token table,
//   so nothing here maps `DISPLACEMENT/Ux` to an apeGmsh name;
// - a nodal result whose datasets disagree (DATA's step axis against TIME and
//   STEP, its node axis against ID, its component axis against COMPONENTS) is
//   refused by name in `NodalInfo.refused`, with the reason, never read;
// - element results, envelopes and the domain and region buckets are listed in
//   `unread`, not skipped silently (Gauss points are slice 8);
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

import type { NodalInfo, ResultsH5, StageInfo } from "./results.ts";

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

/** The six consumer-owned attributes of schema §7b.1, in its order. */
export const PROVENANCE_ATTRS = ["model_hash", "fem_hash", "stage_names", "element_index_space", "unsupported", "writer"] as const;
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
  /** `INFO@SPATIAL_DIM` when it is 1, 2 or 3, else null */
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
  warnings: string[];
}

/** Where one component of one stage lives: a 3-D DATA[T, n, c] or a 2-D MODE_<k>[n, c]. */
interface Column {
  path: string;
  col: number;
  modal: boolean;
}

/** Open a `.ladruno` and index it; reads go through the returned object. */
export function openLadruno(h5: ResultsH5, path: string): LadrunoFile {
  const base = path.split(/[\\/]/).pop() ?? path;
  const part = PART_FILE.exec(base);
  if (part) {
    throw new LadrunoError(
      "unsupported",
      `${path} is part ${part.groups!.idx} of the partitioned set ${part.groups!.stem}.part-<N>.ladruno; this reader reads one file, not a part set`,
    );
  }
  const f = new h5.File(path, "r") as unknown as LFile;
  // h5wasm does not throw when HDF5 cannot open a file: it returns a File whose id is negative.
  if (Number(f.file_id) < 0) throw new LadrunoError("open", `HDF5 could not open ${path} (truncated, or not an HDF5 file)`);
  try {
    return new LadrunoFile(h5, f, path);
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

  constructor(h5: ResultsH5, f: LFile, path: string) {
    this.h5 = h5;
    this.f = f;
    const warnings: string[] = [];
    const info = this.gate(path);
    const formatMinorVersion = optionalInt(info, "FORMAT_MINOR_VERSION") ?? 0;
    const formatVersion = optionalInt(info, "FORMAT_VERSION")!;
    const partitions = optionalInt(info, "NUM_PARTITIONS");
    if (optionalInt(info, "PARTITIONED") === 1 || (partitions !== null && partitions > 1)) {
      throw new LadrunoError("unsupported", `${path}: INFO says this file is one part of a partitioned run (PARTITIONED, NUM_PARTITIONS = ${partitions}); this reader reads one file`);
    }
    const dim = optionalInt(info, "SPATIAL_DIM");
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
      spatialDim: dim !== null && dim >= 1 && dim <= 3 ? dim : null,
      stages,
      fileKinds: Object.fromEntries(hosts.map((h) => [h.stage.id, h.fileKind])),
      modes: modes.facts,
      provenance,
      lineage,
      lineageReason: reason,
      warnings,
    };
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

  /** One array per entry of `components` (`<RESULT>/<label>`), each (N,) at `step`, by hyperslab. */
  readNodalStep(stageId: string, components: readonly string[], step: number): Float64Array[] {
    const s = this.stage(stageId);
    checkStep(s, step);
    return components.map((c) => {
      const col = this.column(s, c);
      const ds = this.datasetAt(col.path);
      return toFloat64(col.modal ? ds.slice([[], [col.col, col.col + 1]]) : ds.slice([[step, step + 1], [], [col.col, col.col + 1]]), ds.path);
    });
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
    if (results) {
      for (const b of results.keys()) {
        if (b === "ON_NODES") onNodes = this.group(results, b);
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
      gauss: [],
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
      if (fact.state === "present" && (name === "model_hash" || name === "fem_hash") && fact.value === "") {
        attrs[name] = { state: "malformed", reason: `@${name} is an empty string` };
        continue;
      }
      attrs[name] = fact;
    }
    const known = new Set<string>(PROVENANCE_ATTRS);
    const other = Object.keys(g.attrs).filter((k) => !known.has(k)).sort();
    return { state: "present", attrs, stageNames, unsupported, other };
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

/** A scalar attribute, or the one element of a one-element array (the fork writes shape (1,)). */
function oneValue(v: unknown): unknown {
  if (Array.isArray(v)) return v.length === 1 ? v[0] : v;
  if (ArrayBuffer.isView(v) && !(v instanceof DataView)) {
    const a = v as unknown as ArrayLike<unknown>;
    return a.length === 1 ? a[0] : v;
  }
  return v;
}

function intOf(v: unknown): number | null {
  if (typeof v === "bigint") return Number(v);
  if (typeof v === "number" && Number.isInteger(v)) return v;
  return null;
}

function optionalInt(g: LGroup, key: string): number | null {
  const a = g.attrs[key];
  if (a === undefined) return null;
  const v = intOf(oneValue(a.value));
  if (v === null) throw new LadrunoError("layout", `${g.path}@${key} is not an integer`);
  return v;
}

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
