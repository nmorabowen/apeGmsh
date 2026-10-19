// Read a partitioned `.ladruno` run as one file (ADR 0122 slice 8, #2188).
//
// An OpenSeesMP run writes one file per rank, `<stem>.part-<N>.ladruno`, N 0-indexed
// and contiguous from 0 (fork schema "File identity", D5), each with the manifest
// INFO@PARTITIONED, PARTITION_ID and NUM_PARTITIONS (§2). This module opens every
// part of the set the opened part belongs to (reader/ladruno.ts `openLadruno`
// with `part`), checks that they are one run, and answers the reads main makes of
// a LadrunoFile, over one logical node index. It is the app's port of Python's
// `_ladruno_multi.LadrunoMultiPartitionReader` node stitch.
//
// Rules this file keeps:
// - the whole set or nothing: a missing part, a part past NUM_PARTITIONS, parts
//   whose NUM_PARTITIONS disagree, or a PARTITION_ID that is not the file's own
//   index refuses the set, naming the indices; a partial set is never read; parts
//   whose INFO@RUN_ID differ are refused under Python's rule (`_validate_part_set`);
// - every part has the same stages, kinds and TIME axis; a part with a shorter
//   step axis (one still being written) refuses the set, and the common prefix is
//   never read;
// - the node index is the sorted union of the parts' node ids, so a node on a
//   partition interface appears once. Each nodal result's PARTITION_REDUCTION
//   (§7.1) says how its copies combine: NONE keeps the first part's copy, SUM adds
//   the copies in part order. A result with no PARTITION_REDUCTION, a value outside
//   the §7.1 table, UNSUPPORTED, parts that disagree, or a result some part does not
//   record, is refused by name in `NodalInfo.refused`, never assumed NONE;
// - pairing: the set's lineage is the parts' PROVENANCE `model_hash` and `fem_hash`
//   only when every part carries the same both (§7b.1 "Partitioned sets"); an
//   unstamped part, or parts that disagree, leave the set unpaired, with the part
//   indices named;
// - element results (Gauss points, line stations) are not stitched by this slice:
//   they are listed in `unread`, and the element-id verdict is unverified, so no
//   element row of a part set leaves main.
//
// Browser-safe: no Node import. The directory listing is the caller's (`list`).

import type { ElementIndexVerdict } from "./ladrunoElements.ts";
import { LadrunoError, openLadruno, type LadrunoFile, type LadrunoIndex, type LadrunoLineage, type LadrunoLineStationSlab, type LadrunoLineStationStep, type LadrunoMode, type LineStationComponent } from "./ladruno.ts";
import type { ElementIds, NodalInfo, ResultsH5, StageInfo } from "./results.ts";

/** A partitioned run's part file name (schema "File identity", D5); the suffix in any case. */
const PART_NAME = /^(?<stem>.+?)\.part-(?<idx>\d+)\.ladruno$/i;

/** The §7.1 values a stitch combines; UNSUPPORTED is in the table, and refused. */
export const PARTITION_REDUCTIONS = ["NONE", "SUM", "UNSUPPORTED"] as const;
type Reduction = "NONE" | "SUM";

/** Why no element result of a part set is read. */
export const PART_SET_ELEMENTS_UNREAD = "element results of a partitioned .ladruno set are not stitched by this version (ADR 0122 slice 8 stitches nodal results)";

const splitPath = (path: string): { dir: string; base: string } => {
  const cut = Math.max(path.lastIndexOf("/"), path.lastIndexOf("\\"));
  return { dir: cut < 0 ? "" : path.slice(0, cut + 1), base: path.slice(cut + 1) };
};

/** The stem and index of a part file's name, or null when `path` is not one. */
export function partOf(path: string): { dir: string; stem: string; index: number } | null {
  const { dir, base } = splitPath(path);
  const m = PART_NAME.exec(base);
  return m ? { dir, stem: m.groups!.stem!, index: Number(m.groups!.idx) } : null;
}

/** Every `<stem>.part-<N>.ladruno` beside `path` (the same stem), by index. `list(dir)` names the directory's files. */
export function partSiblings(path: string, list: (dir: string) => readonly string[]): { index: number; path: string }[] {
  const me = partOf(path);
  if (!me) throw new LadrunoError("argument", `${path} is not a <stem>.part-<N>.ladruno file`);
  const found: { index: number; path: string }[] = [];
  for (const name of list(me.dir === "" ? "." : me.dir)) {
    const m = PART_NAME.exec(name);
    if (m && m.groups!.stem === me.stem) found.push({ index: Number(m.groups!.idx), path: `${me.dir}${name}` });
  }
  found.sort((a, b) => a.index - b.index);
  for (let i = 1; i < found.length; i++) {
    if (found[i]!.index === found[i - 1]!.index) {
      throw new LadrunoError("layout", `${me.stem}: two files are part ${found[i]!.index} (${baseOf(found[i - 1]!.path)}, ${baseOf(found[i]!.path)})`);
    }
  }
  return found;
}

const baseOf = (p: string): string => splitPath(p).base;
const listing = (items: readonly (readonly [string, unknown])[]): string => items.map(([n, v]) => `${n} = ${String(v)}`).join(", ");
const indexList = (xs: readonly number[]): string => xs.join(", ");

/** Open the part set `path` belongs to (any one of its parts). Every refusal names the parts. */
export function openLadrunoSet(h5: ResultsH5, path: string, list: (dir: string) => readonly string[]): LadrunoPartSet {
  const me = partOf(path);
  if (!me) throw new LadrunoError("argument", `${path} is not a <stem>.part-<N>.ladruno file`);
  const sibs = partSiblings(path, list);
  const parts: LadrunoFile[] = [];
  try {
    for (const s of sibs) parts.push(openLadruno(h5, s.path, { part: true }));
    return new LadrunoPartSet(path, me.stem, sibs.map((s) => s.index), parts);
  } catch (err) {
    for (const p of parts) p.close();
    throw err;
  }
}

/** One nodal component of the set: each part's column, and how the copies combine. */
interface Stitch {
  reduction: Reduction;
}

/** Each part's node positions in the set's index, per stage. */
interface StageMap {
  /** for part j, the union column of its i-th node */
  pos: Int32Array[];
  n: number;
}

export class LadrunoPartSet {
  readonly index: LadrunoIndex;
  /** the part files, in index order */
  readonly paths: string[];
  private readonly parts: LadrunoFile[];
  private readonly names: string[];
  private readonly maps = new Map<string, StageMap>();
  private readonly stitches = new Map<string, Map<string, Stitch>>();

  constructor(path: string, stem: string, indices: readonly number[], parts: LadrunoFile[]) {
    this.parts = parts;
    this.paths = parts.map((p) => p.index.path);
    this.names = this.paths.map(baseOf);
    this.checkManifest(stem, indices);
    const p0 = parts[0]!.index;
    const warnings: string[] = [];
    parts.forEach((p, j) => warnings.push(...p.index.warnings.map((w) => `${this.names[j]}: ${w}`)));
    this.checkStages();
    const stages = p0.stages.map((s) => this.stitchStage(s.id));
    const { lineage, reason } = this.lineage();
    const units = parts.every((p) => p.index.units === p0.units) ? p0.units : null;
    if (units === null && parts.some((p) => p.index.units !== null)) {
      warnings.push(`the parts declare different units (${listing(parts.map((p, j) => [this.names[j]!, p.index.units]))}); the set declares none`);
    }
    this.index = {
      ...p0,
      path,
      stages,
      modes: p0.modes.map((m) => ({ ...m })),
      lineage,
      lineageReason: reason,
      units,
      partition: null,
      warnings,
    };
  }

  /** What a re-open compares: every part's INFO/PROVENANCE as read, in index order. */
  identity(): string {
    return `parts:${JSON.stringify(this.parts.map((p) => p.index.provenance))}`;
  }

  close(): void {
    for (const p of this.parts) p.close();
  }

  stage(id: string): StageInfo {
    const s = this.index.stages.find((x) => x.id === id);
    if (!s) throw new LadrunoError("argument", `no stage ${JSON.stringify(id)}; stages: ${this.index.stages.map((x) => x.id).join(", ")}`);
    return s;
  }

  mode(stageId: string): LadrunoMode | null {
    this.stage(stageId);
    return this.index.modes.find((m) => m.stageId === stageId) ?? null;
  }

  /** One array per entry of `components`, each (N,) over the set's node index at `step`. */
  readNodalStep(stageId: string, components: readonly string[], step: number): Float64Array[] {
    const s = this.stage(stageId);
    if (!Number.isInteger(step) || step < 0 || step >= s.nSteps) throw new LadrunoError("argument", `step ${step} is outside stage ${s.id} (0..${s.nSteps - 1})`);
    return this.readNodalRows(stageId, components, step, step + 1);
  }

  /** Steps `k0` to `k1` (exclusive) of each entry of `components`: ((k1 - k0) * N,), step-major, over the set's node index. */
  readNodalRows(stageId: string, components: readonly string[], k0: number, k1: number): Float64Array[] {
    const s = this.stage(stageId);
    if (!Number.isInteger(k0) || !Number.isInteger(k1) || k0 < 0 || k1 > s.nSteps || k0 >= k1) {
      throw new LadrunoError("argument", `steps ${k0}..${k1} are not a range inside stage ${s.id} (0..${s.nSteps})`);
    }
    const map = this.maps.get(s.id)!;
    const T = k1 - k0;
    return components.map((c) => {
      const st = this.stitches.get(s.id)?.get(c);
      if (!st) throw new LadrunoError("argument", `stage ${s.id} of the part set has no component ${JSON.stringify(c)}; it has ${s.nodes?.components.join(", ") || "none"}`);
      const out = new Float64Array(T * map.n);
      // NONE: the first part's copy wins, so the parts are written last to first. SUM: added in part order.
      const order = st.reduction === "SUM" ? this.parts.map((_, j) => j) : this.parts.map((_, j) => this.parts.length - 1 - j);
      for (const j of order) {
        const rows = this.parts[j]!.readNodalRows(s.id, [c], k0, k1)[0]!;
        const pos = map.pos[j]!;
        const nj = pos.length;
        for (let k = 0; k < T; k++) {
          for (let i = 0; i < nj; i++) {
            const at = k * map.n + pos[i]!;
            out[at] = st.reduction === "SUM" ? out[at]! + rows[k * nj + i]! : rows[k * nj + i]!;
          }
        }
      }
      return out;
    });
  }

  elementIndexVerdict(): ElementIndexVerdict {
    return { verified: false, reason: PART_SET_ELEMENTS_UNREAD };
  }

  elementIds(stageId: string, groupId: string): ElementIds {
    throw new LadrunoError("unsupported", `stage ${stageId}, group ${groupId}: ${PART_SET_ELEMENTS_UNREAD}`);
  }

  readGaussStep(stageId: string, groupId: string): Float64Array {
    throw new LadrunoError("unsupported", `stage ${stageId}, group ${groupId}: ${PART_SET_ELEMENTS_UNREAD}`);
  }

  readGaussRows(stageId: string, groupId: string): Float64Array {
    throw new LadrunoError("unsupported", `stage ${stageId}, group ${groupId}: ${PART_SET_ELEMENTS_UNREAD}`);
  }

  availableLineStations(stageId: string): LineStationComponent[] {
    this.stage(stageId);
    return [];
  }

  readLineStations(stageId: string, component: string): LadrunoLineStationSlab {
    throw new LadrunoError("unsupported", `stage ${stageId}, line stations ${component}: ${PART_SET_ELEMENTS_UNREAD}`);
  }

  readLineStationStep(stageId: string, component: string): LadrunoLineStationStep {
    throw new LadrunoError("unsupported", `stage ${stageId}, line stations ${component}: ${PART_SET_ELEMENTS_UNREAD}`);
  }

  // -- the set ----------------------------------------------------------------

  /** One run's parts: one NUM_PARTITIONS, each PARTITION_ID its file's index, every index 0..N-1 present once. */
  private checkManifest(stem: string, indices: readonly number[]): void {
    const counts = this.parts.map((p) => p.index.partition!.count);
    if (new Set(counts).size > 1) {
      throw new LadrunoError("layout", `${stem}.part-<N>.ladruno: the parts disagree on INFO@NUM_PARTITIONS (${listing(this.names.map((n, j) => [n, counts[j]]))}); they are not from one run`);
    }
    const n = counts[0]!;
    const wrongId = this.parts.map((p, j) => [j, p.index.partition!.id] as const).filter(([j, id]) => id !== indices[j]);
    if (wrongId.length) {
      throw new LadrunoError("layout", `${stem}.part-<N>.ladruno: INFO@PARTITION_ID is not the file's own index (${listing(wrongId.map(([j, id]) => [this.names[j]!, id]))})`);
    }
    const missing = Array.from({ length: n }, (_, k) => k).filter((k) => !indices.includes(k));
    const extra = indices.filter((k) => k >= n);
    if (missing.length || extra.length) {
      const what = [missing.length ? `part(s) ${indexList(missing)} missing` : "", extra.length ? `part(s) ${indexList(extra)} past NUM_PARTITIONS` : ""].filter(Boolean).join("; ");
      throw new LadrunoError(
        "layout",
        `${stem}.part-<N>.ladruno: INFO@NUM_PARTITIONS = ${n}, and the directory holds part(s) ${indexList(indices)}: ${what}. A partial set is not read; re-run, or remove the stale part files`,
      );
    }
    // Python `_validate_part_set`: RUN_ID is compared only when every part has one and no
    // part's RUN_ID_SCOPE is absent or "process" (schema §2: a process-unique id is not a run identity).
    const runs = this.parts.map((p) => p.index.partition!);
    if (runs.every((r) => r.runId !== null) && !runs.some((r) => r.runIdScope === null || r.runIdScope === "process")) {
      if (new Set(runs.map((r) => r.runId)).size > 1) {
        throw new LadrunoError(
          "layout",
          `${stem}.part-<N>.ladruno: the parts come from different runs (INFO@RUN_ID differs: ${listing(this.names.map((nm, j) => [nm, runs[j]!.runId]))}). Stale part files from an earlier run are mixed in; remove them and re-run`,
        );
      }
    }
  }

  /** The parts' stages pair by position: the same count, kinds, modes and TIME axis. */
  private checkStages(): void {
    const ref = this.parts[0]!.index;
    this.parts.forEach((p, j) => {
      const ix = p.index;
      if (ix.stages.length !== ref.stages.length) {
        throw new LadrunoError("layout", `${this.names[j]} has ${ix.stages.length} stage(s) and ${this.names[0]} ${ref.stages.length}: the parts are not one run`);
      }
      ix.stages.forEach((s, k) => {
        const r = ref.stages[k]!;
        if (s.id !== r.id || s.kind !== r.kind) {
          throw new LadrunoError("layout", `stage ${k} is ${r.id} (${r.kind}) in ${this.names[0]} and ${s.id} (${s.kind}) in ${this.names[j]}: the parts are not one run`);
        }
        if (s.time.length !== r.time.length) {
          throw new LadrunoError(
            "layout",
            `stage ${r.id}: ${this.names[j]} holds ${s.time.length} step(s) and ${this.names[0]} ${r.time.length}: the step axes disagree (a part still being written, or a stopped rank); the set is not read`,
          );
        }
        if (s.time.some((t, i) => t !== r.time[i])) {
          throw new LadrunoError("layout", `stage ${r.id}: the TIME axes of ${this.names[0]} and ${this.names[j]} differ; the parts are not one run`);
        }
      });
      const modes = (m: LadrunoMode[]) => JSON.stringify(m.map((x) => [x.stageId, x.modeIndex, x.eigenvalue]));
      if (modes(ix.modes) !== modes(ref.modes)) {
        throw new LadrunoError("layout", `${this.names[0]} and ${this.names[j]} record different modes: they are not from one eigen solution`);
      }
    });
  }

  /** The set's stage `id`: the union node index, and each component the parts stitch. */
  private stitchStage(id: string): StageInfo {
    const per = this.parts.map((p) => p.stage(id));
    const r = per[0]!;
    const unread = [...new Set(per.flatMap((s) => s.unread))];
    for (const g of [...new Set(per.flatMap((s) => s.gauss.map((x) => x.id)))]) unread.push(`${g} (${PART_SET_ELEMENTS_UNREAD})`);
    const stations = [...new Set(this.parts.flatMap((p) => p.availableLineStations(id)))];
    if (stations.length) unread.push(`line stations ${stations.join(", ")} (${PART_SET_ELEMENTS_UNREAD})`);
    const stitches = new Map<string, Stitch>();
    this.stitches.set(id, stitches);
    if (per.every((s) => s.nodes === null)) {
      this.maps.set(id, { pos: per.map(() => new Int32Array(0)), n: 0 });
      return { ...r, nodes: null, gauss: [], unread };
    }
    const refused: NodalInfo["refused"] = [];
    per.forEach((s, j) => {
      for (const x of s.nodes?.refused ?? []) refused.push({ component: x.component, reason: `${this.names[j]}: ${x.reason}` });
    });
    const refusedNames = new Set(refused.map((x) => x.component));
    const components: string[] = [];
    const candidates = [...new Set(per.flatMap((s) => s.nodes?.components ?? []))];
    for (const c of candidates) {
      const result = c.slice(0, c.lastIndexOf("/"));
      if (refusedNames.has(result)) continue;
      const lacking = per.map((s, j) => (s.nodes?.components.includes(c) ? -1 : j)).filter((j) => j >= 0);
      if (lacking.length) {
        refused.push({ component: c, reason: `not recorded in ${lacking.map((j) => this.names[j]).join(", ")}: a component of a part set is read only when every part records it` });
        continue;
      }
      const verdict = this.reductionOf(id, c);
      if (typeof verdict === "object") refused.push({ component: c, reason: verdict.reason });
      else {
        components.push(c);
        stitches.set(c, { reduction: verdict });
      }
    }
    const union = [...new Set(per.flatMap((s) => Array.from(s.nodes?.ids ?? [])))].sort((a, b) => a - b);
    const at = new Map(union.map((n, i) => [n, i]));
    this.maps.set(id, { pos: per.map((s) => Int32Array.from(s.nodes?.ids ?? [], (n) => at.get(n)!)), n: union.length });
    const nodes: NodalInfo = { path: r.nodes?.path ?? per.find((s) => s.nodes)!.nodes!.path, ids: Float64Array.from(union), components, refused };
    return { ...r, nodes, gauss: [], unread };
  }

  /** How the parts' copies of `component` combine, or why they are refused (schema §7.1). */
  private reductionOf(stageId: string, component: string): Reduction | { reason: string } {
    const result = component.slice(0, component.lastIndexOf("/"));
    const kinds = this.parts.map((p) => p.nodalReduction(stageId, component));
    const absent = kinds.map((k, j) => (k === null ? j : -1)).filter((j) => j >= 0);
    if (absent.length) {
      return { reason: `${result} has no PARTITION_REDUCTION in ${absent.map((j) => this.names[j]).join(", ")} (schema §7.1); a part set never assumes NONE` };
    }
    const unknown = kinds.map((k, j) => [j, k!] as const).filter(([, k]) => !(PARTITION_REDUCTIONS as readonly string[]).includes(k));
    if (unknown.length) {
      return { reason: `${result}@PARTITION_REDUCTION is ${unknown.map(([j, k]) => `${JSON.stringify(k)} in ${this.names[j]}`).join(", ")}; expected one of ${PARTITION_REDUCTIONS.join(", ")} (schema §7.1)` };
    }
    if (new Set(kinds).size > 1) return { reason: `the parts disagree on ${result}@PARTITION_REDUCTION (${listing(this.names.map((n, j) => [n, kinds[j]]))})` };
    if (kinds[0] === "UNSUPPORTED") {
      return { reason: `${result} is PARTITION_REDUCTION = UNSUPPORTED: each part holds its own rank's partial, and no sum of them is the model value (schema §7.1)` };
    }
    return kinds[0] as Reduction;
  }

  /** The parts' common lineage, or why the set is unpaired, naming the parts (schema §7b.1 "Partitioned sets"). */
  private lineage(): { lineage: LadrunoLineage | null; reason: string | null } {
    const ls = this.parts.map((p) => p.index.lineage);
    const unstamped = ls.map((l, j) => (l === null ? j : -1)).filter((j) => j >= 0);
    if (unstamped.length) {
      const why = this.parts[unstamped[0]!]!.index.lineageReason;
      return {
        lineage: null,
        reason: `part(s) ${indexList(unstamped)} of the set (${unstamped.map((j) => this.names[j]).join(", ")}) carry no pairing stamp (${why}); a part set pairs only when every part carries the same model_hash and fem_hash`,
      };
    }
    const key = (l: LadrunoLineage) => `${l.modelHash}|${l.femHash}`;
    const groups = new Map<string, number[]>();
    ls.forEach((l, j) => groups.set(key(l!), [...(groups.get(key(l!)) ?? []), j]));
    if (groups.size > 1) {
      const said = [...groups.values()].map((js) => `part(s) ${indexList(js)} model_hash ${ls[js[0]!]!.modelHash.slice(0, 8)}…, fem_hash ${ls[js[0]!]!.femHash.slice(0, 8)}…`).join("; ");
      return { lineage: null, reason: `the parts of the set carry different stamps (${said}): they were stamped against different models, and the set pairs with none` };
    }
    return { lineage: { ...ls[0]! }, reason: null };
  }
}
