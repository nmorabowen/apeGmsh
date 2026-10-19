// The results file of the view window, read lazily in main (ADR 0112 D5,
// the #1283 ruling of 2026-10-11; V4b, #1728).
//
// Main reads one results file (reader/results.ts). The renderer gets its
// index as plain metadata (`open`) and one step at a time (`readStep`): the
// store never holds a component array, and nothing is read eagerly.
//
// The HDF5 handle is not held for the window's life (V4b review round 1): on
// Windows an open handle makes a writer's temp-file-and-`os.replace` onto the
// path fail. It is closed after `idleMs` without a read (≤ 1 s) and opened
// again by the next read. The file's identity is taken when it is indexed
// (its size and mtime, and `/meta/lineage@results_hash`); every read checks
// the size and mtime first, and a re-open checks the hash too. A file that
// changed is never read under the old index: the read answers `changed`, and
// the renderer re-indexes it. No Electron import: the tests drive this class
// in Node.
//
// A `.ladruno` (ADR 0122 D1; #2127) is read by reader/ladruno.ts under the
// same rules. It has no results_hash: its identity on re-open is its whole
// INFO/PROVENANCE as read (the `model_hash` and every other stamped
// attribute), so a file re-stamped while open is re-indexed, never read
// under the old pairing. Its lineage is the PROVENANCE `model_hash` and
// `fem_hash`, only when both are stamped (a stamp written in part reads as
// unpaired, schema §7b.1); `element_index_space` stands in for the native
// root stamp, and `stage_names` names the MODEL_STAGE groups when it counts
// as many.

import { existsSync, statSync } from "node:fs";
import {
  LadrunoFile,
  openLadruno,
  type LadrunoLineStationStep,
  type LineStationComponent,
  type ProvenanceFact,
} from "../reader/ladruno.ts";
import { unopenedSibling } from "./pairing.ts";
import { openResults, type ResultsFile, type ResultsH5, type StageInfo } from "../reader/results.ts";
import { gaussCellValues } from "../state/contour.ts";
import { applyPlan, planRange, type ExtrapPlan } from "../state/extrapolation.ts";
import type { ModeFacts, UnitSystemId } from "../state/types.ts";

export type AnyResultsFile = ResultsFile | LadrunoFile;

function openAnyResults(h5: ResultsH5, path: string): AnyResultsFile {
  return path.toLowerCase().endsWith(".ladruno") ? openLadruno(h5, path) : openResults(h5, path);
}

/** What a re-open compares: a native file's results_hash; a `.ladruno`'s whole INFO/PROVENANCE as read. */
function identityOf(rf: AnyResultsFile): string | null {
  if (rf instanceof LadrunoFile) return `provenance:${JSON.stringify(rf.index.provenance)}`;
  return rf.index.lineage?.resultsHash ?? null;
}

/** One stage as the renderer's store keeps it (state/types.ts `ResultsStage`). */
export interface ResultsStageMeta {
  id: string;
  name: string;
  kind: StageInfo["kind"];
  time: number[];
  nodal: string[];
  /** V4c1: nodal components refused one by one (value columns that disagree with `_ids`) */
  refusedNodal: { component: string; reason: string }[];
  /**
   * V4c2a: each gauss group's components, its components refused one by one
   * (a partial write, or a group with no Gauss point), its size, and whether
   * it carries shell local axes.
   */
  gauss: GaussGroupMeta[];
  unread: string[];
  /** .ladruno INFO/PROVENANCE@element_index_space ("fem" or "ops_tag"), or null */
  elementIndexSpace?: string | null;
  /** 1-D beam line-station components recorded in this stage (.ladruno) */
  lineStations?: LineStationComponent[];
  /** a `.ladruno` mode stage's eigen scalars (reader/ladruno.ts `LadrunoMode`); absent on every other stage */
  mode?: ModeFacts;
}

export interface GaussGroupMeta {
  id: string;
  components: string[];
  refused: { component: string; reason: string }[];
  nElements: number;
  nGauss: number;
  localAxes: boolean;
  /** V4c2b: `_natural_coords` (G rows), null when not floating point (reader/results.ts) */
  naturalCoords: number[][] | null;
}

/** Whether a file's `_element_index` rows are FEM element ids: only the root stamp says so (#1768). */
export type ElementIndexVerdict = { verified: true } | { verified: false; reason: string };

export interface ResultsMeta {
  path: string;
  version: string;
  sourceType: string | null;
  lineage: { femHash: string | null; modelHash: string | null; resultsHash: string | null } | null;
  /** .ladruno: why `lineage` is null (the file is not stamped for pairing), else null; absent for a native file */
  lineageReason?: string | null;
  partial: string | null;
  stages: ResultsStageMeta[];
  unavailableZones: string[];
  /** V4h: the embedded opensees zone's version stamp; null when the file embeds no /opensees */
  openseesVersion: string | null;
  /** V4h: the embedded /opensees/decls@program_decl marker ("absent" when missing); null with no /opensees */
  programDecl: string | null;
  /** V4c2a: the element-id verdict; element contours draw only for a verified file */
  elementIndex: ElementIndexVerdict;
  /** Consistent unit system declared in `/model/meta/@units`, `@units`, or `INFO/PROVENANCE@units`, or null when undeclared */
  units: UnitSystemId | null;
  /** .ladruno INFO/PROVENANCE@element_index_space ("fem" or "ops_tag"), or null */
  elementIndexSpace?: string | null;
}

export type ResultsOpenAnswer =
  /**
   * `nodeIds`: each stage's nodal `_ids` by stage id (V4c1), outside `meta` so
   * the store never holds an array. `elementIds` (V4c2a): each gauss group's
   * `_element_index` by stage and group id, sent only for a verified file; an
   * unverified file's rows never leave main, so nothing can join them.
   */
  | {
      ok: true;
      meta: ResultsMeta;
      nodeIds: Record<string, Float64Array>;
      elementIds: Record<string, Record<string, Float64Array>>;
      warnings: string[];
      sizeBytes: number;
      readMs: number;
    }
  | { ok: false; error: string };

/** A step request: the stage by id (and index, echoed back), the step, and the renderer's request number. */
export interface StepRequest {
  path: string;
  stage: string;
  step: number;
  seq: number;
  /** V4c2a: one element (Gauss) component to read too, from every group that has it; verified files only */
  element?: string;
  /** .ladruno: one 1-D beam line-station component to read for this step */
  lineDiagram?: LineStationComponent;
}

export type StepAnswer =
  /** `gauss`: the requested element component by group id, (E * G,) element-major; {} with no `element` */
  | {
      ok: true;
      seq: number;
      stage: string;
      step: number;
      nodal: Record<string, Float64Array>;
      gauss: Record<string, Float64Array>;
      lineDiagram?: LadrunoLineStationStep | null;
    }
  /** `changed`: the file on disk is not the one indexed; re-open it (`open`) before reading */
  | { ok: false; seq: number; error: string; changed?: true };

/**
 * A history-range scan (V4c1, #1756): the finite min and max of one nodal
 * component, or of the magnitude of several, over every step of a stage, as
 * the Qt contour's whole-history colour range (`_compute_initial_clim`).
 */
export interface ScanRequest {
  path: string;
  stage: string;
  components: string[];
  magnitude: boolean;
  scanId: number;
  /**
   * V4c2a: `element` scans one Gauss component (`components` has one name):
   * the per-element value (a one-point group's value, else the mean of the
   * element's Gauss points; state/contour.ts `gaussCellValues`) over every
   * group that has it. Verified files only. `line` scans one `.ladruno`
   * line-station component across all steps. Absent: a nodal scan.
   */
  source?: "element" | "line";
  /**
   * V4c2b: with `source: "element"`, a node-located topology's plan
   * (state/extrapolation.ts): every step is carried through it and the range
   * is of the values drawn (the nodes', or every element corner's).
   */
  plan?: ExtrapPlan;
}

export type ScanAnswer =
  /** `min`/`max` null: no finite value in the whole history */
  | { ok: true; scanId: number; min: number | null; max: number | null; steps: number; ms: number }
  /** `cancelled`: a newer field, file or stage replaced it; `changed`: the file on disk is not the one indexed */
  | { ok: false; scanId: number; error: string; cancelled?: true; changed?: true };

export interface ScanProgress {
  scanId: number;
  done: number;
  total: number;
}

/** How long the handle stays open after the last read (the D5 revisit trigger's 30 steps/s is 33 ms a step). */
export const IDLE_CLOSE_MS = 500;

const signature = (path: string): string => {
  const st = statSync(path);
  return `${st.size}:${st.mtimeMs}`;
};

const message = (err: unknown) => (err instanceof Error ? err.message : String(err));
const baseName = (p: string): string => p.split(/[\\/]/).pop() ?? p;

/** Why a `.ladruno`'s element rows are unverified: no `element_index_space`, or one that is not "fem" (#2079 Q6). */
function unverifiedSpace(space: ProvenanceFact | null): string {
  if (space === null) return "the file has no INFO/PROVENANCE group, so its element rows are not known to be FEM element ids";
  if (space.state === "malformed") return `${space.reason}; its element rows are not known to be FEM element ids`;
  return 'INFO/PROVENANCE has no @element_index_space = "fem", so its element rows are not known to be FEM element ids';
}

/**
 * Each stage's name (aligned with `index.stages`). A well-formed
 * INFO/PROVENANCE@stage_names names the MODEL_STAGE (host) stages when it
 * counts as many; a mode stage keeps its own name, its host prefix renamed.
 * Otherwise every stage keeps the file's name, and a malformed or miscounted
 * stage_names warns. An absent one is silent (the pairing says unstamped).
 */
function ladrunoStageNames(rf: LadrunoFile, warningsOut?: string[]): string[] {
  const ix = rf.index;
  const own = ix.stages.map((s) => s.name);
  const prov = ix.provenance;
  if (prov.state !== "present") return own;
  const fact = prov.attrs.stage_names;
  if (fact.state === "absent") return own;
  const warn = (w: string) => warningsOut?.push(`${ix.path}: ${w}; the file's own stage names are shown`);
  if (fact.state === "malformed" || prov.stageNames === null) {
    warn(fact.state === "malformed" ? fact.reason : "INFO/PROVENANCE@stage_names could not be read");
    return own;
  }
  const hosts = ix.stages.filter((s) => Object.hasOwn(ix.fileKinds, s.id));
  if (prov.stageNames.length !== hosts.length) {
    warn(`INFO/PROVENANCE@stage_names names ${prov.stageNames.length} stage(s), the file has ${hosts.length} MODEL_STAGE group(s)`);
    return own;
  }
  const renamed = new Map(hosts.map((h, k) => [h.id, { from: h.name, to: prov.stageNames![k]! }]));
  const hostOf = new Map(ix.modes.map((m) => [m.stageId, m.hostStageId]));
  return ix.stages.map((s) => {
    const r = renamed.get(s.id);
    if (r) return r.to;
    const host = hostOf.get(s.id);
    const h = host === undefined ? undefined : renamed.get(host);
    return h && s.name.startsWith(`${h.from}/`) ? `${h.to}/${s.name.slice(h.from.length + 1)}` : s.name;
  });
}

export function metaOf(rf: AnyResultsFile, warningsOut?: string[]): ResultsMeta {
  if (rf instanceof LadrunoFile) {
    const ix = rf.index;
    const prov = ix.provenance;
    // Lineage only when both hashes are stamped (reader/ladruno.ts `lineageOf`); else unpairable, with the reader's reason.
    const lineage = ix.lineage ? { femHash: ix.lineage.femHash, modelHash: ix.lineage.modelHash, resultsHash: null } : null;
    const space = prov.state === "present" ? prov.attrs.element_index_space : null;
    const elementIndexSpace = space?.state === "present" ? space.value : null;
    const names = ladrunoStageNames(rf, warningsOut);
    // #2128: a mode stage carries its eigen scalars, so the renderer can say which mode it draws.
    const modes = new Map<string, ModeFacts>(
      ix.modes.map((m) => [m.stageId, { index: m.modeIndex, eigenvalue: m.eigenvalue, frequencyHz: m.frequencyHz, periodS: m.periodS }]),
    );
    return {
      path: ix.path,
      version: "1.0.0",
      sourceType: "ladruno",
      lineage,
      lineageReason: ix.lineageReason,
      partial: null,
      unavailableZones: [],
      openseesVersion: null,
      programDecl: null,
      elementIndex: elementIndexSpace === "fem" ? { verified: true } : { verified: false, reason: unverifiedSpace(space) },
      units: ix.units,
      elementIndexSpace,
      stages: ix.stages.map((s, k) => ({
        id: s.id,
        name: names[k]!,
        kind: s.kind,
        time: Array.from(s.time),
        nodal: s.nodes ? [...s.nodes.components] : [],
        refusedNodal: s.nodes ? s.nodes.refused.map((r) => ({ ...r })) : [],
        gauss: [],
        unread: [...s.unread],
        elementIndexSpace,
        lineStations: rf.availableLineStations(s.id, warningsOut),
        ...(modes.has(s.id) ? { mode: modes.get(s.id)! } : {}),
      })),
    };
  }
  const i = rf.index;
  const verified = rf.elementIndexVerdict().verified;
  return {
    path: i.path,
    version: i.version,
    sourceType: i.sourceType,
    lineage: i.lineage,
    partial: i.partial,
    unavailableZones: [...i.unavailableZones],
    openseesVersion: i.openseesVersion,
    programDecl: i.programDecl,
    stages: i.stages.map((s) => ({
      id: s.id,
      name: s.name,
      kind: s.kind,
      time: Array.from(s.time),
      nodal: s.nodes ? [...s.nodes.components] : [],
      refusedNodal: s.nodes ? s.nodes.refused.map((r) => ({ ...r })) : [],
      gauss: s.gauss.map((g) => ({
        id: g.id,
        components: [...g.components],
        refused: g.refused.map((r) => ({ ...r })),
        nElements: g.nElements,
        nGauss: g.nGauss,
        localAxes: g.localAxes,
        naturalCoords: g.naturalCoords ? g.naturalCoords.map((r) => [...r]) : null,
      })),
      unread: [...s.unread],
      elementIndexSpace: i.elementIndexSpace,
      lineStations: verified ? rf.availableLineStations(s.id) : [],
    })),
    elementIndex: { ...rf.elementIndexVerdict() },
    units: i.units,
    elementIndexSpace: i.elementIndexSpace,
  };
}

export class ResultsHost {
  private readonly h5: ResultsH5;
  private readonly ready: Promise<unknown>;
  private readonly idleMs: number;
  /** the open handle, or null while idle */
  private file: AnyResultsFile | null = null;
  /** the file indexed by the last `open`, and its identity then; null before any open */
  private indexed: { path: string; signature: string; identity: string | null } | null = null;
  private timer: ReturnType<typeof setTimeout> | null = null;
  /** scans running now, and those asked to stop (each checks between chunks) */
  private readonly running = new Set<number>();
  private readonly cancelled = new Set<number>();

  constructor(h5: ResultsH5, ready: Promise<unknown>, opts: { idleMs?: number } = {}) {
    this.h5 = h5;
    this.ready = ready;
    this.idleMs = opts.idleMs ?? IDLE_CLOSE_MS;
  }

  /** Whether the HDF5 handle is open now (it closes after `idleMs` without a read). */
  isOpen(): boolean {
    return this.file !== null;
  }

  /** Open (or re-open) `path` and answer its index; any failure is an answer, never a throw. */
  async open(path: string): Promise<ResultsOpenAnswer> {
    this.close();
    this.indexed = null;
    // A re-index (a file change, or another file) stops every scan of the old index.
    for (const id of this.running) this.cancelled.add(id);
    try {
      await this.ready;
      const t = performance.now();
      const sig = signature(path);
      const rf = openAnyResults(this.h5, path);
      this.file = rf;
      this.indexed = { path, signature: sig, identity: identityOf(rf) };
      this.armIdle();
      const nodeIds: Record<string, Float64Array> = {};
      for (const st of rf.index.stages) if (st.nodes) nodeIds[st.id] = st.nodes.ids;
      // V4c2a (#1711/#1752 gate): element rows leave main only for a verified file.
      const elementIds: Record<string, Record<string, Float64Array>> = {};
      if (!(rf instanceof LadrunoFile) && rf.elementIndexVerdict().verified) {
        for (const st of rf.index.stages) {
          const byGroup: Record<string, Float64Array> = {};
          for (const g of st.gauss) {
            const e = rf.elementIds(st.id, g.id);
            if (e.mapping !== "fem_eid") throw new Error(`${g.path}: the file is verified but its rows read as ${e.mapping}`);
            byGroup[g.id] = e.ids;
          }
          elementIds[st.id] = byGroup;
        }
      }
      const warnings = [...rf.index.warnings];
      // One results slot per set (pairing.ts): the stem's other results file is never read; say so.
      const other = unopenedSibling(path, existsSync);
      if (other !== null) {
        warnings.push(
          `${baseName(other)} is not opened: a set reads one results file, and ${baseName(path)} is this one ` +
            "(the file opened wins; otherwise the .ladruno, ADR 0122 D1). Open the other file itself to read it",
        );
      }
      const meta = metaOf(rf, warnings);
      return { ok: true, meta, nodeIds, elementIds, warnings, sizeBytes: Number(sig.split(":")[0]), readMs: performance.now() - t };
    } catch (err) {
      this.close();
      return { ok: false, error: message(err) };
    }
  }

  /** The handle for a read of the indexed file, re-opened when idle; null when the file is no longer the one indexed. */
  private handle(path: string): AnyResultsFile | null {
    const ix = this.indexed;
    if (!ix || ix.path !== path) throw new Error(`results:readStep: ${path} is not the open results file`);
    // Size and mtime first, on every read: an in-place rewrite under an open handle shows here.
    if (signature(path) !== ix.signature) {
      this.close();
      return null;
    }
    if (this.file) return this.file;
    const rf = openAnyResults(this.h5, path);
    if (identityOf(rf) !== ix.identity) {
      rf.close();
      return null;
    }
    this.file = rf;
    return rf;
  }

  private armIdle(): void {
    if (this.timer !== null) clearTimeout(this.timer);
    this.timer = setTimeout(() => {
      this.timer = null;
      this.file?.close();
      this.file = null;
    }, this.idleMs);
    // An idle timer never keeps the process alive (tests, app quit).
    (this.timer as { unref?: () => void }).unref?.();
  }

  /** The nodal components of one step of the indexed file. Another path is refused; a changed file answers `changed`. */
  readStep(req: StepRequest): StepAnswer {
    try {
      const rf = this.handle(req.path);
      if (!rf) {
        return { ok: false, seq: req.seq, changed: true, error: `${req.path} changed on disk since it was indexed; it is read again before any step is shown` };
      }
      this.armIdle();
      const s = rf.stage(req.stage);
      const comps = s.nodes ? s.nodes.components : [];
      const values = comps.length ? rf.readNodalStep(s.id, comps, req.step) : [];
      // A stage with no nodal results still checks the step against its time vector.
      if (!comps.length && (!Number.isInteger(req.step) || req.step < 0 || req.step >= s.nSteps)) {
        throw new Error(`step ${req.step} is outside stage ${s.id} (0..${s.nSteps - 1})`);
      }
      const nodal: Record<string, Float64Array> = {};
      comps.forEach((c, i) => (nodal[c] = values[i]!));
      const gauss: Record<string, Float64Array> = {};
      if (req.element !== undefined) {
        if (rf instanceof LadrunoFile) {
          throw new Error(`element gauss results are not served from .ladruno file ${req.path}`);
        }
        const st = s as StageInfo;
        const v = rf.elementIndexVerdict();
        // Fail closed: an unverified file's element values are never served (the renderer refuses them first).
        if (!v.verified) throw new Error(`element results of ${req.path} are refused: ${v.reason}`);
        for (const g of st.gauss) if (g.components.includes(req.element)) gauss[g.id] = rf.readGaussStep(s.id, g.id, req.element, req.step);
        if (!Object.keys(gauss).length) throw new Error(`stage ${s.id} has no gauss group with component ${JSON.stringify(req.element)}`);
      }
      let lineDiagram: LadrunoLineStationStep | null | undefined;
      if (req.lineDiagram !== undefined) {
        if (rf instanceof LadrunoFile) {
          lineDiagram = rf.readLineStationStep(s.id, req.lineDiagram, req.step);
        } else {
          const v = rf.elementIndexVerdict();
          if (!v.verified) throw new Error(`line-station results of ${req.path} are refused: ${v.reason}`);
          lineDiagram = rf.readLineStationStep(s.id, req.lineDiagram, req.step);
        }
      }
      return {
        ok: true,
        seq: req.seq,
        stage: s.id,
        step: req.step,
        nodal,
        gauss,
        ...(lineDiagram !== undefined ? { lineDiagram } : {}),
      };
    } catch (err) {
      return { ok: false, seq: req.seq, error: message(err) };
    }
  }

  /** Ask scan `scanId` to stop; it answers `cancelled` at its next chunk. */
  cancelScan(scanId: number): void {
    if (this.running.has(scanId)) this.cancelled.add(scanId);
  }

  /** Scans running now (tests check that a cancel stops one). */
  runningScans(): number[] {
    return [...this.running];
  }

  /**
   * The history range of `req.components` (their magnitude when
   * `req.magnitude`) over every step of `req.stage`, read in chunks of
   * `chunkSteps` steps (about 4 MB each by default) with a yield between
   * chunks, so step reads are served while it runs. NaN values (a node a
   * merged record did not cover, h5-schema.md) are not values and are
   * skipped, as the Qt contour skips them. `paceMs` waits between chunks
   * (capture mode only, so a still can show a scan in progress).
   */
  async scan(req: ScanRequest, progress: (p: ScanProgress) => void, opts: { chunkSteps?: number; paceMs?: number; progressMs?: number } = {}): Promise<ScanAnswer> {
    const id = req.scanId;
    this.running.add(id);
    const t = performance.now();
    try {
      let rf = this.handle(req.path);
      if (!rf) return { ok: false, scanId: id, changed: true, error: `${req.path} changed on disk since it was indexed` };
      const st = rf.stage(req.stage);
      if (req.source === "line") return await this.scanLine(req, rf, st, progress, opts, t);
      if (req.source === "element") {
        if (rf instanceof LadrunoFile) throw new Error(`element gauss scan is not supported on .ladruno file ${req.path}`);
        return await this.scanElement(req, rf, st as StageInfo, progress, opts, t);
      }
      const n = st.nodes?.ids.length ?? 0;
      if (!st.nodes) throw new Error(`stage ${st.id} has no nodal results`);
      if (req.components.length === 0) throw new Error("a scan needs at least one component");
      if (!req.magnitude && req.components.length !== 1) throw new Error(`a component scan reads one component; got ${req.components.length}`);
      const T = st.nSteps;
      const chunk = opts.chunkSteps ?? Math.max(1, Math.floor(4e6 / (8 * Math.max(1, n) * req.components.length)));
      let lo = Infinity;
      let hi = -Infinity;
      let lastProgress = -Infinity;
      for (let k0 = 0; k0 < T; k0 += chunk) {
        if (this.cancelled.has(id)) return { ok: false, scanId: id, cancelled: true, error: "cancelled" };
        rf = this.handle(req.path);
        if (!rf) return { ok: false, scanId: id, changed: true, error: `${req.path} changed on disk during the scan` };
        this.armIdle();
        const k1 = Math.min(T, k0 + chunk);
        const rows = rf.readNodalRows(st.id, req.components, k0, k1);
        const len = rows[0]!.length;
        if (req.magnitude) {
          for (let i = 0; i < len; i++) {
            let sq = 0;
            for (const r of rows) sq += r[i]! * r[i]!;
            const v = Math.sqrt(sq);
            if (!Number.isFinite(v)) continue;
            if (v < lo) lo = v;
            if (v > hi) hi = v;
          }
        } else {
          const r = rows[0]!;
          for (let i = 0; i < len; i++) {
            const v = r[i]!;
            if (!Number.isFinite(v)) continue;
            if (v < lo) lo = v;
            if (v > hi) hi = v;
          }
        }
        const now = performance.now();
        if (k1 < T && now - lastProgress >= (opts.progressMs ?? 100)) {
          lastProgress = now;
          progress({ scanId: id, done: k1, total: T });
        }
        await new Promise((r) => (opts.paceMs ? setTimeout(r, opts.paceMs) : setImmediate(r)));
      }
      if (this.cancelled.has(id)) return { ok: false, scanId: id, cancelled: true, error: "cancelled" };
      const finite = lo <= hi;
      return { ok: true, scanId: id, min: finite ? lo : null, max: finite ? hi : null, steps: T, ms: performance.now() - t };
    } catch (err) {
      return { ok: false, scanId: id, error: message(err) };
    } finally {
      this.running.delete(id);
      this.cancelled.delete(id);
    }
  }

  /**
   * The line-station history range (.ladruno or verified results.h5): one 1-D beam
   * line-station component across all steps of `st.id`. Runs inside `scan`.
   */
  private async scanLine(
    req: ScanRequest,
    first: AnyResultsFile,
    st: StageInfo,
    progress: (p: ScanProgress) => void,
    opts: { chunkSteps?: number; paceMs?: number; progressMs?: number },
    t: number,
  ): Promise<ScanAnswer> {
    const id = req.scanId;
    if (!(first instanceof LadrunoFile)) {
      const v = first.elementIndexVerdict();
      if (!v.verified) throw new Error(`line-station results of ${req.path} are refused: ${v.reason}`);
    }
    if (req.components.length !== 1 || req.magnitude) throw new Error("a line-station scan reads one component, never a magnitude");
    const comp = req.components[0]!;
    const T = st.nSteps;
    if (this.cancelled.has(id)) return { ok: false, scanId: id, cancelled: true, error: "cancelled" };
    this.armIdle();
    const slab = first.readLineStations(st.id, comp);
    const nRows = slab.elementIndex.length;
    const chunk = opts.chunkSteps ?? Math.max(1, Math.floor(4e6 / (8 * Math.max(1, nRows))));
    let lo = Infinity;
    let hi = -Infinity;
    let lastProgress = -Infinity;
    for (let k0 = 0; k0 < T; k0 += chunk) {
      if (this.cancelled.has(id)) return { ok: false, scanId: id, cancelled: true, error: "cancelled" };
      const rf = this.handle(req.path);
      if (!rf) return { ok: false, scanId: id, changed: true, error: `${req.path} changed on disk during the scan` };
      this.armIdle();
      const k1 = Math.min(T, k0 + chunk);
      const start = k0 * nRows;
      const end = k1 * nRows;
      for (let i = start; i < end; i++) {
        const v = slab.values[i]!;
        if (!Number.isFinite(v)) continue;
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
      const now = performance.now();
      if (k1 < T && now - lastProgress >= (opts.progressMs ?? 100)) {
        lastProgress = now;
        progress({ scanId: id, done: k1, total: T });
      }
      await new Promise((r) => (opts.paceMs ? setTimeout(r, opts.paceMs) : setImmediate(r)));
    }
    if (this.cancelled.has(id)) return { ok: false, scanId: id, cancelled: true, error: "cancelled" };
    const finite = lo <= hi;
    return { ok: true, scanId: id, min: finite ? lo : null, max: finite ? hi : null, steps: T, ms: performance.now() - t };
  }

  /**
   * The element history range (V4c2a): one Gauss component of every group
   * that has it, reduced per element (gaussCellValues: NaN when any of the
   * element's points is not finite), in chunks of steps. Runs inside `scan`.
   */
  private async scanElement(
    req: ScanRequest,
    first: ResultsFile,
    st: StageInfo,
    progress: (p: ScanProgress) => void,
    opts: { chunkSteps?: number; paceMs?: number; progressMs?: number },
    t: number,
  ): Promise<ScanAnswer> {
    const id = req.scanId;
    const v = first.elementIndexVerdict();
    if (!v.verified) throw new Error(`element results of ${req.path} are refused: ${v.reason}`);
    if (req.components.length !== 1 || req.magnitude) throw new Error("an element scan reads one component, never a magnitude");
    const comp = req.components[0]!;
    const groups = st.gauss.filter((g) => g.components.includes(comp));
    if (!groups.length) throw new Error(`stage ${st.id} has no gauss group with component ${JSON.stringify(comp)}`);
    const plan = req.plan;
    if (plan) {
      // The plan names the field's groups, in its order and sizes; anything else is refused, never re-mapped.
      const same = plan.groups.length === groups.length && plan.groups.every((pg, i) => pg.id === groups[i]!.id && pg.nElements === groups[i]!.nElements && pg.nGauss === groups[i]!.nGauss);
      if (!same) throw new Error(`the extrapolation plan's groups (${plan.groups.map((g) => g.id).join(", ")}) are not stage ${st.id}'s groups with ${comp}`);
    }
    const T = st.nSteps;
    const per = groups.reduce((a, g) => a + g.nElements * g.nGauss, 0);
    const chunk = opts.chunkSteps ?? Math.max(1, Math.floor(4e6 / (8 * Math.max(1, per))));
    let lo = Infinity;
    let hi = -Infinity;
    let lastProgress = -Infinity;
    for (let k0 = 0; k0 < T; k0 += chunk) {
      if (this.cancelled.has(id)) return { ok: false, scanId: id, cancelled: true, error: "cancelled" };
      const rf = this.handle(req.path);
      if (!rf || rf instanceof LadrunoFile) return { ok: false, scanId: id, changed: true, error: `${req.path} changed on disk during the scan` };
      this.armIdle();
      const k1 = Math.min(T, k0 + chunk);
      if (plan) {
        const rows = groups.map((g) => rf.readGaussRows(st.id, g.id, comp, k0, k1));
        for (let k = 0; k < k1 - k0; k++) {
          const r = planRange(applyPlan(plan, rows.map((x, i) => x.subarray(k * groups[i]!.nElements * groups[i]!.nGauss, (k + 1) * groups[i]!.nElements * groups[i]!.nGauss))));
          if (r && r.min < lo) lo = r.min;
          if (r && r.max > hi) hi = r.max;
        }
      } else {
        for (const g of groups) {
          const rows = rf.readGaussRows(st.id, g.id, comp, k0, k1);
          const vals = gaussCellValues(rows, (k1 - k0) * g.nElements, g.nGauss);
          for (let i = 0; i < vals.length; i++) {
            const x = vals[i]!;
            if (!Number.isFinite(x)) continue;
            if (x < lo) lo = x;
            if (x > hi) hi = x;
          }
        }
      }
      const now = performance.now();
      if (k1 < T && now - lastProgress >= (opts.progressMs ?? 100)) {
        lastProgress = now;
        progress({ scanId: id, done: k1, total: T });
      }
      await new Promise((r) => (opts.paceMs ? setTimeout(r, opts.paceMs) : setImmediate(r)));
    }
    if (this.cancelled.has(id)) return { ok: false, scanId: id, cancelled: true, error: "cancelled" };
    const finite = lo <= hi;
    return { ok: true, scanId: id, min: finite ? lo : null, max: finite ? hi : null, steps: T, ms: performance.now() - t };
  }

  /** Close the handle now (the index is kept: the next read re-opens and checks it). */
  close(): void {
    if (this.timer !== null) clearTimeout(this.timer);
    this.timer = null;
    this.file?.close();
    this.file = null;
  }
}
