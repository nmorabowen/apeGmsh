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

import { statSync } from "node:fs";
import { openResults, type ResultsFile, type ResultsH5, type StageInfo } from "../reader/results.ts";

/** One stage as the renderer's store keeps it (state/types.ts `ResultsStage`). */
export interface ResultsStageMeta {
  id: string;
  name: string;
  kind: StageInfo["kind"];
  time: number[];
  nodal: string[];
  /** V4c1: nodal components refused one by one (value columns that disagree with `_ids`) */
  refusedNodal: { component: string; reason: string }[];
  gauss: { id: string; components: string[] }[];
  unread: string[];
}

export interface ResultsMeta {
  path: string;
  version: string;
  sourceType: string | null;
  lineage: { femHash: string | null; modelHash: string | null; resultsHash: string | null } | null;
  partial: string | null;
  stages: ResultsStageMeta[];
  unavailableZones: string[];
}

export type ResultsOpenAnswer =
  /** `nodeIds`: each stage's nodal `_ids` by stage id (V4c1), outside `meta` so the store never holds an array */
  | { ok: true; meta: ResultsMeta; nodeIds: Record<string, Float64Array>; warnings: string[]; sizeBytes: number; readMs: number }
  | { ok: false; error: string };

/** A step request: the stage by id (and index, echoed back), the step, and the renderer's request number. */
export interface StepRequest {
  path: string;
  stage: string;
  step: number;
  seq: number;
}

export type StepAnswer =
  | { ok: true; seq: number; stage: string; step: number; nodal: Record<string, Float64Array> }
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

export function metaOf(rf: ResultsFile): ResultsMeta {
  const i = rf.index;
  return {
    path: i.path,
    version: i.version,
    sourceType: i.sourceType,
    lineage: i.lineage,
    partial: i.partial,
    unavailableZones: [...i.unavailableZones],
    stages: i.stages.map((s) => ({
      id: s.id,
      name: s.name,
      kind: s.kind,
      time: Array.from(s.time),
      nodal: s.nodes ? [...s.nodes.components] : [],
      refusedNodal: s.nodes ? s.nodes.refused.map((r) => ({ ...r })) : [],
      gauss: s.gauss.map((g) => ({ id: g.id, components: [...g.components] })),
      unread: [...s.unread],
    })),
  };
}

export class ResultsHost {
  private readonly h5: ResultsH5;
  private readonly ready: Promise<unknown>;
  private readonly idleMs: number;
  /** the open handle, or null while idle */
  private file: ResultsFile | null = null;
  /** the file indexed by the last `open`, and its identity then; null before any open */
  private indexed: { path: string; signature: string; resultsHash: string | null } | null = null;
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
      const rf = openResults(this.h5, path);
      this.file = rf;
      this.indexed = { path, signature: sig, resultsHash: rf.index.lineage?.resultsHash ?? null };
      this.armIdle();
      const nodeIds: Record<string, Float64Array> = {};
      for (const st of rf.index.stages) if (st.nodes) nodeIds[st.id] = st.nodes.ids;
      return { ok: true, meta: metaOf(rf), nodeIds, warnings: [...rf.index.warnings], sizeBytes: Number(sig.split(":")[0]), readMs: performance.now() - t };
    } catch (err) {
      this.close();
      return { ok: false, error: message(err) };
    }
  }

  /** The handle for a read of the indexed file, re-opened when idle; null when the file is no longer the one indexed. */
  private handle(path: string): ResultsFile | null {
    const ix = this.indexed;
    if (!ix || ix.path !== path) throw new Error(`results:readStep: ${path} is not the open results file`);
    // Size and mtime first, on every read: an in-place rewrite under an open handle shows here.
    if (signature(path) !== ix.signature) {
      this.close();
      return null;
    }
    if (this.file) return this.file;
    const rf = openResults(this.h5, path);
    if ((rf.index.lineage?.resultsHash ?? null) !== ix.resultsHash) {
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
      return { ok: true, seq: req.seq, stage: s.id, step: req.step, nodal };
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

  /** Close the handle now (the index is kept: the next read re-opens and checks it). */
  close(): void {
    if (this.timer !== null) clearTimeout(this.timer);
    this.timer = null;
    this.file?.close();
    this.file = null;
  }
}
