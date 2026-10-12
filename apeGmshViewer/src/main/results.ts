// The results file of the view window, read lazily in main (ADR 0112 D5,
// the #1283 ruling of 2026-10-11; V4b, #1728).
//
// Main holds the one open ResultsFile (reader/results.ts). The renderer gets
// its index as plain metadata (`open`) and one step at a time (`readStep`):
// the store never holds a component array, and nothing is read eagerly. A
// re-read of the same path (the file changed) closes the old handle first.
// No Electron import: the tests drive this class in Node.

import { statSync } from "node:fs";
import { openResults, type ResultsFile, type ResultsH5, type StageInfo } from "../reader/results.ts";

/** One stage as the renderer's store keeps it (state/types.ts `ResultsStage`). */
export interface ResultsStageMeta {
  id: string;
  name: string;
  kind: StageInfo["kind"];
  time: number[];
  nodal: string[];
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
  | { ok: true; meta: ResultsMeta; warnings: string[]; sizeBytes: number; readMs: number }
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
  | { ok: false; seq: number; error: string };

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
      gauss: s.gauss.map((g) => ({ id: g.id, components: [...g.components] })),
      unread: [...s.unread],
    })),
  };
}

export class ResultsHost {
  private readonly h5: ResultsH5;
  private readonly ready: Promise<unknown>;
  private file: ResultsFile | null = null;

  constructor(h5: ResultsH5, ready: Promise<unknown>) {
    this.h5 = h5;
    this.ready = ready;
  }

  /** Open (or re-open) `path` and answer its index; any failure is an answer, never a throw. */
  async open(path: string): Promise<ResultsOpenAnswer> {
    this.close();
    try {
      await this.ready;
      const t = performance.now();
      const rf = openResults(this.h5, path);
      this.file = rf;
      return { ok: true, meta: metaOf(rf), warnings: [...rf.index.warnings], sizeBytes: statSync(path).size, readMs: performance.now() - t };
    } catch (err) {
      return { ok: false, error: message(err) };
    }
  }

  /** The nodal components of one step of the open file. A request for another path is refused. */
  readStep(req: StepRequest): StepAnswer {
    try {
      const rf = this.file;
      if (!rf || rf.index.path !== req.path) throw new Error(`results:readStep: ${req.path} is not the open results file`);
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

  close(): void {
    this.file?.close();
    this.file = null;
  }
}
