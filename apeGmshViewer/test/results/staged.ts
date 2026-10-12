// The V4b staged pair (fixtures/staged_activation.h5 and .results.h5, written
// by the repository's scripts/make_viewer_results_fixtures.py --staged) read
// the way the app reads it: the model through reader/node.ts, its lineage and
// /sequence element tags through main/zones.ts, the results through
// main/results.ts, all into one store state. Shared by the V4b tests.

import { fileURLToPath } from "node:url";
import * as h5wasm from "h5wasm/node";
import { modelExtras } from "../../src/effects.ts";
import { ResultsHost } from "../../src/main/results.ts";
import { openLineage, openStageElements } from "../../src/main/zones.ts";
import { openModel } from "../../src/reader/node.ts";
import type { ResultsH5 } from "../../src/reader/results.ts";
import { BlobStore } from "../../src/state/blobs.ts";
import { loadModel } from "../../src/state/load.ts";
import { initialState, reduce, type State } from "../../src/state/store.ts";
import type { ModelLoad, ResultsInfo } from "../../src/state/types.ts";

export const MODEL = fileURLToPath(new URL("../../fixtures/staged_activation.h5", import.meta.url));
export const RESULTS = fileURLToPath(new URL("../../fixtures/staged_activation.results.h5", import.meta.url));
export const ORACLE = fileURLToPath(new URL("../../fixtures/staged_activation.oracle.json", import.meta.url));

export const host = new ResultsHost(h5wasm as unknown as ResultsH5, h5wasm.ready);

/** The model as `model:open` answers it, loaded as the effects load it. */
export async function stagedModelLoad(path = MODEL, blobs = new BlobStore()): Promise<ModelLoad> {
  const model = await openModel(path);
  const extras = modelExtras({ lineage: await openLineage(path), stageElements: await openStageElements(path, model.sequence) });
  return loadModel(model, blobs, null, null, extras);
}

/** The results as `results:open` answers them. */
export async function stagedResults(path = RESULTS): Promise<{ info: ResultsInfo; sizeBytes: number; readMs: number; warnings: string[] }> {
  const a = await host.open(path);
  if (!a.ok) throw new Error(a.error);
  return { info: a.meta, sizeBytes: a.sizeBytes, readMs: a.readMs, warnings: a.warnings };
}

/** The store with the staged model and its results read (the phase on the mesh). */
export async function stagedState(): Promise<State> {
  const load = await stagedModelLoad();
  const r = await stagedResults();
  let s = reduce(initialState, { type: "fileLoaded", artifact: "model", load });
  s = reduce(s, { type: "fileOpened", artifact: "results", path: r.info.path });
  return reduce(s, { type: "resultsLoaded", ...r });
}

/** The FEM ids of the element declarations whose OpenSees tag is in `tags` (every element, drawn or not). */
export function femIdsOf(s: State, tags: ReadonlySet<number>): number[] {
  const out: number[] = [];
  for (const d of Object.values(s.decls)) {
    if (d.kind !== "element" || !d.element || d.element.femId === null) continue;
    if (d.element.metas.some((m) => tags.has(m.tag))) out.push(d.element.femId);
  }
  return out.sort((a, b) => a - b);
}
