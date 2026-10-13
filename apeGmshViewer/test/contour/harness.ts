// The V4c1 contour fixtures (fixtures/contour_staged.* and contour_nodata.*,
// written by the repository's scripts/make_viewer_results_fixtures.py
// --contour) read through the app's own code in Node: the real Effects over a
// bridge that calls main's readers directly (main/results.ts, main/zones.ts,
// reader/node.ts), so a test drives the whole renderer-side path without
// Electron. The closed form is the generator's, restated here (not copied
// from the app): s = 0.01 (stage + 1) (k + 1) / T, displacement_x = s z,
// displacement_y = 0.5 s x, displacement_z = -0.2 s (x + z).

import { fileURLToPath } from "node:url";
import * as h5wasm from "h5wasm/node";
import { Effects, type Bridge } from "../../src/effects.ts";
import { ResultsHost, type ScanProgress } from "../../src/main/results.ts";
import { openLineage, openStageElements } from "../../src/main/zones.ts";
import { openModel } from "../../src/reader/node.ts";
import type { ResultsH5 } from "../../src/reader/results.ts";
import { BlobStore } from "../../src/state/blobs.ts";
import { Store } from "../../src/state/store.ts";

export const STAGED = fileURLToPath(new URL("../../fixtures/contour_staged.h5", import.meta.url));
export const STAGED_RESULTS = fileURLToPath(new URL("../../fixtures/contour_staged.results.h5", import.meta.url));
export const NODATA = fileURLToPath(new URL("../../fixtures/contour_nodata.h5", import.meta.url));
export const NODATA_RESULTS = fileURLToPath(new URL("../../fixtures/contour_nodata.results.h5", import.meta.url));

/** The generator's stage order and step counts (STAGED_STEPS). */
export const STAGES = [
  { name: "base", T: 3 },
  { name: "erect", T: 2 },
  { name: "unprop", T: 4 },
] as const;

/** The closed form at one node: [ux, uy, uz]. */
export function closedForm(x: number, z: number, stage: number, k: number, T: number): [number, number, number] {
  const s = (0.01 * (stage + 1) * (k + 1)) / T;
  return [s * z, 0.5 * s * x, -0.2 * s * (x + z)];
}

export const tick = () => new Promise((r) => setTimeout(r, 0));

/** Wait (polling every 5 ms, at most 5 s) until `ok` holds. */
export async function until(ok: () => boolean, what: string): Promise<void> {
  for (let i = 0; i < 1000; i++) {
    if (ok()) return;
    await new Promise((r) => setTimeout(r, 5));
  }
  throw new Error(`timed out waiting for ${what}`);
}

/**
 * The app's renderer-side path over main's code: a store, the BlobStore, the
 * Effects, and a bridge whose calls go straight to the readers. `scan` lets a
 * test hold or wrap the scan; `calls` records the cancels.
 */
export function harness(opts: { scanOpts?: { chunkSteps?: number; paceMs?: number } } = {}) {
  const host = new ResultsHost(h5wasm as unknown as ResultsH5, h5wasm.ready, { idleMs: 50 });
  const progress: ((p: ScanProgress) => void)[] = [];
  const calls = { cancels: [] as number[], scans: [] as { scanId: number; components: string[]; magnitude: boolean }[] };
  const bridge: Bridge = {
    config: async () => ({ mode: "view", file: null, t0: 0, appReadyMs: 0, configMs: 0, pick: null }),
    openModel: async (path: string) => {
      const model = await openModel(path);
      return { ok: true, model, lineage: await openLineage(path), stageElements: await openStageElements(path, model.sequence) };
    },
    openResults: (path: string) => host.open(path),
    readResultsStep: async (req) => host.readStep(req),
    scanResults: (req) => {
      calls.scans.push({ scanId: req.scanId, components: req.components, magnitude: req.magnitude });
      return host.scan(req, (p) => progress.forEach((cb) => cb(p)), opts.scanOpts ?? {});
    },
    cancelResultsScan: async (id: number) => {
      calls.cancels.push(id);
      host.cancelScan(id);
    },
    onScanProgress: (cb) => {
      progress.push(cb);
      return () => progress.splice(progress.indexOf(cb), 1);
    },
    pathForFile: () => "",
    metrics: async () => ({ mainMB: 0, rendererMB: 0, gpuProcessMB: 0, gpuDevices: [] }),
    measureDone: async () => undefined,
    captureStill: async () => null,
    captureDone: async () => undefined,
    fail: async () => undefined,
  };
  const store = new Store();
  const blobs = new BlobStore();
  const effects = new Effects(store, blobs, bridge);
  return { host, store, blobs, effects, calls };
}

/** The model and its results opened through the effects, as an open set would. */
export async function opened(model = STAGED, results = STAGED_RESULTS, opts: Parameters<typeof harness>[0] = {}) {
  const h = harness(opts);
  if (!(await h.effects.open(model))) throw new Error(`model ${model} did not load: ${h.store.get().artifacts.model?.error}`);
  if (!(await h.effects.openResults(results))) throw new Error(`results ${results} did not load: ${h.store.get().artifacts.results?.error}`);
  return h;
}
