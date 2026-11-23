// The loader process (ADR 0137 D2, #3058): the HDF5 reads that take seconds on a large model,
// run outside the main process, so the thread that owns the windows keeps answering while a
// file opens or a history scans. Main starts it with Electron's `utilityProcess.fork`
// (dist/loader.mjs, ./fork.ts) and talks to it through ./client.ts. The door tests fork it as a
// plain Node child instead (test/main/electronStub.ts), so it imports no Electron and speaks
// over either port.
//
// It reads only the paths main sends. Main has already checked each one against the window's
// set, the compare slot or the user's gesture; the loader checks nothing of that.
//
// It holds the two results hosts main used to hold (./results.ts `ResultsHost`): the window
// set's and the compare set's, each with its open file, index, idle close and running scans.
// Requests are not queued: a scan yields between chunks, and a step read or a cancel that
// arrives meanwhile is served then, as it was in main.

import * as h5wasm from "h5wasm/node";
import { openModel } from "../../reader/node.ts";
import type { ResultsH5 } from "../../reader/results.ts";
import { readParticleSet, ResultsHost, type HingeHistoryRequest, type ParticleOutputRequest, type ScanRequest, type StepRequest } from "../results.ts";
import { openCellCount, openGeometry, openLineage, openProvenance, openStageElements } from "../zones.ts";
import type { LoaderMessage, LoaderOps, LoaderRequest, ResultsCall, ResultsHostName, ScanOptions } from "./protocol.ts";

interface Port {
  post(m: LoaderMessage): void;
  listen(fn: (m: unknown) => void): void;
}

/** Electron's `process.parentPort` in a utility process; Node's IPC channel in a forked child. */
function parentPort(): Port {
  const pp = (process as { parentPort?: { postMessage(m: unknown): void; on(e: "message", fn: (e: { data: unknown }) => void): void } }).parentPort;
  if (pp) return { post: (m) => pp.postMessage(m), listen: (fn) => pp.on("message", (e) => fn(e.data)) };
  const send = process.send?.bind(process);
  if (send) return { post: (m) => void send(m), listen: (fn) => void process.on("message", fn) };
  throw new Error("the loader runs only as an Electron utility process or a forked Node child");
}

const h5 = h5wasm as unknown as ResultsH5;

const reads: { [K in keyof LoaderOps]: (path: string) => Promise<LoaderOps[K]> } = {
  model: async (path) => {
    const model = await openModel(path);
    // /provenance rides with the model; a refused or broken zone does not stop the model from loading.
    return { model, provenance: await openProvenance(path), lineage: await openLineage(path), stageElements: await openStageElements(path, model.sequence) };
  },
  modelCounts: async (path) => {
    const model = await openModel(path);
    return { nodes: model.nodeIds.length, elements: model.blocks.reduce((n, b) => n + b.ids.length, 0) };
  },
  cells: openCellCount,
  geometry: openGeometry,
  provenance: openProvenance,
  particleSet: (path) => readParticleSet(h5, h5wasm.ready, path),
};

const hosts: Record<ResultsHostName, ResultsHost> = {
  window: new ResultsHost(h5, h5wasm.ready),
  compare: new ResultsHost(h5, h5wasm.ready),
};

const port = parentPort();

/** One call on a host. The arguments are main's, already checked there (main.ts validates every request's shape). */
function call(host: ResultsHostName, name: ResultsCall, args: unknown[]): unknown {
  const h = hosts[host];
  switch (name) {
    case "open":
      return h.open(args[0] as string);
    case "readStep":
      return h.readStep(args[0] as StepRequest);
    case "readHingeHistory":
      return h.readHingeHistory(args[0] as HingeHistoryRequest);
    case "scan":
      return h.scan(args[0] as ScanRequest, (progress) => port.post({ event: "scanProgress", host, progress }), args[1] as ScanOptions);
    case "cancelScan":
      return h.cancelScan(args[0] as number);
    case "particleIndex":
      return h.particleIndex(args[0] as string);
    case "readParticleOutput":
      return h.readParticleOutput(args[0] as ParticleOutputRequest);
    case "close":
      return h.close();
  }
}

const CALLS = new Set<string>(["open", "readStep", "readHingeHistory", "scan", "cancelScan", "particleIndex", "readParticleOutput", "close"]);

function isRequest(m: unknown): m is LoaderRequest {
  const r = m as LoaderRequest | null;
  if (r === null || typeof r !== "object" || typeof r.id !== "number") return false;
  if (r.op === "results") return (r.host === "window" || r.host === "compare") && CALLS.has(r.call) && Array.isArray(r.args);
  return typeof r.path === "string" && Object.hasOwn(reads, r.op);
}

// h5wasm compiles its module on start, before the first request arrives (main starts the loader with its window).
void h5wasm.ready;
port.listen((m) => {
  if (!isRequest(m)) return; // main sends only requests; anything else has no id to answer
  void (async () => {
    try {
      const value = m.op === "results" ? await call(m.host, m.call, m.args) : await reads[m.op](m.path);
      port.post({ id: m.id, ok: true, value });
    } catch (err) {
      port.post({ id: m.id, ok: false, error: err instanceof Error ? err.message : String(err) });
    }
  })();
});
