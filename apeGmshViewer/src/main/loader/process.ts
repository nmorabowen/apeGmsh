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
import { readParticleSet, ResultsHost } from "../results.ts";
import { openCellCount, openGeometry, openLineage, openProvenance, openStageElements } from "../zones.ts";
import type { LoaderMessage, LoaderOps, LoaderRequest, ResultsCall, ResultsCalls, ResultsHostName, ResultsRequest } from "./protocol.ts";

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

/**
 * Each results call, on a host. The one list of calls: the mapped type makes it name every call of
 * protocol.ts `ResultsCalls` with that call's arguments, and `isRequest` reads its keys. The
 * arguments are main's, already checked there (main.ts validates every request's shape).
 */
const CALLS: { [K in ResultsCall]: (h: ResultsHost, host: ResultsHostName, ...args: ResultsCalls[K]["args"]) => ResultsCalls[K]["answer"] | Promise<ResultsCalls[K]["answer"]> } = {
  open: (h, _host, path) => h.open(path),
  readStep: (h, _host, req) => h.readStep(req),
  readHingeHistory: (h, _host, req) => h.readHingeHistory(req),
  readNodeHistory: (h, _host, req) => h.readNodeHistory(req),
  stepRange: (h, _host, req) => h.stepRange(req),
  stepCounts: (h) => h.stepCounts(),
  scan: (h, host, req, opts) => h.scan(req, (progress) => port.post({ event: "scanProgress", host, progress }), opts),
  cancelScan: (h, _host, scanId) => h.cancelScan(scanId),
  particleIndex: (h, _host, path) => h.particleIndex(path),
  readParticleOutput: (h, _host, req) => h.readParticleOutput(req),
  close: (h) => h.close(),
};

function call(r: ResultsRequest): unknown {
  const run = CALLS[r.call] as (h: ResultsHost, host: ResultsHostName, ...args: unknown[]) => unknown;
  return run(hosts[r.host], r.host, ...r.args);
}

/** Why `m` is not a request this loader serves, or null when it is one. */
export function whyNotRequest(m: unknown): string | null {
  const r = m as Record<string, unknown> | null;
  if (r === null || typeof r !== "object" || typeof r["id"] !== "number") return "not a request";
  if (r["op"] === "results") {
    if (r["host"] !== "window" && r["host"] !== "compare") return `no results host ${JSON.stringify(r["host"])}`;
    if (typeof r["call"] !== "string" || !Object.hasOwn(CALLS, r["call"])) return `the loader has no results call ${JSON.stringify(r["call"])}`;
    return Array.isArray(r["args"]) ? null : "a results call takes an args array";
  }
  if (typeof r["op"] !== "string" || !Object.hasOwn(reads, r["op"])) return `the loader has no op ${JSON.stringify(r["op"])}`;
  return typeof r["path"] === "string" ? null : `op ${r["op"]} takes a path`;
}

// h5wasm compiles its module on start, before the first request arrives (main starts the loader with its window).
void h5wasm.ready;
port.listen((m) => {
  const why = whyNotRequest(m);
  if (why !== null) {
    // Answer anything with an id, so main's promise settles with why; a message with no id has nothing to answer.
    const id = (m as { id?: unknown } | null)?.id;
    if (typeof id === "number") port.post({ id, ok: false, error: `loader: ${why}` });
    return;
  }
  const req = m as LoaderRequest;
  void (async () => {
    try {
      const value = req.op === "results" ? await call(req) : await reads[req.op](req.path);
      port.post({ id: req.id, ok: true, value });
    } catch (err) {
      port.post({ id: req.id, ok: false, error: err instanceof Error ? err.message : String(err) });
    }
  })();
});
