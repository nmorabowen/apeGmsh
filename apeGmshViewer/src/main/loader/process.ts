// The loader process (ADR 0137 D2, #3058): the HDF5 reads that take seconds on a large model,
// run outside the main process, so the thread that owns the windows keeps answering while a
// file opens. Main starts it with Electron's `utilityProcess.fork` (dist/loader.mjs) and talks
// to it through ./client.ts. The door tests fork it as a plain Node child instead
// (test/main/electronStub.ts), so it imports no Electron and speaks over either port.
//
// It reads only the path main sends. Main has already checked that path against the window's
// set and the user's gesture; the loader checks nothing and keeps no file open between requests.

import * as h5wasm from "h5wasm/node";
import { openModel } from "../../reader/node.ts";
import { openGeometry, openLineage, openProvenance, openStageElements } from "../zones.ts";
import type { LoaderOps, LoaderReply, LoaderRequest } from "./protocol.ts";

interface Port {
  post(m: LoaderReply): void;
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
  geometry: openGeometry,
  provenance: openProvenance,
};

function isRequest(m: unknown): m is LoaderRequest {
  const r = m as LoaderRequest | null;
  return r !== null && typeof r === "object" && typeof r.id === "number" && typeof r.path === "string" && Object.hasOwn(reads, r.op);
}

const port = parentPort();
// h5wasm compiles its module on start, before the first request arrives (main starts the loader with its window).
void h5wasm.ready;
// One read at a time, in the order main asked: h5wasm is synchronous, and a queue keeps the
// awaits of one request (h5wasm.ready, the zones after the model) from interleaving with another's.
let queue: Promise<void> = Promise.resolve();
port.listen((m) => {
  if (!isRequest(m)) return; // main sends only requests; anything else has no id to answer
  queue = queue.then(async () => {
    try {
      port.post({ id: m.id, ok: true, value: await reads[m.op](m.path) });
    } catch (err) {
      port.post({ id: m.id, ok: false, error: err instanceof Error ? err.message : String(err) });
    }
  });
});
