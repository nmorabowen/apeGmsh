// The ADR 0112 zones, read from disk in the main process (like
// ../reader/node.ts for the model): the /geometry sibling and a model's
// /provenance. Each answer is plain data for the IPC; a reader error becomes
// `{ok: false, error}` with the reader's message, which names the zone and
// its version when it is a refusal (effects.parseRefusal reads it).

import { statSync } from "node:fs";
import * as h5wasm from "h5wasm/node";
import { readGeometry, type GeometryZone } from "../reader/geometry.ts";
import { filePath, isPseudoFile, readProvenance, type ProvenanceZone } from "../reader/provenance.ts";

const isFile = (p: string): boolean => {
  try {
    return statSync(p).isFile();
  } catch {
    return false;
  }
};
import type { H5Module } from "../reader/read.ts";

const h5 = h5wasm as unknown as H5Module;
const message = (err: unknown) => (err instanceof Error ? err.message : String(err));

export type GeometryAnswer =
  | { ok: true; geometry: GeometryZone | null; sizeBytes: number; readMs: number }
  | { ok: false; error: string };

export type ProvenanceAnswer = { ok: true; zone: ProvenanceZone | null } | { ok: false; error: string };

export async function openGeometry(path: string): Promise<GeometryAnswer> {
  try {
    await h5wasm.ready;
    const t = performance.now();
    const geometry = readGeometry(h5, path);
    return { ok: true, geometry, sizeBytes: statSync(path).size, readMs: performance.now() - t };
  } catch (err) {
    return { ok: false, error: message(err) };
  }
}

export async function openProvenance(path: string): Promise<ProvenanceAnswer> {
  try {
    await h5wasm.ready;
    const zone = readProvenance(h5, path);
    // Only main can see the disk: a source with no digest opens only when its
    // path is a file now (provenance.ts, SourceSite.recorded).
    if (zone) zone.present = zone.files.path.map((p, i) => !isPseudoFile(p) && isFile(filePath(zone, i)));
    return { ok: true, zone };
  } catch (err) {
    return { ok: false, error: message(err) };
  }
}
