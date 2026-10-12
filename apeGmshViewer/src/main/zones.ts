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
import type { SequenceZone } from "../model/types.ts";

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

/** `/meta/lineage` of a model.h5 (ADR 0021), the pairing key for its results (V4b). */
export type LineageAnswer = { ok: true; lineage: { femHash: string | null; modelHash: string | null; resultsHash: string | null } | null } | { ok: false; error: string };

/** The element tags a `/sequence` element row's source holds, by source path (V4b, state/activation.ts). */
export type StageElementsAnswer = { ok: true; sets: Record<string, number[]> } | { ok: false; error: string };

interface H5Handle {
  file_id: bigint | number;
  attrs: Record<string, { value: unknown }>;
  keys(): string[];
  get(name: string): unknown;
  close(): unknown;
}

function openFile(path: string): H5Handle {
  const f = new h5wasm.File(path, "r") as unknown as H5Handle;
  if (Number(f.file_id) < 0) throw new Error(`HDF5 could not open ${path}`);
  return f;
}

export async function openLineage(path: string): Promise<LineageAnswer> {
  try {
    await h5wasm.ready;
    const f = openFile(path);
    try {
      if (!f.keys().includes("meta")) return { ok: true, lineage: null };
      const meta = f.get("meta") as H5Handle;
      if (!(meta instanceof h5wasm.Group) || !meta.keys().includes("lineage")) return { ok: true, lineage: null };
      const g = f.get("meta/lineage");
      if (!(g instanceof h5wasm.Group)) throw new Error("/meta/lineage is not a group");
      const hash = (key: string): string | null => {
        const v = (g as unknown as H5Handle).attrs[key]?.value;
        if (v === undefined) return null;
        if (typeof v !== "string" || v === "") throw new Error(`/meta/lineage@${key} is not a non-empty string`);
        return v;
      };
      return { ok: true, lineage: { femHash: hash("fem_hash"), modelHash: hash("model_hash"), resultsHash: hash("results_hash") } };
    } finally {
      f.close();
    }
  } catch (err) {
    return { ok: false, error: message(err) };
  }
}

/** A `/sequence/changes` element row's source: a stage's `owned_element_ids` or `remove_element`, nothing else. */
const ELEMENT_SOURCE = /^\/opensees\/stages\/stage_\d+\/(owned_element_ids|remove_element)$/;

/**
 * Read the tags every `/sequence` element change row names. Only the two
 * stage datasets a row may name are read; any other source refuses the read
 * (activation is then not applied, and the banner says why).
 */
export async function openStageElements(path: string, sequence: SequenceZone): Promise<StageElementsAnswer> {
  if (sequence.status !== "ready") return { ok: true, sets: {} };
  const t = sequence.changes;
  const wc = t.columns.indexOf("what"), sc = t.columns.indexOf("source");
  if (wc < 0 || sc < 0) return { ok: true, sets: {} }; // activation.ts names the missing column
  const sources = [...new Set(t.rows.filter((r) => r[wc] === "element").map((r) => r[sc]))];
  if (sources.length === 0) return { ok: true, sets: {} };
  try {
    await h5wasm.ready;
    const f = openFile(path);
    try {
      const sets: Record<string, number[]> = {};
      for (const src of sources) {
        if (typeof src !== "string" || !ELEMENT_SOURCE.test(src)) throw new Error(`/sequence/changes names ${JSON.stringify(src)} for an element row, not a stage's owned_element_ids or remove_element`);
        const ds = f.get(src.slice(1));
        if (!(ds instanceof h5wasm.Dataset)) throw new Error(`${src} is missing or not a dataset`);
        if (ds.shape?.length !== 1) throw new Error(`${src} is not 1-D`);
        const v = ds.value;
        if (!(v instanceof BigInt64Array || v instanceof Int32Array)) throw new Error(`${src} is not an integer dataset`);
        sets[src] = Array.from(v as ArrayLike<number | bigint>, Number);
      }
      return { ok: true, sets };
    } finally {
      f.close();
    }
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
