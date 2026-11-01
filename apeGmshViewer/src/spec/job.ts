// The render job main starts from (ADR 0128 D2, D4): the spec file read,
// validated (./spec.ts) and its files resolved (./files.ts), or the whole
// spec refused. A refused spec writes `render.json` naming the key and the
// reason into the out folder, and nothing else: no window opens, no PNG is
// written. The file system comes in as functions, so the tests run it on a
// temporary folder.

import { dirname, join } from "node:path";
import { resolveSpecFiles } from "./files.ts";
import { validateSpec, type ViewSpec } from "./spec.ts";

export interface RenderJobFiles {
  spec: ViewSpec;
  model: string;
  geometry: string | null;
  results: string | null;
}

export interface JobFs {
  readFile(path: string): string;
  exists(path: string): boolean;
  mkdir(path: string): void;
  writeFile(path: string, text: string): void;
}

export type Prepared = { ok: true; job: RenderJobFiles } | { ok: false; key: string; reason: string };

/** Read and check the spec at `specPath`; on a refusal, write `<outDir>/render.json` and nothing else. */
export function prepareRender(specPath: string, outDir: string, fs: JobFs): Prepared {
  const refuse = (key: string, reason: string): Prepared => {
    fs.mkdir(outDir);
    fs.writeFile(join(outDir, "render.json"), JSON.stringify({ render: { spec: specPath, refused: { key, reason } } }, null, 2));
    return { ok: false, key, reason };
  };
  let raw: unknown;
  try {
    raw = JSON.parse(fs.readFile(specPath));
  } catch (err) {
    return refuse("spec", `${specPath} is not a readable JSON document: ${err instanceof Error ? err.message : String(err)}`);
  }
  const v = validateSpec(raw);
  if (!v.ok) return refuse(v.key, v.reason);
  const f = resolveSpecFiles(v.spec, dirname(specPath), fs.exists);
  if (!f.ok) return refuse(f.key, f.reason);
  return { ok: true, job: { spec: v.spec, model: f.set.model, geometry: f.set.geometry, results: f.set.results } };
}
