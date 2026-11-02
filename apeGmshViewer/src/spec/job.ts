// The render job main starts from (ADR 0128 D2, D4): the spec file read,
// validated (./spec.ts) and its files resolved (./files.ts), or the whole
// spec refused. A refused spec writes `manifest.json` naming the key and the
// reason into the out folder, and nothing else: no window opens, no PNG is
// written. The file system comes in as functions, so the tests run it on a
// temporary folder.
//
// One run, one folder: `--out` must not exist or be empty (`outRefusal`). A
// still from an earlier run could otherwise stand beside the record of a
// shot this run refused, or of a spec it refused (review of #2660). The app
// itself then writes everything it writes into that folder: the stills,
// `manifest.json` or `describe.json`, and Chromium's profile under `.chromium`
// (main.ts, INV-1), so a second run into the same folder is refused too.

import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { resolveSpecFiles } from "./files.ts";
import { MANIFEST_FILE, refusedManifest } from "./manifest.ts";
import { validateSpec, type ViewSpec } from "./spec.ts";

/** Chromium's profile and crash dumps of an agent run, inside its `--out` (main.ts). */
export const PROFILE_DIR = ".chromium";

export interface RenderJobFiles {
  spec: ViewSpec;
  model: string;
  geometry: string | null;
  results: string | null;
}

export interface JobFs {
  readFile(path: string): string;
  exists(path: string): boolean;
  /** whether an existing path is a folder */
  isDir(path: string): boolean;
  /** the names in a folder */
  list(path: string): string[];
  mkdir(path: string): void;
  writeFile(path: string, text: string): void;
}

export type Prepared = { ok: true; job: RenderJobFiles } | { ok: false; key: string; reason: string; wrote: boolean };

/** Why `outDir` cannot take a run (it is a file, or a folder that holds files), or null when it is absent or empty. Writes nothing. */
export function outRefusal(outDir: string, fs: Pick<JobFs, "exists" | "isDir" | "list">): string | null {
  if (!fs.exists(outDir)) return null;
  if (!fs.isDir(outDir)) return `${outDir} is a file, not a folder; an agent run writes into an absent or empty folder`;
  const names = fs.list(outDir);
  if (names.length === 0) return null;
  const shown = names.slice(0, 5).join(", ") + (names.length > 5 ? `, and ${names.length - 5} more` : "");
  return `${outDir} is not empty (${shown}); an agent run writes into an absent or empty folder, so no file of an earlier run stands beside its record`;
}

/**
 * The app's version for a spec refused whole. Main's handler states it for a
 * run (`app.getVersion()`), but a refusal is written at startup, before that
 * path: so it is read from the same file, the app's package.json, found
 * walking up from this module (`src/spec/` in a test, `dist/` in a build).
 * Not found: null, and the manifest says so.
 */
export function appVersion(fs: Pick<JobFs, "readFile" | "exists">, from = dirname(fileURLToPath(import.meta.url))): { version: string | null; version_reason: string | null } {
  for (let dir = from; ; dir = dirname(dir)) {
    const path = join(dir, "package.json");
    if (fs.exists(path)) {
      const pkg = JSON.parse(fs.readFile(path)) as { name?: unknown; version?: unknown };
      if (pkg.name === "apegmsh-viewer" && typeof pkg.version === "string") return { version: pkg.version, version_reason: null };
    }
    if (dirname(dir) === dir) return { version: null, version_reason: `no package.json of apegmsh-viewer at or above ${from}, so this refusal cannot state the app's version` };
  }
}

/**
 * Read and check the spec at `specPath`. A non-empty `outDir` is refused
 * before anything is read, and nothing is written; any other refusal writes
 * `<outDir>/manifest.json` and nothing else.
 */
export function prepareRender(specPath: string, outDir: string, fs: JobFs): Prepared {
  const busy = outRefusal(outDir, fs);
  if (busy) return { ok: false, key: "--out", reason: busy, wrote: false };
  const refuse = (key: string, reason: string): Prepared => {
    fs.mkdir(outDir);
    fs.writeFile(join(outDir, MANIFEST_FILE), refusedManifest(specPath, key, reason, appVersion(fs), new Date().toISOString()));
    return { ok: false, key, reason, wrote: true };
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
