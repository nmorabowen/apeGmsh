// What an agent mode writes into its `--out` (VH3, #2868: moved out of
// ../main.ts unchanged), shared by ./render.ts and ./reach.ts: the file
// system a job is checked against, the manifest written last and whole, and
// main's stamp on a manifest the renderer composed.

import { app } from "electron";
import { existsSync, mkdirSync, readdirSync, readFileSync, renameSync, statSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import type { JobFs } from "../../spec/job.ts";
import { MANIFEST_FILE } from "../../spec/manifest.ts";

export const jobFs: JobFs = {
  readFile: (path) => readFileSync(path, "utf8"),
  exists: existsSync,
  isDir: (path) => statSync(path).isDirectory(),
  list: (path) => readdirSync(path),
  mkdir: (path) => mkdirSync(path, { recursive: true }),
  writeFile: (path, text) => writeFileSync(path, text),
};

/** A run's manifest as the renderer composed it; main adds the stamp (`stamped`). */
export interface RunManifest {
  timing: object;
}

/** ADR 0128 D3: the app's version is main's to know; `written_at` lies inside `timing`, the one part that varies between runs (src/spec/manifest.ts). */
export function stamped(m: RunManifest) {
  const { timing, ...rest } = m;
  return { ...rest, app: { version: app.getVersion() }, timing: { ...timing, written_at: new Date().toISOString() } };
}

/** The manifest last, and whole: its presence is what the waiting `--snapshot` process reads as done. */
export function writeManifestLast(dir: string, text: string): void {
  const path = join(dir, MANIFEST_FILE);
  writeFileSync(`${path}.tmp`, text);
  renameSync(`${path}.tmp`, path);
}
