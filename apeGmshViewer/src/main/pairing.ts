// File pairing (ADR 0112 D1): a run leaves its artifacts side by side under
// one stem, and opening any one of them opens the set.
//
//   <stem>.h5            the model (neutral + /opensees)
//   <stem>.geometry.h5   the /geometry sibling (V0 Q1)
//   <stem>.results.h5    results
//
// The suffix is matched case-insensitively. The opened file keeps its own
// spelling in the set; the siblings are probed under the conventional
// (lower-case) suffixes the writers use. Paths compare case-insensitively
// where the file system does (Windows, macOS): `samePath`.
//
// Pure apart from the `exists` probe, which the caller passes in, so the
// rules are tested without a disk. No Electron import: the tests run in Node.

import { isAbsolute, resolve } from "node:path";

export type Kind = "model" | "geometry" | "results";

/** The artifacts of one stem. `null` means absent on disk. */
export interface OpenSet {
  model: string | null;
  geometry: string | null;
  results: string | null;
}

/** The one path each kind can have for a stem, present or not. */
export type Candidates = Record<Kind, string>;

// Longest suffix first: `.geometry.h5` must win over `.h5`.
const SUFFIXES: readonly (readonly [string, Kind])[] = [
  [".geometry.h5", "geometry"],
  [".results.h5", "results"],
  [".h5", "model"],
];

/** Whether this platform's file systems compare names case-insensitively. */
export const FOLD_CASE = process.platform === "win32" || process.platform === "darwin";

/** The same file name, folding case where the file system does. */
export function samePath(a: string | null, b: string | null, fold: boolean = FOLD_CASE): boolean {
  if (a === null || b === null) return a === b;
  return fold ? a.toLowerCase() === b.toLowerCase() : a === b;
}

/** The path's kind and stem; throws on a file the app does not open. */
export function classify(path: string): { kind: Kind; stem: string } {
  const low = path.toLowerCase();
  for (const [suffix, kind] of SUFFIXES) {
    if (low.endsWith(suffix) && low.length > suffix.length) {
      return { kind, stem: path.slice(0, path.length - suffix.length) };
    }
  }
  throw new Error(`apeGmshViewer opens <stem>.h5, <stem>.geometry.h5 or <stem>.results.h5; got ${path}`);
}

/**
 * The paths of the set `opened` belongs to: the conventional name for each
 * kind, except the opened file's own kind, which keeps the opened spelling.
 */
export function candidates(opened: string): Candidates {
  const { kind, stem } = classify(opened);
  const c: Candidates = { model: `${stem}.h5`, geometry: `${stem}.geometry.h5`, results: `${stem}.results.h5` };
  c[kind] = opened;
  return c;
}

/** The set as the disk holds it now: each candidate that exists. */
export function stemSet(c: Candidates, exists: (p: string) => boolean): OpenSet {
  const has = (p: string) => (exists(p) ? p : null);
  return { model: has(c.model), geometry: has(c.geometry), results: has(c.results) };
}

/** The set that opening `path` opens. The opened file must exist. */
export function pairSet(path: string, exists: (p: string) => boolean): OpenSet {
  if (!isAbsolute(path)) throw new Error(`pairSet needs an absolute path; got ${path}`);
  const c = candidates(path);
  if (!exists(path)) throw new Error(`no such file: ${path}`);
  return stemSet(c, exists);
}

export function sameSet(a: OpenSet | null, b: OpenSet | null, fold: boolean = FOLD_CASE): boolean {
  if (a === null || b === null) return a === b;
  return samePath(a.model, b.model, fold) && samePath(a.geometry, b.geometry, fold) && samePath(a.results, b.results, fold);
}

/** Whether `path` is one of the set's files. */
export function inSet(set: OpenSet, path: string, fold: boolean = FOLD_CASE): boolean {
  return [set.model, set.geometry, set.results].some((p) => samePath(p, path, fold));
}

/**
 * The file a launch asks for. `--file=<path>` (scripts/launch.mjs) wins;
 * otherwise the first positional argument after `skip` leading entries (the
 * executable, and the app path when Electron runs unpackaged). Chromium
 * switches that Electron appends to a second instance's argv start with `--`
 * and are ignored. A relative path resolves against `cwd`.
 */
export function fileFromArgv(argv: readonly string[], skip: number, cwd: string): string | null {
  const flag = argv.find((a) => a.startsWith("--file="));
  if (flag) return resolve(cwd, flag.slice("--file=".length));
  const positional = argv.slice(skip).filter((a) => !a.startsWith("--"));
  if (positional.length > 1) {
    process.stderr.write(`apeGmshViewer: opening ${positional[0]}; ignoring ${positional.slice(1).join(", ")}\n`);
  }
  return positional[0] ? resolve(cwd, positional[0]) : null;
}
