// File pairing (ADR 0112 D1): a run leaves its artifacts side by side under
// one stem, and opening any one of them opens the set.
//
//   <stem>.h5            the model (neutral + /opensees)
//   <stem>.geometry.h5   the /geometry sibling (V0 Q1)
//   <stem>.results.h5    results, or <stem>.mpco
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

// Longest suffix first: `.geometry.h5` must win over `.h5`.
const SUFFIXES: readonly (readonly [string, Kind])[] = [
  [".geometry.h5", "geometry"],
  [".results.h5", "results"],
  [".mpco", "results"],
  [".h5", "model"],
];

/** The path's kind and stem; throws on a file the app does not open. */
export function classify(path: string): { kind: Kind; stem: string } {
  const low = path.toLowerCase();
  for (const [suffix, kind] of SUFFIXES) {
    if (low.endsWith(suffix) && low.length > suffix.length) {
      return { kind, stem: path.slice(0, path.length - suffix.length) };
    }
  }
  throw new Error(
    `apeGmshViewer opens <stem>.h5, <stem>.geometry.h5, <stem>.results.h5 or <stem>.mpco; got ${path}`,
  );
}

/** Every path the set of this stem can hold, present or not (what to watch). */
export function candidates(stem: string): string[] {
  return [`${stem}.h5`, `${stem}.geometry.h5`, `${stem}.results.h5`, `${stem}.mpco`];
}

/**
 * The set that opening `path` opens. The opened file must exist; each
 * sibling is included only if it exists. When the opened file is a results
 * file, it is the set's results even if the other results name also exists.
 */
export function pairSet(path: string, exists: (p: string) => boolean): OpenSet {
  if (!isAbsolute(path)) throw new Error(`pairSet needs an absolute path; got ${path}`);
  const { kind, stem } = classify(path);
  if (!exists(path)) throw new Error(`no such file: ${path}`);
  return stemSet(stem, kind === "results" ? path : null, exists);
}

/**
 * The set of `stem` as the disk holds it now. `preferredResults` (the results
 * file the user opened) is the results when it exists; otherwise
 * `<stem>.results.h5`, then `<stem>.mpco`.
 */
export function stemSet(stem: string, preferredResults: string | null, exists: (p: string) => boolean): OpenSet {
  const [model, geometry, resultsH5, mpco] = candidates(stem) as [string, string, string, string];
  const has = (p: string | null) => (p !== null && exists(p) ? p : null);
  return {
    model: has(model),
    geometry: has(geometry),
    results: has(preferredResults) ?? has(resultsH5) ?? has(mpco),
  };
}

export function sameSet(a: OpenSet | null, b: OpenSet | null): boolean {
  if (a === null || b === null) return a === b;
  return a.model === b.model && a.geometry === b.geometry && a.results === b.results;
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
