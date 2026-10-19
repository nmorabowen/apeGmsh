// File pairing (ADR 0112 D1): a run leaves its artifacts side by side under
// one stem, and opening any one of them opens the set.
//
//   <stem>.h5            the model (neutral + /opensees)
//   <stem>.geometry.h5   the /geometry sibling (V0 Q1)
//   <stem>.results.h5    results (native)
//   <stem>.ladruno       results (fork-written; ADR 0122 D1, the canonical one)
//
// A set has one results slot (#2127; the T ruling of 2026-10-16): when both
// results files sit beside the model, the file the user opened wins if it is
// one of them; otherwise the `.ladruno` does. The other one is never read;
// `unopenedSibling` names it so the banner can say so.
//
// A partitioned run (ADR 0122 slice 8, #2188) leaves `<stem>.part-<N>.ladruno`
// beside the model. Opening any part opens the set under the stem; the results
// slot keeps the opened part's path, and main reads the whole part set from it
// (reader/ladrunoParts.ts, which refuses an incomplete set by name). Opened from
// the model, the slot is `<stem>.ladruno`, else `<stem>.part-0.ladruno`, else
// `<stem>.results.h5`.
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
  [".ladruno", "results"],
  [".h5", "model"],
];

/** Whether this platform's file systems compare names case-insensitively. */
export const FOLD_CASE = process.platform === "win32" || process.platform === "darwin";

/** The same file name, folding case where the file system does. */
export function samePath(a: string | null, b: string | null, fold: boolean = FOLD_CASE): boolean {
  if (a === null || b === null) return a === b;
  return fold ? a.toLowerCase() === b.toLowerCase() : a === b;
}

/** A partitioned run's part file (fork schema "File identity"); the same rule as reader/ladrunoParts.ts. */
const PART_FILE = /\.part-(\d+)\.ladruno$/i;

/** The path's kind and stem; throws on a file the app does not open. A part file is the results of its stem. */
export function classify(path: string): { kind: Kind; stem: string } {
  const low = path.toLowerCase();
  const part = PART_FILE.exec(path);
  if (part && part.index > 0 && !/[\\/]$/.test(path.slice(0, part.index))) {
    return { kind: "results", stem: path.slice(0, part.index) };
  }
  for (const [suffix, kind] of SUFFIXES) {
    if (low.endsWith(suffix) && low.length > suffix.length) {
      return { kind, stem: path.slice(0, path.length - suffix.length) };
    }
  }
  throw new Error(`apeGmshViewer opens <stem>.h5, <stem>.geometry.h5, <stem>.ladruno or <stem>.results.h5; got ${path}`);
}

/** The kind of the opened file, for each Candidates `candidates` made. */
const openedKind = new WeakMap<Candidates, Kind>();

/**
   * The paths of the set `opened` belongs to: the conventional name for each
   * kind, except the opened file's own kind, which keeps the opened spelling.
   */
export function candidates(opened: string): Candidates {
  const { kind, stem } = classify(opened);
  const c: Candidates = { model: `${stem}.h5`, geometry: `${stem}.geometry.h5`, results: `${stem}.results.h5` };
  c[kind] = opened;
  // The results slot's rule needs to know whether the results were opened;
  // kept beside the object, so `Object.values(c)` (watch.ts) stays the paths.
  openedKind.set(c, kind);
  return c;
}

/** The stem's results files other than `results`, in slot order: `.ladruno`, the part set (part 0), `.results.h5`. */
function otherResults(results: string): string[] {
  // A part file stands for its whole set: the set's other parts are not "other results".
  const own = PART_FILE.test(results) ? 1 : /\.ladruno$/i.test(results) ? 0 : /\.results\.h5$/i.test(results) ? 2 : -1;
  if (own < 0) throw new Error(`not a results file: ${results}`);
  const { stem } = classify(results);
  return [`${stem}.ladruno`, `${stem}.part-0.ladruno`, `${stem}.results.h5`].filter((_, i) => i !== own);
}

/**
 * The set as the disk holds it now: each candidate that exists, and one
 * results file. An opened results file wins while it exists; otherwise the
 * `.ladruno` (ADR 0122 D1), else part 0 of a part set, else the `.results.h5`.
 */
export function stemSet(c: Candidates, exists: (p: string) => boolean): OpenSet {
  const kind = openedKind.get(c);
  if (kind === undefined) throw new Error("stemSet needs the Candidates that candidates() made");
  const has = (p: string) => (exists(p) ? p : null);
  // Opened from the model or geometry, `c.results` is `<stem>.results.h5`, so the others are the `.ladruno` and part 0.
  const first = (ps: string[]) => ps.map(has).find((p) => p !== null) ?? null;
  const others = otherResults(c.results);
  const results = kind === "results" ? first([c.results, ...others]) : first([...others, c.results]);
  return { model: has(c.model), geometry: has(c.geometry), results };
}

/**
 * The stem's first other results file when it exists beside `results`: the set
 * reads one results file (or one part set), so this one is not opened (the banner names it).
 */
export function unopenedSibling(results: string, exists: (p: string) => boolean): string | null {
  // A file read as results under another name (a fixture, `--mode=measure-results`) has no stem pair.
  if (!/\.(ladruno|results\.h5)$/i.test(results)) return null;
  return otherResults(results).find((p) => exists(p)) ?? null;
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
