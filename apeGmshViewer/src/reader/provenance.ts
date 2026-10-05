// Read the /provenance zone of any artifact (ADR 0112 D3) and find where in
// the user's code a declaration was made, for go-to-source.
//
// Layout and rules: architecture/h5-schema.md, "/provenance". Every artifact
// may carry the zone; absent, it is ignored (`null`). The checks follow
// geometry.ts: a broken table, an index out of range, an unknown `kind` or
// `origin`, or a duplicate declaration path raises, naming its HDF5 path.
//
// 1.1.0 (#1378) added `records/origin`: `user` for a declaration the user
// made, `synthesised` for an object apeGmsh created inside a verb the user
// called (keys `<verb>:<owner>[/<role>]`). From 1.1.0 the column is required;
// a file below it reads with every record as `user`, as the Python reader does
// (`_femdata_h5_io._read_provenance`, `schema_version.PROVENANCE_ORIGIN_FROM`).
//
// Browser-safe: no Node import.

import { PROVENANCE_TARGET, readAttrs, SchemaError, ZONE_FLOOR, type H5File, type H5Module } from "./read.ts";
import { dataset, group, has, int32s, sameLength, strs, zoneVersion } from "./geometry.ts";

// The target and the floor live in read.ts's one version table (#1303).
export { PROVENANCE_TARGET } from "./read.ts";

/** The first version whose `records` carry `origin` (= Python's `PROVENANCE_ORIGIN_FROM`, 1.1.0). */
export const PROVENANCE_ORIGIN_FROM = { major: 1, minor: 1, patch: 0 } as const;

/** `user`: a declaration the user made; `synthesised`: an object apeGmsh made inside a verb the user called. */
export type Origin = "user" | "synthesised";
const ORIGINS: readonly Origin[] = ["user", "synthesised"];

export interface ProvenanceZone {
  version: string;
  /** `@base_dir`: the directory relative `files/path` entries are under. */
  baseDir: string;
  files: { path: string[]; sha256: string[]; kind: ("script" | "module")[] };
  sites: { file: Int32Array; line: Int32Array; function: string[] };
  records: { path: string[]; site: Int32Array; script: Int32Array; seq: Int32Array; origin: Origin[] };
  warnings: string[];
}

/** Read `/provenance` from an open file; `null` when the file has no such zone. */
export function readProvenanceZone(h5: H5Module, f: H5File): ProvenanceZone | null {
  const warnings: string[] = [];
  const meta = has(f, "meta") ? readAttrs(group(h5, f, "meta")) : {};
  const version = zoneVersion(meta, "provenance_schema_version", PROVENANCE_TARGET, ZONE_FLOOR.provenance, warnings);
  const present = has(f, "provenance");
  if (version === null && !present) return null;
  if (version === null) throw new SchemaError(`/provenance is present but /meta has no provenance_schema_version`);
  if (!present) throw new SchemaError(`/meta/provenance_schema_version is ${version} but /provenance is missing`);

  const g = group(h5, f, "provenance");
  const baseDir = readAttrs(g)["base_dir"];
  if (typeof baseDir !== "string") throw new SchemaError(`/provenance@base_dir is not a string`);

  const fg = group(h5, g, "files");
  const kinds = strs(dataset(h5, fg, "kind"));
  const files = {
    path: strs(dataset(h5, fg, "path")),
    sha256: strs(dataset(h5, fg, "sha256")),
    kind: kinds.map((k, i) => {
      if (k !== "script" && k !== "module") {
        throw new SchemaError(`${fg.path}/kind[${i}] = ${JSON.stringify(k)}; expected script or module`);
      }
      return k;
    }),
  };
  const nFiles = sameLength(fg.path, files);
  files.sha256.forEach((h, i) => {
    if (!/^[0-9a-f]{64}$/.test(h)) throw new SchemaError(`${fg.path}/sha256[${i}] is not a hex sha256`);
  });
  // The spec stores paths POSIX: a backslash is a writer fault, refused rather
  // than guessed at (on POSIX it is a legal file-name character).
  files.path.forEach((p, i) => {
    if (p.includes("\\")) throw new SchemaError(`${fg.path}/path[${i}] = ${JSON.stringify(p)} has a backslash; paths are POSIX`);
  });
  if (baseDir.includes("\\")) throw new SchemaError(`/provenance@base_dir = ${JSON.stringify(baseDir)} has a backslash; paths are POSIX`);

  const sg = group(h5, g, "sites");
  const sites = {
    file: int32s(dataset(h5, sg, "file")),
    line: int32s(dataset(h5, sg, "line")),
    function: strs(dataset(h5, sg, "function")),
  };
  const nSites = sameLength(sg.path, sites);
  for (let i = 0; i < nSites; i++) {
    if (sites.file[i]! < 0 || sites.file[i]! >= nFiles) {
      throw new SchemaError(`${sg.path}/file[${i}] = ${sites.file[i]} is not a row of files (0..${nFiles - 1})`);
    }
    if (sites.line[i]! < 1) throw new SchemaError(`${sg.path}/line[${i}] = ${sites.line[i]}; lines are 1-based`);
  }

  const rg = group(h5, g, "records");
  const path = strs(dataset(h5, rg, "path"));
  // From 1.1.0 the column is required; below it, an absent column reads as
  // every record the user's.
  const { major, minor, patch } = PROVENANCE_ORIGIN_FROM;
  if (atLeast(version, PROVENANCE_ORIGIN_FROM) && !has(rg, "origin")) {
    throw new SchemaError(`${rg.path}/origin is missing (required from provenance_schema_version ${major}.${minor}.${patch}; this file is ${version})`);
  }
  const rawOrigin = has(rg, "origin") ? strs(dataset(h5, rg, "origin")) : path.map(() => "user");
  const records = {
    path,
    site: int32s(dataset(h5, rg, "site")),
    script: int32s(dataset(h5, rg, "script")),
    seq: int32s(dataset(h5, rg, "seq")),
    origin: rawOrigin.map((o, i): Origin => {
      if (!(ORIGINS as readonly string[]).includes(o)) {
        throw new SchemaError(`${rg.path}/origin[${i}] = ${JSON.stringify(o)}; expected ${ORIGINS.join(" or ")}`);
      }
      return o as Origin;
    }),
  };
  const nRecords = sameLength(rg.path, records);
  const seen = new Set<string>();
  // `seq` is the capture order in the run: one per declaration, so unique.
  const seqs = new Set<number>();
  for (let i = 0; i < nRecords; i++) {
    if (seqs.has(records.seq[i]!)) throw new SchemaError(`${rg.path}/seq has ${records.seq[i]} twice; it must be unique`);
    seqs.add(records.seq[i]!);
    const p = records.path[i]!;
    if (seen.has(p)) throw new SchemaError(`${rg.path}/path has ${JSON.stringify(p)} twice; it must be unique`);
    seen.add(p);
    for (const col of ["site", "script"] as const) {
      const v = records[col][i]!;
      if (v < -1 || v >= nSites) {
        throw new SchemaError(`${rg.path}/${col}[${i}] = ${v} is neither -1 nor a row of sites (0..${nSites - 1})`);
      }
    }
    if (records.seq[i]! < 0) throw new SchemaError(`${rg.path}/seq[${i}] = ${records.seq[i]}; expected >= 0`);
  }
  return { version, baseDir, files, sites, records, warnings };
}

/** `version` ("X.Y.Z", already checked by zoneVersion) is at or above `v`. */
function atLeast(version: string, v: { major: number; minor: number; patch: number }): boolean {
  const [a, b, c] = version.split(".").map(Number) as [number, number, number];
  return a !== v.major ? a > v.major : b !== v.minor ? b > v.minor : c >= v.patch;
}

/** Open `path` and read its `/provenance` zone (see readProvenanceZone). */
export function readProvenance(h5: H5Module, path: string): ProvenanceZone | null {
  const f = new h5.File(path, "r");
  try {
    return readProvenanceZone(h5, f);
  } finally {
    f.close();
  }
}

/** One source location, ready for `goToSource(file, line)`. */
export interface SourceSite {
  /** Absolute path (POSIX separators; Windows accepts them). */
  file: string;
  line: number;
  function: string;
  /** The file's sha256 when it was captured: compare to detect an edit. */
  sha256: string;
  kind: "script" | "module";
}

export type SourceOf =
  | { ok: true; site: SourceSite | null; script: SourceSite | null; seq: number; origin: Origin }
  | { ok: false; reason: string };

const isAbsolutePosix = (p: string) => p.startsWith("/") || /^[A-Za-z]:\//.test(p);

/**
 * Where the declaration at `declPath` (`<zone>/<family>/<name|#k>`) was
 * made: its `site` (the first frame outside apeGmsh) and its `script` line
 * (the outermost `__main__` frame); either is null when the record has -1.
 * A synthesised object's site is the user's call of the verb that made it.
 */
export function sourceOf(zone: ProvenanceZone, declPath: string): SourceOf {
  const i = zone.records.path.indexOf(declPath);
  if (i < 0) return { ok: false, reason: `no provenance record for ${declPath}` };
  const at = (row: number): SourceSite | null => {
    if (row === -1) return null;
    const fi = zone.sites.file[row]!;
    const rel = zone.files.path[fi]!;
    return {
      file: isAbsolutePosix(rel) ? rel : `${zone.baseDir.replace(/\/+$/, "")}/${rel}`,
      line: zone.sites.line[row]!,
      function: zone.sites.function[row]!,
      sha256: zone.files.sha256[fi]!,
      kind: zone.files.kind[fi]!,
    };
  };
  const site = at(zone.records.site[i]!);
  const script = at(zone.records.script[i]!);
  if (site === null && script === null) return { ok: false, reason: `${declPath} has no source frame` };
  return { ok: true, site, script, seq: zone.records.seq[i]!, origin: zone.records.origin[i]! };
}
