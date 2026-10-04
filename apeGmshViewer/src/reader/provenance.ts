// Read the /provenance zone of any artifact (ADR 0112 D3) and find where in
// the user's code a declaration was made, for go-to-source.
//
// Layout and rules: architecture/h5-schema.md, "/provenance". Every artifact
// may carry the zone; absent, it is ignored (`null`). The checks follow
// geometry.ts: a broken table, an index out of range, an unknown `kind` or a
// duplicate declaration path raises, naming its HDF5 path.
//
// Browser-safe: no Node import.

import { readAttrs, SchemaError, type H5File, type H5Module } from "./read.ts";
import { dataset, group, has, int32s, sameLength, strs, zoneVersion } from "./geometry.ts";

/** The provenance zone version this reader was written against. */
export const PROVENANCE_TARGET = { major: 1, minor: 0 } as const;
// TODO(#1313): read the floor from read.ts's ZONE_FLOOR table once #1313 lands.
export const PROVENANCE_FLOOR = 0;

export interface ProvenanceZone {
  version: string;
  /** `@base_dir`: the directory relative `files/path` entries are under. */
  baseDir: string;
  files: { path: string[]; sha256: string[]; kind: ("script" | "module")[] };
  sites: { file: Int32Array; line: Int32Array; function: string[] };
  records: { path: string[]; site: Int32Array; script: Int32Array; seq: Int32Array };
  warnings: string[];
}

/** Read `/provenance` from an open file; `null` when the file has no such zone. */
export function readProvenanceZone(h5: H5Module, f: H5File): ProvenanceZone | null {
  const warnings: string[] = [];
  const meta = has(f, "meta") ? readAttrs(group(h5, f, "meta")) : {};
  const version = zoneVersion(meta, "provenance_schema_version", PROVENANCE_TARGET, PROVENANCE_FLOOR, warnings);
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
  const records = {
    path: strs(dataset(h5, rg, "path")),
    site: int32s(dataset(h5, rg, "site")),
    script: int32s(dataset(h5, rg, "script")),
    seq: int32s(dataset(h5, rg, "seq")),
  };
  const nRecords = sameLength(rg.path, records);
  const seen = new Set<string>();
  for (let i = 0; i < nRecords; i++) {
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
  | { ok: true; site: SourceSite | null; script: SourceSite | null; seq: number }
  | { ok: false; reason: string };

const isAbsolutePosix = (p: string) => p.startsWith("/") || /^[A-Za-z]:\//.test(p);

/**
 * Where the declaration at `declPath` (`<zone>/<family>/<name|#k>`) was
 * made: its `site` (the first frame outside apeGmsh) and its `script` line
 * (the outermost `__main__` frame); either is null when the record has -1.
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
  return { ok: true, site, script, seq: zone.records.seq[i]! };
}
