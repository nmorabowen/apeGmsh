// Read one model.h5 into a ModelFile, with h5wasm and nothing else.
//
// The layout comes from architecture/h5-schema.md. Rules this file keeps:
// - an optional child is probed by name (`keys().includes`), never by a
//   `get()` that returns null; a child that is listed but cannot be
//   opened is a broken file and raises;
// - a required child that is missing raises;
// - anything the reader does not recognise is a loud warning in
//   `ModelFile.warnings`, never a silent skip.

import type {
  ElementBlock,
  ElementMeta,
  Group as ModelGroup,
  ModelFile,
  NameAlias,
  OpenSeesZone,
  OpsFamily,
  OpsObject,
  Param,
  Table,
} from "../model/types.ts";

/** The parts of the h5wasm module this reader uses. */
export interface H5Module {
  File: new (filename: string, mode?: "r") => H5File;
  Group: abstract new (...args: never[]) => H5Group;
  Dataset: abstract new (...args: never[]) => H5Dataset;
}
export interface H5Attr {
  value: unknown;
  json_value: unknown;
}
export interface H5Node {
  path: string;
  attrs: Record<string, H5Attr>;
}
export interface H5Group extends H5Node {
  keys(): string[];
  get(name: string): unknown;
}
export interface H5Dataset extends H5Node {
  shape: number[] | null;
  dtype: unknown;
  value: unknown;
  json_value: unknown;
}
export interface H5File extends H5Group {
  close(): unknown;
}

/** The schema versions this reader was written against (ADR 0023). */
export const NEUTRAL_TARGET = { major: 2, minor: 35 } as const;
export const OPENSEES_TARGET = { major: 2, minor: 22 } as const;
/** The ADR 0112 zones (V2a specs): read by geometry.ts and provenance.ts. */
export const GEOMETRY_TARGET = { major: 1, minor: 0 } as const;
export const PROVENANCE_TARGET = { major: 1, minor: 0 } as const;
/**
 * The lowest minor of each zone this reader opens, under the zone's target
 * major (ADR 0113 D1/D7: one floor table, the only place the version check
 * reads; it equals the Python writer constants). Neutral 2.10 is the B2 layout
 * split (physical groups and labels side-partitioned); opensees 2.12 is the
 * first era whose files open (2.11 is the 0-based rank flip, but every 2.11
 * writer stamped a neutral zone below the neutral floor, so no 2.11 file
 * opens); the ADR 0112 zones start at their first version.
 * TODO(V4): results 1.0 joins this table with the app's results reader.
 */
export const ZONE_FLOOR = { neutral: 10, opensees: 12, geometry: 0, provenance: 0 } as const;

/**
 * ADR 0113 D7, the app's rule for one zone's `/meta/<key>` stamp:
 * - another major: refused;
 * - same major, below the floor: refused, naming the floor;
 * - same major, floor to target: opens, no banner;
 * - same major, newer than the target: opens with one banner (warning)
 *   naming the file's stamp and the app's target (an app-only deviation
 *   from INV-4: the app reads, and new data goes in new zones).
 * The refusal is `<key> <stamp>: <reason>`, the shape effects.parseRefusal reads.
 */
export function checkZoneVersion(
  raw: Param | Param[] | undefined,
  key: string,
  target: { major: number; minor: number },
  floor: number,
  warnings: string[],
): string {
  if (typeof raw !== "string") throw new SchemaError(`/meta has no string attribute ${key}`);
  const m = /^(\d+)\.(\d+)\.(\d+)$/.exec(raw);
  if (!m) throw new SchemaError(`/meta/${key} = ${JSON.stringify(raw)} is not X.Y.Z`);
  const major = Number(m[1]);
  const minor = Number(m[2]);
  if (major !== target.major) throw new SchemaError(`${key} ${raw}: this app reads major ${target.major} only`);
  if (minor < floor) {
    throw new SchemaError(`${key} ${raw}: layouts before ${target.major}.${floor} are not supported`);
  }
  if (minor > target.minor) {
    warnings.push(
      `${key} ${raw} is newer than this app (${target.major}.${target.minor}.x): ` +
        `the file opens, and what that apeGmsh added is not shown`,
    );
  }
  return raw;
}

export class SchemaError extends Error {}

export function readModel(
  h5: H5Module,
  path: string,
  sizeBytes: number,
): ModelFile {
  const t0 = performance.now();
  const f = new h5.File(path, "r");
  try {
    const r = new Reader(h5, f);
    const model = r.read(path, sizeBytes);
    model.readMs = performance.now() - t0;
    return model;
  } finally {
    f.close();
  }
}

class Reader {
  readonly warnings: string[] = [];
  private readonly h5: H5Module;
  private readonly f: H5File;

  constructor(h5: H5Module, f: H5File) {
    this.h5 = h5;
    this.f = f;
  }

  read(path: string, sizeBytes: number): ModelFile {
    const root = this.f;
    const meta = this.group(root, "meta");
    const metaAttrs = readAttrs(meta);
    const neutralVersion = this.checkVersion(
      metaAttrs,
      "neutral_schema_version",
      NEUTRAL_TARGET,
      ZONE_FLOOR.neutral,
    );

    const nodes = this.group(root, "nodes");
    const nodeIds = numbers(this.dataset(nodes, "ids"));
    const nodeCoords = numbers(this.dataset(nodes, "coords"));
    if (nodeCoords.length !== nodeIds.length * 3) {
      throw new SchemaError(
        `/nodes/coords has ${nodeCoords.length} values for ${nodeIds.length} ids; expected (N, 3)`,
      );
    }

    const blocks: ElementBlock[] = [];
    if (has(root, "elements")) {
      const elements = this.group(root, "elements");
      for (const alias of elements.keys()) {
        blocks.push(this.readBlock(this.group(elements, alias), alias));
      }
    }

    const physicalGroups = has(root, "physical_groups")
      ? this.readSidePartitioned(this.group(root, "physical_groups"))
      : [];
    const labels = has(root, "labels")
      ? this.readSidePartitioned(this.group(root, "labels"))
      : [];

    let opensees: OpenSeesZone | null = null;
    if (has(root, "opensees")) {
      const v = this.checkVersion(
        metaAttrs,
        "opensees_schema_version",
        OPENSEES_TARGET,
        ZONE_FLOOR.opensees,
      );
      opensees = this.readOpenSees(this.group(root, "opensees"), v);
    }

    return {
      path,
      sizeBytes,
      neutralVersion,
      meta: metaAttrs,
      nodeIds,
      nodeCoords,
      blocks,
      physicalGroups,
      labels,
      opensees,
      warnings: this.warnings,
      readMs: 0,
    };
  }

  private checkVersion(
    meta: Record<string, Param | Param[]>,
    key: string,
    target: { major: number; minor: number },
    floor: number,
  ): string {
    return checkZoneVersion(meta[key], key, target, floor, this.warnings);
  }

  private readBlock(g: H5Group, alias: string): ElementBlock {
    const a = readAttrs(g);
    const npe = num(a["npe"], `${g.path}@npe`);
    const ids = numbers(this.dataset(g, "ids"));
    const connectivity = numbers(this.dataset(g, "connectivity"));
    if (connectivity.length !== ids.length * npe) {
      throw new SchemaError(
        `${g.path}/connectivity has ${connectivity.length} values for ${ids.length} x ${npe}`,
      );
    }
    return {
      alias,
      code: num(a["code"], `${g.path}@code`),
      dim: num(a["dim"], `${g.path}@dim`),
      npe,
      ids,
      connectivity,
    };
  }

  private readSidePartitioned(g: H5Group): ModelGroup[] {
    const out: ModelGroup[] = [];
    if (!has(g, "element_side")) return out;
    const side = this.group(g, "element_side");
    for (const name of side.keys()) {
      const e = this.group(side, name);
      const a = readAttrs(e);
      out.push({
        name,
        dim: num(a["dim"], `${e.path}@dim`),
        tag: num(a["tag"], `${e.path}@tag`),
        path: e.path,
        // element_ids is optional by schema: a dim-0 group lists nodes only.
        elementIds: has(e, "element_ids")
          ? numbers(this.dataset(e, "element_ids"))
          : new Float64Array(0),
      });
    }
    return out;
  }

  private readOpenSees(g: H5Group, version: string): OpenSeesZone {
    const objects: OpsObject[] = [];
    const families: [string, OpsFamily][] = [
      ["sections", "section"],
      ["transforms", "geomTransf"],
      ["beam_integration", "beamIntegration"],
    ];
    if (has(g, "materials")) {
      const mats = this.group(g, "materials");
      for (const sub of mats.keys()) {
        const fam: OpsFamily | null =
          sub === "uniaxial" ? "uniaxialMaterial" : sub === "nd" ? "nDMaterial" : null;
        if (fam === null) {
          this.warnings.push(
            `unknown material family ${mats.path}/${sub}; its members are not shown`,
          );
          continue;
        }
        const fg = this.group(mats, sub);
        for (const name of fg.keys()) objects.push(this.readObject(this.group(fg, name), fam));
      }
    }
    for (const [key, fam] of families) {
      if (!has(g, key)) continue;
      const fg = this.group(g, key);
      for (const name of fg.keys()) objects.push(this.readObject(this.group(fg, name), fam));
    }

    const elementMeta: ElementMeta[] = [];
    if (has(g, "element_meta")) {
      const em = this.group(g, "element_meta");
      for (const type of em.keys()) elementMeta.push(this.readElementMeta(this.group(em, type), type));
    }

    const names: NameAlias[] = [];
    if (has(g, "names")) {
      const ng = this.group(g, "names");
      const nm = strings(this.dataset(ng, "name"));
      const kd = strings(this.dataset(ng, "kind"));
      const tg = numbers(this.dataset(ng, "tag"));
      if (nm.length !== kd.length || nm.length !== tg.length) {
        throw new SchemaError(`${ng.path}: name/kind/tag lengths differ`);
      }
      for (let i = 0; i < nm.length; i++) {
        names.push({ name: nm[i]!, kind: kd[i]!, tag: tg[i]! });
      }
    }
    return { version, objects, elementMeta, names };
  }

  private readObject(g: H5Group, family: OpsFamily): OpsObject {
    const a = readAttrs(g);
    const type = a["type"];
    const tag = a["tag"];
    if (typeof type !== "string") throw new SchemaError(`${g.path} has no string 'type' attribute`);
    if (typeof tag !== "number") throw new SchemaError(`${g.path} has no integer 'tag' attribute`);
    const params = mergeParams(a["params"], a["params_str"]);
    const attrs: Record<string, Param | Param[]> = {};
    for (const [k, v] of Object.entries(a)) {
      if (k !== "type" && k !== "tag" && k !== "params" && k !== "params_str") attrs[k] = v;
    }
    const tables: Record<string, Table> = {};
    for (const key of g.keys()) {
      const node = g.get(key);
      if (!(node instanceof this.h5.Dataset)) {
        this.warnings.push(`${g.path}/${key} is not a dataset; not shown`);
        continue;
      }
      tables[key] = readTable(node as H5Dataset);
    }
    return {
      family,
      path: g.path,
      groupName: g.path.slice(g.path.lastIndexOf("/") + 1),
      type,
      tag,
      params,
      attrs,
      tables,
    };
  }

  private readElementMeta(g: H5Group, type: string): ElementMeta {
    const ids = numbers(this.dataset(g, "ids"));
    const femEids = numbers(this.dataset(g, "fem_eids"));
    if (femEids.length !== ids.length) {
      throw new SchemaError(`${g.path}: fem_eids and ids lengths differ`);
    }
    const argsDs = this.dataset(g, "args");
    const shape = argsDs.shape ?? [];
    const nCols = shape.length === 2 ? shape[1]! : 0;
    const flat = numbers(argsDs);
    const flatStr = has(g, "args_str") ? strings(this.dataset(g, "args_str")) : null;
    const args: Param[][] = [];
    for (let i = 0; i < ids.length; i++) {
      const row: Param[] = [];
      for (let j = 0; j < nCols; j++) {
        const k = i * nCols + j;
        const s = flatStr ? flatStr[k] : "";
        row.push(s ? s : flat[k]!);
      }
      // Trailing NaN with no string are padding to the type's max tail.
      while (row.length && typeof row[row.length - 1] === "number" && Number.isNaN(row[row.length - 1])) {
        row.pop();
      }
      args.push(row);
    }
    let inlineConnectivity: number[][] | null = null;
    if (has(g, "inline_connectivity")) {
      const v = this.dataset(g, "inline_connectivity").value;
      if (!Array.isArray(v)) throw new SchemaError(`${g.path}/inline_connectivity is not a ragged array`);
      inlineConnectivity = v.map((row) => Array.from(row as ArrayLike<number | bigint>, Number));
    }
    return { type, path: g.path, ids, femEids, args, inlineConnectivity };
  }

  private group(parent: H5Group, name: string): H5Group {
    if (!has(parent, name)) throw new SchemaError(`${joinPath(parent.path, name)} is missing`);
    const node = parent.get(name);
    if (!(node instanceof this.h5.Group)) {
      throw new SchemaError(`${joinPath(parent.path, name)} is listed but is not a readable group`);
    }
    return node as H5Group;
  }

  private dataset(parent: H5Group, name: string): H5Dataset {
    if (!has(parent, name)) throw new SchemaError(`${joinPath(parent.path, name)} is missing`);
    const node = parent.get(name);
    if (!(node instanceof this.h5.Dataset)) {
      throw new SchemaError(`${joinPath(parent.path, name)} is listed but is not a readable dataset`);
    }
    return node as H5Dataset;
  }
}

function joinPath(parent: string, name: string): string {
  return parent === "/" ? `/${name}` : `${parent}/${name}`;
}

function has(g: H5Group, name: string): boolean {
  return g.keys().includes(name);
}

function num(v: Param | Param[] | undefined, where: string): number {
  if (typeof v !== "number") throw new SchemaError(`${where} is not a number`);
  return v;
}

/** One attribute value as a Param or Param[] (int64 becomes number). */
function attrValue(raw: unknown): Param | Param[] {
  if (typeof raw === "bigint") return Number(raw);
  if (typeof raw === "number" || typeof raw === "string") return raw;
  if (typeof raw === "boolean") return raw ? 1 : 0;
  if (ArrayBuffer.isView(raw)) return Array.from(raw as unknown as ArrayLike<number | bigint>, Number);
  if (Array.isArray(raw)) return raw.map((x) => (typeof x === "bigint" ? Number(x) : (x as Param)));
  throw new SchemaError(`unsupported attribute value ${String(raw)}`);
}

export function readAttrs(n: H5Node): Record<string, Param | Param[]> {
  const out: Record<string, Param | Param[]> = {};
  for (const [k, a] of Object.entries(n.attrs)) out[k] = attrValue(a.value);
  return out;
}

function numbers(ds: H5Dataset): Float64Array {
  const v = ds.value;
  if (v instanceof Float64Array) return v;
  if (ArrayBuffer.isView(v)) return Float64Array.from(v as unknown as ArrayLike<number | bigint>, Number);
  throw new SchemaError(`${ds.path} is not a numeric dataset`);
}

function strings(ds: H5Dataset): string[] {
  const v = ds.json_value;
  if (Array.isArray(v) && v.every((x) => typeof x === "string")) return v as string[];
  throw new SchemaError(`${ds.path} is not a string dataset`);
}

function mergeParams(p: Param | Param[] | undefined, s: Param | Param[] | undefined): Param[] {
  const nums = p === undefined ? [] : Array.isArray(p) ? p : [p];
  const strs = s === undefined ? [] : Array.isArray(s) ? s : [s];
  const n = Math.max(nums.length, strs.length);
  const out: Param[] = [];
  for (let i = 0; i < n; i++) {
    const t = strs[i];
    out.push(typeof t === "string" && t !== "" ? t : (nums[i] ?? Number.NaN));
  }
  return out;
}

/** A dataset as a table. Compound rows keep their field order. */
function readTable(ds: H5Dataset): Table {
  const dt = ds.dtype;
  const shape = ds.shape ?? [];
  if (Array.isArray(dt)) {
    const columns = (dt as unknown[]).map((m) => String((m as unknown[])[0]));
    const rows = (ds.json_value as unknown[][]).map((r) => r.map((x) => (x === null ? Number.NaN : x)));
    return { path: ds.path, columns, rows };
  }
  const v = ds.json_value;
  const flat = Array.isArray(v) ? (v as unknown[]) : [v];
  const width = shape.length === 2 ? shape[1]! : 1;
  const rows: unknown[][] = [];
  if (width === 0) return { path: ds.path, columns: [], rows: [] };
  for (let i = 0; i < flat.length; i += width) rows.push(flat.slice(i, i + width));
  return { path: ds.path, columns: width === 1 ? ["value"] : Array.from({ length: width }, (_, j) => `[${j}]`), rows };
}
