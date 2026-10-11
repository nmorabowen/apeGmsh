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
  AnalysisDef,
  BcsDef,
  DeclParams,
  DeclRecord,
  DeclTable,
  DeclTagRun,
  DeclValue,
  Definitions,
  DefObject,
  SequenceZone,
  StageDef,
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
/** opensees 2.26.0: `/opensees/decls` (2.25) and `/opensees/decl_params` (2.26, K1-7). */
export const OPENSEES_TARGET = { major: 2, minor: 26 } as const;
/** The ADR 0112 zones (V2a specs): read by geometry.ts and provenance.ts. */
export const GEOMETRY_TARGET = { major: 1, minor: 0 } as const;
export const PROVENANCE_TARGET = { major: 1, minor: 1 } as const;
/**
 * The root `/sequence` zone (ADR 0112 D2(b), V0 decision 8: "1.0.0"), read
 * here when a file carries it. Its floor is `SEQUENCE_FLOOR`, not a
 * `ZONE_FLOOR` row, until V3a ships Python's writer and its floor constant:
 * tests/test_app_floor_drift.py compares every `ZONE_FLOOR` row with
 * Python's `reader_floor(zone)`, which has no sequence zone yet.
 * TODO(V3a): move `sequence: 0` into ZONE_FLOOR when Python's floor lands.
 */
export const SEQUENCE_TARGET = { major: 1, minor: 0 } as const;
export const SEQUENCE_FLOOR = 0;
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

/** The definitions of an `/opensees` zone that holds none (a synthetic model in the tests). */
export const emptyDefinitions = (): Definitions => ({
  decls: null,
  analysis: null,
  timeSeries: [],
  patterns: [],
  recorders: [],
  bcs: { fix: null, mass: null },
  stages: [],
});

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
    const sequence = this.readSequence(root, metaAttrs);

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
      sequence,
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
    return { version, objects, elementMeta, names, definitions: this.readDefinitions(g) };
  }

  /**
   * The model-definition data of `/opensees` (h5-schema.md): the declarations
   * (2.25) and their parameters by field name (2.26), the analysis chain, the
   * time series, patterns and recorders, the fix and mass records, and the
   * stages. Every group here is optional by schema; a listed one that is
   * broken raises, naming its path.
   */
  private readDefinitions(g: H5Group): Definitions {
    const members = (key: string, tagged: boolean): DefObject[] => {
      if (!has(g, key)) return [];
      const fg = this.group(g, key);
      return fg.keys().map((name) => this.readDef(this.group(fg, name), tagged));
    };
    if (has(g, "decl_params") && !has(g, "decls")) {
      throw new SchemaError(`${joinPath(g.path, "decl_params")} is present but ${joinPath(g.path, "decls")}, the rows it indexes, is missing`);
    }
    return {
      decls: has(g, "decls") ? this.readDecls(g) : null,
      analysis: has(g, "analysis") ? this.readAnalysis(this.group(g, "analysis")) : null,
      timeSeries: members("time_series", true),
      patterns: members("patterns", true),
      recorders: members("recorders", false),
      bcs: has(g, "bcs") ? this.readBcs(this.group(g, "bcs")) : { fix: null, mass: null },
      stages: has(g, "stages") ? this.readStages(this.group(g, "stages")) : [],
    };
  }

  /** A series, pattern or recorder group: `type` (and `tag`, unless a recorder), params, attrs, datasets. */
  private readDef(g: H5Group, tagged: boolean): DefObject {
    const a = readAttrs(g);
    const type = a["type"];
    const tag = a["tag"];
    if (typeof type !== "string") throw new SchemaError(`${g.path} has no string 'type' attribute`);
    if (tagged && typeof tag !== "number") throw new SchemaError(`${g.path} has no integer 'tag' attribute`);
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
      path: g.path,
      groupName: g.path.slice(g.path.lastIndexOf("/") + 1),
      type,
      tag: typeof tag === "number" ? tag : null,
      params: mergeParams(a["params"], a["params_str"]),
      attrs,
      tables,
    };
  }

  private readAnalysis(g: H5Group): AnalysisDef {
    for (const key of g.keys()) this.warnings.push(`${g.path}/${key}: the analysis group holds attributes only; not shown`);
    return { path: g.path, attrs: readAttrs(g) };
  }

  private readBcs(g: H5Group): BcsDef {
    const out: BcsDef = { fix: null, mass: null };
    for (const key of g.keys()) {
      if (key === "fix" || key === "mass") out[key] = readTable(this.dataset(g, key));
      else this.warnings.push(`unknown bcs record ${g.path}/${key}; not shown`);
    }
    return out;
  }

  /** `/opensees/stages/stage_NNN`, in ordinal order; `@n_stages`, when stamped, must count them. */
  private readStages(g: H5Group): StageDef[] {
    const stages: StageDef[] = [];
    for (const name of g.keys()) {
      const m = /^stage_(\d{3,})$/.exec(name);
      if (!m) {
        this.warnings.push(`unknown stage group ${g.path}/${name}; not shown`);
        continue;
      }
      stages.push(this.readStage(this.group(g, name), Number(m[1])));
    }
    stages.sort((a, b) => a.index - b.index);
    const n = readAttrs(g)["n_stages"];
    if (n !== undefined && n !== stages.length) {
      throw new SchemaError(`${g.path}@n_stages is ${JSON.stringify(n)} but the group holds ${stages.length} stage_NNN groups`);
    }
    stages.forEach((st, i) => {
      if (st.index !== i) throw new SchemaError(`${g.path}: stage ordinals are not 0..${stages.length - 1} (found ${st.path})`);
    });
    return stages;
  }

  private readStage(g: H5Group, index: number): StageDef {
    const attrs = readAttrs(g);
    const name = attrs["name"];
    if (typeof name !== "string") throw new SchemaError(`${g.path} has no string 'name' attribute`);
    const st: StageDef = { index, path: g.path, name, attrs, analysis: null, patterns: [], recorders: [], bcs: { fix: null, mass: null }, other: [] };
    for (const key of g.keys()) {
      if (key === "analysis") st.analysis = this.readAnalysis(this.group(g, key));
      else if (key === "patterns" || key === "recorders") {
        const fg = this.group(g, key);
        st[key] = fg.keys().map((n) => this.readDef(this.group(fg, n), key === "patterns"));
      } else if (key === "bcs") st.bcs = this.readBcs(this.group(g, key));
      else if (STAGE_OTHER.has(key)) st.other.push(key);
      else this.warnings.push(`unknown stage record ${g.path}/${key}; not shown`);
    }
    return st;
  }

  /**
   * `/opensees/decls` (2.25.0) joined with `/opensees/decl_params` (2.26.0),
   * checked as the Python reader checks them (`H5Model.declarations()`, which
   * raises `MalformedH5Error`): equal column lengths, every row index in
   * range, `params` a JSON object whose tagged values are `$decl` (to a
   * declared key), `$struct` or `$opaque`, `params_names` "" or a JSON list
   * of strings, and every `*_ref` run inside its keys and naming declared keys.
   */
  private readDecls(g: H5Group): DeclTable {
    const d = this.group(g, "decls");
    const key = strings(this.dataset(d, "key"));
    const family = strings(this.dataset(d, "family"));
    const name = strings(this.dataset(d, "name"));
    const synth = numbers(this.dataset(d, "synth"));
    const n = key.length;
    if (family.length !== n || name.length !== n || synth.length !== n) {
      throw new SchemaError(`${d.path}: key, family, name and synth lengths differ (${n}, ${family.length}, ${name.length}, ${synth.length})`);
    }
    const keys = new Set(key);
    if (keys.size !== n) throw new SchemaError(`${d.path}/key holds a key twice`);
    const inRange = (i: number, where: string): number => {
      if (!Number.isInteger(i) || i < 0 || i >= n) throw new SchemaError(`${where} is ${i}, outside the ${n} rows of ${d.path}`);
      return i;
    };
    const records: DeclRecord[] = key.map((k, i) => ({ row: i, key: k, family: family[i]!, name: name[i]!, synth: synth[i] !== 0, params: [] }));

    const tags: DeclTagRun[] = [];
    if (has(d, "tags")) {
      const t = this.group(d, "tags");
      const kind = strings(this.dataset(t, "kind"));
      const first = numbers(this.dataset(t, "first"));
      const count = numbers(this.dataset(t, "count"));
      const decl = numbers(this.dataset(t, "decl"));
      if (first.length !== kind.length || count.length !== kind.length || decl.length !== kind.length) {
        throw new SchemaError(`${t.path}: kind, first, count and decl lengths differ`);
      }
      kind.forEach((k, i) => tags.push({ kind: k, first: first[i]!, count: count[i]!, decl: inRange(decl[i]!, `${t.path}/decl[${i}]`) }));
    }

    const rows: Record<string, number[]> = Object.create(null) as Record<string, number[]>;
    if (has(d, "rows")) {
      const walk = (grp: H5Group, prefix: string) => {
        for (const k of grp.keys()) {
          const node = grp.get(k);
          const p = prefix ? `${prefix}/${k}` : k;
          if (node instanceof this.h5.Group) walk(node as H5Group, p);
          else rows[p] = Array.from(numbers(this.dataset(grp, k)), (v, i) => inRange(v, `${grp.path}/${k}[${i}]`));
        }
      };
      walk(this.group(d, "rows"), "");
    }

    const hasParams = has(g, "decl_params");
    if (hasParams) {
      for (const p of this.readDeclParams(this.group(g, "decl_params"), keys, n)) records[p.decl]!.params.push(p.params);
    }
    return { hasParams, records, tags, rows };
  }

  private readDeclParams(g: H5Group, keys: Set<string>, nDecls: number): { decl: number; params: DeclParams }[] {
    const decl = numbers(this.dataset(g, "decl"));
    const type = strings(this.dataset(g, "type"));
    const params = strings(this.dataset(g, "params"));
    const names = strings(this.dataset(g, "params_names"));
    const P = decl.length;
    if (type.length !== P || params.length !== P || names.length !== P) {
      throw new SchemaError(`${g.path}: decl, type, params and params_names lengths differ (${P}, ${type.length}, ${params.length}, ${names.length})`);
    }
    const refCol = (col: "transf_ref" | "integration_ref" | "section_ref"): string[][] => {
      if (!has(g, col)) throw new SchemaError(`${g.path}/${col} is missing`);
      const r = this.group(g, col);
      const first = numbers(this.dataset(r, "first"));
      const count = numbers(this.dataset(r, "count"));
      const key = strings(this.dataset(r, "key"));
      if (first.length !== P || count.length !== P) {
        throw new SchemaError(`${r.path}: first and count have ${first.length} and ${count.length} rows, not ${P}`);
      }
      return Array.from({ length: P }, (_, i) => {
        const a = first[i]!, c = count[i]!;
        if (!Number.isInteger(a) || !Number.isInteger(c) || a < 0 || c < 0 || a + c > key.length) {
          throw new SchemaError(`${r.path} row ${i} runs ${a}..${a + c}, outside its ${key.length} keys`);
        }
        const run = key.slice(a, a + c);
        for (const k of run) if (!keys.has(k)) throw new SchemaError(`${r.path} row ${i} is ${JSON.stringify(k)}, which /opensees/decls does not declare`);
        return run;
      });
    };
    const transf = refCol("transf_ref");
    const integration = refCol("integration_ref");
    const section = refCol("section_ref");
    const out: { decl: number; params: DeclParams }[] = [];
    for (let i = 0; i < P; i++) {
      const di = decl[i]!;
      if (!Number.isInteger(di) || di < 0 || di >= nDecls) {
        throw new SchemaError(`${g.path} row ${i} points at declaration ${di}, outside the ${nDecls} rows of /opensees/decls`);
      }
      const where = `${g.path}/params[${i}]`;
      let parsed: unknown;
      try {
        parsed = JSON.parse(params[i]!);
      } catch (e) {
        throw new SchemaError(`${where} is not JSON: ${(e as Error).message}`);
      }
      if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) throw new SchemaError(`${where} is not a JSON object`);
      checkDeclValue(parsed, keys, where);
      let paramsNames: string[] | null = null;
      if (names[i] !== "") {
        let pn: unknown;
        try {
          pn = JSON.parse(names[i]!);
        } catch (e) {
          throw new SchemaError(`${g.path}/params_names[${i}] is not JSON: ${(e as Error).message}`);
        }
        if (!Array.isArray(pn) || !pn.every((x) => typeof x === "string")) {
          throw new SchemaError(`${g.path}/params_names[${i}] is not "" or a JSON list of strings`);
        }
        paramsNames = pn as string[];
      }
      out.push({
        decl: di,
        params: {
          row: i,
          type: type[i]!,
          params: parsed as Record<string, DeclValue>,
          paramsNames,
          refs: { transf: transf[i]!, integration: integration[i]!, section: section[i]! },
        },
      });
    }
    return out;
  }

  /**
   * The root `/sequence` zone, when the file carries it: `absent` otherwise
   * (never an error). A version the app does not read is `refused`; a broken
   * zone raises, naming its path, like every other zone of the model file.
   */
  private readSequence(root: H5Group, meta: Record<string, Param | Param[]>): SequenceZone {
    if (!has(root, "sequence")) return { status: "absent" };
    const raw = meta["sequence_schema_version"];
    const key = "sequence_schema_version";
    let version: string;
    try {
      version = checkZoneVersion(raw, key, SEQUENCE_TARGET, SEQUENCE_FLOOR, this.warnings);
    } catch (e) {
      const m = typeof raw === "string" ? /^(\d+)\.(\d+)\.(\d+)$/.exec(raw) : null;
      if (!m || !(e instanceof SchemaError)) throw e;
      const t = SEQUENCE_TARGET;
      return {
        status: "refused",
        version: raw as string,
        accepted: `${t.major}.${SEQUENCE_FLOOR} and any later ${t.major}.x`,
        newer: Number(m[1]) > t.major,
        reason: e.message.slice(e.message.indexOf(": ") + 2),
      };
    }
    const g = this.group(root, "sequence");
    return { status: "ready", version, attrs: readAttrs(g), stages: this.flatTable(g, "stages"), changes: this.flatTable(g, "changes") };
  }

  /** A flat table: a compound dataset, or a group of equal-length 1-D column datasets. */
  private flatTable(parent: H5Group, name: string): Table {
    if (!has(parent, name)) throw new SchemaError(`${joinPath(parent.path, name)} is missing`);
    const node = parent.get(name);
    if (node instanceof this.h5.Dataset) return readTable(node as H5Dataset);
    const g = this.group(parent, name);
    const columns = g.keys();
    const cols = columns.map((c) => {
      const v = this.dataset(g, c).json_value;
      if (!Array.isArray(v)) throw new SchemaError(`${g.path}/${c} is not a 1-D column`);
      return v as unknown[];
    });
    const n = cols[0]?.length ?? 0;
    cols.forEach((c, j) => {
      if (c.length !== n) throw new SchemaError(`${g.path}: column ${columns[j]} has ${c.length} rows, not ${n}`);
    });
    const rows = Array.from({ length: n }, (_, i) => cols.map((c) => c[i]));
    return { path: g.path, columns, rows };
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

/**
 * The children of a `stage_NNN` group the reader lists by name and does not
 * interpret (h5-schema.md, opensees 2.18.0 and 2.21.0); the stage windows
 * that need them read them. Any other child is a loud warning.
 */
const STAGE_OTHER = new Set([
  "owned_node_ids",
  "owned_element_ids",
  "regions",
  "constraints",
  "rayleigh",
  "rayleigh_emit_index",
  "remove_sp",
  "remove_element",
  "activated_pgs",
  "initial_stress",
  "activate_absorbing",
  "update_material_stage",
]);

/** The tags a `decl_params` JSON object may carry (h5-schema.md, 2.26.0). */
function checkDeclValue(v: unknown, keys: Set<string>, where: string): void {
  if (Array.isArray(v)) {
    v.forEach((x, i) => checkDeclValue(x, keys, `${where}[${i}]`));
    return;
  }
  if (v === null || typeof v !== "object") return;
  const o = v as Record<string, unknown>;
  const fields = Object.keys(o);
  const tags = fields.filter((k) => k.startsWith("$")).sort();
  if (tags.length === 0) {
    for (const k of fields) checkDeclValue(o[k], keys, `${where}.${k}`);
    return;
  }
  const only = (...want: string[]) => fields.length === want.length && want.every((w) => w in o);
  if (only("$decl")) {
    const key = o["$decl"];
    if (typeof key !== "string" || !keys.has(key)) {
      throw new SchemaError(`${where} references ${JSON.stringify(key)}, which /opensees/decls does not declare`);
    }
    return;
  }
  if (only("$struct", "fields") && o["fields"] !== null && typeof o["fields"] === "object" && !Array.isArray(o["fields"])) {
    const f = o["fields"] as Record<string, unknown>;
    for (const k of Object.keys(f)) checkDeclValue(f[k], keys, `${where}.${k}`);
    return;
  }
  if (only("$opaque")) return;
  throw new SchemaError(`${where} carries the unknown tag(s) ${JSON.stringify(tags)}; the reader knows $decl, $struct and $opaque`);
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
