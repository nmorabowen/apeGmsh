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
  EmbeddedRebarTies,
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
  ProgramDecls,
  Table,
} from "../model/types.ts";
import { SIDE_CHANNEL_DECL_VERBS } from "../state/results.ts";
import { openH5, type H5FS } from "./h5open.ts";

/** The parts of the h5wasm module this reader uses. */
export interface H5Module {
  File: new (filename: string, mode?: "r") => H5File;
  readonly FS: H5FS | null;
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
/**
 * opensees 2.27.0: `/opensees/decls` (2.25), `/opensees/decl_params` (2.26,
 * K1-7), and 2.27's additions (K1-8, h5-schema.md "Parameter tags (opensees
 * 2.27.0)"), none of which this app needs to read:
 * - `@param_tags` on an initial stress and a `param_tags` dataset under an
 *   absorbing flip: the build's `parameter` tags, stored so a deck replay
 *   mints none. Replay data: no window shows a parameter tag, and the stage
 *   children that carry them (`initial_stress`, `activate_absorbing`) are
 *   listed by name, not interpreted (`STAGE_OTHER`).
 * - two new optional stage children, `update_parameter/update_NNN` and
 *   `zero_velocities/zero_NNN` (`s.update_parameter`, `s.zero_velocities`):
 *   listed by name in `STAGE_OTHER` like the other uninterpreted stage
 *   records, so a 2.27 file that has them opens with no warning. `/sequence`
 *   does not record either verb (archive gap #1646), which the stage diagram
 *   already says on every file (`NOT_RECORDED`).
 *
 * opensees 2.28.0 (ADR 0115 amendment R2-f, h5-schema.md "`label`: provenance
 * only (opensees 2.28.0)"): the `decl_params` rows of fix, mass and typed
 * Node / Element recorder declarations gain a `label` field, the names a
 * `label=` declaration was given (`null` for `pg=` or explicit ids). No path
 * or dtype changes, so the reader needs no new read: `decl_params` rows are
 * read field by field and `label` arrives as a field. The rule for every
 * window: `label` is provenance, not an OpenSees parameter. No window lists it
 * as a parameter or an unset field; the Recorders window, the one window that
 * renders those declarations' fields, shows it on the targets line, and a null
 * label shows nothing. The ids stay authoritative: nothing the app draws or
 * selects resolves a label.
 *
 * opensees 2.29.0 (K2b #1795, h5-schema.md "`/opensees/constraints/
 * embeddedRebar` (optional, opensees 2.29.0)"): a new optional group holding
 * every `g.reinforce` LadrunoEmbeddedRebar tie the build wrote (its tag and
 * positional arguments). The ties are not elements (`element_meta` does not
 * list them, so the chain and the element tables never meet one) and get no
 * `/opensees/decls` row. Since V4m (#1846) the reader reads the group for one
 * fact, the bond material each tie names (`tieBonds`, below; the other
 * columns are replay data it does not read), and the materials window counts
 * those ties as users. The gap this comment used to name is closed: a bond
 * material only the ties use no longer reads "used by nothing", and a tie
 * whose bond tag cannot be read makes the window say "users unknown", never
 * 0. A file without the group (before 2.29, or with no tie) reads as before.
 *
 * opensees 2.30.0 (R3-S5 #1789, h5-schema.md "`precision` (opensees
 * 2.30.0)"): typed Node / Element recorder `decl_params` rows gain
 * `precision`, a real OpenSees option (`-precision N`). Unlike `label` it is a
 * parameter: a value is a parameter row in the Recorders window, and `null`
 * means "not given; the deck carries no `-precision`, OpenSees' default",
 * so it is listed among the unset fields, truthfully. No new read.
 *
 * opensees 2.31.0 (A7 #1839, h5-schema.md "`/opensees/program`"): the
 * program's `decl` column now holds the `/opensees/decls` row that made each
 * call (`-1` where unproved), and `/opensees/decls@program_decl`
 * ("complete" | "absent") says whether those joins are recorded. This app
 * reads the marker only for the results pairing (V4h, #1856: it says whether
 * the model_hash includes declaration ids), never for a join. It reads
 * `/opensees/program` only for that pairing too (`programDecls`, #2189), and
 * reads only the datasets of `/opensees/decls`, so `decl` can never appear as an
 * OpenSees parameter in any window, and no join is made from it. The joins the
 * windows make are unchanged and do not depend on the marker: recorders join
 * through `/opensees/decls/rows` (recorded by the writer, not guessed); the
 * analysis window's chain ↔ declaration join is by count and type order, and
 * a refused or ambiguous join says the archive records no link (#1646). Using `program.decl` for that join when the
 * marker is "complete" is a follow-up, not this target bump.
 *
 * opensees 2.32.0 (#1853, h5-schema.md "`decl`: each call's declaration"):
 * the six side-channel verbs (`step_hook_ramp`, `addToParameter`,
 * `flip_element_stage`, `update_parameter`, `set_node_vel`, `set_node_accel`)
 * now carry their record's `decl` instead of `-1`. No path, dtype or marker
 * change, so no window reads differently. The column is hashed: a 2.32 re-emit
 * of a model with any of those calls has a new model_hash. The results pairing
 * names that change (#2189): it reads each side's `/opensees/program` methods
 * and `decl` (`programDecls`, below) and applies Python's rule
 * (state/results.ts `sideChannelHashCause`).
 *
 * opensees 2.33.0 (#2372, h5-schema.md "Solve stamp"): `/opensees@requires`
 * may name the value tokens `"mpco"` (a `recorder mpco`) and `"mpi"` (a
 * Mumps-family or `Parallel*` `system` or a partitioned `@solve_mode`). No path or dtype
 * change, and this app does not read the solve stamp, so nothing here reads
 * differently. `@requires` is hashed: a 2.33 re-emit of a model with an MPCO
 * recorder, a Mumps system or partitions has a new model_hash, which the
 * results pairing shows as a plain mismatch.
 *
 * opensees 2.34.0 (#2404, h5-schema.md "Large arrays as datasets"): a
 * `params` / `params_str` pair, or a declared recorder's `ids`, past the
 * HDF5 attribute limit is stored as same-named datasets in its group.
 * `readDef` reads either form and refuses a group holding both; smaller
 * arrays keep the attribute form and read as before.
 *
 * opensees 2.35.0 (h5-schema.md "/opensees/transforms"): a transform group
 * whose `geomTransf` line carries `-jntOffset` gains a `jnt_offset` dataset
 * ((1, 6) in 3-D, (1, 4) in 2-D). `readObject` reads it as one more table;
 * the scene still draws each beam between its nodes, not its offset ends.
 *
 * tests/test_app_target_drift.py fails when any `*_TARGET` here (or
 * `RESULTS_TARGET` in results.ts) differs from its Python writer's version:
 * bump it only after reading that zone's history entry in h5-schema.md.
 */
export const OPENSEES_TARGET = { major: 2, minor: 35 } as const;
/** The ADR 0112 zones (V2a specs): read by geometry.ts and provenance.ts. */
export const GEOMETRY_TARGET = { major: 1, minor: 0 } as const;
export const PROVENANCE_TARGET = { major: 1, minor: 1 } as const;
/** The root `/sequence` zone (ADR 0112 D2(b), V0 decision 8: "1.0.0"), read here when a file carries it. */
export const SEQUENCE_TARGET = { major: 1, minor: 0 } as const;
/**
 * The lowest minor of each zone this reader opens, under the zone's target
 * major (ADR 0113 D1/D7: one floor table, the only place the version check
 * reads; it equals the Python writer constants). Neutral 2.10 is the B2 layout
 * split (physical groups and labels side-partitioned); opensees 2.12 is the
 * first era whose files open (2.11 is the 0-based rank flip, but every 2.11
 * writer stamped a neutral zone below the neutral floor, so no 2.11 file
 * opens); the ADR 0112 zones (geometry, provenance, sequence) start at their
 * first version. tests/test_app_floor_drift.py compares every row with
 * Python's `reader_floor(zone)`.
 * TODO(V4): results 1.0 joins this table with the app's results reader.
 */
export const ZONE_FLOOR = { neutral: 10, opensees: 12, geometry: 0, provenance: 0, sequence: 0 } as const;

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
  embeddedRebar: null,
  program: null,
});

export function readModel(
  h5: H5Module,
  path: string,
  sizeBytes: number,
): ModelFile {
  const t0 = performance.now();
  const f = openH5(h5, path);
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
  /** The opensees zone's stamp while `/opensees` is read (gates the 2.34.0 dataset form). */
  private opsVersion: string | null = null;

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
    this.opsVersion = version;
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
    const definitions = this.readDefinitions(g);
    // A 2.25+ file declares every tag (the writer refuses a row no declaration
    // claims): an object no `decls/tags` run owns is a broken file, never a
    // fallback to an app-made key.
    const table = definitions.decls;
    if (table) {
      for (const o of objects) {
        if (!table.tags.some((r) => r.kind === o.family && o.tag >= r.first && o.tag < r.first + r.count)) {
          throw new SchemaError(`${o.path}: /opensees/decls/tags has no run for ${o.family} ${o.tag}; a 2.25+ file declares every tag`);
        }
      }
    }
    return { version, objects, elementMeta, names, definitions };
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
      embeddedRebar: this.readEmbeddedRebar(g),
      program: this.readProgram(g),
    };
  }

  /**
   * `/opensees/program`'s side-channel calls and their declaration proof
   * (`programDecls`), read for the results pairing only (#2189). Read only
   * where the 2.32.0 rule can use it: the decls marker reads "complete", and
   * the table itself only when `@methods` names a side-channel verb. null
   * when the marker is not "complete" or the zone has no program (a rewrite
   * below 2.23.0 writes none); a table that cannot be read is a loud warning
   * and null, so the pairing names no cause from it, and the model still opens.
   */
  private readProgram(g: H5Group): ProgramDecls | null {
    if (!has(g, "program") || !has(g, "decls")) return null;
    if (this.group(g, "decls").attrs["program_decl"]?.value !== "complete") return null;
    const node = g.get("program");
    if (!(node instanceof this.h5.Dataset)) {
      this.warnings.push(`${joinPath(g.path, "program")} is not a dataset; the results pairing does not read it`);
      return null;
    }
    const ds = node as H5Dataset;
    const dt = ds.dtype;
    const columns = Array.isArray(dt) ? (dt as unknown[]).map((m) => String((m as unknown[])[0])) : [];
    const out = programDecls(ds.path, ds.attrs["methods"]?.value, columns, () => ds.value, SIDE_CHANNEL_DECL_VERBS);
    if (typeof out === "string") {
      this.warnings.push(`${out}; the results pairing does not read it`);
      return null;
    }
    return out;
  }

  /**
   * `/opensees/constraints/embeddedRebar` (opensees 2.29.0; V4m #1846): the
   * four columns that carry a tie's arguments, reduced by `tieBonds` to the
   * bond material each tie names. `ele_tag` and `name` are replay data and
   * are not read. A missing column is an unknown, not an error: the file
   * opened before V4m, and the window says "users unknown" instead.
   */
  private readEmbeddedRebar(g: H5Group): EmbeddedRebarTies | null {
    if (!has(g, "constraints")) return null;
    const c = this.group(g, "constraints");
    if (!has(c, "embeddedRebar")) return null;
    const er = this.group(c, "embeddedRebar");
    const opt = <T>(name: string, read: (ds: H5Dataset) => T): T | null => (has(er, name) ? read(this.dataset(er, name)) : null);
    return tieBonds(er.path, {
      arg_offsets: opt("arg_offsets", numbers),
      args: opt("args", numbers),
      args_str: opt("args_str", strings),
      arg_kinds: opt("arg_kinds", numbers),
    });
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
    const { params, large } = this.readLargeArrays(g, a, ["params", "params_str", "ids"]);
    if (large["ids"] !== undefined) attrs["ids"] = large["ids"];
    const tables: Record<string, Table> = {};
    for (const key of g.keys()) {
      if (key in large) continue;
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
      params,
      attrs,
      tables,
    };
  }

  /**
   * A typed group's `params` pair, in either form, and the same-named
   * datasets among `keys` (opensees 2.34.0: an array HDF5 refuses as an
   * attribute is written as a dataset). Refuses a key in both forms, a pair
   * split across them, a lone `params_str` dataset, and any dataset form in
   * a file stamped below 2.34.0, as the Python reader does.
   */
  private readLargeArrays(
    g: H5Group, a: Record<string, Param | Param[]>, keys: readonly string[],
  ): { params: Param[]; large: Record<string, Param[]> } {
    const children = g.keys();
    const large: Record<string, Param[]> = {};
    for (const key of keys) {
      if (!children.includes(key)) continue;
      if (key in a) throw new SchemaError(`${joinPath(g.path, key)} is both an attribute and a dataset; the writer stores one form`);
      if (!atLeast(this.opsVersion, LARGE_ARRAY_DATASETS_FROM)) {
        throw new SchemaError(`${joinPath(g.path, key)} is a dataset in an opensees ${this.opsVersion} file; a large array is stored as a dataset from opensees ${LARGE_ARRAY_DATASETS_FROM.major}.${LARGE_ARRAY_DATASETS_FROM.minor}.0 on`);
      }
      const ds = this.dataset(g, key);
      large[key] = key === "params_str" ? strings(ds) : Array.from(numbers(ds));
    }
    const pairInDatasets = "params" in large || "params_str" in large;
    if (pairInDatasets && ("params" in a || "params_str" in a)) {
      throw new SchemaError(`${g.path} holds its params pair in both forms; the writer stores one`);
    }
    if ("params_str" in large && !("params" in large)) {
      throw new SchemaError(`${joinPath(g.path, "params_str")} is a dataset with no 'params' beside it`);
    }
    const params = pairInDatasets ? mergeParams(large["params"], large["params_str"]) : mergeParams(a["params"], a["params_str"]);
    return { params, large };
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

    // `tags` and `rows` are required children (Python reads both unguarded).
    const tags: DeclTagRun[] = [];
    const t = this.group(d, "tags");
    const kind = strings(this.dataset(t, "kind"));
    const first = numbers(this.dataset(t, "first"));
    const count = numbers(this.dataset(t, "count"));
    const decl = numbers(this.dataset(t, "decl"));
    if (first.length !== kind.length || count.length !== kind.length || decl.length !== kind.length) {
      throw new SchemaError(`${t.path}: kind, first, count and decl lengths differ`);
    }
    kind.forEach((k, i) => tags.push({ kind: k, first: first[i]!, count: count[i]!, decl: inRange(decl[i]!, `${t.path}/decl[${i}]`) }));

    // `rows/<store>` holds one declaration row per record of `/opensees/<store>`:
    // as many as the store has (a dataset's rows, a group's children, the
    // columnar `commands` group's `method` entries; 0 for a store not in the file).
    const rows: Record<string, number[]> = Object.create(null) as Record<string, number[]>;
    const walk = (grp: H5Group, prefix: string) => {
      for (const k of grp.keys()) {
        const node = grp.get(k);
        const p = prefix ? `${prefix}/${k}` : k;
        if (node instanceof this.h5.Group) {
          walk(node as H5Group, p);
          continue;
        }
        const column = Array.from(numbers(this.dataset(grp, k)), (v, i) => inRange(v, `${grp.path}/${k}[${i}]`));
        const expected = this.storeCount(g, p);
        if (column.length !== expected) {
          throw new SchemaError(`${d.path}/rows/${p} has ${column.length} entries; ${joinPath(g.path, p)} has ${expected} records`);
        }
        rows[p] = column;
      }
    };
    walk(this.group(d, "rows"), "");

    const hasParams = has(g, "decl_params");
    if (hasParams) {
      for (const p of this.readDeclParams(this.group(g, "decl_params"), keys, n)) records[p.decl]!.params.push(p.params);
    }
    // V4h (#1856): the completeness marker, read for the results pairing only.
    const marker = d.attrs["program_decl"]?.value;
    const programDecl = marker === undefined ? "absent" : typeof marker === "string" ? marker : JSON.stringify(marker);
    return { hasParams, records, tags, rows, programDecl };
  }

  /** The record count of the store at `path` under `ops`, as Python's `declarations()` counts it. */
  private storeCount(ops: H5Group, path: string): number {
    let node: unknown = ops;
    for (const seg of path.split("/")) {
      if (!(node instanceof this.h5.Group) || !has(node as H5Group, seg)) return 0;
      node = (node as H5Group).get(seg);
    }
    if (node instanceof this.h5.Group) {
      const grp = node as H5Group;
      if (has(grp, "method")) return this.dataset(grp, "method").shape?.[0] ?? 0;
      return grp.keys().length;
    }
    if (node instanceof this.h5.Dataset) return (node as H5Dataset).shape?.[0] ?? 1;
    throw new SchemaError(`${joinPath(ops.path, path)} is neither a group nor a dataset`);
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
      version = checkZoneVersion(raw, key, SEQUENCE_TARGET, ZONE_FLOOR.sequence, this.warnings);
    } catch (e) {
      const m = typeof raw === "string" ? /^(\d+)\.(\d+)\.(\d+)$/.exec(raw) : null;
      if (!m || !(e instanceof SchemaError)) throw e;
      const t = SEQUENCE_TARGET;
      return {
        status: "refused",
        version: raw as string,
        accepted: `${t.major}.${ZONE_FLOOR.sequence} and any later ${t.major}.x`,
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
    const { params, large } = this.readLargeArrays(g, a, ["params", "params_str"]);
    const attrs: Record<string, Param | Param[]> = {};
    for (const [k, v] of Object.entries(a)) {
      if (k !== "type" && k !== "tag" && k !== "params" && k !== "params_str") attrs[k] = v;
    }
    const tables: Record<string, Table> = {};
    for (const key of g.keys()) {
      if (key in large) continue;
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
 * interpret (h5-schema.md, opensees 2.18.0, 2.21.0 and 2.27.0); the stage windows
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
  // opensees 2.27.0 (OPENSEES_TARGET says why the app does not read them)
  "update_parameter",
  "zero_velocities",
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

/** The four argument columns of `/opensees/constraints/embeddedRebar`; null for one the group lacks. */
export interface TieColumns {
  arg_offsets: ArrayLike<number> | null;
  args: ArrayLike<number> | null;
  args_str: readonly string[] | null;
  arg_kinds: ArrayLike<number> | null;
}

/**
 * The bond material each LadrunoEmbeddedRebar tie names (V4m, #1846). The
 * interpreted tier, like chain/signatures.ts: h5-schema.md names no bond
 * column, so the tag is read from the tie's arguments by the command grammar
 * of `src/apeGmsh/opensees/element/embedded_rebar.py` (the single source of
 * the emit grammar), which mirrors the fork's `OPS_LadrunoEmbeddedRebar.cpp`:
 *
 *     ... ( -bond matTag [-bondScale bs] | -perfect kAxial ) ...
 *
 * Row i owns `args[arg_offsets[i]:arg_offsets[i+1]]`, encoded as
 * `/opensees/commands` is (`arg_kinds` 0 int, 1 float, 2 str; a string in
 * `args_str`). The tag is the integer right after the one `-bond` string
 * token of the row; it is matched to a uniaxialMaterial by tag in the
 * window. A row with no `-bond` names no material. A row whose `-bond` has no
 * value, a non-integer value, or that carries `-bond` twice is unresolved, as
 * is every row when a column is missing or the columns disagree in length:
 * the window then says "users unknown", never 0.
 */
export function tieBonds(path: string, c: TieColumns): EmbeddedRebarTies {
  const off = c.arg_offsets;
  const count = off && off.length >= 1 ? off.length - 1 : null;
  const whole = (why: string): EmbeddedRebarTies => ({ path, count, bond: [], unbonded: 0, unresolved: [{ why, ties: count }] });
  const missing = (["arg_offsets", "args", "args_str", "arg_kinds"] as const).filter((k) => c[k] === null);
  if (missing.length) return whole(`${path} has no ${missing.join(", ")}`);
  const args = c.args!, str = c.args_str!, kinds = c.arg_kinds!;
  if (count === null) return whole(`${path}/arg_offsets is empty`);
  const a = args.length;
  if (str.length !== a || kinds.length !== a) {
    return whole(`${path}: args (${a}), args_str (${str.length}) and arg_kinds (${kinds.length}) differ in length`);
  }
  for (let i = 0; i <= count; i++) {
    const o = off![i]!;
    if (!Number.isInteger(o) || o < 0 || o > a || (i === 0 && o !== 0) || (i > 0 && o < off![i - 1]!) || (i === count && o !== a)) {
      return whole(`${path}/arg_offsets does not run from 0 to ${a} without decreasing`);
    }
  }
  const tags = new Map<number, number>();
  const why = new Map<string, number>();
  let unbonded = 0;
  for (let i = 0; i < count; i++) {
    const lo = off![i]!, hi = off![i + 1]!;
    const at: number[] = [];
    for (let k = lo; k < hi; k++) if (kinds[k] === 2 && str[k] === "-bond") at.push(k);
    let reason: string | null = null;
    if (at.length === 0) {
      unbonded++;
      continue;
    }
    if (at.length > 1) reason = "-bond appears more than once in the tie's arguments";
    else {
      const k = at[0]! + 1;
      if (k >= hi) reason = "-bond has no tag after it";
      else if (kinds[k] !== 0 || !Number.isInteger(args[k])) reason = "the value after -bond is not an integer tag";
      else tags.set(args[k]!, (tags.get(args[k]!) ?? 0) + 1);
    }
    if (reason) why.set(reason, (why.get(reason) ?? 0) + 1);
  }
  return {
    path,
    count,
    bond: [...tags].sort((x, y) => x[0] - y[0]).map(([tag, ties]) => ({ tag, ties })),
    unbonded,
    unresolved: [...why].map(([w, ties]) => ({ why: w, ties })),
  };
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

/** The first opensees minor whose writer stores an array HDF5 refuses as an attribute as a same-named dataset (#2404). */
const LARGE_ARRAY_DATASETS_FROM = { major: 2, minor: 34 } as const;

/** True when `version` ("M.m.p") is at or above `from`; false for null or unparsable. */
function atLeast(version: string | null, from: { major: number; minor: number }): boolean {
  const m = version === null ? null : /^(\d+)\.(\d+)\./.exec(version);
  if (m === null) return false;
  const major = Number(m[1]), minor = Number(m[2]);
  return major > from.major || (major === from.major && minor >= from.minor);
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

/**
 * `/opensees/program` (h5-schema.md, opensees 2.23.0) as `ProgramDecls`: for
 * each of `verbs` a run calls (`method` indexes `@methods`), whether one of
 * its runs carries a declaration (`decl >= 0`). Shared by this reader (the
 * model) and reader/results.ts (the results' embedded zone), so both sides
 * read the table one way. `rows` is called only when `@methods` names one of
 * `verbs`: a program that cannot call one is `{}` without reading its table.
 *
 * Returns the reason, naming `path`, when the table is not the schema's: no
 * `method` or `decl` column, an `@methods` that is not a 1-D list of strings
 * (a scalar string is refused: h5wasm reads a one-name list as `["x"]`, and
 * Python reads a scalar as its characters, so it would name no verb), or a
 * method index outside it.
 */
export function programDecls(path: string, methods: unknown, columns: readonly string[], rows: () => unknown, verbs: readonly string[]): ProgramDecls | string {
  const im = columns.indexOf("method");
  const id = columns.indexOf("decl");
  if (im < 0 || id < 0) return `${path} has no method and decl columns (it has ${JSON.stringify(columns)})`;
  if (!Array.isArray(methods) || !methods.every((x) => typeof x === "string")) return `${path}@methods is not a list of strings`;
  const names = methods as string[];
  const wanted = new Set(verbs);
  const out: Record<string, boolean> = Object.create(null) as Record<string, boolean>;
  if (!names.some((n) => wanted.has(n))) return out;
  const table = rows();
  if (!Array.isArray(table)) return `${path} is not a compound table`;
  for (const [k, row] of table.entries()) {
    const m = Array.isArray(row) ? Number(row[im]) : Number.NaN;
    const d = Array.isArray(row) ? Number(row[id]) : Number.NaN;
    if (!Number.isInteger(m) || m < 0 || m >= names.length) return `${path}[${k}].method is ${String(m)}, outside the ${names.length} names of @methods`;
    if (!Number.isInteger(d)) return `${path}[${k}].decl is not an integer`;
    const name = names[m]!;
    if (wanted.has(name)) out[name] = (out[name] ?? false) || d >= 0;
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

function flattenNumericRows(rows: readonly (readonly unknown[])[]): number[] {
  const out: number[] = [];
  const visit = (x: unknown): void => {
    if (Array.isArray(x)) {
      for (const item of x) visit(item);
    } else if (typeof x === "number") {
      out.push(x);
    } else if (typeof x === "bigint") {
      out.push(Number(x));
    }
  };
  visit(rows);
  return out;
}

/**
 * Extract `(emittedTag, vecxz)` pairs from a `/opensees/transforms/<name>`
 * `OpsObject` (`per_element_vecxz` of shape `(n, 3)` and
 * `per_element_emitted_tag` of shape `(n,)`, matching Python's
 * `H5Model.element_local_axes_vecxz`). A 2-D transform writes
 * `per_element_vecxz` of shape `(1, 0)` and returns `[]`.
 */
export function transformVecxzOf(obj: OpsObject): { tag: number; vecxz: [number, number, number] }[] {
  if (obj.family !== "geomTransf") return [];
  const vecTable = obj.tables["per_element_vecxz"];
  if (vecTable !== undefined) {
    if (vecTable.columns.length !== 3 || vecTable.rows.length === 0) return [];
    const flatVec = flattenNumericRows(vecTable.rows);
    if (flatVec.length < 3 || flatVec.length % 3 !== 0) return [];
    const nVec = flatVec.length / 3;
    const tagTable = obj.tables["per_element_emitted_tag"];
    const flatTags = tagTable ? flattenNumericRows(tagTable.rows) : [obj.tag];
    const n = Math.min(nVec, flatTags.length);
    const out: { tag: number; vecxz: [number, number, number] }[] = [];
    for (let r = 0; r < n; r++) {
      const tag = flatTags[r]!;
      const vx = flatVec[r * 3]!;
      const vy = flatVec[r * 3 + 1]!;
      const vz = flatVec[r * 3 + 2]!;
      if (Number.isInteger(tag) && Number.isFinite(vx) && Number.isFinite(vy) && Number.isFinite(vz)) {
        out.push({ tag, vecxz: [vx, vy, vz] });
      }
    }
    return out;
  }
  // Fallback for synthetic geomTransf objects constructed without `tables` in tests:
  if (
    (obj.params.length === 3 || obj.params.length === 9) &&
    typeof obj.params[0] === "number" &&
    typeof obj.params[1] === "number" &&
    typeof obj.params[2] === "number" &&
    Number.isFinite(obj.params[0]) &&
    Number.isFinite(obj.params[1]) &&
    Number.isFinite(obj.params[2])
  ) {
    return [{ tag: obj.tag, vecxz: [obj.params[0], obj.params[1], obj.params[2]] }];
  }
  return [];
}

/**
 * Map every emitted 3-D `geomTransf` tag in `model` to its `vecxz` `(3,)`
 * reference vector from `/opensees/transforms/<name>/{per_element_vecxz, per_element_emitted_tag}`.
 */
export function modelTransformVecxz(model: ModelFile): Map<number, [number, number, number]> {
  const out = new Map<number, [number, number, number]>();
  for (const obj of model.opensees?.objects ?? []) {
    if (obj.family !== "geomTransf") continue;
    for (const { tag, vecxz } of transformVecxzOf(obj)) {
      out.set(tag, vecxz);
    }
  }
  return out;
}

