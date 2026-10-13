// The in-memory form of one model.h5, as the reader builds it.
//
// Every value here was read from the file. Nothing is derived with
// apeGmsh semantics: the reader copies datasets and attributes, and the
// chain resolver (src/chain) records where each link came from.

/** One `/elements/{alias}` block of the neutral zone. */
export interface ElementBlock {
  alias: string;
  /** gmsh element-type code (`code` attr). */
  code: number;
  dim: number;
  /** nodes per element (`npe` attr). */
  npe: number;
  ids: Float64Array;
  /** row-major (ids.length, npe) node tags. */
  connectivity: Float64Array;
}

/** One element-side physical group (or label) entry. */
export interface Group {
  name: string;
  dim: number;
  tag: number;
  /** HDF5 path of the group entry. */
  path: string;
  elementIds: Float64Array;
}

/** A positional parameter: a number, or a string token (flag or name). */
export type Param = number | string;

/** `/opensees/{materials,sections,transforms,beam_integration}/...` member. */
export interface OpsObject {
  family: OpsFamily;
  /** HDF5 path of the group, e.g. `/opensees/sections/Fiber_1`. */
  path: string;
  /** HDF5 group name (the writer names it `{type}_{tag}`). */
  groupName: string;
  type: string;
  tag: number;
  /** `params` merged with `params_str` (a string slot wins). */
  params: Param[];
  /** Every other scalar or array attribute, verbatim. */
  attrs: Record<string, Param | Param[]>;
  /** Compound datasets under the group (patches, fibers, layers, ...). */
  tables: Record<string, Table>;
}

export type OpsFamily =
  | "uniaxialMaterial"
  | "nDMaterial"
  | "section"
  | "geomTransf"
  | "beamIntegration";

/** A compound dataset as named columns of rows. */
export interface Table {
  path: string;
  columns: string[];
  rows: unknown[][];
}

/** `/opensees/element_meta/{type}`: per-type OpenSees element rows. */
export interface ElementMeta {
  type: string;
  path: string;
  /** OpenSees element tags. */
  ids: Float64Array;
  /** FEM element ids; -1 for elements emitted outside a mesh fan-out. */
  femEids: Float64Array;
  /** Per-row positional args after the connectivity prefix. */
  args: Param[][];
  /** Present when some rows have no neutral-zone cell (node-pair, truss rebar). */
  inlineConnectivity: number[][] | null;
}

/** One `/opensees/names` alias row. */
export interface NameAlias {
  name: string;
  kind: string;
  tag: number;
}

export interface OpenSeesZone {
  version: string;
  objects: OpsObject[];
  elementMeta: ElementMeta[];
  names: NameAlias[];
  /** the model-definition data: declarations, analysis, series, patterns, recorders, stages */
  definitions: Definitions;
}

/**
 * A JSON value of `/opensees/decl_params/params` (h5-schema.md, 2.26.0): a
 * plain JSON value, or one of the three tagged objects, `{"$decl": key}` (a
 * reference to another declaration, by its `/opensees/decls` key),
 * `{"$struct": class, "fields": {...}}` (a value dataclass) or
 * `{"$opaque": class}`. The reader checks every tag and every `$decl` key.
 */
export type DeclValue = null | boolean | number | string | DeclValue[] | { [field: string]: DeclValue };

/** One owner's row of `/opensees/decl_params`: a declaration's parameters by field name. */
export interface DeclParams {
  /** the row in `/opensees/decl_params` */
  row: number;
  /** the primitive class (`Steel01`, `Fiber`, `FixRecord`, ...) */
  type: string;
  /** field name -> value, in the writer's field order */
  params: Record<string, DeclValue>;
  /** the argv slot names of the declaration's store row; null when the archive names none (`""`) */
  paramsNames: string[] | null;
  /** the GeomTransf / BeamIntegration / Section declaration keys the fields reference, in field order */
  refs: { transf: string[]; integration: string[]; section: string[] };
}

/** One `/opensees/decls` row (opensees 2.25.0): a user declaration of the bridge. */
export interface DeclRecord {
  /** the row in `/opensees/decls`; `tags/decl`, `rows/*` and `decl_params/decl` index it */
  row: number;
  /** `opensees/<family>/<name|#k>`, or a synthesised `<verb>:<owner>[/<role>]` key */
  key: string;
  /** the allocator kind (`uniaxialMaterial`, `pattern`, `recorder`, ...) or the tagless verb (`fix`, `mass`) */
  family: string;
  /** the user's `name=`; "" when unnamed */
  name: string;
  /** an object apeGmsh synthesised inside a user verb */
  synth: boolean;
  /** its `/opensees/decl_params` rows, one per owner in call order; [] when the file has no decl_params (opensees < 2.26) */
  params: DeclParams[];
}

/** A run of tags of one allocator kind that one declaration owns (`/opensees/decls/tags`). */
export interface DeclTagRun {
  kind: string;
  first: number;
  count: number;
  /** the `/opensees/decls` row */
  decl: number;
}

/** `/opensees/decls` and `/opensees/decl_params`, joined. */
export interface DeclTable {
  /** `/opensees/decl_params` is in the file (opensees 2.26.0) */
  hasParams: boolean;
  records: DeclRecord[];
  tags: DeclTagRun[];
  /**
   * `rows/<store>`: one declaration row per record of a tagless store, keyed
   * by the store's path under `rows/` (`bcs/fix`, `recorders`,
   * `stages/stage_000/bcs/fix`, ...).
   */
  rows: Record<string, number[]>;
  /**
   * `/opensees/decls@program_decl` (opensees 2.31.0): "complete" when the
   * program's hashed `decl` column records each call's declaration, "absent"
   * when it does not; a file below 2.31.0 has no marker and reads as
   * "absent" (h5-schema.md). Another value is kept verbatim (a non-string as
   * its JSON) and names no era. Read only for the results pairing (V4h).
   */
  programDecl?: string;
}

/**
 * A member of `/opensees/{time_series,patterns,recorders}` or of a stage's
 * `patterns/` or `recorders/`: its attributes verbatim and its datasets as
 * tables. `tag` is null for a recorder (recorders carry none).
 */
export interface DefObject {
  path: string;
  groupName: string;
  type: string;
  tag: number | null;
  /** `params` merged with `params_str` (a string slot wins) */
  params: Param[];
  /** every other attribute, verbatim */
  attrs: Record<string, Param | Param[]>;
  tables: Record<string, Table>;
}

/** `/opensees/analysis` or a stage's `analysis/`: the chain tokens and their `*_args`, verbatim. */
export interface AnalysisDef {
  path: string;
  attrs: Record<string, Param | Param[]>;
}

/** The fix and mass records of `/opensees/bcs` or of a stage's `bcs/`. */
export interface BcsDef {
  fix: Table | null;
  mass: Table | null;
}

/** One `/opensees/stages/stage_NNN` group. */
export interface StageDef {
  /** NNN, the stage's ordinal */
  index: number;
  path: string;
  /** the `name` attribute */
  name: string;
  /** every attribute verbatim (`analyze_steps`, `analyze_dt`, `set_time`, `domain_change`, ...) */
  attrs: Record<string, Param | Param[]>;
  analysis: AnalysisDef | null;
  patterns: DefObject[];
  recorders: DefObject[];
  bcs: BcsDef;
  /** the stage's other children (h5-schema.md, opensees 2.18.0), by name, not interpreted here */
  other: string[];
}

/** The model-definition data of the opensees zone. */
export interface Definitions {
  /** null when the file has no `/opensees/decls` (opensees < 2.25) */
  decls: DeclTable | null;
  /** null when the file has no global `/opensees/analysis` (a staged file carries one per stage) */
  analysis: AnalysisDef | null;
  timeSeries: DefObject[];
  patterns: DefObject[];
  recorders: DefObject[];
  bcs: BcsDef;
  /** `/opensees/stages/stage_NNN` in ordinal order; [] for an unstaged file */
  stages: StageDef[];
}

/**
 * The root `/sequence` zone (ADR 0112 D2(b), V3a; `/meta/sequence_schema_version`):
 * a flat `stages` table and a flat `changes` table derived from the archive.
 * `absent` is the typed "not in this file" state, never an error; a version
 * the app does not read is `refused`, and the model opens without it.
 */
export type SequenceZone =
  | { status: "absent" }
  | { status: "ready"; version: string; attrs: Record<string, Param | Param[]>; stages: Table; changes: Table }
  | { status: "refused"; version: string; accepted: string; newer: boolean; reason: string };

export interface ModelFile {
  path: string;
  sizeBytes: number;
  neutralVersion: string;
  meta: Record<string, Param | Param[]>;
  nodeIds: Float64Array;
  /** row-major (n, 3) coordinates. */
  nodeCoords: Float64Array;
  blocks: ElementBlock[];
  physicalGroups: Group[];
  labels: Group[];
  opensees: OpenSeesZone | null;
  /** the root `/sequence` zone; `absent` when the file has none */
  sequence: SequenceZone;
  /** Loud, user-visible warnings raised while reading. */
  warnings: string[];
  /** Milliseconds spent in the reader (open to return). */
  readMs: number;
}
