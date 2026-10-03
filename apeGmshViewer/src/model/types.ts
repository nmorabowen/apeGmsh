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
}

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
  /** Loud, user-visible warnings raised while reading. */
  warnings: string[];
  /** Milliseconds spent in the reader (open to return). */
  readMs: number;
}
