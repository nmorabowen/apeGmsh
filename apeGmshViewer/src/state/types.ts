// The state and event types of the one store (ADR 0112 D6; the reconciled
// V0 design on #1283, decisions 15-17).
//
// Everything in `State` is plain data: records, arrays, strings, numbers,
// booleans and null. Large arrays live in the BlobStore (blobs.ts), outside
// the snapshot, and the state holds only a `BlobRef` to each. The
// reducer-purity test (test/store.test.ts) walks every state the reducer
// returns and fails on a TypedArray, an ArrayBuffer, a Map or a Set.

import type { Field } from "../chain/resolve.ts";
import type { OpsFamily, Param } from "../model/types.ts";

/**
 * A declaration path, `<zone>/<family>/<name|#k>` (decision 11): the key of
 * `decls`, of provenance (V2c/V2d) and of a selection. Never an HDF5 group
 * name and never an OpenSees tag.
 *
 *   mesh/element/<femId>            a neutral-zone cell (its FEM id is the mesh identity)
 *   opensees/element/<type>#<row>   an OpenSees-only element row (no cell)
 *   opensees/<family>/<name>        an object named in /opensees/names
 *   opensees/<family>/#<k>          an unnamed object, k = its order in the reader's listing
 *   mesh/physical_group/<name>      an element-side physical group
 *   mesh/label/<name>               a label
 */
export type DeclPath = string;

export type DeclKind = "element" | "physical_group" | "label" | OpsFamily;

/** A reference into the BlobStore. The array itself is never in the state. */
export interface BlobRef {
  readonly id: string;
  /** what the array is, as a path-like name (`model/mesh/linePositions`) */
  readonly path: string;
  readonly dtype: "f32" | "f64" | "i32" | "u8";
  readonly shape: readonly number[];
}

/** The facts of a cell or OpenSees-only row that the inspector shows. */
export interface ElementFacts {
  /** FEM id of the cell; null for an OpenSees-only row */
  femId: number | null;
  /** `/elements/{alias}` block and row of the cell; null for an OpenSees-only row */
  cell: { alias: string; block: number; row: number } | null;
  /** node tags, from the cell's connectivity or the row's inline_connectivity */
  nodes: number[];
  /** element-side physical groups that contain the cell, as decl paths */
  groups: DeclPath[];
  /** every `/opensees/element_meta/{type}` row joined to the cell by fem_eids (one for an OpenSees-only row) */
  metas: { type: string; tag: number; h5: string; row: number; args: Param[] }[];
}

export interface Decl {
  path: DeclPath;
  kind: DeclKind;
  /** OpenSees type, cell alias, or the group's family */
  type: string;
  name: string;
  /** where the name came from (`/opensees/names`, or the HDF5 path whose group name was used) */
  nameSource: string;
  /** HDF5 path of the group or block the declaration was read from */
  h5: string;
  /** the OpenSees tag of an object (`@tag`); elements and groups have none here */
  tag: number | null;
  params: Param[];
  /** declarations this one references, by field (`transfTag`, `secTag`, `material_ref#0`, ...) */
  refs: Record<string, DeclPath>;
  /**
   * How each ref was followed (label, value, source, interpreted, note), keyed
   * like `refs`, and the links that could not be followed. For an element
   * these are derived by the chain selector from `element` (kept out of the
   * snapshot: one entry per element would be most of the state's weight), so
   * an element decl carries `links: {}` and `problems: []`.
   */
  links: Record<string, Field>;
  problems: string[];
  /** display-ready read facts of an object: type, tag, params, attrs, tables (elements derive theirs) */
  fields: Field[];
  element?: ElementFacts;
  group?: { dim: number; tag: number; count: number; ids: BlobRef };
  /** reserved for /provenance (V2c, V2d) */
  provenance?: { file: string; line: number; function: string };
}

export type ArtifactKind = "geometry" | "model" | "results";

export type ZoneStatus =
  | { status: "absent" }
  | { status: "loading" }
  | { status: "ready"; version: string }
  | { status: "refused"; version: string; window: string; reason: string };

export interface ArtifactInfo {
  path: string;
  /** opened: the path is known and nothing has been read yet (a sibling V2f names, or a file being loaded) */
  status: "opened" | "ready" | "failed";
  error: string | null;
  /** the file changed on disk after it was read (fileChanged); the effects reload it */
  stale: boolean;
  sizeBytes: number;
  readMs: number;
  zones: Record<string, ZoneStatus>;
  /** loud reader warnings, shown in the banner */
  warnings: string[];
  counts: { nodes: number; cells: number; opsOnly: number };
}

/** One row of the legend: a colour group, in display order. */
export interface LegendEntry {
  /** the group's decl path, or null for the two synthetic rows */
  decl: DeclPath | null;
  name: string;
  color: readonly [number, number, number];
  elements: number;
}

/** The render derivation of the model artifact: blob refs, per-primitive indices, legend and bounds. */
export interface MeshInfo {
  /** 6 floats per segment; 9 per triangle; 6 per outline edge */
  linePositions: BlobRef;
  triPositions: BlobRef;
  edgePositions: BlobRef;
  /** index into `elements` per drawn segment / triangle (i32) */
  lineElement: BlobRef;
  triElement: BlobRef;
  /** index into `legend` per drawn segment / triangle (i32) */
  lineGroup: BlobRef;
  triGroup: BlobRef;
  /** the decl path of every drawn element, in first-drawn order */
  elements: DeclPath[];
  legend: LegendEntry[];
  center: readonly [number, number, number];
  radius: number;
  counts: { lineCells: number; faceCells: number; solidCells: number; opsOnly: number; points: number };
  warnings: string[];
}

export type PhaseKey =
  | { kind: "geometry" }
  | { kind: "mesh" }
  | { kind: "stage"; index: number }
  | { kind: "results"; step: number };

/** What the user clicked: the declaration and the hit point. */
export interface Pick {
  decl: DeclPath;
  at: readonly [number, number, number] | null;
}

export type WindowKind = "stages" | "analysis" | "patterns" | "recorders" | "materials";

export interface Layout {
  open: WindowKind[];
}

export interface State {
  artifacts: Record<ArtifactKind, ArtifactInfo | null>;
  decls: Record<DeclPath, Decl>;
  /** user names (/opensees/names, /labels, /physical_groups) to the declarations that carry them */
  names: Record<string, DeclPath[]>;
  mesh: MeshInfo | null;
  phase: { axis: PhaseKey[]; at: PhaseKey | null };
  selection: { decls: DeclPath[]; picks: Pick[] };
  hover: Pick | DeclPath | null;
  visibility: { hidden: DeclPath[]; edges: boolean; opacity: number };
  inspector: { pinned: DeclPath[] };
  windows: Layout;
  /** reserved for the D3 ADR; always empty */
  overrides: Record<DeclPath, never>;
}

/** The payload of `fileLoaded` for the model artifact: what the loader derived from the file. */
export interface ModelLoad {
  info: ArtifactInfo;
  decls: Record<DeclPath, Decl>;
  names: Record<string, DeclPath[]>;
  mesh: MeshInfo;
}

export type Event =
  | { type: "fileOpened"; artifact: ArtifactKind; path: string }
  | { type: "fileLoaded"; artifact: "model"; load: ModelLoad }
  | { type: "fileFailed"; artifact: ArtifactKind; path: string; error: string }
  | { type: "fileChanged"; path: string }
  | { type: "zoneRefused"; artifact: ArtifactKind; zone: string; version: string; window: string; reason: string }
  | { type: "select"; pick: Pick }
  | { type: "selectAdd"; pick: Pick }
  | { type: "clearSelection" }
  | { type: "hover"; at: Pick | DeclPath | null }
  | { type: "setHidden"; decls: DeclPath[]; hidden: boolean }
  | { type: "isolate"; decls: DeclPath[] }
  | { type: "showAll" }
  | { type: "setEdges"; on: boolean }
  | { type: "setOpacity"; value: number }
  | { type: "setPhase"; at: PhaseKey }
  | { type: "setResultStep"; step: number }
  | { type: "inspectorPin"; decl: DeclPath }
  | { type: "inspectorUnpin"; decl: DeclPath }
  | { type: "openWindow"; window: WindowKind }
  | { type: "closeWindow"; window: WindowKind };

export type EventType = Event["type"];
