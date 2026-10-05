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

// Panels import their chain types from here, never from src/chain (the
// panel allow-list: state/store, state/selectors, state/types, ui/).
export type { Chain, ChainNode, Field } from "../chain/resolve.ts";

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
  /**
   * `/elements/{alias}` block and row of the cell; null for an OpenSees-only
   * row. The cell's node tags are the range `row * npe .. (row + 1) * npe` of
   * `State.blocks[block].connectivity` (decision 15: no copy per element);
   * a `Pick` carries them when the viewport reads them for the inspector.
   */
  cell: { alias: string; block: number; row: number } | null;
  /** node tags of an OpenSees-only row (`inline_connectivity`, the reader's array, not a copy); [] for a cell */
  inlineNodes: readonly number[];
  /** element-side physical groups that contain the cell, as decl paths */
  groups: readonly DeclPath[];
  /** every `/opensees/element_meta/{type}` row joined to the cell by fem_eids (one for an OpenSees-only row) */
  metas: readonly { type: string; tag: number; h5: string; row: number; args: readonly Param[] }[];
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
  params: readonly Param[];
  /** declarations this one references, by field (`transfTag`, `secTag`, `material_ref#0`, ...) */
  refs: Readonly<Record<string, DeclPath>>;
  /**
   * How each ref was followed (label, value, source, interpreted, note), keyed
   * like `refs`, and the links that could not be followed. For an element
   * these are derived by the chain selector from `element` (kept out of the
   * snapshot: one entry per element would be most of the state's weight), so
   * an element decl carries `links: {}` and `problems: []`.
   */
  links: Readonly<Record<string, Field>>;
  problems: readonly string[];
  /** display-ready read facts of an object: type, tag, params, attrs, tables (elements derive theirs) */
  fields: readonly Field[];
  element?: ElementFacts;
  group?: { dim: number; tag: number; count: number; ids: BlobRef };
  /** where the user's code made this declaration (/provenance, joined by decl path); absent when the file has no record for it */
  provenance?: DeclSource;
}

/** A declaration's source location, from /provenance (src/reader/provenance.ts). */
export interface DeclSource {
  /** absolute path (POSIX separators) */
  file: string;
  line: number;
  function: string;
  /** the file's sha256 when it was captured */
  sha256: string;
  /** the outermost script line, when the call came through a helper; null when there is none */
  script: { file: string; line: number } | null;
}

/**
 * Who made a /provenance record (schema 1.1.0, `records/origin`): `user` for a
 * declaration the user made, `synthesised` for an object apeGmsh created
 * inside a verb the user called (a stage's HOLD series and support pattern).
 * A 1.0.x file reads every record as `user`.
 */
export type Origin = "user" | "synthesised";

/**
 * One /provenance record, as the sources listing shows it. The listing has
 * every record of the file, synthesised ones included (maintainer ruling on
 * #1378: shown by default), in capture order (`seq`).
 */
export interface ProvenanceEntry {
  /** the record's declaration path, `<zone>/<family>/<name|#k>`, or `<zone>/<family>/<verb>:<owner>[/<role>]` when synthesised */
  key: DeclPath;
  origin: Origin;
  /** 0-based capture order in the run */
  seq: number;
  /** where go-to-source jumps: the call site (for a synthesised object, the user's verb call); null when the record has no frame */
  source: DeclSource | null;
}

export type ArtifactKind = "geometry" | "model" | "results";

export type ZoneStatus =
  | { status: "absent" }
  | { status: "loading" }
  | { status: "ready"; version: string }
  /** `accepted` is the version range the reader opens; `newer` says the file is ahead of this app, not behind it */
  | { status: "refused"; version: string; accepted: string; newer: boolean; reason: string }
  /** the zone's version is fine but its content is broken (the reader's message names the HDF5 path) */
  | { status: "malformed"; reason: string };

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
  /** `/meta/session_id`: the key that pairs a model with its geometry sibling; null when the file has none */
  sessionId: string | null;
}

/** One row of the legend: a colour group, in display order. */
export interface LegendEntry {
  /** the group's decl path, or null for the two synthetic rows */
  decl: DeclPath | null;
  name: string;
  color: readonly [number, number, number];
  elements: number;
  /** a second cue beside the colour: stripes on the chip of a group whose slot is in the second ring (the same hue as an office colour, one lightness step away; taken when no free office colour contrasts with the group's neighbours) */
  cue: "stripe" | "stripe2" | null;
  /** the palette slot the group was assigned (state/palette.ts); null for the synthetic rows */
  slot: number | null;
}

/** What colours the model: its physical groups, or the structural role each element's file records. */
export type ColourBy = "group" | "role";

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
  /** legend rows whose elements share a node (each pair once, a < b): what the slot assignment keeps apart */
  adjacency: [number, number][];
  center: readonly [number, number, number];
  radius: number;
  counts: { lineCells: number; faceCells: number; solidCells: number; opsOnly: number; points: number };
  warnings: string[];
}

/** The render derivation of the geometry artifact (the `/geometry` sibling, V2a spec). */
export interface GeometryInfo {
  path: string;
  /** `/meta/session_id` of the geometry file; the pairing rule compares it with the model's */
  sessionId: string | null;
  source: "mesh" | "temp_mesh";
  status: "ok" | "partial";
  /** 6 floats per curve segment, 9 per surface triangle, 3 per point */
  curvePositions: BlobRef;
  surfacePositions: BlobRef;
  pointPositions: BlobRef;
  counts: { points: number; curves: number; surfaces: number; volumes: number };
  center: readonly [number, number, number];
  radius: number;
}

export type PhaseKey =
  | { kind: "geometry" }
  | { kind: "mesh" }
  | { kind: "stage"; index: number }
  | { kind: "results"; step: number };

/** What the user clicked: the declaration, the hit point, and the cell's node tags as the viewport read them from the connectivity blob. */
export interface Pick {
  decl: DeclPath;
  at: readonly [number, number, number] | null;
  nodes: readonly number[] | null;
}

/** One `/elements/{alias}` block of the neutral zone, its connectivity as a blob. */
export interface BlockInfo {
  alias: string;
  npe: number;
  /** row-major (count, npe) node tags */
  connectivity: BlobRef;
}

export type WindowKind = "stages" | "analysis" | "patterns" | "recorders" | "materials";

export interface Layout {
  open: WindowKind[];
}

export interface State {
  artifacts: Record<ArtifactKind, ArtifactInfo | null>;
  /** keyed by declaration path; a null-prototype record, so no path collides with `Object.prototype` */
  decls: Record<DeclPath, Decl>;
  /** user names (/opensees/names, /labels, /physical_groups) to the declarations that carry them; null-prototype */
  names: Record<string, DeclPath[]>;
  /** the neutral-zone element blocks, in file order (`ElementFacts.cell.block` indexes it) */
  blocks: BlockInfo[];
  mesh: MeshInfo | null;
  phase: { axis: PhaseKey[]; at: PhaseKey | null };
  selection: { decls: DeclPath[]; picks: Pick[] };
  hover: Pick | DeclPath | null;
  visibility: { hidden: DeclPath[]; edges: boolean; opacity: number; colourBy: ColourBy };
  inspector: { pinned: DeclPath[] };
  windows: Layout;
  /** reserved for the D3 ADR; always empty */
  overrides: Record<DeclPath, never>;
  /** the geometry sibling as read; drawn only when it pairs with the model (selectors.geometryPairing) */
  geometry: GeometryInfo | null;
  /** the model file's /provenance records in capture order; [] when it has none (selectors.sourcesOf lists them) */
  provenance: ProvenanceEntry[];
  /**
   * go-to-source: the latest request (the effects act on a new `seq`) and the
   * latest answer. `seq` counts requests for the whole session and is never
   * reset (a model re-read clears `request` and `last`, not the count), so a
   * request after a re-read can never reuse a seq the effects already saw.
   */
  source: {
    seq: number;
    request: { decl: DeclPath; seq: number } | null;
    last: { decl: DeclPath; ok: boolean; reason: string | null } | null;
  };
  /** view requests: `frameSeq` counts `frameSelection` events; the frame effect acts on each change */
  view: { frameSeq: number };
}

/** The payload of `fileLoaded` for the model artifact: what the loader derived from the file. */
export interface ModelLoad {
  info: ArtifactInfo;
  decls: Record<DeclPath, Decl>;
  names: Record<string, DeclPath[]>;
  blocks: BlockInfo[];
  mesh: MeshInfo;
  /** the /provenance records, in capture order; [] when the file has none or the zone was not read */
  provenance: ProvenanceEntry[];
}

/** The payload of `fileLoaded` for the geometry artifact. */
export interface GeometryLoad {
  info: ArtifactInfo;
  geometry: GeometryInfo;
}

/**
 * Decision 17's twenty events, plus `fileClosed` (added by V2e review: an
 * `onOpen` set that names no results file closes the results artifact), and
 * V2f's go-to-source pair: `requestSource` (a panel asks; the effects act)
 * and `sourceResult` (the effects answer), and `setLoaded`: the model and its
 * geometry sibling re-read together after a D1 re-run, applied as one step so
 * no state pairs a new file with an old one.
 */
export type Event =
  | { type: "fileOpened"; artifact: ArtifactKind; path: string }
  | { type: "setLoaded"; model: ModelLoad | null; geometry: GeometryLoad | null }
  | { type: "fileLoaded"; artifact: "model"; load: ModelLoad }
  | { type: "fileLoaded"; artifact: "geometry"; load: GeometryLoad }
  | { type: "fileFailed"; artifact: ArtifactKind; path: string; error: string }
  | { type: "fileClosed"; artifact: ArtifactKind }
  | { type: "fileChanged"; path: string }
  | { type: "zoneRefused"; artifact: ArtifactKind; zone: string; version: string; accepted: string; newer: boolean; reason: string }
  | { type: "select"; pick: Pick }
  | { type: "selectAdd"; pick: Pick }
  | { type: "clearSelection" }
  | { type: "hover"; at: Pick | DeclPath | null }
  | { type: "setHidden"; decls: DeclPath[]; hidden: boolean }
  | { type: "isolate"; decls: DeclPath[] }
  | { type: "showAll" }
  | { type: "setEdges"; on: boolean }
  | { type: "setOpacity"; value: number }
  | { type: "setColourBy"; by: ColourBy }
  | { type: "setPhase"; at: PhaseKey }
  | { type: "setResultStep"; step: number }
  | { type: "inspectorPin"; decl: DeclPath }
  | { type: "inspectorUnpin"; decl: DeclPath }
  | { type: "openWindow"; window: WindowKind }
  | { type: "closeWindow"; window: WindowKind }
  | { type: "requestSource"; decl: DeclPath }
  | { type: "sourceResult"; decl: DeclPath; ok: boolean; reason: string | null }
  /** frame the selection, or the whole model when nothing is selected (`F`, or any panel); the frame effect moves the camera */
  | { type: "frameSelection" };

export type EventType = Event["type"];
