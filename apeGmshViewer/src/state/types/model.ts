// The model slice: declarations, the mesh and geometry, provenance, the artifacts, the selection's picks, and what a model load carries.
// One slice of the store's types; src/state/types.ts re-exports it, and the
// panels import it from there. A slice file imports sibling files, never the barrel.

import type { Field } from "../../chain/resolve.ts";
import type { Definitions, NeutralLoads, OpsFamily, Param, SectionProfile, SequenceZone } from "../../model/types.ts";

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
  /** for a 3-D geomTransf object: full `(emittedTag, vecxz)` rows from `per_element_vecxz` and `per_element_emitted_tag` (untruncated) */
  vecxz?: readonly { readonly tag: number; readonly vecxz: readonly [number, number, number] }[];
  /** for a geomTransf object with `-jntOffset`: full `(emittedTag, dI, dJ)` rows in global coordinates */
  jntOffset?: readonly {
    readonly tag: number;
    readonly dI: readonly [number, number, number];
    readonly dJ: readonly [number, number, number];
  }[];
  /** for a section object: 2-D cross-section profile loops in local (y, z) coordinates */
  sectionProfile?: SectionProfile;
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
  /** the file's sha256 when it was captured; "" when the writer had none to take */
  sha256: string;
  /** the source can be opened; false for a pseudo-file, or a path with no digest that is not on disk ("source not recorded") */
  recorded: boolean;
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

/**
 * What colours the model: its physical groups, the structural role each
 * element's file records, or (V4c1) a nodal results field at the current step.
 */
export type ColourBy = "group" | "role" | "contour";

/** Pre-computed 3-D cross-section prism extrusion and rigid joint-offset buffers (stored in BlobStore). */
export interface FrameExtrusionBlobs {
  readonly triPositions: BlobRef;
  readonly triNodes: BlobRef;
  readonly triSegments: BlobRef;
  readonly triElementIds: BlobRef;
  readonly triElement: BlobRef;
  readonly triGroup: BlobRef;
  readonly edgePositions: BlobRef;
  readonly edgeNodes: BlobRef;
  readonly edgeSegments: BlobRef;
  readonly extrudedSegmentMask: BlobRef;
  readonly clearLinePositions: BlobRef;
  readonly offsetLinePositions: BlobRef;
  readonly offsetLineNodes: BlobRef;
  readonly offsetLineSegments: BlobRef;
  readonly offsetJointPositions: BlobRef;
  readonly offsetJointNodes: BlobRef;
  readonly offsetJointSegments: BlobRef;
  readonly hasExtrudableFrames: boolean;
  readonly hasJointOffsets: boolean;
  readonly extrusionSkippedUnoriented: number;
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
  /** legend rows whose elements share a node (each pair once, a < b): what the slot assignment keeps apart */
  adjacency: [number, number][];
  center: readonly [number, number, number];
  radius: number;
  counts: { lineCells: number; faceCells: number; solidCells: number; opsOnly: number; points: number };
  warnings: string[];
  /**
   * V4c1: the node tag at every drawn vertex (f64: 2 per segment, 3 per
   * triangle), as mesh/build.ts read it from the connectivity; a contour
   * colours through it. V4d adds `edge` (2 per outline edge), which the
   * deformed shape moves the outline edges through.
   */
  vertexNodes: { line: BlobRef; tri: BlobRef; edge: BlobRef };
  /**
   * V4e3: every node of the model, interior ones too (the cut face of a solid
   * meets nodes no drawn face has): `ids` (f64) and `xyz` (f64, 3 per node),
   * in `/nodes` order. Put by the effects' model read; absent when the mesh
   * was built without it, and then the cut face says so instead of a cap.
   */
  nodes?: { ids: BlobRef; xyz: BlobRef };
  /** 3-D cross-section prism extrusion and rigid joint-offset geometry blobs for 1-D frame elements. */
  frameExtrusion?: FrameExtrusionBlobs;
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

/**
 * A position on the phase axis. A results position names a stage of the
 * results file (`stage`, an index into `State.results.stages`) and a step of
 * it; the axis holds one results entry per stage (at step 0), and two results
 * keys are the same phase when they name the same stage (the step is the
 * position within it, moved by `setResultStep`).
 */
export type PhaseKey =
  | { kind: "geometry" }
  | { kind: "mesh" }
  | { kind: "stage"; index: number }
  | { kind: "results"; stage: number; step: number }
  /**
   * VU7b-2 (ADR 0130 D1): a User Combination of the set's `.user.h5`, by name,
   * shown as the current state. It is never on the axis: the phase bar's User
   * Combinations family picks it, and it stays shown while the user file still
   * lists it (reduce.ts `withPhase`). Off the results phases, a stage's layers
   * (a step, a deformed shape, a line diagram) are not drawn.
   */
  | { kind: "combination"; name: string };

/** `/meta/lineage` of a model.h5 or a results.h5: the ADR 0021 hashes, null where the writer stamped none. */
export interface Lineage {
  femHash: string | null;
  modelHash: string | null;
  resultsHash: string | null;
}

/**
 * The element tags a model's `/sequence` element change rows name, read from
 * each row's `source` dataset (`/opensees/stages/stage_NNN/owned_element_ids`,
 * `.../remove_element`). `error` when a source could not be read, or its
 * length disagrees with its rows' counts; activation is then not applied.
 */
export type StageElements =
  | { status: "none" }
  | { status: "ready"; sets: Record<string, number[]> }
  | { status: "error"; reason: string };

/** The seven consistent unit systems of `baseUnits.systems` (state/units.ts). */
export type UnitSystemId = "kN_m_s" | "tf_m_s" | "kgf_m_s" | "N_mm_s" | "N_m_s" | "kip_in_s" | "dyne_cm_s";

/**
 * The model's spatial dimension, as the deformed shape needs it (V4d): `/meta/ndm`
 * read from neutral 2.34.0 on, where it is the `ops.model(ndm=)` dimension
 * (h5-schema.md, the `/meta` table). Null, with the reason, when the file
 * declares none (`0`: a broker-only file) or stamps the mesh dimension (an
 * older file): then nothing is taken to be 2-D.
 */
export type SpatialNdm = { ndm: 2 | 3; source: string } | { ndm: null; reason: string };

/** What the user clicked: the declaration, the hit point, and the cell's node tags as the viewport read them from the connectivity blob. */
export interface Pick {
  decl: DeclPath;
  at: readonly [number, number, number] | null;
  nodes: readonly number[] | null;
}

/** One `/elements/{alias}` block of the neutral zone, its connectivity as a blob. */
export interface BlockInfo {
  alias: string;
  /** V4c2b: the block's gmsh element-type code (`/elements/{alias}@code`): the Gauss-to-node extrapolation's shape functions */
  code: number;
  npe: number;
  /** row-major (count, npe) node tags */
  connectivity: BlobRef;
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
  /** the zone carries `records/origin` (false with no zone) */
  provenanceOrigin: boolean;
  /** the opensees zone's definition data; null when the file has no /opensees */
  definitions: Definitions | null;
  /** neutral `/loads/{nodal,element,sp}/{case}` tables; null when the file has none */
  neutralLoads?: NeutralLoads | null;
  sequence: SequenceZone;
  lineage: Lineage | null;
  stageElements: StageElements;
  /** V4d: the model's spatial dimension; absent means not read (nothing is taken to be 2-D) */
  ndm?: SpatialNdm;
  /** the model's declared consistent unit system (`/meta/@units`), or null when undeclared */
  units?: UnitSystemId | null;
}

/** The payload of `fileLoaded` for the geometry artifact. */
export interface GeometryLoad {
  info: ArtifactInfo;
  geometry: GeometryInfo;
}
