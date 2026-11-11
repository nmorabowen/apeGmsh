// The results slice: a results file's stages and metadata, and the fetch of the step shown.
// One slice of the store's types; src/state/types.ts re-exports it, and the
// panels import it from there. A slice file imports sibling files, never the barrel.

import type { ProgramDecls } from "../../model/types.ts";
import type { LineStationComponent } from "./lineDiagram.ts";
import type { BlobRef, Lineage, UnitSystemId } from "./model.ts";

/** One `/stages/{id}` of a results file, as the store holds it: metadata only, never a component array. */
export interface ResultsStage {
  id: string;
  /**
   * The label every site shows: the user's stage name (`begin_stage(name=)`),
   * `mode_<k>` for a mode stage, or a `.ladruno`'s display name
   * (`INFO/PROVENANCE@stage_display_names`, a `stage_names=` rename). Never
   * matched against the model: `programName` is.
   */
  name: string;
  /**
   * The program's stage name (W4d #2843): what every match against the model
   * uses (`/sequence` activation, the loads overlay, span loads, a compare
   * set). Equal to `name` unless a `.ladruno` carries display names. "" is a
   * name too (and pairs with nothing).
   */
  programName: string;
  kind: "static" | "transient" | "mode";
  /** the stage's `time` vector: step k is time[k] */
  time: number[];
  /** nodal components (`displacement_x`, ...) */
  nodal: string[];
  /** V4c1: nodal components the reader refused one by one (value columns that disagree with `_ids`: a partial write) */
  refusedNodal: { component: string; reason: string }[];
  /** V4c1: the stage's nodal `_ids` (f64), put in the BlobStore by the effects; absent before that, or with no nodal results */
  nodeIds?: BlobRef;
  /** gauss groups and their components (V4c2a: element contours read them) */
  gauss: GaussGroup[];
  /** element categories the reader does not read */
  unread: string[];
  /** .ladruno INFO/PROVENANCE@element_index_space ("fem" or "ops_tag"), or null */
  elementIndexSpace?: string | null;
  /** 1-D beam line-station components recorded in this stage (.ladruno) */
  lineStations?: LineStationComponent[];
  /** zero-length springs this stage records a global `force` for (.ladruno); absent or 0 for none */
  springs?: number;
  /** a `.ladruno` mode stage's eigen scalars, as the file records them (its LAMBDA); absent elsewhere */
  mode?: ModeFacts;
}

/**
 * A mode's eigen scalars (reader/ladruno.ts `LadrunoMode`): the file's
 * `LAMBDA` (omega^2) and the frequency and period derived from it, both 0 for
 * a non-positive eigenvalue (no frequency or period).
 */
export interface ModeFacts {
  /** 1-based, as the stage name `mode_<k>` */
  index: number;
  eigenvalue: number;
  frequencyHz: number;
  periodS: number;
}

/** One `gauss_points` group of a results stage (V4c2a): metadata, and its element ids as a blob ref. */
export interface GaussGroup {
  id: string;
  components: string[];
  /** components refused one by one (a partial write), or every component of a group with no Gauss point */
  refused: { component: string; reason: string }[];
  nElements: number;
  nGauss: number;
  /** the group carries `_local_axes_quaternion` (shells) */
  localAxes: boolean;
  /** the group's `_element_index` (f64) in the BlobStore: present only for a verified file, put there by the effects */
  elementIds?: BlobRef;
  /** V4c2b: `_natural_coords`, one parent-space row per Gauss point; null when the file's dataset is not floating point */
  naturalCoords: number[][] | null;
}

/** Whether a results file's `_element_index` rows are FEM element ids: only its root stamp says so (#1768). */
export type ElementIndexVerdict = { verified: true } | { verified: false; reason: string };

/** The results artifact as read (V4b): what the store keeps of a results.h5. Step arrays stay in main. */
export interface ResultsInfo {
  path: string;
  version: string;
  sourceType: string | null;
  /**
   * null: the file has no /meta/lineage (it cannot be paired by content). A
   * `.ladruno`'s lineage is its INFO/PROVENANCE `model_hash` and `fem_hash`,
   * null unless both are stamped (#2127)
   */
  lineage: Lineage | null;
  /** .ladruno: why `lineage` is null (main/results.ts, from reader/ladruno.ts), else null; absent for a native file */
  lineageReason?: string | null;
  /** why the file looks unfinished (no results_hash), or null; it is read as it stands */
  partial: string | null;
  stages: ResultsStage[];
  unavailableZones: string[];
  /**
   * V4h: the embedded opensees zone's version stamp (unhashed), null when the
   * file embeds no /opensees. The pairing rule reads it to name the 2.31.0
   * model_hash change (state/results.ts); null keeps the plain message.
   */
  openseesVersion: string | null;
  /**
   * V4h: the embedded `/opensees/decls@program_decl` marker ("absent" when the
   * zone has no decls group or no marker); null with no /opensees. It, not the
   * version stamp, says whether the results' model_hash includes declaration ids.
   */
  programDecl: string | null;
  /**
   * #2189: the embedded `/opensees/program`'s side-channel calls and whether
   * each carries a declaration (reader/read.ts `programDecls`). The pairing
   * names the opensees 2.32.0 model_hash change from it (state/results.ts).
   * null (no program, a marker that is not "complete", a .ladruno) or absent
   * (an info built without it) names no cause.
   */
  program?: ProgramDecls | null;
  /**
   * V4c2a: the element-id verdict; element contours draw only for a verified
   * file (#1711/#1752). Absent (a results info built without it) reads as
   * unverified: selectors `elementVerdictOf`, never as verified.
   */
  elementIndex?: ElementIndexVerdict;
  /** Consistent unit system declared in `/model/meta/@units` or `@units`, or null when undeclared. */
  units?: UnitSystemId | null;
  /** .ladruno INFO/PROVENANCE@element_index_space ("fem" or "ops_tag"), or null */
  elementIndexSpace?: string | null;
}

/** The current results step's data, fetched lazily from main (the values themselves stay in the effects). */
export interface ResultStepStatus {
  stage: number;
  step: number;
  status: "loading" | "ready" | "failed";
  error: string | null;
}
