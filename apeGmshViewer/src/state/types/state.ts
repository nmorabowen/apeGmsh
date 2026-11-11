// The store's root state: one field per slice (types/<slice>.ts), plus the
// fields no single slice owns. src/state/types.ts re-exports it.

import type { Definitions, NeutralLoads, SequenceZone } from "../../model/types.ts";
import type { ThemeMode } from "../../theme/tokens.ts";
import type { CameraState } from "./camera.ts";
import type { ClipState } from "./clip.ts";
import type { ContourState } from "./contour.ts";
import type { DeformState } from "./deform.ts";
import type { LineDiagramState } from "./lineDiagram.ts";
import type { ArtifactInfo, ArtifactKind, BlockInfo, Decl, DeclPath, GeometryInfo, Lineage, MeshInfo, PhaseKey, Pick, ProvenanceEntry, SpatialNdm, StageElements, UnitSystemId } from "./model.ts";
import type { OffersState } from "./offers.ts";
import type { LoadsState, PrincipalState, ReactionsState, SpringsState, SupportsState, VectorsState } from "./overlays.ts";
import type { ResultStepStatus, ResultsInfo } from "./results.ts";
import type { CombinationRead, UserFileState, UserResultRead } from "./user.ts";
import type { VariantsState } from "./variants.ts";
import type { Visibility } from "./visibility.ts";
import type { Layout, SearchState, SpotlightState } from "./windows.ts";

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
  visibility: Visibility;
  /** The node markers, apart from `visibility`: what watches that (the hidden set, the banner) need not wake for a size. */
  nodes: {
    on: boolean;
    /** in pixels; null until the effects choose one (state/nodes.ts `smartNodeSize`) */
    size: number | null;
    /** true while the size is the chosen one, false once the user sets it */
    auto: boolean;
  };
  /** The Gauss point and beam station markers (state/gaussPoints.ts), with the same size rules as the nodes. */
  gaussPoints: {
    on: boolean;
    size: number | null;
    auto: boolean;
    /** what the viewport placed on the stage `stage` (its id), and what it could not place */
    measured: { stage: string; count: number; refused: { reason: string; elements: number }[]; notes: string[] } | null;
  };
  inspector: { pinned: DeclPath[] };
  windows: Layout;
  /** reserved for the D3 ADR; always empty */
  overrides: Record<DeclPath, never>;
  /** the geometry sibling as read; drawn only when it pairs with the model (selectors.geometryPairing) */
  geometry: GeometryInfo | null;
  /** the model file's /provenance records in capture order; [] when it has none (selectors.sourcesOf lists them) */
  provenance: ProvenanceEntry[];
  /** the records carry `records/origin` (1.1.0 on); false when they do not, so none is marked synthesised */
  provenanceOrigin: boolean;
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
  /** the model's definition data (decls, decl_params, analysis, series, patterns, recorders, stages); null with no model or no /opensees */
  definitions: Definitions | null;
  /** the model's neutral `/loads/{nodal,element,sp}/{case}` tables; null when the file has none */
  neutralLoads: NeutralLoads | null;
  /** the model's root /sequence zone; `absent` when the file has none (or no model is loaded) */
  sequence: SequenceZone;
  /**
   * The selected stage: a row index into `/sequence/stages` (0-based, file
   * order), or null. A selection of its own, not a phase: the phase axis is
   * what the viewport draws (D7), and a stage is not drawable yet. A re-read
   * of the same file keeps it only while that row still names the same stage
   * (selectors.stageIdentity); any other load clears it.
   */
  stage: number | null;
  /** the model's `/meta/lineage`; null with no model, or a model file without one */
  lineage: Lineage | null;
  /** the element tags of the model's `/sequence` element change rows (state/activation.ts reads them) */
  stageElements: StageElements;
  /** the results artifact's metadata (V4b); null until a results file is read. Paired with the model by selectors.resultsPairing */
  results: ResultsInfo | null;
  /** the fetch of the current results step; null when the phase is not a results step */
  resultStep: ResultStepStatus | null;
  /** ADR 0130: the set's `.user.h5` (the phase bar's Spectra and User Combinations, the User Results group); null when the set has none */
  user: UserFileState | null;
  /** VU7b-2: the read of the User Combination shown, for the quantity picked; null off a combination, and after the user file is read again */
  combination: CombinationRead | null;
  /** VU16: the read of the user result picked (`contour.source` `user`); null with none, and after the user file is read again */
  userResult: UserResultRead | null;
  /** VU10 (ADR 0132 D9): the study's variants, the compare set and the difference; it lasts across the window set's loads */
  variants: VariantsState;
  /** V4c1: the nodal contour (shown when `visibility.colourBy` is `contour`) */
  contour: ContourState;
  /** V4d: the model's spatial dimension (`/meta/ndm`), or why it is not known */
  ndm: SpatialNdm;
  /** the loaded model's declared consistent unit system (`/meta/@units`), or null when undeclared */
  units: UnitSystemId | null;
  /** V4d: the deformed shape */
  deform: DeformState;
  /** V4e1: the section planes */
  clip: ClipState;
  /** 1-D beam line diagram state */
  lineDiagram: LineDiagramState;
  /** 3-D structural loading and prescribed-SP visualization state */
  loads: LoadsState;
  reactions: ReactionsState;
  principal: PrincipalState;
  vectors: VectorsState;
  springs: SpringsState;
  supports: SupportsState;
  /** Spotlight (`Ctrl+K`) command and search palette */
  spotlight: SpotlightState;
  /** Search window query builder and text filter */
  search: SearchState;
  /** VA1 (ADR 0128 D1): the camera the viewport shows, and the latest request to move it */
  camera: CameraState;
  /** VA1: the theme of the page and the viewport (an effect applies it with theme/tokens.ts `applyTheme`) */
  theme: ThemeMode;
  /** VA5 (ADR 0128 D5): the views agents offered, and the way back from one the human applied */
  offers: OffersState;
}
