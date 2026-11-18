// The model slice's events: its part of the store's Event union (types/event.ts).
// Not re-exported by src/state/types.ts, which exports only the whole union.

import type { ThemeMode } from "../../../theme/tokens.ts";
import type { ArtifactKind, ColourBy, DeclPath, GeometryLoad, ModelLoad, PhaseKey, Pick } from "../model.ts";

export type ModelEvent =
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
  | { type: "selectDecls"; decls: readonly DeclPath[]; add?: boolean }
  | { type: "clearSelection" }
  | { type: "hover"; at: Pick | DeclPath | null }
  | { type: "setHidden"; decls: readonly DeclPath[]; hidden: boolean }
  | { type: "isolate"; decls: readonly DeclPath[] }
  | { type: "showAll" }
  | { type: "setEdges"; on: boolean }
  | { type: "setOpacity"; value: number }
  | { type: "setColourBy"; by: ColourBy }
  | { type: "toggleExtrudeFrames"; on?: boolean }
  | { type: "toggleShowOffsets"; on?: boolean }
  /** turning them on while the size is automatic chooses it again */
  | { type: "setNodes"; on: boolean }
  /** a number from the user; `auto` from the effects' choice; null asks for a new choice */
  | { type: "setNodeSize"; size: number | null; auto?: boolean }
  /** the Gauss point markers, as setNodes and setNodeSize */
  | { type: "setGaussPoints"; on: boolean }
  | { type: "setGaussSize"; size: number | null; auto?: boolean }
  /** dispatched by the viewport once it has placed the Gauss markers of a stage */
  | { type: "gaussMeasured"; stage: string; count: number; refused: { reason: string; elements: number }[]; notes: string[] }
  | { type: "setPhase"; at: PhaseKey }
  | { type: "inspectorPin"; decl: DeclPath }
  | { type: "inspectorUnpin"; decl: DeclPath }
  | { type: "requestSource"; decl: DeclPath }
  | { type: "sourceResult"; decl: DeclPath; ok: boolean; reason: string | null }
  /** frame the selection, or the whole model when nothing is selected (`F`, or any panel); the frame effect moves the camera */
  | { type: "frameSelection" }
  /** VA1: the page and viewport theme */
  | { type: "setTheme"; theme: ThemeMode }
  /** select a `/sequence/stages` row by index (a stage card click), or clear the stage selection with null */
  | { type: "selectStage"; index: number | null };
