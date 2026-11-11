// The workspace slice's events: its part of the store's Event union (types/event.ts).
// Not re-exported by src/state/types.ts, which exports only the whole union.

import type { SetKind, WorkspaceIn, WorkspacePageEdit, WorkspaceSaveIn, WorkspaceState } from "../workspace.ts";

export type WorkspaceEvent =
  /** VU9a (the effects): a read of the window set's workspace started */
  | { type: "workspaceReading" }
  /** VU9a (the effects): main's read of the window set's workspace; null with no window set */
  | { type: "workspaceRead"; read: WorkspaceIn | null }
  /** VU9a: what a new set takes from the selection (the sets panel's toggle) */
  | { type: "setWorkspaceKind"; kind: SetKind }
  /** VU9a: the sets panel asks for one edit; the effects carry it to main (a create takes its members from the selection) */
  | { type: "workspaceSave"; edit: WorkspacePageEdit }
  /** VU9a (the effects): main answered save `seq`; an answer for another save is dropped */
  | { type: "workspaceSaved"; seq: number; answer: WorkspaceSaveIn }
  /** VU9a (the effects): the members a save would take from the selection now (state/workspace.ts `membersOf`) */
  | { type: "workspacePreview"; preview: NonNullable<WorkspaceState["preview"]> }
  /** VU9a: the banner naming the file an unreadable workspace was renamed aside to is closed */
  | { type: "dismissWorkspaceAside" };
