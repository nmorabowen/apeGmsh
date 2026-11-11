// The workspace slice of the reducer (state/reduce.ts delegates here): VU9a's
// read of the window set's `<stem>.workspace.h5`, the pick kind, and the save
// in flight (ADR 0132 D6, D7). It writes `workspace` only.

import type { WorkspaceEvent } from "../types/events/workspace.ts";
import { SET_KINDS, type WorkspaceIn, type WorkspaceState } from "../types/workspace.ts";

/** VU9a: nothing read, node picks, no save asked. */
export const initialWorkspace: WorkspaceState = { read: null, reading: false, kind: "nodes", seq: 0, request: null, saving: null, notice: null, aside: null, ownSaveId: null };

const STATUSES = ["none", "absent", "unreadable", "refused", "ready"];

/** The file a read names, or null. */
function fileOf(r: WorkspaceIn | null): string | null {
  return r !== null && "file" in r ? r.file : null;
}

/** Why the page may not ask for a save now, or null: no file to write, or one this app must not write over. */
export function cannotSave(w: WorkspaceState): string | null {
  const r = w.read;
  if (r === null || r.status === "none") return r?.status === "none" ? r.reason : "no set is open";
  if (r.status === "refused") return `${r.file} is refused (${r.reason}): ${r.detail}`;
  if (r.status === "ready" && r.readOnly !== null) return r.readOnly;
  if (r.status === "ready" && r.broken.length) return `${r.file}: /${r.broken[0]!.table} cannot be copied verbatim (${r.broken[0]!.reason})`;
  if (w.saving !== null) return "a save is running";
  return null;
}

/** The workspace after `e`: the same object when nothing changed. */
export function reduceWorkspace(w: WorkspaceState, e: WorkspaceEvent): WorkspaceState {
  switch (e.type) {
    case "workspaceReading":
      return w.reading ? w : { ...w, reading: true };
    case "workspaceRead": {
      const r = e.read;
      if (r !== null && (typeof r !== "object" || !STATUSES.includes((r as { status: unknown }).status as string))) {
        throw new RangeError(`workspaceRead: ${JSON.stringify(r)} is not a workspace read`);
      }
      // The watcher's report of this window's own save: the definitions on screen are that save's.
      const own = r?.status === "ready" && w.ownSaveId !== null && r.saveId === w.ownSaveId && w.read?.status === "ready" && w.read.saveId === r.saveId;
      if (own) return w.reading ? { ...w, reading: false } : w;
      // Another set's file: the last one's notice and aside banner leave with it.
      const same = fileOf(r) === fileOf(w.read);
      return { ...w, read: r, reading: false, notice: same ? w.notice : null, aside: same ? w.aside : null, ownSaveId: same ? w.ownSaveId : null };
    }
    case "setWorkspaceKind":
      if (!(SET_KINDS as readonly string[]).includes(e.kind)) throw new RangeError(`setWorkspaceKind: ${JSON.stringify(e.kind)} is not nodes or elements`);
      return e.kind === w.kind ? w : { ...w, kind: e.kind };
    case "workspaceSave": {
      const why = cannotSave(w);
      if (why !== null) throw new Error(`workspaceSave: ${why}`);
      const seq = w.seq + 1;
      return { ...w, seq, request: { seq, edit: e.edit }, saving: seq, notice: null };
    }
    case "workspaceSaved": {
      // Only the save in flight: a late answer for another is dropped.
      if (w.saving !== e.seq) return w;
      const a = e.answer;
      if (!a.ok) return { ...w, saving: null, notice: { reason: a.reason, detail: a.detail, file: a.file, held: a.held } };
      return { ...w, saving: null, notice: null, read: a.read, ownSaveId: a.read.status === "ready" ? a.read.saveId : null, aside: a.aside ?? w.aside };
    }
    case "dismissWorkspaceAside":
      return w.aside === null ? w : { ...w, aside: null };
    default: {
      const unknown: never = e;
      throw new Error(`reduceWorkspace: unknown event ${JSON.stringify(unknown)}`);
    }
  }
}
