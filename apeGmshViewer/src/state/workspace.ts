// What a new workspace set takes from the selection (VU9a, ADR 0132 D3).
//
// A set's members are OpenSees tags (`id_space` `opensees`). An element set
// takes every `/opensees/element_meta` tag the selected elements carry (a
// cell can carry more than one; a cell that carries none is no OpenSees
// element and is left out, said by a note). A node set takes the node kind's
// selection (ADR 0138 D2): a node's tag is its FEM id (the bridge emits every
// node under it). Main takes each member's coordinates from the model file,
// never from here.
//
// Pure: the node table comes in from the BlobStore (the effects pass it).

import { selectedElementsOf } from "./search.ts";
import type { SetKind, State, WorkspaceSaveIn } from "./types.ts";

/** A save another program blocked: the typed refusal of HG9 rule 2 (main/workspace.ts answers it; the panel shows it). */
export function heldRefusal(file: string): Extract<WorkspaceSaveIn, { ok: false }> {
  return {
    ok: false,
    reason: "held",
    detail: `${file} is open in another program (for example a Python session), which keeps this app from replacing it. Close it there and save again. Nothing was replaced.`,
    file,
    held: true,
  };
}

/** The model's node ids and coordinates (3 per node), as `MeshInfo.nodes` holds them. */
export interface NodeTable {
  ids: ArrayLike<number>;
  xyz: ArrayLike<number>;
}

export type Members = { ok: true; ids: number[]; notes: string[] } | { ok: false; reason: string };

function uniq(ids: number[]): number[] {
  return [...new Set(ids)];
}

/** The OpenSees tags of the selected elements, in selection order. */
export function elementMembersOf(s: State): Members {
  const decls = selectedElementsOf(s);
  if (decls.length === 0) return { ok: false, reason: "select elements first (click one; Shift+click adds)" };
  const ids: number[] = [];
  let bare = 0;
  for (const p of decls) {
    const metas = s.decls[p]?.element?.metas ?? [];
    if (metas.length === 0) bare++;
    for (const m of metas) ids.push(m.tag);
  }
  const notes = bare ? [`${bare} selected cell${bare === 1 ? " is" : "s are"} no OpenSees element (no /opensees/element_meta row) and ${bare === 1 ? "is" : "are"} left out`] : [];
  if (ids.length === 0) return { ok: false, reason: "no selected element is an OpenSees element: an element set holds OpenSees tags" };
  return { ok: true, ids: uniq(ids), notes };
}

/**
 * The node kind's tags, in pick order (ADR 0138 D2). A click picks the node
 * where it is drawn, the deformed shape included, so nothing here reads where
 * a click landed. With the node table, a tag the model lacks is left out and
 * said by a note.
 */
export function nodeMembersOf(s: State, nodes: NodeTable | null): Members {
  if (s.selection.nodes.length === 0) return { ok: false, reason: "pick nodes first: choose Nodes, then click near a node (Shift+click adds)" };
  let ids = uniq(s.selection.nodes);
  const notes: string[] = [];
  if (nodes !== null) {
    const known = new Set<number>();
    for (let i = 0; i < nodes.ids.length; i++) known.add(nodes.ids[i]!);
    const missing = ids.filter((t) => !known.has(t));
    if (missing.length) {
      ids = ids.filter((t) => known.has(t));
      notes.push(`${missing.length} selected node${missing.length === 1 ? " is" : "s are"} not in the model's node table and left out`);
    }
  }
  if (ids.length === 0) return { ok: false, reason: "no selected node is in the model's node table" };
  return { ok: true, ids, notes };
}

/** The set kind a new set takes: the active selection kind (ADR 0138 D2, D12). */
export function setKindOf(s: State): SetKind {
  return s.selection.active === "node" ? "nodes" : "elements";
}

export function membersOf(s: State, kind: SetKind, nodes: NodeTable | null): Members {
  return kind === "elements" ? elementMembersOf(s) : nodeMembersOf(s, nodes);
}

/** What the sets panel says the selection gives: the effects' preview of the members (tags), else the members now, or why nothing. */
export function pickSummaryOf(s: State): { count: number; reason: string | null; ids: readonly number[] | null } {
  const p = s.workspace.preview, kind = setKindOf(s);
  if (p !== null && p.kind === kind) return "ids" in p ? { count: p.ids.length, reason: null, ids: p.ids } : { count: 0, reason: p.reason, ids: null };
  const m = membersOf(s, kind, null);
  return m.ok ? { count: m.ids.length, reason: null, ids: m.ids } : { count: 0, reason: m.reason, ids: null };
}
