// What a new workspace set takes from the selection (VU9a, ADR 0132 D3).
//
// A set's members are OpenSees tags (`id_space` `opensees`). An element set
// takes every `/opensees/element_meta` tag the selected elements carry (a
// cell can carry more than one; a cell that carries none is no OpenSees
// element and is left out, said by a note). A node set takes, for each click
// on the model, the node of the clicked element nearest the point clicked: a
// node's tag is its FEM id (the bridge emits every node under it). Main takes
// each member's coordinates from the model file, never from here.
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

/** For each click, the node of the clicked element nearest the point clicked, in click order. */
export function nodeMembersOf(s: State, nodes: NodeTable | null): Members {
  const picks = s.selection.picks.filter((p) => p.at !== null && p.nodes !== null && p.nodes.length > 0);
  if (picks.length === 0) return { ok: false, reason: "click the model near each node first (Shift+click adds): each click takes the nearest node of the element clicked" };
  if (nodes === null) return { ok: false, reason: "the model was read without its node table, so no node can be placed" };
  // A click lands where the cell is drawn; with the deformed shape that is not where the node table puts the nodes.
  if (s.deform.on || s.workspace.deformedPicks) {
    return {
      ok: false,
      reason: s.deform.on
        ? "the deformed shape shows: a click lands on the displaced cell, not where its nodes are, so Nodes picks nothing; turn the deformed shape off and click again"
        : "a click was made on the deformed shape: turn it off and click the nodes again",
    };
  }
  const index = new Map<number, number>();
  for (let i = 0; i < nodes.ids.length; i++) if (!index.has(nodes.ids[i]!)) index.set(nodes.ids[i]!, i);
  const ids: number[] = [];
  for (const p of picks) {
    let best: number | null = null;
    let bestD = Infinity;
    for (const n of p.nodes!) {
      const i = index.get(n);
      if (i === undefined) continue;
      const d = Math.hypot(nodes.xyz[3 * i]! - p.at![0], nodes.xyz[3 * i + 1]! - p.at![1], nodes.xyz[3 * i + 2]! - p.at![2]);
      if (d < bestD) [best, bestD] = [n, d];
    }
    if (best !== null) ids.push(best);
  }
  const skipped = s.selection.picks.length - picks.length;
  const notes = skipped ? [`${skipped} selected item${skipped === 1 ? " was" : "s were"} not clicked on the model and ${skipped === 1 ? "names" : "name"} no node`] : [];
  if (ids.length === 0) return { ok: false, reason: "no clicked element has a node the model places" };
  return { ok: true, ids: uniq(ids), notes };
}

export function membersOf(s: State, kind: SetKind, nodes: NodeTable | null): Members {
  return kind === "elements" ? elementMembersOf(s) : nodeMembersOf(s, nodes);
}

/** What the sets panel says the selection gives: the effects' preview of the members (tags), else a count of the clicks, or why nothing. */
export function pickSummaryOf(s: State): { count: number; reason: string | null; ids: readonly number[] | null } {
  const p = s.workspace.preview;
  if (p !== null && p.kind === s.workspace.kind) return "ids" in p ? { count: p.ids.length, reason: null, ids: p.ids } : { count: 0, reason: p.reason, ids: null };
  if (s.workspace.kind === "elements") {
    const m = elementMembersOf(s);
    return m.ok ? { count: m.ids.length, reason: null, ids: m.ids } : { count: 0, reason: m.reason, ids: null };
  }
  if (s.deform.on || s.workspace.deformedPicks) {
    const m = nodeMembersOf(s, { ids: [], xyz: [] });
    if (!m.ok && /deformed/.test(m.reason)) return { count: 0, reason: m.reason, ids: null };
  }
  const n = s.selection.picks.filter((p) => p.at !== null && p.nodes !== null && p.nodes.length > 0).length;
  return n ? { count: n, reason: null, ids: null } : { count: 0, reason: "click the model near each node (Shift+click adds)", ids: null };
}
