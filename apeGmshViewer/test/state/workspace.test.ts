// VU9a (ADR 0132; #2803): the workspace slice, what a set takes from the
// selection, and what the sets panel says. Pure: the store's reducer, the
// member functions of state/workspace.ts and panels/sets.ts `setsLines`.

import assert from "node:assert/strict";
import { test } from "node:test";
import { initialState, reduce } from "../../src/state/reduce.ts";
import { heldRefusal, membersOf } from "../../src/state/workspace.ts";
import { setsLines } from "../../src/panels/sets.ts";
import { noticesOf } from "../../src/state/selectors.ts";
import type { Decl, Event, State, WorkspaceIn } from "../../src/state/types.ts";

const FILE = "C:/runs/frame.workspace.h5";
const verdict = (v: "current" | "moved" | "partial" | "lost", matched: number, total: number) => ({ verdict: v, matched, total, members: Array.from({ length: total }, (_, i) => i < matched), checked: v !== "current", maxDistance: matched ? 0 : null });
const READY: Extract<WorkspaceIn, { status: "ready" }> = {
  status: "ready",
  file: FILE,
  token: "python-bytes",
  version: "1.0.0",
  revision: 4,
  writer: "apeGmsh 1.4.0",
  saveId: "python-save",
  readOnly: null,
  sets: [
    { row: 0, name: "roof", kind: "nodes", ids: [31, 32, 33], verdict: verdict("current", 3, 3) },
    { row: 1, name: "corners", kind: "nodes", ids: [1, 3, 31, 33], verdict: verdict("moved", 4, 4) },
    { row: 2, name: "base", kind: "nodes", ids: [1, 2, 3], verdict: verdict("partial", 2, 3) },
    { row: 3, name: "column line", kind: "nodes", ids: [1, 11, 21, 31], verdict: verdict("lost", 0, 4) },
  ],
  plots: [{ table: "plots", row: 0, name: "roof drift", grammar: "apegmsh-plot/1", body: "{}" }],
  views: [],
  refused: [{ table: "sets", row: 4, name: "twin", reason: "duplicate", detail: "2 sets are named 'twin'" }],
  broken: [],
};

const run = (events: Event[], s: State = initialState) => events.reduce(reduce, s);

test("the panel states every verdict with its reason, the refused row, and the stored plot", () => {
  const lines = setsLines(run([{ type: "workspaceRead", read: READY }]))!;
  assert.deepEqual(lines.sets.map((x) => [x.name, x.chip, x.canAccept]), [
    ["roof", "current", false],
    ["corners", "moved", false],
    ["base", "partial · 2 of 3", true],
    ["column line", "lost · 0 of 4", false],
  ]);
  assert.match(lines.sets[2]!.reason, /2 of 3 members sit at their stored positions; a plot that reads this set refuses/);
  assert.match(lines.sets[3]!.reason, /no member sits at its stored position/);
  assert.deepEqual(lines.refused, ["sets row 4 (twin) refused, duplicate: 2 sets are named 'twin'"]);
  assert.deepEqual(lines.plots, ["roof drift"]);
  assert.equal(lines.canSave, true);
  assert.equal(setsLines(initialState), null, "no room with no set open");
});

test("a save: the request carries the edit; main's answer lands only for the save in flight", () => {
  const s = run([{ type: "workspaceRead", read: READY }, { type: "workspaceSave", edit: { op: "create", name: "picked" } }]);
  assert.deepEqual(s.workspace.request, { seq: 1, edit: { op: "create", name: "picked" }, expect: "python-bytes" }, "the save carries the token of the read it was made on");
  assert.equal(s.workspace.saving, 1);
  assert.throws(() => reduce(s, { type: "workspaceSave", edit: { op: "delete", table: "sets", name: "roof" } }), /a save is running/);
  const late = reduce(s, { type: "workspaceSaved", seq: 7, answer: heldRefusal(FILE) });
  assert.equal(late, s, "an answer for another save is dropped");
  const held = reduce(s, { type: "workspaceSaved", seq: 1, answer: heldRefusal(FILE) });
  assert.equal(held.workspace.saving, null);
  assert.deepEqual(held.workspace.notice, { reason: "held", detail: heldRefusal(FILE).detail, file: FILE, held: true });
  const lines = setsLines(held)!;
  assert.equal(lines.notice?.held, true);
  assert.match(lines.notice!.text, /frame\.workspace\.h5 is open in another program .* Close it there and save again\. Nothing was replaced\./);
});

test("this window's own save, reported back by the watcher, is recognised; another writer's is taken", () => {
  const saved: WorkspaceIn = { ...READY, revision: 5, saveId: "app-save", token: "app-bytes" };
  let s = run([{ type: "workspaceRead", read: READY }, { type: "workspaceSave", edit: { op: "accept", name: "base" } }]);
  s = reduce(s, { type: "workspaceSaved", seq: 1, answer: { ok: true, read: saved, aside: null } });
  assert.equal(s.workspace.ownSaveId, "app-save");
  const again = reduce(s, { type: "workspaceRead", read: { ...saved } });
  assert.equal(again.workspace.read, s.workspace.read, "the watcher's report of the own save keeps the read on screen");
  const other = reduce(s, { type: "workspaceRead", read: { ...READY, revision: 6, saveId: "python-again", token: "python-again-bytes" } });
  assert.equal((other.workspace.read as { revision: number }).revision, 6);
});

test("the aside banner: named once per rename, closed by the user, gone with another set", () => {
  let s = run([{ type: "workspaceRead", read: { status: "unreadable", file: FILE, reason: "not-hdf5", detail: "not HDF5", token: "garbage-bytes" } }, { type: "workspaceSave", edit: { op: "create", name: "roof" } }]);
  s = reduce(s, { type: "workspaceSaved", seq: 1, answer: { ok: true, read: READY, aside: `${FILE}.unreadable-20261111T090807Z` } });
  assert.ok(noticesOf(s).includes("frame.workspace.h5 was not a readable workspace: it was moved aside to frame.workspace.h5.unreadable-20261111T090807Z, unchanged, before the save wrote a new frame.workspace.h5. Nothing was overwritten."));
  assert.equal(reduce(s, { type: "dismissWorkspaceAside" }).workspace.aside, null);
  assert.equal(reduce(s, { type: "workspaceRead", read: { status: "absent", file: "C:/runs/other.workspace.h5", token: "absent" } }).workspace.aside, null);
});

test("a newer file is read-only: the banner says so, and no save is asked", () => {
  const s = run([{ type: "workspaceRead", read: { ...READY, readOnly: "newer than this app's 1.0.0: it is shown read-only" } }]);
  assert.ok(noticesOf(s).some((n) => n.includes("shown read-only")));
  assert.equal(setsLines(s)!.canSave, false);
  assert.throws(() => reduce(s, { type: "workspaceSave", edit: { op: "delete", table: "sets", name: "roof" } }), /read-only/);
});

/** A model state with two cells: one carrying two OpenSees elements, one carrying none. */
function withCells(): State {
  const decl = (path: string, metas: number[]): Decl =>
    ({ path, kind: "element", type: "tri3", name: path, nameSource: "", h5: "", tag: null, params: [], refs: {}, links: {}, problems: [], fields: [], element: { femId: 1, cell: null, inlineNodes: [], groups: [], metas: metas.map((tag) => ({ type: "ShellMITC3", tag, h5: "", row: 0, args: [] })) } }) as Decl;
  const decls = Object.assign(Object.create(null), { "mesh/element/1": decl("mesh/element/1", [501, 502]), "mesh/element/2": decl("mesh/element/2", []) }) as State["decls"];
  return { ...initialState, decls };
}

test("an element set takes the OpenSees tags of the selected cells; a cell with none is left out, said by a note", () => {
  const s = run(
    [
      { type: "selectAdd", pick: { decl: "mesh/element/1", at: [0, 0, 0], nodes: [1, 2, 3] } },
      { type: "selectAdd", pick: { decl: "mesh/element/2", at: [0, 0, 0], nodes: [2, 3, 4] } },
    ],
    withCells(),
  );
  const m = membersOf(s, "elements", null);
  assert.ok(m.ok);
  assert.deepEqual(m.ids, [501, 502]);
  assert.match(m.notes[0]!, /1 selected cell is no OpenSees element/);
});

test("a node set takes, per click, the node of the clicked element nearest the point clicked", () => {
  const nodes = { ids: [1, 2, 3, 4], xyz: [0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0] };
  const s = run(
    [
      { type: "selectAdd", pick: { decl: "mesh/element/1", at: [0.9, 0.1, 0], nodes: [1, 2, 3] } },
      { type: "selectAdd", pick: { decl: "mesh/element/2", at: [0.1, 0.8, 0], nodes: [2, 3, 4] } },
    ],
    withCells(),
  );
  const m = membersOf(s, "nodes", nodes);
  assert.ok(m.ok);
  assert.deepEqual(m.ids, [2, 3]);
  assert.deepEqual(membersOf(initialState, "nodes", nodes).ok, false);
  assert.equal(membersOf(s, "nodes", null).ok, false, "no node table: refused, never guessed");
});

test("a set switch: the last set's read leaves at once, nothing is editable until the new read lands, and a late answer about the last file is never shown", () => {
  const model = (path: string): Event => ({ type: "fileOpened", artifact: "model", path });
  let s = run([model("C:/runs/frame.h5"), { type: "workspaceRead", read: READY }, { type: "workspaceSave", edit: { op: "delete", table: "sets", name: "roof" } }]);
  assert.equal(s.workspace.saving, 1);
  // The same model again (a re-read) keeps the read.
  assert.equal(reduce(s, model("C:/runs/frame.h5")).workspace.read, s.workspace.read);
  // Another model: the read, the save in flight and the notices go; the request count stays.
  s = reduce(s, model("C:/runs/other.h5"));
  assert.equal(s.workspace.read, null);
  assert.equal(s.workspace.saving, null);
  assert.equal(s.workspace.seq, 1);
  const opening = setsLines(s)!;
  assert.match(opening.status ?? "", /Reading the workspace of other\.h5, the set now open: nothing can be saved/);
  assert.deepEqual([opening.canSave, opening.sets.length], [false, 0], "the last set's sets are gone and nothing is saveable");
  assert.throws(() => reduce(s, { type: "workspaceSave", edit: { op: "delete", table: "sets", name: "roof" } }), /no set is open/);
  // The last set's save answers late: dropped.
  assert.equal(reduce(s, { type: "workspaceSaved", seq: 1, answer: { ok: true, read: { ...READY, revision: 5 }, aside: null } }), s);
  // The new set's read lands; an answer about the old file, for a save asked on the new one, is never shown as the new set's.
  const OTHER: WorkspaceIn = { status: "absent", file: "C:/runs/other.workspace.h5", token: "absent" };
  s = run([{ type: "workspaceRead", read: OTHER }, { type: "workspaceSave", edit: { op: "create", name: "base" } }], s);
  assert.deepEqual(s.workspace.request, { seq: 2, edit: { op: "create", name: "base" }, expect: "absent" });
  const late = reduce(s, { type: "workspaceSaved", seq: 2, answer: { ok: true, read: { ...READY, revision: 9 }, aside: null } });
  assert.equal(late.workspace.read, OTHER, "an answer naming frame.workspace.h5 never replaces other.workspace.h5's read");
  assert.equal(late.workspace.saving, null);
  // Closing the model clears the workspace too, keeping the request count (no seq is reused).
  const closed = reduce(s, { type: "fileClosed", artifact: "model" });
  assert.equal(closed.workspace.read, null);
  assert.equal(closed.workspace.seq, 2);
});
