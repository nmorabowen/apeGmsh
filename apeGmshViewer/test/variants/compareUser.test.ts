// VU19 (#2861; ADR 0132 D9.5): the compare slot's user results, end to end on
// the renderer side. The window set (Uncracked) opens through the real Effects
// and its own `.user.h5` is read as `user:read` reads it; the compare set
// (Cracked) answers through a VariantDoor whose compare doors call main's
// readers on the compare set's own files, as main.ts's `compare:*` doors do
// (test/main/compareDoors.test.ts holds main's side). The oracle is Python's
// read of the same files (test/fixtures/vu19/vu19.golden.json):
//
// - the compare slot lists the compare set's cases with their OWN A6 verdicts,
//   and the window set's user list is the same object before and after;
// - a compare user result is drawn on the window's mesh by node id with the
//   compare file's values (never the window file's under the same name), and
//   only when both models state one mesh (fem_hash): two meshes that do not
//   pair are refused with the reason, never interpolated;
// - a stale compare entry is refused and offered no Draw; an answer for a
//   closed or replaced slot is dropped; compare:close and the difference end it;
// - a legend field pick ends it too and draws the picked field (#2943): the
//   reducer clears the compare result as Hide does, a refused pick excepted;
//   Back's view memo records it and restoreView draws it again.

import assert from "node:assert/strict";
import { copyFileSync, cpSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, test } from "node:test";
import { fileURLToPath } from "node:url";
import * as h5wasm from "h5wasm/node";
import { compareStagesOf, ResultsHost } from "../../src/main/results.ts";
import { openLineage } from "../../src/main/zones.ts";
import { variantLines } from "../../src/panels/variants.ts";
import { openModel } from "../../src/reader/node.ts";
import type { ResultsH5 } from "../../src/reader/results.ts";
import { readUserResultColumn, readUserSet, type UserH5 } from "../../src/reader/userFile.ts";
import { UserResultFeed, userResultOnHand, VariantRunner, type VariantDoor } from "../../src/renderer/contour.ts";
import { restoreView, viewMemoOf } from "../../src/spec/snapshot.ts";
import { reduce } from "../../src/state/reduce.ts";
import { compareMeshOf, compareUserRequestOf, contourFieldsOf, contourView, deformOf, userResultRequestOf } from "../../src/state/selectors.ts";
import type { CompareIn, State, VariantsIn } from "../../src/state/types.ts";
import { harness, until } from "../contour/harness.ts";

const VU10 = fileURLToPath(new URL("../fixtures/vu10/", import.meta.url));
const VU19 = fileURLToPath(new URL("../fixtures/vu19/", import.meta.url));
const GOLDEN = JSON.parse(readFileSync(join(VU19, "vu19.golden.json"), "utf8")) as {
  compare_own: { results: Record<string, { stale: boolean; reason: string }> };
  compare_against_window: { results: Record<string, { stale: boolean; reason: string }> };
  compare_footfall: { ids: number[]; data: (number | null)[][] };
  window_footfall: { ids: number[]; data: (number | null)[][] };
};

const root = mkdtempSync(join(tmpdir(), "vu19-state-"));
after(() => rmSync(root, { recursive: true, force: true }));
cpSync(VU10, root, { recursive: true, filter: (src) => !src.endsWith(".py") });
copyFileSync(join(VU19, "plate_cracked.user.h5"), join(root, "cracked", "plate_cracked.user.h5"));
copyFileSync(join(VU19, "plate_uncracked.user.h5"), join(root, "plate_uncracked.user.h5"));
const UNCRACKED = { model: join(root, "plate_uncracked.h5"), results: join(root, "plate_uncracked.ladruno"), user: join(root, "plate_uncracked.user.h5") };
const CRACKED = { model: join(root, "cracked", "plate_cracked.h5"), results: join(root, "cracked", "plate_cracked.ladruno"), user: join(root, "cracked", "plate_cracked.user.h5") };
const SLOT = { index: 1, label: "Cracked" };
const h5 = h5wasm as unknown as UserH5;
const list = () => [] as string[];

const STUDY: VariantsIn = {
  manifest: join(root, "study.variants.json"),
  version: "1.0.0",
  newer: null,
  rows: [
    { index: 0, label: "Uncracked", path: "plate_uncracked.h5", params: { "Ie/Ig": 1 } },
    { index: 1, label: "Cracked", path: "cracked/plate_cracked.h5", params: { "Ie/Ig": 0.35 } },
  ],
  refused: [],
  active: 0,
  compare: null,
};

/** A VariantDoor whose compare doors read the compare set's own files with main's readers, as main.ts's do. */
function compareDoor(opts: { userFile?: string } = {}) {
  const host = new ResultsHost(h5wasm as unknown as ResultsH5, h5wasm.ready, { idleMs: 50 });
  const asked: string[] = [];
  const user = opts.userFile ?? CRACKED.user;
  const door: VariantDoor = {
    variantsSwitch: async () => ({ ok: false, reason: "not in this test" }),
    variantsCompare: async (index, label) => {
      const a = await host.open(CRACKED.results);
      if (!a.ok) return { ok: false, reason: a.error };
      const compare: CompareIn = { index, label, results: CRACKED.results, femHash: a.meta.lineage?.femHash ?? null, stages: compareStagesOf(a.meta) };
      return { ok: true, compare, nodeIds: a.nodeIds };
    },
    compareClose: async () => host.close(),
    readCompareStep: async (req) => {
      const r = host.readStep({ ...req, path: CRACKED.results });
      return r.ok ? { ok: true, seq: r.seq, nodal: r.nodal } : { ok: false, seq: r.seq, error: r.error };
    },
    compareModel: async () => {
      asked.push("model");
      const m = await openModel(CRACKED.model);
      const l = await openLineage(CRACKED.model);
      if (!l.ok || !l.lineage?.femHash || !l.lineage.modelHash) return { ok: false, slot: SLOT, reason: "no lineage" };
      return { ok: true, slot: SLOT, femHash: l.lineage.femHash, modelHash: l.lineage.modelHash, nodes: m.nodeIds.length, elements: m.blocks.reduce((n, b) => n + b.ids.length, 0) };
    },
    compareUserRead: async () => {
      asked.push("user");
      await h5wasm.ready;
      return { slot: SLOT, path: user, read: readUserSet(h5, user, CRACKED.model, CRACKED.results, list) };
    },
    compareUserReadResult: async (req) => {
      asked.push(`result ${Object.keys(req).sort().join(",")}`);
      await h5wasm.ready;
      return { slot: SLOT, read: readUserResultColumn(h5, user, CRACKED.model, CRACKED.results, list, req.name, req.component) };
    },
  };
  return { door, asked };
}

/** The window set (Uncracked) open with its own user file read, the study read, the compare set open and its user file read. */
async function compared(opts: { userFile?: string } = {}) {
  const h = harness();
  if (!(await h.effects.open(UNCRACKED.model))) throw new Error("the window model did not load");
  if (!(await h.effects.openResults(UNCRACKED.results))) throw new Error("the window results did not load");
  await h5wasm.ready;
  h.store.dispatch({ type: "userOpened", path: UNCRACKED.user });
  h.store.dispatch({ type: "userRead", path: UNCRACKED.user, read: readUserSet(h5, UNCRACKED.user, UNCRACKED.model, UNCRACKED.results, list) });
  // The window's own user-result feed runs beside the runner: a compare result shown must never read through it.
  const windowReads: string[] = [];
  const feed = new UserResultFeed(h.store, async (req) => {
    windowReads.push(`${req.name} ${req.component}`);
    return readUserResultColumn(h5, req.path, UNCRACKED.model, UNCRACKED.results, list, req.name, req.component);
  });
  const { door, asked } = compareDoor(opts);
  const runner = new VariantRunner(h.store, h.blobs, door, () => h.effects.currentStep());
  const offs = [feed.start(), runner.start()];
  h.store.dispatch({ type: "variantsRead", study: STUDY });
  h.store.dispatch({ type: "variantCompare", row: SLOT });
  await until(() => h.store.get().variants.compareUser?.status !== "reading" && h.store.get().variants.compareModel?.status === "ready", "the compare set's model and user file");
  return { h, asked, windowReads, off: () => offs.forEach((o) => o()) };
}

const column = (g: { data: (number | null)[][] }) => g.data.map((row) => (row[0] === null ? NaN : row[0]!));

test("the compare slot lists the compare set's own cases, each with its own A6 verdict; the window set's user list is unchanged", async () => {
  const { h, off, asked } = await compared();
  const s = h.store.get();
  const windowUser = s.user;
  assert.equal(windowUser?.status, "ready");
  const lines = variantLines(s.variants, compareMeshOf(s))!;
  const cu = lines.compareUser!;
  assert.match(cu.title, /Cracked's user file: plate_cracked\.user\.h5/);
  assert.deepEqual(cu.cases.map((c) => [c.name, c.verdict]), Object.entries(GOLDEN.compare_own.results).map(([n, v]) => [n, v.stale ? "stale" : "current"]));
  // Both current against the compare set's own files, where Python says both are stale against the window set's.
  for (const [n, v] of Object.entries(GOLDEN.compare_against_window.results)) assert.equal(v.stale, true, `precondition: ${n}`);
  assert.deepEqual(cu.cases.find((c) => c.name === "footfall")!.draws, [{ component: "ratio", shown: false }]);
  assert.equal(cu.combinations, "User Combinations: an empty zone (no row)");
  assert.equal(cu.mesh, null, "one mesh by fem_hash");
  // The doors named neither a path nor a row.
  assert.deepEqual(asked.slice().sort(), ["model", "user"]);
  // The window set's user file state is the very object it was: the compare read wrote nothing into it.
  assert.equal(h.store.get().user, windowUser);
  off();
});

test("Draw: the compare file's footfall ratio on the window's mesh by node id, never the window file's under the same name", async () => {
  const { h, off, windowReads, asked } = await compared();
  // The window's own footfall is drawn first, through the window feed.
  h.store.dispatch({ type: "setColourBy", by: "contour" });
  h.store.dispatch({ type: "setContourField", field: "footfall", component: "ratio", source: "user" });
  await until(() => contourView(h.store.get()).state === "user", "the window's user result");
  const w = contourView(h.store.get());
  assert.equal(w.state === "user" && w.name, "footfall");
  assert.deepEqual(Array.from(userResultOnHand(w as Extract<typeof w, { state: "user" }>)!.values), column(GOLDEN.window_footfall));
  // Now the compare set's footfall.
  h.store.dispatch({ type: "showCompareUser", pick: { name: "footfall", component: "ratio" } });
  assert.equal(userResultRequestOf(h.store.get()), null, "the window file is not read under a compare result");
  await until(() => {
    const v = contourView(h.store.get());
    return v.state === "user" && v.name === "Cracked: footfall";
  }, "the compare user result");
  const v = contourView(h.store.get()) as Extract<ReturnType<typeof contourView>, { state: "user" }>;
  const held = userResultOnHand(v)!;
  assert.deepEqual(Array.from(held.ids), GOLDEN.compare_footfall.ids);
  assert.deepEqual(Array.from(held.values), column(GOLDEN.compare_footfall));
  assert.equal(v.clim.min, Math.min(...column(GOLDEN.compare_footfall)));
  assert.equal(v.clim.max, Math.max(...column(GOLDEN.compare_footfall)));
  assert.ok(asked.includes("result component,name"), "the compare result door took a name and a component only");
  assert.deepEqual(windowReads, ["footfall ratio"], "the window feed read once, for the window's own result");
  // A compare user result is a contour only.
  const d = deformOf(h.store.get());
  assert.equal(d.ok, false);
  assert.match(!d.ok ? d.reason : "", /compare variant's user result is shown as a contour only/);
  // Hide: the window's own result comes back, read again through the window door.
  h.store.dispatch({ type: "showCompareUser", pick: null });
  await until(() => {
    const x = contourView(h.store.get());
    return x.state === "user" && x.name === "footfall" && userResultOnHand(x) !== null;
  }, "the window's user result again");
  assert.deepEqual(windowReads, ["footfall ratio", "footfall ratio"], "read again through the window's own door");
  assert.deepEqual(Array.from(userResultOnHand(contourView(h.store.get()) as Extract<ReturnType<typeof contourView>, { state: "user" }>)!.values), column(GOLDEN.window_footfall));
  off();
});

test("a legend field pick replaces a compare user result drawn, the same field as under it too; a refused pick keeps it (#2943)", async () => {
  const { h, off } = await compared();
  const PICK = { name: "footfall", component: "ratio" };
  const compareDrawn = () => {
    const v = contourView(h.store.get());
    return v.state === "user" && v.name === "Cracked: footfall" && userResultOnHand(v) !== null;
  };
  h.store.dispatch({ type: "setColourBy", by: "contour" });
  h.store.dispatch({ type: "setPhase", at: { kind: "results", stage: 0, step: 0 } });
  // The window's own footfall is the pick under the compare result.
  h.store.dispatch({ type: "setContourField", field: "footfall", component: "ratio", source: "user" });
  h.store.dispatch({ type: "showCompareUser", pick: PICK });
  await until(compareDrawn, "the compare user result");
  // A pick the reducer refuses changes nothing: the compare result stays drawn.
  assert.throws(() => h.store.dispatch({ type: "setContourField", field: "no_such_field", component: "x", source: "nodal" }), /is not a nodal field component/);
  assert.deepEqual(h.store.get().variants.shown, PICK);
  assert.ok(compareDrawn(), "still the compare result after a refused pick");
  // The legend picks the field the contour already holds: a pick all the same, so the window's footfall is drawn.
  h.store.dispatch({ type: "setContourField", field: "footfall", component: "ratio", source: "user" });
  assert.deepEqual([h.store.get().variants.shown, h.store.get().variants.shownRead], [null, null]);
  await until(() => {
    const x = contourView(h.store.get());
    return x.state === "user" && x.name === "footfall" && userResultOnHand(x) !== null;
  }, "the window's user result after the pick");
  const w = contourView(h.store.get()) as Extract<ReturnType<typeof contourView>, { state: "user" }>;
  assert.deepEqual(Array.from(userResultOnHand(w)!.values), column(GOLDEN.window_footfall));
  // Draw the compare result again, then pick a nodal field from the legend's own rows.
  h.store.dispatch({ type: "showCompareUser", pick: PICK });
  await until(compareDrawn, "the compare user result again");
  const rows = contourFieldsOf(h.store.get());
  assert.ok(rows !== null && rows.fields.length > 0, "precondition: the results stage shown offers nodal fields");
  const f = rows.fields[0]!;
  const key = f.components[0]!.key;
  h.store.dispatch({ type: "setContourField", field: f.name, component: key, source: "nodal" });
  assert.equal(h.store.get().variants.shown, null);
  const v = contourView(h.store.get());
  assert.equal(v.state, "ready", `the picked field is drawn: ${JSON.stringify(v)}`);
  assert.deepEqual(v.state === "ready" ? [v.field, v.component.key] : null, [f.name, key]);
  off();
});

test("Back's restoreView draws again the compare user result its memo held, and hides one drawn since (#2943 review F1)", async () => {
  const { h, off } = await compared();
  const PICK = { name: "footfall", component: "ratio" };
  const compareDrawn = () => {
    const v = contourView(h.store.get());
    return v.state === "user" && v.name === "Cracked: footfall" && userResultOnHand(v) !== null;
  };
  h.store.dispatch({ type: "setColourBy", by: "contour" });
  h.store.dispatch({ type: "setPhase", at: { kind: "results", stage: 0, step: 0 } });
  h.store.dispatch({ type: "showCompareUser", pick: PICK });
  await until(compareDrawn, "the compare user result");
  const model = UNCRACKED.model;
  const memo = viewMemoOf(h.store.get(), null, model);
  assert.deepEqual(memo.compareUser, PICK, "the memo records the compare user result drawn");
  // An offer's contour pick ends it (OfferRunner.apply dispatches the plan's setContourField).
  const f = contourFieldsOf(h.store.get())!.fields[0]!;
  h.store.dispatch({ type: "setContourField", field: f.name, component: f.components[0]!.key, source: "nodal" });
  assert.equal(h.store.get().variants.shown, null, "precondition: the offer's pick ended it");
  // Back: the memo's view, the compare result included, drawn again.
  restoreView(h.store, memo);
  assert.deepEqual(h.store.get().variants.shown, PICK);
  await until(compareDrawn, "the compare user result after Back");
  // Restoring an unchanged view (a refused offer's rollback) keeps it.
  restoreView(h.store, viewMemoOf(h.store.get(), null, model));
  assert.deepEqual(h.store.get().variants.shown, PICK);
  // A memo that drew none hides one drawn since.
  restoreView(h.store, { ...memo, compareUser: null });
  assert.equal(h.store.get().variants.shown, null);
  // A compare set closed since the memo: the reducer refuses the Draw with its reason (putBack reports it).
  h.store.dispatch({ type: "variantCompare", row: null });
  assert.throws(() => restoreView(h.store, memo), /no compare set's user file is read/);
  off();
});

test("two meshes that do not pair are refused with the reason, never interpolated: no read, no Draw", async () => {
  const { h, off } = await compared();
  h.store.dispatch({ type: "setColourBy", by: "contour" });
  h.store.dispatch({ type: "showCompareUser", pick: { name: "footfall", component: "ratio" } });
  const base = h.store.get();
  const m = base.variants.compareModel as Extract<NonNullable<State["variants"]["compareModel"]>, { status: "ready" }>;
  const other: State = { ...base, variants: { ...base.variants, compareModel: { ...m, femHash: "0".repeat(32) } } };
  assert.equal(compareUserRequestOf(other), null, "nothing is read for two meshes");
  const v = contourView(other);
  assert.equal(v.state, "refused");
  assert.match(v.state === "refused" ? v.reason : "", /not one mesh .*nodes are never interpolated/);
  const lines = variantLines(other.variants, compareMeshOf(other))!;
  const ff = lines.compareUser!.cases.find((c) => c.name === "footfall")!;
  assert.deepEqual(ff.draws, []);
  assert.match(ff.why ?? "", /not one mesh/);
  // A window model with no fem_hash cannot place the compare nodes either.
  const noHash: State = { ...base, lineage: base.lineage ? { ...base.lineage, femHash: null } : null };
  assert.match((compareMeshOf(noHash) as { reason: string }).reason, /stamps no fem_hash/);
  off();
});

test("a compare entry stale by its own verdict is listed with the reason, offered no Draw, and refused when picked", async () => {
  const { h, off } = await compared();
  // The compare file judged against the window set's model and results: what a door that crossed the slots would answer.
  const crossed = readUserSet(h5, CRACKED.user, UNCRACKED.model, UNCRACKED.results, list);
  h.store.dispatch({ type: "compareUserRead", answer: { slot: SLOT, path: CRACKED.user, read: crossed } });
  const s = h.store.get();
  const ff = variantLines(s.variants, compareMeshOf(s))!.compareUser!.cases.find((c) => c.name === "footfall")!;
  assert.equal(ff.verdict, "stale");
  assert.equal(ff.reason, GOLDEN.compare_against_window.results["footfall"]!.reason);
  assert.deepEqual(ff.draws, []);
  h.store.dispatch({ type: "setColourBy", by: "contour" });
  h.store.dispatch({ type: "showCompareUser", pick: { name: "footfall", component: "ratio" } });
  assert.equal(compareUserRequestOf(h.store.get()), null, "a stale entry is never read");
  const v = contourView(h.store.get());
  assert.equal(v.state, "refused");
  assert.match(v.state === "refused" ? v.reason : "", /is stale \(A6, against its own model and results\): model_hash differs/);
  off();
});

test("a compare set with no .user.h5 says so; the window set's file is not shown in its place", async () => {
  const { h, off } = await compared({ userFile: join(root, "cracked", "absent.user.h5") });
  const s = h.store.get();
  assert.equal(s.variants.compareUser?.status, "none");
  assert.match(variantLines(s.variants, compareMeshOf(s))!.compareUser!.title, /Cracked has no user file \(absent\.user\.h5\)/);
  assert.equal(s.user?.status, "ready", "the window's own file stays the window's");
  off();
});

test("an answer for a closed or replaced slot is dropped; compare:close and the difference end the compare result", async () => {
  const { h, off } = await compared();
  const s0 = h.store.get();
  // Another slot's answers are dropped.
  assert.equal(reduce(s0, { type: "compareUserRead", answer: { slot: { index: 0, label: "Uncracked" }, path: UNCRACKED.user, read: { ok: false, error: "x" } } }), s0);
  assert.equal(reduce(s0, { type: "compareModelRead", answer: { ok: false, slot: { index: 1, label: "Other" }, reason: "x" } }), s0);
  assert.equal(reduce(s0, { type: "compareUserResultRead", read: { key: "another", status: "loading" } }), s0);
  // A study main sends without the compare row (D9.3: a manifest re-read dropped it) takes the compare reads along.
  const dropped = reduce(s0, { type: "variantsRead", study: { ...STUDY, compare: null } }).variants;
  assert.deepEqual([dropped.compare, dropped.compareModel, dropped.compareUser], [null, null, null]);
  // The difference replaces a compare result, and a compare result the difference.
  h.store.dispatch({ type: "setColourBy", by: "contour" });
  h.store.dispatch({ type: "showCompareUser", pick: { name: "footfall", component: "ratio" } });
  h.store.dispatch({ type: "setDifference", on: true });
  assert.equal(h.store.get().variants.shown, null);
  h.store.dispatch({ type: "showCompareUser", pick: { name: "footfall", component: "ratio" } });
  assert.equal(h.store.get().variants.difference, false);
  // compare:close empties the compare set's model, file and the result drawn; a late answer then is dropped.
  h.store.dispatch({ type: "variantCompare", row: null });
  const closed = h.store.get().variants;
  assert.deepEqual([closed.compare, closed.compareModel, closed.compareUser, closed.shown, closed.shownRead], [null, null, null, null, null]);
  const s1 = h.store.get();
  assert.equal(reduce(s1, { type: "compareUserRead", answer: { slot: SLOT, path: CRACKED.user, read: readUserSet(h5, CRACKED.user, CRACKED.model, CRACKED.results, list) } }), s1);
  assert.throws(() => reduce(s1, { type: "showCompareUser", pick: { name: "footfall", component: "ratio" } }), /no compare set's user file is read/);
  off();
});

test("fail closed: a variant door without the compare doors is a broken build", () => {
  const h = harness();
  const { door } = compareDoor();
  const { compareUserRead: _drop, ...partial } = door;
  assert.throws(() => new VariantRunner(h.store, h.blobs, partial as unknown as VariantDoor, () => null), /has no compareUserRead/);
});
