// VU16 (#2800, ADR 0130 D7): a state-free `ON_NODES` user result of the set's
// `.user.h5`, picked in the Contour card and drawn as a contour. Footfall's
// ratio map (W3, #2799) is the first: one entry `footfall`, three components,
// `DATA (1, N, 3)` over every model node, NaN off the response set.
//
// The oracles:
// - W3's own writer. test/fixtures/vu16 holds files `FootfallResult.to_results`
//   wrote, and each golden is what Python's `UserFile.read_result("footfall")`
//   reads back (the generator is text in that directory's README.md).
//   `w3_bay` is W3's unit fixture itself (five ids, a value of 1e-300, NaN rows).
//   The column the reader gives, the column the feed holds, and the value each
//   drawn vertex takes equal the golden to 1e-12 relative; a NaN node is no data.
// - A6. The same user file against another fem's archive is stale; it is
//   refused with the reason, and its DATA is never read.
// - ADR 0130 D7. A user result is a contour only: never a deformed shape.

import assert from "node:assert/strict";
import { readdirSync, readFileSync } from "node:fs";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import * as h5wasm from "h5wasm/node";
import { deformLines, userResultsGroup } from "../../src/panels/legend.ts";
import { readUserNodalColumn, readUserResultColumn, readUserSet, type UserH5 } from "../../src/reader/userFile.ts";
import { contourShownKey, UserResultFeed, userResultColours, userResultContourKey, userResultHeld, type UserResultFetch } from "../../src/renderer/contour.ts";
import { compileShot, type UserResultOf } from "../../src/spec/compile.ts";
import type { SetDescription } from "../../src/spec/describe.ts";
import { manifestEntry } from "../../src/spec/manifest.ts";
import { validateSpec } from "../../src/spec/spec.ts";
import { viewVerdict } from "../../src/spec/verdict.ts";
import { columnOf, NO_DATA_RGB, primitiveColours, vertexValues } from "../../src/state/contour.ts";
import { reduce, userEventOf } from "../../src/state/reduce.ts";
import { contourView, deformOf, deformView, differencePlanOf, noticesOf, userResultPickOf, userResultRequestOf } from "../../src/state/selectors.ts";
import type { State } from "../../src/state/types.ts";
import { harness } from "../contour/harness.ts";

await h5wasm.ready;
const H5 = h5wasm as unknown as UserH5;
const at = (name: string) => fileURLToPath(new URL(`../fixtures/vu16/${name}`, import.meta.url));
const list = (d: string) => readdirSync(d);
const RTOL = 1e-12;
const RATIO = "footfall_ratio";

interface Golden {
  name: string;
  components: string[];
  ids: number[];
  shape: [number, number, number];
  /** DATA[0], row by row; null is NaN */
  data: (number | null)[][];
}
const golden = (name: string): Golden => JSON.parse(readFileSync(at(name), "utf8")) as Golden;

/** `got` equals the golden column `k` to 1e-12 relative, NaN where the golden is null. */
function sameColumn(got: Float64Array, g: Golden, k: number, what: string): void {
  assert.equal(got.length, g.ids.length, `${what}: length`);
  g.data.forEach((row, i) => {
    const want = row[k]!;
    if (want === null) assert.ok(Number.isNaN(got[i]!), `${what}: node ${g.ids[i]} is NaN in the file, got ${got[i]}`);
    else assert.ok(Math.abs(got[i]! - want) <= RTOL * Math.abs(want), `${what}: node ${g.ids[i]} is ${got[i]}, want ${want}`);
  });
}

// ---- the reader: W3's fixture -------------------------------------------------

test("W3's fixture: each footfall column reads as W3 wrote it, 1e-300 and NaN rows included", () => {
  const g = golden("w3_bay.golden.json");
  assert.deepEqual([g.name, g.components, g.shape], ["footfall", ["footfall_ap", RATIO, "footfall_fdom"], [1, 5, 3]]);
  for (const [k, component] of g.components.entries()) {
    const r = readUserResultColumn(H5, at("w3_bay.user.h5"), at("w3_bay.h5"), null, list, "footfall", component);
    assert.ok(r.ok, r.ok ? "" : r.detail);
    assert.deepEqual([...r.value.ids], g.ids);
    assert.deepEqual(r.value.components, g.components);
    sameColumn(r.value.values, g, k, component);
  }
  // The index carries the columns, so the picker offers them without reading DATA.
  const set = readUserSet(H5, at("w3_bay.user.h5"), at("w3_bay.h5"), null, list);
  assert.ok(set.ok && set.user.results.ok);
  assert.deepEqual(set.user.results.value.entries.map((e) => [e.name, e.target, e.basis, e.perState, e.components]), [["footfall", "nodes", "model", false, g.components]]);
});

test("A6: against another fem's archive, or none, the entry is stale and refused before its DATA is read", () => {
  const other = readUserResultColumn(H5, at("w3_bay.user.h5"), at("w3_other.h5"), null, list, "footfall", RATIO);
  assert.equal(other.ok, false);
  assert.ok(!other.ok && other.reason === "stale");
  assert.match(other.detail, /^user result 'footfall' is stale \(A6\): fem_hash differs: \S+ here, \S+ in the model$/);
  const none = readUserResultColumn(H5, at("w3_bay.user.h5"), null, null, list, "footfall", RATIO);
  assert.deepEqual(none, { ok: false, reason: "stale", detail: "user result 'footfall' is stale (A6): no model file to compare with" });
});

test("a name the file does not list, a component it does not have, or a refused zone is refused by name", () => {
  const p = at("w3_bay.user.h5");
  const read = (name: string, component: string) => readUserResultColumn(H5, p, at("w3_bay.h5"), null, list, name, component);
  assert.deepEqual(read("nope", RATIO), { ok: false, reason: "refused", detail: `${p} holds no user result 'nope'` });
  assert.deepEqual(read("footfall", "x"), { ok: false, reason: "refused", detail: "user result 'footfall' has no component 'x'; its components are footfall_ap, footfall_ratio, footfall_fdom" });
  assert.throws(() => readUserNodalColumn(H5, p, { ok: false, reason: "refused: broken" }, "footfall", RATIO), /\/results refused: broken$/);
  const entry = { name: "m", target: "elements" as const, basis: "model" as const, perState: false, linear: false, link: "r0", states: [], components: [] };
  assert.throws(() => readUserNodalColumn(H5, p, { ok: true, value: { entries: [entry], refused: [] } }, "m", "x"), /is on elements; this app draws a user result on nodes only/);
  const perState = { ...entry, target: "nodes" as const, perState: true, components: ["x"] };
  assert.throws(() => readUserNodalColumn(H5, p, { ok: true, value: { entries: [perState], refused: [] } }, "m", "x"), /is per state \(0 states\); this app draws a state-free one only/);
});

// ---- the app: the plate, its user file, and the feed --------------------------

const PLATE = at("plate.h5");
const PLATE_USER = at("plate.user.h5");
const PG = golden("plate.golden.json");
const K = PG.components.indexOf(RATIO);

/** The plate opened as a set is, and its user file read as renderer/app.ts reads it, judged against `model`. */
async function plateSet(model: string | null = PLATE) {
  const h = harness();
  if (!(await h.effects.open(PLATE))) throw new Error(`plate did not load: ${h.store.get().artifacts.model?.error}`);
  h.store.dispatch({ type: "userOpened", path: PLATE_USER });
  h.store.dispatch(userEventOf(PLATE_USER, readUserSet(H5, PLATE_USER, model, null, list)));
  const reads: { path: string; name: string; component: string }[] = [];
  // main's `user:readResult`, without Electron: the same call on the same files (main confines the path; test/main/userDoors.test.ts).
  const fetch: UserResultFetch = async (req) => {
    reads.push(req);
    return readUserResultColumn(H5, req.path, model, null, list, req.name, req.component);
  };
  const feed = new UserResultFeed(h.store, fetch);
  return { ...h, feed, reads };
}

/** Pick footfall's ratio in the Contour card, as the picker does. */
const pick = (s: State, component = RATIO): State =>
  reduce(reduce(s, { type: "setColourBy", by: "contour" }), { type: "setContourField", field: "footfall", component, source: "user" });

test("the User Results group offers footfall, state-free on the nodes, with its three columns", async () => {
  const { store } = await plateSet();
  const g = userResultsGroup(store.get())!;
  assert.deepEqual(g.rows.map((r) => [r.text, r.pick]), [["footfall (nodal, state-free)", { name: "footfall", components: PG.components }]]);
  assert.equal(g.note, null, "nothing is listed only");
  assert.deepEqual(userResultPickOf(store.get(), "footfall").ok, true);
});

test("footfall_ratio drawn on the plate equals W3's DATA to 1e-12, and a NaN node is no data", async () => {
  const { store, blobs, feed, reads } = await plateSet();
  store.dispatch({ type: "setColourBy", by: "contour" });
  store.dispatch({ type: "setContourField", field: "footfall", component: RATIO, source: "user" });
  // A user result is shown at the mesh phase: it has no state, and this set has no results file.
  assert.equal(store.get().phase.at?.kind, "mesh");
  assert.equal(contourView(store.get()).state, "unavailable", "being read");
  await feed.check();
  assert.deepEqual(reads, [{ path: PLATE_USER, name: "footfall", component: RATIO }]);
  const s = store.get();
  const v = contourView(s);
  assert.equal(v.state, "user", JSON.stringify(v));
  if (v.state !== "user") return;
  assert.deepEqual([v.name, v.component.key, v.basis, v.nodes], ["footfall", RATIO, "model", PG.ids.length]);
  const finite = PG.data.filter((r) => r[K] !== null).map((r) => r[K]!);
  assert.equal(v.finite, finite.length);
  assert.deepEqual([v.clim.min, v.clim.max], [Math.min(...finite), Math.max(...finite)], "the range is the column's own: one state, no history");
  // The column the feed holds.
  const held = userResultHeld.current!;
  assert.deepEqual([held.key, held.read], [v.key, v.read]);
  assert.deepEqual([...held.ids], PG.ids);
  sameColumn(held.values, PG, K, "held");
  // The value each drawn vertex takes, through the mesh's own vertex-to-node map.
  const mesh = s.mesh!;
  const byNode = new Map(PG.ids.map((id, i) => [id, PG.data[i]![K]!]));
  for (const which of ["tri", "line"] as const) {
    const nodes = blobs.get(mesh.vertexNodes[which]) as Float64Array;
    const drawn = vertexValues(nodes, columnOf(held.ids), [held.values], false);
    assert.ok(nodes.length > 0 || which === "line");
    nodes.forEach((id, i) => {
      const want = byNode.get(id);
      if (want === undefined || want === null) assert.ok(Number.isNaN(drawn[i]!), `${which} vertex ${i} (node ${id}) is no data`);
      else assert.ok(Math.abs(drawn[i]! - want) <= RTOL * Math.abs(want), `${which} vertex ${i} (node ${id}): ${drawn[i]} vs ${want}`);
    });
  }
  // The colours: a triangle that touches a NaN node is no data, every other one is on the ramp.
  const tri = blobs.get(mesh.vertexNodes.tri) as Float64Array;
  const nTri = tri.length / 3;
  const triMap = Int32Array.from({ length: nTri }, (_, i) => i);
  const colours = userResultColours(v, mesh, blobs, new Int32Array(0), triMap)!;
  assert.equal(colours.key, userResultContourKey(v));
  assert.equal(contourShownKey(v), colours.key);
  let touching = 0;
  for (let t = 0; t < nTri; t++) if ([0, 1, 2].some((j) => byNode.get(tri[3 * t + j]!) == null)) touching++;
  assert.ok(touching > 0 && touching < nTri, "the plate has both");
  const { noData } = primitiveColours(vertexValues(tri, columnOf(held.ids), [held.values], false), triMap, 3, v.clim, v.cmap);
  assert.equal(noData, touching);
  const grey = NO_DATA_RGB.map((c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4));
  let greys = 0;
  for (let t = 0; t < nTri; t++) if ([0, 1, 2].every((j) => grey.every((c, ch) => Math.abs(colours.tri[(3 * t + j) * 3 + ch]! - c) < 1e-6))) greys++;
  assert.equal(greys, touching);
});

test("a user result is never a deformed shape: the toggle is refused with the reason", async () => {
  const { store, feed } = await plateSet();
  store.dispatch({ type: "setColourBy", by: "contour" });
  store.dispatch({ type: "setContourField", field: "footfall", component: RATIO, source: "user" });
  await feed.check();
  const d = deformOf(store.get());
  assert.deepEqual(d, { ok: false, state: "refused", reason: "a user result is shown as a contour only, never as a deformed shape" });
  store.dispatch({ type: "setDeform", on: true });
  assert.deepEqual(deformView(store.get()), { state: "refused", reason: "a user result is shown as a contour only, never as a deformed shape" });
  assert.equal(deformLines(store.get()).available, false);
  // Nor is it subtracted across variants.
  const diff = differencePlanOf({ ...store.get(), variants: { ...store.get().variants, difference: true, compare: { status: "open" } as never } });
  assert.deepEqual(diff, { ok: false, reason: "a user result is of the window variant's own file, so it is not subtracted across variants" });
});

test("A6 in the app: a stale footfall is picked, refused with its reason on the contour and the banner, and never read", async () => {
  const { store, feed, reads } = await plateSet(null);
  store.dispatch({ type: "setColourBy", by: "contour" });
  store.dispatch({ type: "setContourField", field: "footfall", component: RATIO, source: "user" });
  assert.equal(userResultRequestOf(store.get()), null);
  assert.equal(await feed.check(), null);
  assert.deepEqual(reads, []);
  const v = contourView(store.get());
  assert.deepEqual(v, { state: "refused", code: "user", reason: 'user result "footfall" is stale (A6): no model file to compare with' });
  assert.ok(noticesOf(store.get()).includes('Contour not drawn: user result "footfall" is stale (A6): no model file to compare with.'));
});

test("main's refusal and a read that raises are stated on the contour, never a blank view", async () => {
  const { store } = await plateSet();
  const s = pick(store.get());
  const key = userResultRequestOf(s)!.key;
  const refused = reduce(s, { type: "userResultRead", read: { key, status: "refused", reason: "stale", detail: "user result 'footfall' is stale (A6): fem_hash differs" } });
  assert.deepEqual(contourView(refused), { state: "refused", code: "user", reason: `user result "footfall" is refused: user result 'footfall' is stale (A6): fem_hash differs` });
  const failed = reduce(s, { type: "userResultRead", read: { key, status: "failed", error: "boom" } });
  assert.deepEqual(contourView(failed), { state: "refused", code: "user", reason: 'user result "footfall" could not be read: boom' });
  // An answer for another key is dropped; a malformed one raises.
  assert.equal(reduce(s, { type: "userResultRead", read: { key: "other", status: "loading" } }), s);
  assert.throws(() => reduce(s, { type: "userResultRead", read: { key, status: "refused", reason: "nope" as "stale", detail: "" } }), /refusal reason "nope"/);
  // A feed whose door raises states it as a failed read.
  const h = await plateSet();
  h.store.dispatch({ type: "setColourBy", by: "contour" });
  h.store.dispatch({ type: "setContourField", field: "footfall", component: RATIO, source: "user" });
  const broken = new UserResultFeed(h.store, () => {
    throw new Error("this build's bridge has no readUserResult (main's user:readResult)");
  });
  await broken.check();
  assert.deepEqual(contourView(h.store.get()), { state: "refused", code: "user", reason: `user result "footfall" could not be read: this build's bridge has no readUserResult (main's user:readResult)` });
});

test("the reducer refuses a user result the file does not list, a component it does not have, and one over a User Combination", async () => {
  const { store } = await plateSet();
  const s = reduce(store.get(), { type: "setColourBy", by: "contour" });
  assert.throws(() => reduce(s, { type: "setContourField", field: "nope", component: RATIO, source: "user" }), /^RangeError: setContourField: plate\.user\.h5 lists no user result "nope"$/);
  assert.throws(() => reduce(s, { type: "setContourField", field: "footfall", component: "x", source: "user" }), /has no component "x"; its components are footfall_ap, footfall_ratio, footfall_fdom/);
  const onComb = { ...s, phase: { ...s.phase, at: { kind: "combination" as const, name: "C" } } };
  assert.throws(() => reduce(onComb, { type: "setContourField", field: "footfall", component: RATIO, source: "user" }), /User Combination "C" is shown; a user result is not drawn over it/);
  // A re-read of the user file drops the read, so the feed reads it again.
  const picked = pick(s);
  const key = userResultRequestOf(picked)!.key;
  const ready = reduce(picked, { type: "userResultRead", read: { key, status: "ready", read: 1, range: null, nodes: 0, finite: 0 } });
  assert.equal(reduce(ready, { type: "userOpened", path: PLATE_USER + ".x" }).userResult, null);
});

// ---- the spec: a `--render` shot names it --------------------------------------

const NO_SET = { files: { results: null }, pairing: { state: "none" }, stages: [], groups: [] } as unknown as SetDescription;
const spec = (shot: object) => validateSpec({ spec: "apegmsh-view/1", files: { model: "m.h5" }, shots: [{ name: "a", ...shot }] });

test("a spec shot names a user result by source user, on no stage, step, combination, compare or shape", () => {
  const ok = spec({ contour: { field: "footfall", component: RATIO, source: "user" } });
  assert.ok(ok.ok, ok.ok ? "" : ok.reason);
  const refused = (shot: object) => {
    const v = spec(shot);
    assert.ok(!v.ok);
    return [v.key, v.reason];
  };
  const c = { contour: { field: "footfall", component: RATIO, source: "user" } };
  assert.deepEqual(refused({ ...c, stage: 0 }), ["shots[0].stage", "a user result has no state: a shot of one names no stage"]);
  assert.deepEqual(refused({ ...c, combination: "C" }), ["shots[0].combination", "a shot shows a User Combination or a user result, not both"]);
  assert.deepEqual(refused({ ...c, deform: {} }), ["shots[0].deform", "a user result is shown as a contour only, never as a deformed shape"]);
});

test("the compiler turns a user-result shot into the pick, and refuses one the file cannot draw", async () => {
  const { store, feed, blobs } = await plateSet();
  const of: UserResultOf = (name) => {
    const p = userResultPickOf(store.get(), name);
    return p.ok ? { ok: true, components: p.entry.components } : p;
  };
  const v = spec({ contour: { field: "footfall", component: RATIO, source: "user", cmap: "jet" } });
  assert.ok(v.ok);
  const shot = v.spec.shots[0]!;
  const plan = compileShot(shot, NO_SET, undefined, undefined, of);
  assert.ok(plan.ok, plan.ok ? "" : plan.reason);
  assert.deepEqual([plan.plan.phase, plan.plan.stage, plan.plan.contour], [{ kind: "mesh" }, null, { field: "footfall", component: RATIO, label: RATIO, source: "user" }]);
  for (const e of plan.plan.events) store.dispatch(e);
  // Still coming until the read lands; shown once its colours are applied.
  assert.equal(viewVerdict(store.get(), shot, plan.plan, { contour: false, deform: false }).kind, "waiting");
  assert.equal((viewVerdict(store.get(), shot, plan.plan, { contour: false, deform: false }) as { loading: boolean }).loading, true);
  await feed.check();
  assert.deepEqual(viewVerdict(store.get(), shot, plan.plan, { contour: true, deform: false }), { kind: "shown" });
  // The manifest states it from the column on screen.
  const entry = manifestEntry(store.get(), { shot, png: "a.png", image: [16, 10], camera: { eye: [1, 1, 1], target: [0, 0, 0], up: [0, 0, 1], fov: 30 }, layout: null, blobs, step: null });
  const m = entry.contour;
  assert.ok(m && m.drawn, JSON.stringify(m));
  if (!m.drawn) return;
  assert.deepEqual([m.source, m.user, m.field, m.component, m.range.mode, m.topology], ["user", { name: "footfall", basis: "model" }, "footfall", RATIO, "state", "nodal"]);
  const finite = PG.data.filter((r) => r[K] !== null).map((r) => r[K]!);
  assert.deepEqual([m.range.min, m.range.max], [Math.min(...finite), Math.max(...finite)]);
  assert.equal(m.no_data.count, PG.data.filter((r) => r[K] === null).length, "every NaN node is counted as no data");
  // Refused at compile time, with the app's reason.
  const c = (field: string, component = RATIO) => {
    const w = spec({ contour: { field, component, source: "user" } });
    assert.ok(w.ok);
    return compileShot(w.spec.shots[0]!, NO_SET, undefined, undefined, of);
  };
  assert.deepEqual(c("nope"), { ok: false, reason: 'user result "nope" cannot be drawn: plate.user.h5 lists no user result "nope"' });
  assert.deepEqual(c("footfall", "x"), { ok: false, reason: 'user result "footfall" has no component "x"; its components are footfall_ap, footfall_ratio, footfall_fdom' });
  assert.deepEqual(compileShot(shot, NO_SET), { ok: false, reason: 'this view has no user file to find user result "footfall" in' });
});
