// VU7b-2 (#2747, ADR 0130 D1, D4, A3): a User Combination shown as the current
// state. It is picked from the phase bar's User Combinations family, evaluated
// by VU4's evaluator through main's `user:readCase` (the feed in
// renderer/contour.ts), and drawn as a contour. The deformed shape refuses an
// unsigned combination, with the reason on screen.
//
// The oracles:
// - Python. test/fixtures/vu7b/sanramon_case.golden.json is what
//   `Results.combinations().evaluate(name, component, level="nodes")` answers for
//   the fixture's rows (its generator is text in that directory's README.md).
//   The column the feed holds equals it to 1e-12 relative, node for node.
// - A closed form. A `linear` row over one increment of the case is that
//   increment times its factor, read straight from the case's DATA.
// - ADR 0130 D4 and A3. Sign comes from the rule: `cqc` is unsigned, so it is
//   refused as a deformed shape; a `linear` over increments is signed.
//
// The fixture's `/spectra` was written against the San Ramon model, which is not
// in the repository. Staleness against it is main's (A6); here the zone is
// marked current, or stale where a test says so.

import assert from "node:assert/strict";
import { readdirSync, readFileSync } from "node:fs";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import * as h5wasm from "h5wasm/node";
import { CombinationRefused } from "../../src/combination/evaluate.ts";
import { FieldError } from "../../src/combination/field.ts";
import { userCombinationColumn, type Bridge } from "../../src/effects.ts";
import { familiesOf, mountPhase } from "../../src/panels/phase.ts";
import { deformLines } from "../../src/panels/legend.ts";
import { readUserFile, readUserSet, type UserH5 } from "../../src/reader/userFile.ts";
import { readCaseValues, spectrumSource, type CaseRead } from "../../src/reader/userSpectra.ts";
import { combinationColours, CombinationFeed, combinationContourKey, contourShownKey, toLinear, type CombinationFetch } from "../../src/renderer/contour.ts";
import { compileShot } from "../../src/spec/compile.ts";
import { validateSpec } from "../../src/spec/spec.ts";
import { viewVerdict } from "../../src/spec/verdict.ts";
import { BlobStore } from "../../src/state/blobs.ts";
import { NO_DATA_RGB, primitiveColours } from "../../src/state/contour.ts";
import { initialState, reduce, userEventOf } from "../../src/state/reduce.ts";
import { combinationRequestOf, combinationTableOf, contourFieldsOf, contourView, deformOf, deformView, noticesOf } from "../../src/state/selectors.ts";
import type { Store } from "../../src/state/store.ts";
import type { Event, MeshInfo, State, UserReadIn } from "../../src/state/types.ts";

await h5wasm.ready;
const H5 = h5wasm as unknown as UserH5;
const USER = fileURLToPath(new URL("../fixtures/vu7b/sanramon_case.user.h5", import.meta.url));
const GOLDEN = JSON.parse(readFileSync(fileURLToPath(new URL("../fixtures/vu7b/sanramon_case.golden.json", import.meta.url)), "utf8")) as {
  case: string;
  combinations: Record<string, Record<string, { ids: number[]; values: number[]; signed: boolean }>>;
};
const CASE = GOLDEN.case;
const RTOL = 1e-12;
const UX = "DISPLACEMENT/Ux";

/** A mesh as far as these tests reach it: a triangle and a line over the fixture's nodes (`vertexNodes`). */
function meshOf(blobs: BlobStore, tri: number[], line: number[]): MeshInfo {
  return { vertexNodes: { tri: blobs.put("tri", Float64Array.from(tri)), line: blobs.put("line", Float64Array.from(line)) } } as unknown as MeshInfo;
}

/** main's `user:read` answer for the fixture, with `/spectra` current (or stale). */
function userRead(stale: string | null = null): UserReadIn {
  const read = readUserSet(H5, USER, null, null, (d) => readdirSync(d));
  assert.ok(read.ok, read.ok ? "" : read.error);
  const combinations = stale === null ? [] : read.staleness.combinations;
  return { ...read, staleness: { ...read.staleness, combinations, zones: { ...read.staleness.zones, spectra: { stale: stale !== null, reason: stale ?? "" } } } };
}

/** The store after the fixture was read as renderer/app.ts reads it, over a mesh. */
function base(mesh: MeshInfo | null, stale: string | null = null): State {
  const s = { ...initialState, mesh };
  return reduce(reduce(s, { type: "userOpened", path: USER }), userEventOf(USER, userRead(stale)));
}

/** main's `user:readCase`, without Electron: the same two calls on the same file. */
const bridge: Pick<Bridge, "readUserCase"> = {
  readUserCase: async (req) => readCaseValues(H5, req.path, readUserFile(H5, req.path, null).spectra, req.case, req.target, req.label),
};
const fetchOf = (b: Pick<Bridge, "readUserCase">): CombinationFetch => (req) => userCombinationColumn(b, req);

/** A store over `reduce`, starting from `s`. */
function storeOf(s: State): Store & { events: Event[] } {
  let state = s;
  const listeners: ((n: State, p: State) => void)[] = [];
  const events: Event[] = [];
  return {
    events,
    get: () => state,
    dispatch: (e: Event) => {
      events.push(e);
      const prev = state;
      const next = reduce(prev, e);
      if (next === prev) return;
      state = next;
      for (const l of listeners) l(next, prev);
    },
    subscribe: (l: (n: State, p: State) => void) => {
      listeners.push(l);
      return () => listeners.splice(listeners.indexOf(l), 1);
    },
  } as unknown as Store & { events: Event[] };
}

/** Show combination `name` as the phase bar does: the state, then colouring by contour. */
const show = (s: State, name: string): State => reduce(reduce(s, { type: "setPhase", at: { kind: "combination", name } }), { type: "setColourBy", by: "contour" });

function close(got: Float64Array, want: readonly number[], what: string): void {
  assert.equal(got.length, want.length, `${what}: length`);
  want.forEach((w, i) => assert.ok(Math.abs(got[i]! - w) <= RTOL * Math.abs(w), `${what}: value ${i} is ${got[i]}, want ${w} to ${RTOL} relative`));
}

// ---- picking ---------------------------------------------------------------

test("the User Combinations family is drawable over a mesh, and lists the standing rows a click can show", () => {
  const f = familiesOf(base(meshOf(new BlobStore(), [], []))).find((x) => x.key === "combinations")!;
  assert.equal(f.drawable, true);
  assert.deepEqual(f.members, ["CQC Ex 2%", "CQC Ex", "SRSS Ex", "CQC Ex modes 3-6"]);
  assert.match(f.title, /^User Combinations; a click shows the first as a contour/);
  const none = familiesOf(base(null)).find((x) => x.key === "combinations")!;
  assert.deepEqual([none.drawable, none.members], [false, []]);
});

test("a combination is shown as the current state: off the axis, its pick its first nodal component, never a magnitude", () => {
  const s = show(base(meshOf(new BlobStore(), [], [])), "CQC Ex");
  assert.deepEqual(s.phase.at, { kind: "combination", name: "CQC Ex" });
  assert.ok(!s.phase.axis.some((k) => k.kind === "combination"), "a combination is never on the axis");
  assert.deepEqual([s.contour.source, s.contour.field, s.contour.component], ["nodal", "DISPLACEMENT", UX]);
  const picker = contourFieldsOf(s)!;
  assert.deepEqual(picker.fields.map((f) => [f.name, f.components.map((c) => c.key)]), [["DISPLACEMENT", [UX, "DISPLACEMENT/Uy", "DISPLACEMENT/Uz"]]]);
  assert.match(picker.fields[0]!.magnitudeRefused!, /evaluated per component/);
  assert.deepEqual(combinationRequestOf(s)?.quantity, { label: "DISPLACEMENT", component: "Ux" });
  // Another component of it, and not one it lacks.
  const uy = reduce(s, { type: "setContourField", field: "DISPLACEMENT", component: "DISPLACEMENT/Uy", source: "nodal" });
  assert.deepEqual(combinationRequestOf(uy)?.quantity, { label: "DISPLACEMENT", component: "Uy" });
  assert.throws(() => reduce(s, { type: "setContourField", field: "DISPLACEMENT", component: "DISPLACEMENT/Rx", source: "nodal" }), /is not a nodal component User Combination "CQC Ex" offers/);
  assert.throws(() => reduce(s, { type: "setContourField", field: "DISPLACEMENT", component: UX, source: "element" }), /is not a element component/);
});

test("setPhase refuses a combination the file does not list, one with no mesh, and one before the file is read", () => {
  const s = base(meshOf(new BlobStore(), [], []));
  assert.throws(() => reduce(s, { type: "setPhase", at: { kind: "combination", name: "nope" } }), /lists no User Combination "nope"/);
  assert.throws(() => reduce(base(null), { type: "setPhase", at: { kind: "combination", name: "CQC Ex" } }), /no mesh is loaded/);
  const reading = reduce({ ...initialState, mesh: s.mesh }, { type: "userOpened", path: USER });
  assert.throws(() => reduce(reading, { type: "setPhase", at: { kind: "combination", name: "CQC Ex" } }), /the user file is not read yet/);
});

test("a re-read keeps the combination shown but not its value; a file that no longer lists it leaves for the mesh", () => {
  const blobs = new BlobStore();
  let s = show(base(meshOf(blobs, [], [])), "CQC Ex");
  s = { ...s, phase: { ...s.phase, axis: [{ kind: "mesh" }] } };
  const key = combinationRequestOf(s)!.key;
  s = reduce(s, { type: "combinationRead", read: { key, status: "ready", label: UX, range: { min: 0, max: 1 }, nodes: 40, signed: false } });
  assert.equal(contourView(s).state, "combination");
  const pending = reduce(s, { type: "userOpened", path: USER });
  assert.deepEqual([pending.phase.at, pending.combination], [{ kind: "combination", name: "CQC Ex" }, null], "pending: still shown, its old value gone");
  assert.equal(contourView(pending).state, "unavailable");
  const gone = reduce(pending, { type: "userClosed" });
  assert.deepEqual(gone.phase.at, { kind: "mesh" });
});

// ---- drawn: the feed, Python's value, the colours --------------------------

test("CQC Ex is evaluated through user:readCase and drawn: the column equals Python's Results to 1e-12 relative", async () => {
  const blobs = new BlobStore();
  const want = GOLDEN.combinations["CQC Ex"]!["displacement_x"]!;
  const store = storeOf(show(base(meshOf(blobs, [want.ids[0]!, want.ids[1]!, want.ids[2]!], [want.ids[3]!, -7])), "CQC Ex"));
  const feed = new CombinationFeed(store, fetchOf(bridge));
  const busy = feed.check();
  assert.ok(busy, "the state asks for a read");
  assert.equal(store.get().combination?.status, "loading");
  assert.match((contourView(store.get()) as { reason: string }).reason, /is being evaluated/);
  await busy;
  const s = store.get();
  const r = s.combination!;
  assert.equal(r.status, "ready");
  assert.ok(r.status === "ready" && r.signed === false && r.nodes === want.ids.length);
  assert.deepEqual([...feed.held!.column.ids], want.ids);
  close(feed.held!.column.values, want.values, "CQC Ex displacement_x");
  const v = contourView(s);
  assert.equal(v.state, "combination");
  assert.ok(v.state === "combination");
  const finite = want.values.filter(Number.isFinite);
  assert.deepEqual([v.clim.min, v.clim.max], [Math.min(...finite), Math.max(...finite)]);
  assert.match(v.clim.label, /one state, no history/);
  assert.equal(contourShownKey(v), combinationContourKey(v));
  // The colours join the vertices by node id: the triangle takes its nodes' values; the line's unknown node is no data.
  const c = combinationColours(v, s.mesh!, blobs, feed.held, Int32Array.of(0), Int32Array.of(0))!;
  const expect = primitiveColours(Float64Array.from([want.values[0]!, want.values[1]!, want.values[2]!]), Int32Array.of(0), 3, v.clim, v.cmap).rgb.map(toLinear);
  assert.deepEqual([...c.tri], [...expect]);
  const grey = primitiveColours(Float64Array.of(want.values[3]!, NaN), Int32Array.of(0), 2, v.clim, v.cmap).rgb.map(toLinear);
  assert.deepEqual([...c.line], [...grey]);
  assert.deepEqual([...grey], [...Float32Array.from([...NO_DATA_RGB, ...NO_DATA_RGB]).map(toLinear)], "a primitive with one no-data vertex is no data whole");
  // Another read's column is not drawn.
  assert.equal(combinationColours(v, s.mesh!, blobs, { ...feed.held!, key: "other" }, Int32Array.of(0), Int32Array.of(0)), null);
  // No second read for the same key.
  assert.equal(feed.check(), null);
});

test("a signed combination (a linear over one increment) is that increment times its factor, the closed form", async () => {
  const blobs = new BlobStore();
  let s = base(meshOf(blobs, [], []));
  const u = s.user as Extract<NonNullable<State["user"]>, { status: "ready" }>;
  const row = { name: "-2 x mode 3", rule: "linear", terms: [{ kind: "spectrum", source: CASE, step: -1, mode: 3, factor: -2 }], envelope: "", fraction: null, damping: null };
  const c = u.info.combinations!;
  assert.ok(c.ok);
  s = { ...s, user: { ...u, info: { ...u.info, combinations: { ...c, items: [...c.items, { name: row.name, rule: "linear", origin: "user", terms: 1, stale: false }] }, table: { ...u.info.table!, rows: [...u.info.table!.rows, row] } } } };
  const store = storeOf(show(s, row.name));
  const feed = new CombinationFeed(store, fetchOf(bridge));
  await feed.check();
  const read = store.get().combination!;
  assert.ok(read.status === "ready" && read.signed, JSON.stringify(read));
  const raw = readCaseValues(H5, USER, readUserFile(H5, USER, null).spectra, CASE, "nodes", "DISPLACEMENT");
  assert.ok(raw.ok);
  const [, n, k] = raw.value.quantity.shape;
  const m = raw.value.case.modes.findIndex((x) => x.mode === 3);
  const ux = raw.value.quantity.components.indexOf("Ux");
  const want = Array.from({ length: n }, (_, i) => -2 * raw.value.data[(m * n + i) * k + ux]!);
  close(feed.held!.column.values, want, "linear -2 x mode 3");
  // Signed is not refused as unsigned: its deformed shape is not drawn yet, and says so.
  const d = deformOf(store.get());
  assert.ok(!d.ok && d.state === "unavailable" && /is signed; this app draws the deformed shape of a results step only/.test(d.reason), JSON.stringify(d));
});

// ---- the deformed shape refuses an unsigned combination (the mutation target) ----

test("the deformed shape refuses an unsigned combination and says why: the toggle, the Deformed card and the banner", () => {
  const s = show(base(meshOf(new BlobStore(), [], [])), "CQC Ex");
  const d = deformOf(s);
  const why = 'User Combination "CQC Ex" (cqc) is unsigned, so it is shown as a contour only, never as a deformed shape (ADR 0130 D4, A3)';
  assert.deepEqual(d, { ok: false, state: "refused", reason: why });
  // The legend disables the toggle with the reason, and states it on the Deformed card.
  const lines = deformLines(s);
  assert.deepEqual([lines.available, lines.why, lines.notes], [false, why, [{ text: `Unavailable: ${why}.`, refused: true }]]);
  // Turned on anyway (a key, a spec, a spotlight command): nothing deforms, and the banner says why.
  const on = reduce(s, { type: "setDeform", on: true });
  assert.deepEqual(deformView(on), { state: "refused", reason: why });
  assert.ok(noticesOf(on).includes(`Deformed shape not drawn: ${why}.`), noticesOf(on).join("\n"));
  // Every rule of the fixture is modal, so every row is refused the same way.
  for (const name of ["CQC Ex 2%", "SRSS Ex", "CQC Ex modes 3-6"]) {
    const x = deformOf(show(s, name));
    assert.ok(!x.ok && x.state === "refused" && x.reason.includes("is unsigned"), name);
  }
});

test("a --render shot that asks a User Combination for a deformed shape is refused with the banner's sentence", () => {
  const v = validateSpec({ spec: "apegmsh-view/1", files: { model: "m.h5" }, shots: [{ name: "d", combination: "CQC Ex", contour: { field: "DISPLACEMENT", component: "Ux" }, deform: {} }] });
  assert.ok(v.ok);
  const shot = v.spec.shots[0]!;
  const plan = compileShot(shot, { files: { results: null }, pairing: { state: "none" }, stages: [], groups: [] } as never);
  assert.ok(plan.ok, plan.ok ? "" : plan.reason);
  let s: State = { ...base(meshOf(new BlobStore(), [], [])), phase: { axis: [{ kind: "mesh" }], at: { kind: "mesh" } } };
  for (const e of plan.plan.events) s = reduce(s, e);
  const key = combinationRequestOf(s)!.key;
  s = reduce(s, { type: "combinationRead", read: { key, status: "ready", label: UX, range: { min: 0, max: 1 }, nodes: 40, signed: false } });
  const verdict = viewVerdict(s, shot, plan.plan, { contour: true, deform: false });
  assert.deepEqual(verdict, { kind: "waiting", why: 'Deformed shape not drawn: User Combination "CQC Ex" (cqc) is unsigned, so it is shown as a contour only, never as a deformed shape (ADR 0130 D4, A3).', loading: false });
});

// ---- refusals and failures are stated, never a blank view ------------------

test("a stale /spectra refuses stale, and the contour and banner say so (A6)", async () => {
  const store = storeOf(show(base(meshOf(new BlobStore(), [], []), "fem_hash differs"), "CQC Ex"));
  await new CombinationFeed(store, fetchOf(bridge)).check();
  const s = store.get();
  assert.deepEqual(s.combination && { status: s.combination.status, reason: (s.combination as { reason?: string }).reason }, { status: "refused", reason: "stale" });
  const v = contourView(s);
  assert.ok(v.state === "refused" && v.code === "combination" && /refused \(stale\): \/spectra is stale \(fem_hash differs\)/.test(v.reason), JSON.stringify(v));
  assert.ok(noticesOf(s).some((l) => l.startsWith('Contour not drawn: User Combination "CQC Ex" is refused (stale)')));
});

test("a case that repeats a node id raises in the evaluator, as Python's Field does; the app states it, never a blank view", async () => {
  const twice: Pick<Bridge, "readUserCase"> = {
    readUserCase: async (req) => {
      const r = await bridge.readUserCase!(req);
      if (!r.ok) return r;
      const ids = [...r.value.quantity.ids!];
      ids[1] = ids[0]!;
      return { ok: true, value: { ...r.value, quantity: { ...r.value.quantity, ids } } };
    },
  };
  await assert.rejects(userCombinationColumn(twice, { path: USER, table: combinationTableOf(base(null))!, name: "CQC Ex", quantity: { label: "DISPLACEMENT", component: "Ux" }, staleness: { stale: false, reason: "" } }), FieldError);
  const store = storeOf(show(base(meshOf(new BlobStore(), [], [])), "CQC Ex"));
  await new CombinationFeed(store, fetchOf(twice)).check();
  const s = store.get();
  assert.equal(s.combination?.status, "failed");
  const v = contourView(s);
  assert.ok(v.state === "refused" && /could not be evaluated: a field repeats the row/.test(v.reason), JSON.stringify(v));
});

test("a late answer is dropped: leaving the combination, or another pick, before the read answers", async () => {
  let release!: () => void;
  const gate = new Promise<void>((r) => (release = r));
  const slow: CombinationFetch = async (req) => {
    await gate;
    return fetchOf(bridge)(req);
  };
  const store = storeOf(show(base(meshOf(new BlobStore(), [], [])), "CQC Ex"));
  const feed = new CombinationFeed(store, slow);
  const busy = feed.check()!;
  store.dispatch({ type: "setContourField", field: "DISPLACEMENT", component: "DISPLACEMENT/Uy", source: "nodal" });
  const second = feed.check()!;
  release();
  await busy;
  await second;
  const s = store.get();
  assert.ok(s.combination?.status === "ready" && s.combination.label === "DISPLACEMENT/Uy", JSON.stringify(s.combination));
  assert.equal(feed.held!.column.label, "DISPLACEMENT/Uy");
  // An answer for another key never lands.
  assert.equal(reduce(s, { type: "combinationRead", read: { key: "[\"elsewhere\"]", status: "loading" } }), s);
  assert.throws(() => reduce(s, { type: "combinationRead", read: { key: "x", status: "maybe" } as never }), /is not a combination read/);
});

// ---- the deferred VU7b-1 review items (#2748) -----------------------------

const table = () => combinationTableOf(base(null))!;
const CURRENT = { stale: false, reason: "" };

test("#2748: a /spectra that cannot be read refuses before staleness, as Python does; a refused case on a stale zone stays stale", () => {
  const stale = { stale: true, reason: "a newer model" };
  const zone: CaseRead = readCaseValues(H5, USER, { ok: false, reason: "spectra_schema_version 2.0.0: newer" }, CASE, "nodes", "DISPLACEMENT");
  assert.ok(!zone.ok && zone.zone === true);
  const refusal = (r: CaseRead, st: typeof stale) => {
    try {
      spectrumSource(new Map([[CASE, r]]), { label: "DISPLACEMENT", component: "Ux" }, st, false).case(CASE);
    } catch (err) {
      assert.ok(err instanceof CombinationRefused);
      return err.reason;
    }
    throw new Error("it must refuse");
  };
  assert.equal(refusal(zone, stale), "refused", "a zone refusal is not hidden behind stale");
  const one = readCaseValues(H5, USER, { ok: true, value: { cases: [], refused: [{ name: CASE, reason: "bad-param", detail: "x", origin: "user", row: 0 }] } }, CASE, "nodes", "DISPLACEMENT");
  assert.ok(!one.ok && one.zone === undefined);
  assert.equal(refusal(one, stale), "stale");
  assert.equal(refusal(one, CURRENT), "refused");
});

test("#2748: `Ux` and `ux` in one result are two columns for one quantity, refused as ambiguous as Python refuses them", async () => {
  const both: Pick<Bridge, "readUserCase"> = {
    readUserCase: async (req) => {
      const r = await bridge.readUserCase!(req);
      if (!r.ok) return r;
      const components = r.value.quantity.components.map((c) => (c === "Uy" ? "ux" : c));
      return { ok: true, value: { ...r.value, quantity: { ...r.value.quantity, components } } };
    },
  };
  const read = await userCombinationColumn(both, { path: USER, table: table(), name: "CQC Ex", quantity: { label: "DISPLACEMENT", component: "Ux" }, staleness: CURRENT });
  assert.ok(!read.ok);
  assert.equal(read.refusal.reason, "unresolved");
  assert.match(read.refusal.detail, /holds 2 columns for nodes quantity 'DISPLACEMENT\/Ux'/);
  // A spelling the file does not use is not taken for one it does.
  const lower = await userCombinationColumn(bridge, { path: USER, table: table(), name: "CQC Ex", quantity: { label: "DISPLACEMENT", component: "ux" }, staleness: CURRENT });
  assert.ok(!lower.ok && /holds no nodes quantity/.test(lower.refusal.detail));
});

test("#2748: a mode or stage term refuses unresolved and says the app does not combine it yet, where Python's Results does", () => {
  const src = spectrumSource(new Map(), { label: "DISPLACEMENT", component: "Ux" }, CURRENT, false);
  for (const kind of ["mode", "stage"]) {
    try {
      src.field({ kind, source: "x", step: -1, mode: 1 });
      assert.fail("it must refuse");
    } catch (err) {
      assert.ok(err instanceof CombinationRefused && err.reason === "unresolved");
      assert.match(err.detail, /Python's Results also reads stage and mode terms; this app does not yet/);
    }
  }
});

// ---- the spec ----------------------------------------------------------------

test("the spec names a combination as the state shown: no stage beside it, and no step", () => {
  const spec = (shot: object) => validateSpec({ spec: "apegmsh-view/1", files: { model: "m.h5" }, shots: [{ name: "a", ...shot }] });
  const ok = spec({ combination: "CQC Ex", contour: { field: "DISPLACEMENT", component: "Ux" } });
  assert.ok(ok.ok);
  assert.equal(ok.spec.shots[0]!.combination, "CQC Ex");
  const plain = spec({});
  assert.ok(plain.ok && !("combination" in plain.spec.shots[0]!), "absent when not asked for, so a spec without one normalises as before");
  assert.deepEqual(spec({ combination: "CQC Ex", stage: 0 }), { ok: false, key: "shots[0].combination", reason: "names the state shown, as stage does: a shot shows a stage or a User Combination, not both" });
  assert.deepEqual(spec({ combination: "CQC Ex", step: 0 }), { ok: false, key: "shots[0].step", reason: "a User Combination is one state: it has no step" });
  assert.deepEqual(spec({ combination: "" }), { ok: false, key: "shots[0].combination", reason: 'must be a non-empty string; got ""' });
  const plan = compileShot(ok.spec.shots[0]!, { files: { results: null }, pairing: { state: "none" }, stages: [], groups: [] } as never);
  assert.ok(plan.ok);
  assert.deepEqual([plan.plan.phase, plan.plan.combination, plan.plan.stage, plan.plan.contour], [{ kind: "combination", name: "CQC Ex" }, "CQC Ex", null, { field: "DISPLACEMENT", component: UX, label: "Ux", source: "nodal" }]);
  assert.ok(plan.plan.events.some((e) => e.type === "setContourField" && e.component === UX));
});

// ---- the phase bar, mounted ---------------------------------------------------

class FakeEl {
  tagName: string;
  className = "";
  title = "";
  value = "";
  hidden = false;
  disabled = false;
  selected = false;
  dataset: Record<string, string | undefined> = {};
  style: Record<string, string> = {};
  children: (FakeEl | string)[] = [];
  listeners = new Map<string, (() => void)[]>();
  constructor(tag: string) {
    this.tagName = tag.toUpperCase();
  }
  append(...xs: (FakeEl | string)[]) {
    this.children.push(...xs);
  }
  replaceChildren(...xs: (FakeEl | string)[]) {
    this.children = [...xs];
  }
  addEventListener(type: string, fn: () => void) {
    this.listeners.set(type, [...(this.listeners.get(type) ?? []), fn]);
  }
  removeEventListener(type: string, fn: () => void) {
    this.listeners.set(type, (this.listeners.get(type) ?? []).filter((f) => f !== fn));
  }
  fire(type: string) {
    for (const fn of this.listeners.get(type) ?? []) fn();
  }
  get textContent(): string {
    return this.children.map((c) => (typeof c === "string" ? c : c.textContent)).join("");
  }
  set textContent(v: string) {
    this.children = [v];
  }
  getBoundingClientRect() {
    return { height: 40 };
  }
  all(pred: (e: FakeEl) => boolean): FakeEl[] {
    const out: FakeEl[] = pred(this) ? [this] : [];
    for (const c of this.children) if (typeof c !== "string") out.push(...c.all(pred));
    return out;
  }
}

test("mounted: a click on User Combinations shows the first as a contour; the picker shows another and says it is contour only", () => {
  const prevDoc = (globalThis as { document?: unknown }).document;
  const bar = new FakeEl("div");
  (globalThis as { document?: unknown }).document = {
    createElement: (t: string) => new FakeEl(t),
    getElementById: (id: string) => (id === "phase" ? bar : null),
    documentElement: { style: { setProperty: () => {}, removeProperty: () => {} } },
  };
  const s0 = base(meshOf(new BlobStore(), [], []));
  const store = storeOf({ ...s0, phase: { axis: [{ kind: "mesh" }], at: { kind: "mesh" } } });
  const off = mountPhase(store);
  try {
    const family = () => bar.all((e) => e.dataset["family"] === "combinations")[0]!;
    assert.equal(family().disabled, false);
    family().fire("click");
    assert.deepEqual(store.events.slice(-2), [
      { type: "setPhase", at: { kind: "combination", name: "CQC Ex 2%" } },
      { type: "setColourBy", by: "contour" },
    ]);
    assert.match(family().className, /active/);
    const pick = bar.all((e) => e.className === "combination-pick")[0]!;
    assert.deepEqual(pick.children.map((o) => (o as FakeEl).textContent), ["CQC Ex 2%", "CQC Ex", "SRSS Ex", "CQC Ex modes 3-6"]);
    assert.equal(bar.all((e) => e.className === "combination-readout")[0]!.textContent, "cqc · unsigned: contour only");
    assert.equal(bar.all((e) => e.className === "scrubber").length, 0, "a combination is one state: no step scrubber");
    pick.value = "SRSS Ex";
    pick.fire("change");
    assert.deepEqual(store.get().phase.at, { kind: "combination", name: "SRSS Ex" });
    assert.equal(bar.all((e) => e.className === "combination-readout")[0]!.textContent, "srss · unsigned: contour only");
  } finally {
    off();
    (globalThis as { document?: unknown }).document = prevDoc;
  }
});
