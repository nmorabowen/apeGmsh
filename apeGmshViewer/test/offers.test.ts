// ADR 0128 slice 5 (VA5): offers and snapshots through the app's own path in
// Node (test/contour/harness.ts: the real Effects over main's readers), with
// the viewport's part played by a small camera double.
//
// The oracles:
//   INV-3   the spec compiled from the fixture and the spec read back from
//           the state compile to one plan (./src/spec/compile.ts), and the
//           document read back is the spec applied, key for key.
//   INV-5   the state before an offer and after it, every key but `offers`,
//           are the same objects; only a click moves the view.
//   D5      "the same compile as D4": an offer applied over a view a human
//           has changed gives the state the same offer gives in a new window.
//   Back    the view memo before the click equals the memo after Back.
// The second-instance path and the exit codes need Electron: main.ts, with
// the end-to-end run once VA3 hands that file over.

import assert from "node:assert/strict";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, test } from "node:test";
import { fileURLToPath } from "node:url";
import { arrivalLabel, mountOfferBadge, mountOffers } from "../src/panels/offers.ts";
import { compileShot } from "../src/spec/compile.ts";
import { describeSet, SPEC_MAGNITUDE } from "../src/spec/describe.ts";
import { resolveSpecFiles } from "../src/spec/files.ts";
import { agentArgsOf, applyClip, MAX_PRESENT_SHOTS, offersOfDocument, presentDocument, OfferRunner, sameView, samePoseWithin, snapshotManifest, snapshotOf, toDocument, viewMemoOf, viewShotOf, type OfferPayload } from "../src/spec/snapshot.ts";
import { manifestEntry, MANIFEST_VERSION, type ManifestEntry } from "../src/spec/manifest.ts";
import { SPEC_VERSION, validateSpec, type SpecShot } from "../src/spec/spec.ts";
import { MAGNITUDE } from "../src/state/contour.ts";
import { canonicalPose, initialState, MAX_OFFERS, reduce } from "../src/state/reduce.ts";
import { Store, type Event, type State } from "../src/state/store.ts";
import type { CameraPose, CameraRequest, ClipPlane, OfferIn, Vec3, ViewMemo } from "../src/state/types.ts";
import { opened, STAGED, STAGED_RESULTS, until } from "./contour/harness.ts";

const FIX = fileURLToPath(new URL("../fixtures/", import.meta.url));
const DEFORM_2D = join(FIX, "deform_2d.h5");
const DEFORM_2D_RESULTS = join(FIX, "deform_2d.ladruno");

type Opened = Awaited<ReturnType<typeof opened>>;
const sets: Opened[] = [];
after(() => {
  for (const h of sets) {
    h.effects.dispose();
    h.host.close();
  }
});

/** Canonical poses (the reducer's form): the viewport's camera writes nothing else back. */
const canon = (eye: Vec3): CameraPose => canonicalPose({ eye, target: [0, 0, 0], up: [0, 0, 1] }, "test");
const POSES: Record<string, CameraPose> = {
  start: canon([1, 2, 3]),
  home: canon([9, 9, 9]),
  iso: canon([8, 8, 8]),
  "+x": canon([7, 0, 0]),
  fit: canon([6, 6, 6]),
};

/** The viewport's camera, as the runner sees it: a request is applied at the next paint, and written back. */
function cameraDouble(store: Store) {
  let shown: CameraPose = POSES["start"]!;
  const resolve = (to: CameraRequest): CameraPose => {
    if (typeof to === "object" && "eye" in to) return to;
    if (typeof to === "string" && POSES[to]) return POSES[to]!;
    throw new Error(`the camera double has no pose for ${JSON.stringify(to)}`);
  };
  return {
    pose: () => shown,
    settle: async () => {
      const r = store.get().camera.request;
      if (r) shown = resolve(r.to);
      store.dispatch({ type: "cameraMoved", pose: shown });
    },
    move: (p: CameraPose) => {
      shown = p;
      store.dispatch({ type: "cameraMoved", pose: p });
    },
  };
}

async function rig(model = STAGED, results = STAGED_RESULTS) {
  const h = await opened(model, results);
  sets.push(h);
  // The page the theme effect styles (renderer/app.ts hands it the real one).
  h.effects.setThemeTarget({ documentElement: { style: { setProperty() {} }, dataset: {}, setAttribute() {} }, getElementById: () => ({}) } as unknown as Document);
  const camera = cameraDouble(h.store);
  await camera.settle();
  const opens: string[] = [];
  // Fast clock: a view that stays refused settles (SETTLED_MS) without the test waiting for it.
  let t = 0;
  const runner = new OfferRunner({
    store: h.store,
    open: async (path) => {
      opens.push(path);
      if (path.toLowerCase().includes("deform_2d")) h.effects.openSet({ model: DEFORM_2D, geometry: null, results: DEFORM_2D_RESULTS });
      else h.effects.openSet({ model: STAGED, geometry: null, results: STAGED_RESULTS });
      return { ok: true };
    },
    onScreen: () => ({ contour: true, deform: true }),
    pose: camera.pose,
    settle: camera.settle,
    sleep: () => new Promise((r) => setTimeout(r, 1)),
    now: () => (t += 50),
  });
  const stop = runner.start();
  after(stop);
  const tick = () => new Promise((r) => setTimeout(r, 0));
  /** The human clicks Apply, and the click runs to its end. */
  const click = async (id: number) => {
    h.store.dispatch({ type: "applyOffer", id });
    await tick();
    await runner.whenIdle();
  };
  const back = async () => {
    h.store.dispatch({ type: "backOffer" });
    await tick();
    await runner.whenIdle();
  };
  return { ...h, camera, runner, opens, tick, click, back };
}

/** The offers of a spec on the fixture, through the present process's own function. */
function offers(shots: Record<string, unknown>[], top: Record<string, unknown> = {}, model = STAGED, results: string | null = STAGED_RESULTS): OfferIn[] {
  const r = offersOfDocument({ spec: SPEC_VERSION, files: results ? { model, results } : { model }, shots, ...top }, (spec) => resolveSpecFiles(spec, FIX, () => true), true);
  if (!r.ok) throw new Error(`${r.key}: ${r.reason}`);
  return r.offers;
}

const offerNamed = (s: State, name: string) => {
  const o = s.offers.list.find((x) => x.name === name);
  assert.ok(o, `offer ${name} is listed`);
  return o;
};
const shotOf = (o: OfferIn): SpecShot => (o.payload as OfferPayload).shot;

/**
 * The memo as a view: without the planes' creation ids, and without a contour
 * pick that colours nothing. Back cannot un-pick a field the offer picked (no
 * event clears a pick), and a pick that is not drawn is not part of the view.
 * The camera to INV-3's float tolerance: `canonicalPose` is exact only to rounding.
 */
function comparable(m: ViewMemo) {
  const c = m.contour;
  const contour = m.visibility.colourBy === "contour" ? c : { ...c, field: null, component: null, source: "nodal", fixed: null };
  const near = (v: Vec3) => v.map((x) => Math.round(x * 1e9) / 1e9);
  const camera = m.camera && { eye: near(m.camera.eye), target: near(m.camera.target), up: near(m.camera.up) };
  return { ...m, contour, camera, clip: { ...m.clip, planes: m.clip.planes.map(({ id: _id, ...p }) => p) } };
}

// ---- --present: the document becomes offers, or is refused whole (D2) -----------

test("D2: --present refuses a whole spec, naming the key, and lists nothing", () => {
  const go = (doc: unknown) => offersOfDocument(doc, (spec) => resolveSpecFiles(spec, FIX, () => true), true);
  const bad = go({ spec: SPEC_VERSION, files: { model: STAGED }, shots: [{ name: "a", colur_by: "group" }] });
  assert.equal(bad.ok, false);
  if (!bad.ok) assert.equal(bad.key, "shots[0].colur_by");
  const version = go({ spec: "apegmsh-view/2", files: { model: STAGED }, shots: [{ name: "a" }] });
  assert.equal(version.ok, false);
  if (!version.ok) assert.equal(version.key, "spec");
  const missing = offersOfDocument({ spec: SPEC_VERSION, files: { model: "nowhere.h5" }, shots: [{ name: "a" }] }, (spec) => resolveSpecFiles(spec, FIX, () => false), true);
  assert.equal(missing.ok, false);
  if (!missing.ok) assert.equal(missing.key, "files.model");
});

test("D5: one offer per shot, in the spec's order; a spec longer than the list is refused, not cut", () => {
  const two = offers([{ name: "a", caption: "First view." }, { name: "b" }]);
  assert.deepEqual(two.map((o) => [o.name, o.caption]), [["a", "First view."], ["b", null]]);
  const long = Array.from({ length: MAX_PRESENT_SHOTS + 1 }, (_, i) => ({ name: `s${i}` }));
  const r = offersOfDocument({ spec: SPEC_VERSION, files: { model: STAGED }, shots: long }, (spec) => resolveSpecFiles(spec, FIX, () => true), true);
  assert.equal(r.ok, false);
  if (!r.ok) assert.equal(r.key, "shots");
  assert.equal(offers(long.slice(0, MAX_PRESENT_SHOTS)).length, MAX_PRESENT_SHOTS);
});

test("D8: an offer knows whether its spec named a theme; D9: it says when still-only keys are ignored", () => {
  assert.equal((offers([{ name: "a" }])[0]!.payload as OfferPayload).theme, null, "the validator's batch default is not the spec's word");
  assert.equal((offers([{ name: "a" }], { theme: "light" })[0]!.payload as OfferPayload).theme, "light");
  assert.equal(offers([{ name: "a" }])[0]!.ignored, null);
  for (const shot of [{ chrome: "model" }, { chrome: "model", background: "transparent" }, { layers: { header: false } }]) {
    assert.match(offers([{ name: "a", ...shot }])[0]!.ignored ?? "", /not applied to this window/);
  }
});

// ---- the list (D5) ---------------------------------------------------------------

const plain = (name: string): OfferIn => ({ name, caption: null, ignored: null, payload: null });

test("D5: the list is newest first, capped at eight; a ninth drops the oldest", () => {
  let s = initialState;
  for (let i = 1; i <= MAX_OFFERS; i++) s = reduce(s, { type: "offerReceived", offers: [plain(`o${i}`)], at: i });
  assert.equal(MAX_OFFERS, 8);
  assert.deepEqual(s.offers.list.map((o) => o.name), ["o8", "o7", "o6", "o5", "o4", "o3", "o2", "o1"]);
  s = reduce(s, { type: "offerReceived", offers: [plain("o9")], at: 9 });
  assert.deepEqual(s.offers.list.map((o) => o.name), ["o9", "o8", "o7", "o6", "o5", "o4", "o3", "o2"]);
  assert.equal(s.offers.list[0]!.arrivedAt, 9);
  assert.deepEqual([...new Set(s.offers.list.map((o) => o.id))].length, 8, "ids are unique");
});

test("D5: offers last the window session: a closed or failed model keeps them", () => {
  let s = reduce(initialState, { type: "fileOpened", artifact: "model", path: "a.h5" });
  s = reduce(s, { type: "offerReceived", offers: [plain("kept")], at: 1 });
  const closed = reduce(s, { type: "fileClosed", artifact: "model" });
  assert.deepEqual(closed.offers.list.map((o) => o.name), ["kept"]);
  const failed = reduce(s, { type: "fileFailed", artifact: "model", path: "a.h5", error: "broken" });
  assert.deepEqual(failed.offers.list.map((o) => o.name), ["kept"]);
});

test("fail closed: a click on an unknown or a refused offer raises; Back with nothing applied raises; a click while one runs does nothing", () => {
  let s = reduce(initialState, { type: "offerReceived", offers: [plain("a"), plain("b")], at: 1 });
  assert.throws(() => reduce(s, { type: "applyOffer", id: 99 }), /not on the list/);
  assert.throws(() => reduce(s, { type: "backOffer" }), /no applied offer/);
  const [a, b] = s.offers.list;
  const refused = reduce(s, { type: "offerAssessed", id: b!.id, refused: "no such stage", opens: [] });
  assert.throws(() => reduce(refused, { type: "applyOffer", id: b!.id }), /refused and cannot be applied: no such stage/);
  s = reduce(s, { type: "applyOffer", id: a!.id });
  assert.equal(reduce(s, { type: "applyOffer", id: b!.id }), s, "a second click while one runs is dropped");
  assert.throws(() => reduce(s, { type: "offerRefused", id: a!.id, reason: "" }), /names its reason/);
});

// ---- INV-5: no takeover -----------------------------------------------------------

test("INV-5: an offer reaching the window changes nothing but the list; every key of the state is the same object", async () => {
  const r = await rig();
  r.store.dispatch({ type: "setPhase", at: { kind: "results", stage: 1, step: 0 } });
  await until(() => r.store.get().resultStep?.status === "ready", "the step");
  const before = r.store.get();
  const seen: (keyof State)[] = [];
  const off = r.store.subscribe((s, prev) => {
    for (const k of Object.keys(s) as (keyof State)[]) if (s[k] !== prev[k]) seen.push(k);
  });
  r.runner.receive(offers([{ name: "takeover", stage: "unprop", contour: { field: "displacement", component: "z" }, camera: "+x" }, { name: "refused", stage: "nowhere" }], { theme: "light" }));
  await r.tick();
  await r.tick();
  off();
  const after_ = r.store.get();
  assert.deepEqual([...new Set(seen)], ["offers"], "the offer and its assessment touch only the list");
  for (const k of Object.keys(before) as (keyof State)[]) if (k !== "offers") assert.equal(after_[k], before[k], `${k} is the same object`);
  assert.deepEqual(r.opens, [], "nothing opened");
  assert.deepEqual(r.camera.pose(), POSES["start"], "the camera did not move");
  assert.equal(offerNamed(after_, "takeover").status, "pending");
  assert.equal(offerNamed(after_, "refused").status, "refused");
});

test("INV-5: only the panel's click dispatches Apply or Back; main and the spec side never do", () => {
  const root = fileURLToPath(new URL("../src/", import.meta.url));
  const walk = (dir: string): string[] => readdirSync(dir, { withFileTypes: true }).flatMap((d) => (d.isDirectory() ? walk(join(dir, d.name)) : d.name.endsWith(".ts") ? [join(dir, d.name)] : []));
  const senders = walk(root).filter((f) => /type: "(applyOffer|backOffer)"/.test(readFileSync(f, "utf8")) && !/state[\\/]types\.ts$/.test(f));
  assert.deepEqual(senders.map((f) => f.slice(root.length).replace(/\\/g, "/")), ["panels/offers.ts"]);
});

test("INV-5: an agent's call returns before anything in the window takes focus, restores or opens", () => {
  const src = readFileSync(fileURLToPath(new URL("../src/main/main.ts", import.meta.url)), "utf8").replace(/\r\n/g, "\n");
  const start = src.indexOf('app.on("second-instance"');
  const heard = src.indexOf("heardFromAgent(data as Reach);\n      return;", start);
  assert.ok(start >= 0 && heard > start, "the handler hands an agent's call over, and returns");
  for (const opener of ["fileFromArgv(argv", "openPath(next", "win.restore()", "win.focus()"]) {
    assert.ok(src.indexOf(opener, start) > heard, `${opener} comes after the agent's return`);
  }
  const reach = src.slice(src.indexOf("function heardFromAgent"), src.indexOf('ipcMain.on("viewer:offersReady"'));
  assert.doesNotMatch(reach, /focus|restore|openPath|show\(|reload/, "the agent's path touches none of them");
});

// ---- an offer that would refuse is listed as refused (D5, D2) -----------------------

test("D5: an offer the set cannot draw is listed as refused, with the compiler's own reason, and cannot be applied", async () => {
  const r = await rig();
  const spec = { name: "nostage", stage: "nowhere" };
  r.runner.receive(offers([spec, { name: "nostep", stage: "base", step: 99 }, { name: "nofield", stage: "base", contour: { field: "nope", component: "z" } }, { name: "ok" }]));
  await r.tick();
  await r.tick();
  const s = r.store.get();
  const d = describeSet(s, []);
  for (const name of ["nostage", "nostep", "nofield"]) {
    const o = offerNamed(s, name);
    const c = compileShot(shotOf(o), d);
    assert.equal(c.ok, false);
    assert.equal(o.status, "refused");
    if (!c.ok) assert.equal(o.reason, c.reason, "the reason is the compiler's, word for word");
    assert.throws(() => r.store.dispatch({ type: "applyOffer", id: o.id }), /cannot be applied/);
  }
  assert.match(offerNamed(s, "nostage").reason!, /no stage named "nowhere"/);
  assert.match(offerNamed(s, "nostep").reason!, /step 99 is not one of them/);
  assert.equal(offerNamed(s, "ok").status, "pending");
});

// VU15 (#2776): the offer path has no User Combination resolver, and says so. Before, the compiler's
// no-resolver reason ("this view has no user file ...") reached the list, which is wrong whenever the set's
// user file is open. The oracle is the named reason, word for word, on both of D5's paths.
const COMBINATION_REASON = 'User Combination "C1": combination offers are not supported yet';

test("VU15: an offer of a combination contour is listed as refused, with the reason that it is a combination offer", async () => {
  const r = await rig();
  const contour = { field: "displacement", component: "z" };
  r.runner.receive(offers([{ name: "combo", combination: "C1", contour }, { name: "combo-phase", combination: "C1" }]));
  await r.tick();
  await r.tick();
  const s = r.store.get();
  const o = offerNamed(s, "combo");
  assert.equal(o.status, "refused");
  assert.equal(o.reason, COMBINATION_REASON);
  assert.doesNotMatch(o.reason!, /user file/);
  assert.throws(() => r.store.dispatch({ type: "applyOffer", id: o.id }), /cannot be applied/);
  // Only a combination contour is refused here: a shot that only shows a combination is the reducer's to judge.
  assert.notEqual(offerNamed(s, "combo-phase").reason, COMBINATION_REASON);
});

test("VU15: a combination contour offered on another set is refused at the click with the same reason, and the view goes back", async () => {
  const r = await rig();
  r.runner.receive(offers([{ name: "combo-there", combination: "C1", contour: { field: "displacement", component: "z" } }], {}, DEFORM_2D, DEFORM_2D_RESULTS));
  await r.tick();
  await r.tick();
  const listed = offerNamed(r.store.get(), "combo-there");
  assert.equal(listed.status, "pending", "it cannot be judged until its set is open");
  await r.click(listed.id);
  const s = r.store.get();
  assert.equal(offerNamed(s, "combo-there").status, "refused");
  assert.equal(offerNamed(s, "combo-there").reason, COMBINATION_REASON);
  assert.equal(s.offers.back.length, 0, "a refused click leaves nothing to go back from");
});

test("D5: an offer naming files outside the open set says so; its click opens them first, and Back returns", async () => {
  const r = await rig();
  r.store.dispatch({ type: "setPhase", at: { kind: "results", stage: 1, step: 1 } });
  await until(() => r.store.get().resultStep?.status === "ready", "the step");
  await r.camera.settle();
  const before = viewMemoOf(r.store.get(), r.camera.pose(), STAGED);
  r.runner.receive(offers([{ name: "other", stage: 0, step: "first" }], {}, DEFORM_2D, DEFORM_2D_RESULTS));
  await r.tick();
  await r.tick();
  const listed = offerNamed(r.store.get(), "other");
  assert.equal(listed.status, "pending", "it cannot be judged until its set is open");
  assert.deepEqual(listed.opens, [DEFORM_2D, DEFORM_2D_RESULTS]);
  assert.equal(r.store.get().artifacts.model!.path, STAGED, "listing it opened nothing");
  await r.click(listed.id);
  const s = r.store.get();
  assert.deepEqual(r.opens, [DEFORM_2D_RESULTS], "the click opened the set, through the results it names");
  assert.equal(s.artifacts.model!.path, DEFORM_2D);
  assert.equal(offerNamed(s, "other").status, "applied");
  assert.deepEqual(s.phase.at, { kind: "results", stage: 0, step: 0 });
  await r.back();
  const t = r.store.get();
  assert.equal(t.artifacts.model!.path, STAGED, "Back opened the set it came from");
  assert.deepEqual(r.opens, [DEFORM_2D_RESULTS, STAGED_RESULTS]);
  assert.deepEqual(t.phase.at, { kind: "results", stage: 1, step: 1 });
  assert.deepEqual(comparable(viewMemoOf(t, r.camera.pose(), STAGED)), comparable(before));
});

test("D2/D5: a view refused only when drawn is refused at the click; the view goes back, and reading the set again does not revive the offer", async () => {
  const r = await rig();
  const before = viewMemoOf(r.store.get(), r.camera.pose(), STAGED);
  // Compiles (a contour without a stage is the view's refusal, not the compiler's).
  r.runner.receive(offers([{ name: "nostep", contour: { field: "displacement", component: "z" } }]));
  await r.tick();
  await r.tick();
  const o = offerNamed(r.store.get(), "nostep");
  assert.equal(o.status, "pending");
  await r.click(o.id);
  const s = r.store.get();
  assert.equal(offerNamed(s, "nostep").status, "refused");
  assert.equal(offerNamed(s, "nostep").reason, "the contour is not drawn: contours colour a results step: choose results in the phase bar");
  assert.deepEqual(comparable(viewMemoOf(s, r.camera.pose(), STAGED)), comparable(before), "INV-4: the view it could not draw does not stay");
  assert.equal(s.offers.back.length, 0, "a refused click leaves nothing to go back from");
  assert.equal(s.offers.request, null);
  r.store.dispatch({ type: "offerReceived", offers: [], at: 2 });
  await r.tick();
  r.store.dispatch({ type: "fileChanged", path: STAGED });
  await until(() => r.store.get().artifacts.model!.stale === false, "the re-read");
  await r.tick();
  assert.equal(offerNamed(r.store.get(), "nostep").status, "refused");
});

// ---- D5: the same view the still showed ---------------------------------------------

const VIEWS: Record<string, Record<string, unknown>> = {
  mesh: { camera: "iso" },
  "mesh by role": { colour_by: "role", camera: "+x" },
  "last step, history range": { stage: "unprop", contour: { field: "displacement", component: "z" }, camera: "fit" },
  "first step, by index, step range": { stage: 0, step: "first", contour: { field: "displacement", component: "x", range: "step", cmap: "viridis" } },
  "a step, fixed range": { stage: "erect", step: 1, contour: { field: "displacement", component: "y", range: { min: -1, max: 1 } } },
  magnitude: { stage: "base", contour: { field: "displacement", component: SPEC_MAGNITUDE } },
  deformed: { stage: "unprop", step: 2, deform: { scale: 25 }, camera: { eye: [5, -4, 3], target: [1, 1, 0], up: [0, 0, 1] } },
  cut: { clip: [{ normal: "z", offset: 1.5, flip: true }, { normal: [0, 1, 0], offset: 0.25 }] },
  "a caption": { caption: "Vertical displacement, last step." },
};

for (const [title, view] of Object.entries(VIEWS)) {
  test(`INV-3: ${title}: the spec applied reads back as that spec, and the view alone compiles to the same plan`, async () => {
    const r = await rig();
    const groups = describeSet(r.store.get(), []).groups;
    r.runner.receive(offers([{ name: "v", ...view }], { theme: "light" }));
    await r.tick();
    await r.tick();
    const o = offerNamed(r.store.get(), "v");
    assert.equal(o.status, "pending", o.reason ?? "");
    await r.click(o.id);
    const s = r.store.get();
    assert.equal(offerNamed(s, "v").status, "applied", offerNamed(s, "v").reason ?? "");
    assert.equal(s.theme, "light", "the theme the spec names");
    const want = shotOf(o);
    const got = snapshotOf(s, r.camera.pose(), [1280, 720], r.runner["applied"].at(-1) ?? null);
    assert.ok(got.ok, got.ok ? "" : got.reason);
    if (!got.ok) return;
    assert.deepEqual(got.snapshot.outside, [], "the spec says all of this view");
    const v = validateSpec(got.snapshot.view);
    assert.ok(v.ok, v.ok ? "" : `${v.key}: ${v.reason}`);
    if (!v.ok) return;
    assert.deepEqual(v.spec.shots[0], want, "the document read back is the spec applied, key for key");
    assert.deepEqual(v.spec.files, { model: STAGED, results: STAGED_RESULTS });
    assert.equal(v.spec.theme, "light");
    assert.deepEqual(v.spec.size, [1280, 720], "the window's size, not the spec's (an offer never resizes the window)");
    // The state alone, with no memory of the offer: the same plan, and the camera within tolerance.
    const pose = r.camera.pose();
    const { shot: derived, outside } = viewShotOf(s, pose);
    assert.deepEqual(outside, []);
    assert.ok(sameView(derived, want, describeSet(s, [])), "the view read from the state compiles to the plan applied");
    const again = compileShot(derived, describeSet(s, []));
    assert.ok(again.ok);
    if (again.ok) assert.ok(typeof again.plan.camera === "object" && "eye" in again.plan.camera && samePoseWithin(again.plan.camera, pose));
    assert.ok(groups.length > 0);
  });
}

test("INV-3: an isolate reads back as an isolate", async () => {
  const r = await rig();
  const paths = describeSet(r.store.get(), []).groups.map((g) => g.path);
  r.runner.receive(offers([{ name: "iso", isolate: [paths[1]!, paths[0]!] }]));
  await r.tick();
  await r.tick();
  await r.click(offerNamed(r.store.get(), "iso").id);
  const got = r.runner.snapshot([800, 600]);
  assert.ok(got.ok);
  if (!got.ok) return;
  assert.deepEqual((got.snapshot.view["shots"] as { isolate: string[] }[])[0]!.isolate, [paths[1], paths[0]], "the offer's own order");
  const { shot, outside } = viewShotOf(r.store.get(), r.camera.pose());
  assert.deepEqual(outside, []);
  assert.deepEqual([...shot.isolate].sort(), [paths[0], paths[1]].sort());
});

test("INV-3: an element contour reads back as itself", async () => {
  const r = await rig(join(FIX, "elem_solid.h5"), join(FIX, "elem_solid.results.h5"));
  const d = describeSet(r.store.get(), []);
  const f = d.stages[0]!.fields.find((x) => x.source === "element");
  assert.ok(f, "the fixture offers an element field");
  r.runner.receive(offers([{ name: "el", stage: 0, contour: { field: f.name, component: f.components[0]!.key, source: "element", gauss: "cell_averaged", range: "step" } }], {}, join(FIX, "elem_solid.h5"), join(FIX, "elem_solid.results.h5")));
  await r.tick();
  await r.tick();
  await r.click(offerNamed(r.store.get(), "el").id);
  const s = r.store.get();
  assert.equal(offerNamed(s, "el").status, "applied", offerNamed(s, "el").reason ?? "");
  const { shot, outside } = viewShotOf(s, r.camera.pose());
  assert.deepEqual(outside.filter((x) => !x.includes("opacity")), []);
  assert.equal(shot.contour!.source, "element");
  assert.equal(shot.contour!.gauss, "cell_averaged");
  assert.ok(sameView(shot, shotOf(offerNamed(s, "el")), describeSet(s, [])));
});

test("toDocument: every normal form reads back unchanged through the validator (D1)", () => {
  const doc = {
    spec: SPEC_VERSION,
    files: { model: "m.h5", results: "m.results.h5" },
    size: [640, 480],
    theme: "dark",
    shots: [
      { name: "a", caption: "c", stage: "s", step: 3, camera: { eye: [1, 2, 3], target: [0, 0, 0], up: [0, 0, 1] }, contour: { field: "f", component: "z", source: "element", range: { min: 0, max: 1 }, cmap: "viridis", gauss: "node" }, deform: { scale: 2 }, clip: [{ normal: "x", offset: 1, flip: true }], isolate: ["/physical/A"], chrome: "model", background: "transparent", layers: { axes: false, caption: true } },
      { name: "b", colour_by: "role", chrome: "app", layers: { header: false } },
      { name: "c", stage: 1, step: "first", contour: { field: "f", component: "z", range: "step" } },
    ],
  };
  const v = validateSpec(doc);
  assert.ok(v.ok, v.ok ? "" : `${v.key}: ${v.reason}`);
  if (!v.ok) return;
  const back = validateSpec(toDocument(v.spec));
  assert.ok(back.ok, back.ok ? "" : `${back.key}: ${back.reason}`);
  if (back.ok) assert.deepEqual(back.spec, v.spec);
});

test("D5: an offer over a view the human has changed gives the view it gives a new window (the agent's still)", async () => {
  const shot = { name: "v", stage: "unprop", contour: { field: "displacement", component: "z" }, deform: { scale: 5 }, clip: [{ normal: "z", offset: 1 }], camera: "iso" };
  const fresh = await rig();
  fresh.runner.receive(offers([shot]));
  await fresh.tick();
  await fresh.tick();
  await fresh.click(offerNamed(fresh.store.get(), "v").id);

  const human = await rig();
  const d = human.store.dispatch.bind(human.store);
  const paths = describeSet(human.store.get(), []).groups.map((g) => g.path);
  d({ type: "setEdges", on: false });
  d({ type: "setOpacity", value: 0.2 });
  d({ type: "toggleExtrudeFrames", on: true });
  d({ type: "toggleShowOffsets", on: true });
  d({ type: "setLoads", on: true });
  d({ type: "setLineDiagram", on: true });
  d({ type: "setApplyCuts", on: false });
  d({ type: "setCutFace", on: false });
  d({ type: "setContourUnits", source: "kN_m_s", target: "tf_m_s" });
  d({ type: "setHidden", decls: [paths[0]!], hidden: true });
  d({ type: "addClipPlane", normal: [1, 0, 0] });
  d({ type: "setColourBy", by: "role" });
  d({ type: "setPhase", at: { kind: "results", stage: 0, step: 0 } });
  d({ type: "setContourCmap", cmap: "viridis" });
  await until(() => human.store.get().resultStep?.status === "ready", "the step");
  const messy = viewMemoOf(human.store.get(), human.camera.pose(), STAGED);
  human.runner.receive(offers([shot]));
  await human.tick();
  await human.tick();
  await human.click(offerNamed(human.store.get(), "v").id);

  const a = viewMemoOf(fresh.store.get(), fresh.camera.pose(), STAGED);
  const b = viewMemoOf(human.store.get(), human.camera.pose(), STAGED);
  assert.equal(offerNamed(human.store.get(), "v").status, "applied", offerNamed(human.store.get(), "v").reason ?? "");
  // The planes' creation counts differ; the planes themselves do not.
  assert.deepEqual(comparable(b), comparable(a));
  // And Back returns the view the human had built.
  await human.back();
  const c = viewMemoOf(human.store.get(), human.camera.pose(), STAGED);
  assert.deepEqual(comparable(c), comparable(messy));
});

test("D8: an offer that names no theme leaves the human's alone; a snapshot keeps it", async () => {
  const r = await rig();
  r.store.dispatch({ type: "setTheme", theme: "dark" });
  r.runner.receive(offers([{ name: "v" }]));
  await r.tick();
  await r.tick();
  await r.click(offerNamed(r.store.get(), "v").id);
  assert.equal(r.store.get().theme, "dark");
  const got = r.runner.snapshot([800, 600]);
  assert.ok(got.ok);
  if (got.ok) assert.equal(got.snapshot.view["theme"], "dark");
});

test("ADR 0128 Amendment 2: the camera after Apply is the spec's camera, the last request made, and nothing reframes it", async () => {
  const r = await rig();
  const eye: Vec3 = [30, -40, 25];
  const target: Vec3 = [2, 1, 3];
  const spec = { eye, target, up: [0, 0, 1] as Vec3 };
  const want = canonicalPose(spec, "test");
  r.runner.receive(offers([{ name: "pose", stage: "unprop", deform: { scale: 5 }, camera: spec }, { name: "named", camera: "+x" }]));
  await r.tick();
  await r.tick();
  const requests: unknown[] = [];
  const off = r.store.subscribe((s, prev) => {
    if (s.camera.seq !== prev.camera.seq) requests.push(s.camera.request?.to);
  });
  await r.click(offerNamed(r.store.get(), "pose").id);
  off();
  const s = r.store.get();
  assert.equal(offerNamed(s, "pose").status, "applied", offerNamed(s, "pose").reason ?? "");
  assert.deepEqual(s.camera.pose!.eye, eye);
  assert.deepEqual(s.camera.pose!.target, target);
  assert.ok(samePoseWithin(s.camera.pose!, want, 1e-12), "up, to `canonicalPose`'s rounding");
  assert.deepEqual(requests, ["home", want], "home, as a still does, then the spec's camera, last");
  const got = r.runner.snapshot([800, 600]);
  assert.ok(got.ok);
  if (got.ok) assert.deepEqual(got.snapshot.spec.shots[0]!.camera, spec, "and it reads back in the spec's words");
  // A named view: the name is the last request, whatever the window's panels.
  await r.click(offerNamed(r.store.get(), "named").id);
  assert.equal(r.store.get().camera.request?.to, "+x");
});

// ---- Back ---------------------------------------------------------------------------

test("D5: Back returns to the view from before the last applied offer, one click at a time", async () => {
  const r = await rig();
  r.store.dispatch({ type: "setPhase", at: { kind: "results", stage: 0, step: 1 } });
  r.store.dispatch({ type: "setDeform", on: true });
  r.store.dispatch({ type: "setDeformScale", scale: 3 });
  await until(() => r.store.get().resultStep?.status === "ready", "the step");
  const m0 = viewMemoOf(r.store.get(), r.camera.pose(), STAGED);
  r.runner.receive(offers([{ name: "a", stage: "unprop", camera: "+x" }, { name: "b", clip: [{ normal: "z", offset: 2 }], camera: "fit" }]));
  await r.tick();
  await r.tick();
  await r.click(offerNamed(r.store.get(), "a").id);
  const m1 = viewMemoOf(r.store.get(), r.camera.pose(), STAGED);
  await r.click(offerNamed(r.store.get(), "b").id);
  assert.equal(r.store.get().offers.back.length, 2);
  assert.notDeepEqual(comparable(viewMemoOf(r.store.get(), r.camera.pose(), STAGED)), comparable(m1));
  await r.back();
  assert.deepEqual(comparable(viewMemoOf(r.store.get(), r.camera.pose(), STAGED)), comparable(m1));
  assert.equal(offerNamed(r.store.get(), "b").status, "pending", "it is on offer again");
  assert.equal(offerNamed(r.store.get(), "a").status, "applied");
  await r.back();
  assert.deepEqual(comparable(viewMemoOf(r.store.get(), r.camera.pose(), STAGED)), comparable(m0));
  assert.equal(r.store.get().offers.back.length, 0);
  assert.equal(r.store.get().offers.notice, null);
});

test("D5: Back that cannot restore says so; the view is never quietly something else", async () => {
  const r = await rig();
  r.runner.receive(offers([{ name: "a" }]));
  await r.tick();
  await r.tick();
  await r.click(offerNamed(r.store.get(), "a").id);
  // The file the view came from now has one stage fewer than the memo names.
  const top = r.store.get().offers.back[0]!;
  const broken = { ...top.memo, at: { kind: "results" as const, stage: 99, step: 0 } };
  const stateWithBrokenMemo = { ...top, memo: broken };
  const orig = r.store.get.bind(r.store);
  r.store.get = () => {
    const s = orig();
    return s.offers.back[0] === top ? { ...s, offers: { ...s.offers, back: [stateWithBrokenMemo, ...s.offers.back.slice(1)] } } : s;
  };
  r.store.dispatch({ type: "backOffer" });
  await r.tick();
  await r.runner.whenIdle();
  r.store.get = orig;
  assert.match(r.store.get().offers.notice ?? "", /Back could not restore the view: setPhase/);
});

// ---- the snapshot (D6) --------------------------------------------------------------

test("D6: a snapshot changes nothing, and carries the selection beside the spec", async () => {
  const r = await rig();
  const decl = r.store.get().mesh!.elements[0]!;
  r.store.dispatch({ type: "select", pick: { decl, at: null, nodes: null } });
  const before = r.store.get();
  const got = r.runner.snapshot([1024, 768]);
  assert.equal(r.store.get(), before, "the state is the same object");
  assert.ok(got.ok);
  if (!got.ok) return;
  assert.deepEqual(got.snapshot.selection, [decl]);
  assert.deepEqual(got.snapshot.camera, POSES["start"]);
  const v = validateSpec(got.snapshot.view);
  assert.ok(v.ok);
  if (!v.ok) return;
  assert.equal(v.spec.shots[0]!.name, "view");
  assert.deepEqual(v.spec.shots[0]!.camera, POSES["start"]);
  assert.equal(v.spec.shots[0]!.stage, null);
  assert.deepEqual(Object.keys(got.snapshot.view).sort(), ["files", "shots", "size", "spec", "theme"], "the closed key set: no selection key in the spec");
});

test("D6: the human moves after applying: the snapshot is the view as it is, no longer the offer's words", async () => {
  const r = await rig();
  r.runner.receive(offers([{ name: "v", caption: "C.", stage: "unprop", step: "last", camera: "iso" }]));
  await r.tick();
  await r.tick();
  await r.click(offerNamed(r.store.get(), "v").id);
  const shotIn = (x: ReturnType<OfferRunner["snapshot"]>) => {
    if (!x.ok) throw new Error(x.reason);
    return (x.snapshot.view["shots"] as Record<string, unknown>[])[0]!;
  };
  assert.equal(shotIn(r.runner.snapshot([800, 600]))["name"], "v");
  assert.equal(shotIn(r.runner.snapshot([800, 600]))["camera"], "iso");
  // Orbit, within tolerance: still the offer's camera.
  const iso = POSES["iso"]!;
  r.camera.move({ ...iso, eye: [8 + 1e-7, 8, 8] });
  assert.equal(shotIn(r.runner.snapshot([800, 600]))["camera"], "iso");
  // Orbit for real: an explicit pose, and no longer the offer's name.
  r.camera.move(POSES["+x"]!);
  const moved = shotIn(r.runner.snapshot([800, 600]));
  assert.equal(moved["name"], "view");
  assert.equal(moved["caption"], undefined);
  assert.deepEqual(moved["camera"], POSES["+x"]);
  // Scrub: the step as it is.
  r.camera.move(iso);
  r.store.dispatch({ type: "setResultStep", step: 1 });
  const scrubbed = shotIn(r.runner.snapshot([800, 600]));
  assert.equal(scrubbed["name"], "view");
  assert.equal(scrubbed["step"], 1);
  assert.equal(scrubbed["stage"], "unprop");
});

test("D6: what the spec cannot say is named, never silently lost", async () => {
  const r = await rig();
  const paths = describeSet(r.store.get(), []).groups.map((g) => g.path);
  const d = r.store.dispatch.bind(r.store);
  d({ type: "setOpacity", value: 0.5 });
  d({ type: "setEdges", on: false });
  d({ type: "setLineDiagram", on: true });
  d({ type: "setLoads", on: true });
  d({ type: "setPhase", at: { kind: "results", stage: 0, step: 0 } });
  d({ type: "setDeform", on: true });
  d({ type: "setReferenceGhost", on: true });
  d({ type: "setHidden", decls: [r.store.get().mesh!.elements[0]!], hidden: true });
  d({ type: "addClipPlane", normal: [0, 0, 1] });
  d({ type: "setApplyCuts", on: false });
  const got = r.runner.snapshot([800, 600]);
  assert.ok(got.ok, got.ok ? "" : got.reason);
  if (!got.ok) return;
  for (const word of [/opacity is 0.5/, /edges are off/, /line diagram is on/, /loads are on/, /reference ghost/, /hidden declarations/, /Apply cuts off/]) {
    assert.ok(got.snapshot.outside.some((x) => word.test(x)), `${word} in ${JSON.stringify(got.snapshot.outside)}`);
  }
  assert.ok(validateSpec(got.snapshot.view).ok, "and it is still a spec the validator takes");
  assert.ok(paths.length);
  // Colouring by a magnitude: the spec's word, not the internal key.
  d({ type: "setContourField", field: "displacement", component: MAGNITUDE });
  d({ type: "setColourBy", by: "contour" });
  assert.equal(viewShotOf(r.store.get(), r.camera.pose()).shot.contour!.component, SPEC_MAGNITUDE);
});

test("D6: with nothing open, or nothing framed, the snapshot is refused with the reason", () => {
  const none = snapshotOf(initialState, POSES["start"]!, [800, 600], null);
  assert.deepEqual(none, { ok: false, reason: "the window has no model open, so it has no view to read back" });
});

test("D6: a stage name two stages share reads back as its index", async () => {
  const r = await rig();
  r.store.dispatch({ type: "setPhase", at: { kind: "results", stage: 1, step: 0 } });
  const s = r.store.get();
  const stages = s.results!.stages.map((st) => ({ ...st, name: "same" }));
  const dup: State = { ...s, results: { ...s.results!, stages } };
  assert.equal(viewShotOf(dup, POSES["start"]!).shot.stage, 1);
  assert.equal(viewShotOf(s, POSES["start"]!).shot.stage, "erect");
});

// ---- the section plane a spec names (D1) ------------------------------------------

test("D1: a plane is normal . x = offset with the normal as given: normal [2,0,0], offset 4 cuts at x = 2", async () => {
  const r = await rig();
  const offerOf = (clip: unknown) => compileShot(shotOf(offers([{ name: "c", clip }])[0]!), describeSet(r.store.get(), []));
  const c = offerOf([{ normal: [2, 0, 0], offset: 4, flip: true }, { normal: [0, -3, 4], offset: 10 }]);
  assert.ok(c.ok);
  if (!c.ok) return;
  applyClip(r.store, c.plan);
  const [a, b] = r.store.get().clip.planes as [ClipPlane, ClipPlane];
  assert.deepEqual(a.normal, [1, 0, 0]);
  assert.equal(a.offset, 2, "the plane through x = 2: 2 x 2 = 4");
  assert.equal(a.flipped, true);
  // The point 10 n / |n|^2 is on the spec's plane; the state's unit-normal plane holds it too.
  const on = [0, (-3 * 10) / 25, (4 * 10) / 25];
  assert.ok(Math.abs(b.normal[0] * on[0]! + b.normal[1] * on[1]! + b.normal[2] * on[2]! - b.offset) < 1e-12);
});

test("INV-3: a plane given with a non-unit normal reads back as the same plane", async () => {
  const r = await rig();
  r.runner.receive(offers([{ name: "c", clip: [{ normal: [2, 0, 0], offset: 4 }], camera: "iso" }]));
  await r.tick();
  await r.tick();
  await r.click(offerNamed(r.store.get(), "c").id);
  const s = r.store.get();
  assert.equal(offerNamed(s, "c").status, "applied", offerNamed(s, "c").reason ?? "");
  const { shot } = viewShotOf(s, r.camera.pose());
  assert.deepEqual(shot.clip, [{ normal: [1, 0, 0], offset: 2, flip: false }], "the state's own form of that plane");
  assert.ok(sameView(shot, shotOf(offerNamed(s, "c")), describeSet(s, [])), "[1,0,0] at 2 is [2,0,0] at 4");
  // A different plane is a different view: the comparison is not blind to the offset.
  assert.equal(sameView({ ...shot, clip: [{ normal: [2, 0, 0], offset: 2, flip: false }] }, shotOf(offerNamed(s, "c")), describeSet(s, [])), false);
  const back = r.runner.snapshot([800, 600]);
  assert.ok(back.ok);
  if (back.ok) assert.deepEqual(back.snapshot.spec.shots[0]!.clip, shotOf(offerNamed(s, "c")).clip, "the offer's own words, until the human moves");
});

// ---- the panel ----------------------------------------------------------------------

class FakeEl {
  children: FakeEl[] = [];
  parent: FakeEl | null = null;
  className = "";
  id = "";
  title = "";
  type = "";
  hidden = false;
  disabled = false;
  dataset: Record<string, string> = {};
  private own = "";
  private handlers = new Map<string, (() => void)[]>();
  get textContent(): string {
    return this.own + this.children.map((c) => c.textContent).join(" ");
  }
  set textContent(t: string) {
    this.own = t;
  }
  append(...xs: FakeEl[]) {
    for (const x of xs) x.parent = this;
    this.children.push(...xs);
  }
  prepend(x: FakeEl) {
    x.parent = this;
    this.children.unshift(x);
  }
  replaceChildren(...xs: FakeEl[]) {
    this.children = [];
    this.append(...xs);
  }
  remove() {
    if (this.parent) this.parent.children = this.parent.children.filter((c) => c !== this);
  }
  addEventListener(type: string, fn: () => void) {
    this.handlers.set(type, [...(this.handlers.get(type) ?? []), fn]);
  }
  removeEventListener(type: string, fn: () => void) {
    this.handlers.set(type, (this.handlers.get(type) ?? []).filter((f) => f !== fn));
  }
  click() {
    if (this.disabled) return;
    for (const f of this.handlers.get("click") ?? []) f();
  }
  all(cls: string): FakeEl[] {
    const here = this.className.split(" ").includes(cls) ? [this] : [];
    return [...here, ...this.children.flatMap((c) => c.all(cls))];
  }
}

function withFakeDom<T>(fn: (rail: FakeEl) => T): T {
  const rail = new FakeEl();
  rail.id = "rail";
  const toggle = new FakeEl();
  const g = globalThis as unknown as { document?: unknown };
  const before = g.document;
  g.document = { createElement: () => new FakeEl(), getElementById: (id: string) => (id === "rail" ? rail : id === "rail-toggle" ? toggle : null) };
  try {
    return fn(rail);
  } finally {
    g.document = before;
  }
}

test("the panel: no box with nothing on offer; name, caption, arrival, status; a refused offer's Apply is off; a click dispatches and nothing else moves", () => {
  withFakeDom((rail) => {
    const store = new Store();
    const sent: Event["type"][] = [];
    const dispose = mountOffers(store);
    const panel = rail.children[0]!;
    assert.equal(panel.id, "offers");
    assert.equal(panel.hidden, true, "nothing on offer: no box");
    const at = new Date(2026, 10, 1, 9, 7, 3).getTime();
    assert.equal(arrivalLabel(at), "09:07:03");
    store.dispatch({ type: "offerReceived", offers: [{ name: "peak-drift", caption: "Peak drift at column line C.", ignored: "A still's chrome.", payload: null }], at });
    store.dispatch({ type: "offerReceived", offers: [{ name: "no-stage", caption: null, ignored: null, payload: null }], at: at + 60000 });
    const [refusedId, okId] = store.get().offers.list.map((o) => o.id) as [number, number];
    store.dispatch({ type: "offerAssessed", id: refusedId, refused: "the results have no stage named \"x\"", opens: [] });
    store.dispatch({ type: "offerAssessed", id: okId, refused: null, opens: ["C:\\m\\other.h5"] });
    assert.equal(panel.hidden, false);
    const rows = panel.all("offer-row");
    assert.deepEqual(rows.map((r) => r.dataset["offer"]), [String(refusedId), String(okId)], "newest first");
    assert.match(rows[0]!.textContent, /no-stage 09:08:03 refused/);
    assert.match(rows[0]!.textContent, /no stage named/);
    assert.match(rows[1]!.textContent, /peak-drift 09:07:03 offered/);
    assert.match(rows[1]!.textContent, /Peak drift at column line C\./);
    assert.match(rows[1]!.textContent, /Applying opens other\.h5\./);
    assert.match(rows[1]!.textContent, /A still's chrome\./);
    const [refusedBtn, okBtn] = panel.all("offer-apply") as [FakeEl, FakeEl];
    assert.equal(refusedBtn.disabled, true);
    assert.equal(okBtn.disabled, false);
    assert.equal(panel.all("offer-back")[0]!.disabled, true, "nothing applied: Back is off");
    const off = store.subscribe((s, prev) => {
      if (s.offers.request !== prev.offers.request) sent.push(s.offers.request?.kind === "apply" ? "applyOffer" : "backOffer");
    });
    refusedBtn.click();
    assert.deepEqual(sent, []);
    okBtn.click();
    assert.deepEqual(sent, ["applyOffer"]);
    const running = panel.all("offer-apply");
    assert.ok(running.every((b) => b.disabled), "one click at a time");
    assert.match(panel.all("offer-row")[1]!.textContent, /applying/);
    off();
    dispose();
    assert.equal(rail.children.length, 0);
  });
});

test("the panel: Back is on after an applied offer, and a failed Back's notice shows", () => {
  withFakeDom(() => {
    const store = new Store();
    mountOffers(store);
    store.dispatch({ type: "offerReceived", offers: [plain("a")], at: 1 });
    const id = store.get().offers.list[0]!.id;
    store.dispatch({ type: "applyOffer", id });
    const memo = viewMemoOf(store.get(), null, "m.h5");
    store.dispatch({ type: "offerApplied", id, memo });
    const doc = (globalThis as unknown as { document: { getElementById(id: string): FakeEl } }).document;
    const panel = doc.getElementById("rail").children[0]!;
    const back = panel.all("offer-back")[0]!;
    assert.equal(back.disabled, false);
    assert.match(panel.all("offer-row")[0]!.textContent, /applied/);
    back.click();
    assert.equal(store.get().offers.request?.kind, "back");
    store.dispatch({ type: "offerBackDone", notice: "Back could not restore the view: gone" });
    assert.match(panel.textContent, /Back could not restore the view: gone/);
  });
});

test("the collapsed rail's toggle counts what is on offer: not the refused, not the applied", () => {
  withFakeDom(() => {
    const store = new Store();
    const dispose = mountOfferBadge(store);
    const doc = (globalThis as unknown as { document: { getElementById(id: string): FakeEl } }).document;
    const toggle = doc.getElementById("rail-toggle");
    assert.equal(toggle.dataset["offers"], undefined, "nothing on offer: no count");
    store.dispatch({ type: "offerReceived", offers: [plain("a")], at: 1 });
    assert.equal(toggle.dataset["offers"], "1 offer");
    store.dispatch({ type: "offerReceived", offers: [plain("b"), plain("c")], at: 2 });
    assert.equal(toggle.dataset["offers"], "3 offers");
    const [b, c] = store.get().offers.list;
    store.dispatch({ type: "offerAssessed", id: b!.id, refused: "no such stage", opens: [] });
    assert.equal(toggle.dataset["offers"], "2 offers");
    store.dispatch({ type: "applyOffer", id: c!.id });
    store.dispatch({ type: "offerApplied", id: c!.id, memo: null });
    assert.equal(toggle.dataset["offers"], "1 offer");
    dispose();
    assert.equal(toggle.dataset["offers"], undefined);
  });
});

test("D3/D6: the snapshot's manifest is the one D3 entry, with the selection and what the spec cannot say", async () => {
  const r = await rig();
  r.runner.receive(offers([{ name: "m", stage: "unprop", contour: { field: "displacement", component: "z" }, camera: "iso" }]));
  await r.tick();
  await r.tick();
  await r.click(offerNamed(r.store.get(), "m").id);
  await until(() => r.effects.currentStep()?.step === 3, "the step's values");
  const decl = r.store.get().mesh!.elements[0]!;
  r.store.dispatch({ type: "select", pick: { decl, at: null, nodes: null } });
  r.store.dispatch({ type: "setOpacity", value: 0.5 });
  const got = r.runner.snapshot([900, 700]);
  assert.ok(got.ok, got.ok ? "" : got.reason);
  if (!got.ok) return;
  const m = snapshotManifest(r.store.get(), got.snapshot, { image: [900, 700], fov: 45, layout: null, blobs: r.blobs, step: r.effects.currentStep() });
  assert.equal(m.manifest, MANIFEST_VERSION);
  assert.equal(m.shots.length, 1);
  const e = m.shots[0]! as ManifestEntry & { selection: string[]; outside: string[] };
  const direct = manifestEntry(r.store.get(), { shot: got.snapshot.spec.shots[0]!, png: "view.png", image: [900, 700], camera: { ...got.snapshot.camera, fov: 45 }, layout: null, blobs: r.blobs, step: r.effects.currentStep() });
  const { selection, outside, ...rest } = e;
  assert.deepEqual(rest, direct, "no second manifest: the entry is VA3's");
  assert.deepEqual(selection, [decl]);
  assert.deepEqual(outside, ["the opacity is 0.5"]);
  assert.equal(e.png, "view.png");
  assert.ok(e.contour && e.contour.drawn && e.contour.extrema.recorded !== null, "the extrema are stated");
});

// ---- nothing an agent sends raises in the human's window (review of #2697) ----------

test("no modal: a spec the window cannot judge is listed as refused, never a throw (wrong suffix, folder, missing, not JSON, not a spec)", () => {
  const dir = mkdtempSync(join(tmpdir(), "agv-offers-"));
  after(() => rmSync(dir, { recursive: true, force: true }));
  mkdirSync(join(dir, "folder"));
  writeFileSync(join(dir, "x.txt"), "x");
  const go = (files: unknown) => presentDocument(() => JSON.stringify({ spec: SPEC_VERSION, files, shots: [{ name: "n", caption: "C." }] }), (spec) => resolveSpecFiles(spec, dir, existsSync), true);
  // Each of these throws out of `resolveSpecFiles` (main/pairing.ts classify) or refuses; none may reach the caller as a throw.
  for (const files of [{ model: join(dir, "x.txt") }, { model: "x.txt" }, { model: join(dir, "folder") }, { model: STAGED, results: join(dir, "x.txt") }, { model: join(dir, "none.txt") }, { model: join(FIX, "contour_staged.results.h5") }, { model: STAGED, results: STAGED }]) {
    const p = go(files);
    assert.ok(p.refused, `${JSON.stringify(files)} is refused`);
    assert.equal(p.offers.length, 1);
    assert.equal(p.offers[0]!.name, "n", "the row keeps the name it could read");
    assert.equal(p.offers[0]!.caption, "C.");
    assert.ok(p.offers[0]!.refused?.includes(p.refused.reason));
  }
  // The throw is real: guarded here, not absent.
  assert.throws(() => offersOfDocument({ spec: SPEC_VERSION, files: { model: join(dir, "x.txt") }, shots: [{ name: "n" }] }, (spec) => resolveSpecFiles(spec, dir, existsSync), true));
  // No name to read: one row that says so.
  for (const text of ["{ not json", "null", "[]", `{"spec":"${SPEC_VERSION}"}`, `{"spec":"${SPEC_VERSION}","files":{"model":"m.h5"},"shots":[{"name":7}]}`]) {
    const p = presentDocument(() => text, (spec) => resolveSpecFiles(spec, dir, existsSync), true);
    assert.ok(p.refused, text);
    assert.deepEqual(p.offers.map((o) => o.name), ["unreadable spec"], text);
  }
});

test("no modal: the swap race: the file changes, or goes, between the sender's check and the window's read", () => {
  const good = JSON.stringify({ spec: SPEC_VERSION, files: { model: STAGED }, shots: [{ name: "n" }] });
  const bad = JSON.stringify({ spec: SPEC_VERSION, files: { model: "C:/Windows/win.ini" }, shots: [{ name: "n" }] });
  const resolve = (spec: Parameters<typeof resolveSpecFiles>[0]) => resolveSpecFiles(spec, FIX, () => true);
  assert.equal(presentDocument(() => good, resolve, true).refused, null, "what the sender saw");
  const swapped = presentDocument(() => bad, resolve, true);
  assert.equal(swapped.refused?.key, "files");
  assert.equal(swapped.offers[0]!.name, "n");
  const gone = presentDocument(() => {
    throw new Error("ENOENT");
  }, resolve, true);
  assert.equal(gone.refused?.key, "spec");
  assert.deepEqual(gone.offers.map((o) => o.name), ["unreadable spec"]);
});

test("a row that arrived refused stays refused, cannot be applied, and is not judged again", async () => {
  const r = await rig();
  const p = presentDocument(() => "{ not json", (spec) => resolveSpecFiles(spec, FIX, () => true), true);
  r.runner.receive(p.offers);
  await r.tick();
  await r.tick();
  const o = r.store.get().offers.list[0]!;
  assert.equal(o.status, "refused");
  assert.match(o.reason!, /^spec: not a readable JSON document/);
  assert.throws(() => r.store.dispatch({ type: "applyOffer", id: o.id }), /cannot be applied/);
  r.store.dispatch({ type: "fileChanged", path: STAGED });
  await r.tick();
  assert.equal(r.store.get().offers.list[0]!.status, "refused");
});

test("no modal: a click whose set cannot be opened refuses in the list, and the view stays", async () => {
  const h = await opened();
  sets.push(h);
  h.effects.setThemeTarget({ documentElement: { style: { setProperty() {} }, dataset: {}, setAttribute() {} }, getElementById: () => ({}) } as unknown as Document);
  const camera = cameraDouble(h.store);
  await camera.settle();
  const asked: [string, readonly string[]][] = [];
  const runner = new OfferRunner({
    store: h.store,
    open: async (path, needs) => {
      asked.push([path, needs]);
      return { ok: false, reason: `no such file: ${needs[0]}` };
    },
    onScreen: () => ({ contour: true, deform: true }),
    pose: camera.pose,
    settle: camera.settle,
  });
  after(runner.start());
  const before = viewMemoOf(h.store.get(), camera.pose(), STAGED);
  runner.receive(offers([{ name: "gone" }], {}, DEFORM_2D, DEFORM_2D_RESULTS));
  await new Promise((ok) => setTimeout(ok, 0));
  const o = offerNamed(h.store.get(), "gone");
  h.store.dispatch({ type: "applyOffer", id: o.id });
  await new Promise((ok) => setTimeout(ok, 0));
  await runner.whenIdle();
  const s = h.store.get();
  assert.deepEqual(asked, [[DEFORM_2D_RESULTS, [DEFORM_2D, DEFORM_2D_RESULTS]]], "the offer names every file it needs");
  assert.equal(offerNamed(s, "gone").status, "refused");
  assert.equal(offerNamed(s, "gone").reason, `no such file: ${DEFORM_2D}`);
  assert.equal(s.artifacts.model!.path, STAGED);
  assert.deepEqual(comparable(viewMemoOf(s, camera.pose(), STAGED)), comparable(before));
});

// ---- INV-3 over the forms a spec can take (review of #2697) --------------------------

test("INV-3: every small-integer normal, at several offsets, reads back as the same plane", async () => {
  const r = await rig();
  const d = describeSet(r.store.get(), []);
  let checked = 0;
  const lost: string[] = [];
  for (let x = -3; x <= 3; x++) for (let y = -3; y <= 3; y++) for (let z = -3; z <= 3; z++) {
    if (x === 0 && y === 0 && z === 0) continue;
    for (const offset of [1, 4, -7.3, 0.1]) {
      for (const flip of [false, true]) {
        const want = shotOf(offers([{ name: "c", clip: [{ normal: [x, y, z], offset, flip }] }])[0]!);
        const c = compileShot(want, d);
        assert.ok(c.ok);
        if (!c.ok) continue;
        applyClip(r.store, c.plan);
        const { shot } = viewShotOf(r.store.get(), r.camera.pose());
        checked++;
        if (!sameView(shot, want, d)) lost.push(JSON.stringify([x, y, z, offset, flip]));
      }
    }
  }
  assert.equal(checked, 342 * 8);
  assert.deepEqual(lost, []);
  // Tolerance is not blindness: the next plane over is another view.
  const base = shotOf(offers([{ name: "c", clip: [{ normal: [1, 2, 3], offset: 1 }] }])[0]!);
  const near = shotOf(offers([{ name: "c", clip: [{ normal: [1, 2, 3], offset: 1 + 1e-6 }] }])[0]!);
  assert.equal(sameView(base, near, d), false);
  const tilted = shotOf(offers([{ name: "c", clip: [{ normal: [1, 2, 3.00001], offset: 1 }] }])[0]!);
  assert.equal(sameView(base, tilted, d), false);
});

test("INV-3: isolating every group reads back as the offer: nothing is hidden, and it is the same view", async () => {
  const r = await rig();
  const paths = describeSet(r.store.get(), []).groups.map((g) => g.path);
  r.runner.receive(offers([{ name: "all", isolate: paths, camera: "iso" }, { name: "some", isolate: paths.slice(0, 1), camera: "iso" }]));
  await r.tick();
  await r.tick();
  await r.click(offerNamed(r.store.get(), "all").id);
  const s = r.store.get();
  assert.equal(offerNamed(s, "all").status, "applied");
  assert.deepEqual(s.visibility.hidden, []);
  const d = describeSet(s, []);
  assert.ok(sameView(viewShotOf(s, r.camera.pose()).shot, shotOf(offerNamed(s, "all")), d));
  assert.equal(sameView(viewShotOf(s, r.camera.pose()).shot, shotOf(offerNamed(s, "some")), d), false, "an isolate of some groups is another view");
  const got = r.runner.snapshot([800, 600]);
  assert.ok(got.ok);
  if (!got.ok) return;
  assert.deepEqual(got.snapshot.spec.shots[0], shotOf(offerNamed(s, "all")), "the offer's own words");
});

// ---- the ADR's spellings (review of #2697) ---------------------------------------------

test("D4, D5, D6: the ADR's spellings are the app's modes, and an agent's request never falls into view mode", () => {
  const argvOf = (...a: string[]) => {
    const r = agentArgsOf(a);
    assert.ok(r && "argv" in r, `${a.join(" ")}: ${JSON.stringify(r)}`);
    return r.argv;
  };
  assert.deepEqual(argvOf("--present", "s.json"), ["--mode=present", "--spec=s.json"]);
  assert.deepEqual(argvOf("--present=s.json"), ["--mode=present", "--spec=s.json"]);
  assert.deepEqual(argvOf("--snapshot", "--out", "d"), ["--mode=snapshot", "--out=d"]);
  assert.deepEqual(argvOf("--snapshot", "--out=d"), ["--mode=snapshot", "--out=d"]);
  assert.deepEqual(argvOf("--out", "d", "--snapshot"), ["--mode=snapshot", "--out=d"]);
  assert.deepEqual(argvOf("--render", "s.json", "--out", "d"), ["--mode=render", "--spec=s.json", "--out=d"]);
  assert.deepEqual(argvOf("--describe", "m.h5", "--out", "d"), ["--mode=describe", "--file=m.h5", "--out=d"]);
  assert.deepEqual(argvOf("--user-data-dir=u", "--present", "s.json"), ["--mode=present", "--spec=s.json"]);
  for (const [a, why] of [
    [["--present"], /needs a spec file/],
    [["--present", "--out", "d"], /needs a spec file/],
    [["--present", "s", "--out", "d"], /takes no --out/],
    [["--snapshot"], /needs --out/],
    [["--snapshot", "d"], /takes no value/],
    [["--snapshot", "--out"], /needs --out/],
    [["--render", "s"], /needs --out/],
    [["--render", "--out", "d"], /needs a spec file/],
    [["--describe", "--out", "d"], /needs a file/],
    [["--present", "s", "--snapshot", "--out", "d"], /only one of/],
    [["--present", "s", "--mode=view"], /two modes/],
    [["--spec=s"], /names no agent mode/],
    [["--out=d"], /names no agent mode/],
    [["m.h5", "--out", "d"], /names no agent mode/],
    [["--mode=view", "--spec=s"], /names no agent mode/],
  ] as [string[], RegExp][]) {
    const r = agentArgsOf(a);
    assert.ok(r && "refused" in r, `${a.join(" ")} is refused; got ${JSON.stringify(r)}`);
    assert.match(r.refused, why, a.join(" "));
  }
  // What stays as it was.
  for (const a of [[], ["m.h5"], ["--file=m.h5"], ["--mode=view", "--file=m.h5"], ["--mode=render", "--spec=s", "--out=d"], ["--mode=capture", "--file=m", "--out=o.png"], ["--mode=present", "--spec=s"]]) {
    assert.equal(agentArgsOf(a), null, a.join(" "));
  }
});
