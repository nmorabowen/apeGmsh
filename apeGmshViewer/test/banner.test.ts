// The banner's summary line: the open or closed arrow is drawn by style.css
// (`.banner-summary::before`), so an app-chrome still can draw it open. A text
// arrow here would draw two.

//
// VF2 (#2924 item 6): a derived contour on elements the model does not
// classify as plane stress / plane strain puts the manifest's plane line on
// the banner, wired as renderer/app.ts wires it; the line follows the field,
// the step and the set.

import assert from "node:assert/strict";
import { after, test } from "node:test";
import { fileURLToPath } from "node:url";
import { mountBanner } from "../src/panels/banner.ts";
import { elementContourKey } from "../src/renderer/contour.ts";
import { takeShots, type RenderDeps } from "../src/renderer/agentRender.ts";
import { describeSet } from "../src/spec/describe.ts";
import { manifestEntry, refusedEntry } from "../src/spec/manifest.ts";
import { validateSpec, type SpecShot } from "../src/spec/spec.ts";
import { contourFieldsOf, contourView, derivedPlaneOf } from "../src/state/selectors.ts";
import { Store } from "../src/state/store.ts";
import type { Event } from "../src/state/types.ts";
import { opened, until } from "./contour/harness.ts";
import * as h5wasm from "h5wasm/node";
import { metaOf } from "../src/main/results.ts";
import { openLadruno } from "../src/reader/ladruno.ts";
import type { ResultsH5 } from "../src/reader/results.ts";
import { pairingNotice, pairResults } from "../src/state/results.ts";

class FakeEl {
  className = "";
  title = "";
  type = "";
  hidden = false;
  textContent = "";
  dataset: Record<string, string | undefined> = {};
  style: Record<string, string> = {};
  children: FakeEl[] = [];
  listeners: (() => void)[] = [];
  replaceChildren(...xs: FakeEl[]) {
    this.children = xs;
  }
  addEventListener(_: string, fn: () => void) {
    this.listeners.push(fn);
  }
  removeEventListener(_: string, fn: () => void) {
    this.listeners = this.listeners.filter((f) => f !== fn);
  }
}

test("the banner's summary carries no arrow of its own, closed or open", () => {
  const real = new Store().get();
  const s = { ...real, artifacts: { ...real.artifacts, model: { path: "m.h5", status: "ready", error: null, stale: false, sizeBytes: 0, readMs: 0, zones: {}, warnings: ["first", "second"] } } } as typeof real;
  const fake = { get: () => s, subscribe: () => () => {}, dispatch: () => {} };
  const banner = new FakeEl();
  const prevDoc = (globalThis as { document?: unknown }).document;
  (globalThis as { document?: unknown }).document = { getElementById: () => banner, createElement: () => new FakeEl() };
  try {
    const unmount = mountBanner(fake as never);
    const summary = () => banner.children[0]!;
    assert.equal(banner.hidden, false, "two warnings show the banner");
    assert.equal(summary().className, "banner-summary");
    assert.match(summary().textContent, /^2 warnings/, "closed: the count, with no arrow in front");
    for (const fn of summary().listeners) fn();
    assert.equal(banner.dataset["open"], "true");
    assert.match(summary().textContent, /^2 warnings/, "open: the count, with no arrow in front");
    assert.doesNotMatch(summary().textContent, /[▸▾]/);
    unmount();
  } finally {
    (globalThis as { document?: unknown }).document = prevDoc;
  }
});

// ---- VF2: the derived contour's plane line ----

const fx = (name: string) => fileURLToPath(new URL(`../fixtures/${name}`, import.meta.url));
type Opened = Awaited<ReturnType<typeof opened>>;
const sets: Opened[] = [];
after(() => {
  for (const h of sets) {
    h.effects.dispose();
    h.host.close();
  }
});
async function openSet(name: string): Promise<Opened> {
  const h = await opened(fx(`${name}.h5`), fx(`${name}.results.h5`));
  sets.push(h);
  return h;
}

/** v1 warns on `extrap_plane` (no /opensees zone, so none of its 15 quads is classified); test/contour/derived.test.ts pins the manifest's copy. */
const EXTRAP_PLANE_LINE =
  "Out-of-plane stress_zz recovery is incomplete: 15 of 15 element(s) could not be classified as plane stress / plane strain. stress_zz is left at 0 there.";
const PLANE_LINE = /^Out-of-plane .*_zz recovery is incomplete/;

/** The banner mounted on a real store in a fake document; `lines()` reads its rows after the summary. */
function mounted(store: Opened["store"], planeLine?: Parameters<typeof mountBanner>[1]) {
  const banner = new FakeEl();
  const prevDoc = (globalThis as { document?: unknown }).document;
  (globalThis as { document?: unknown }).document = { getElementById: () => banner, createElement: () => new FakeEl() };
  const restore = () => {
    (globalThis as { document?: unknown }).document = prevDoc;
  };
  try {
    const unmount = mountBanner(store as never, planeLine);
    return {
      lines: () => banner.children.slice(1).map((c) => c.textContent),
      close: () => {
        unmount();
        restore();
      },
    };
  } catch (err) {
    restore();
    throw err;
  }
}

function shotOf(shot: Record<string, unknown>): SpecShot {
  const v = validateSpec({ spec: "apegmsh-view/1", files: { model: "m.h5" }, shots: [{ name: "s", ...shot }] });
  if (!v.ok) throw new Error(`${v.key}: ${v.reason}`);
  return v.spec.shots[0]!;
}

const pick = (h: Opened, component: string, step: number) => {
  for (const e of [
    { type: "setColourBy", by: "contour" },
    { type: "setPhase", at: { kind: "results", stage: 0, step } },
    { type: "setContourField", field: "stress", component, source: "element" },
  ] as Event[]) {
    h.store.dispatch(e);
  }
};
const shown = (h: Opened, component: string, step: number) => () => {
  const v = contourView(h.store.get());
  const at = h.store.get().phase.at;
  return v.state === "element" && v.component.key === component && at?.kind === "results" && at.step === step && h.effects.currentStep()?.step === step;
};

test("VF2: a derived contour on unclassified elements puts the manifest's plane line on the banner; a stored component, another step and contours off follow", async () => {
  const h = await openSet("extrap_plane");
  const b = mounted(h.store, (s) => derivedPlaneOf(s, h.blobs)?.warning ?? null);
  try {
    pick(h, "von_mises_stress", 1);
    await until(shown(h, "von_mises_stress", 1), "von_mises_stress at step 1");
    assert.deepEqual(b.lines().filter((l) => PLANE_LINE.test(l)), [EXTRAP_PLANE_LINE], b.lines().join("\n"));
    // One wording, built once: the manifest's `contour.derived.warning` and its banner list are the banner's line.
    const entry = manifestEntry(h.store.get(), {
      shot: shotOf({ stage: "load", step: 1, contour: { field: "stress", component: "von_mises_stress" } }),
      png: "s.png",
      image: [1600, 1000],
      camera: { eye: [1, 1, 1], target: [0, 0, 0], up: [0, 0, 1], fov: 30 },
      layout: null,
      blobs: h.blobs,
      step: h.effects.currentStep(),
    });
    assert.ok(entry.contour?.drawn);
    assert.equal(entry.contour.drawn && entry.contour.derived?.warning, EXTRAP_PLANE_LINE);
    assert.deepEqual(entry.banner.filter((l) => PLANE_LINE.test(l)), [EXTRAP_PLANE_LINE]);
    // A stored component of the same field: no derived statement, so no line (never a stale one).
    const storedKey = contourFieldsOf(h.store.get())!.elements.find((f) => f.name === "stress")!.components.find((c) => !c.derived)!.key;
    pick(h, storedKey, 1);
    await until(shown(h, storedKey, 1), `${storedKey} at step 1`);
    assert.deepEqual(b.lines().filter((l) => PLANE_LINE.test(l)), [], b.lines().join("\n"));
    assert.equal(derivedPlaneOf(h.store.get(), h.blobs), null);
    // Back to von Mises on another step: the line is recomputed, once.
    pick(h, "von_mises_stress", 2);
    await until(shown(h, "von_mises_stress", 2), "von_mises_stress at step 2");
    assert.deepEqual(b.lines().filter((l) => PLANE_LINE.test(l)), [EXTRAP_PLANE_LINE], b.lines().join("\n"));
    // Contours off: no line.
    h.store.dispatch({ type: "setColourBy", by: "role" } as Event);
    assert.deepEqual(b.lines().filter((l) => PLANE_LINE.test(l)), [], b.lines().join("\n"));
  } finally {
    b.close();
  }
  // A banner mounted without the line refuses a derived contour, never a silent drop.
  pick(h, "von_mises_stress", 1);
  await until(shown(h, "von_mises_stress", 1), "von_mises_stress again");
  assert.throws(() => mounted(h.store), /mounted without its out-of-plane line/);
});

test("VF3 (#2959): a shot refused after its view was drawn states the same banner as the drawn shot, the plane line included", async () => {
  const h = await openSet("extrap_plane");
  const frame = globalThis.requestAnimationFrame;
  globalThis.requestAnimationFrame = ((cb: FrameRequestCallback) => setTimeout(() => cb(0), 0)) as unknown as typeof requestAnimationFrame;
  const canvas = { width: 640, height: 480, getBoundingClientRect: () => ({ x: 0, y: 0, width: 640, height: 480 }) };
  // The render mode's own path (renderer/agentRender.ts `takeShots`), with a viewport that has applied whatever element contour the state shows.
  const deps = {
    store: h.store,
    blobs: h.blobs,
    effects: h.effects,
    viewport: {
      renderer: { domElement: canvas },
      camera: {},
      get contourApplied() {
        const v = contourView(h.store.get());
        return v.state === "element" ? elementContourKey(v) : null;
      },
      deformApplied: null,
      renderNow: () => {},
      currentPose: () => ({ eye: [1, 1, 1], target: [0, 0, 0], up: [0, 0, 1] }),
    },
    bridge: { renderCapture: async (name: string) => ({ path: name, width: 640, height: 480 }) },
    layout: { apply: () => {}, inLayout: () => [] },
    combination: () => null,
  } as unknown as RenderDeps;
  try {
    const vm = { stage: "load", step: 1, camera: "top", contour: { field: "stress", component: "von_mises_stress", gauss: "node" } };
    // extrap_plane records no nodal displacement, so the second shot is refused once its contour is drawn (the D7 spec's `vm_refused`).
    const v = validateSpec({ spec: "apegmsh-view/1", files: { model: "m.h5" }, size: [640, 480], shots: [{ name: "vm", ...vm }, { name: "vm_deform", ...vm, deform: { scale: 1 } }] });
    if (!v.ok) throw new Error(`${v.key}: ${v.reason}`);
    const [ok, refused] = await takeShots(deps, v.spec, describeSet(h.store.get(), []), null);
    assert.equal(ok!.entry.status, "ok", ok!.entry.reason ?? "");
    assert.equal(refused!.entry.status, "refused");
    assert.match(refused!.entry.reason!, /the deformed shape is not drawn: stage load records no nodal displacement/);
    assert.deepEqual(ok!.entry.banner.filter((l) => PLANE_LINE.test(l)), [EXTRAP_PLANE_LINE]);
    assert.deepEqual(refused!.entry.banner, ok!.entry.banner, "the refused entry's banner is the drawn entry's, line for line");
  } finally {
    globalThis.requestAnimationFrame = frame;
  }
});

test("VF3 (#2959): a refused entry whose plane line cannot be stated throws, and keeps the refusal's reason", async () => {
  const h = await openSet("extrap_plane");
  pick(h, "von_mises_stress", 1);
  await until(shown(h, "von_mises_stress", 1), "von_mises_stress at step 1");
  const at = { stage: null, step: 1, time: null };
  const shot = shotOf({ stage: "load", step: 1, contour: { field: "stress", component: "von_mises_stress" } });
  // A store that holds no element ids: never a banner silently short of its plane line.
  const empty = { get: () => undefined, i32: () => undefined } as unknown as typeof h.blobs;
  assert.throws(() => refusedEntry(h.store.get(), shot, "the view did not settle", at, empty), /banner cannot be stated .*the shot was refused: the view did not settle/);
  // A shot that did not compile (`at` null) reports no view, so it needs no ids.
  assert.deepEqual(refusedEntry(h.store.get(), shot, "no", null, empty).banner, []);
});

test("VF2: the statement is computed once across steps and the manifest, and again for a new set (elem_mixed: PlaneStrain, nu 0.2), which leaves no stale plane line", async () => {
  const h = await openSet("extrap_plane");
  // planeStatement returns a new object on every run, so the distinct statements seen count its runs (#2952 review).
  const runs = new Set<object>();
  let asked = 0;
  const b = mounted(h.store, (s) => {
    const d = derivedPlaneOf(s, h.blobs);
    if (d) {
      asked++;
      runs.add(d.plane);
    }
    return d?.warning ?? null;
  });
  try {
    for (const step of [0, 1, 2, 1]) {
      pick(h, "von_mises_stress", step);
      await until(shown(h, "von_mises_stress", step), `extrap_plane's von_mises_stress at step ${step}`);
      assert.deepEqual(b.lines().filter((l) => PLANE_LINE.test(l)), [EXTRAP_PLANE_LINE]);
    }
    const entry = manifestEntry(h.store.get(), {
      shot: shotOf({ stage: "load", step: 1, contour: { field: "stress", component: "von_mises_stress" } }),
      png: "s.png",
      image: [1600, 1000],
      camera: { eye: [1, 1, 1], target: [0, 0, 0], up: [0, 0, 1], fov: 30 },
      layout: null,
      blobs: h.blobs,
      step: h.effects.currentStep(),
    });
    assert.ok(entry.contour?.drawn && entry.contour.derived);
    runs.add(entry.contour.derived.plane);
    assert.ok(asked >= 4, `the banner asked ${asked} time(s) over four steps`);
    assert.equal(runs.size, 1, "one statement for every step, the banner and the manifest");
    // The same window opens another set: its model classifies every quad and triangle, so its zz is recovered.
    assert.ok(await h.effects.open(fx("elem_mixed.h5")));
    assert.ok(await h.effects.openResults(fx("elem_mixed.results.h5")));
    pick(h, "von_mises_stress", 1);
    await until(shown(h, "von_mises_stress", 1), "von_mises_stress at step 1");
    assert.deepEqual(derivedPlaneOf(h.store.get(), h.blobs)?.plane, { elements: 15, recorded: 0, recovered: 15, zero_exact: 0, unclassified: 0, no_nu: 0 });
    assert.equal(runs.size, 2, "the new set's statement is computed again");
    assert.deepEqual(b.lines().filter((l) => PLANE_LINE.test(l)), [], b.lines().join("\n"));
  } finally {
    b.close();
  }
});

// ---- M2-S3 (#2982): particles read without a model (ADR 0134 D1, D3) ----
// Before this slice an all-particle .ladruno opened alone said "Results not drawn: no model
// is open". The info is the one main hands the renderer (main/results.ts `metaOf`) of the
// fork's fixtures; the banner is mounted on a real store with no model open.

await h5wasm.ready;
const mpm = (name: string) => fileURLToPath(new URL(`../../tests/fixtures/mpm/${name}`, import.meta.url));

function metaOfFixture(name: string) {
  const f = openLadruno(h5wasm as unknown as ResultsH5, mpm(name));
  try {
    return metaOf(f);
  } finally {
    f.close();
  }
}

/** The banner rows of a store that has read `info` as its results, with no model open. */
function bannerLines(info: ReturnType<typeof metaOfFixture>): string[] {
  const store = new Store();
  store.dispatch({ type: "fileOpened", artifact: "results", path: info.path } as Event);
  store.dispatch({ type: "resultsLoaded", info, sizeBytes: 0, readMs: 0, warnings: [] } as Event);
  const b = mounted(store as never);
  try {
    return b.lines();
  } finally {
    b.close();
  }
}

const PARTICLES_ONLY = (name: string) => `No model is open for ${name}: it holds particles only, and particles read without a model (ADR 0134).`;

test("M2-S3: an all-particle .ladruno opened with no model says its particles read without one, not \"Results not drawn\"", () => {
  const info = metaOfFixture("particles_v1m2.ladruno");
  assert.deepEqual(pairResults(null, info), { state: "noModel", particlesOnly: true });
  const lines = bannerLines(info);
  assert.ok(lines.includes(PARTICLES_ONLY("particles_v1m2.ladruno")), lines.join("\n"));
  assert.ok(!lines.some((l) => /Results not drawn/.test(l)), lines.join("\n"));
});

test("M2-S3: a .ladruno with an FE stage keeps today's sentence: its FE results need the model", () => {
  const info = metaOfFixture("particles_mixed_v1m2.ladruno");
  assert.deepEqual(pairResults(null, info), { state: "noModel" });
  const lines = bannerLines(info);
  assert.ok(lines.includes("Results not drawn: no model is open for particles_mixed_v1m2.ladruno; open the model.h5 the run was emitted from."), lines.join("\n"));
});

test("M2-S3: a <stem>.particles.h5 with no model reads the particle sentence; with a model the pairing is the lineage's, as before", () => {
  assert.equal(pairingNotice({ state: "noModel" }, "case1.particles.h5"), PARTICLES_ONLY("case1.particles.h5"));
  assert.equal(pairingNotice({ state: "noModel" }, "Case1.PARTICLES.H5"), PARTICLES_ONLY("Case1.PARTICLES.H5"));
  // A model open: particlesOnly never applies (pairing is the lineage's; this unstamped fixture is unpairable).
  const info = metaOfFixture("particles_v1m2.ladruno");
  assert.equal(pairResults({ lineage: null, openseesVersion: null, programDecl: null }, info).state, "unpairable");
});
