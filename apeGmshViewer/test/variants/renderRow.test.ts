// VU18 (#2856; ADR 0132 D9.2, D9.3; ADR 0128 D2, D4): a render shot names a
// variant row of the spec's `files.variants` by index and label (`variant`),
// and one `--render` call covers a study. Each piece on its own oracle:
// - the spec (spec/spec.ts): the key's shape is refused whole, naming the key
//   (D2); a spec without it validates exactly as before (shared namespace);
// - main's `Study.rowNow` (main/session.ts): the manifest is read again for
//   every shot that names a row (D9.3), so a rewrite between two shots is
//   seen; a label that differs only in case, an index out of range, a refused
//   row and a row that no longer resolves inside the study are refused;
// - the render mode (renderer/agentRender.ts `takeShots`) over the committed
//   VU10 pair (test/fixtures/vu10): each shot draws the set main names for its
//   row, read through the real Effects; a refused row writes no PNG and its
//   reason is the entry's; a shot without a row draws the spec's set again.
// Main's door itself (main/modes/render.ts `render:variant`) is judged in
// test/main/renderVariant.test.ts.

import assert from "node:assert/strict";
import { join, resolve } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import { Study } from "../../src/main/session.ts";
import type { VariantRow, VariantsRead } from "../../src/reader/variants.ts";
import { takeShots, type RenderDeps, type VariantAnswer } from "../../src/renderer/agentRender.ts";
import { describeSet } from "../../src/spec/describe.ts";
import { SPEC_VERSION, validateSpec, type SpecVariant } from "../../src/spec/spec.ts";
import type { VariantsIn } from "../../src/state/types.ts";
import { opened } from "../contour/harness.ts";

// ---- the spec ----------------------------------------------------------------------

const doc = (shot: Record<string, unknown>, variants: string | null = "s.variants.json", more: Record<string, unknown>[] = []) => ({
  spec: SPEC_VERSION,
  files: { model: "u.h5", ...(variants ? { variants } : {}) },
  shots: [{ name: "a", ...shot }, ...more],
});
const verdict = (raw: unknown) => {
  const v = validateSpec(raw);
  return v.ok ? "ok" : `${v.key}: ${v.reason}`;
};

test("the spec: a shot names a variant row by { index, label }, and the normalised shot states it", () => {
  const v = validateSpec(doc({ variant: { index: 1, label: "Cracked" } }));
  assert.ok(v.ok, JSON.stringify(v));
  assert.deepEqual(v.ok && v.spec.shots[0]!.variant, { index: 1, label: "Cracked" });
  // null is none, as compare's null is.
  const none = validateSpec(doc({ variant: null }));
  assert.ok(none.ok);
  assert.equal(none.ok && "variant" in none.spec.shots[0]!, false);
});

test("shared namespace: a spec without the key validates exactly as before, with no variant key in any shot", () => {
  const plain = validateSpec(doc({ stage: 0, contour: { field: "DISPLACEMENT", component: "y" } }));
  assert.ok(plain.ok);
  assert.equal(JSON.stringify(plain.spec).includes("variant\""), false, "no `variant` key is added to a shot that names none");
  // The key is a shot's only: `files` and the spec's top level still refuse it.
  assert.match(verdict({ ...doc({}), files: { model: "u.h5", variant: "x" } }), /^files\.variant: is not a key/);
  assert.match(verdict({ ...doc({}), variant: { index: 0, label: "U" } }), /^variant: is not a key/);
});

test("D2: the key's shape refuses the whole spec, naming the key", () => {
  assert.match(verdict(doc({ variant: { index: 1, label: "Cracked" } }, null)), /^shots\[0\]\.variant: names a variant row, and the spec has no files\.variants/);
  assert.match(verdict(doc({ variant: { index: -1, label: "Cracked" } })), /^shots\[0\]\.variant\.index: must be a row index .* got -1/);
  assert.match(verdict(doc({ variant: { index: 1.5, label: "Cracked" } })), /^shots\[0\]\.variant\.index: must be a row index/);
  assert.match(verdict(doc({ variant: { index: "1", label: "Cracked" } })), /^shots\[0\]\.variant\.index: must be a row index/);
  assert.match(verdict(doc({ variant: { index: 1, label: "" } })), /^shots\[0\]\.variant\.label: must be a non-empty string/);
  assert.match(verdict(doc({ variant: { index: 1 } })), /^shots\[0\]\.variant\.label: is required/);
  assert.match(verdict(doc({ variant: { label: "Cracked" } })), /^shots\[0\]\.variant\.index: is required/);
  assert.match(verdict(doc({ variant: { index: 1, label: "Cracked", path: "c.h5" } })), /^shots\[0\]\.variant\.path: is not a key/);
  assert.match(verdict(doc({ variant: "cracked/plate_cracked.h5" })), /^shots\[0\]\.variant: must be an object/);
  assert.match(verdict(doc({ variant: 1 })), /^shots\[0\]\.variant: must be an object/);
});

test("D2: a shot never compares with the row it draws, and a spec that draws rows shows no user file", () => {
  const contour = { stage: 0, contour: { field: "DISPLACEMENT", component: "y" } };
  assert.match(verdict(doc({ ...contour, variant: { index: 1, label: "Cracked" }, compare: "Cracked" })), /^shots\[0\]\.compare: is "Cracked", the variant the shot draws/);
  assert.equal(verdict(doc({ ...contour, variant: { index: 1, label: "Cracked" }, compare: "Uncracked" })), "ok");
  // The render mode reads the spec set's `.user.h5` once and never a row's: no shot of such a spec may show one.
  assert.match(verdict(doc({ variant: { index: 1, label: "Cracked" } }, "s.variants.json", [{ name: "b", combination: "D+L" }])), /^shots\[1\]\.combination: shots\[0\] names a variant row/);
  assert.match(verdict(doc({ variant: { index: 1, label: "Cracked" } }, "s.variants.json", [{ name: "b", contour: { field: "U", component: "x", source: "user" } }])), /^shots\[1\]\.contour\.source: shots\[0\] names a variant row/);
  // Without a row, a combination shot is as before.
  assert.equal(verdict(doc({ combination: "D+L" })), "ok");
});

// ---- main's Study.rowNow ----------------------------------------------------------

const M = resolve("/study/s.variants.json");
const row = (index: number, label: string, path: string): VariantRow => ({ index, label, path, resolved: join(resolve("/study"), path), real: join(resolve("/study"), path), params: {} });
const readOf = (rows: VariantRow[], refused: Extract<VariantsRead, { ok: true }>["refused"] = []): VariantsRead => ({ ok: true, manifest: M, version: "1.0.0", newer: null, rows, refused });

function studyOver(first: VariantsRead) {
  let next = first;
  let reads = 0;
  let recheck: (r: VariantRow) => { ok: true; real: string } | { ok: false; code: string; reason: string } = (r) => ({ ok: true, real: r.real });
  const study = new Study({
    read: () => (reads++, next),
    // An agent run watches nothing (main.ts): only rowNow reads again.
    watch: () => ({ close: () => undefined }),
    note: () => undefined,
    recheck: (_m, r) => recheck(r),
    confined: () => null,
  });
  assert.ok(study.open(M).ok);
  return {
    study,
    reads: () => reads,
    rewrite: (r: VariantsRead) => (next = r),
    setRecheck: (f: typeof recheck) => (recheck = f),
  };
}

test("reload: a manifest rewritten between two shots is read again for the second, never the first shot's rows", () => {
  const s = studyOver(readOf([row(0, "Uncracked", "u.h5"), row(1, "Cracked", "c.h5")]));
  const a = s.study.rowNow(1, "Cracked");
  assert.ok(a.ok, JSON.stringify(a));
  assert.equal(a.ok && a.open, row(1, "Cracked", "c.h5").real);
  const before = s.reads();
  // The writer relabels row 1 and points it at another file.
  s.rewrite(readOf([row(0, "Uncracked", "u.h5"), row(1, "Cracked x2", "c2.h5")]));
  const b = s.study.rowNow(1, "Cracked");
  assert.equal(s.reads(), before + 1, "the manifest is read again for the shot");
  assert.equal(b.ok, false);
  assert.match(!b.ok ? b.reason : "", /variant row 1 is "Cracked x2", not "Cracked"/);
  const c = s.study.rowNow(1, "Cracked x2");
  assert.ok(c.ok);
  assert.equal(c.ok && c.open, row(1, "Cracked x2", "c2.h5").real, "the rewritten row's file, not the first read's");
});

test("a manifest refused on re-read refuses the row with the file's reason, never the rows read before", () => {
  const s = studyOver(readOf([row(0, "Uncracked", "u.h5"), row(1, "Cracked", "c.h5")]));
  s.rewrite({ ok: false, manifest: M, code: "bad-json", reason: `${M}: is not a JSON document (torn write)` });
  const r = s.study.rowNow(1, "Cracked");
  assert.equal(r.ok, false);
  assert.match(!r.ok ? r.reason : "", /refused on re-read \(bad-json\): .*torn write/);
});

test("degenerate names: a label that differs only in case, an index out of range or negative, a refused row, a non-integer", () => {
  const s = studyOver(readOf([row(0, "Uncracked", "u.h5"), row(2, "Cracked", "c.h5")], [{ index: 1, label: "Bad", code: "escapes", reason: "variants[1]: path \"../x.h5\" resolves outside the manifest's folder" }]));
  const why = (index: unknown, label: unknown) => {
    const r = s.study.rowNow(index, label);
    return r.ok ? "ok" : r.reason;
  };
  assert.match(why(2, "cracked"), /variant row 2 is "Cracked", not "cracked"/);
  assert.match(why(7, "Cracked"), /the manifest has no variant row 7/);
  assert.match(why(-1, "Cracked"), /the manifest has no variant row -1/);
  assert.match(why(1, "Bad"), /variant row 1 is refused \(escapes: .*outside the manifest's folder\)/);
  assert.match(why(1.5, "Cracked"), /named by its row index and label/);
  assert.match(why(2, ""), /variant row 2 is "Cracked", not ""/);
  assert.equal(why(2, "Cracked"), "ok");
});

test("D9 confinement: a row that no longer resolves inside the study since the read is refused, not followed", () => {
  const s = studyOver(readOf([row(0, "Uncracked", "u.h5"), row(1, "Cracked", "c.h5")]));
  s.setRecheck(() => ({ ok: false, code: "escapes", reason: "path \"c.h5\" resolves to D:\\elsewhere\\c.h5, outside the manifest's folder" }));
  const r = s.study.rowNow(1, "Cracked");
  assert.equal(r.ok, false);
  assert.match(!r.ok ? r.reason : "", /variant row 1 \("Cracked"\) no longer opens \(escapes\)/);
});

// ---- the render mode over the VU10 pair -----------------------------------------------

const FIX = (p: string) => fileURLToPath(new URL(`../fixtures/vu10/${p}`, import.meta.url));
const SETS = {
  Uncracked: { model: FIX("plate_uncracked.h5"), geometry: null, results: FIX("plate_uncracked.ladruno") },
  Cracked: { model: FIX("cracked/plate_cracked.h5"), geometry: null, results: FIX("cracked/plate_cracked.ladruno") },
};
const STUDY = (active: number): VariantsIn => ({
  manifest: FIX("study.variants.json"),
  version: "1.0.0",
  newer: null,
  rows: [
    { index: 0, label: "Uncracked", path: "plate_uncracked.h5", params: {} },
    { index: 1, label: "Cracked", path: "cracked/plate_cracked.h5", params: {} },
  ],
  refused: [],
  active,
  compare: null,
});

test("one render call: each shot draws its row's set, a refused row writes no PNG and states why, and a shot without a row draws the spec's set again", async () => {
  // The spec's set (row 0) is open, as runAgentMode opens it before the shots.
  const h = await opened(SETS.Uncracked.model, SETS.Uncracked.results);
  const frame = globalThis.requestAnimationFrame;
  globalThis.requestAnimationFrame = ((cb: FrameRequestCallback) => setTimeout(() => cb(0), 0)) as unknown as typeof requestAnimationFrame;
  const written: string[] = [];
  const asked: (SpecVariant | null)[] = [];
  let active = 0;
  const canvas = { width: 640, height: 480, getBoundingClientRect: () => ({ x: 0, y: 0, width: 640, height: 480 }) };
  // Main's door, as main/modes/render.ts answers it on this study: a row by index and label, exactly.
  const renderVariant = async (want: SpecVariant | null): Promise<VariantAnswer> => {
    asked.push(want);
    if (want === null) return ((active = 0), { ok: true, row: null, set: SETS.Uncracked });
    const rows = STUDY(active).rows;
    const r = rows.find((x) => x.index === want.index);
    if (!r) return { ok: false, reason: `render:variant: the manifest has no variant row ${want.index}` };
    if (r.label !== want.label) return { ok: false, reason: `render:variant: variant row ${r.index} is ${JSON.stringify(r.label)}, not ${JSON.stringify(want.label)}` };
    active = r.index;
    return { ok: true, row: { index: r.index, label: r.label }, set: SETS[r.label as keyof typeof SETS] };
  };
  const deps = {
    store: h.store,
    blobs: h.blobs,
    effects: h.effects,
    viewport: { renderer: { domElement: canvas }, camera: {}, renderNow: () => {}, currentPose: () => ({ eye: [1, 1, 1], target: [0, 0, 0], up: [0, 0, 1] }) },
    bridge: {
      renderCapture: async (name: string) => (written.push(name), { path: name, width: 640, height: 480 }),
      renderVariant,
      variants: async () => STUDY(active),
    },
    layout: { apply: () => undefined, inLayout: () => [] },
    combination: () => null,
  } as unknown as RenderDeps;
  try {
    const v = validateSpec({
      spec: SPEC_VERSION,
      files: { model: SETS.Uncracked.model, variants: FIX("study.variants.json") },
      size: [640, 480],
      shots: [
        { name: "cracked", variant: { index: 1, label: "Cracked" } },
        { name: "cracked-again", variant: { index: 1, label: "Cracked" } },
        { name: "wrong-case", variant: { index: 1, label: "cracked" } },
        { name: "no-row", variant: { index: 5, label: "Cracked" } },
        { name: "spec-set" },
        { name: "uncracked", variant: { index: 0, label: "Uncracked" } },
      ],
    });
    assert.ok(v.ok, JSON.stringify(v));
    const shots = await takeShots(deps, v.spec, describeSet(h.store.get(), []), null);
    const model = (i: number) => shots[i]!.entry.files.model?.path ?? null;
    assert.deepEqual(
      shots.map((s) => [s.entry.name, s.entry.status, s.entry.png]),
      [
        ["cracked", "ok", "cracked.png"],
        ["cracked-again", "ok", "cracked-again.png"],
        ["wrong-case", "refused", null],
        ["no-row", "refused", null],
        ["spec-set", "ok", "spec-set.png"],
        ["uncracked", "ok", "uncracked.png"],
      ],
    );
    // Drawn from its row's set: the entry's files are the row's, read through the effects.
    assert.equal(model(0), SETS.Cracked.model);
    assert.equal(shots[0]!.entry.files.results?.path, SETS.Cracked.results);
    assert.equal(model(1), SETS.Cracked.model);
    assert.equal(model(4), SETS.Uncracked.model, "a shot without a row draws the spec's set again");
    assert.equal(shots[4]!.entry.files.results?.path, SETS.Uncracked.results);
    assert.equal(model(5), SETS.Uncracked.model);
    assert.notEqual(shots[0]!.entry.files.model?.model_hash, shots[4]!.entry.files.model?.model_hash, "the two variants are two models");
    // Partial writes: a refused row has no PNG, and its entry carries main's reason.
    assert.deepEqual(written, ["cracked", "cracked-again", "spec-set", "uncracked"]);
    assert.match(shots[2]!.entry.reason!, /^variant row 1 \("cracked"\) is not drawn: render:variant: variant row 1 is "Cracked", not "cracked"/);
    assert.match(shots[3]!.entry.reason!, /^variant row 5 \("Cracked"\) is not drawn: .*no variant row 5/);
    // D9.3: every shot that names a row asks main again (the manifest is read for it), the same row twice included;
    // the spec's set is asked for back after a failed switch.
    assert.deepEqual(asked, [{ index: 1, label: "Cracked" }, { index: 1, label: "Cracked" }, { index: 1, label: "cracked" }, { index: 5, label: "Cracked" }, null, { index: 0, label: "Uncracked" }]);
    assert.equal(h.store.get().variants.study?.active, 0, "the page's study follows the row main drew");
  } finally {
    globalThis.requestAnimationFrame = frame;
    h.effects.dispose();
    h.host.close();
  }
});

test("a spec that names no row asks main for no set: its shots render exactly as before", async () => {
  const h = await opened(SETS.Uncracked.model, SETS.Uncracked.results);
  const frame = globalThis.requestAnimationFrame;
  globalThis.requestAnimationFrame = ((cb: FrameRequestCallback) => setTimeout(() => cb(0), 0)) as unknown as typeof requestAnimationFrame;
  const canvas = { width: 640, height: 480, getBoundingClientRect: () => ({ x: 0, y: 0, width: 640, height: 480 }) };
  const deps = {
    store: h.store,
    blobs: h.blobs,
    effects: h.effects,
    viewport: { renderer: { domElement: canvas }, camera: {}, renderNow: () => {}, currentPose: () => ({ eye: [1, 1, 1], target: [0, 0, 0], up: [0, 0, 1] }) },
    bridge: {
      renderCapture: async (name: string) => ({ path: name, width: 640, height: 480 }),
      renderVariant: async () => assert.fail("a spec with no row asks for no set"),
      variants: async () => assert.fail("a spec with no row reads no study"),
    },
    layout: { apply: () => undefined, inLayout: () => [] },
    combination: () => null,
  } as unknown as RenderDeps;
  try {
    const v = validateSpec({ spec: SPEC_VERSION, files: { model: SETS.Uncracked.model }, size: [640, 480], shots: [{ name: "a" }, { name: "b" }] });
    assert.ok(v.ok);
    const shots = await takeShots(deps, v.spec, describeSet(h.store.get(), []), null);
    assert.deepEqual(shots.map((s) => s.entry.status), ["ok", "ok"]);
  } finally {
    globalThis.requestAnimationFrame = frame;
    h.effects.dispose();
    h.host.close();
  }
});
