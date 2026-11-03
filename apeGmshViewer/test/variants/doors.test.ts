// VU10 (#2694; ADR 0132 D9): the pieces around the variant doors, each on its
// own oracle:
// - the view spec's `files.variants` and `compare` (spec/spec.ts): a compare
//   needs the manifest, a nodal contour of a stage, and no combination
//   (a combination across variants is refused) and no deformed shape;
// - the compiler (spec/compile.ts) asks for the compare set by row and turns
//   the difference on; the base view turns it off;
// - `variantRowOf` refuses the window's own row and a label the study lacks;
// - main's `Study` (main/session.ts) re-reads the same path and drops a compare
//   row the new read no longer holds with its label and file;
// - the variants panel's lines (panels/variants.ts);
// - the preload sends a compare step's stage, step and request number only.

import assert from "node:assert/strict";
import { registerHooks } from "node:module";
import { join, resolve } from "node:path";
import { test } from "node:test";
import { Study } from "../../src/main/session.ts";
import { paramsLine, variantLines } from "../../src/panels/variants.ts";
import type { VariantsRead } from "../../src/reader/variants.ts";
import { BASE_EVENTS, compileShot } from "../../src/spec/compile.ts";
import type { SetDescription } from "../../src/spec/describe.ts";
import { validateSpec } from "../../src/spec/spec.ts";
import { initialState, initialVariants, reduce } from "../../src/state/reduce.ts";
import { variantRowOf } from "../../src/state/selectors.ts";
import type { VariantsIn } from "../../src/state/types.ts";

const STUDY: VariantsIn = {
  manifest: resolve("/study/s.variants.json"),
  version: "1.0.0",
  newer: null,
  rows: [
    { index: 0, label: "Uncracked", path: "u.h5", params: { "Ie/Ig": 1, state: "uncracked" } },
    { index: 2, label: "Cracked", path: "sub/c.h5", params: { "Ie/Ig": 0.35 } },
  ],
  refused: [{ index: 1, label: "Bad", code: "bad-path", reason: "variants[1]: path is absolute" }],
  active: 0,
  compare: null,
};

const shot = { name: "d", stage: 0, contour: { field: "DISPLACEMENT", component: "y" } };
const spec = (s: Record<string, unknown>, variants: string | null = "s.variants.json") =>
  validateSpec({ spec: "apegmsh-view/1", files: { model: "u.h5", ...(variants ? { variants } : {}) }, shots: [{ ...shot, ...s }] });

test("the spec: a compare needs files.variants, a nodal contour of a stage, and never a combination or a deformed shape", () => {
  const ok = spec({ compare: "Cracked" });
  assert.ok(ok.ok, JSON.stringify(ok));
  assert.equal(ok.ok && ok.spec.shots[0]!.compare, "Cracked");
  assert.equal(ok.ok && ok.spec.files.variants, "s.variants.json");
  const key = (v: ReturnType<typeof spec>) => (v.ok ? "ok" : `${v.key}: ${v.reason}`);
  assert.match(key(spec({ compare: "Cracked" }, null)), /^shots\[0\]\.compare: .*no files\.variants/);
  assert.match(key(spec({ compare: "Cracked", stage: undefined, combination: "D+L" })), /^shots\[0\]\.compare: a combination across variants is refused/);
  assert.match(key(spec({ compare: "Cracked", deform: { scale: 1 } })), /^shots\[0\]\.deform: a difference is never a deformed shape/);
  assert.match(key(spec({ compare: "Cracked", contour: undefined })), /^shots\[0\]\.compare: needs a contour/);
  assert.match(key(spec({ compare: "Cracked", contour: { field: "S", component: "xx", source: "element" } })), /nodal component/);
  assert.match(key(spec({}, "s.json")), /^files\.variants: must name a <study>\.variants\.json/);
});

const d = { stages: [], groups: [], files: { results: "u.ladruno" }, pairing: { state: "paired", reason: null } } as unknown as SetDescription;

test("the compiler asks for the compare set by row and turns the difference on; every shot starts with it off", () => {
  assert.deepEqual(BASE_EVENTS.at(-1), { type: "setDifference", on: false });
  const v = spec({ compare: "Cracked", stage: null as unknown as number, contour: undefined, colour_by: "group" });
  assert.equal(v.ok, false, "a compare with no stage is refused by the validator first");
  // The compiler on its own: a shot whose stage the description lacks is refused for that; here only the compare branch is judged.
  const s = spec({ compare: "Cracked" });
  assert.ok(s.ok);
  const noStudy = compileShot({ ...s.spec.shots[0]!, stage: null, contour: null, colour_by: "group" }, d, undefined, undefined);
  assert.equal(noStudy.ok, false);
  assert.match(!noStudy.ok ? noStudy.reason : "", /no variants manifest/);
  const c = compileShot({ ...s.spec.shots[0]!, stage: null, contour: null, colour_by: "group" }, d, undefined, () => ({ ok: true, row: { index: 2, label: "Cracked" }, open: false }));
  assert.ok(c.ok);
  const tail = c.ok ? c.plan.events.slice(-2) : [];
  assert.deepEqual(tail, [{ type: "variantCompare", row: { index: 2, label: "Cracked" } }, { type: "setDifference", on: true }]);
  const again = compileShot({ ...s.spec.shots[0]!, stage: null, contour: null, colour_by: "group" }, d, undefined, () => ({ ok: true, row: { index: 2, label: "Cracked" }, open: true }));
  assert.deepEqual(again.ok ? again.plan.events.slice(-1) : [], [{ type: "setDifference", on: true }], "an open compare set is not opened again");
});

test("variantRowOf: the window's own row, a refused row and an unknown label are refused, each with why", () => {
  const s = reduce(initialState, { type: "variantsRead", study: STUDY });
  assert.deepEqual(variantRowOf(s, "Cracked"), { ok: true, row: { index: 2, label: "Cracked" }, open: false });
  assert.match((variantRowOf(s, "Uncracked") as { reason: string }).reason, /is the window set/);
  assert.match((variantRowOf(s, "Bad") as { reason: string }).reason, /is refused \(bad-path/);
  assert.match((variantRowOf(s, "Nope") as { reason: string }).reason, /no variant "Nope"; its variants are "Uncracked", "Cracked"/);
  assert.match((variantRowOf(initialState, "Cracked") as { reason: string }).reason, /no variants manifest is open/);
});

test("Study: a re-read of the same path drops a compare row whose label or file moved, and keeps one that stayed", () => {
  const m = resolve("/study/s.variants.json");
  const row = (index: number, label: string, path: string) => ({ index, label, path, resolved: join(resolve("/study"), path), params: {} });
  let next: VariantsRead = { ok: true, manifest: m, version: "1.0.0", newer: null, rows: [row(0, "U", "u.h5"), row(1, "C", "c.h5")], refused: [] };
  const changes: string[] = [];
  let watched: (() => void) | null = null;
  const study = new Study({ read: () => next, watch: (_m, cb) => ((watched = cb), { close: () => changes.push("closed") }), note: (x) => changes.push(x) });
  study.onChange = () => changes.push("changed");
  assert.ok(study.open(m).ok);
  study.compare = { row: row(1, "C", "c.h5"), set: { model: row(1, "C", "c.h5").resolved, geometry: null, results: null, user: null } };
  watched!();
  assert.notEqual(study.compare, null, "the same row, label and file: kept");
  next = { ...next, rows: [row(0, "U", "u.h5"), row(1, "C2", "c.h5")] };
  watched!();
  assert.equal(study.compare, null, "a relabelled row leaves the slot");
  assert.deepEqual(study.view(row(0, "U", "u.h5").resolved)?.rows.map((r) => r.label), ["U", "C2"]);
  assert.equal(study.view(row(0, "U", "u.h5").resolved)?.active, 0);
  next = { ok: false, manifest: m, code: "not-json", reason: "torn write" };
  watched!();
  const v = study.view(null)!;
  assert.deepEqual(v.rows, []);
  assert.equal(v.refused[0]!.code, "not-json");
  assert.ok(changes.includes("changed"));
});

test("the panel: the window and compare rows, the refused rows, and the difference toggle only with an open compare set", () => {
  assert.equal(variantLines(initialVariants), null);
  const s = reduce(initialState, { type: "variantsRead", study: STUDY });
  const l = variantLines(s.variants)!;
  assert.equal(l.title, "s.variants.json · 1.0.0");
  assert.deepEqual(l.rows.map((r) => [r.label, r.role, r.canSwitch, r.canCompare]), [["Uncracked", "window", false, false], ["Cracked", null, true, true]]);
  assert.deepEqual(l.refused, ["Bad (refused, bad-path): variants[1]: path is absolute"]);
  assert.equal(l.difference, "unavailable");
  assert.equal(paramsLine({ "Ie/Ig": 0.35, state: "cracked" }), "Ie/Ig = 0.35 · state = cracked");
  const opening = reduce(s, { type: "variantCompare", row: { index: 2, label: "Cracked" } });
  assert.deepEqual(variantLines(opening.variants)!.rows.map((r) => [r.role, r.canSwitch]), [["window", false], ["compare", false]]);
  const open = reduce(opening, { type: "compareOpened", result: { index: 2, label: "Cracked", results: "c.ladruno", femHash: "f", stages: [] } });
  assert.equal(variantLines(open.variants)!.difference, "off");
  assert.match(variantLines(open.variants)!.compare!, /Comparing with Cracked \(read-only; never the window set\)/);
  // A late answer for another row is dropped.
  assert.equal(reduce(opening, { type: "compareOpened", result: { index: 0, label: "Uncracked", reason: "x" } }), opening);
  // Main's study without the compare row empties the page's slot.
  const gone = reduce(open, { type: "variantsRead", study: { ...STUDY, compare: null } });
  assert.equal(gone.variants.compare, null);
});

test("the preload sends a compare step's stage, step and request number only, never a path the page added", async () => {
  const STUB = new URL("../main/electronStub.ts", import.meta.url).href;
  registerHooks({ resolve: (specifier, context, next) => (specifier === "electron" ? { url: STUB, shortCircuit: true } : next(specifier, context)) });
  const { exposed, sent } = (await import(STUB)) as typeof import("../main/electronStub.ts");
  await import(new URL("../../src/main/preload.ts", import.meta.url).href);
  const viewer = exposed.get("viewer") as unknown as { readCompareStep(req: object): Promise<unknown>; variantsCompare(i: number, l: string): Promise<unknown> };
  sent.length = 0;
  await viewer.readCompareStep({ stage: "stage_0", step: 2, seq: 1, path: "C:\\elsewhere\\x.ladruno" });
  await viewer.variantsCompare(2, "Cracked");
  assert.deepEqual(sent, [
    { how: "invoke", channel: "compare:readStep", args: [{ stage: "stage_0", step: 2, seq: 1 }] },
    { how: "invoke", channel: "variants:compare", args: [2, "Cracked"] },
  ]);
});
