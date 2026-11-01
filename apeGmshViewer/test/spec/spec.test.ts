// ADR 0128 slice 2 (VA2): the view spec `apegmsh-view/1` fails closed (D2).
// Every spec-level refusal names its key and, through the render job main
// starts from (src/spec/job.ts), writes render.json and no PNG; a valid spec
// states every default (D1, D8: light theme for batch stills; D9 layers).

import assert from "node:assert/strict";
import { existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { after, test } from "node:test";
import { prepareRender } from "../../src/spec/job.ts";
import { APP_LAYERS, MODEL_LAYERS, SPEC_VERSION, validateSpec } from "../../src/spec/spec.ts";

const FIX = fileURLToPath(new URL("../../fixtures/", import.meta.url));
const MODEL = join(FIX, "contour_staged.h5");
const tmp = mkdtempSync(join(tmpdir(), "agv-spec-"));
after(() => rmSync(tmp, { recursive: true, force: true }));

const base = (over: Record<string, unknown> = {}, shot: Record<string, unknown> = {}) => ({
  spec: SPEC_VERSION,
  files: { model: MODEL },
  shots: [{ name: "a", ...shot }],
  ...over,
});

function refused(raw: unknown, key: string, reason: RegExp) {
  const v = validateSpec(raw);
  assert.equal(v.ok, false, `expected a refusal at ${key}`);
  if (v.ok) return;
  assert.equal(v.key, key);
  assert.match(v.reason, reason);
}

// ---- D2: each spec-level refusal names its key ---------------------------------

test("D2: an unknown spec version refuses the spec at `spec`, before any other key is read", () => {
  refused({ ...base(), spec: "apegmsh-view/2", bogus: 1 }, "spec", /must be "apegmsh-view\/1"; got "apegmsh-view\/2"/);
  refused({ files: { model: MODEL }, shots: [] }, "spec", /got nothing/);
  refused([1, 2], "spec", /must be a JSON object/);
});

test("D2: an unknown key refuses the spec and names the key, at every level", () => {
  refused(base({ colour: "red" }), "colour", /is not a key of apegmsh-view\/1 here/);
  refused(base({}, { zoom: 2 }), "shots[0].zoom", /not a key/);
  refused(base({}, { stage: "s", contour: { field: "f", component: "x", palette: "jet" } }), "shots[0].contour.palette", /not a key/);
  refused(base({ files: { model: MODEL, geometry: "x.geometry.h5" } }), "files.geometry", /not a key/);
  refused(base({}, { stage: "s", deform: { scale: 2, factor: 3 } }), "shots[0].deform.factor", /not a key/);
  refused(base({}, { clip: [{ normal: "z", offset: 0, cap: true }] }), "shots[0].clip[0].cap", /not a key/);
  refused(base({}, { camera: { eye: [1, 0, 0], target: [0, 0, 0], up: [0, 0, 1], fov: 30 } }), "shots[0].camera.fov", /not a key/);
});

test("D2: a wrong type refuses the spec and names the key", () => {
  refused(base({ size: [1600, "1000"] }), "size[1]", /integer from 64 to 8192; got "1000"/);
  refused(base({ size: [1600] }), "size", /\[width, height\]/);
  refused(base({ shots: "a" }), "shots", /non-empty list/);
  refused(base({ shots: [] }), "shots", /non-empty list/);
  refused(base({ files: { model: 3 } }), "files.model", /non-empty string; got 3/);
  refused(base({}, { name: 7 }), "shots[0].name", /non-empty string/);
  refused(base({}, { stage: "s", step: -1 }), "shots[0].step", /"first", "last" or a step index/);
  refused(base({}, { stage: "s", step: 1.5 }), "shots[0].step", /step index/);
  refused(base({}, { stage: true }), "shots[0].stage", /stage name or its index/);
  refused(base({}, { stage: "s", deform: { scale: "10" } }), "shots[0].deform.scale", /finite number; got "10"/);
  refused(base({}, { stage: "s", deform: { scale: 0 } }), "shots[0].deform.scale", /must be positive/);
  refused(base({}, { stage: "s", deform: { ghost: "yes" } }), "shots[0].deform.ghost", /true or false/);
  refused(base({}, { camera: { eye: [1, 0], target: [0, 0, 0], up: [0, 0, 1] } }), "shots[0].camera.eye", /three numbers/);
  refused(base({}, { camera: { eye: [0, 0, 0], target: [0, 0, 0], up: [0, 0, 1] } }), "shots[0].camera.eye", /differ from the target/);
  refused(base({}, { isolate: [] }), "shots[0].isolate", /non-empty list/);
  refused(base({}, { isolate: ["g", "g"] }), "shots[0].isolate", /names g twice/);
  refused(base({}, { clip: [{ normal: [0, 0, 0], offset: 1 }] }), "shots[0].clip[0].normal", /zero vector/);
  refused(base({}, { clip: Array.from({ length: 7 }, () => ({ normal: "z", offset: 0 })) }), "shots[0].clip", /at most 6/);
  refused(base({}, { stage: "s", contour: { field: "f", component: "x", range: { min: 2, max: 1 } } }), "shots[0].contour.range.max", /greater than min/);
});

test("D2/D8: a value outside its closed set refuses the spec; a colour can only be a named colormap", () => {
  refused(base({ theme: "sepia" }), "theme", /one of dark, light; got "sepia"/);
  refused(base({}, { camera: "home" }), "shots[0].camera", /one of iso, \+x/);
  refused(base({}, { stage: "s", contour: { field: "f", component: "x", cmap: "#ff0000" } }), "shots[0].contour.cmap", /one of jet, turbo/);
  refused(base({}, { stage: "s", contour: { field: "f", component: "x", gauss: "nodes" } }), "shots[0].contour.gauss", /one of cell/);
  refused(base({}, { colour_by: "material" }), "shots[0].colour_by", /one of group, role, contour/);
  refused(base({ background: "white" }), "background", /one of theme, transparent/);
});

test("D2: contradictory keys refuse the spec: colour_by against contour, a step without a stage, two shots of one file name", () => {
  refused(base({}, { colour_by: "contour" }), "shots[0].colour_by", /no contour key/);
  refused(base({}, { stage: "s", colour_by: "group", contour: { field: "f", component: "x" } }), "shots[0].colour_by", /a contour colours by "contour"/);
  refused(base({}, { step: 0 }), "shots[0].step", /needs a stage/);
  refused(base({}, { name: "../up" }), "shots[0].name", /file name/);
  refused(base({ shots: [{ name: "Uz" }, { name: "uz" }] }), "shots[1].name", /already the name of shots\[0\]/);
});

test("D9: layers are a closed set per chrome; transparent needs model mode", () => {
  refused(base({}, { layers: { legend: false } }), "shots[0].layers.legend", /not a key/);
  refused(base({}, { layers: { scale_bar: false } }), "shots[0].layers.scale_bar", /not a layer of chrome "app"/);
  refused(base({}, { chrome: "model", layers: { header: false } }), "shots[0].layers.header", /not a layer of chrome "model"/);
  refused(base({}, { chrome: "model", layers: { scale_bar: true } }), "shots[0].layers.scale_bar", /draws no contour to scale/);
  refused(base({}, { chrome: "model", layers: { caption: true } }), "shots[0].layers.caption", /no caption/);
  refused(base({}, { background: "transparent" }), "shots[0].background", /needs chrome "model"/);
  refused(base({ background: "transparent" }), "shots[0].chrome", /needs chrome "model"/);
  // A spec-level default that does not fit a shot's chrome is refused at the spec's key.
  refused(base({ layers: { header: false } }, { chrome: "model" }), "layers.header", /not a layer of chrome "model"/);
  refused(base({}, { layers: { header: "no" } }), "shots[0].layers.header", /true or false/);
});

// ---- the defaults ---------------------------------------------------------------

test("D1/D8/D9: a minimal spec states every default (light theme, iso, last step, every app panel on)", () => {
  const v = validateSpec(base());
  assert.ok(v.ok);
  if (!v.ok) return;
  assert.equal(v.spec.theme, "light", "D8: batch stills default to the light theme");
  assert.deepEqual(v.spec.size, [1600, 1000]);
  assert.deepEqual(v.spec.files, { model: MODEL, results: null });
  const s = v.spec.shots[0]!;
  assert.deepEqual(
    { camera: s.camera, step: s.step, stage: s.stage, colour_by: s.colour_by, contour: s.contour, deform: s.deform, clip: s.clip, isolate: s.isolate, chrome: s.chrome, background: s.background },
    { camera: "iso", step: "last", stage: null, colour_by: "group", contour: null, deform: null, clip: [], isolate: [], chrome: "app", background: "theme" },
  );
  assert.deepEqual(Object.keys(s.layers), [...APP_LAYERS]);
  assert.ok(Object.values(s.layers).every((x) => x === true));
});

test("D9: model mode resolves every layer; the scale bar is on whenever a contour is drawn, unless turned off", () => {
  const v = validateSpec(
    base({ chrome: "model", layers: { axes: false } }, {}),
  );
  assert.ok(v.ok);
  if (!v.ok) return;
  assert.deepEqual(v.spec.shots[0]!.layers, { scale_bar: false, axes: false, banner: false, caption: false });
  const c = validateSpec({
    ...base({ chrome: "model" }),
    shots: [
      { name: "on", stage: "s", contour: { field: "f", component: "x" } },
      { name: "off", stage: "s", contour: { field: "f", component: "x" }, layers: { scale_bar: false }, caption: "A caption.", background: "transparent" },
      { name: "app", chrome: "app", layers: { header: false, rail: false } },
    ],
  });
  assert.ok(c.ok);
  if (!c.ok) return;
  const [on, off, app] = c.spec.shots;
  assert.deepEqual(Object.keys(on!.layers), [...MODEL_LAYERS]);
  assert.equal(on!.layers["scale_bar"], true, "a contour carries its scale bar by default");
  assert.equal(off!.layers["scale_bar"], false, "an agent turns it off explicitly");
  assert.equal(off!.background, "transparent");
  assert.deepEqual(app!.layers, { header: false, phase: true, viewcube: true, left_rail: true, rail: false, windows: true, banner: true });
  assert.equal(on!.contour!.cmap, "jet", "D8: contours default to jet");
  assert.deepEqual(on!.contour!.range, { mode: "history" });
});

test("D1: an axis-name normal is its unit vector; an explicit camera is kept as given", () => {
  const v = validateSpec(base({}, { clip: [{ normal: "y", offset: 1.5, flip: true }], camera: { eye: [5, 5, 5], target: [0, 0, 0], up: [0, 0, 1] } }));
  assert.ok(v.ok);
  if (!v.ok) return;
  assert.deepEqual(v.spec.shots[0]!.clip, [{ normal: [0, 1, 0], offset: 1.5, flip: true }]);
  assert.deepEqual(v.spec.shots[0]!.camera, { eye: [5, 5, 5], target: [0, 0, 0], up: [0, 0, 1] });
});

// ---- the render job: a refused spec writes render.json and no PNG ------------------

const fsOf = {
  readFile: (p: string) => readFileSync(p, "utf8"),
  exists: existsSync,
  mkdir: (p: string) => mkdirSync(p, { recursive: true }),
  writeFile: (p: string, t: string) => writeFileSync(p, t),
};

function job(name: string, body: string) {
  const dir = join(tmp, name);
  mkdirSync(dir, { recursive: true });
  const spec = join(dir, "spec.json");
  writeFileSync(spec, body);
  const out = join(dir, "out");
  return { spec, out, p: prepareRender(spec, out, fsOf) };
}

for (const [name, body, key] of [
  ["not-json", "{ not json", "spec"],
  ["version", JSON.stringify({ ...base(), spec: "apegmsh-view/0" }), "spec"],
  ["unknown-key", JSON.stringify(base({}, { zoom: 2 })), "shots[0].zoom"],
  ["wrong-type", JSON.stringify(base({ size: "big" })), "size"],
  ["no-model", JSON.stringify(base({ files: { model: "missing.h5" } })), "files.model"],
  ["not-a-model", JSON.stringify(base({ files: { model: join(FIX, "contour_staged.results.h5") } })), "files.model"],
  ["results-elsewhere", JSON.stringify(base({ files: { model: MODEL, results: join(FIX, "deform_2d.results.h5") } })), "files.results"],
] as const) {
  test(`D2: a refused spec (${name}) writes render.json naming ${key}, and no PNG`, () => {
    const { out, p } = job(name, body);
    assert.equal(p.ok, false);
    if (p.ok) return;
    assert.equal(p.key, key);
    const files = readdirSync(out);
    assert.deepEqual(files, ["render.json"], "nothing but render.json: no PNG");
    const r = JSON.parse(readFileSync(join(out, "render.json"), "utf8"));
    assert.deepEqual(r.render.refused, { key: p.key, reason: p.reason });
  });
}

test("D1: files resolve against the spec's folder and pair by ADR 0112's rules; a valid spec writes nothing yet", () => {
  const dir = join(tmp, "relative");
  mkdirSync(dir, { recursive: true });
  for (const f of ["contour_staged.h5", "contour_staged.results.h5"]) writeFileSync(join(dir, f), readFileSync(join(FIX, f)));
  const spec = join(dir, "spec.json");
  writeFileSync(spec, JSON.stringify(base({ files: { model: "contour_staged.h5" } })));
  const out = join(dir, "out");
  const p = prepareRender(spec, out, fsOf);
  assert.ok(p.ok, p.ok ? "" : `${p.key}: ${p.reason}`);
  if (!p.ok) return;
  assert.equal(p.job.model, join(dir, "contour_staged.h5"));
  assert.equal(p.job.results, join(dir, "contour_staged.results.h5"), "the results sibling is found by the pairing rule");
  assert.equal(existsSync(out), false, "a valid spec writes nothing before the window draws");
  // The results named explicitly must be the set's own.
  writeFileSync(spec, JSON.stringify(base({ files: { model: "contour_staged.h5", results: "contour_staged.results.h5" } })));
  const q = prepareRender(spec, out, fsOf);
  assert.ok(q.ok && q.job.results === join(dir, "contour_staged.results.h5"));
});
