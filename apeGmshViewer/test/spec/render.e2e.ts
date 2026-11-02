// ADR 0128 slices 2 and 3 (VA2, VA3), end to end: `--render` and `--describe` in the
// real app, on the fixtures. Not part of `npm test`: it starts Electron, so
// it needs a display (Windows today; Linux capture is an open question of
// the ADR). Run it after a build:
//
//   node scripts/build.mjs && node test/spec/render.e2e.ts [<outdir>]
//
// It checks, and exits non-zero on the first failure:
//   1. the mesh, contour and deformed views of ADR 0094, and a reaction
//      contour at the supports (ADR 0094's reactions view is a diagram at the
//      supports, which the app does not draw yet: a parity gap for slice 6),
//      render in both chromes, each PNG exactly the spec's size; a model still
//      lays out no panel; an app still with its layers off keeps the
//      viewport's box;
//   2. the same spec rendered twice gives identical PNGs, every shot (the
//      kill criterion: it stops here when not), byte-identical manifests except
//      `timing` (ADR 0128 Acceptance), and each shot's time (kill criterion
//      ~10 s); the reference ghost likewise, shot 5 times in each of 2 runs
//      (#2664): every model still alike, each app still alike run to run,
//      over the whole image, the phase readout included;
//   INV-4 (#2696): an app still is the page as it settled: a capture that
//      first returns a stale frame still writes the right still, a page that
//      never settles refuses the shot with no PNG, and `--capture-tries` below
//      2 is refused (exit 2);
//   INV-1: Chromium's profile goes to `<out>/.chromium`, and each run's own
//      `--user-data-dir`, where Chromium writes by default, stays empty (a
//      check no other window on the machine can disturb, unlike %APPDATA%);
//   3. INV-7: an app and a model still of one shot have the same camera pose
//      and the same contour; the scale bar is drawn unless turned off, and
//      it is the only difference; `transparent` writes PNG alpha;
//   4. each D2 shot refusal refuses with its reason, the same in both
//      chromes, and writes no PNG, while the other shots still run (exit 1);
//      each spec refusal writes manifest.json with the app's version and no PNG (exit 2), and a
//      non-empty --out is refused with nothing written (exit 2);
//   5. a render beside an open view window leaves that window's pixels,
//      inspector and header as they were (INV-5, ADR 0128 D4);
//   6. D3, INV-6: on the fixtures, the manifest a real run writes equals
//      Python's `Results` (test/spec/manifest.oracle.json), in both chromes,
//      with one entry per PNG.
// The stills stay in <outdir> (default: a new temporary folder) for review.
// Every process starts through a generated app folder (the real app's name and
// version, its built main) that turns a modal into a stderr line, so nothing
// stops on the desktop of whoever runs this, and the run fails on any such
// line; `AGV_FAULT` forces a capture fault into it.

import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { copyFileSync, existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import type { ManifestEntry } from "../../src/spec/manifest.ts";
import { caseShot, ORACLE, sameAsOracle } from "./manifestOracle.ts";
import { readPng, rgbDiff } from "./png.ts";
import { makeReactionSet } from "./reactions.ts";
import * as h5wasm from "h5wasm/node";

const ROOT = fileURLToPath(new URL("../../", import.meta.url));
const FIX = join(ROOT, "fixtures");
const electron = createRequire(import.meta.url)("electron") as string;
const OUT = resolve(process.argv[2] ?? mkdtempSync(join(tmpdir(), "agv-render-e2e-")));
mkdirSync(OUT, { recursive: true });
const log = (s: string) => process.stdout.write(`${s}\n`);

/** The app every run starts: the real app's name and version, dialogs turned into stderr lines, a forced capture fault when `AGV_FAULT` names one. */
const APP = join(OUT, "app");
mkdirSync(APP, { recursive: true });
{
  const pkg = JSON.parse(readFileSync(join(ROOT, "package.json"), "utf8")) as { name: string; version: string };
  writeFileSync(join(APP, "package.json"), JSON.stringify({ name: pkg.name, version: pkg.version, private: true, type: "module", main: "entry.mjs" }));
}
writeFileSync(
  join(APP, "entry.mjs"),
  `import { app, dialog, ipcMain, nativeImage } from "electron";
const nl = String.fromCharCode(10);
dialog.showErrorBox = (t, c) => process.stderr.write("MODAL " + t + " " + c + nl);
dialog.showMessageBox = async (...a) => (process.stderr.write("MODAL showMessageBox" + nl), { response: 0, checkboxChecked: false });
dialog.showMessageBoxSync = (...a) => (process.stderr.write("MODAL showMessageBoxSync" + nl), 0);
const fault = process.env.AGV_FAULT ?? "";
// A hidden window's capture of the frame before the page's last changes (#2696): each app still's
// first capture returns the image the capture before it returned (a grey page for the first one).
if (fault === "stale-first") {
  let first = false;
  let last = null;
  const handle = ipcMain.handle.bind(ipcMain);
  ipcMain.handle = (channel, fn) => handle(channel, channel !== "render:capture" ? fn : (e, ...a) => ((first = true), fn(e, ...a)));
  app.on("web-contents-created", (_e, wc) => {
    const real = wc.capturePage.bind(wc);
    wc.capturePage = async (...a) => {
      const img = await real(...a);
      const stale = first ? (last ?? nativeImage.createFromBitmap(Buffer.alloc(img.toBitmap().length, 0x80), img.getSize())) : img;
      first = false;
      last = img;
      return stale;
    };
  });
}
// A page that never settles: every capture differs from the one before it in one byte.
if (fault === "never-settles") {
  app.on("web-contents-created", (_e, wc) => {
    const real = wc.capturePage.bind(wc);
    let n = 0;
    wc.capturePage = async (...a) => {
      const img = await real(...a);
      const px = Buffer.from(img.toBitmap());
      px[0] = n++ % 251;
      return nativeImage.createFromBitmap(px, img.getSize());
    };
  });
}
await import(${JSON.stringify(pathToFileURL(join(ROOT, "dist", "main.mjs")).href)});
`,
);

/** Every run's own profile folder (`--user-data-dir`); INV-1 holds while each stays empty. */
const userDirs: string[] = [];
let runs = 0;
function run(args: string[], env: Record<string, string> = {}): Promise<{ code: number; ms: number }> {
  const t = performance.now();
  const user = join(OUT, "users", String(++runs).padStart(3, "0"));
  userDirs.push(user);
  return new Promise((done) => {
    const child = spawn(electron, [APP, `--user-data-dir=${user}`, ...args], { stdio: ["ignore", "ignore", "pipe"], env: { ...process.env, ...env } });
    let err = "";
    child.stderr.on("data", (d) => {
      err += String(d);
      process.stderr.write(d);
    });
    child.on("exit", (code) => {
      assert.doesNotMatch(err, /MODAL/, `a modal would have stopped on the desktop: ${err}`);
      done({ code: code ?? -1, ms: performance.now() - t });
    });
  });
}

interface Rendered {
  code: number;
  ms: number;
  dir: string;
  shots: ManifestEntry[];
  /** each shot's time; `timing` is the one part of a manifest that varies between runs */
  timing: Record<string, { ms: number }>;
  /** the manifest as written, without `timing` */
  stable: string;
  version: string | undefined;
  refused?: { key: string; reason: string };
}

async function render(name: string, spec: object, env: Record<string, string> = {}): Promise<Rendered> {
  const dir = join(OUT, name);
  mkdirSync(dir, { recursive: true });
  const path = join(dir, "spec.json");
  writeFileSync(path, JSON.stringify(spec, null, 2));
  const out = join(dir, "out");
  const r = await run([`--mode=render`, `--spec=${path}`, `--out=${out}`], env);
  const j = JSON.parse(readFileSync(join(out, "manifest.json"), "utf8"));
  const { timing, ...stable } = j;
  return { ...r, dir: out, shots: j.shots, timing: timing?.shots ?? {}, stable: JSON.stringify(stable), version: j.app?.version, refused: j.refused };
}

const pngs = (dir: string) => readdirSync(dir).filter((f) => f.endsWith(".png")).sort();


// ---- the sets -----------------------------------------------------------------------

const reactions = await makeReactionSet(join(OUT, "sets", "reactions"));
function copySet(dir: string, model: string, results: string, stem = "set"): string {
  mkdirSync(dir, { recursive: true });
  copyFileSync(join(FIX, model), join(dir, `${stem}.h5`));
  copyFileSync(join(FIX, results), join(dir, `${stem}.results.h5`));
  return join(dir, `${stem}.h5`);
}
const activation = copySet(join(OUT, "sets", "activation"), "staged_activation.h5", "staged_activation.results.h5");
const badid = copySet(join(OUT, "sets", "badid"), "elem_badid.h5", "elem_badid.results.h5");
const mismatch = copySet(join(OUT, "sets", "mismatch"), "contour_staged.h5", "elem_mixed.results.h5");
// #2664: contour_staged with one 2-node line added between two of its nodes, so the model has solids and lines and opens
// translucent (reduce.ts, opacity 0.35). The stored lineage is untouched, so its results still pair.
const translucent = copySet(join(OUT, "sets", "translucent"), "contour_staged.h5", "contour_staged.results.h5");
{
  await h5wasm.ready;
  const f = new h5wasm.File(translucent, "a");
  try {
    const ids = Array.from((f.get("nodes/ids") as h5wasm.Dataset).value as ArrayLike<number | bigint>, Number);
    const elem = Array.from((f.get("elements/tet4/ids") as h5wasm.Dataset).value as ArrayLike<number | bigint>, Number);
    const g = f.create_group("elements/line2");
    for (const [k, v] of [["code", 1], ["dim", 1], ["npe", 2], ["order", 1]] as const) g.create_attribute(k, v, [], "<q");
    g.create_attribute("gmsh_name", "Line 2");
    g.create_dataset({ name: "connectivity", data: new BigInt64Array([BigInt(ids[0]!), BigInt(ids[1]!)]), shape: [1, 2], dtype: "<q" });
    g.create_dataset({ name: "ids", data: new BigInt64Array([BigInt(Math.max(...elem) + 1)]), shape: [1], dtype: "<q" });
    g.create_dataset({ name: "module_label", data: [""], shape: [1] });
  } finally {
    f.close();
  }
}

// ---- describe -------------------------------------------------------------------------

{
  const out = join(OUT, "describe");
  const r = await run(["--mode=describe", `--file=${reactions.results}`, `--out=${out}`]);
  assert.equal(r.code, 0, "describe exits 0");
  const d = JSON.parse(readFileSync(join(out, "describe.json"), "utf8")).describe;
  assert.equal(d.pairing.state, "paired");
  assert.deepEqual(d.stages.map((s: { name: string }) => s.name), ["base", "erect", "unprop"]);
  assert.ok(d.stages[2].fields.some((f: { name: string }) => f.name === "reaction_force"));
  assert.deepEqual(pngs(out), [], "describe draws nothing");
  assert.ok(existsSync(join(out, ".chromium")), "INV-1: Chromium's profile is inside --out");
  log(`describe: ok (${Math.round(r.ms)} ms)`);
}

// ---- 1-3: the views, twice -----------------------------------------------------------------

const SIZE: [number, number] = [1200, 800];
const contour = { stage: "unprop", contour: { field: "displacement", component: "z" } };
const views = [
  { name: "mesh", camera: "iso" },
  { name: "contour", ...contour },
  { name: "deformed", stage: "unprop", deform: { scale: 10 } },
  // A nodal contour of the reactions, recorded at the held nodes only: not ADR 0094's diagram at the supports.
  { name: "reactions", stage: "unprop", contour: { field: "reaction_force", component: "magnitude" } },
];
const viewSpec = {
  spec: "apegmsh-view/1",
  files: { model: reactions.model },
  size: SIZE,
  shots: [
    ...views,
    ...views.map((v) => ({ ...v, name: `${v.name}.model`, chrome: "model", caption: `The ${v.name} view, model mode.` })),
    { ...contour, name: "contour.bare", layers: { header: false, phase: false, viewcube: false, left_rail: false, rail: false, windows: false, banner: false } },
    { ...contour, name: "contour.model-noscale", chrome: "model", layers: { scale_bar: false } },
    { ...contour, name: "contour.model-transparent", chrome: "model", background: "transparent" },
    { ...contour, name: "contour.model-captioned", chrome: "model", caption: "Vertical displacement, stage unprop, last step.", layers: { caption: true, banner: true } },
    { ...contour, name: "contour.clip.model", chrome: "model", clip: [{ normal: "y", offset: 0.5 }], isolate: ["mesh/physical_group/Col", "mesh/physical_group/Top"] },
  ],
};
const A = await render("views-a", viewSpec);
const B = await render("views-b", viewSpec);
for (const R of [A, B]) {
  assert.equal(R.code, 0, `every view renders (exit ${R.code}): ${R.shots.filter((s) => s.status !== "ok").map((s) => `${s.name}: ${s.reason}`).join("; ")}`);
  assert.deepEqual(pngs(R.dir), viewSpec.shots.map((s) => `${s.name}.png`).sort());
}
const byName = (R: typeof A) => new Map(R.shots.map((s) => [s.name, s]));
const a = byName(A);
let worst = 0;
for (const s of A.shots) {
  const pa = readPng(readFileSync(join(A.dir, `${s.name}.png`)));
  const pb = readPng(readFileSync(join(B.dir, `${s.name}.png`)));
  assert.deepEqual([pa.width, pa.height], SIZE, `${s.name} is exactly the spec's size`);
  const d = rgbDiff(pa, pb);
  // Kill criterion (#2651): the same spec must give the same pixels on one machine, every shot.
  if (d.count) throw new Error(`KILL: ${s.name} differs between two runs of one spec in ${d.count} pixels (box ${d.bbox?.join(", ")})`);
  if (s.shot.chrome === "model") assert.deepEqual(s.layout!.panels, [], `${s.name}: a model still lays out no panel (DOM check)`);
  assert.equal(s.png, `${s.name}.png`, "a still's path is relative to its manifest");
  const ms = A.timing[s.name]!.ms;
  worst = Math.max(worst, ms);
  log(`  ${s.name}: ${pa.width}x${pa.height}, identical twice, ${Math.round(ms)} ms`);
}
// ADR 0128 Acceptance: the same spec twice gives byte-identical manifests, except for timestamps.
assert.equal(A.stable, B.stable, "two runs of one spec: the manifests differ outside `timing`");
assert.ok(A.version, "D3: the manifest states the app's version");
// A camera as resolved, from the real viewport.
for (const s of A.shots) {
  assert.ok(s.camera, `${s.name}: a drawn shot states its camera`);
  assert.ok([...s.camera.eye, ...s.camera.target, ...s.camera.up].every(Number.isFinite), `${s.name}: a finite pose`);
  assert.ok(typeof s.camera.fov === "number" && s.camera.fov > 0, `${s.name}: a field of view`);
}
// INV-6: a plane cuts this still, so nothing is stated as drawn.
{
  const cut = a.get("contour.clip.model")!.contour;
  assert.ok(cut?.drawn);
  assert.equal(cut.extrema.drawn, null);
  assert.match(cut.extrema.drawn_reason ?? "", /section planes are cutting/);
  assert.match(cut.no_data.limit ?? "", /counted before the section planes/);
  assert.ok(cut.extrema.recorded, "the recorded extrema stand");
}
for (const R of [A, B]) assert.ok(existsSync(join(R.dir, ".chromium")), "INV-1: Chromium's profile is inside --out");
log(`views: ${A.shots.length} stills, pixel-identical over two runs; run ${Math.round(A.ms)} / ${Math.round(B.ms)} ms; slowest shot ${Math.round(worst)} ms`);
if (worst > 10000) throw new Error(`KILL: a shot took ${Math.round(worst)} ms (> ~10 s)`);

// D9: app layers off leave the viewport's box alone and leave only what no layer names.
assert.deepEqual(a.get("contour.bare")!.layout!.viewport, a.get("contour")!.layout!.viewport);
assert.deepEqual(a.get("contour.bare")!.layout!.panels, [], "every named panel omitted from the layout");
assert.ok(a.get("contour")!.layout!.panels.includes("#header"), "the default app still keeps its panels");
// INV-7: the same manifest entry in both chromes, apart from the keys that name the chrome, the layers and the background.
const inv7 = (s: ManifestEntry) => ({ ...s, name: null, png: null, shot: null, layout: null });
for (const v of views) {
  assert.deepEqual(inv7(a.get(`${v.name}.model`)!), inv7(a.get(v.name)!), `${v.name}: one entry in both chromes`);
}
// D9: the scale bar is drawn by default and is the only difference to a still without it.
{
  const withBar = readPng(readFileSync(join(A.dir, "contour.model.png")));
  const without = readPng(readFileSync(join(A.dir, "contour.model-noscale.png")));
  const d = rgbDiff(withBar, without);
  assert.ok(d.count > 1000, "the scale bar is drawn");
  const [x0, y0, x1, y1] = d.bbox!;
  // The bar's panel is 284 x 86 at (16, 16) (renderer/stillLayers.ts); its anti-aliased border reaches one pixel out.
  assert.ok(x0 >= 15 && y0 >= 15 && x1 <= 16 + 284 + 2 && y1 <= 16 + 86 + 2, `the only difference is the scale bar's box (top left): ${d.bbox}`);
  assert.equal(a.get("contour.model")!.shot.layers["scale_bar"], true);
  assert.equal(a.get("contour.model-noscale")!.shot.layers["scale_bar"], false, "the manifest records the layer turned off");
  const t = readPng(readFileSync(join(A.dir, "contour.model-transparent.png")));
  let clear = 0;
  for (let i = 3; i < t.rgba.length; i += 4) if (t.rgba[i] === 0) clear++;
  assert.equal(t.colorType, 6);
  assert.ok(clear > t.width * t.height * 0.5, `transparent: most pixels have alpha 0 (${clear})`);
  let opaque = true;
  for (let i = 3; i < withBar.rgba.length; i += 4) if (withBar.rgba[i] !== 255) opaque = false;
  assert.ok(opaque, "background theme: every pixel opaque");
  assert.equal(a.get("contour.model-transparent")!.shot.background, "transparent");
}
log("D9 layers: ok");

// #2664: the reference ghost draws the same pixels every time, 5 shots in each of 2 runs, in both chromes.
{
  const shot = { stage: "unprop", deform: { scale: 10, ghost: true } };
  const shots = [...Array(5).keys()].flatMap((i) => [
    { ...shot, name: `ghost${i}.model`, chrome: "model" },
    { ...shot, name: `ghost${i}.app` },
  ]);
  const spec = { spec: "apegmsh-view/1", files: { model: reactions.model }, size: SIZE, shots: [...shots, { ...shot, deform: { scale: 10 }, name: "noghost.model", chrome: "model" }] };
  const G1 = await render("ghost-a", spec);
  const G2 = await render("ghost-b", spec);
  for (const R of [G1, G2]) assert.equal(R.code, 0, `the ghost renders (exit ${R.code}): ${R.shots.map((s) => s.reason).filter(Boolean).join("; ")}`);
  const still = (R: typeof G1, name: string) => readPng(readFileSync(join(R.dir, `${name}.png`)));
  assert.equal((byName(G1).get("ghost0.model")!.deform as { ghost: boolean }).ghost, true, "the result records the ghost");
  // A ghost that is not drawn would be stable too.
  assert.ok(rgbDiff(still(G1, "ghost0.model"), still(G1, "noghost.model")).count > 1000, "the ghost is drawn");
  // The model still is the renderer's own canvas: every shot of both runs is the first one, pixel for pixel.
  const first = still(G1, "ghost0.model");
  for (const R of [G1, G2]) {
    for (let i = 0; i < 5; i++) {
      const d = rgbDiff(first, still(R, `ghost${i}.model`));
      if (d.count) throw new Error(`KILL: ghost${i}.model differs from the first shot in ${d.count} pixels (box ${d.bbox?.join(", ")})`);
    }
  }
  // The app still is the page as the compositor settled it (#2696), the phase readout included: each shot is the
  // same shot of the other run, over the whole image. Shot to shot within a run it is not held: with the ghost
  // drawn, the settled page of one shot can differ from its neighbours by 1 in a channel at ~200 edge pixels while
  // the canvas the renderer drew is identical (the single capture before #2696 took the frame before and hid it).
  for (let i = 0; i < 5; i++) {
    const d = rgbDiff(still(G1, `ghost${i}.app`), still(G2, `ghost${i}.app`));
    if (d.count) throw new Error(`KILL: ghost${i}.app differs between two runs of one spec in ${d.count} pixels (box ${d.bbox?.join(", ")})`);
  }
  log("reference ghost: 5 shots in each of 2 runs; model stills pixel-identical, app stills pixel-identical run to run");
}

// INV-4 (#2696): an app still is the page as it settled, or the shot is refused.
{
  // App stills that differ from one to the next, so a stale frame is a different picture; a model shot between them omits every panel.
  const spec = {
    spec: "apegmsh-view/1",
    files: { model: reactions.model },
    size: [640, 480],
    shots: [
      { name: "iso", camera: "iso" },
      { name: "between.model", chrome: "model", camera: "top" },
      { name: "top", camera: "top", stage: "unprop", deform: { scale: 10 } },
      { name: "bare", ...contour, layers: { header: false, rail: false } },
    ],
  };
  const app = spec.shots.filter((s) => s.chrome !== "model").map((s) => s.name);
  const ref = await render("settle-ref", spec);
  assert.equal(ref.code, 0, `the reference run: ${ref.shots.map((s) => s.reason).filter(Boolean).join("; ")}`);
  // Each still's first capture returns the frame before: the still written is still the settled page.
  const stale = await render("settle-stale-first", spec, { AGV_FAULT: "stale-first" });
  assert.equal(stale.code, 0, `a stale first capture: ${stale.shots.map((s) => s.reason).filter(Boolean).join("; ")}`);
  for (const name of app) {
    const d = rgbDiff(readPng(readFileSync(join(ref.dir, `${name}.png`))), readPng(readFileSync(join(stale.dir, `${name}.png`))));
    if (d.count) throw new Error(`INV-4: ${name} written from a stale capture: it differs from the settled page in ${d.count} pixels (box ${d.bbox?.join(", ")})`);
  }
  // A page that never settles refuses each app shot, names it, and writes no PNG for it; the model shot still runs.
  const never = await render("settle-never", spec, { AGV_FAULT: "never-settles" });
  assert.equal(never.code, 1, "a shot refused: exit 1");
  assert.deepEqual(pngs(never.dir), ["between.model.png"], "no PNG for a still that never settled");
  for (const name of app) {
    const s = byName(never).get(name)!;
    assert.equal(s.status, "refused", `${name}: a page that never settles is refused`);
    assert.equal(s.png, null);
    assert.match(s.reason ?? "", new RegExp(`render:capture ${name}: the page did not settle: no two consecutive of 10 captures were pixel-identical \\(--capture-tries\\)`));
  }
  // The bound is the caller's, checked like any argument.
  for (const v of ["1", "0", "-3", "5x", ""]) {
    const out = join(OUT, `settle-tries-${v || "empty"}`, "out");
    const r = await run(["--mode=render", `--spec=${join(OUT, "settle-ref", "spec.json")}`, `--out=${out}`, `--capture-tries=${v}`]);
    assert.equal(r.code, 2, `--capture-tries=${v}: exit 2`);
    assert.equal(existsSync(out), false, `--capture-tries=${v}: nothing written`);
  }
  log(`INV-4: ${app.length} app stills written right from a stale first capture; a page that never settles refused (${byName(never).get("iso")!.reason}); --capture-tries below 2 refused`);
}

// ---- 4: D2 refusals ---------------------------------------------------------------------

const both = (name: string, shot: object) => [
  { ...shot, name: `${name}.app` },
  { ...shot, name: `${name}.model`, chrome: "model" },
];
async function refusals(name: string, model: string, cases: [string, object, RegExp][]) {
  const spec = { spec: "apegmsh-view/1", files: { model }, size: [640, 480], shots: [{ name: "control", camera: "top" }, ...cases.flatMap(([n, s]) => both(n, s))] };
  const R = await render(name, spec);
  assert.equal(R.code, 1, `${name}: a refused shot makes the exit code 1`);
  assert.deepEqual(pngs(R.dir), ["control.png"], `${name}: only the shot that was not refused has a PNG`);
  const m = byName(R);
  assert.equal(m.get("control")!.status, "ok", "the other shots still run");
  for (const [n, , reason] of cases) {
    const [x, y] = [m.get(`${n}.app`)!, m.get(`${n}.model`)!];
    assert.equal(x.status, "refused");
    assert.match(x.reason ?? "", reason, `${n}`);
    assert.equal(y.reason, x.reason, `${n}: refused identically in both chromes`);
    assert.equal(x.png, null);
    log(`  refused ${n}: ${x.reason}`);
  }
}
await refusals("refuse-files", reactions.model, [
  ["stage-name", { stage: "gravity" }, /no stage named "gravity"/],
  ["stage-index", { stage: 5 }, /stage 5 is not one of them/],
  ["step", { stage: "erect", step: 9 }, /step 9 is not one of them/],
  ["field", { stage: "base", contour: { field: "velocity", component: "x" } }, /has no field "velocity"/],
  ["component", { stage: "base", contour: { field: "displacement", component: "rz" } }, /has no component "rz"/],
  ["group", { isolate: ["mesh/physical_group/Beams"] }, /no physical group/],
  ["no-step", { contour: { field: "displacement", component: "z" } }, /contours colour a results step/],
]);
await refusals("refuse-deform", activation, [["deform", { stage: "base", deform: { scale: 10 } }, /^Deformed shape not drawn: stage base records no displacement_y/]]);
await refusals("refuse-element", badid, [["element", { stage: 0, contour: { field: "stress", component: "stress_xx" } }, /no cell for \(999999\)/]]);
await refusals("refuse-pairing", mismatch, [["pairing", { stage: 0 }, /results are not drawn on this model: .*model_hash differs/]]);
await refusals("refuse-ghost-translucent", translucent, [["ghost", { stage: "unprop", deform: { scale: 10, ghost: true } }, /^the reference ghost is not drawn over translucent surfaces \(surface opacity 0\.35\): a still of it is not reproducible$/]]);
for (const [name, spec, key] of [
  ["spec-version", { spec: "apegmsh-view/9", files: { model: reactions.model }, shots: [{ name: "a" }] }, "spec"],
  ["spec-key", { spec: "apegmsh-view/1", files: { model: reactions.model }, shots: [{ name: "a", zoom: 2 }] }, "shots[0].zoom"],
  ["spec-type", { spec: "apegmsh-view/1", files: { model: reactions.model }, size: [10, 10], shots: [{ name: "a" }] }, "size[0]"],
  ["spec-fit-selection", { spec: "apegmsh-view/1", files: { model: reactions.model }, shots: [{ name: "a", camera: "fit-selection" }] }, "shots[0].camera"],
] as const) {
  const R = await render(name, spec);
  assert.equal(R.code, 2, `${name}: a refused spec exits 2`);
  assert.equal(R.refused?.key, key);
  assert.equal(R.version, A.version, `${name}: the same app version as a run, from the built app`);
  assert.deepEqual(readdirSync(R.dir).filter((f) => f !== ".chromium"), ["manifest.json"], `${name}: no PNG`);
  log(`  refused spec ${name} at ${key} (${Math.round(R.ms)} ms): ${readdirSync(R.dir).join(", ")}`);
}
// A busy --out: an earlier run's still and record stay as they were, and nothing is added (no stale still beside a refusal).
{
  const dir = join(OUT, "busy-out");
  const out = join(dir, "out");
  mkdirSync(out, { recursive: true });
  writeFileSync(join(out, "a.png"), "an earlier run's still");
  const spec = join(dir, "spec.json");
  writeFileSync(spec, JSON.stringify({ spec: "apegmsh-view/1", files: { model: reactions.model }, shots: [{ name: "a", stage: "erect", step: 99 }] }));
  const r = await run(["--mode=render", `--spec=${spec}`, `--out=${out}`]);
  assert.equal(r.code, 2, "a busy --out exits 2");
  assert.deepEqual(readdirSync(out), ["a.png"], "nothing is written into a busy --out");
  const d = await run(["--mode=describe", `--file=${reactions.model}`, `--out=${out}`]);
  assert.equal(d.code, 2, "describe refuses a busy --out too");
  assert.deepEqual(readdirSync(out), ["a.png"]);
  log("  refused a busy --out (render and describe), nothing written");
}
// INV-1: nothing of the agent runs above went to their own profile folders, where Chromium writes by default (the human's window below does write to its own).
{
  const used = userDirs.filter((d) => existsSync(d) && readdirSync(d).length > 0);
  assert.deepEqual(used, [], `INV-1: an agent run wrote to its --user-data-dir: ${used.map((d) => `${d}: ${readdirSync(d).join(", ")}`).join("; ")}`);
  log(`INV-1: ${userDirs.length} agent runs, each --user-data-dir left empty`);
}
log("D2 refusals: ok");

// ---- 5: a render leaves the human's window alone -------------------------------------------

{
  const port = 9300 + Math.floor(Math.random() * 500);
  const human = spawn(electron, [APP, `--user-data-dir=${join(OUT, "users", "human")}`, "--mode=view", `--file=${join(FIX, "shoebuckle.h5")}`, `--remote-debugging-port=${port}`], { stdio: "ignore" });
  try {
    let ws: WebSocket | null = null;
    for (let i = 0; i < 100 && !ws; i++) {
      await new Promise((r) => setTimeout(r, 200));
      try {
        const pages = (await (await fetch(`http://127.0.0.1:${port}/json/list`)).json()) as { type: string; webSocketDebuggerUrl: string }[];
        const page = pages.find((p) => p.type === "page");
        if (page) {
          ws = new WebSocket(page.webSocketDebuggerUrl);
          await new Promise((r, j) => ((ws!.onopen = r), (ws!.onerror = j)));
        }
      } catch {
        ws = null;
      }
    }
    if (!ws) throw new Error("the view window's debugging port did not answer");
    let id = 0;
    const call = (method: string, params: object = {}) =>
      new Promise<{ result?: { result?: { value?: unknown } } }>((done) => {
        const n = ++id;
        const on = (e: MessageEvent) => {
          const m = JSON.parse(String(e.data));
          if (m.id === n) {
            ws!.removeEventListener("message", on);
            done(m);
          }
        };
        ws!.addEventListener("message", on);
        ws!.send(JSON.stringify({ id: n, method, params }));
      });
    const evaluate = async (expr: string) => (await call("Runtime.evaluate", { expression: expr, returnByValue: true, awaitPromise: true })).result?.result?.value;
    for (let i = 0; i < 100 && !(await evaluate("!document.getElementById('header').hidden")); i++) await new Promise((r) => setTimeout(r, 100));
    // The human picks an element: a click through the page's own input path, tried over a grid until one hits the model.
    const [w, h] = (await evaluate("[innerWidth, innerHeight]")) as [number, number];
    const selected = async () => (await evaluate("!document.getElementById('inspector').hidden && document.getElementById('inspector').innerText.length > 0")) === true;
    await new Promise((r) => setTimeout(r, 2000));
    grid: for (let gy = 1; gy < 12; gy++) {
      for (let gx = 1; gx < 16; gx++) {
        const at = { x: Math.round((gx * w) / 16), y: Math.round((gy * h) / 12) };
        await call("Input.dispatchMouseEvent", { type: "mouseMoved", ...at });
        for (const type of ["mousePressed", "mouseReleased"]) await call("Input.dispatchMouseEvent", { type, ...at, button: "left", buttons: type === "mousePressed" ? 1 : 0, clickCount: 1 });
        await new Promise((r) => setTimeout(r, 120));
        if (await selected()) break grid;
      }
    }
    await new Promise((r) => setTimeout(r, 1500));
    const look = () =>
      evaluate(
        "(async () => { const c = document.querySelector('#viewport canvas'); const b = new Uint8Array(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(c.toDataURL()))); " +
          "return { pixels: [...b].map(x => x.toString(16).padStart(2, '0')).join(''), inspector: document.getElementById('inspector').innerText, header: document.getElementById('header').innerText }; })()",
      );
    const before = (await look()) as { pixels: string; inspector: string; header: string };
    assert.ok(before.inspector.length > 0, "the human's window shows a selection in the inspector");
    const R = await render("beside-human", { spec: "apegmsh-view/1", files: { model: reactions.model }, size: [800, 600], shots: views });
    assert.equal(R.code, 0);
    await new Promise((r) => setTimeout(r, 1000));
    const after = await look();
    assert.deepEqual(after, before, "the human's window: the same pixels, selection and header after a render");
    log(`INV-5: a render beside the open window left it unchanged (selection: ${before.inspector.split("\n")[0]})`);
    ws.close();
  } finally {
    human.kill();
  }
}

// ---- 6: D3, the manifest a real run writes equals Python's Results -----------------------

for (const set of Object.keys(ORACLE.sets)) {
  const cases = ORACLE.cases.filter((c) => c.set === set);
  const R = await render(`oracle-${set}`, {
    spec: "apegmsh-view/1",
    files: { model: join(FIX, `${set}.h5`) },
    size: [800, 600],
    shots: cases.flatMap((c) => [
      { ...caseShot(c), name: c.name },
      { ...caseShot(c), name: `${c.name}.model`, chrome: "model" },
    ]),
  });
  assert.equal(R.code, 0, `${set}: ${R.shots.filter((s) => s.status !== "ok").map((s) => `${s.name}: ${s.reason}`).join("; ")}`);
  assert.deepEqual(pngs(R.dir), R.shots.map((s) => s.png).sort(), "INV-6: every PNG has its entry, and every entry its PNG");
  const m = byName(R);
  for (const c of cases) {
    sameAsOracle(m.get(c.name)!, c);
    sameAsOracle(m.get(`${c.name}.model`)!, c);
  }
  log(`  ${set}: ${cases.length} cases equal Python's Results, in both chromes`);
}
log("D3 manifest oracle: ok");

log(`render e2e: all checks passed; stills in ${OUT}`);
