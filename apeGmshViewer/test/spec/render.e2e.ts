// ADR 0128 slice 2 (VA2), end to end: `--render` and `--describe` in the
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
//      kill criterion: it stops here when not), and each shot's time (kill
//      criterion ~10 s); the reference ghost likewise, shot 5 times in each
//      of 2 runs in both chromes (#2664);
//   INV-1: Chromium's profile goes to `<out>/.chromium`, and the app's
//      folder under APPDATA is unchanged by a render or a describe;
//   3. INV-7: an app and a model still of one shot have the same camera pose
//      and the same contour; the scale bar is drawn unless turned off, and
//      it is the only difference; `transparent` writes PNG alpha;
//   4. each D2 shot refusal refuses with its reason, the same in both
//      chromes, and writes no PNG, while the other shots still run (exit 1);
//      each spec refusal writes render.json and no PNG (exit 2), and a
//      non-empty --out is refused with nothing written (exit 2);
//   5. a render beside an open view window leaves that window's pixels,
//      inspector and header as they were (INV-5, ADR 0128 D4).
// The stills stay in <outdir> (default: a new temporary folder) for review.

import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { copyFileSync, existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { readPng, rgbDiff } from "./png.ts";
import { makeReactionSet } from "./reactions.ts";

const ROOT = fileURLToPath(new URL("../../", import.meta.url));
const FIX = join(ROOT, "fixtures");
const electron = createRequire(import.meta.url)("electron") as string;
const OUT = resolve(process.argv[2] ?? mkdtempSync(join(tmpdir(), "agv-render-e2e-")));
mkdirSync(OUT, { recursive: true });
const log = (s: string) => process.stdout.write(`${s}\n`);

interface Shot {
  name: string;
  status: string;
  reason: string | null;
  png: string | null;
  chrome: string;
  layers: Record<string, boolean>;
  background: string;
  camera: unknown;
  contour: unknown;
  deform: unknown;
  layout: { viewport: number[]; panels: string[] } | null;
  image: [number, number] | null;
  ms: number;
}

function run(args: string[]): Promise<{ code: number; ms: number }> {
  const t = performance.now();
  return new Promise((done) => {
    const child = spawn(electron, [ROOT, ...args], { stdio: ["ignore", "ignore", "inherit"] });
    child.on("exit", (code) => done({ code: code ?? -1, ms: performance.now() - t }));
  });
}

async function render(name: string, spec: object): Promise<{ code: number; ms: number; dir: string; shots: Shot[]; refused?: { key: string; reason: string } }> {
  const dir = join(OUT, name);
  mkdirSync(dir, { recursive: true });
  const path = join(dir, "spec.json");
  writeFileSync(path, JSON.stringify(spec, null, 2));
  const out = join(dir, "out");
  const r = await run([`--mode=render`, `--spec=${path}`, `--out=${out}`]);
  const j = JSON.parse(readFileSync(join(out, "render.json"), "utf8")).render;
  return { ...r, dir: out, shots: j.shots ?? [], refused: j.refused };
}

const pngs = (dir: string) => readdirSync(dir).filter((f) => f.endsWith(".png")).sort();

/** INV-1: every file under the app's APPDATA folder, with its size and time; a run that writes there changes it. */
const APPDATA_DIR = join(process.env["APPDATA"] ?? "", "apegmsh-viewer");
function appdataListing(): string[] {
  if (!process.env["APPDATA"] || !existsSync(APPDATA_DIR)) return [];
  return (readdirSync(APPDATA_DIR, { recursive: true }) as string[])
    .map((p) => {
      const st = statSync(join(APPDATA_DIR, p));
      return `${p} ${st.isDirectory() ? "dir" : st.size} ${st.mtimeMs}`;
    })
    .sort();
}
const appdataBefore = appdataListing();

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
  if (s.chrome === "model") assert.deepEqual(s.layout!.panels, [], `${s.name}: a model still lays out no panel (DOM check)`);
  worst = Math.max(worst, s.ms);
  log(`  ${s.name}: ${pa.width}x${pa.height}, identical twice, ${Math.round(s.ms)} ms`);
}
for (const R of [A, B]) assert.ok(existsSync(join(R.dir, ".chromium")), "INV-1: Chromium's profile is inside --out");
log(`views: ${A.shots.length} stills, pixel-identical over two runs; run ${Math.round(A.ms)} / ${Math.round(B.ms)} ms; slowest shot ${Math.round(worst)} ms`);
if (worst > 10000) throw new Error(`KILL: a shot took ${Math.round(worst)} ms (> ~10 s)`);

// D9: app layers off leave the viewport's box alone and leave only what no layer names.
assert.deepEqual(a.get("contour.bare")!.layout!.viewport, a.get("contour")!.layout!.viewport);
assert.deepEqual(a.get("contour.bare")!.layout!.panels, [], "every named panel omitted from the layout");
assert.ok(a.get("contour")!.layout!.panels.includes("#header"), "the default app still keeps its panels");
// INV-7: same camera pose and same contour in both chromes.
for (const v of views) {
  const app = a.get(v.name)!, model = a.get(`${v.name}.model`)!;
  assert.deepEqual(model.camera, app.camera, `${v.name}: one camera pose in both chromes`);
  assert.deepEqual(model.contour, app.contour, `${v.name}: one contour in both chromes`);
  assert.deepEqual(model.deform, app.deform, `${v.name}: one deformed shape in both chromes`);
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
  assert.equal(a.get("contour.model")!.layers["scale_bar"], true);
  assert.equal(a.get("contour.model-noscale")!.layers["scale_bar"], false, "the result records the layer turned off");
  const t = readPng(readFileSync(join(A.dir, "contour.model-transparent.png")));
  let clear = 0;
  for (let i = 3; i < t.rgba.length; i += 4) if (t.rgba[i] === 0) clear++;
  assert.equal(t.colorType, 6);
  assert.ok(clear > t.width * t.height * 0.5, `transparent: most pixels have alpha 0 (${clear})`);
  let opaque = true;
  for (let i = 3; i < withBar.rgba.length; i += 4) if (withBar.rgba[i] !== 255) opaque = false;
  assert.ok(opaque, "background theme: every pixel opaque");
  assert.equal(a.get("contour.model-transparent")!.background, "transparent");
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
  for (const chrome of ["model", "app"]) {
    const first = still(G1, `ghost0.${chrome}`);
    // In the app chrome, below the phase bar: a hidden window's capture can catch its readout a frame old, with or without a ghost (reported on #2664).
    const box: [number, number, number, number] | undefined = chrome === "app" ? [0, 100, SIZE[0], SIZE[1]] : undefined;
    for (const R of [G1, G2]) {
      for (let i = 0; i < 5; i++) {
        const d = rgbDiff(first, still(R, `ghost${i}.${chrome}`), box);
        if (d.count) throw new Error(`KILL: ghost${i}.${chrome} differs from the first shot in ${d.count} pixels (box ${d.bbox?.join(", ")})`);
      }
    }
  }
  log("reference ghost: 5 shots in each of 2 runs, both chromes, pixel-identical");
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
for (const [name, spec, key] of [
  ["spec-version", { spec: "apegmsh-view/9", files: { model: reactions.model }, shots: [{ name: "a" }] }, "spec"],
  ["spec-key", { spec: "apegmsh-view/1", files: { model: reactions.model }, shots: [{ name: "a", zoom: 2 }] }, "shots[0].zoom"],
  ["spec-type", { spec: "apegmsh-view/1", files: { model: reactions.model }, size: [10, 10], shots: [{ name: "a" }] }, "size[0]"],
  ["spec-fit-selection", { spec: "apegmsh-view/1", files: { model: reactions.model }, shots: [{ name: "a", camera: "fit-selection" }] }, "shots[0].camera"],
] as const) {
  const R = await render(name, spec);
  assert.equal(R.code, 2, `${name}: a refused spec exits 2`);
  assert.equal(R.refused?.key, key);
  assert.deepEqual(readdirSync(R.dir).filter((f) => f !== ".chromium"), ["render.json"], `${name}: no PNG`);
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
// INV-1: nothing of the agent runs above went to the app's APPDATA folder (the human's window below does write there).
assert.deepEqual(appdataListing(), appdataBefore, `INV-1: ${APPDATA_DIR} unchanged by the agent runs`);
log(`INV-1: ${APPDATA_DIR} unchanged (${appdataBefore.length} entries before and after)`);
log("D2 refusals: ok");

// ---- 5: a render leaves the human's window alone -------------------------------------------

{
  const port = 9300 + Math.floor(Math.random() * 500);
  const human = spawn(electron, [ROOT, "--mode=view", `--file=${join(FIX, "shoebuckle.h5")}`, `--remote-debugging-port=${port}`], { stdio: "ignore" });
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

log(`render e2e: all checks passed; stills in ${OUT}`);
