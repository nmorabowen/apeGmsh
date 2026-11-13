// VU18 (#2856), end to end: one `--render` call over a two-variant study. Not
// part of `npm test`: it starts Electron, so it needs a display (Windows
// today). Run it after a build, from a source build only:
//
//   node scripts/build.mjs && node test/spec/study.e2e.ts [<study dir>] [<outdir>]
//
// `<study dir>` holds a `study.variants.json` and the VU10 pair as
// scripts/make_viewer_variants_fixture.py writes it (default: a copy of
// test/fixtures/vu10). One spec names both rows, a row by a label that
// differs only in case, a row out of range, and a shot with no row. It checks,
// and exits non-zero on the first failure:
//   1. each row's shot is drawn from its row's set: its manifest entry names
//      the row's model and results, and its contour's recorded maximum is the
//      fixture's closed form at the last step, Uy(x = 2) = 0.06 / r, with
//      r = 1 (Uncracked) and 0.35 (Cracked);
//   2. the two stills differ in their pixels (a fixed contour range, so the
//      colours differ where the values do), and the count is printed;
//   3. the refused rows write no PNG and carry their reason in manifest.json,
//      and the run exits 1 (D2: a refused shot);
//   4. the shot with no row draws the spec's set again: pixel-identical to
//      the Uncracked row's still (the spec's model is that row).
// VF3-b (#2960) then runs a second `--render` over the same study with a
// third row, `Remeshed` (test/fixtures/compare_render: the plate on another
// mesh), and VU19's `.user.h5` beside the cracked model. Its spec draws the
// Uncracked window with two compare user results, and it checks:
//   6. `footfall` of the Cracked row (one mesh with the window by `fem_hash`)
//      is drawn: the manifest names it `user.compare: {variant, entry}`, its
//      recorded extrema are VU19's closed form (0.2 at (0, 0), 1.5 at (2, 1),
//      as vu19.golden.json gives the column), and its still is written;
//   7. `footfall` of the Remeshed row (another `fem_hash`) is refused with the
//      two hashes hashes.golden.json records, writes no PNG, and the run exits 1.
// The stills and manifest.json stay in <outdir>/render and <outdir>/compare-render
// (default: a new temporary folder) for review. The run starts through a generated app folder
// that turns a modal into a stderr line, as test/spec/render.e2e.ts does.

import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { cpSync, existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import type { ManifestEntry } from "../../src/spec/manifest.ts";
import { readPng, rgbDiff } from "./png.ts";

const ROOT = fileURLToPath(new URL("../../", import.meta.url));
const electron = createRequire(import.meta.url)("electron") as string;
const FROM = resolve(process.argv[2] ?? join(ROOT, "test", "fixtures", "vu10"));
const OUT = resolve(process.argv[3] ?? mkdtempSync(join(tmpdir(), "agv-study-e2e-")));
mkdirSync(OUT, { recursive: true });
const log = (s: string) => process.stdout.write(`${s}\n`);

const APP = join(OUT, "app");
mkdirSync(APP, { recursive: true });
{
  const pkg = JSON.parse(readFileSync(join(ROOT, "package.json"), "utf8")) as { name: string; version: string };
  writeFileSync(join(APP, "package.json"), JSON.stringify({ name: pkg.name, version: pkg.version, private: true, type: "module", main: "entry.mjs" }));
}
writeFileSync(
  join(APP, "entry.mjs"),
  `import { dialog } from "electron";
const nl = String.fromCharCode(10);
dialog.showErrorBox = (t, c) => process.stderr.write("MODAL " + t + " " + c + nl);
dialog.showMessageBox = async () => (process.stderr.write("MODAL showMessageBox" + nl), { response: 0, checkboxChecked: false });
dialog.showMessageBoxSync = () => (process.stderr.write("MODAL showMessageBoxSync" + nl), 0);
await import(${JSON.stringify(pathToFileURL(join(ROOT, "dist", "main.mjs")).href)});
`,
);

// The study, copied: the run reads it and never writes it.
const STUDY = join(OUT, "study");
cpSync(FROM, STUDY, { recursive: true, filter: (src) => !src.endsWith(".py") && !src.endsWith("codes.golden.json") });
const UNCRACKED = { model: join(STUDY, "plate_uncracked.h5"), results: join(STUDY, "plate_uncracked.ladruno") };
const CRACKED = { model: join(STUDY, "cracked", "plate_cracked.h5"), results: join(STUDY, "cracked", "plate_cracked.ladruno") };

const view = { stage: 0, step: "last", camera: "top", contour: { field: "DISPLACEMENT", component: "y", range: { min: 0, max: 0.18 } } };
const spec = {
  spec: "apegmsh-view/1",
  files: { model: UNCRACKED.model, variants: join(STUDY, "study.variants.json") },
  size: [800, 500],
  chrome: "model",
  shots: [
    { name: "uncracked", variant: { index: 0, label: "Uncracked" }, ...view },
    { name: "cracked", variant: { index: 1, label: "Cracked" }, ...view },
    { name: "refused-case", variant: { index: 1, label: "cracked" }, ...view },
    { name: "refused-range", variant: { index: 7, label: "Cracked" }, ...view },
    { name: "spec-set", ...view },
    // A difference drawn from a row: the window set is the row drawn (Cracked), the compare set the other row.
    { name: "cracked-minus-uncracked", variant: { index: 1, label: "Cracked" }, compare: "Uncracked", ...view, contour: { ...view.contour, range: { min: 0, max: 0.12 } } },
  ],
};
const specPath = join(OUT, "spec.json");
writeFileSync(specPath, JSON.stringify(spec, null, 2));
const render = join(OUT, "render");

/** One `--render` run of `spec` into `out`; its exit code. A modal fails it. */
const runRender = (spec: string, out: string, user: string) =>
  new Promise<number>((done) => {
    const child = spawn(electron, [APP, `--user-data-dir=${user}`, "--mode=render", `--spec=${spec}`, `--out=${out}`], { stdio: ["ignore", "ignore", "pipe"] });
    let err = "";
    child.stderr.on("data", (d) => {
      err += String(d);
      process.stderr.write(d);
    });
    child.on("exit", (c) => {
      assert.doesNotMatch(err, /MODAL/, `a modal would have stopped on the desktop: ${err}`);
      done(c ?? -1);
    });
  });

const code = await runRender(specPath, render, join(OUT, "user"));

const manifest = JSON.parse(readFileSync(join(render, "manifest.json"), "utf8")) as { shots: ManifestEntry[] };
const entry = (name: string) => manifest.shots.find((s) => s.name === name)!;
const same = (a: string | undefined, b: string) => (a ?? "").toLowerCase() === b.toLowerCase();

// 3. D2: refused shots make the exit code 1; the other shots ran.
assert.equal(code, 1, `a run with refused shots exits 1; got ${code}`);
assert.deepEqual(manifest.shots.map((s) => [s.name, s.status]), [["uncracked", "ok"], ["cracked", "ok"], ["refused-case", "refused"], ["refused-range", "refused"], ["spec-set", "ok"], ["cracked-minus-uncracked", "ok"]]);
assert.match(entry("refused-case").reason!, /variant row 1 \("cracked"\) is not drawn: render:variant: variant row 1 is "Cracked", not "cracked"/);
assert.match(entry("refused-range").reason!, /variant row 7 \("Cracked"\) is not drawn: render:variant: the manifest has no variant row 7/);
const written = readdirSync(render).filter((f) => f.endsWith(".png")).sort();
assert.deepEqual(written, ["cracked-minus-uncracked.png", "cracked.png", "spec-set.png", "uncracked.png"], "a refused shot writes no PNG");
for (const s of manifest.shots) assert.equal(s.png === null, s.status === "refused", s.name);
log("refused rows: no PNG, reasons in manifest.json, exit 1");

// 1. Each row's shot is drawn from its row's set (closed form at the last step: Uy(2) = 0.06 / r).
for (const [name, set, r] of [["uncracked", UNCRACKED, 1], ["cracked", CRACKED, 0.35], ["spec-set", UNCRACKED, 1]] as const) {
  const e = entry(name);
  assert.ok(same(e.files.model?.path, set.model) || same(e.files.model?.path?.replace(/\//g, "\\"), set.model), `${name}: model ${e.files.model?.path}`);
  assert.ok(same(e.files.results?.path, set.results), `${name}: results ${e.files.results?.path}`);
  const c = e.contour!;
  assert.ok(c.drawn, `${name}: the contour is drawn`);
  const max = c.drawn ? c.extrema.recorded!.max.value : NaN;
  assert.ok(Math.abs(max - 0.06 / r) < 1e-9, `${name}: max Uy ${max} vs ${0.06 / r}`);
  log(`${name}: model ${e.files.model?.path}, max Uy ${max} (closed form ${0.06 / r})`);
}

// 5. A difference from a row: Cracked (the row drawn) minus Uncracked (the compare row), 0.06 (1 / 0.35 - 1) at x = 2.
{
  const e = entry("cracked-minus-uncracked");
  const c = e.contour!;
  assert.ok(c.drawn && c.source === "difference", `the difference is drawn: ${JSON.stringify(c)}`);
  if (c.drawn) {
    assert.deepEqual(c.difference, { window: "Cracked", compare: "Uncracked" });
    const max = c.extrema.drawn!.max.value;
    assert.ok(Math.abs(max - 0.06 * (1 / 0.35 - 1)) < 1e-9, `difference max ${max} vs ${0.06 * (1 / 0.35 - 1)}`);
    log(`cracked-minus-uncracked: window ${c.difference!.window}, compare ${c.difference!.compare}, max ${max} (closed form ${0.06 * (1 / 0.35 - 1)})`);
  }
  assert.ok(same(e.files.model?.path, CRACKED.model), `the difference's window set is the row drawn: ${e.files.model?.path}`);
}

// 2. The two stills differ where the variants do; 4. the spec's set again draws as the Uncracked row.
const png = (n: string) => readPng(readFileSync(join(render, `${n}.png`)));
const d = rgbDiff(png("uncracked"), png("cracked"));
assert.ok(d.count > 0, "the two variants' stills differ");
log(`pixel difference uncracked vs cracked: ${d.count} pixels of ${800 * 500}, box ${JSON.stringify(d.bbox)}`);
const back = rgbDiff(png("uncracked"), png("spec-set"));
assert.equal(back.count, 0, `the spec's set drawn again equals the Uncracked row's still; ${back.count} pixels differ`);
log("spec-set still equals the uncracked still (0 pixels differ)");
assert.ok(!existsSync(join(render, "refused-case.png")) && !existsSync(join(render, "refused-range.png")));

// VF3-b (#2960): a compare variant's user result in render mode. The VU10 study again, with the
// Remeshed row laid over it (its manifest replaces VU10's) and VU19's user file beside the cracked model.
const FIXTURES = join(ROOT, "test", "fixtures");
const STUDY2 = join(OUT, "compare-study");
cpSync(join(FIXTURES, "vu10"), STUDY2, { recursive: true, filter: (src) => !src.endsWith(".py") && !src.endsWith("codes.golden.json") });
cpSync(join(FIXTURES, "compare_render"), STUDY2, { recursive: true, filter: (src) => !src.endsWith("README.md") && !src.endsWith("hashes.golden.json") });
cpSync(join(FIXTURES, "vu19", "plate_cracked.user.h5"), join(STUDY2, "cracked", "plate_cracked.user.h5"));
const hashes = JSON.parse(readFileSync(join(FIXTURES, "compare_render", "hashes.golden.json"), "utf8")) as Record<"Uncracked" | "Cracked" | "Remeshed", { fem_hash: string }>;
assert.equal(hashes.Uncracked.fem_hash, hashes.Cracked.fem_hash, "the VU10 pair is one mesh");
assert.notEqual(hashes.Uncracked.fem_hash, hashes.Remeshed.fem_hash, "the Remeshed row is another mesh");
const footfall = JSON.parse(readFileSync(join(FIXTURES, "vu19", "vu19.golden.json"), "utf8")) as { compare_footfall: { data: number[][] } };
const column = footfall.compare_footfall.data.map((row) => row[0]!);

const user = { field: "footfall", component: "ratio", source: "user" };
const spec2 = {
  spec: "apegmsh-view/1",
  files: { model: join(STUDY2, "plate_uncracked.h5"), variants: join(STUDY2, "study.variants.json") },
  size: [800, 500],
  chrome: "model",
  shots: [
    // One mesh with the window (fem_hash): drawn on the window's nodes.
    { name: "cracked-footfall", camera: "top", compare: "Cracked", contour: user },
    // Another mesh: refused, never interpolated.
    { name: "remeshed-footfall", camera: "top", compare: "Remeshed", contour: user },
  ],
};
const spec2Path = join(OUT, "compare-spec.json");
writeFileSync(spec2Path, JSON.stringify(spec2, null, 2));
const render2 = join(OUT, "compare-render");
const code2 = await runRender(spec2Path, render2, join(OUT, "user2"));
const manifest2 = JSON.parse(readFileSync(join(render2, "manifest.json"), "utf8")) as { shots: ManifestEntry[] };
const entry2 = (name: string) => manifest2.shots.find((s) => s.name === name)!;

// 7. The refused shot makes the exit code 1; the drawn one ran.
assert.equal(code2, 1, `a run with a refused shot exits 1; got ${code2}`);
assert.deepEqual(manifest2.shots.map((s) => [s.name, s.status]), [["cracked-footfall", "ok"], ["remeshed-footfall", "refused"]]);

// 6. Drawn: named as the compare variant's entry, apart from its display string, with VU19's values.
{
  const e = entry2("cracked-footfall");
  assert.ok(same(e.files.model?.path, join(STUDY2, "plate_uncracked.h5")) || same(e.files.model?.path?.replace(/\//g, "\\"), join(STUDY2, "plate_uncracked.h5")), `the window set stays Uncracked: ${e.files.model?.path}`);
  const c = e.contour!;
  assert.ok(c.drawn, `the compare user result is drawn: ${JSON.stringify(c)}`);
  if (c.drawn) {
    assert.equal(c.source, "user");
    assert.deepEqual(c.user, { name: "Cracked: footfall", basis: "model", compare: { variant: "Cracked", entry: "footfall" } });
    assert.equal(c.component, "ratio");
    const rec = c.extrema.recorded!;
    // VU19's closed form, 0.2 + 1.2 x / 2 + 0.1 y on the 2 x 1 plate, and the column vu19.golden.json gives.
    assert.ok(Math.abs(rec.min.value - 0.2) < 1e-9 && Math.abs(rec.min.value - Math.min(...column)) < 1e-12, `min ${rec.min.value}`);
    assert.ok(Math.abs(rec.max.value - 1.5) < 1e-9 && Math.abs(rec.max.value - Math.max(...column)) < 1e-12, `max ${rec.max.value}`);
    assert.deepEqual(rec.min.position.map((v) => Math.round(v * 1e9) / 1e9), [0, 0, 0], "the minimum is at (0, 0)");
    assert.deepEqual(rec.max.position.map((v) => Math.round(v * 1e9) / 1e9), [2, 1, 0], "the maximum is at (2, 1)");
    log(`cracked-footfall: user ${JSON.stringify(c.user)}, recorded ${rec.min.value} .. ${rec.max.value} (closed form 0.2 .. 1.5)`);
  }
  assert.equal(e.png, "cracked-footfall.png");
  const still = readPng(readFileSync(join(render2, "cracked-footfall.png")));
  assert.deepEqual([still.width, still.height], [800, 500], "the still is written at the spec's size");
  // The colours drawn (#2968 review): the top camera fits the 2 x 1 plate to x 202..598, y 150..350 of
  // the still (4 x 2 cells, grid lines at x = 300, 400, 500 and y = 250). The ramp rises with x, so with
  // jet the plate is blue near x = 0 and red near x = 2; no plate pixel is the no-data grey
  // (NO_DATA_RGB, #6B7280, shaded): a grey has its channels within 30 and is brighter than a grid line.
  const rgb = (x: number, y: number) => Array.from(still.rgba.subarray((y * still.width + x) * 4, (y * still.width + x) * 4 + 3)) as [number, number, number];
  for (const y of [200, 300]) {
    const [r, g, b] = rgb(230, y);
    assert.ok(b > 2 * r && b > 2 * g, `blue near x = 0: (230, ${y}) is ${[r, g, b]}`);
    const [r2, g2, b2] = rgb(570, y);
    assert.ok(r2 > 2 * g2 && r2 > 2 * b2, `red near x = 2: (570, ${y}) is ${[r2, g2, b2]}`);
  }
  let grey = 0;
  for (let y = 154; y <= 346; y++) {
    for (let x = 206; x <= 594; x++) {
      const c = rgb(x, y);
      if (Math.max(...c) - Math.min(...c) < 30 && Math.max(...c) > 60) grey++;
    }
  }
  assert.equal(grey, 0, `no plate pixel is the no-data grey; ${grey} are`);
  log("cracked-footfall still: blue at x = 0, red at x = 2, no no-data grey on the plate");
}

// 7. Refused: two meshes, by the fem_hash each model stamps; no PNG.
{
  const e = entry2("remeshed-footfall");
  const want = `user result "footfall" of the compare variant "Remeshed" is not drawn: the two variants' models are not one mesh (fem_hash ${hashes.Uncracked.fem_hash} here, ${hashes.Remeshed.fem_hash} in "Remeshed"), so its values are not placed on this mesh; nodes are never interpolated`;
  assert.equal(e.reason, `the contour is not drawn: ${want}`, "the refusal names the two meshes");
  assert.equal(e.png, null);
  log(`remeshed-footfall: refused: ${e.reason}`);
}
assert.deepEqual(readdirSync(render2).filter((f) => f.endsWith(".png")), ["cracked-footfall.png"], "the refused shot writes no PNG");
log(`study e2e: ok (${render}, ${render2})`);
