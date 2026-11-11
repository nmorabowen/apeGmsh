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
// The stills and manifest.json stay in <outdir>/render (default: a new
// temporary folder) for review. The run starts through a generated app folder
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
  ],
};
const specPath = join(OUT, "spec.json");
writeFileSync(specPath, JSON.stringify(spec, null, 2));
const render = join(OUT, "render");

const code = await new Promise<number>((done) => {
  const child = spawn(electron, [APP, `--user-data-dir=${join(OUT, "user")}`, "--mode=render", `--spec=${specPath}`, `--out=${render}`], { stdio: ["ignore", "ignore", "pipe"] });
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

const manifest = JSON.parse(readFileSync(join(render, "manifest.json"), "utf8")) as { shots: ManifestEntry[] };
const entry = (name: string) => manifest.shots.find((s) => s.name === name)!;
const same = (a: string | undefined, b: string) => (a ?? "").toLowerCase() === b.toLowerCase();

// 3. D2: refused shots make the exit code 1; the other shots ran.
assert.equal(code, 1, `a run with refused shots exits 1; got ${code}`);
assert.deepEqual(manifest.shots.map((s) => [s.name, s.status]), [["uncracked", "ok"], ["cracked", "ok"], ["refused-case", "refused"], ["refused-range", "refused"], ["spec-set", "ok"]]);
assert.match(entry("refused-case").reason!, /variant row 1 \("cracked"\) is not drawn: render:variant: variant row 1 is "Cracked", not "cracked"/);
assert.match(entry("refused-range").reason!, /variant row 7 \("Cracked"\) is not drawn: render:variant: the manifest has no variant row 7/);
const written = readdirSync(render).filter((f) => f.endsWith(".png")).sort();
assert.deepEqual(written, ["cracked.png", "spec-set.png", "uncracked.png"], "a refused shot writes no PNG");
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

// 2. The two stills differ where the variants do; 4. the spec's set again draws as the Uncracked row.
const png = (n: string) => readPng(readFileSync(join(render, `${n}.png`)));
const d = rgbDiff(png("uncracked"), png("cracked"));
assert.ok(d.count > 0, "the two variants' stills differ");
log(`pixel difference uncracked vs cracked: ${d.count} pixels of ${800 * 500}, box ${JSON.stringify(d.bbox)}`);
const back = rgbDiff(png("uncracked"), png("spec-set"));
assert.equal(back.count, 0, `the spec's set drawn again equals the Uncracked row's still; ${back.count} pixels differ`);
log("spec-set still equals the uncracked still (0 pixels differ)");
assert.ok(!existsSync(join(render, "refused-case.png")) && !existsSync(join(render, "refused-range.png")));
log(`study e2e: ok (${render})`);
