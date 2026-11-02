// ADR 0128 slice 5 (VA5), end to end: `--present` and `--snapshot` as real
// processes beside a real view window, on the fixtures. Not part of `npm test`:
// it starts Electron, so it needs a display. Run it after a build:
//
//   node scripts/build.mjs && node test/spec/offers.e2e.ts [<outdir>]
//
// Every process gets its own `--user-data-dir`, so the single-instance lock is
// this run's and no human's window on the machine is reached. A click is a DOM
// click through the window's debug port. It stops only the window it started.
//
// It checks, in the dark theme and in a spec-named light theme, and exits
// non-zero on the first failure:
//   1. `--present` and `--snapshot` with no window exit 1 and open nothing; a
//      refused spec and a non-empty `--out` exit 2;
//   2. INV-5: offers arriving leave the view read back as it was, and only the
//      rail's pixels change;
//   3. INV-3: Apply, then `--snapshot`, returns the presented spec, with its
//      D3 manifest entry;
//   4. D5: Back leaves 0 RGB pixels different from before the click;
//   5. D1: a plane given as normal [2,0,0], offset 4 reads back in those words.
// The stills stay in <outdir> (default: a new temporary folder) for review.

import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { readPng, rgbDiff } from "./png.ts";

const ROOT = fileURLToPath(new URL("../../", import.meta.url));
const FIX = join(ROOT, "fixtures");
const electron = createRequire(import.meta.url)("electron") as string;
const OUT = resolve(process.argv[2] ?? mkdtempSync(join(tmpdir(), "agv-offers-e2e-")));
mkdirSync(OUT, { recursive: true });
const log = (s: string) => process.stdout.write(`${s}\n`);
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

const MODEL = join(FIX, "contour_staged.h5");
const RESULTS = join(FIX, "contour_staged.results.h5");
/** The right rail's left edge in a 1584 px wide page at device scale 1: 1584 - 20 (`right`) - 470 (`width`). style.css #rail. */
const RAIL_LEFT = 1094;

type Doc = Record<string, unknown>;

function run(user: string, ...args: string[]): Promise<{ code: number | null; text: string }> {
  return new Promise((done) => {
    const c = spawn(electron, [ROOT, `--user-data-dir=${user}`, ...args], { stdio: ["ignore", "pipe", "pipe"] });
    let text = "";
    c.stdout.on("data", (d) => (text += String(d)));
    c.stderr.on("data", (d) => (text += String(d)));
    const killer = setTimeout(() => c.kill(), 90000);
    c.on("exit", (code) => {
      clearTimeout(killer);
      done({ code, text });
    });
  });
}

/** The window's page, through its debug port. */
async function pageOf(port: number) {
  let ws: WebSocket | null = null;
  for (let i = 0; i < 200 && !ws; i++) {
    try {
      const list = (await (await fetch(`http://127.0.0.1:${port}/json`)).json()) as { type: string; webSocketDebuggerUrl: string }[];
      const page = list.find((t) => t.type === "page");
      if (page) {
        const open = new WebSocket(page.webSocketDebuggerUrl);
        await new Promise((ok, bad) => {
          open.onopen = ok;
          open.onerror = bad;
        });
        ws = open;
      }
    } catch {
      await sleep(100);
    }
    if (!ws) await sleep(100);
  }
  if (!ws) throw new Error(`no page on debug port ${port}`);
  const socket = ws;
  let seq = 0;
  const waiting = new Map<number, (r: { result?: { result?: { value?: unknown }; exceptionDetails?: unknown } }) => void>();
  socket.onmessage = (m) => {
    const r = JSON.parse(String(m.data)) as { id?: number; result?: { result?: { value?: unknown }; exceptionDetails?: unknown } };
    if (r.id) waiting.get(r.id)?.(r);
  };
  const js = async (expression: string): Promise<unknown> => {
    const id = ++seq;
    const r = await new Promise<{ result?: { result?: { value?: unknown }; exceptionDetails?: unknown } }>((ok) => {
      waiting.set(id, ok);
      socket.send(JSON.stringify({ id, method: "Runtime.evaluate", params: { expression, returnByValue: true } }));
    });
    if (r.result?.exceptionDetails) throw new Error(`${expression}: ${JSON.stringify(r.result.exceptionDetails)}`);
    return r.result?.result?.value;
  };
  const until = async (expression: string, what: string, ms = 60000) => {
    const t = Date.now();
    while (Date.now() - t < ms) {
      if (await js(expression).catch(() => false)) return;
      await sleep(100);
    }
    throw new Error(`timed out waiting for ${what}`);
  };
  return { js, until };
}

const rowOf = (name: string) => `[...document.querySelectorAll(".offer-row")].find((r) => r.querySelector(".offer-name").textContent === ${JSON.stringify(name)})`;
const statusIs = (name: string, status: string) => `Boolean(${rowOf(name)}?.querySelector(".offer-status").classList.contains(${JSON.stringify(status)}))`;
const applyClick = (name: string) => `${rowOf(name)}.querySelector(".offer-apply").click()`;
const png = (path: string) => readPng(readFileSync(path));

async function campaign(theme: "dark" | "light", port: number): Promise<void> {
  const dir = join(OUT, theme);
  mkdirSync(dir, { recursive: true });
  const user = join(dir, "user");
  const top = theme === "light" ? { theme: "light" } : {};
  const specFile = (name: string, shots: Doc[]) => {
    const p = join(dir, `${name}.json`);
    writeFileSync(p, JSON.stringify({ spec: "apegmsh-view/1", files: { model: MODEL, results: RESULTS }, shots, ...top }));
    return p;
  };
  const snap = async (name: string) => {
    const out = join(dir, name);
    const r = await run(user, "--mode=snapshot", `--out=${out}`);
    assert.equal(r.code, 0, `snapshot ${name}: ${r.text}`);
    for (const f of ["view.json", "view.png", "manifest.json"]) assert.ok(existsSync(join(out, f)), `snapshot ${name} wrote ${f}`);
    return {
      png: join(out, "view.png"),
      view: JSON.parse(readFileSync(join(out, "view.json"), "utf8")) as { files: Doc; theme: string; shots: Doc[] },
      manifest: JSON.parse(readFileSync(join(out, "manifest.json"), "utf8")) as { manifest: string; app: { version: unknown }; timing: { written_at: unknown }; shots: Doc[] },
    };
  };

  const shown = { name: "uz-last", caption: "Vertical displacement, stage unprop, last step.", stage: "unprop", step: "last", contour: { field: "displacement", component: "z", range: "history", cmap: "jet" }, deform: { scale: 25 }, camera: "iso" };
  const specShown = specFile("shown", [shown]);

  // 1. No window: non-zero, and nothing opens or is written.
  const none = await run(user, "--mode=present", `--spec=${specShown}`);
  assert.equal(none.code, 1, none.text);
  assert.match(none.text, /no apeGmshViewer window is running/);
  const noneOut = join(dir, "no-window");
  const noneSnap = await run(user, "--mode=snapshot", `--out=${noneOut}`);
  assert.equal(noneSnap.code, 1, noneSnap.text);
  assert.equal(existsSync(noneOut), false, "with no window, a snapshot writes nothing");
  log(`${theme}: --present and --snapshot with no window exit 1, nothing opened`);
  const bad = await run(user, "--mode=present", `--spec=${specFile("bad", [{ name: "x", colur_by: "group" }])}`);
  assert.equal(bad.code, 2, bad.text);
  assert.match(bad.text, /shots\[0\]\.colur_by/);

  const view = spawn(electron, [ROOT, `--user-data-dir=${user}`, "--force-device-scale-factor=1", `--remote-debugging-port=${port}`, "--mode=view", `--file=${MODEL}`], { stdio: "ignore" });
  try {
    const page = await pageOf(port);
    await page.until(`document.getElementById("header") && !document.getElementById("header").hidden`, "the model");
    await sleep(1500);

    const busy = join(dir, "busy");
    mkdirSync(busy, { recursive: true });
    writeFileSync(join(busy, "x"), "x");
    assert.equal((await run(user, "--mode=snapshot", `--out=${busy}`)).code, 2, "a non-empty --out is refused");

    const s0 = await snap("before");
    assert.equal(s0.manifest.manifest, "apegmsh-manifest/1");
    assert.equal(typeof s0.manifest.app.version, "string", "the caller adds the app's version");
    assert.equal(typeof s0.manifest.timing.written_at, "string");
    assert.deepEqual(s0.manifest.shots[0]!["selection"], []);

    // 2. INV-5: the offers arrive, one refused, and the view stays.
    for (const [name, shots] of [
      ["refused", [{ name: "no-stage", caption: "A stage the results do not hold.", stage: "roof" }]],
      ["other", [{ name: "deformed", caption: "Deformed shape, stage unprop.", stage: "unprop", deform: { scale: 25 }, camera: "iso" }]],
      ["shown", [shown]],
    ] as [string, Doc[]][]) {
      const r = await run(user, "--mode=present", `--spec=${name === "shown" ? specShown : specFile(name, shots)}`);
      assert.equal(r.code, 0, r.text);
    }
    await page.until(`${statusIs("no-stage", "refused")} && ${statusIs("uz-last", "pending")}`, "the list");
    assert.equal(await page.js(`document.querySelectorAll(".offer-row").length`), 3);
    await sleep(600);
    const s1 = await snap("arrived");
    assert.deepEqual(s1.view, s0.view, "INV-5: the view read back is unchanged by the offers' arrival");
    const size = png(s1.png);
    assert.equal(size.width, 1584, "the rail's edge below assumes this page width");
    const left = rgbDiff(png(s0.png), png(s1.png), [0, 0, RAIL_LEFT, size.height]);
    assert.equal(left.count, 0, `INV-5: arrival changed pixels outside the rail, in ${JSON.stringify(left.bbox)}`);
    assert.ok(rgbDiff(png(s0.png), png(s1.png)).count > 0, "the offers are listed in the rail");
    log(`${theme}: INV-5: arrival changes only the rail's pixels`);

    // 3. INV-3: the click, then the view read back.
    await page.js(applyClick("uz-last"));
    await page.until(statusIs("uz-last", "applied"), "the applied offer");
    await sleep(1000);
    const s2 = await snap("applied");
    const got = s2.view.shots[0]!;
    const { contour, deform, ...plain } = shown;
    for (const [k, v] of Object.entries(plain)) assert.deepEqual(got[k], v, `INV-3: ${k}`);
    assert.deepEqual(got["contour"], { ...contour, gauss: "cell" }, "INV-3: the contour, in its normal form");
    assert.deepEqual(got["deform"], { ...deform, ghost: false }, "INV-3: the deformed shape, in its normal form");
    assert.deepEqual(s2.view.files, { model: MODEL, results: RESULTS });
    assert.equal(s2.view.theme, theme, "D8: a named theme applies; an unnamed one leaves the human's");
    const e = s2.manifest.shots[0] as { stage: { name: string }; step: number; outside: unknown[]; contour: { drawn: boolean; extrema: { recorded: unknown } } };
    assert.equal(e.stage.name, "unprop");
    assert.equal(e.step, 3);
    assert.deepEqual(e.outside, []);
    assert.ok(e.contour.drawn && e.contour.extrema.recorded, "the entry states the extrema");
    log(`${theme}: INV-3: apply then snapshot gives the presented spec`);

    // 4. D5: Back.
    await page.js(`document.querySelector(".offer-back").click()`);
    await page.until(statusIs("uz-last", "pending"), "Back");
    await sleep(1000);
    const s3 = await snap("back");
    const back = rgbDiff(png(s1.png), png(s3.png));
    assert.equal(back.count, 0, `D5: Back left ${back.count} RGB pixels different, in ${JSON.stringify(back.bbox)}`);
    log(`${theme}: D5: Back gives 0 RGB pixels different`);

    // 5. D1: a plane with a non-unit normal, read back in the spec's words.
    assert.equal((await run(user, "--mode=present", `--spec=${specFile("clip", [{ name: "clip-x2", clip: [{ normal: [2, 0, 0], offset: 4 }], camera: "iso" }])}`)).code, 0);
    await page.until(statusIs("clip-x2", "pending"), "the plane's offer");
    await page.js(applyClick("clip-x2"));
    await page.until(statusIs("clip-x2", "applied"), "the plane's click");
    await sleep(800);
    assert.deepEqual((await snap("clip")).view.shots[0]!["clip"], [{ normal: [2, 0, 0], offset: 4, flip: false }]);
    log(`${theme}: D1: the plane reads back`);
  } finally {
    view.kill();
    await new Promise((ok) => view.on("exit", ok));
    await sleep(500);
  }
  // With the window gone, there is nothing to reach again.
  assert.equal((await run(user, "--mode=present", `--spec=${specShown}`)).code, 1, "the window closed: --present exits 1 again");
}

await campaign("dark", 19321);
await campaign("light", 19322);
log(`offers e2e: all checks passed; stills in ${OUT}`);
process.exit(0);
