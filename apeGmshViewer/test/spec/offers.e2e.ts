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
//   5. D1: a plane given as normal [2,0,0], offset 4 reads back in those words;
//   6. nothing an agent sends raises a modal: a spec with a wrong suffix, one
//      swapped under the window's read, a model gone before the click, a folder;
//   7. exit 0 only on the window's acknowledgement: concurrent `--present` with
//      no window all exit non-zero;
//   8. the ADR's spellings work, and a request that cannot be carried out exits
//      2 at once, never view mode.
// A dialog would stop on the desktop of whoever runs this, so every process
// starts through a generated entry that turns `dialog.showErrorBox` and an
// uncaught error into stderr lines, and the run fails on any such line.
// The stills stay in <outdir> (default: a new temporary folder) for review.

import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { copyFileSync, existsSync, mkdirSync, mkdtempSync, readFileSync, renameSync, unlinkSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
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

/** The app, with its dialogs as stderr lines: see the header. */
const ENTRY = join(OUT, "no-dialog-entry.mjs");
writeFileSync(
  ENTRY,
  `import { dialog } from "electron";
dialog.showErrorBox = (t, c) => process.stderr.write("MODAL " + t + " " + c + String.fromCharCode(10));
process.on("uncaughtException", (e) => process.stderr.write("MODAL uncaught " + (e && e.stack || e) + String.fromCharCode(10)));
await import(${JSON.stringify(pathToFileURL(join(ROOT, "dist", "main.mjs")).href)});
`,
);
const noModal = (text: string, what: string) => assert.doesNotMatch(text, /MODAL/, `${what}: ${text}`);

function runWithin(ms: number, user: string, ...args: string[]): Promise<{ code: number | null; text: string }> {
  return new Promise((done) => {
    const c = spawn(electron, [ENTRY, `--user-data-dir=${user}`, ...args], { stdio: ["ignore", "pipe", "pipe"] });
    let text = "";
    c.stdout.on("data", (d) => (text += String(d)));
    c.stderr.on("data", (d) => (text += String(d)));
    const killer = setTimeout(() => {
      text += "\nTIMED OUT: still running";
      c.kill();
    }, ms);
    c.on("exit", (code) => {
      clearTimeout(killer);
      noModal(text, args.join(" "));
      done({ code, text });
    });
  });
}

const run = (user: string, ...args: string[]) => runWithin(90000, user, ...args);

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

  const view = spawn(electron, [ENTRY, `--user-data-dir=${user}`, "--force-device-scale-factor=1", `--remote-debugging-port=${port}`, "--mode=view", `--file=${MODEL}`], { stdio: ["ignore", "ignore", "pipe"] });
  let viewErr = "";
  view.stderr.on("data", (d) => (viewErr += String(d)));
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

    // ADR 0128 Amendment 2: the camera is the contract, with the rail open, in the real viewport.
    const pose = { eye: [30, -40, 25], target: [2, 1, 3], up: [0, 0, 1] };
    assert.equal((await run(user, "--mode=present", `--spec=${specFile("pose", [{ name: "pose", camera: pose }])}`)).code, 0);
    await page.until(statusIs("pose", "pending"), "the camera's offer");
    assert.equal(await page.js(`getComputedStyle(document.getElementById("rail")).display !== "none"`), true, "the rail is open");
    await page.js(applyClick("pose"));
    await page.until(statusIs("pose", "applied"), "the camera's click");
    await sleep(800);
    const cam = (await snap("pose")).manifest.shots[0]!["camera"] as { eye: number[]; target: number[] };
    for (const k of ["eye", "target"] as const) cam[k].forEach((x, i) => assert.ok(Math.abs(x - pose[k][i]!) < 1e-6, `the camera's ${k}: ${cam[k].join(", ")}`));
    assert.equal(await page.js(`getComputedStyle(document.getElementById("rail")).display !== "none"`), true, "Apply did not collapse the rail");
    log(`${theme}: Amendment 2: the camera after Apply is the spec's`);
    noModal(viewErr, "the view window");
  } finally {
    view.kill();
    await new Promise((ok) => view.on("exit", ok));
    await sleep(500);
  }
  // With the window gone, there is nothing to reach again.
  assert.equal((await run(user, "--mode=present", `--spec=${specShown}`)).code, 1, "the window closed: --present exits 1 again");
}

/** 6, 7, 8: what must not stop on the human's desktop, and what an exit code may claim. */
async function robustness(port: number): Promise<void> {
  const dir = join(OUT, "robust");
  mkdirSync(join(dir, "folder.h5"), { recursive: true });
  const user = join(dir, "user");
  const spec = (name: string, doc: Doc) => {
    const p = join(dir, `${name}.json`);
    writeFileSync(p, JSON.stringify({ spec: "apegmsh-view/1", ...doc }));
    return p;
  };
  const good = spec("good", { files: { model: MODEL, results: RESULTS }, shots: [{ name: "plain", camera: "iso" }] });

  // 7. No window, on a fresh profile, under concurrent calls: nobody may claim a window took it (one holds the lock while another is denied it).
  for (const [n, rounds] of [[16, 4], [8, 4], [4, 6]] as const) {
    for (let k = 0; k < rounds; k++) {
      const u = join(dir, `race-${n}-${k}`);
      const rs = await Promise.all(Array.from({ length: n }, () => run(u, "--mode=present", `--spec=${good}`)));
      assert.deepEqual(rs.map((r) => r.code), rs.map(() => 1), `${n} concurrent --present with no window: ${rs.map((r) => r.text).join("|")}`);
    }
  }
  log("exit 0 needs the window: 14 rounds of concurrent --present with no window all exit 1");

  // 8. The spellings with no window, and requests that cannot be carried out: an exit at once, not a view window.
  assert.equal((await runWithin(20000, user, "--present", good)).code, 1, "--present <spec> with no window");
  const noWin = join(dir, "lit-nowin");
  assert.equal((await runWithin(20000, user, "--snapshot", "--out", noWin)).code, 1, "--snapshot --out with no window");
  assert.equal(existsSync(noWin), false);
  for (const a of [["--present"], ["--snapshot"], ["--render", good], ["--describe", MODEL], ["--mode=presnt", `--spec=${good}`], ["--mode=present", `--specc=${good}`], [`--spec=${good}`], ["--present", good, "--out", dir]]) {
    const r = await runWithin(20000, user, ...a);
    assert.equal(r.code, 2, `${a.join(" ")}: ${r.text}`);
  }
  const rendered = join(dir, "lit-render");
  assert.equal((await runWithin(120000, user, "--render", good, "--out", rendered)).code, 0, "--render <spec> --out");
  assert.ok(existsSync(join(rendered, "plain.png")) && existsSync(join(rendered, "manifest.json")));
  const described = join(dir, "lit-describe");
  assert.equal((await runWithin(60000, user, "--describe", MODEL, "--out", described)).code, 0, "--describe <file> --out");
  assert.ok(existsSync(join(described, "describe.json")));
  log("spellings: the ADR's four work; requests that cannot be carried out exit 2");

  const view = spawn(electron, [ENTRY, `--user-data-dir=${user}`, "--force-device-scale-factor=1", `--remote-debugging-port=${port}`, "--mode=view", `--file=${MODEL}`], { stdio: ["ignore", "ignore", "pipe"] });
  let viewErr = "";
  view.stderr.on("data", (d) => (viewErr += String(d)));
  const clean = (what: string) => noModal(viewErr, what);
  try {
    const page = await pageOf(port);
    await page.until(`document.getElementById("header") && !document.getElementById("header").hidden`, "the model");
    await sleep(1200);
    assert.equal((await runWithin(20000, user, "--present", good)).code, 0, "--present <spec> into the window");
    await page.until(statusIs("plain", "pending"), "the literal --present's offer");
    const lit = join(dir, "lit-snap");
    assert.equal((await runWithin(30000, user, "--snapshot", "--out", lit)).code, 0, "--snapshot --out");
    assert.ok(existsSync(join(lit, "manifest.json")));

    // 6. Static: the sender refuses these, whatever the suffix.
    for (const [n, files] of [["win", { model: "C:/Windows/win.ini" }], ["res", { model: MODEL, results: "C:/Windows/win.ini" }], ["txt", { model: "x.txt" }]] as const) {
      const r = await run(user, "--mode=present", `--spec=${spec(n, { files, shots: [{ name: n }] })}`);
      assert.equal(r.code, 2, `${n}: ${r.text}`);
    }
    clean("wrong suffixes");

    // 6. The swap race: the file flips between the sender's check and the window's read.
    const flipped = join(dir, "flip.json");
    const tmp = join(dir, "flip.tmp");
    const goodDoc = JSON.stringify({ spec: "apegmsh-view/1", files: { model: MODEL }, shots: [{ name: "flip" }] });
    const badDoc = JSON.stringify({ spec: "apegmsh-view/1", files: { model: "C:/Windows/win.ini" }, shots: [{ name: "flip" }] });
    writeFileSync(flipped, goodDoc);
    let on = true;
    const flipper = setInterval(() => {
      try {
        writeFileSync(tmp, on ? badDoc : goodDoc);
        renameSync(tmp, flipped);
        on = !on;
      } catch {
        // A rename can lose to a reader on Windows; the next tick flips.
      }
    }, 2);
    const codes: (number | null)[] = [];
    for (let i = 0; i < 12; i++) {
      const rs = await Promise.all([0, 1, 2].map(() => run(user, "--mode=present", `--spec=${flipped}`)));
      codes.push(...rs.map((r) => r.code));
    }
    clearInterval(flipper);
    await sleep(500);
    clean("the swap race");
    assert.ok(codes.every((c) => c === 0 || c === 2), `the swap race exits 0 or 2: ${codes.join("")}`);
    log(`swap race: ${codes.filter((c) => c === 0).length} taken, ${codes.filter((c) => c === 2).length} refused, no modal`);

    // 6. The click: a model gone before it, a folder, a file that is not HDF5.
    const gone = join(dir, "gone.h5");
    copyFileSync(MODEL, gone);
    const garbage = join(dir, "garbage.h5");
    writeFileSync(garbage, "not hdf5");
    for (const [n, model] of [["cgone", gone], ["cfolder", join(dir, "folder.h5")], ["cgarbage", garbage]] as const) {
      assert.equal((await run(user, "--mode=present", `--spec=${spec(n, { files: { model }, shots: [{ name: n }] })}`)).code, 0, n);
      await page.until(`Boolean(${rowOf(n)})`, n);
      await sleep(200);
      if (n === "cgone") unlinkSync(gone);
      await page.js(applyClick(n));
      await page.until(statusIs(n, "refused"), `${n} refused`);
      await page.until(`document.querySelector(".offer-apply") && !document.querySelector(".offer-back").matches(":disabled") || !document.querySelector(".offer-row.applying")`, "the click's end");
      await sleep(600);
      clean(n);
      await page.until(`document.getElementById("file-name").textContent.startsWith("contour_staged")`, `${n}: the set it came from`, 20000);
      log(`click on ${n}: refused in the list, no modal, the view back: ${String(await page.js(`${rowOf(n)}.querySelector(".offer-reason")?.textContent`)).slice(0, 90)}`);
    }
    clean("the view window");
  } finally {
    view.kill();
    await new Promise((ok) => view.on("exit", ok));
    await sleep(500);
  }
}

await robustness(19323);
await campaign("dark", 19321);
await campaign("light", 19322);
log(`offers e2e: all checks passed; stills in ${OUT}`);
process.exit(0);
