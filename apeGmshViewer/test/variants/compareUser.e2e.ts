// VU19 (#2861), end to end on a real view window of the source build: the
// compare slot lists the compare set's own user results and draws one. Not
// part of `npm test`: it starts Electron, so it needs a display. Run it after
// a build:
//
//   node scripts/build.mjs && node test/variants/compareUser.e2e.ts [<outdir>]
//
// The process gets its own `--user-data-dir`, so the single-instance lock is
// this run's and no human's window on the machine is reached. A click is a DOM
// click through the window's debug port; a still is the page's own pixels
// (`Page.captureScreenshot`). It stops only the window it started.
//
// On the VU10 study with a `.user.h5` beside each model (test/fixtures/vu19),
// it writes the D7 stills into <outdir> and checks, exiting non-zero on the
// first failure:
//   1. `1-study.png`: the study open, the compare slot empty;
//   2. `2-compare-user-list.png`: Compare on Cracked; the slot lists Cracked's
//      own cases, each current against its own model (the window's own
//      footfall is untouched); only the rail's pixels change (RGB diff);
//   3. `3-compare-user-drawn.png`: Draw ratio; the legend names
//      "Cracked: footfall" and the model's pixels change (RGB diff);
//   4. `4-closed.png`: Close; the slot is gone, and a compare read through
//      the preload is then refused by main ("no compare set is open").

import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { copyFileSync, cpSync, mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { readPng, rgbDiff } from "../spec/png.ts";

const ROOT = fileURLToPath(new URL("../../", import.meta.url));
const electron = createRequire(import.meta.url)("electron") as string;
const OUT = resolve(process.argv[2] ?? mkdtempSync(join(tmpdir(), "agv-vu19-e2e-")));
mkdirSync(OUT, { recursive: true });
const log = (s: string) => process.stdout.write(`${s}\n`);
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
/** The right rail's left edge in a 1584 px wide page at device scale 1 (style.css #rail; test/spec/offers.e2e.ts). */
const RAIL_LEFT = 1094;

const study = join(OUT, "study");
cpSync(join(ROOT, "test", "fixtures", "vu10"), study, { recursive: true });
copyFileSync(join(ROOT, "test", "fixtures", "vu19", "plate_cracked.user.h5"), join(study, "cracked", "plate_cracked.user.h5"));
copyFileSync(join(ROOT, "test", "fixtures", "vu19", "plate_uncracked.user.h5"), join(study, "plate_uncracked.user.h5"));

/** The app, with its dialogs as stderr lines, so none stops on the desktop of whoever runs this. */
const ENTRY = join(OUT, "no-dialog-entry.mjs");
writeFileSync(
  ENTRY,
  `import { dialog } from "electron";
dialog.showErrorBox = (t, c) => process.stderr.write("MODAL " + t + " " + c + String.fromCharCode(10));
process.on("uncaughtException", (e) => process.stderr.write("MODAL uncaught " + (e && e.stack || e) + String.fromCharCode(10)));
await import(${JSON.stringify(pathToFileURL(join(ROOT, "dist", "main.mjs")).href)});
`,
);

/** The window's page, through its debug port: evaluate, wait, and capture. */
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
  type Reply = { result?: { result?: { value?: unknown }; exceptionDetails?: unknown; data?: string } };
  const waiting = new Map<number, (r: Reply) => void>();
  socket.onmessage = (m) => {
    const r = JSON.parse(String(m.data)) as Reply & { id?: number };
    if (r.id) waiting.get(r.id)?.(r);
  };
  const send = (method: string, params: object): Promise<Reply> => {
    const id = ++seq;
    return new Promise((ok) => {
      waiting.set(id, ok);
      socket.send(JSON.stringify({ id, method, params }));
    });
  };
  const js = async (expression: string): Promise<unknown> => {
    const r = await send("Runtime.evaluate", { expression, returnByValue: true, awaitPromise: true });
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
  const still = async (name: string): Promise<string> => {
    // Two frames, so a change just dispatched is painted.
    await js("new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)))");
    const r = await send("Page.captureScreenshot", { format: "png" });
    const path = join(OUT, name);
    writeFileSync(path, Buffer.from(r.result!.data!, "base64"));
    return path;
  };
  return { js, until, still };
}

/** Pixels that differ in RGB (alpha ignored) left of and right of the rail's edge. */
function changed(a: string, b: string): { model: number; rail: number } {
  const pa = readPng(readFileSync(a));
  const pb = readPng(readFileSync(b));
  assert.equal(pa.width, pb.width);
  return { model: rgbDiff(pa, pb, [0, 0, RAIL_LEFT, pa.height]).count, rail: rgbDiff(pa, pb, [RAIL_LEFT, 0, pa.width, pa.height]).count };
}

const panelText = `document.getElementById("variants")?.innerText ?? ""`;
const variantRow = (label: string) => `[...document.querySelectorAll("#variants [data-variant]")].find((r) => r.querySelector(".offer-name").textContent === ${JSON.stringify(label)})`;
const buttonIn = (row: string, text: string) => `[...${row}.querySelectorAll("button")].find((b) => b.textContent === ${JSON.stringify(text)})`;

const port = 9400 + Math.floor(Math.random() * 400);
const user = join(OUT, "user");
const view = spawn(electron, [ENTRY, `--user-data-dir=${user}`, "--force-device-scale-factor=1", `--remote-debugging-port=${port}`, "--mode=view", `--file=${join(study, "study.variants.json")}`], { stdio: ["ignore", "ignore", "pipe"] });
let viewErr = "";
view.stderr.on("data", (d) => (viewErr += String(d)));
let failed = false;
try {
  const page = await pageOf(port);
  await page.until(`document.getElementById("header") && !document.getElementById("header").hidden`, "the model");
  await page.until(`!document.getElementById("variants").hidden && ${variantRow("Cracked")} !== undefined`, "the variants panel");
  await sleep(1500);
  const s1 = await page.still("1-study.png");
  assert.doesNotMatch(String(await page.js(panelText)), /user file/, "no compare slot yet");
  log(`1. study open, compare slot empty: ${s1}`);

  await page.js(`${buttonIn(variantRow("Cracked"), "Compare")}.click()`);
  await page.until(`document.querySelectorAll("#variants .compare-user [data-compare-user]").length === 2`, "the compare set's user results");
  await sleep(800);
  const text2 = String(await page.js(panelText));
  assert.match(text2, /Cracked's user file: plate_cracked\.user\.h5/);
  assert.match(text2, /footfall · on nodes · basis model\s*current/);
  assert.match(text2, /drift · on nodes · basis results\s*current/);
  assert.match(text2, /User Combinations: an empty zone/);
  const s2 = await page.still("2-compare-user-list.png");
  const d12 = changed(s1, s2);
  assert.ok(d12.rail > 0, "the rail shows the compare slot");
  log(`2. compare slot lists Cracked's own cases, each current against its own model: ${s2} (RGB pixels changed: model ${d12.model}, rail ${d12.rail})`);

  const ff = `[...document.querySelectorAll("#variants [data-compare-user]")].find((r) => r.dataset.compareUser === "footfall")`;
  await page.js(`${buttonIn(ff, "Draw ratio")}.click()`);
  await page.until(`document.body.innerText.includes("Cracked: footfall")`, "the legend naming the compare user result");
  await sleep(1500);
  const s3 = await page.still("3-compare-user-drawn.png");
  const d23 = changed(s2, s3);
  assert.ok(d23.model > 1000, `the model is coloured by the compare user result (${d23.model} pixels changed)`);
  log(`3. Cracked's footfall ratio drawn on the window's mesh: ${s3} (RGB pixels changed: model ${d23.model}, rail ${d23.rail})`);

  await page.js(`${buttonIn(variantRow("Cracked"), "Close")}.click()`);
  await page.until(`!(${panelText}).includes("user file")`, "the compare slot closed");
  await sleep(1000);
  const refused = (await page.js(`window.viewer.compareUserRead().then((a) => JSON.stringify(a))`)) as string;
  assert.match(refused, /no compare set is open/);
  const refusedCol = (await page.js(`window.viewer.compareUserReadResult({ name: "footfall", component: "ratio" }).then((a) => JSON.stringify(a))`)) as string;
  assert.match(refusedCol, /no compare set is open/);
  const s4 = await page.still("4-closed.png");
  writeFileSync(join(OUT, "4-refused-after-close.json"), `${refused}\n${refusedCol}\n`);
  log(`4. after Close, the compare reads are refused by main (${refused}): ${s4}`);
  assert.doesNotMatch(viewErr, /MODAL/, viewErr);
} catch (err) {
  failed = true;
  process.stderr.write(`${err instanceof Error ? err.stack : String(err)}\n${viewErr}\n`);
} finally {
  if (view.exitCode === null && view.signalCode === null) {
    const gone = new Promise((ok) => view.once("exit", ok));
    view.kill();
    await gone;
  }
}
log(`stills in ${OUT}`);
process.exit(failed ? 1 : 0);
