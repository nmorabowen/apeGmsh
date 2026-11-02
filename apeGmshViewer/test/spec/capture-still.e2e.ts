// VA2k (#2716), end to end: a D7 capture (`--mode=capture`) writes each still
// from the page as it settled, as `--render` does for an app still since #2696.
// A hidden window's `capturePage` copies the last frame the window showed, so
// one capture can return a frame from before the page's latest changes. Not
// part of `npm test`: it starts Electron, so it needs a display. Run it after a
// build:
//
//   node scripts/build.mjs && node test/spec/capture-still.e2e.ts [<outdir>]
//
// It checks, and exits non-zero on the first failure:
//   1. a capture whose every still's first `capturePage` returns a stale frame
//      (the frame of the capture before it; a grey page for the first still)
//      still writes every still exactly as a normal run of the same model
//      does, compared in RGB, and the stale frame was served once per still
//      (so the check is not vacuous);
//   2. a page that never settles (each capture differs from the one before it
//      in one byte) fails the run: exit 1, no still written, and
//      `<out>.error.txt` names the still and the `--capture-tries` bound;
//   3. a run that begins ending (`headlessEnd`) as a still's capture settles
//      writes no still beside `<out>.error.txt` (#2736): the failure is forced
//      synchronously inside the settling `capturePage`, so main's code after
//      that await runs with the run already ending;
//   4. a write of a still that throws part way (a full disk) leaves no
//      partial PNG, and the run fails: exit 1 with `<out>.error.txt` (#2736).
//
// Every process starts through a generated entry that forces the fault and
// turns `dialog.showErrorBox` into a stderr line, so nothing stops on the
// desktop of whoever runs this; the run fails on any such line. A process still
// running at its limit is stopped by its own PID (and that PID's tree), never
// by name.

import assert from "node:assert/strict";
import { execFileSync, spawn } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { readPng, rgbDiff } from "./png.ts";

const ROOT = fileURLToPath(new URL("../../", import.meta.url));
const MODEL = join(ROOT, "fixtures", "shoebuckle.h5");
const electron = createRequire(import.meta.url)("electron") as string;
const OUT = resolve(process.argv[2] ?? mkdtempSync(join(tmpdir(), "agv-capture-still-e2e-")));
mkdirSync(OUT, { recursive: true });
const log = (s: string) => process.stdout.write(`${s}\n`);
/** How long a run may take before it counts as hung: it is stopped by its PID, and the check fails. */
const LIMIT_MS = 120000;

const ENTRY = join(OUT, "fault-entry.mjs");
writeFileSync(
  ENTRY,
  `import { app, dialog, ipcMain, nativeImage } from "electron";
import fs from "node:fs";
import { syncBuiltinESMExports } from "node:module";
const nl = String.fromCharCode(10);
dialog.showErrorBox = (t, c) => process.stderr.write("MODAL " + t + " " + c + nl);
const fault = process.env.AGV_FAULT ?? "";
// Each still's first capture returns the image the capture before it returned (a grey page for the first still).
if (fault === "stale-first") {
  let first = false;
  let last = null;
  const handle = ipcMain.handle.bind(ipcMain);
  ipcMain.handle = (channel, fn) => handle(channel, channel !== "capture:still" ? fn : (e, ...a) => ((first = true), fn(e, ...a)));
  app.on("web-contents-created", (_e, wc) => {
    const real = wc.capturePage.bind(wc);
    wc.capturePage = async (...a) => {
      const img = await real(...a);
      const stale = first ? (last ?? nativeImage.createFromBitmap(Buffer.alloc(img.toBitmap().length, 0x80), img.getSize())) : img;
      if (first) process.stderr.write("AGV_STALE served" + nl);
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
// The run begins ending as the first still settles: on the capture that matches the one before it,
// main's uncaught-error listener (headlessEnd) runs before that capture is handed back.
if (fault === "end-as-settles") {
  let fired = false;
  app.on("web-contents-created", (_e, wc) => {
    const real = wc.capturePage.bind(wc);
    let prev = null;
    wc.capturePage = async (...a) => {
      const img = await real(...a);
      if (!fired && prev && img.toBitmap().equals(prev.toBitmap())) {
        fired = true;
        process.stderr.write("AGV_END fired" + nl);
        process.emit("uncaughtException", new Error("AGV_FAULT: the run ends as the still settles"));
      }
      prev = img;
      return img;
    };
  });
}
// A write of the first still that throws part way, as a full disk does.
if (fault === "partial") {
  const target = process.env.AGV_TARGET ?? "";
  const real = fs.writeFileSync;
  fs.writeFileSync = function (path, data, ...rest) {
    if (String(path).toLowerCase() === target.toLowerCase()) {
      real.call(this, path, data.subarray(0, 64));
      throw new Error("AGV_FAULT: ENOSPC, a forced partial write");
    }
    return real.call(this, path, data, ...rest);
  };
  syncBuiltinESMExports();
}
await import(${JSON.stringify(pathToFileURL(join(ROOT, "dist", "main.mjs")).href)});
`,
);

interface Ran {
  code: number | null;
  text: string;
  /** the folder holding `still.png` and the stills beside it */
  dir: string;
  out: string;
  stills: string[];
}

/** One capture of the model into its own folder, with its own profile folder; stopped by its PID tree at the limit. */
function capture(name: string, env: Record<string, string>, ...args: string[]): Promise<Ran> {
  const dir = join(OUT, name);
  mkdirSync(dir, { recursive: true });
  const out = join(dir, "still.png");
  return new Promise((done) => {
    const c = spawn(electron, [ENTRY, `--user-data-dir=${join(OUT, "users", name)}`, "--mode=capture", `--file=${MODEL}`, `--out=${out}`, ...args], {
      stdio: ["ignore", "pipe", "pipe"],
      env: { ...process.env, ...env },
    });
    let text = "";
    c.stdout.on("data", (d) => (text += String(d)));
    c.stderr.on("data", (d) => (text += String(d)));
    const killer = setTimeout(() => {
      try {
        if (process.platform === "win32") execFileSync("taskkill", ["/PID", String(c.pid), "/T", "/F"], { stdio: "ignore" });
        else c.kill("SIGKILL");
      } catch {
        // It exited between the timer and the kill.
      }
    }, LIMIT_MS);
    c.on("exit", (code) => {
      clearTimeout(killer);
      assert.doesNotMatch(text, /MODAL/, `${name}: a modal would have stopped on the desktop: ${text}`);
      const stills = readdirSync(dir).filter((f) => f.endsWith(".png")).sort();
      done({ code, text, dir, out, stills });
    });
  });
}

// ---- the reference: a normal capture -------------------------------------------------------

const ref = await capture("ref", {});
assert.equal(ref.code, 0, `a normal capture exits 0: ${ref.text}`);
assert.ok(ref.stills.includes("still.png") && ref.stills.length > 1, `a normal capture writes still.png and the stills beside it: ${ref.stills.join(", ")}`);
log(`reference: ${ref.stills.length} stills`);

// ---- 1: a stale first capture of every still -------------------------------------------------

const stale = await capture("stale-first", { AGV_FAULT: "stale-first" });
assert.equal(stale.code, 0, `a stale first capture: exit 0: ${stale.text}`);
assert.deepEqual(stale.stills, ref.stills, "the same stills as the normal capture");
const served = (stale.text.match(/AGV_STALE served/g) ?? []).length;
assert.equal(served, ref.stills.length, `a stale frame served once per still (${served} for ${ref.stills.length} stills)`);
for (const name of ref.stills) {
  const d = rgbDiff(readPng(readFileSync(join(ref.dir, name))), readPng(readFileSync(join(stale.dir, name))));
  if (d.count) throw new Error(`#2716: ${name} written from a stale capture: it differs from the settled page in ${d.count} pixels (box ${d.bbox?.join(", ")})`);
}
log(`stale first capture: all ${ref.stills.length} stills RGB-identical to the reference`);

// ---- 2: a page that never settles ------------------------------------------------------------

const never = await capture("never-settles", { AGV_FAULT: "never-settles" }, "--capture-tries=3");
assert.equal(never.code, 1, `a page that never settles fails the run: exit 1: ${never.text}`);
assert.deepEqual(never.stills, [], "no still written");
const reason = new RegExp(
  `capture failed: capture:still ${never.out.replace(/[\\^$.*+?()[\]{}|]/g, "\\$&")}: the page did not settle: no two consecutive of 3 captures were pixel-identical \\(--capture-tries\\); no still had been written`,
);
assert.ok(existsSync(`${never.out}.error.txt`), "the reason in <out>.error.txt");
assert.match(readFileSync(`${never.out}.error.txt`, "utf8"), reason);
assert.match(never.text, reason, "the reason on stderr");
log(`never settles: exit 1, no still, ${never.out}.error.txt names it`);

// ---- 3: the run begins ending as a still settles (#2736) ----------------------------------------

const late = await capture("end-as-settles", { AGV_FAULT: "end-as-settles" });
assert.match(late.text, /AGV_END fired/, `the fault fired (else the check is vacuous): ${late.text}`);
assert.equal(late.code, 1, `a run that ended as a still settled: exit 1: ${late.text}`);
assert.ok(existsSync(`${late.out}.error.txt`), "the reason in <out>.error.txt");
const lateSaid = readFileSync(`${late.out}.error.txt`, "utf8");
assert.match(lateSaid, /capture failed: an uncaught error in the main process: .*AGV_FAULT: the run ends as the still settles/);
assert.match(lateSaid, /; no still had been written$/m);
assert.deepEqual(late.stills, [], `#2736: no still written after headlessEnd, beside ${late.out}.error.txt: ${late.stills.join(", ")}`);
log(`run ending as the still settles: exit 1, no still beside ${late.out}.error.txt`);

// ---- 4: a write that throws part way (a full disk) -------------------------------------------------

const partialOut = join(OUT, "partial", "still.png");
const partial = await capture("partial", { AGV_FAULT: "partial", AGV_TARGET: partialOut });
assert.equal(partial.code, 1, `a still that could not be written fails the run: exit 1: ${partial.text}`);
assert.deepEqual(partial.stills, [], `#2736: no partial PNG left: ${partial.stills.join(", ")}`);
assert.ok(existsSync(`${partial.out}.error.txt`), "the reason in <out>.error.txt");
assert.match(readFileSync(`${partial.out}.error.txt`, "utf8"), /AGV_FAULT: ENOSPC/);
log(`partial write: exit 1, no partial PNG, ${partial.out}.error.txt names the write`);

log(`capture:still e2e: all checks passed; runs in ${OUT}`);
process.exit(0);
