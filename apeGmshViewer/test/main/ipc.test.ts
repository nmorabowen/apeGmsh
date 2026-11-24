// Review finding 5 (2026-11-24): one typed page↔main channel map (src/main/ipc.ts).
//
// The typecheck does most of the work: main answers through `handle`, checked against the map; the
// preload `satisfies PreloadApi`; renderer/app.ts hands each door `PreloadApi` without a cast. These
// tests keep the map and main in step at run time, and keep raw IPC out of the code:
//
// - every channel main registers is in the map, and every channel in the map is registered;
// - no `ipcMain.handle/on`, `ipcRenderer.invoke/send/on` or `webContents.send` outside ipc.ts, but
//   for the few listed here with why.

import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { copyFileSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, relative } from "node:path";
import { after, test } from "node:test";
import { fileURLToPath } from "node:url";
import type { Invoke, Send, ToolInvoke } from "../../src/main/ipc.ts";

// The typecheck makes these lists complete and exact: a channel missing here, or not in the map, fails `npm run typecheck`.
const INVOKE = {
  "app:config": true, "app:fail": true, "model:open": true, "geometry:open": true, "results:open": true, "results:readStep": true,
  "results:hingeHistory": true, "results:scan": true, "results:scanCancel": true, "results:close": true, "results:step": true,
  "particles:index": true, "particles:readOutput": true, "particles:set": true, "particles:harness": true, "particles:harnessStill": true,
  "particles:harnessDone": true, "source:goto": true, "user:read": true, "user:readCase": true, "user:readResult": true,
  "workspace:read": true, "workspace:save": true, "combinations:save": true, "variants:current": true, "variants:switch": true,
  "variants:compare": true, "compare:readStep": true, "compare:close": true, "compare:model": true, "compare:userRead": true,
  "compare:userReadResult": true, "viewer:currentSet": true, "viewer:requestOpen": true, "viewer:offerOpen": true,
  "viewer:snapshotReply": true, "render:job": true, "render:variant": true, "render:png": true, "render:capture": true,
  "render:done": true, "capture:still": true, "capture:done": true, "capture:resize": true, "measure:metrics": true,
  "measure:done": true, "measure:cpu": true, "measure:memory": true, "materialTest:run": true, "materialTest:cancel": true,
  "materialTest:saveCsv": true, "tool:open": true, "tool:request": true,
} satisfies Record<keyof Invoke | keyof ToolInvoke, true>;
const SEND = { "viewer:dropped": true, "viewer:subscribe": true, "viewer:unsubscribe": true, "viewer:offersReady": true, "tool:ready": true } satisfies Record<keyof Send | "tool:ready", true>;

const root = mkdtempSync(join(tmpdir(), "agv-ipc-"));
after(() => rmSync(root, { recursive: true, force: true }));

test("main registers exactly the map's channels: every request it answers and every message it hears", () => {
  const dir = join(root, "set");
  mkdirSync(dir);
  copyFileSync(fileURLToPath(new URL("../../fixtures/bridge_provenance.h5", import.meta.url)), join(dir, "s.h5"));
  const stdout = execFileSync(process.execPath, [fileURLToPath(new URL("./loaderDriver.ts", import.meta.url)), "--case=channels", "--mode=describe", `--file=${join(dir, "s.h5")}`, `--out=${join(dir, "out")}`], {
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
    timeout: 60_000,
  });
  const line = stdout.split(/\r?\n/).find((l) => l.startsWith("LOADER "));
  assert.ok(line, stdout);
  const got = JSON.parse(line.slice("LOADER ".length)) as { handles: string[]; listens: string[] };
  // `results:step` is registered by the measure-results run only (main/modes/measure.ts), so a describe run lacks it.
  const always = Object.keys(INVOKE).filter((c) => c !== "results:step").sort();
  assert.deepEqual(got.handles, always, "the channels main answers are the map's Invoke and ToolInvoke");
  assert.deepEqual(got.listens, Object.keys(SEND).sort(), "the channels main hears are the map's Send and tool:ready");
});

/** Raw IPC that stays outside the typed helpers, each with why. */
const ALLOWED: Record<string, RegExp[]> = {
  // the helpers themselves
  "main/ipc.ts": [/ipc\.(handle|on|invoke|send|removeListener)\(/, /to\.send\(/],
  // a drop is sent synchronously (`sendSync`), so main records it before any open names it
  "main/preload.ts": [/ipcRenderer\.sendSync\("viewer:dropped"/, /ipcRenderer\.on\("tool:port"/],
  // a tool window's own preload: its port, and its two channels (ToolInvoke, tool:ready)
  "main/toolPreload.ts": [/ipcRenderer\.on\("tool:port"/, /ipcRenderer\.send\("tool:ready"\)/, /ipcRenderer\.invoke\("tool:request"/],
  // main hands each end of a tool's port pair over (`postMessage` with a transfer list, not `send`)
  "main/toolWindows.ts": [],
  // the measure-results run replaces its own read handler
  "main/modes/measure.ts": [/ipcMain\.removeHandler\("results:step"\)/],
  // workspace and combinations register through a narrow, tested `ipc` parameter; their answers are typed WorkspaceIn/WorkspaceSaveIn/CombinationSaveIn, the map's
  "main/workspace.ts": [/ipc\.handle\("workspace:(read|save)"/],
  "main/combinations.ts": [/ipc\.handle\("combinations:save"/],
};
const RAW = /\b(ipcMain\.(handle|on|once)|ipcRenderer\.(invoke|send|sendSync|on|once)|webContents\.send|ipc\.handle|ipc\.on|ipc\.invoke)\(/;

function walk(dir: string): string[] {
  return readdirSync(dir).flatMap((n) => {
    const p = join(dir, n);
    return statSync(p).isDirectory() ? walk(p) : p.endsWith(".ts") ? [p] : [];
  });
}

test("no raw IPC outside the typed helpers, but for the few allowed with why", () => {
  const src = fileURLToPath(new URL("../../src/", import.meta.url));
  const stray: string[] = [];
  for (const f of walk(src)) {
    const rel = relative(src, f).replaceAll("\\", "/");
    readFileSync(f, "utf8")
      .split("\n")
      .forEach((line, i) => {
        if (line.trimStart().startsWith("//") || line.trimStart().startsWith("*")) return;
        if (!RAW.test(line)) return;
        if ((ALLOWED[rel] ?? []).some((ok) => ok.test(line))) return;
        stray.push(`${rel}:${i + 1}: ${line.trim()}`);
      });
  }
  assert.deepEqual(stray, []);
});

test("the raw-IPC pattern catches each raw call, and not the typed helpers", () => {
  for (const raw of ['ipcMain.handle("x", f)', 'ipcMain.on("x", f)', 'ipcRenderer.invoke("x")', 'ipcRenderer.send("x")', 'win.webContents.send("x", 1)', "ipcRenderer.on(`viewer:${c}`, h)"]) {
    assert.ok(RAW.test(raw), raw);
  }
  for (const typed of ['handle(ipcMain, "x")(f)', 'listen(ipcMain, "x", f)', 'invoke(ipcRenderer, "x")', 'push(win.webContents, "x", 1)', 'hear(ipcRenderer, "x", cb)']) {
    assert.ok(!RAW.test(typed), typed);
  }
});
