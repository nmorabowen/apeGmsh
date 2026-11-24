// ADR 0137 D9, D11, D13 items 3-5, D4 (H1): main's tool windows, with fake windows, ports and memory.
//
// - D13 item 3: a tool window's webPreferences, and the content security policy of its page.
// - D13 item 4: the tool preload exposes `toolHost.ready` only, not the model window's `viewer`.
// - D13 item 5: `tool:ready` links a window only when main made it for a tool.
// - D11: a renderer that dies or stops responding is reported, and the tool can be opened again.
// - D4: under 1 GB available, a tool window is refused, and open tools are paused until memory frees.

import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import { TOOL_FLOOR_MB, toolWebPreferences, ToolWindows, type ToolContents, type ToolPage, type ToolWin } from "../../src/main/toolWindows.ts";
import type { ToolStatus } from "../../src/state/types.ts";

/** The fakes take any listener, as Electron's emitters do. */
type Handler = (...a: any[]) => void;

class FakeContents implements ToolContents {
  posted: { channel: string; message: unknown; transfer?: unknown[] }[] = [];
  handlers = new Map<string, Handler>();
  postMessage(channel: string, message: unknown, transfer?: unknown[]) {
    this.posted.push({ channel, message, transfer });
  }
  on(event: string, fn: Handler) {
    this.handlers.set(event, fn);
    return this;
  }
  fire(event: string, ...a: unknown[]) {
    this.handlers.get(event)!(...a);
  }
}

class FakeWin implements ToolWin {
  readonly webContents = new FakeContents();
  focused = 0;
  destroyed = false;
  private closedFn: (() => void) | null = null;
  focus() {
    this.focused++;
  }
  close() {
    this.destroyed = true;
    this.closedFn?.();
  }
  isDestroyed() {
    return this.destroyed;
  }
  on(_: "closed", fn: () => void) {
    this.closedFn = fn;
    return this;
  }
}

class FakePage implements ToolPage {
  sent: { channel: string; args: unknown[] }[] = [];
  posted: { channel: string; message: unknown; transfer?: unknown[] }[] = [];
  send(channel: string, ...args: unknown[]) {
    this.sent.push({ channel, args });
  }
  postMessage(channel: string, message: unknown, transfer?: unknown[]) {
    this.posted.push({ channel, message, transfer });
  }
  isDestroyed() {
    return false;
  }
  statuses(): ToolStatus[] {
    return this.sent.filter((s) => s.channel === "tool:status").map((s) => s.args[0] as ToolStatus);
  }
  pauses(): unknown[] {
    return this.sent.filter((s) => s.channel === "tool:paused").map((s) => s.args[0]);
  }
}

function setup(mb: number | null = 8000) {
  const made: FakeWin[] = [];
  let memory = mb;
  let tick: (() => void) | null = null;
  const tools = new ToolWindows(
    {
      create: () => {
        const w = new FakeWin();
        made.push(w);
        return w;
      },
      channel: () => ({ port1: "page-end", port2: "tool-end" }),
      availableMB: () => memory,
      every: (_ms, fn) => {
        tick = fn;
        return () => (tick = null);
      },
    },
    { step: "Step readout" },
  );
  return { tools, made, setMemory: (m: number | null) => (memory = m), tick: () => tick?.(), polling: () => tick !== null };
}

test("D13 item 3: a tool window is sandboxed, context-isolated, without Node, with its own preload", () => {
  assert.deepEqual(toolWebPreferences("/app/dist/toolPreload.cjs"), {
    preload: "/app/dist/toolPreload.cjs",
    sandbox: true,
    contextIsolation: true,
    nodeIntegration: false,
    backgroundThrottling: false,
  });
});

test("D13 item 3: the tool page's content security policy is the ADR's, exactly", () => {
  const html = readFileSync(fileURLToPath(new URL("../../src/tools/tool.html", import.meta.url)), "utf8");
  const csp = /<meta http-equiv="Content-Security-Policy" content="([^"]+)"/.exec(html)?.[1];
  assert.equal(csp, "default-src 'none'; script-src 'self'; style-src 'unsafe-inline'");
  assert.doesNotMatch(html, /<script>[^<]/, "no inline script (script-src 'self' would refuse it)");
});

test("opening a tool makes one window, reports opening, and a second open focuses it", () => {
  const { tools, made } = setup();
  const page = new FakePage();
  assert.deepEqual(tools.openTool(page, "step"), { ok: true });
  assert.equal(made.length, 1);
  assert.deepEqual(page.statuses().map((s) => s.status), ["opening"]);
  assert.deepEqual(tools.openTool(page, "step"), { ok: true });
  assert.equal(made.length, 1, "no second window");
  assert.equal(made[0]!.focused, 1);
});

test("an unknown tool is refused before any window is made", () => {
  const { tools, made } = setup();
  const page = new FakePage();
  assert.deepEqual(tools.openTool(page, "../../evil"), { ok: false, reason: 'no tool named "../../evil"' });
  assert.deepEqual(tools.openTool(page, 7), { ok: false, reason: "no tool named 7" });
  assert.equal(made.length, 0);
});

test("D13 item 5: only the tool's own window is linked when it says it is ready", () => {
  const { tools, made } = setup();
  const page = new FakePage();
  tools.openTool(page, "step");
  assert.equal(tools.ready({}), false, "a sender main did not make for a tool");
  assert.equal(page.posted.length, 0);
  assert.equal(tools.ready(made[0]!.webContents), true);
  assert.deepEqual(made[0]!.webContents.posted, [{ channel: "tool:port", message: { tool: "step" }, transfer: ["tool-end"] }]);
  assert.deepEqual(page.posted, [{ channel: "tool:port", message: { tool: "step" }, transfer: ["page-end"] }]);
  assert.deepEqual(page.statuses().map((s) => s.status), ["opening", "open"]);
});

test("D11: a renderer that dies or hangs is reported; the tool opens again in a new window", () => {
  const { tools, made } = setup();
  const page = new FakePage();
  tools.openTool(page, "step");
  tools.ready(made[0]!.webContents);
  made[0]!.webContents.fire("unresponsive");
  made[0]!.webContents.fire("responsive");
  made[0]!.webContents.fire("render-process-gone", {}, { reason: "crashed", exitCode: -1 });
  const after = page.statuses().slice(2);
  assert.deepEqual(after.map((s) => [s.status, s.reason]), [["unresponsive", null], ["open", null], ["stopped", "its renderer crashed"]]);
  // Opening it again (Spotlight) replaces the dead window, never just focuses it.
  assert.deepEqual(tools.openTool(page, "step"), { ok: true });
  assert.equal(made[0]!.destroyed, true, "the dead window is closed");
  assert.equal(made[0]!.focused, 0);
  assert.equal(made.length, 2);
  assert.equal(page.statuses().at(-1)!.status, "opening", "the replaced window reports no 'closed' over the new one");
  assert.deepEqual(tools.openTools(), ["step"]);
  made[1]!.close();
  assert.equal(page.statuses().at(-1)!.status, "closed");
  assert.deepEqual(tools.openTools(), []);
  tools.openTool(page, "step");
  assert.equal(made.length, 3, "opened again in a new window after the user closed it");
});

test("D4: under 1 GB available a tool window is refused, with how much there is", () => {
  const { tools, made } = setup(TOOL_FLOOR_MB - 1);
  const page = new FakePage();
  const r = tools.openTool(page, "step");
  assert.deepEqual(r, { ok: false, reason: "1.0 GB of RAM is available, under 1.0 GB" });
  assert.equal(made.length, 0);
  assert.deepEqual(page.statuses().map((s) => [s.status, s.reason]), [["refused", "1.0 GB of RAM is available, under 1.0 GB"]]);
  const at = setup(TOOL_FLOOR_MB);
  assert.deepEqual(at.tools.openTool(new FakePage(), "step"), { ok: true }, "at the floor it opens");
});

test("D4: open tools pause when memory falls under 1 GB and resume when it frees; polling stops with the last window", () => {
  const { tools, made, setMemory, tick, polling } = setup(4000);
  const page = new FakePage();
  tools.openTool(page, "step");
  tools.ready(made[0]!.webContents);
  assert.equal(polling(), true);
  setMemory(900);
  tick();
  tick();
  setMemory(1500);
  tick();
  assert.deepEqual(page.pauses(), [true, false], "one message per change");
  made[0]!.close();
  assert.equal(polling(), false);
});

test("a tool window linked while paused is told at once", () => {
  const { tools, made, setMemory, tick } = setup(4000);
  const page = new FakePage();
  tools.openTool(page, "step");
  setMemory(900);
  tick();
  tools.ready(made[0]!.webContents);
  assert.deepEqual(page.pauses(), [true, true]);
});

test("D13 item 5: isTool is true only for a live tool window main made, not for a dead one", () => {
  const { tools, made } = setup();
  const page = new FakePage();
  tools.openTool(page, "step");
  assert.equal(tools.isTool(made[0]!.webContents), true);
  assert.equal(tools.isTool({}), false);
  assert.equal(tools.isTool(page), false, "the model window's page is not a tool");
  made[0]!.webContents.fire("render-process-gone", {}, { reason: "crashed", exitCode: -1 });
  assert.equal(tools.isTool(made[0]!.webContents), false);
});

test("D13 item 4: the tool preload exposes toolHost.ready and request only, and relays its port", () => {
  const STUB = new URL("./electronStub.ts", import.meta.url).href;
  const PRELOAD = new URL("../../src/main/toolPreload.ts", import.meta.url).href;
  const out = execFileSync(
    process.execPath,
    [
      "--input-type=module",
      "-e",
      `import { registerHooks } from "node:module";
registerHooks({ resolve: (s, c, n) => (s === "electron" ? { url: ${JSON.stringify(STUB)}, shortCircuit: true } : n(s, c)) });
globalThis.window = { postMessage() {} };
const stub = await import(${JSON.stringify(STUB)});
await import(${JSON.stringify(PRELOAD)});
const api = stub.exposed;
process.stdout.write(JSON.stringify({ keys: [...api.keys()], tool: Object.keys(api.get("toolHost") ?? {}) }));
api.get("toolHost").ready();
process.stdout.write(" " + JSON.stringify(stub.sent));`,
    ],
    { encoding: "utf8" },
  );
  const [exposed, sent] = out.split(" ");
  assert.deepEqual(JSON.parse(exposed!), { keys: ["toolHost"], tool: ["ready", "request"] });
  assert.deepEqual(JSON.parse(sent!), [{ how: "send", channel: "tool:ready", args: [] }]);
});
