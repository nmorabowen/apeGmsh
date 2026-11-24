// ADR 0137 D10 (H1): the model window's links to its tool windows (src/renderer/tools.ts), over
// Node's own MessageChannel, with a store that runs the real reducer.
//
// - a linked tool is sent the context at once, then only when it changes;
// - nothing is sent while paused, and the newest context on resume;
// - a tool's event reaches the store only when the state allows it;
// - an `openTool` request goes to main, and main's statuses reach the store;
// - a port from another window is ignored.

import assert from "node:assert/strict";
import { MessageChannel, type MessagePort } from "node:worker_threads";
import { test } from "node:test";
import { mountToolLinks } from "../../src/renderer/tools.ts";
import { initialState, reduce } from "../../src/state/store.ts";
import type { Event, State, ToolStatus } from "../../src/state/types.ts";

class TinyStore {
  state: State;
  private listeners: ((s: State, prev: State) => void)[] = [];
  dispatched: Event[] = [];
  constructor(s: State) {
    this.state = s;
  }
  get = () => this.state;
  subscribe = (fn: (s: State, prev: State) => void) => {
    this.listeners.push(fn);
    return () => void (this.listeners = this.listeners.filter((f) => f !== fn));
  };
  dispatch = (e: Event) => {
    this.dispatched.push(e);
    const prev = this.state;
    const next = reduce(prev, e);
    if (next === prev) return;
    this.state = next;
    for (const l of this.listeners) l(next, prev);
  };
}

class FakeWindow {
  listeners: ((e: unknown) => void)[] = [];
  addEventListener(_: string, fn: (e: unknown) => void) {
    this.listeners.push(fn);
  }
  removeEventListener(_: string, fn: (e: unknown) => void) {
    this.listeners = this.listeners.filter((f) => f !== fn);
  }
  deliver(data: unknown, port: MessagePort, source: unknown = this) {
    for (const l of this.listeners) l({ data, ports: [port], source });
  }
}

function onResults(): State {
  const s = reduce(initialState, { type: "fileOpened", artifact: "model", path: "/m/bridge.h5" });
  return {
    ...s,
    results: { stages: [{ id: "stage_0", name: "push", time: [0, 1, 2, 3] }] } as unknown as State["results"],
    phase: { axis: [{ kind: "results", stage: 0, step: 0 }] as unknown as State["phase"]["axis"], at: { kind: "results", stage: 0, step: 0 } },
  };
}

const settle = () => new Promise((r) => setTimeout(r, 20));

function setup() {
  const store = new TinyStore(onResults());
  const win = new FakeWindow();
  const opened: string[] = [];
  let status: ((s: ToolStatus) => void) | null = null;
  let paused: ((p: boolean) => void) | null = null;
  const off = mountToolLinks(
    store as never,
    { openTool: async (t) => void opened.push(t), onToolStatus: (cb) => void (status = cb), onToolsPaused: (cb) => void (paused = cb) },
    win as unknown as Window,
  );
  const { port1, port2 } = new MessageChannel();
  const got: { context?: { step: { step: number } | null; theme: string }; event?: unknown }[] = [];
  port2.on("message", (m) => got.push(m));
  return { store, win, opened, off, port1, port2, got, status: (s: ToolStatus) => status!(s), paused: (p: boolean) => paused!(p) };
}

test("a linked tool gets the context at once, then only when it changes", async () => {
  const h = setup();
  try {
    h.win.deliver({ toolPort: { tool: "step" } }, h.port1 as never);
    await settle();
    assert.equal(h.got.length, 1);
    assert.equal(h.got[0]!.context!.step!.step, 0);
    h.store.dispatch({ type: "setTheme", theme: "light" });
    h.store.dispatch({ type: "hover", pick: null } as unknown as Event);
    await settle();
    assert.deepEqual(h.got.map((m) => m.context!.theme), ["dark", "light"], "a hover changes no context and sends nothing");
  } finally {
    h.off();
    h.port2.close();
  }
});

test("paused: nothing is sent; on resume the newest context is sent", async () => {
  const h = setup();
  try {
    h.win.deliver({ toolPort: { tool: "step" } }, h.port1 as never);
    h.paused(true);
    h.store.dispatch({ type: "setTheme", theme: "light" });
    h.store.dispatch({ type: "setTheme", theme: "dark" });
    h.store.dispatch({ type: "setTheme", theme: "light" });
    await settle();
    assert.equal(h.got.length, 1, "only the first context, from before the pause");
    h.paused(false);
    await settle();
    assert.deepEqual(h.got.map((m) => m.context!.theme), ["dark", "light"]);
  } finally {
    h.off();
    h.port2.close();
  }
});

test("a tool's event reaches the store only when the state allows it", async () => {
  const h = setup();
  try {
    h.win.deliver({ toolPort: { tool: "step" } }, h.port1 as never);
    h.port2.postMessage({ event: { type: "setResultStep", step: 2 } });
    h.port2.postMessage({ event: { type: "setResultStep", step: 9 } });
    h.port2.postMessage({ event: { type: "fileOpened", artifact: "model", path: "/evil.h5" } });
    await settle();
    const fromTool = h.store.dispatched.filter((e) => e.type === "setResultStep" || e.type === "fileOpened");
    assert.deepEqual(fromTool, [{ type: "setResultStep", step: 2 }]);
  } finally {
    h.off();
    h.port2.close();
  }
});

test("an openTool request goes to main once; main's statuses and pauses reach the store", async () => {
  const h = setup();
  try {
    h.store.dispatch({ type: "openTool", tool: "step" });
    h.store.dispatch({ type: "setResultStep", step: 1 });
    await settle();
    assert.deepEqual(h.opened, ["step"]);
    h.status({ tool: "step", title: "Step readout", status: "stopped", reason: "its renderer crashed" });
    h.paused(true);
    assert.equal(h.store.get().tools.windows["step"]?.status, "stopped");
    assert.equal(h.store.get().tools.paused, true);
  } finally {
    h.off();
    h.port2.close();
  }
});

test("a port that does not come from this window is ignored", async () => {
  const h = setup();
  try {
    h.win.deliver({ toolPort: { tool: "step" } }, h.port1 as never, {});
    await settle();
    assert.equal(h.got.length, 0);
  } finally {
    h.off();
    h.port1.close();
    h.port2.close();
  }
});
