// A tool window's page (ADR 0137 D9, D12; H1, H4): a separate sandboxed renderer holding tiles of the
// registered tools, side by side and one above the other, with splitters between them.
//
// Its preload (src/main/toolPreload.ts) hands it one MessagePort, linked to the model window's page,
// with the window's id and the tool of its first tile. Contexts arrive on the port; every tile hears
// each, and draws it at most once per frame, only while it is live and the window is visible
// (./host.ts). Up to `LIVE_TILES` tiles are live; more wait paused, as tabs (./tiles.ts). A tile's
// events go back on the port; its data requests go to main with the tile's id, so each tile's
// newest request wins on its own. The model window adds a tile on the port (`{ addTile }`); the
// window's own "Add tile" picker and each tile's × add and close them here. The page holds no store
// and reads no file.

import type { AddTileMessage, ToolAnswer, ToolContext, ToolContextMessage, ToolEvent, ToolEventMessage, ToolHandle, ToolPortMessage, ToolRequest } from "./contract.ts";
import { ToolScheduler } from "./host.ts";
import { TOOLS } from "./registry.ts";
import { dragRatio, layoutTree, splitTree, TileSet, type SplitNode } from "./tiles.ts";
import { cssVariables } from "../theme/tokens.ts";

const bridge = (window as unknown as { toolHost: { ready(): void; request(q: ToolRequest, seq: number, tile: number): Promise<ToolAnswer> } }).toolHost;
const root = document.getElementById("tool")!;

interface Mounted {
  frame: HTMLElement;
  handle: ToolHandle;
  scheduler: ToolScheduler;
}

function el<K extends keyof HTMLElementTagNameMap>(tag: K, cls: string, text?: string): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

function start(port: MessagePort, windowId: string, firstTool: string): void {
  document.title = `Tool window ${windowId.replace(/^w/, "")} - apeGmshViewer`;
  const bar = el("div", "tiles-bar");
  const add = el("select", "tiles-add");
  add.title = "add a tile of a tool to this window";
  const area = el("div", "tiles-area");
  const tabs = el("div", "tiles-tabs");
  bar.append(add, tabs);
  root.replaceChildren(bar, area);
  add.append(el("option", "", "+ Add tile"), ...Object.values(TOOLS).map((t) => Object.assign(el("option", "", t.title), { value: t.id })));
  add.addEventListener("change", () => {
    if (add.value) addTile(add.value);
    add.selectedIndex = 0;
  });

  const set = new TileSet();
  const mounted = new Map<number, Mounted>();
  let ctx: ToolContext | null = null;
  let tree: SplitNode | null = null;
  let treeIds = "";
  let seq = 0;
  const visible = () => document.visibilityState === "visible";
  const frames = { request: (fn: () => void) => void requestAnimationFrame(() => fn()) };

  function addTile(tool: string): void {
    const def = TOOLS[tool];
    if (!def) return;
    const id = set.add(tool);
    const frame = el("div", "tile");
    const head = el("div", "tile-head");
    const close = el("button", "tile-close", "×");
    close.title = "close this tile";
    head.append(el("span", "tile-title", def.title), close);
    const body = el("div", "tile-body");
    frame.append(head, body);
    area.append(frame);
    const handle = def.mount({ root: body, send: (event: ToolEvent) => port.postMessage({ event } satisfies ToolEventMessage), request: (q) => bridge.request(q, ++seq, id) });
    const scheduler = new ToolScheduler(handle, frames, () => visible() && set.isLive(id));
    mounted.set(id, { frame, handle, scheduler });
    close.addEventListener("click", () => closeTile(id));
    if (ctx) scheduler.push(ctx);
    render();
  }

  function closeTile(id: number): void {
    const m = mounted.get(id);
    if (!m) return;
    m.handle.unmount?.();
    m.frame.remove();
    mounted.delete(id);
    set.remove(id);
    render();
  }

  /** Lay the live tiles out, show the paused ones as tabs, and let each newly live tile draw. */
  function render(): void {
    const live = set.live().map((t) => t.id);
    const ids = live.join(",");
    if (ids !== treeIds) {
      tree = splitTree(live);
      treeIds = ids;
    }
    for (const b of area.querySelectorAll(".split-bar")) b.remove();
    const { tiles, bars } = layoutTree(tree, { x: 0, y: 0, w: area.clientWidth, h: area.clientHeight });
    for (const [id, m] of mounted) {
      const r = tiles.get(id);
      m.frame.hidden = r === undefined;
      if (!r) continue;
      Object.assign(m.frame.style, { left: `${r.x}px`, top: `${r.y}px`, width: `${r.w}px`, height: `${r.h}px` });
      // A tile live again draws the newest context it missed while paused.
      m.scheduler.shown();
    }
    for (const b of bars) {
      const handle = el("div", `split-bar ${b.node.dir}`);
      Object.assign(handle.style, { left: `${b.rect.x}px`, top: `${b.rect.y}px`, width: `${b.rect.w}px`, height: `${b.rect.h}px` });
      handle.addEventListener("pointerdown", (e) => {
        handle.setPointerCapture(e.pointerId);
        const box = area.getBoundingClientRect();
        const move = (m: PointerEvent) => {
          b.node.ratio = dragRatio(b, m.clientX - box.left, m.clientY - box.top);
          requestAnimationFrame(render);
        };
        handle.addEventListener("pointermove", move);
        handle.addEventListener("pointerup", () => handle.removeEventListener("pointermove", move), { once: true });
      });
      area.append(handle);
    }
    tabs.replaceChildren(
      ...set.paused().map((t) => {
        const tab = el("button", "tiles-tab", `${TOOLS[t.tool]?.title ?? t.tool} (paused)`);
        tab.title = "show this tile; the tile shown longest ago pauses in its place";
        tab.addEventListener("click", () => {
          set.bringIn(t.id);
          render();
        });
        return tab;
      }),
    );
  }

  // The app's colour tokens, as the model window's page has them (not applyTheme: its font faces are
  // files, and the tool page's content security policy loads none).
  let theme: string | null = null;
  port.onmessage = (m: MessageEvent) => {
    const data = m.data as Partial<ToolContextMessage & AddTileMessage> | null;
    if (typeof data?.addTile === "string") {
      addTile(data.addTile);
      return;
    }
    const next = data?.context;
    if (!next) return;
    ctx = next;
    if (next.theme !== theme) {
      theme = next.theme;
      for (const [k, v] of Object.entries(cssVariables(next.theme))) document.documentElement.style.setProperty(k, v);
    }
    for (const t of mounted.values()) t.scheduler.push(next);
  };
  document.addEventListener("visibilitychange", () => {
    for (const t of mounted.values()) t.scheduler.shown();
  });
  addEventListener("resize", () => requestAnimationFrame(render));
  addTile(firstTool);
}

window.addEventListener("message", (ev: MessageEvent) => {
  const meta = (ev.data as Partial<ToolPortMessage> | null)?.toolPort;
  const port = ev.ports[0];
  if (!meta || !port) return;
  if (!TOOLS[meta.tool]) {
    root.textContent = `No tool named ${JSON.stringify(meta.tool)}`;
    return;
  }
  start(port, meta.window, meta.tool);
});

bridge.ready();
