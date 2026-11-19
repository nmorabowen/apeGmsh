// A stand-in for the `electron` module, so main.ts loads in plain Node and its
// IPC handlers can be called (test/main/userDoors.test.ts, VU11 #2750). It opens
// no window: `whenReady` never resolves. Only what main.ts touches while it
// loads is here; any other property throws by name, so a new top-level use of
// Electron fails the test loudly instead of being stubbed into silence.
// preload.ts loads against it too (test/main/preload.test.ts, VU12 #2753).

type Handler = (event: unknown, ...args: unknown[]) => unknown;

/** The `ipcMain.handle` registrations, by channel. */
export const handlers = new Map<string, Handler>();

function strict<T extends object>(name: string, known: T): T {
  return new Proxy(known, {
    get(target, key, receiver) {
      if (typeof key === "symbol" || key in target) return Reflect.get(target, key, receiver);
      throw new Error(`electronStub: main.ts used ${name}.${key}, which this stub does not provide`);
    },
  });
}

/** The `app.on` listeners, by event, so a test can deliver a second instance's call (VU12). */
export const appEvents = new Map<string, Handler[]>();

// VU14 (#2771): `whenReady` resolves only when a test calls `ready()`, so a
// driver can let main.ts make its window (a recording stand-in, below) and
// deliver that window's page events. A test that never calls it gets no window.
let markReady: () => void = () => undefined;
const readyOnce = new Promise<void>((r) => (markReady = r));
export const ready = (): void => markReady();

export const app = strict("app", {
  on: (event: string, fn: Handler) => {
    appEvents.set(event, [...(appEvents.get(event) ?? []), fn]);
  },
  exit: (code?: number) => {
    throw new Error(`electronStub: main.ts called app.exit(${code})`);
  },
  quit: () => undefined,
  whenReady: () => readyOnce,
  requestSingleInstanceLock: () => true,
  commandLine: { appendSwitch: () => undefined },
  getVersion: () => "0.0.0-stub",
  // an agent mode's profile under --out (main.ts `agentStartup`); nothing is written
  setPath: () => undefined,
});

/** The `ipcMain.on` listeners, by channel (VU12: `viewer:dropped`, which the preload sends synchronously). */
export const listeners = new Map<string, Handler>();

export const ipcMain = strict("ipcMain", {
  handle: (channel: string, fn: Handler) => {
    handlers.set(channel, fn);
  },
  on: (channel: string, fn: Handler) => {
    listeners.set(channel, fn);
  },
});

// The renderer side, for preload.ts (test/main/preload.test.ts, VU12): what it
// exposes, every message it sends, and the path Electron would give a File.
export const exposed = new Map<string, Record<string, (...args: never[]) => unknown>>();
export const sent: { how: "invoke" | "send" | "sendSync"; channel: string; args: unknown[] }[] = [];
export const filePaths = new WeakMap<object, string>();

export const contextBridge = strict("contextBridge", {
  exposeInMainWorld: (key: string, api: Record<string, (...args: never[]) => unknown>) => {
    exposed.set(key, api);
  },
});

export const ipcRenderer = strict("ipcRenderer", {
  invoke: (channel: string, ...args: unknown[]) => {
    sent.push({ how: "invoke", channel, args });
    return Promise.resolve({ ok: true, channel });
  },
  send: (channel: string, ...args: unknown[]) => {
    sent.push({ how: "send", channel, args });
  },
  sendSync: (channel: string, ...args: unknown[]) => {
    sent.push({ how: "sendSync", channel, args });
    return null;
  },
  on: () => undefined,
  removeListener: () => undefined,
});

/** Electron's rule: a File from the disk (a drop) has its path; one the page made has "". */
export const webUtils = strict("webUtils", { getPathForFile: (f: object) => filePaths.get(f) ?? "" });

/** The page events main.ts listens for on its window (`webContents.on`), and what it sent the page, in order (VU14). */
export const pageEvents = new Map<string, Handler[]>();
export const pageSent: { channel: string; args: unknown[] }[] = [];

/**
 * VU15 (#2776): the window's main frame, as Electron names it (`WebFrameMain`'s
 * `processId` and `routingId`; an IPC event carries the same two as `processId`
 * and `frameId`). A driver puts the frame a load commits with `commitFrame`,
 * then delivers `did-navigate`, and sends from a frame with `pageEvent`.
 */
export type StubFrame = { processId: number; routingId: number };
let frame: StubFrame = { processId: 1, routingId: 1 };
export const commitFrame = (f: StubFrame): void => {
  frame = f;
};
/** The window's page (its `webContents`), once main.ts has made the window. */
export const pages: object[] = [];
/** An IPC event as the page's frame `f` sends it, from `sender` (by default the window's page). */
export const pageEvent = (f: StubFrame, sender: object | undefined = pages[0]) => ({ sender, processId: f.processId, frameId: f.routingId });

/** The window main.ts makes once `ready()` is called: it records, and opens nothing. */
class StubWindow {
  constructor() {
    const webContents = strict("webContents", {
      get mainFrame() {
        return strict("mainFrame", { ...frame });
      },
      on: (event: string, fn: Handler) => {
        pageEvents.set(event, [...(pageEvents.get(event) ?? []), fn]);
      },
      send: (channel: string, ...args: unknown[]) => {
        pageSent.push({ channel, args });
      },
      // a reload is a page load: the driver delivers `did-start-loading` itself
      reload: () => {
        pageSent.push({ channel: "reload", args: [] });
      },
    });
    pages.push(webContents);
    return strict("window", {
      webContents,
      setMenuBarVisibility: () => undefined,
      setContentSize: () => undefined,
      on: () => undefined,
      loadFile: () => Promise.resolve(),
      isMinimized: () => false,
      restore: () => undefined,
      focus: () => undefined,
    }) as unknown as StubWindow;
  }
  static getAllWindows = () => [] as unknown[];
}

export const BrowserWindow = strict("BrowserWindow", StubWindow);

export const dialog = strict("dialog", { showErrorBox: () => undefined });
