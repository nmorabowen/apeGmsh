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

export const app = strict("app", {
  on: (event: string, fn: Handler) => {
    appEvents.set(event, [...(appEvents.get(event) ?? []), fn]);
  },
  exit: (code?: number) => {
    throw new Error(`electronStub: main.ts called app.exit(${code})`);
  },
  quit: () => undefined,
  whenReady: () => new Promise<void>(() => undefined),
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

export const BrowserWindow = strict("BrowserWindow", { getAllWindows: () => [] as unknown[] });

export const dialog = strict("dialog", { showErrorBox: () => undefined });
