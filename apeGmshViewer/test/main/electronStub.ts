// A stand-in for the `electron` module, so main.ts loads in plain Node and its
// IPC handlers can be called (test/main/userDoors.test.ts, VU11 #2750). It opens
// no window: `whenReady` never resolves. Only what main.ts touches while it
// loads is here; any other property throws by name, so a new top-level use of
// Electron fails the test loudly instead of being stubbed into silence.

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

export const app = strict("app", {
  on: () => undefined,
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

export const ipcMain = strict("ipcMain", {
  handle: (channel: string, fn: Handler) => {
    handlers.set(channel, fn);
  },
  on: () => undefined,
});

export const BrowserWindow = strict("BrowserWindow", { getAllWindows: () => [] as unknown[] });

export const dialog = strict("dialog", { showErrorBox: () => undefined });
