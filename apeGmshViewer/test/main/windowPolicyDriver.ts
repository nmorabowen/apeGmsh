// Loads main.ts in plain Node, with `electron` resolved to ./electronStub.ts,
// and reports what ADR 0137 D13 (H0) needs of it: that it listens for
// `web-contents-created` before any window exists, that the listener guards a
// contents it is handed (a fake one here), and the `webPreferences` of the
// model window it makes once ready (test/main/windowPolicy.test.ts). It runs in
// a child process because main.ts reads its mode from argv; it prints one JSON
// line and exits.
//
//   node test/main/windowPolicyDriver.ts

import { registerHooks } from "node:module";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

const STUB = new URL("./electronStub.ts", import.meta.url).href;
registerHooks({
  resolve: (specifier, context, next) => (specifier === "electron" ? { url: STUB, shortCircuit: true } : next(specifier, context)),
});

const { appEvents, ready, windowOptions } = (await import(STUB)) as typeof import("./electronStub.ts");
await import(pathToFileURL(join(import.meta.dirname, "..", "..", "src", "main", "main.ts")).href);

const createdBeforeReady = (appEvents.get("web-contents-created") ?? []).length;

// A fake contents showing the app's page, handed to main's listener as Electron would.
const PAGE = "file:///C:/app/dist/index.html";
let openHandler: ((d: { url: string }) => { action: string }) | null = null;
const listeners = new Map<string, ((e: { url?: string; preventDefault(): void }, url?: string) => void)[]>();
const contents = {
  getURL: () => PAGE,
  setWindowOpenHandler: (h: (d: { url: string }) => { action: string }) => {
    openHandler = h;
  },
  on: (event: string, fn: (e: { url?: string; preventDefault(): void }, url?: string) => void) => {
    listeners.set(event, [...(listeners.get(event) ?? []), fn]);
  },
};
for (const fn of appEvents.get("web-contents-created") ?? []) fn({}, contents);

function prevented(event: string, url: string): boolean {
  let stopped = false;
  for (const fn of listeners.get(event) ?? []) fn({ url, preventDefault: () => (stopped = true) }, url);
  return stopped;
}

ready();
// `whenReady().then(...)` runs on a later microtask; let it make its window.
await new Promise((r) => setTimeout(r, 50));
const model = windowOptions[0] as { webPreferences?: Record<string, unknown> } | undefined;

process.stdout.write(
  `WINDOW_POLICY ${JSON.stringify({
    createdBeforeReady,
    windowOpen: openHandler === null ? null : (openHandler as (d: { url: string }) => { action: string })({ url: PAGE }).action,
    navigateAway: prevented("will-navigate", "https://example.com/"),
    navigateSelf: prevented("will-navigate", PAGE),
    redirectAway: prevented("will-redirect", "file:///C:/elsewhere/index.html"),
    webPreferences: model?.webPreferences ?? null,
  })}\n`,
);
process.exit(0);
