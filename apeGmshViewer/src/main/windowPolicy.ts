// ADR 0137 D13 items 1 and 2 (H0): the window and navigation policy, on every
// `webContents` the app creates. main.ts applies it from
// `app.on("web-contents-created")`, so a window added later inherits it
// without asking.
//
//   1. `setWindowOpenHandler` denies every request: `window.open`, a link with
//      `target=_blank`, a shift-click. Only main creates windows. (The ADR's
//      spike: `window.open` from the page made windows for `about:blank` and
//      for the app's own index.html.)
//   2. `will-navigate` and `will-redirect` are prevented unless the target is
//      the page already loaded. Main's own loads (`loadFile`, `loadURL`) and
//      `webContents.reload()` (main.ts `deliver`, the watch reload) emit
//      neither event, so they are untouched; an in-page move (a `#hash`) emits
//      neither either.
//
// "The page already loaded" is the URL the contents shows now, without its
// fragment. A contents that shows nothing yet has no page, so nothing matches.
//
// Items 3-5 of D13 (the tool window's preload, its CSP, sender checks on tool
// requests) land with H1, when a tool window first exists.

/** What the policy needs of an Electron `WebContents`; a test hands it a fake. */
export interface GuardedContents {
  getURL(): string;
  setWindowOpenHandler(handler: (details: { url: string }) => { action: "deny" }): void;
  on(event: "will-navigate" | "will-redirect", listener: (event: NavigationEvent, url?: string) => void): unknown;
}

/** The part of Electron's navigation event the policy reads: `url` (Electron 25+) and `preventDefault`. */
export interface NavigationEvent {
  url?: string;
  preventDefault(): void;
}

/** `url` without its fragment, or null when it does not parse. */
function page(url: string): string | null {
  try {
    const u = new URL(url);
    u.hash = "";
    return u.href;
  } catch {
    return null;
  }
}

/** True when `target` is the page `current` already shows (fragments ignored). */
export function isLoadedPage(current: string, target: string): boolean {
  if (!current) return false;
  const here = page(current);
  return here !== null && here === page(target);
}

/** Apply D13 items 1 and 2 to `contents`. */
export function guardContents(contents: GuardedContents): void {
  contents.setWindowOpenHandler(() => ({ action: "deny" }));
  const stay = (event: NavigationEvent, url?: string) => {
    const target = event.url ?? url ?? "";
    if (!isLoadedPage(contents.getURL(), target)) event.preventDefault();
  };
  contents.on("will-navigate", stay);
  contents.on("will-redirect", stay);
}
