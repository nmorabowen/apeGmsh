// ADR 0137 D13 items 1 and 2 (H0): every webContents the app creates denies
// every new-window request, and prevents `will-navigate` / `will-redirect`
// unless the target is the page already loaded.
//
// Two oracles. src/main/windowPolicy.ts against a fake webContents that records
// the handlers it is given; and main.ts itself, loaded in a child process
// against ./electronStub.ts (./windowPolicyDriver.ts), which must register the
// policy on `web-contents-created` before any window exists and make its model
// window with D13's posture.

import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import { guardContents, isLoadedPage, type NavigationEvent } from "../../src/main/windowPolicy.ts";

const PAGE = "file:///C:/Program%20Files/apeGmshViewer/resources/app/dist/index.html";

type Listener = (event: NavigationEvent, url?: string) => void;

/** A webContents that shows `url` and records what the policy hands it. */
function fakeContents(url: string) {
  const listeners = new Map<string, Listener[]>();
  let openHandler: ((d: { url: string }) => { action: "deny" }) | null = null;
  const contents = {
    url,
    getURL: () => contents.url,
    setWindowOpenHandler: (h: (d: { url: string }) => { action: "deny" }) => {
      openHandler = h;
    },
    on: (event: string, fn: Listener) => {
      listeners.set(event, [...(listeners.get(event) ?? []), fn]);
    },
  };
  guardContents(contents);
  /** Deliver `event` for `target` as Electron 25+ does (the URL on the event and as the deprecated second argument); true when a listener prevented it. */
  const deliver = (event: "will-navigate" | "will-redirect", target: string): boolean => {
    let stopped = false;
    const fns = listeners.get(event) ?? [];
    assert.ok(fns.length > 0, `no ${event} listener`);
    for (const fn of fns) fn({ url: target, preventDefault: () => (stopped = true) }, target);
    return stopped;
  };
  return { contents, deliver, open: (target: string) => openHandler?.({ url: target }) ?? null };
}

test("D13.1: every new-window request is denied, the app's own page included", () => {
  const { open } = fakeContents(PAGE);
  for (const target of ["about:blank", PAGE, "https://example.com/", "file:///C:/Windows/System32/", "javascript:alert(1)"]) {
    assert.deepEqual(open(target), { action: "deny" }, target);
  }
});

test("D13.2: a navigation or a redirect away from the loaded page is prevented", () => {
  const { deliver } = fakeContents(PAGE);
  const away = [
    "https://example.com/",
    "about:blank",
    "file:///C:/Users/someone/model.h5",
    "file:///C:/Program%20Files/apeGmshViewer/resources/app/dist/other.html",
    `${PAGE}?mode=evil`,
    "data:text/html,<p>hi</p>",
    "not a url",
    "",
  ];
  for (const target of away) {
    assert.equal(deliver("will-navigate", target), true, `will-navigate ${target}`);
    assert.equal(deliver("will-redirect", target), true, `will-redirect ${target}`);
  }
});

test("D13.2: the loaded page itself still loads and reloads", () => {
  const { contents, deliver } = fakeContents(PAGE);
  // A navigation to the page already loaded (a reload by navigation), with or without a fragment.
  assert.equal(deliver("will-navigate", PAGE), false);
  assert.equal(deliver("will-navigate", `${PAGE}#inspector`), false);
  assert.equal(deliver("will-redirect", PAGE), false);
  // The page shown with a fragment is still the same page.
  contents.url = `${PAGE}#a`;
  assert.equal(deliver("will-navigate", PAGE), false);
  // The watch reload is `webContents.reload()` (main.ts `deliver`), which emits neither event:
  // the policy registers no reload or load listener that could stop it.
});

test("D13.2: a contents that shows nothing yet has no page to stay on", () => {
  const { deliver } = fakeContents("");
  assert.equal(deliver("will-navigate", PAGE), true);
  assert.equal(isLoadedPage("", ""), false);
});

test("isLoadedPage ignores the fragment and nothing else", () => {
  assert.equal(isLoadedPage(PAGE, `${PAGE}#x`), true);
  assert.equal(isLoadedPage(`${PAGE}#y`, PAGE), true);
  assert.equal(isLoadedPage(PAGE, PAGE.replace("index", "Index")), false);
  assert.equal(isLoadedPage(PAGE, `${PAGE}?a=1`), false);
});

test("main.ts registers the policy on web-contents-created before any window, and its model window keeps D13's posture", () => {
  const driver = fileURLToPath(new URL("./windowPolicyDriver.ts", import.meta.url));
  const stdout = execFileSync(process.execPath, [driver], { encoding: "utf8", stdio: ["ignore", "pipe", "pipe"], timeout: 60_000 });
  const line = stdout.split(/\r?\n/).find((l) => l.startsWith("WINDOW_POLICY "));
  assert.ok(line, `the driver printed nothing:\n${stdout}`);
  const seen = JSON.parse(line.slice("WINDOW_POLICY ".length)) as Record<string, unknown>;
  assert.equal(seen.createdBeforeReady, 1);
  assert.equal(seen.windowOpen, "deny");
  assert.equal(seen.navigateAway, true);
  assert.equal(seen.navigateSelf, false);
  assert.equal(seen.redirectAway, true);
  const prefs = seen.webPreferences as Record<string, unknown>;
  assert.equal(prefs.contextIsolation, true);
  assert.equal(prefs.nodeIntegration, false);
  assert.equal(prefs.sandbox, true);
});
